"""Read-only independent K5 artifact and event arithmetic verification."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.data_loader import sha256

def verify():
    p=ROOT/'experiments/k5_entry_diagnostic'
    manifest=json.loads((p/'artifact_manifest.json').read_text())
    for file,r in manifest.items():assert sha256(p/file)==r['sha256'] and (p/file).stat().st_size==r['bytes'],file
    meta=json.loads((p/'metadata.json').read_text())
    for file,h in meta['source_sha256'].items():assert sha256(ROOT/file)==h,file
    assert meta['no_new_portfolio_backtest']
    e=pd.read_csv(p/'events.csv');s=pd.read_csv(p/'signals.csv');c=pd.read_csv(p/'cycles.csv')
    beta=s.beta.to_numpy();g=s.g.to_numpy();n=len(s)
    assert n==70176 and pd.to_datetime(s.date,utc=True).iloc[-1]<pd.Timestamp('2022-01-01',tz='UTC')
    x=np.arange(5);windows=np.lib.stride_tricks.sliding_window_view(beta,5);valid=np.isfinite(windows).all(axis=1)
    independent=np.linalg.lstsq(np.column_stack([np.ones(5),x]),windows[valid].T,rcond=None)[0][1]
    np.testing.assert_allclose(g[4:][valid],independent,atol=1e-15,rtol=0)
    long=(beta[1:]<0)&(g[:-1]<=0)&(g[1:]>0);short=(beta[1:]>0)&(g[:-1]>=0)&(g[1:]<0)
    expected=np.flatnonzero(np.r_[False,long|short]);expected=expected[(expected>=95)&(expected<n-1)]
    np.testing.assert_array_equal(e.source_index,expected)
    assert e.event_id.is_unique and (e.entry_index==e.source_index+1).all()
    assert (pd.to_datetime(e.entry_time,utc=True)==pd.to_datetime(e.source_bar,utc=True)+pd.Timedelta(minutes=15)).all()
    np.testing.assert_allclose(e.gross_usdt-e.fees_usdt-e.slippage_usdt,e.net_usdt,rtol=0,atol=1e-7)
    assert (e.mae_pct>=0).all() and (e.mfe_pct>=0).all()
    assert e.censored.eq(~e.confirmation_executable).all()
    assert e.is_first.eq(e.ordinal_in_cycle==1).all()
    assert c.events.sum()==len(e)
    for r in e.itertuples():
        i=r.source_index;hits=np.flatnonzero(np.sign(beta[i+1:])==r.side)
        cross=i+1+int(hits[0]) if len(hits) else None
        if cross is None:assert not r.confirmation_observed
        else:
            assert r.confirmation_source_index==cross
            if cross+1<n:assert r.confirmation_execution_index==cross+1 and r.lead_bars==cross-i
    summary=pd.read_csv(p/'event_summary.csv')
    for side in ['long','short']:
        z=summary[(summary.direction==side)&(summary.period=='overall')].set_index('group')
        assert z.loc['all','events']==z.loc['first','events']+z.loc['later','events']
        for group in ['all','first','later']:
            a=summary[(summary.direction==side)&(summary.group==group)]
            assert a.loc[a.period=='overall','events'].iloc[0]==a.loc[a.period!='overall','events'].sum()
    html=(p/'event_explorer.html').read_text(encoding='utf-8')
    data=json.loads(html.split('<script id="data" type="application/json">')[1].split('</script>')[0])
    assert len(data['events'])==len(e) and len(data['raw']['open'])==n
    for r in e.itertuples():
        closed=r.confirmation_executable;j=r.entry_index;k=int(r.endpoint_index)
        raw=data['raw'];entry=raw['open'][j]*(1+r.side*.0002);qty=2000/entry
        end=raw['open'][k] if closed else raw['close'][k];stop=k if closed else k+1
        low=min(min(raw['low'][j:stop]),end,entry);high=max(max(raw['high'][j:stop]),end,entry)
        adverse=entry-low if r.side==1 else high-entry
        np.testing.assert_allclose(r.mae_usdt,qty*adverse,rtol=0,atol=1e-6)
    print('PASS hashes, all K5 regressions/events, exact next-open timing, first/later/year counts, censoring, confirmation search, costs and every MAE path.')

if __name__=='__main__':verify()
