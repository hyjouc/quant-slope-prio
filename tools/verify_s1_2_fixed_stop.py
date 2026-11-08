"""Delivery and frozen-source audit; no strategy search or holdout calculation."""
from pathlib import Path
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
from src.data_loader import sha256

def main():
    p=ROOT/'experiments/s1_2_fixed_stop';manifest=json.loads((p/'artifact_manifest.json').read_text())
    for name,item in manifest.items():
        q=ROOT/name;assert q.stat().st_size==item['bytes'] and sha256(q)==item['sha256'],name
    meta=json.loads((p/'diagnostic_metadata.json').read_text())
    for name,digest in meta['frozen_files_sha256'].items():assert sha256(ROOT/name)==digest,name
    d=pd.read_csv(p/'parameter_comparison.csv');t=pd.read_csv(p/'trades_all.csv');e=pd.read_csv(p/'executions_all.csv')
    assert d.scenario.tolist()==['no_stop','L01pct','L02pct','L03pct','L04pct','L05pct','L06pct']
    assert len(pd.read_csv(p/'annual_comparison.csv'))==14 and len(pd.read_csv(p/'monthly_comparison.csv'))==168
    curve=pd.read_csv(p/'equity_curves.csv');dd=pd.read_csv(p/'drawdown_curves.csv')
    assert len(curve)==len(dd)==70080 and pd.to_datetime(curve.date,utc=True).max()<pd.Timestamp('2022-01-01',tz='UTC')
    for field in ['entry_time','exit_time']:
        assert (pd.to_datetime(t[field],utc=True).dropna()<pd.Timestamp('2022-01-01',tz='UTC')).all()
    q=pd.to_datetime(e.time,utc=True);assert (q<pd.Timestamp('2022-01-01',tz='UTC')).all()
    np.testing.assert_allclose(e.loc[e.action=='open','notional'],2000,atol=1e-7,rtol=0)
    top=pd.read_csv(p/'original_top10_winner_audit.csv');assert len(top)==70
    isolated=top.groupby('scenario').original_entry_time.apply(lambda x:tuple(sorted(x)))
    assert all(x==isolated.iloc[0] for x in isolated)
    touch=e[(e.action=='close')&(e.reason=='stop_touch')]
    assert touch.time_precision.eq('within_15m_unknown').all()
    closed=t[t.status=='closed']
    unknown=closed.exit_time_precision.eq('within_15m_unknown')
    np.testing.assert_allclose(closed.holding_hours_upper-closed.holding_hours,np.where(unknown,.25,0.),atol=1e-9,rtol=0)
    for side,sign in [('long',1),('short',-1)]:
        g=t[(t.direction==side)&t.L.notna()]
        np.testing.assert_allclose(g.fixed_stop_price,g.entry_price*(1-sign*g.L),atol=1e-7,rtol=0)
        g=e[(e.action=='close')&(e.direction==side)]
        np.testing.assert_allclose(g.fill_price,g.raw_price*(1-sign*.0002),atol=1e-7,rtol=0)
    annual=pd.read_csv(p/'annual_comparison.csv');monthly=pd.read_csv(p/'monthly_comparison.csv')
    episode=pd.read_csv(p/'all_original_episode_audit.csv')
    baseline_open=t[(t.scenario=='no_stop')&(t.status!='closed')]
    assert len(baseline_open)==1
    final_original_entry=pd.Timestamp(baseline_open.entry_time.iloc[0])
    for r in d.itertuples():
        trades=t[t.scenario==r.scenario];events=e[e.scenario==r.scenario];c=curve[r.scenario]
        np.testing.assert_allclose(trades.net_pnl.sum(),r.net_profit,atol=1e-6,rtol=0)
        np.testing.assert_allclose(c.iloc[-1]-10000,r.net_profit,atol=1e-6,rtol=0)
        np.testing.assert_allclose(events.fee.sum(),r.fees,atol=1e-6,rtol=0)
        np.testing.assert_allclose(annual[annual.scenario==r.scenario].net_profit.sum(),r.net_profit,atol=1e-6,rtol=0)
        np.testing.assert_allclose(monthly[monthly.scenario==r.scenario].net_profit.sum(),r.net_profit,atol=1e-6,rtol=0)
        z=top[top.scenario==r.scenario]
        np.testing.assert_allclose(z.episode_net_profit.sum(),r.original_top10_episode_net_profit,atol=1e-6,rtol=0)
        assert z.original_entry_stopped_early.sum()==r.original_top10_entry_early_stop_count
        assert len(z)==10
        ep=episode[episode.scenario==r.scenario]
        assert len(ep)==1147 and ep.original_entry_time.is_unique
        final_actual=trades[pd.to_datetime(trades.entry_time,utc=True)>=final_original_entry]
        np.testing.assert_allclose(ep.episode_net_profit.sum()+final_actual.net_pnl.sum(),r.net_profit,atol=1e-6,rtol=0)
        if r.scenario!='no_stop':
            stops=trades[trades.exit_reason.isin(['stop_gap','stop_touch'])]
            for stop in stops.itertuples():
                nxt=trades[trades.entry_time>=stop.exit_time]
                # Exclude same trade when entry=exit on a stop-touch entry candle.
                nxt=nxt[nxt.trade_sequence>stop.trade_sequence]
                if len(nxt):assert pd.Timestamp(nxt.entry_time.iloc[0])>=pd.Timestamp(stop.exit_time)+pd.Timedelta(minutes=15)
    assert len(list((p/'figures').glob('*.png')))==6
    print(f'PASS: seven scenarios, 14 annual/168 monthly rows, 70 fixed-original-top10 rows, all ledger/fee/curve reconciliations; frozen S0/S1.1 unchanged; {len(manifest)} hashes; six figures.')

if __name__=='__main__':main()
