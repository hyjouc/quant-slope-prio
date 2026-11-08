"""Build an offline raw-market/trade explorer; never run or alter a strategy."""
from pathlib import Path
import sys,json,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.data_loader import load_training,sha256
from src.slope_estimators import estimate

OUT=ROOT/'experiments/s1_3_recent_weighted_slope'
TEMPLATE=ROOT/'tools/s1_3_explorer_template.html'

def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,(float,np.floating)) and not math.isfinite(value):return None
    if isinstance(value,np.generic):return value.item()
    return value

def hourly_reference(d):
    """Hourly N32 OLS, available only after each UTC hour closes.

    Return per-15m-equivalent units for the existing chart's internal coordinate;
    chart/hover convert all series to bp/hour. No signal or trade is generated.
    """
    hourly=d.set_index('date').resample('1h',closed='left',label='left',origin='start_day').agg(
        close=('close','last'),count=('close','size'))
    hourly=hourly[hourly['count']==4]
    beta=estimate(hourly.close,32,'B_ols')
    available=pd.Series(beta/4,index=hourly.index+pd.Timedelta(hours=1))
    closes=pd.DatetimeIndex(d.date)+pd.Timedelta(minutes=15)
    return available.reindex(closes,method='ffill').to_numpy()

def build():
    protocol=json.loads((ROOT/'config/s1_3_recent_weighted_slope.json').read_text())
    d,meta=load_training(protocol['data_path'])
    slopes=pd.read_csv(OUT/'slope_signal_paths.csv');eq=pd.read_csv(OUT/'equity_curves.csv')
    t=pd.read_csv(OUT/'trades_all.csv');e=pd.read_csv(OUT/'executions_all.csv')
    models=['OLS','H64','H32','H16','H8']
    dates=pd.DatetimeIndex(d.date)
    assert len(d)==70176 and len(eq)==70080
    np.testing.assert_array_equal(pd.to_datetime(slopes.date,utc=True),d.date)
    np.testing.assert_array_equal(pd.to_datetime(eq.date,utc=True),d.date.iloc[96:])
    stamp=lambda s:int(pd.Timestamp(s).value//10**9)
    index=lambda s:int(dates.get_indexer([pd.Timestamp(s)])[0])
    data=dict(first=stamp(d.date.iloc[0]),step=900,performance=96,models=models,
        referenceSlope=hourly_reference(d).tolist(),
        raw={c:d[c].tolist() for c in ['open','high','low','close','volume']},
        slopes={name:slopes[name+'_slope'].tolist() for name in models},
        signals={name:slopes[name+'_signal'].tolist() for name in models},
        equity={name:[None]*96+eq[name].tolist() for name in models},
        summary=pd.read_csv(OUT/'parameter_comparison.csv').to_dict('records'),
        annual=pd.read_csv(OUT/'annual_comparison.csv').to_dict('records'),
        source=meta,trades={},executions={},duration=[])
    for name in models:
        g=t[t.model==name]
        data['trades'][name]=[dict(id=i,entry=index(r.entry_time),exit=index(r.exit_time) if r.status=='closed' else None,
            side=1 if r.direction=='long' else -1,entryPrice=r.entry_price,
            exitPrice=r.exit_price if r.status=='closed' else None,pnl=r.net_pnl,hours=r.holding_hours,
            fee=r.entry_fee+r.exit_fee,slip=r.entry_slippage_usdt+r.exit_slippage_usdt,
            closed=r.status=='closed') for i,r in enumerate(g.itertuples())]
        data['executions'][name]=[dict(index=index(r.time),action=r.action,side=1 if r.direction=='long' else -1,
            price=r.fill_price,fee=r.fee,slip=r.slippage_usdt,zero=bool(r.zero_volume)) for r in e[e.model==name].itertuples()]
        z=g[g.status=='closed']
        for label,lo,hi in [('＜1小时',0,1),('1–4小时',1,4),('4–16小时',4,16),('≥16小时',16,np.inf)]:
            b=z[(z.holding_hours>=lo)&(z.holding_hours<hi)]
            data['duration'].append(dict(model=name,group=label,count=len(b),net=b.net_pnl.sum(),
                win_rate=(b.net_pnl>0).mean(),share=len(b)/len(z)))
    payload=json.dumps(clean(data),ensure_ascii=False,separators=(',',':'),allow_nan=False)
    template=TEMPLATE.read_text(encoding='utf-8')
    assert template.count('__DATA__')==1
    target=OUT/'market_explorer.html'
    target.write_text(template.replace('__DATA__',payload),encoding='utf-8')
    evidence=dict(html=str(target.relative_to(ROOT)).replace('\\','/'),html_sha256=sha256(target),
        raw_sha256=meta['sha256'],raw_rows=len(d),start=str(d.date.iloc[0]),end=str(d.date.iloc[-1]),
        data_columns=list(data['raw']),models=models,raw_prices='unrounded original training OHLCV',
        source_hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in [TEMPLATE,Path(__file__),OUT/'slope_signal_paths.csv',OUT/'trades_all.csv',OUT/'executions_all.csv',OUT/'equity_curves.csv']},
        strategy_rerun=False,holdout_used=False,display_only_aggregation='pixel groups of OHLCV / min-max derived-series envelopes; hover always raw bar',
        slope_display_unit='bp/hour: original 15m slope * 40000; hourly reference internal value = hourly slope / 4, display * 40000; no signal scaling',
        reference_slope='OLS log-close N32 on complete UTC 1h candles (four contiguous raw bars each), trailing thirty-two hourly closes; available only at hour close; held until next hour close; display only, no new backtest',
        raw_and_derived_separated=True)
    (OUT/'explorer_provenance.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False),encoding='utf-8')
    manifest=json.loads((OUT/'artifact_manifest.json').read_text())
    for p in OUT.rglob('*'):
        if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts:
            manifest['artifacts'][str(p.relative_to(OUT)).replace('\\','/')]=dict(sha256=sha256(p),bytes=p.stat().st_size)
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Built {target} | {target.stat().st_size:,} bytes | {len(d)} raw training bars | no backtest executed')

if __name__=='__main__':build()
