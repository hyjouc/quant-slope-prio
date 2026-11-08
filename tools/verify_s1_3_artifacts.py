"""Read-only independent checks on delivered S1.3 CSVs, hashes and frozen files."""
from pathlib import Path
import sys,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.data_loader import sha256

def verify():
    folder=ROOT/'experiments/s1_3_recent_weighted_slope'
    manifest=json.loads((folder/'artifact_manifest.json').read_text())
    for p,r in manifest['artifacts'].items():
        assert sha256(folder/p)==r['sha256'],p
        assert (folder/p).stat().st_size==r['bytes'],p
    required=['s1_3_report.md','parameter_comparison.csv','annual_comparison.csv','monthly_comparison.csv',
        'signal_response_analysis.csv','signal_flip_comparison.csv','long_short_comparison.csv','trade_level_comparison.csv',
        'profit_concentration.csv','synthetic_response_tests.csv','equity_curves.csv','drawdown_curves.csv',
        'diagnostic_metadata.json','run.log','artifact_manifest.json']
    assert all((folder/p).is_file() for p in required)
    meta=json.loads((folder/'diagnostic_metadata.json').read_text())
    for p,h in (meta['frozen_base_file_sha256']|meta['active_code_sha256']).items():assert sha256(ROOT/p)==h,p
    assert meta['selected_H'] is None and not meta['validation_2022_2023_run'] and not meta['read_2024_plus']
    table=pd.read_csv(folder/'parameter_comparison.csv');annual=pd.read_csv(folder/'annual_comparison.csv')
    monthly=pd.read_csv(folder/'monthly_comparison.csv');directions=pd.read_csv(folder/'long_short_comparison.csv')
    trades=pd.read_csv(folder/'trades_all.csv');events=pd.read_csv(folder/'executions_all.csv')
    eq=pd.read_csv(folder/'equity_curves.csv');dd=pd.read_csv(folder/'drawdown_curves.csv')
    assert table.model.tolist()==['OLS','H64','H32','H16','H8']
    assert len(eq)==70080 and len(monthly)==120 and len(annual)==10 and len(directions)==30
    assert len(pd.read_csv(folder/'synthetic_response_tests.csv'))==50
    assert len(list((folder/'figures').glob('*.png')))==9
    dates=pd.to_datetime(eq.date,utc=True);assert dates.iloc[0]==pd.Timestamp('2020-01-02',tz='UTC')
    assert dates.iloc[-1]<pd.Timestamp('2022-01-01',tz='UTC')
    for r in table.itertuples():
        t=trades[trades.model==r.model];e=events[events.model==r.model]
        np.testing.assert_allclose(t.net_pnl.sum(),r.net_profit,atol=1e-6,rtol=0)
        np.testing.assert_allclose(eq[r.model].iloc[-1]-10000,r.net_profit,atol=1e-6,rtol=0)
        peak=np.maximum.accumulate(np.r_[10000,eq[r.model]])[1:]
        np.testing.assert_allclose(peak-eq[r.model],dd[r.model],atol=1e-6,rtol=0)
        np.testing.assert_allclose(dd[r.model].max(),r.max_drawdown_usdt,atol=1e-6,rtol=0)
        assert (t.status=='closed').sum()==r.trade_count
        np.testing.assert_allclose(e.fee.sum(),r.fees,atol=1e-6,rtol=0)
        np.testing.assert_allclose(e.slippage_usdt.sum(),r.slippage_usdt,atol=1e-6,rtol=0)
        np.testing.assert_allclose(r.long_net_profit+r.short_net_profit,r.net_profit,atol=1e-6,rtol=0)
        for frame in [annual,monthly]:
            for field in ['net_profit','long_net_profit','short_net_profit','fees','slippage_usdt','trade_count']:
                np.testing.assert_allclose(frame.loc[frame.model==r.model,field].sum(),getattr(r,field),atol=1e-6,rtol=0)
        for direction in ['long','short']:
            z=directions[(directions.model==r.model)&(directions.period=='overall')&(directions.direction==direction)].iloc[0]
            np.testing.assert_allclose(z.net_profit,getattr(r,direction+'_net_profit'),atol=1e-6,rtol=0)
            np.testing.assert_allclose(t.loc[t.direction==direction,'net_pnl'].sum(),z.net_profit,atol=1e-6,rtol=0)
        assert (pd.to_datetime(e.time,utc=True)==pd.to_datetime(e.source_signal_bar,utc=True)+pd.Timedelta(minutes=15)).all()
    cmp=pd.read_csv(folder/'trade_level_comparison.csv');assert len(cmp)==1147*5
    np.testing.assert_allclose(cmp.loc[cmp.model=='OLS','interval_pnl_change_vs_original'],0,atol=1e-8)
    m=pd.read_csv(folder/'signal_flip_comparison.csv')
    for name,g in m.groupby('model'):
        matched=g.dropna(subset=['ols_event_id','ewls_event_id'])
        assert matched.ols_event_id.is_unique and matched.ewls_event_id.is_unique
        assert g.ols_event_id.dropna().is_unique and g.ewls_event_id.dropna().is_unique
    print('PASS: manifest, required files, nine figures, frozen/active hashes, five candidates, accounting/directions/years/months, next-open timing and event uniqueness.')

if __name__=='__main__':verify()
