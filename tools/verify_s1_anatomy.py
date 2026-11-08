"""Independent delivery validation; no original S0 files or trading rules modified."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.data_loader import sha256

def main():
    p=ROOT/'experiments/s1_trade_anatomy'
    manifest=json.loads((p/'artifact_manifest.json').read_text())
    for path,item in manifest.items():
        q=ROOT/path
        assert q.stat().st_size==item['bytes'] and sha256(q)==item['sha256'],path
    for path,item in json.loads((ROOT/'experiments/s0_from_scratch_slope/artifact_manifest.json').read_text()).items():
        assert sha256(ROOT/path)==item['sha256'],path
    required=['s1_trade_anatomy_report.md','trade_level_metrics.csv','mae_mfe_analysis.csv','holding_duration_analysis.csv',
        'profit_giveback_analysis.csv','long_short_comparison.csv','loss_cluster_analysis.csv','run.log','diagnostic_metadata.json']
    assert all((p/x).is_file() for x in required)
    t=pd.read_csv(p/'trade_level_metrics.csv');c=t[t.status=='closed'];o=t[t.status!='closed']
    assert c.groupby('candidate').size().to_dict()=={'B_ols_N64':1147,'B_ols_N96':751}
    assert len(o)==2 and not o.trade_id.isin(c.trade_id).any()
    for field in ['entry_time','exit_time']:
        assert (pd.to_datetime(t[field],utc=True).dropna()<pd.Timestamp('2022-01-01',tz='UTC')).all()
    np.testing.assert_allclose(t.path_bar_count*.25,t.holding_hours,rtol=0,atol=1e-9)
    np.testing.assert_allclose(c.giveback_execution_usdt-c.giveback_price_usdt,c.exit_slippage_usdt,atol=1e-7,rtol=0)
    h=pd.read_csv(p/'holding_duration_analysis.csv')
    side=pd.read_csv(p/'long_short_comparison.csv')
    runs=pd.read_csv(p/'loss_streaks.csv');risk=pd.read_csv(p/'loss_cluster_analysis.csv')
    period=pd.read_csv(p/'period_pnl.csv');curve=pd.read_csv(p/'anatomy_equity_curves.csv')
    source=pd.read_csv(ROOT/'experiments/s0_from_scratch_slope/parameter_comparison.csv').set_index('candidate')
    for candidate,g in c.groupby('candidate'):
        z=h[(h.candidate==candidate)&(h.direction=='all')];assert z['count'].sum()==len(g)
        np.testing.assert_allclose(z.net_profit.sum(),g.net_pnl.sum(),atol=1e-6,rtol=0)
        z=side[(side.candidate==candidate)&(side.direction!='all')];assert z.trade_count.sum()==len(g)
        np.testing.assert_allclose(z.closed_net_profit.sum(),g.net_pnl.sum(),atol=1e-6,rtol=0)
        z=runs[(runs.candidate==candidate)&(runs.direction=='all')]
        assert z.losses.sum()==g.net_pnl.lt(0).sum()
        for frequency in ['day','week','month']:
            z=period[(period.candidate==candidate)&(period.direction=='all')&(period.frequency==frequency)]
            np.testing.assert_allclose(z.net_profit.sum(),source.loc[candidate,'net_profit'],atol=1e-6,rtol=0)
        z=curve[(curve.candidate==candidate)&(curve.direction=='all')]
        assert len(z)==70080
        np.testing.assert_allclose(z.equity.iloc[-1]-10000,source.loc[candidate,'net_profit'],atol=1e-6,rtol=0)
        np.testing.assert_allclose(risk.query('candidate==@candidate and direction=="all"').max_drawdown_usdt.iloc[0],source.loc[candidate,'max_drawdown_usdt'],atol=1e-6,rtol=0)
    assert len(list((p/'figures').glob('*.png')))==6
    print(f'PASS: {len(c)} closed + {len(o)} separate open positions; six figures; all S1 hashes and original S0 hashes; duration/direction/streak/period/equity reconciliation.')

if __name__=='__main__':main()
