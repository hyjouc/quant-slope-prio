"""Validate delivered files and public source isolation without rerunning experiments."""
from pathlib import Path
import sys,json,ast,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.data_loader import sha256

def main():
    out=ROOT/'experiments/s0_from_scratch_slope'
    required=['s0_report.md','estimator_comparison.csv','parameter_comparison.csv','annual_comparison.csv',
        'monthly_comparison.csv','direction_comparison.csv','trades_all.csv','equity_curves.csv',
        'drawdown_curves.csv','parameter_stability.csv','zero_volume_execution_audit.csv',
        'diagnostic_metadata.json','run.log','artifact_manifest.json']
    assert all((out/name).is_file() for name in required)
    manifest=json.loads((out/'artifact_manifest.json').read_text(encoding='utf-8'))
    for path,item in manifest.items():
        p=ROOT/path
        assert p.stat().st_size==item['bytes'] and sha256(p)==item['sha256'],path
    table=pd.read_csv(out/'parameter_comparison.csv');annual=pd.read_csv(out/'annual_comparison.csv');monthly=pd.read_csv(out/'monthly_comparison.csv')
    assert len(table)==10 and len(annual)==20 and len(monthly)==240
    assert set(annual.year)=={2020,2021}
    assert set(table.method)=={'A_endpoint','B_ols'} and set(table.N)=={8,16,32,64,96}
    eq=pd.read_csv(out/'equity_curves.csv');dd=pd.read_csv(out/'drawdown_curves.csv')
    assert len(eq)==70080 and len(dd)==70080
    assert pd.to_datetime(eq.date,utc=True).max()<pd.Timestamp('2022-01-01',tz='UTC')
    events=pd.read_csv(out/'executions_all.csv');zero=pd.read_csv(out/'zero_volume_execution_audit.csv')
    pd.testing.assert_frame_equal(events[events.zero_volume].reset_index(drop=True),zero,check_dtype=False)
    times=pd.to_datetime(events.time,utc=True);source=pd.to_datetime(events.source_signal_bar,utc=True)
    assert (times-source==pd.Timedelta(minutes=15)).all()
    assert times.max()<pd.Timestamp('2022-01-01',tz='UTC')
    np.testing.assert_allclose(events.loc[events.action=='open','notional'],2000,atol=1e-7,rtol=0)
    for row in table.itertuples():
        np.testing.assert_allclose(eq[row.candidate].iloc[-1]-10000,row.net_profit,atol=1e-6)
        np.testing.assert_allclose(annual.loc[annual.candidate==row.candidate,'net_profit'].sum(),row.net_profit,atol=1e-6)
        np.testing.assert_allclose(monthly.loc[monthly.candidate==row.candidate,'net_profit'].sum(),row.net_profit,atol=1e-6)
    # All active Python imports must stay outside the archived tree.
    for folder in ['src','tests','tools','experiments/s0_from_scratch_slope']:
        for p in (ROOT/folder).glob('*.py'):
            tree=ast.parse(p.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):names=[n.name for n in node.names]
                elif isinstance(node,ast.ImportFrom):names=[node.module or '']
                else:continue
                assert not any(any(term in name.lower() for term in ['kalman','legacy','archive','ridge','freqtrade']) for name in names),p
    assert len(list((out/'figures').glob('*.png')))==7
    print(f'PASS: 14 required outputs, 7 figures, {len(manifest)} hashes; ten candidates, 20 years/240 months; training-only fill timestamps; zero-volume audit exact; no archived imports.')

if __name__=='__main__':main()
