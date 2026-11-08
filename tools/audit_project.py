"""Read-only source/data audit. No orders, trade execution, PnL, or parameter search."""
from pathlib import Path
import sys,ast,json,platform
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.data_loader import load_data,sha256
from src.kalman_filter import kalman_slope
from src.slope_signal import standardize

def reference_function(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_features')
    ns={'np':np,'pd':pd}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    return ns['build_features']

def main():
    cfg=json.loads((ROOT/'config/s0_config.json').read_text(encoding='utf-8-sig'))
    assert cfg['stage']=='stage1_audit_only' and cfg['backtest_authorized'] is False
    paths=[cfg['data_path'],cfg['reference_path'],cfg['execution_reference_path']]
    before={str(p):sha256(p) for p in paths}
    if before[cfg['reference_path']]!='39b5ee58b32a75c9a775e57fd5dba9bdb67d426393c9bcd90642276c11502fd9':
        raise ValueError('V18 source differs from audited remote commit')
    if before[cfg['execution_reference_path']]!='4b805f4068a168d66722a66821c90f9f497ea012942f5aaf6b7c620a2e56c782':
        raise ValueError('Execution reference differs from audited remote commit')
    d,meta=load_data(cfg['data_path']);v=kalman_slope(d.close);s=standardize(v)
    original=reference_function(cfg['reference_path'])(d)
    np.testing.assert_array_equal(v,original.slope)
    np.testing.assert_allclose(s,original.slope_scaled,rtol=0,atol=0,equal_nan=True)
    # Recalculate actual year-boundary prefixes; no resets or future observations.
    boundaries=[]
    for year in [2021,2022,2023]:
        n=int(d.date.searchsorted(pd.Timestamp(f'{year}-01-01',tz='UTC')))+2
        prefix=kalman_slope(d.close.iloc[:n])
        np.testing.assert_array_equal(prefix,v[:n])
        np.testing.assert_allclose(standardize(prefix),s[:n],rtol=0,atol=0,equal_nan=True)
        boundaries.append(dict(year=year,prefix_bars=n,exact=True))
    after={str(p):sha256(p) for p in paths};assert before==after
    annual=d.groupby(d.date.dt.year).agg(bars=('date','size'),first=('date','min'),last=('date','max'),zero_volume=('volume',lambda x:int(x.eq(0).sum())))
    annual.to_csv(ROOT/'audit/data_annual.csv')
    d.loc[d.volume.eq(0),['date','open','high','low','close','volume']].to_csv(ROOT/'audit/zero_volume_bars.csv',index=False)
    output=dict(project='quant-slope-prio',stage='stage1_audit_only',data=meta,
        audited_file_hashes_before=before,audited_file_hashes_after=after,
        checks=dict(v18_slope_exact_all_bars=True,v18_standardization_exact_all_bars=True,
                    year_boundary_prefixes=boundaries,sources_unchanged=True,parameter_search_run=False,
                    backtest_results_produced=False,post2023_market_data_read=False),
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        zero_volume_policy='retain observed candles; no forward-fill/drop/filter; original exchange provenance not independently re-downloaded',
        limitations=['not independent untouched OOS','audit does not prove exchange authenticity','97 zero-volume candles require research-lead review',
                    'legacy executable drafts from superseded full-experiment scope excluded; no ranking or performance conclusion retained'])
    (ROOT/'audit/audit_metadata.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PASS: exact V18 slope/standardization on all 140256 bars; three year-boundary prefixes; source hashes unchanged. No backtest.')

if __name__=='__main__':main()
