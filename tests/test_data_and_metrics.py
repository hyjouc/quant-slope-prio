import numpy as np
import pytest
from src.data_loader import load_training
from src.metrics import summarize
from src.backtester import simulate
from test_execution import fixture

def test_hash_mismatch_stops_before_deserialization(tmp_path):
    bad=tmp_path/'bad.feather';bad.write_bytes(b'not approved')
    with pytest.raises(ValueError,match='SHA256 mismatch'):load_training(bad)

def test_accounting_year_boundary_and_metrics_reconcile():
    d,cfg=fixture(prices=[100,101,103,105,106,102,100,99],start='2020-12-31 23:00')
    t,c,e=simulate(d,[1,1,1,1,-1,-1,-1,-1],'A_endpoint',8,cfg)
    whole=summarize(c,t,e,cfg)
    y20=summarize(c,t,e,cfg,'2020-01-01T00:00:00Z','2021-01-01T00:00:00Z','2020')
    y21=summarize(c,t,e,cfg,'2021-01-01T00:00:00Z','2022-01-01T00:00:00Z','2021')
    np.testing.assert_allclose(y20['net_profit']+y21['net_profit'],whole['net_profit'])
    np.testing.assert_allclose(whole['long_net_profit']+whole['short_net_profit'],whole['net_profit'])
    np.testing.assert_allclose(t.net_pnl.sum(),whole['net_profit'])
    np.testing.assert_allclose(whole['gross_before_cost']-whole['fees']-whole['slippage_usdt'],whole['net_profit'])
    assert whole['trade_count']==1 and y20['trade_count']==0 and y21['trade_count']==1
    assert whole['funded_execution_feasible']
