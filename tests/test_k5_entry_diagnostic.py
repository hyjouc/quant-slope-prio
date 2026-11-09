import numpy as np
import pandas as pd
from src.k5_entry_diagnostic import slope_change,event_flags,cycle_ids,interval_outcome,build_events

def bars(n):
    return pd.DataFrame(dict(date=pd.date_range('2020-01-01',periods=n,freq='15min',tz='UTC'),
        open=np.full(n,100.),close=np.full(n,100.),low=np.full(n,99.),high=np.full(n,101.),volume=np.ones(n)))

def test_k5_formula_prefix_future_and_scale():
    b=np.random.default_rng(5).normal(size=300);g=slope_change(b)
    assert np.isnan(g[:4]).all()
    x=np.arange(5)
    for i in range(4,300):
        expected=np.linalg.lstsq(np.column_stack([np.ones(5),x]),b[i-4:i+1],rcond=None)[0][1]
        np.testing.assert_allclose(g[i],expected,atol=1e-15)
    np.testing.assert_array_equal(g[:150],slope_change(b[:150]))
    b[150:]+=100;np.testing.assert_array_equal(g[:150],slope_change(b)[:150])

def test_cross_exact_zero_invalid_and_symmetry():
    b=np.array([-1.,-1,-1,-1,1,1,1,np.nan])
    g=np.array([np.nan,-1,0,1,1,0,-1,1.])
    np.testing.assert_array_equal(event_flags(b,g),[0,0,0,1,0,0,-1,0])
    np.testing.assert_array_equal(event_flags(-b,-g),-event_flags(b,g))
    np.testing.assert_array_equal(cycle_ids([-1.,0,-2,1,0,-1]),[0,0,0,1,1,2])

def test_excursion_exit_candle_exclusion_costs_and_censored():
    d=bars(8);d.loc[5,['low','high']]=[1,1000];d.loc[5,'open']=102
    r=interval_outcome(d,2,5,1,True)
    np.testing.assert_allclose(r['mfe_usdt'],r['quantity']*(102-r['entry_fill']))
    np.testing.assert_allclose(r['mae_usdt'],r['quantity']*(r['entry_fill']-99))
    np.testing.assert_allclose(r['gross_usdt']-r['fees_usdt']-r['slippage_usdt'],r['net_usdt'],atol=1e-10)
    closed=interval_outcome(d,2,7,-1,False)
    assert closed['fees_usdt']==1 and closed['mfe_pct']>90

def test_repeated_events_first_warmup_confirm_and_censor():
    d=bars(16);b=np.array([-1.]*9+[1.]*7)
    g=np.array([-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1.])
    e,_,_=build_events(d,b,g,first=4)
    assert e.iloc[0].ordinal_in_cycle==2 and not e.iloc[0].is_first
    assert e.iloc[0].entry_index==4 and e.iloc[0].confirmation_execution_index==10
    assert e.iloc[-1].censored and e.iloc[-1].status=='censored_marked_at_training_end'
    assert (e.entry_time==e.source_bar+pd.Timedelta(minutes=15)).all()

def test_prefix_events_causal_not_outcomes():
    b=np.random.default_rng(9).normal(size=100);g=slope_change(b)
    np.testing.assert_array_equal(event_flags(b,g)[:60],event_flags(b[:60],slope_change(b[:60])))
