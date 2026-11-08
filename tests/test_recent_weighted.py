import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.recent_weighted_slope import weights,ewls_log,estimate_ewls
from src.recent_weighted_analysis import match_flips,interval_pnl,original_trade_comparison,synthetic_experiments
from src.slope_estimators import estimate
from src.signal_generator import generate_signals
from src.backtester import simulate
from test_execution import fixture

@pytest.mark.parametrize('h',[None,64,32,16,8])
def test_formula_window_causality_translation_scale(h):
    p=np.exp(5+np.random.default_rng(73).normal(0,.02,250).cumsum())
    b=estimate_ewls(p,h);y=np.log(p);w=weights(h);x=np.arange(64)
    assert np.isnan(b[:63]).all()
    for i in [63,100,249]:
        # Independent weighted least squares, including fitted intercept.
        design=np.column_stack([np.ones(64),x])*np.sqrt(w[:,None])
        expected=np.linalg.lstsq(design,y[i-63:i+1]*np.sqrt(w),rcond=None)[0][1]
        np.testing.assert_allclose(b[i],expected,atol=1e-15,rtol=0)
    np.testing.assert_array_equal(b[:170],estimate_ewls(p[:170],h))
    altered=p.copy();altered[170:]*=np.linspace(1,20,80)
    np.testing.assert_array_equal(b[:170],estimate_ewls(altered,h)[:170])
    np.testing.assert_allclose(b,ewls_log(y+7,h),atol=1e-15,rtol=0)
    np.testing.assert_array_equal(np.sign(b),np.sign(estimate_ewls(p*13,h)))
    np.testing.assert_array_equal(b,estimate_ewls(p,h))

def test_infinity_exact_and_weights():
    p=np.exp(np.random.default_rng(72).normal(4,.3,1200))
    np.testing.assert_array_equal(estimate_ewls(p,None),estimate(p,64,'B_ols'))
    np.testing.assert_array_equal(estimate_ewls(p,np.inf),estimate(p,64,'B_ols'))
    for h in [8,16,32,64]:
        w=weights(h);assert w[-1]==1 and (np.diff(w)>0).all()
        if h<64:assert w[-1-h]==.5
        np.testing.assert_allclose(w[1:]/w[:-1],2**(1/h))
    for h in [0,-1,np.nan,-np.inf]:
        with pytest.raises(ValueError):weights(h)
    p[70]=0;assert np.isnan(estimate_ewls(p,16)[70:134]).all()

def test_sign_close_next_open_and_reconcile():
    d,cfg=fixture(count=170)
    d['close']=np.exp(4+np.r_[np.arange(85)*.002, .168-np.arange(85)*.004])
    b=estimate_ewls(d.close,16);sig=generate_signals(b)
    np.testing.assert_array_equal(generate_signals(np.array([np.nan,1.,0.,-1.,0.,np.nan])),[np.nan,1,1,-1,-1,np.nan])
    t,c,e=simulate(d,sig,'EWLS_H16',64,cfg)
    assert (e.time==e.source_signal_bar+pd.Timedelta(minutes=15)).all()
    assert e.time.iloc[0]==d.date.iloc[64]
    assert e.direction.iloc[0]=='long' and t.direction.iloc[1]=='short'
    np.testing.assert_allclose(t.net_pnl.sum(),c.equity.iloc[-1]-10000,atol=1e-9)
    np.testing.assert_allclose(e.fee.sum(),c.fees.iloc[-1])
    np.testing.assert_allclose(e.slippage_usdt.sum(),c.slippage_usdt.iloc[-1])
    pd.testing.assert_frame_equal(t,simulate(d,sig,'EWLS_H16',64,cfg)[0],check_exact=True)

def test_interval_endpoint_and_straddling():
    d,cfg=fixture(count=12);d['open']=np.arange(12)+100.;d['close']=d.open+.1
    sig=np.array([1,1,1,-1,-1,-1,1,1,1,-1,-1,-1.])
    t,c,e=simulate(d,sig,'B_ols',64,cfg)
    for r in t[t.status=='closed'].itertuples():
        p,side=interval_pnl(t,r.entry_time,r.exit_time,d)
        np.testing.assert_allclose(p,r.net_pnl,atol=1e-10)
        np.testing.assert_allclose(side[r.direction],p,atol=1e-10)
    a=d.date.iloc[2];b=d.date.iloc[3];row=t.iloc[0]
    p,_=interval_pnl(t,a,b,d)
    np.testing.assert_allclose(p,row.quantity*(d.open.iloc[3]-d.open.iloc[2]))
    cmp=original_trade_comparison(t,t,d)
    np.testing.assert_allclose(cmp.interval_pnl_change_vs_original,0,atol=1e-10)
    assert not cmp.first_covering_position_exited_early.any()

def test_flip_one_to_one_midpoint_and_unmatched():
    a=np.ones(40);a[10:20]=-1;a[30:]=-1
    b=np.ones(40);b[8:11]=-1;b[15:17]=-1;b[29:]=-1
    m=match_flips(a,b,0,40)
    matched=m.dropna(subset=['ols_event_id','ewls_event_id'])
    assert matched.ols_event_id.is_unique and matched.ewls_event_id.is_unique
    assert set(m.classification)>={'early','unmatched_ewls'}
    np.testing.assert_array_equal(m.ols_index.dropna().sort_values(),[10,20,30])

def test_synthetic_repeat_and_clean_delay():
    cfg=json.loads((Path(__file__).parents[1]/'config/s1_3_recent_weighted_slope.json').read_text())['synthetic']
    models=[('OLS',None),('H64',64),('H32',32),('H16',16),('H8',8)]
    a,p=synthetic_experiments(cfg,models);b,q=synthetic_experiments(cfg,models)
    pd.testing.assert_frame_equal(a,b,check_exact=True);pd.testing.assert_frame_equal(p,q,check_exact=True)
    clean=a[(a.family=='trend_reversal')&(a.noise_std==0)]
    assert (clean.response_delay_bars>=0).all()
    assert clean[clean.model=='H8'].response_delay_bars.max()<clean[clean.model=='OLS'].response_delay_bars.min()
