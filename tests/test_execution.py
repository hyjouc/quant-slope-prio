import numpy as np
import pandas as pd
import pytest
from src.backtester import simulate
from src.signal_generator import generate_signals

def fixture(prices=None,start='2020-01-01',count=8):
    dates=pd.date_range(start,periods=count,freq='15min',tz='UTC')
    op=np.array(prices if prices is not None else [100.]*count,dtype=float)
    d=pd.DataFrame(dict(date=dates,open=op,high=op+1,low=op-1,close=op,volume=1.))
    cfg=dict(performance_start=str(dates[1]),performance_end_exclusive=str(dates[-1]+pd.Timedelta(minutes=15)),
        initial_account=10000.,margin_budget=1000.,notional=2000.,fee=.0005,slippage=.0002)
    return d,cfg

def test_sign_zero_carries_invalid_no_order():
    np.testing.assert_equal(generate_signals([0.,1.,0.,np.nan,0.,-1.,0.]),[0.,1.,1.,np.nan,1.,-1.,-1.])

def test_reversal_next_open_and_double_fees_and_slippage():
    d,cfg=fixture([100,110,120,115,108,109,110,112])
    sig=generate_signals([1,1,-1,-1,-1,-1,-1,-1])
    t,c,e=simulate(d,sig,'A_endpoint',8,cfg)
    assert t.iloc[0].entry_time==d.date.iloc[1]
    assert t.iloc[0].exit_time==d.date.iloc[3]
    assert list(e.action)==['open','close','open']
    assert e.time.iloc[1]==e.time.iloc[2]==d.date.iloc[3]
    ep=110*1.0002;xp=115*.9998;qty=2000/ep
    expected=qty*(xp-ep)-1-qty*xp*.0005
    np.testing.assert_allclose(t.iloc[0].net_pnl,expected)
    assert e.fill_price.iloc[2]==115*.9998 # short entry sells at adverse lower price
    assert e.fee.iloc[0]==e.fee.iloc[2]==1.
    np.testing.assert_allclose(c.equity.iloc[-1]-10000,t.net_pnl.sum())
    np.testing.assert_allclose(t.gross_before_cost-t.entry_slippage_usdt-t.exit_slippage_usdt-t.entry_fee-t.exit_fee,t.net_pnl,atol=1e-10)

def test_short_cover_adverse_slippage():
    d,cfg=fixture();sig=np.array([-1,-1,1,1,1,1,1,1.])
    t,c,e=simulate(d,sig,'B_ols',8,cfg)
    assert e.fill_price.iloc[0]==100*.9998
    assert e.fill_price.iloc[1]==100*1.0002
    assert e.fill_price.iloc[2]==100*1.0002
    assert t.net_pnl.iloc[0]<-2.

def test_invalid_preserves_position_zero_preserves_signal():
    d,cfg=fixture();sig=generate_signals([1,0,np.nan,0,1,0,0,0])
    t,c,e=simulate(d,sig,'A_endpoint',8,cfg)
    assert len(e)==1 and t.status.iloc[0]=='open_marked_at_training_end'
    assert c.position.eq(1).all()

def test_final_signal_unexecuted_open_position_marked_no_exit_fee():
    d,cfg=fixture();sig=np.ones(8);sig[-1]=-1
    t,c,e=simulate(d,sig,'A_endpoint',8,cfg)
    assert len(e)==1 and len(t)==1
    assert pd.isna(t.exit_time.iloc[0]) and t.exit_fee.iloc[0]==0.
    np.testing.assert_allclose(c.equity.iloc[-1]-10000,t.net_pnl.iloc[0])

def test_year_boundary_no_flatten_and_zero_volume_audit():
    d,cfg=fixture(start='2020-12-31 23:00');d.loc[1,'volume']=0
    t,c,e=simulate(d,np.ones(8),'A_endpoint',8,cfg)
    assert len(e)==1 and e.zero_volume.iloc[0]
    assert c.position.eq(1).all()
    assert t.holding_hours.iloc[0]==(len(d)-1)*.25

def test_future_close_high_low_cannot_affect_current_open_fill():
    d,cfg=fixture();sig=generate_signals([1,1,-1,-1,1,1,-1,-1])
    t,c,e=simulate(d,sig,'B_ols',8,cfg)
    changed=d.copy();changed.loc[3:,['close','high','low']]*=2
    _,_,e2=simulate(changed,sig,'B_ols',8,cfg)
    pd.testing.assert_frame_equal(e,e2)

def test_refuse_gap():
    d,cfg=fixture();d.loc[4:,'date']+=pd.Timedelta(minutes=15)
    with pytest.raises(ValueError,match='missing'):simulate(d,np.ones(8),'A_endpoint',8,cfg)
