import numpy as np
import pandas as pd
import pytest
from src.fixed_stop_backtester import simulate_fixed_stop
from src.backtester import simulate,TRADE_COLUMNS,EXEC_COLUMNS
from test_execution import fixture

def setup(direction=1,count=8):
    d,cfg=fixture(count=count);d['high']=100.1;d['low']=99.9
    return d,cfg,np.full(count,float(direction))

@pytest.mark.parametrize('direction',[1,-1])
def test_no_stop_identical_s0(direction):
    d,cfg,sig=setup(direction);sig[3:]=-direction
    old=simulate(d,sig,'B_ols',64,cfg);new=simulate_fixed_stop(d,sig,cfg,None)
    pd.testing.assert_frame_equal(old[0],new[0][TRADE_COLUMNS])
    pd.testing.assert_frame_equal(old[1],new[1])
    pd.testing.assert_frame_equal(old[2],new[2][EXEC_COLUMNS])

@pytest.mark.parametrize('direction',[1,-1])
def test_price_stop_entry_fill_anchor_both_costs_and_next_open_reentry(direction):
    d,cfg,sig=setup(direction)
    ep=100*(1+direction*.0002);stop=ep*(1-direction*.01)
    d.loc[2,'low' if direction==1 else 'high']=stop-.1 if direction==1 else stop+.1
    t,c,e=simulate_fixed_stop(d,sig,cfg,.01)
    r=t.iloc[0];xp=stop*(1-direction*.0002);q=2000/ep
    assert r.exit_reason=='stop_touch' and r.exit_time==d.date.iloc[2]
    assert r.exit_time_precision=='within_15m_unknown'
    np.testing.assert_allclose(r.exit_raw_price,stop)
    np.testing.assert_allclose(r.net_pnl,direction*q*(xp-ep)-1-q*xp*.0005)
    assert t.entry_time.iloc[1]==d.date.iloc[3]
    assert r.holding_hours_upper-r.holding_hours==.25
    assert len(e[e.time==d.date.iloc[2]])==1 # no same-bar reentry

@pytest.mark.parametrize('direction',[1,-1])
def test_gap_uses_open_not_better_stop_no_same_open_reentry(direction):
    d,cfg,sig=setup(direction)
    d.loc[2,'open']=95 if direction==1 else 105
    d.loc[2,'low']=94.;d.loc[2,'high']=106.
    t,c,e=simulate_fixed_stop(d,sig,cfg,.01)
    r=t.iloc[0]
    assert r.exit_reason=='stop_gap' and r.exit_time_precision=='open'
    assert r.exit_raw_price==d.open.iloc[2]
    assert t.entry_time.iloc[1]==d.date.iloc[3]
    assert r.holding_hours_upper==r.holding_hours

def test_open_signal_exit_not_overridden_by_later_extremes():
    d,cfg,sig=setup();sig[1:]=-1;d.loc[2,'low']=50
    t,c,e=simulate_fixed_stop(d,sig,cfg,.02)
    assert t.exit_reason.iloc[0]=='signal' and t.exit_time.iloc[0]==d.date.iloc[2]
    assert t.entry_time.iloc[1]==d.date.iloc[2]
    assert t.status.iloc[1]=='open_marked_at_training_end'

def test_gap_priority_when_opposite_scheduled_signal():
    d,cfg,sig=setup();sig[1:]=-1;d.loc[2,['open','low']]=[95,94]
    t,c,e=simulate_fixed_stop(d,sig,cfg,.02)
    assert t.exit_reason.iloc[0]=='stop_gap'
    assert t.entry_time.iloc[1]==d.date.iloc[3] and t.direction.iloc[1]=='short'

def test_entry_bar_stop_active_no_fictional_open_holding_time():
    d,cfg,sig=setup();d.loc[1,'low']=90
    t,c,e=simulate_fixed_stop(d,sig,cfg,.01)
    assert t.entry_time.iloc[0]==t.exit_time.iloc[0]
    assert t.holding_hours.iloc[0]==0 and t.holding_hours_upper.iloc[0]==.25
    assert t.entry_time.iloc[1]==d.date.iloc[2]

def test_invalid_signal_does_not_block_protective_stop_or_trigger_new_entry():
    d,cfg,sig=setup();sig[2]=np.nan;d.loc[2,'low']=90
    t,c,e=simulate_fixed_stop(d,sig,cfg,.02)
    assert t.exit_reason.iloc[0]=='stop_touch'
    assert t.entry_time.iloc[1]==d.date.iloc[4] # index3 no valid previous signal

def test_exact_touch_equality_triggers_and_new_signal_after_stop():
    d,cfg,sig=setup();d.loc[2,'low']=100*1.0002*.98;sig[2:]=-1
    t,c,e=simulate_fixed_stop(d,sig,cfg,.02)
    assert t.exit_reason.iloc[0]=='stop_touch'
    assert t.direction.iloc[1]=='short' and t.entry_time.iloc[1]==d.date.iloc[3]

def test_final_stop_closes_without_reading_next_candle():
    d,cfg,sig=setup();d.loc[len(d)-1,'low']=90
    t,c,e=simulate_fixed_stop(d,sig,cfg,.02)
    assert t.status.eq('closed').all() and len(t)==1 and c.position.iloc[-1]==0
    np.testing.assert_allclose(t.net_pnl.sum(),c.equity.iloc[-1]-10000)

def test_prefix_future_perturbation_preserves_prior_curves_and_fills():
    d,cfg,sig=setup(count=10);d.loc[2,'low']=90
    _,c,e=simulate_fixed_stop(d,sig,cfg,.02)
    changed=d.copy();changed.loc[6:,['open','high','low','close']]*=1.5
    _,c2,e2=simulate_fixed_stop(changed,sig,cfg,.02)
    cutoff=d.date.iloc[6]
    pd.testing.assert_frame_equal(c[c.date<cutoff],c2[c2.date<cutoff])
    pd.testing.assert_frame_equal(e[e.time<cutoff],e2[e2.time<cutoff])
    _,prefix,_=simulate_fixed_stop(d.iloc[:6],sig[:6],cfg,.02)
    pd.testing.assert_frame_equal(c[c.date<cutoff],prefix)

def test_annual_boundary_no_forced_close_and_zero_volume_event():
    d,cfg,sig=setup();d['date']=pd.date_range('2020-12-31 23:00',periods=len(d),freq='15min',tz='UTC')
    cfg['performance_start']=str(d.date.iloc[1]);cfg['performance_end_exclusive']=str(d.date.iloc[-1]+pd.Timedelta(minutes=15))
    d.loc[4,'low']=90.;d.loc[4,'volume']=0
    t,c,e=simulate_fixed_stop(d,sig,cfg,.02)
    assert t.exit_time.iloc[0]==d.date.iloc[4] and t.holding_hours.iloc[0]==.75
    assert e[e.reason=='stop_touch'].zero_volume.all()

def test_no_double_charge_and_marked_end():
    d,cfg,sig=setup();t,c,e=simulate_fixed_stop(d,sig,cfg,.06)
    assert len(e)==1 and t.exit_fee.iloc[0]==0 and pd.isna(t.exit_time.iloc[0])
    np.testing.assert_allclose(t.net_pnl.sum(),c.equity.iloc[-1]-10000)
    np.testing.assert_allclose(t.gross_before_cost-t.entry_slippage_usdt-t.exit_slippage_usdt-t.entry_fee-t.exit_fee,t.net_pnl)
