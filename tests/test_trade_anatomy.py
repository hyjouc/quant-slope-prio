import numpy as np
import pandas as pd
from src.trade_anatomy import excursions,loss_streaks,drawdown_episodes,duration_analysis

def toy(direction='long',closed=True):
    date=pd.date_range('2020-12-31 23:30',periods=5,freq='15min',tz='UTC')
    b=pd.DataFrame(dict(date=date,open=[100,100,102,105,103],high=[101,110,108,9999,104],
        low=[99,95,98,1,100],close=[100,105,104,103,102],volume=1.))
    b=b.astype({field:float for field in ['open','high','low','close']})
    s=1 if direction=='long' else -1;ep=100.02 if s==1 else 99.98
    xp=105*(1-s*.0002) if closed else 102
    q=2000/ep;gross=s*q*(xp-ep)
    t=pd.DataFrame([dict(candidate='B_ols_N64',direction=direction,entry_time=date[1],exit_time=date[3] if closed else pd.NaT,
        entry_price=ep,quantity=q,gross_after_slippage=gross,net_pnl=gross-1-(q*xp*.0005 if closed else 0),
        holding_hours=.5 if closed else 1.,status='closed' if closed else 'open_marked_at_training_end')])
    return t,b

def test_long_excursion_uses_fill_and_excludes_exit_high_low():
    t,b=toy();a=excursions(t,b).iloc[0];ep=t.entry_price.iloc[0];q=t.quantity.iloc[0]
    np.testing.assert_allclose(a.mae_pct,(ep-95)/ep*100)
    np.testing.assert_allclose(a.mfe_pct,(110-ep)/ep*100)
    np.testing.assert_allclose(a.mae_usdt,(ep-95)*q)
    np.testing.assert_allclose(a.giveback_price_usdt,(110-105)*q)
    np.testing.assert_allclose(a.giveback_execution_usdt-a.giveback_price_usdt,105*.0002*q)
    assert a.path_bar_count==2 and a.exit_bar_exclusive==3

def test_short_direction_and_usdt():
    t,b=toy('short');a=excursions(t,b).iloc[0];ep=t.entry_price.iloc[0];q=t.quantity.iloc[0]
    np.testing.assert_allclose(a.mae_usdt,(110-ep)*q)
    np.testing.assert_allclose(a.mfe_usdt,(ep-95)*q)
    np.testing.assert_allclose(a.terminal_raw_gross,(ep-105)*q)
    assert a.giveback_price_ratio>1 # once favorable, ultimately loses; ratio may exceed100%

def test_exit_candle_future_extremes_and_suffix_do_not_change_closed_metrics():
    t,b=toy();a=excursions(t,b)
    changed=b.copy();changed.loc[3:,['high','low','close']]=[1e9,.01,1e8]
    result=excursions(t,changed)
    pd.testing.assert_frame_equal(a,result)
    pd.testing.assert_frame_equal(a,excursions(t,b.iloc[:4]))

def test_raw_exit_open_is_path_endpoint_even_outside_previous_range():
    t,b=toy();b.loc[3,'open']=120
    a=excursions(t,b).iloc[0]
    np.testing.assert_allclose(a.mfe_usdt,(120-t.entry_price.iloc[0])*t.quantity.iloc[0])

def test_open_trade_separate_and_includes_last_held_bar():
    t,b=toy(closed=False);a=excursions(t,b).iloc[0]
    assert a.outcome=='open_marked' and a.path_bar_count==4 and a.exit_bar_exclusive==5
    assert a.path_high==9999 and a.path_low==1

def test_nonpositive_mfe_ratio_is_unavailable():
    t,b=toy();b.loc[1:2,'high']=99.;b.loc[1:2,'low']=95.;b.loc[3,'open']=97
    a=excursions(t,b).iloc[0]
    assert a.mfe_usdt==0 and np.isnan(a.giveback_price_ratio)

def test_loss_runs_nonloss_break_and_direction_definition():
    times=pd.date_range('2020-01-01',periods=6,freq='1h',tz='UTC')
    g=pd.DataFrame(dict(candidate='B_ols_N64',entry_time=times,exit_time=times+pd.Timedelta(minutes=30),
        direction=['long','short']*3,net_pnl=[-1,-2,0,-3,-4,5],trade_id=list('abcdef')))
    s=loss_streaks(g);all_runs=s[s.direction=='all']
    assert all_runs.losses.tolist()==[2,2] and all_runs.net_loss.tolist()==[-3,-7]
    assert s[s.direction=='short'].losses.tolist()==[2]
    assert all_runs.ended_by_nonloss.all()

def test_drawdown_recovery_and_unrecovered_duration():
    dates=pd.date_range('2020-01-01',periods=5,freq='15min',tz='UTC')
    c=pd.DataFrame(dict(candidate='B_ols_N64',direction='all',date=dates,equity=[10000,9990,9980,10000,9995]))
    ep=drawdown_episodes(c)
    assert ep.depth_usdt.tolist()==[20,5]
    assert ep.recovered.tolist()==[True,False]
    np.testing.assert_allclose(ep.duration_hours,[.75,.25])

def test_duration_bins_are_fixed_and_exclusive():
    g=pd.DataFrame(dict(candidate='B_ols_N64',direction='long',holding_hours=[.25,1,2,4,8,16,32],
        net_pnl=[1,-1,1,-1,1,-1,1],mae_usdt=2.,mfe_usdt=3.))
    a=duration_analysis(g);all_bins=a[a.direction=='all']
    assert all_bins['count'].tolist()==[1]*7 and all_bins.net_profit.sum()==1
