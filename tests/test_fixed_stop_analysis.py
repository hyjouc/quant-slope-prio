import numpy as np
import pandas as pd
from test_execution import fixture
from src.fixed_stop_backtester import simulate_fixed_stop
from src.fixed_stop_analysis import original_winner_audit,streak_summary

def test_original_winner_early_stop_and_reentry_are_separate_from_episode_pnl():
    d,cfg=fixture(prices=[100,100,101,102,103,104,105,106]);d['low']=d.open-.1;d['high']=d.open+.1
    sig=np.array([1,1,1,1,1,-1,-1,-1.]);d.loc[2,'low']=95.
    original,_,_=simulate_fixed_stop(d,sig,cfg,None)
    stopped,_,_=simulate_fixed_stop(d,sig,cfg,.02)
    audit=original_winner_audit(original,stopped,d,.02)
    r=audit.iloc[0]
    assert r.original_entry_stopped_early and r.isolated_original_entry_early_touch
    assert r.episode_entries==2 and r.episode_reentries==1
    assert not r.original_entry_held_to_normal_exit
    np.testing.assert_allclose(r.episode_net_profit,stopped.iloc[:2].net_pnl.sum())

def test_isolated_audit_excludes_exit_bar_future_extremes():
    d,cfg=fixture(prices=[100,100,101,102,103,104,105,106]);d['low']=d.open-.1;d['high']=d.open+.1
    sig=np.array([1,1,1,1,1,-1,-1,-1.]);original,_,_=simulate_fixed_stop(d,sig,cfg,None)
    a=original_winner_audit(original,original,d,.02)
    changed=d.copy();changed.loc[6:,'low']=1.;changed.loc[6:,'high']=9999.
    b=original_winner_audit(original,original,changed,.02)
    pd.testing.assert_frame_equal(a,b)
    assert not a.isolated_original_entry_early_touch.iloc[0]

def test_stop_and_signal_exit_same_bar_sequence_controls_loss_run():
    t=pd.DataFrame(dict(status=['closed']*5,trade_sequence=[0,1,2,3,4],net_pnl=[-20,-15,5,-10,-12]))
    r=streak_summary(t)
    assert r['max_consecutive_losses']==2 and r['max_loss_streak_usdt']==35 and r['loss_streak_count']==2
