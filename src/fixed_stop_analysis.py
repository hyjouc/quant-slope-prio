"""Fixed-stop comparison diagnostics; no parameter selection."""
import numpy as np
import pandas as pd
from .trade_anatomy import drawdown_episodes

def streak_summary(t):
    g=t[t.status=='closed'].sort_values('trade_sequence');length=0;amount=0.;runs=[]
    for r in g.itertuples():
        if r.net_pnl<0:length+=1;amount+=r.net_pnl
        elif length:runs.append((length,amount));length=0;amount=0.
    if length:runs.append((length,amount))
    return dict(max_consecutive_losses=max((a for a,b in runs),default=0),
        max_loss_streak_usdt=-min((b for a,b in runs),default=0.),loss_streak_count=len(runs))

def risk_summary(c,t):
    curve=c.copy();curve['direction']='all'
    dd=drawdown_episodes(curve)
    g=t[t.status=='closed'];w=g[g.net_pnl>0]
    net=c.equity.iloc[-1]-10000
    row=streak_summary(t)
    row.update(max_single_trade_loss=g.net_pnl.min(),max_single_trade_win=g.net_pnl.max(),
        longest_underwater_days=dd.duration_hours.max()/24 if len(dd) else 0.,
        underwater_at_end=bool(len(dd) and not dd.recovered.iloc[-1]),
        stop_exits=int(g.exit_reason.isin(['stop_gap','stop_touch']).sum()),
        gap_stop_exits=int(g.exit_reason.eq('stop_gap').sum()),intrabar_stop_exits=int(g.exit_reason.eq('stop_touch').sum()),
        normal_signal_exits=int(g.exit_reason.eq('signal').sum()),
        average_holding_hours_upper=g.holding_hours_upper.mean())
    for k in [1,5,10,20]:
        amount=w.nlargest(k,'net_pnl').net_pnl.sum()
        row.update({f'top{k}_profit_usdt':amount,f'top{k}_share_winning_profit':amount/w.net_pnl.sum() if len(w) else np.nan,
            f'net_without_top{k}_winning_trades':net-amount})
    return row,dd

def original_winner_audit(baseline,t,bars,stop_loss):
    """Original top10 fixed identities; original-entry touch vs realized changed path separately."""
    winners=baseline[baseline.status=='closed'].nlargest(10,'net_pnl').copy()
    dates=pd.DatetimeIndex(bars.date);rows=[]
    for rank,b in enumerate(winners.itertuples(),1):
        entry=pd.Timestamp(b.entry_time);exit=pd.Timestamp(b.exit_time)
        ei=dates.get_indexer([entry])[0];xi=dates.get_indexer([exit])[0]
        s=1 if b.direction=='long' else -1
        stop=b.entry_price*(1-s*stop_loss) if stop_loss is not None else np.nan
        hit=None;hit_reason='none';precision='none'
        if stop_loss is not None:
            for i in range(ei,xi):
                op=float(bars.open.iloc[i]);lo=float(bars.low.iloc[i]);hi=float(bars.high.iloc[i])
                if i>ei and ((s==1 and op<=stop) or (s==-1 and op>=stop)):
                    hit=dates[i];hit_reason='stop_gap';precision='open';break
                if (s==1 and lo<=stop) or (s==-1 and hi>=stop):
                    hit=dates[i];hit_reason='stop_touch';precision='within_15m_unknown';break
        # All actual entries within original same-sign episode; new opposite entry
        # at original exit excluded. Stop exits at its exact endpoint not "early".
        episode=t[(t.entry_time>=entry)&(t.entry_time<exit)&(t.direction==b.direction)]
        if episode.exit_time.isna().any() or (episode.exit_time>exit).any():raise ValueError('Episode crosses original sign reversal')
        stops=episode[episode.exit_reason.isin(['stop_gap','stop_touch'])]
        early=stops[stops.exit_time<exit]
        aligned=episode[episode.entry_time==entry]
        origin_early=bool(len(aligned) and aligned.exit_reason.iloc[0] in ['stop_gap','stop_touch'] and aligned.exit_time.iloc[0]<exit)
        kept=bool(len(aligned) and aligned.exit_reason.iloc[0]=='signal' and aligned.exit_time.iloc[0]==exit)
        same_entry_skipped=len(aligned)==0
        amount=episode.net_pnl.sum()
        rows.append(dict(original_winner_rank=rank,original_entry_time=entry,original_exit_time=exit,direction=b.direction,
            original_net_profit=b.net_pnl,original_entry_price=b.entry_price,original_quantity=b.quantity,
            fixed_L=stop_loss,original_entry_stop_price=stop,isolated_original_entry_early_touch=hit is not None,
            isolated_touch_time=hit,isolated_touch_precision=precision,isolated_touch_reason=hit_reason,
            same_original_entry_taken=not same_entry_skipped,original_entry_stopped_early=origin_early,
            original_entry_held_to_normal_exit=kept,any_actual_early_stop_during_episode=bool(len(early)),
            actual_early_stop_count=len(early),episode_entries=len(episode),episode_reentries=max(0,len(episode)-1),
            episode_net_profit=amount,episode_profit_retention_ratio=amount/b.net_pnl,
            episode_pnl_difference=amount-b.net_pnl,
            first_actual_stop_time=early.exit_time.iloc[0] if len(early) else pd.NaT))
    return pd.DataFrame(rows)
