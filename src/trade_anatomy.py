"""Post-trade descriptive analysis. No signal, estimator, order or strategy changes."""
import numpy as np
import pandas as pd

def excursions(trades,bars):
    dates=pd.DatetimeIndex(bars.date);rows=[]
    for number,t in enumerate(trades.itertuples(index=False)):
        entry=pd.Timestamp(t.entry_time);ei=dates.get_indexer([entry])[0]
        if ei<0:raise ValueError('Missing exact entry candle')
        closed=t.status=='closed'
        end=pd.Timestamp(t.exit_time) if closed else dates[-1]+pd.Timedelta(minutes=15)
        xi=dates.get_indexer([end])[0] if closed else len(bars)
        if xi<0 or xi<=ei:raise ValueError('Invalid exit or empty holding interval')
        b=bars.iloc[ei:xi] # EXIT CANDLE IS EXCLUDED, including its close
        s=1 if t.direction=='long' else -1
        terminal=float(bars.open.iloc[xi]) if closed else float(bars.close.iloc[-1])
        ep=float(t.entry_price);qty=float(t.quantity)
        high=max(ep,float(b.high.max()),terminal);low=min(ep,float(b.low.min()),terminal)
        mae=(ep-low)/ep if s==1 else (high-ep)/ep
        mfe=(high-ep)/ep if s==1 else (ep-low)/ep
        mae_usdt=mae*ep*qty;mfe_usdt=mfe*ep*qty
        raw_gross=s*qty*(terminal-ep)
        exec_gross=float(t.gross_after_slippage)
        primary=mfe_usdt-raw_gross;execution=mfe_usdt-exec_gross
        row=t._asdict()
        row.update(trade_id=f'{t.candidate}_{number:04d}',entry_bar_index=int(ei),exit_bar_exclusive=int(xi),
            path_bar_count=len(b),path_start=str(entry),path_end=str(end),terminal_raw_price=terminal,
            path_high=high,path_low=low,mae_pct=100*mae,mfe_pct=100*mfe,mae_usdt=mae_usdt,mfe_usdt=mfe_usdt,
            terminal_raw_gross=raw_gross,final_gross_after_slippage=exec_gross,
            giveback_price_usdt=primary,giveback_price_ratio=primary/mfe_usdt if mfe_usdt>0 else np.nan,
            giveback_execution_usdt=execution,giveback_execution_ratio=execution/mfe_usdt if mfe_usdt>0 else np.nan,
            outcome=('winner' if t.net_pnl>0 else 'loser' if t.net_pnl<0 else 'breakeven') if closed else 'open_marked',
            zero_volume_path_bars=int(b.volume.eq(0).sum()),
            gross_positive_net_loss=bool(closed and exec_gross>0 and t.net_pnl<0),
            positive_mfe_final_net_loss=bool(closed and mfe_usdt>0 and t.net_pnl<0))
        rows.append(row)
    return pd.DataFrame(rows)

def group_views(closed):
    for candidate,c in closed.groupby('candidate',sort=False):
        for direction in ['all','long','short']:
            d=c if direction=='all' else c[c.direction==direction]
            for outcome in ['all','winner','loser','breakeven']:
                g=d if outcome=='all' else d[d.outcome==outcome]
                if len(g):yield candidate,direction,outcome,g

def distributions(closed,fields):
    rows=[]
    for candidate,direction,outcome,g in group_views(closed):
        for field in fields:
            v=g[field].dropna()
            row=dict(candidate=candidate,direction=direction,outcome=outcome,metric=field,
                trade_count=len(g),valid_count=len(v),sum=v.sum(),mean=v.mean(),median=v.median(),
                min=v.min(),max=v.max())
            row.update({f'p{int(q*100):02}':v.quantile(q) for q in [.1,.25,.75,.9,.95]})
            rows.append(row)
    return pd.DataFrame(rows)

def duration_analysis(closed):
    edges=[0,1,2,4,8,16,32,np.inf]
    labels=['[0,1)','[1,2)','[2,4)','[4,8)','[8,16)','[16,32)','[32,inf)']
    rows=[]
    for candidate,c in closed.groupby('candidate',sort=False):
        for direction in ['all','long','short']:
            d=c if direction=='all' else c[c.direction==direction]
            bins=pd.cut(d.holding_hours,edges,right=False,labels=labels)
            for label in labels:
                g=d[bins==label]
                rows.append(dict(candidate=candidate,direction=direction,duration_bin=label,count=len(g),
                    winners=int(g.net_pnl.gt(0).sum()),losers=int(g.net_pnl.lt(0).sum()),
                    net_profit=g.net_pnl.sum(),mean_net=g.net_pnl.mean(),mean_hours=g.holding_hours.mean(),
                    median_hours=g.holding_hours.median(),mean_mae_usdt=g.mae_usdt.mean(),mean_mfe_usdt=g.mfe_usdt.mean()))
    return pd.DataFrame(rows)

def loss_streaks(closed):
    rows=[]
    for candidate,c in closed.groupby('candidate',sort=False):
        for direction in ['all','long','short']:
            g=(c if direction=='all' else c[c.direction==direction]).sort_values('exit_time')
            streak=[]
            def flush():
                if streak:
                    rows.append(dict(candidate=candidate,direction=direction,streak_id=len(rows),
                        start_entry=streak[0].entry_time,first_exit=streak[0].exit_time,last_exit=streak[-1].exit_time,
                        losses=len(streak),net_loss=sum(t.net_pnl for t in streak),
                        elapsed_hours=(streak[-1].exit_time-streak[0].entry_time).total_seconds()/3600,
                        first_trade_id=streak[0].trade_id,last_trade_id=streak[-1].trade_id,
                        ended_by_nonloss=False))
            for t in g.itertuples():
                if t.net_pnl<0:streak.append(t)
                elif streak:
                    flush();rows[-1]['ended_by_nonloss']=True;streak=[]
            flush()
    return pd.DataFrame(rows)

def pnl_summary(closed,streaks):
    rows=[];concentration=[]
    for candidate,c in closed.groupby('candidate',sort=False):
        for direction in ['all','long','short']:
            g=c if direction=='all' else c[c.direction==direction]
            w=g[g.net_pnl>0];l=g[g.net_pnl<0];ss=streaks[(streaks.candidate==candidate)&(streaks.direction==direction)]
            gross_win=w.net_pnl.sum();gross_loss=-l.net_pnl.sum()
            row=dict(candidate=candidate,direction=direction,trade_count=len(g),winning_count=len(w),losing_count=len(l),
                breakeven_count=int(g.net_pnl.eq(0).sum()),win_rate=len(w)/len(g),
                mean_win=w.net_pnl.mean(),mean_loss=l.net_pnl.mean(),median_win=w.net_pnl.median(),median_loss=l.net_pnl.median(),
                payoff_ratio=w.net_pnl.mean()/(-l.net_pnl.mean()) if len(w) and len(l) else np.nan,
                profit_factor=gross_win/gross_loss if gross_loss else np.nan,
                max_win=g.net_pnl.max(),max_loss=g.net_pnl.min(),closed_net_profit=g.net_pnl.sum(),
                mean_mae_pct=g.mae_pct.mean(),mean_mfe_pct=g.mfe_pct.mean(),mean_mae_usdt=g.mae_usdt.mean(),mean_mfe_usdt=g.mfe_usdt.mean(),
                mean_holding_hours=g.holding_hours.mean(),median_holding_hours=g.holding_hours.median(),
                winner_mean_hours=w.holding_hours.mean(),winner_median_hours=w.holding_hours.median(),
                loser_mean_hours=l.holding_hours.mean(),loser_median_hours=l.holding_hours.median(),
                mean_giveback_price_usdt=g.giveback_price_usdt.mean(),median_giveback_price_ratio=g.giveback_price_ratio.median(),
                max_consecutive_losses=int(ss.losses.max()) if len(ss) else 0,
                worst_streak_net=ss.net_loss.min() if len(ss) else 0.,
                gross_positive_net_loss_count=int(g.gross_positive_net_loss.sum()),
                winner_gross_profit=gross_win,loser_gross_loss=gross_loss)
            rows.append(row)
            for k in [1,5,10,20]:
                top=w.nlargest(k,'net_pnl')
                amount=top.net_pnl.sum()
                concentration.append(dict(candidate=candidate,direction=direction,top_k=k,actual_count=len(top),top_net_profit=amount,
                    share_winning_profit=amount/gross_win if gross_win else np.nan,
                    share_closed_net=amount/row['closed_net_profit'] if row['closed_net_profit']>0 else np.nan,
                    closed_net_without_top=row['closed_net_profit']-amount,
                    top_mean_mae_usdt=top.mae_usdt.mean(),top_mean_mfe_usdt=top.mfe_usdt.mean(),
                    top_mean_price_giveback_usdt=top.giveback_price_usdt.mean(),top_median_price_giveback_ratio=top.giveback_price_ratio.median(),
                    top_mean_holding_hours=top.holding_hours.mean()))
    return pd.DataFrame(rows),pd.DataFrame(concentration)

def reconstruct_curves(trades,bars,reference,initial=10000.):
    """Value EXISTING trades; no signals/orders recomputed. Assert equality to S0."""
    dates=pd.DatetimeIndex(bars.date);cl=bars.close.to_numpy(float);curves=[]
    start=pd.Timestamp(reference.date.iloc[0]);first=int(dates.get_indexer([start])[0])
    for candidate,t in trades.groupby('candidate',sort=False):
        side_pnl={'long':np.zeros(len(bars)),'short':np.zeros(len(bars))}
        for r in t.itertuples():
            ei=dates.get_indexer([r.entry_time])[0]
            xi=dates.get_indexer([r.exit_time])[0] if r.status=='closed' else len(bars)
            if ei<first or xi<0:raise ValueError('Trade date outside approved interval')
            s=1 if r.direction=='long' else -1
            side_pnl[r.direction][ei:xi]+=s*r.quantity*(cl[ei:xi]-r.entry_price)-r.entry_fee
            if r.status=='closed':side_pnl[r.direction][xi:]+=r.net_pnl
        account=initial+side_pnl['long'][first:]+side_pnl['short'][first:]
        np.testing.assert_allclose(account,reference[candidate],rtol=0,atol=1e-6)
        for direction in ['all','long','short']:
            pnl=account-initial if direction=='all' else side_pnl[direction][first:]
            equity=initial+pnl;peak=np.maximum.accumulate(np.r_[initial,equity])[1:]
            c=pd.DataFrame(dict(date=dates[first:],candidate=candidate,direction=direction,equity=equity,
                pnl_change=np.diff(np.r_[0.,pnl]),drawdown_usdt=peak-equity))
            curves.append(c)
    return pd.concat(curves,ignore_index=True)

def drawdown_episodes(curve,initial=10000.):
    peak=initial;peak_time=curve.date.iloc[0];active=None;rows=[]
    for r in curve.itertuples():
        clock=r.date+pd.Timedelta(minutes=15)
        if r.equity>=peak-1e-8:
            if active is not None:
                rows.append(dict(active,end=clock,recovered=True,duration_hours=(clock-active['peak_time']).total_seconds()/3600))
                active=None
            peak=max(peak,r.equity);peak_time=clock
        else:
            if active is None:active=dict(candidate=r.candidate,direction=r.direction,peak_time=peak_time,first_underwater=clock,
                trough_time=clock,depth_usdt=peak-r.equity,peak_equity=peak)
            if peak-r.equity>active['depth_usdt']:active.update(trough_time=clock,depth_usdt=peak-r.equity)
    if active is not None:
        clock=curve.date.iloc[-1]+pd.Timedelta(minutes=15)
        rows.append(dict(active,end=clock,recovered=False,duration_hours=(clock-active['peak_time']).total_seconds()/3600))
    return pd.DataFrame(rows)

def risk_analysis(curves,streaks,closed):
    summaries=[];periods=[];episodes=[]
    for (candidate,direction),c in curves.groupby(['candidate','direction'],sort=False):
        dd=drawdown_episodes(c);episodes.append(dd)
        period_values={}
        for name,freq in [('day','D'),('week','W-SUN'),('month','M')]:
            keys=c.date.dt.tz_localize(None).dt.to_period(freq)
            z=c.groupby(keys).pnl_change.sum()
            period_values[name]=z
            for period,value in z.items():
                periods.append(dict(candidate=candidate,direction=direction,frequency=name,period=str(period),
                    start=str(period.start_time.tz_localize('UTC')),net_profit=value))
        ss=streaks[(streaks.candidate==candidate)&(streaks.direction==direction)]
        g=closed[closed.candidate==candidate]
        if direction!='all':g=g[g.direction==direction]
        month=period_values['month'];negative=-month[month<0]
        deepest=dd.loc[dd.depth_usdt.idxmax()];longest=dd.loc[dd.duration_hours.idxmax()]
        row=dict(candidate=candidate,direction=direction,closed_trades=len(g),
            max_consecutive_losses=int(ss.losses.max()),max_consecutive_loss_usdt=float(-ss.net_loss.min()),
            total_loss_streaks=len(ss),median_loss_streak_length=ss.losses.median(),
            max_drawdown_usdt=float(c.drawdown_usdt.max()),
            max_drawdown_duration_hours=longest.duration_hours,max_drawdown_duration_days=longest.duration_hours/24,
            deepest_dd_duration_days=deepest.duration_hours/24,deepest_dd_peak=str(deepest.peak_time),
            deepest_dd_trough=str(deepest.trough_time),deepest_dd_end=str(deepest.end),deepest_dd_recovered=bool(deepest.recovered),
            unrecovered_at_end=bool(not dd.recovered.iloc[-1]),
            losing_months=int((month<0).sum()),total_months=len(month),
            worst3_months_loss_share=float(negative.nlargest(3).sum()/negative.sum()) if len(negative) else np.nan,
            flips_within_1h_count=int(g.holding_hours.lt(1).sum()),
            losses_within_1h_count=int((g.holding_hours.lt(1)&g.net_pnl.lt(0)).sum()))
        for name,z in period_values.items():row.update({f'worst_{name}_pnl':z.min(),f'worst_{name}_period':str(z.idxmin())})
        summaries.append(row)
    return pd.DataFrame(summaries),pd.DataFrame(periods),pd.concat(episodes,ignore_index=True)
