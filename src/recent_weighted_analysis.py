"""Descriptive event matching, synthetic response and original-winner attribution."""
import numpy as np
import pandas as pd
from .recent_weighted_slope import estimate_ewls
from .signal_generator import generate_signals

def flips(signal,start=0,end=None):
    end=len(signal) if end is None else end
    last=0;rows=[]
    for i in range(start,end):
        s=signal[i]
        if not np.isfinite(s) or s==0:continue
        if last and s!=last:rows.append(dict(index=i,from_sign=int(last),to_sign=int(s),event_id=len(rows)))
        last=int(s)
    return pd.DataFrame(rows,columns=['index','from_sign','to_sign','event_id'])

def match_flips(original,candidate,start,end,dates=None):
    """Disjoint OLS Voronoi cells, no tuned bar-distance or arbitrary matched count."""
    a=flips(original,start,end);b=flips(candidate,start,end);rows=[];used=set()
    idx=a['index'].to_numpy()
    for i,r in enumerate(a.itertuples(index=False)):
        left=start-.5 if i==0 else (idx[i-1]+r.index)/2
        right=end-.5 if i==len(a)-1 else (r.index+idx[i+1])/2
        # Midpoint tie goes to earlier OLS cell, guaranteeing unique membership.
        near=b[(b['index']>left)&(b['index']<=right)&(b.to_sign==r.to_sign)]
        if len(near):
            near=near.assign(distance=(near['index']-r.index).abs()).sort_values(['distance','index'])
            hit=near.iloc[0];j=int(hit['index']);used.add(int(hit.event_id));delay=j-r.index
            rows.append(dict(ols_event_id=r.event_id,ewls_event_id=int(hit.event_id),ols_index=r.index,ewls_index=j,
                to_sign=r.to_sign,delta_bars=delay,classification='early' if delay<0 else 'late' if delay>0 else 'same',
                cell_left_exclusive=left,cell_right_inclusive=right))
        else:rows.append(dict(ols_event_id=r.event_id,ewls_event_id=np.nan,ols_index=r.index,ewls_index=np.nan,
            to_sign=r.to_sign,delta_bars=np.nan,classification='unmatched_ols',cell_left_exclusive=left,cell_right_inclusive=right))
    for r in b.itertuples(index=False):
        if r.event_id not in used:rows.append(dict(ols_event_id=np.nan,ewls_event_id=r.event_id,ols_index=np.nan,ewls_index=r.index,
            to_sign=r.to_sign,delta_bars=np.nan,classification='unmatched_ewls',cell_left_exclusive=np.nan,cell_right_inclusive=np.nan))
    out=pd.DataFrame(rows)
    if out.empty:return out
    for name in ['ols','ewls']:
        out[name+'_source_bar']=[dates[int(i)] if dates is not None and pd.notna(i) else pd.NaT for i in out[name+'_index']]
        out[name+'_signal_available_at']=pd.to_datetime(out[name+'_source_bar'],utc=True)+pd.Timedelta(minutes=15)
    return out

def response_summary(original,candidate,matched,start,end):
    o=np.asarray(original[start:end]);c=np.asarray(candidate[start:end]);good=np.isfinite(o)&np.isfinite(c)
    delta=matched.delta_bars.dropna() if len(matched) else pd.Series(dtype=float)
    early=delta[delta<0];late=delta[delta>0]
    af=flips(original,start,end);bf=flips(candidate,start,end)
    return dict(used_signal_bars=end-start,valid_comparison_bars=int(good.sum()),
        signal_agreement_ratio=float(np.mean(o[good]==c[good])),ols_flip_count=len(af),candidate_flip_count=len(bf),
        matched_reversals=len(delta),early_reversals=len(early),late_reversals=len(late),same_reversals=int(delta.eq(0).sum()),
        unmatched_ols_reversals=int(matched.classification.eq('unmatched_ols').sum()) if len(matched) else 0,
        unmatched_candidate_reversals=int(matched.classification.eq('unmatched_ewls').sum()) if len(matched) else 0,
        signed_delay_mean_bars=delta.mean(),signed_delay_median_bars=delta.median(),
        early_lead_mean_bars=-early.mean(),late_delay_mean_bars=late.mean(),
        flip_rate_per_1000_bars=len(bf)/(end-start)*1000,
        long_to_short_flips=int(bf.to_sign.eq(-1).sum()),short_to_long_flips=int(bf.to_sign.eq(1).sum()))

def synthetic_paths(cfg):
    n=cfg['bars'];r=cfg['reversal_bar'];drift=cfg['log_return_per_bar'];x=np.arange(n)
    for direction,kind in [(1,'up_to_down'),(-1,'down_to_up')]:
        inc=np.where(x<r,direction*drift,-direction*drift)
        latent=np.log(100.)+np.cumsum(inc)
        # Same random draw across amplitudes/directions, only after reversal.
        noise=np.random.default_rng(cfg['seed']).normal(size=n);noise[:r]=0
        for sd in cfg['log_price_noise_std']:
            yield dict(case=f'{kind}_noise{sd:g}',family='trend_reversal',noise_std=sd,direction=direction,
                y=latent+sd*noise,true_sign=np.sign(inc),change=r,unshocked=None)
    latent=np.log(100.)+cfg['oscillation_log_amplitude']*np.sin(2*np.pi*x/cfg['oscillation_period_bars'])
    for direction,kind in [(1,'oscillation_up_shock'),(-1,'oscillation_down_shock')]:
        y=latent.copy();s=cfg['shock_start'];e=s+cfg['shock_length_bars'];y[s:e]+=direction*cfg['shock_log_offset']
        yield dict(case=kind,family='oscillation_shock',noise_std=0.,direction=direction,y=y,
            true_sign=None,change=s,unshocked=latent)

def synthetic_experiments(cfg,models):
    rows=[];paths=[]
    for path in synthetic_paths(cfg):
        n=len(path['y']);change=path['change']
        for name,h in models:
            beta=estimate_ewls(np.exp(path['y']),h);signal=generate_signals(beta)
            f=flips(signal,63,n)
            if path['family']=='trend_reversal':
                target=-path['direction'];hits=np.flatnonzero((np.arange(n)>=change)&(signal==target))
                first=int(hits[0]) if len(hits) else None
                wrong=sum(int(e.to_sign!=path['true_sign'][e.index]) for e in f.itertuples())
                switchbacks=sum(int(e.index>first and e.to_sign!=target) for e in f.itertuples()) if first is not None else 0
                counter=np.nan;extra=np.nan
            else:
                target=-int(signal[change-1]) if np.isfinite(signal[change-1]) else 0
                hits=np.flatnonzero((np.arange(n)>=change)&(signal==target));first=int(hits[0]) if len(hits) else None
                un=generate_signals(estimate_ewls(np.exp(path['unshocked']),h))
                counter=len(flips(un,63,n));extra=len(f)-counter;wrong=np.nan;switchbacks=np.nan
            rows.append(dict(case=path['case'],family=path['family'],model=name,H=h,noise_std=path['noise_std'],
                seed=cfg['seed'],true_change_index=change if path['family']=='trend_reversal' else np.nan,
                shock_start_index=change if path['family']=='oscillation_shock' else np.nan,
                first_new_direction_index=first,response_delay_bars=first-change if first is not None else np.nan,
                wrong_destination_flips=wrong,post_response_wrong_switchbacks=switchbacks,total_flips=len(f),
                unshocked_total_flips=counter,additional_flips_vs_unshocked=extra))
            paths.append(pd.DataFrame(dict(case=path['case'],family=path['family'],model=name,index=np.arange(n),
                log_price=path['y'],slope=beta,signal=signal,change_index=change)))
    out=pd.DataFrame(rows)
    base=out[out.model=='OLS'].set_index('case').response_delay_bars
    out['delay_change_vs_ols']=out.response_delay_bars-out.case.map(base)
    return out,pd.concat(paths,ignore_index=True)

def interval_pnl(trades,start,end,bars):
    """Actual candidate cash/MTM change on [start-open,end-open], no imaginary fills.

    Pre-entry exposure marked at raw start open; entry charges at start included;
    end-open closes included but new entries exactly at end excluded. All positions
    still alive at end marked at raw end open without hypothetical exit charges.
    """
    dates=pd.DatetimeIndex(bars.date);sp=float(bars.open.iloc[dates.get_indexer([start])[0]]);ep=float(bars.open.iloc[dates.get_indexer([end])[0]])
    total=0.;by_side={'long':0.,'short':0.}
    for t in trades.itertuples():
        en=pd.Timestamp(t.entry_time);ex=pd.Timestamp(t.exit_time) if t.status=='closed' else None
        if en>=end or (ex is not None and ex<=start):continue
        s=1 if t.direction=='long' else -1
        if en>=start:
            # Entry fill inside interval; already contains entry slippage.
            begin=t.entry_price;entry_fee=t.entry_fee
        else:begin=sp;entry_fee=0.
        if ex is not None and ex<=end:
            terminal=t.exit_price;exit_fee=t.exit_fee
        else:terminal=ep;exit_fee=0.
        pnl=s*t.quantity*(terminal-begin)-entry_fee-exit_fee
        total+=pnl;by_side[t.direction]+=pnl
    return total,by_side

def original_trade_comparison(original,candidate,bars):
    rows=[]
    top=set(original[original.status=='closed'].nlargest(10,'net_pnl').entry_time)
    for i,o in enumerate(original[original.status=='closed'].itertuples()):
        start=pd.Timestamp(o.entry_time);end=pd.Timestamp(o.exit_time)
        # Position held immediately after the original entry-open execution.
        covering=candidate[(candidate.entry_time<=start)&((candidate.exit_time>start)|candidate.exit_time.isna())]
        same=covering[covering.direction==o.direction]
        aligned=candidate[(candidate.entry_time==start)&(candidate.direction==o.direction)]
        overlap_first_exit=same.exit_time.iloc[0] if len(same) else pd.NaT
        early=bool(pd.notna(overlap_first_exit) and overlap_first_exit<end)
        direct_net=float(same.net_pnl.iloc[0]) if len(same) else np.nan
        pnl,side=interval_pnl(candidate,start,end,bars)
        rows.append(dict(original_trade_id=i,original_entry_time=start,original_exit_time=end,direction=o.direction,
            original_net_profit=o.net_pnl,original_holding_hours=o.holding_hours,is_original_top10=start in top,
            exact_same_entry_taken=bool(len(aligned)),same_direction_position_at_original_entry=bool(len(same)),
            covering_trade_entry=same.entry_time.iloc[0] if len(same) else pd.NaT,
            covering_trade_exit=overlap_first_exit,first_covering_position_exited_early=early,
            covering_trade_net_profit=direct_net,covering_trade_smaller_positive_winner=bool(o.net_pnl>0 and direct_net>0 and direct_net<o.net_pnl),
            covering_trade_became_loss=bool(o.net_pnl>0 and direct_net<0),
            actual_interval_pnl=pnl,actual_interval_long_pnl=side['long'],actual_interval_short_pnl=side['short'],
            interval_pnl_change_vs_original=pnl-o.net_pnl,
            interval_smaller_positive_profit=bool(o.net_pnl>0 and 0<pnl<o.net_pnl),
            interval_became_loss=bool(o.net_pnl>0 and pnl<0)))
    return pd.DataFrame(rows)
