"""Causal K5 entry events and descriptive outcomes, no portfolio simulator."""
import numpy as np
import pandas as pd

def slope_change(beta):
    b=np.asarray(beta,float);out=np.full(len(b),np.nan)
    if len(b)<5:return out
    win=np.lib.stride_tricks.sliding_window_view(b,5)
    out[4:]=np.where(np.isfinite(win).all(axis=1),np.sum((win-win[:,:1])*np.arange(-2,3),axis=1)/10,np.nan)
    return out

def event_flags(beta,g):
    b=np.asarray(beta,float);g=np.asarray(g,float)
    result=np.zeros(len(b),int)
    valid=np.isfinite(b[1:])&np.isfinite(g[1:])&np.isfinite(g[:-1])
    result[1:]=np.where(valid&(b[1:]<0)&(g[:-1]<=0)&(g[1:]>0),1,
        np.where(valid&(b[1:]>0)&(g[:-1]>=0)&(g[1:]<0),-1,0))
    return result

def cycle_ids(beta):
    ids=np.full(len(beta),-1,int);sign=0;cycle=-1
    for i,b in enumerate(beta):
        if not np.isfinite(b):continue
        new=int(np.sign(b)) or sign
        if not new:continue
        if new!=sign:cycle+=1;sign=new
        ids[i]=cycle
    return ids

def interval_outcome(d,entry,end,side,closed=True,notional=2000.,fee=.0005,slip=.0002):
    """end is confirmation execution bar, or last bar index for censored MTM."""
    raw=float(d.open.iloc[entry]);fill=raw*(1+side*slip);qty=notional/fill
    terminal=float(d.open.iloc[end] if closed else d.close.iloc[end])
    stop=end if closed else end+1
    lows=d.low.iloc[entry:stop].to_numpy();highs=d.high.iloc[entry:stop].to_numpy()
    low=min(float(lows.min()) if len(lows) else terminal,terminal,fill)
    high=max(float(highs.max()) if len(highs) else terminal,terminal,fill)
    adverse=(fill-low) if side==1 else (high-fill)
    favorable=(high-fill) if side==1 else (fill-low)
    exit_fill=terminal*(1-side*slip) if closed else terminal
    exit_fee=qty*exit_fill*fee if closed else 0.
    entry_fee=notional*fee
    net=side*qty*(exit_fill-fill)-entry_fee-exit_fee
    gross=side*qty*(terminal-raw)
    slip_cost=qty*abs(fill-raw)+(qty*abs(exit_fill-terminal) if closed else 0.)
    return dict(entry_raw=raw,entry_fill=fill,quantity=qty,terminal_raw=terminal,terminal_fill=exit_fill,
        mae_pct=100*adverse/fill,mfe_pct=100*favorable/fill,mae_usdt=qty*adverse,mfe_usdt=qty*favorable,
        raw_signed_bps=side*(terminal/raw-1)*10000,net_usdt=net,gross_usdt=gross,
        fees_usdt=entry_fee+exit_fee,slippage_usdt=slip_cost,entry_zero_volume=bool(d.volume.iloc[entry]==0),
        terminal_zero_volume=bool(d.volume.iloc[end]==0),path_zero_volume_bars=int(d.volume.iloc[entry:stop].eq(0).sum()))

def build_events(d,beta,g,first=96):
    flags=event_flags(beta,g);ids=cycle_ids(beta);n=len(d);dates=pd.DatetimeIndex(d.date)
    positions={c:np.flatnonzero(ids==c) for c in np.unique(ids[ids>=0])}
    ordinals={};rows=[]
    for i in np.flatnonzero(flags):
        c=int(ids[i]);ordinals[c]=ordinals.get(c,0)+1
        if i+1<first or i+1>=n:continue
        side=int(flags[i]);j=i+1;members=positions[c];last=int(members[-1]);cross=last+1
        # Exact zero belongs to current cycle; reject invalid or absent next cycle.
        if cross>=n or not np.isfinite(beta[cross]) or np.sign(beta[cross])!=side:cross=None
        executable=cross is not None and cross+1<n
        end=cross+1 if executable else n-1
        outcome=interval_outcome(d,j,end,side,executable)
        row=dict(event_id=len(rows)+1,direction='long' if side==1 else 'short',side=side,
            cycle_id=c,ordinal_in_cycle=ordinals[c],is_first=ordinals[c]==1,
            cycle_left_truncated=bool(members[0]<first-1),source_index=int(i),entry_index=int(j),
            source_bar=dates[i],entry_time=dates[j],entry_year=int(dates[j].year),
            beta=beta[i],b_bp_hour=beta[i]*40000,g=g[i],g_bp_hour_per_15m=g[i]*40000,
            confirmation_source_index=cross if cross is not None else np.nan,
            confirmation_execution_index=cross+1 if executable else np.nan,
            confirmation_observed=cross is not None,confirmation_executable=executable,censored=not executable,
            confirmation_signal_time=dates[cross]+pd.Timedelta(minutes=15) if cross is not None else pd.NaT,
            confirmation_execution_time=dates[cross+1] if executable else pd.NaT,
            lead_bars=cross-i if executable else np.nan,
            wait_hours=(cross-i)/4 if executable else np.nan,
            observed_duration_hours=(end-j)/4 if executable else (n-j)/4,
            endpoint_index=end,endpoint_time=dates[end] if executable else dates[end]+pd.Timedelta(minutes=15),
            status='closed_at_confirmation_open' if executable else 'censored_marked_at_training_end',**outcome)
        between=g[i+1:cross+1 if cross is not None else n]
        row['g_reversed_before_confirmation']=bool(np.any(side*between<0))
        rows.append(row)
    return pd.DataFrame(rows),flags,ids
