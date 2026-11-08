"""S1.2 isolated extension. Frozen S0 modules remain untouched."""
import numpy as np
import pandas as pd
from .backtester import TRADE_COLUMNS,EXEC_COLUMNS

def simulate_fixed_stop(d,signals,config,stop_loss=None):
    if stop_loss is not None and not 0<stop_loss<1:raise ValueError('L must be a PRICE fraction in (0,1)')
    dates=pd.DatetimeIndex(d.date);sig=np.asarray(signals,dtype=float)
    if dates.tz is None or not dates.is_unique or not dates.is_monotonic_increasing:raise ValueError('Need chronological UTC candles')
    if len(d)>1 and not (dates[1:]-dates[:-1]==pd.Timedelta(minutes=15)).all():raise ValueError('Missing candles')
    if len(sig)!=len(d) or not np.isin(sig[np.isfinite(sig)],[-1,0,1]).all():raise ValueError('Invalid signal')
    if dates[-1]>=pd.Timestamp(config['performance_end_exclusive']):raise ValueError('Unapproved performance period')
    first=int(dates.searchsorted(pd.Timestamp(config['performance_start'])))
    if first>=len(d):raise ValueError('Empty interval')
    op=d.open.to_numpy(float);cl=d.close.to_numpy(float);hi=d.high.to_numpy(float);lo=d.low.to_numpy(float)
    zero=d.volume.to_numpy()==0
    if not np.isfinite(np.column_stack([op,cl,hi,lo])).all():raise ValueError('Invalid OHLC')
    initial=float(config['initial_account']);notional=float(config['notional']);fee_rate=float(config['fee']);slip=float(config['slippage'])
    candidate='B_ols_N64' # retain frozen S0 IDs; scenario is a separate column in caller
    cash=initial;side=0;qty=0.;entry=None;side_cash={1:0.,-1:0.};fees=0.;slippage=0.
    trades=[];executions=[];rows=[]
    def event(i,action,s,raw,fill,quantity,fee,slip_cost,account,reason,precision,source,order_available):
        executions.append(dict(candidate=candidate,method='B_ols',N=64,time=dates[i],source_signal_bar=source,
            action=action,direction='long' if s==1 else 'short',raw_price=raw,fill_price=fill,
            quantity=quantity,notional=quantity*fill,fee=fee,slippage_usdt=slip_cost,
            zero_volume=bool(zero[i]),account_before_fill=account,
            margin_budget_shortfall=bool(action=='open' and account<config['margin_budget']+fee),
            execution_uncertainty=('zero-volume candle; hypothetical fill not exchange-verified' if zero[i] else 'OHLCV open-price fill assumption') if precision=='open' else
                ('zero-volume candle; stop touch and fill hypothetical' if zero[i] else '15m stop touch; exact intrabar time unknown'),
            reason=reason,time_precision=precision,order_available_at=order_available,event_sequence=len(executions)))
    def close(i,raw,reason,precision):
        nonlocal cash,side,qty,entry,fees,slippage
        fill=raw*(1-side*slip);fee=qty*fill*fee_rate;sc=qty*abs(fill-raw)
        account=cash+side*qty*(raw-entry['entry_price'])
        source=dates[i-1] if reason=='signal' else pd.NaT
        available=dates[i] if reason=='signal' else entry['entry_time']
        event(i,'close',side,raw,fill,qty,fee,sc,account,reason,precision,source,available)
        gross=side*qty*(fill-entry['entry_price']);pnl=gross-fee
        cash+=pnl;side_cash[side]+=pnl;fees+=fee;slippage+=sc
        lower=(dates[i]-entry['entry_time']).total_seconds()/3600
        trades.append(dict(entry,exit_source_bar=source,exit_signal_available_at=dates[i] if reason=='signal' else pd.NaT,
            exit_time=dates[i],exit_raw_price=raw,exit_price=fill,exit_fee=fee,exit_slippage_usdt=sc,
            gross_before_cost=side*qty*(raw-entry['entry_raw_price']),gross_after_slippage=gross,
            net_pnl=gross-fee-entry['entry_fee'],holding_hours=lower,
            holding_hours_upper=lower+(.25 if precision!='open' else 0.),
            exit_zero_volume=bool(zero[i]),status='closed',exit_reason=reason,exit_time_precision=precision,
            stop_price=entry['fixed_stop_price'],trade_sequence=len(trades)))
        side=0;qty=0.;entry=None
    for i in range(first,len(d)):
        target=sig[i-1] if i else np.nan
        gap_stopped=False
        if side and stop_loss is not None:
            stop=entry['fixed_stop_price']
            if (side==1 and op[i]<=stop) or (side==-1 and op[i]>=stop):
                close(i,op[i],'stop_gap','open');gap_stopped=True
        if not gap_stopped and np.isfinite(target) and target!=side:
            if side:close(i,op[i],'signal','open')
            if target:
                side=int(target);raw=op[i];fill=raw*(1+side*slip);qty=notional/fill
                fee=notional*fee_rate;sc=qty*abs(fill-raw)
                event(i,'open',side,raw,fill,qty,fee,sc,cash,'signal','open',dates[i-1],dates[i])
                cash-=fee;side_cash[side]-=fee;fees+=fee;slippage+=sc
                entry=dict(candidate=candidate,method='B_ols',N=64,direction='long' if side==1 else 'short',
                    entry_source_bar=dates[i-1],entry_signal_available_at=dates[i],entry_time=dates[i],
                    entry_raw_price=raw,entry_price=fill,quantity=qty,entry_fee=fee,
                    entry_slippage_usdt=sc,entry_zero_volume=bool(zero[i]),
                    fixed_stop_price=fill*(1-side*stop_loss) if stop_loss is not None else np.nan)
        # Open exits precede current candle extremes. Stops active on entry candle.
        if side and stop_loss is not None:
            stop=entry['fixed_stop_price']
            if (side==1 and lo[i]<=stop) or (side==-1 and hi[i]>=stop):close(i,stop,'stop_touch','within_15m_unknown')
        mark=side*qty*(cl[i]-entry['entry_price']) if side else 0.
        rows.append((dates[i],cash+mark,side_cash[1]+(mark if side==1 else 0.),
            side_cash[-1]+(mark if side==-1 else 0.),fees,slippage,side))
    if side:
        mark=side*qty*(cl[-1]-entry['entry_price'])
        held=(dates[-1]+pd.Timedelta(minutes=15)-entry['entry_time']).total_seconds()/3600
        trades.append(dict(entry,exit_source_bar=pd.NaT,exit_signal_available_at=pd.NaT,exit_time=pd.NaT,
            exit_raw_price=np.nan,exit_price=np.nan,exit_fee=0.,exit_slippage_usdt=0.,
            gross_before_cost=side*qty*(cl[-1]-entry['entry_raw_price']),gross_after_slippage=mark,
            net_pnl=mark-entry['entry_fee'],holding_hours=held,holding_hours_upper=held,
            exit_zero_volume=False,status='open_marked_at_training_end',exit_reason='training_end_mark',
            exit_time_precision='unclosed',stop_price=entry['fixed_stop_price'],trade_sequence=len(trades)))
    c=pd.DataFrame(rows,columns=['date','equity','long_pnl','short_pnl','fees','slippage_usdt','position'])
    c['pnl_change']=np.diff(np.r_[initial,c.equity.to_numpy()])
    for field in ['long_pnl','short_pnl','fees','slippage_usdt']:c[field+'_change']=np.diff(np.r_[0.,c[field].to_numpy()])
    peak=np.maximum.accumulate(np.r_[initial,c.equity])[1:]
    c['drawdown_usdt']=peak-c.equity;c['drawdown_pct_peak']=100*(peak-c.equity)/peak
    c['candidate']=candidate;c['method']='B_ols';c['N']=64
    extra=['fixed_stop_price','holding_hours_upper','exit_reason','exit_time_precision','stop_price','trade_sequence']
    ex=['reason','time_precision','order_available_at','event_sequence']
    return pd.DataFrame(trades,columns=TRADE_COLUMNS+extra),c,pd.DataFrame(executions,columns=EXEC_COLUMNS+ex)
