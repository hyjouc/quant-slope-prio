"""Independent next-open fixed-notional ledger, written for endpoint/OLS S0."""
import numpy as np
import pandas as pd

TRADE_COLUMNS=['candidate','method','N','direction','entry_source_bar','entry_signal_available_at',
    'entry_time','entry_raw_price','entry_price','quantity','entry_fee','entry_slippage_usdt',
    'exit_source_bar','exit_signal_available_at','exit_time','exit_raw_price','exit_price',
    'exit_fee','exit_slippage_usdt','gross_before_cost','gross_after_slippage','net_pnl',
    'holding_hours','entry_zero_volume','exit_zero_volume','status']
EXEC_COLUMNS=['candidate','method','N','time','source_signal_bar','action','direction','raw_price',
    'fill_price','quantity','notional','fee','slippage_usdt','zero_volume','account_before_fill',
    'margin_budget_shortfall','execution_uncertainty']

def simulate(d,signals,method,n,config):
    dates=pd.DatetimeIndex(d.date)
    if dates.tz is None or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Require unique chronological UTC bars')
    if len(d)>1 and not (dates[1:]-dates[:-1]==pd.Timedelta(minutes=15)).all():
        raise ValueError('Refuse execution across missing bars')
    if dates[-1]>=pd.Timestamp(config['performance_end_exclusive']):
        raise ValueError('Execution input includes unapproved performance period')
    sig=np.asarray(signals,dtype=float)
    if len(sig)!=len(d) or not np.isin(sig[np.isfinite(sig)],[-1,0,1]).all():
        raise ValueError('Invalid signal array')
    first=int(dates.searchsorted(pd.Timestamp(config['performance_start'])))
    if first>=len(d):raise ValueError('No performance data')
    op=d.open.to_numpy(float);cl=d.close.to_numpy(float);zero=d.volume.to_numpy()==0
    initial=float(config['initial_account']);notional=float(config['notional'])
    fee_rate=float(config['fee']);slip=float(config['slippage'])
    candidate=f'{method}_N{n}';cash=initial;side=0;qty=0.;entry=None
    # Arrays store incremental per-direction net attribution and incurred costs.
    side_cash={1:0.,-1:0.};fees=0.;slippage=0.;trades=[];executions=[];rows=[]
    def fill_record(i,action,s,raw,fill,quantity,fee,slip_cost,account):
        shortfall=action=='open' and account<config['margin_budget']+fee
        executions.append(dict(candidate=candidate,method=method,N=n,time=dates[i],source_signal_bar=dates[i-1],
            action=action,direction='long' if s==1 else 'short',raw_price=raw,fill_price=fill,
            quantity=quantity,notional=quantity*fill,fee=fee,slippage_usdt=slip_cost,
            zero_volume=bool(zero[i]),account_before_fill=account,margin_budget_shortfall=bool(shortfall),
            execution_uncertainty='zero-volume candle; hypothetical fill not exchange-verified' if zero[i] else 'OHLCV open-price fill assumption'))
    for i in range(first,len(d)):
        target=sig[i-1] if i else np.nan
        if np.isfinite(target) and target!=side:
            if side:
                raw=op[i];fill=raw*(1-side*slip);fee=qty*fill*fee_rate
                sc=qty*abs(fill-raw);account=cash+side*qty*(raw-entry['entry_price'])
                fill_record(i,'close',side,raw,fill,qty,fee,sc,account)
                gross=side*qty*(fill-entry['entry_price']);pnl=gross-fee
                cash+=pnl;side_cash[side]+=pnl;fees+=fee;slippage+=sc
                trades.append(dict(entry,exit_source_bar=dates[i-1],exit_signal_available_at=dates[i],
                    exit_time=dates[i],exit_raw_price=raw,exit_price=fill,exit_fee=fee,exit_slippage_usdt=sc,
                    gross_before_cost=side*qty*(raw-entry['entry_raw_price']),gross_after_slippage=gross,
                    net_pnl=gross-fee-entry['entry_fee'],holding_hours=(dates[i]-entry['entry_time']).total_seconds()/3600,
                    exit_zero_volume=bool(zero[i]),status='closed'))
                side=0;qty=0.;entry=None
            if target:
                side=int(target);raw=op[i];fill=raw*(1+side*slip);qty=notional/fill
                fee=notional*fee_rate;sc=qty*abs(fill-raw)
                fill_record(i,'open',side,raw,fill,qty,fee,sc,cash)
                cash-=fee;side_cash[side]-=fee;fees+=fee;slippage+=sc
                entry=dict(candidate=candidate,method=method,N=n,direction='long' if side==1 else 'short',
                    entry_source_bar=dates[i-1],entry_signal_available_at=dates[i],entry_time=dates[i],
                    entry_raw_price=raw,entry_price=fill,quantity=qty,entry_fee=fee,
                    entry_slippage_usdt=sc,entry_zero_volume=bool(zero[i]))
        mark=side*qty*(cl[i]-entry['entry_price']) if side else 0.
        rows.append((dates[i],cash+mark,side_cash[1]+(mark if side==1 else 0.),
            side_cash[-1]+(mark if side==-1 else 0.),fees,slippage,side))
    if side:
        mark=side*qty*(cl[-1]-entry['entry_price'])
        trades.append(dict(entry,exit_source_bar=pd.NaT,exit_signal_available_at=pd.NaT,
            exit_time=pd.NaT,exit_raw_price=np.nan,exit_price=np.nan,exit_fee=0.,exit_slippage_usdt=0.,
            gross_before_cost=side*qty*(cl[-1]-entry['entry_raw_price']),gross_after_slippage=mark,
            net_pnl=mark-entry['entry_fee'],holding_hours=(dates[-1]+pd.Timedelta(minutes=15)-entry['entry_time']).total_seconds()/3600,
            exit_zero_volume=False,status='open_marked_at_training_end'))
    c=pd.DataFrame(rows,columns=['date','equity','long_pnl','short_pnl','fees','slippage_usdt','position'])
    c['pnl_change']=np.diff(np.r_[initial,c.equity.to_numpy()])
    for field in ['long_pnl','short_pnl','fees','slippage_usdt']:
        c[field+'_change']=np.diff(np.r_[0.,c[field].to_numpy()])
    peak=np.maximum.accumulate(np.r_[initial,c.equity])[1:]
    c['drawdown_usdt']=peak-c.equity
    c['drawdown_pct_peak']=100*(peak-c.equity)/peak
    c['candidate']=candidate;c['method']=method;c['N']=n
    return pd.DataFrame(trades,columns=TRADE_COLUMNS),c,pd.DataFrame(executions,columns=EXEC_COLUMNS)
