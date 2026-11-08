"""Daily fixed-initial-capital risk statistics and transparent MTM accounting."""
import numpy as np
import pandas as pd

def summarize(c,t,e,config,start=None,end=None,label='training_2020_2021'):
    start=pd.Timestamp(start or config['performance_start']);end=pd.Timestamp(end or config['performance_end_exclusive'])
    g=c[(c.date>=start)&(c.date<end)]
    if g.empty:raise ValueError('Empty metrics interval')
    closed=t[(t.status=='closed')&(t.exit_time>=start)&(t.exit_time<end)]
    fills=e[(e.time>=start)&(e.time<end)]
    daily=g.set_index('date').pnl_change.resample('1D').sum()/config['initial_account']
    sd=daily.std(ddof=1);downside=np.sqrt(np.mean(np.minimum(daily,0.)**2))
    win=closed.loc[closed.net_pnl>0,'net_pnl'].sum();loss=-closed.loc[closed.net_pnl<0,'net_pnl'].sum()
    eq=config['initial_account']+g.pnl_change.cumsum().to_numpy()
    peak=np.maximum.accumulate(np.r_[config['initial_account'],eq])[1:]
    monthly=g.set_index('date').pnl_change.resample('MS').sum()
    row=dict(candidate=g.candidate.iloc[0],method=g.method.iloc[0],N=int(g.N.iloc[0]),period=label,
        long_net_profit=g.long_pnl_change.sum(),short_net_profit=g.short_pnl_change.sum(),net_profit=g.pnl_change.sum(),
        return_pct_initial=g.pnl_change.sum()/config['initial_account']*100,
        max_drawdown_usdt=float(np.max(peak-eq)),max_drawdown_pct_peak=float(np.max(100*(peak-eq)/peak)),
        max_drawdown_pct_initial=float(np.max(peak-eq)/config['initial_account']*100),
        profit_factor=float(win/loss) if loss else (float('inf') if win else np.nan),
        sharpe=float(np.sqrt(365)*daily.mean()/sd) if sd else np.nan,
        sortino=float(np.sqrt(365)*daily.mean()/downside) if downside else np.nan,
        win_rate=float((closed.net_pnl>0).mean()) if len(closed) else np.nan,
        trade_count=len(closed),entry_count=int(fills.action.eq('open').sum()),execution_count=len(fills),
        average_holding_hours=float(closed.holding_hours.mean()),monthly_profit_ratio=float((monthly>0).mean()),
        months=len(monthly),fees=g.fees_change.sum(),slippage_usdt=g.slippage_usdt_change.sum(),
        gross_before_cost=g.pnl_change.sum()+g.fees_change.sum()+g.slippage_usdt_change.sum(),
        zero_volume_execution_count=int(fills.zero_volume.sum()),min_actual_equity=float(g.equity.min()),
        margin_budget_shortfall_entries=int(fills.margin_budget_shortfall.sum()),
        first_margin_shortfall_time=str(fills.loc[fills.margin_budget_shortfall,'time'].iloc[0]) if fills.margin_budget_shortfall.any() else '',
        first_nonpositive_equity_time=str(g.loc[g.equity<=0,'date'].iloc[0]) if (g.equity<=0).any() else '',
        funded_execution_feasible=bool(not fills.margin_budget_shortfall.any() and (g.equity>0).all()),
        gross_closed_profits=float(win),gross_closed_losses=float(loss))
    winners=closed[closed.net_pnl>0].net_pnl.sort_values(ascending=False)
    for k in [1,5,10]:
        top=winners.iloc[:k].sum()
        row[f'top{k}_profit_usdt']=top
        row[f'top{k}_share_winning_profit']=top/win if win else np.nan
        row[f'top{k}_share_net_profit']=top/row['net_profit'] if row['net_profit']>0 else np.nan
        row[f'net_without_top{k}_winning_trades']=row['net_profit']-top
    return row

def neighborhood(table,annual):
    rows=[]
    for method,g in table.groupby('method',sort=False):
        g=g.sort_values('N').reset_index(drop=True)
        for i,r in g.iterrows():
            neighbors=g.iloc[max(0,i-1):min(len(g),i+2)]
            years=annual[annual.candidate==r.candidate]
            row=dict(candidate=r.candidate,method=method,N=int(r.N),neighbor_N=','.join(map(str,neighbors.N)),
                neighborhood_count=len(neighbors),net_profit=r.net_profit,
                neighborhood_net_min=neighbors.net_profit.min(),neighborhood_net_max=neighbors.net_profit.max(),
                neighborhood_net_range=neighbors.net_profit.max()-neighbors.net_profit.min(),
                neighborhood_profitable_fraction=float((neighbors.net_profit>0).mean()),
                neighborhood_min_profit_factor=neighbors.profit_factor.min(),
                neighborhood_worst_drawdown_usdt=neighbors.max_drawdown_usdt.max(),
                both_training_years_positive=bool((years.net_profit>0).all()),
                worst_year_profit=years.net_profit.min(),best_year_profit=years.net_profit.max(),
                long_short_profit_gap=abs(r.long_net_profit-r.short_net_profit),
                strict_interior_local_net_peak=bool(i>0 and i<len(g)-1 and r.net_profit>g.iloc[i-1].net_profit and r.net_profit>g.iloc[i+1].net_profit),
                endpoint_caution=bool(i==0 or i==len(g)-1))
            rows.append(row)
    return pd.DataFrame(rows)
