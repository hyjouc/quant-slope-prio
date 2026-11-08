"""S1.1 existing-ledger anatomy; no estimator/backtester invocation or parameter search."""
from pathlib import Path
import sys,json,logging,subprocess,platform,importlib.metadata
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data_loader import sha256
from src.trade_anatomy import (excursions,distributions,duration_analysis,loss_streaks,pnl_summary,
    reconstruct_curves,risk_analysis)

OUT=Path(__file__).parent;S0=ROOT/'experiments/s0_from_scratch_slope'
BASE='fd6658dbde9d046488577062dc53b84ef621c4bf'
CANDIDATES=['B_ols_N64','B_ols_N96']

def frame_hash(d):
    import hashlib
    return hashlib.sha256(d.to_csv(index=False,float_format='%.14g',lineterminator='\n').encode()).hexdigest()

def verify_s0():
    manifest=json.loads((S0/'artifact_manifest.json').read_text())
    for path,item in manifest.items():
        if sha256(ROOT/path)!=item['sha256']:raise ValueError('S0 artifact changed: '+path)
    # Also pin the manifest itself, all S0 outputs/config/source files to the named Git commit.
    paths=[str(p.relative_to(ROOT)).replace('\\','/') for p in S0.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths+=['config/s0_from_scratch.json']
    for path in paths:
        saved=subprocess.check_output(['git','show',BASE+':'+path])
        if (ROOT/path).read_bytes()!=saved:raise ValueError('S0 byte mismatch against base commit: '+path)
    return {p:sha256(ROOT/p) for p in paths}

def load_inputs(cfg):
    s0_hashes=verify_s0()
    if sha256(cfg['data_path'])!=cfg['data_sha256']:raise ValueError('Raw data SHA256 mismatch: STOP')
    # Only six raw columns; no features. Discard 2022/2023 BEFORE any analysis.
    raw=pd.read_feather(cfg['data_path'],columns=['date','open','high','low','close','volume'])
    raw['date']=pd.to_datetime(raw.date,utc=True)
    bars=raw.loc[(raw.date>=pd.Timestamp('2020-01-01',tz='UTC'))&(raw.date<pd.Timestamp('2022-01-01',tz='UTC'))].copy().reset_index(drop=True)
    del raw
    expected=pd.date_range('2020-01-01','2021-12-31 23:45',freq='15min',tz='UTC')
    if not np.array_equal(bars.date.array,expected.array):raise ValueError('Training raw dates incomplete')
    all_trades=pd.read_csv(S0/'trades_all.csv')
    t=all_trades[all_trades.candidate.isin(CANDIDATES)].copy().reset_index(drop=True)
    for field in ['entry_time','exit_time','entry_source_bar','entry_signal_available_at','exit_source_bar','exit_signal_available_at']:
        t[field]=pd.to_datetime(t[field],utc=True)
    for field in ['entry_time','exit_time']:
        if (t[field].dropna()>=pd.Timestamp('2022-01-01',tz='UTC')).any():raise ValueError('Holdout trade detected')
    eq=pd.read_csv(S0/'equity_curves.csv',usecols=['date']+CANDIDATES);eq['date']=pd.to_datetime(eq.date,utc=True)
    if not np.array_equal(eq.date.array,bars.date.iloc[96:].array):raise ValueError('S0 equity clock mismatch')
    summary=pd.read_csv(S0/'parameter_comparison.csv').set_index('candidate')
    events=pd.read_csv(S0/'executions_all.csv');events=events[events.candidate.isin(CANDIDATES)]
    reference_drawdown=pd.read_csv(S0/'drawdown_curves.csv',usecols=['date']+CANDIDATES)
    for candidate in CANDIDATES:
        g=t[t.candidate==candidate];s=summary.loc[candidate];e=events[events.candidate==candidate]
        if len(g[g.status=='closed'])!=int(s.trade_count):raise ValueError('Closed-trade count mismatch')
        np.testing.assert_allclose(g.net_pnl.sum(),s.net_profit,atol=1e-6,rtol=0)
        np.testing.assert_allclose(g.entry_fee.sum()+g.exit_fee.sum(),s.fees,atol=1e-6,rtol=0)
        np.testing.assert_allclose(e.fee.sum(),s.fees,atol=1e-6,rtol=0)
        np.testing.assert_allclose(g.entry_slippage_usdt.sum()+g.exit_slippage_usdt.sum(),s.slippage_usdt,atol=1e-6,rtol=0)
        np.testing.assert_allclose(g.gross_after_slippage-g.entry_fee-g.exit_fee,g.net_pnl,atol=1e-7,rtol=0)
        np.testing.assert_allclose(g.quantity*g.entry_price,2000,atol=1e-7,rtol=0)
        for r in g.itertuples():
            sgn=1 if r.direction=='long' else -1
            np.testing.assert_allclose(r.entry_price,r.entry_raw_price*(1+sgn*.0002),atol=1e-8,rtol=0)
            if r.status=='closed':
                np.testing.assert_allclose(r.exit_price,r.exit_raw_price*(1-sgn*.0002),atol=1e-8,rtol=0)
                np.testing.assert_allclose(r.exit_fee,r.quantity*r.exit_price*.0005,atol=1e-8,rtol=0)
        np.testing.assert_allclose(eq[candidate].iloc[-1]-10000,s.net_profit,atol=1e-6,rtol=0)
    curves=reconstruct_curves(t,bars,eq)
    for candidate in CANDIDATES:
        c=curves[(curves.candidate==candidate)&(curves.direction=='all')]
        np.testing.assert_allclose(c.drawdown_usdt,reference_drawdown[candidate],atol=1e-6,rtol=0)
    return bars,t,curves,s0_hashes

def compute(t,bars,curves):
    level=excursions(t,bars)
    np.testing.assert_allclose(level.path_bar_count*.25,level.holding_hours,atol=1e-9,rtol=0)
    closed=level[level.status=='closed'].copy();opened=level[level.status!='closed'].copy()
    streak=loss_streaks(closed);side,concentration=pnl_summary(closed,streak)
    risk,periods,episodes=risk_analysis(curves,streak,closed)
    counts=streak.groupby(['candidate','direction','losses'],sort=False).agg(streak_count=('streak_id','size'),
        total_loss_usdt=('net_loss','sum'),worst_streak_net=('net_loss','min')).reset_index()
    mae=distributions(closed,['mae_pct','mfe_pct','mae_usdt','mfe_usdt','net_pnl','holding_hours'])
    give=distributions(closed,['mfe_usdt','terminal_raw_gross','final_gross_after_slippage','net_pnl',
        'giveback_price_usdt','giveback_price_ratio','giveback_execution_usdt','giveback_execution_ratio'])
    holding=duration_analysis(closed)
    # Top10/20 flags describe original winners only; no deletion/reoptimization.
    level['top10_closed_winner']=False;level['top20_closed_winner']=False
    for candidate in CANDIDATES:
        winners=closed[(closed.candidate==candidate)&(closed.net_pnl>0)]
        for k in [10,20]:level.loc[winners.nlargest(k,'net_pnl').index,f'top{k}_closed_winner']=True
    # Reconcile overlapping cohort reports using only disjoint slices.
    for candidate in CANDIDATES:
        c=closed[closed.candidate==candidate];h=holding[(holding.candidate==candidate)&(holding.direction=='all')]
        assert h['count'].sum()==len(c)
        np.testing.assert_allclose(h.net_profit.sum(),c.net_pnl.sum(),atol=1e-7,rtol=0)
        s=side[(side.candidate==candidate)&(side.direction!='all')]
        assert s.trade_count.sum()==len(c)
        np.testing.assert_allclose(s.closed_net_profit.sum(),c.net_pnl.sum(),atol=1e-7,rtol=0)
        runs=streak[(streak.candidate==candidate)&(streak.direction=='all')]
        assert runs.losses.sum()==int(c.net_pnl.lt(0).sum())
        np.testing.assert_allclose(runs.net_loss.sum(),c.loc[c.net_pnl<0,'net_pnl'].sum(),atol=1e-7,rtol=0)
    return dict(trade_level_metrics=level,open_positions=opened,mae_mfe_analysis=mae,
        holding_duration_analysis=holding,profit_giveback_analysis=give,long_short_comparison=side,
        profit_concentration=concentration,loss_cluster_analysis=risk,loss_streaks=streak,
        loss_streak_distribution=counts,period_pnl=periods,drawdown_episodes=episodes,anatomy_equity_curves=curves)

def charts(tables):
    path=OUT/'figures';path.mkdir(exist_ok=True)
    closed=tables['trade_level_metrics'].query("status=='closed'")
    colors={'winner':'#0072B2','loser':'#D55E00','breakeven':'#777777'}
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
        for row,candidate in enumerate(CANDIDATES):
            for col,direction in enumerate(['long','short']):
                ax=axs[row,col];g=closed[(closed.candidate==candidate)&(closed.direction==direction)]
                for outcome,marker in [('winner','o'),('loser','x')]:
                    z=g[g.outcome==outcome];ax.scatter(z.mae_pct,z.mfe_pct,s=14,alpha=.5,color=colors[outcome],marker=marker,label=outcome)
                ax.set(title=f'{candidate} {direction}',xlabel='MAE (%)',ylabel='MFE (%)');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.savefig(path/'mae_mfe_scatter.png',dpi=170);plt.close(fig)
        fig,axs=plt.subplots(2,1,figsize=(12,8),layout='constrained')
        bins=np.linspace(closed.net_pnl.min(),closed.net_pnl.max(),51)
        for ax,candidate in zip(axs,CANDIDATES):
            for outcome,hatch in [('winner','//'),('loser','..')]:
                z=closed[(closed.candidate==candidate)&(closed.outcome==outcome)]
                ax.hist(z.net_pnl,bins=bins,label=outcome,alpha=.6,color=colors[outcome],hatch=hatch)
            ax.set(title=candidate,xlabel='Actual closed net PnL (USDT); full range',ylabel='Trades (log count)',yscale='log');ax.legend();ax.grid(alpha=.2)
        fig.savefig(path/'pnl_distribution.png',dpi=170);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        labels=['[0,1)','[1,2)','[2,4)','[4,8)','[8,16)','[16,32)','[32,inf)']
        for ax,candidate in zip(axs,CANDIDATES):
            g=tables['holding_duration_analysis'].query("candidate==@candidate and direction=='all'")
            x=np.arange(7);ax.bar(x-.18,g.winners,.36,label='Winner',hatch='//');ax.bar(x+.18,g.losers,.36,label='Loser',hatch='..')
            ax.set_xticks(x,labels,rotation=35);ax.set(title=candidate,xlabel='Holding hours, fixed bins',ylabel='Closed trades');ax.legend()
        fig.savefig(path/'holding_duration.png',dpi=170);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for ax,candidate in zip(axs,CANDIDATES):
            g=closed[closed.candidate==candidate]
            for direction,marker,color in [('long','o','#0072B2'),('short','x','#D55E00')]:
                z=g[g.direction==direction];ax.scatter(z.mfe_usdt,z.net_pnl,s=14,alpha=.5,marker=marker,color=color,label=direction)
            limit=g.mfe_usdt.max();ax.plot([0,limit],[0,limit],'--',color='gray',label='Net = MFE reference (not realizable)')
            ax.axhline(0,color='black',linewidth=.6);ax.set(title=candidate,xlabel='Maximum favorable gross excursion (USDT)',ylabel='Actual closed net PnL (USDT)');ax.legend(fontsize=7)
        fig.savefig(path/'mfe_actual_profit.png',dpi=170);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained',sharey=True)
        for ax,candidate in zip(axs,CANDIDATES):
            g=closed[closed.candidate==candidate]
            ax.boxplot([g[g.direction==side].net_pnl for side in ['long','short']],tick_labels=['Long','Short'],showfliers=True)
            ax.axhline(0,color='gray',linewidth=.6);ax.set(title=candidate,ylabel='Actual net PnL (USDT), all outliers shown')
        fig.savefig(path/'long_short_distribution.png',dpi=170);plt.close(fig)
        fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
        for row,candidate in enumerate(CANDIDATES):
            s=tables['loss_streaks'].query("candidate==@candidate and direction=='all'")
            ax=axs[row,0];ax.vlines(s.last_exit,0,s.losses,linewidth=.9);ax.set(title=candidate,xlabel='Loss-run final exit (UTC)',ylabel='Consecutive net losses')
            ax=axs[row,1]
            for direction,style in [('all','-'),('long','--'),('short',':')]:
                c=tables['anatomy_equity_curves'].query('candidate==@candidate and direction==@direction')
                ax.plot(c.date+pd.Timedelta(minutes=15),c.drawdown_usdt,label=direction,linestyle=style,linewidth=.8)
            ax.set(title=candidate,xlabel='Actual close time (UTC)',ylabel='MTM drawdown (USDT)');ax.legend(fontsize=8)
        fig.savefig(path/'loss_runs_drawdowns.png',dpi=170);plt.close(fig)

def md(table,columns):
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,r in table.iterrows():
        values=[]
        for col in columns:
            v=r[col];values.append(f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)

def make_report(tables,meta):
    level=tables['trade_level_metrics'];closed=level[level.status=='closed'];opened=tables['open_positions']
    side=tables['long_short_comparison'];con=tables['profit_concentration'];risk=tables['loss_cluster_analysis']
    # Compact cohort evidence; all rows trace to exact underlying trade IDs.
    cohorts=[]
    for (candidate,direction,outcome),g in closed.groupby(['candidate','direction','outcome'],sort=False):
        cohorts.append(dict(candidate=candidate,direction=direction,outcome=outcome,count=len(g),net=g.net_pnl.sum(),
            mean_mae_pct=g.mae_pct.mean(),median_mae_pct=g.mae_pct.median(),mean_mfe_pct=g.mfe_pct.mean(),
            mean_hours=g.holding_hours.mean(),median_hours=g.holding_hours.median(),
            mean_price_giveback=g.giveback_price_usdt.mean(),median_giveback_ratio=g.giveback_price_ratio.median(),
            p90_mae_pct=g.mae_pct.quantile(.9),p90_mfe_pct=g.mfe_pct.quantile(.9),
            positive_mfe_count=int(g.mfe_usdt.gt(0).sum()),gross_positive_net_loss_count=int(g.gross_positive_net_loss.sum()),
            max_mae_pct=g.mae_pct.max()))
    cohort=pd.DataFrame(cohorts);cohort.to_csv(OUT/'cohort_summary.csv',index=False,float_format='%.12g')
    answers=[]
    for candidate in CANDIDATES:
        g=closed[closed.candidate==candidate];w=g[g.net_pnl>0];l=g[g.net_pnl<0]
        r=risk[(risk.candidate==candidate)&(risk.direction=='all')].iloc[0]
        c=con[(con.candidate==candidate)&(con.direction=='all')].set_index('top_k')
        summary=side[(side.candidate==candidate)&(side.direction=='all')].iloc[0]
        open_net=opened.loc[opened.candidate==candidate,'net_pnl'].sum()
        top=g.nlargest(10,'net_pnl')
        long_losses=l[l.holding_hours>=32]
        answers.append(f'''### {candidate}

- 已平仓{len(g)}笔，盈利{len(w)}、亏损{len(l)}、持平{int(g.net_pnl.eq(0).sum())}；已平仓净收益{g.net_pnl.sum():.2f} USDT，末期未平仓MTM {open_net:.2f} USDT，合计{g.net_pnl.sum()+open_net:.2f} USDT。PF={summary.profit_factor:.3f}，平均盈利{summary.mean_win:.2f}、平均亏损{summary.mean_loss:.2f}、盈亏比{summary.payoff_ratio:.3f}。
- 盈利平均/中位持仓{w.holding_hours.mean():.2f}/{w.holding_hours.median():.2f}小时，亏损{l.holding_hours.mean():.2f}/{l.holding_hours.median():.2f}小时；盈利平均MAE {w.mae_pct.mean():.3f}%、亏损{l.mae_pct.mean():.3f}%，两类分布存在重叠，不能把均值当可用止损线。
- {int(l.mfe_usdt.gt(0).sum())}/{len(l)}笔亏损交易曾有正的毛MFE，但正MFE不代表曾出现可扣费兑现的净利润。仅{int(g.gross_positive_net_loss.sum())}笔是实际退出毛收益为正但扣双边手续费后转亏，故亏损不能只归因于手续费。
- 持仓≥32小时的已平仓亏损交易{len(long_losses)}笔，合计{long_losses.net_pnl.sum():.2f} USDT；不等于这些仓位可通过时间退出避免。前10大赢家中最大MAE为{top.mae_pct.max():.3f}%，显示简单风险截断也可能伤及大赢家。
- 前10笔盈利合计{c.loc[10,'top_net_profit']:.2f} USDT，占所有盈利金额{100*c.loc[10,'share_winning_profit']:.2f}%；从已平仓净收益扣除后{c.loc[10,'closed_net_without_top']:.2f} USDT。加回未平仓MTM后{c.loc[10,'closed_net_without_top']+open_net:.2f} USDT（与S0口径对齐）。
- 全部盈利交易价格回吐均值{w.giveback_price_usdt.mean():.2f} USDT、回吐比例中位数{100*w.giveback_price_ratio.median():.2f}%；前10大赢家价格回吐均值{c.loc[10,'top_mean_price_giveback_usdt']:.2f} USDT、比例中位数{100*c.loc[10,'top_median_price_giveback_ratio']:.2f}%。这些回吐不是可直接兑现的额外利润。
- 最大连亏{int(r.max_consecutive_losses)}笔、最差连亏金额{r.max_consecutive_loss_usdt:.2f} USDT；最差日{r.worst_day_pnl:.2f}（{r.worst_day_period}）、周{r.worst_week_pnl:.2f}（{r.worst_week_period}）、月{r.worst_month_pnl:.2f}（{r.worst_month_period}）。
- 最大收盘回撤{r.max_drawdown_usdt:.2f} USDT；最长水下时长{r.max_drawdown_duration_days:.2f}天，最大深度那次回撤历时{r.deepest_dd_duration_days:.2f}天，是否已修复={r.deepest_dd_recovered}。{int(r.losing_months)}/{int(r.total_months)}个月MTM亏损；最差3个亏损月占全部亏损月绝对损失{100*r.worst3_months_loss_share:.2f}%。
''')
    duration=tables['holding_duration_analysis'].query("direction=='all'")
    concentration=con.query("direction=='all'")
    top20=level[level.top20_closed_winner].sort_values(['candidate','net_pnl'],ascending=[True,False])
    top20.to_csv(OUT/'top20_winner_paths.csv',index=False,float_format='%.12g')
    text=f'''# S1.1 Trade Anatomy：2020—2021现有斜率交易行为

## 审计范围与口径

基准S0提交 `{BASE}`，只研究B_ols_N64（主）及B_ols_N96（辅）。不改变S0任何文件、公式、N、规则或结果；不调用估计器或执行器，不计算2022/2023指标、交易或绩效，未读取2024+。本轮只是历史探索性交易描述，四年历史已被研究使用过。
S0原产物清单、Git基准逐字节对照通过；两组交易数量、净收益、费用、滑点、每根权益及每根回撤对账通过，不需要重建交易。金额差异容差1e-6 USDT仅用于S0 CSV的小数序列化误差。
两个候选是研究对象，不是已冻结最终策略。记录中的零成交量假设成交（两组共{meta['zero_volume_executions']}项）仍有执行不确定性。

{chr(10).join(answers)}

## 1. MAE/MFE：从实际入场成交价出发

MAE与MFE以实际入场fill为分母；多头看最低/最高、空头看最高/最低，百分比乘100，USDT=比例×实际入场价×实际数量。路径含入场fill锚点，故MAE/MFE非负；费用不混入价格波动。
使用区间 **[入场K线,退出K线)** 内全部OHLC极值，并将退出原始open加入终点；退出K线其后的high/low/close完全不使用。另保留实际退出fill和退出滑点。入场bar的高低价可用，因为入场在其open；不假定同根high/low先后。
若MFE出现在MAE之前或之后，OHLC不能区分；本报告不推断“先扛亏再盈利”、盘中触发次序或一条连续盘中路径。
期末持仓只到训练末根close，单列open_positions.csv，不进入以下任何已平仓分布、连亏或方向交易统计；账户MTM风险则包含这部分浮动盈亏。

{md(cohort,['candidate','direction','outcome','count','mean_mae_pct','median_mae_pct','mean_mfe_pct','p90_mae_pct','p90_mfe_pct'])}

亏损形成的两类可观测现象（毛MFE只说明有利极值，不说明可成交净浮盈）：

{md(cohort.query("outcome=='loser'"),['candidate','direction','count','positive_mfe_count','gross_positive_net_loss_count','max_mae_pct'])}

完整分布及分位数见mae_mfe_analysis.csv；每笔路径索引、端点及USDT见trade_level_metrics.csv。

## 2. 持仓时间

盈利形成时间与亏损持仓时间的均值/中位数：

{md(cohort,['candidate','direction','outcome','count','mean_hours','median_hours','net'])}

固定分组左闭右开，最后一组[32,∞)，不根据结果调边界。末期未平仓不进入分组。

{md(duration,['candidate','duration_bin','count','winners','losers','net_profit','mean_hours'])}

盈利平均/中位持仓普遍长于亏损，收益需要捕获持续方向段，不能仅因持仓久就视为无效；[32,∞)中的亏损笔数及金额同时列出，说明长持仓亦可能失败。该关联是事后描述，不是时间止损效果。

## 3. 浮盈回吐（重点）

最大浮盈为MFE_USDT，未扣费用。主口径价格回吐=MFE_USDT−方向×数量×(退出原始open−实际入场fill)。执行口径回吐=MFE_USDT−账本gross_after_slippage=主回吐+退出滑点，不含手续费。
回吐比例以MFE为分母，MFE≤0留空。比例可超过100%（先有有利极值但终点转为毛亏损），不能截断。实际net另外扣双边手续费，不得拿MFE−net误称价格回吐。

{md(cohort,['candidate','direction','outcome','count','mean_price_giveback','median_giveback_ratio'])}

profit_giveback_analysis.csv分别保存价格及执行口径、盈利/亏损/方向分布；top20_winner_paths.csv可逐笔核查大赢家回吐。
OHLC最高浮盈是假设价格极值，并不保证该时点存在足够流动性可退出。任何止盈/移动止损还会改变后续持仓与大赢家，不可将回吐金额直接视为可挽回利润。

## 4. 盈亏分布与收益集中度

{md(side,['candidate','direction','trade_count','winning_count','losing_count','mean_win','mean_loss','median_win','median_loss','payoff_ratio','profit_factor','max_win','max_loss','closed_net_profit'])}

{md(concentration,['candidate','top_k','top_net_profit','share_winning_profit','share_closed_net','closed_net_without_top','top_mean_mae_usdt','top_mean_mfe_usdt','top_mean_price_giveback_usdt'])}

该表使用已平仓净收益，S0的全期总收益另外含末期未平仓MTM，因此去掉top10后的数值须加回该MTM才与S0的19.47/−1219.16口径一致。profit_concentration.csv有全部多空细分。
删除交易后的数字只是描述性扣除，不是可执行回测。正净收益对尾部趋势赢家有实质依赖；不宜以提高胜率为唯一退出目标。

## 5. 亏损聚集与风险

连亏按实际退出顺序，净收益≥0终止亏损序列；方向独立序列会忽略对侧交易，不能将方向连亏误称账户同时连续亏损。
本策略没有止损，因此亏损串是斜率翻向退出/反向的损失，不称“反复止损”。短时反向交易数量提供行为证据，未增加震荡或其他技术指标定义市场状态。

{md(risk,['candidate','direction','max_consecutive_losses','max_consecutive_loss_usdt','worst_day_pnl','worst_day_period','worst_week_pnl','worst_week_period','worst_month_pnl','worst_month_period','max_drawdown_usdt','max_drawdown_duration_days','flips_within_1h_count','losses_within_1h_count'])}

完整亏损序列在loss_streaks.csv，序列长度频数在loss_streak_distribution.csv。日/周/月风险来自逐K线账户MTM（含末期持仓），不是只按平仓日汇总。周为UTC周一至周日；期间归属沿S0开盘标签。
回撤用实际收盘时刻计算水下时长，初始10000纳入峰值；最长时长与最大深度可能属于不同回撤，未修复时长为截至训练末端的下界。方向曲线是单独的贡献账本加10000，不能简单相加方向最大回撤。
亏损在表中列出的月份和连亏串内聚集，最差3个月损失比例提供定量描述；没有随机基准或独立重采样检验，不能据此宣布“统计上非随机”或确定某种市场状态导致损失。

## 6. 多空结构差异

{md(side.query("direction!='all'"),['candidate','direction','trade_count','win_rate','mean_win','mean_loss','profit_factor','mean_mae_pct','mean_mfe_pct','mean_holding_hours','max_consecutive_losses','closed_net_profit'])}

表中净收益是已平仓净收益，末期未平仓的方向与MTM独立列出：

{md(opened,['candidate','direction','entry_time','holding_hours','net_pnl','mae_pct','mfe_pct','status'])}

多头在这段历史呈现正向趋势捕获证据，空头在两种N下均拖累组合；这支持“历史方向不对称”的描述，不证明多头具有稳定因果优势或空头在其他阶段也必然失效。
现有证据不足以证明多空必须使用不同退出机制，也不足以新增多空独立参数。方向分离只能作为后续机制假设与报告维度；本轮不关闭任何方向。

## 三个核心问题与研究建议

**利润来自哪里？** 主要来自较长持仓、多头方向的大趋势赢家；较低胜率由较高盈亏比和少数尾部收益补偿。top1/5/10/20、时间分组及多空PF表给出量化证据。N96的年度退化不能被累计盈利掩盖。

**最大风险是什么？** 大赢家依赖与空头负贡献并存，反向亏损串和长时间水下会侵蚀耐受度；大幅MFE之后仍可能回吐。实际MAE/MFE路径无盘中先后，不能用最高浮盈构造可实现的止盈利润。资金费、强平、盘口及成交假设仍不完整。

**若下一步只能增加一个风险/退出参数，最支持研究什么？** 建议研究一个**多空共用、按入场成交价定义的固定最大亏损比例**，作为S1.2的独立公平对照假设；理由是已量化的亏损MAE、连亏和尾部回撤需要风险约束，而盈利更长持仓且依赖少数大赢家，过早固定止盈/时间退出可能损伤收益来源。该建议只确定研究方向，不指定、估计或测试任何止损阈值，也不声称止损一定改善结果。
盈利与亏损MAE分布有重叠，大赢家也承受不利波动与浮盈回吐；不能用亏损MAE的均值/分位数直接宣布止损线。下一阶段若获授权，应先冻结单一对照设计、成本及盘中成交处理，再评价风险约束与大赢家损伤，不能回看2022/2023挑阈值。多空独立退出、移动止盈均只是未验证假设，本轮不开展。

## 质量控制与交付

所有聚合均能追溯trade_id和CSV；独立重复分析逐表精确一致。MAE/MFE公式、方向、退出bar排除、未来扰动、末期持仓、连亏边界和回撤时长有单元测试；全部S0测试继续通过。
数据、原S0配置/产物及本轮配置/源码SHA256见diagnostic_metadata.json与artifact_manifest.json。S0原始输出保持字节不变。完整图表如下；散点包含全部已平仓交易，直方图全范围无截尾，回撤为完整15分钟序列。

![MAE MFE](figures/mae_mfe_scatter.png)
![盈亏分布](figures/pnl_distribution.png)
![持仓时长](figures/holding_duration.png)
![浮盈与收益](figures/mfe_actual_profit.png)
![多空收益分布](figures/long_short_distribution.png)
![连亏与回撤](figures/loss_runs_drawdowns.png)

本轮完成即停止，等待负责人审阅，不进入止损优化或S1.2。
'''
    (OUT/'s1_trade_anatomy_report.md').write_text(text,encoding='utf-8')

def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log',mode='w',encoding='utf-8'),logging.StreamHandler(sys.stdout)],force=True)
    cfg_path=ROOT/'config/s1_trade_anatomy.json';cfg=json.loads(cfg_path.read_text());config_hash=sha256(cfg_path)
    assert cfg['candidates']==CANDIDATES and cfg['s0_commit']==BASE
    logging.info('Start existing-ledger anatomy only; no new trades, estimator or backtester calls')
    bars,t,curves,s0_hashes=load_inputs(cfg)
    logging.info('S0 artifacts, counts, costs, equity and drawdown reconcile; raw input %s training bars',len(bars))
    tables=compute(t,bars,curves);repeated=compute(t,bars,curves)
    for name,frame in tables.items():
        pd.testing.assert_frame_equal(frame,repeated[name],check_exact=True)
        frame.to_csv(OUT/f'{name}.csv',index=False,float_format='%.12g')
    # Ledger raw exit price matches actual bar open; MAE endpoint is not fabricated.
    closed=tables['trade_level_metrics'].query("status=='closed'")
    np.testing.assert_allclose(closed.terminal_raw_price,closed.exit_raw_price,atol=1e-8,rtol=0)
    np.testing.assert_allclose(closed.giveback_execution_usdt-closed.giveback_price_usdt,closed.exit_slippage_usdt,atol=1e-7,rtol=0)
    assert (closed.mae_usdt>=0).all() and (closed.mfe_usdt>=0).all() and (closed.giveback_price_usdt>=-1e-8).all()
    assert verify_s0()==s0_hashes and sha256(cfg['data_path'])==cfg['data_sha256'] and sha256(cfg_path)==config_hash
    e=pd.read_csv(S0/'zero_volume_execution_audit.csv');zero_count=int(e.candidate.isin(CANDIDATES).sum())
    meta=dict(task=cfg['task'],s0_base_commit=BASE,data_sha256=cfg['data_sha256'],s0_input_hashes=s0_hashes,
        config_sha256=config_hash,training_bars=len(bars),closed_trades=len(closed),open_positions=len(tables['open_positions']),
        zero_volume_executions=zero_count,analysis_repeated_exactly=True,s0_unchanged=True,
        holdout_used=False,post2023_read=False,parameter_search=False,strategy_modified=False,
        ledger_equity_cost_reconciliation_tolerance_usdt=1e-6,
        path_policy=cfg['excursion_path'],giveback_policy=cfg['giveback_primary'],
        execution_giveback_policy=cfg['giveback_execution_adjusted'],intrabar_order_inferred=False,
        versions={x:importlib.metadata.version(x) for x in ['numpy','pandas','pyarrow','matplotlib']},python=platform.python_version(),
        repeat_table_hashes={name:frame_hash(frame) for name,frame in tables.items()})
    charts(tables);make_report(tables,meta)
    (OUT/'diagnostic_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    for row in tables['long_short_comparison'].query("direction=='all'").itertuples():
        logging.info('%s closed=%s closed-net=%.2f win-rate=%.3f PF=%.3f mean-win=%.2f mean-loss=%.2f',row.candidate,row.trade_count,row.closed_net_profit,row.win_rate,row.profit_factor,row.mean_win,row.mean_loss)
    logging.info('Complete six analyses; exact-repeat and group reconciliations PASS; original S0 bytes unchanged. STOP before S1.2.')
    logging.shutdown()
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]
    files+=[ROOT/'src/trade_anatomy.py',ROOT/'tests/test_trade_anatomy.py',cfg_path,
        ROOT/'docs/s1_trade_anatomy_protocol.md',ROOT/'tools/verify_s1_anatomy.py']
    manifest={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(files)}
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':main()
