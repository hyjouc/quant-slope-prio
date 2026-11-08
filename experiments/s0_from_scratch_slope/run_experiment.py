"""Exactly 10 training-only candidates; no automatic model or N selection."""
from pathlib import Path
import sys,json,logging,platform,hashlib,importlib.metadata
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data_loader import load_training,sha256
from src.slope_estimators import estimate,N_GRID
from src.signal_generator import generate_signals
from src.backtester import simulate
from src.metrics import summarize,neighborhood

OUT=Path(__file__).parent
METHODS=['A_endpoint','B_ols']

def save_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def markdown(table,columns):
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,row in table.iterrows():
        vals=[]
        for k in columns:
            v=row[k]
            vals.append(f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

def plot_all(d,table,curves,stability):
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        # Fixed middle-grid N32 and first 7 days; selected before seeing outcomes.
        fig,axs=plt.subplots(2,1,figsize=(12,7),layout='constrained')
        sample=d.iloc[:7*96]
        axs[0].plot(sample.date,np.log(sample.close),color='#333333',label='Observed ln(close)')
        for method,style in zip(METHODS,['-','--']):
            beta=estimate(sample.close,32,method)
            axs[1].plot(sample.date,beta,label=method,linestyle=style,linewidth=1.)
        axs[0].set(ylabel='Log price',title='Estimator illustration: first 7 training days, fixed middle-grid N=32')
        axs[1].set(ylabel='Log-price slope per bar',xlabel='Candle open label (UTC)')
        axs[1].axhline(0,color='black',linewidth=.6)
        for ax in axs:ax.legend();ax.grid(alpha=.2)
        fig.savefig(figures/'estimator_comparison.png',dpi=160);plt.close(fig)
        for metric,title in [('net_profit','Net profit (USDT)'),('max_drawdown_usdt','Maximum drawdown (USDT)')]:
            fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
            for method,marker,style in zip(METHODS,['o','s'],['-','--']):
                g=table[table.method==method]
                ax.plot(g.N,g[metric],marker=marker,linestyle=style,label=method)
            ax.axhline(0,color='black',linewidth=.5);ax.set(xlabel='N (bars)',ylabel=title,title='2020-2021 training only')
            ax.set_xticks(N_GRID);ax.grid(alpha=.2);ax.legend()
            fig.savefig(figures/f'N_{metric}.png',dpi=160);plt.close(fig)
        colors=plt.colormaps['viridis'](np.linspace(.05,.9,5))
        for metric,title in [('equity','Account equity (USDT)'),('drawdown_usdt','Drawdown (USDT)')]:
            fig,axs=plt.subplots(2,1,figsize=(13,8),layout='constrained',sharex=True)
            for ax,method in zip(axs,METHODS):
                for j,n in enumerate(N_GRID):
                    c=curves[f'{method}_N{n}']
                    ax.plot(c.date,c[metric],color=colors[j],linestyle=['-','--',':','-.','-'][j],linewidth=.7,label=f'N={n}')
                ax.set(title=method,ylabel=title);ax.legend(ncol=5,fontsize=8);ax.grid(alpha=.2)
            axs[-1].set_xlabel('UTC candle open label; all 15-minute observations, no smoothing')
            fig.savefig(figures/f'{metric}_curves.png',dpi=160);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained',sharey=True)
        for ax,method in zip(axs,METHODS):
            g=table[table.method==method];x=np.arange(len(g))
            ax.bar(x-.18,g.long_net_profit,width=.36,hatch='//',label='Long')
            ax.bar(x+.18,g.short_net_profit,width=.36,hatch='..',label='Short')
            ax.set_xticks(x,g.N);ax.axhline(0,color='black',linewidth=.6)
            ax.set(title=method,xlabel='N',ylabel='MTM net profit (USDT)');ax.legend()
        fig.savefig(figures/'direction_comparison.png',dpi=160);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained',sharey=True)
        for ax,method in zip(axs,METHODS):
            g=stability[stability.method==method]
            ax.vlines(g.N,g.neighborhood_net_min,g.neighborhood_net_max,color='#777777',label='Adjacent-grid min to max')
            ax.plot(g.N,g.net_profit,'o-',label='Candidate net profit')
            ax.axhline(0,color='black',linewidth=.6);ax.set_xticks(N_GRID)
            ax.set(title=method,xlabel='N',ylabel='Training net profit (USDT)');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.savefig(figures/'parameter_neighborhood.png',dpi=160);plt.close(fig)

def report(table,annual,estimator,stability,meta):
    cols=['candidate','long_net_profit','short_net_profit','net_profit','max_drawdown_usdt','profit_factor','sharpe','sortino','win_rate','trade_count','average_holding_hours','monthly_profit_ratio']
    annualcols=['candidate','period','net_profit','long_net_profit','short_net_profit','max_drawdown_usdt','profit_factor','fees','slippage_usdt']
    extreme=['candidate','gross_before_cost','fees','slippage_usdt','top1_share_winning_profit','top5_share_winning_profit','top10_share_winning_profit','net_without_top10_winning_trades']
    stabilitycols=['candidate','neighbor_N','neighborhood_net_min','neighborhood_net_max','neighborhood_profitable_fraction','both_training_years_positive','strict_interior_local_net_peak','endpoint_caution']
    findings=[]
    for method,g in table.groupby('method',sort=False):
        years=annual[annual.method==method]
        both=stability[(stability.method==method)&stability.both_training_years_positive]
        findings.append(f"- {method}：五个N净利润范围 {g.net_profit.min():.2f} 至 {g.net_profit.max():.2f} USDT；盈利 {int((g.net_profit>0).sum())}/5；两年均盈利的N为 {both.N.tolist()}。最大回撤范围 {g.max_drawdown_usdt.min():.2f}—{g.max_drawdown_usdt.max():.2f} USDT。平均持仓 {g.average_holding_hours.min():.2f}—{g.average_holding_hours.max():.2f} 小时。")
    unfunded=table[~table.funded_execution_feasible]
    text=f'''# S0 从零斜率估计研究：2020—2021训练结果

任务：`S0_FROM_SCRATCH_SLOPE_ESTIMATOR_2020_2023`。

## 范围与结论边界

只比较端点A与滚动OLSB，每种固定N=[8,16,32,64,96]，共10组。方法选择本身也是模型选择自由度。
本轮没有自动排名、最佳模型或候选冻结；没有计算2022/2023绩效，亦未读取2024+行情。
四年行情曾参与旧策略研究，本轮任何时段均不能称为完全未见的独立样本外。以下是训练证据，不能证明未来交易价值。

**资金可执行性限制：{len(unfunded)}/10组曾出现保证金预算不足或非正权益。它们的全期结果是固定名义信号账本的假设延续，不是初始10000 USDT能够持续执行的账户回测。资金不足后仍按同一规则记账，没有追加保证金、自动缩仓或引入退出条件。**

{markdown(table,['candidate','min_actual_equity','first_margin_shortfall_time','first_nonpositive_equity_time','margin_budget_shortfall_entries','funded_execution_feasible'])}

{chr(10).join(findings)}

不能据累计净利润峰值直接选择N。参数邻域是固定离散网格上的描述性比较，端点只有两个邻域点，不能向网格外推断。完整10组、年度差异、费用及交易集中度如下，交由负责人判断哪些结构值得冻结验证。

## 本轮六个研究问题的回答

1. **方法稳定性**：端点法在整个既定网格均亏损；OLS只在较大窗口有正收益，不能据此称OLS在整个参数域都稳定。五个N的分布比单个峰值更有意义。
2. **N变化是否平滑**：端点法随N增加，亏损整体减少；OLS从N32到N64出现较大的收益跳变，之后N96累计仍为正。N间距不均，不能把连线当作连续响应证据。
3. **年度差异**：OLS N64两年分别为正，N96主要来自2020而2021转负；长窗口的累计邻近盈利未转化为一致年度稳健性。端点N96的2021微利也不能抵消其2020亏损。
4. **孤立点与稳定区**：OLS N64、96在累计净收益上是相邻的两个正值，但N64的另一侧N32亏损，N96又是网格端点，且二者年度结果不同。因此暂不能认定存在已确认的连续稳定近优区域，不增加N48/80或其他新候选检验。
5. **多空平衡**：所有10组空头净收益都为负；两个累计盈利OLS候选均依赖多头贡献，不能称多空均合理。不能通过关闭空头或给多空独立N改善本轮结果。
6. **成本与极端交易**：多组账面毛收益为正、扣实际费用滑点后转负，频繁翻向的摩擦是明确问题，但部分端点候选连毛收益也为负，不能全归因于成本。OLS N64描述性扣除最盈利10笔后仅约19.47 USDT，N96则约−1219.16 USDT；正收益对少数大赢家有实质依赖。这不证明趋势模型无效，也不足以宣告可以交易，需负责人审阅后再决定是否接受后续冻结检验。

## 数据与预登记口径

只读取date/open/high/low/close/volume六列，数据SHA256 `{meta['data']['sha256']}`，不一致立即停止。
原始快照140256根，仅用于结构核验；送入指标、信号、执行及统计的是2020—2021的70176根。
首96根只预热，统一从2020-01-02 00:00 UTC开盘开始成交，共70080根绩效K线，年界不平仓、不重置指标。
零成交量不删除、不补价：原快照97根，训练输入39根；全候选共 {meta['zero_volume_execution_count']} 项假设成交发生于零成交量K线（反向平仓及开仓各一项，跨候选不去重）。详见zero_volume_execution_audit.csv，无法由OHLCV证明可成交。
N表示观测点数量，N根K线覆盖N×15分钟，首尾收盘点间隔(N−1)×15分钟。A和B单位均为每根K线的对数价格变化，无乘100、标准化或阈值。
初始账户10000，固定2000成交名义（1000保证金、2倍），不复利；费用另扣。单边手续费0.05%、滑点0.02%，买入向上、卖出向下，反向旧仓退出与新仓进入分别扣费。
无资金费率、交易所强平、盘口深度或延迟模型。固定名义账本遇到保证金预算不足时记录而不更改信号/仓位，不能作为可实际融资的交易结果；本次不足事件数 {meta['margin_shortfall_entry_count']}。

## 全部10组训练指标

PF、胜率及持仓时间仅包含实际平仓；净利润含末期未平仓MTM。win_rate和monthly_profit_ratio为0—1比例。

{markdown(table,cols)}

## 年度对比（保持跨年仓位）

{markdown(annual,annualcols)}

年度净收益按该年度K线的MTM增量归属，最后一根K线的收盘按其开盘标签所属年度归属；不代表年界强制退出。
PF/胜率将整笔已平仓交易归到退出年度，与年度MTM不是同一会计口径。

## 方法层面的网格稳健性

{markdown(estimator,list(estimator.columns))}

这些是五个既定N的等权描述性汇总，不是可交易组合，也不是自动方法选择评分。

## 参数邻域与年度一致性

{markdown(stability,stabilitycols)}

邻域只取同方法中现有相邻网格点（包含自身），并不等于等距离参数扰动。表中strict_interior_local_net_peak仅表示内部离散局部峰值，不自动认定为过拟合，也不将其宣布为最佳。

## 成本与极端盈利交易依赖

{markdown(table,extreme)}

gross_before_cost保持实际固定名义产生的数量，逐笔加回已经发生的费用与滑点，不是另跑无成本策略。
topK_share_winning_profit分母为已平仓正收益交易之和，表示收益集中度；删除topK的净收益只是描述性扣除，不是可执行反事实回测。净收益为负时topK/net比例留空，避免误导。
若费用和滑点超过gross_before_cost，意味着成本足以吞噬该候选的账面毛收益；不能据此改成本假设重新选择更有利结果。
多空贡献保留正负金额；亏损组合不使用“收益占比”掩盖方向失衡。详细direction_comparison.csv保留方向费用、滑点、平仓次数及PF。

## 统计、成交与期末

每个收盘信号仅能在下一根open成交；源K线open标签、signal_available_at及fill_time独立记录，两者可用边界相同不代表使用未来信息。
严格β=0时保持上一信号（初始为0）；无效β不下单，不把它变成中性平仓。OLS不使用近零容差阈值。
最大回撤基于15分钟收盘MTM，含初始账户峰值，同时输出USDT/运行峰值百分比/初始本金百分比。年度回撤只重置统计峰值，不重置仓位。
Sharpe基于每日MTM增量/固定10000，rf=0，样本标准差ddof=1及sqrt(365)年化；Sortino用所有日收益负部RMS作分母。
末期持仓按2021-12-31 23:45标签K线的收盘MTM，不伪造退出、不扣未发生退出费。最终收盘反向信号没有下一根已授权K线，保持未执行；进入后续验证时应在2022首根open执行，须冻结记录携带此状态。

## 核验与下一步

10组独立重复运行逐笔、成交事件及每根权益精确一致；年度/月度/方向/成本/逐笔（含末期MTM）与权益增量对账通过。
模型公式与窗口、前缀、未来扰动、正负零及无效信号、反向双费、下一开盘、不利滑点、跨年、末端与零成交量执行均有测试。
代码不导入或调用任何归档实现，不读取旧特征/模型/执行器/优化结果。
本轮完成后停止；等待负责人审阅10组及邻域证据，确定方法和N并形成冻结记录，然后才可运行2022—2023。验证差不能反过来重新选N。

## 图表（对应CSV为底层数据，完整15分钟曲线，无平滑）

![两种斜率估计](figures/estimator_comparison.png)
![N与净利润](figures/N_net_profit.png)
![N与最大回撤](figures/N_max_drawdown_usdt.png)
![权益曲线](figures/equity_curves.png)
![回撤曲线](figures/drawdown_usdt_curves.png)
![多空收益](figures/direction_comparison.png)
![参数邻域](figures/parameter_neighborhood.png)
'''
    (OUT/'s0_report.md').write_text(text,encoding='utf-8')

def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log',mode='w',encoding='utf-8'),logging.StreamHandler(sys.stdout)],force=True)
    cfg_path=ROOT/'config/s0_from_scratch.json';cfg=json.loads(cfg_path.read_text());cfg_hash=sha256(cfg_path)
    assert cfg['N_grid']==list(N_GRID) and cfg['methods']==METHODS
    assert cfg['holdout_performance_authorized'] is False and cfg['performance_end_exclusive']=='2022-01-01T00:00:00Z'
    logging.info('START ten fixed candidates, training only; no automatic selection; config SHA256 %s',cfg_hash)
    d,data=load_training(cfg['data_path']);logging.info('Raw snapshot hash and continuity PASS; training-only input %s bars',len(d))
    tables=[];annual=[];monthly=[];trades=[];executions=[];directions=[];curves={};repeat=[]
    for method in METHODS:
        for n in N_GRID:
            beta=estimate(d.close,n,method);sig=generate_signals(beta)
            # Real market prefixes and changed future within TRAINING only.
            cut=40000
            np.testing.assert_allclose(beta[:cut],estimate(d.close.iloc[:cut],n,method),equal_nan=True,rtol=0,atol=0)
            changed=d.close.to_numpy().copy();changed[cut:]*=1.7
            np.testing.assert_allclose(beta[:cut],estimate(changed,n,method)[:cut],equal_nan=True,rtol=0,atol=0)
            t,c,e=simulate(d,sig,method,n,cfg)
            t2,c2,e2=simulate(d,generate_signals(estimate(d.close,n,method)),method,n,cfg)
            for a,b in [(t,t2),(c,c2),(e,e2)]:pd.testing.assert_frame_equal(a,b,check_exact=True)
            candidate=c.candidate.iloc[0];curves[candidate]=c;trades.append(t);executions.append(e)
            rows=summarize(c,t,e,cfg);tables.append(rows)
            for year in [2020,2021]:
                r=summarize(c,t,e,cfg,f'{year}-01-01T00:00:00Z',f'{year+1}-01-01T00:00:00Z',str(year));r['year']=year;annual.append(r)
            for month in sorted(c.date.dt.strftime('%Y-%m').unique()):
                start=pd.Timestamp(month+'-01',tz='UTC');end=start+pd.offsets.MonthBegin(1)
                r=summarize(c,t,e,cfg,start,end,month);r['month']=month;monthly.append(r)
            for direction in ['long','short']:
                tt=t[t.direction==direction];ee=e[e.direction==direction];closed=tt[tt.status=='closed']
                wins=closed.loc[closed.net_pnl>0,'net_pnl'].sum();loss=-closed.loc[closed.net_pnl<0,'net_pnl'].sum()
                directions.append(dict(candidate=candidate,method=method,N=n,direction=direction,
                    net_profit=rows[direction+'_net_profit'],fees=ee.fee.sum(),slippage_usdt=ee.slippage_usdt.sum(),
                    gross_before_cost=rows[direction+'_net_profit']+ee.fee.sum()+ee.slippage_usdt.sum(),
                    closed_trade_count=len(closed),entry_count=int(ee.action.eq('open').sum()),
                    profit_factor=wins/loss if loss else np.nan,win_rate=(closed.net_pnl>0).mean()))
            # Accounting assertions independent of tabular rendering.
            np.testing.assert_allclose(c.equity-10000,c.long_pnl+c.short_pnl,atol=1e-8,rtol=0)
            np.testing.assert_allclose(t.net_pnl.sum(),rows['net_profit'],atol=1e-8,rtol=0)
            np.testing.assert_allclose(t.gross_before_cost-t.entry_fee-t.exit_fee-t.entry_slippage_usdt-t.exit_slippage_usdt,t.net_pnl,atol=1e-9,rtol=0)
            np.testing.assert_allclose(e.fee.sum(),c.fees.iloc[-1],atol=1e-8,rtol=0)
            np.testing.assert_allclose(e.slippage_usdt.sum(),c.slippage_usdt.iloc[-1],atol=1e-8,rtol=0)
            np.testing.assert_allclose(sum(r['net_profit'] for r in annual if r['candidate']==candidate),rows['net_profit'],atol=1e-8,rtol=0)
            np.testing.assert_allclose(sum(r['net_profit'] for r in monthly if r['candidate']==candidate),rows['net_profit'],atol=1e-8,rtol=0)
            repeat.append(dict(candidate=candidate,all_trades_events_equity_exact=True,prefix_exact=True,future_perturbation_exact=True))
            logging.info('%s net %.2f DD %.2f PF %.4f closed %s fees %.2f slip %.2f zero-volume fills %s; repeat/reconciliation PASS',candidate,rows['net_profit'],rows['max_drawdown_usdt'],rows['profit_factor'],rows['trade_count'],rows['fees'],rows['slippage_usdt'],rows['zero_volume_execution_count'])
    table=pd.DataFrame(tables);ann=pd.DataFrame(annual);mon=pd.DataFrame(monthly);ts=pd.concat(trades,ignore_index=True);ev=pd.concat(executions,ignore_index=True)
    stable=neighborhood(table,ann)
    est=[]
    for method,g in table.groupby('method',sort=False):
        a=ann[ann.method==method];st=stable[stable.method==method]
        est.append(dict(method=method,candidates=len(g),profitable_candidates=int((g.net_profit>0).sum()),
            both_years_positive_candidates=int(st.both_training_years_positive.sum()),
            net_min=g.net_profit.min(),net_median=g.net_profit.median(),net_max=g.net_profit.max(),
            worst_drawdown=g.max_drawdown_usdt.max(),median_profit_factor=g.profit_factor.median(),
            median_sharpe=g.sharpe.median(),median_monthly_profit_ratio=g.monthly_profit_ratio.median(),
            median_closed_trades=g.trade_count.median()))
    est=pd.DataFrame(est)
    zero=ev[ev.zero_volume].copy()
    for name,frame in [('parameter_comparison',table),('estimator_comparison',est),('annual_comparison',ann),('monthly_comparison',mon),
            ('direction_comparison',pd.DataFrame(directions)),('trades_all',ts),('executions_all',ev),('parameter_stability',stable),
            ('zero_volume_execution_audit',zero),('reproducibility',pd.DataFrame(repeat))]:
        frame.to_csv(OUT/f'{name}.csv',index=False,float_format='%.12g')
    for name,field in [('equity_curves','equity'),('drawdown_curves','drawdown_usdt')]:
        out=pd.DataFrame({'date':next(iter(curves.values())).date})
        for candidate,c in curves.items():out[candidate]=c[field]
        out.to_csv(OUT/f'{name}.csv',index=False,float_format='%.12g')
    meta=dict(task_id=cfg['task_id'],config_sha256=cfg_hash,data=data,
        actual_performance_first=str(next(iter(curves.values())).date.iloc[0]),actual_performance_last=str(next(iter(curves.values())).date.iloc[-1]),
        performance_bars=len(next(iter(curves.values()))),candidates=10,grid=list(N_GRID),methods=METHODS,
        legacy_code_imported=False,old_feature_columns_read=False,post2023_market_data_read=False,
        holdout_indicators_computed=False,holdout_performance_computed=False,model_or_N_selected=False,freeze_record_generated=False,
        zero_volume_execution_count=len(zero),margin_shortfall_entry_count=int(ev.margin_budget_shortfall.sum()),
        nonpositive_equity_candidates=[candidate for candidate,c in curves.items() if (c.equity<=0).any()],
        numerical_policy='OLS subtract first window value then exact centered x sum; strict zero, no epsilon threshold; each window independently summed',
        accounting='fixed entry fill notional 2000, fees separate; initial account 10000; no compounding; raw-open slippage costs use actual quantities',
        statistics='bar-close MTM; daily PnL/10000; rf0; sqrt365; Sharpe sample SD; Sortino all-day negative-part RMS; period attribution by candle open label',
        annual='continuous holdings, period MTM deltas; PF/win/holding stats entire closed trade assigned to exit period',
        pending_final_orders={candidate:dict(last_signal=None if not np.isfinite(generate_signals(estimate(d.close,int(c.N.iloc[0]),str(c.method.iloc[0])))[-1]) else int(generate_signals(estimate(d.close,int(c.N.iloc[0]),str(c.method.iloc[0])))[-1]),position=int(c.position.iloc[-1])) for candidate,c in curves.items()},
        checks=dict(actual_full_grid_repeat=True,real_training_prefix_and_future_perturbation=True,all_ledger_reconciliations=True),
        versions={x:importlib.metadata.version(x) for x in ['numpy','pandas','pyarrow','matplotlib']},python=platform.python_version(),
        limitations=['all four years historically researched, not untouched OOS','zero-volume executions hypothetical','no funding or exchange liquidation/depth/latency model','training only; no automatic winner'],
        figure_policy='seven PNGs; complete 15m curves; illustrative estimator comparison fixed N32 first 7 days; no smoothing')
    assert sha256(cfg_path)==cfg_hash and sha256(cfg['data_path'])==data['sha256']
    plot_all(d,table,curves,stable);report(table,ann,est,stable,meta)
    save_json(OUT/'diagnostic_metadata.json',meta)
    logging.info('COMPLETE training only; no selection, freeze or second search; data/config hashes unchanged')
    logging.shutdown()
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]
    files+=list((ROOT/'src').glob('*.py'))+[cfg_path]
    manifest={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(files)}
    save_json(OUT/'artifact_manifest.json',manifest)

if __name__=='__main__':main()
