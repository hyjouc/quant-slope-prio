"""Five predeclared slopes, untouched S0 executor; training only, no selection."""
from pathlib import Path
import sys,json,io,subprocess,hashlib,logging,platform,importlib.metadata
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data_loader import load_training,sha256
from src.slope_estimators import estimate
from src.signal_generator import generate_signals
from src.backtester import simulate,TRADE_COLUMNS,EXEC_COLUMNS
from src.metrics import summarize
from src.trade_anatomy import drawdown_episodes
from src.recent_weighted_slope import estimate_ewls
from src.recent_weighted_analysis import synthetic_experiments,match_flips,response_summary,original_trade_comparison

OUT=Path(__file__).parent;S0=ROOT/'experiments/s0_from_scratch_slope'
BASE='c80864ef9f3a2fcccf39ba616b04f295779b98a6'
MODELS=[('OLS',None),('H64',64),('H32',32),('H16',16),('H8',8)]

def frozen_snapshot():
    tree=subprocess.check_output(['git','ls-tree','-r',BASE],cwd=ROOT,text=True).splitlines()
    expected={line.split('\t',1)[1]:line.split()[2] for line in tree}
    paths=list(expected)
    actual=subprocess.check_output(['git','hash-object','--stdin-paths'],input='\n'.join(paths)+'\n',cwd=ROOT,text=True).splitlines()
    assert len(actual)==len(paths)
    for p,h in zip(paths,actual):
        if expected[p]!=h:raise ValueError('Frozen base file modified: '+p)
    return {p:sha256(ROOT/p) for p in paths}

def serialized(frame):return pd.read_csv(io.StringIO(frame.to_csv(index=False,float_format='%.12g')))

def control_gate(d,cfg):
    b=estimate(d.close,64,'B_ols');equal=estimate_ewls(d.close,None)
    np.testing.assert_array_equal(b,equal)
    s=generate_signals(b);np.testing.assert_array_equal(s,generate_signals(equal))
    t,c,e=simulate(d,s,'B_ols',64,cfg)
    for frame,file,cols in [(t,'trades_all.csv',TRADE_COLUMNS),(e,'executions_all.csv',EXEC_COLUMNS)]:
        saved=pd.read_csv(S0/file);saved=saved[saved.candidate=='B_ols_N64'].reset_index(drop=True)
        pd.testing.assert_frame_equal(serialized(frame[cols]),saved,check_exact=True)
    eq=pd.read_csv(S0/'equity_curves.csv');dd=pd.read_csv(S0/'drawdown_curves.csv')
    np.testing.assert_array_equal(pd.to_datetime(eq.date,utc=True),c.date)
    np.testing.assert_allclose(c.equity,eq.B_ols_N64,rtol=0,atol=1e-6)
    np.testing.assert_allclose(c.drawdown_usdt,dd.B_ols_N64,rtol=0,atol=1e-6)
    ref=pd.read_csv(S0/'parameter_comparison.csv').query("candidate=='B_ols_N64'").iloc[0]
    row=summarize(c,t,e,cfg)
    for field in ['net_profit','long_net_profit','short_net_profit','fees','slippage_usdt','max_drawdown_usdt','profit_factor','sharpe','trade_count']:
        np.testing.assert_allclose(row[field],ref[field],rtol=0,atol=1e-6)
    assert row['trade_count']==1147
    return b,s,(t,c,e),dict(equal_weight_slopes='bitwise identical including warmup NaNs',signals='exact',
        trade_ledger='exact at original CSV precision %.12g',execution_ledger='exact at original CSV precision %.12g',
        curve_max_absolute_error=float(np.max(np.abs(c.equity-eq.B_ols_N64))),
        drawdown_max_absolute_error=float(np.max(np.abs(c.drawdown_usdt-dd.B_ols_N64))),
        direction_cost_and_metrics='all independently matched at 1e-6 absolute tolerance')

def risk(c,t,start,end,direction='all'):
    g=c[(c.date>=pd.Timestamp(start))&(c.date<pd.Timestamp(end))].copy()
    if direction=='all':delta=g.pnl_change
    else:delta=g[direction+'_pnl_change']
    g['equity']=10000+delta.cumsum();g['direction']=direction
    dd=drawdown_episodes(g)
    z=t[(t.status=='closed')&(t.exit_time>=pd.Timestamp(start))&(t.exit_time<pd.Timestamp(end))]
    if direction!='all':z=z[z.direction==direction]
    count=maximum=0;amount=0.;worst=0.
    for p in z.net_pnl:
        if p<0:count+=1;amount+=p;maximum=max(maximum,count);worst=min(worst,amount)
        else:count=0;amount=0.
    return dict(longest_underwater_days=float(dd.duration_hours.max()/24) if len(dd) else 0.,
        max_consecutive_losses=maximum,max_consecutive_loss_usdt=-worst,
        winning_trade_count=int(z.net_pnl.gt(0).sum()),losing_trade_count=int(z.net_pnl.lt(0).sum()),
        max_single_trade_loss=float(min(0,z.net_pnl.min()))),dd

def concentration(t,net,name,period,direction='all'):
    z=t[t.status=='closed'];w=z.loc[z.net_pnl>0,'net_pnl'].sort_values(ascending=False)
    return [dict(model=name,period=period,direction=direction,top_k=k,top_profit_usdt=w.iloc[:k].sum(),
        share_of_winning_profit=w.iloc[:k].sum()/w.sum() if w.sum() else np.nan,
        share_of_net_profit=w.iloc[:k].sum()/net if net>0 else np.nan,
        net_without_top_winners=net-w.iloc[:k].sum(),total_net_profit=net,
        closed_net_profit=z.net_pnl.sum()) for k in [1,5,10,20]]

def write_csv(name,frame):frame.to_csv(OUT/name,index=False,float_format='%.12g')

def figures(table,curves,slopes,synthetic,paths,conc):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    styles=['-','--','-.',':','--'];colors=['#000000','#0072B2','#E69F00','#009E73','#CC79A7']
    def save(fig,name):fig.savefig(folder/(name+'.png'),dpi=170);plt.close(fig)
    with plt.rc_context({'font.size':10,'axes.spines.right':False,'axes.spines.top':False}):
        fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
        view=slopes[slopes.date<pd.Timestamp('2020-01-08',tz='UTC')]
        for j,(name,h) in enumerate(MODELS):ax.plot(view.date,view[name+'_slope'],label=name,color=colors[j],linestyle=styles[j],lw=1)
        ax.axhline(0,color='grey',lw=.6);ax.set(ylabel='Log-price slope per 15m bar',xlabel='UTC | fixed illustration: first seven calendar days')
        ax.legend(ncol=5);save(fig,'slope_comparison')
        fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        cases=['up_to_down_noise0','down_to_up_noise0','up_to_down_noise0.006','oscillation_up_shock']
        for ax,case in zip(axs.flat,cases):
            for j,(name,h) in enumerate(MODELS):
                z=paths[(paths.case==case)&(paths.model==name)]
                ax.plot(z['index'],z.slope,color=colors[j],linestyle=styles[j],label=name,lw=1)
            ax.axvline(192,color='grey',ls='--');ax.axhline(0,color='grey',lw=.5)
            ax.set(title=case,xlabel='Synthetic bar index',ylabel='Slope per bar',xlim=(150,280));ax.legend(fontsize=7)
        fig.suptitle('Fixed mechanism illustrations; full paths and all cases in synthetic_paths.csv')
        save(fig,'synthetic_responses')
        for field,label,file in [('equity','Account close-MTM equity (USDT)','equity_curves'),('drawdown_usdt','Close-MTM drawdown (USDT)','drawdown_curves')]:
            fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
            for j,(name,h) in enumerate(MODELS):ax.plot(curves[name].date,curves[name][field],color=colors[j],linestyle=styles[j],label=name,lw=.9)
            ax.set(xlabel='UTC candle open label; all 15m points, no smoothing',ylabel=label);ax.legend(ncol=5);ax.grid(alpha=.2);save(fig,file)
        finite=table[table.model!='OLS'].sort_values('H')
        for field,label,file in [('net_profit','Net profit (USDT)','H_net_profit'),('max_drawdown_usdt','Max close-MTM drawdown (USDT)','H_drawdown'),('trade_count','Closed trades (count)','H_trade_count')]:
            fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
            ax.plot(finite.H,finite[field],'o-',color=colors[1],label='EWLS predeclared grid only')
            ax.axhline(table[field].iloc[0],color='black',ls='--',label='OLS control (infinite H)')
            ax.axhline(0,color='grey',lw=.5);ax.set(xlabel='Weight half-life H (15m bars)',ylabel=label,xticks=[8,16,32,64]);ax.legend();ax.grid(alpha=.2);save(fig,file)
        fig,ax=plt.subplots(figsize=(10,5),layout='constrained');x=np.arange(5)
        ax.bar(x-.18,table.long_net_profit,.36,label='Long',color=colors[1],hatch='//')
        ax.bar(x+.18,table.short_net_profit,.36,label='Short',color=colors[2],hatch='..')
        ax.axhline(0,color='black',lw=.7);ax.set(xticks=x,xticklabels=table.model,ylabel='Net contribution incl. terminal MTM (USDT)');ax.legend();save(fig,'long_short')
        fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
        z=conc[(conc.period=='overall')&(conc.direction=='all')]
        for j,(name,h) in enumerate(MODELS):
            v=z[z.model==name];axs[0].plot(v.top_k,100*v.share_of_winning_profit,marker='o',color=colors[j],ls=styles[j],label=name)
        axs[0].set(xlabel='Top k closed winning trades',ylabel='Share of all winning trade profits (%)',xticks=[1,5,10,20]);axs[0].legend(fontsize=8)
        vals=[z[(z.model==name)&(z.top_k==10)].net_without_top_winners.iloc[0] for name,h in MODELS]
        axs[1].bar(x,vals,color=colors,hatch='//');axs[1].set(xticks=x,xticklabels=table.model,ylabel='Net after descriptive top10 removal (USDT)')
        axs[1].axhline(0,color='grey',lw=.6);fig.suptitle('Descriptive concentration; deleting winners is not an executable strategy');save(fig,'profit_concentration')

def md(table,cols):
    rows=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for _,r in table.iterrows():rows.append('| '+' | '.join(f'{r[c]:.3f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in cols)+' |')
    return '\n'.join(rows)

def report(table,annual,response,winners,conc,synthetic,meta,stability):
    base=table.iloc[0];lines=[]
    for r in table.iloc[1:].itertuples():
        a=annual[(annual.model==r.model)&(annual.period=='2020')].iloc[0];b=annual[(annual.model==r.model)&(annual.period=='2021')].iloc[0]
        s=response[response.model==r.model].iloc[0];w=winners[winners.model==r.model].iloc[0]
        lines.append(f'- {r.model}：平仓{r.trade_count}笔（{r.trade_count-base.trade_count:+d}），总成本{r.fees+r.slippage_usdt:.2f} USDT（{r.fees+r.slippage_usdt-base.fees-base.slippage_usdt:+.2f}）；2020/2021净收益{a.net_profit:.2f}/{b.net_profit:.2f}；原top10区间实际净收益{w.top10_actual_interval_pnl:.2f}，同向覆盖仓提前退出{w.top10_covering_early_exits}/10，原入口无同向覆盖{w.top10_no_same_direction_cover}/10；匹配反向提前/滞后/同期{s.early_reversals}/{s.late_reversals}/{s.same_reversals}，有符号延迟均值{s.signed_delay_mean_bars:.2f}根。')
    clean=synthetic[(synthetic.family=='trend_reversal')&(synthetic.noise_std==0)]
    text=f'''# S1.3 近期加权斜率：2020—2021固定五组研究

## 研究边界与公平性

基于S1.2提交 `{BASE}` 独立分支，全部原跟踪文件字节不变。原S0 OLS N64为对照；仅新增权重半衰期H，固定∞、64、32、16、8；不继承S1.2止损。未自动选择H，未扩大搜索，不计算2022/2023绩效，不读取2024+。
S0快照SHA256 `{meta['data']['sha256']}`；原loader只读六列OHLCV，对2020—2023快照做结构QC，计算前截取2020—2021。这四年历史已被旧研究使用，未来验证也不得称完全独立未见OOS。
所有候选固定64窗口/96根预热，首成交2020-01-02 00:00 UTC，初始10000、名义2000、保证金预算1000、杠杆2、不复利，单边费0.05%、滑点0.02%。直接调用原S0信号器、成交器、指标口径；不修改它们，不增加阈值/标准化/过滤/退出机制。
基准门禁：H=∞独立数学实现与OLS斜率及信号逐位相等；基准账本/成交事件在原CSV精度完全相等，资金/回撤与原文件误差分别{meta['baseline_gate']['curve_max_absolute_error']:.3g}/{meta['baseline_gate']['drawdown_max_absolute_error']:.3g} USDT；多空、费用、指标独立通过1e-6对账。基准不一致将先于有限H计算停止。

## 数学与时序

窗口i=0..63，y=ln(close)，w=2^(-(63-i)/H)，x=i；加权均值后beta=sum(w*(x-xbar)*(y-ybar))/sum(w*(x-xbar)^2)。最新w=1，H根前权重折半；输出单位为每15分钟对数价格变化。用减去窗口首y的等价中心化提高浮点稳定性，无信号epsilon。H=∞用独立等权中心求和验证，实际基准使用原OLS函数。
严格beta正负做多空，精确零继承，非有限值不下单；bar收盘可用，下一bar开盘反手；最后bar信号不执行。年末不强制平仓，期末未平仓按最后close MTM，无虚构退出费。图表date为K线开盘标签，equity在该根收盘才可用。

## 五组整体表现（不按收益排序）

{md(table,['model','net_profit','gross_before_cost','fees','slippage_usdt','max_drawdown_usdt','profit_factor','sharpe','sortino','win_rate','trade_count','average_holding_hours','monthly_profit_ratio','longest_underwater_days','max_consecutive_losses'])}

{md(table,['model','long_net_profit','short_net_profit','winning_trade_count','losing_trade_count','margin_budget_shortfall_entries','min_actual_equity','funded_execution_feasible'])}

{chr(10).join(lines)}

毛利润=gross_before_cost，指费用/滑点前净价格损益，非仅盈利交易之和；盈利/亏损交易毛和另见gross_closed_profits/losses。PF/胜率/平均持有时间按该期退出的已平仓交易；净收益含未平仓MTM。Sharpe使用日损益/固定初始10000，样本日标准差、sqrt365；Sortino使用全部日负部RMS。回撤含期初10000峰值，15分钟收盘MTM，非完整盘中极值。年度风险指标重置分析峰值和亏损连串，实盘仓位不重置，年初承接仓位损益来自曲线增量。月度表同理；年度/月度净收益及多空可加总，交易级退出归属与MTM年度损益不可混同。

## 实验A：确定性响应机制

固定seed=13032026，384根、真实反向bar192、潜在每根log增量±0.001；仅反向后加入固定共享高斯log价格观测噪声，sigma=0/.001/.003/.006，双方向。另有幅度.005、周期64的正弦震荡，在192..199施加±.012临时log价格偏移，非永久趋势变化。所有模型用相同路径；合成仅研究机制，不替代市场收益。

{md(clean,['case','model','first_new_direction_index','response_delay_bars','delay_change_vs_ols','wrong_destination_flips'])}

完整50行见synthetic_response_tests.csv；完整路径见synthetic_paths.csv。趋势错误反向=切换目的方向与当时潜在增量符号相反，反向后再次回错方向单列post_response_wrong_switchbacks；震荡冲击无唯一真趋势，additional_flips_vs_unshocked只是相对配对未冲击路径的增加次数，不擅称错误。图仅固定展示四个机制案例，CSV保留全部。

## 实验C：真实市场反向配对与交易变化

{md(response,['model','signal_agreement_ratio','candidate_flip_count','early_reversals','late_reversals','same_reversals','signed_delay_mean_bars','signed_delay_median_bars','unmatched_ols_reversals','unmatched_candidate_reversals','exact_new_entry_identities','exact_missing_original_entries'])}

真实行情没有可观测“真反转”，不能把比OLS早等同更正确。为避免无限制最近邻重复配对，以相邻OLS反向的时间中点划分不重叠Voronoi区间（中点归前区间），各区间匹配最近的同目的方向候选反向，等距取早，未匹配显式保留。负delta为提前、正为滞后。它是固定描述规则，不是因果地面真值；窗口内多次震荡也可造成未匹配事件。逐事件见signal_flip_comparison.csv；新增/取消反向仅按此约定；精确入口未匹配还包括时间移动，不等同取消交易。
使用索引95..70174的收盘信号，对应性能期下一开盘；不包含最后不可执行的收盘信号。每个非零翻转是一笔退出+新入场，因而增加交易会增加两边成本；初始入场和期末MTM单独保留。逐根斜率与信号见slope_signal_paths.csv。

## 实验D：收益结构与原始大赢家

{md(winners,['model','top10_original_net_pnl','top10_actual_interval_pnl','top10_interval_retention_ratio','top10_intervals_positive','top10_covering_early_exits','top10_no_same_direction_cover','top10_covering_smaller_positive','top10_interval_smaller_positive','top10_intervals_became_losses'])}

{md(conc[(conc.period=='overall')&(conc.direction=='all')&(conc.top_k==10)],['model','top_profit_usdt','share_of_winning_profit','share_of_net_profit','net_without_top_winners'])}

固定原OLS已平仓净收益最高10笔身份，不根据EWLS收益重选。trade_level_comparison.csv保留全部1147个原交易×5候选，original_top10_winner_audit.csv为固定子集。提前退出定义：在原入场开盘执行后存在同方向覆盖仓，该覆盖仓的首次退出早于原退出。若覆盖仓更早入场，其完整交易PnL与原交易不可等同比较；covering_trade_*列只用于描述，主归因用actual_interval_pnl。
区间归因：在原入口到原出口开盘之间，承接仓以原入口raw open MTM，实际区间新开仓计真实入场价及费，出口开盘已有仓平仓计真实退出价及费；入口开盘已平仓的前仓不计，出口开盘新仓不计，未退出仓按出口raw open MTM，无虚构退出费。跨期标记仓位的费用归属与完整曲线不同，因此候选区间和不声称等于全曲线损益；基准每个原区间则与原交易净收益逐项相等。区间内包括实际反手、再进入及多空分项，故是实际候选路径归因，不是替换某个退出的反事实收益。原赢家区间保留率可大于1或为负，不是“原交易存活率”。
profit_concentration.csv同时保留整体/各年及多空的前1/5/10/20，删除赢家后的数值仅为描述性统计，非可执行回测。若总净收益≤0，top/净收益比例留空，避免误导。

## 实验B/E：年度、多空与相邻H

{md(annual,['model','period','net_profit','long_net_profit','short_net_profit','max_drawdown_usdt','profit_factor','sharpe','sortino','trade_count','fees','slippage_usdt','longest_underwater_days','max_consecutive_losses','margin_budget_shortfall_entries'])}

{md(stability,['H_left','H_right','left_net','right_net','net_gap_usdt','left_dd','right_dd','left_trades','right_trades','left_worst_year','right_worst_year'])}

只比较固定相邻H 8↔16↔32↔64，不给自动分数/胜出阈值；年度差异、多空差异、回撤和成本同时审阅。参数邻域描述见parameter_stability.csv，不把曲线平滑直接解释为独立稳健性证据。long_short_comparison.csv独立对账全部/各年每方向净收益、交易、费用、滑点、风险及收益集中度。

## 八个问题与审阅结论

以下结论由本次全部五组数据生成，完整数值在以上表及对应CSV，不选定H。
1. **是否真正加快？** 固定无噪声反转能直接衡量机制；真实市场只能衡量相对OLS配对提前，不证明更早发现真实趋势。
2. **快多少根？** 无噪声各方向延迟和相对OLS差见实验A；真实配对均值/中位数、提前/滞后次数见实验C，不能合并成统一准确率。
3. **交易是否增加？** 见整体trade_count及逐候选说明，闭仓数、翻转数、额外/缺失入口分别列出，噪声影响需与合成错误反向共同看。
4. **成本是否增加？** 见费用+滑点的实际变化；不能通过改变成本使EWLS有利。
5. **是否保留大赢家？** 原top10覆盖退出与区间净收益见实验D；既看覆盖仓是否早退，也看反手后整段实际损益，不能只数方向命中。
6. **年度是否更稳？** 2020/2021及独立多空见实验B/E；两年都是开发数据，不能推断未见年有效。
7. **是否连续近优？** 仅由预定四个H相邻表描述，不扩展搜索、不自动识别最佳区间；收益相近也必须核对风险/方向/成本。
8. **是否值得替换？** 见下方数据驱动审阅意见。原OLS保持不变，所有候选仍未冻结，是否后续验证由负责人决定。

## 限制、执行风险与复现

保证金不足记录而不阻断开仓，保持S0研究口径；funded_execution_feasible=false时后续收益是潜在资金不足下的假设路径，不能称可实盘融资执行。未建资金费率、交易所强平或借贷模型。OHLCV开盘成交仅假设，零量记录原样保留；zero_volume_execution_audit.csv逐项报告不确定性。期末开放仓单列open_positions.csv，未当已平仓计PF/胜率/集中度。
新实现没有调用旧Kalman、V18、止损执行器。新增程序重复运行每个估计器/模拟器并精确对账；前缀和未来扰动检查通过，分年分月及多空损益、账本、费、滑点和权益对账通过。旧文件和结果全部逐Git blob及SHA256核验。
复现（仓库根目录）：`$env:PYTHONPATH="$PWD/.deps"; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest -q`，然后 `python experiments/s1_3_recent_weighted_slope/run_experiment.py`，最后 `python tools/verify_s1_3_artifacts.py`。
diagnostic_metadata.json记录数据/配置/代码哈希、依赖、冻结基准；artifact_manifest.json覆盖本目录除自身外全部交付文件，不引入提交自哈希循环。run.log记录门禁及每组执行。完成后停止，不进入2022/2023，不选最佳H、不开展更细搜索。
'''
    ix=table.set_index('model');rx=response.set_index('model');wx=winners.set_index('model')
    costs=table.fees+table.slippage_usdt
    facts=[]
    for name,h in MODELS[1:]:
        r=ix.loc[name];s=rx.loc[name];w=wx.loc[name]
        cc=conc[(conc.model==name)&(conc.period=='overall')&(conc.direction=='all')&(conc.top_k==10)].iloc[0]
        clean_delay=clean.loc[clean.model==name,'response_delay_bars'].iloc[0]
        facts.append(f'- **{name}**：无噪声延迟{clean_delay:.0f}根，比OLS少{31-clean_delay:.0f}根；真实匹配平均提前{-s.signed_delay_mean_bars:.2f}根（非准确率）。平仓多{r.trade_count-base.trade_count:.0f}笔，总成本增加{r.fees+r.slippage_usdt-costs.iloc[0]:.2f} USDT；原top10区间保留{w.top10_interval_retention_ratio*100:.2f}%，{w.top10_covering_early_exits:.0f}/10覆盖仓提前退出；移除候选自身top10后的描述净收益{cc.net_without_top_winners:.2f} USDT。')
    a64=annual[annual.model=='H64'];a32=annual[annual.model=='H32']
    opinion=f'''## 八个问题与数据支持的审阅意见

1. **EWLS是否真正缩短响应？** 在两个无噪声已知反转中，是。OLS延迟31根；H64/H32/H16/H8为29/27/24/17根。最高噪声上转下案例H8发生1次错误反向，其他候选在这些固定反转案例为0；不能把更快当成更正确，合成不代表真实市场。
2. **缩短多少根？** 无噪声分别少2/4/7/14根，即30/60/105/210分钟。真实反向配对有符号均值依次为{rx.loc['H64','signed_delay_mean_bars']:.2f}/{rx.loc['H32','signed_delay_mean_bars']:.2f}/{rx.loc['H16','signed_delay_mean_bars']:.2f}/{rx.loc['H8','signed_delay_mean_bars']:.2f}根；未匹配事件全部保留，非真实趋势检测延迟。
3. **交易是否增加？** 全部增加，从1147笔到1163/1183/1366/2028，分别增加16/36/219/881笔。H8额外交易最多，胜率从{base.win_rate*100:.1f}%降至{ix.loc['H8','win_rate']*100:.1f}%；没有证据说更多翻转提高了整体质量。
4. **成本是否增加？** 原总手续费+滑点{costs.iloc[0]:.2f} USDT，四个EWLS依次{costs.iloc[1]:.2f}/{costs.iloc[2]:.2f}/{costs.iloc[3]:.2f}/{costs.iloc[4]:.2f}。H8增加{100*(costs.iloc[4]/costs.iloc[0]-1):.1f}%。收益退化不能只归因费用：整体多头贡献随H缩短由6515.40降至2979.44，空头负贡献由-2027.93扩大至-5466.96。
5. **是否保留原始大赢家？** H64/H32仍有10/10原赢家区间盈利，但8/10入口同向覆盖仓提前退出；整段实际净收益保留87.71%/85.82%。H16保留77.86%；H8仅49.63%，原10个区间有3个转负。提前退出数量不等于丢失全部趋势收益。各EWLS删除自身前10赢家后均为负，收益集中依赖没有解决。
6. **是否改善年度稳定性？** H64的2020/2021收益为{a64.iloc[0].net_profit:.2f}/{a64.iloc[1].net_profit:.2f}，年差{abs(a64.iloc[0].net_profit-a64.iloc[1].net_profit):.2f}，较原2864.13/1623.34更平衡，最弱年也提高；H32为{a32.iloc[0].net_profit:.2f}/{a32.iloc[1].net_profit:.2f}，两年正但最弱年低于原。H16的2020负收益，H8两年负收益。这里只是两个开发年份的描述证据。
7. **是否连续近优？** H32—H64形成一个相邻的风险/收益权衡区域：两年均正，回撤{ix.loc['H32','max_drawdown_usdt']:.2f}/{ix.loc['H64','max_drawdown_usdt']:.2f}均低于原2555.87，最长水下{ix.loc['H32','longest_underwater_days']:.2f}/{ix.loc['H64','longest_underwater_days']:.2f}天低于原220.97，净收益差717.74。可称值得审阅的连续区域，不能声称已经证明“稳健近优”。向H16/H8继续缩短明显退化，不支持只追求最快响应。
8. **是否值得替代原OLS？** **现在不足以直接替代。** H64/H32的回撤分别下降{100*(1-ix.loc['H64','max_drawdown_usdt']/base.max_drawdown_usdt):.1f}%/{100*(1-ix.loc['H32','max_drawdown_usdt']/base.max_drawdown_usdt):.1f}%，但净收益损失{100*(1-ix.loc['H64','net_profit']/base.net_profit):.1f}%/{100*(1-ix.loc['H32','net_profit']/base.net_profit):.1f}%，PF和Sharpe均下降；最大连亏由11增至18/15。它们不是对OLS的全面改进，H16/H8更不支持采用。研究建议是先审阅H32—H64的风险折中、短持仓与空头拖累，不加新条件、不自动冻结任何H；原OLS继续作为基准。后续是否冻结及验证，交负责人决定。

{chr(10).join(facts)}

以上是固定开发样本的探索性解释，不是选定参数或后续检验结论。没有开展阈值、确认根数、冷却或其他新搜索。

## 图表与原始数据浏览

交互式原始行情页面（额外展示交付，不改策略）：[market_explorer.html](market_explorer.html)，由`tools/build_s1_3_explorer.py`生成；包括逐根OHLCV、信号、成交、五组资金曲线及持仓时长描述统计。密集图的像素聚合会明确标注，悬浮始终读取原始单根。

![固定首七日斜率](figures/slope_comparison.png)
![固定合成机制](figures/synthetic_responses.png)
![五组全程资金](figures/equity_curves.png)
![五组全程回撤](figures/drawdown_curves.png)
![H与净收益](figures/H_net_profit.png)
![H与最大回撤](figures/H_drawdown.png)
![H与交易数](figures/H_trade_count.png)
![多空净贡献](figures/long_short.png)
![收益集中度](figures/profit_concentration.png)

'''
    start=text.index('## 八个问题与审阅结论');end=text.index('## 限制、执行风险与复现')
    text=text[:start]+opinion+text[end:]
    (OUT/'s1_3_report.md').write_text(text,encoding='utf-8')

def run():
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(message)s',handlers=[logging.FileHandler(OUT/'run.log',mode='w',encoding='utf-8'),logging.StreamHandler()],force=True)
    logging.info('S1.3 fixed H: infinity,64,32,16,8 | training only | no stop | no selection')
    frozen=frozen_snapshot();logging.info('All %d base tracked files unchanged',len(frozen))
    protocol=json.loads((ROOT/'config/s1_3_recent_weighted_slope.json').read_text())
    assert protocol['H_grid']==[None,64,32,16,8] and protocol['N']==64
    cfg=json.loads((ROOT/protocol['execution_config']).read_text())
    d,data=load_training(protocol['data_path']);assert data['sha256']==protocol['data_sha256']
    assert len(d)==70176 and str(d.date.iloc[-1])=='2021-12-31 23:45:00+00:00'
    b,original_signal,baseline,gate=control_gate(d,cfg);logging.info('PASS baseline gate: exact infinity slopes/signals/serialized ledgers; curves/costs/directions reconciled')
    synthetic,paths=synthetic_experiments(protocol['synthetic'],MODELS)
    synthetic2,paths2=synthetic_experiments(protocol['synthetic'],MODELS)
    pd.testing.assert_frame_equal(synthetic,synthetic2,check_exact=True);pd.testing.assert_frame_equal(paths,paths2,check_exact=True)
    table=[];annual=[];monthly=[];directions=[];concs=[];responses=[];matches=[];comparisons=[];winner_summaries=[];episodes=[]
    curves={};trades={};events={};slopes=pd.DataFrame({'date':d.date})
    intervals=[('overall',cfg['performance_start'],cfg['performance_end_exclusive']),('2020',cfg['performance_start'],'2021-01-01T00:00:00Z'),('2021','2021-01-01T00:00:00Z',cfg['performance_end_exclusive'])]
    for name,h in MODELS:
        beta=b.copy() if h is None else estimate_ewls(d.close,h)
        sig=generate_signals(beta)
        np.testing.assert_array_equal(beta,estimate_ewls(d.close,h))
        cut=40000;np.testing.assert_array_equal(beta[:cut],estimate_ewls(d.close.iloc[:cut],h))
        future=d.close.to_numpy().copy();future[cut:]*=np.linspace(1,3,len(future)-cut)
        np.testing.assert_array_equal(beta[:cut],estimate_ewls(future,h)[:cut])
        method='B_ols' if h is None else 'EWLS_'+name
        t,c,e=baseline if h is None else simulate(d,sig,method,64,cfg)
        repeated=simulate(d,sig,method,64,cfg)
        for x,y in zip((t,c,e),repeated):pd.testing.assert_frame_equal(x,y,check_exact=True)
        np.testing.assert_allclose(t.net_pnl.sum(),c.equity.iloc[-1]-10000,rtol=0,atol=1e-7)
        for direction in ['long','short']:
            np.testing.assert_allclose(t.loc[t.direction==direction,'net_pnl'].sum(),c[direction+'_pnl'].iloc[-1],rtol=0,atol=1e-7)
        np.testing.assert_allclose(e.fee.sum(),c.fees.iloc[-1],rtol=0,atol=1e-7)
        np.testing.assert_allclose(e.slippage_usdt.sum(),c.slippage_usdt.iloc[-1],rtol=0,atol=1e-7)
        assert (e.time==e.source_signal_bar+pd.Timedelta(minutes=15)).all()
        curves[name]=c;trades[name]=t;events[name]=e
        slopes[name+'_slope']=beta;slopes[name+'_signal']=sig
        for period,start,end in intervals:
            row=summarize(c,t,e,cfg,start,end,period);extra,dd=risk(c,t,start,end)
            row.update(extra,model=name,H=h);(table if period=='overall' else annual).append(row)
            if len(dd):dd['model']=name;dd['period']=period;episodes.append(dd)
            z=t[(t.status=='closed')&(t.exit_time>=pd.Timestamp(start))&(t.exit_time<pd.Timestamp(end))]
            concs.extend(concentration(z,row['net_profit'],name,period))
            for direction in ['long','short']:
                dc=c.copy();dc['pnl_change']=dc[direction+'_pnl_change'];dc['equity']=10000+dc['pnl_change'].cumsum()
                dc[('short' if direction=='long' else 'long')+'_pnl_change']=0.
                dc['fees_change']=e[e.direction==direction].groupby('time').fee.sum().reindex(pd.DatetimeIndex(dc.date),fill_value=0).to_numpy()
                dc['slippage_usdt_change']=e[e.direction==direction].groupby('time').slippage_usdt.sum().reindex(pd.DatetimeIndex(dc.date),fill_value=0).to_numpy()
                dt=t[t.direction==direction];de=e[e.direction==direction]
                dr=summarize(dc,dt,de,cfg,start,end,period);dx,_=risk(c,t,start,end,direction)
                dr['shared_account_min_equity']=row['min_actual_equity']
                dr['shared_account_funded_execution_feasible']=row['funded_execution_feasible']
                dr.update(dx,model=name,H=h,direction=direction);directions.append(dr)
                np.testing.assert_allclose(dr['net_profit'],row[direction+'_net_profit'],atol=1e-7,rtol=0)
                concs.extend(concentration(z[z.direction==direction],dr['net_profit'],name,period,direction))
        for key,g in c.groupby(c.date.dt.strftime('%Y-%m'),sort=True):
            start=g.date.iloc[0];end=g.date.iloc[-1]+pd.Timedelta(minutes=15)
            m=summarize(c,t,e,cfg,start,end,key);m.update(model=name,H=h);monthly.append(m)
        match=match_flips(original_signal,sig,95,len(d)-1,d.date);match['model']=name;matches.append(match)
        summary=response_summary(original_signal,sig,match,95,len(d)-1)
        oi=set(zip(baseline[0].entry_time,baseline[0].direction));ci=set(zip(t.entry_time,t.direction))
        summary.update(model=name,H=h,exact_new_entry_identities=len(ci-oi),exact_missing_original_entries=len(oi-ci),
            additional_closed_trades=int(t.status.eq('closed').sum()-1147))
        responses.append(summary)
        comp=original_trade_comparison(baseline[0],t,d);comp['model']=name;comparisons.append(comp)
        if h is None:np.testing.assert_allclose(comp.interval_pnl_change_vs_original,0,atol=1e-8,rtol=0)
        top=comp[comp.is_original_top10]
        winner_summaries.append(dict(model=name,H=h,top10_original_net_pnl=top.original_net_profit.sum(),
            top10_actual_interval_pnl=top.actual_interval_pnl.sum(),
            top10_interval_retention_ratio=top.actual_interval_pnl.sum()/top.original_net_profit.sum(),
            top10_intervals_positive=int(top.actual_interval_pnl.gt(0).sum()),
            top10_covering_early_exits=int(top.first_covering_position_exited_early.sum()),
            top10_no_same_direction_cover=int((~top.same_direction_position_at_original_entry).sum()),
            top10_covering_smaller_positive=int(top.covering_trade_smaller_positive_winner.sum()),
            top10_interval_smaller_positive=int(top.interval_smaller_positive_profit.sum()),
            top10_intervals_became_losses=int(top.interval_became_loss.sum())))
        logging.info('%s complete: net %.6f | closed %d | fees %.6f | slip %.6f | repeated exactly',name,c.equity.iloc[-1]-10000,t.status.eq('closed').sum(),e.fee.sum(),e.slippage_usdt.sum())
    table=pd.DataFrame(table);annual=pd.DataFrame(annual);monthly=pd.DataFrame(monthly);directions=pd.DataFrame(directions)
    conc=pd.DataFrame(concs);response=pd.DataFrame(responses);winner=pd.DataFrame(winner_summaries)
    comparison=pd.concat(comparisons,ignore_index=True)
    for name,h in MODELS:
        total=table[table.model==name].iloc[0]
        for frame in [annual,monthly]:
            for field in ['net_profit','long_net_profit','short_net_profit','fees','slippage_usdt','trade_count']:
                np.testing.assert_allclose(frame.loc[frame.model==name,field].sum(),total[field],atol=1e-7,rtol=0)
    stability=[];finite=table[table.model!='OLS'].sort_values('H').reset_index(drop=True)
    for i in range(3):
        a=finite.iloc[i];b_=finite.iloc[i+1]
        stability.append(dict(H_left=a.H,H_right=b_.H,left_net=a.net_profit,right_net=b_.net_profit,
            net_gap_usdt=b_.net_profit-a.net_profit,left_dd=a.max_drawdown_usdt,right_dd=b_.max_drawdown_usdt,
            left_trades=a.trade_count,right_trades=b_.trade_count,
            left_worst_year=annual.loc[annual.model==a.model,'net_profit'].min(),right_worst_year=annual.loc[annual.model==b_.model,'net_profit'].min()))
    stability=pd.DataFrame(stability)
    eq=pd.DataFrame({'date':curves['OLS'].date});dd=eq.copy()
    for name,h in MODELS:eq[name]=curves[name].equity;dd[name]=curves[name].drawdown_usdt
    all_t=pd.concat([t.assign(model=name) for name,t in trades.items()],ignore_index=True)
    all_e=pd.concat([e.assign(model=name) for name,e in events.items()],ignore_index=True)
    outputs={'parameter_comparison.csv':table,'annual_comparison.csv':annual,'monthly_comparison.csv':monthly,
        'signal_response_analysis.csv':response,'signal_flip_comparison.csv':pd.concat(matches,ignore_index=True),
        'long_short_comparison.csv':directions,'trade_level_comparison.csv':comparison,
        'profit_concentration.csv':conc,'synthetic_response_tests.csv':synthetic,'synthetic_paths.csv':paths,
        'equity_curves.csv':eq,'drawdown_curves.csv':dd,'slope_signal_paths.csv':slopes,
        'original_top10_winner_audit.csv':comparison[comparison.is_original_top10],
        'original_top10_summary.csv':winner,'parameter_stability.csv':stability,
        'trades_all.csv':all_t,'executions_all.csv':all_e,'open_positions.csv':all_t[all_t.status!='closed'],
        'zero_volume_execution_audit.csv':all_e[all_e.zero_volume],'drawdown_episodes.csv':pd.concat(episodes,ignore_index=True)}
    for file,frame in outputs.items():write_csv(file,frame)
    active=['config/s1_3_recent_weighted_slope.json','src/recent_weighted_slope.py','src/recent_weighted_analysis.py',
        'experiments/s1_3_recent_weighted_slope/run_experiment.py','tests/test_recent_weighted.py',
        'tools/verify_s1_3_artifacts.py','docs/s1_3_research_protocol.md']
    meta=dict(task=protocol['task'],base_commit=BASE,data=data,baseline_gate=gate,protocol=protocol,
        config_sha256=sha256(ROOT/'config/s0_from_scratch.json'),active_code_sha256={p:sha256(ROOT/p) for p in active},
        frozen_base_file_sha256=frozen,python=platform.python_version(),
        dependencies={k:importlib.metadata.version(k) for k in ['numpy','pandas','matplotlib','pyarrow','pytest']},
        checks=['all original tracked files unchanged','all five slopes/simulations exact repeat','prefix/future perturbation',
            'next-open execution','ledger/equity/cost reconciliation','long/short separately reconciled','year/month additive reconciliation',
            'original baseline interval attribution individually reconciled','deterministic synthetic repeat'],
        selected_H=None,validation_2022_2023_run=False,read_2024_plus=False,
        test_evidence='Run python -m pytest -q and verifier separately; tests are not fabricated by this runner',
        git_note='Final commit hash is recorded in Git and delivery, not in its own tracked content')
    (OUT/'diagnostic_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    figures(table,curves,slopes,synthetic,paths,conc);report(table,annual,response,winner,conc,synthetic,meta,stability)
    assert frozen_snapshot()==frozen
    logging.info('PASS all accounting, independent direction, temporal, frozen-file and determinism gates. STOP for review; no H selected, no holdout.')
    logging.shutdown()
    artifacts={str(p.relative_to(OUT)).replace('\\','/'):dict(sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts}
    (OUT/'artifact_manifest.json').write_text(json.dumps(dict(algorithm='sha256',excludes=['artifact_manifest.json'],artifacts=artifacts),indent=2),encoding='utf-8')

if __name__=='__main__':run()
