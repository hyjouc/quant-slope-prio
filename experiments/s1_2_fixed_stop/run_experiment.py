"""Seven predeclared N64 training scenarios; isolated stop extension, no holdout use."""
from pathlib import Path
import sys,json,logging,subprocess,io,platform,importlib.metadata
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
from src.fixed_stop_backtester import simulate_fixed_stop
from src.metrics import summarize
from src.fixed_stop_analysis import risk_summary,original_winner_audit

OUT=Path(__file__).parent;S0=ROOT/'experiments/s0_from_scratch_slope'
BASE='476be2cb349f85732a498e4bd0a7ddefdd44432b'
GRID=[None,.01,.02,.03,.04,.05,.06]

def frozen_snapshot():
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],text=True).splitlines()
    import hashlib
    hashes={}
    for path in paths:
        expected=subprocess.check_output(['git','show',BASE+':'+path])
        if (ROOT/path).read_bytes()!=expected:raise ValueError('Frozen S0/S1.1 file changed: '+path)
        hashes[path]=hashlib.sha256(expected).hexdigest()
    return hashes

def serialized(frame):
    return pd.read_csv(io.StringIO(frame.to_csv(index=False,float_format='%.12g')))

def assert_control(t,c,e,original):
    ot,oc,oe=original
    pd.testing.assert_frame_equal(t[TRADE_COLUMNS],ot,check_exact=True)
    pd.testing.assert_frame_equal(c,oc,check_exact=True)
    pd.testing.assert_frame_equal(e[EXEC_COLUMNS],oe,check_exact=True)
    # Compare both reconstructed and serialized control to original saved S0 CSV.
    saved_t=pd.read_csv(S0/'trades_all.csv');saved_t=saved_t[saved_t.candidate=='B_ols_N64'].reset_index(drop=True)
    saved_e=pd.read_csv(S0/'executions_all.csv');saved_e=saved_e[saved_e.candidate=='B_ols_N64'].reset_index(drop=True)
    pd.testing.assert_frame_equal(serialized(t[TRADE_COLUMNS]),saved_t,check_exact=True)
    pd.testing.assert_frame_equal(serialized(e[EXEC_COLUMNS]),saved_e,check_exact=True)
    eq=pd.read_csv(S0/'equity_curves.csv');dd=pd.read_csv(S0/'drawdown_curves.csv')
    np.testing.assert_allclose(c.equity,eq.B_ols_N64,rtol=0,atol=1e-6)
    np.testing.assert_allclose(c.drawdown_usdt,dd.B_ols_N64,rtol=0,atol=1e-6)
    r=pd.read_csv(S0/'parameter_comparison.csv').query("candidate=='B_ols_N64'").iloc[0]
    assert int(t.status.eq('closed').sum())==r.trade_count==1147
    for value,ref in [(t.net_pnl.sum(),r.net_profit),(e.fee.sum(),r.fees),(e.slippage_usdt.sum(),r.slippage_usdt)]:
        np.testing.assert_allclose(value,ref,atol=1e-6,rtol=0)

def plots(table,curves,winners):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        x=np.arange(7);labels=table.scenario
        for ax,field,title in zip(axs.flat,['net_profit','max_drawdown_usdt','longest_underwater_days','profit_factor'],
            ['Net profit (USDT)','Max close-MTM drawdown (USDT)','Longest underwater period (days)','Profit factor']):
            ax.plot(x,table[field],'o-');ax.axhline(table[field].iloc[0],color='gray',linestyle='--',label='Frozen no-stop control')
            ax.set_xticks(x,labels,rotation=25);ax.set_ylabel(title);ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('Seven fixed N64 scenarios | 2020-2021 only | no automatic selection')
        fig.savefig(folder/'seven_scenario_comparison.png',dpi=170);plt.close(fig)
        for field,title in [('equity','Account equity (USDT)'),('drawdown_usdt','Close-MTM drawdown (USDT)')]:
            fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
            for j,(name,c) in enumerate(curves.items()):
                ax.plot(c.date,c[field],label=name,linewidth=1.3 if j==0 else .7,linestyle=['-','--',':','-.'][j%4],color='black' if j==0 else plt.colormaps['viridis'](j/7))
            ax.set(xlabel='UTC candle open label; all 15m observations, no smoothing',ylabel=title)
            ax.legend(ncol=4,fontsize=8);ax.grid(alpha=.2);fig.savefig(folder/f'{field}_curves.png',dpi=170);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        axs[0].bar(x,table.original_top10_entry_early_stop_count,hatch='//',label='Same original entry actually stopped early')
        axs[0].plot(x,table.original_top10_isolated_touch_count,'o--',color='black',label='Isolated original-entry touch (not strategy PnL)')
        axs[0].set(ylabel='Original top10 winners (count)',ylim=(0,10.5));axs[0].legend(fontsize=7)
        axs[1].bar(x,table.original_top10_episode_net_profit,hatch='..',label='Actual episode PnL incl. reentries')
        axs[1].axhline(table.original_top10_episode_net_profit.iloc[0],color='gray',linestyle='--',label='Original top10 sum')
        axs[1].set_ylabel('Net profit on original winner episodes (USDT)');axs[1].legend(fontsize=7)
        for ax in axs:ax.set_xticks(x,labels,rotation=25);ax.grid(axis='y',alpha=.2)
        fig.savefig(folder/'original_top10_retention.png',dpi=170);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for ax,fields,title in [(axs[0],['long_net_profit','short_net_profit'],'Direction net contributions (USDT)'),
            (axs[1],['fees','slippage_usdt'],'Incurred execution costs (USDT)')]:
            ax.bar(x-.18,table[fields[0]],.36,label=fields[0],hatch='//');ax.bar(x+.18,table[fields[1]],.36,label=fields[1],hatch='..')
            ax.set_xticks(x,labels,rotation=25);ax.axhline(0,color='black',linewidth=.6);ax.set_ylabel(title);ax.legend(fontsize=8)
        fig.savefig(folder/'directions_and_costs.png',dpi=170);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        axs[0].plot(x,table.max_consecutive_losses,'o-',label='Max consecutive net losses')
        axs[1].plot(x,-table.max_single_trade_loss,'o-',label='Largest closed net loss (absolute USDT)')
        for ax in axs:ax.set_xticks(x,labels,rotation=25);ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.savefig(folder/'loss_risk_comparison.png',dpi=170);plt.close(fig)

def md(table,columns):
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,r in table.iterrows():
        vals=[f'{r[c]:.3f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in columns]
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

def make_report(table,annual,meta):
    baseline=table.iloc[0];rows=[]
    for r in table.iloc[1:].itertuples():
        rows.append(f'- {r.scenario}：净收益{r.net_profit:.2f} USDT（相对对照{r.net_profit-baseline.net_profit:+.2f}），最大收盘回撤{r.max_drawdown_usdt:.2f}（{100*(r.max_drawdown_usdt/baseline.max_drawdown_usdt-1):+.1f}%），最长水下{r.longest_underwater_days:.2f}天；原始top10同入口提前止损{r.original_top10_entry_early_stop_count}/10，经历重入后的原赢家区间净收益{r.original_top10_episode_net_profit:.2f} USDT。')
    text=f'''# S1.2 固定价格止损：2020—2021七组结果

## 范围与冻结

S0策略提交fd6658dbde9d046488577062dc53b84ef621c4bf保持字节不变，S1.1提交{BASE}及全部已跟踪文件也不改动。
只研究OLS N64，唯一新增参数L=[1%,2%,3%,4%,5%,6%]，多空共用；另保留无止损对照。N96不重跑、不作为新搜索维度。
严格只计算2020—2021。2022/2023未计算绩效、未用于选L；它们曾用于旧研究，不能称独立未见OOS。未读取2024+。本轮不自动选参数、不冻结候选、不运行第二轮。

## 七组总表（全部展示，不按收益排名）

{md(table,['scenario','net_profit','max_drawdown_usdt','max_drawdown_pct_peak','longest_underwater_days','max_single_trade_loss','max_consecutive_losses','profit_factor','sharpe','sortino','trade_count','stop_exits','fees','slippage_usdt'])}

{chr(10).join(rows)}

## 四个研究问题

本轮最明确的共同结果是：**六个L的最大单笔亏损都减少，但净利润、PF、Sharpe全部低于无止损对照，最大收盘回撤全部高于对照，最长水下期没有任何一组缩短。** 这不满足“风险曲线明显改善且利润不过度牺牲”的初衷，不能因单笔亏损变小就宣布策略质量提高。
1%—2%邻域均呈更高回撤、更长水下和更低PF；4%—6%的原top10保持完整，但整体指标仍未超过对照，邻近表现接近原策略不等于有新增收益。
因此，这组训练证据暂不支持为整体质量改善而采用所测固定止损。保留原无止损作为审阅对照，不自动将任何L写入策略配置；是否继续及如何解释风险预算由负责人决定，不能为改善结果扩大网格。

### 1. 是否减少大亏损

最大单笔亏损和最大回撤见总表；价格L不是账户或杠杆后亏损。2000名义、1000保证金下，止损价处退出的毛损失约2000×L，保证金亏损约2L，再计滑点及双边费。
跳空按更差open成交，固定L不是最大实际损失保证。盘中触发再计退出不利滑点，因此最大净亏损也可超过2000×L。stop_gap_audit.csv保留每项跳空。
回撤为15分钟close MTM，含初始10000峰值，不是交易所强平或完整盘中回撤；止损不能据此证明交易所强平风险已解决。

### 2. 是否误杀原始大赢家（重点top10）

{md(table,['scenario','original_top10_entry_early_stop_count','original_top10_same_entry_skipped_count','original_top10_held_intact_count','original_top10_isolated_touch_count','original_top10_episode_net_profit','original_top10_episode_profit_retention','original_top10_episodes_still_positive'])}

原始top10固定为S0 N64已平仓净收益最高10笔，原总净收益{baseline.original_top10_episode_net_profit:.2f} USDT；不是各L重新选出10笔。
“同入口提前止损”必须实际模拟中在原入场时刻开仓且在原退出之前被止损；“完整保留”是同入口持有至原反向退出；若之前gap-stop使原入口被跳过，会单列，不冒充提前止损。
“独立原入口触及”仅用原交易入场fill与原路径做固定止损是否触及的审计，排除原退出K线后续极值。它不含策略止损后的重入，不当作真实候选净收益。
原赢家区间收益是候选在该原始同向区间内所有实际开仓（含多次重入）的净收益，不能只数首次提前退出就认定整段利润永久丢失。all_original_episode_audit.csv覆盖所有原始已平仓区间；original_top10_winner_audit.csv逐笔列出10笔。

3%组原top10中仅1笔同入口被提前止损，而十个原区间计入实际重入后的净收益为{table.loc[table.scenario=='L03pct','original_top10_episode_net_profit'].iloc[0]:.2f} USDT，保留{100*table.loc[table.scenario=='L03pct','original_top10_episode_profit_retention'].iloc[0]:.2f}%。1%组首次提前止损5/10，但这些区间仍保留{100*table.loc[table.scenario=='L01pct','original_top10_episode_profit_retention'].iloc[0]:.2f}%。实际重入使“首次误杀比例”不能直接等同永久失去利润的比例。
这一结果不能证明止损无害：区间外的重入与小额亏损串也会累积，须同时看全体账本。1%组实际手续费+滑点共{(table.loc[table.scenario=='L01pct','fees']+table.loc[table.scenario=='L01pct','slippage_usdt']).iloc[0]:.2f} USDT，而对照为{baseline.fees+baseline.slippage_usdt:.2f} USDT。成本上升与更频繁的止损重入相伴；交易路径毛收益也改变，不能把净利润差额全部归因于费用。

### 3. 资金曲线、连亏及水下期

{md(table,['scenario','longest_underwater_days','underwater_at_end','max_consecutive_losses','max_loss_streak_usdt','entry_count','stop_exits','gap_stop_exits','intrabar_stop_exits','margin_budget_shortfall_entries','min_actual_equity'])}

年/月界不清仓、不重置斜率、信号、持仓或止损；每次新开仓按新实际fill固定止损。止损后无冷却，最早下一根open使用上一close的原信号重入。
连亏按模拟事件顺序统计（同根正常反向退出、新仓又触及止损可产生两笔），不捏造盘中时刻。水下期按实际close，未恢复时长为截至训练末端的下界。

### 4. 是否提高综合质量、是否牺牲利润

{md(table,['scenario','long_net_profit','short_net_profit','profit_factor','sharpe','monthly_profit_ratio','top10_profit_usdt','top10_share_winning_profit','net_without_top10_winning_trades','net_without_top20_winning_trades'])}

这张集中度表的top10/top20为每个候选自身的已平仓赢家，与固定“原始top10审计”不同。净收益保留期末未平仓MTM；删除赢家仅为描述性扣除，不是可执行回测。
判定不能只看最高净利润，应联合回撤、水下期、原赢家区间保留、年度、多空、成本与邻域。没有事后构造评分权重或接受阈值。neighborhood_stability.csv只描述既定相邻L的区间，不插入新L、不外推。

## 年度结果与边界

{md(annual,['scenario','period','net_profit','long_net_profit','short_net_profit','max_drawdown_usdt','profit_factor','trade_count'])}

年/月收益是S0原开盘标签归属的MTM增量；PF/胜率/持仓时长只看实际平仓，整笔归退出K线所属期间。末期持仓标记不伪造退出或扣未发生退出费。
日Sharpe=日MTM变化/固定10000，rf=0、sqrt365及样本SD；Sortino负部RMS。参数仅改变价格止损，不改资金成本、名义本金或N。

## 事件顺序与执行不确定性

1. 每根open先检查旧仓是否gap穿越/等于原固定止损；若触发，按raw open加不利滑点退出，该open不重入，最早下一根open重入。即使恰有反向收盘信号，也先标gap-stop，不在同一时刻重入。
2. 无gap-stop时，按上一close原信号执行正常反向平旧开新；此前已经在open退出的仓位不再使用本根高低价判止损。
3. 对open后持仓（含新仓），本根不利high/low触及止损价即按stop加不利滑点退出；不允许本根再次入场。只有固定止损，无止盈，因此无需推测high/low先后。
4. 盘中触及具体时刻未知。trade/执行的time是bar-open标签，time_precision明确为within_15m_unknown；持仓时长同时保存下界及上界（相差0.25小时），总表平均持仓是下界，另列average_holding_hours_upper。不存在精确盘中路径或精确分钟持仓推断。

输入SHA256 `{meta['data_sha256']}`，只读六列原始OHLCV，先核验哈希，之后只传2020—2021给指标与执行。
原始零成交量记录保留，每个L上的假设成交均在zero_volume_execution_audit.csv单列；对应总事件数见diagnostic_metadata.json。资金费、交易所强平、盘口深度及撮合延迟不建模。
账户10000、固定名义2000、单边手续费0.05%/滑点0.02%、费用另扣、不复利。若资金不足只标记，不偷偷添加拒单或缩仓规则；全期延续结果须按资金可行性标记审阅。

## 质量核验与停止

**在运行任何L之前，无止损扩展器与冻结S0执行器逐笔、执行、逐根权益精确一致，并与原CSV的交易数/费用/滑点/权益/回撤对照通过。**
七组都完整运行两次，交易、事件、每根曲线精确相同；逐笔/多空/年度/月度/费用/滑点对账通过。完整训练前缀及未来扰动检查通过；新增事件顺序与成本等测试和全部旧测试通过。
S0/S1.1所有原跟踪文件与基准Git内容字节一致，未修改原策略、结果或配置。data/config/code哈希和产物清单留存。

完成七组即停止，等待负责人审阅。若六个L均不能兼顾风险改善、赢家保留、利润及邻域，应保留原无止损；本轮不为强行改善结果增加条件，也不自动进入2022—2023。

## 图表

![七组比较](figures/seven_scenario_comparison.png)
![权益](figures/equity_curves.png)
![回撤](figures/drawdown_usdt_curves.png)
![原始top10保留](figures/original_top10_retention.png)
![方向与成本](figures/directions_and_costs.png)
![连亏与单笔风险](figures/loss_risk_comparison.png)
'''
    (OUT/'s1_2_fixed_stop_report.md').write_text(text,encoding='utf-8')

def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log',mode='w',encoding='utf-8'),logging.StreamHandler(sys.stdout)],force=True)
    cfg_path=ROOT/'config/s1_2_fixed_stop.json';cfg=json.loads(cfg_path.read_text());cfg_hash=sha256(cfg_path)
    assert cfg['stop_grid']==GRID and cfg['N']==64 and cfg['holdout_performance_authorized'] is False
    frozen=frozen_snapshot();s0cfg=json.loads((ROOT/'config/s0_from_scratch.json').read_text())
    for field in ['initial_account','margin_budget','leverage','notional','fee','slippage','compound','funding_model','exchange_liquidation_model']:
        assert cfg[field]==s0cfg[field],f'Frozen execution setting changed: {field}'
    logging.info('Frozen config %s; seven predeclared training scenarios; no selection',cfg_hash)
    d,data=load_training(cfg['data_path']);assert data['sha256']==cfg['data_sha256']
    beta=estimate(d.close,64,'B_ols');sig=generate_signals(beta)
    # CONTROL GATE: abort immediately on any mismatch before stopped scenarios.
    original=simulate(d,sig,'B_ols',64,s0cfg)
    control=simulate_fixed_stop(d,sig,s0cfg,None)
    assert_control(*control,original)
    logging.info('CONTROL GATE PASS: no-stop exact frozen executor + serialized original ledgers + equity/drawdown/costs; stopped runs now allowed')
    base=control[0].copy();curves={};trades=[];events=[];rows=[];annual=[];monthly=[];ddrows=[];toprows=[];episode_rows=[];repeat=[]
    for loss in GRID:
        scenario='no_stop' if loss is None else f'L{int(loss*100):02d}pct'
        t,c,e=control if loss is None else simulate_fixed_stop(d,sig,s0cfg,loss)
        t2,c2,e2=simulate_fixed_stop(d,sig,s0cfg,loss)
        for a,b in [(t,t2),(c,c2),(e,e2)]:pd.testing.assert_frame_equal(a,b,check_exact=True)
        # Actual training-time causal prefixes (contains no 2022/2023) and future perturbation.
        cut=40000;pre=simulate_fixed_stop(d.iloc[:cut],sig[:cut],s0cfg,loss)
        pd.testing.assert_frame_equal(c.iloc[:len(pre[1])],pre[1],check_exact=True)
        pd.testing.assert_frame_equal(e[e.time<d.date.iloc[cut]].reset_index(drop=True),pre[2],check_exact=True)
        changed=d.copy();changed.loc[cut:,['open','high','low','close']]*=1.5
        changed_sig=generate_signals(estimate(changed.close,64,'B_ols'))
        np.testing.assert_allclose(sig[:cut],changed_sig[:cut],rtol=0,atol=0,equal_nan=True)
        pert=simulate_fixed_stop(changed,changed_sig,s0cfg,loss)
        pd.testing.assert_frame_equal(c.iloc[:len(pre[1])],pert[1].iloc[:len(pre[1])],check_exact=True)
        pd.testing.assert_frame_equal(pre[2],pert[2][pert[2].time<d.date.iloc[cut]].reset_index(drop=True),check_exact=True)
        # Restore candidate identifiers only after exact control gate and causal tests.
        t=t.copy();c=c.copy();e=e.copy()
        for frame in [t,c,e]:frame['scenario']=scenario;frame['L']=loss;frame['candidate']=scenario
        top=original_winner_audit(base,t,d,loss);top['scenario']=scenario;toprows.append(top)
        # Reuse top10 audit mechanism on ALL original closed episodes (ranks chosen per batch)
        all_orig=base[base.status=='closed']
        episode_parts=[]
        for offset in range(0,len(all_orig),10):
            z=original_winner_audit(all_orig.iloc[offset:offset+10],t,d,loss)
            z['scenario']=scenario;episode_parts.append(z)
        all_epi=pd.concat(episode_parts,ignore_index=True).drop(columns='original_winner_rank')
        all_epi['original_trade_sequence']=all_epi.original_entry_time.map(all_orig.set_index('entry_time').trade_sequence)
        episode_rows.append(all_epi)
        metrics=summarize(c,t,e,s0cfg);risk,dd=risk_summary(c,t)
        metrics.update(risk);metrics.update(scenario=scenario,L=loss)
        metrics.update(original_top10_entry_early_stop_count=int(top.original_entry_stopped_early.sum()),
            original_top10_entry_early_stop_fraction=float(top.original_entry_stopped_early.mean()),
            original_top10_same_entry_skipped_count=int((~top.same_original_entry_taken).sum()),
            original_top10_held_intact_count=int(top.original_entry_held_to_normal_exit.sum()),
            original_top10_isolated_touch_count=int(top.isolated_original_entry_early_touch.sum()),
            original_top10_episode_net_profit=top.episode_net_profit.sum(),
            original_top10_episode_profit_retention=top.episode_net_profit.sum()/top.original_net_profit.sum(),
            original_top10_episodes_still_positive=int(top.episode_net_profit.gt(0).sum()))
        rows.append(metrics);curves[scenario]=c;trades.append(t);events.append(e);dd['scenario']=scenario;ddrows.append(dd)
        for year in [2020,2021]:
            r=summarize(c,t,e,s0cfg,f'{year}-01-01T00:00:00Z',f'{year+1}-01-01T00:00:00Z',str(year));r.update(scenario=scenario,L=loss);annual.append(r)
        for month in sorted(c.date.dt.strftime('%Y-%m').unique()):
            start=pd.Timestamp(month+'-01',tz='UTC');r=summarize(c,t,e,s0cfg,start,start+pd.offsets.MonthBegin(1),month);r.update(scenario=scenario,L=loss);monthly.append(r)
        net=c.equity.iloc[-1]-10000
        np.testing.assert_allclose(t.net_pnl.sum(),net,atol=1e-7,rtol=0)
        np.testing.assert_allclose(c.long_pnl+c.short_pnl,c.equity-10000,atol=1e-7,rtol=0)
        np.testing.assert_allclose(t.gross_before_cost-t.entry_fee-t.exit_fee-t.entry_slippage_usdt-t.exit_slippage_usdt,t.net_pnl,atol=1e-8,rtol=0)
        np.testing.assert_allclose(e.fee.sum(),c.fees.iloc[-1],atol=1e-7,rtol=0)
        np.testing.assert_allclose(e.slippage_usdt.sum(),c.slippage_usdt.iloc[-1],atol=1e-7,rtol=0)
        np.testing.assert_allclose(sum(x['net_profit'] for x in annual if x['scenario']==scenario),net,atol=1e-7,rtol=0)
        np.testing.assert_allclose(sum(x['net_profit'] for x in monthly if x['scenario']==scenario),net,atol=1e-7,rtol=0)
        # All closed original episodes + final original open interval partition strategy PnL.
        # The entry-gap priority may skip first fill; audit keeps that explicitly.
        repeat.append(dict(scenario=scenario,exact_repeated=True,training_prefix_exact=True,future_perturbation_exact=True))
        logging.info('%s net=%.2f DD=%.2f underwater=%.2fd PF=%.3f stops=%d top10-early=%d/10 top10-episode-net=%.2f; repeat/causality/ledger PASS',
            scenario,net,metrics['max_drawdown_usdt'],metrics['longest_underwater_days'],metrics['profit_factor'],metrics['stop_exits'],metrics['original_top10_entry_early_stop_count'],metrics['original_top10_episode_net_profit'])
    table=pd.DataFrame(rows);ann=pd.DataFrame(annual);mon=pd.DataFrame(monthly);ts=pd.concat(trades,ignore_index=True);ev=pd.concat(events,ignore_index=True);top=pd.concat(toprows,ignore_index=True)
    neighborhood=[];stopped=table.iloc[1:].reset_index(drop=True)
    for i,r in stopped.iterrows():
        near=stopped.iloc[max(0,i-1):min(len(stopped),i+2)]
        neighborhood.append(dict(scenario=r.scenario,L=r.L,neighbor_L=','.join(f'{x:.2f}' for x in near.L),
            endpoint_caution=i in [0,5],net_min=near.net_profit.min(),net_max=near.net_profit.max(),
            dd_worst=near.max_drawdown_usdt.max(),underwater_worst_days=near.longest_underwater_days.max(),
            pf_min=near.profit_factor.min(),original_top10_episode_retention_min=near.original_top10_episode_profit_retention.min()))
    for name,frame in [('parameter_comparison',table),('annual_comparison',ann),('monthly_comparison',mon),
        ('trades_all',ts),('executions_all',ev),('original_top10_winner_audit',top),('all_original_episode_audit',pd.concat(episode_rows,ignore_index=True)),
        ('drawdown_episodes',pd.concat(ddrows,ignore_index=True)),('neighborhood_stability',pd.DataFrame(neighborhood)),
        ('zero_volume_execution_audit',ev[ev.zero_volume]),('stop_gap_audit',ev[ev.reason=='stop_gap']),('reproducibility',pd.DataFrame(repeat))]:
        frame.to_csv(OUT/f'{name}.csv',index=False,float_format='%.12g')
    for filename,field in [('equity_curves','equity'),('drawdown_curves','drawdown_usdt')]:
        z=pd.DataFrame({'date':next(iter(curves.values())).date})
        for name,c in curves.items():z[name]=c[field]
        z.to_csv(OUT/f'{filename}.csv',index=False,float_format='%.12g')
    directions=[]
    for scenario,g in ts.groupby('scenario',sort=False):
        for side in ['long','short']:
            x=g[g.direction==side];closed=x[x.status=='closed'];w=closed.loc[closed.net_pnl>0,'net_pnl'].sum();l=-closed.loc[closed.net_pnl<0,'net_pnl'].sum()
            directions.append(dict(scenario=scenario,direction=side,net_profit=x.net_pnl.sum(),closed_trades=len(closed),
                profit_factor=w/l if l else np.nan,win_rate=closed.net_pnl.gt(0).mean(),stop_exits=int(closed.exit_reason.isin(['stop_gap','stop_touch']).sum())))
    pd.DataFrame(directions).to_csv(OUT/'direction_comparison.csv',index=False,float_format='%.12g')
    assert frozen_snapshot()==frozen and sha256(cfg_path)==cfg_hash and sha256(cfg['data_path'])==data['sha256']
    meta=dict(task=cfg['task'],base_commit=BASE,data_sha256=data['sha256'],data=data,
        config_sha256=cfg_hash,frozen_files_sha256=frozen,control_exact_s0=True,all_scenarios_repeated=True,
        all_causal_prefix_perturbation_checks=True,all_ledger_reconciliations=True,
        original_top10_fixed_entries=[str(x) for x in base.query("status=='closed'").nlargest(10,'net_pnl').entry_time],
        scenario_count=7,only_new_parameter='L',N=64,N96_researched=False,selected_L=None,
        holdout_performance_computed=False,post2023_data_read=False,
        zero_volume_execution_counts=ev.groupby('scenario').zero_volume.sum().astype(int).to_dict(),
        timing=cfg['intrabar_time_policy'],priority=cfg['open_event_priority'],
        versions={x:importlib.metadata.version(x) for x in ['numpy','pandas','pyarrow','matplotlib']},python=platform.python_version(),
        limitations=['historical reused training data','15m OHLC stop touch timing unknown','no funding/exchange liquidation/depth/latency','no automatic candidate selection'])
    plots(table,curves,top);make_report(table,ann,meta)
    (OUT/'diagnostic_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    logging.info('COMPLETE seven fixed scenarios; S0/S1.1 frozen bytes unchanged; STOP before selection/holdout')
    logging.shutdown();manifest={}
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]
    files+=[cfg_path,ROOT/'src/fixed_stop_backtester.py',ROOT/'src/fixed_stop_analysis.py',ROOT/'tests/test_fixed_stop.py',
        ROOT/'tests/test_fixed_stop_analysis.py',ROOT/'docs/s1_2_fixed_stop_protocol.md',ROOT/'tools/verify_s1_2_fixed_stop.py']
    for p in sorted(files):manifest[p.relative_to(ROOT).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha256(p))
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':main()
