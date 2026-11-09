"""K5 entry-only diagnostic. No strategy fitting, no complete strategy backtest."""
from pathlib import Path
import sys,json,logging,importlib.util,platform,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data_loader import load_training,sha256
from src.k5_entry_diagnostic import slope_change,build_events,interval_outcome

OUT=Path(__file__).parent
def save(name,df):df.to_csv(OUT/name,index=False,float_format='%.17g')
def md(df,cols):
    rows=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for _,r in df.iterrows():rows.append('| '+' | '.join(f'{r[c]:.3f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in cols)+' |')
    return '\n'.join(rows)

def summarize_events(e):
    rows=[]
    for direction in ['long','short']:
        for period in ['overall','2020','2021']:
            for group in ['all','first','later']:
                z=e[e.direction==direction]
                if period!='overall':z=z[z.entry_year==int(period)]
                if group!='all':z=z[z.is_first==(group=='first')]
                confirmed=z[z.confirmation_executable];censored=z[z.censored]
                rows.append(dict(direction=direction,period=period,group=group,events=len(z),
                    sign_cycles=z.cycle_id.nunique(),confirmation_executable=len(confirmed),censored=len(censored),
                    eventual_confirmation_ratio=len(confirmed)/len(z) if len(z) else np.nan,
                    confirm_within_1h_ratio=float((z.lead_bars<=4).mean()) if len(z) else np.nan,
                    confirm_within_4h_ratio=float((z.lead_bars<=16).mean()) if len(z) else np.nan,
                    confirm_within_16h_ratio=float((z.lead_bars<=64).mean()) if len(z) else np.nan,
                    wait_median_hours=confirmed.wait_hours.median(),wait_mean_hours=confirmed.wait_hours.mean(),
                    wait_p95_hours=confirmed.wait_hours.quantile(.95),wait_max_hours=confirmed.wait_hours.max(),
                    mae_median_pct=z.mae_pct.median(),mae_p95_pct=z.mae_pct.quantile(.95),mae_max_pct=z.mae_pct.max(),
                    mae_mean_usdt=z.mae_usdt.mean(),mae_max_usdt=z.mae_usdt.max(),
                    mfe_median_pct=z.mfe_pct.median(),net_to_confirmation_mean=confirmed.net_usdt.mean(),
                    net_to_confirmation_median=confirmed.net_usdt.median(),
                    positive_net_to_confirmation_ratio=confirmed.net_usdt.gt(0).mean(),
                    entry_price_advantage_ratio=confirmed.raw_signed_bps.gt(0).mean(),
                    endpoint_price_adverse_ratio=confirmed.raw_signed_bps.lt(0).mean(),
                    g_reversal_ratio=z.g_reversed_before_confirmation.mean(),
                    censored_mtm_mean=censored.net_usdt.mean(),censored_mae_max_pct=censored.mae_pct.max(),
                    endpoint_zero_volume_count=int(z.entry_zero_volume.sum()+z.terminal_zero_volume.sum())))
    return pd.DataFrame(rows)

def fixed_horizons(d,beta,g,e,grid):
    lookup={int(r.source_index):r for r in e.itertuples()};details=[]
    op=d.open.to_numpy();n=len(d);dates=pd.DatetimeIndex(d.date)
    for h in grid:
        for i in range(95,n-1):
            if not np.isfinite(g[i]) or not np.isfinite(g[i-1]) or beta[i]==0:continue
            side=1 if beta[i]<0 else -1;j=i+1;k=j+h
            event=lookup.get(i)
            row=dict(source_index=i,direction='long' if side==1 else 'short',entry_year=int(dates[j].year),
                is_event=event is not None,is_first=bool(event.is_first) if event else False,
                event_id=event.event_id if event else np.nan,horizon_bars=h,horizon_hours=h/4,
                entry_time=dates[j],endpoint_available=k<n)
            if k<n:
                raw=op[j];terminal=op[k];fill=raw*(1+side*.0002);qty=2000/fill;xf=terminal*(1-side*.0002)
                row.update(net_usdt=side*qty*(xf-fill)-1-qty*xf*.0005,
                    raw_signed_bps=side*(terminal/raw-1)*10000,exit_time=dates[k],
                    entry_zero_volume=bool(d.volume.iloc[j]==0),exit_zero_volume=bool(d.volume.iloc[k]==0))
            else:row.update(net_usdt=np.nan,raw_signed_bps=np.nan,exit_time=pd.NaT,entry_zero_volume=bool(d.volume.iloc[j]==0),exit_zero_volume=False)
            details.append(row)
    detail=pd.DataFrame(details);summary=[]
    for direction in ['long','short']:
        for h in grid:
            z=detail[(detail.direction==direction)&(detail.horizon_bars==h)]
            for group in ['background','all_events','first_events','later_events']:
                w=z if group=='background' else z[z.is_event]
                if group in ['first_events','later_events']:w=w[w.is_first==(group=='first_events')]
                for period in ['overall','2020','2021']:
                    a=w if period=='overall' else w[w.entry_year==int(period)]
                    b=a[a.endpoint_available]
                    summary.append(dict(direction=direction,horizon_bars=h,horizon_hours=h/4,group=group,period=period,
                        observations=len(a),available=len(b),unavailable=len(a)-len(b),
                        mean_net_usdt=b.net_usdt.mean(),median_net_usdt=b.net_usdt.median(),
                        net_positive_ratio=b.net_usdt.gt(0).mean(),mean_raw_signed_bps=b.raw_signed_bps.mean(),
                        raw_positive_ratio=b.raw_signed_bps.gt(0).mean(),
                        zero_volume_endpoint_count=int(b.entry_zero_volume.sum()+b.exit_zero_volume.sum())))
    return detail,pd.DataFrame(summary)

def run():
    OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(message)s',handlers=[logging.FileHandler(OUT/'run.log',mode='w',encoding='utf-8'),logging.StreamHandler()],force=True)
    protocol=json.loads((ROOT/'config/k5_entry_diagnostic.json').read_text())
    assert protocol['N']==64 and protocol['K']==5 and protocol['fixed_horizon_bars']==[4,16,64]
    cfg=json.loads((ROOT/'config/s0_from_scratch.json').read_text())
    d,data=load_training(cfg['data_path']);assert len(d)==70176
    # Baseline is the ONLY portfolio reconstruction; enforce original S0 identity.
    source=ROOT/'experiments/s1_3_recent_weighted_slope/run_experiment.py'
    spec=importlib.util.spec_from_file_location('s13_gate',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    beta,_,baseline,gate=module.control_gate(d,cfg)
    logging.info('PASS original S0 N64 control gate; no new strategy simulations')
    g=slope_change(beta);e,flags,cycles=build_events(d,beta,g)
    np.testing.assert_array_equal(g[:40000],slope_change(beta[:40000]))
    altered=beta.copy();altered[40000:]+=np.arange(len(beta)-40000)
    np.testing.assert_array_equal(g[:40000],slope_change(altered)[:40000])
    for i in [67,95,1000,40000,70175]:
        expected=np.linalg.lstsq(np.column_stack([np.ones(5),np.arange(5)]),beta[i-4:i+1],rcond=None)[0][1]
        np.testing.assert_allclose(g[i],expected,atol=1e-15,rtol=0)
    assert (e.entry_time==e.source_bar+pd.Timedelta(minutes=15)).all()
    np.testing.assert_allclose(e.gross_usdt-e.fees_usdt-e.slippage_usdt,e.net_usdt,atol=1e-9,rtol=0)
    summary=summarize_events(e)
    cycle_rows=[]
    for cid in np.unique(cycles[95:-1]):
        if cid<0:continue
        ix=np.flatnonzero(cycles==cid);z=e[e.cycle_id==cid];side='long' if beta[ix[0]]<0 else 'short'
        times=z.entry_time.sort_values()
        cycle_rows.append(dict(cycle_id=int(cid),direction=side,start_index=int(ix[0]),end_index=int(ix[-1]),
            source_start=d.date.iloc[ix[0]],source_end=d.date.iloc[ix[-1]],
            left_truncated=bool(ix[0]<95),right_truncated=bool(ix[-1]==len(d)-1),events=len(z),
            first_event_visible=bool(z.is_first.any()),repeat_events=int((~z.is_first).sum()),
            median_event_gap_hours=times.diff().dt.total_seconds().median()/3600 if len(times)>1 else np.nan))
    cycle_table=pd.DataFrame(cycle_rows);stats=[]
    for direction,z in cycle_table.groupby('direction'):
        with_event=z[z.events>0]
        stats.append(dict(direction=direction,cycles=len(z),cycles_with_events=len(with_event),cycles_without_event=int(z.events.eq(0).sum()),
            cycles_with_repeated_events=int(z.events.gt(1).sum()),events=int(z.events.sum()),
            first_events=int(e[(e.direction==direction)&e.is_first].shape[0]),
            later_events=int(e[(e.direction==direction)&~e.is_first].shape[0]),
            median_events_per_active_cycle=with_event.events.median(),max_events_per_cycle=z.events.max(),
            repeated_cycle_ratio=with_event.events.gt(1).mean(),
            median_within_cycle_event_gap_hours=e[e.direction==direction].groupby('cycle_id').entry_time.diff().dt.total_seconds().median()/3600))
    cycle_summary=pd.DataFrame(stats)
    details,horizons=fixed_horizons(d,beta,g,e,protocol['fixed_horizon_bars'])
    # Original winners: no simulated replacement exit and no best-event hindsight selection.
    original=baseline[0];closed=original[original.status=='closed'];top=closed.nlargest(10,'net_pnl')
    winners=[];winner_events=[]
    for rank,t in enumerate(top.itertuples(),1):
        source_index=int(pd.DatetimeIndex(d.date).get_indexer([t.entry_source_bar])[0])
        preceding=cycles[source_index-1] if source_index else -1
        z=e[(e.cycle_id==preceding)&(e.direction==t.direction)&(e.confirmation_execution_time==t.entry_time)]
        winners.append(dict(original_winner_rank=rank,direction=t.direction,original_entry=t.entry_time,original_exit=t.exit_time,
            original_net_usdt=t.net_pnl,preceding_cycle=int(preceding),eligible_events=len(z),
            first_event_exists=bool(z.is_first.any()),first_event_id=int(z[z.is_first].event_id.iloc[0]) if z.is_first.any() else np.nan))
        if len(z):winner_events.append(z.assign(original_winner_rank=rank,original_trade_net_usdt=t.net_pnl))
    winners=pd.DataFrame(winners);winner_event_table=pd.concat(winner_events,ignore_index=True) if winner_events else e.iloc[:0].assign(original_winner_rank=[],original_trade_net_usdt=[])
    outputs={'events.csv':e,'event_summary.csv':summary,'cycles.csv':cycle_table,'cycle_summary.csv':cycle_summary,
        'fixed_horizon_details.csv':details,'fixed_horizon_summary.csv':horizons,'original_top10_mapping.csv':winners,
        'original_top10_preentry_events.csv':winner_event_table,'censored_events.csv':e[e.censored],
        'zero_volume_event_endpoints.csv':e[e.entry_zero_volume|e.terminal_zero_volume],
        'signals.csv':pd.DataFrame(dict(date=d.date,beta=beta,b_bp_hour=beta*40000,g=g,g_bp_hour_per_15m=g*40000,early_event=flags,cycle_id=cycles))}
    for file,df in outputs.items():save(file,df)
    logging.info('Complete all events=%d | first=%d | censored=%d; these are NOT portfolio trade counts',len(e),int(e.is_first.sum()),int(e.censored.sum()))
    plots(d,beta,g,e,summary,horizons)
    report(e,summary,cycle_summary,horizons,winners,winner_event_table,data)
    build_html(d,beta,g,e,summary,data)
    active=['config/k5_entry_diagnostic.json','src/k5_entry_diagnostic.py','tests/test_k5_entry_diagnostic.py',
        'experiments/k5_entry_diagnostic/run_analysis.py','tools/k5_entry_template.html','tools/verify_k5_entry.py']
    frozen=['src/data_loader.py','src/slope_estimators.py','src/signal_generator.py','src/backtester.py','src/metrics.py',
        'experiments/s1_3_recent_weighted_slope/market_explorer.html']
    base=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()
    (OUT/'metadata.json').write_text(json.dumps(dict(protocol=protocol,data=data,baseline_gate=gate,base_commit=base,
        python=platform.python_version(),source_sha256={p:sha256(ROOT/p) for p in active+frozen},
        checks=['S0 gate','K5 independent regression','prefix and future perturbation','next-open time','gross/cost/net reconciliation',
            'all/first/later groups exhaustive','right censoring retained'],no_new_portfolio_backtest=True,
        interpretation='event counts not trade counts; overlapping outcomes not summed; future confirmation used only to describe outcomes, never select events'),indent=2,ensure_ascii=False),encoding='utf-8')
    logging.info('PASS entry-only diagnostic complete. STOP; no tau selection, no exits optimized, no holdout results.')
    logging.shutdown();manifest()

def plots(d,beta,g,e,summary,horizons):
    first=96;end=first+30*96;ix=np.arange(first,end);dates=pd.DatetimeIndex(d.date)
    fig,axs=plt.subplots(3,1,figsize=(15,10),sharex=True,layout='constrained')
    axs[0].plot(dates[ix],d.open.iloc[ix],lw=.8,color='#25364b',label='Raw open price')
    axs[1].plot(dates[ix],beta[ix-1]*40000,lw=.9,color='#25364b',label='b from last completed bar (bp/hour)')
    axs[2].plot(dates[ix],g[ix-1]*40000,lw=.8,color='#25364b',label='K5 change from last completed bar')
    for side,col,mark in [('long','#007e73','^'),('short','#bd5143','v')]:
        z=e[(e.direction==side)&(e.entry_index>=first)&(e.entry_index<end)];j=z.entry_index.to_numpy(int)
        axs[0].scatter(dates[j],d.open.iloc[j],s=23,c=col,marker=mark,label=f'{side} early event next-open (not portfolio fills)')
    hit=e[e.confirmation_executable].drop_duplicates('confirmation_execution_index')
    hit=hit[(hit.confirmation_execution_index>=first)&(hit.confirmation_execution_index<end)]
    j=hit.confirmation_execution_index.to_numpy(int)
    axs[0].scatter(dates[j],d.open.iloc[j],s=24,c='#ae7c0d',marker='x',label='OLS confirmation next-open')
    for ax in axs[1:]:ax.axhline(0,color='gray',lw=.6)
    for ax,label in zip(axs,['Price (USDT)','Slope (bp/hour)','Slope trend change (bp/hour per 15m step)']):ax.set_ylabel(label);ax.legend(fontsize=8);ax.grid(alpha=.2)
    axs[-1].set_xlabel('Actual UTC availability / open time; fixed first30 performance days, no selected winners')
    fig.savefig(OUT/'figures/first30days.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,side in zip(axs,['long','short']):
        for group,color in [('background','#66788a'),('all_events','#007e73'),('first_events','#2868b8'),('later_events','#b4781b')]:
            z=horizons[(horizons.direction==side)&(horizons.group==group)&(horizons.period=='overall')].sort_values('horizon_hours')
            ax.plot(z.horizon_hours,z.mean_net_usdt,marker='o',color=color,label=group)
        ax.axhline(0,color='gray',lw=.7);ax.set(title=side,xlabel='Fixed descriptive horizon (hours)',ylabel='Mean net per hypothetical $2000 entry (USDT)',xticks=[1,4,16]);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Overlapping diagnostic intervals, NOT an executable portfolio / optimal holding period')
    fig.savefig(OUT/'figures/fixed_horizons.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,side in zip(axs,['long','short']):
        for first,color,label in [(True,'#2868b8','First'),(False,'#b4781b','Later')]:
            z=e[(e.direction==side)&(e.is_first==first)].mae_pct.sort_values().to_numpy()
            if len(z):ax.step(z,np.arange(1,len(z)+1)/len(z),where='post',color=color,label=f'{label}, n={len(z)}')
        ax.set(title=side,xlabel='MAE to confirmation / censored endpoint (%)',ylabel='Empirical cumulative fraction',ylim=(0,1));ax.legend();ax.grid(alpha=.2)
    fig.suptitle('All excursions retained; no intrabar high/low order assumed')
    fig.savefig(OUT/'figures/mae_distribution.png',dpi=160);plt.close(fig)

def report(e,summary,cycles,horizons,winners,we,data):
    overall=summary[summary.period=='overall'];h=horizons[(horizons.period=='overall')&(horizons.group!='background')]
    observations=[]
    for side in ['long','short']:
        a=overall[(overall.direction==side)&(overall.group=='all')].iloc[0]
        f=overall[(overall.direction==side)&(overall.group=='first')].iloc[0]
        hs=h[(h.direction==side)&(h.group=='first_events')].sort_values('horizon_hours')
        observations.append(f'- {side}：全部{a.events}次、首次{f.events}次；首次事件到可执行过零的中位等待{f.wait_median_hours:.2f}小时，确认前MAE中位{f.mae_median_pct:.2f}%、95分位{f.mae_p95_pct:.2f}%、最大{f.mae_max_pct:.2f}%。首次事件1/4/16小时双边成本后平均损益依次为'+ '/'.join(f'{v:.2f}' for v in hs.mean_net_usdt)+' USDT（每次假设2000名义，区间重叠，不能累加成账户收益）。')
    winner_first=we[we.is_first]
    save('original_top10_first_events.csv',winner_first)
    extremes=e.sort_values('mae_pct',ascending=False).head(10)
    save('extreme_mae_events.csv',extremes)
    facts='''**本轮证据不足以支持直接把原过零入场替换为“每段首次K5上拐就入场”。** 它更早，但首次做多的提前阶段平均价格收益扣费后为负，首次做空平均损失更大；极端浮亏远非可以忽略。固定16小时首次做多虽为正，也没有超过所有负斜率状态的粗略背景均值，不能据此宣称入场择时优势。

具体解释：首次做多到过零的平均净损益2020为-3.53、2021为+1.52 USDT，年度不一致；首次做空分别-16.29/-12.26 USDT。首次做多有58.26%的提前区间净收益为正，但均值-1.05，说明较多小优势并不足以抵消尾部不利路径；首次做空有53.63%为正、均值-14.30，同样不能用超过50%的比例证明优势。相同终点只是把原过零开盘作为早入仓的假设退出，并非原策略的完整持仓回报。

后续做多事件提前阶段均值+5.97，高于首次-1.05，但其出现时点不同、可能一段多个事件、无法事先知道哪次是真反转；这只提示“一次试入可能放弃后续机会”的代价，不能直接改成挑第二次/最后一次或自动放开重复入场。

全部做多1072个机会、首次539个；做空1224个、首次537个。零区间的反复还存在：有事件的负区间49.17%、正区间52.33%包含多次机会，最多分别10/11次。首次机会之中，49.54%做多、52.89%做空在确认前g又反向；K5仍不能消除反复。

这份样本恰好2296个事件均在2021结束前出现可执行确认，所以删失事件表本轮为空；不代表确认必然发生或风险很低，删失分支已用合成测试覆盖。首次做多最长等待71.75小时、做空72.25小时；MAE极值56.92%/60.50%，分别对应约1138.47/1210.03 USDT假设名义损失幅度，未计潜在资金费率和交易所强平，不能将假设持仓路径当可融资执行保证。

原OLS前10赢家前都有首次K5事件，说明机会覆盖10/10，**不等于保留10/10利润或改善10/10入场**；逐笔早入阶段的负收益/MAE仍应审阅。完整原赢家身份和首次事件均保存，不将赢家PnL直接加到新模型。

建议先由专家审阅首次做多的尾部风险与后续机会差异，暂不实施对称早入策略。若继续，只冻结一种明确失败退出，检验它降低尾部的同时会损失多少原赢家；它仍需独立公平对照。不要把本轮16小时观察窗转成最优持仓，不自动改N/K或搜索tau。
'''
    text=f'''# K=5提前入场诊断：供专家评审

## 核心结论与边界

本轮只诊断提前入场，没有实施完整策略、失败退出、动量退出或tau优化；事件数不是实际交易数。固定15分钟、N=64、K=5、tau=0边界，未选参数。全部2020—2021是已反复研究的开发样本，N64本身也来自既往研究选择，不能称独立未见验证。不计算2022/2023收益，不读2024+。

{chr(10).join(observations)}

{facts}

必须将“到某个未来时刻终究穿零”与“提前买入确实有优势”区分；长时间接飞刀后最终穿零也算观察到确认，不能称入场成功。首次/后续的完整结果如下，专家应同时看MAE、等待、净价格优势及分年，而非只看确认率。

## 冻结事件公式与时序

beta=原S0最近64根log(close)的OLS；b=40000beta，bp/小时。g=最近5根beta回归斜率，等价[-2,-1,0,1,2]/10的因果中心求和；计算事件用未缩放beta，避免单位变换浮点改变符号。
做多：beta[t]<0且g[t-1]<=0且g[t]>0；做空镜像。双方g有限才发事件，g恰零本身不发事件、不前移标记。事件在t收盘才知道，t+1开盘是最早假设入场。首可执行时点2020-01-02 00:00 UTC，96根暖机；最后不可执行收盘信号不计事件。全周期转向计数包括暖机，避免将已有周期的第二次误说成第一次；边界截断标记cycle_left_truncated。

sign-cycle为连续beta符号区间，beta恰零沿用前符号，零本身不是确认；后续第一次严格相反符号才是该方向的确认。周期切分和未来确认只用于事后描述，事件本身绝不看未来；没有事后挑谷底、峰顶或最赚钱一次。第1次/后续事件按原始时间顺序划分，首次不自动成为策略。

## 全部 / 首次 / 后续事件

{md(overall,['direction','group','events','sign_cycles','confirmation_executable','censored','confirm_within_1h_ratio','confirm_within_4h_ratio','confirm_within_16h_ratio','wait_median_hours','wait_p95_hours','wait_max_hours','mae_median_pct','mae_p95_pct','mae_max_pct','net_to_confirmation_mean','net_to_confirmation_median','positive_net_to_confirmation_ratio','g_reversal_ratio'])}

来源event_summary.csv。确认率期限的分母包含右删失事件，按未观察到期限内确认计；这不是Kaplan-Meier生存估计，尾部可观察时长不同，不能作无偏概率推断。可执行确认才进入等待/确认端点收益统计，右删失保留在单独表和MAE中，禁止删除后声称全部成功。

{md(cycles,['direction','cycles','cycles_with_events','cycles_without_event','cycles_with_repeated_events','first_events','later_events','median_events_per_active_cycle','max_events_per_cycle','median_within_cycle_event_gap_hours'])}

来源cycles.csv/cycle_summary.csv。未出现K5转向的周期也保留。高事件数说明同一趋势内存在多次改善，不是本次模拟了对应数量实际交易。多事件常对应同一个过零点，观测强相关。

## 按入场年份检查

{md(summary[(summary.period!='overall')&(summary.group=='first')],['direction','period','events','censored','wait_median_hours','mae_median_pct','mae_p95_pct','mae_max_pct','net_to_confirmation_mean','positive_net_to_confirmation_ratio'])}

事件按入场年分组；2020入场到2021确认仍保留，不年度强平；样本结束仍未确认按末根close标记，不读2022。这里没有账户年度净收益，只有事件级描述。

## 固定未来1/4/16小时的净价格检验

{md(h,['direction','group','horizon_hours','observations','available','unavailable','mean_net_usdt','median_net_usdt','net_positive_ratio','mean_raw_signed_bps'])}

来源fixed_horizon_summary.csv；固定4/16/64根未来bar，均从下一开盘入场到entry+horizon开盘退出，含双边手续费/滑点。尾部不能退出的保留unavailable，不补2022数据。background为所有相同beta符号、g前后可计算bar，包含事件自身；仅粗略描述背景，不是匹配独立对照组，没有因果识别、置信区间或显著性声称。全部、首次、后续的时点分布不同，也不能直接归因一次试入规则优劣。持有期只是观察窗口，不自动转成最优退出或搜索维度。完整可用/不可用明细全部保存。

## 费用、MAE/MFE和删失定义

固定名义2000、entry_fill=open*(1+s*0.0002)、q=2000/entry_fill；单边手续费0.0005，退出也按不利方向滑点。gross、fee、slippage、net逐事件对账。价格优势raw_signed_bps=s*(confirmation_open/entry_open-1)*10000为无成本端点方向收益；net_to_confirmation在原过零可入场开盘假设退出早入仓，不是原策略完整交易收益。

MAE/MFE以实际假设滑点入场价为基准，并计入数量对应USDT；含入场K线至退出前一K线高低，退出根只纳入开盘价，不含其后高低。右删失包含最后根高低并按末close MTM，仅收取已发生的入场费及滑点、不造平仓费；其最终损益和MAE都是截至观察终点，不是假设永久持仓结果。MAE/MFE下限0；OHLCV无法恢复盘中高低先后。比较窗口到过零的MAE与长期持仓MAE不同，不能将它直接推导为止损阈值。

## 原S0前10大赢家的提前机会

{md(winners,['original_winner_rank','direction','original_net_usdt','eligible_events','first_event_exists','first_event_id'])}

{md(winner_first,['original_winner_rank','event_id','direction','wait_hours','mae_pct','net_usdt','raw_signed_bps'])}

来源original_top10_mapping.csv和original_top10_preentry_events.csv。固定原OLS N64已平仓净收益前10身份；仅检查它们入场前的相反符号周期中有哪些合格事件。不能选择其中价格最好的一次，不能把原赢家PnL认领为提前入场贡献；本轮没有定义新的完整持仓退出，所以不计算所谓保留率或修改后的top10利润。是否有首次事件只是机会覆盖，不是获利保证。专家可查看每个前置事件的等待、MAE和提前阶段损益。

## 不能省略的最不利路径

{md(extremes,['event_id','direction','entry_time','ordinal_in_cycle','wait_hours','mae_pct','mae_usdt','net_usdt'])}

来源extreme_mae_events.csv，只用于暴露尾部，不用于选择参数；完整事件始终保留。到确认的观察窗口长短不同，MAE是条件于该窗口的风险，不能与固定持有期无条件比较。

## 图形与交互复核

打开 [event_explorer.html](event_explorer.html)：价格、b、g按真实UTC可用时间对齐；所有提前事件与确认点可见，可筛选方向/年份/首次后续/删失，点击事件查看完整区间，不会只展示成功例子。K线价格为raw open；在同一open坐标绘制上一根已收盘的b/g，避免将收盘信号提前15分钟。详细提示有原OHLC和时间标记。密集图使用像素组最小最大包络而非平滑，放大恢复逐点；全部原始数据内嵌可离线打开。

![固定首30日完整事件](figures/first30days.png)
![固定时距事件检验](figures/fixed_horizons.png)
![全部MAE分布](figures/mae_distribution.png)

## 专家评审建议与停止点

首先判断首次做多事件是否在两年都体现足够价格优势，代价是否为较大MAE/等待，以及后续事件是否仅更靠近未来确认；后续更好不意味着可以实时选择“最后一次”。做空单独判断，不因对称数学而假设对称优势。若优势只在较长窗口出现，也可能来自ETH这两年的价格漂移和条件选择，不能立即构建策略。
事件反复、风险暴露、方向与年份不同，需要在选择失败退出前先审阅。若本轮出现正向均值，下一轮只应固定入场后检验一种失败退出与原基准，不能同时优化入场/退出/tau。若没有稳定优势，保留原过零入场，不强迫增加复杂性。
本轮不自动冻结新模型、不开参数搜索、不做完整新策略回测。审阅后再决定是否继续。

## 复现和原结果保护

数据hash `{data['sha256']}`，原loader仅读六列OHLCV，原2020—2023快照结构QC后去掉2022/2023再计算，无2024+。原S0控制账本、成本、多空、权益门禁通过，唯一组合模拟是原基准复现。本轮没有计算新策略Sharpe/PF/总净利润，以免将事件相加伪装为可执行回测。

仓库根目录：设置`.deps`到PYTHONPATH，运行`python -m pytest -q`、`python experiments/k5_entry_diagnostic/run_analysis.py`、`python tools/verify_k5_entry.py`。配置config/k5_entry_diagnostic.json在结果计算前固定；metadata.json记录数据/代码/配置hash，manifest覆盖本目录除自身外全部交付。原S0/S1.3/观察页不修改。旧K1探索属于先前误扩展解释，不能当作本轮K5结果。
'''
    (OUT/'report.md').write_text(text,encoding='utf-8')

def build_html(d,beta,g,e,summary,data):
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,(np.integer,np.bool_)):return v.item()
        if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
        return v
    payload=dict(first=int(pd.Timestamp(d.date.iloc[0]).value//10**9),step=900,
        raw={c:d[c].tolist() for c in ['open','high','low','close','volume']},
        b=(beta*40000).tolist(),g=(g*40000).tolist(),
        events=e.drop(columns=['source_bar','entry_time','confirmation_signal_time','confirmation_execution_time','endpoint_time']).to_dict('records'),
        summary=summary[summary.period=='overall'].to_dict('records'),data_sha256=data['sha256'])
    template=(ROOT/'tools/k5_entry_template.html').read_text(encoding='utf-8')
    assert template.count('__DATA__')==1
    (OUT/'event_explorer.html').write_text(template.replace('__DATA__',json.dumps(clean(payload),separators=(',',':'),ensure_ascii=False,allow_nan=False)),encoding='utf-8')

def manifest():
    files={str(p.relative_to(OUT)).replace('\\','/'):dict(sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts}
    (OUT/'artifact_manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8')

if __name__=='__main__':run()
