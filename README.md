# quant-slope-prio · S0 从零斜率研究

任务：`S0_FROM_SCRATCH_SLOPE_ESTIMATOR_2020_2023`。

**少参数 > 多参数，稳定性 > 样本内收益，简单有效 > 复杂优化。**

本分支从零比较端点斜率A和滚动OLS斜率B，各自唯一参数N=[8,16,32,64,96]，共10组。使用ln(close)、严格符号信号、下一根开盘反向执行。无旧模型、标准化、技术指标堆叠或旧执行器。

本轮仅2020—2021训练，不计算2022/2023绩效、不自动宣布最佳模型、不扩大网格。负责人审阅并冻结方法/N后才能进入后续历史验证。2020—2023均有历史研究复用，不能称为完全未见独立样本外。

- [训练报告与全部10组结果](experiments/s0_from_scratch_slope/s0_report.md)
- [新模型与交易定义](docs/model_definition.md)
- [研究规范及统计口径](docs/research_protocol.md)
- [阶段执行计划](docs/experiment_plan.md)
- [固定实验配置](config/s0_from_scratch.json)
- [机器可读诊断](experiments/s0_from_scratch_slope/diagnostic_metadata.json)
- [历史归档：SUPERSEDED - NOT USED IN NEW MODEL DESIGN](docs/archive/legacy_kalman_audit/README.md)

## 结构

```text
src/data_loader.py          # 指定快照哈希及六列OHLCV，输出仅训练数据
src/slope_estimators.py     # A端点、B滚动OLS
src/signal_generator.py     # 严格符号、零保持、无效不下单
src/backtester.py           # 独立执行器与成交事件/权益账本
src/metrics.py              # MTM/费用/方向/年度/邻域/极端盈利依赖
tests/                     # 独立公式、因果、执行及会计测试
experiments/s0_from_scratch_slope/
docs/archive/legacy_kalman_audit/  # 原始审计记录保留但不参与新设计
```

## 重现

Python 3.14，`python -m pip install -r requirements.txt`。

```powershell
python -m pytest -q
python experiments/s0_from_scratch_slope/run_experiment.py
```

本机补充依赖在Git忽略目录`.deps`，可先设置 `$env:PYTHONPATH = "$PWD\.deps"`。大型原始行情不提交；其他主机需在配置中指定同SHA256的快照。哈希不一致则立即停止。
固定输入仅六列date/open/high/low/close/volume；SHA256为`23972d31dd5ff90b46da05703e1b75dbfbbbbfde6efcd26b6b9ad24ee3b65503`。
两年输入70176根，首96根公共预热，从2020-01-02 00:00 UTC开始计绩效，年界连续。97根原始零成交量（训练39根）原样保留，所有其上的假设成交单独审计。

账户10000、每笔保证金预算1000/2倍/固定名义2000，费用另扣，不复利。手续费单边0.05%、滑点单边0.02%；不模拟资金费和交易所强平。详细会计及资金不足限制见报告。
完整15分钟曲线、逐笔交易、每项开/平执行事件均输出CSV。清单覆盖报告/图表/结果/活跃源码/配置的SHA256，不把清单自哈希或Git提交自引用写入自身。

## Git边界

基准提交`2021e1699f0221f923783806071c909dfc3e352e`，研究分支`research/s0-from-scratch-slope-2020-2023`。main保持基准状态，旧项目只读。
本地目录仍为最初创建的`eth-kalman-slope-lab`，远程项目为`hyjouc/quant-slope-prio`。旧版历史状态仅见归档，不代表本轮结论。
