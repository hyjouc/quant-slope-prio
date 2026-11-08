# 仓库与旧实现审计报告

审计日期：2026-11-08（Asia/Shanghai）。当前目标为用户最新指定的公开仓库 [hyjouc/quant-slope-prio](https://github.com/hyjouc/quant-slope-prio)。

## 仓库身份与边界

GitHub API首次核查：`private=false`、默认分支`main`、size=0，commits接口返回409（空仓库）。没有覆盖已有提交，不Fork，不导入旧Git历史。本地独立初始化main，与目标同名分支对齐。
最初目标`eth-kalman-slope-lab`曾按最初要求改为Private，收到用户改为Public后已恢复；未向该仓库推送本项目内容。
原项目仅通过本地文件读取与GitHub GET API审计，没有修改或推送旧仓库。

旧仓库：[hyjouc/quant-strategy-slope](https://github.com/hyjouc/quant-strategy-slope)。审计时默认分支 `research/v19-step1-baseline-audit-2020-2023`，对应提交 `03a71cf8c587bf7bf67b17aef7513fab3e6719e4`。
新项目读取的本地冻结源码与该提交的远程文件逐字节一致：

| 文件 | SHA256 |
|---|---|
| inputs/ETH_KalmanMomentum_V18C3_TwoConfirm.py | 39b5ee58b32a75c9a775e57fd5dba9bdb67d426393c9bcd90642276c11502fd9 |
| inputs/reference_replay.py | 4b805f4068a168d66722a66821c90f9f497ea012942f5aaf6b7c620a2e56c782 |

核验证据见 `audit/repository_snapshot.json`。本地参考位于旧项目`stage4_trade_attribution/inputs`；运行前后对源码及行情计算哈希一致，记录在 `audit/audit_metadata.json`。

## 代码审计发现

1. `build_features`第21—36行定义dt=.25、R=.01、积分白噪声Q、三状态F/H、Joseph更新、前96根shift标准化。准确数学定义见 kalman_definition.md。
2. 第29行`if i`表示首根不预测。它优先于最初文字中“每根都先预测”的笼统描述。
3. 原函数另外计算加速度标准化、Momentum、trend8及4小时EMA，后续还有Ridge质量评分；原策略还含多空独立参数、两根确认、止损与早退。这些都是旧策略的一部分，S0一律不继承。审计脚本只调用纯函数作对照，不加载策略类、模型、seed或runtime文件。
4. 旧策略文件第85—87行预留开仓费；旧回放第83行`units=sizes/(ep*(1+FEE))`，故名义预算并非全部成交仓位。按2000 USDT预算计算，实际入场成交名义约1999.00050、入场费约0.99950 USDT。
5. 新S0最初要求是2000 USDT严格成交名义、费用另扣，入场费为1 USDT。这是有意定义的差异，不能默默声称成本及仓位口径完全照旧；留待负责人确认。
6. 旧回放单边FEE=.0005、SLIP=.0002，开仓`open*(1+s*SLIP)`、平仓`open*(1-s*SLIP)`，费用双边扣。S0提案沿用费率与不利滑点方向，反向必须计算平旧开新两笔执行。
7. 旧回放包含月界清除pending/counts、平仓当根不再开仓、两根确认，以及单独年度调用时flat start；这些不是Kalman数学要求，不应直接移植为S0行为。S0提案跨年保持指标及仓位、允许下一开盘直接反向。
8. 原代码存在资金可用性断言及复利/不复利并行模式，S0只提议固定名义不复利。资金不足与强平模型尚未冻结，不得绕过此限制宣称实盘结果。

## 本轮核验与状态

全部140,256根的Kalman slope及标准化slope与原V18精确一致；三个跨年时点重新计算的前缀亦精确一致。5项自动测试通过，1项执行验收明确跳过，因为第一阶段不交付执行器。
本轮未完成参数搜索、未形成θ排名或交易价值结论。最新范围调整前旧完整实验流程已进入首个θ训练执行但在统计日期解析时失败，没有保留其绩效产物；该草稿已隔离，不纳入提交。
审计不证明策略盈利。后续必须先审阅配置及执行计划，才开始回测。
