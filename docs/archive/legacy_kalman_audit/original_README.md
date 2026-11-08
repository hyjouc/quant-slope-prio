# quant-slope-prio

极简 Kalman Slope 研究。**少参数 > 多参数，稳定性 > 样本内收益，简单有效 > 复杂优化。**

本轮只完成第一阶段：只读审计旧项目、确认数学定义与行情、建立规范及配置。没有完成参数优化，也没有选定 θ。审阅通过前禁止运行回测。

- [仓库与旧实现审计](docs/repository_audit.md)
- [行情审计](docs/data_audit.md)
- [Kalman Slope 数学定义](docs/kalman_definition.md)
- [S0 最小策略定义](docs/strategy_definition.md)
- [研究规范](docs/research_protocol.md)
- [实验执行计划](docs/experiment_plan.md)
- [机器可读审计](audit/audit_metadata.json)
- [实验配置](config/s0_config.json)

## 重现第一阶段核验

Python 3.14；`python -m pip install -r requirements.txt`，然后：

```powershell
python tools/audit_project.py
python -m pytest -q
```

当前主机为避免更改全局环境，将补充依赖安装在被Git忽略的 `.deps`，运行时可设置 `$env:PYTHONPATH = "$PWD\.deps"`。
其他主机需在配置中提供同一哈希的 pre2024 快照与只读V18源码路径；原始行情不随仓库提交。
数据加载器在反序列化前校验登记哈希，拒绝未知混合年份文件。脚本仅计算指标对照、时间完整性与未来扰动核验，不执行交易回测。

## 项目状态

远程为公开独立仓库 `hyjouc/quant-slope-prio`，全新Git历史。当前本地目录保留最初创建时的名称 `eth-kalman-slope-lab`，项目名、配置与远程均以 `quant-slope-prio` 为准。
原项目 `quant-strategy-slope` 及其本地参考文件保持只读。

最新指令已将任务从“S0完整实验”缩减为“第一阶段审计”。切换前旧流程已完成指标对照并进入首个θ训练执行，随后在时间字符串解析处失败；没有形成排名、收益结论或参数选择。未完成草稿及日志已隔离到Git忽略目录 `.superseded`，不作为第一阶段交付。
`experiments/s0_single_threshold/run_experiment.py` 当前只输出阻止执行提示；回测执行器与指标绩效模块暂不纳入项目。

历史复用不等于独立样本外：2020—2023均存在研究复用痕迹，2023尤其经历多轮使用；2024及之后本轮不读取，是否存在真正未使用的可信检验集尚未确认。
