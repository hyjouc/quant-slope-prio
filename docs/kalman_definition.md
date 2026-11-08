# 实际代码中的 Kalman Slope

依据旧项目冻结 V18 的 `build_features` 逐行审计；源路径、哈希及远程对应关系见 audit 元数据。以下为实际实现，不重设滤波器。

观测为 `y[t] = 100 ln(close[t])`，时间单位小时，`dt=0.25`。状态为 `[position, velocity, acceleration]ᵀ`；S0只使用velocity，模型内部保留加速度状态并不代表新增交易指标。

```text
F = [[1, dt, dt²/2],
     [0,  1, dt],
     [0,  0, 1]]
H = [[1, 0, 0]]
Q = 1e-7 * [[dt⁵/20, dt⁴/8, dt³/6],
             [dt⁴/8,  dt³/3, dt²/2],
             [dt³/6,  dt²/2, dt]]
R = 0.01
x_initial = [y[0], 0, 0]ᵀ
P_initial = diag([R, 1, 0.1])
```

首根以初始化状态直接进入更新；后续根执行：

```text
x_prior = F x_previous
P_prior = F P_previous Fᵀ + Q
innovation = y[t] - H x_prior
innovation_variance = H P_prior Hᵀ + R
K = P_prior Hᵀ / innovation_variance
x = x_prior + K innovation
P = (I-KH) P_prior (I-KH)ᵀ + K R Kᵀ
v[t] = x[1]
```

首根“不预测”是与最初文字要求“每根先预测”存在的差异。以实际V18代码为准，不能悄悄改变初始化。第一根v为0。

标准化采用前96根、排除当前值、样本标准差（ddof=1）：

```text
history_mean[t] = mean(v[t-96:t])
sd[t] = sqrt(sum((v[j]-history_mean[t])² for j=t-96..t-1) / 95)
S[t] = v[t] / sd[t]
```

标准差内部当然需要均值；“不减均值”指标准化分子保持v[t]，不写成v[t]-history_mean[t]。不足96根、sd=0或非有限值时指标无效。
没有双向平滑、居中窗口、未来价格或年度重初始化。`tools/audit_project.py` 对全样本V18输出精确对照，并在三个跨年边界重新运行前缀核验。

K线date是开盘标签，close及指标在date+15分钟可用；下一根open在同一时刻边界，因果顺序是上一根已收盘后才发单。
