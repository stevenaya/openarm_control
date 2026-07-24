# 回缩时关节速度额度与肘部分支选择

## 结论先行

本轮仿真的目标是：快速回缩时，让肘部横移和关节分支尽量接近
“不设关节速度限制”的参考解，同时不改变反向伸出、慢速回缩和近伸直圆周。

当前最有效的方案不是只给奇异弱方向额外速度，而是：

1. 保留正常基础速度上限；
2. 只有当目标手同时满足“到 J1 转轴的垂距缩短”和“到肩部的总距离缩短”
   时，才临时提高 J1/J4 的**总速度额度**；
3. 对额度本身做变化率限制；
4. 原有 joint-limit braking 和 singularity-approach limit 继续保留。

本次 left arm、episode 202、fast retract segment 13 的最佳折中参数为：

```text
J1 base / high       = 2.0 / 5.5 rad/s
J4 base / high       = 3.8 / 4.4 rad/s
retract deadband     = 0.02 m/s
retract full speed   = 0.15 m/s
cap slew             = 10 rad/s^2
gate                 = perpendicular shrink AND reach shrink
```

相对“不设关节速度限制”参考：

```text
elbow lateral RMSE   = 0.572 mm
elbow xyz RMSE       = 1.438 mm
joint q RMSE         = 0.00557 rad
EEF position RMSE    = 19.38 vs 19.44 mm
EEF orientation RMSE = 0.11430 vs 0.11481 rad
```

反向 segment 13 和 `0.4/0.8/1.2/1.6 m/s` 等臂长圆周中，双条件门控均为
0，结果与原始基础限速仅有数值精度差异。

这些数值是仿真调试结果，不是实机安全定值。该方案只有在 driver 和电机确实
允许相应速度时才能复现参考分支。如果 J1 的物理上限仍是 `2 rad/s`，那么
不能靠重新分配弱方向速度得到同一条快速轨迹，必须降低任务空间回缩速度。

## 1. 已比较的方案

| 方案 | 作用位置 | 结果 |
|---|---|---|
| 原始逐关节固定限速 | QP box limit | 安全直接，但 fast retract 中 J1/J4 饱和后 QP 换分支 |
| 完全移除 joint velocity limit | QP reference | 肘部横移最小，作为目标分支参考，不适合作为实机方案 |
| 固定提高 J1 或 J1/J4 上限 | QP box limit | 能逐步接近参考分支，但会永久改变所有动作 |
| joint-limit braking | QP inequality | 解决接近关节位置限位时的制动，不解决回缩分支选择 |
| singularity approach limit | QP inequality | 只限制接近奇异点的速度分量，适合安全保护，不负责离开奇异点时选哪条分支 |
| exact-null posture task | QP task | 只调节 \(z\)，无法约束主要分支变化所在的 \(v_{\mathrm{near}}\) |
| 两个最弱非零任务方向放宽 | QP modal limit | 第二个方向在当前构型并不弱，容易给正常任务方向不必要的额度 |
| \(v_{\mathrm{near}}\) 单方向放宽 | QP modal limit | 只改善一部分，仍无法复现无 joint-limit 分支 |
| \(\{z,v_{\mathrm{near}}\}\) 二维放宽 | QP modal limit | 横向误差下降，但三维肘轨迹、关节轨迹和姿态误差恶化，且加速度很高 |
| 弱任务方向 target governor | QP 前的目标速度 | 适合避免进入弱构型时的速度放大，但会主动改变目标，不能直接复现快速参考分支 |
| 仅垂距缩短门控 | 动态 QP total cap | 常用速度下有效，但高速等臂长圆周会误激活 |
| 仅总臂长缩短门控 | 动态 QP total cap | 圆周中关闭，但 reverse segment 13 有 11.3% 误激活 |
| 垂距与总臂长同时缩短 | 动态 QP total cap | fast retract 保持参考分支，所有交叉场景门控均正确关闭 |

## 2. “二维弱方向”到底指什么

对只包含 7 个 arm DoF 的归一化几何 Jacobian 做 SVD：

\[
J_{\mathrm{norm}}
=U\Sigma V^\mathsf{T},
\qquad
J_{\mathrm{norm}}\in\mathbb{R}^{6\times 7}
\]

其中：

```text
U      : 6 x 6，任务空间方向
Sigma  : 6 个非负奇异值
V      : 7 x 7，关节空间方向
```

令奇异值按从大到小排列：

\[
\sigma_1\ge\cdots\ge\sigma_6\ge0
\]

右奇异向量有三类：

\[
v_{\mathrm{near}}=V[:,6]
\]

它对应最小的非零奇异值：

\[
J_{\mathrm{norm}}v_{\mathrm{near}}
=\sigma_6u_6
\]

当 \(\sigma_6\) 很小时，沿 \(v_{\mathrm{near}}\) 的较大关节速度只能产生很小的
末端速度，因此会出现肩肘速度放大。

\[
z=V[:,7]
\]

是 7 DoF arm 在 full 6D task 下的精确零空间：

\[
J_{\mathrm{norm}}z=0
\]

### 2.1 早期的“两个弱任务方向”

早期 pair 实验使用：

\[
P_{\mathrm{task2}}
=v_5v_5^\mathsf{T}+v_6v_6^\mathsf{T}
\]

它不包含 exact nullspace。测试圆周中：

```text
sigma_5 / sigma_max = 0.338
sigma_6 / sigma_max = 0.057
```

所以只有 \(v_6\) 真正较弱，\(v_5\) 仍是健康任务方向。给这两个方向同时放宽
没有明显优于只放宽 \(v_6\)，高速时还可能鼓励 QP 使用低可控方向。

### 2.2 与肘部分支更相关的二维近零空间

针对本次分支问题，更合理的二维子空间是：

\[
P_{\mathrm{near2d}}
=zz^\mathsf{T}
+v_{\mathrm{near}}v_{\mathrm{near}}^\mathsf{T}
\]

它包含：

```text
z       : 当前精确零空间方向
v_near  : 即将随着 rank 下降扩展成第二个零空间的方向
```

对任意关节速度：

\[
\dot q
=P_{\mathrm{near2d}}\dot q
+(I-P_{\mathrm{near2d}})\dot q
\]

J1 分量相应写成：

\[
\dot q_1
=e_1^\mathsf{T}P_{\mathrm{near2d}}\dot q
+e_1^\mathsf{T}(I-P_{\mathrm{near2d}})\dot q
\]

“只给二维弱方向额外额度”的 QP 约束为：

\[
|\dot q_1|\le v_{1,\mathrm{high}}
\]

\[
\left|
e_1^\mathsf{T}(I-P_{\mathrm{near2d}})\dot q
\right|
\le v_{1,\mathrm{base}}
\]

J4 同理。第一条限制总速度，第二条要求超过基础额度的部分只能来自选定
子空间。

### 2.3 为什么二维放宽仍没有复现参考分支

fast retract 的低奇异度区间中：

```text
mean |z^T dq|       ~= 0.43 rad/s
mean |v_near^T dq|  ~= 2.01 rad/s
```

约 38% 的速度能量位于 \(v_{\mathrm{near}}\)，exact null \(z\) 只有约 3%。
所以 exact-null posture task 确实管不到主要分支变化。

但是反过来，只允许 \(v_{\mathrm{near}}\) 或
\(\{z,v_{\mathrm{near}}\}\) 超过基础额度也不够。无 joint-limit 参考解中，
J1 的非 \(v_{\mathrm{near}}\) 分量统计为：

```text
mean abs = 1.34 rad/s
p90      = 3.90 rad/s
p99      = 4.60 rad/s
max      = 4.72 rad/s
```

约 27.9% 的帧超过原 J1 基础上限 `2 rad/s`。也就是说，参考回缩不仅需要
近零空间运动，还需要正常任务方向上的大 J1 运动。

仿真结果也对应这一点：

| Strategy | Elbow y RMSE | Elbow xyz RMSE | q RMSE | Position RMSE | Orientation RMSE | J1 actual accel p99 |
|---|---:|---:|---:|---:|---:|---:|
| fixed base limits | 54.33 mm | 104.87 mm | 0.362 rad | 46.42 mm | 0.258 rad | 26.1 |
| \(v_{\mathrm{near}}\) only | 31.47 mm | 47.32 mm | 0.143 rad | 29.39 mm | 0.177 rad | 76.8 |
| \(\{z,v_{\mathrm{near}}\}\) | 7.74 mm | 41.86 mm | 0.095 rad | 39.80 mm | 0.203 rad | 119.6 |
| no joint velocity limit | 0 | 0 | 0 | 19.44 mm | 0.115 rad | 34.7 |

二维近零空间方案虽然让 elbow y 更接近参考，但 elbow xyz 和整条 q 轨迹仍然
差很多；开放 exact null 后还出现了更强的分支自由度和加速度。因此它不适合
作为本问题的最终限速策略。

## 3. 双几何条件动态总额度

### 3.1 回缩上下文

使用当前 J1 世界坐标转轴 \(a\)、肩部锚点 \(s\) 和经过 VR 映射/滤波后的
目标手位置 \(p_d\)：

\[
r=p_d-s
\]

到 J1 转轴的垂直分量和垂距为：

\[
r_\perp=(I-aa^\mathsf{T})r,
\qquad
R_\perp=\|r_\perp\|
\]

肩到手的总伸展距离为：

\[
L=\|r\|
\]

分别估计收缩速度：

\[
v_\perp
=\max\left(
-\frac{R_{\perp,k}-R_{\perp,k-1}}{\Delta t},
0
\right)
\]

\[
v_L
=\max\left(
-\frac{L_k-L_{k-1}}{\Delta t},
0
\right)
\]

双条件门控使用：

\[
v_{\mathrm{retract}}=\min(v_\perp,v_L)
\]

因此只有两者都大于零时才会激活：

```text
只绕肩做等臂长圆周：v_L = 0，关闭
反向伸出：至少一个收缩速度为 0，关闭
真正向肩回缩：两者均大于 0，开启
```

### 3.2 平滑激活与 cap slew

\[
u=
\operatorname{clip}
\left(
\frac{v_{\mathrm{retract}}-0.02}
{0.15-0.02},
0,1
\right)
\]

\[
\alpha=3u^2-2u^3
\]

这个 smoothstep 在区间两端的一阶导数为 0，避免进入或离开门控时 cap
斜率突变。

期望上限：

\[
\bar v_1
=2.0+\alpha(5.5-2.0)
\]

\[
\bar v_4
=3.8+\alpha(4.4-3.8)
\]

然后限制上限本身每个控制周期的变化：

\[
|v_{j,k}-v_{j,k-1}|
\le 10\Delta t
\]

最终直接作为 QP total velocity inequality：

\[
|\dot q_1|\le v_{1,k},
\qquad
|\dot q_4|\le v_{4,k}
\]

这里不再增加 modal/nonweak 约束。QP 可以按 FrameTask、nullspace task、
kinetic-energy task 和其余 limits 自己分配所有关节分量。

## 4. 参数扫描

fast retract segment 13：

| J1 high / slew | Elbow y RMSE | Elbow xyz RMSE | q RMSE | Position RMSE | J1 cmd peak | J1 actual accel p99 |
|---|---:|---:|---:|---:|---:|---:|
| 5.0 / 5 | 9.39 mm | 12.80 mm | 0.0351 rad | 20.84 mm | 4.81 | 48.0 |
| 5.0 / 7.5 | 1.38 mm | 2.31 mm | 0.00760 rad | 19.50 mm | 5.00 | 49.0 |
| 5.0 / 10 | 1.36 mm | 2.29 mm | 0.00755 rad | 19.49 mm | 5.00 | 49.1 |
| 5.5 / 10 | 0.57 mm | 1.44 mm | 0.00557 rad | 19.38 mm | 5.38 | 44.0 |
| 6.0 / 10 | 0.46 mm | 1.33 mm | 0.00531 rad | 19.34 mm | 5.54 | 42.5 |
| no joint limit | 0 | 0 | 0 | 19.44 mm | 5.97 | 34.7 |

`slew=5` 提升额度太慢，QP 已经选到另一条分支后再增加额度无法恢复。
`J1 high=6` 的拟合略好，但比 `5.5` 暴露更多速度；因此 `5.5/10` 是当前
更合适的实机前仿真起点。

J4 不能保持原 `3.8 rad/s` 不变，因为参考解的 J4 command peak 为
`4.29 rad/s`。`4.4 rad/s` 足以覆盖本次参考，同时没有无限放开 J4。

## 5. 交叉场景

| Case | Combined gate active | Candidate effect |
|---|---:|---|
| slow retract segment 12 | 100% | J1/J4 实际需求低于 base，轨迹基本不变 |
| fast retract segment 13 | 73.5% | elbow xyz RMSE to reference 1.44 mm |
| reverse segment 13 | 0% | 与原 base-limit trace 基本一致 |
| circle 0.4 m/s | 0% | 与原 base-limit trace 基本一致 |
| circle 0.8 m/s | 0% | 与原 base-limit trace 基本一致 |
| circle 1.2 m/s | 0% | 与原 base-limit trace 基本一致 |
| circle 1.6 m/s | 0% | 与原 base-limit trace 基本一致 |

对比单条件门控：

```text
perpendicular-only:
  circle 1.2 m/s active 44.3%
  circle 1.6 m/s active 45.8%

reach-only:
  reverse segment 13 active 11.3%

combined:
  上述场景均为 0%
```

## 6. 实现建议

### 6.1 QP 与 driver 分层

动态值是 IK/QP 的“解分配额度”，不能替代 driver 的物理安全上限：

```text
IK dynamic cap      : 为特定回缩上下文避免过早换分支
driver hard limit   : 电机和机构真正允许的最大速度
joint braking       : 接近关节位置限位时保证可制动
singularity limit   : 接近奇异点时限制危险方向
```

如果 driver 最终仍把 J1 截到 `2 rad/s`，IK 内部允许 `5.5 rad/s` 只会重新
制造 command/actual 偏差，不能得到本报告中的轨迹。

### 6.2 如果硬件不能达到所需速度

无 joint-limit 参考在这条轨迹中需要：

```text
J1 command peak = 5.97 rad/s
J4 command peak = 4.29 rad/s
```

若 J1 物理上限只有 `2 rad/s`，要保持相近的关节路径只能对回缩动作做时间
缩放。线性近似下至少需要：

\[
s
\le
\min_j
\frac{v_{j,\mathrm{physical}}}
{|\dot q_{j,\mathrm{preview}}|}
\]

本例仅看 J1：

\[
s\lesssim \frac{2.0}{5.97}\approx0.335
\]

也就是把目标回缩速度降到约三分之一。可用一次高额度 preview solve 估计
\(\dot q_{\mathrm{preview}}\)，再按同一个标量缩放整条 task-space target
increment，最后使用真实基础速度上限重解。这样会变慢，但比只放宽某个弱
模态更容易保持空间分支。

### 6.3 采样与状态

几何速度必须按真正的新 VR 样本更新：

```text
dt = 两个 fresh VR pose 的 monotonic 时间差
无新样本时 hold 上一次 retract-speed estimate
不要用 250 Hz tick 重复对同一 UDP 包求差分
```

建议对 \(v_\perp,v_L\) 做轻量低通或直接平滑 \(\alpha\)，并保留 cap slew。
clutch、disconnect、INVALID 和重新接管时，应把历史距离、速度估计和动态
cap 重置到当前 base 状态。

### 6.4 推荐实施顺序

1. 先在 runtime 中实现 combined geometry gate 和 total J1/J4 cap；
2. 所有参数通过 args/config 暴露，默认保持 base limits；
3. 仿真复测左右臂和更多 intervention 记录；
4. 实机从低于 `5.5/4.4` 的 high cap 开始，记录 command/actual q、dq；
5. 只有确认 driver 和电机能跟上，才继续提高 high cap；
6. 若物理速度不足，停止提高 cap，改用 combined-gated target time scaling。

## 数据

- `../retract_subspace_budget_tune_seg13_20260724/summary.csv`
- `../retract_subspace_budget_refine_seg13_20260724/summary.csv`
- `../retract_subspace_budget_crosscheck_20260724/summary.csv`
- `../retract_subspace_budget_candidate_crosscheck_20260724/summary.csv`
- `../retract_subspace_budget_candidate_circle_stress_20260724/summary.csv`
- `../retract_gate_metric_crosscheck_20260724/summary.csv`
- `../modal_shoulder_limit_ablation_20260724/summary.csv`
- `../j1_weak_pair_horizontal_retract_20260724/REPORT.md`

本轮只修改了 `dev/sim_retract_subspace_budget_comparison.py` 和本报告，没有
修改运行时 `src/`。
