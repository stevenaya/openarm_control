# J1 二维弱方向与几何回缩门控仿真

## 范围

- 仿真侧：left arm，episode 202 的 recorded retract，加近伸直球面圆周轨迹。
- 控制侧：沿用当前 Mink QP、joint braking、singularity approach limit、
  nullspace task、kinetic-energy task 和 MuJoCo position-control plant。
- 本轮只修改 `dev/sim_modal_shoulder_limit_comparison.py`，没有修改运行时
  `src/`。
- 修正了实验脚本的 J1 multiplier：自定义绝对上限现在相对真实默认
  `2 rad/s` 换算。此前只在 `base != 2` 的低额度实验中会造成名义值与实际
  box 不一致。

## 1. 二维弱方向额度

对归一化几何 Jacobian 做 SVD：

\[
J_{\mathrm{norm}} = U\Sigma V^\mathsf{T}
\]

去掉 7 DoF arm 的 exact nullspace 后，取两个最小非零奇异值对应的右奇异
向量 \(v_5,v_6\)：

\[
P_w = [v_5\ v_6][v_5\ v_6]^\mathsf{T}
\]

J1 速度可写成：

\[
\dot q_1
= e_1^\mathsf{T}P_w\dot q
+ e_1^\mathsf{T}(I-P_w)\dot q
\]

实验中的 pair limit 是：

\[
|\dot q_1| \le 1.5\ \mathrm{rad/s}
\]

\[
\left|e_1^\mathsf{T}(I-P_w)\dot q\right|
\le 0.75\ \mathrm{rad/s}
\]

因此总 J1 可以超过 `0.75`，但超出的部分必须来自二维弱任务子空间。

这里的“二维弱”只是按奇异值排序取最后两维，不表示两维都已经接近奇异。
`0.4 m/s` 圆周核心段的实测比例为：

```text
sigma_5 / sigma_max = 0.338
sigma_6 / sigma_max = 0.057
```

因此当前姿势只有 \(v_6\) 是真正的弱模态；\(v_5\) 仍是健康、且圆周动作
确实需要的任务方向。

### 近伸直圆周结果

| EEF speed | fixed 0.75 J1 peak | pair 1.5/0.75 J1 peak | pair nonweak binding | pair / fixed-low position RMSE | pair / fixed-low orientation RMSE |
|---:|---:|---:|---:|---:|---:|
| 0.2 m/s | 0.557 | 0.570 | 0.0% | 25.80 / 25.81 mm | 0.089 / 0.090 rad |
| 0.4 m/s | 0.750 | 1.096 | 55.1% | 28.66 / 28.19 mm | 0.114 / 0.111 rad |
| 0.6 m/s | 0.750 | 0.892 | 90.4% | 37.15 / 34.07 mm | 0.187 / 0.147 rad |
| 0.8 m/s | 0.750 | 0.951 | 98.3% | 53.74 / 51.00 mm | 0.243 / 0.196 rad |

结论：

- 二维约束确实起效。`0.4 m/s` 时总 J1 已从 `0.75` 放大到
  `1.10 rad/s`，而非弱分量长期贴住 `0.75`。
- 当前正常额度 `total=3, nonweak=2` 在测试圆周中没有绑定，所以不会增加
  可见的 J1 限速。
- 人为压低额度后，QP 会更多借用两个低可控任务方向。高速时姿态误差明显
  增大，因此不能把“弱方向允许更快”直接当作性能优化。
- pair limit 更适合作为针对已确认瓶颈的保护性余量，不能替代正常的 total
  velocity budget。

### 为什么等臂长圆周仍有弱模态速度

圆周目标到固定肩点的距离变化小于 `3e-16 m`，目标姿态在圆周核心段也完全
不变。理想零误差构型下，你的判断是对的：目标速度在径向奇异方向上的分量
应为零。

但 Mink 每帧基于当前 command configuration 的 6D Jacobian 和当前 pose
error 求解，不是基于目标球面的理想 Jacobian 做速度前馈。本次核心段中：

```text
当前最弱任务方向与目标球面径向的对齐度约 0.99
目标速度在当前最弱模态上的平均归一化投影约 0.108 1/s
等效纯平移约 0.032 m/s
当前肩手臂长误差平均约 5.6 mm，最大约 14.9 mm
```

小的方向偏差和跟踪误差经过较小的 `sigma_6` 反解后，会产生明显的
joint-space coefficient。补充的 single/pair A/B 也确认了这一点：

| EEF speed | single J1 peak | pair J1 peak | single/pair position RMSE |
|---:|---:|---:|---:|
| 0.4 m/s | 1.099 | 1.096 | 29.00 / 28.66 mm |
| 0.6 m/s | 0.926 | 0.892 | 39.04 / 37.15 mm |
| 0.8 m/s | 0.875 | 0.951 | 55.60 / 53.74 mm |

`0.4 m/s` 时 single 和 pair 的放行量几乎相同，说明额外 J1 主要来自真正
最弱的 \(v_6\)，不是误纳入的 \(v_5\)。不过低 nonweak cap 会鼓励 QP
更多使用这个低可控方向，所以高速时仍会恶化姿态跟踪。

完整数据：

- `../modal_pair_lowcap_circle_20260724/summary.csv`
- `../modal_pair_lowcap_circle_20260724/modal_circle_*`
- `../modal_single_pair_lowcap_circle_20260724/summary.csv`

## 2. J1 几何回缩门控

### 几何量

使用当前 J1 世界坐标转轴 \(a\)、肩部转轴锚点 \(s\) 和原始目标手位置
\(p_d\)：

\[
r_\perp = (I-aa^\mathsf{T})(p_d-s)
\]

\[
R = \|r_\perp\|,
\qquad
\dot R_k = \frac{R_k-R_{k-1}}{\Delta t}
\]

只在 \(R\) 减小时激活：

\[
v_r = \max(-\dot R,0)
\]

\[
u =
\operatorname{clip}
\left(
\frac{v_r-0.02}{0.15-0.02},
0,1
\right),
\qquad
\alpha = 3u^2-2u^3
\]

\[
v_{1,\mathrm{desired}}
=2.0+\alpha(3.2-2.0)
\]

再限制 cap 自身的变化率：

\[
|v_{1,k}-v_{1,k-1}|
\le 5.0\Delta t
\]

最后在 QP 中加入对称约束：

\[
|\dot q_1| \le v_{1,k}
\]

这里有意对 J1 正负方向同时放开。绕 J1 自身旋转不会改变 \(R\)，所以
\(\partial R/\partial q_1\) 理论上为零，不能用它可靠地判断应该放开 J1
正向还是负向。\(R\) 只负责识别“目标正在回缩”这个运动上下文。

旧 `reach` 实验曾把世界坐标肩手向量与 Mink 的末端局部坐标 Jacobian 直接
点乘。两者坐标系不一致，不能解释成真实距离梯度。本轮没有复用该判据。

### 快回缩参数扫描

| Strategy | Position RMSE | Orientation RMSE | Elbow lateral range | J1 cmd accel p99 | J1 actual accel p99 |
|---|---:|---:|---:|---:|---:|
| fixed 2.0 | 46.42 mm | 0.258 rad | 178.5 mm | 25.1 | 26.1 |
| fixed 3.0 | 32.04 mm | 0.188 rad | 141.6 mm | 56.8 | 29.6 |
| gated high 3.2, no slew | 31.11 mm | 0.179 rad | 141.8 mm | 101.9 | 42.3 |
| gated high 3.2, slew 20 | 30.53 mm | 0.177 rad | 139.4 mm | 65.9 | 35.4 |
| gated high 3.2, slew 10 | 30.25 mm | 0.177 rad | 135.6 mm | 39.6 | 26.8 |
| gated high 3.2, slew 5 | 30.05 mm | 0.176 rad | 131.5 mm | 45.4 | 26.8 |

当前单条 recorded retract 上，推荐实验起点为：

```text
base_j1_speed        = 2.0 rad/s
retract_high_speed   = 3.2 rad/s
retract_deadband     = 0.02 m/s
retract_full_speed   = 0.15 m/s
cap_slew_rate        = 5.0 rad/s^2
```

`slew=5` 在这条轨迹上综合结果最好，但它还不是硬件安全定值。真实机械臂应
先从更保守的 `high=2.6..3.0` 和 `slew=5` 开始，并观察实际 J1 速度，因为
MuJoCo position-control plant 中 `high=3.2` 时实际 J1 峰值约
`3.72 rad/s`。

### 交叉场景

| Case | Gate active | Mean cap | Effect versus fixed 2.0 |
|---|---:|---:|---|
| slow retract | 100% | 3.03 | J1 peak only 1.96，未绑定，position 差 0.003 mm |
| fast retract | 96.1% | 2.95 | position RMSE 46.42 -> 30.05 mm |
| reverse extension | 0% | 2.00 | command/actual trace 逐样本完全一致 |
| straight circle 0.4 m/s | 35.6% | 2.11 | J1 peak 1.11，未绑定，trace 完全一致 |
| straight circle 0.8 m/s | 40.7% | 2.27 | J1 peak 1.59，未绑定，EEF 差小于 0.5 mm |

门控显示 active 不代表一定改变动作。只有当 QP 原本需要超过基础
`2 rad/s` 时，额外额度才真正影响解。

完整数据：

- `../horizontal_retract_slew5_crosscheck_20260724/summary.csv`
- `../horizontal_retract_slew5_crosscheck_20260724/modal_*`
- `../horizontal_retract_tune_seg13_20260724/summary.csv`
- `../horizontal_retract_tune_high_seg13_20260724/summary.csv`
- `../horizontal_retract_slew10_seg13_20260724/summary.csv`
- `../horizontal_retract_slew20_seg13_20260724/summary.csv`
- `../horizontal_retract_slew50_seg13_20260724/summary.csv`

## 建议

1. 不把二维弱方向扩增作为默认方案。它能工作，但高速近奇异动作中会诱导
   QP 使用低可控任务方向，姿态跟踪变差。
2. 若继续验证 J1 选择性放宽，优先使用“垂距减小门控 + 对称 J1 cap +
   cap slew”。它对反向伸出严格关闭，在普通圆周中即使判定为回缩半周期，
   只要 J1 未到基础上限就不会改变动作。
3. 上硬件前补充 measured-state 门控、实际速度 overshoot 统计和双臂数据。
   本轮结论来自单侧 MuJoCo position-control plant，不能直接替代硬件限速。
