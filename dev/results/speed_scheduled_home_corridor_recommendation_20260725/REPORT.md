# Home-referenced speed-scheduled elbow corridor

## 结论

这版结构可以实现预期语义：

- 零空间与肘部参考都固定为 MuJoCo `home` keyframe；
- 低速时增广项完全退出，数值上恢复现有 Mink 行为；
- 高速时 exact-nullspace home cost、home-swivel 回正代价和方向性软走廊逐渐增强；
- 不依赖 Ruckig feed-forward，累计 FrameTask error 通过每子步 error leak 限幅后进入增广 QP；
- 走廊始终包含“当前肘部位置到 home”的整段，继续远离 home 才使用 slack；
- 所有原有位置、速度、braking 和 singularity constraints 仍是硬约束。

候选 `home_final_h1p5_0p3mm_e0p03_c5` 在左臂困难回缩上优于上一版
`damping_e0p06_r0p75`，但在右臂中速回缩上会产生更多 tracking lag，因此目前只适合
guarded A/B，不应直接成为双臂统一默认参数。

## Home 坐标

`ElbowSwivelCoordinate` 的参考构型改为：

```text
q_ref = solver._posture_task.target_q
```

而不是 sync 后的当前构型。数值验证：

```text
psi_home_left  = -7.1e-18 rad
psi_home_right =  7.1e-18 rad
```

因此 `psi=0` 表示 home 肘部方向。

exact-nullspace task 仍使用固定 `home_qpos`。高速时只把它的 base cost 从 `1.0x`
平滑提高到 `1.5x`，不修改目标。

## Error leak

不使用 nominal feed-forward twist。每个 `0.4 ms` 子步从完整 FrameTask error 中只取：

```text
e_p_leak = sat_norm(e_p, 0.0003 m)
e_R_leak = sat_norm(e_R, 0.0024 rad)
Delta x_cmd = -[e_p_leak, e_R_leak]
```

这等价于在单个子步中最多请求约 `0.75 m/s` 平移和 `6 rad/s` 转动，但不会把几厘米
累计误差一次性交给 `s`。完整误差继续保留在 FrameTask target 中，后续周期会继续追赶。

低速 `alpha_v=0` 时 explicit equality 的 Cartesian slack 不计成本，原始
FrameTask objective 决定 `Delta q`，所以 error leak 不改变低速结果。

## 速度激活

激活仍由原始目标 pose 的有限差分速度计算，仅用于调权，不作为 QP feed-forward：

```text
r = max(norm(v_target) / 0.6, norm(omega_target) / 4.0)
u = clip((r - 0.75) / 0.25, 0, 1)
alpha_target = 3*u^2 - 2*u^3
```

所以平移在 `0.45 -> 0.6 m/s`、旋转在 `3 -> 4 rad/s` 之间激活。`alpha_v` 的
上升/下降 rate limit 分别为 `4 /s` 和 `2 /s`。

## 方向性 home corridor

每个外层 `4 ms` 周期记录当前 home-referenced swivel `psi_0`。余量为：

```text
m(alpha) = m_fast + (m_slow - m_fast) * (1 - alpha)^2
m_slow = pi rad
m_fast = 0 rad
```

走廊上下界为：

```text
lower = min(0, psi_0) - m(alpha)
upper = max(0, psi_0) + m(alpha)
```

因此 `alpha=1` 时，免费区域恰好是当前位置到 home 的区间。QP 约束：

```text
psi + J_psi Delta q <= upper + epsilon_psi
psi + J_psi Delta q >= lower - epsilon_psi
epsilon_psi >= 0
```

同一个 `epsilon_psi` 允许任一侧违反。由于每个周期都会以当前 `psi_0` 重建区间，
持续远离 home 并非不可行，但每一帧都要为 outward motion 支付 slack 代价。

slack 以 `1 deg` 归一化，代价从低速的零平滑增加到：

```text
5 * epsilon_bar + 0.5 * 50 * epsilon_bar^2
```

## Home-swivel 回正

几何肘部的期望回正速度为：

```text
v_psi_des = clip(-0.5 * psi, -0.1, 0.1) rad/s
Delta psi_des = v_psi_des * dt_sub
```

其归一化速度尺度是 `0.25 rad/s`，权重从低速的 `0` 增加到高速的 `0.03`。
这部分对实际肘部分支的影响明显大于 exact 1D nullspace cost。

把 exact-nullspace cost 从 `1x` 扫到 `3x` 时，困难回缩指标几乎不变。这再次说明
主要分支变化包含 near-null direction，而不是仅发生在结构性的 1D `z` 上。

## 左臂结果

比较对象是上一轮增广 QP 候选 `damping_e0p06_r0p75`，不是当前生产控制器。

| Case | Method | elbow path cm | position cm | orientation rad | J1 accel p99 | J4 accel p99 |
|---|---|---:|---:|---:|---:|---:|
| slow retract 12 | previous | 0.015 | 1.61 | 0.063 | 28.6 | 32.5 |
| slow retract 12 | home corridor | 0.015 | 1.61 | 0.063 | 28.6 | 32.5 |
| fast retract 13 | previous | 6.03 | 7.93 | 0.117 | 25.1 | 43.4 |
| fast retract 13 | home corridor | 5.14 | 8.34 | 0.106 | 28.1 | 61.4 |
| reverse 13 | previous | 2.15 | 9.41 | 0.132 | 60.3 | 85.9 |
| reverse 13 | home corridor | 1.84 | 9.03 | 0.104 | 60.3 | 49.4 |
| circle 0.4 m/s | previous | 1.44 | 2.83 | 0.108 | 15.0 | 19.3 |
| circle 0.4 m/s | home corridor | 1.44 | 2.83 | 0.108 | 15.0 | 19.3 |
| circle 0.8 m/s | previous | 2.47 | 4.92 | 0.159 | 41.3 | 23.0 |
| circle 0.8 m/s | home corridor | 2.34 | 7.56 | 0.174 | 44.1 | 27.8 |
| circle 1.2 m/s | previous | 2.41 | 7.71 | 0.154 | 79.9 | 59.8 |
| circle 1.2 m/s | home corridor | 2.42 | 9.64 | 0.175 | 64.5 | 43.1 |

所有左臂候选最终配置均为零 QP failure，硬速度包络未被突破。

## 右臂交叉验证

右臂记录只有两个回缩段。慢段 `0` 与旧行为一致。较快段 `1`：

| Case | Method | elbow path cm | position cm | orientation rad | J1 accel p99 | J4 accel p99 |
|---|---|---:|---:|---:|---:|---:|
| retract 1 | previous | 0.98 | 1.52 | 0.077 | 200.1 | 164.6 |
| retract 1 | home corridor | 1.25 | 3.36 | 0.124 | 130.9 | 135.1 |
| reverse 1 | previous | 2.09 | 4.50 | 0.227 | 144.6 | 94.9 |
| reverse 1 | home corridor | 2.03 | 5.98 | 0.266 | 239.2 | 176.8 |
| circle 0.8 m/s | previous | 1.88 | 5.68 | 0.082 | 42.9 | 44.9 |
| circle 0.8 m/s | home corridor | 1.24 | 7.71 | 0.070 | 45.6 | 34.7 |

上一版候选在右臂 `0.8 m/s` circle 中有一次 QP failure，home-corridor 候选没有。
但右臂 retract/reverse 的 tracking 与 acceleration 结果说明该参数还不适合直接全局启用。

## 消融结论

- `0.2 mm` leak 可以进一步约束肘部分支，但位置 lag 明显增大。
- `0.4/0.5 mm` leak 并未恢复持续圆周进度，QP 会相应降低 `s`，同时左臂困难回缩变差。
- exact-nullspace cost `1x -> 3x` 几乎没有可测影响。
- cheap corridor slack 基本被直接使用；`5/50` 才开始产生稳定但仍柔性的约束。
- 强 corridor 可以把 slack 压到零，但会增加 J4 command acceleration。
- `0.06/0.1` 的主动 home-swivel 回正能显著缩小肘部范围，但会产生不可接受的肩肘加速度。
- `0.03`、`0.1 rad/s` 上限和较慢 alpha rate 是当前较温和的组合。

## 建议

1. 保留此实现为离线/guarded A/B，不修改生产默认。
2. 生产接入前把 error leak、home-swivel weight、corridor slack cost 都做成参数。
3. 记录左右臂的 `alpha_v`、`s`、`epsilon_psi`、`J_psi dq`、dominant velocity limit、
   command acceleration 和 measured lead。
4. 如果要降低高速 tracking lag，下一步应加入受限的 target feed-forward 或 target governor，
   而不是继续增大 error leak；本轮已经验证增大 leak 会被更小的 `s` 抵消。
5. 右臂 reverse 中的 acceleration 退化需要单独处理后，才考虑双臂默认启用。

## Artifacts

- 实验脚本：`dev/sim_speed_scheduled_elbow_qp.py`
- 初始扫描：`dev/results/speed_scheduled_home_corridor_scan_20260725/`
- 权重调优：`dev/results/speed_scheduled_home_corridor_tune_20260725/`
- 左臂交叉验证：`dev/results/speed_scheduled_home_corridor_crosscheck_20260725/`
- leak 最终扫描：`dev/results/speed_scheduled_home_corridor_final_scan_20260725/`
- 右臂交叉验证：`dev/results/speed_scheduled_home_corridor_right_crosscheck_20260725/`
- 右臂最终扫描：`dev/results/speed_scheduled_home_corridor_right_final_scan_20260725/`
