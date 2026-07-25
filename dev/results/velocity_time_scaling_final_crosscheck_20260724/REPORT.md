# 速度软约束、时间缩放与增广 QP 对比

## 结论

当前硬关节速度约束会在高需求回缩中改变 QP 的速度分配，实验段 13
相对“仅保留高数值 guard、不施加物理速度上限”的参考解产生了约
9.54 cm 的肘部路径偏差。

先求预览解、再统一时间缩放是最干净的保分支基线：

- 最终命令严格满足物理速度包络。
- 不引入 exact-null 或 near-null 重分配。
- 段 13 的肘部路径偏差从 9.54 cm 降到 6.34 cm。
- 代价是目标继续前进时会积累跟踪误差，位置 RMSE 从 4.64 cm
  增到 7.78 cm。

在 Mink 原 QP 内加入软超速 slack 后再做时间缩放，能够进一步减少
肘部路径偏差并降低 J1 命令加速度，但它依赖沿 `v_near` 的重新分配：

- `w_soft=30`：肘部路径偏差 5.57 cm，`v_near` 残差
  1.24 rad/s RMS。
- `w_soft=100`：肘部路径偏差 4.31 cm，`v_near` 残差
  2.31 rad/s RMS。

因此更大的软速度 cost 并不是“更接近纯时间缩放”，而是在鼓励 QP
用其他关节和弱奇异方向绕开超速。它可能碰巧更接近无限速参考分支，
但不能作为通用保证。

带显式泄漏预算的第二层增广 QP 是较可控的柔性方案：

- `|r_i| <= 0.15 rad/s`
- `|z^T r| <= 0.02 rad/s`
- `|v_near^T r| <= 0.05 rad/s`

它在段 13 上将位置 RMSE 从纯缩放的 7.78 cm 降到 7.18 cm，
但肘部路径偏差从 6.34 cm 略增到 6.77 cm。它提供的是可解释的
“少量换分支换进度”，不是同时改善所有指标。

直接把物理速度硬边界、progress 和 residual 全部塞回 Mink 原 QP 的
版本目前不可靠：两个圆周场景各出现 79 帧不可行。原因是原位置/奇异度
恢复约束、物理速度包络和有限 residual budget 的交集可能为空。

建议先实现事务式预览加统一时间缩放，再把第二层增广 QP 作为可选实验。
暂不把“原 Mink QP 增广版”接入运行时。

## Annotation 1：什么是增广 QP

Mink 当前每个子步求解：

```text
min  1/2 Δqᵀ H Δq + cᵀ Δq
s.t. G Δq <= h
     A Δq  = b
```

“增广”是把决策变量从 `Δq` 扩为：

```text
x = [Δq, alpha, xi]
```

- `alpha in [0, 1]`：本帧任务进度或时间缩放量。
- `xi >= 0`：允许速度包络被软超出的 slack。

对 task 的期望反馈 `b_task = -gain * error`，使用：

```text
min ||W (J Δq - alpha b_task)||²
    + w_alpha (1 - alpha)²
    + w_xi ||xi||²
```

方向性软速度约束为：

```text
 Δq_i <= dt * v_upper_i + xi_i
-Δq_i <= dt * v_lower_i + xi_i
```

Mink 原来的 `G/h/A/b` 只需补零列即可保留。若原 QP 的线性项是 `c`，
扩展 Hessian 中 `Δq-alpha` 的交叉块就是 `c`；`alpha-alpha` 项还要补上
所有 task 加权反馈的平方范数，否则优化器会错误地偏向 `alpha=0`。

## 第二层增广 QP

实验中更稳定的版本不改 Mink 原 QP，而是先取得 guard-limited
预览速度 `v_ref`，再求：

```text
v = alpha * v_ref + r
```

其中 `r` 是允许的小幅速度重分配：

```text
min ||r||²
    + w_task ||J r||²
    + w_alpha (1 - alpha)²
    + w_xi ||xi||²

s.t. -v_lower <= v <= v_upper
     |r_i| <= r_joint_max
     |zᵀ r| <= r_null_max
     |v_nearᵀ r| <= r_near_max
     0 <= alpha <= 1
```

泄漏预算全部为零时，它退化为纯时间缩放。实验中严格增广 QP 与纯缩放
的轨迹数值一致，验证了预览和最终状态提交的实现。

## 内部状态提交

预览过程必须是事务式的：

```text
q_start
  -> 临时求解 q_preview
  -> 得到 v_ref
  -> 求 alpha / r
  -> q_final = integrate(q_start, v_final, control_dt)
  -> configuration = q_final
```

不能先让 Mink 保留 `q_preview`，再只缩小发送给 driver 的命令。否则下一
帧 Jacobian、限位距离和 task error 都从未发送的状态计算，内部 lead 会
继续积累。

## 仿真设置

- MuJoCo 动力学 plant，位置控制命令，不使用速度前馈。
- 控制频率 250 Hz，Mink 每帧 10 个 `0.4 ms` 子步。
- 轨迹来自 intervention episode 202。
- `12`：普通慢速段。
- `13`：困难快速回缩段。
- `reverse_13`：反向伸展。
- `circle_0p4mps`、`circle_0p8mps`：伸展构型圆周运动。
- reference 保留关节位置、奇异度和 4 倍数值 guard，但不施加物理速度
  上限，仅用于比较分支，不能发送给实机。

所有可发送候选的最终峰值速度利用率均不超过 1.0。

## 关键结果

| case | strategy | elbow path error [cm] | position RMSE [cm] | orientation RMSE [rad] | `v_near` residual [rad/s] | failures |
|---|---|---:|---:|---:|---:|---:|
| 12 | current hard | 0.01 | 1.61 | 0.063 | 0.000 | 0 |
| 12 | pure scale | 0.02 | 1.62 | 0.062 | 0.000 | 0 |
| 13 | current hard | 9.54 | 4.64 | 0.258 | 0.000 | 0 |
| 13 | pure scale | 6.34 | 7.78 | 0.373 | 0.000 | 0 |
| 13 | bounded augmented | 6.77 | 7.18 | 0.367 | 0.042 | 0 |
| 13 | Mink soft `w=30` | 5.57 | 7.78 | 0.311 | 1.239 | 0 |
| 13 | Mink soft `w=100` | 4.31 | 8.62 | 0.251 | 2.307 | 0 |
| reverse 13 | current hard | 3.15 | 6.31 | 0.152 | 0.000 | 0 |
| reverse 13 | pure scale | 2.91 | 9.56 | 0.214 | 0.000 | 0 |
| reverse 13 | bounded augmented | 3.04 | 9.60 | 0.209 | 0.039 | 0 |
| reverse 13 | Mink soft `w=30` | 2.77 | 9.38 | 0.210 | 1.258 | 0 |
| circle 0.4 | pure scale | 0.95 | 4.47 | 0.113 | 0.000 | 0 |
| circle 0.4 | bounded augmented | 0.92 | 3.83 | 0.108 | 0.020 | 0 |
| circle 0.8 | pure scale | 1.92 | 7.41 | 0.153 | 0.000 | 0 |
| circle 0.8 | bounded augmented | 1.93 | 6.98 | 0.157 | 0.028 | 0 |

段 13 的 J1 加速度也体现出区别：

| strategy | command p99 [rad/s²] | actual p99 [rad/s²] |
|---|---:|---:|
| current hard | 25.10 | 26.14 |
| pure scale | 41.99 | 33.93 |
| bounded augmented | 34.79 | 26.52 |
| Mink soft `w=30` | 26.71 | 26.16 |
| Mink soft `w=100` | 25.95 | 26.09 |
| original-QP hard augmented | 65.14 | 47.48 |

## 推荐实现顺序

1. 将预览 QP 的物理速度限制替换为较高的数值 guard，但保留位置和
   奇异度安全限制。
2. 从 `q_start` 到 `q_preview` 计算 `v_ref`。
3. 用 command 与实测 `q/dq` 共同计算方向性 `v_lower/v_upper`。
4. 首版使用统一时间缩放，且只提交 `q_final`。
5. 记录 scale、主导关节、`z/v_near` 分量和实测 command lead。
6. 若纯缩放跟踪滞后不可接受，再启用第二层 bounded augmented QP。
7. 不使用未限制 `v_near` 的高权重软速度 task。
8. 在解决圆周不可行前，不采用原 Mink QP 的硬包络增广版本。

