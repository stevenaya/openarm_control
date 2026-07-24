# Episode 202 intervention replay

## Scope and limitation

Episode 202 stores the IK joint command and measured robot state, but not the
raw VR target pose. The replay therefore:

1. Computes the recorded left-arm command's full 6D end-effector pose with FK.
2. Resamples that pose stream at 250 Hz.
3. Feeds the same position and orientation stream into Mink.
4. Drives a separate MuJoCo position-controlled plant with the resulting joint
   command.

This tests redundancy resolution for the recorded geometric path. It cannot
reconstruct any VR target lead that existed beyond the recorded IK command.

## Why a moderate Cartesian speed can saturate the shoulder

For the characteristic-length-normalized geometric Jacobian:

```text
J_norm = U diag(sigma) V^T
```

`z = V[:, -1]` is the exact structural nullspace direction. The smallest
nonzero singular direction is `v_near = V[:, -2]`. For a normalized desired
twist `xi_norm`, the unconstrained weak-direction joint velocity is:

```text
dq_near = (u_min^T xi_norm / sigma_min) v_near
```

Therefore total Cartesian speed is not a universal saturation threshold. The
required joint speed also depends on configuration, motion direction, and
orientation.

At the weakest configurations in replay segments 12 and 13:

```text
segment 12 singular values:
[2.2250, 1.8720, 1.4214, 0.7919, 0.7413, 0.0837]

segment 13 singular values:
[2.2230, 1.8706, 1.4129, 0.7910, 0.7542, 0.0912]
```

The weak joint direction was approximately:

```text
v_near = [-0.40, 0.03, 0.02, -0.82, 0.01, 0.41, -0.03]
```

It is dominated by J1, J4, and J6. Its task-space direction is almost entirely
translational, so this is the elbow-extension reach singularity rather than a
wrist-orientation singularity. J4 was still about `0.28-0.31 rad`, but the arm
was already geometrically close enough to straight for `rho` to fall to
`0.038-0.041`.

The measured weak-direction demand was:

| Segment | Mean weak twist | Mean unconstrained `v_near` | Peak |
|---|---:|---:|---:|
| 12, slow | 0.570 1/s | 1.75 rad/s | 2.98 rad/s |
| 13, fast | 1.037 1/s | 3.68 rad/s | 9.27 rad/s |

This is why the shoulder can reach its `2 rad/s` bound while total end-effector
speed is well below `1.6 m/s`. The earlier `1.6 m/s` result was one point in a
different synthetic trajectory sweep, not a universal onset speed.

## Posture-cost replay

Segments 12 and 13 start with nearly the same elbow lateral coordinate. Segment
12 is the slow/back branch; segment 13 is the fast/left-back branch.

| Segment | Cost | Any joint saturated | Elbow lateral delta | Position RMSE | Orientation RMSE |
|---|---:|---:|---:|---:|---:|
| 12 | 0.00 | 0.8% | 0.0685 m | 0.0026 m | 0.0012 rad |
| 12 | 0.10 | 0.8% | 0.0701 m | 0.0024 m | 0.0084 rad |
| 13 | 0.00 | 49.5% | 0.1927 m | 0.0277 m | 0.0930 rad |
| 13 | 0.01 | 49.5% | 0.1926 m | 0.0277 m | 0.0930 rad |
| 13 | 0.03 | 49.5% | 0.1924 m | 0.0275 m | 0.0927 rad |
| 13 | 0.10 | 47.5% | 0.1895 m | 0.0263 m | 0.0895 rad |

Costs up to 0.1 do not materially alter this branch decision.

Higher constant full-posture costs eventually hold the elbow, but by sacrificing
the frame task:

| Cost | Elbow lateral delta | Position RMSE | Orientation RMSE |
|---:|---:|---:|---:|
| 0.3 | 0.157 m | 0.0287 m | 0.0924 rad |
| 1.0 | 0.051 m | 0.0579 m | 0.374 rad |
| 3.0 | 0.005 m | 0.1178 m | 0.718 rad |

This reproduces why increasing a soft-limit or posture cost can make the bump
heavier: it does not remove the incompatible demand; it redistributes it among
the frame task and active velocity bounds.

## Same-path temporal scaling

Segment 13 was replayed along exactly the same 6D path at three durations:

| Duration scale | Peak target speed | Saturated frames | `mean |z^T dq|` | `mean |v_near^T dq|` | Elbow delta |
|---:|---:|---:|---:|---:|---:|
| 1x | 1.390 m/s | 49.5% | 1.159 rad/s | 1.787 rad/s | 0.193 m |
| 2x | 0.778 m/s | 14.4% | 0.187 rad/s | 0.949 rad/s | 0.104 m |
| 4x | 0.389 m/s | 0.25% | 0.033 rad/s | 0.455 rad/s | 0.086 m |

Thus speed is a real trigger even for an identical path. The mechanism is the
Jacobian amplification followed by joint-velocity saturation.

## Rate versus direct exact-nullspace error

Increasing the existing physical exact-nullspace return rate helped only
partially:

| Exact-z `k / vmax` | Elbow delta | Orientation RMSE |
|---|---:|---:|
| 0.8 / 0.8 | 0.193 m | 0.093 rad |
| 3.0 / 3.0 | 0.176 m | 0.111 rad |
| 10.0 / 3.8 | 0.155 m | 0.133 rad |

The emerging `v_near` mode carries most of the large bending motion, while the
exact `z` coordinate selects much of the elbow plane. Increasing only `k_ns`
cannot remove the weak-direction velocity demand.

A direct exact-nullspace soft error was also tested:

```text
cost^2 || z^T Delta q + beta e_z ||^2
e_z = z^T (q minus q_home)
```

| Per-substep `beta` | Saturated frames | Elbow delta | Position RMSE | Orientation RMSE |
|---:|---:|---:|---:|---:|
| rate baseline | 49.5% | 0.193 m | 0.0277 m | 0.093 rad |
| 0.003 | 61.9% | 0.123 m | 0.0313 m | 0.161 rad |
| 0.010 | 63.6% | 0.087 m | 0.0434 m | 0.306 rad |
| 0.030 | 63.6% | 0.087 m | 0.0525 m | 0.356 rad |

Direct error can force the preferred branch, but a hard constraint would
transfer even more conflict into frame error and joint limits.

## Velocity-limit causal sweep

The same segment-13 target path was replayed while multiplying all seven
velocity limits. The elbow delta below is measured only over the 0.816 s
retraction core, rather than over the recovery context after it.

| Limit scale | J1 cap | Any-limit frames | J1 core-limit frames | Core elbow delta | Core exact-z drift | Position RMSE | Orientation RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0x | 2 rad/s | 49.5% | 86.3% | 0.181 m | +1.127 rad | 0.0277 m | 0.0930 rad |
| 1.5x | 3 rad/s | 21.5% | 41.7% | 0.164 m | +0.694 rad | 0.0143 m | 0.0637 rad |
| 2.0x | 4 rad/s | 15.1% | 28.9% | 0.109 m | +0.085 rad | 0.0064 m | 0.0225 rad |
| 4.0x | 8 rad/s | 0.25% | 0.0% | 0.080 m | -0.138 rad | 0.0032 m | 0.0031 rad |

The unconstrained branch moves J1 by about `2.16 rad` during the 0.816 s
core, or `2.65 rad/s` even as a simple interval average. Its peak command is
about `5.9 rad/s`. With a `2 rad/s` hard cap, that J1 path is infeasible.
The constrained solution instead changes the joint coordination:

```text
                Delta J1  Delta J2  Delta J3  Delta J4
2 rad/s J1 cap    +1.527    -1.013    +0.877    +1.859 rad
relaxed limits    +2.158    -0.375    -0.108    +1.950 rad
```

The extra J2/J3 motion is the observed sideways elbow branch. This is not an
intentional path planner selecting an elbow pose. It is the new constrained
least-squares optimum after the velocity box changes the QP active set.

A per-joint ablation isolates the responsible bound:

| Limits relaxed by 4x | Core elbow delta | Core exact-z drift | Position RMSE | Orientation RMSE |
|---|---:|---:|---:|---:|
| none | 0.181 m | +1.127 rad | 0.0277 m | 0.0930 rad |
| J1 only | 0.080 m | -0.137 rad | 0.0055 m | 0.0032 rad |
| J2 only | 0.201 m | +1.248 rad | 0.0216 m | 0.0936 rad |
| J4 only | 0.181 m | +1.125 rad | 0.0272 m | 0.0963 rad |
| J1 and J4 | 0.080 m | -0.136 rad | 0.0032 m | 0.0034 rad |

For this trajectory, J1 is the branch-changing constraint. Relaxing J4
improves residual frame tracking only after J1 has enough headroom.

The dynamic MuJoCo plant follows the command-level branch closely: at the
baseline, command and actual core elbow deltas are `0.1812 m` and `0.1785 m`.
Thus the branch change is already present in the IK command; plant lag is not
required to create it, although real dynamics can amplify the visible bump.

### Why a weak task-space direction asks for high joint speed

Here "weak" describes the kinematic transmission ratio, not a small target
velocity or a low task cost. For

```text
J_norm = U diag(sigma) V^T
```

the target twist component along the weakest task-space direction satisfies:

```text
u_min^T xi = sigma_min v_min^T dq
v_min^T dq = (u_min^T xi) / sigma_min
```

The VR target is currently not projected or slowed along `u_min`. It can have
a finite or large `u_min^T xi`; dividing by the small `sigma_min` then asks for
large joint speed. The velocity limit clips the joint result only afterward.

### Why direct exact-z regulation conflicts under velocity limits

Without active limits and to first order:

```text
dq = J_pinv xi + z nu
J z = 0
```

so changing `nu` does not change the frame velocity. With per-joint limits,
however, the sum must also satisfy:

```text
|J_pinv xi + z nu| <= dq_max
```

The frame solution and nullspace correction use the same physical joints. If
their sum exceeds a joint cap, no feasible solution can satisfy both desired
motions. The velocity limit is hard while both Mink tasks are soft, so the QP
trades frame error against exact-z error.

The tested direct task is also stronger than its small `beta` suggests. With
ten IK substeps:

```text
dt_sub = 0.004 / 10 = 0.0004 s
equivalent z speed = beta e_z / dt_sub
```

For `beta=0.003` and `e_z=1 rad`, this requests `7.5 rad/s` along `z`.
At the original limits it increased saturated frames from `49.5%` to `61.9%`
and orientation RMSE from `0.093` to `0.161 rad`. At 4x limits, both the
baseline and direct task had only `0.25%` saturated frames, and their
orientation RMSE values were `0.0031` and `0.0036 rad`. This confirms that
most of the observed conflict comes from shared velocity headroom, not from
`z` failing to be a geometric nullspace direction.

During the fast core, `beta=0.003` at the original limits still produced
`0.180 m` elbow motion versus `0.181 m` for the baseline. Its smaller
full-window endpoint delta came mainly from correction after the fast core,
not from preventing the branch change while it happened.

## Recommended direction

Do not use total handle speed as the only modulation variable. Use the
configuration-aware predicted weak-direction demand:

```text
r_weak = |u_min^T xi_norm| / sigma_min
```

The first protection should slow only the target component that requires
excessive `v_near`, before the joint box constraints become active.

Relaxing the J1 limit is a useful causal diagnostic, but `6 rad/s` should not
become a production value unless the motor, driver, acceleration, torque, and
stopping-distance limits all support it. A safer controller should preserve
the real J1 limit and reduce the weak target component before the QP reaches
that active set. Any direct exact-z correction should also be gated by
remaining joint-velocity headroom.

For branch selection, retain the exact-nullspace task but add a gently gated
direct-error component rather than only increasing its physical return rate:

```text
a_rho   = near-singularity activation
a_load  = smooth activation from predicted velocity utilization
beta_eff = beta_max a_rho a_load

cost^2 || z^T Delta q + beta_eff e_z ||^2
```

Start below the tested `beta=0.003`, and use a soft objective rather than a hard
constraint. The reference should be configurable: `q_home` is not necessarily
the desired elbow plane at every workspace pose, so a clutch-time branch anchor
or a deliberately chosen elbow-plane reference may be more appropriate.

The existing singularity-approach limit only constrains decreasing `rho`.
During retraction the arm is leaving the straight singularity and `rho` is
increasing, so that limit does not govern which elbow branch is selected.
