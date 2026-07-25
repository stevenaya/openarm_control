# Speed-scheduled elbow augmented QP experiment

## Conclusion

The weighted augmented QP is numerically feasible, but the full proposed
combination should not replace the current runtime controller yet.

The intervention record does not contain the original VR or Ruckig stream.
The experiment therefore uses the finite difference of the recorded-command FK
target as the nominal feed-forward twist. This is the closest replayable proxy,
but hardware A/B must log and use the real post-Ruckig nominal twist.

The useful part is:

- task progress `s`;
- high-cost Cartesian slack used only for recovery;
- a very light, speed-scheduled elbow-swivel velocity damping term.

The fixed clutch-time elbow corridor is not recommended in the first runtime
version. It did not improve the difficult retract once light damping was
active, and it harmed sustained circular motion.

The best experimental setting was `damping_e0p06_r0p75`. Across all tested
cases it had zero QP failures and respected the hard joint velocity envelope.
It is suitable for a guarded real-arm A/B test, not an unconditional production
default.

## Corrected formulation

The first literal implementation exposed three important details.

### 1. `Delta x_cmd` must be a bounded twist command

Using the complete FrameTask pose error as the displacement requested in every
`0.4 ms` substep made `s` collapse to about `0.003`:

```text
Delta x_cmd != -error
```

The experiment instead uses:

```text
v_cmd = sat(v_ff + K_task * error)
Delta x_cmd = -v_cmd * dt_sub
```

The minus sign follows Mink's FrameTask error-Jacobian convention. Translation
and rotation are norm-limited independently.

### 2. Low speed must reduce exactly to the existing Mink task

Using the explicit task equality at every speed changed the redundancy branch
even in an unconstrained slow retract. The final experiment blends the task
formulation:

```text
(1 - alpha_v) * original Mink FrameTask objective
    + alpha_v * augmented task behavior
```

The equality remains present, but Cartesian slack is unpenalized at
`alpha_v=0`, so it only absorbs the duplicate task expression. At
`alpha_v=0`, `s=1` and the solved `Delta q` matches the existing Mink QP.
The maximum command-joint difference was below `4e-6 rad` in slow retract 12
and below `2e-6 rad` in the `0.4 m/s` circle.

### 3. Scale the decision variables

Mink solves substep displacement, while `s` and normalized slacks are order
one. The experiment solves for:

```text
Delta q = D y
D_i = velocity_limit_i * dt_sub
```

All `H/c/G/h/A/b` blocks are transformed consistently. This keeps DAQP's
numerical scale reasonable.

## Elbow coordinate

`psi` is the geometric elbow swivel around the shoulder-to-end-effector axis.
The clutch-time elbow normal defines zero:

```text
a = normalize(p_wrist - p_shoulder)
n = normalize((p_elbow - p_shoulder)
              - a * a.T * (p_elbow - p_shoulder))
psi = signed_angle(project(n_ref, plane_normal=a), n, axis=a)
```

`J_psi` is computed by central finite difference once per 4 ms control tick and
reused in the ten Mink substeps. The term is disabled if the elbow-plane radius
falls below `0.02 m`. The minimum observed radius in the selected difficult
retract was about `0.046 m`.

## Speed activation

Recommended activation:

```text
r = max(norm(v_ff) / 0.6, norm(omega_ff) / 4.0)
u = clip((r - 0.75) / (1.0 - 0.75), 0, 1)
alpha_target = 3*u^2 - 2*u^3
```

This means:

- linear activation starts at `0.45 m/s` and is full at `0.6 m/s`;
- angular activation starts at `3 rad/s` and is full at `4 rad/s`.

`alpha_v` itself is rate-limited:

```text
rise: 8 /s
fall: 4 /s
```

Without this rate limit, replayed target speed made `alpha_v` jump by more than
`200 /s`, and the elbow damping weight produced a command acceleration spike.

## Recommended experimental parameters

| Parameter | Value |
|---|---:|
| `linear_fast_m_s` | `0.6` |
| `angular_fast_rad_s` | `4.0` |
| `speed_ratio_slow / fast` | `0.75 / 1.0` |
| `activation rise / fall` | `8 /s`, `4 /s` |
| task feedback gain | `10 /s` |
| task command linear limit | `1.0 m/s` |
| task command angular limit | `8.0 rad/s` |
| task-scale weight | `10` |
| Cartesian slack scales | `0.002 m`, `0.01 rad` |
| Cartesian slack high-speed weight | `1e6` |
| elbow velocity normalization | `0.25 rad/s` |
| elbow velocity weight slow / fast | `0 / 0.06` |
| elbow return gain / max speed | `0.5 /s`, `0.3 rad/s` |
| elbow corridor | disabled |
| swivel minimum radius | `0.02 m` |

The raw elbow weight is meaningful only with the normalization and variable
scaling used by the experiment.

## Main comparison

The "current hard" baseline below uses the fixed physical velocity limits, not
the newer dynamic retract velocity governor. Elbow path error is relative to
the high-guard reference path.

| Case | Method | elbow path [cm] | position RMSE [cm] | orientation RMSE [rad] | J1 accel p99 [rad/s2] |
|---|---|---:|---:|---:|---:|
| slow retract 12 | current hard | 0.03 | 1.60 | 0.062 | 28.6 |
| slow retract 12 | recommended | 0.03 | 1.60 | 0.062 | 28.6 |
| fast retract 13 | current hard | 9.53 | 4.64 | 0.258 | 25.1 |
| fast retract 13 | recommended | 5.97 | 7.96 | 0.116 | 25.1 |
| reverse 13 | current hard | 3.16 | 6.32 | 0.152 | 38.1 |
| reverse 13 | recommended | 2.14 | 9.41 | 0.132 | 60.4 |
| circle 0.4 m/s | current hard | 1.44 | 2.83 | 0.108 | 15.0 |
| circle 0.4 m/s | recommended | 1.44 | 2.83 | 0.108 | 15.0 |
| circle 0.8 m/s | current hard | 2.50 | 5.27 | 0.196 | 53.2 |
| circle 0.8 m/s | recommended | 2.47 | 4.92 | 0.159 | 41.3 |
| circle 1.2 m/s | current hard | 2.42 | 10.34 | 0.271 | 80.9 |
| circle 1.2 m/s | recommended | 2.41 | 7.71 | 0.154 | 79.9 |

For fast retract 13:

- actual swivel range fell from about `59 deg` in the fixed-limit baseline to
  `28.6 deg`;
- high-activation task scale averaged about `0.58`;
- high-activation Cartesian slack p99 was below `0.5 um` and `0.1 urad`;
- J1, J2, J3, J4 and J6 touched their hard velocity envelopes;
- there were no QP failures.

The cost is deliberate time lag: fast-retract position RMSE increased by about
`3.3 cm`. Reverse motion also showed a higher J1 acceleration p99 than the
current hard baseline, so real-arm testing needs an acceleration/command-step
abort threshold.

## Why the fixed elbow corridor was rejected

With cheap corridor slack, the QP simply used `epsilon_psi`, so the corridor did
not constrain the branch. Making it strong enough to matter reduced task
progress.

A `45 deg -> 25/30 deg` speed-scheduled clutch corridor:

- made no additional improvement in fast retract 13 beyond light damping;
- on the `0.8 m/s` circle, increased elbow path error from `2.48 cm` to about
  `4.85 cm`;
- increased orientation RMSE from `0.160 rad` to about `0.221 rad`.

The shoulder-wrist axis changes continuously in a circle, so a fixed
clutch-time swivel reference is not a generally valid posture corridor.

## Runtime recommendation

1. Do not enable the elbow corridor penalty yet.
2. Keep the existing hard position, velocity, braking, collision and
   singularity constraints.
3. If this augmented controller is tested on hardware, start with the
   `0.06` elbow damping setting and the `0.45 -> 0.6 m/s` activation band.
4. Log `alpha_v`, `s`, Cartesian slack, `psi`, `J_psi dq`, dominant velocity
   limit, command acceleration and measured command lead.
5. Abort or fall back to current IK if QP fails, Cartesian slack exceeds the
   configured emergency scale, or command acceleration exceeds the observed
   simulation envelope.
6. Compare against the current dynamic retract velocity governor before any
   production replacement. This experiment isolated fixed physical limits and
   does not establish superiority over that newer production path.

## Artifacts

- Experiment driver:
  `dev/sim_speed_scheduled_elbow_qp.py`
- Fast/reverse weight edge:
  `dev/results/speed_scheduled_elbow_qp_weight_edge_20260724/`
- Recommended circle cross-check:
  `dev/results/speed_scheduled_elbow_qp_recommended_circles_20260724/`
- Fixed-corridor A/B:
  `dev/results/speed_scheduled_elbow_qp_guard_crosscheck_20260724/`
- Earlier ablations and parameter scans:
  `dev/results/speed_scheduled_elbow_qp_*_20260724/`
