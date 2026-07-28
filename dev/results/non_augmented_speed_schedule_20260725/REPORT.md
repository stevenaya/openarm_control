# Low-Velocity-Cap IK Study

Date: 2026-07-25

## Goal

The reference behavior uses the locally raised joint-velocity caps:

```text
[3.0, 3.0, 4.14, 4.14, 12.6, 12.6, 12.6] rad/s
```

The safety-oriented comparison uses:

```text
[1.57, 1.57, 3.14, 3.14, 12.6, 12.6, 12.6] rad/s
```

The objective was to preserve the raised-cap elbow branch as closely as possible
under the lower caps, without the augmented QP, while keeping wrist flips,
reach/retract motion, and ordinary workspace trajectories smooth.

## Why The Elbow Branch Changes

For the normalized geometric end-effector Jacobian:

\[
J = U \Sigma V^\mathsf{T}
\]

the exact nullspace direction of a full-rank 6-by-7 arm Jacobian is:

\[
z = V[:, -1], \qquad Jz = 0
\]

Near arm extension, the next right-singular vector:

\[
v_{\mathrm{near}} = V[:, -2]
\]

has a small but nonzero singular value. It is not an exact nullspace direction,
but it can produce large shoulder/elbow joint motion for a modest Cartesian
motion. In the recorded fast retraction, considerably more joint-velocity
energy was carried by `v_near` than by `z`.

This explains the difference between the two candidate elbow objectives:

\[
J_\psi \Delta q
\]

sees exact-null, near-null, and range-space contributions to elbow swivel,
whereas:

\[
J_\psi N \Delta q, \qquad N = zz^\mathsf{T}
\]

only sees the exact one-dimensional nullspace. The projected form therefore
cannot prevent all speed-limit-induced branch changes. The direct form can see
them, but it also competes with the Cartesian task and can redistribute
velocity into other joints.

## Geometric Jacobian Requirement

`FrameTask.compute_jacobian()` includes the target-error-dependent log-map
factor:

\[
J_{\mathrm{task}}
=
-J_{\log}(T_{tb})J_{\mathrm{geo}}
\]

When `Jlog` is invertible, this does not change the exact nullspace. It does
change the singular values and therefore corrupts singularity activation with
the current target error. An unreachable target can then make nullspace return
weaker even when the physical arm configuration is unchanged.

Singularity ratio, exact nullspace, and near-null directions must therefore be
computed from the normalized geometric Jacobian:

\[
\rho(q)
=
\frac{\sigma_{\min}(J_{\mathrm{geo,norm}})}
{\sigma_{\max}(J_{\mathrm{geo,norm}})}
\]

`NullspacePostureTask` now follows this rule.

## Approaches Tested

### Direct elbow-swivel damping

A speed-dependent task on `J_psi Delta q` can reduce branch error, but stronger
costs increase Cartesian path error and shoulder/elbow acceleration. The
finite-difference swivel Jacobian also raised one-arm solve p95 from roughly
1.3 ms to 2.3 ms.

### Exact-null projected elbow damping

The `J_psi N Delta q` form had little effect on the recorded branch change
because the dominant motion was along `v_near`, not `z`.

### Soft elbow corridor

Low corridor costs were ineffective. High costs caused sharp redistribution
when the solution reached the corridor boundary. Recorded J4 acceleration
spikes rose to roughly 441-988 rad/s^2 in the stronger settings. This is not
recommended for the current velocity-QP architecture.

### Near-singular `V[:, -2]` task

A moderate dev-only task reduced acceleration in a synthetic straight
retraction, but did not recover the recorded elbow branch. Expanding its
activation range or cost worsened tracking. Suppressing `v_near` alone does
not solve the hard velocity-box feasibility and time-scaling problem.

### Always-on Cartesian error limiting

Always limiting every first-order Cartesian request improved phase-aligned
elbow similarity, but accumulated 13-24 cm of lag and degraded pure wrist
rotation. It was rejected.

## Selected Conservative Mechanism

The stable non-augmented candidate limits the Cartesian error entering each QP
only while the raw desired target has high translational speed.

The activation input is the raw desired translation speed:

\[
v
=
\frac{\|p_{\mathrm{desired},k}
       -p_{\mathrm{desired},k-1}\|}
      {\Delta t}
\]

Between configured slow and fast translation speeds:

\[
u
=
\operatorname{clip}
\left(
\frac{v-v_{\mathrm{slow}}}
{v_{\mathrm{fast}}-v_{\mathrm{slow}}},
0,1
\right)
\]

\[
\alpha_v = 3u^2 - 2u^3
\]

`alpha_v` is rate-limited separately while rising and falling. Rotation does
not activate it, so a stationary-position wrist flip retains the native IK
behavior.

The first-order task error is norm-clamped independently:

\[
e_{p,\mathrm{clip}}
=
\operatorname{sat}_{L_p}(e_p)
\]

\[
e_{R,\mathrm{clip}}
=
\operatorname{sat}_{L_R}(e_R)
\]

and blended continuously:

\[
e_{\mathrm{used}}
=
e_{\mathrm{full}}
+\alpha_v(e_{\mathrm{clip}}-e_{\mathrm{full}})
\]

At `alpha_v = 0`, the implementation calls the native Mink
`FrameTask.compute_qp_objective()` exactly. A position-lag latch prevents
accumulated lag from being released as one large request immediately after the
measured target speed falls.

## Simulation Summary

Selected candidate:

```text
position error limit:       0.003 m per QP substep
orientation error limit:    disabled
translation slow/fast:      0.5 / 1.0 m/s
activation rise/fall rate:  4.0 / 2.0 1/s
```

Synthetic results under low joint caps:

| Scenario | Low-cap baseline | Scheduled candidate |
|---|---:|---:|
| Reach/retract 0.30 m/s, elbow gap | 0.26 cm | 0.26 cm |
| Reach/retract 0.30 m/s, ddq p99 | 19.8 | 19.8 |
| Circle 0.40 m/s, elbow gap | 0.28 cm | 0.28 cm |
| Circle 0.40 m/s, ddq p99 | 27.5 | 27.5 |
| Reach/retract 0.60 m/s, ddq p99 | 69.3 | 64.6 |
| Circle 0.80 m/s, ddq p99 | 101.2 | 87.6 |
| Wrist flip, elbow gap | 0.55 cm | 0.55 cm |
| Wrist flip, ddq p99 | 348.4 | 348.4 |

Solve p95 remained about 1.25-1.31 ms, with no meaningful overhead compared
with the native task.

For the aggressive reconstructed left-arm intervention segment:

| Metric | Low-cap baseline | Scheduled candidate |
|---|---:|---:|
| Same-time elbow gap to raised-cap reference | 9.42 cm | 10.05 cm |
| EEF-position-aligned elbow gap | 7.77 cm | 2.64 cm |
| Cartesian path gap | 4.48 cm | 9.60 cm |

The same-time metric gets worse because the candidate deliberately slows the
motion. Once aligned by end-effector path phase, its elbow branch is much
closer to the raised-cap reference. The reconstructed target peaked near
1.56 m/s, above the current VR target limit, so this replay is intentionally
more aggressive than normal operation.

The right-arm replay already matched the raised-cap reference closely under
the low caps; the scheduled candidate preserved that behavior. No left-only
or right-only activation was found.

## Integration Status

The conservative limiter is implemented behind default-off arguments:

```text
--frame-position-error-limit 0.003
--frame-orientation-error-limit 0
--frame-error-limit-linear-slow 0.5
--frame-error-limit-linear-fast 1.0
--frame-error-limit-activation-rise-rate 4
--frame-error-limit-activation-fall-rate 2
```

These arguments are intentionally not active in
`dora-openarm-evaluation-ui/dataflow.yaml`. The currently active dataflow
remains the rolled-back version, and the old augmented-QP argument line and
state-feedback notes remain preserved as comments.

The direct speed-scheduled elbow task also remains default-off for controlled
experiments. It is not the recommended production setting from this study.

## Recommendation

1. Keep the current rolled-back dataflow unchanged.
2. Retain the geometric-Jacobian fix for nullspace and singularity logic.
3. Trial the translation-scheduled 3 mm limiter explicitly through command-line
   arguments before enabling it in a deployed dataflow.
4. Do not enable the soft corridor or `V[:, -2]` task by default.
5. If lower caps must preserve both timing and Cartesian tracking, revisit a
   properly time-scaled or augmented formulation; a secondary task cannot
   create joint-speed feasibility that the hard velocity box removes.
