# IK Safety Limits

This note documents the preventive joint-limit and singularity limits used by
the OpenArm Mink IK solver. Both mechanisms are QP inequalities. They shape the
next differential-IK displacement before it is integrated, rather than adding a
soft task that competes with the end-effector task after a boundary is crossed.

## QP variable and timing

Mink solves for a tangent displacement `Delta q` over one integration substep:

```text
Delta q = qdot * dt_sub
dt_sub = control_dt / max_iters
```

`mink.solve_ik()` returns `qdot = Delta q / dt_sub`, and the caller integrates
that velocity by `dt_sub`. With ten substeps in a 4 ms control tick, all ten
substeps together still represent one 4 ms command interval.

Therefore, a velocity bound `a^T qdot <= v_max` is passed to Mink as:

```text
a^T Delta q <= v_max * dt_sub
```

## Joint-limit approach envelope

For a scalar joint with lower and upper guard positions:

```text
d_lower = q - q_lower_guard
d_upper = q_upper_guard - q
```

The allowed speed toward either guard is:

```text
u = clip(d / d_slow, 0, 1)
s(u) = 3 u^2 - 2 u^3
v_allowed(d) = v_joint_max * s(u)^p
```

The default exponent is `p = 2`. This gives:

- zero approach speed at the guard;
- the normal joint velocity cap outside the slowdown interval;
- zero slope at both ends of the interval;
- a stronger final slowdown than plain smoothstep.

The two one-sided QP inequalities are:

```text
 Delta q_i <= v_upper_allowed * dt_sub
-Delta q_i <= v_lower_allowed * dt_sub
```

Only motion toward a nearby guard is reduced. Motion away from that guard is
still allowed by the opposite inequality.

### Measured-state prediction

The command configuration can lead the physical arm. To avoid activating the
envelope only when the command reaches a guard, the effective distance is the
more conservative of command distance and a short measured-state prediction:

```text
d_lower_pred =
    q_measured - q_lower_guard
    - reaction_time * max(-qdot_measured, 0)
    - distance_buffer

d_upper_pred =
    q_upper_guard - q_measured
    - reaction_time * max(qdot_measured, 0)
    - distance_buffer

d_effective = min(d_command, d_pred)
```

The measured state only tightens the safety envelope. It does not overwrite the
Mink command configuration and therefore does not erase the command integrator
on every control tick.

### Legacy acceleration profile

The previous elbow-only profile used stopping distance:

```text
v_allowed(d) = min(v_joint_max, sqrt(2 * a_brake * max(d, 0)))
```

This follows from `v^2 = 2 a d`. It is not an explicit inter-frame acceleration
constraint. It is another distance-to-maximum-speed envelope whose shape is
parameterized by `a_brake`. The powered-smoothstep profile is easier to specify
in terms of a visible slowdown distance and reaches exactly zero at the guard.

## Singularity metric

For one 7-DoF arm, let the end-effector geometric Jacobian be:

```text
J(q) in R^(6 x 7)
```

Its translational rows have units of meters per radian, while its rotational
rows have units of radians per radian. They cannot be compared directly by an
SVD. A characteristic arm length `L` converts translation into a dimensionless
quantity:

```text
J_norm = [ J_position / L ]
         [ J_rotation     ]
```

Changing `L` changes the relative importance of translation and rotation, so it
does affect the numerical singular values. That is intentional: `L` defines
what one meter of translational motion means relative to one radian of
rotational motion. The same `L` must be used for logging, nullspace modulation,
and singularity braking.

The scale-independent condition ratio is:

```text
rho(q) = sigma_min(J_norm(q)) / sigma_max(J_norm(q))
```

`rho` approaches zero when one Cartesian motion direction becomes unavailable.
It is not a geometric distance in meters or radians; it is a local
dimensionless conditioning measure.

### Geometric Jacobian versus FrameTask error Jacobian

Mink exposes two related Jacobians that serve different purposes.

`Configuration.get_frame_jacobian()` returns the geometric frame Jacobian:

```text
v_frame = J_geo(q) qdot
```

It maps joint velocity to the current frame twist. It depends on the robot
model, frame, and current configuration `q`, but not on an end-effector target.

`FrameTask.compute_jacobian()` returns the Jacobian of the task error
coordinates. Mink defines the body-frame pose error using the SE(3) logarithm:

```text
T_error = inverse(T_target) T_frame(q)
e(q, T_target) = -Log(T_error)
```

Linearizing that error gives:

```text
J_task(q, T_target)
    = partial e / partial q
    = -JLog(T_error) J_geo(q)
```

`JLog(T_error)` is the Jacobian of the SE(3) logarithm. It depends on the
relative transform between the target and the current frame. Therefore,
`J_task` changes when the target changes, even if the physical robot
configuration remains exactly the same:

```text
q fixed, target A != target B

J_geo(q)                         unchanged
J_task(q, target A)              generally differs from
J_task(q, target B)
```

Left multiplication by `JLog` is exactly what Mink needs to predict reduction
of its nonlinear pose-error coordinates. It is the correct Jacobian for
building the `FrameTask` QP objective. Its singular values, however, combine
two effects:

1. geometric conditioning of the manipulator at `q`;
2. conditioning and scaling of the target-dependent SE(3) error
   parameterization.

This distinction matters when the VR target moves outside the reachable
workspace. The robot can remain at the same `q`, while `T_error` grows and
`JLog(T_error)` changes. A ratio computed from `J_task` can consequently change
only because the handle target moved farther away. The resulting quantity would
really be:

```text
rho_task(q, T_target)
```

rather than a robot-only singularity metric:

```text
rho_geo(q)
```

Using `rho_task` in a safety limit creates an undesirable feedback path:

```text
unreachable target moves
  -> JLog changes
  -> reported singularity changes
  -> safety velocity bound changes
```

The target must not be able to make an unchanged arm appear more or less
geometrically singular. The singularity approach limit therefore obtains
`J_geo` from `Configuration.get_frame_jacobian()`, selects the seven arm
columns, normalizes the translational rows, and computes `rho` from that matrix.
Its finite-difference gradient is consequently:

```text
g(q) = grad_q rho_geo(q)
```

and not a gradient of target error.

This does not mean `FrameTask.compute_jacobian()` should be removed elsewhere.
The frame tracking objective must use `J_task`. The nullspace posture task also
uses the same task linearization so that its selected direction is numerically
consistent with the primary QP objective. The API choice is therefore:

| Use | Jacobian |
| --- | --- |
| Frame pose-error objective | `FrameTask.compute_jacobian()` |
| Nullspace relative to that objective | `FrameTask.compute_jacobian()` |
| Robot geometric singularity and `rho` | `Configuration.get_frame_jacobian()` |

A regression test also constructs an untargeted `FrameTask` and verifies that
the singularity ratio and its finite-difference gradient are identical to those
obtained with a targeted task. This catches accidental reintroduction of
target-dependent `JLog` into the safety metric.

## Gradient and direction of motion

Define the configuration-space gradient:

```text
g(q) = grad_q rho(q) in R^7
```

For a small arm displacement `Delta q`, first-order Taylor expansion gives:

```text
rho(q + Delta q)
  = rho(q) + g(q)^T Delta q + O(||Delta q||^2)
```

Ignoring the second-order remainder:

```text
Delta rho ~= g^T Delta q
```

This gives an immediate directional interpretation:

- `g^T Delta q < 0`: `rho` decreases, so the motion approaches poorer
  conditioning.
- `g^T Delta q = 0`: the motion is tangent to a local constant-`rho` surface.
- `g^T Delta q > 0`: `rho` increases, so the motion leaves the singular region.

The implementation computes `g` with a central finite difference in MuJoCo's
tangent space:

```text
g_i ~= [rho(q plus eps e_i) - rho(q minus eps e_i)] / (2 eps)
```

`mj_integratePos()` is used for the perturbation, so this remains valid for
MuJoCo configuration coordinates rather than assuming that every `qpos` can be
modified by ordinary array addition.

## Explicit component decomposition

When `g` is nonzero, any arm velocity can be split into its component normal to
the constant-`rho` surface and its tangent component:

```text
qdot_normal =
    (g^T qdot / g^T g) g

qdot_tangent =
    qdot - qdot_normal
```

The reconstruction is exact:

```text
qdot = qdot_tangent + qdot_normal
```

and:

```text
g^T qdot_tangent = 0
```

If `g^T qdot < 0`, only `qdot_normal` is approaching the singularity. A
post-solve implementation could scale that component:

```text
qdot_final = qdot_tangent + gamma_rho * qdot_normal
```

with `0 <= gamma_rho <= 1`. This leaves tangent and escaping motion unchanged.
However, post-solve scaling can invalidate other QP constraints and tasks. It is
cleaner to impose the same directional bound inside the existing Mink QP.

## One-sided singularity QP inequality

Let `v_rho_allowed` be the maximum allowed rate at which `rho` may decrease:

```text
rho_dot >= -v_rho_allowed
```

Using `rho_dot ~= g^T qdot`:

```text
g^T qdot >= -v_rho_allowed
```

Converting this lower bound into the standard QP form `G Delta q <= h`:

```text
-g^T Delta q <= v_rho_allowed * dt_sub
```

This is one row in the QP:

```text
G[arm_dofs] = -g
h = v_rho_allowed * dt_sub
```

Consequences:

- tangent motion has a left-hand side of zero and remains free;
- motion that increases `rho` has a negative left-hand side and remains free;
- only excessive decrease of `rho` is clipped;
- no explicit nullspace basis or Jacobian inverse is required.

## Distance-dependent singularity approach rate

The allowed decrease rate is reduced as the arm enters a configured
conditioning band:

```text
u = clip(
    (rho_effective - rho_stop) / (rho_slow - rho_stop),
    0,
    1,
)

v_rho_allowed = v_rho_max * (3 u^2 - 2 u^3)^p
```

At and below `rho_stop`, first-order motion further into the singularity is
stopped. At and above `rho_slow`, the normal IK solution is unaffected. The
smoothstep has zero slope at both thresholds, avoiding an abrupt change of QP
feasible set as `rho` crosses either boundary. The exponent controls how
strongly the final portion is slowed.

For state-aware activation:

```text
rho_effective = min(rho_command, rho_measured)
```

The gradient is still evaluated at the command configuration because that is
where Mink linearizes and solves its QP. The measured ratio only chooses the
conservative allowed rate. This preserves a coherent QP tangent point while
reacting earlier when the physical arm is already less well conditioned than
the command model suggests.

## Scope and limitations

These constraints are kinematic. They do not model actuator torque, gravity,
motor bandwidth, link inertia, or contact. MuJoCo dynamics are used in the
separate reach experiment to measure how a position-only actuator follows the
kinematic command. A kinetic-energy regularization task can be added after the
kinematic envelopes are validated, but it remains a soft objective and does not
replace state-aware preventive limits.

The singularity constraint is based on a local first-order model. Near points
where singular values cross or the gradient becomes very small, higher-order
effects matter. The implementation therefore records `rho`, singular values,
`||g||`, activation, and allowed approach rate so thresholds can be selected
from simulation and real records rather than treated as universal constants.
