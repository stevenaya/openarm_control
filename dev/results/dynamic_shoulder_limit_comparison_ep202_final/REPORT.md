# Dynamic shoulder velocity-limit simulation

## Setup

- Source: intervention episode 202, left-arm retract segments 12 and 13.
- Segment 12 core target speed: 0.241 m/s.
- Segment 13 core target speed: 0.580 m/s.
- Targets are reconstructed by FK from the recorded command trajectory.
- Mink uses the current nullspace, joint braking, singularity approach, kinetic
  energy, and measured-state settings.
- A separate MuJoCo position-controlled plant produces actual `q`, `dq`, elbow,
  end-effector, and mass-matrix signals.
- The driver/hardware limit is not changed by this experiment.

The tested asymmetric limit is a Mink QP inequality:

```text
-v1_negative(q) * dt <= delta_q1 <= v1_positive(q) * dt
```

The shoulder-to-end-effector reach is:

```text
r(q) = ||p_ee(q) - p_shoulder(q)||
dr/dq1 = r_hat^T J_position[:, q1]
```

The sign of `-dr/dq1` identifies the J1 direction that decreases reach. That
side receives the higher speed cap; the opposite side remains at 2 rad/s. The
sign is derived from geometry independently for the left and right arms.

## Main comparison

Slow segment 12 never saturates J1. Baseline, fixed 4 rad/s, and asymmetric
4/2 rad/s remain effectively identical:

| Strategy | Peak J1 cmd | Elbow lateral delta | Actual position RMSE |
| --- | ---: | ---: | ---: |
| Baseline 2/2 | 1.933 rad/s | 45.79 mm | 16.14 mm |
| Fixed 4/4 | 1.981 rad/s | 45.94 mm | 16.15 mm |
| Asymmetric retract 4/2 | 1.962 rad/s | 45.77 mm | 16.14 mm |

Fast segment 13 exposes the active-set change:

| Retract-side cap | J1 saturation | Other-joint saturation | Elbow lateral delta | Actual position RMSE | J1 cmd accel p99 | J1 actual accel p99 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 86.3% | 51.5% | 178.5 mm | 46.4 mm | 35.4 rad/s^2 | 26.2 rad/s^2 |
| 3.0 | 46.6% | 54.9% | 141.4 mm | 32.0 mm | 42.5 rad/s^2 | 29.5 rad/s^2 |
| 3.5 | 38.2% | 49.0% | 116.9 mm | 26.6 mm | 82.3 rad/s^2 | 32.3 rad/s^2 |
| 4.0 | 28.4% | 40.2% | 103.9 mm | 22.9 mm | 90.0 rad/s^2 | 37.4 rad/s^2 |

The end-effector acceleration p99 is 13.25, 12.32, 12.87, and 12.98 m/s^2
respectively. The branch and tracking improvement therefore comes primarily
from avoiding sustained J1 saturation, not from making the whole arm more
aggressive.

Fixed symmetric 4/4 and asymmetric 4/2 behave almost identically during the
retract itself. However, fixed 4/4 reaches a J1 command-acceleration peak of
651 rad/s^2, while asymmetric 4/2 remains at 256 rad/s^2, equal to the
baseline peak. The lower reverse-direction cap removes an unnecessary fast
release/reversal.

## Direction criteria

Three direction tests were compared:

1. Singularity escape: `d rho / d q1`.
2. Decreasing shoulder inertia: `-d M11 / d q1`.
3. Decreasing shoulder-to-EEF reach: `-d r / d q1`.

On the recorded paths:

```text
abs(d rho / d q1) ~= 1e-12
d M11 / d q1      = 0 numerically
-d r / d q1       = 0.20 to 0.43 m/rad
```

J1 mainly changes the arm's global shoulder direction. It leaves the Jacobian
singular values and shoulder diagonal mass term nearly invariant, so neither
`rho` nor `M11` can identify which J1 sign is retracting. Their apparent sign
changes are numerical noise.

The actual shoulder mass-matrix term still drops strongly during retraction:

```text
segment 12: M11 0.444 -> 0.180
segment 13: M11 0.448 -> 0.119
```

That decrease is caused by the coordinated J2/J4 configuration change, not by
J1 itself. It supports the intuition that the retracted arm is dynamically
easier to move, but it is not a usable J1 direction classifier.

## Reverse and outward tests

The fast diagonal retract was replayed in reverse:

| Strategy | Peak J1 cmd | J1 saturation | Elbow lateral range | Position RMSE | EEF accel p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline 2/2 | 2.00 rad/s | 80.9% | 189.0 mm | 63.1 mm | 9.35 m/s^2 |
| Fixed 4/4 | 3.97 rad/s | 0.5% | 173.1 mm | 40.3 mm | 12.59 m/s^2 |
| Asymmetric retract 4/2 | 2.00 rad/s | 80.9% | 189.0 mm | 63.1 mm | 9.35 m/s^2 |

The asymmetric limit intentionally reproduces the baseline on extension,
while fixed 4/4 also enables the aggressive direction toward the extended
configuration.

Pure shoulder-height radial reaches at 0.1, 0.2, and 0.4 m/s require less than
1 rad/s from J1, so all strategies are identical there. This trajectory does
not stress the shoulder velocity bound; the reversed diagonal path does.

The right-arm intervention has much less J1 saturation. Its slow segment is
identical across strategies. On its faster segment, raising the retract-side
cap changes peak J1 from 1.96 to 2.26 rad/s and changes position RMSE only from
12.50 to 12.41 mm. This confirms that the large gain is specific to the
left-arm path where J1 is persistently active.

## Acceleration variants

Applying a global J1 acceleration window at 20 rad/s^2 preserves the fast
retract branch:

```text
elbow lateral delta: 97.5 mm
position RMSE:        24.1 mm
J1 command accel:     exactly bounded to 20 rad/s^2
J1 actual accel p99:  32.2 rad/s^2
```

It does not reduce MuJoCo end-effector acceleration, and it slightly changes
the slow trajectory. Windows of 30, 50, and 100 rad/s^2 produce no better
dynamic result because the high-PD plant response is not determined only by
the command finite difference.

An experimental "limit acceleration only above 2 rad/s" state machine is
rejected. Releasing the state at 2 rad/s creates a new discontinuity and
produces command acceleration peaks of 178 to 349 rad/s^2 in the core, with
an overall peak near 983 rad/s^2.

A 50 rad/s^2 slew on the cap itself is also ineffective here: the cap reaches
4 rad/s before J1 needs it, so its trace is identical to the unslewed
asymmetric limit.

## Recommendation

1. Keep the driver absolute J1 velocity limit unchanged.
2. Keep the full-vector singularity approach inequality and all joint braking
   limits in the Mink QP.
3. Add an asymmetric Mink `Limit` for J1:
   - normal and extension side: 2 rad/s;
   - geometrically retracting side: initially 3 rad/s;
   - derive the sign from `-r_hat^T J_position[:, q1]`, never hard-code left
     and right signs.
4. Do not gate the higher cap with the current `rho=0.02/0.08` window. In the
   fast replay it only raises the cap to 2.35 rad/s and leaves the branch jump
   almost unchanged.
5. Do not add the experimental acceleration state machine initially.
6. Log positive/negative J1 caps, J1 command/actual velocity, reach derivative,
   `rho`, and active QP bounds during hardware validation.
7. Start hardware validation at 3 rad/s. Move to 3.5 rad/s only if the elbow
   branch remains visible and the measured motor/driver limits permit it.

The 3 rad/s variant gives the best conservative tradeoff in this simulation:
it removes about 37 mm of elbow lateral motion and 14 mm of position RMSE,
while keeping actual J1 acceleration and end-effector acceleration close to
the 2 rad/s baseline.

## Artifacts and limitations

- Main summary: `summary.csv`
- Per-run arrays: `trace_*.npz`
- Reverse test:
  `../dynamic_shoulder_limit_comparison_ep202_reverse13/summary.csv`
- Right-arm test:
  `../dynamic_shoulder_limit_comparison_ep202_right/summary.csv`
- Cap sweep:
  `../dynamic_shoulder_limit_comparison_ep202_high3/summary.csv` and
  `../dynamic_shoulder_limit_comparison_ep202_high3p5/summary.csv`
- Simulation source:
  `../../sim_dynamic_shoulder_limit_comparison.py`

MuJoCo includes gravity and the model's position actuator dynamics, but it
does not reproduce all real driver latency, friction, current limiting, or
structural compliance. The targets are reconstructed from recorded commands,
not raw VR packets. No runtime `src/` code was changed by this experiment.
