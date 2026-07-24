# J1 speed-budget and no-limit ablation

## Scope

This report extends the recorded retract comparison with a normal tangential
motion near the straight-arm configuration.

The synthetic circle:

- starts from the bent state associated with left-arm episode 202 segment 13;
- extends toward the recorded near-straight pose;
- follows a 0.06 m small circle at constant shoulder-to-EEF reach;
- uses a 0.3 s smooth speed ramp before and after the measured revolution;
- holds EEF orientation fixed; and
- tests 0.1, 0.2, 0.4, 0.6, and 0.8 m/s.

Only the constant-speed revolution is included in the metrics. The MuJoCo
plant, current Mink tasks, measured-state safety logic, gravity, and 250 Hz
control period are the same as in the recorded replay experiment.

## Variants

- A: current baseline, including the ordinary velocity box, joint-limit
  braking, and singularity-approach limit.
- H: remove only the ordinary velocity box. Joint-limit braking and the
  singularity limit remain.
- F: remove the ordinary velocity box and `JointBrakingLimit`, but retain the
  singularity limit.
- G: remove all velocity inequalities, including the singularity limit. Hard
  configuration position limits and all IK tasks remain.
- B: allow total J1 speed 3 rad/s in the geometric retract direction and
  2 rad/s in the opposite direction.
- C-pair: allow total J1 speed 3 rad/s while limiting the J1 component outside
  `span(v5, v6)` to 2 rad/s.

G is an unsafe diagnostic reference, not a proposed runtime setting.

## Limit hierarchy

Removing only the ordinary velocity box has no measurable effect: A and H are
identical on every recorded and circular case. The reason is that the current
joint-limit braking envelope is always no looser than the nominal velocity cap:

```text
d_eff = min(d_command, d_measured - reaction_term - distance_buffer)
v_approach = v_max * smoothstep(d_eff / 0.5)^2
```

It therefore subsumes the ordinary box while additionally reducing approach
speed near a guard.

On the circle, J2 approaches its upper guard of 0.17453 rad. J2 braking is the
only joint-limit braking row that binds. The singularity constraint also binds:

```text
rho = sigma_min(J_norm) / sigma_max(J_norm)
-grad(rho)^T qdot <= allowed_approach_rate(rho)
```

| Target speed | J2 braking bind | Singularity bind | A elbow range | G elbow range |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 m/s | 28.3% | 33.8% | 56.7 mm | 67.7 mm |
| 0.2 m/s | 35.2% | 23.1% | 56.9 mm | 68.1 mm |
| 0.4 m/s | 42.4% | 22.5% | 59.6 mm | 72.5 mm |
| 0.6 m/s | 45.9% | 22.9% | 66.9 mm | 76.9 mm |
| 0.8 m/s | 46.6% | 22.9% | 76.6 mm | 93.9 mm |

No baseline circle case reaches the ordinary velocity box. Even at 0.8 m/s,
the peak baseline J1 speed is 1.58 rad/s. B and C-pair are consequently
indistinguishable from A on this motion.

## Circular motion

| Speed | A effective min rho | F effective min rho | G effective min rho | A position RMSE | G position RMSE | A/G EEF accel p99 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 m/s | 0.0326 | 0.0303 | 0.0000 | 25.2 mm | 24.3 mm | 0.8 / 1.0 m/s^2 |
| 0.2 m/s | 0.0371 | 0.0322 | 0.0007 | 25.8 mm | 25.5 mm | 1.7 / 1.3 m/s^2 |
| 0.4 m/s | 0.0418 | 0.0331 | 0.0055 | 28.3 mm | 27.3 mm | 4.9 / 4.6 m/s^2 |
| 0.6 m/s | 0.0427 | 0.0318 | 0.0070 | 35.5 mm | 34.2 mm | 11.4 / 10.2 m/s^2 |
| 0.8 m/s | 0.0423 | 0.0314 | 0.0086 | 52.7 mm | 53.8 mm | 25.1 / 24.0 m/s^2 |

Removing all speed limits does not materially improve EEF tracking or
acceleration on the smooth circle. It does increase elbow lateral travel and
lets the arm enter an almost singular configuration. Removing joint braking
while retaining singularity braking gives an intermediate elbow range:
61.6, 63.0, 69.2, 75.2, and 92.4 mm across the five speeds.

The 0.8 m/s case has target centripetal acceleration of about 10.7 m/s^2 and
is a stress case, not a suggested normal VR speed.

## Recorded paths

The no-limit result is direction dependent:

| Case | Variant | Elbow lateral range | Position RMSE | Peak J1 / J4 command |
| --- | --- | ---: | ---: | ---: |
| slow retract 12 | A | 45.8 mm | 16.1 mm | 1.93 / 3.52 rad/s |
| slow retract 12 | G | 46.1 mm | 16.2 mm | 1.99 / 3.61 rad/s |
| fast retract 13 | A | 178.5 mm | 46.4 mm | 2.00 / 3.80 rad/s |
| fast retract 13 | B | 141.4 mm | 32.0 mm | 3.00 / 3.80 rad/s |
| fast retract 13 | C-pair | 141.6 mm | 32.1 mm | 3.00 / 3.80 rad/s |
| fast retract 13 | G | 79.6 mm | 19.5 mm | 5.97 / 4.32 rad/s |
| reverse 13 | A | 189.0 mm | 63.1 mm | 2.00 / 2.94 rad/s |
| reverse 13 | C-pair | 179.5 mm | 41.0 mm | 3.00 / 3.54 rad/s |
| reverse 13 | G | 176.2 mm | 25.4 mm | 5.32 / 4.11 rad/s |

Fast retract 13 confirms that the large elbow branch change is substantially
caused by saturated joint coordination. The unconstrained QP avoids most of
that elbow travel by demanding unsafe J1 and J4 speeds. B and C-pair recover
part of this tracking ability with a bounded 3 rad/s J1 budget.

Slow retract 12 has no meaningful saturation and therefore shows no meaningful
change. On the near-straight circle, by contrast, removing limits increases
elbow travel. A no-limit reference is useful for attribution but is not a
uniformly better redundancy policy.

## Conclusion

1. Keep joint-limit braking and singularity-approach braking. They are active
   and useful during ordinary near-straight circular motion.
2. The ordinary velocity box is currently behaviorally redundant, but it
   should remain as a simple defense-in-depth constraint.
3. Extra J1 budget addresses the fast-retract bottleneck without changing the
   tested circular motion, because J1 is not the circular-motion bottleneck.
4. Use geometric B for a selective retract-only exception. Use C-pair only if
   bidirectional weak-subspace assistance is required and its higher reverse
   acceleration is acceptable.
5. If circular motion feels overconstrained, investigate the J2 upper-guard
   braking profile. Changing J1 modal allocation will not address that case.

All 30 final circular solves and all 30 recorded-ablation solves completed
without a QP failure. Runtime source files were not modified.
