# Shoulder-height reach braking experiment

## Setup

- Initial end-effector height equals the right shoulder height.
- The target moves along world +X by 0.25 m, beyond the reachable workspace.
- Target speeds: 0.05, 0.10, and 0.20 m/s.
- The dynamic MuJoCo model uses the current high-PD actuator parameters, gravity,
  no gravity compensation, and the current 250 Hz IK command loop.
- Command state stays open-loop during active IK, matching the current node.
- Profiles:
  - no elbow braking
  - acceleration profile with `a_brake=20 rad/s^2`
  - distance profiles with slowdown distances 0.50, 0.70, and 1.00 rad

Dashed lines in the comparison plots are commands. Solid lines are simulated
actual states.

## Main observations

At 0.10 m/s, the no-braking profile reaches the straight-arm singular
configuration and abruptly stops its shoulder command:

- peak command shoulder acceleration: 291 rad/s^2
- peak simulated shoulder acceleration: 106 rad/s^2
- peak simulated shoulder speed: 1.75 rad/s
- actual elbow minimum: -0.0227 rad
- elbow hard-limit constraint force: about 44 Nm

The shoulder actuator itself does not saturate. The actual elbow penetrates the
zero lower limit, and the resulting MuJoCo constraint impulse couples through
the robot dynamics into the shoulder.

The acceleration profile prevents hard-limit contact, but still changes the
shoulder command relatively quickly:

- peak command shoulder acceleration: about 10.5 rad/s^2
- peak simulated shoulder acceleration: about 18.8 rad/s^2
- actual elbow minimum: 0.0415 rad

The distance profile progressively smooths the shoulder motion. At 0.10 m/s:

| Profile | Peak command shoulder acceleration | Peak actual shoulder acceleration | Minimum actual elbow |
| --- | ---: | ---: | ---: |
| distance 0.50 | 3.69 rad/s^2 | 7.53 rad/s^2 | 0.0647 rad |
| distance 0.70 | 1.15 rad/s^2 | 2.58 rad/s^2 | 0.0714 rad |
| distance 1.00 | 0.32 rad/s^2 | 0.68 rad/s^2 | 0.0850 rad |

The same ranking holds at 0.05 and 0.20 m/s. A 1.00 rad slowdown distance gives
the smoothest shoulder state in this test.

## Command versus simulated actual

Gravity creates a steady command-to-state offset even before moving:

- shoulder q1 error: approximately -0.0475 rad
- elbow q4 error: approximately -0.0222 rad
- end-effector sag: approximately 24 mm

Therefore, a braking envelope computed only from command q4 cannot guarantee
that measured q4 remains above the same guard. Increasing the slowdown distance
reduces transient lead, but it does not remove this steady offset.

## End-effector behavior

The command end-effector height remains constant, but the simulated actual
end-effector rises and then falls during the approach. This directional
transient was hidden by the original norm-only error metric.

Near full extension:

| Target speed | Profile | Upward excursion | Fall from peak |
| --- | --- | ---: | ---: |
| 0.10 m/s | none | 6.90 mm | 1.44 mm |
| 0.10 m/s | acceleration 20 | 3.39 mm | 2.63 mm |
| 0.10 m/s | distance 0.50 | 4.46 mm | 2.18 mm |
| 0.10 m/s | distance 1.00 | 1.06 mm | 0.93 mm |
| 0.20 m/s | none | 5.22 mm | 4.69 mm |
| 0.20 m/s | distance 0.50 | 8.71 mm | 4.78 mm |
| 0.20 m/s | distance 1.00 | 1.22 mm | 1.02 mm |

This is not vertical leakage in the kinematic command. For the 0.20 m/s,
distance-0.50 run at 0.308 s, the command Jacobian contributions are:

- q1: +89.38 mm/s
- q4: -89.38 mm/s
- total command vertical velocity: approximately zero

The simulated actual joints at the same time produce:

- q1: +195.24 mm/s
- q4: -69.60 mm/s
- total actual vertical velocity: +125.64 mm/s

At 0.472 s, q1 has already stopped while q4 is still moving, producing about
-65.13 mm/s downward velocity. The upward-then-downward motion therefore comes
from phase mismatch between the independently controlled shoulder and elbow.
Their kinematic contributions cancel in the QP command but not after different
joint inertia, gravity loading, PD gains, and tracking lag act on them.

No sustained end-effector oscillation appears in the final hold stage for any
profile: the X and Z error have zero sign crossings. The distance profiles show
small monotonic settling drift, below 0.1 mm peak-to-peak in the worst tested
tail window.

The simulation includes rigid-body dynamics, gravity, joint limits, and the
current PD actuators. It does not include communication delay and jitter,
elasticity, backlash, or a detailed motor current loop. Those effects can make
physical-arm oscillation larger than this result.

## Conclusion

The observed shoulder bump is not only a singularity metric issue or a QP
position-task sacrifice. In the no-braking case it is the combination of:

1. a discontinuous shoulder command near the fully extended IK solution;
2. shoulder/elbow tracking phase mismatch that breaks their vertical
   cancellation;
3. actual elbow hard-limit impact and dynamic coupling into the shoulder.

The distance-dependent velocity envelope improves all three transient metrics.
For this trajectory, 0.70 rad is a useful compromise and 1.00 rad is the
smoothest tested value. Because the real state lags under gravity, measured
state feedback or gravity compensation is still needed if the 0.08 rad elbow
guard must be respected physically rather than only in command space.
