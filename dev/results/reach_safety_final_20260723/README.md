# Shoulder-height reach safety experiment

## Setup

- MuJoCo OpenArm v2 model, right arm.
- Initial arm position:
  `[1.224145, 0, 0, 0.7, 0, 0, 0]`.
- The end-effector target remains at shoulder height and moves `0.25 m` in
  world `+x`, beyond the reachable workspace.
- Target speeds: `0.05`, `0.10`, and `0.20 m/s`.
- Mink runs at `250 Hz` with ten substeps per control tick.
- A separate MuJoCo dynamics state tracks the Mink position command using the
  model position actuators. No velocity feed-forward or gravity compensation is
  applied.
- `summary.csv` contains scalar metrics. Each `trace_*.csv` contains command and
  actual `q/dq/ddq`, end-effector pose, conditioning ratio, actuator force,
  constraint force, and bias force.

## Compared profiles

- `none`: configuration and fixed joint-velocity limits only.
- `acceleration_20`: legacy joint4 `sqrt(2 a d)` envelope.
- `distance_0p50`: legacy joint4 smoothstep distance envelope.
- `all_joint_command`: powered-smoothstep envelopes on both limits of all arm
  joints, using command position only.
- `all_joint_state`: all-joint envelopes additionally tightened by measured
  `q/dq`.
- `all_joint_state_singularity_energy`: final candidate, adding the one-sided
  singularity approach constraint and kinetic-energy regularization.

Final candidate parameters:

```text
joint slowdown distance       0.5 rad
joint smoothstep exponent     2
measured reaction time        0.04 s
measured distance buffer      0.01 rad
rho stop / slow               0.02 / 0.08
maximum rho decrease          0.25 /s
rho smoothstep exponent       2
kinetic-energy cost           3e-5
```

## Main results

The table reports actual joint4 minimum, actual minimum conditioning ratio,
maximum command-to-actual joint1 error, upward end-effector excursion relative
to the start of the reach, and total vertical peak-to-peak motion.

| Speed | Profile | min actual q4 | min actual rho | max q1 error | EE up | EE z P2P |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.05 | none | -0.023 rad | 0.0001 | 0.078 rad | 2.72 mm | 5.33 mm |
| 0.05 | distance_0p50 | 0.062 rad | 0.0082 | 0.060 rad | 1.00 mm | 6.29 mm |
| 0.05 | all_joint_state | 0.144 rad | 0.0191 | 0.058 rad | 1.00 mm | 4.93 mm |
| 0.05 | final | 0.206 rad | 0.0274 | 0.057 rad | 0.64 mm | 4.29 mm |
| 0.10 | none | -0.023 rad | 0.0001 | 0.080 rad | 3.21 mm | 6.92 mm |
| 0.10 | distance_0p50 | 0.065 rad | 0.0086 | 0.067 rad | 2.50 mm | 9.04 mm |
| 0.10 | all_joint_state | 0.154 rad | 0.0205 | 0.066 rad | 2.50 mm | 9.46 mm |
| 0.10 | final | 0.217 rad | 0.0288 | 0.062 rad | 1.71 mm | 7.68 mm |
| 0.20 | none | -0.022 rad | 0.0001 | 0.087 rad | 6.54 mm | 15.49 mm |
| 0.20 | distance_0p50 | 0.068 rad | 0.0090 | 0.087 rad | 5.52 mm | 17.18 mm |
| 0.20 | all_joint_state | 0.164 rad | 0.0218 | 0.086 rad | 4.94 mm | 13.57 mm |
| 0.20 | final | 0.227 rad | 0.0301 | 0.071 rad | 2.96 mm | 10.89 mm |

## Reproduced upward bump

At `0.20 m/s`, the no-braking dynamic end effector reaches a vertical minimum
at `t=0.236 s`, then rebounds by `15.49 mm` to a later maximum at
`t=0.388 s`. The kinematic command keeps end-effector vertical velocity near
zero.

At the peak upward-velocity sample:

```text
no braking, command Jz*dq:
joint1 +0.2 mm/s, joint4 -0.2 mm/s, sum approximately 0

no braking, actual Jz*dq:
joint1 +110.2 mm/s, joint4 +49.9 mm/s, sum +160.1 mm/s
```

The command has already reached the straight-arm boundary, while the actual
joint1 still moves forward and the actual joint4 is recovering from its limit
overshoot. Those two actual contributions point upward instead of cancelling.
The no-braking run reaches about `41.66 Nm` of joint4 constraint force.

For the final candidate at its peak upward-velocity sample:

```text
final, command Jz*dq:
joint1 +72.1 mm/s, joint4 -71.9 mm/s, sum +0.1 mm/s

final, actual Jz*dq:
joint1 +155.9 mm/s, joint4 -72.1 mm/s, sum +83.8 mm/s
```

The final constraints avoid joint4 contact and preserve the command-level
vertical cancellation. A residual dynamic mismatch remains because the
position-only actuator tracks the shoulder and elbow with different phase lag.

## Interpretation

The original bump is not a QP decision to sacrifice vertical position. Its
command-space end-effector vertical motion remains approximately zero. It is a
dynamic tracking failure triggered by reaching the straight-arm/joint boundary:

1. the kinematic command coordinates large, opposing joint1 and joint4
   contributions;
2. the real position-controlled joints lag by different amounts;
3. joint4 reaches or crosses its boundary and reverses while joint1 is still
   catching up;
4. the intended Cartesian cancellation disappears, so the actual end effector
   moves upward.

Gravity is a contributor to tracking lag: joint1 bias torque is about `11.5 Nm`
near extension in this model. It is not the sole trigger, and neither shoulder
actuator reaches its `40 Nm` force cap in these runs. Gravity compensation
should be evaluated separately after the kinematic safety envelopes, because it
changes the plant tracking error rather than the QP geometry.

The kinetic-energy task is a soft regularizer. A cost of `3e-5` reduced the
large command acceleration while adding about `2.4 mm` of command x lag in the
regular (`rho > 0.06`) part of the `0.20 m/s` run. A cost of `1e-3` nearly
removed the vertical transient but reduced mean command x speed from roughly
`0.099` to `0.033 m/s`, so it was rejected as excessive damping.

## Solver timing

An isolated 200-sample bimanual benchmark, after 20 warm-up samples, measured:

| Configuration | Mean | P50 | P95 | Max |
| --- | ---: | ---: | ---: | ---: |
| Previous IK objectives and limits | 1.597 ms | 1.600 ms | 1.617 ms | 1.635 ms |
| Final state-aware safety configuration | 2.539 ms | 2.554 ms | 2.584 ms | 2.643 ms |

The finite-difference singularity gradients add about `0.94 ms` per bimanual
solve on this machine. The isolated P95 remains below the `4 ms` period, but the
full Dora process should still be checked under normal system load.
