# Weak-direction target governor replay

## Goal

Test whether limiting a VR target before Mink can prevent the J1 velocity box
from forcing the elbow onto a different redundancy branch.

The experiment replays episode 202 segments 12 and 13. Raw VR poses were not
recorded, so the target is the full 6D FK pose of the recorded IK command.

## Geometry

At the current command configuration:

```text
J_norm = U diag(sigma) V^T
e_norm = [translation / characteristic_length, rotation]
```

For one raw target increment, the modal joint displacement is:

```text
c = U^T e_norm
a_i = c_i / sigma_i
dq_pred = sum_i a_i v_i
```

`v_5 = V[:, 5]` is the smallest nonzero singular direction for the 6-by-7
arm Jacobian. `V[:, 6]` is the exact structural nullspace direction.

The weak-first limiter finds the largest `gamma` in `[0, 1]` satisfying:

```text
|dq_nonweak + gamma a_5 v_5|
    <= velocity_margin dq_max dt
```

Only the weak coefficient is reduced when this interval is feasible. If the
other five modes already exceed the budget, the complete target increment is
uniformly scaled as a fallback. This fallback was required in about 8% of the
fast-segment frames, confirming that `u_min` alone does not explain all J1
demand.

The governed increments are integrated into a persistent target. When the raw
target is nearly stationary, the remaining full 6D error is advanced using
only the unused predicted joint-velocity budget. This allows the robot to
catch up without discarding operator intent.

## Rejected prototypes

### Clamp current-to-raw absolute error along only `u_min`

The suppressed error accumulated and was interpreted as a new one-step
velocity request on every frame. Requested weak speed grew above
`2000 rad/s`, the target lag exceeded `0.16 m`, and even the slow trajectory
was damaged.

### Project current-to-raw error into one-period row-space feasibility

This required the entire accumulated frame error to be closed in one 4 ms
period and initially omitted the seventh exact-nullspace coefficient. It
therefore classified normal multi-period tracking error as infeasible and
held the target about `0.14-0.17 m` behind even on the slow segment.

The usable design must filter target increments, not reinterpret accumulated
tracking error as instantaneous target velocity.

## Recorded-path results

### Slow segment 12

| Variant | Position RMSE | Orientation RMSE | Core elbow delta | J1 core saturation |
|---|---:|---:|---:|---:|
| Baseline | 0.0026 m | 0.0012 rad | 0.0472 m | 0.0% |
| Governor, margin 0.9 | 0.0027 m | 0.0012 rad | 0.0472 m | 1.6% |
| Governor, margin 0.8 | 0.0031 m | 0.0013 rad | 0.0471 m | 0.0% |

The governor is effectively transparent on the slow trajectory.

### Fast segment 13

| Variant | Core elbow delta | Exact-z core drift | J1 core saturation | Any-limit frames | Position RMSE | Target-position lag RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.1812 m | +1.1269 rad | 86.3% | 49.5% | 0.0277 m | 0.0000 m |
| Margin 1.00 | 0.0817 m | -0.2195 rad | 80.9% | 56.2% | 0.0840 m | 0.0808 m |
| Margin 0.98 | 0.0770 m | -0.2637 rad | 63.2% | 47.0% | 0.0852 m | 0.0833 m |
| Margin 0.95 | 0.0733 m | -0.2828 rad | 32.8% | 24.8% | 0.0886 m | 0.0871 m |
| Margin 0.92 | 0.0714 m | -0.2840 rad | 21.1% | 14.6% | 0.0929 m | 0.0916 m |
| Margin 0.90 | 0.0702 m | -0.2876 rad | 10.8% | 8.9% | 0.0967 m | 0.0954 m |

The branch change is strongly reduced, but this cannot be free: the original
path asks for more J1 velocity than the retained safety limit permits. The
governor converts the infeasible joint motion into temporary task-space lag.

## Same path at different speeds

The margin-0.9 governor was replayed on the same 6D path:

| Duration | Peak target speed | Variant | Any-limit frames | Elbow delta | Target-position lag RMSE |
|---:|---:|---|---:|---:|---:|
| 1x | 1.390 m/s | Baseline | 49.5% | 0.1927 m | 0.0000 m |
| 1x | 1.390 m/s | Governor | 8.9% | 0.0671 m | 0.0954 m |
| 2x | 0.778 m/s | Baseline | 14.4% | 0.1043 m | 0.0000 m |
| 2x | 0.778 m/s | Governor | 0.62% | 0.0817 m | 0.0208 m |
| 4x | 0.389 m/s | Baseline | 0.25% | 0.0861 m | 0.0000 m |
| 4x | 0.389 m/s | Governor | 0.25% | 0.0861 m | below 0.1 um |

At 4x duration, the governor never activates and reproduces the baseline to
floating-point precision. It is driven by predicted feasibility, not merely
by proximity to a singular configuration.

## Catch-up after the handle stops

For the 1x fast path with margin 0.9 and a 1.5 s final hold:

```text
target lag at hold start:       0.0345 m
time to target lag <= 0.020 m:  0.020 s
time to target lag <= 0.005 m:  0.040 s
final target lag:               below numerical precision
hold J1 saturation:             0%
hold J1 peak utilization:       0.935
additional elbow lateral move:  0.0078 m
```

The original target is retained and recovered without a second branch jump.

## Remaining issue: target acceleration continuity

The prototype clips each target increment independently. This improves branch
consistency but can create discontinuities when the limiter activates or
releases:

```text
                              Baseline   Margin 0.9
actual EEF acceleration p99   13.1       12.3 m/s^2
actual EEF acceleration peak  13.3       14.5 m/s^2
command J1 acceleration peak  256        340 rad/s^2
command J3 acceleration peak  177        472 rad/s^2
```

Before runtime integration, the governed target velocity needs a continuous
attack/release slope or an acceleration bound. A practical implementation
should use a safety margin below the true velocity limit so it can start
decelerating before the hard QP box becomes active.

## Recommendation

The experiment supports a runtime target governor, with these constraints:

1. Filter fresh target increments, not accumulated FrameTask error.
2. Use the geometric Jacobian and characteristic-length normalization.
3. Reduce the weakest mode first.
4. Fall back to scaling the full increment only when weak-mode reduction alone
   cannot make the predicted motion feasible.
5. Preserve the raw target and perform bounded full-6D catch-up when input
   motion slows or stops.
6. Add target-velocity acceleration continuity before real-arm testing.

`velocity_margin=0.9` is the safer tested starting point. At the approximately
`0.78 m/s` replay speed, it reduced limit activity to `0.62%` with about
`2.1 cm` RMS target lag. It should remain configurable rather than becoming a
hard-coded value.
