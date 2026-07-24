# J1 modal velocity-budget simulation

## Question

Can J1 keep a conservative speed budget for nonweak task motion while using a
higher total speed budget when the motion belongs to the weak singular mode?

The tested decomposition starts from the normalized geometric Jacobian:

```text
J = U diag(sigma) V^T
```

For the single-mode version:

```text
P_weak = v6 v6^T
qdot1_weak = e1^T P_weak qdot
qdot1_nonweak = e1^T (I - P_weak) qdot
```

The QP keeps the existing total J1 box limit at 3 rad/s and adds:

```text
abs(qdot1_nonweak) <= 2 rad/s
```

The runtime source was not modified. The custom limit exists only in
`dev/sim_modal_shoulder_limit_comparison.py`.

## Compared strategies

- A: original total J1 limit of 2 rad/s.
- E: fixed total J1 limit of 3 rad/s in both directions.
- B: geometric reach-direction limit, 3 rad/s while retracting and 2 rad/s in
  the opposite direction.
- C-single: total 3 rad/s, with only `v6` receiving the extra budget.
- C-pair: total 3 rad/s, with the `span(v5, v6)` weak subspace receiving the
  extra budget.
- C-spectral: continuously weighted weak projector using singular-value ratios
  and the 0.02/0.08 window.
- D: C-single plus the existing weakest-direction target governor at margins
  1.0 and 0.9.

The left-arm tests use intervention episode 202, slow retract segment 12, fast
retract segment 13, and the reversed segment 13 target stream. A right-arm
cross-check uses segments 0 and 1 plus reversed segment 1.

Every replay uses the same current Mink tasks and limits, measured state,
MuJoCo position-controlled plant, gravity, and reconstructed 6D target stream.

## Fast left retract

| Strategy | J1 total saturation | Elbow lateral range | Position RMSE | J1 command accel p99 | J1 actual accel p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A total 2 | 86.3% | 178.5 mm | 46.4 mm | 25.1 rad/s^2 | 26.1 rad/s^2 |
| B reach 3/2 | 46.6% | 141.4 mm | 32.0 mm | 57.4 rad/s^2 | 29.5 rad/s^2 |
| C single | 38.2% | 145.2 mm | 33.8 mm | 90.1 rad/s^2 | 38.7 rad/s^2 |
| C pair | 46.6% | 141.6 mm | 32.1 mm | 58.8 rad/s^2 | 29.6 rad/s^2 |
| C spectral | 0.0% | 178.3 mm | 46.0 mm | 26.9 rad/s^2 | 26.8 rad/s^2 |
| D single + governor 1.0 | 33.8% | 90.5 mm | 53.3 mm | 93.4 rad/s^2 | 29.1 rad/s^2 |

The single-mode QP constraint works numerically: its nonweak J1 p99 is
2.00001 rad/s while the weak contribution reaches 2.24 rad/s. It does not
improve the behavior, because the identity of the single weakest direction is
not stable on this path.

The target governor reduces elbow travel, but does so by accumulating 44.2 mm
of target-position RMS backlog. Margin 0.9 increases that backlog to 60.9 mm
and actual position RMSE to 66.3 mm.

## Weak-mode degeneracy

During fast segment 13:

```text
max(sigma6 / sigma5) = 0.979
```

Therefore `v5` and `v6` are almost degenerate. The individual `v6` vector can
rotate inside their two-dimensional subspace even when the robot geometry
changes smoothly. The sign-invariant projector `v6 v6^T` removes sign flips,
but it cannot remove this mode exchange.

Consequences observed in C-single:

- J1 command acceleration p99 rises from 57.4 to 90.1 rad/s^2 relative to B.
- Actual J1 acceleration p99 rises from 29.5 to 38.7 rad/s^2.
- Tracking and elbow motion are slightly worse than B.

The two-dimensional projector:

```text
P_weak = [v5 v6] [v5 v6]^T
```

is invariant to rotations between `v5` and `v6`. C-pair consequently matches B
on the fast retract within simulation noise.

The tested spectral 0.02/0.08 weighting is too conservative. As rho rises, its
weak weights close and the nonweak bound remains active for 84.3% of the fast
core, so it behaves almost like the original total-2 limit.

## Reversed fast path

| Strategy | Peak J1 command | Elbow lateral range | Position RMSE | EEF accel p99 |
| --- | ---: | ---: | ---: | ---: |
| A total 2 | 2.00 rad/s | 189.0 mm | 63.1 mm | 9.35 m/s^2 |
| B reach 3/2 | 2.00 rad/s | 189.0 mm | 63.1 mm | 9.35 m/s^2 |
| E fixed 3 | 3.00 rad/s | 179.5 mm | 41.0 mm | 12.71 m/s^2 |
| C single | 3.00 rad/s | 182.7 mm | 42.4 mm | 12.89 m/s^2 |
| C pair | 3.00 rad/s | 179.5 mm | 41.0 mm | 12.70 m/s^2 |
| D governor 1.0 | 2.91 rad/s | 163.6 mm | 62.2 mm | 10.55 m/s^2 |

B intentionally preserves the original 2 rad/s limit in the reverse direction.
C-pair releases J1 whenever the requested motion lies in the weak subspace,
regardless of retract or extension direction. It improves reverse tracking but
also raises the acceleration envelope.

This is the main policy difference:

- B is the conservative low-inertia/retract-only rule.
- C-pair is a task-modal, bidirectional rule.

## Slow and right-arm checks

The slow left segment never reaches the J1 speed bound. Position RMSE is about
16.15 mm for every strategy and differs by less than 0.02 mm across variants;
elbow ranges differ by less than 0.2 mm.

The right-arm segments are also below or only briefly touch the J1 limit:

- Segment 0 is identical across all variants.
- Segment 1 position RMSE remains within 12.43 to 12.51 mm.
- Reversed segment 1 improves from 16.15 to 15.77 mm with C-pair, with a small
  increase in acceleration.

No left- or right-arm replay produced a QP solve failure.

## Conclusion

The proposed modal budget is feasible, but a single `v_min` implementation is
not robust enough for this trajectory because the two smallest nonzero
singular values are nearly equal.

C-pair is the valid form of the idea on this data:

```text
total J1 speed <= 3 rad/s
nonweak J1 contribution outside span(v5, v6) <= 2 rad/s
```

It matches the geometric 3/2 strategy on the problematic fast retract and
improves reverse tracking. However, on this trajectory the nonweak constraint
is inactive, so C-pair is behaviorally close to a fixed 3 rad/s J1 limit. Its
benefit is not greater than B during retract.

Recommended interpretation:

1. Use B when the extra J1 budget should only be available in the
   inertia-reducing retract direction.
2. Use C-pair when better bidirectional weak-mode tracking justifies the higher
   reverse-direction acceleration.
3. Do not use C-single or the current single-mode target governor near a
   `sigma5 ~= sigma6` cluster.
4. Keep the hardware total J1 cap, joint braking, and singularity-approach
   inequality in every case.

Before a runtime implementation, weak-subspace dimension should be selected
from a singular-value cluster rather than hard-coded to one or two modes.
