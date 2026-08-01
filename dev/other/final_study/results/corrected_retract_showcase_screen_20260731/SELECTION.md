# Corrected Right-Arm Retract Showcase Selection

## Comparison Conditions

The screening used right-arm retract windows from intervention episodes
73, 75, 77, 79, 81, 83, 85, 87, 89, 91, and 93. A semantic sampler retained
264 windows covering the highest retract score, elbow excursion, velocity
utilization, reach drop, and wrist angular speed.

The corrected mainline profile is:

- `ori/main` task defaults:
  `position_cost=1`, `orientation_cost=1`, `lm_damping=0.01`,
  `damping=0.25`, and `posture_cost=0.01`
- standard Mink `ConfigurationLimit + VelocityLimit`
- physical IK velocity caps:
  `[2, 2, 3.14, 3.14, 6.3, 6.3, 6.3] rad/s`
- joint braking disabled
- measured-state safety disabled
- no frame-error, nullspace, singularity-approach, or kinetic-energy additions

All profiles used the same frozen Cartesian target and dynamic plant.

## Selected Case

`recorded_ep75_right_retract30_straight_retract` is the selected replacement
for the previous event-32 retract example.

| Result | Current PR | Corrected ori/main | PR without nullspace | Recorded hardware |
|---|---:|---:|---:|---:|
| Right-elbow lateral range | 4.0 cm | 15.5 cm | 17.6 cm | 4.9 cm |
| Most outward elbow y | -0.217 m | -0.330 m | -0.346 m | -0.225 m |
| Maximum position error | 4.4 cm | 5.3 cm | 3.1 cm | 7.6 cm |

The mainline and no-nullspace solutions move the elbow outward while the full
PR and recorded hardware preserve the compact branch. The mainline position
error remains close to the PR result, so the elbow difference is not an
artifact of a completely failed Cartesian trajectory.

Selected video:

`/home/hilab/workdir.evaluation/note/openarm_control/final_report/videos/showcase_selected_v3_retract_ep75_event30_corrected_ori_main_comparison.mp4`

## Repeated Evidence

The same behavior appears in independent windows:

| Scenario | Current PR | Corrected ori/main | Difference |
|---|---:|---:|---:|
| episode 75 event 31 | 6.7 cm | 17.6 cm | +11.0 cm |
| episode 75 event 57 | 7.5 cm | 17.5 cm | +10.0 cm |
| episode 79 event 38 | 2.7 cm | 12.4 cm | +9.8 cm |
| episode 85 event 58 | 6.8 cm | 10.9 cm | +4.1 cm |

Candidate videos use the
`showcase_retract_screen_v1_*_corrected_ori_main.mp4` naming pattern in the
final report video directory.
