# OpenArm VR IK Experiment and Asset Index

Language: **English** | [中文](experiment_index.zh.md) | [日本語](experiment_index.ja.md)

Date: 2026-08-01<br>
Decision brief: [README.md](README.md) · Technical report: [detailed_report.md](detailed_report.md) · Reproduction guide: [run_experiment.md](run_experiment.md) · Top-level manifest: [manifests/manifest.json](manifests/manifest.json)

Experiment scripts and frozen inputs are under [`src`](src/README.md). Local raw results used by the report are under `results/final_report_20260801`.

## 1. Fixed Experimental Conditions

| Condition | Value |
|---|---|
| Model | `openarm_mujoco/v2/cell.xml` |
| Model SHA-256 | `cb0322c264b2acd781ea08e873970ef21dd1db491e64ed0c02844aa8c1a3bfa7` |
| Coordinates | EEF target/error in `arm_origin`; plant integration in world frame |
| Outer period | `4 ms` |
| Substeps | `5 x 0.8 ms` |
| IK velocity limits | `[2, 2, 3.14, 3.14, 6.3, 6.3, 6.3] rad/s` |
| Driver velocity limits | `[2, 2, 3.14, 3.14, 6.3, 6.3, 6.3] rad/s` |
| Random sampling | Disabled; recorded seed `0` |

Source comparison uses upstream merge-base `d543cedeec5f` (tag `0.2.0`), and the evaluated PR revision is `f983a0eb5cae`. Experiments were generated from `d006ece506f2` plus recorded diff `0ef6a402...`; that diff was later committed unchanged as `f983a0eb5cae`, so runtime code and defaults are identical.

The complete PR-default parameters are stored in the top-level [manifest](manifests/manifest.json) and each suite metadata file. Suite metadata also records the revision, dirty-diff hash, generation command, dependency versions, model path and hash, controller profiles, and trajectory inventory.

An IK cap is a joint-velocity constraint inside the QP. A driver cap is per-joint clipping after the QP.

## 2. Controller Profile Naming

| Report name | Internal profile | Definition |
|---|---|---|
| PR default | `current_deployment` | PR mechanisms enabled by the evaluated code defaults |
| PR w/o IK velocity limits | `driver_only_velocity` | QP velocity envelope disabled; driver limiter retained |
| Mainline-task baseline | `strict_mainline` | Mainline task cost/damping/posture with the common recoverable envelope and `0.8 ms` substeps; shortened to Mainline baseline in the text |
| PR w/o posture regulation | `no_branch_regulation` | `nullspace_cost=0, posture_cost=0` |
| PR: full-home posture 0.01 | `full_home_replacement_0p01` | Exact-nullspace regulation disabled; `posture_cost=0.01` enabled |
| PR w/o position bound | `no_position_error_bound` | Position-error budget disabled |
| PR w/o orientation bound | `no_orientation_error_bound` | Orientation-error budget disabled |
| PR w/o 6D error bound | `no_frame_error_bounds` | Position- and orientation-error budgets both disabled |
| PR w/o singularity limit | `no_singularity_limit` | Singularity-approach limit disabled |
| PR w/o braking | `no_joint_braking` | Distance braking disabled; velocity envelope retained |
| PR w/o kinetic regularizer | `no_kinetic_regularization` | Kinetic-energy cost set to zero |

Mainline-task baseline is a structural control, not a byte-for-byte replay of the old commit. It uses `1/1` FrameTask cost, `0.25` damping, `0.01` LM damping, and `0.01` PostureTask, together with the same velocity limits, recoverable joint envelope, and `0.8 ms` substeps as PR default.

## 3. Target Trajectories

### 3.1 Diverse Test Trajectories

The screening suite uses a fixed comparison set of `42` trajectories. Other suites add `28` targeted trajectories, giving `70` unique targets after deduplication by `trajectory_sha256`. The catalog video selects `21` representative clips from these targets. Clip numbers below are their playback order.

The generation entry point, scenario name, suite usage, and hash for every one of the `70` targets are listed in the [trajectory catalog](src/TRAJECTORIES.md). Of these, `67` are generated deterministically, `2` come from frozen hardware commands, and `1` is a time-compressed version of the frozen fast-retract path. Procedural trajectories need no additional input `.npz` files.

| Action family / target type | Catalog clips (number and on-screen name) | Clips | Fixed set (42) | All unique targets (70) |
|---|---|---:|---:|---:|
| `reach` | 01 `Arm extension beyond reach: center`<br>02 `Arm extension beyond reach: lateral +`<br>03 `Arm extension beyond reach: lateral -` | 3 | 5 | 11 |
| `retract` | 04 `Fast diagonal retract: lateral +`<br>05 `Fast diagonal retract: lateral -` | 2 | 6 | 6 |
| `extended_translation` | 06 `Extended-arm circle` | 1 | 3 | 5 |
| `extended_axis` | 07 `Extended: up and return`<br>08 `Extended: down and return`<br>09 `Extended: left and return`<br>10 `Extended: right and return` | 4 | 8 | 8 |
| `extended_wrist` | 11 `Extended-arm wrist roll` | 1 | 3 | 5 |
| `normal_wrist` | 12 `Normal-workspace wrist roll`<br>13 `Near-chest wrist roll only` | 2 | 3 | 8 |
| `chest_wrist_outward` | 14 `Near-chest roll + forward translation`<br>15 `Near-chest roll + lateral translation`<br>16 `Near-chest roll + downward translation`<br>17 `Near-chest roll + diagonal translation` | 4 | 9 | 13 |
| `normal_workspace` | 18 `Normal-workspace motion`<br>19 `Bimanual workspace motion` | 2 | 5 | 7 |
| `frozen_near_chest_wrist_roll` | 20 `Recorded near-chest fast wrist-roll benchmark` | 1 | 0 | 1 |
| Recorded retract variants | 21 `Recorded fast-retract elbow-posture benchmark` | 1 | 0 | 2 |
| `joint_braking` | No catalog clip | 0 | 0 | 4 |
| **Total** | - | **21** | **42** | **70** |

The two `Recorded retract variants` are the original recording and the same path compressed by `2x`; the catalog plays only the original. The additional `28` targeted trajectories also contain left/right mirrors, additional speeds and directions, near-chest stress cases, `4` joint-braking targets, and one deep-start singularity trajectory whose start is shifted back by `0.10 m` while retaining the same farthest point.

Speed levels are selected from `0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8 m/s` and `2, 4, 6, 8, 10, 12 rad/s` as appropriate for each trajectory. The fixed `42` contain `40` right-arm targets and `2` bimanual targets.

The [21-action animated catalog](assets/video_previews/ideal_reference_trajectory_catalog.webp) is a visual preview only. Each segment is compressed to `2.5 s`; the upper-left overlay gives the playback multiplier, and the second line shows the family's count in the `42/70` target sets. This playback rate does not represent the speed of the `1,921` simulations.

### 3.2 Recorded-Command-Derived Benchmarks

| Benchmark | Source SHA-256 | Duration | Peak target speed | Evaluation focus |
|---|---|---:|---:|---|
| Near-chest fast wrist-roll | `915973...eb391` | 3.94 s | 0.027 m/s, 12.02 rad/s | Position/orientation tracking, joint acceleration, residual vibration |
| Fast-retract elbow-branch | `a0667f...53a19` | 2.92 s | 0.589 m/s, 6.25 rad/s | Elbow lateral range, nullspace branch consistency, Cartesian error, joint dynamics |

Both benchmarks are extracted from hardware recordings; each frame is a commanded pose relative to `arm_origin`. Absolute source paths and complete hashes are in the [top-level manifest](manifests/manifest.json). Every controller uses the same frame-by-frame target, MuJoCo plant, and driver velocity limits.

## 4. Experiment Suites and Run Counts

The full study covers `119` controller profiles and `70` unique targets. Each suite runs only the combinations relevant to its question rather than the full Cartesian product, for a total of `1,921` dynamic simulations.

| Suite | Metadata | Scenarios | Profiles | Runs | Primary purpose |
|---|---|---:|---:|---:|---|
| Boundary | [JSON](manifests/boundary/metadata.json) | 378 static conditions | 3 constraint forms | - | Feasibility of recovery from an out-of-bound state |
| Braking | [JSON](manifests/braking/metadata.json) | 4 | 6 | 24 | Braking distance and position margin |
| Candidates | [JSON](manifests/candidates/metadata.json) | 42 | 5 | 210 | Pareto trade-offs among combined parameter candidates |
| Chest | [JSON](manifests/chest/metadata.json) | 14 | 17 | 238 | Stress tests combining wrist rotation and translation |
| Driver | [JSON](manifests/driver/metadata.json) | 11 | 4 | 44 | PR default vs PR w/o IK velocity limits |
| Recorded benchmarks | [JSON](manifests/frozen/metadata.json) | 2 | 7 | 14 | Two recorded-command-derived targets |
| Accelerated fast-retract | [JSON](manifests/fast_retract_error_bound/metadata.json) | 1 | 2 | 2 | `2x` time-compressed fast-retract error-bound A/B |
| Deep-start singularity video | [JSON](manifests/singularity_video/metadata.json) | 1 | 2 | 2 | Singularity-limit A/B with the same farthest point and a start shifted back `0.10 m` |
| Nullspace sweep | [JSON](manifests/nullspace_sweep/metadata.json) | 1 | 44 | 44 | Cost / return-rate / maximum-speed sweep |
| Nullspace validation | [JSON](manifests/nullspace_validation/metadata.json) | 11 | 6 | 66 | Cross-scenario validation of candidates |
| Parameters | [JSON](manifests/parameters/metadata.json) | 11 | 54 | 594 | Single-parameter sensitivity |
| Robustness | [JSON](manifests/robustness/metadata.json) | 5 | 9 | 45 | Delay, gain, and measured-state robustness |
| Screening | [JSON](manifests/screening/metadata.json) | 42 | 14 | 588 | Overall systems and single-feature ablation |
| Symmetry | [JSON](manifests/symmetry/metadata.json) | 25 | 2 | 50 | Left/right mirrors and single/bimanual operation |

The study completes `1,921` controller-profile × target-trajectory dynamic simulations, all with `solver_failures=0`. Bimanual runs compute metrics for each side but count as one run.

## 5. Figure Index

| Figure | File | Data source | Description |
|---:|---|---|---|
| 01 | [VR IK control chain](assets/01_control_layers.png) | Schematic | Target, IK command, driver command, and actual state |
| 02 | [3D trajectory catalog](assets/02_trajectory_catalog.png) | Target generators | Representative `arm_origin` 3D target/state paths; not every speed or profile |
| 03 | [IK controllers](assets/03_headline_baseline_comparison.png) | screening | Mean performance of four controllers on the 42 test trajectories |
| 04 | [Single-feature ablation](assets/04_feature_ablation_heatmap.png) | screening | Percent change from PR default after removing one feature |
| 05 | [Feature effect by action family](assets/05_feature_effect_by_trajectory_family.png) | screening | Family breakdown of joint acceleration and elbow lateral range |
| 06 | [Fast wrist response](assets/06_chest_wrist_error_modulation_timeseries.png) | chest | `1.2 m/s + 8 rad/s` near-chest diagonal variant; four error/posture profiles |
| 07 | [Near-chest EEF paths](assets/07_chest_wrist_eef_paths.png) | chest | EEF paths for error-bound A/B |
| 08 | [Posture regulation](assets/08_nullspace_branch_control.png) | recorded benchmark | Three secondary-posture profiles on the same fast retract |
| 09 | [Singularity approach](assets/09_singularity_reach_timeseries.png) | screening | Blue extension, gray retraction, yellow slowdown interval |
| 10 | [Velocity limits inside/outside QP](assets/10_driver_limit_coupling_timeseries.png) | driver | Along-path lag, lateral deviation, and raw IK / actual J1 speed during the same fast retract |
| 11 | [Joint-limit recovery](assets/11_recoverable_joint_limit.png) | boundary | Solved-condition count and remaining out-of-bound angle after one `4 ms` outer solve |
| 12 | [Joint braking](assets/12_joint_braking_envelope.png) | braking | Braking distance, joint margin, and activation rate |
| 13 | [Single-parameter sweep](assets/13_parameter_sweep_summary.png) | parameters | Includes position/orientation budgets and velocity-limit scale |
| 14 | [Combined-parameter trade-offs](assets/14_combined_tuning_candidates.png) | candidates | A-D metric heatmap and Pareto scatter; green improves and red degrades |
| 15 | [Symmetry and robustness](assets/15_symmetry_and_robustness.png) | symmetry/robustness | Left/right mirrors and plant perturbations |
| 16 | [Solve time](assets/16_solver_timing.png) | screening | Mixed trajectories; each outer solve contains five substeps |
| 17 | [Nullspace sweep](assets/17_nullspace_targeted_tuning.png) | nullspace sweep | Blue frame shows PR-default absolute value; other cells show signed changes, green better and red worse |
| 18 | [Nullspace cross-validation](assets/18_nullspace_cross_validation.png) | nullspace validation | Candidate change from PR default across families, green better and red worse |
| 33 | [Representative case overview](assets/33_final_showcase_controller_comparison.png) | recorded benchmark | Summary of the two recorded-command-derived benchmarks |
| 34 | [Near-chest wrist benchmark](assets/34_chest_flip_benchmark_timeseries_and_path.png) | recorded benchmark | Position/orientation error, actual joint acceleration, and EEF path |
| 35 | [Fast-retract benchmark](assets/35_fast_retract_benchmark_timeseries_and_path.png) | recorded benchmark | Elbow Y-Z path, `y-y0`, EEF position error, and maximum absolute acceleration across seven joints |

## 6. Video Index

Every controller-comparison video shows the profile/parameters, `arm_origin` coordinates, playback multiplier, and legends for command/state/elbow paths. Red axes are the target; translucent arms are IK or driver commands; opaque arms are simulated actual states. WebP is the higher-resolution `8-12 fps` preview used by the reports; GIF is the compact compatibility version.

| Video | Type | Playback | Content |
|---|---|---:|---|
| [Reference-action catalog preview](assets/video_previews/ideal_reference_trajectory_catalog.webp) ([GIF fallback](assets/video_previews/ideal_reference_trajectory_catalog.gif), [MP4 download](videos/ideal_reference_trajectory_catalog.mp4?raw=1)) | Target catalog | Per segment | 21 representative clips with family counts in the 42/70 sets; not a controller A/B |
| [Near-chest controller comparison preview](assets/video_previews/near_chest_fast_wrist_roll_controller_comparison.webp) ([GIF fallback](assets/video_previews/near_chest_fast_wrist_roll_controller_comparison.gif), [MP4 download](videos/near_chest_fast_wrist_roll_controller_comparison.mp4?raw=1)) | One recorded-command-derived trajectory | 0.5x | PR default / Mainline baseline / PR w/o 6D error bound |
| [Fast-retract 6D error bound preview](assets/video_previews/fast_retract_frame_error_bound_comparison.webp) ([GIF fallback](assets/video_previews/fast_retract_frame_error_bound_comparison.gif), [MP4 download](videos/fast_retract_frame_error_bound_comparison.mp4?raw=1)) | `2x` time-compressed recorded path | 0.5x | PR default / PR w/o 6D error bound, two columns |
| [Near-chest roll with diagonal translation preview](assets/video_previews/near_chest_roll_translation_error_bound_comparison.webp) ([GIF fallback](assets/video_previews/near_chest_roll_translation_error_bound_comparison.gif), [MP4 download](videos/near_chest_roll_translation_error_bound_comparison.mp4?raw=1)) | One synthetic trajectory | 0.5x | PR default / PR w/o 6D error bound |
| [Fast-retract controller comparison preview](assets/video_previews/fast_retract_controller_comparison.webp) ([GIF fallback](assets/video_previews/fast_retract_controller_comparison.gif), [MP4 download](videos/fast_retract_controller_comparison.mp4?raw=1)) | One recorded-command-derived trajectory | 0.5x | PR default / Mainline baseline / PR w/o posture regulation, two views |
| [Posture-regulation comparison preview](assets/video_previews/fast_retract_posture_regulation_comparison.webp) ([GIF fallback](assets/video_previews/fast_retract_posture_regulation_comparison.gif), [MP4 download](videos/fast_retract_posture_regulation_comparison.mp4?raw=1)) | One recorded-command-derived trajectory | 0.5x | PR default / PR w/o posture regulation / PR: full-home posture 0.01 / 0.03 |
| [Nullspace parameters 2x2 preview](assets/video_previews/fast_retract_nullspace_parameter_comparison.webp) ([GIF fallback](assets/video_previews/fast_retract_nullspace_parameter_comparison.gif), [MP4 download](videos/fast_retract_nullspace_parameter_comparison.mp4?raw=1)) | One recorded-command-derived trajectory | 0.5x | PR default and three nullspace candidates |
| [IK velocity limits preview](assets/video_previews/fast_retract_ik_velocity_limit_comparison.webp) ([GIF fallback](assets/video_previews/fast_retract_ik_velocity_limit_comparison.gif), [MP4 download](videos/fast_retract_ik_velocity_limit_comparison.mp4?raw=1)) | One recorded-command-derived trajectory | 0.5x | PR default / PR w/o IK velocity limits |
| [Extension singularity preview](assets/video_previews/straight_reach_singularity_limit_comparison.webp) ([GIF fallback](assets/video_previews/straight_reach_singularity_limit_comparison.gif), [MP4 download](videos/straight_reach_singularity_limit_comparison.mp4?raw=1)) | Deep-start shoulder-height synthetic trajectory | 0.5x | Start shifted back `0.10 m`, same farthest point; singularity limit on/off; displays actual J1 acceleration |

## 7. CSV Index

| CSV | Content |
|---|---|
| [headline_baseline_means.csv](tables/headline_baseline_means.csv) | Aggregate values for four IK controllers |
| [ablation_relative_effects_percent.csv](tables/ablation_relative_effects_percent.csv) | Percent changes for single-feature ablations |
| [ablation_ddq_delta_by_family.csv](tables/ablation_ddq_delta_by_family.csv) | Joint-acceleration changes by action family |
| [ablation_elbow_delta_by_family_cm.csv](tables/ablation_elbow_delta_by_family_cm.csv) | Elbow changes by action family |
| [chest_profile_means.csv](tables/chest_profile_means.csv) | Aggregate over 14 near-chest targets |
| [near_chest_frozen_controller_metrics.csv](tables/near_chest_frozen_controller_metrics.csv) | Recorded near-chest target metrics |
| [fast_retract_frozen_controller_metrics.csv](tables/fast_retract_frozen_controller_metrics.csv) | Recorded fast-retract target metrics |
| [branch_regulation_frozen_metrics.csv](tables/branch_regulation_frozen_metrics.csv) | Secondary-posture comparison |
| [boundary_recovery_solved_counts.csv](tables/boundary_recovery_solved_counts.csv) | Solved conditions during joint-limit recovery |
| [braking_profile_means.csv](tables/braking_profile_means.csv) | Joint-braking aggregate |
| [driver_coupling_profile_means.csv](tables/driver_coupling_profile_means.csv) | Interaction between QP and driver velocity limits |
| [driver_path_tracking_components.csv](tables/driver_path_tracking_components.csv) | Maximum absolute spatial lag along the commanded direction and maximum lateral path deviation for `Fast diagonal retract: lateral +` |
| [parameter_profile_means.csv](tables/parameter_profile_means.csv) | All single-parameter profiles |
| [combined_candidate_definitions.csv](tables/combined_candidate_definitions.csv) | Parameter definitions for combined candidates A-D |
| [combined_candidate_means.csv](tables/combined_candidate_means.csv) | Absolute metrics for combined candidates |
| [combined_candidate_relative_percent.csv](tables/combined_candidate_relative_percent.csv) | Candidate changes relative to PR default |
| [nullspace_cost_return_grid.csv](tables/nullspace_cost_return_grid.csv) | Nullspace cost / return-rate grid |
| [nullspace_frozen_sweep_metrics.csv](tables/nullspace_frozen_sweep_metrics.csv) | 44 nullspace profiles on the recorded retract |
| [nullspace_cross_validation_deltas.csv](tables/nullspace_cross_validation_deltas.csv) | Candidate changes across action families |
| [exact_mirror_side_means.csv](tables/exact_mirror_side_means.csv) | Left/right mirror means |
| [robustness_profile_means.csv](tables/robustness_profile_means.csv) | Robustness means |
| [solver_timing_means.csv](tables/solver_timing_means.csv) | Mean and p95 solve times |
| [report_traceability.csv](tables/report_traceability.csv) | Mapping from public assets to suites, CSV files, scenarios, and profiles |
| [asset_checksums.sha256](tables/asset_checksums.sha256) | Checksums for final public assets |

## 8. Data Boundaries and Interpretation

- `solver_failures=0` means only that the solver returned successfully; it does not prove collision, torque, or every tracking metric is hardware-safe.
- Position and orientation RMSE are computed from the simulated actual state, not the raw IK command.
- Joint acceleration is a velocity difference at fixed `4 ms` sampling; p99 is more suitable than a one-frame maximum for cross-trajectory comparison.
- Relative MuJoCo A/B results cannot replace hardware validation of friction, compliance, and firmware position loops.
- Early hardware recordings lack raw and filtered VR targets, so only controller responses to the same final target can be compared.
- The posture-regulation figure and fast-retract controller figure share the same target. The former changes only the secondary posture task to isolate exact-nullspace regulation; the latter compares PR default, Mainline baseline, and PR w/o posture regulation to assess the complete PR against the Mainline-task baseline.
