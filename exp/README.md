# OpenArm VR IK Simulation Evaluation Brief

Language: **English** | [中文](README.zh.md) | [日本語](README.ja.md)

Date: 2026-08-01<br>
Upstream mainline baseline: `d543cedeec5f` (upstream `main` branch, tag `0.2.0`)<br>
Evaluated PR revision: `f983a0eb5cae` (the later clean commit of experiment snapshot `d006ece506f2` plus recorded diff `0ef6a402...`)<br>
Full materials: [technical report](detailed_report.md) · [experiment and asset index](experiment_index.md) · [reproduction guide](run_experiment.md) · [experiment manifest](manifests/manifest.json)

## Decision

**Use the current PR defaults as the deployment baseline for VR IK.** Across `42` diverse simulation trajectories, relative to the **Mainline-task baseline**, they reduce position RMSE by about `71%`, actual joint-acceleration p99 by about `49%`, elbow Cartesian-acceleration p99 by about `47%`, and end-of-motion end-effector (EEF) residual motion by about `75%`.

The main cost is transient orientation lag during fast wrist rotation. This is an explicit control trade-off: preserve the EEF position path, shoulder-elbow motion branch, and whole-arm stability instead of consuming an excessive rotation error in one QP solve.

## Compared Systems and PR Changes

The report distinguishes three objects:

| Object | Version or name | Purpose |
|---|---|---|
| Upstream mainline baseline | `d543cedeec5f` (tag `0.2.0`) | PR merge-base; documents upstream capabilities and source changes |
| Evaluated PR | `f983a0eb5cae` | PR source evaluated by this report |
| Mainline-task baseline | `strict_mainline` profile | Restores mainline task parameters inside the common substep timing and joint envelope; an algorithmic control, not a byte-for-byte replay of the old revision |

Upstream already provides MuJoCo- and Mink-based differential IK, 6D EEF tracking, `arm_origin` relative coordinates, damping, a full-home posture task, and basic position and velocity limits. The PR adds handling for large target errors, 7-DoF redundancy branches, singularity approach, and recovery from small joint-limit violations. See [Section 1](detailed_report.md#1-research-background-and-evaluation-scope) for context and [Section 2](detailed_report.md#2-pr-solver-architecture-and-new-mechanisms) for the algorithms.

| Mechanism | Recommendation | Primary role |
|---|---|---|
| 6D frame-error modulation | **Keep** | Prevent large position or orientation errors from redistributing motion into the shoulder and elbow under joint-speed saturation |
| Exact-nullspace home regulation | **Keep** | Regulate the elbow branch only along the 7-DoF arm's one-dimensional exact nullspace |
| One-sided singularity-approach limit | **Keep** | Slow only motion that further reduces geometric manipulability |
| QP joint-velocity envelope | **Keep** | Solve Cartesian and secondary objectives together within executable joint speeds |
| Recoverable joint envelope | **Keep** | Let a slightly out-of-bounds joint return within its velocity cap |
| Distance-dependent joint braking | **Keep; independently disable for diagnosis** | Reduce velocity toward a physical joint boundary before reaching it |
| Kinetic-energy regularization | **Keep at low weight** | Provide a weak preference among kinematically similar solutions |

## Experimental Design and Data

The evaluation distinguishes four states in the control chain:

- **Target pose**: EEF target submitted to IK after VR mapping;
- **Raw IK command**: joint configuration after one outer Mink solve;
- **Driver command**: IK command after the driver's per-joint velocity limiter;
- **Actual state**: dynamic MuJoCo plant response to the driver command.

Unless noted otherwise, tracking metrics compare the target pose with the **simulated actual state**. IK velocity limits act inside the QP; driver velocity limits clip each axis after the QP. Both command layers are recorded. A dedicated A/B in [Section 5.5](detailed_report.md#55-qp-vs-driver-velocity-limits) tests the effect of retaining only the driver limiter.

![Figure 1: VR IK control chain and the target, command, and actual states](assets/01_control_layers.png)

The experiments use `70` unique target trajectories. A fixed set of `42` supports the main controller comparison, feature ablations, and combined-parameter candidates; another `28` supports targeted tests. The report selects `21` representative clips from these trajectories for visual inspection.

| Experiment content | Composition | Use |
|---|---|---|
| Fixed comparison set | `42`: reach/extended `19`, retract `6`, wrist/chest `12`, normal workspace `5` | Identical inputs for the main systems, single-feature ablations, and combined candidates |
| Additional targeted trajectories | `28`, bringing the union to `70` unique targets | Left/right mirrors, additional speeds and directions, near-chest fast wrist rotation, joint braking, recorded-command replay, and deep-start singularity motion |
| Example-action video | `21` clips selected from the `70` targets | Shows target motion and representative responses; not a separate quantitative dataset |

Thirteen dynamic experiment groups execute `1,921` controller-profile × target-trajectory simulations across `119` controller profiles with no QP solver failures. A separate static suite contains `378` joint-boundary conditions. The action-family accounting, names of all 21 clips, and suite inventory are in the [experiment and asset index](experiment_index.md#3-target-trajectories). The [animated catalog preview](assets/video_previews/ideal_reference_trajectory_catalog.webp) provides a quick visual overview; the [original MP4](videos/ideal_reference_trajectory_catalog.mp4) is also available.

Two focal targets are derived from recorded hardware commands and replayed using a common target, MuJoCo plant, and driver velocity limits:

- **Near-chest fast wrist-roll**: fast wrist rotation near the chest with about `4.8 cm` of translation and `12.02 rad/s` peak angular speed;
- **Fast-retract elbow-branch**: right-arm retraction from a nearly extended posture with `0.589 m/s` peak linear speed.

These targets retain the frame-by-frame pose, speed, and direction changes of the recorded command. Their actual state and quantitative metrics still come from simulation and do not represent hardware closed-loop performance.

## Headline Results

All four systems use the same `42` trajectories. Metrics are computed per trajectory and then averaged; terminal EEF peak-to-peak motion uses the final `0.35 s`. Lower is better except that orientation error reflects an intentional trade-off.

| IK controller | Position RMSE ↓ | Orientation RMSE ↓ | Joint acceleration p99 ↓ | Elbow acceleration p99 ↓ | Elbow lateral range ↓ | Terminal EEF p2p ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **PR default** | **3.97 cm** | 31.27° | **29.43 rad/s²** | **5.46 m/s²** | 6.21 cm | **0.84 cm** |
| PR w/o IK velocity limits | 4.01 cm | 30.84° | 31.91 rad/s² | 6.22 m/s² | 6.50 cm | 0.89 cm |
| Mainline-task baseline | 13.78 cm | **11.13°** | 57.49 rad/s² | 10.21 m/s² | 6.12 cm | 3.37 cm |
| PR: full-home posture 0.01 | 3.94 cm | 29.03° | 33.42 rad/s² | 6.52 m/s² | 7.38 cm | 0.91 cm |

![Figure 2: Aggregate comparison of four IK controllers. PR default accepts some orientation lag to reduce position error, actual acceleration, and terminal residual motion](assets/03_headline_baseline_comparison.png)

The cross-trajectory mean elbow range cannot identify which nullspace branch was selected in one fast retract, so the targeted fast-retract case remains necessary.

## Feature Ablation

Each ablation below removes one feature from PR default. Values are relative changes; positive means that the metric increases. Since orientation lag is intentional, a reduction in a single metric does not necessarily mean a better overall controller.

![Figure 3: Single-feature ablation. The benefits of 6D error bounds, exact-nullspace regulation, and singularity limiting concentrate in different scenarios](assets/04_feature_ablation_heatmap.png)

Key observations:

- Removing the orientation-error bound raises position RMSE by `125%`, terminal EEF residual motion by `181%`, and driver-limit activation by `212%`;
- Removing exact-nullspace regulation raises elbow lateral range by `19%`, joint-acceleration p99 by `13%`, and driver-limit activation by `108%`;
- Removing the singularity limit raises joint-acceleration p99 by `12%`, mainly on reach and extended-arm trajectories;
- Joint braking and kinetic regularization have small effects on cross-trajectory means; their benefits concentrate near position boundaries or where several kinematic solutions have similar cost.

The family-level breakdown appears in [Section 4.2](detailed_report.md#42-single-feature-ablation).

## Three Representative Cases

### 1. Fast Wrist Rotation Near the Chest: 6D Error Modulation

The target translates only about `4.8 cm`, but reaches `12.02 rad/s` peak angular speed. The A/B result shows that aggressively tracking the full rotation error, together with joint-speed saturation, changes shoulder-elbow allocation and moves the EEF away from its intended position path.

| Controller | Position RMSE / maximum ↓ | Orientation RMSE | Joint acceleration p99 ↓ | Driver-limit activation ↓ |
|---|---:|---:|---:|---:|
| **PR default** | **1.28 / 1.78 cm** | 28.4° | **39.3 rad/s²** | **15.3%** |
| Mainline-task baseline | 5.81 / 17.11 cm | **14.5°** | 73.0 rad/s² | 29.3% |
| PR w/o 6D error bound | 1.79 / 4.81 cm | 22.0° | 47.3 rad/s² | 22.1% |

![Figure 4: Simulation replay derived from a recorded near-chest fast wrist roll. PR default limits maximum position deviation and actual joint acceleration at the cost of transient orientation lag](assets/34_chest_flip_benchmark_timeseries_and_path.png)

[View the animated three-controller preview](assets/video_previews/near_chest_fast_wrist_roll_controller_comparison.webp) · [Download the MP4](videos/near_chest_fast_wrist_roll_controller_comparison.mp4) · [See the near-chest stress-test A/B](detailed_report.md#511-near-chest-stress-test)

### 2. Fast Retract: Exact-Nullspace Branch Regulation

The exact-nullspace task regulates only the projection of home-configuration error onto the current one-dimensional nullspace. It does not apply the full-home preference directly to every joint direction.

| Secondary regularizer | Position RMSE / maximum | Elbow lateral range ↓ | Joint acceleration p99 | Driver-limit activation |
|---|---:|---:|---:|---:|
| **Exact nullspace (PR default)** | 1.80 / 4.38 cm | **4.04 cm** | 40.90 rad/s² | 36.9% |
| No posture regulation | 1.51 / 3.08 cm | 17.59 cm | 38.92 rad/s² | 33.7% |
| Full-home posture 0.01 | **1.49 / 3.00 cm** | 17.61 cm | **38.84 rad/s²** | **33.2%** |

![Figure 5: Elbow Y-Z path, lateral displacement, EEF position error, and actual joint acceleration during fast retract](assets/08_nullspace_branch_control.png)

For about `3 mm` additional position RMSE, exact-nullspace regulation reduces elbow lateral range by about `13.6 cm`. Full-home `PostureTask` produces about `17.6 cm` range at `posture_cost=0.003/0.01/0.03` and does not provide an equivalent branch constraint.

[Posture-regulation preview](assets/video_previews/fast_retract_posture_regulation_comparison.webp) · [Full-controller preview](assets/video_previews/fast_retract_controller_comparison.webp) · [MP4 files](experiment_index.md#6-video-index)

### 3. Extension Beyond Reach: Singularity-Approach Limiting

The singularity limit uses a dimensionless singular-value ratio determined only by the current geometric configuration:

$$
\rho(q)=\frac{\sigma_{\min}(J_{\mathrm{norm}})}{\sigma_{\max}(J_{\mathrm{norm}})}.
$$

The QP limits only joint motion that further decreases $\rho$; motion away from the singularity or tangent to an iso-singularity surface remains unrestricted. On the `0.8 m/s` shoulder-height extension, PR default increases minimum $\rho$ from `0.0047` to `0.0323` and lowers joint-acceleration p99 from `48.7` to `40.6 rad/s²`, with essentially unchanged position RMSE.

![Figure 6: The singularity-approach limit changes only the extension phase and releases automatically during retraction](assets/09_singularity_reach_timeseries.png)

[View the animated extension-singularity A/B](assets/video_previews/straight_reach_singularity_limit_comparison.webp) · [Download the MP4](videos/straight_reach_singularity_limit_comparison.mp4)

## Other Key Validation

| Question | Result | Conclusion |
|---|---|---|
| Do position and velocity constraints conflict after a small joint-limit violation? | Recoverable envelope solves `126/126` conditions; separate position and velocity constraints solve only `78/126` | The combined envelope repairs recovery feasibility |
| Can driver velocity limiting replace the QP velocity envelope? | On fast diagonal retract, disabling IK limits raises position RMSE from `3.93 cm` to `5.03 cm` and maximum lateral path deviation from `2.68 cm` to `9.43 cm` | In this A/B, QP limits preserve task allocation closer to the intended path |
| Does joint braking add physical boundary margin? | For a `12 rad/s` wrist target, a `0.20 rad` braking distance increases minimum joint margin from `26` to `62 mrad` | Adds boundary margin at the cost of earlier tracking lag |
| Does kinetic regularization provide a large dynamics benefit? | Removing it raises joint- and elbow-acceleration p99 by about `2.3%` and `3.6%` | It is a weak tie-breaker, not inverse dynamics or gravity compensation |

The complete timing and path decomposition for QP and driver limits is in [Section 5.5](detailed_report.md#55-qp-vs-driver-velocity-limits).

## Parameters and Deployment Recommendation

Neither the single-parameter sweeps, combined tuning, nor targeted exact-nullspace sweep finds a setting that simultaneously improves tracking, joint dynamics, elbow branch, and driver-limit activation across scenarios. The current defaults are not individually optimal on every trajectory, but provide a stable cross-scenario compromise.

| Parameter group | Current deployment value |
|---|---|
| EEF task / damping | `position_cost=12`, `orientation_cost=1.5`, `damping=0.1`, `lm_damping=0.01` |
| Outer timing | `4 ms`, `5` QP substeps |
| Total 6D error budget | `0.020 m / 0.25 rad`; position-speed interval `0.6 -> 0.9 m/s`; latch threshold `0.006 m` |
| Exact-nullspace regulation | cost `8.5`; return rate `1.6 s⁻¹`; maximum speed `1.0 rad/s`; activation `0.02 -> 0.05` |
| Singularity-approach limit | stop/slow `0.02 / 0.08`; maximum approach rate `0.25 s⁻¹` |
| Joint braking | distance `0.20 rad`; exponent `2`; measured-state buffer `0.01 rad` |
| Kinetic regularization | `2e-5` |
| IK / driver velocity limits J1-J7 | `[2, 2, 3.14, 3.14, 6.3, 6.3, 6.3] rad/s` |

See [Section 6.1](detailed_report.md#61-nullspace-parameter-sweep-and-cross-validation) for the exact-nullspace sweep and cross-validation, and [Section 6.2 onward](detailed_report.md#62-single-parameter-sensitivity) for other parameter curves and combined candidates.

## Current Limitations

- During fast wrist rotation near the chest with a sudden outward translation, some weakly controllable directions still excite large shoulder-elbow motion. Tightening the orientation budget protects position further but increases orientation lag.
- Measured joint positions conservatively affect only braking and singularity envelopes; they do not continuously overwrite Mink's integrated command. Command lead over actual state can still accumulate.
- The joint envelope does not provide environmental collision protection. Table and base clearance still require independent Cartesian or collision constraints.
- MuJoCo A/B supports relative comparison between controllers but cannot replace hardware validation with real friction, structural compliance, motor bandwidth, and firmware position loops.
- Early hardware recordings did not synchronously retain the raw VR target, filtered target, and final limited target, so a few compound wrist-rotation anomalies cannot be uniquely attributed.

## Reproducibility

- [Experiment reproduction guide](run_experiment.md): complete steps for environment setup, trajectories, simulation, figures, videos, and final validation;
- [Experiment source and commands](src/README.md): simulation, analysis, video generation, frozen inputs, and local raw results;
- [Full technical report](detailed_report.md): equations, experimental conditions, targeted results, parameter sweeps, performance, and API validation;
- [Experiment and asset index](experiment_index.md): mapping among figures, videos, CSV files, suites, and generation scripts;
- [Top-level experiment manifest](manifests/manifest.json): revision, dependencies, model hash, profiles, and scenario inventory;
- [Headline controller CSV](tables/headline_baseline_means.csv);
- [Single-feature ablation CSV](tables/ablation_relative_effects_percent.csv);
- [Fast-retract posture-regulation CSV](tables/branch_regulation_frozen_metrics.csv).
