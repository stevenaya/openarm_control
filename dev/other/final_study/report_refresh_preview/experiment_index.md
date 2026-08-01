# OpenArm IK 最终报告实验与资产索引

日期：2026-07-31  
简明报告：[README.md](README.md) · 技术报告：[detailed_report.md](detailed_report.md) · 顶层 manifest：[manifests/manifest.json](manifests/manifest.json)

## 1. 固定实验条件

| 条件 | 值 |
|---|---|
| Model | `openarm_mujoco/v2/cell.xml` |
| Model SHA-256 | `cb0322c264b2acd781ea08e873970ef21dd1db491e64ed0c02844aa8c1a3bfa7` |
| Coordinate | EEF target/error in `arm_origin`; plant in world |
| Outer period | `4 ms` |
| Substeps | `5 x 0.8 ms` |
| IK caps | `[2, 2, 3.14, 3.14, 6.3, 6.3, 6.3] rad/s` |
| Driver caps | `[2, 2, 3.14, 3.14, 12.6, 12.6, 12.6] rad/s` |
| Random sampling | disabled, seed `0` recorded |

Current deployment 的完整参数在 [manifest](manifests/manifest.json) 和各 suite metadata 中重复保存。Suite metadata 还记录代码 revision、dirty diff hash、生成命令、依赖版本、模型路径/hash、profiles 和 scenario inventory。

## 2. Controller profile 约定

| Public label | Internal profile | 定义 |
|---|---|---|
| Current deployment | `current_deployment` | 当前代码/dataflow 默认的全部 PR mechanisms |
| Driver cap only | `driver_only_velocity` | 关闭 IK velocity envelope；保留 driver cap |
| Strict ori/main | `strict_mainline` | 主线 task cost/damping/posture，统一当前 recoverable envelope 和 0.8 ms substeps |
| No branch | `no_branch_regulation` | `nullspace_cost=0, posture_cost=0` |
| Full-home 0.01 | `full_home_replacement_0p01` | 关闭 exact-nullspace，启用 `posture_cost=0.01` |
| No position bound | `no_position_error_bound` | position budget disabled |
| No orientation bound | `no_orientation_error_bound` | orientation budget disabled |
| No 6D bound | `no_frame_error_bounds` | both budgets disabled |
| No singularity limit | `no_singularity_limit` | singularity approach rate disabled |
| No braking | `no_joint_braking` | distance braking disabled, velocity envelope retained |
| No kinetic | `no_kinetic_regularization` | kinetic cost zero |

Strict ori/main 是结构对照，不是旧 commit 的逐行 replay。它使用 `1/1` FrameTask cost、`0.25` damping、`0.01` LM damping 和 `0.01` PostureTask，同时与 Current 共用 caps、recoverable position/velocity envelope 和每子步 `0.8 ms`。

## 3. Target 轨迹

### 3.1 广覆盖轨迹

Screening suite 包含 42 条独立 targets：

| Family | 数量 | 内容 |
|---|---:|---|
| `chest_wrist_outward` | 9 | 胸前 wrist roll 与不同方向/速度平移 |
| `extended_axis` | 8 | 伸直状态上下左右轴向移动 |
| `extended_translation` | 3 | 伸直状态平移组合 |
| `extended_wrist` | 3 | 伸直状态 wrist rotations |
| `normal_workspace` | 5 | 正常工作区单/双臂轨迹 |
| `normal_wrist` | 3 | 正常姿态 wrist rotations |
| `reach` | 5 | 同肩高度、左右偏置和不同速度伸直 |
| `retract` | 6 | 正前/斜向快速回缩 |

Speed inventory 为 `0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8 m/s` 与 `2, 4, 6, 8, 10, 12 rad/s` 中适用于各轨迹的档位。40 条为 right-arm target，2 条为 bimanual。其他 suites 使用该集合的子集、左右镜像、专门 stress targets 或冻结 recorded targets，因此最终有 68 个不同 target hashes。

[视频：21 个几何参考模式](videos/ideal_reference_trajectory_catalog.mp4) 只用于目录预览；每段压缩到 `2.5 s`，左上角显示倍率，不代表 1,939 runs 都具有相同播放速度。

### 3.2 冻结 benchmark

| Benchmark | 来源 SHA-256 | 时长 | 峰值 target speed | 评价重点 |
|---|---|---:|---:|---|
| Near-chest fast wrist-roll | `915973...eb391` | 3.94 s | 0.027 m/s, 12.02 rad/s | position/orientation tracking、ddq、残余振动 |
| Fast-retract elbow-branch | `a0667f...53a19` | 2.92 s | 0.589 m/s, 6.25 rad/s | elbow lateral branch、Cartesian error、joint dynamics |

绝对源路径和完整 hash 见 [顶层 manifest](manifests/manifest.json)。所有 profiles 使用同一逐帧 target、同一 MuJoCo plant 和同一 downstream driver caps。

## 4. 实验矩阵

| Suite | Metadata | Scenarios | Profiles | Runs | 主要用途 |
|---|---|---:|---:|---:|---|
| Boundary | [JSON](manifests/boundary/metadata.json) | 378 static cases | 3 formulations | - | 越界恢复可行性 |
| Braking | [JSON](manifests/braking/metadata.json) | 4 | 6 | 24 | braking distance 与 margin |
| Candidates | [JSON](manifests/candidates/metadata.json) | 42 | 5 | 210 | 组合参数 Pareto trade-off |
| Chest | [JSON](manifests/chest/metadata.json) | 14 | 17 | 238 | wrist-roll + translation stress |
| Driver | [JSON](manifests/driver/metadata.json) | 11 | 6 | 66 | IK/driver cap interaction |
| Frozen | [JSON](manifests/frozen/metadata.json) | 2 | 7 | 14 | 两条冻结 controller benchmark |
| Nullspace sweep | [JSON](manifests/nullspace_sweep/metadata.json) | 1 | 44 | 44 | cost/return/max sweep |
| Nullspace validation | [JSON](manifests/nullspace_validation/metadata.json) | 11 | 6 | 66 | 跨场景候选复验 |
| Parameters | [JSON](manifests/parameters/metadata.json) | 11 | 54 | 594 | 单参数 sensitivity |
| Robustness | [JSON](manifests/robustness/metadata.json) | 5 | 9 | 45 | delay/gain/state path |
| Screening | [JSON](manifests/screening/metadata.json) | 42 | 14 | 588 | headline 与 feature ablation |
| Symmetry | [JSON](manifests/symmetry/metadata.json) | 25 | 2 | 50 | 左右镜像和单/双臂 |

总计 `1,939` dynamic controller-target runs，逐侧指标 `2,045` rows，全部 `solver_failures=0`。

## 5. 图像索引

| 图 | 文件 | 数据来源 | 读法 |
|---:|---|---|---|
| 01 | [Control layers](assets/01_control_layers.png) | schematic | target -> IK command -> driver command -> actual state |
| 02 | [Trajectory catalog](assets/02_trajectory_catalog.png) | target generator | 21 个几何模式，不含全部速度/profile |
| 03 | [Controller profiles](assets/03_headline_baseline_comparison.png) | screening | 四个 headline profiles 跨 42 targets 平均 |
| 04 | [Single-feature ablation](assets/04_feature_ablation_heatmap.png) | screening | 只移除一项后相对 Current 的百分比 |
| 05 | [Feature by family](assets/05_feature_effect_by_trajectory_family.png) | screening | ddq/elbow range 的 family localization |
| 06 | [Near-chest stress](assets/06_chest_wrist_error_modulation_timeseries.png) | chest | suite 中代表性 `1.2 m/s + 8 rad/s` target |
| 07 | [Near-chest paths](assets/07_chest_wrist_eef_paths.png) | chest | error-bound A/B EEF path |
| 08 | [Branch regulation](assets/08_nullspace_branch_control.png) | frozen | exact / none / full-home 0.01，同一 retract target |
| 09 | [Singularity approach](assets/09_singularity_reach_timeseries.png) | screening | 蓝色伸直、灰色回缩、黄色 slowdown band |
| 10 | [IK/driver caps](assets/10_driver_limit_coupling_timeseries.png) | driver | actual 七轴 max speed/acceleration 和 command lead |
| 11 | [Joint-limit recovery](assets/11_recoverable_joint_limit.png) | boundary | solved count 与界外剩余角距离 |
| 12 | [Joint braking](assets/12_joint_braking_envelope.png) | braking | braking distance、margin 和激活率 |
| 13 | [Single parameters](assets/13_parameter_sweep_summary.png) | parameters | 含 position/orientation budgets 和 velocity scale |
| 14 | [Combined trade-offs](assets/14_combined_tuning_candidates.png) | candidates | A-D 精确参数、Pareto scatter 和 metric heatmap |
| 15 | [Symmetry/robustness](assets/15_symmetry_and_robustness.png) | symmetry/robustness | mirror side 与 plant perturbation |
| 16 | [Solver timing](assets/16_solver_timing.png) | screening | mixed trajectories，每 solve 含 5 substeps |
| 17 | [Nullspace sweep](assets/17_nullspace_targeted_tuning.png) | nullspace sweep | swivel、dynamic cost、frozen metrics |
| 18 | [Nullspace validation](assets/18_nullspace_cross_validation.png) | nullspace validation | 候选相对 Current 的跨 family 变化 |
| 33 | [Showcase overview](assets/33_final_showcase_controller_comparison.png) | frozen | 两条冻结 benchmark 摘要 |
| 34 | [Near-chest benchmark](assets/34_chest_flip_benchmark_timeseries_and_path.png) | frozen | position/orientation error、actual ddq、EEF path |
| 35 | [Fast-retract benchmark](assets/35_fast_retract_benchmark_timeseries_and_path.png) | frozen | `y-y0`、七轴最大绝对速度和 Y-Z branch path |

## 6. 视频索引

所有 controller 对照视频均显示 profile/参数、`arm_origin` 坐标、播放倍率和 command/state/elbow 轨迹图例。红色 transform 是 target；半透明 arm 是 IK/driver command ghost；不透明 arm 是 actual plant。

| 视频 | 类型 | 倍率 | 内容 |
|---|---|---:|---|
| [Reference catalog](videos/ideal_reference_trajectory_catalog.mp4) | target catalog | 分段显示 | 21 个几何模式；非 controller A/B |
| [Near-chest controller](videos/near_chest_fast_wrist_roll_controller_comparison.mp4) | single frozen trajectory | 0.5x | Current / strict main / no 6D bound |
| [Near-chest synthetic bound](videos/near_chest_roll_translation_error_bound_comparison.mp4) | single synthetic trajectory | 0.5x | Current / no 6D bound |
| [Fast-retract controller](videos/fast_retract_controller_comparison.mp4) | single frozen trajectory | 0.5x | Current / strict main / no branch，双视角 |
| [Branch regulator](videos/fast_retract_branch_regulator_comparison.mp4) | single frozen trajectory | 0.5x | exact / none / full-home 0.01 / 0.03 |
| [Nullspace 2x2](videos/fast_retract_nullspace_parameter_comparison.mp4) | single frozen trajectory | 0.5x | deployment 与 3 个 nullspace candidates |
| [IK velocity caps](videos/fast_retract_ik_velocity_limit_comparison.mp4) | single frozen trajectory | 0.5x | IK+driver / driver-only |
| [Singularity reach](videos/straight_reach_singularity_limit_comparison.mp4) | single synthetic trajectory | 0.5x | singularity limit on/off |

## 7. CSV 表索引

| CSV | 内容 |
|---|---|
| [headline_baseline_means.csv](tables/headline_baseline_means.csv) | 图 03 headline 数值 |
| [ablation_relative_effects_percent.csv](tables/ablation_relative_effects_percent.csv) | 图 04 百分比变化 |
| [ablation_ddq_delta_by_family.csv](tables/ablation_ddq_delta_by_family.csv) | 图 05 ddq family deltas |
| [ablation_elbow_delta_by_family_cm.csv](tables/ablation_elbow_delta_by_family_cm.csv) | 图 05 elbow family deltas |
| [chest_profile_means.csv](tables/chest_profile_means.csv) | near-chest 14-target aggregate |
| [near_chest_frozen_controller_metrics.csv](tables/near_chest_frozen_controller_metrics.csv) | 图 34 冻结胸前 metrics |
| [fast_retract_frozen_controller_metrics.csv](tables/fast_retract_frozen_controller_metrics.csv) | 图 35 冻结回缩 metrics |
| [branch_regulation_frozen_metrics.csv](tables/branch_regulation_frozen_metrics.csv) | 图 08 regulator-only 对照 |
| [boundary_recovery_solved_counts.csv](tables/boundary_recovery_solved_counts.csv) | 图 11 solved counts |
| [braking_profile_means.csv](tables/braking_profile_means.csv) | 图 12 braking aggregate |
| [driver_coupling_profile_means.csv](tables/driver_coupling_profile_means.csv) | 图 10 driver interaction |
| [parameter_profile_means.csv](tables/parameter_profile_means.csv) | 图 13 全部单参数 profiles |
| [combined_candidate_definitions.csv](tables/combined_candidate_definitions.csv) | 图 14 A-D 参数定义 |
| [combined_candidate_means.csv](tables/combined_candidate_means.csv) | 图 14 absolute metrics |
| [combined_candidate_relative_percent.csv](tables/combined_candidate_relative_percent.csv) | 图 14 相对 Current metrics |
| [nullspace_cost_return_grid.csv](tables/nullspace_cost_return_grid.csv) | 图 17 cost/return grid |
| [nullspace_frozen_sweep_metrics.csv](tables/nullspace_frozen_sweep_metrics.csv) | 图 17 冻结 44 profiles |
| [nullspace_cross_validation_deltas.csv](tables/nullspace_cross_validation_deltas.csv) | 图 18 跨 family deltas |
| [exact_mirror_side_means.csv](tables/exact_mirror_side_means.csv) | 图 15 左右镜像 means |
| [robustness_profile_means.csv](tables/robustness_profile_means.csv) | 图 15 robustness means |
| [solver_timing_means.csv](tables/solver_timing_means.csv) | 图 16 timing means/p95 |
| [report_traceability.csv](tables/report_traceability.csv) | 公开资产到 suite/CSV/scenario/profile 的映射 |
| [asset_checksums.sha256](tables/asset_checksums.sha256) | 最终报告公开资产 checksum |

## 8. 数据边界与解释限制

- `solver_failures=0` 只表示求解器返回成功，不表示碰撞、扭矩或所有 tracking 指标满足实机安全要求。
- Position/orientation RMSE 来自 actual plant，不是 raw IK command。
- Joint acceleration 是固定 `4 ms` 采样下的速度差分，p99 比单帧 max 更适合跨轨迹比较。
- MuJoCo 的相对 A/B 结果不能替代实机摩擦、柔性和 firmware loop 验证。
- 冻结 recorded target 并不包含早期记录中缺失的 raw/filtered VR stages；只能比较 controller 对同一最终 target 的响应。
- 图 08 与图 35 故意复用同一 fast-retract target：图 08 隔离 branch regulator，图 35 比较 controller-level structures。

## 9. 公开目录边界

`final_report` 仅保留三份 Markdown 实际引用的 figures、videos、summary CSV 和 manifests。旧报告与旧资产归档到相邻 `other_exp/report_refresh_pre_20260731`；完整逐帧 NPZ、raw metric rows 和生成脚本保留在 `openarm_control/dev/final_study`，不复制进公开报告目录。
