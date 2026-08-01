# OpenArm VR IK 实验与资产索引

语言：[English](experiment_index.md) | **中文** | [日本語](experiment_index.ja.md)

日期：2026-08-01<br>
决策简报：[README.zh.md](README.zh.md) · 详细报告：[detailed_report.zh.md](detailed_report.zh.md) · 复现指南：[run_experiment.zh.md](run_experiment.zh.md) · 顶层 manifest：[manifests/manifest.json](manifests/manifest.json)

实验脚本和冻结输入位于 [`src`](src/README.md)，报告使用的本地原始结果
位于 `results/final_report_20260801`。

## 1. 固定实验条件

| 条件 | 值 |
|---|---|
| 模型 | `openarm_mujoco/v2/cell.xml` |
| 模型 SHA-256 | `cb0322c264b2acd781ea08e873970ef21dd1db491e64ed0c02844aa8c1a3bfa7` |
| 坐标 | EEF 目标/误差使用 `arm_origin`；被控对象在 world frame 中积分 |
| 外层周期 | `4 ms` |
| 子步 | `5 x 0.8 ms` |
| IK 速度上限 | `[2, 2, 3.14, 3.14, 6.3, 6.3, 6.3] rad/s` |
| 驱动器速度上限 | `[2, 2, 3.14, 3.14, 6.3, 6.3, 6.3] rad/s` |
| 随机采样 | 关闭；记录 seed `0` |

代码比较以上游 merge-base `d543cedeec5f`（tag `0.2.0`）为基准，受测 PR revision 为 `f983a0eb5cae`。实验生成时使用 `d006ece506f2` 与已记录差异 `0ef6a402...`；该差异随后原样提交为 `f983a0eb5cae`，运行时代码与默认参数相同。

PR default 的完整参数同时保存在顶层 [manifest](manifests/manifest.json) 和各实验组 metadata 中。后者还记录 revision、dirty-diff hash、生成命令、依赖版本、模型路径与 hash、控制器配置和轨迹清单。

IK cap 指 QP 内的关节速度约束；driver cap 指 QP 后的逐关节裁剪。

## 2. 控制器配置约定

| 报告名称 | 内部 profile | 定义 |
|---|---|---|
| PR default | `current_deployment` | 受测代码默认启用的 PR 机制 |
| PR w/o IK velocity limits | `driver_only_velocity` | 关闭 QP 内速度包络；保留驱动器限速 |
| Mainline-task baseline | `strict_mainline` | 主线 task cost/damping/posture，并使用统一的可恢复包络和 `0.8 ms` 子步；正文简称 Mainline baseline |
| PR w/o posture regulation | `no_branch_regulation` | `nullspace_cost=0, posture_cost=0` |
| PR: full-home posture 0.01 | `full_home_replacement_0p01` | 关闭精确零空间正则，启用 `posture_cost=0.01` |
| PR w/o position bound | `no_position_error_bound` | 关闭位置误差预算 |
| PR w/o orientation bound | `no_orientation_error_bound` | 关闭朝向误差预算 |
| PR w/o 6D error bound | `no_frame_error_bounds` | 同时关闭位置和朝向误差预算 |
| PR w/o singularity limit | `no_singularity_limit` | 关闭奇异点接近限速 |
| PR w/o braking | `no_joint_braking` | 关闭距离制动，保留速度包络 |
| PR w/o kinetic regularizer | `no_kinetic_regularization` | 动能 cost 设为零 |

Mainline-task baseline 是结构对照，不是旧 commit 的原样回放。它使用 `1/1` FrameTask cost、`0.25` damping、`0.01` LM damping 和 `0.01` PostureTask，并与 PR default 共用速度上限、可恢复关节包络和 `0.8 ms` 子步。

## 3. 目标轨迹

### 3.1 多类型测试轨迹

Screening 实验组使用 `42` 条固定对比轨迹。其他实验组为专项问题增加 `28` 条轨迹；全部动态实验按 `trajectory_sha256` 去重后共有 `70` 条唯一目标。目录视频从这些目标中选取 `21` 个代表片段，下表给出动作组、轨迹数量和视频片段之间的对应关系。片段编号即目录视频的播放顺序。

全部 `70` 条目标的生成入口、场景名称、使用它们的实验组及轨迹 hash
见[轨迹生成目录](src/TRAJECTORIES.md)。其中 `67` 条由脚本
确定性生成，`2` 条来自冻结的实机指令，另 `1` 条是冻结快速回缩路径的
时间压缩版本；无需为程序化轨迹额外保存输入 `.npz`。

| 动作组/目标类型 | 目录视频片段（编号、画面名称） | 片段数 | 固定对比轨迹（42） | 全部唯一目标（70） |
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
| `joint_braking` | 无目录片段 | 0 | 0 | 4 |
| **合计** | - | **21** | **42** | **70** |

`Recorded retract variants` 的两条目标分别为原始记录和相同路径的 `2x` 时间压缩版本；目录只播放原始版本。额外的 `28` 条专项轨迹还包括左右镜像、额外速度与方向、胸前压力测试、`4` 条关节制动目标，以及一条最远点不变、起点后移 `0.10 m` 的深起点奇异轨迹。

速度档位取自 `0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8 m/s` 和 `2, 4, 6, 8, 10, 12 rad/s`，各轨迹使用其中适用的组合。`42` 条固定对比轨迹含 `40` 条右臂目标和 `2` 条双臂目标。

[视频：21 个几何参考动作](videos/ideal_reference_trajectory_catalog.mp4)仅用于目录预览。每段压缩至 `2.5 s`；左上角显示倍率，第二行显示所属动作族在 `42/70` 条目标中的数量。该播放速度不代表 `1,921` 次仿真的实际速度。

### 3.2 实机记录派生基准

| 基准 | 来源 SHA-256 | 时长 | 峰值目标速度 | 评价重点 |
|---|---|---:|---:|---|
| Near-chest fast wrist-roll | `915973...eb391` | 3.94 s | 0.027 m/s, 12.02 rad/s | 位置/朝向跟踪、关节加速度、残余振动 |
| Fast-retract elbow-branch | `a0667f...53a19` | 2.92 s | 0.589 m/s, 6.25 rad/s | 肘横向范围、零空间分支一致性、笛卡尔误差、关节动态 |

两条基准均从实机记录提取，每帧目标为相对 `arm_origin` 的指令位姿。绝对源路径和完整 hash 见[顶层 manifest](manifests/manifest.json)。所有配置使用相同的逐帧目标、MuJoCo 被控对象和驱动器速度上限。

## 4. 实验组与运行次数

全部实验覆盖 `119` 种控制器配置和 `70` 条唯一目标。各实验组只运行与其问题相关的组合，并非完整笛卡尔积，因此共完成 `1,921` 次动态仿真。

| 实验组 | Metadata | 场景数 | 配置数 | 运行数 | 主要用途 |
|---|---|---:|---:|---:|---|
| Boundary | [JSON](manifests/boundary/metadata.json) | 378 个静态条件 | 3 种约束形式 | - | 越界恢复可行性 |
| Braking | [JSON](manifests/braking/metadata.json) | 4 | 6 | 24 | 制动距离与位置余量 |
| Candidates | [JSON](manifests/candidates/metadata.json) | 42 | 5 | 210 | 组合参数 Pareto 取舍 |
| Chest | [JSON](manifests/chest/metadata.json) | 14 | 17 | 238 | 翻腕伴随平移的压力测试 |
| Driver | [JSON](manifests/driver/metadata.json) | 11 | 4 | 44 | PR default 与 PR w/o IK velocity limits 的差异 |
| Recorded benchmarks | [JSON](manifests/frozen/metadata.json) | 2 | 7 | 14 | 两条实机记录派生基准 |
| Accelerated fast-retract | [JSON](manifests/fast_retract_error_bound/metadata.json) | 1 | 2 | 2 | 快速回缩指令路径的 `2x` 时间压缩误差上限 A/B |
| Deep-start singularity video | [JSON](manifests/singularity_video/metadata.json) | 1 | 2 | 2 | 同一最远点、起点后移 `0.10 m` 的奇异点限速 A/B 视频 |
| Nullspace sweep | [JSON](manifests/nullspace_sweep/metadata.json) | 1 | 44 | 44 | cost/回正率/最大速度扫描 |
| Nullspace validation | [JSON](manifests/nullspace_validation/metadata.json) | 11 | 6 | 66 | 跨场景候选复验 |
| Parameters | [JSON](manifests/parameters/metadata.json) | 11 | 54 | 594 | 单参数敏感性 |
| Robustness | [JSON](manifests/robustness/metadata.json) | 5 | 9 | 45 | 延迟、增益和状态链路鲁棒性 |
| Screening | [JSON](manifests/screening/metadata.json) | 42 | 14 | 588 | 整体方案与单功能消融 |
| Symmetry | [JSON](manifests/symmetry/metadata.json) | 25 | 2 | 50 | 左右镜像和单/双臂 |

共完成 `1,921` 次“控制器配置 × 目标轨迹”动态仿真，全部为 `solver_failures=0`。双臂运行分别计算左右臂指标，但不重复计数。

## 5. 图像索引

| 图 | 文件 | 数据来源 | 说明 |
|---:|---|---|---|
| 01 | [VR IK 控制链](assets/01_control_layers.png) | 示意图 | 目标、IK 指令、驱动器指令与实际状态 |
| 02 | [3D 轨迹目录](assets/02_trajectory_catalog.png) | 目标生成器 | 典型动作的 `arm_origin` 3D 目标/状态路径；不含全部速度和配置 |
| 03 | [IK 控制方案](assets/03_headline_baseline_comparison.png) | screening | 四种 IK 控制方案在 42 条测试轨迹上的平均表现 |
| 04 | [单功能消融](assets/04_feature_ablation_heatmap.png) | screening | 每次移除一项后相对 PR default 的百分比变化 |
| 05 | [按动作族分解功能收益](assets/05_feature_effect_by_trajectory_family.png) | screening | 关节加速度和肘横向范围的动作族分布 |
| 06 | [快速翻腕响应](assets/06_chest_wrist_error_modulation_timeseries.png) | chest | `Near-chest roll + diagonal translation` 的 `1.2 m/s + 8 rad/s` 变体；四种误差/姿态配置对照 |
| 07 | [胸前 EEF 路径](assets/07_chest_wrist_eef_paths.png) | chest | 误差上限 A/B 的 EEF 路径 |
| 08 | [姿态正则](assets/08_nullspace_branch_control.png) | recorded benchmark | 同一快速回缩目标下的三种次级姿态配置 |
| 09 | [奇异点接近](assets/09_singularity_reach_timeseries.png) | screening | 蓝色为伸直，灰色为回缩，黄色为减速区间 |
| 10 | [QP 内外速度限制](assets/10_driver_limit_coupling_timeseries.png) | driver | 同一快速回缩下的沿轨迹滞后、横向偏离，以及 raw IK/实际 J1 速度 |
| 11 | [关节越界恢复](assets/11_recoverable_joint_limit.png) | boundary | 可解条件数，以及一次 `4 ms` 外层求解后的界外剩余角距离 |
| 12 | [关节制动](assets/12_joint_braking_envelope.png) | braking | 制动距离、关节余量和激活率 |
| 13 | [单参数扫描](assets/13_parameter_sweep_summary.png) | parameters | 含位置/朝向预算和速度上限比例 |
| 14 | [组合参数取舍](assets/14_combined_tuning_candidates.png) | candidates | 上方为 A-D 相对 PR default 的指标热图，下方为 Pareto 散点；绿色改善、红色恶化 |
| 15 | [镜像与鲁棒性](assets/15_symmetry_and_robustness.png) | symmetry/robustness | 左右镜像和被控对象扰动 |
| 16 | [求解耗时](assets/16_solver_timing.png) | screening | 混合轨迹；每次外层求解含 5 个子步 |
| 17 | [零空间参数扫描](assets/17_nullspace_targeted_tuning.png) | nullspace sweep | 蓝框显示 PR default 绝对值，其余格显示相对 PR default 的有符号变化；绿色改善，红色恶化 |
| 18 | [零空间交叉验证](assets/18_nullspace_cross_validation.png) | nullspace validation | 候选相对 PR default 的跨动作族变化；绿色改善，红色恶化 |
| 33 | [代表案例总览](assets/33_final_showcase_controller_comparison.png) | recorded benchmark | 两条实机记录派生基准摘要 |
| 34 | [胸前翻腕基准](assets/34_chest_flip_benchmark_timeseries_and_path.png) | recorded benchmark | 位置/朝向误差、实际关节加速度和 EEF 路径 |
| 35 | [快速回缩基准](assets/35_fast_retract_benchmark_timeseries_and_path.png) | recorded benchmark | Y-Z 肘部路径、`y-y0`、EEF 位置误差和七轴最大绝对加速度 |

## 6. 视频索引

所有控制器对照视频均显示 profile/参数、`arm_origin` 坐标、播放倍率及指令/状态/肘部轨迹图例。红色坐标系表示目标，半透明机械臂表示 IK 或驱动器指令，不透明机械臂表示仿真实际状态。

| 视频 | 类型 | 倍率 | 内容 |
|---|---|---:|---|
| [参考动作目录预览](assets/video_previews/ideal_reference_trajectory_catalog.gif) ([MP4](videos/ideal_reference_trajectory_catalog.mp4)) | 目标目录 | 分段显示 | 21 个代表片段；标明动作族在 42/70 条目标中的数量；非控制器 A/B |
| [胸前控制器对照预览](assets/video_previews/near_chest_fast_wrist_roll_controller_comparison.gif) ([MP4](videos/near_chest_fast_wrist_roll_controller_comparison.mp4)) | 单条记录派生轨迹 | 0.5x | PR default / Mainline baseline / PR w/o 6D error bound |
| [快速回缩 6D 误差上限预览](assets/video_previews/fast_retract_frame_error_bound_comparison.gif) ([MP4](videos/fast_retract_frame_error_bound_comparison.mp4)) | `2x` 时间压缩的记录派生路径 | 0.5x | PR default / PR w/o 6D error bound，双栏 |
| [胸前翻腕伴随斜向平移预览](assets/video_previews/near_chest_roll_translation_error_bound_comparison.gif) ([MP4](videos/near_chest_roll_translation_error_bound_comparison.mp4)) | 单条合成轨迹 | 0.5x | PR default / PR w/o 6D error bound |
| [快速回缩控制器对照预览](assets/video_previews/fast_retract_controller_comparison.gif) ([MP4](videos/fast_retract_controller_comparison.mp4)) | 单条记录派生轨迹 | 0.5x | PR default / Mainline baseline / PR w/o posture regulation，双视角 |
| [姿态正则对照预览](assets/video_previews/fast_retract_posture_regulation_comparison.gif) ([MP4](videos/fast_retract_posture_regulation_comparison.mp4)) | 单条记录派生轨迹 | 0.5x | PR default / PR w/o posture regulation / PR: full-home posture 0.01 / 0.03 |
| [零空间参数 2x2预览](assets/video_previews/fast_retract_nullspace_parameter_comparison.gif) ([MP4](videos/fast_retract_nullspace_parameter_comparison.mp4)) | 单条记录派生轨迹 | 0.5x | PR default 与 3 个零空间候选 |
| [IK 速度上限预览](assets/video_previews/fast_retract_ik_velocity_limit_comparison.gif) ([MP4](videos/fast_retract_ik_velocity_limit_comparison.mp4)) | 单条记录派生轨迹 | 0.5x | PR default / PR w/o IK velocity limits |
| [伸直奇异点预览](assets/video_previews/straight_reach_singularity_limit_comparison.gif) ([MP4](videos/straight_reach_singularity_limit_comparison.mp4)) | 深起点肩高合成轨迹 | 0.5x | 起点后移 `0.10 m`、最远点不变；奇异点限速开/关；显示实际 J1 加速度 |

## 7. CSV 表索引

| CSV | 内容 |
|---|---|
| [headline_baseline_means.csv](tables/headline_baseline_means.csv) | 四种 IK 控制方案的整体对比数值 |
| [ablation_relative_effects_percent.csv](tables/ablation_relative_effects_percent.csv) | 单功能消融百分比变化 |
| [ablation_ddq_delta_by_family.csv](tables/ablation_ddq_delta_by_family.csv) | 按动作族分解的关节加速度变化 |
| [ablation_elbow_delta_by_family_cm.csv](tables/ablation_elbow_delta_by_family_cm.csv) | 按动作族分解的肘部变化 |
| [chest_profile_means.csv](tables/chest_profile_means.csv) | 胸前 14 条目标汇总 |
| [near_chest_frozen_controller_metrics.csv](tables/near_chest_frozen_controller_metrics.csv) | 胸前记录派生轨迹指标 |
| [fast_retract_frozen_controller_metrics.csv](tables/fast_retract_frozen_controller_metrics.csv) | 快速回缩记录派生轨迹指标 |
| [branch_regulation_frozen_metrics.csv](tables/branch_regulation_frozen_metrics.csv) | 次级姿态正则对照 |
| [boundary_recovery_solved_counts.csv](tables/boundary_recovery_solved_counts.csv) | 关节越界恢复可解条件数 |
| [braking_profile_means.csv](tables/braking_profile_means.csv) | 制动实验汇总 |
| [driver_coupling_profile_means.csv](tables/driver_coupling_profile_means.csv) | QP 与驱动器限速交互 |
| [driver_path_tracking_components.csv](tables/driver_path_tracking_components.csv) | `Fast diagonal retract: lateral +` 沿指令前进方向的最大绝对空间滞后与最大横向路径偏离 |
| [parameter_profile_means.csv](tables/parameter_profile_means.csv) | 全部单参数配置 |
| [combined_candidate_definitions.csv](tables/combined_candidate_definitions.csv) | 组合候选 A-D 参数定义 |
| [combined_candidate_means.csv](tables/combined_candidate_means.csv) | 组合候选绝对指标 |
| [combined_candidate_relative_percent.csv](tables/combined_candidate_relative_percent.csv) | 组合候选相对 PR default 的变化 |
| [nullspace_cost_return_grid.csv](tables/nullspace_cost_return_grid.csv) | 零空间 cost/回正率网格 |
| [nullspace_frozen_sweep_metrics.csv](tables/nullspace_frozen_sweep_metrics.csv) | 记录派生回缩轨迹的 44 个零空间配置 |
| [nullspace_cross_validation_deltas.csv](tables/nullspace_cross_validation_deltas.csv) | 零空间候选的跨动作族变化 |
| [exact_mirror_side_means.csv](tables/exact_mirror_side_means.csv) | 左右镜像均值 |
| [robustness_profile_means.csv](tables/robustness_profile_means.csv) | 鲁棒性实验均值 |
| [solver_timing_means.csv](tables/solver_timing_means.csv) | 求解耗时均值与 p95 |
| [report_traceability.csv](tables/report_traceability.csv) | 公开资产到实验组、CSV、场景和配置的映射 |
| [asset_checksums.sha256](tables/asset_checksums.sha256) | 最终公开资产校验和 |

## 8. 数据边界与解释限制

- `solver_failures=0` 仅表示求解器成功返回，不代表碰撞、扭矩或所有跟踪指标满足实机安全要求。
- 位置和朝向 RMSE 来自仿真实际状态，不是 raw IK command。
- 关节加速度由固定 `4 ms` 采样下的速度差分得到；p99 比单帧最大值更适合跨轨迹比较。
- MuJoCo 的相对 A/B 结果不能替代实机摩擦、柔性和固件位置环验证。
- 早期实机记录缺少原始/滤波后 VR 目标，只能比较控制器对同一最终目标的响应。
- 姿态正则图与快速回缩控制器图共用同一目标。前者只改变次级姿态 task，用于隔离精确零空间正则；后者比较 PR default、Mainline baseline 和 PR w/o posture regulation，用于评估完整 PR 相对 Mainline-task baseline 的表现。
