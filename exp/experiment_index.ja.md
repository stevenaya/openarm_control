# OpenArm VR IK 実験・アセット索引

言語：[English](experiment_index.md) | [中文](experiment_index.zh.md) | **日本語**

日付：2026-08-01<br>
判断概要：[README.ja.md](README.ja.md) · 詳細レポート：[detailed_report.ja.md](detailed_report.ja.md) · 再現手順：[run_experiment.ja.md](run_experiment.ja.md) · Top-level manifest：[manifests/manifest.json](manifests/manifest.json)

実験 script と frozen input は [`src`](src/README.md) にあり、レポートで使用した local raw result は `results/final_report_20260801` にある。

## 1. 固定実験条件

| 条件 | 値 |
|---|---|
| Model | `openarm_mujoco/v2/cell.xml` |
| Model SHA-256 | `cb0322c264b2acd781ea08e873970ef21dd1db491e64ed0c02844aa8c1a3bfa7` |
| 座標 | EEF target/error は `arm_origin`、plant integration は world frame |
| Outer period | `4 ms` |
| Substep | `5 x 0.8 ms` |
| IK velocity limit | `[2, 2, 3.14, 3.14, 6.3, 6.3, 6.3] rad/s` |
| Driver velocity limit | `[2, 2, 3.14, 3.14, 6.3, 6.3, 6.3] rad/s` |
| Random sampling | 無効。記録 seed `0` |

ソース比較の上流 merge-base は `d543cedeec5f`（tag `0.2.0`）、評価対象 PR revision は `f983a0eb5cae` である。実験は `d006ece506f2` と記録済み差分 `0ef6a402...` から生成され、この差分は後にそのまま `f983a0eb5cae` として commit されたため、runtime code と default parameter は同一である。

PR default の全 parameter は top-level [manifest](manifests/manifest.json) と各 suite metadata に保存される。Suite metadata には revision、dirty-diff hash、生成 command、dependency version、model path と hash、controller profile、trajectory inventory も記録される。

IK cap は QP 内の joint-velocity constraint、driver cap は QP 後の関節別 clipping を指す。

## 2. Controller profile の名称

| レポート名 | Internal profile | 定義 |
|---|---|---|
| PR default | `current_deployment` | 評価対象 code の default で有効な PR mechanism |
| PR w/o IK velocity limits | `driver_only_velocity` | QP velocity envelope を無効化し、driver limiter を維持 |
| Mainline-task baseline | `strict_mainline` | Mainline の task cost/damping/posture と共通 recoverable envelope、`0.8 ms` substep。本文では Mainline baseline と略記 |
| PR w/o posture regulation | `no_branch_regulation` | `nullspace_cost=0, posture_cost=0` |
| PR: full-home posture 0.01 | `full_home_replacement_0p01` | Exact-nullspace regulation を無効化し、`posture_cost=0.01` を有効化 |
| PR w/o position bound | `no_position_error_bound` | Position-error budget を無効化 |
| PR w/o orientation bound | `no_orientation_error_bound` | Orientation-error budget を無効化 |
| PR w/o 6D error bound | `no_frame_error_bounds` | Position と orientation の両 budget を無効化 |
| PR w/o singularity limit | `no_singularity_limit` | Singularity-approach limit を無効化 |
| PR w/o braking | `no_joint_braking` | Distance braking を無効化し、velocity envelope を維持 |
| PR w/o kinetic regularizer | `no_kinetic_regularization` | Kinetic-energy cost を 0 に設定 |

Mainline-task baseline は構造比較であり、旧 commit の完全 replay ではない。`1/1` FrameTask cost、`0.25` damping、`0.01` LM damping、`0.01` PostureTask を使用し、PR default と同じ velocity limit、recoverable joint envelope、`0.8 ms` substep を共有する。

## 3. 目標軌道

### 3.1 多様な試験軌道

Screening suite は固定比較用の `42` trajectory を使用する。他 suite は個別問題向けに `28` trajectory を追加し、`trajectory_sha256` で重複除去すると `70` unique target になる。Catalog video はその中から `21` representative clip を選択する。下表の clip number は再生順である。

全 `70` target の生成 entry point、scenario name、使用 suite、hash は[軌道 catalog](src/TRAJECTORIES.md)に記載する。`67` 本は決定論的に生成され、`2` 本は frozen hardware command、`1` 本は frozen fast-retract path の時間圧縮版である。Procedural trajectory に追加 `.npz` input は不要。

| Action family / target type | Catalog clip（番号、画面名） | Clip 数 | 固定セット (42) | 全 unique target (70) |
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
| `joint_braking` | Catalog clip なし | 0 | 0 | 4 |
| **合計** | - | **21** | **42** | **70** |

`Recorded retract variants` は元の記録と同じ path の `2x` 時間圧縮版で、catalog は元の版だけを再生する。追加 `28` trajectory には左右 mirror、追加速度・方向、胸前 stress case、`4` joint-braking target、最遠点を維持したまま start を `0.10 m` 後退させた deep-start singularity trajectory も含む。

速度 level は trajectory に応じて `0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8 m/s` および `2, 4, 6, 8, 10, 12 rad/s` から選ぶ。固定 `42` 本には右腕 target `40` 本と bimanual target `2` 本が含まれる。

[21-action animated catalog](assets/video_previews/ideal_reference_trajectory_catalog.webp)は visual preview 専用である。各 segment は `2.5 s` に圧縮され、左上に playback multiplier、2 行目にその family の `42/70` target 内の本数を表示する。この再生速度は `1,921` simulation の実速度ではない。

### 3.2 実機 command 由来 benchmark

| Benchmark | Source SHA-256 | Duration | Peak target speed | 評価項目 |
|---|---|---:|---:|---|
| Near-chest fast wrist-roll | `915973...eb391` | 3.94 s | 0.027 m/s, 12.02 rad/s | Position/orientation tracking、joint acceleration、residual vibration |
| Fast-retract elbow-branch | `a0667f...53a19` | 2.92 s | 0.589 m/s, 6.25 rad/s | Elbow lateral range、nullspace branch consistency、Cartesian error、joint dynamics |

両 benchmark は実機記録から抽出され、各 frame は `arm_origin` 相対の command pose である。絶対 source path と完全 hash は [top-level manifest](manifests/manifest.json) にある。全 controller は同一の frame-by-frame target、MuJoCo plant、driver velocity limit を使用する。

## 4. 実験 suite と run 数

全 study は `119` controller profile と `70` unique target を対象とする。各 suite は問題に必要な組み合わせのみを実行し、全 Cartesian product ではないため、動的 simulation は合計 `1,921` 回となる。

| Suite | Metadata | Scenario | Profile | Run | 主な用途 |
|---|---|---:|---:|---:|---|
| Boundary | [JSON](manifests/boundary/metadata.json) | 378 static condition | 3 constraint form | - | 限界超過状態からの復帰可能性 |
| Braking | [JSON](manifests/braking/metadata.json) | 4 | 6 | 24 | Braking distance と position margin |
| Candidates | [JSON](manifests/candidates/metadata.json) | 42 | 5 | 210 | 組み合わせ parameter candidate の Pareto trade-off |
| Chest | [JSON](manifests/chest/metadata.json) | 14 | 17 | 238 | 手首回転と並進を組み合わせた stress test |
| Driver | [JSON](manifests/driver/metadata.json) | 11 | 4 | 44 | PR default と PR w/o IK velocity limits |
| Recorded benchmarks | [JSON](manifests/frozen/metadata.json) | 2 | 7 | 14 | 2 本の記録 command 由来 target |
| Accelerated fast-retract | [JSON](manifests/fast_retract_error_bound/metadata.json) | 1 | 2 | 2 | `2x` 時間圧縮 fast-retract error-bound A/B |
| Deep-start singularity video | [JSON](manifests/singularity_video/metadata.json) | 1 | 2 | 2 | 同じ最遠点、start を `0.10 m` 後退させた singularity-limit A/B |
| Nullspace sweep | [JSON](manifests/nullspace_sweep/metadata.json) | 1 | 44 | 44 | Cost / return-rate / maximum-speed sweep |
| Nullspace validation | [JSON](manifests/nullspace_validation/metadata.json) | 11 | 6 | 66 | Candidate の cross-scenario validation |
| Parameters | [JSON](manifests/parameters/metadata.json) | 11 | 54 | 594 | Single-parameter sensitivity |
| Robustness | [JSON](manifests/robustness/metadata.json) | 5 | 9 | 45 | Delay、gain、measured-state robustness |
| Screening | [JSON](manifests/screening/metadata.json) | 42 | 14 | 588 | Controller 全体と single-feature ablation |
| Symmetry | [JSON](manifests/symmetry/metadata.json) | 25 | 2 | 50 | 左右 mirror と single/bimanual operation |

`1,921` 回の controller-profile × target-trajectory dynamic simulation はすべて `solver_failures=0` だった。Bimanual run は左右の metric を別々に算出するが、run 数は重複計上しない。

## 5. 図索引

| 図 | File | Data source | 説明 |
|---:|---|---|---|
| 01 | [VR IK control chain](assets/01_control_layers.png) | Schematic | Target、IK command、driver command、actual state |
| 02 | [3D trajectory catalog](assets/02_trajectory_catalog.png) | Target generator | 代表的な `arm_origin` 3D target/state path。全速度・profile ではない |
| 03 | [IK controller](assets/03_headline_baseline_comparison.png) | screening | 42 test trajectory 上の 4 controller 平均性能 |
| 04 | [Single-feature ablation](assets/04_feature_ablation_heatmap.png) | screening | 機能を 1 つ外した際の PR default からの変化率 |
| 05 | [Action family 別機能効果](assets/05_feature_effect_by_trajectory_family.png) | screening | Joint acceleration と elbow lateral range の family 別内訳 |
| 06 | [高速手首応答](assets/06_chest_wrist_error_modulation_timeseries.png) | chest | `1.2 m/s + 8 rad/s` 胸前 diagonal variant、4 error/posture profile |
| 07 | [胸前 EEF path](assets/07_chest_wrist_eef_paths.png) | chest | Error-bound A/B の EEF path |
| 08 | [Posture regulation](assets/08_nullspace_branch_control.png) | recorded benchmark | 同じ fast retract に対する 3 secondary-posture profile |
| 09 | [Singularity approach](assets/09_singularity_reach_timeseries.png) | screening | 青は伸展、灰は retract、黄は減速区間 |
| 10 | [QP 内外 velocity limit](assets/10_driver_limit_coupling_timeseries.png) | driver | 同じ fast retract の along-path lag、lateral deviation、raw IK / actual J1 speed |
| 11 | [Joint-limit recovery](assets/11_recoverable_joint_limit.png) | boundary | Solved condition 数と `4 ms` outer solve 1 回後の境界外残角度 |
| 12 | [Joint braking](assets/12_joint_braking_envelope.png) | braking | Braking distance、joint margin、activation rate |
| 13 | [Single-parameter sweep](assets/13_parameter_sweep_summary.png) | parameters | Position/orientation budget と velocity-limit scale を含む |
| 14 | [組み合わせ parameter trade-off](assets/14_combined_tuning_candidates.png) | candidates | A-D metric heatmap と Pareto scatter。緑は改善、赤は悪化 |
| 15 | [Symmetry と robustness](assets/15_symmetry_and_robustness.png) | symmetry/robustness | 左右 mirror と plant perturbation |
| 16 | [Solve time](assets/16_solver_timing.png) | screening | 混合 trajectory。outer solve ごとに 5 substep |
| 17 | [Nullspace sweep](assets/17_nullspace_targeted_tuning.png) | nullspace sweep | 青枠は PR-default absolute value、他 cell は signed change。緑は改善、赤は悪化 |
| 18 | [Nullspace cross-validation](assets/18_nullspace_cross_validation.png) | nullspace validation | Family 間の candidate change。緑は改善、赤は悪化 |
| 33 | [代表 case overview](assets/33_final_showcase_controller_comparison.png) | recorded benchmark | 2 recorded-command-derived benchmark の summary |
| 34 | [胸前手首 benchmark](assets/34_chest_flip_benchmark_timeseries_and_path.png) | recorded benchmark | Position/orientation error、actual joint acceleration、EEF path |
| 35 | [Fast-retract benchmark](assets/35_fast_retract_benchmark_timeseries_and_path.png) | recorded benchmark | Elbow Y-Z path、`y-y0`、EEF position error、7 関節の maximum absolute acceleration |

## 6. 動画索引

全 controller-comparison video は profile/parameter、`arm_origin` 座標、playback multiplier、command/state/elbow path legend を表示する。赤い座標軸が target、半透明の腕が IK または driver command、不透明の腕が simulated actual state である。Report では高解像度の `8-12 fps` WebP preview を使用し、GIF は小容量の互換版として残している。

| Video | Type | Playback | 内容 |
|---|---|---:|---|
| [Reference-action catalogプレビュー](assets/video_previews/ideal_reference_trajectory_catalog.webp) ([GIF 互換版](assets/video_previews/ideal_reference_trajectory_catalog.gif), [MP4 ダウンロード](videos/ideal_reference_trajectory_catalog.mp4?raw=1)) | Target catalog | Segment ごと | 21 representative clip と 42/70 set 内の family count。Controller A/B ではない |
| [胸前 controller 比較プレビュー](assets/video_previews/near_chest_fast_wrist_roll_controller_comparison.webp) ([GIF 互換版](assets/video_previews/near_chest_fast_wrist_roll_controller_comparison.gif), [MP4 ダウンロード](videos/near_chest_fast_wrist_roll_controller_comparison.mp4?raw=1)) | 1 recorded-command-derived trajectory | 0.5x | PR default / Mainline baseline / PR w/o 6D error bound |
| [Fast-retract 6D error boundプレビュー](assets/video_previews/fast_retract_frame_error_bound_comparison.webp) ([GIF 互換版](assets/video_previews/fast_retract_frame_error_bound_comparison.gif), [MP4 ダウンロード](videos/fast_retract_frame_error_bound_comparison.mp4?raw=1)) | `2x` time-compressed recorded path | 0.5x | PR default / PR w/o 6D error bound、2 column |
| [胸前 roll + diagonal translationプレビュー](assets/video_previews/near_chest_roll_translation_error_bound_comparison.webp) ([GIF 互換版](assets/video_previews/near_chest_roll_translation_error_bound_comparison.gif), [MP4 ダウンロード](videos/near_chest_roll_translation_error_bound_comparison.mp4?raw=1)) | 1 synthetic trajectory | 0.5x | PR default / PR w/o 6D error bound |
| [Fast-retract controller 比較プレビュー](assets/video_previews/fast_retract_controller_comparison.webp) ([GIF 互換版](assets/video_previews/fast_retract_controller_comparison.gif), [MP4 ダウンロード](videos/fast_retract_controller_comparison.mp4?raw=1)) | 1 recorded-command-derived trajectory | 0.5x | PR default / Mainline baseline / PR w/o posture regulation、2 view |
| [Posture-regulation 比較プレビュー](assets/video_previews/fast_retract_posture_regulation_comparison.webp) ([GIF 互換版](assets/video_previews/fast_retract_posture_regulation_comparison.gif), [MP4 ダウンロード](videos/fast_retract_posture_regulation_comparison.mp4?raw=1)) | 1 recorded-command-derived trajectory | 0.5x | PR default / PR w/o posture regulation / PR: full-home posture 0.01 / 0.03 |
| [Nullspace parameter 2x2プレビュー](assets/video_previews/fast_retract_nullspace_parameter_comparison.webp) ([GIF 互換版](assets/video_previews/fast_retract_nullspace_parameter_comparison.gif), [MP4 ダウンロード](videos/fast_retract_nullspace_parameter_comparison.mp4?raw=1)) | 1 recorded-command-derived trajectory | 0.5x | PR default と 3 nullspace candidate |
| [IK velocity limitプレビュー](assets/video_previews/fast_retract_ik_velocity_limit_comparison.webp) ([GIF 互換版](assets/video_previews/fast_retract_ik_velocity_limit_comparison.gif), [MP4 ダウンロード](videos/fast_retract_ik_velocity_limit_comparison.mp4?raw=1)) | 1 recorded-command-derived trajectory | 0.5x | PR default / PR w/o IK velocity limits |
| [伸展 singularityプレビュー](assets/video_previews/straight_reach_singularity_limit_comparison.webp) ([GIF 互換版](assets/video_previews/straight_reach_singularity_limit_comparison.gif), [MP4 ダウンロード](videos/straight_reach_singularity_limit_comparison.mp4?raw=1)) | Deep-start shoulder-height synthetic trajectory | 0.5x | Start を `0.10 m` 後退、同じ最遠点。Singularity limit on/off。Actual J1 acceleration を表示 |

## 7. CSV 索引

| CSV | 内容 |
|---|---|
| [headline_baseline_means.csv](tables/headline_baseline_means.csv) | 4 IK controller の aggregate value |
| [ablation_relative_effects_percent.csv](tables/ablation_relative_effects_percent.csv) | Single-feature ablation の変化率 |
| [ablation_ddq_delta_by_family.csv](tables/ablation_ddq_delta_by_family.csv) | Action family 別 joint-acceleration change |
| [ablation_elbow_delta_by_family_cm.csv](tables/ablation_elbow_delta_by_family_cm.csv) | Action family 別 elbow change |
| [chest_profile_means.csv](tables/chest_profile_means.csv) | 胸前 14 target の aggregate |
| [near_chest_frozen_controller_metrics.csv](tables/near_chest_frozen_controller_metrics.csv) | 記録胸前 target metric |
| [fast_retract_frozen_controller_metrics.csv](tables/fast_retract_frozen_controller_metrics.csv) | 記録 fast-retract target metric |
| [branch_regulation_frozen_metrics.csv](tables/branch_regulation_frozen_metrics.csv) | Secondary-posture comparison |
| [boundary_recovery_solved_counts.csv](tables/boundary_recovery_solved_counts.csv) | Joint-limit recovery の solved condition |
| [braking_profile_means.csv](tables/braking_profile_means.csv) | Joint-braking aggregate |
| [driver_coupling_profile_means.csv](tables/driver_coupling_profile_means.csv) | QP と driver velocity limit の相互作用 |
| [driver_path_tracking_components.csv](tables/driver_path_tracking_components.csv) | `Fast diagonal retract: lateral +` の command direction に沿う最大絶対空間 lag と最大 lateral path deviation |
| [parameter_profile_means.csv](tables/parameter_profile_means.csv) | 全 single-parameter profile |
| [combined_candidate_definitions.csv](tables/combined_candidate_definitions.csv) | Combined candidate A-D の parameter definition |
| [combined_candidate_means.csv](tables/combined_candidate_means.csv) | Combined candidate の absolute metric |
| [combined_candidate_relative_percent.csv](tables/combined_candidate_relative_percent.csv) | PR default に対する candidate change |
| [nullspace_cost_return_grid.csv](tables/nullspace_cost_return_grid.csv) | Nullspace cost / return-rate grid |
| [nullspace_frozen_sweep_metrics.csv](tables/nullspace_frozen_sweep_metrics.csv) | 記録 retract に対する 44 nullspace profile |
| [nullspace_cross_validation_deltas.csv](tables/nullspace_cross_validation_deltas.csv) | Action family 間の candidate change |
| [exact_mirror_side_means.csv](tables/exact_mirror_side_means.csv) | 左右 mirror mean |
| [robustness_profile_means.csv](tables/robustness_profile_means.csv) | Robustness mean |
| [solver_timing_means.csv](tables/solver_timing_means.csv) | Mean と p95 solve time |
| [report_traceability.csv](tables/report_traceability.csv) | Public asset と suite、CSV、scenario、profile の対応 |
| [asset_checksums.sha256](tables/asset_checksums.sha256) | Final public asset の checksum |

## 8. データの範囲と解釈上の注意

- `solver_failures=0` は solver が正常に返ったことだけを意味し、collision、torque、全 tracking metric が実機で安全であることを証明しない。
- Position と orientation RMSE は simulated actual state から計算し、raw IK command ではない。
- Joint acceleration は固定 `4 ms` sample の速度差分であり、trajectory 間比較には 1 frame maximum より p99 が適する。
- MuJoCo の相対 A/B は実機の friction、compliance、firmware position loop の検証を置き換えない。
- 初期の実機記録には raw / filtered VR target がなく、同じ final target に対する controller response だけを比較できる。
- Posture-regulation 図と fast-retract controller 図は同じ target を共有する。前者は secondary posture task のみを変更して exact-nullspace regulation を分離し、後者は PR default、Mainline baseline、PR w/o posture regulation を比較して PR 全体を Mainline-task baseline に対して評価する。
