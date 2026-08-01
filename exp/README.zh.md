# OpenArm VR IK 仿真评估简报

语言：[English](README.md) | **中文** | [日本語](README.ja.md)

日期：2026-08-01<br>
上游主线基准：`d543cedeec5f`（上游 `main` 分支，tag `0.2.0`）<br>
受测 PR revision：`f983a0eb5cae`（将实验快照 `d006ece506f2` 及其已记录差异 `0ef6a402...` 原样提交后的版本）<br>
完整资料：[详细技术报告](detailed_report.zh.md) · [实验与资产索引](experiment_index.zh.md) · [实验复现指南](run_experiment.zh.md) · [实验 manifest](manifests/manifest.json)

## 决策结论

**建议将当前 PR 默认配置作为 VR IK 的部署基线。** 在 `42` 条多类型仿真轨迹上，与 **Mainline-task baseline** 相比，位置 RMSE 降低约 `71%`，实际关节加速度 p99 降低约 `49%`，肘部笛卡尔加速度 p99 降低约 `47%`，动作末段的末端执行器（EEF）残余运动降低约 `75%`。

主要代价是快速翻腕时出现暂态姿态滞后。这是明确的设计取舍：优先保持末端位置路径、肩肘运动分支和整臂稳定性，而非在一次 QP 中尽快消化过大的旋转误差。

## 比较对象与 PR 改动

报告涉及三个不同的比较对象：

| 对象 | 版本或名称 | 用途 |
|---|---|---|
| 上游主线基准 | `d543cedeec5f`（tag `0.2.0`） | PR 的 merge-base；用于说明上游能力和计算代码差异 |
| 受测 PR | `f983a0eb5cae` | 本报告实际评价的 PR 源码 |
| Mainline-task baseline | `strict_mainline` profile | 在统一子步时序和关节包络中恢复主线 task 参数；用于算法对照，不是旧版本的原样重放 |

上游已经具备基于 MuJoCo 和 Mink 的微分 IK、6D 末端跟踪、`arm_origin` 相对坐标、damping、full-home posture task，以及基本的位置和速度限制。PR 在此基础上处理大目标误差、7-DoF 冗余分支、奇异点迫近和越界恢复。具体背景见[详细报告第 1 节](detailed_report.zh.md#1-研究背景与评估对象)，算法见[第 2 节](detailed_report.zh.md#2-pr-求解架构与新增机制)。

| 机制 | 建议 | 主要作用 |
|---|---|---|
| 6D 末端误差调制 | **保留** | 避免大位置或姿态误差在关节限速下触发肩肘运动重分配 |
| 精确零空间 home 正则 | **保留** | 仅沿 7-DoF 手臂的一维精确零空间约束肘部分支 |
| 单侧奇异点接近限速 | **保留** | 只减慢使几何奇异度继续下降的运动 |
| QP 内关节速度包络 | **保留** | 使末端任务与次级任务在可执行速度范围内共同求解 |
| 可恢复关节包络 | **保留** | 允许略微越界的关节在速度上限内逐步回界 |
| 距离相关关节制动 | **保留，可独立关闭诊断** | 在物理位置边界前逐步降低朝向边界的允许速度 |
| 动能正则 | **低权重保留** | 在运动学代价近似相同的解之间提供弱选择偏好 |

## 实验设计与数据来源

评估区分控制链中的四类状态：

- **目标位姿（target）**：VR 映射后提交给 IK 的末端目标；
- **IK 指令（raw IK command）**：Mink 完成一次外层求解后得到的关节配置；
- **驱动器指令（driver command）**：IK 指令经驱动器逐关节限速后的参考；
- **仿真实际状态（actual state）**：MuJoCo 被控对象对驱动器指令的动态响应。

除特别说明外，跟踪指标均由目标位姿与**仿真实际状态**计算。IK 速度上限在 QP 内生效；驱动器速度上限在 QP 后逐轴裁剪。报告分别记录两层指令，并通过[详细报告第 5.5 节](detailed_report.zh.md#55-qp-与驱动器速度限制的-ab)的专项 A/B 检验只保留驱动器限速的影响。

![图 1：VR IK 控制链及目标、指令与实际状态](assets/01_control_layers.png)

实验共使用 `70` 条唯一目标轨迹。其中，`42` 条固定轨迹用于主要方案、功能消融和组合参数的统一对比；另外 `28` 条轨迹用于专项测试。报告还从这些轨迹中选取 `21` 个动作片段制作视频，便于查看目标运动和典型控制响应。

| 实验内容 | 构成 | 用途 |
|---|---|---|
| 固定对比轨迹 | `42` 条：reach/extended `19`、retract `6`、wrist/chest `12`、normal workspace `5` | 所有主要方案、单功能消融和组合候选使用相同输入 |
| 额外专项轨迹 | `28` 条；与固定对比轨迹合计为 `70` 条唯一目标 | 测试左右镜像、额外速度与方向、胸前快速翻腕、关节制动、实机记录回放和深起点奇异动作 |
| 动作示例视频 | 从 `70` 条目标中选取 `21` 个片段 | 展示目标运动和典型控制响应；视频不是额外的定量实验集 |

13 个实验组共执行 `1,921` 次“控制器配置 × 目标轨迹”动态仿真，覆盖 `119` 种控制器配置，未出现 QP 求解失败。另有 `378` 个静态关节边界条件。完整动作族加总、21 个片段名称和实验组清单见[实验与资产索引](experiment_index.zh.md#3-目标轨迹)；[动画目录预览](assets/video_previews/ideal_reference_trajectory_catalog.webp)可用于快速浏览动作，也可下载[原始 MP4](videos/ideal_reference_trajectory_catalog.mp4)。

其中两条重点轨迹由实机指令记录派生，并使用统一目标、MuJoCo 被控对象和驱动器速度上限进行仿真回放：

- **Near-chest fast wrist-roll**：胸前快速翻腕并伴随约 `4.8 cm` 平移，峰值角速度 `12.02 rad/s`；
- **Fast-retract elbow-branch**：右臂从接近伸直状态快速回缩，峰值线速度 `0.589 m/s`。

这两条轨迹保留了实机指令逐帧的位姿、速度和方向变化，但实际状态和量化指标仍来自仿真，不代表实机闭环性能。

## 核心结果

四种方案使用同一组 `42` 条轨迹。指标先在单条轨迹内计算，再跨轨迹平均；末段 EEF 峰峰值取最后 `0.35 s`。除姿态误差体现主动取舍外，其余指标均越低越好。

| IK 控制方案 | 位置 RMSE ↓ | 朝向 RMSE ↓ | 关节加速度 p99 ↓ | 肘部加速度 p99 ↓ | 肘横向范围 ↓ | 末段 EEF p2p ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **PR default** | **3.97 cm** | 31.27° | **29.43 rad/s²** | **5.46 m/s²** | 6.21 cm | **0.84 cm** |
| PR w/o IK velocity limits | 4.01 cm | 30.84° | 31.91 rad/s² | 6.22 m/s² | 6.50 cm | 0.89 cm |
| Mainline-task baseline | 13.78 cm | **11.13°** | 57.49 rad/s² | 10.21 m/s² | 6.12 cm | 3.37 cm |
| PR: full-home posture 0.01 | 3.94 cm | 29.03° | 33.42 rad/s² | 6.52 m/s² | 7.38 cm | 0.91 cm |

![图 2：四种 IK 控制方案的整体对照。PR default 主动接受部分朝向滞后，以降低位置误差、实际加速度和末段残余运动](assets/03_headline_baseline_comparison.png)

跨轨迹平均的肘横向范围无法判断单次快速回缩选择了哪条零空间运动分支，因此还需结合后文的 fast-retract 定向案例。

## 功能消融

下图每次仅从 PR 默认配置中移除一个功能。数值表示相对变化；正值表示指标增大。由于姿态滞后是主动取舍，单个指标下降不一定代表整体更优。

![图 3：单功能消融。6D 误差上限、精确零空间正则和奇异点限速的收益集中在不同场景](assets/04_feature_ablation_heatmap.png)

主要观察如下：

- 移除姿态误差上限后，位置 RMSE 增加 `125%`，末段 EEF 残余运动增加 `181%`，驱动器限速激活率增加 `212%`；
- 移除精确零空间正则后，肘部横向范围增加 `19%`，关节加速度 p99 增加 `13%`，驱动器限速激活率增加 `108%`；
- 移除奇异点限速后，关节加速度 p99 增加 `12%`；收益主要集中在伸直和 extended-arm 轨迹；
- 关节制动和动能正则对跨轨迹均值的影响较小，收益集中在接近位置边界或存在近似等价运动学解的场景。

按动作族分解的消融结果见[详细报告第 4.2 节](detailed_report.zh.md#42-单功能消融)。

## 三个代表案例

### 1. 胸前快速翻腕：6D 误差调制

该轨迹的平移仅约 `4.8 cm`，但峰值角速度达到 `12.02 rad/s`。A/B 结果表明，若 QP 尽快追踪完整旋转误差，旋转需求与关节限速会共同改变肩肘分配，使末端偏离原位置路径。

| 控制方案 | 位置 RMSE / 最大值 ↓ | 朝向 RMSE | 关节加速度 p99 ↓ | 驱动器限速激活率 ↓ |
|---|---:|---:|---:|---:|
| **PR default** | **1.28 / 1.78 cm** | 28.4° | **39.3 rad/s²** | **15.3%** |
| Mainline-task baseline | 5.81 / 17.11 cm | **14.5°** | 73.0 rad/s² | 29.3% |
| PR w/o 6D error bound | 1.79 / 4.81 cm | 22.0° | 47.3 rad/s² | 22.1% |

![图 4：实机记录派生的胸前快速翻腕回放。PR default 限制最大位置偏离和实际关节加速度，代价是暂态朝向滞后](assets/34_chest_flip_benchmark_timeseries_and_path.png)

[查看三种控制方案动画预览](assets/video_previews/near_chest_fast_wrist_roll_controller_comparison.webp) · [下载 MP4](videos/near_chest_fast_wrist_roll_controller_comparison.mp4) · [查看胸前压力测试 A/B](detailed_report.zh.md#511-胸前压力测试)

### 2. 快速回缩：精确零空间分支约束

精确零空间 task 只调节 home 构型误差在当前一维零空间上的投影，不把 full-home 偏好直接施加到全部关节方向。

| 次级正则 | 位置 RMSE / 最大值 | 肘横向范围 ↓ | 关节加速度 p99 | 驱动器限速激活率 |
|---|---:|---:|---:|---:|
| **精确零空间（PR default）** | 1.80 / 4.38 cm | **4.04 cm** | 40.90 rad/s² | 36.9% |
| 无 posture regulation | 1.51 / 3.08 cm | 17.59 cm | 38.92 rad/s² | 33.7% |
| Full-home posture 0.01 | **1.49 / 3.00 cm** | 17.61 cm | **38.84 rad/s²** | **33.2%** |

![图 5：快速回缩中的肘部 Y-Z 路径、横移、末端位置误差和实际关节加速度](assets/08_nullspace_branch_control.png)

精确零空间正则以约 `3 mm` 的额外位置 RMSE，将肘部横向范围减少约 `13.6 cm`。当 `posture_cost=0.003/0.01/0.03` 时，full-home `PostureTask` 的肘部横向范围均约为 `17.6 cm`，未形成等价的分支约束。

[姿态正则动画预览](assets/video_previews/fast_retract_posture_regulation_comparison.webp) · [完整控制方案动画预览](assets/video_previews/fast_retract_controller_comparison.webp) · [MP4 文件](experiment_index.zh.md#6-视频索引)

### 3. 向可达域外伸直：奇异点接近限速

奇异点限制使用只由当前几何构型决定的无量纲奇异值比

$$
\rho(q)=\frac{\sigma_{\min}(J_{\mathrm{norm}})}{\sigma_{\max}(J_{\mathrm{norm}})}.
$$

QP 只限制使 $\rho$ 继续下降的关节运动；离开奇异点或沿等奇异度方向移动不受影响。在 `0.8 m/s` 肩高伸直轨迹上，PR 默认配置将最小 $\rho$ 从 `0.0047` 提高到 `0.0323`，关节加速度 p99 从 `48.7` 降至 `40.6 rad/s²`，位置 RMSE 基本不变。

![图 6：奇异点接近限速只改变伸直阶段，并在回缩时自动释放](assets/09_singularity_reach_timeseries.png)

[查看伸直奇异区 A/B 动画预览](assets/video_previews/straight_reach_singularity_limit_comparison.webp) · [下载 MP4](videos/straight_reach_singularity_limit_comparison.mp4)

## 其他关键验证

| 问题 | 验证结果 | 结论 |
|---|---|---|
| 关节略微越界后，位置与速度约束是否冲突 | 可恢复包络解出 `126/126` 个条件；独立位置与速度约束仅解出 `78/126` | 联合包络修复了越界恢复的可行性问题 |
| 驱动器限速能否替代 QP 内限速 | 快速斜向回缩中，关闭 QP 内限速后，位置 RMSE 从 `3.93 cm` 增至 `5.03 cm`，最大横向路径偏差从 `2.68 cm` 增至 `9.43 cm` | 在该 A/B 场景中，QP 内限速能保持更接近原路径的任务分配 |
| 关节制动是否增加物理边界余量 | 在 `12 rad/s` 翻腕目标中，`0.20 rad` 制动距离将最小关节余量从 `26` 提高到 `62 mrad` | 提高位置边界余量，代价是更早产生跟踪滞后 |
| 动能正则是否提供明显动力学收益 | 移除后，关节和肘部加速度 p99 分别增加约 `2.3%` 和 `3.6%` | 仅为弱选择偏好，不是逆动力学或重力补偿 |

QP 与驱动器限速的完整时序和路径分解见[详细报告第 5.5 节](detailed_report.zh.md#55-qp-与驱动器速度限制的-ab)。

## 参数与部署建议

单参数扫描、组合调参和精确零空间专项扫描均未找到能够同时改善跟踪、关节动态、肘部分支和驱动器限速激活率的配置。当前默认值不是每条轨迹上的单项最优点，但在不同场景之间提供了较稳定的折中。

| 参数组 | 当前部署值 |
|---|---|
| 末端 task / damping | `position_cost=12`, `orientation_cost=1.5`, `damping=0.1`, `lm_damping=0.01` |
| 外层时序 | `4 ms`, `5` 个 QP 子步 |
| 6D 总误差预算 | `0.020 m / 0.25 rad`；位置速度区间 `0.6 -> 0.9 m/s`；保持阈值 `0.006 m` |
| 精确零空间正则 | cost `8.5`；return `1.6 s⁻¹`；max speed `1.0 rad/s`；activation `0.02 -> 0.05` |
| 奇异点接近限速 | stop/slow `0.02 / 0.08`；max approach rate `0.25 s⁻¹` |
| 关节制动 | distance `0.20 rad`；exponent `2`；measured-state buffer `0.01 rad` |
| 动能正则 | `2e-5` |
| IK / 驱动器速度上限 J1-J7 | `[2, 2, 3.14, 3.14, 6.3, 6.3, 6.3] rad/s` |

精确零空间专项扫描与交叉验证见[详细报告第 6.1 节](detailed_report.zh.md#61-零空间参数扫描与交叉验证)；其余单参数曲线和组合候选见[第 6.2 节起](detailed_report.zh.md#62-单参数敏感性)。

## 当前局限

- 胸前快速翻腕并突然外摆时，某些弱可控方向仍会激发较大的肩肘运动。收紧姿态误差预算可进一步保护位置，但会增加姿态滞后。
- 实测关节位置仅保守影响制动和奇异点包络，不持续覆盖 Mink 的积分指令；指令与实际状态之间的领先量仍可能累积。
- 关节包络不提供环境碰撞保护；桌面和底板余量仍需独立的笛卡尔空间或碰撞约束。
- MuJoCo A/B 支持控制方案之间的相对比较，但不能替代包含真实摩擦、结构柔性、电机带宽和固件位置环的实机验证。
- 早期实机记录未同步保存原始 VR、滤波后目标和最终限幅目标，因此无法唯一归因少数复合翻腕异常。

## 可复现性

- [实验复现指南](run_experiment.zh.md)：从环境安装到轨迹、仿真、图表、视频和最终校验的完整步骤；
- [实验代码与复现命令](src/README.md)：仿真、分析、视频生成、冻结输入和本地原始结果；
- [完整技术报告](detailed_report.zh.md)：算法公式、实验条件、全部专项结果、参数扫描、性能与 API 验证；
- [实验与资产索引](experiment_index.zh.md)：图、视频、CSV、实验组和生成脚本映射；
- [顶层实验 manifest](manifests/manifest.json)：revision、依赖、模型 hash、profile 与场景清单；
- [整体方案 CSV](tables/headline_baseline_means.csv)；
- [单功能消融 CSV](tables/ablation_relative_effects_percent.csv)；
- [快速回缩姿态正则 CSV](tables/branch_regulation_frozen_metrics.csv)。
