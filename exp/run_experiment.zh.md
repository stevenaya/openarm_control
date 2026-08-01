# OpenArm VR IK 实验复现指南

语言：[English](run_experiment.md) | **中文** | [日本語](run_experiment.ja.md)

本文说明如何在本地重新生成本报告使用的目标轨迹、MuJoCo 仿真结果、
图表和视频。除特别说明外，所有命令均从 `openarm_control` 仓库根目录执行。

## 1. 复现范围

根据目的选择下列流程：

| 目的 | 需要执行的部分 | 是否重新运行仿真 |
|---|---|---|
| 检查报告完整性 | 环境安装、轨迹校验、报告校验 | 否 |
| 重建图表和视频 | 环境安装、图表生成、视频生成、报告校验 | 否，使用 `exp/results/` 中保留的结果 |
| 完整复现实验 | 环境安装、全部实验、图表、视频、报告校验 | 是 |
| 调试单项机制 | 环境安装、指定 suite、按需生成图表或视频 | 是 |

发布报告使用：

- 上游基准：`d543cedeec5f`；
- 受测 PR revision：`f983a0eb5cae`；
- Python：`3.14.6`；
- Python 依赖：由 `exp/src/uv.lock` 固定；
- MuJoCo 模型：`openarm-mujoco==2.0.1`；
- 视频编码：`ffmpeg 6.1.1-3ubuntu5`。

每个实验组的 manifest 还记录了实际源码 revision、tracked diff hash、模型
XML hash、冻结的 driver 速度包络、IK 参数和运行时依赖版本。

## 2. 目录结构

```text
exp/
├── README.md                    # 简明报告，英文主版本
├── README.zh.md                 # 简明报告，中文
├── README.ja.md                 # 简明报告，日文
├── detailed_report.md           # 详细报告，英文主版本
├── detailed_report.zh.md        # 详细报告，中文
├── detailed_report.ja.md        # 详细报告，日文
├── experiment_index.md          # 实验与资产索引，英文主版本
├── experiment_index.zh.md       # 实验与资产索引，中文
├── experiment_index.ja.md       # 实验与资产索引，日文
├── run_experiment.md            # 本指南，英文主版本
├── run_experiment.zh.md         # 本指南，中文
├── run_experiment.ja.md         # 本指南，日文
├── assets/                      # 发布图表
├── videos/                      # 发布视频
├── tables/                      # 汇总 CSV 与资产校验和
├── manifests/                   # 公开实验清单
├── results/                     # 本地原始结果，Git 忽略
└── src/
    ├── run_experiments.py       # 实验组入口
    ├── trajectory_catalog.py    # 轨迹生成与哈希校验
    ├── build_figures.py         # 图表和派生 CSV
    ├── build_videos.py          # 视频
    ├── validate_report.py       # 报告完整性校验
    ├── inputs/                  # 两条记录目标及冻结的 driver 速度包络
    ├── TRAJECTORIES.md          # 70 条唯一轨迹目录
    ├── pyproject.toml
    └── uv.lock
```

`exp/results/final_report_20260801/` 约为 `400 MB`，包含报告图表和视频所需
的 summary、metadata 和 trace。该目录不提交到 Git。除两条记录派生输入外，
轨迹均由代码确定性生成，不保存额外 NPZ。

## 3. 前置条件

### 3.1 实验配置

实验包可以独立复现，不读取部署 dataflow，也不依赖同级仓库中的配置文件。
IK 标量参数来自受测版本的 `IKParams` 默认值。部署通过 `--limit-velocity`
启用 IK 速度包络；由于没有提供 `--config` 覆盖，该参数读取受测控制器 revision
中的内置 caps，实验 profile 则直接注入同一 mapping。仿真的下游限速器读取
`exp/src/inputs/experiment_driver_config.yaml` 中冻结的部署速度包络。两层包络
分别配置并分别校验；本次部署中数值恰好相同，但并不要求始终相等。

### 3.2 系统工具

需要：

- Linux x86_64；
- `uv`，本报告使用 `uv 0.11.24`；
- 支持 MuJoCo EGL 离屏渲染的 OpenGL/EGL 环境；
- 带 `libwebp_anim` 编码器的 `ffmpeg`，仅生成视频时需要；
- DejaVu Sans 字体，用于复现相同的图表和视频排版。

Python 和 Python 包不需要手动安装。

## 4. 创建锁定环境

```bash
uv sync --project exp/src --frozen
```

该命令按照 `.python-version` 创建 Python `3.14.6` 环境，并严格使用
`exp/src/uv.lock`。`openarm-control` 从当前仓库根目录以 editable 方式安装，
外部 Python 依赖则全部由锁文件固定。

锁文件只固定外部 Python 依赖，`openarm-control` 则从当前仓库 checkout 安装。
报告提交叠加在受测 PR revision 之上，只新增 `exp/` 和本地结果忽略规则，
不会改变 `f983a0eb5cae` 之后的控制器包源码。复现时应 checkout 报告分支以
取得实验脚本；上述 revision 仍是受测控制器的来源标识。若控制器源码发生
变化，则必须重新生成结果并更新 manifest。

先运行快速检查：

```bash
uv run --project exp/src --frozen python exp/src/trajectory_catalog.py \
  --check-report \
  --output /tmp/openarm-trajectories.md

uv run --project exp/src --frozen python exp/src/validate_report.py exp
```

第一条命令重新生成全部轨迹并校验其哈希；第二条检查报告链接、manifest、
冻结输入、实验参数和依赖锁文件。

## 5. 重新生成目标轨迹

查看全部轨迹的 JSON 描述：

```bash
uv run --project exp/src --frozen python exp/src/trajectory_catalog.py \
  --format json
```

重新生成可读目录并与公开实验清单核对：

```bash
uv run --project exp/src --frozen python exp/src/trajectory_catalog.py \
  --check-report \
  --output exp/src/TRAJECTORIES.md
```

该过程会在内存中生成并采样所有目标，不会写出大量轨迹 NPZ。目录应报告：

- `70` 条唯一目标轨迹；
- `67` 条程序生成轨迹；
- `2` 条冻结记录轨迹；
- `1` 条由冻结轨迹时间压缩得到的派生轨迹。

具体生成函数、参数、实验组和 SHA-256 见
[`src/TRAJECTORIES.md`](src/TRAJECTORIES.md)。

## 6. 运行 MuJoCo 实验

### 6.1 完整实验

以下命令运行全部 14 个 suite（13 个动态实验组和 1 个静态边界实验组），
并将结果写入默认位置：

```bash
uv run --project exp/src --frozen python exp/src/run_experiments.py \
  --suite all \
  --workers 8
```

完整报告包含 `1,921` 次动态“控制器配置 × 轨迹”仿真，以及 `378` 个静态
关节边界条件。运行时间取决于 CPU 和 worker 数量。动态结果按 profile 和
scenario 缓存，因此中断后使用同一输出目录可以继续。

若要保留发布结果并进行独立复现，指定新目录：

```bash
uv run --project exp/src --frozen python exp/src/run_experiments.py \
  --suite all \
  --output-dir exp/results/reproduction \
  --workers 8
```

### 6.2 单个实验组

例如，只运行胸前快速翻腕实验：

```bash
uv run --project exp/src --frozen python exp/src/run_experiments.py \
  --suite chest \
  --output-dir exp/results/reproduction \
  --workers 8
```

可选 suite：

```text
screening
parameters
driver
symmetry
chest
braking
robustness
candidates
boundary
frozen
fast_retract_error_bound
singularity_video
nullspace_sweep
nullspace_validation
```

单 suite 输出适合调试和专项分析。完整执行 `build_figures.py` 和
`build_videos.py` 时，结果目录仍需包含它们引用的全部实验组。

每个动态 suite 生成：

- `metadata.json`：环境、模型、配置、profile、轨迹及哈希；
- `summary.csv`：每个 profile/trajectory 的指标；
- `traces/`：需要绘图或视频的逐帧 target、IK command、driver command 和
  simulated actual state。

顶层 `manifest.json` 汇总各实验组、总运行数、唯一轨迹数、冻结输入和依赖
锁文件哈希。

## 7. 重新生成图表和表格

使用发布结果直接覆盖生成 `exp/assets/` 和派生 CSV：

```bash
uv run --project exp/src --frozen python exp/src/build_figures.py
```

使用独立复现结果并先输出到临时报告目录：

```bash
uv run --project exp/src --frozen python exp/src/build_figures.py \
  --results exp/results/reproduction \
  --report-dir /tmp/openarm-report-reproduction
```

脚本会生成报告中的 21 张 PNG 以及对应的汇总 CSV。`report_traceability.csv`
是公开资产与实验来源的人工审阅索引，不由绘图脚本覆盖。

## 8. 重新生成视频

生成全部 8 个控制器对比视频和 1 个 21 动作目录视频：

```bash
uv run --project exp/src --frozen python exp/src/build_videos.py
```

仅生成动作目录视频：

```bash
uv run --project exp/src --frozen python exp/src/build_videos.py \
  --catalog-only
```

仅生成控制器对比视频：

```bash
uv run --project exp/src --frozen python exp/src/build_videos.py \
  --skip-catalog
```

仅根据现有 MP4 重新生成可在 GitHub 直接显示的动画预览：

```bash
uv run --project exp/src --frozen python exp/src/build_videos.py \
  --previews-only
```

使用独立结果和输出目录：

```bash
uv run --project exp/src --frozen python exp/src/build_videos.py \
  --results exp/results/reproduction \
  --output-dir /tmp/openarm-video-reproduction \
  --preview-dir /tmp/openarm-video-previews
```

渲染脚本默认使用 `MUJOCO_GL=egl`，并将 RGB frame 直接输送给 FFmpeg，不
保留中间图片序列。由于 GitHub 不会内嵌播放仓库中的 MP4，脚本还会在
`exp/assets/video_previews/` 下生成高分辨率 WebP 动画预览和紧凑的 GIF
兼容版本。

## 9. 更新和校验公开报告

重新生成发布资产后，更新资产校验和并检查报告：

```bash
uv run --project exp/src --frozen python exp/src/validate_report.py \
  exp \
  --write-checksums

uv run --project exp/src --frozen python exp/src/validate_report.py exp
```

还可从 `exp/` 目录直接验证公开文件 SHA-256：

```bash
cd exp
sha256sum --check tables/asset_checksums.sha256
```

校验通过后，返回仓库根目录继续操作：

```bash
cd ..
```

## 10. 推荐的完整执行顺序

```text
1. uv sync --project exp/src --frozen
2. trajectory_catalog.py --check-report
3. run_experiments.py --suite all              # 需要重新仿真时
4. build_figures.py
5. build_videos.py
6. validate_report.py exp --write-checksums
7. validate_report.py exp
```

若只需从保留结果重建公开资产，可跳过第 3 步。

## 11. 常见问题

### 配置哈希不一致

说明仓库内的实验 driver 配置、IK 默认值或依赖锁与 report manifest 不一致。
若目标是复现本报告，应恢复 manifest 中记录的源码 revision；若目标是评价新
配置，应输出到新的 results 目录并保留新的 manifest，不要覆盖既有实验口径。

### 找不到冻结轨迹

确认以下文件存在：

```text
exp/src/inputs/near_chest_fast_wrist_roll.npz
exp/src/inputs/fast_retract_elbow_branch.npz
```

它们的 SHA-256 记录在 `exp/manifests/manifest.json`。

### EGL 初始化失败

确认机器具有可用的 EGL/OpenGL 驱动。仿真指标生成本身不要求视频渲染；可以
先运行实验和图表，之后在具备 EGL 的机器上单独执行 `build_videos.py`。

### Matplotlib 缓存目录不可写

设置一个可写缓存目录后重新生成图表：

```bash
MPLCONFIGDIR=/tmp/openarm-matplotlib \
  uv run --project exp/src --frozen python exp/src/build_figures.py
```

### 磁盘占用

当前发布实验的原始结果约为 `400 MB`，独立 Python 环境约为 `420 MB`。
`exp/results/` 和 `.venv/` 均已加入 Git ignore，不会进入提交。
