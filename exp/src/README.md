# Final Report Reproduction Source

Report: **[English](../README.md)** | [中文](../README.zh.md) | [日本語](../README.ja.md)<br>
Reproduction guide: **[English](../run_experiment.md)** | [中文](../run_experiment.zh.md) | [日本語](../run_experiment.ja.md)

This directory contains the code, frozen inputs, and locked Python environment
used to reproduce the public report in [`..`](../README.md). The complete
step-by-step workflow is documented in
[`../run_experiment.md`](../run_experiment.md). Run all commands below from the
`openarm_control` repository root.

## Contents

| Path | Purpose |
|---|---|
| `run_experiments.py` | Run all report experiment suites or one selected suite |
| `trajectory_catalog.py` | Enumerate and validate every target generator without writing trajectory arrays |
| `TRAJECTORIES.md` | Generated catalog of the 70 unique targets, builders, suites, and hashes |
| `build_figures.py` | Regenerate report figures and CSV tables from retained results |
| `build_videos.py` | Regenerate public MP4 videos, high-resolution WebP previews, and compact GIF fallbacks from retained traces |
| `validate_report.py` | Validate links, parameters, manifests, inputs, and asset checksums |
| `study.py` | Shared MuJoCo plant, trajectories, profiles, metrics, and matrix runner |
| `targeted_study.py` | Chest, braking, and robustness scenarios |
| `boundary_study.py` | Static recoverable-joint-limit experiment |
| `candidate_study.py` | Combined parameter candidates |
| `video_trajectory_study.py` | Deep-start reach and retract video trajectories |
| `render_videos.py` | Shared MuJoCo video rendering helpers |
| `inputs/` | Two recorded targets and the frozen experiment driver envelope |
| `pyproject.toml`, `uv.lock` | Exact Python dependency environment for experiments and report generation |

The procedural targets are generated deterministically in memory. The two
trajectory files in [`inputs/`](inputs/README.md) are retained because they
were derived from recorded commands. Their hashes are pinned in
[`../manifests/manifest.json`](../manifests/manifest.json).

Raw traces and per-run metrics are retained under
`../results/final_report_20260801/`. This approximately 400 MB directory is
excluded from Git but is used directly by the figure and video builders.

## Environment

The experiment project pins Python `3.14.6` in `.python-version` and all Python
packages in `uv.lock`. The lock was generated and validated with `uv 0.11.24`.
Create the isolated environment with:

```bash
uv sync --project exp/src --frozen
```

The report videos additionally require FFmpeg with the `libwebp_anim` encoder.
The published assets were rendered with `ffmpeg 6.1.1-3ubuntu5`, MuJoCo EGL
rendering, and DejaVu Sans.
Solver timing also depends on CPU, BLAS, and operating-system scheduling; the
measured platform remains recorded in each suite manifest.

`openarm-control` is installed from the repository root rather than from a
package index. The suite manifests separately record the tested Git revision
and tracked-diff hash, while `uv.lock` fixes the external Python dependencies.

## Commands

Regenerate all experiment results:

```bash
uv run --project exp/src --frozen python exp/src/run_experiments.py \
  --suite all \
  --workers 8
```

List all unique target trajectories as JSON without creating NPZ files:

```bash
uv run --project exp/src --frozen python exp/src/trajectory_catalog.py \
  --format json
```

Regenerate and verify the readable trajectory catalog:

```bash
uv run --project exp/src --frozen python exp/src/trajectory_catalog.py \
  --check-report \
  --output exp/src/TRAJECTORIES.md
```

Regenerate figures and tables from the retained results:

```bash
uv run --project exp/src --frozen python exp/src/build_figures.py
```

Regenerate all public videos and GitHub-renderable animated previews, or only
the 21-action trajectory catalog:

```bash
uv run --project exp/src --frozen python exp/src/build_videos.py
uv run --project exp/src --frozen python exp/src/build_videos.py --catalog-only
uv run --project exp/src --frozen python exp/src/build_videos.py --previews-only
```

Validate the complete published report:

```bash
uv run --project exp/src --frozen python exp/src/validate_report.py exp
```

The experiment runner records the IK parameters, frozen driver envelope,
model hash, source revision, and runtime package versions in the suite
manifests. It does not read a deployment dataflow or any sibling repository.
