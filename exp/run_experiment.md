# Reproducing the OpenArm VR IK Experiments

Language: **English** | [中文](run_experiment.zh.md) | [日本語](run_experiment.ja.md)

This guide explains how to regenerate the target trajectories, MuJoCo simulation
results, figures, and videos used by this report. Unless stated otherwise, run
all commands from the root of the `openarm_control` repository.

## 1. Reproduction Scope

Choose a workflow according to the intended result:

| Goal | Required steps | Rerun simulations? |
|---|---|---|
| Check report integrity | Install environment, validate trajectories, validate report | No |
| Rebuild figures and videos | Install environment, build figures, build videos, validate report | No; uses retained results under `exp/results/` |
| Reproduce the complete study | Install environment, run all experiments, build figures and videos, validate report | Yes |
| Debug one mechanism | Install environment, run a selected suite, optionally build figures or videos | Yes |

The published report uses:

- upstream baseline: `d543cedeec5f`;
- evaluated PR revision: `f983a0eb5cae`;
- Python: `3.14.6`;
- Python dependencies: locked by `exp/src/uv.lock`;
- MuJoCo model package: `openarm-mujoco==2.0.1`;
- video encoder: `ffmpeg 6.1.1-3ubuntu5`.

Each experiment-suite manifest additionally records the source revision,
tracked-diff hash, model XML hash, frozen driver envelope, IK parameters, and
runtime dependency versions used for that run.

The lockfile freezes external Python dependencies, while `openarm-control` is
installed from the current repository checkout. The report commit is layered
on the evaluated PR revision and adds only `exp/` plus the local-results ignore
rule; it does not change the controller package after `f983a0eb5cae`. Checkout
the report branch to obtain the experiment harness. The revision above remains
the provenance identifier for the controller under test; new controller
changes require new results and updated manifests.

## 2. Directory Layout

```text
exp/
├── README.md                    # concise report, English primary
├── README.zh.md                 # concise report, Chinese
├── README.ja.md                 # concise report, Japanese
├── detailed_report.md           # detailed report, English primary
├── detailed_report.zh.md        # detailed report, Chinese
├── detailed_report.ja.md        # detailed report, Japanese
├── experiment_index.md          # experiment and asset index, English primary
├── experiment_index.zh.md       # experiment and asset index, Chinese
├── experiment_index.ja.md       # experiment and asset index, Japanese
├── run_experiment.md            # this guide, English primary
├── run_experiment.zh.md         # this guide, Chinese
├── run_experiment.ja.md         # this guide, Japanese
├── assets/                      # published figures
├── videos/                      # published videos
├── tables/                      # summary CSV files and asset checksums
├── manifests/                   # public experiment manifests
├── results/                     # local raw results, ignored by Git
└── src/
    ├── run_experiments.py       # experiment-suite entry point
    ├── trajectory_catalog.py    # trajectory generation and hash validation
    ├── build_figures.py         # figures and derived CSV files
    ├── build_videos.py          # videos
    ├── validate_report.py       # report-integrity checks
    ├── inputs/                  # two recorded targets and frozen driver envelope
    ├── TRAJECTORIES.md          # catalog of 70 unique trajectories
    ├── pyproject.toml
    └── uv.lock
```

`exp/results/final_report_20260801/` is approximately `400 MB` and contains the
summaries, metadata, and traces needed to rebuild the published figures and
videos. It is not committed to Git. Except for the two recorded inputs, all
trajectories are generated deterministically in code, so no additional bulk
NPZ files are stored.

## 3. Prerequisites

### 3.1 Experiment Configuration

The experiment package is self-contained. It does not read a deployment
dataflow or a configuration from a sibling repository. IK values come from the
evaluated `IKParams` defaults and are recorded in every manifest. The simulated
downstream limiter reads the frozen velocity envelope in
`exp/src/inputs/experiment_driver_config.yaml`; its hash is also recorded and
validated.

### 3.2 System Tools

Required:

- Linux x86_64;
- `uv` (the report used `uv 0.11.24`);
- an OpenGL/EGL environment capable of MuJoCo offscreen rendering;
- `ffmpeg`, required only for video generation;
- the DejaVu Sans font, to reproduce the same figure and video typography.

Python and Python packages do not need to be installed manually.

## 4. Create the Locked Environment

```bash
uv sync --project exp/src --frozen
```

This creates a Python `3.14.6` environment according to `.python-version` and
uses `exp/src/uv.lock` without updating it. `openarm-control` is installed in
editable mode from the repository root; all external Python dependencies are
frozen by the lockfile.

Run the fast checks first:

```bash
uv run --project exp/src --frozen python exp/src/trajectory_catalog.py \
  --check-report \
  --output /tmp/openarm-trajectories.md

uv run --project exp/src --frozen python exp/src/validate_report.py exp
```

The first command regenerates every trajectory and verifies its hash. The
second validates report links, manifests, frozen inputs, experiment parameters,
and the dependency lockfile.

## 5. Regenerate Target Trajectories

Print the JSON description of every trajectory:

```bash
uv run --project exp/src --frozen python exp/src/trajectory_catalog.py \
  --format json
```

Regenerate the human-readable catalog and compare it with the public
experiment manifest:

```bash
uv run --project exp/src --frozen python exp/src/trajectory_catalog.py \
  --check-report \
  --output exp/src/TRAJECTORIES.md
```

The generator creates and samples all targets in memory; it does not write a
large set of trajectory NPZ files. The catalog should report:

- `70` unique target trajectories;
- `67` procedurally generated trajectories;
- `2` frozen recorded trajectories;
- `1` trajectory derived by time-compressing a frozen trajectory.

Generation functions, parameters, suite membership, and SHA-256 values are
listed in [`src/TRAJECTORIES.md`](src/TRAJECTORIES.md).

## 6. Run the MuJoCo Experiments

### 6.1 Complete Study

The following command runs all 14 suites (13 dynamic suites and one static
boundary suite) and writes to the default result directory:

```bash
uv run --project exp/src --frozen python exp/src/run_experiments.py \
  --suite all \
  --workers 8
```

The complete report includes `1,921` dynamic controller-profile-by-trajectory
simulations and `378` static joint-boundary cases. Runtime depends on CPU and
worker count. Dynamic results are cached by profile and scenario, so an
interrupted run can resume in the same output directory.

To retain the published results and create an independent reproduction, choose
a new directory:

```bash
uv run --project exp/src --frozen python exp/src/run_experiments.py \
  --suite all \
  --output-dir exp/results/reproduction \
  --workers 8
```

### 6.2 One Experiment Suite

For example, run only the near-chest fast wrist-motion suite:

```bash
uv run --project exp/src --frozen python exp/src/run_experiments.py \
  --suite chest \
  --output-dir exp/results/reproduction \
  --workers 8
```

Available suites:

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

A single suite is useful for debugging and focused analysis. To run
`build_figures.py` and `build_videos.py` in full, the selected result directory
must still contain every suite referenced by those scripts.

Each dynamic suite generates:

- `metadata.json`: environment, model, configuration, profiles, trajectories,
  and hashes;
- `summary.csv`: metrics for every profile and trajectory;
- `traces/`: per-frame target, IK command, driver command, and simulated actual
  state needed by figures or videos.

The top-level `manifest.json` summarizes suites, run counts, unique
trajectories, frozen inputs, and the dependency-lock hash.

## 7. Regenerate Figures and Tables

Use the published result set to rebuild `exp/assets/` and the derived CSV files
in place:

```bash
uv run --project exp/src --frozen python exp/src/build_figures.py
```

Use an independent result set and write first to a temporary report directory:

```bash
uv run --project exp/src --frozen python exp/src/build_figures.py \
  --results exp/results/reproduction \
  --report-dir /tmp/openarm-report-reproduction
```

The script creates the 21 PNG figures used by the report and their associated
summary CSV files. `report_traceability.csv` is a manually reviewed index from
published assets to experiment sources and is not overwritten by the plotting
script.

## 8. Regenerate Videos

Generate all eight controller-comparison videos and the one 21-motion catalog
video:

```bash
uv run --project exp/src --frozen python exp/src/build_videos.py
```

Generate only the motion-catalog video:

```bash
uv run --project exp/src --frozen python exp/src/build_videos.py \
  --catalog-only
```

Generate only the controller-comparison videos:

```bash
uv run --project exp/src --frozen python exp/src/build_videos.py \
  --skip-catalog
```

Regenerate only the GitHub-renderable animated previews from existing MP4 files:

```bash
uv run --project exp/src --frozen python exp/src/build_videos.py \
  --previews-only
```

Use an independent result set and output directory:

```bash
uv run --project exp/src --frozen python exp/src/build_videos.py \
  --results exp/results/reproduction \
  --output-dir /tmp/openarm-video-reproduction \
  --preview-dir /tmp/openarm-video-previews
```

The renderer uses `MUJOCO_GL=egl` by default and streams RGB frames directly to
FFmpeg without retaining an intermediate image sequence. It also writes compact
animated GIF previews to `exp/assets/video_previews/`, because GitHub does not
render repository MP4 files inline.

## 9. Update and Validate the Published Report

After regenerating published assets, update their checksums and validate the
report:

```bash
uv run --project exp/src --frozen python exp/src/validate_report.py \
  exp \
  --write-checksums

uv run --project exp/src --frozen python exp/src/validate_report.py exp
```

Public file SHA-256 values can also be checked directly from `exp/`:

```bash
cd exp
sha256sum --check tables/asset_checksums.sha256
```

Return to the repository root afterward:

```bash
cd ..
```

## 10. Recommended Full Sequence

```text
1. uv sync --project exp/src --frozen
2. trajectory_catalog.py --check-report
3. run_experiments.py --suite all              # when simulations must be rerun
4. build_figures.py
5. build_videos.py
6. validate_report.py exp --write-checksums
7. validate_report.py exp
```

Skip step 3 when only rebuilding published assets from retained results.

## 11. Troubleshooting

### Configuration Hash Mismatch

The committed experiment driver configuration, IK defaults, or dependency
lock differs from the report manifest. Restore the recorded source revision to
reproduce the published report. To evaluate a new configuration, write to a
new results directory and retain its new manifest; do not overwrite the
published experiment definition.

### Frozen Trajectory Not Found

Confirm that these files exist:

```text
exp/src/inputs/near_chest_fast_wrist_roll.npz
exp/src/inputs/fast_retract_elbow_branch.npz
```

Their SHA-256 values are recorded in `exp/manifests/manifest.json`.

### EGL Initialization Failure

Confirm that the machine has a working EGL/OpenGL driver. Simulation metric
generation itself does not require video rendering: run the experiments and
figures first, then execute `build_videos.py` on a machine with EGL support.

### Matplotlib Cache Directory Is Not Writable

Choose a writable cache directory and regenerate the figures:

```bash
MPLCONFIGDIR=/tmp/openarm-matplotlib \
  uv run --project exp/src --frozen python exp/src/build_figures.py
```

### Disk Usage

The raw published experiment results use approximately `400 MB`; the isolated
Python environment uses approximately `420 MB`. Both `exp/results/` and
`.venv/` are ignored by Git and do not enter the commit.
