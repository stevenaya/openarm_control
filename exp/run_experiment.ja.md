# OpenArm VR IK 実験の再現手順

言語：[English](run_experiment.md) | [中文](run_experiment.zh.md) | **日本語**

本書では、レポートで使用した目標軌道、MuJoCo シミュレーション結果、図、
動画をローカルで再生成する方法を説明します。特記がない限り、すべての
コマンドは `openarm_control` リポジトリのルートから実行してください。

## 1. 再現範囲

目的に応じて以下の手順を選択します。

| 目的 | 実行する項目 | シミュレーションの再実行 |
|---|---|---|
| レポートの整合性確認 | 環境構築、軌道検証、レポート検証 | 不要 |
| 図と動画の再生成 | 環境構築、図生成、動画生成、レポート検証 | 不要。`exp/results/` の保持済み結果を使用 |
| 実験全体の再現 | 環境構築、全実験、図、動画、レポート検証 | 必要 |
| 個別メカニズムのデバッグ | 環境構築、指定 suite、必要に応じて図または動画 | 必要 |

公開レポートでは以下を使用しています。

- upstream baseline：`d543cedeec5f`；
- 評価対象 PR revision：`f983a0eb5cae`；
- Python：`3.14.6`；
- Python 依存関係：`exp/src/uv.lock` で固定；
- MuJoCo モデルパッケージ：`openarm-mujoco==2.0.1`；
- 動画エンコーダ：`ffmpeg 6.1.1-3ubuntu5`。

各 experiment suite の manifest には、実行時のソース revision、tracked diff
hash、model XML hash、固定 driver velocity envelope、IK parameter、runtime
dependency の version も記録されます。

lockfile は外部 Python 依存関係を固定し、`openarm-control` は現在の
repository checkout から install されます。Report commit は評価対象 PR
revision の上に `exp/` と local result の ignore rule だけを追加し、
`f983a0eb5cae` 以降の controller package source は変更しません。実験 harness
を取得するには report branch を checkout し、上記 revision を評価対象
controller の provenance ID として扱います。Controller source を変更した
場合は result を再生成し、manifest も更新する必要があります。

## 2. ディレクトリ構成

```text
exp/
├── README.md                    # 簡明版、英語を主版とする
├── README.zh.md                 # 簡明版、中国語
├── README.ja.md                 # 簡明版、日本語
├── detailed_report.md           # 詳細版、英語を主版とする
├── detailed_report.zh.md        # 詳細版、中国語
├── detailed_report.ja.md        # 詳細版、日本語
├── experiment_index.md          # 実験・asset index、英語を主版とする
├── experiment_index.zh.md       # 実験・asset index、中国語
├── experiment_index.ja.md       # 実験・asset index、日本語
├── run_experiment.md            # 本手順、英語を主版とする
├── run_experiment.zh.md         # 本手順、中国語
├── run_experiment.ja.md         # 本手順、日本語
├── assets/                      # 公開図
├── videos/                      # 公開動画
├── tables/                      # 集計 CSV と asset checksum
├── manifests/                   # 公開実験 manifest
├── results/                     # ローカル raw result、Git ignore 対象
└── src/
    ├── run_experiments.py       # experiment suite entry point
    ├── trajectory_catalog.py    # 軌道生成と hash 検証
    ├── build_figures.py         # 図と派生 CSV
    ├── build_videos.py          # 動画
    ├── validate_report.py       # レポート整合性検証
    ├── inputs/                  # 2 本の記録 target と固定 driver velocity envelope
    ├── TRAJECTORIES.md          # 70 本の unique trajectory catalog
    ├── pyproject.toml
    └── uv.lock
```

`exp/results/final_report_20260801/` は約 `400 MB` で、公開図と動画の再生成に
必要な summary、metadata、trace を含みます。このディレクトリは Git に
commit されません。2 本の記録入力を除き、軌道はすべてコードから決定論的に
生成されるため、大量の追加 NPZ は保存しません。

## 3. 前提条件

### 3.1 実験設定

実験 package は自己完結しており、deployment dataflow や sibling repository
の設定を読みません。IK の scalar parameter は評価対象 `IKParams` の default
から取得します。Deployment は `--limit-velocity` で IK velocity envelope を
有効化し、`--config` override を指定しないため、評価対象 controller revision
の built-in caps を読みます。実験 profile は同じ mapping を直接注入します。
simulation の downstream limiter は
`exp/src/inputs/experiment_driver_config.yaml` に固定した deployment envelope を
読みます。2 つの envelope は独立に設定・検証されます。本 deployment では
同じ値ですが、常に一致する必要はありません。

### 3.2 System Tool

必要なもの：

- Linux x86_64；
- `uv`（本レポートでは `uv 0.11.24`）；
- MuJoCo の offscreen rendering に対応する OpenGL/EGL 環境；
- `libwebp_anim` encoder を含む `ffmpeg`（動画生成時のみ）；
- 同じ図・動画の typography を再現する DejaVu Sans font。

Python と Python package を手動で install する必要はありません。

## 4. 固定環境の作成

```bash
uv sync --project exp/src --frozen
```

このコマンドは `.python-version` に従って Python `3.14.6` 環境を作成し、
`exp/src/uv.lock` を更新せずに使用します。`openarm-control` は repository root
から editable mode で install され、外部 Python 依存関係は lockfile で固定
されます。

最初に簡易検証を実行します。

```bash
uv run --project exp/src --frozen python exp/src/trajectory_catalog.py \
  --check-report \
  --output /tmp/openarm-trajectories.md

uv run --project exp/src --frozen python exp/src/validate_report.py exp
```

最初のコマンドは全軌道を再生成して hash を照合します。2 番目は report link、
manifest、固定入力、experiment parameter、dependency lockfile を検証します。

## 5. 目標軌道の再生成

全軌道の JSON 記述を表示します。

```bash
uv run --project exp/src --frozen python exp/src/trajectory_catalog.py \
  --format json
```

可読な catalog を再生成し、公開 experiment manifest と照合します。

```bash
uv run --project exp/src --frozen python exp/src/trajectory_catalog.py \
  --check-report \
  --output exp/src/TRAJECTORIES.md
```

generator は全 target を memory 上で生成・sampling し、大量の trajectory NPZ
を書き出しません。catalog は次の内容を報告します。

- `70` 本の unique target trajectory；
- `67` 本の procedurally generated trajectory；
- `2` 本の固定 recorded trajectory；
- 固定軌道を時間圧縮して生成する `1` 本の派生 trajectory。

生成関数、parameter、suite membership、SHA-256 は
[`src/TRAJECTORIES.md`](src/TRAJECTORIES.md) に記載されています。

## 6. MuJoCo 実験の実行

### 6.1 全実験

次のコマンドは全 14 suite（13 dynamic suite と 1 static boundary suite）を
実行し、既定の result directory に保存します。

```bash
uv run --project exp/src --frozen python exp/src/run_experiments.py \
  --suite all \
  --workers 8
```

詳細レポートは、controller profile と trajectory の組み合わせによる `1,921`
回の dynamic simulation と `378` 個の static joint-boundary case を含みます。
実行時間は CPU と worker 数に依存します。dynamic result は profile と scenario
ごとに cache されるため、中断後も同じ output directory から再開できます。

公開結果を保持したまま独立再現を行う場合は、新しい directory を指定します。

```bash
uv run --project exp/src --frozen python exp/src/run_experiments.py \
  --suite all \
  --output-dir exp/results/reproduction \
  --workers 8
```

### 6.2 個別 Experiment Suite

例として、胸前の高速手首動作だけを実行します。

```bash
uv run --project exp/src --frozen python exp/src/run_experiments.py \
  --suite chest \
  --output-dir exp/results/reproduction \
  --workers 8
```

選択可能な suite：

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

単一 suite の出力は debugging と個別解析に適しています。`build_figures.py` と
`build_videos.py` を完全実行するには、result directory に両 script が参照する
全 suite が必要です。

各 dynamic suite は次を生成します。

- `metadata.json`：environment、model、configuration、profile、trajectory、hash；
- `summary.csv`：profile/trajectory ごとの metric；
- `traces/`：図や動画に必要な per-frame target、IK command、driver command、
  simulated actual state。

最上位の `manifest.json` は suite、run count、unique trajectory、固定入力、
dependency-lock hash を集約します。

## 7. 図と Table の再生成

公開 result set を使って `exp/assets/` と派生 CSV を上書き生成します。

```bash
uv run --project exp/src --frozen python exp/src/build_figures.py
```

独立再現結果を使い、まず一時 report directory に出力します。

```bash
uv run --project exp/src --frozen python exp/src/build_figures.py \
  --results exp/results/reproduction \
  --report-dir /tmp/openarm-report-reproduction
```

script はレポートで使用する 21 枚の PNG と対応する集計 CSV を生成します。
`report_traceability.csv` は公開 asset と実験 source の対応を人手で確認した
index であり、plot script は上書きしません。

## 8. 動画の再生成

8 本の controller comparison 動画と、21 動作をまとめた catalog 動画 1 本を
生成します。

```bash
uv run --project exp/src --frozen python exp/src/build_videos.py
```

動作 catalog 動画のみ生成します。

```bash
uv run --project exp/src --frozen python exp/src/build_videos.py \
  --catalog-only
```

controller comparison 動画のみ生成します。

```bash
uv run --project exp/src --frozen python exp/src/build_videos.py \
  --skip-catalog
```

既存 MP4 から GitHub で表示可能な animated preview のみ再生成します。

```bash
uv run --project exp/src --frozen python exp/src/build_videos.py \
  --previews-only
```

独立 result と output directory を使用します。

```bash
uv run --project exp/src --frozen python exp/src/build_videos.py \
  --results exp/results/reproduction \
  --output-dir /tmp/openarm-video-reproduction \
  --preview-dir /tmp/openarm-video-previews
```

renderer は既定で `MUJOCO_GL=egl` を使用し、中間画像列を保持せず RGB frame を
FFmpeg に直接送ります。GitHub は repository 内の MP4 を inline 再生しないため、
script は `exp/assets/video_previews/` に高解像度の animated WebP preview と、
小容量の GIF 互換版も生成します。

## 9. 公開レポートの更新と検証

公開 asset を再生成した後、checksum を更新してレポートを検証します。

```bash
uv run --project exp/src --frozen python exp/src/validate_report.py \
  exp \
  --write-checksums

uv run --project exp/src --frozen python exp/src/validate_report.py exp
```

`exp/` directory から公開ファイルの SHA-256 を直接検証することもできます。

```bash
cd exp
sha256sum --check tables/asset_checksums.sha256
```

検証後は repository root に戻ります。

```bash
cd ..
```

## 10. 推奨する全実行順序

```text
1. uv sync --project exp/src --frozen
2. trajectory_catalog.py --check-report
3. run_experiments.py --suite all              # simulation を再実行する場合
4. build_figures.py
5. build_videos.py
6. validate_report.py exp --write-checksums
7. validate_report.py exp
```

保持済み結果から公開 asset だけを再生成する場合は、手順 3 を省略できます。

## 11. よくある問題

### Configuration Hash が一致しない

repository 内の experiment driver 設定、IK default、または dependency lock
が report manifest と一致しません。本レポートを再現する場合は、manifest に
記録された source revision を復元してください。新しい設定を評価する場合は、
新しい results directory に出力して manifest を保持し、既存の実験定義を
上書きしないでください。

### 固定軌道が見つからない

次のファイルが存在することを確認してください。

```text
exp/src/inputs/near_chest_fast_wrist_roll.npz
exp/src/inputs/fast_retract_elbow_branch.npz
```

各 SHA-256 は `exp/manifests/manifest.json` に記録されています。

### EGL の初期化に失敗する

有効な EGL/OpenGL driver があることを確認してください。simulation metric の
生成自体には動画 rendering は不要です。先に実験と図を実行し、その後 EGL を
利用できる machine で `build_videos.py` を実行できます。

### Matplotlib Cache Directory に書き込めない

書き込み可能な cache directory を指定して図を再生成します。

```bash
MPLCONFIGDIR=/tmp/openarm-matplotlib \
  uv run --project exp/src --frozen python exp/src/build_figures.py
```

### Disk 使用量

公開実験の raw result は約 `400 MB`、独立 Python 環境は約 `420 MB` です。
`exp/results/` と `.venv/` はいずれも Git ignore 対象で、commit には入りません。
