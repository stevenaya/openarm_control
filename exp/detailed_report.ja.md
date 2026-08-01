# OpenArm VR IK シミュレーション評価：アルゴリズム、実験、パラメータ

言語：[English](detailed_report.md) | [中文](detailed_report.zh.md) | **日本語**

日付：2026-08-01<br>
Upstream baseline：`d543cedeec5f`（upstream `main` branch、tag `0.2.0`）<br>
評価対象 PR revision：`f983a0eb5cae`（experiment snapshot `d006ece506f2` と記録済み差分 `0ef6a402...` をそのまま commit した revision）<br>
Decision brief：[README.ja.md](README.ja.md) · Asset index：[experiment_index.ja.md](experiment_index.ja.md) · 再現手順：[run_experiment.ja.md](run_experiment.ja.md) · Experiment manifest：[manifest.json](manifests/manifest.json)

## 概要

**現在の PR default を VR IK の deployment baseline として採用することを推奨します。** `42` 本の多様な simulation trajectory において、Mainline-task baseline と比較して position RMSE を約 `71%`、actual joint acceleration p99 を約 `49%`、elbow Cartesian acceleration p99 を約 `47%`、動作終端の EEF residual motion を約 `75%` 低減しました。

主な代償は、高速な手首回転中に生じる一時的な orientation lag です。これは意図した control tradeoff であり、過大な rotation error を 1 回の QP で急いで消化する代わりに、end-effector の position path、shoulder/elbow branch、whole-arm stability を優先します。actual-state metric はすべて MuJoCo simulation に基づき、hardware closed-loop performance を表すものではありません。

## 読み方

本レポートは algorithm、実験条件、完全な A/B、parameter sweep、実装検証、performance data を記録します。merge または deployment の判断だけが必要な場合は、先に [decision brief](README.ja.md) を参照してください。

| 確認したい内容 | 参照先 |
|---|---|
| Upstream の既存機能、比較対象、PR の変更理由 | 第 1 章 |
| PR の統一 solver architecture と algorithm | 第 2 章 |
| Test trajectory、comparison profile、metric の定義 | 第 3 章 |
| Controller 全体と各機能の比較 | 第 4 章 |
| 各 mechanism の個別 A/B | 第 5 章 |
| Parameter 選定、correctness、performance | 第 6 章 |
| 最終結論、制約、deployment recommendation | 第 7 章 |
| Raw result の追跡と再現 | Appendix と [experiment index](experiment_index.ja.md) |

## 1. 研究背景と評価範囲

### 1.1 評価する問い

本研究は現在の OpenArm differential-IK PR について、次の問いを検証します。

1. Mainline の task structure と parameter を用いる共通 baseline に比べ、PR は singularity 近傍、高速 retract、高速 wrist rotation における shoulder/elbow branch の急変を減らせるか。
2. Frame-error modulation、exact-nullspace regulation、singularity-approach limit、recoverable joint envelope、joint braking、kinetic-energy regularization は、それぞれ何を解決するか。
3. 個別 A/B 条件において、QP 内 velocity limit を同じ数値の post-QP driver joint limit で置き換えられるか。
4. 現在の default より単純または積極的で、複数 scenario にわたり一貫して優れる parameter combination は存在するか。
5. 新しい task、limit、Jacobian fast path は `arm_origin` relative-frame semantics、左右対称性、single-arm freezing、正しい `qpos`/`dof` mapping を維持するか。

PR default が simulation 全体の大域最適であるとは主張しません。これは評価対象 code と hardware deployment で使用する configuration です。実験 profile にその値を固定し、一定条件下で利点と代償を評価します。

### 1.2 役割の異なる三つの比較対象

Source baseline、評価対象 PR、実験上の comparison profile は目的が異なります。

| 対象 | 固定 revision または名称 | 本レポートでの意味 |
|---|---|---|
| **Upstream baseline** | `d543cedeec5f742d08a817999d430c4a87f7660f`（upstream `main` branch、tag `0.2.0`） | PR と upstream の merge base。upstream の機能説明と source diff の算出に使用 |
| **評価対象 PR revision** | `f983a0eb5caeb3d96783416b21219a1f31fd3046` | 本レポートが評価する source。実験で使用した `d006ece506f2` と記録済み差分 `0ef6a402...` は、後に変更なしでこの revision として commit |
| **Mainline-task baseline** | `strict_mainline` profile | PR の共通 substep timing と joint envelope の中で mainline の task structure と parameter を復元し、task と regularization の差を分離。`d543cede` の literal replay ではない |

したがって、**Mainline-task baseline** の数値は一つの統制された実験 framework 内での algorithm comparison です。静的 source analysis を除き、本レポートは過去の `d543cede` binary performance と PR を直接比較しません。

### 1.3 Upstream に既にあった機能

`d543cede` は、MuJoCo + Mink differential IK の基本構成を既に備えていました。

- `FrameTask` または `RelativeFrameTask` による 6D end-effector tracking。model に `arm_origin` がある場合は relative coordinates が default で、world-frame operation も選択可能；
- position/orientation task cost、task-level LM damping、global damping から構成される soft QP objective；
- IK 初期構型を target とする default の full-home `PostureTask`；
- `ConfigurationLimit` と optional Mink `VelocityLimit`；
- inactive DoF を固定する `DofFreezingTask`；
- outer event ごとの複数回の `solve_ik()` と configuration integration；
- bimanual FK/IK、driver state からの同期、gripper pass-through、`arm_origin` relative-pose API。

これらは通常 workspace の teleoperation には十分です。現場での異常は主に、大きな 6D target error、reachable-workspace boundary、joint-position limit、actual-state lag が同時に作用する条件で現れました。

### 1.4 元の構成で未対応だった Control Problem

次表は、upstream source structure で未対応であり、現場解析または静的検証から直接示された control problem に限定しています。

| Upstream mechanism | 未対応の問題 | PR の処理 |
|---|---|---|
| Frame task が 6D error 全量を使用 | 高速 wrist rotation、高速 retract、unreachable target が 1 回の outer solve で過大な動作を要求し、joint velocity saturation 時に shoulder/elbow allocation を変える | Position/orientation の total error budget と、speed-activated・latched position clipping |
| Full-home `PostureTask` が joint space 全体に作用 | 7-DoF redundant arm で、home bias が EEF pose を保つ 1D nullspace だけに限定されない | Exact-nullspace home regulation。元 task は保持するが default で無効 |
| Damping は全体の動作量のみ抑制 | Singularity に近づく動作と離れる動作を区別する幾何学的制約がない | One-sided singularity-approach limit |
| Position limit と velocity limit を独立に重ねる | 構型がわずかに範囲外の場合、「1 step で戻す」と one-step velocity bound が競合しうる | Recoverable combined position/velocity envelope |
| Position limit は boundary で displacement だけを制約 | Mechanical limit に達する前に、その方向の許容速度が滑らかにゼロへ近づかない | Distance-dependent joint braking |
| Solve failure 後に limit を外して retry | 最も必要なときに safety constraint が bypass されうる | Outer solve 全体を rollback。unconstrained retry は行わない |
| 各 iteration が full `dt` を使用 | `max_iters` 回の integration が複数 physical period を表し、velocity cap に別の補正が必要 | `dt_sub = dt_outer / max_iters` |
| Active qpos から frozen DoF を間接推定 | Single-arm mode と一般の MuJoCo model では `nq == nv` や両腕 active を仮定できない | Active qpos と tangent-space DoF を明示的に処理 |
| Safety envelope が integrated command のみを見る | Command state が actual state より先行すると braking と singularity activation が遅れる | Measured $q$ は state-aware constraint に保守的に反映し、integrated command は置換しない |
| Euclidean damping のみ | Kinematic cost が近い解の間に configuration-dependent mass-matrix preference がない | Low-weight kinetic-energy regularization |

## 2. PR のソルバ構成と新規メカニズム

### 2.1 統一 QP と Physical Substep

Mink 内部の定数項を除くと、各 substep は次のように表せます。

$$
\min_{\Delta q}
\sum_i \left\|W_i\left(J_i\Delta q-r_i\right)\right\|^2
+\lambda\|\Delta q\|^2,
\qquad G\Delta q\le h.
$$

$\Delta q$ は MuJoCo tangent space 上の one-step displacement、$r_i$ は各 task がその substep で求める correction です。Task は cost function を通じて調整可能な soft objective を与え、limit は $G\Delta q\le h$ を満たす必要がある one-step bound を与えます。

PR は 1 outer control period を全 QP substep に均等配分します。

$$
\Delta t_{\mathrm{sub}}=
\frac{\Delta t_{\mathrm{outer}}}{N}.
$$

現在は $\Delta t_{\mathrm{outer}}=4\,\mathrm{ms}$、$N=5$ なので、各 substep は `0.8 ms`、5 回の integration の合計が 1 physical control period です。

### 2.2 6D End-Effector Error Modulation

$e_p,e_R\in\mathbb{R}^3$ を frame task の position error と orientation error とし、方向を保つ norm saturation を定義します。

$$
\mathrm{sat}_b(x)=
\begin{cases}
x, & \lVert x\rVert\le b,\\
b\dfrac{x}{\lVert x\rVert}, & \lVert x\rVert>b.
\end{cases}
$$

Position と orientation の parameter $B_p,B_R$ は 1 outer solve の total budget で、$N$ substep に均等配分されます。

$$
\bar e_p=\mathrm{sat}_{B_p/N}(e_p),\qquad
\bar e_R=\mathrm{sat}_{B_R/N}(e_R).
$$

Orientation は常に $\bar e_R$ を使用します。Position は target linear speed から $\alpha_p\in[0,1]$ を求め、full error と clipped error を連続的に混合します。

$$
\hat e_p=(1-\alpha_p)e_p+\alpha_p\bar e_p,
\qquad \hat e_R=\bar e_R.
$$

$v_t$ を outer period 間の target-position finite-difference velocity とすると、

$$
u_p=\mathrm{clip}\left(
\frac{\lVert v_t\rVert-v_{\mathrm{slow}}}
{v_{\mathrm{fast}}-v_{\mathrm{slow}}},0,1\right),
\qquad \alpha_p=3u_p^2-2u_p^3.
$$

現在の speed-scheduling interval は `0.6 -> 0.9 m/s` です。Position clipping の activation 後、累積 position error が `6 mm` を超える間は latch が現在の activation を維持し、error が threshold 内へ戻ると解除します。この task は現在の QP が消化する error 量だけを変え、元の target を上書きしません。

### 2.3 Exact-Nullspace Home Regulation

$J_p,J_R$ を geometric linear/angular velocity Jacobian とします。Characteristic length $l_c=0.3\,\mathrm{m}$ により行の数値 scale をそろえます。

$$
J_{\mathrm{norm}}=
\begin{bmatrix}J_p/l_c\\J_R\end{bmatrix}.
$$

この可逆な row scaling は nullspace を変えません。7-DoF arm の $6\times7$ Jacobian を full SVD すると、

$$
J_{\mathrm{norm}}=U\Sigma V^\mathsf{T},
\qquad V=[v_1,\ldots,v_7],\quad z=v_7,
\qquad J_{\mathrm{norm}}z=0.
$$

Home error は MuJoCo configuration difference で計算し、$z$ 方向の scalar component だけを残します。

$$
e_q=q\ominus q_{\mathrm{home}},
\qquad e_{\mathrm{ns}}=z^\mathsf{T}e_q,
$$

$$
v_{\mathrm{ns}}=\mathrm{clip}
\left(-k_{\mathrm{ns}}e_{\mathrm{ns}},
-v_{\mathrm{ns,max}},v_{\mathrm{ns,max}}\right).
$$

Secondary objective は

$$
L_{\mathrm{ns}}=w_{\mathrm{eff}}^2
\left(z^\mathsf{T}\Delta q-v_{\mathrm{ns}}\Delta t_{\mathrm{sub}}\right)^2.
$$

これは 1 次近似で 6D EEF pose を保つ nullspace component だけを regulate し、他方向へ full-home preference を加えません。Singularity 近傍では次を用います。

$$
\rho=\frac{\sigma_{\min}(J_{\mathrm{norm}})}
{\sigma_{\max}(J_{\mathrm{norm}})},
\quad
u_{\mathrm{ns}}=\mathrm{clip}
\left(\frac{\rho-\rho_{\mathrm{low}}}
{\rho_{\mathrm{high}}-\rho_{\mathrm{low}}},0,1\right),
$$

$$
\alpha_{\mathrm{ns}}=3u_{\mathrm{ns}}^2-2u_{\mathrm{ns}}^3,
\qquad w_{\mathrm{eff}}=w_0\sqrt{\alpha_{\mathrm{ns}}}.
$$

Smooth にするのは task weight だけで、$z$ は smooth しないため $Jz=0$ を維持します。$\rho$ が低いと home preference を徐々に弱め、nullspace dimension が変化する直前の不安定な方向が solve を支配することを防ぎます。

### 2.4 One-Sided Singularity-Approach Limit

Singularity は current configuration の **geometric** Jacobian から計算する必要があります。Target-dependent $J_{\log}$ を含む `FrameTask.compute_jacobian()` は singular value を変えるため使用しません。

$$
\rho(q)=\frac{\sigma_{\min}(J_{\mathrm{norm}})}
{\sigma_{\max}(J_{\mathrm{norm}})},
\qquad g=\nabla_q\rho.
$$

実装は joint tangent-space direction に沿う central finite difference で $g$ を計算します。1 次近似では $\dot\rho\approx g^\mathsf{T}\dot q$ であり、$g^\mathsf{T}\dot q<0$ のときだけ arm は singularity に接近します。許容 approach rate は current $\rho$ に応じて滑らかに縮小します。

$$
u_\rho=\mathrm{clip}
\left(\frac{\rho-\rho_{\mathrm{stop}}}
{\rho_{\mathrm{slow}}-\rho_{\mathrm{stop}}},0,1\right),
\qquad
v_{\rho,\mathrm{allowed}}
=v_{\rho,\max}(3u_\rho^2-2u_\rho^3)^p.
$$

QP には one-sided constraint を追加します。

$$
g^\mathsf{T}\Delta q
\ge -v_{\rho,\mathrm{allowed}}\Delta t_{\mathrm{sub}}.
$$

したがって、$\rho$ を下げる component だけが減速され、singularity から離れる motion と equal-singularity contour に沿う motion は制限されません。Measured $q$ がある場合、activation は command state と measured state のうち低い方の $\rho$ を使い、gradient は current QP configuration で linearize したままです。

### 2.5 Recoverable Joint Envelope と Preventive Braking

各 scalar arm joint について、$q_{min},q_{max}$ を position limit、$v_{max}$ を physical velocity limit、$k_q\in(0,1]$ を position gain とします。Recoverable envelope は position recovery と velocity bound を一組の one-step limit に統合します。

$$
\Delta q_{low}=\mathrm{clip}
\left(k_q(q_{min}-q),-v_{max}\Delta t_{\mathrm{sub}},
v_{max}\Delta t_{\mathrm{sub}}\right),
$$

$$
\Delta q_{high}=\mathrm{clip}
\left(k_q(q_{max}-q),-v_{max}\Delta t_{\mathrm{sub}},
v_{max}\Delta t_{\mathrm{sub}}\right).
$$

Joint が position range のわずか外にあり、必要な recovery が velocity-limited one step を超える場合、両 bound は最大安全 recovery step に収束します。Solver は「1 step で戻す」と「step velocity limit 内に保つ」という矛盾した要求を受けません。

Braking 有効時、$m$ を motion direction にある position limit までの effective distance、$d_b$ を braking distance とします。

$$
u=\mathrm{clip}\left(\frac{\max(m,0)}{d_b},0,1\right),
\qquad
v_{\mathrm{allowed}}(m)=v_{max}(3u^2-2u^3)^p.
$$

Distance が小さくなるほど limit 方向の permitted velocity はゼロへ近づき、limit から離れる motion は影響を受けません。Measured $q$ は fixed buffer を差し引いたうえで、command state より保守的な margin を選択できます。

### 2.6 QP Velocity Envelope と Measured State

`--limit-velocity` 有効時、per-joint velocity limit は第 2.5 節の hard QP envelope に入ります。Cartesian task、nullspace task、その他 soft objective は、一つの executable velocity set 内で solve されます。Driver には execution-layer safeguard として同じ post-QP limit を残せます。Driver limit だけを残す方法が等価かどうかは第 5.5 節で検証します。

Measured $q$ は Mink の integrated command configuration を上書きせず、braking margin と singularity-limit activation だけに保守的に影響します。これにより、tick ごとの強制同期で incremental position command を繰り返し縮小せずに、real state で safety boundary を補正できます。

### 2.7 Kinetic-Energy Regularization

Low-weight kinetic task は MuJoCo mass-matrix metric を Hessian に追加します。

$$
H_{\mathrm{kin}}=w_{\mathrm{kin}}
\frac{M(q)}{\Delta t_{\mathrm{sub}}^2}.
$$

これは kinematic cost が近い解の間に弱い preference を与えるだけです。Torque command を生成せず、gravity compensation、contact dynamics、inverse dynamics も含みません。

### 2.8 Solver、Coordinate、Indexing の Semantics

上記 task/limit に加え、PR は次の behavior を変更します。

- constrained QP のどれか 1 substep が失敗した場合、limit を外して retry せず outer solve 全体を rollback；
- MuJoCo `qpos`（`nq`）と tangent-space DoF（`nv`）の index を分離；
- single-arm mode は inactive DoF だけを freeze；
- `sync()` は独立した gripper command を上書きしない；
- upstream の `arm_origin` / `RelativeFrameTask` API を保持し、別の hard-coded coordinate transform は追加しない；
- 条件を満たす static root では relative-Jacobian fast path を使い、moving root では general calculation を維持。

## 3. 実験設計と指標計算

### 3.1 実験環境と Control Chain

#### 3.1.1 Software、Model、Hardware

| 項目 | Version または条件 |
|---|---|
| CPU | Intel Core Ultra 7 265F |
| Kernel | Linux 7.0.0-28 x86_64 |
| Python | 3.14.6 |
| MuJoCo | 3.11.0 |
| Mink | 1.1.0 |
| DAQP | 0.7.2 |
| `openarm_mujoco` | 2.0.1 |
| Model | `openarm_mujoco/v2/cell.xml` |
| Model SHA-256 | `cb0322c264b2acd781ea08e873970ef21dd1db491e64ed0c02844aa8c1a3bfa7` |
| Outer control period | `4 ms`（`250 Hz`） |
| QP substep | `5`、各 `0.8 ms` |
| Random sampling | 無効。記録 seed `0` |

各 suite の revision、dirty-diff hash、command、dependency version、model hash、profile、scenario list は [manifest](manifests/manifest.json) に保存されています。

#### 3.1.2 Frame、用語、Control-Chain State

Target pose、IK command に対応する EEF pose、error は `arm_origin` relative coordinates で表します。MuJoCo plant は world frame で integration します。レポートの elbow Y-Z path は actual elbow joint position を `arm_origin` に変換して算出します。

![VR IK control chain](assets/01_control_layers.png)

Control chain の各量は次を意味します。

- **Target pose**：VR-side processing 後に IK へ渡される end-effector target；
- **Raw IK command**：Mink の 1 outer solve 後の integrated configuration；
- **Driver command**：driver の per-joint velocity limit 後の reference configuration；
- **Actual state**：MuJoCo actuator、inertia、control delay が共同で生む state。

特記がない限り、tracking metric は target と simulated actual state から計算します。Command metric は `raw IK` または `driver` と明記し、IK command を robot state として扱いません。Profile name、target、parameter identifier は code、CSV、動画との対応を保つため原文の英語を使用します。

### 3.2 Default Configuration と Test Trajectory

#### 3.2.1 PR Default

| Parameter | 値 |
|---|---:|
| Position/orientation cost | `12 / 1.5` |
| Global damping / LM damping | `0.1 / 0.01` |
| Full-home posture cost | `0` |
| Total position/orientation error budget | `0.020 m / 0.25 rad` |
| Position speed-scheduling interval | `0.6 -> 0.9 m/s` |
| Position latch threshold | `0.006 m` |
| Nullspace cost / return rate / maximum speed | $8.5 / 1.6\,\mathrm{s}^{-1} / 1.0\,\mathrm{rad/s}$ |
| Nullspace activation interval（dimensionless） | `0.02 -> 0.05` |
| Singularity stop/slow interval（dimensionless） | `0.02 / 0.08` |
| Maximum singularity-approach rate | $0.25\,\mathrm{s}^{-1}$ |
| Joint braking distance / measured-state buffer | `0.20 / 0.01 rad` |
| Joint braking exponent | `2` |
| Kinetic regularization cost | `2e-5` |
| IK velocity limits J1-J7 | `[2, 2, 3.14, 3.14, 6.3, 6.3, 6.3] rad/s` |
| Driver velocity limits J1-J7 | `[2, 2, 3.14, 3.14, 6.3, 6.3, 6.3] rad/s` |

**IK velocity limits** は QP 内の joint constraint、**driver velocity limits** は QP 後の per-joint clipping です。文脈が明確な箇所では IK cap、driver cap と略記します。

#### 3.2.2 Test Trajectory の構成

実験では `70` 本の unique target trajectory を使用します。

1. **Fixed comparison set（42 本）**：headline controller、single-feature ablation、combined candidate が同じ target を使い、直接比較可能にする；
2. **Additional targeted set（28 本）**：左右 mirror、追加の速度・方向、胸前高速 wrist motion、joint braking、recorded-command replay、deep-start singular motion を検証；
3. **Motion-example video（21 clip）**：上記 `70` 本から代表動作を選び、path geometry、movement direction、典型 response を示す。追加の定量 dataset ではない。

各 motion family の本数と目的を次表に示します。完全な scenario name、family count、21 clip の順序は [Experiment Index 第 3 章](experiment_index.ja.md#3-目標軌道) にあります。

| Motion family | Fixed comparison | Additional targeted | Unique total | Video clip | 目的 |
|---|---:|---:|---:|---:|---|
| Reach and extended arm | 19 | 10 | 29 | 9 | Extension singularity、target beyond reach、extended state の translation/wrist rotation |
| Fast retract | 6 | 2 | 8 | 3 | Joint velocity limit 下の nullspace branch と elbow lateral motion |
| Normal/near-chest wrist motion | 12 | 10 | 22 | 7 | Fast rotation と translation が同時に起きる場合の 6D error modulation |
| Normal workspace and bimanual/mirror motion | 5 | 2 | 7 | 2 | Routine tracking、左右 consistency、regression |
| Joint braking | 0 | 4 | 4 | 0 | Physical position limit 近傍の velocity margin |
| **合計** | **42** | **28** | **70** | **21** | - |

![Reference target-trajectory catalog](assets/02_trajectory_catalog.png)

GitHub は repository 内の MP4 を inline 再生しないため、本 report では `8-12 fps` の高解像度 animated WebP preview を、panel layout に応じて `760-980 px` で表示する。Preview をクリックすると元の MP4 にアクセスでき、動画索引には小容量の GIF 互換版も残している。

**動画：21 reference-motion clip。画面内 label は 42 本の fixed set と全 70 target における各 family の本数を表示**

<details>
<summary>動画プレビュー（クリックして展開）</summary>

<a href="videos/ideal_reference_trajectory_catalog.mp4"><img src="assets/video_previews/ideal_reference_trajectory_catalog.webp" alt="動画プレビュー" width="820"></a>

</details>

[高解像度プレビューを開く](assets/video_previews/ideal_reference_trajectory_catalog.webp) · [MP4 をダウンロード](videos/ideal_reference_trajectory_catalog.mp4)

#### 3.2.3 Experiment Suite と Run Count

最終評価は `13` experiment group で構成されます。

| Experiment group | Trajectory | Profile | Dynamic simulation |
|---|---:|---:|---:|
| Screening | 42 | 14 | 588 |
| Parameters | 11 | 54 | 594 |
| Driver coupling | 11 | 4 | 44 |
| Symmetry | 25 | 2 | 50 |
| Chest stress | 14 | 17 | 238 |
| Braking | 4 | 6 | 24 |
| Robustness | 5 | 9 | 45 |
| Combined candidates | 42 | 5 | 210 |
| Recorded benchmarks | 2 | 7 | 14 |
| Accelerated recorded fast-retract | 1 | 2 | 2 |
| Deep-start singularity video | 1 | 2 | 2 |
| Recorded fast-retract nullspace sweep | 1 | 44 | 44 |
| Nullspace cross-validation | 11 | 6 | 66 |
| **合計** | - | - | **1,921** |

Dynamic experiment は `119` controller profile と `70` unique target を含みます。各 group は問いに必要な組み合わせだけを実行し、$119\times70$ の完全な Cartesian product は実行しません。`1,921` 回の dynamic simulation はすべて `solver_failures=0` で完了し、さらに `378` static boundary case があります。Bimanual metric は arm ごとに計算しますが、run count は重複させません。

### 3.3 Comparison Profile と評価指標

#### 3.3.1 主な IK Controller Profile

- **PR default**：第 3.2.1 節の完全な PR 実装と parameter。
- **PR w/o IK velocity limits**：QP velocity envelope を無効にし、driver limit とその他の PR-default task は保持。
- **Mainline-task baseline**（以下 **Mainline baseline**）：mainline の `cost=1/1`、`damping=0.25`、`lm_damping=0.01`、`posture_cost=0.01` を使用し、PR の新 task を無効化。Task structure を分離するため、同じ recoverable joint envelope、velocity limit、`0.8 ms` substep は保持。
- **PR: full-home posture 0.01**：他の PR-default mechanism を保持し、exact-nullspace regulation を full-home `PostureTask(0.01)` で置換。

したがって Mainline baseline は、共通 timing と joint envelope の下で mainline task parameter と structure を比較する統制された algorithm profile であり、旧 commit の binary replay ではありません。

#### 3.3.2 Metric

- `position RMSE/max`：target と actual EEF の Euclidean position error；
- `orientation RMSE`：target と actual EEF の SO(3) geodesic angle；
- $\max_i |\dot q_{i,\mathrm{actual}}|$：各時刻における 7 joint の actual velocity 絶対値の最大；
- $\max_i |\ddot q_{i,\mathrm{actual}}|$：actual velocity を finite difference した後の各時刻の最大絶対値；
- `joint ddq p99`：1 trajectory の全 actual joint acceleration 絶対値の 99th percentile；
- `elbow lateral range`：`arm_origin y` 方向の elbow joint peak-to-peak displacement；
- `elbow acceleration p99`：elbow Cartesian acceleration norm の 99th percentile；
- `tail EEF p2p`：最後の `0.35 s` における actual EEF position の最大 axis-wise peak-to-peak；
- `driver velocity-cap occupancy`：いずれかの driver joint velocity limit が active な時間割合；
- `swivel`：shoulder-wrist axis 周りの elbow angle が初期値から最も離れた量。

Hardware record 由来の 2 benchmark を除き、table metric は trajectory ごとに計算してから target 間で平均します。長い trajectory が frame 数だけで大きな weight を持つことを防ぎます。

#### 3.3.3 Hardware Record 由来の二つの Simulation-Replay Benchmark

二つの target は hardware command record から抽出し、frame ごとに `arm_origin` relative coordinates へ変換しました。全 controller は同じ target、MuJoCo plant、driver velocity limit を使用します。結果は controller structure の比較であり、hardware closed-loop performance ではありません。

| Benchmark | Trajectory の特徴 | 主な評価内容 |
|---|---|---|
| **Near-chest fast wrist-roll** | `3.94 s`；total translation 約 `4.8 cm`；peak linear/angular speed `0.027 m/s / 12.02 rad/s` | 高速 wrist rotation が shoulder、elbow、EEF を意図した position path から引き離すか |
| **Fast-retract elbow-branch** | `2.92 s`；peak linear/angular speed `0.589 m/s / 6.25 rad/s` | Retract speed と joint velocity limit が異なる elbow/nullspace branch を誘発するか |

## 4. 全体結果

### 4.1 Controller 全体の比較

| IK controller | Position RMSE ↓ | Orientation RMSE ↓ | Joint acceleration p99 ↓ | Elbow acceleration p99 ↓ | Elbow lateral range ↓ | Tail EEF p2p ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **PR default** | **3.97 cm** | 31.27° | **29.43 rad/s²** | **5.46 m/s²** | 6.21 cm | **0.84 cm** |
| PR w/o IK velocity limits | 4.01 cm | 30.84° | 31.91 rad/s² | 6.22 m/s² | 6.50 cm | 0.89 cm |
| Mainline-task baseline | 13.78 cm | **11.13°** | 57.49 rad/s² | 10.21 m/s² | **6.12 cm** | 3.37 cm |
| PR: full-home posture 0.01 | 3.94 cm | 29.03° | 33.42 rad/s² | 6.52 m/s² | 7.38 cm | 0.91 cm |

![Overall comparison of four IK controllers](assets/03_headline_baseline_comparison.png)

PR default はすべての個別 metric を最適化するものではありません。一部の orientation lag を意図的に許容し、position error、actual acceleration、動作終端の residual movement を低減します。Mainline baseline は orientation RMSE が低い一方、position RMSE は PR default の約 `3.5` 倍です。

### 4.2 単一機能 Ablation

次の図は、PR default から毎回一つの機能だけを外します。各 column の metric について、cell は次を表します。

$$
100\%\times\frac{m_{\mathrm{without\ feature}}-m_{\mathrm{PR}}}
{|m_{\mathrm{PR}}|}.
$$

Column は position RMSE、orientation RMSE、joint-acceleration p99、elbow-acceleration p99、elbow lateral range、tail EEF movement、driver-cap occupancy の順です。正値は metric の増加を表します。負値は tracking/stability tradeoff による場合があり、必ずしも全体改善を意味しません。

![Single-feature ablation](assets/04_feature_ablation_heatmap.png)

![Feature effects by trajectory family](assets/05_feature_effect_by_trajectory_family.png)

主な観察：

- orientation-error budget は胸前 wrist rotation の安定性を支える主要機能；
- exact-nullspace regulation は retract と near-chest motion の elbow branch、acceleration、driver-cap occupancy を改善；
- singularity limit の効果は reach/extended family に集中し、one-sided geometric constraint の意味と一致；
- braking と kinetic regularization の `42` trajectory 平均での効果は小さい。効果が position limit 近傍または kinematic solution がほぼ等価な条件に集中するためである。

## 5. 個別メカニズムの評価

### 5.1 End-Effector Task Shaping

Algorithm は第 2.2 節で定義しました。本節では、胸前の synthetic stress test と、胸前 wrist rotation、高速 retract の二つの recorded-command replay を用いて error budget を評価します。

#### 5.1.1 胸前 Stress Test

この suite は `14` trajectory で構成されます。**Near-chest roll + diagonal translation** の `9` speed combination（`0.3/0.8/1.2 m/s` と `4/8/12 rad/s`）、peak `1.2 m/s + 8 rad/s` の forward/lateral/downward/diagonal direction variant `4` 本、**Near-chest wrist roll only** `1` 本です。名称は [reference trajectory-catalog video](videos/ideal_reference_trajectory_catalog.mp4) と一致します。Metric は各 trajectory 全体で計算してから `14` 本で平均します。

| Metric | PR default | PR w/o 6D error bound |
|---|---:|---:|
| Position RMSE | **1.95 cm** | 6.93 cm |
| Orientation RMSE | 2.12 rad | **1.66 rad** |
| Joint acceleration p99 | **40.97** | 50.18 |
| Elbow acceleration p99 | **5.31** | 8.05 |
| Elbow lateral range | **10.96 cm** | 16.31 cm |
| Driver velocity-cap occupancy | **3.9%** | 52.3% |
| Tail EEF p2p | **0.24 cm** | 2.34 cm |

Orientation RMSE が `1.66 rad` から `2.12 rad` に増えるのは意図した結果です。高速 rotation が消費する joint capacity を制限し、一時的な orientation lag と引き換えに position error、joint acceleration、elbow excursion、driver-cap occupancy、動作終端の residual movement を低減します。

次図は peak linear/angular speed が `1.2 m/s / 8 rad/s` の **Near-chest roll + diagonal translation** を選び、PR default、PR: orientation budget 0.15 rad、PR w/o posture regulation、PR w/o 6D error bound を比較します。Orientation budget、secondary posture task、complete error modulation が tracking、joint dynamics、driver limit に与える効果を分離します。

$\max_i |\dot q_{i,\mathrm{actual}}|$ と $\max_i |\ddot q_{i,\mathrm{actual}}|$ は、各時刻の 7 joint における actual velocity と acceleration の最大絶対値です。`elbow y-y0` は actual elbow の初期位置からの lateral displacement、`any driver velocity cap active` は少なくとも一つの driver joint が velocity-limited であることを表します。

![Time response during fast wrist rotation with translation](assets/06_chest_wrist_error_modulation_timeseries.png)

6D error bound を外すと、position error、elbow lateral motion、joint acceleration、driver limit が active な時間がすべて明確に増加します。より小さい orientation budget は orientation lag を増やす代わりに、より保守的な joint request を生成します。Posture regulation の無効化は elbow motion を変えますが、6D error modulation の代わりにはなりません。次図は target と actual EEF path を比較します。

![EEF paths during near-chest wrist rotation](assets/07_chest_wrist_eef_paths.png)

**動画：Near-chest roll + diagonal translation、0.5x playback**

<details>
<summary>動画プレビュー（クリックして展開）</summary>

<a href="videos/near_chest_roll_translation_error_bound_comparison.mp4"><img src="assets/video_previews/near_chest_roll_translation_error_bound_comparison.webp" alt="動画プレビュー" width="900"></a>

</details>

[高解像度プレビューを開く](assets/video_previews/near_chest_roll_translation_error_bound_comparison.webp) · [MP4 をダウンロード](videos/near_chest_roll_translation_error_bound_comparison.mp4)

#### 5.1.2 Hardware Record 由来の胸前 Simulation Replay

**Near-chest fast wrist-roll benchmark** は `3.94 s`、total translation 約 `4.8 cm`、peak linear/angular speed `0.027 m/s` と `12.02 rad/s` です。各 target frame は `arm_origin` relative coordinates で表されます。PR default、Mainline baseline、PR w/o 6D error bound が同じ command を replay し、Cartesian tracking、joint acceleration、driver-cap occupancy を比較します。

| Profile | Position RMSE / maximum | Orientation RMSE | Joint acceleration p99 | Driver velocity-cap occupancy |
|---|---:|---:|---:|---:|
| **PR default** | **1.28 / 1.78 cm** | 0.495 rad | **39.3** | **15.3%** |
| Mainline baseline | 5.81 / 17.11 cm | **0.253 rad** | 73.0 | 29.3% |
| PR w/o 6D error bound | 1.79 / 4.81 cm | 0.384 rad | 47.3 | 22.1% |

![Near-chest fast wrist-roll benchmark](assets/34_chest_flip_benchmark_timeseries_and_path.png)

Mainline baseline は orientation RMSE が最小ですが、EEF position path は target から大きく離れてから戻ります。Maximum position error は `17.11 cm`、joint-acceleration p99 は `73.0 rad/s²` に達します。PR default は maximum position error を `1.78 cm` に下げ、より安定した spatial path を保ちますが、一時的な orientation error は増えます。PR w/o 6D error bound はその中間です。

**動画：胸前高速 wrist rotation の controller comparison、hardware record 由来の単一 trajectory、0.5x playback**

<details>
<summary>動画プレビュー（クリックして展開）</summary>

<a href="videos/near_chest_fast_wrist_roll_controller_comparison.mp4"><img src="assets/video_previews/near_chest_fast_wrist_roll_controller_comparison.webp" alt="動画プレビュー" width="980"></a>

</details>

[高解像度プレビューを開く](assets/video_previews/near_chest_fast_wrist_roll_controller_comparison.webp) · [MP4 をダウンロード](videos/near_chest_fast_wrist_roll_controller_comparison.mp4)

この結果は、error modulation の主目的が orientation を急いで追従することではなく、高速 wrist rotation が引き起こす whole-arm instability の抑制であることを示します。遅い orientation tracking は明示的な control tradeoff です。

#### 5.1.3 Hardware Record 由来の高速 Retract Replay

高速 retract における 6D error bound の効果だけを見るため、**Fast-retract elbow-branch benchmark** の完全な position/orientation path を維持しながら、time axis を `1/2` に圧縮して `2x` command speed で controller に入力します。これは hardware-recorded command path の accelerated simulation replay であり、hardware-state video ではありません。二つの列は position/orientation error bound の有無以外すべて同じ parameter です。

**動画：高速 retract の 6D error-bound side-by-side comparison、trajectory 2x、0.5x playback**

<details>
<summary>動画プレビュー（クリックして展開）</summary>

<a href="videos/fast_retract_frame_error_bound_comparison.mp4"><img src="assets/video_previews/fast_retract_frame_error_bound_comparison.webp" alt="動画プレビュー" width="900"></a>

</details>

[高解像度プレビューを開く](assets/video_previews/fast_retract_frame_error_bound_comparison.webp) · [MP4 をダウンロード](videos/fast_retract_frame_error_bound_comparison.mp4)

### 5.2 Redundancy と Singularity

#### 5.2.1 高速 Retract における Exact-Nullspace Regulation

幾何学的定義、home-return speed、singularity activation は第 2.3 節にあります。本実験は、高速 retract 中の elbow branch をこの mechanism が安定化できるか検証します。

Hardware record 由来の **Fast-retract elbow-branch benchmark** は `2.92 s`、peak linear/angular speed `0.589 m/s` と `6.25 rad/s` で、全 frame が `arm_origin` relative coordinates です。三つの profile は完全に同じ target を replay し、他の PR parameter を固定したまま secondary posture task だけを変えます。

1. PR default：exact-nullspace `8.5 / 1.6 / 1.0`；
2. PR w/o posture regulation：`nullspace_cost=0, posture_cost=0`；
3. PR: full-home posture 0.01：`nullspace_cost=0, posture_cost=0.01`。

| Secondary posture regulation | Position RMSE / maximum | Orientation RMSE | Elbow lateral range | Joint acceleration p99 | Elbow acceleration p99 | Driver velocity-cap occupancy |
|---|---:|---:|---:|---:|---:|---:|
| **PR default** | 1.80 / 4.38 cm | **0.107** | **4.04 cm** | 40.90 | 10.21 | 36.9% |
| PR w/o posture regulation | 1.51 / 3.08 cm | 0.115 | 17.59 cm | 38.92 | **9.55** | 33.7% |
| PR: full-home posture 0.01 | **1.49 / 3.00 cm** | 0.115 | 17.61 cm | **38.84** | 9.55 | **33.2%** |

Table は trajectory 全体の集計です。次図は elbow Y-Z path、初期値からの lateral displacement、EEF position error、各時刻の maximum actual joint acceleration を示し、各 secondary posture task が nullspace branch と Cartesian tracking に与える影響を可視化します。

![Posture regulation during fast retraction](assets/08_nullspace_branch_control.png)

Y-Z は `arm_origin` plane における actual elbow path、`y-y0` は初期 lateral position からの displacement です。$\max_i |\ddot q_{i,\mathrm{actual}}|$ は各時刻における 7 joint actual acceleration の最大絶対値です。

**動画：secondary-posture 4-way comparison、hardware record 由来の単一 trajectory、0.5x playback**

<details>
<summary>動画プレビュー（クリックして展開）</summary>

<a href="videos/fast_retract_posture_regulation_comparison.mp4"><img src="assets/video_previews/fast_retract_posture_regulation_comparison.webp" alt="動画プレビュー" width="760"></a>

</details>

[高解像度プレビューを開く](assets/video_previews/fast_retract_posture_regulation_comparison.webp) · [MP4 をダウンロード](videos/fast_retract_posture_regulation_comparison.mp4)

`posture_cost=0.003/0.01/0.03` の elbow lateral range は `17.60/17.61/17.76 cm` で、いずれも exact-nullspace regulation と等価な branch constraint を形成しません。Exact-nullspace regulation は約 `3 mm` の追加 position RMSE を受け入れ、elbow excursion を約 `13.6 cm` 減らします。また、home preference を joint-space の全方向へ直接加えません。

上の実験は secondary posture task だけを変えています。次図は同じ target で PR default、Mainline-task baseline、PR w/o posture regulation を比較し、完全な PR controller を評価します。Panel の順序は同じです。

![Fast-retract elbow-branch benchmark](assets/35_fast_retract_benchmark_timeseries_and_path.png)

**動画：高速 retract controller comparison、hardware record 由来の単一 trajectory、0.5x playback**

<details>
<summary>動画プレビュー（クリックして展開）</summary>

<a href="videos/fast_retract_controller_comparison.mp4"><img src="assets/video_previews/fast_retract_controller_comparison.webp" alt="動画プレビュー" width="980"></a>

</details>

[高解像度プレビューを開く](assets/video_previews/fast_retract_controller_comparison.webp) · [MP4 をダウンロード](videos/fast_retract_controller_comparison.mp4)

#### 5.2.2 Extension と Retract における Singularity-Approach Limit

One-sided singularity constraint は第 2.4 節で定義しました。Target を shoulder level から reachable workspace の外まで伸ばし、その後 retract する trajectory を使い、singularity への接近だけを減速し、離脱時に自動解除されるか検証します。

`0.8 m/s` の正面 extension target では、

| Profile | Minimum $\rho$ | Joint acceleration p99 | Position RMSE |
|---|---:|---:|---:|
| PR default | **0.0323** | **40.6** | 18.88 cm |
| PR w/o singularity limit | 0.0047 | 48.7 | 18.58 cm |

Target は最終的に reachability boundary を越えるため、大きな position RMSE は本実験の主結論ではありません。Blue は extension、gray は retract、yellow は singularity slow zone です。$\rho$ curve は全体を表示し、velocity/acceleration trace は最遠点以降を淡くして extension を強調します。

![Singularity approach during arm extension](assets/09_singularity_reach_timeseries.png)

Limit を有効にすると、blue の extension segment が yellow slow zone に入った後も高い geometric $\rho$ を保ち、最遠点付近の joint acceleration が低下します。Gray の retract segment は同じ one-sided constraint によって減速されません。

動画は同じ最遠 target を持つ deep-start variant です。Start と return endpoint を `arm_origin x=0.410 m` から `x=0.310 m` に移し、最遠点は `x=0.710 m` のままです。First-frame error を避けるため、独立に求めた initial IK configuration を使用します。右上の yellow trace は actual J1 acceleration で、両列の vertical scale は共通です。動画実験は二つの profile の全 time series を別々に保存しますが、前の図と table は standard scenario `reach_right_p0p00_v0p80` を集計しています。

**動画：Extended-arm singular region の A/B、0.5x playback**

<details>
<summary>動画プレビュー（クリックして展開）</summary>

<a href="videos/straight_reach_singularity_limit_comparison.mp4"><img src="assets/video_previews/straight_reach_singularity_limit_comparison.webp" alt="動画プレビュー" width="900"></a>

</details>

[高解像度プレビューを開く](assets/video_previews/straight_reach_singularity_limit_comparison.webp) · [MP4 をダウンロード](videos/straight_reach_singularity_limit_comparison.mp4)

### 5.3 Joint Safety Envelope

#### 5.3.1 Recoverable Joint Envelope

Combined position/velocity bound は第 2.5 節で定義しました。Static boundary case により、configuration が joint limit のわずか外から始まる場合も constraint が feasible で、velocity-bounded rate で valid range へ戻るか検証します。

| Constraint form | Feasible case |
|---|---:|
| **Recoverable position + velocity envelope** | **126 / 126** |
| Independent native position + velocity constraints | 78 / 126 |
| Position constraint only | 126 / 126 |

Position-only constraint は feasible ですが、recovery speed を制限しません。図の “remaining distance outside position limit” は 1 回の完全な `Kinematics.solve()` 後も command configuration が physical boundary の外に残る absolute angular distance です。この solve は 5 回の `0.8 ms` QP substep を含む 1 回の `4 ms` control period です。ゼロは boundary 上または valid range 内、正値は範囲外を表します。

通常の position constraint が即時復帰を要求し、velocity constraint が必要な step を許さない場合、QP は infeasible になりえます。Recoverable envelope は `126/126` case すべてで feasible を保ち、violation を段階的に減らします。Independent constraint は `48` case で失敗し、position-only constraint は overspeed recovery を要求する場合があります。

![Feasibility and response during recovery from a joint-limit violation](assets/11_recoverable_joint_limit.png)

#### 5.3.2 Distance-Dependent Joint Braking

One-sided distance-dependent velocity envelope は第 2.5 節で定義しました。他の PR parameter を固定し、braking distance だけを変えて、physical position margin と早い tracking lag の tradeoff を定量化します。

`12 rad/s` wrist-rotation command の個別実験では、

| Braking profile | Minimum joint margin | Maximum J6 speed | Joint acceleration p99 |
|---|---:|---:|---:|
| Off | 26 mrad | 6.03 rad/s | 56.0 |
| 0.08 rad | 34 mrad | 5.99 rad/s | 55.5 |
| 0.12 rad | 42 mrad | 5.95 rad/s | 54.5 |
| **0.20 rad** | **62 mrad** | 5.90 rad/s | 53.2 |
| 0.30 rad | 90 mrad | 5.80 rad/s | 52.1 |
| PR w/o IK velocity limits / braking | 22 mrad | 6.01 rad/s | 59.3 |

![Distance-dependent joint braking](assets/12_joint_braking_envelope.png)

Curve と scatter point は同じ braking distance に同じ色を使用します。`PR w/o braking` は distance braking だけを無効にし、`PR w/o IK velocity limits / braking` は QP velocity constraint と distance braking の両方を無効にします。Braking distance が大きいほど position margin は早く増えますが、tracking lag も早く始まります。この parameter は mechanical-limit risk に基づいて選ぶべきです。Position limit から遠い通常動作では braking は active にならないため、smoothness 調整には使いません。

### 5.4 Weak Dynamics Regularization

Mass-matrix regularizer の定義と範囲は第 2.7 節にあります。PR default は `2e-5` です。`42` trajectory で無効にすると、joint-acceleration p99 は約 `2.3%`、elbow-acceleration p99 は約 `3.6%`、elbow lateral range は約 `4.2%` 増加します。

この項は torque command を計算せず、gravity compensation、contact dynamics、inverse dynamics も含みません。Dynamics controller ではなく、弱い solution preference です。

### 5.5 QP 内速度制限と Driver 速度制限の A/B

本節は QP 内と driver の velocity limit を比較します。QP envelope を無効にした場合、driver の per-axis clipping は同じ path を一様に遅くするだけなのか、それとも Cartesian task と secondary task の path を変えるのかを検証します。これは独立に設計した A/B であり、第 1.4 節の upstream limitation には含めません。

実験は `1` target trajectory と `3` limit profile を使用します。Target は [reference trajectory-catalog video](videos/ideal_reference_trajectory_catalog.mp4) の右腕 **Fast diagonal retract: lateral +** で、peak linear speed は `0.8 m/s` です。Profile は PR default、PR w/o IK velocity limits、PR w/o velocity limits です。

| Profile | Position RMSE | Actual joint acceleration p99 | Elbow acceleration p99 | Elbow lateral range |
|---|---:|---:|---:|---:|
| **QP + driver velocity limits** | **3.93 cm** | **46.5** | **9.89** | **2.58 cm** |
| PR w/o IK velocity limits | 5.03 cm | 51.1 | 10.78 | 3.84 cm |
| PR w/o velocity limits | **1.36 cm** | 68.4 | 16.34 | 3.71 cm |

PR w/o velocity limits は tracking が良い一方、actual dynamics が大幅に積極的です。PR w/o IK velocity limits は raw IK command と driver command の lead を残し、QP 内で task を executable envelope に合わせて配分する能力を失います。この scenario では position RMSE が `3.93 cm` から `5.03 cm`、elbow lateral range が `2.58 cm` から `3.84 cm` に増え、actual acceleration も増加します。

QP と driver は同じ per-joint numerical limit を使いますが、役割は等価ではありません。QP output が既に limit 内なら driver は通常再 clipping しません。QP limit を無効にすると、driver は optimization 終了後に overspeed command を処理することしかできません。

次図は同じ **Fast diagonal retract: lateral +** を詳しく示します。**Along-track lag** は local command direction に沿った actual EEF と raw IK command の signed spatial distance で、正は遅れ、負は先行を意味します。単位は cm で communication latency ではありません。**Cross-track gap** は local command direction に垂直な距離です。下段は raw IK と actual J1 velocity、gray dotted line は `±2 rad/s` bound です。

全体の maximum absolute along-track lag は PR default の `2.57 cm` から PR w/o IK velocity limits の `9.15 cm` へ、maximum cross-track gap は `2.68 cm` から `9.43 cm` へ増加します。この A/B では driver per-axis clipping は uniform time scaling と等価ではありません。

![Effect of moving velocity limiting out of IK](assets/10_driver_limit_coupling_timeseries.png)

**動画：高速 retract の QP velocity-limit A/B。両列で driver limit 有効、0.5x playback**

<details>
<summary>動画プレビュー（クリックして展開）</summary>

<a href="videos/fast_retract_ik_velocity_limit_comparison.mp4"><img src="assets/video_previews/fast_retract_ik_velocity_limit_comparison.webp" alt="動画プレビュー" width="900"></a>

</details>

[高解像度プレビューを開く](assets/video_previews/fast_retract_ik_velocity_limit_comparison.webp) · [MP4 をダウンロード](videos/fast_retract_ik_velocity_limit_comparison.mp4)

## 6. パラメータ、正しさ、性能

本章では default parameter の選定根拠、implementation correctness、plant robustness、solve time をまとめます。

### 6.1 Nullspace Parameter Sweep と Cross-Validation

最初に **Fast-retract elbow-branch benchmark** で `44` profile の cost、return rate、maximum return speed を sweep し、その後 `11` cross-scenario trajectory で `6` candidate を検証します。`cost` は singularity activation 前の base weight、`return rate` は unsaturated home-return rate、`max return speed` は return velocity の bound です。

図の `swivel departure` は shoulder-wrist axis 周りの elbow rotation が初期値から最大に離れた量、`dynamic cost` は $\alpha_{ns}$ による modulation 後の real-time weight、`driver velocity-cap occupancy` は少なくとも一つの driver joint が limited である時間割合です。

この図では maximum return speed を `1.0 rad/s` に固定します。Panel title と blue outline cell は PR default（`cost=8.5, return=1.6`）の absolute value、他 cell は relative change です。6 metric はすべて低い方が良いため、green は低下、red は増加を表します。Panel ごとに unit と color scale が異なり、panel 間で直接比較できません。

![Nullspace-regulation parameter sweep](assets/17_nullspace_targeted_tuning.png)

この trajectory では `cost=12, return=0.8, max=0.6` が elbow lateral range を約 `3.52 cm` まで下げますが、複数 scenario で tracking、joint dynamics、nullspace branch、cap occupancy を同時に改善する candidate はありません。

Generalization を確認するため、次図は七つの motion type にまたがる `11` direction/speed variant で candidate を cross-validate します。**Near-chest roll + diagonal translation**、**Extended-arm circle**、**Extended-arm wrist roll**、**Bimanual workspace motion**、**Normal-workspace wrist roll**、**Arm extension beyond reach**、**Fast diagonal retract** です。Row は `cost / return rate / max return speed` で識別し、CSV の `c/r/v` も同じ略記です。各 cell は PR default に対する変化で、green は改善、red は悪化です。

![Nullspace-parameter cross-validation](assets/18_nullspace_cross_validation.png)

**動画：高速 retract の exact-nullspace parameter 2x2、hardware record 由来の単一 trajectory、0.5x playback**

<details>
<summary>動画プレビュー（クリックして展開）</summary>

<a href="videos/fast_retract_nullspace_parameter_comparison.mp4"><img src="assets/video_previews/fast_retract_nullspace_parameter_comparison.webp" alt="動画プレビュー" width="760"></a>

</details>

[高解像度プレビューを開く](assets/video_previews/fast_retract_nullspace_parameter_comparison.webp) · [MP4 をダウンロード](videos/fast_retract_nullspace_parameter_comparison.mp4)

見た目の差は小さい結果です。

### 6.2 単一 Parameter 感度

Sweep は `11` representative trajectory を使用します。Arm extension beyond reach `3` 本、Fast diagonal retract `2` 本、Extended-arm circle `2` 本、および Extended-arm wrist roll、Normal-workspace wrist roll、Near-chest roll + diagonal translation、Bimanual workspace motion を各 `1` 本です。対象 parameter は total position/orientation error budget、nullspace cost/return rate/maximum speed、singularity envelope、braking distance、kinetic cost、QP velocity-limit scale です。

各 curve は one-dimensional sweep です。Horizontal-axis parameter だけを変え、他は PR default に固定します。Position/orientation budget は 1 outer solve の total budget で、5 substep に配分されます。QP velocity-limit scale は 7 IK cap すべてを同時に scale し、driver limit は固定します。

Black vertical dashed line は PR default、star はその metric で sampling した点の最良値です。RMSE、acceleration、lateral range、cap occupancy は低い方がよく、minimum $\rho$ と physical joint margin は高い方がよいため、metric ごとの best parameter は通常一致しません。

![Single-parameter sensitivity](assets/13_parameter_sweep_summary.png)

主な傾向：

- Position budget が小さすぎると normal-workspace lag が増え、大きすぎると高速 retract の branch stability が弱まる；
- Orientation budget が大きいと胸前 position error と driver-cap occupancy が明確に増え、小さいと orientation lag が増える；
- 高い nullspace cost は cross-scenario stability を保証しない。Return rate と maximum speed は、それぞれ unsaturated regime と saturated regime でのみ支配的；
- 小さい singularity-approach rate は保守的だが、extension を早く制限する；
- Braking distance は position margin と tracking lag の tradeoff を決める；
- QP velocity limit を緩めると一部の command-tracking error は減るが、actual acceleration と driver-cap occupancy は増える。

### 6.3 Combined Candidate

Combined candidate は controller 全体比較と同じ `42` trajectory を使用します。Extension、retract、extended-arm circle/axial motion、normal/extended configuration の wrist rotation、normal single/bimanual workspace motion、translation を伴う near-chest wrist rotation を含みます。図の略記は、`orientation` が total orientation-error budget、`singularity rate` が maximum singularity-approach rate、`nullspace` が exact-nullspace base cost、`braking` が braking distance、`kinetic` が kinetic-regularization cost です。完全な parameter name と unit は図の下にあります。

図は metric heatmap と Pareto plot を組み合わせます。A-D は single-parameter trend から選んだ representative configuration で、combined space の winner ではありません。最も sensitive な parameter だけを tightening、中程度の joint adjustment、積極的な joint adjustment、default orientation budget を保ちながら secondary mechanism だけを変更、という範囲を含みます。Heatmap は PR default に対する percentage change で、green は低下、red は増加です。Position RMSE と joint-acceleration p99 の Pareto plot では左下がよい領域です。

![Combined-parameter tradeoffs](assets/14_combined_tuning_candidates.png)

4 candidate は joint-acceleration p99 を `2.0–7.1%` 下げる一方、elbow-acceleration p99 を `5.1–8.2%`、position RMSE を `0.3–0.7%` 増やします。Tracking、nullspace branch、dynamics、driver-cap occupancy を同時に改善する candidate はないため、PR default を保持します。

### 6.4 Implementation Correctness と Robustness

#### 6.4.1 左右 Arm と Mirror

`25` mirrored scenario で左右 consistency を検証します。Exact-mirror aggregate position RMSE は `0.6815/0.6821 cm`、joint-acceleration p99 は `86.9166/86.9161 rad/s²` で、差は numerical-noise scale です。

![Left/right symmetry and robustness](assets/15_symmetry_and_robustness.png)

#### 6.4.2 Single-Arm Mode と Relative Coordinates

Code test は次を含みます。

- `arm_origin` の `RelativeFrameTask` と world-frame formulation の等価性；
- static-root Jacobian fast path と moving-root general path の数値一致；
- right-only/left-only mode が inactive DoF だけを freeze；
- MuJoCo `nq` qpos index と `nv` dof index の分離；
- measured-state mapping が gripper command を上書きしないこと；
- constrained-solve failure 時の outer step 全体の rollback。

`openarm_mujoco>=2.0.1` は `arm_origin` site を提供し、control package は別の origin を hard-code しません。明示的な `origin_frame=world` は world-frame behavior を維持します。

#### 6.4.3 Plant Robustness

Robustness experiment は actuator gain、state/command delay、state rate/dropout、gravity compensation を変化させます。相対的な傾向が一つの idealized plant に依存しないかを検証するもので、hardware friction や structural compliance の同定ではありません。Measured state は現在 braking と singularity constraint に保守的に反映され、Mink の integrated command を連続的に上書きしません。

### 6.5 Solve Time

次図は `42` mixed screening trajectory の統計です。各 outer solve は `5` QP substep を含みます。単一の bimanual microbenchmark ではありません。

| Profile | Mean | p95 |
|---|---:|---:|
| PR default | 1.140 ms | 1.195 ms |
| PR w/o IK velocity limits | 1.077 ms | 1.145 ms |
| PR w/o 6D error bound | 1.103 ms | 1.163 ms |
| Mainline baseline | 0.483 ms | 0.511 ms |

![Solve time](assets/16_solver_timing.png)

同じ machine の別の bimanual relative-coordinate microbenchmark では、static-root Jacobian fast path により 5-substep outer solve が約 `2.04 ms` から `1.39 ms` へ短縮され、joint-command array は同一でした。二つの timing dataset は trajectory と aggregation method が異なるため、直接比較できません。

## 7. 結論、制約、Deployment Recommendation

### 7.1 主な結論

**現在の PR default を VR IK の deployment baseline として保持することを推奨します。** 個々の metric に対する simulation optimum ではありませんが、EEF position path、shoulder/elbow branch、joint dynamics、physical envelope、高速 rotation 時の orientation lag の間で安定した compromise です。

- `42` common comparison trajectory で orientation-error bound を無効にすると、position RMSE は `125%`、tail EEF residual movement は `181%`、driver-cap occupancy は `212%` 増加。
- Hardware record 由来の高速 retract benchmark で exact-nullspace task は elbow lateral range を、posture regulation なしの `17.59 cm` から `4.04 cm` へ低減。Low-weight full-home task は等価な constraint を形成しない。
- Singularity limit は `0.8 m/s` extension trajectory の minimum $\rho$ を `0.0047` から `0.0323` へ上げ、joint-acceleration p99 を `48.7` から `40.6 rad/s²` へ低減。
- Recoverable envelope は範囲外からの recovery case を `126/126` 解き、独立 position/velocity constraint は `78/126` のみ。
- Joint braking は主に position limit 近傍の margin を増やし、kinetic regularization は弱い preference のみを提供。
- 第 5.5 節の個別 A/B は、その scenario では post-QP limit だけを残すと同じ動作を一様に遅くするのではなく、path と dynamics が変わることを示す。
- `54` single-parameter profile、`5` combined profile、`44` fast-retract nullspace profile のいずれも、複数 scenario で PR default を全面的に上回る candidate を示さない。

### 7.2 現在の制約

1. 胸前で高速 wrist rotation と急な outward translation が同時に起きると、一部の weakly controllable direction が大きな shoulder/elbow motion を依然として励起しうる。Orientation budget を小さくすると position は保護されるが orientation lag が増える。
2. Measured state は integrated command を連続的に上書きしないため、command lead over actual state は蓄積しうる。一方、tick ごとの直接同期は incremental position command を縮小する。完全な処理には独立した reference governor または state prediction が必要。
3. Joint envelope は collision constraint を提供しない。Extension/retract 中の table/base clearance には独立した Cartesian-space または collision layer が必要。
4. MuJoCo actuator は hardware friction、structural compliance、motor bandwidth、firmware position loop、gravity compensation を完全には再現しない。Simulation は controller structure の比較には適するが、hardware validation の代替ではない。
5. 初期の hardware record は raw VR、filtered target、final limited target を同期保存しておらず、少数の three-stage wrist-rotation anomaly を一意に帰属できない。

### 7.3 Deployment Recommendation

- PR default を説明可能な baseline として保持し、deployment argument を記録済み code default と同期させる。
- QP velocity envelope を保持し、PR w/o IK velocity limits で置き換えない。
- Total orientation-error budget を保持する。胸前高速 rotation を安定化する主要 mechanism である。
- Exact-nullspace posture task を保持する。高速 retract benchmark で full-home posture の position RMSE が低いことは、等価な elbow-branch constraint であることを意味しない。
- One-sided singularity constraint を保持し、その geometric Jacobian を target-dependent FrameTask Jacobian に戻さない。
- Recoverable position/velocity envelope を feasibility correction として hard constraint のまま保持する。
- Diagnosis のため braking は独立に無効化できる。`0.20 rad` は現在の deployment safety value であり、すべての hardware に対する普遍的 optimum ではない。
- Kinetic task は low-cost weak regularizer として保持する。明確な dynamics benefit には、この cost を上げるのではなく acceleration/torque-level whole-body QP または MPC が必要。

## Appendix：Data と再現

- Experiment code、固定入力、local raw result：[`exp/src`](src/README.md)
- Top-level experiment manifest：[manifests/manifest.json](manifests/manifest.json)
- Figure、video、CSV mapping：[experiment_index.ja.md](experiment_index.ja.md)
- Motion type 全体の controller comparison：[tables/headline_baseline_means.csv](tables/headline_baseline_means.csv)
- Single-feature ablation：[tables/ablation_relative_effects_percent.csv](tables/ablation_relative_effects_percent.csv)
- 高速 retract posture-regulation comparison：[tables/branch_regulation_frozen_metrics.csv](tables/branch_regulation_frozen_metrics.csv)
- File-level traceability：[tables/report_traceability.csv](tables/report_traceability.csv)
