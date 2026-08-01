# OpenArm Control Final Study Log

This file is the durable notebook for the final PR evaluation. It is updated
as experiments complete so interrupted runs can resume without relying on chat
history.

## Study Identity

- Started: 2026-07-30
- Production repository: `/home/hilab/workdir.evaluation/openarm_control`
- Production branch: `pr/control-for-ik`
- Initial production revision: `d006ece`
- Upstream comparison revision: `ori/main` (`d543ced` when the study started)
- MuJoCo model: `openarm_mujoco.v2.openarm_cell_xml()`
- Control period: 4 ms (250 Hz)
- MuJoCo integration period: 1 ms
- QP backend: DAQP

Production source is not modified by this study. Development scripts and
generated artifacts live under `dev/final_study/`.

## Configuration Facts To Preserve

### Current PR defaults

```text
Frame task costs              12 / 1.5
LM / global damping           0.01 / 0.1
Outer period / substeps       0.004 s / 5
Position/orientation budget   0.015 m / 0.25 rad per outer solve
Position activation           smoothstep 0.6 -> 0.9 m/s + 0.006 m latch
Nullspace                     cost 7, rate 1.6 1/s, vmax 1 rad/s
Nullspace rho window          0.02 -> 0.05
Singularity                   rho 0.02 -> 0.08, max approach 0.25 1/s
Joint braking                 distance 0.2 rad, exponent 2, buffer 0.01 rad
Kinetic energy                2e-5
Built-in IK velocity caps     [2, 2, 3.14, 3.14, 6.3, 6.3, 6.3] rad/s
```

The IK velocity envelope remains opt-in via `--limit-velocity`.

### Active evaluation dataflow

`dora-openarm-evaluation-ui/dataflow.yaml` currently passes
`--limit-velocity`, so its IK algorithm and parameters match the `pr_full`
profile. The profile initially named `active_dataflow` in this study was based
on an incorrect configuration reading. It is a valid driver-only velocity
ablation, but not the active deployment; it has been renamed
`driver_only_ablation`.

### Active driver configuration

`openarm_cell_higher_pd_gr00t.yaml` applies downstream command velocity limits:

```text
[2, 2, 3.14, 3.14, 12.6, 12.6, 12.6] rad/s
```

The driver sends position commands with zero velocity and torque feedforward to
the MIT position loop. The MuJoCo model contains the corresponding position
actuators and is used as the dynamic plant in this study.

This means three distinct command paths must be reported:

1. raw IK command;
2. driver-limited position reference;
3. simulated physical state.

## Existing Evidence

The 2026-07-29 reports cover more than 5,000 aggregate rows plus dedicated
frame-error, chest-wrist, normal-workspace, simplification, and boundary
studies. They are treated as prior evidence, not silently mixed with current-PR
results. A result is cited as historical unless it is reproduced on the current
revision.

Reusable trajectory families from the archived study:

- shoulder-height reach beyond the workspace and retract;
- diagonal retract from an extended weak configuration;
- nearly extended vertical/lateral circles;
- normal and extended wrist flips;
- normal-workspace single-arm and bimanual Lissajous motion;
- joint-boundary recovery and braking stress.

## Current-Revision Experiment Matrix

### Baselines

- Active dataflow / full PR: IK velocity and braking plus downstream driver cap.
- Driver-only ablation: PR tasks with no IK velocity limit and a downstream cap.
- Full PR safety: IK velocity/braking enabled, downstream driver cap.
- Upstream-style baseline: full-home PostureTask, damping, normal configuration
  limit, no new tasks or limits, with corrected outer-period semantics.
- No-limit diagnostic: no IK velocity limit and no downstream driver limit.

### Feature Ablations

- recoverable position/velocity limit;
- joint braking;
- exact-nullspace home regulation;
- singularity-approach limit;
- position and orientation frame-error modulation;
- kinetic-energy regularization;
- full-home posture baseline.

### Driver Coupling

- matching IK and driver caps;
- driver-only velocity limiting;
- IK-only velocity limiting;
- no velocity limiting diagnostic;
- current built-in wrist caps versus the active driver wrist caps.

### Parameter Sweeps

- frame position/orientation budgets and task costs;
- nullspace cost and return rate;
- singularity maximum approach rate;
- braking distance;
- kinetic-energy cost;
- velocity cap scaling and proximal-joint variants.

### Required Stress Paths

- straight reach through the reachable boundary;
- left/right biased reach and diagonal retract;
- extended up/down/left/right and circular motion;
- normal and extended wrist roll;
- chest wrist roll with simultaneous outward displacement;
- normal single-arm and bimanual activity;
- synchronized state just outside a position limit.

## Stage Results

### Harness smoke validation

The current-revision harness now records:

1. raw integrated Mink command;
2. downstream driver-limited position reference;
3. independent MuJoCo dynamic state.

Four reach smoke runs (`pr_full` and the driver-only ablation, 0.4 and 0.8 m/s)
completed with zero solver failures. A representative distinction at 0.8 m/s:

```text
profile          raw max dq    driver max dq    plant max dq
pr_full          3.14 rad/s    3.14 rad/s       4.16 rad/s
driver-only A/B  8.53 rad/s    4.31 rad/s       4.26 rad/s
```

The distal driver limit is 12.6 rad/s, so the 4.31 rad/s driver maximum can
occur on a wrist joint while proximal joints remain capped at 2/2/3.14/3.14.
The dynamic plant can transiently exceed the reference velocity because it is
a finite-bandwidth position servo rather than a velocity-controlled plant.

This confirms that IK-side and driver-side saturation must not be conflated in
the final analysis.

### Completed current-PR matrices

The following matrices completed with zero IK solver failures:

| Suite | Profile/scenario combinations | Per-arm rows |
|---|---:|---:|
| Broad feature screening | 630 | 660 |
| One-factor parameter sweep | 528 | 576 |
| Driver coupling | 77 | 84 |
| Mirrored side equivalence | 50 | 52 |
| Static boundary recovery | 378 | 378 |
| Chest translation plus wrist rotation | 210 | 210 |
| Joint braking | 24 | 24 |
| Plant/state robustness | 45 | 45 |
| Combined tuning candidates | 210 | 220 |

The corrected chest suite advances translation and rotation independently and
uses unique trajectory identifiers for forward, lateral, downward, and
diagonal displacements.

### Main baseline result

Across the broad screening set, the full PR profile improves the dynamic
smoothness metrics relative to both upstream-style baselines. Representative
global means:

```text
profile                       actual pos RMSE   actual ddq p99   tail EEF p2p
pr_full                           0.0397 m         30.0 rad/s2      0.0085 m
upstream_style                    0.0935 m         46.4 rad/s2      0.0212 m
upstream_tasks_tuned_costs        0.0567 m         57.9 rad/s2      0.0253 m
```

The upstream-style profile uses the original full-home posture task and normal
configuration limits, but the corrected four-millisecond outer timing. This
keeps the comparison focused on task and limit behavior instead of retaining
the former substep timing bug.

### Feature attribution

- Frame-error modulation is the strongest protection on chest wrist motion.
  In the corrected 14-path chest matrix, disabling it changes position RMSE
  from `0.0198 m` to `0.0695 m`, joint acceleration p99 from `42.2` to
  `51.5 rad/s2`, elbow lateral range from `0.115` to `0.163 m`, and downstream
  driver-limit occupancy from `2.6%` to `53.1%`. Orientation RMSE improves
  because the unbounded solver spends the joint envelope on rotation; this is
  the intended tracking/stability trade.
- Exact-nullspace home regulation primarily controls branch selection.
  Disabling it in the same chest matrix adds `0.046 m` elbow lateral travel,
  about `7.8 rad/s2` joint acceleration p99, and `28.6 percentage points` of
  driver-limit occupancy, while marginally reducing orientation lag.
- The singularity approach limit is localized to extended configurations.
  Removing it lowers minimum geometric singularity ratio by about `0.026` in
  reach tests and adds roughly `7.5 rad/s2` actual acceleration p99 on the
  extended-axis family, with negligible normal-workspace effect.
- Position error limiting is most useful during fast diagonal retract.
  Disabling it improves position tracking by only a few millimeters but adds
  up to `14.9 rad/s2` joint acceleration p99, `4.4 m/s2` elbow acceleration
  p99, and `0.029 m` elbow lateral range.
- Kinetic-energy regularization is a weak tie-breaker. The default
  `2e-5` provides small average gains; larger values smooth more but begin to
  trade tracking accuracy. It is not a dynamics controller.
- Braking preserves joint-limit margin rather than monotonically reducing all
  acceleration. A shorter `0.12 rad` slowdown distance is less intrusive than
  the current conservative `0.20 rad`; both remain valid policy choices.

### Recoverable joint-limit result

For configurations initialized just outside a joint position bound:

```text
limit                              solved cases
recoverable position+velocity      126 / 126
native configuration+velocity       78 / 126
configuration only                 126 / 126
```

At J4 lower bound minus `0.020 rad`, the recoverable limit remains feasible,
requests the capped `3.14 rad/s` recovery, and reduces the violation to
`0.0074 rad` in the tested step sequence. Native separate position and
velocity limits are infeasible; configuration-only requests about `5 rad/s`.

### IK and driver velocity coupling

On the `0.8 m/s` fast diagonal retract:

```text
profile             pos RMSE   actual ddq p99   elbow accel p99
full PR              0.038 m       47.5 rad/s2      10.2 m/s2
driver-only limit     0.050 m       51.1 rad/s2      10.8 m/s2
no velocity limits    0.014 m       68.5 rad/s2      16.3 m/s2
```

Removing the IK velocity envelope can improve raw geometric tracking, but the
downstream driver then clips a different trajectory and increases plant lag.
This does not preserve the QP branch and is not a generally safer replacement.

### Parameter and combined-candidate conclusion

- Position budgets `0.010-0.020 m` occupy a broad performance plateau; the
  current `0.015 m` is not sensitive.
- Orientation budgets `0.15-0.20 rad` reduce chest elbow travel and global
  acceleration at the cost of intentional orientation lag. The current
  `0.25 rad` remains a balanced general default.
- Nullspace costs `7-12` are the useful plateau; higher values add little.
- Singularity maximum approach rate `0.18 rad/s` was smoother than the current
  `0.25 rad/s` in the focused sweep, but the difference is trajectory-specific.
- Kinetic-energy costs `2e-5` to `5e-5` remain appropriately weak.

Combining every one-factor winner did not dominate the current defaults. The
combined candidates reduce global acceleration by roughly `7-9%`, but increase
chest elbow acceleration by roughly `1.8-2.2 m/s2`. Therefore the current PR
configuration remains the report's balanced default; lower orientation budgets
are documented as an application-specific conservative profile.

### Extended video trajectories

Two dedicated trajectories were added under `video_trajectories/` and were
simulated again rather than produced by editing existing footage:

- `reach_deep_start_right_p0p00_v0p80` moves the start and return pose from
  `x=0.410 m` to `x=0.310 m`; the original far target remains at `x=0.710 m`.
  The initial joint configuration was solved to match the new pose, so the
  trajectory starts without artificial frame error.
- `retract_deep_p0p10_v0p80` changes the retract endpoint from `x=0.202 m` to
  `x=0.102 m`; lateral and vertical displacements remain `+0.10 m` and
  `-0.12 m`.

All targeted runs completed without solver failure. On the deeper retract, the
actual elbow world-y path span is `9.3 cm` with the full PR and `11.4 cm`
without frame-error modulation. This absolute span includes the commanded
`10 cm` lateral wrist motion and the intentionally deep folded configuration;
it is not a pure elbow-branch displacement or a prediction of hardware
amplitude. The same-trajectory A/B difference is `2.1 cm`. These traces are
illustrative video stress tests; the broad-study aggregate tables remain based
on the original screening suite. Hardware intervention data could not be
rechecked because `/hdd_data/rollout` was not mounted in the current session.

All retract videos now use this deep endpoint. The PR, no-frame-error,
driver-only velocity, no-nullspace, upstream baseline, and upstream plus IK
velocity-cap profiles were rerun on the positive-lateral path. The ideal
trajectory catalog uses matching positive- and negative-lateral deep retracts.

### Dynamic-model caveats

Adding MuJoCo gravity compensation improves average tracking and command-state
agreement, but does not remove the branch and saturation effects above.
Measured-state delay of `8-20 ms` has a small effect because measured state is
used conservatively by braking and singularity limits rather than continuously
resetting the integrated IK configuration. Command delay and lower actuator
gain increase the expected driver-to-plant lag.

## 2026-07-30 targeted nullspace retuning

The deeper positive-lateral retract was used for 69 additional cost,
return-rate, and maximum-return-speed runs. A geometric elbow-swivel metric
was added using link2/link4/link6 and the arm plane around the shoulder-wrist
axis; this avoids interpreting the commanded 10 cm wrist lateral motion as
pure branch motion.

Increasing only `nullspace_cost` from 7 to 7.5 reduced maximum swivel by just
0.8%, while increasing orientation RMSE by 9.1%, joint acceleration p99 by
11.4%, and elbow acceleration p99 by 13.7%. Costs of 9 and above produced
larger tracking and acceleration trade-offs. The behavior is non-monotonic
because the nullspace term is a weighted objective inside the same constrained
QP, and its instantaneous direction rotates with configuration.

Lowering the proportional return rate improved the stress-path swivel but
also left much larger residual nullspace error after motion stopped. Keeping
the 1.6 s^-1 return rate and capping maximum nullspace return speed was more
semantically appropriate. The best stress candidate was:

```text
cost          7.5
return rate   1.6 s^-1
max speed     0.6 rad/s
```

On the positive deep retract it reduced swivel by 7.3% and elbow acceleration
p99 by 3.3%, with 1.3%/2.6% higher position/orientation RMSE. A subsequent
96-run, 12-scenario cross-validation showed only 2.6% mean swivel reduction,
0.4% higher position and orientation RMSE, 1.5% higher joint acceleration,
and 0.5% higher elbow acceleration. It also slowed post-motion home return and
increased elbow acceleration in one chest outward-flip case by 7.4%.

All 165 targeted runs completed without QP failure. The production defaults
remain `cost=7`, `return_rate=1.6 s^-1`, and `max_speed=1.0 rad/s`.

## 2026-07-31 episode 73 tracking-gap and direct-plant study

The intervention dataset became available at:

```text
/hdd_data/rollout/pillow_0702_tune_llm
```

Episode 73 was analyzed over its complete 142-second right-arm record. Two
tracking errors were kept separate:

```text
same-time error = ||FK(q_measured(t)) - FK(q_action(t))||
cross-track      = distance from FK(q_measured(t)) to the recent command path
```

Right-arm p95/max same-time error was `2.37/5.41 cm`; p95/max cross-track
error was `1.72/3.73 cm`. Therefore the episode contains both acceptable path
lag and genuine short off-path excursions.

The most relevant outward-motion event is event 85. The recorded IK command
path already bends about `9.4 cm` away from its endpoint chord before moving
outward; measured hardware bends about `10.5 cm`. Since raw VR targets are not
recorded, this identifies the problem as pre-driver but cannot yet separate VR
processing from IK command generation.

Feeding the exact recorded joint command into the MuJoCo position-actuator
plant gave:

```text
profile                  right EEF RMSE   aligned RMSE
nominal, no gravity comp      2.57 mm        2.51 mm
no driver cap                 2.58 mm        2.51 mm
20 ms command delay           3.73 mm        2.51 mm
40 ms command delay           6.54 mm        2.51 mm
gravity compensation          6.29 mm        6.25 mm
```

This establishes that the prior large apparent sim-to-real difference came
mainly from replaying a smoothed `FK(action)` intent proxy rather than the
original unrecorded VR target. With identical command input, the simulated
plant reproduces the real path closely.

Recorded hardware SVD energy for events 40/50/85 was:

```text
event    exact z    near-weak V[-2]    other
40         2.5%          46.4%         51.1%
50         5.2%          27.3%         67.5%
85         6.2%          11.9%         82.0%
```

These events have minimum geometric singularity ratios of roughly
`0.16/0.13/0.10`, so they are not straight-arm singularity-limit events.
Exact-nullspace home regulation cannot directly suppress the dominant
near-weak or other task directions.

Focused report:

```text
note/openarm_control/final_report/exp_notes/episode73_tracking_and_sim_gap.md
```

## 2026-07-31 correction: triphasic chest motion and episode 75 plant gap

The previous event-85 interpretation answered an automatic cross-track
ranking, not the user's reported retract-pause-extend symptom. Event 85 is a
positive Current-PR example. A dedicated reach-pattern scan found the actual
triphasic command windows around 34, 98, and 132 seconds:

```text
window   retract   recover   pause    orientation travel
34 s      16.83 cm  18.99 cm  0.296 s  6.36 rad
98 s      17.17 cm  17.98 cm  0.324 s  7.24 rad
132 s     17.96 cm  18.17 cm  0.248 s  6.34 rad
```

All three reversals are present in `FK(action)`. Counterfactual replay kept
the recorded `FK(action)` orientation proxy but replaced position with a
smooth, non-reversing path. Current PR then produced only 3.5-3.7 mm maximum
inward deviation, while mainline plus IK velocity limits produced
3.5-10.8 cm. The output orientation proxy may underestimate the original
target error. This supports the 6D frame bound, but the missing final
Cartesian target still prevents unique attribution between upstream target
processing and the recorded runtime IK.

The July-31 VR implementation was also reconstructed at 60/72/90/120 Hz. Its
repeated-packet/shared-cutoff path raised peak orientation-filter error to
0.74-0.84 rad. Fresh-packet processing with an independent angular cutoff
reduced this to 0.45 rad. No tested pipeline reversed a monotonic position
target, so the old filter is a real input-quality issue but not a sufficient
cause of the 17-18 cm command reversal.

Episode 75 was replayed with the exact recorded joint command. The nominal
MuJoCo plant matched the right hardware EEF with 2.20 mm full-episode RMSE.
For events 5/41:

```text
command chord dip              11.20 / 5.91 cm
hardware below command max      2.03 / 2.00 cm
simulation below command max    2.32 / 2.27 cm
hardware below simulation max   0.88 / 1.00 cm
```

Therefore the previously different synthetic retract used a different IK
command. With identical command input, MuJoCo reproduces the main sink. The
remaining plant offset is still safety-relevant near the tabletop.

Focused report:

```text
note/openarm_control/final_report/exp_notes/residual_failure_followup.md
```

## 2026-07-31 final-report baseline and showcase freeze

The final report comparison was rerun with one strict mainline definition:

```text
ori/main task defaults:
  position/orientation cost = 1 / 1
  LM/global damping         = 0.01 / 0.25
  full-home posture cost    = 0.01

shared comparison mechanics:
  outer/substep dt          = 4.0 / 0.8 ms
  recoverable position + velocity limit
  IK caps                   = [2, 2, 3.14, 3.14, 6.3, 6.3, 6.3] rad/s
  driver caps               = [2, 2, 3.14, 3.14, 12.6, 12.6, 12.6] rad/s

disabled for mainline:
  frame bound, nullspace, singularity approach, kinetic energy,
  joint braking, measured-state safety
```

The corrected mainline was run over all 42 broad trajectories (44 per-arm
rows). Two frozen right-arm target trajectories were then replayed under
Current PR, strict mainline, and one targeted ablation:

```text
chest:
  episode73_triphasic_98s_straight_position
  PR/mainline/no-frame max position error = 1.8 / 17.1 / 4.8 cm

retract:
  recorded_ep75_right_retract30_straight_retract
  PR/mainline/no-nullspace elbow lateral  = 4.0 / 15.5 / 17.6 cm
  PR/mainline/no-nullspace max position   = 4.4 / 5.3 / 3.1 cm
```

Four independent retract windows reproduced a strict-mainline elbow-range
increase of `11.0/10.0/9.8/4.1 cm` relative to Current PR. The final-study
confound check additionally found:

```text
chest max position error:
  PR / no-frame                   = 1.78 / 4.81 cm
  both with braking disabled      = 1.82 / 4.99 cm

retract elbow lateral range:
  PR / no-nullspace               = 4.04 / 17.59 cm
  both with braking disabled      = 4.04 / 17.59 cm

measured-state path disabled:
  chest max position error        = 1.786 cm
  retract core metrics            = unchanged

strict mainline with PR braking:
  chest max position error        = 16.3 cm (17.1 cm without braking)
  retract elbow lateral range     = 15.5 cm (unchanged)
```

The complete final-study addition contains 68 per-arm rows and zero solver
failures.

Artifacts:

```text
dev/final_study/results/final_report_mainline_broad_20260731/
dev/final_study/results/final_report_showcases_20260731/
dev/final_study/results/final_report_retract_replication_20260731/
dev/final_study/results/final_report_showcase_sensitivity_20260731/
note/openarm_control/final_report/assets/33_final_showcase_controller_comparison.png
note/openarm_control/final_report/videos/showcase_selected_v3_chest_ep73_triphasic98_final_report_comparison.mp4
note/openarm_control/final_report/videos/showcase_selected_v4_retract_ep75_event30_final_report_comparison.mp4
```

## 2026-07-31 final-report trajectory/figure audit

Replacing the two final showcase targets changes only their six controller
traces and case-specific numbers. The broad-suite and corrected-chest
ablation matrices are independent and were retained. To prevent the report
from presenting an older representative stress trace as the final showcase,
two trace-matched figures were generated:

```text
note/openarm_control/final_report/assets/34_selected_chest_timeseries_and_path.png
note/openarm_control/final_report/assets/35_selected_retract_timeseries_and_path.png
```

Figures 06-08 are now explicitly labeled as broad/corrected-chest evidence;
figures 33-35 and the two controller videos contain the frozen final targets.
The ideal reference catalog was extended from 19 to 21 segments by appending
the episode-73-derived chest wrist flip and episode-75 event-30 retract:

```text
note/openarm_control/final_report/videos/ideal_reference_trajectory_catalog.mp4
duration = 52.5 s, frames = 1575, 640x480, H.264 yuv420p
```

Each catalog segment is normalized to 2.5 s and now displays its true
playback rate. The final chest/retract segments are `1.58x` and `1.17x`;
the controller showcase videos are `0.5x` and render actual plant state
with a command ghost rather than a solid raw-IK arm.
