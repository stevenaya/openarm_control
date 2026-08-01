#!/usr/bin/env python3
"""Study episode-73 wrist flips with retract-pause-extend IK commands.

Raw VR poses were not recorded.  The first stage therefore detects the
three-phase pattern in FK(action q) without claiming that it was absent from
the operator input.  The second stage is a controller counterfactual: it keeps
the recorded orientation history but replaces the command-side position loop
with a smooth start-to-end path before rerunning IK.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import recorded_intervention_study as recorded
import study

EPISODE = 73
SIDE = "right"
DT = study.CONTROL_DT
OUTPUT_DIR = HERE / "results" / "episode73_triphasic_20260731"
RECORDED_DIR = HERE / "results" / "recorded_interventions_20260731"
FINAL_ASSET_DIR = (
    HERE.parents[2]
    / "note"
    / "openarm_control"
    / "final_report"
    / "assets"
)
WINDOWS_S = (
    ("triphasic_34s", 33.876, 36.556),
    ("triphasic_98s", 97.703, 100.343),
    ("triphasic_132s", 131.946, 133.934),
)


@dataclass(frozen=True)
class Pattern:
    """One command-side retract-pause-extend sequence."""

    name: str
    start: int
    end: int
    start_s: float
    end_s: float
    retract_m: float
    recover_m: float
    min_reach_s: float
    pause_duration_s: float
    peak_angular_speed_rad_s: float
    orientation_travel_rad: float
    command_path_length_m: float
    hardware_cross_track_max_m: float


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _record() -> recorded.ArmRecord:
    return recorded.load_record(
        EPISODE,
        SIDE,
        RECORDED_DIR / "cache",
    )


def _index(record: recorded.ArmRecord, seconds: float) -> int:
    return int(np.argmin(np.abs(record.time - seconds)))


def detect_patterns() -> list[Pattern]:
    record = _record()
    command = recorded._smooth(record.action_world_eef, 0.032)
    actual = recorded._smooth(record.obs_world_eef, 0.032)
    reach = np.linalg.norm(command - record.shoulder, axis=1)
    reach_rate = recorded._derivative(reach[:, None], 0.052)[:, 0]
    angular_speed = recorded._angular_speed(record.action_pose)
    patterns: list[Pattern] = []
    for name, start_s, end_s in WINDOWS_S:
        start = _index(record, start_s)
        end = _index(record, end_s) + 1
        local_reach = reach[start:end]
        minimum_local = int(np.argmin(local_reach))
        minimum = start + minimum_local
        pre_max = float(np.max(local_reach[: minimum_local + 1]))
        post_max = float(np.max(local_reach[minimum_local:]))
        retract = pre_max - float(reach[minimum])
        recover = post_max - float(reach[minimum])
        pause_mask = (
            np.abs(reach_rate[minimum:end]) < 0.035
        )
        pause_runs = recorded._runs(pause_mask)
        pause_duration = (
            max((run_end - run_start) * DT for run_start, run_end in pause_runs)
            if pause_runs
            else 0.0
        )
        command_steps = np.linalg.norm(np.diff(command[start:end], axis=0), axis=1)
        cross_track, _ = _local_path_match(
            command[start:end],
            actual[start:end],
        )
        patterns.append(
            Pattern(
                name=name,
                start=start,
                end=end,
                start_s=float(record.time[start]),
                end_s=float(record.time[end - 1]),
                retract_m=retract,
                recover_m=recover,
                min_reach_s=float(record.time[minimum]),
                pause_duration_s=float(pause_duration),
                peak_angular_speed_rad_s=float(
                    np.max(angular_speed[start:end])
                ),
                orientation_travel_rad=float(
                    np.sum(angular_speed[start:end]) * DT
                ),
                command_path_length_m=float(np.sum(command_steps)),
                hardware_cross_track_max_m=float(np.max(cross_track)),
            )
        )
    _write_csv(
        OUTPUT_DIR / "patterns.csv",
        [asdict(pattern) for pattern in patterns],
    )
    return patterns


def _local_path_match(
    command: np.ndarray,
    actual: np.ndarray,
    *,
    max_shift_s: float = 0.40,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest local command point for each measured point."""
    count = command.shape[0]
    best = np.full(count, np.inf, dtype=np.float64)
    best_index = np.arange(count, dtype=np.int64)
    max_shift = round(max_shift_s / DT)
    index = np.arange(count)
    for shift in range(-max_shift, max_shift + 1):
        candidate = index - shift
        valid = (candidate >= 0) & (candidate < count)
        delta = actual[valid] - command[candidate[valid]]
        squared = np.einsum("ij,ij->i", delta, delta)
        update = squared < best[valid]
        selected = index[valid][update]
        best[selected] = squared[update]
        best_index[selected] = candidate[valid][update]
    return np.sqrt(best), best_index


def _smooth_path(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, count)
    progress = 3.0 * np.square(u) - 2.0 * np.power(u, 3)
    return start[None, :] + progress[:, None] * (end - start)[None, :]


def _scenario(
    record: recorded.ArmRecord,
    pattern: Pattern,
) -> study.Scenario:
    prefix = round(0.50 / DT)
    suffix = round(0.80 / DT)
    source_pose = record.action_pose[pattern.start : pattern.end].copy()
    source_pose[:, :3] = _smooth_path(
        source_pose[0, :3],
        source_pose[-1, :3],
        source_pose.shape[0],
    )
    target = np.concatenate(
        [
            np.repeat(source_pose[:1], prefix, axis=0),
            source_pose,
            np.repeat(source_pose[-1:], suffix, axis=0),
        ],
        axis=0,
    )
    phase = np.concatenate(
        [
            np.zeros(prefix, dtype=np.int8),
            np.ones(source_pose.shape[0], dtype=np.int8),
            np.full(suffix, 2, dtype=np.int8),
        ]
    )
    initial = record.action_q[pattern.start].copy()
    return study.Scenario(
        name=f"episode73_{pattern.name}_straight_position",
        family="recorded_chest_counterfactual",
        mode=SIDE,
        speed=float(
            np.max(
                np.linalg.norm(
                    recorded._derivative(source_pose[:, :3]),
                    axis=1,
                )
            )
        ),
        times=np.arange(target.shape[0], dtype=np.float64) * DT,
        phase=phase,
        target_right=target,
        target_left=np.repeat(target[:1], target.shape[0], axis=0),
        initial_right=initial,
        initial_left=study.HOME_Q.copy(),
        description=(
            "Recorded episode-73 orientation with a smooth start-to-end "
            "position path; raw VR target was unavailable."
        ),
    )


def _profiles() -> list[study.Profile]:
    return [
        study.make_profile(
            "pr_recorded_config",
            nullspace_cost=8.0,
            description="Current PR with episode-73 dataflow settings.",
        ),
        study.make_profile(
            "pr_orientation_budget_015",
            nullspace_cost=8.0,
            frame_orientation_error_limit=0.15,
            description="Current PR with a 0.15 rad total orientation budget.",
        ),
        study.make_profile(
            "pr_orientation_budget_010",
            nullspace_cost=8.0,
            frame_orientation_error_limit=0.10,
            description="Current PR with a 0.10 rad total orientation budget.",
        ),
        study.make_profile(
            "pr_no_frame_error",
            nullspace_cost=8.0,
            frame_position_error_limit=0.0,
            frame_orientation_error_limit=0.0,
            description="Current PR with both frame-error budgets disabled.",
        ),
        study.make_profile(
            "pr_no_ik_velocity",
            nullspace_cost=8.0,
            velocity_caps=None,
            limit_style="configuration_only",
            description=(
                "Current PR tasks without the IK velocity envelope; driver "
                "velocity limiting remains enabled."
            ),
        ),
        study.make_profile(
            "pr_position_cost_20",
            nullspace_cost=8.0,
            position_cost=20.0,
            description="Current PR with position cost increased to 20.",
        ),
        study.make_profile(
            "pr_no_nullspace",
            nullspace_cost=0.0,
            description="Current PR without nullspace home regulation.",
        ),
        study.upstream_style_ik_velocity_profile("mainline_ik_velocity"),
        study.upstream_style_profile("mainline"),
    ]


def _trace_path(
    output_dir: Path,
    profile: study.Profile,
    scenario: study.Scenario,
) -> Path:
    matches = list(
        (output_dir / "traces").glob(
            f"*_{profile.name}_{scenario.name}.npz"
        )
    )
    if not matches:
        raise FileNotFoundError(
            f"No trace for {profile.name}/{scenario.name}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _orientation_error(target: np.ndarray, command: np.ndarray) -> np.ndarray:
    target_rotation = Rotation.from_quat(target[:, [4, 5, 6, 3]])
    command_rotation = Rotation.from_quat(command[:, [4, 5, 6, 3]])
    return (target_rotation.inv() * command_rotation).magnitude()


def _repeat_edges(values: np.ndarray, prefix: int, suffix: int) -> np.ndarray:
    return np.concatenate(
        [
            np.repeat(values[:1], prefix, axis=0),
            values,
            np.repeat(values[-1:], suffix, axis=0),
        ],
        axis=0,
    )


def _save_recorded_video_trace(
    record: recorded.ArmRecord,
    pattern: Pattern,
    scenario: study.Scenario,
) -> None:
    prefix = round(0.50 / DT)
    suffix = round(0.80 / DT)
    window = slice(pattern.start, pattern.end)
    target_pose = _repeat_edges(record.action_pose[window], prefix, suffix)
    command_q = _repeat_edges(record.action_q[window], prefix, suffix)
    actual_q = _repeat_edges(record.obs_q[window], prefix, suffix)
    command_pose = _repeat_edges(record.action_pose[window], prefix, suffix)
    actual_pose = _repeat_edges(record.obs_pose[window], prefix, suffix)
    command_elbow = _repeat_edges(record.action_elbow[window], prefix, suffix)
    actual_elbow = _repeat_edges(record.obs_elbow[window], prefix, suffix)
    payload = {
        "times": scenario.times,
        "phase": scenario.phase,
        "right_target_pose": target_pose,
        "right_target": target_pose,
        "right_command_q": command_q,
        "right_driver_q": command_q,
        "right_actual_q": actual_q,
        "right_command_pose": command_pose,
        "right_driver_pose": command_pose,
        "right_actual_pose": actual_pose,
        "right_command_elbow": command_elbow,
        "right_driver_elbow": command_elbow,
        "right_actual_elbow": actual_elbow,
    }
    np.savez_compressed(
        OUTPUT_DIR
        / "traces"
        / f"recorded_recorded_hardware_{scenario.name}.npz",
        **payload,
    )


def replay() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    record = _record()
    patterns = detect_patterns()
    profiles = _profiles()
    rows: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    for pattern in patterns:
        scenario = _scenario(record, pattern)
        _save_recorded_video_trace(record, pattern, scenario)
        scenarios.append(
            {
                **asdict(pattern),
                "scenario": scenario.name,
                "target_semantics": (
                    "recorded command orientation + smooth start/end position"
                ),
            }
        )
        print(f"Scenario {scenario.name}", flush=True)
        for profile in profiles:
            print(f"  {profile.name}", flush=True)
            study.run_cached(
                profile,
                scenario,
                OUTPUT_DIR,
                save_full_trace=True,
            )
            trace = np.load(_trace_path(OUTPUT_DIR, profile, scenario))
            phase = trace["phase"] == 1
            target = trace["right_target_pose"][phase]
            command = trace["right_command_pose"][phase]
            actual = trace["right_actual_pose"][phase]
            command_position_error = np.linalg.norm(
                command[:, :3] - target[:, :3],
                axis=1,
            )
            actual_position_error = np.linalg.norm(
                actual[:, :3] - target[:, :3],
                axis=1,
            )
            target_radius = np.linalg.norm(target[:, :3], axis=1)
            command_radius = np.linalg.norm(command[:, :3], axis=1)
            command_dq = trace["right_command_dq"][phase]
            utilization = np.abs(command_dq) / np.asarray(study.CONTROL_CAPS)
            rows.append(
                {
                    "scenario": scenario.name,
                    "profile": profile.name,
                    "target_position_path_m": float(
                        np.sum(
                            np.linalg.norm(
                                np.diff(target[:, :3], axis=0),
                                axis=1,
                            )
                        )
                    ),
                    "command_position_error_rmse_m": float(
                        np.sqrt(np.mean(np.square(command_position_error)))
                    ),
                    "command_position_error_max_m": float(
                        np.max(command_position_error)
                    ),
                    "actual_position_error_rmse_m": float(
                        np.sqrt(np.mean(np.square(actual_position_error)))
                    ),
                    "actual_position_error_max_m": float(
                        np.max(actual_position_error)
                    ),
                    "command_inward_deviation_max_m": float(
                        np.max(target_radius - command_radius)
                    ),
                    "command_orientation_error_rmse_rad": float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    _orientation_error(target, command)
                                )
                            )
                        )
                    ),
                    "peak_joint_velocity_utilization": float(
                        np.max(utilization)
                    ),
                    "joint_velocity_saturated_pct": float(
                        100.0 * np.mean(np.any(utilization >= 0.999, axis=1))
                    ),
                    "solver_failed_count": int(
                        np.count_nonzero(trace["solver_failed"])
                    ),
                }
            )
    _write_csv(OUTPUT_DIR / "counterfactual_metrics.csv", rows)
    (OUTPUT_DIR / "scenarios.json").write_text(
        json.dumps(scenarios, indent=2),
        encoding="utf-8",
    )


def plot_summary() -> None:
    """Plot the actual three-phase commands and controller counterfactuals."""
    record = _record()
    patterns = detect_patterns()
    profiles = {profile.name: profile for profile in _profiles()}
    selected = (
        "pr_recorded_config",
        "pr_orientation_budget_010",
        "pr_no_frame_error",
        "mainline_ik_velocity",
    )
    colors = {
        "pr_recorded_config": "#1f77b4",
        "pr_orientation_budget_010": "#2ca02c",
        "pr_no_frame_error": "#d62728",
        "mainline_ik_velocity": "#9467bd",
    }
    labels = {
        "pr_recorded_config": "Current PR (0.25 rad)",
        "pr_orientation_budget_010": "Current PR (0.10 rad)",
        "pr_no_frame_error": "PR, frame bound off",
        "mainline_ik_velocity": "Mainline + IK cap",
    }

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for row, pattern in enumerate(patterns):
        scenario = _scenario(record, pattern)
        recorded_time = (
            record.time[pattern.start : pattern.end]
            - record.time[pattern.start]
        )
        command = record.action_world_eef[pattern.start : pattern.end]
        actual = record.obs_world_eef[pattern.start : pattern.end]
        shoulder = record.shoulder[pattern.start : pattern.end]
        command_reach = np.linalg.norm(command - shoulder, axis=1)
        actual_reach = np.linalg.norm(actual - shoulder, axis=1)
        angular_speed = recorded._angular_speed(record.action_pose)[
            pattern.start : pattern.end
        ]

        axis = axes[row, 0]
        axis.plot(
            recorded_time,
            100.0 * command_reach,
            color="#d62728",
            linewidth=2.0,
            label="FK(IK command)",
        )
        axis.plot(
            recorded_time,
            100.0 * actual_reach,
            color="#00a6a6",
            linewidth=1.8,
            label="hardware",
        )
        minimum_time = pattern.min_reach_s - pattern.start_s
        axis.axvline(minimum_time, color="#555555", linestyle="--", linewidth=1.0)
        axis.set_ylabel("shoulder-to-EEF reach [cm]")
        axis.set_title(
            f"{pattern.name.removeprefix('triphasic_')}: recorded "
            "retract-pause-extend"
        )
        axis.grid(alpha=0.25)

        axis = axes[row, 1]
        command_progress = np.linalg.norm(command - command[0], axis=1)
        actual_progress = np.linalg.norm(actual - actual[0], axis=1)
        axis.plot(
            recorded_time,
            100.0 * command_progress,
            color="#d62728",
            linewidth=2.0,
            label="command travel",
        )
        axis.plot(
            recorded_time,
            100.0 * actual_progress,
            color="#00a6a6",
            linewidth=1.8,
            label="hardware travel",
        )
        twin = axis.twinx()
        twin.plot(
            recorded_time,
            angular_speed,
            color="#f0a000",
            linewidth=1.1,
            alpha=0.8,
            label="command angular speed",
        )
        axis.set_ylabel("distance from start [cm]")
        twin.set_ylabel("angular speed [rad/s]", color="#a66f00")
        axis.set_title(
            f"rotation travel {pattern.orientation_travel_rad:.1f} rad; "
            f"pause {pattern.pause_duration_s:.2f} s"
        )
        axis.grid(alpha=0.25)

        axis = axes[row, 2]
        for profile_name in selected:
            trace = np.load(
                _trace_path(
                    OUTPUT_DIR,
                    profiles[profile_name],
                    scenario,
                )
            )
            phase = trace["phase"] == 1
            target = trace["right_target_pose"][phase, :3]
            solved = trace["right_command_pose"][phase, :3]
            target_radius = np.linalg.norm(target, axis=1)
            solved_radius = np.linalg.norm(solved, axis=1)
            counterfactual_time = (
                trace["times"][phase] - trace["times"][phase][0]
            )
            axis.plot(
                counterfactual_time,
                100.0 * (target_radius - solved_radius),
                color=colors[profile_name],
                linewidth=1.6,
                label=labels[profile_name],
            )
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        axis.set_ylabel("command inward deviation [cm]")
        axis.set_title("Same rotation, smooth non-reversing position target")
        axis.grid(alpha=0.25)

    for axis in axes[-1, :]:
        axis.set_xlabel("window time [s]")
    axes[0, 0].legend(loc="best", fontsize=8)
    handles, legend_labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        fontsize=9,
    )
    fig.suptitle(
        "Episode 73: the actual triphasic command vs an IK counterfactual",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "episode73_triphasic_summary.png"
    fig.savefig(output, dpi=180)
    FINAL_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FINAL_ASSET_DIR / "30_episode73_triphasic_summary.png",
        dpi=180,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("detect", "replay", "plot", "all"),
        default="all",
        nargs="?",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage in {"detect", "all"}:
        patterns = detect_patterns()
        for pattern in patterns:
            print(pattern)
    if args.stage in {"replay", "all"}:
        replay()
    if args.stage in {"plot", "all"}:
        plot_summary()


if __name__ == "__main__":
    main()
