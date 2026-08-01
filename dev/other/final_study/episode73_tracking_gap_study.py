#!/usr/bin/env python3
"""Separate path lag from off-path motion in intervention episode 73.

The recorder stores the joint-space IK command before the hardware driver and
the measured motor state after the plant.  It does not store the raw VR target.
This study therefore evaluates command-following behavior only:

* temporal error compares measured and commanded FK at the same timestamp;
* cross-track error measures distance to the recent commanded end-effector path;
* progress lag measures how far behind the measured state is along that path;
* radial contradiction detects measured motion toward the shoulder while the
  command is moving away from it.

It also feeds the recorded joint command directly into the same MuJoCo position
actuator model used by the main simulation study.  This isolates plant-model
error from IK/controller differences.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import recorded_intervention_study as recorded
import study

DT = study.CONTROL_DT
EPISODE = 73
SIDES = ("right", "left")
OUTPUT_DIR = (
    HERE
    / "results"
    / "episode73_tracking_gap_20260731"
)
FINAL_ASSET_DIR = (
    HERE.parents[2]
    / "note"
    / "openarm_control"
    / "final_report"
    / "assets"
)
DIRECT_TRACE_DIR = (
    HERE
    / "results"
    / "current_pr_20260730"
    / "episode73_direct_plant"
    / "traces"
)
DIRECT_EVENT_IDS = (40, 50)


@dataclass
class TrackingSeries:
    """Derived command-following signals for one arm."""

    temporal_error_m: np.ndarray
    cross_track_error_m: np.ndarray
    nearest_command_index: np.ndarray
    progress_lag_m: np.ndarray
    match_lag_s: np.ndarray
    orientation_error_rad: np.ndarray
    command_speed_m_s: np.ndarray
    actual_speed_m_s: np.ndarray
    velocity_cosine: np.ndarray
    command_reach_rate_m_s: np.ndarray
    actual_reach_rate_m_s: np.ndarray
    opposite_motion: np.ndarray
    radial_contradiction: np.ndarray


@dataclass(frozen=True)
class Excursion:
    """One interval with substantial off-path or opposite motion."""

    side: str
    rank: int
    start: int
    end: int
    core_start: int
    core_end: int
    start_s: float
    end_s: float
    duration_s: float
    score: float
    max_temporal_error_m: float
    max_cross_track_error_m: float
    p95_cross_track_error_m: float
    max_progress_lag_m: float
    median_match_lag_s: float
    max_orientation_error_rad: float
    opposite_duration_s: float
    radial_contradiction_duration_s: float
    min_actual_reach_rate_m_s: float
    max_command_reach_rate_m_s: float


@dataclass(frozen=True)
class PlantProfile:
    """One direct-command dynamic replay configuration."""

    name: str
    gravity_compensation: bool = False
    command_delay_s: float = 0.0
    actuator_kp_scale: float = 1.0
    actuator_kv_scale: float = 1.0
    driver_velocity_limit: bool = True


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _orientation_error(
    command_pose: np.ndarray,
    actual_pose: np.ndarray,
) -> np.ndarray:
    command = Rotation.from_quat(command_pose[:, [4, 5, 6, 3]])
    actual = Rotation.from_quat(actual_pose[:, [4, 5, 6, 3]])
    return (command.inv() * actual).magnitude()


def _local_path_match(
    command: np.ndarray,
    actual: np.ndarray,
    *,
    max_lag_s: float = 1.5,
    max_lead_s: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """Match each actual point to the nearest recent command sample.

    A local temporal window prevents looped command paths from matching an
    unrelated visit to the same Cartesian point.  Positive lag means the plant
    most closely resembles an older command and is therefore behind.
    """
    count = command.shape[0]
    best_squared = np.full(count, np.inf, dtype=np.float64)
    best_index = np.arange(count, dtype=np.int64)
    lag_ticks = round(max_lag_s / DT)
    lead_ticks = round(max_lead_s / DT)
    indices = np.arange(count, dtype=np.int64)
    for lag in range(-lead_ticks, lag_ticks + 1):
        command_index = indices - lag
        valid = (command_index >= 0) & (command_index < count)
        if not np.any(valid):
            continue
        delta = actual[valid] - command[command_index[valid]]
        squared = np.einsum("ij,ij->i", delta, delta)
        update = squared < best_squared[valid]
        if not np.any(update):
            continue
        valid_indices = indices[valid][update]
        best_squared[valid_indices] = squared[update]
        best_index[valid_indices] = command_index[valid][update]
    return np.sqrt(best_squared), best_index


def compute_tracking(
    record: recorded.ArmRecord,
) -> TrackingSeries:
    """Compute temporal, path-relative, and directional tracking metrics."""
    command = recorded._smooth(record.action_world_eef, 0.032)
    actual = recorded._smooth(record.obs_world_eef, 0.032)
    shoulder = recorded._smooth(record.shoulder, 0.032)
    command_velocity = recorded._derivative(command, 0.044)
    actual_velocity = recorded._derivative(actual, 0.044)
    command_speed = np.linalg.norm(command_velocity, axis=1)
    actual_speed = np.linalg.norm(actual_velocity, axis=1)
    velocity_dot = np.einsum("ij,ij->i", command_velocity, actual_velocity)
    velocity_cosine = velocity_dot / np.maximum(
        command_speed * actual_speed,
        1.0e-9,
    )

    cross_track, nearest_index = _local_path_match(command, actual)
    command_step = np.linalg.norm(np.diff(command, axis=0), axis=1)
    command_arclength = np.concatenate([[0.0], np.cumsum(command_step)])
    progress_lag = command_arclength - command_arclength[nearest_index]
    match_lag_s = (np.arange(command.shape[0]) - nearest_index) * DT

    command_reach = np.linalg.norm(command - shoulder, axis=1)
    actual_reach = np.linalg.norm(actual - shoulder, axis=1)
    command_reach_rate = recorded._derivative(
        command_reach[:, None],
        0.052,
    )[:, 0]
    actual_reach_rate = recorded._derivative(
        actual_reach[:, None],
        0.052,
    )[:, 0]
    opposite = (
        (command_speed > 0.05)
        & (actual_speed > 0.03)
        & (velocity_cosine < -0.20)
    )
    radial_contradiction = (
        (command_reach_rate > 0.035)
        & (actual_reach_rate < -0.035)
    )

    return TrackingSeries(
        temporal_error_m=np.linalg.norm(actual - command, axis=1),
        cross_track_error_m=cross_track,
        nearest_command_index=nearest_index,
        progress_lag_m=progress_lag,
        match_lag_s=match_lag_s,
        orientation_error_rad=_orientation_error(
            record.action_pose,
            record.obs_pose,
        ),
        command_speed_m_s=command_speed,
        actual_speed_m_s=actual_speed,
        velocity_cosine=velocity_cosine,
        command_reach_rate_m_s=command_reach_rate,
        actual_reach_rate_m_s=actual_reach_rate,
        opposite_motion=opposite,
        radial_contradiction=radial_contradiction,
    )


def detect_excursions(
    side: str,
    record: recorded.ArmRecord,
    tracking: TrackingSeries,
) -> list[Excursion]:
    """Find and rank off-path or directionally contradictory intervals."""
    core_mask = (
        (tracking.cross_track_error_m > 0.025)
        | (
            tracking.opposite_motion
            & (tracking.temporal_error_m > 0.030)
        )
        | tracking.radial_contradiction
    )
    runs = recorded._merge_runs(recorded._runs(core_mask), 0.20)
    context = round(0.45 / DT)
    candidates: list[Excursion] = []
    for core_start, core_end in runs:
        if (core_end - core_start) * DT < 0.08:
            continue
        start = max(0, core_start - context)
        end = min(record.time.shape[0], core_end + context)
        core = slice(core_start, core_end)
        duration = (core_end - core_start) * DT
        max_cross = float(np.max(tracking.cross_track_error_m[core]))
        max_temporal = float(np.max(tracking.temporal_error_m[core]))
        opposite_duration = float(
            np.count_nonzero(tracking.opposite_motion[core]) * DT
        )
        contradiction_duration = float(
            np.count_nonzero(tracking.radial_contradiction[core]) * DT
        )
        score = (
            40.0 * max_cross
            + 20.0 * max_temporal
            + 2.0 * opposite_duration
            + 3.0 * contradiction_duration
            + 0.25
            * float(np.max(tracking.orientation_error_rad[core]))
        )
        candidates.append(
            Excursion(
                side=side,
                rank=0,
                start=start,
                end=end,
                core_start=core_start,
                core_end=core_end,
                start_s=float(record.time[start]),
                end_s=float(record.time[end - 1]),
                duration_s=duration,
                score=score,
                max_temporal_error_m=max_temporal,
                max_cross_track_error_m=max_cross,
                p95_cross_track_error_m=float(
                    np.percentile(tracking.cross_track_error_m[core], 95)
                ),
                max_progress_lag_m=float(
                    np.max(tracking.progress_lag_m[core])
                ),
                median_match_lag_s=float(
                    np.median(tracking.match_lag_s[core])
                ),
                max_orientation_error_rad=float(
                    np.max(tracking.orientation_error_rad[core])
                ),
                opposite_duration_s=opposite_duration,
                radial_contradiction_duration_s=contradiction_duration,
                min_actual_reach_rate_m_s=float(
                    np.min(tracking.actual_reach_rate_m_s[core])
                ),
                max_command_reach_rate_m_s=float(
                    np.max(tracking.command_reach_rate_m_s[core])
                ),
            )
        )
    candidates.sort(key=lambda value: value.score, reverse=True)
    return [
        Excursion(**{**asdict(value), "rank": rank})
        for rank, value in enumerate(candidates, start=1)
    ]


def _plot_overview(
    records: dict[str, recorded.ArmRecord],
    tracking: dict[str, TrackingSeries],
    excursions: list[Excursion],
    path: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    colors = {"right": "#d1495b", "left": "#00798c"}
    for side in SIDES:
        time = records[side].time
        values = tracking[side]
        color = colors[side]
        axes[0].plot(
            time,
            100.0 * values.temporal_error_m,
            color=color,
            alpha=0.85,
            label=f"{side} temporal",
        )
        axes[0].plot(
            time,
            100.0 * values.cross_track_error_m,
            color=color,
            linestyle="--",
            alpha=0.85,
            label=f"{side} cross-track",
        )
        axes[1].plot(
            time,
            values.command_speed_m_s,
            color=color,
            alpha=0.55,
            label=f"{side} command",
        )
        axes[1].plot(
            time,
            values.actual_speed_m_s,
            color=color,
            linestyle="--",
            alpha=0.85,
            label=f"{side} measured",
        )
        axes[2].plot(
            time,
            values.orientation_error_rad,
            color=color,
            label=side,
        )
        if side == "right":
            axes[3].fill_between(
                time,
                0.0,
                0.42 * values.opposite_motion.astype(float),
                color="#d1495b",
                alpha=0.55,
                label="velocity points opposite to command",
            )
            axes[3].fill_between(
                time,
                0.58,
                0.58
                + 0.42 * values.radial_contradiction.astype(float),
                color="#f6bd60",
                alpha=0.75,
                label="moves toward chest while command moves out",
            )
    for event in excursions[:10]:
        if event.side != "right":
            continue
        for axis in axes:
            axis.axvspan(
                records[event.side].time[event.core_start],
                records[event.side].time[event.core_end - 1],
                color="#f6bd60",
                alpha=0.10,
            )
    axes[0].set_ylabel("error [cm]")
    axes[1].set_ylabel("EEF speed [m/s]")
    axes[2].set_ylabel("orientation [rad]")
    axes[3].set_ylabel("event")
    axes[3].set_xlabel("episode time [s]")
    axes[3].set_yticks(
        (0.21, 0.79),
        ("opposite", "radial"),
    )
    axes[3].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.grid(alpha=0.20)
        axis.legend(loc="upper right", ncol=2, fontsize=8)
    fig.suptitle(
        "Episode 73: temporal lag versus true off-path motion",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_excursion_catalog(
    records: dict[str, recorded.ArmRecord],
    tracking: dict[str, TrackingSeries],
    excursions: list[Excursion],
    path: Path,
) -> None:
    selected = excursions[:6]
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    for axis, event in zip(axes.flat, selected, strict=False):
        record = records[event.side]
        values = tracking[event.side]
        window = slice(event.start, event.end)
        command = record.action_world_eef[window]
        actual = record.obs_world_eef[window]
        shoulder = record.shoulder[event.core_start]
        axis.plot(
            command[:, 0],
            command[:, 2],
            color="#d1495b",
            linewidth=2.0,
            label="command FK",
            zorder=4,
        )
        axis.plot(
            actual[:, 0],
            actual[:, 2],
            color="#00798c",
            linewidth=1.8,
            label="measured FK",
            zorder=5,
        )
        axis.scatter(
            [shoulder[0]],
            [shoulder[2]],
            marker="x",
            color="#303030",
            s=45,
            label="shoulder",
            zorder=6,
        )
        core = slice(
            event.core_start - event.start,
            event.core_end - event.start,
        )
        axis.scatter(
            actual[core, 0],
            actual[core, 2],
            c=values.cross_track_error_m[
                event.core_start : event.core_end
            ],
            cmap="magma",
            s=8,
            zorder=7,
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world z [m]")
        axis.set_title(
            f"#{event.rank} {event.side} {event.start_s:.2f}-{event.end_s:.2f}s\n"
            f"cross={100 * event.max_cross_track_error_m:.1f} cm, "
            f"same-time={100 * event.max_temporal_error_m:.1f} cm, "
            f"opposite={event.opposite_duration_s:.2f}s"
        )
        axis.legend(fontsize=8, loc="best")
    for axis in axes.flat[len(selected) :]:
        axis.axis("off")
    fig.suptitle(
        "Largest episode-73 command-following excursions (side view)",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plant_profiles() -> list[PlantProfile]:
    return [
        PlantProfile("nominal_no_gc"),
        PlantProfile("no_driver_cap", driver_velocity_limit=False),
        PlantProfile("delay_8ms", command_delay_s=0.008),
        PlantProfile("delay_20ms", command_delay_s=0.020),
        PlantProfile("delay_40ms", command_delay_s=0.040),
        PlantProfile("gravity_compensation", gravity_compensation=True),
        PlantProfile(
            "kp_075_kv_100",
            actuator_kp_scale=0.75,
        ),
        PlantProfile(
            "kp_100_kv_150",
            actuator_kv_scale=1.50,
        ),
    ]


def _run_plant(
    records: dict[str, recorded.ArmRecord],
    profile: PlantProfile,
) -> dict[str, np.ndarray]:
    count = min(record.time.shape[0] for record in records.values())
    plant = study.DynamicPlant(
        records["right"].obs_q[0],
        records["left"].obs_q[0],
        gravity_compensation=profile.gravity_compensation,
        actuator_kp_scale=profile.actuator_kp_scale,
        actuator_kv_scale=profile.actuator_kv_scale,
    )
    for side in SIDES:
        plant.data.qvel[plant._dofs_by_side[side]] = records[side].obs_dq[0]
    mujoco.mj_forward(plant.model, plant.data)

    delay_ticks = round(profile.command_delay_s / DT)
    delay_queue: deque[dict[str, np.ndarray]] = deque()
    previous = {
        side: records[side].action_q[0].copy()
        for side in SIDES
    }
    outputs: dict[str, np.ndarray] = {}
    for side in SIDES:
        outputs[f"{side}_q"] = np.empty((count, 7), dtype=np.float64)
        outputs[f"{side}_dq"] = np.empty((count, 7), dtype=np.float64)
        outputs[f"{side}_pose"] = np.empty((count, 7), dtype=np.float64)
        outputs[f"{side}_elbow"] = np.empty((count, 3), dtype=np.float64)
        outputs[f"{side}_driver_q"] = np.empty(
            (count, 7),
            dtype=np.float64,
        )
        outputs[f"{side}_limited"] = np.zeros(
            (count, 7),
            dtype=bool,
        )

    first_command = {
        side: records[side].action_q[0].copy()
        for side in SIDES
    }
    for _ in range(delay_ticks):
        delay_queue.append(
            {side: value.copy() for side, value in first_command.items()}
        )

    for index in range(count):
        driver_command: dict[str, np.ndarray] = {}
        for side in SIDES:
            desired = records[side].action_q[index]
            if profile.driver_velocity_limit:
                limited, active = study._limit_driver_command(
                    desired,
                    previous[side],
                    study.DRIVER_CAPS,
                )
            else:
                limited = desired.copy()
                active = np.zeros(7, dtype=bool)
            previous[side] = limited
            driver_command[side] = limited
            outputs[f"{side}_driver_q"][index] = limited
            outputs[f"{side}_limited"][index] = active

        delay_queue.append(
            {
                side: command.copy()
                for side, command in driver_command.items()
            }
        )
        applied = delay_queue.popleft()
        plant.set_command(applied)
        plant.step_control_period()
        for side in SIDES:
            outputs[f"{side}_q"][index] = plant.q(side)
            outputs[f"{side}_dq"][index] = plant.dq(side)
            outputs[f"{side}_pose"][index] = plant.ee(side)
            outputs[f"{side}_elbow"][index] = plant.elbow(side)
    return outputs


def _best_alignment(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    max_shift_s: float = 0.12,
) -> tuple[int, float]:
    max_shift = round(max_shift_s / DT)
    best_shift = 0
    best_rmse = np.inf
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            ref = reference[shift:]
            value = candidate[: candidate.shape[0] - shift]
        else:
            ref = reference[: reference.shape[0] + shift]
            value = candidate[-shift:]
        rmse = float(np.sqrt(np.mean(np.square(ref - value))))
        if rmse < best_rmse:
            best_rmse = rmse
            best_shift = shift
    return best_shift, best_rmse


def _plant_metrics(
    profile: PlantProfile,
    records: dict[str, recorded.ArmRecord],
    output: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for side in SIDES:
        record = records[side]
        count = output[f"{side}_q"].shape[0]
        sim_pose = output[f"{side}_pose"]
        actual_pose = record.obs_pose[:count]
        shift, aligned_position_rmse = _best_alignment(
            actual_pose[:, :3],
            sim_pose[:, :3],
        )
        q_shift, aligned_q_rmse = _best_alignment(
            record.obs_q[:count],
            output[f"{side}_q"],
        )
        rows.append(
            {
                **asdict(profile),
                "side": side,
                "position_rmse_m": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                sim_pose[:, :3] - actual_pose[:, :3]
                            )
                        )
                    )
                ),
                "position_rmse_aligned_m": aligned_position_rmse,
                "best_position_shift_ms": shift * DT * 1000.0,
                "joint_rmse_rad": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                output[f"{side}_q"]
                                - record.obs_q[:count]
                            )
                        )
                    )
                ),
                "joint_rmse_aligned_rad": aligned_q_rmse,
                "best_joint_shift_ms": q_shift * DT * 1000.0,
                "orientation_rmse_rad": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                _orientation_error(
                                    sim_pose,
                                    actual_pose,
                                )
                            )
                        )
                    )
                ),
                "driver_limit_active_pct": float(
                    100.0
                    * np.mean(
                        np.any(output[f"{side}_limited"], axis=1)
                    )
                ),
                "driver_limit_joint1_pct": float(
                    100.0
                    * np.mean(output[f"{side}_limited"][:, 0])
                ),
                "driver_limit_joint4_pct": float(
                    100.0
                    * np.mean(output[f"{side}_limited"][:, 3])
                ),
            }
        )
    return rows


def _plot_plant_case(
    event: Excursion,
    record: recorded.ArmRecord,
    tracking: TrackingSeries,
    plant_outputs: dict[str, dict[str, np.ndarray]],
    path: Path,
) -> None:
    window = slice(event.start, event.end)
    time = record.time[window] - record.time[event.start]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    command = record.action_pose[window, :3]
    actual = record.obs_pose[window, :3]
    axes[0, 0].plot(
        command[:, 0],
        command[:, 2],
        color="#d1495b",
        linewidth=2.2,
        label="recorded command FK",
    )
    axes[0, 0].plot(
        actual[:, 0],
        actual[:, 2],
        color="#00798c",
        linewidth=2.0,
        label="real measured FK",
    )
    for name, color in (
        ("nominal_no_gc", "#edae49"),
        ("delay_20ms", "#5f4b8b"),
        ("gravity_compensation", "#59a14f"),
    ):
        sim = plant_outputs[name][f"{event.side}_pose"][window, :3]
        axes[0, 0].plot(
            sim[:, 0],
            sim[:, 2],
            color=color,
            linewidth=1.3,
            label=f"MuJoCo {name}",
        )
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].set_xlabel("relative x [m]")
    axes[0, 0].set_ylabel("relative z [m]")
    axes[0, 0].set_title("End-effector path")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        time,
        100.0 * tracking.temporal_error_m[window],
        label="real same-time error",
        color="#00798c",
    )
    axes[0, 1].plot(
        time,
        100.0 * tracking.cross_track_error_m[window],
        label="real cross-track error",
        color="#e15759",
    )
    for name, color in (
        ("nominal_no_gc", "#edae49"),
        ("delay_20ms", "#5f4b8b"),
        ("gravity_compensation", "#59a14f"),
    ):
        sim = plant_outputs[name][f"{event.side}_pose"][window, :3]
        error = np.linalg.norm(sim - command, axis=1)
        axes[0, 1].plot(
            time,
            100.0 * error,
            label=f"{name} same-time",
            color=color,
            alpha=0.8,
        )
    axes[0, 1].set_ylabel("error [cm]")
    axes[0, 1].set_title("Tracking error")
    axes[0, 1].legend(fontsize=8)

    for joint, axis in ((0, axes[1, 0]), (3, axes[1, 1])):
        axis.plot(
            time,
            record.action_q[window, joint],
            color="#d1495b",
            linewidth=2.0,
            label="command",
        )
        axis.plot(
            time,
            record.obs_q[window, joint],
            color="#00798c",
            linewidth=1.8,
            label="real",
        )
        for name, color in (
            ("nominal_no_gc", "#edae49"),
            ("delay_20ms", "#5f4b8b"),
            ("gravity_compensation", "#59a14f"),
        ):
            axis.plot(
                time,
                plant_outputs[name][f"{event.side}_q"][window, joint],
                color=color,
                linewidth=1.0,
                alpha=0.85,
                label=name,
            )
        axis.set_xlabel("window time [s]")
        axis.set_ylabel("angle [rad]")
        axis.set_title(f"Joint {joint + 1}")
        axis.legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"Episode 73 excursion #{event.rank}: real plant versus direct MuJoCo replay",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _trace_payload(
    record: recorded.ArmRecord,
    start: int,
    end: int,
    *,
    actual_q: np.ndarray,
    actual_pose: np.ndarray,
    actual_elbow: np.ndarray,
) -> dict[str, np.ndarray]:
    count = end - start
    command_q = record.action_q[start:end]
    command_pose = record.action_pose[start:end]
    command_elbow = record.action_elbow[start:end]
    return {
        "times": np.arange(count, dtype=np.float64) * DT,
        "phase": np.ones(count, dtype=np.int8),
        "right_target_pose": command_pose,
        "right_target": command_pose,
        "right_command_q": command_q,
        "right_driver_q": command_q,
        "right_actual_q": actual_q[start:end],
        "right_command_pose": command_pose,
        "right_driver_pose": command_pose,
        "right_actual_pose": actual_pose[start:end],
        "right_command_elbow": command_elbow,
        "right_driver_elbow": command_elbow,
        "right_actual_elbow": actual_elbow[start:end],
    }


def _save_direct_comparison_traces(
    record: recorded.ArmRecord,
    nominal_output: dict[str, np.ndarray],
) -> None:
    """Save real and MuJoCo traces with the exact same joint command."""
    source_events = json.loads(
        (
            HERE
            / "results"
            / "recorded_interventions_20260731"
            / "events.json"
        ).read_text(encoding="utf-8")
    )
    DIRECT_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    for event_id in DIRECT_EVENT_IDS:
        event = next(
            value
            for value in source_events
            if value["episode"] == EPISODE
            and value["side"] == "right"
            and value["index"] == event_id
        )
        start = int(event["start"])
        end = int(event["end"])
        scenario = f"episode73_event{event_id:02d}_direct_command"
        real_payload = _trace_payload(
            record,
            start,
            end,
            actual_q=record.obs_q,
            actual_pose=record.obs_pose,
            actual_elbow=record.obs_elbow,
        )
        sim_payload = _trace_payload(
            record,
            start,
            end,
            actual_q=nominal_output["right_q"],
            actual_pose=nominal_output["right_pose"],
            actual_elbow=nominal_output["right_elbow"],
        )
        np.savez_compressed(
            DIRECT_TRACE_DIR
            / f"direct_recorded_real_{scenario}.npz",
            **real_payload,
        )
        np.savez_compressed(
            DIRECT_TRACE_DIR
            / f"direct_mujoco_nominal_{scenario}.npz",
            **sim_payload,
        )


def run_tracking(output_dir: Path) -> tuple[
    dict[str, recorded.ArmRecord],
    dict[str, TrackingSeries],
    list[Excursion],
]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        HERE
        / "results"
        / "recorded_interventions_20260731"
        / "cache"
    )
    records = {
        side: recorded.load_record(EPISODE, side, cache_dir)
        for side in SIDES
    }
    tracking = {
        side: compute_tracking(record)
        for side, record in records.items()
    }
    excursions = [
        event
        for side in SIDES
        for event in detect_excursions(
            side,
            records[side],
            tracking[side],
        )
    ]
    excursions.sort(key=lambda value: value.score, reverse=True)
    excursions = [
        Excursion(**{**asdict(value), "rank": rank})
        for rank, value in enumerate(excursions, start=1)
    ]
    _write_csv(
        output_dir / "excursions.csv",
        [asdict(value) for value in excursions],
    )
    (output_dir / "excursions.json").write_text(
        json.dumps(
            [asdict(value) for value in excursions],
            indent=2,
        ),
        encoding="utf-8",
    )
    for side, values in tracking.items():
        np.savez_compressed(
            output_dir / f"episode73_{side}_tracking.npz",
            **asdict(values),
        )
    _plot_overview(
        records,
        tracking,
        excursions,
        output_dir / "episode73_tracking_overview.png",
    )
    _plot_excursion_catalog(
        records,
        tracking,
        excursions,
        output_dir / "episode73_excursion_catalog.png",
    )
    return records, tracking, excursions


def run_plant(
    output_dir: Path,
    records: dict[str, recorded.ArmRecord],
    tracking: dict[str, TrackingSeries],
    excursions: list[Excursion],
) -> None:
    plant_dir = output_dir / "plant"
    plant_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, np.ndarray]] = {}
    metric_rows: list[dict[str, Any]] = []
    for profile in _plant_profiles():
        cache_path = plant_dir / f"{profile.name}.npz"
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            output = {name: cached[name] for name in cached.files}
        else:
            print(f"Replaying direct command: {profile.name}", flush=True)
            output = _run_plant(records, profile)
            np.savez_compressed(cache_path, **output)
        outputs[profile.name] = output
        metric_rows.extend(_plant_metrics(profile, records, output))
    _write_csv(plant_dir / "profile_metrics.csv", metric_rows)

    selected = [
        event
        for event in excursions
        if event.side == "right"
    ][:4]
    for event in selected:
        _plot_plant_case(
            event,
            records[event.side],
            tracking[event.side],
            outputs,
            plant_dir / f"excursion_{event.rank:02d}_plant_replay.png",
        )
    _save_direct_comparison_traces(
        records["right"],
        outputs["nominal_no_gc"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("tracking", "plant", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    args = parser.parse_args()
    records, tracking, excursions = run_tracking(args.output_dir)
    if args.command in {"plant", "all"}:
        run_plant(
            args.output_dir,
            records,
            tracking,
            excursions,
        )


if __name__ == "__main__":
    main()
