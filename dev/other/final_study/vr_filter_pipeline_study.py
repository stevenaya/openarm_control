#!/usr/bin/env python3
"""Factor the July-31 VR filter path and replay its output through current IK.

The recorded intervention episodes contain joint command/state but not raw VR
poses.  This synthetic study therefore answers a narrower causal question:
can the old repeated-packet One Euro implementation, by itself, turn a
stationary or smoothly moving handle into the large reverse translation seen
in episode 73?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import study

DT = study.CONTROL_DT
OUTPUT_DIR = HERE / "results" / "vr_filter_pipeline_20260731"
FINAL_ASSET_DIR = (
    HERE.parents[2]
    / "note"
    / "openarm_control"
    / "final_report"
    / "assets"
)
PACKET_RATES_HZ = (60.0, 72.0, 90.0, 120.0)
PIPELINES = (
    "unfiltered_fresh",
    "old_repeated_shared_cutoff",
    "old_fresh_shared_cutoff",
    "fixed_fresh_separate_cutoff",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _alpha(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def _slerp(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
    q2 = q2.copy()
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q1 + alpha * (q2 - q1)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (
        math.sin((1.0 - alpha) * theta) / sin_theta * q1
        + math.sin(alpha * theta) / sin_theta * q2
    )


def _quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = abs(float(np.dot(q1, q2)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def _rotation(pose: np.ndarray) -> Rotation:
    return Rotation.from_quat(pose[[4, 5, 6, 3]])


def _quaternion(rotation: Rotation) -> np.ndarray:
    quaternion = rotation.as_quat()
    return quaternion[[3, 0, 1, 2]]


class OldPoseSmoother:
    """Exact July-31 smoother: shared cutoff and filtered-output velocity."""

    def __init__(self) -> None:
        self.p_prev: np.ndarray | None = None
        self.q_prev: np.ndarray | None = None
        self.dp_prev = np.zeros(3)
        self.t_prev: float | None = None

    def smooth(self, time_s: float, pose: np.ndarray) -> np.ndarray:
        position = pose[:3]
        quaternion = pose[3:]
        if self.t_prev is None or self.p_prev is None or self.q_prev is None:
            self.p_prev = position.copy()
            self.q_prev = quaternion.copy()
            self.t_prev = time_s
            return pose.copy()
        dt = time_s - self.t_prev
        if dt <= 0.0:
            return pose.copy()
        raw_velocity = (position - self.p_prev) / dt
        derivative_alpha = _alpha(dt, 1.5)
        filtered_velocity = (
            derivative_alpha * raw_velocity
            + (1.0 - derivative_alpha) * self.dp_prev
        )
        cutoff = 2.0 + 0.04 * float(np.linalg.norm(filtered_velocity))
        pose_alpha = _alpha(dt, cutoff)
        filtered_position = self.p_prev + pose_alpha * (
            position - self.p_prev
        )
        filtered_quaternion = _slerp(
            self.q_prev,
            quaternion,
            pose_alpha,
        )
        self.p_prev = filtered_position
        self.q_prev = filtered_quaternion
        self.dp_prev = filtered_velocity
        self.t_prev = time_s
        return np.concatenate([filtered_position, filtered_quaternion])


class FixedPoseSmoother:
    """Fresh-sample, raw-to-raw position and independent rotation filter."""

    def __init__(self) -> None:
        self.p_prev: np.ndarray | None = None
        self.p_raw_prev: np.ndarray | None = None
        self.q_prev: np.ndarray | None = None
        self.q_raw_prev: np.ndarray | None = None
        self.dp_prev = np.zeros(3)
        self.omega_prev = 0.0
        self.t_prev: float | None = None

    def smooth(self, time_s: float, pose: np.ndarray) -> np.ndarray:
        position = pose[:3]
        quaternion = pose[3:].astype(np.float64)
        quaternion /= np.linalg.norm(quaternion)
        if (
            self.q_raw_prev is not None
            and np.dot(self.q_raw_prev, quaternion) < 0.0
        ):
            quaternion = -quaternion
        if (
            self.t_prev is None
            or self.p_prev is None
            or self.q_prev is None
        ):
            self.p_prev = position.copy()
            self.p_raw_prev = position.copy()
            self.q_prev = quaternion.copy()
            self.q_raw_prev = quaternion.copy()
            self.t_prev = time_s
            return np.concatenate([position, quaternion])
        dt = time_s - self.t_prev
        if dt <= 0.0:
            return pose.copy()
        assert self.p_raw_prev is not None
        assert self.q_raw_prev is not None
        derivative_alpha = _alpha(dt, 1.5)
        raw_velocity = (position - self.p_raw_prev) / dt
        filtered_velocity = (
            derivative_alpha * raw_velocity
            + (1.0 - derivative_alpha) * self.dp_prev
        )
        position_cutoff = 2.0 + 0.04 * float(
            np.linalg.norm(filtered_velocity)
        )
        position_alpha = _alpha(dt, position_cutoff)
        raw_angular_speed = (
            _quat_angle(self.q_raw_prev, quaternion) / dt
        )
        angular_alpha = _alpha(dt, 3.0)
        filtered_angular_speed = (
            angular_alpha * raw_angular_speed
            + (1.0 - angular_alpha) * self.omega_prev
        )
        rotation_cutoff = 3.0 + 0.25 * filtered_angular_speed
        rotation_alpha = _alpha(dt, rotation_cutoff)
        filtered_position = self.p_prev + position_alpha * (
            position - self.p_prev
        )
        filtered_quaternion = _slerp(
            self.q_prev,
            quaternion,
            rotation_alpha,
        )
        self.p_prev = filtered_position
        self.p_raw_prev = position.copy()
        self.q_prev = filtered_quaternion
        self.q_raw_prev = quaternion.copy()
        self.dp_prev = filtered_velocity
        self.omega_prev = filtered_angular_speed
        self.t_prev = time_s
        return np.concatenate([filtered_position, filtered_quaternion])


@dataclass(frozen=True)
class Trajectory:
    name: str
    duration_s: float
    evaluator: Callable[[float], np.ndarray]
    initial_pose: np.ndarray


def _smoothstep(value: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _trajectory(
    name: str,
    *,
    displacement: tuple[float, float, float],
    angular_speed: float,
    linear_speed: float,
) -> Trajectory:
    factory = study.PoseFactory()
    base, _ = factory.bimanual(study.CHEST_Q_RIGHT, study.START_Q_LEFT)
    translation = np.asarray(displacement, dtype=np.float64)
    translation_duration = (
        0.0
        if np.linalg.norm(translation) == 0.0
        else 1.5 * float(np.linalg.norm(translation)) / linear_speed
    )
    rotation_duration = 1.5 * math.pi / angular_speed
    move_duration = max(translation_duration, rotation_duration)
    start_rotation = _rotation(base)
    rotation_delta = np.array([0.0, 0.0, -math.pi])
    hold_before = 0.30
    hold_after = 0.70

    def evaluator(time_s: float) -> np.ndarray:
        pose = base.copy()
        elapsed = time_s - hold_before
        position_amount = (
            1.0
            if translation_duration == 0.0 and elapsed >= 0.0
            else _smoothstep(elapsed / max(translation_duration, 1e-9))
        )
        if translation_duration == 0.0:
            position_amount = 0.0
        rotation_amount = _smoothstep(elapsed / rotation_duration)
        pose[:3] += position_amount * translation
        pose[3:] = _quaternion(
            start_rotation
            * Rotation.from_rotvec(rotation_amount * rotation_delta)
        )
        return pose

    return Trajectory(
        name=name,
        duration_s=hold_before + move_duration + hold_after,
        evaluator=evaluator,
        initial_pose=base,
    )


def trajectories() -> tuple[Trajectory, ...]:
    return (
        _trajectory(
            "chest_pure_roll_w10",
            displacement=(0.0, 0.0, 0.0),
            angular_speed=10.0,
            linear_speed=1.0,
        ),
        _trajectory(
            "chest_roll_forward_w10_v1",
            displacement=(0.07, 0.0, 0.0),
            angular_speed=10.0,
            linear_speed=1.0,
        ),
        _trajectory(
            "chest_roll_diagonal_w10_v1",
            displacement=(0.06, -0.05, -0.025),
            angular_speed=10.0,
            linear_speed=1.0,
        ),
    )


def _sample_filter(
    trajectory: Trajectory,
    pipeline: str,
    packet_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(0.0, trajectory.duration_s + 0.5 * DT, DT)
    packet_times = np.arange(
        0.0,
        trajectory.duration_s + 1.0 / packet_rate_hz,
        1.0 / packet_rate_hz,
    )
    packet_poses = np.asarray(
        [trajectory.evaluator(time_s) for time_s in packet_times]
    )
    raw_tick = np.asarray(
        [trajectory.evaluator(time_s) for time_s in times]
    )
    output = np.empty_like(raw_tick)
    latest_packet = -1
    cached: np.ndarray | None = None
    smoother: OldPoseSmoother | FixedPoseSmoother | None
    if pipeline in (
        "old_repeated_shared_cutoff",
        "old_fresh_shared_cutoff",
    ):
        smoother = OldPoseSmoother()
    elif pipeline == "fixed_fresh_separate_cutoff":
        smoother = FixedPoseSmoother()
    elif pipeline == "unfiltered_fresh":
        smoother = None
    else:
        raise ValueError(pipeline)

    for index, time_s in enumerate(times):
        packet_index = int(
            np.searchsorted(packet_times, time_s, side="right") - 1
        )
        packet_index = max(packet_index, 0)
        is_fresh = packet_index != latest_packet
        latest_packet = packet_index
        pose = packet_poses[packet_index]
        if pipeline == "old_repeated_shared_cutoff":
            assert smoother is not None
            cached = smoother.smooth(time_s, pose)
        elif is_fresh:
            cached = pose.copy() if smoother is None else smoother.smooth(
                packet_times[packet_index],
                pose,
            )
        assert cached is not None
        output[index] = cached
    return times, raw_tick, output


def _limit_targets(
    poses: np.ndarray,
    initial_pose: np.ndarray,
    *,
    linear_speed: float = 1.0,
    angular_speed: float = 6.0,
) -> np.ndarray:
    limited = np.empty_like(poses)
    previous = initial_pose.copy()
    for index, pose in enumerate(poses):
        position_delta = pose[:3] - previous[:3]
        position_norm = float(np.linalg.norm(position_delta))
        max_position_step = linear_speed * DT
        next_pose = pose.copy()
        if position_norm > max_position_step:
            next_pose[:3] = (
                previous[:3]
                + position_delta / position_norm * max_position_step
            )
        relative = _rotation(previous).inv() * _rotation(pose)
        rotation_vector = relative.as_rotvec()
        angle = float(np.linalg.norm(rotation_vector))
        max_rotation_step = angular_speed * DT
        if angle > max_rotation_step:
            next_pose[3:] = _quaternion(
                _rotation(previous)
                * Rotation.from_rotvec(
                    rotation_vector / angle * max_rotation_step
                )
            )
        previous = next_pose
        limited[index] = next_pose
    return limited


def _scenario(
    trajectory: Trajectory,
    pipeline: str,
    packet_rate_hz: float,
    times: np.ndarray,
    targets: np.ndarray,
) -> study.Scenario:
    phase = np.ones(times.size, dtype=np.int8)
    phase[times < 0.30] = 0
    phase[times > trajectory.duration_s - 0.70] = 2
    name = (
        f"vr_{trajectory.name}_{pipeline}_{packet_rate_hz:g}hz"
        .replace(".", "p")
    )
    return study.Scenario(
        name=name,
        family="vr_filter_pipeline",
        mode="right",
        speed=packet_rate_hz,
        times=times,
        phase=phase,
        target_right=targets,
        target_left=np.repeat(targets[:1], times.size, axis=0),
        initial_right=study.CHEST_Q_RIGHT.copy(),
        initial_left=study.START_Q_LEFT.copy(),
        description=(
            "Synthetic chest wrist flip passed through a selected VR "
            "sampling/filter pipeline and the production target rate limiter."
        ),
    )


def _orientation_error(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (_rotation(first[index]).inv() * _rotation(second[index])).magnitude()
            for index in range(first.shape[0])
        ]
    )


def _trace_path(
    profile: study.Profile,
    scenario: study.Scenario,
) -> Path:
    matches = list(
        (OUTPUT_DIR / "ik" / "traces").glob(
            f"*_{profile.name}_{scenario.name}.npz"
        )
    )
    if not matches:
        raise FileNotFoundError(scenario.name)
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profile = study.make_profile(
        "current_pr",
        nullspace_cost=8.0,
        description="Current PR and July-31 evaluation dataflow settings.",
    )
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories():
        for packet_rate_hz in PACKET_RATES_HZ:
            for pipeline in PIPELINES:
                print(
                    f"{trajectory.name}: {pipeline}, {packet_rate_hz:g} Hz",
                    flush=True,
                )
                times, raw, filtered = _sample_filter(
                    trajectory,
                    pipeline,
                    packet_rate_hz,
                )
                governed = _limit_targets(filtered, trajectory.initial_pose)
                scenario = _scenario(
                    trajectory,
                    pipeline,
                    packet_rate_hz,
                    times,
                    governed,
                )
                study.run_cached(
                    profile,
                    scenario,
                    OUTPUT_DIR / "ik",
                    save_full_trace=True,
                )
                trace = np.load(_trace_path(profile, scenario))
                phase = trace["phase"] == 1
                command = trace["right_command_pose"]
                actual = trace["right_actual_pose"]
                raw_orientation_error = _orientation_error(raw, filtered)
                governed_orientation_error = _orientation_error(raw, governed)
                command_error = np.linalg.norm(
                    command[:, :3] - governed[:, :3],
                    axis=1,
                )
                actual_error = np.linalg.norm(
                    actual[:, :3] - governed[:, :3],
                    axis=1,
                )
                governed_radius = np.linalg.norm(governed[:, :3], axis=1)
                command_radius = np.linalg.norm(command[:, :3], axis=1)
                actual_radius = np.linalg.norm(actual[:, :3], axis=1)
                displacement = raw[-1, :3] - raw[0, :3]
                displacement_norm = float(np.linalg.norm(displacement))
                if displacement_norm > 1e-9:
                    direction = displacement / displacement_norm
                    filtered_progress = (
                        filtered[:, :3] - filtered[0, :3]
                    ) @ direction
                    governed_progress = (
                        governed[:, :3] - governed[0, :3]
                    ) @ direction
                    filter_reverse_travel = float(
                        np.sum(
                            np.maximum(
                                -np.diff(filtered_progress),
                                0.0,
                            )
                        )
                    )
                    governor_reverse_travel = float(
                        np.sum(
                            np.maximum(
                                -np.diff(governed_progress),
                                0.0,
                            )
                        )
                    )
                else:
                    filter_reverse_travel = 0.0
                    governor_reverse_travel = 0.0
                command_dq = trace["right_command_dq"]
                utilization = np.abs(command_dq) / np.asarray(
                    study.CONTROL_CAPS
                )
                rows.append(
                    {
                        "trajectory": trajectory.name,
                        "pipeline": pipeline,
                        "packet_rate_hz": packet_rate_hz,
                        "filter_position_error_max_m": float(
                            np.max(
                                np.linalg.norm(
                                    filtered[:, :3] - raw[:, :3],
                                    axis=1,
                                )
                            )
                        ),
                        "filter_orientation_error_rmse_rad": float(
                            np.sqrt(np.mean(np.square(raw_orientation_error)))
                        ),
                        "filter_orientation_error_max_rad": float(
                            np.max(raw_orientation_error)
                        ),
                        "filter_reverse_travel_m": filter_reverse_travel,
                        "governor_reverse_travel_m": governor_reverse_travel,
                        "governed_orientation_error_rmse_rad": float(
                            np.sqrt(
                                np.mean(
                                    np.square(governed_orientation_error)
                                )
                            )
                        ),
                        "command_position_error_max_m": float(
                            np.max(command_error[phase])
                        ),
                        "actual_position_error_max_m": float(
                            np.max(actual_error[phase])
                        ),
                        "command_inward_deviation_max_m": float(
                            np.max(
                                governed_radius[phase]
                                - command_radius[phase]
                            )
                        ),
                        "actual_inward_deviation_max_m": float(
                            np.max(
                                governed_radius[phase]
                                - actual_radius[phase]
                            )
                        ),
                        "joint_velocity_saturated_pct": float(
                            100.0
                            * np.mean(np.any(utilization[phase] >= 0.999, axis=1))
                        ),
                        "solver_failed_count": int(
                            np.count_nonzero(trace["solver_failed"])
                        ),
                    }
                )
    _write_csv(OUTPUT_DIR / "metrics.csv", rows)
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "packet_rates_hz": PACKET_RATES_HZ,
                "pipelines": PIPELINES,
                "target_limits": {
                    "linear_m_s": 1.0,
                    "angular_rad_s": 6.0,
                },
                "production_vr_revision": "5c0cb0f",
                "fixed_reference_revision": "8346f2d with shortest-path SLERP",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def plot() -> None:
    rows = list(csv.DictReader((OUTPUT_DIR / "metrics.csv").open()))
    by_key = {
        (
            row["trajectory"],
            row["pipeline"],
            float(row["packet_rate_hz"]),
        ): row
        for row in rows
    }
    trajectory = next(
        item
        for item in trajectories()
        if item.name == "chest_roll_diagonal_w10_v1"
    )
    rate = 72.0
    profile = study.make_profile("current_pr", nullspace_cost=8.0)
    colors = {
        "unfiltered_fresh": "#555555",
        "old_repeated_shared_cutoff": "#d62728",
        "old_fresh_shared_cutoff": "#ff8c00",
        "fixed_fresh_separate_cutoff": "#1f77b4",
    }
    labels = {
        "unfiltered_fresh": "fresh packets, no smoothing",
        "old_repeated_shared_cutoff": "July-31 old path",
        "old_fresh_shared_cutoff": "old filter, fresh only",
        "fixed_fresh_separate_cutoff": "fresh + raw-to-raw + rotation filter",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for pipeline in PIPELINES:
        times, raw, filtered = _sample_filter(trajectory, pipeline, rate)
        governed = _limit_targets(filtered, trajectory.initial_pose)
        scenario = _scenario(
            trajectory,
            pipeline,
            rate,
            times,
            governed,
        )
        trace = np.load(_trace_path(profile, scenario))
        orientation_error = _orientation_error(raw, governed)
        command_error = np.linalg.norm(
            trace["right_command_pose"][:, :3] - governed[:, :3],
            axis=1,
        )
        inward = (
            np.linalg.norm(governed[:, :3], axis=1)
            - np.linalg.norm(trace["right_command_pose"][:, :3], axis=1)
        )
        axes[0, 0].plot(
            times,
            orientation_error,
            color=colors[pipeline],
            label=labels[pipeline],
        )
        axes[0, 1].plot(
            times,
            100.0 * np.linalg.norm(filtered[:, :3] - raw[:, :3], axis=1),
            color=colors[pipeline],
        )
        axes[1, 0].plot(
            times,
            100.0 * command_error,
            color=colors[pipeline],
        )
        axes[1, 1].plot(
            times,
            100.0 * inward,
            color=colors[pipeline],
        )
    axes[0, 0].set_title("VR/intervention orientation lag")
    axes[0, 0].set_ylabel("orientation error [rad]")
    axes[0, 1].set_title("VR filter translation lag")
    axes[0, 1].set_ylabel("position error [cm]")
    axes[1, 0].set_title("Current PR command vs filtered target")
    axes[1, 0].set_ylabel("position error [cm]")
    axes[1, 1].set_title("Current PR inward position sacrifice")
    axes[1, 1].set_ylabel("inward deviation [cm]")
    for axis in axes.flat:
        axis.set_xlabel("time [s]")
        axis.grid(alpha=0.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
    )
    fig.suptitle(
        "72 Hz chest roll + diagonal move: old and corrected VR pipelines",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    output = OUTPUT_DIR / "vr_filter_pipeline_summary.png"
    fig.savefig(output, dpi=180)
    FINAL_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FINAL_ASSET_DIR / "31_vr_filter_pipeline_summary.png",
        dpi=180,
    )
    plt.close(fig)

    summary_rows: list[dict[str, Any]] = []
    for pipeline in PIPELINES:
        matching = [
            row
            for (name, candidate, _), row in by_key.items()
            if name == trajectory.name and candidate == pipeline
        ]
        summary_rows.append(
            {
                "pipeline": pipeline,
                "filter_orientation_error_max_rad": max(
                    float(row["filter_orientation_error_max_rad"])
                    for row in matching
                ),
                "command_inward_deviation_max_m": max(
                    float(row["command_inward_deviation_max_m"])
                    for row in matching
                ),
                "actual_inward_deviation_max_m": max(
                    float(row["actual_inward_deviation_max_m"])
                    for row in matching
                ),
            }
        )
    _write_csv(OUTPUT_DIR / "summary.csv", summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("run", "plot", "all"),
        default="all",
        nargs="?",
    )
    args = parser.parse_args()
    if args.stage in {"run", "all"}:
        run()
    if args.stage in {"plot", "all"}:
        plot()


if __name__ == "__main__":
    main()
