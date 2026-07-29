"""Jointly tune position and orientation FrameTask error modulation.

This is a development-only experiment. It compares the current instantaneous
position-speed schedule with the historical rate-limited schedule, no-latch,
always-on, and disabled position limiting. Orientation limiting is applied
independently of the position schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import matplotlib.pyplot as plt
import mink
import mujoco
import numpy as np
from chest_wrist_flip_study import (
    CONTROL_DT,
    DEFAULT_CAPS,
    FLIP_START_Q,
    DynamicPlant,
    driver_state,
    make_setup,
    pack_pose,
    pose_from_configuration,
    velocity_mapping,
)
from scipy.spatial.transform import Rotation

from openarm_control import IKParams, Kinematics
from openarm_control.bounded_frame_task import BoundedFrameTask

EXTENDED_Q_RIGHT = np.array(
    [1.224145, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ErrorProfile:
    """One position schedule and pair of Cartesian error caps."""

    name: str
    position_mode: str
    position_limit_m: float
    orientation_limit_rad: float
    activation_low_m_s: float = 0.6
    activation_high_m_s: float = 0.9


@dataclass(frozen=True)
class Scenario:
    """One sampled target path."""

    name: str
    kind: str
    speed: float
    lateral_m: float = -0.1
    post_hold_s: float = 1.0


def _smoothstep(value: float, low: float, high: float) -> float:
    unit = float(np.clip((value - low) / (high - low), 0.0, 1.0))
    return unit * unit * (3.0 - 2.0 * unit)


def _make_kinematics(profile: ErrorProfile, *, braking: bool) -> Kinematics:
    params = IKParams(
        position_cost=10.0,
        orientation_cost=1.0,
        lm_damping=0.02,
        damping=0.1,
        dt=CONTROL_DT,
        max_iters=5,
        velocity_limits=velocity_mapping(tuple(DEFAULT_CAPS)),
        # Keep the wrapper present even when the position part is disabled.
        frame_position_error_limit=max(profile.position_limit_m, 1e-12),
        joint_braking=braking,
        joint_braking_distance=0.2,
        nullspace_cost=12.0,
        nullspace_return_rate=1.6,
        nullspace_max_speed=1.0,
        singularity_max_approach_rate=0.25,
        kinetic_energy_cost=5e-5,
    )
    return Kinematics(make_setup("right"), params)


def _install_error_policy(
    task: BoundedFrameTask,
    profile: ErrorProfile,
) -> None:
    task.position_error_limit = profile.position_limit_m
    task.orientation_error_limit = profile.orientation_limit_rad

    def compute_independent_objective(
        self: BoundedFrameTask,
        configuration: mink.Configuration,
    ) -> mink.Objective:
        full_error = self.compute_full_error(configuration)
        limited_error = self.compute_limited_error(configuration)
        used_error = full_error.copy()
        used_error[:3] += self.limit_activation * (
            limited_error[:3] - full_error[:3]
        )
        if self.orientation_error_limit > 0.0:
            used_error[3:] = limited_error[3:]
        return self._assemble_qp(
            used_error,
            self.compute_jacobian(configuration),
            configuration._eye_nv,
        )

    task.compute_qp_objective = MethodType(compute_independent_objective, task)

    state: dict[str, Any] = {
        "previous_position": None,
        "activation": 1.0 if profile.position_mode == "always_cap" else 0.0,
    }
    task.set_limit_activation(float(state["activation"]))

    def update_schedule(
        self: BoundedFrameTask,
        transform: mink.SE3,
        configuration: mink.Configuration,
    ) -> None:
        self.set_target(transform)
        mode = profile.position_mode
        if mode == "always_cap":
            activation = 1.0
        elif mode == "position_off":
            activation = 0.0
        else:
            position = transform.translation()
            previous = state["previous_position"]
            speed = (
                0.0
                if previous is None
                else float(np.linalg.norm(position - previous)) / CONTROL_DT
            )
            state["previous_position"] = position.copy()

            if mode.startswith("current_"):
                target = _smoothstep(
                    speed,
                    profile.activation_low_m_s,
                    profile.activation_high_m_s,
                )
                rise_rate = math.inf
                fall_rate = math.inf
            elif mode.startswith("historical_"):
                target = _smoothstep(speed, 0.2, 0.5)
                rise_rate = 4.0
                fall_rate = 2.0
            else:
                raise ValueError(f"Unknown position mode: {mode}")

            current = float(state["activation"])
            if mode in {"current_latch", "historical_latch"} and current > 0.0:
                position_error = float(
                    np.linalg.norm(self.compute_full_error(configuration)[:3])
                )
                if position_error > 2.0 * self.position_error_limit:
                    target = max(target, current)

            difference = target - current
            rate = rise_rate if difference > 0.0 else fall_rate
            if np.isfinite(rate):
                difference = float(
                    np.clip(
                        difference,
                        -rate * CONTROL_DT,
                        rate * CONTROL_DT,
                    )
                )
            activation = current + difference

        state["activation"] = float(activation)
        self.set_limit_activation(float(activation))

    task.set_target_and_update_schedule = MethodType(update_schedule, task)


def _initial_q(scenario: Scenario) -> np.ndarray:
    if scenario.kind == "chest_flip":
        return FLIP_START_Q.copy()
    if scenario.kind == "fast_retract":
        return EXTENDED_Q_RIGHT.copy()
    raise ValueError(f"Unknown scenario kind: {scenario.kind}")


def _build_targets(
    scenario: Scenario,
    initial_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scenario.kind == "chest_flip":
        pre_hold_s = 0.2
        translation = np.array([0.05, 0.0, 0.0])
        translation_s = float(np.linalg.norm(translation))
        angle = 4.5
        flip_s = angle / scenario.speed
        post_hold_s = scenario.post_hold_s
        total_s = pre_hold_s + translation_s + flip_s + post_hold_s
        count = math.ceil(total_s / CONTROL_DT)
        time = np.arange(count, dtype=np.float64) * CONTROL_DT
        targets = np.empty((count, 7), dtype=np.float64)
        phase = np.zeros(count, dtype=np.int8)
        initial_rotation = Rotation.from_quat(initial_pose[[4, 5, 6, 3]])
        axis = np.array([0.0, 0.0, -1.0])
        flip_start_s = pre_hold_s + translation_s
        for index, elapsed in enumerate(time):
            translation_fraction = np.clip(
                (elapsed - pre_hold_s) / translation_s,
                0.0,
                1.0,
            )
            theta = np.clip(
                (elapsed - flip_start_s) * scenario.speed,
                0.0,
                angle,
            )
            targets[index] = pack_pose(
                initial_pose[:3] + translation_fraction * translation,
                initial_rotation * Rotation.from_rotvec(axis * theta),
            )
            if elapsed >= pre_hold_s:
                phase[index] = 1
            if elapsed >= flip_start_s:
                phase[index] = 2
            if elapsed >= flip_start_s + flip_s:
                phase[index] = 3
        return time, phase, targets

    if scenario.kind == "fast_retract":
        pre_hold_s = 0.3
        displacement = np.array([-0.22, scenario.lateral_m, -0.12])
        distance = float(np.linalg.norm(displacement))
        move_s = 1.5 * distance / scenario.speed
        post_hold_s = 1.0
        total_s = pre_hold_s + move_s + post_hold_s
        count = math.ceil(total_s / CONTROL_DT)
        time = np.arange(count, dtype=np.float64) * CONTROL_DT
        targets = np.repeat(initial_pose[None, :], count, axis=0)
        phase = np.zeros(count, dtype=np.int8)
        for index, elapsed in enumerate(time):
            unit = float(np.clip((elapsed - pre_hold_s) / move_s, 0.0, 1.0))
            progress = unit * unit * (3.0 - 2.0 * unit)
            targets[index, :3] = initial_pose[:3] + progress * displacement
            if elapsed >= pre_hold_s:
                phase[index] = 1
            if elapsed >= pre_hold_s + move_s:
                phase[index] = 2
        return time, phase, targets

    raise ValueError(f"Unknown scenario kind: {scenario.kind}")


def _orientation_error(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    target_rotation = Rotation.from_quat(target[:, [4, 5, 6, 3]])
    actual_rotation = Rotation.from_quat(actual[:, [4, 5, 6, 3]])
    return (actual_rotation.inv() * target_rotation).magnitude()


def _elbow_body_id(model: mujoco.MjModel) -> int:
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "openarm_right_link4",
    )


def simulate(
    profile: ErrorProfile,
    scenario: Scenario,
    *,
    braking: bool,
) -> dict[str, np.ndarray]:
    kin = _make_kinematics(profile, braking=braking)
    initial_q = _initial_q(scenario)
    state = driver_state(kin.setup)
    state[:7] = initial_q
    state[7] = 0.0
    plant = DynamicPlant(initial_q, gravity=True)
    kin.sync(plant.qpos16())

    solver = kin._ik
    assert solver is not None
    task = solver._tasks["right"]
    if not isinstance(task, BoundedFrameTask):
        raise TypeError("Expected a BoundedFrameTask in the tuning study.")
    _install_error_policy(task, profile)

    initial_pose = pose_from_configuration(kin, "right")
    time, phase, targets = _build_targets(scenario, initial_pose)
    count = time.size
    arrays = {
        "time": time,
        "phase": phase,
        "target_pose": targets,
        "command_pose": np.empty((count, 7)),
        "actual_pose": np.empty((count, 7)),
        "command_q": np.empty((count, 7)),
        "actual_q": np.empty((count, 7)),
        "actual_dq": np.empty((count, 7)),
        "command_elbow": np.empty((count, 3)),
        "actual_elbow": np.empty((count, 3)),
        "activation": np.empty(count),
        "full_error": np.empty((count, 6)),
        "used_error": np.empty((count, 6)),
        "failed": np.zeros(count, dtype=bool),
    }

    command_q = initial_q.copy()
    elbow_body = _elbow_body_id(kin.setup.model)
    actual_elbow_body = _elbow_body_id(plant.model)
    for tick, target in enumerate(targets):
        kin.update_measured_state(plant.qpos16(), plant.qvel16())
        kin.set_target("right", target)
        result = kin.solve()
        if result is None:
            arrays["failed"][tick] = True
        else:
            command_q = result[:7].astype(np.float64)

        command_pose = pose_from_configuration(kin, "right")
        command_elbow = kin.setup.data.xpos[elbow_body].copy()
        plant.set_command(command_q)
        plant.step()

        full_error = task.compute_full_error(solver._config)
        limited_error = task.compute_limited_error(solver._config)
        used_error = full_error.copy()
        used_error[:3] += task.limit_activation * (
            limited_error[:3] - full_error[:3]
        )
        if task.orientation_error_limit > 0.0:
            used_error[3:] = limited_error[3:]

        arrays["command_pose"][tick] = command_pose
        arrays["actual_pose"][tick] = plant.pose("right")
        arrays["command_q"][tick] = command_q
        arrays["actual_q"][tick] = plant.q("right")
        arrays["actual_dq"][tick] = plant.dq("right")
        arrays["command_elbow"][tick] = command_elbow
        arrays["actual_elbow"][tick] = plant.data.xpos[actual_elbow_body]
        arrays["activation"][tick] = task.limit_activation
        arrays["full_error"][tick] = full_error
        arrays["used_error"][tick] = used_error
    return arrays


def _settle_time(
    time: np.ndarray,
    error: np.ndarray,
    motion_end: int,
    threshold: float,
) -> float:
    indices = np.flatnonzero(
        (np.arange(time.size) >= motion_end) & (error <= threshold)
    )
    if not indices.size:
        return math.nan
    return float(time[indices[0]] - time[motion_end])


def summarize(
    profile: ErrorProfile,
    scenario: Scenario,
    trace: dict[str, np.ndarray],
    *,
    braking: bool,
) -> dict[str, Any]:
    phase = trace["phase"]
    active_phase = 2 if scenario.kind == "chest_flip" else 1
    active = phase == active_phase
    motion_end = int(np.flatnonzero(active)[-1])
    tail = np.arange(trace["time"].size) >= max(
        motion_end,
        trace["time"].size - round(0.4 / CONTROL_DT),
    )

    command_position_error = np.linalg.norm(
        trace["command_pose"][:, :3] - trace["target_pose"][:, :3],
        axis=1,
    )
    actual_position_error = np.linalg.norm(
        trace["actual_pose"][:, :3] - trace["target_pose"][:, :3],
        axis=1,
    )
    command_orientation_error = _orientation_error(
        trace["target_pose"],
        trace["command_pose"],
    )
    actual_orientation_error = _orientation_error(
        trace["target_pose"],
        trace["actual_pose"],
    )
    command_dq = np.vstack(
        [
            np.zeros(7),
            np.diff(trace["command_q"], axis=0) / CONTROL_DT,
        ]
    )
    command_ddq = np.vstack(
        [
            np.zeros(7),
            np.diff(command_dq, axis=0) / CONTROL_DT,
        ]
    )
    actual_ddq = np.vstack(
        [
            np.zeros(7),
            np.diff(trace["actual_dq"], axis=0) / CONTROL_DT,
        ]
    )
    command_elbow_velocity = np.vstack(
        [
            np.zeros(3),
            np.diff(trace["command_elbow"], axis=0) / CONTROL_DT,
        ]
    )
    actual_elbow_velocity = np.vstack(
        [
            np.zeros(3),
            np.diff(trace["actual_elbow"], axis=0) / CONTROL_DT,
        ]
    )
    caps = np.asarray(DEFAULT_CAPS, dtype=np.float64)
    saturation = np.abs(command_dq[active]) >= 0.98 * caps

    tail_centered = trace["actual_pose"][tail, :3] - np.mean(
        trace["actual_pose"][tail, :3],
        axis=0,
    )
    tail_vibration = np.max(np.ptp(tail_centered, axis=0))

    row: dict[str, Any] = {
        **asdict(profile),
        "scenario": scenario.name,
        "scenario_kind": scenario.kind,
        "scenario_speed": scenario.speed,
        "scenario_lateral_m": scenario.lateral_m,
        "braking": braking,
        "command_position_rmse_active_m": float(
            np.sqrt(np.mean(command_position_error[active] ** 2))
        ),
        "actual_position_rmse_active_m": float(
            np.sqrt(np.mean(actual_position_error[active] ** 2))
        ),
        "command_position_p95_active_m": float(
            np.percentile(command_position_error[active], 95)
        ),
        "actual_position_p95_active_m": float(
            np.percentile(actual_position_error[active], 95)
        ),
        "max_command_position_error_m": float(np.max(command_position_error)),
        "max_actual_position_error_m": float(np.max(actual_position_error)),
        "final_command_position_error_m": float(command_position_error[-1]),
        "final_actual_position_error_m": float(actual_position_error[-1]),
        "command_orientation_rmse_active_rad": float(
            np.sqrt(np.mean(command_orientation_error[active] ** 2))
        ),
        "actual_orientation_rmse_active_rad": float(
            np.sqrt(np.mean(actual_orientation_error[active] ** 2))
        ),
        "max_command_orientation_error_rad": float(
            np.max(command_orientation_error)
        ),
        "max_actual_orientation_error_rad": float(np.max(actual_orientation_error)),
        "final_command_orientation_error_rad": float(
            command_orientation_error[-1]
        ),
        "final_actual_orientation_error_rad": float(actual_orientation_error[-1]),
        "position_settle_time_1cm_s": _settle_time(
            trace["time"],
            actual_position_error,
            motion_end,
            0.01,
        ),
        "orientation_settle_time_0p05rad_s": _settle_time(
            trace["time"],
            actual_orientation_error,
            motion_end,
            0.05,
        ),
        "command_joint_speed_max_active_rad_s": float(
            np.max(np.abs(command_dq[active]))
        ),
        "actual_joint_speed_max_active_rad_s": float(
            np.max(np.abs(trace["actual_dq"][active]))
        ),
        "command_joint_accel_p99_active_rad_s2": float(
            np.percentile(np.abs(command_ddq[active]), 99)
        ),
        "actual_joint_accel_p99_active_rad_s2": float(
            np.percentile(np.abs(actual_ddq[active]), 99)
        ),
        "command_proximal_accel_p99_active_rad_s2": float(
            np.percentile(np.abs(command_ddq[active, :4]), 99)
        ),
        "actual_proximal_accel_p99_active_rad_s2": float(
            np.percentile(np.abs(actual_ddq[active, :4]), 99)
        ),
        "command_elbow_speed_max_active_m_s": float(
            np.max(np.linalg.norm(command_elbow_velocity[active], axis=1))
        ),
        "actual_elbow_speed_max_active_m_s": float(
            np.max(np.linalg.norm(actual_elbow_velocity[active], axis=1))
        ),
        "command_elbow_lateral_range_active_m": float(
            np.ptp(trace["command_elbow"][active, 1])
        ),
        "actual_elbow_lateral_range_active_m": float(
            np.ptp(trace["actual_elbow"][active, 1])
        ),
        "j1_velocity_saturation_fraction": float(np.mean(saturation[:, 0])),
        "any_velocity_saturation_fraction": float(
            np.mean(np.any(saturation, axis=1))
        ),
        "mean_activation_active": float(np.mean(trace["activation"][active])),
        "mean_activation_tail": float(np.mean(trace["activation"][tail])),
        "max_full_position_error_m": float(
            np.max(np.linalg.norm(trace["full_error"][:, :3], axis=1))
        ),
        "max_used_position_error_m": float(
            np.max(np.linalg.norm(trace["used_error"][:, :3], axis=1))
        ),
        "max_full_orientation_error_rad": float(
            np.max(np.linalg.norm(trace["full_error"][:, 3:], axis=1))
        ),
        "max_used_orientation_error_rad": float(
            np.max(np.linalg.norm(trace["used_error"][:, 3:], axis=1))
        ),
        "tail_position_vibration_p2p_m": float(tail_vibration),
        "solve_failures": int(np.sum(trace["failed"])),
    }
    return row


def _tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def primary_profiles() -> list[ErrorProfile]:
    profiles: dict[str, ErrorProfile] = {}

    def add(mode: str, position: float, orientation: float) -> None:
        name = f"{mode}_p{_tag(position)}_r{_tag(orientation)}"
        profiles[name] = ErrorProfile(name, mode, position, orientation)

    for mode in (
        "current_latch",
        "current_no_latch",
        "historical_latch",
        "historical_no_latch",
        "always_cap",
        "position_off",
    ):
        for orientation in (0.0, 0.02, 0.03, 0.04, 0.06):
            add(mode, 0.0 if mode == "position_off" else 0.003, orientation)

    for mode in (
        "current_latch",
        "current_no_latch",
        "historical_latch",
        "always_cap",
    ):
        for position in (0.0015, 0.002, 0.004, 0.006, 0.009, 0.012):
            add(mode, position, 0.03)

    for mode in ("current_latch", "current_no_latch", "always_cap"):
        for orientation in (0.01, 0.015, 0.025, 0.05, 0.08, 0.12):
            add(mode, 0.003, orientation)
    return list(profiles.values())


def primary_scenarios() -> list[Scenario]:
    return [
        Scenario("retract_m0p10_v0p8", "fast_retract", 0.8, -0.1),
        Scenario("chest_flip_w8", "chest_flip", 8.0),
    ]


def robustness_scenarios() -> list[Scenario]:
    scenarios = [
        Scenario(
            f"retract_{'m' if lateral < 0.0 else 'p'}0p10_v{_tag(speed)}",
            "fast_retract",
            speed,
            lateral,
        )
        for lateral in (-0.1, 0.1)
        for speed in (0.4, 0.8, 1.0)
    ]
    scenarios.extend(
        Scenario(f"chest_flip_w{_tag(speed)}", "chest_flip", speed)
        for speed in (2.0, 4.0, 6.0, 8.0, 10.0)
    )
    return scenarios


def robustness_profiles() -> list[ErrorProfile]:
    return [
        ErrorProfile("current_baseline", "current_latch", 0.003, 0.0),
        ErrorProfile("current_p3_r3", "current_latch", 0.003, 0.03),
        ErrorProfile("current_p2_r4", "current_latch", 0.002, 0.04),
        ErrorProfile("current_p3_r4", "current_latch", 0.003, 0.04),
        ErrorProfile("current_p3_r5", "current_latch", 0.003, 0.05),
        ErrorProfile("current_p3_r6", "current_latch", 0.003, 0.06),
        ErrorProfile("current_no_latch_p3_r4", "current_no_latch", 0.003, 0.04),
        ErrorProfile("historical_p3_r4", "historical_latch", 0.003, 0.04),
        ErrorProfile("always_p3_r4", "always_cap", 0.003, 0.04),
        ErrorProfile("always_p6_r4", "always_cap", 0.006, 0.04),
        ErrorProfile("always_p9_r4", "always_cap", 0.009, 0.04),
        ErrorProfile("position_off_r4", "position_off", 0.0, 0.04),
    ]


def schedule_profiles() -> list[ErrorProfile]:
    profiles = [
        ErrorProfile("current_baseline", "current_latch", 0.003, 0.0),
        ErrorProfile("historical_p3_r4", "historical_latch", 0.003, 0.04),
    ]
    profiles.extend(
        ErrorProfile(
            f"current_p3_r4_v{_tag(low)}_{_tag(high)}",
            "current_latch",
            0.003,
            0.04,
            low,
            high,
        )
        for low, high in (
            (0.4, 0.7),
            (0.5, 0.8),
            (0.6, 0.9),
            (0.7, 1.0),
            (0.8, 1.1),
        )
    )
    return profiles


def settling_profiles() -> list[ErrorProfile]:
    return [
        ErrorProfile("current_baseline", "current_latch", 0.003, 0.0),
        ErrorProfile("current_p3_r3", "current_latch", 0.003, 0.03),
        ErrorProfile("current_p3_r4", "current_latch", 0.003, 0.04),
        ErrorProfile("current_p3_r5", "current_latch", 0.003, 0.05),
        ErrorProfile("current_p3_r6", "current_latch", 0.003, 0.06),
        ErrorProfile("current_no_latch_p3_r4", "current_no_latch", 0.003, 0.04),
        ErrorProfile("historical_p3_r4", "historical_latch", 0.003, 0.04),
    ]


def settling_scenarios() -> list[Scenario]:
    return [
        Scenario(
            f"chest_flip_w{_tag(speed)}_hold3",
            "chest_flip",
            speed,
            post_hold_s=3.0,
        )
        for speed in (8.0, 10.0)
    ]


def _plot_scenario(
    path: Path,
    scenario: Scenario,
    traces: dict[str, dict[str, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for name, trace in traces.items():
        command_position_error = np.linalg.norm(
            trace["command_pose"][:, :3] - trace["target_pose"][:, :3],
            axis=1,
        )
        actual_position_error = np.linalg.norm(
            trace["actual_pose"][:, :3] - trace["target_pose"][:, :3],
            axis=1,
        )
        actual_orientation_error = _orientation_error(
            trace["target_pose"],
            trace["actual_pose"],
        )
        axes[0, 0].plot(trace["time"], command_position_error, label=name)
        axes[0, 1].plot(trace["time"], actual_position_error, label=name)
        axes[1, 0].plot(trace["time"], actual_orientation_error, label=name)
        axes[1, 1].plot(trace["time"], trace["activation"], label=name)
        axes[2, 0].plot(trace["time"], trace["actual_q"][:, 0], label=name)
        axes[2, 1].plot(trace["time"], trace["actual_elbow"][:, 1], label=name)
    axes[0, 0].set_ylabel("target-command pos [m]")
    axes[0, 1].set_ylabel("target-actual pos [m]")
    axes[1, 0].set_ylabel("target-actual rot [rad]")
    axes[1, 1].set_ylabel("position activation")
    axes[2, 0].set_ylabel("actual J1 [rad]")
    axes[2, 1].set_ylabel("actual elbow y [m]")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 1].set_xlabel("time [s]")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(scenario.name)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_study(
    output_dir: Path,
    profiles: list[ErrorProfile],
    scenarios: list[Scenario],
    *,
    braking: bool,
    save_traces: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    if save_traces:
        trace_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    selected_names = {
        "current_latch_p0p003_r0",
        "current_latch_p0p003_r0p03",
        "current_no_latch_p0p003_r0p03",
        "historical_latch_p0p003_r0p03",
        "always_cap_p0p003_r0p03",
        "position_off_p0_r0p03",
        "current_baseline",
        "current_p3_r3",
        "current_p3_r4",
        "current_no_latch_p3_r4",
        "historical_p3_r4",
        "always_p3_r4",
        "always_p6_r4",
        "always_p9_r4",
        "position_off_r4",
    }
    plot_traces: dict[str, dict[str, dict[str, np.ndarray]]] = {
        scenario.name: {} for scenario in scenarios
    }

    total = len(profiles) * len(scenarios)
    completed = 0
    for scenario in scenarios:
        for profile in profiles:
            completed += 1
            print(
                f"[{completed:03d}/{total:03d}] {scenario.name} {profile.name}",
                flush=True,
            )
            trace = simulate(profile, scenario, braking=braking)
            rows.append(
                summarize(
                    profile,
                    scenario,
                    trace,
                    braking=braking,
                )
            )
            if save_traces:
                np.savez_compressed(
                    trace_dir / f"{scenario.name}__{profile.name}.npz",
                    **trace,
                )
            if profile.name in selected_names:
                plot_traces[scenario.name][profile.name] = trace

    fields = sorted({key for row in rows for key in row})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "profiles": [asdict(profile) for profile in profiles],
                "scenarios": [asdict(scenario) for scenario in scenarios],
                "braking": braking,
                "control_dt": CONTROL_DT,
                "velocity_caps": DEFAULT_CAPS.tolist(),
                "active_dataflow_values": {
                    "position_cost": 10.0,
                    "orientation_cost": 1.0,
                    "lm_damping": 0.02,
                    "damping": 0.1,
                    "max_iters": 5,
                    "joint_braking_distance": 0.2,
                    "nullspace_cost": 12.0,
                    "nullspace_return_rate": 1.6,
                    "singularity_max_approach_rate": 0.25,
                    "kinetic_energy_cost": 5e-5,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for scenario in scenarios:
        traces = plot_traces[scenario.name]
        if traces:
            _plot_scenario(
                output_dir / f"{scenario.name}.png",
                scenario,
                traces,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("primary", "robustness", "schedule", "settling"),
        default="primary",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dev/results/frame_error_joint_tuning_20260729"),
    )
    parser.add_argument(
        "--no-braking",
        action="store_true",
        help="Disable joint braking while retaining position and velocity limits.",
    )
    parser.add_argument("--save-traces", action="store_true")
    args = parser.parse_args()

    if args.stage == "primary":
        profiles = primary_profiles()
        scenarios = primary_scenarios()
        output_dir = args.output_dir / "primary"
    elif args.stage == "robustness":
        profiles = robustness_profiles()
        scenarios = robustness_scenarios()
        output_dir = args.output_dir / "robustness"
    elif args.stage == "schedule":
        profiles = schedule_profiles()
        scenarios = robustness_scenarios()
        output_dir = args.output_dir / "schedule"
    else:
        profiles = settling_profiles()
        scenarios = settling_scenarios()
        output_dir = args.output_dir / "settling"
    if args.no_braking:
        output_dir = output_dir.with_name(f"{output_dir.name}_no_braking")
    run_study(
        output_dir,
        profiles,
        scenarios,
        braking=not args.no_braking,
        save_traces=args.save_traces,
    )


if __name__ == "__main__":
    main()
