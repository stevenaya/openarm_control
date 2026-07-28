#!/usr/bin/env python3
"""Replay intervention targets through the non-augmented elbow task.

The recorded intervention episode stores joint commands rather than raw VR
poses. The companion replay helper reconstructs end-effector targets with FK
and resamples them at 250 Hz. This script compares:

* the locally raised velocity limits that currently behave well;
* the original lower velocity limits;
* light direct ``J_psi Delta q`` regularization;
* exact-nullspace-only ``J_psi N Delta q`` regularization.

All candidates keep the production Mink QP dimensions and hard safety limits.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from openarm_control import IKParams, Kinematics
from openarm_control.speed_scheduled_elbow_task import ElbowSwivelCoordinate

from sim_intervention_posture_replay import (
    DEFAULT_RUN_DIR,
    DynamicSideArm,
    _load_recorded_commands,
    _source_geometry,
    detect_retract_segments,
)
from sim_reach_braking_experiment import CONTROL_DT, _setup
from speed_scheduled_near_singular_task import (
    SpeedScheduledNearSingularTask,
)


LOW_CAPS = np.array(
    [1.57, 1.57, 3.14, 3.14, 12.6, 12.6, 12.6],
    dtype=np.float64,
)
RAISED_CAPS = np.array(
    [3.0, 3.0, 4.14, 4.14, 12.6, 12.6, 12.6],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Candidate:
    """One velocity envelope and ordinary-Mink elbow-task configuration."""

    name: str
    caps: tuple[float, ...]
    enabled: bool = False
    projection: str = "direct"
    velocity_cost: float = 0.0
    return_rate: float = 0.0
    max_return_speed: float = 0.1
    max_return_acceleration: float = 1.0
    corridor_cost: float = 0.0
    position_error_limit: float = 0.0
    orientation_error_limit: float = 0.0
    frame_limit_linear_slow: float = 0.2
    frame_limit_linear_fast: float = 0.5
    frame_limit_activation_rise_rate: float = 4.0
    frame_limit_activation_fall_rate: float = 2.0
    near_singular_cost: float = 0.0
    near_singular_return_rate: float = 0.0
    near_singular_max_return_speed: float = 0.0
    near_singular_low: float = 0.02
    near_singular_high: float = 0.08
    linear_slow: float = 0.45
    linear_fast: float = 0.6
    activation_rise_rate: float = 4.0
    activation_fall_rate: float = 2.0


@dataclass
class Trace:
    """Signals retained for branch, tracking, smoothness, and timing checks."""

    target: np.ndarray
    command_q: np.ndarray
    command_dq: np.ndarray
    command_ddq: np.ndarray
    actual_q: np.ndarray
    actual_dq: np.ndarray
    actual_ddq: np.ndarray
    command_ee: np.ndarray
    actual_ee: np.ndarray
    command_elbow: np.ndarray
    actual_elbow: np.ndarray
    command_swivel: np.ndarray
    actual_swivel: np.ndarray
    activation: np.ndarray
    frame_limit_activation: np.ndarray
    target_speed: np.ndarray
    solve_ms: np.ndarray
    failed: np.ndarray


@dataclass(frozen=True)
class Metrics:
    """One candidate's comparison against the raised-limit reference."""

    case: str
    candidate: str
    actual_elbow_path_rmse_to_raised_m: float
    actual_elbow_phase_rmse_to_raised_m: float
    actual_position_path_gap_to_raised_m: float
    actual_position_rmse_m: float
    actual_orientation_rmse_rad: float
    actual_swivel_path_rmse_to_raised_deg: float
    actual_swivel_range_deg: float
    target_speed_mean_m_s: float
    target_speed_peak_m_s: float
    activation_mean: float
    activation_high_fraction: float
    frame_limit_activation_mean: float
    frame_limit_activation_high_fraction: float
    any_velocity_limit_fraction: float
    peak_velocity_utilization: float
    dominant_saturated_joint: int
    j1_command_accel_p99_rad_s2: float
    j4_command_accel_p99_rad_s2: float
    command_accel_p99_rad_s2: float
    actual_accel_p99_rad_s2: float
    solve_mean_ms: float
    solve_p95_ms: float
    failure_count: int


def _velocity_limits(caps: tuple[float, ...]) -> dict[str, float]:
    """Expand seven scalar arm caps to the names expected by IKParams."""
    return {
        f"openarm_{side}_joint{index + 1}": float(cap)
        for side in ("left", "right")
        for index, cap in enumerate(caps)
    }


def _make_kinematics(side: str, candidate: Candidate) -> Kinematics:
    """Build the same non-augmented solve structure used by the VR IK node."""
    kinematics = Kinematics(
        _setup(side),
        IKParams(
            position_cost=10.0,
            orientation_cost=1.0,
            frame_position_error_limit=candidate.position_error_limit,
            frame_orientation_error_limit=candidate.orientation_error_limit,
            frame_error_limit_linear_slow=candidate.frame_limit_linear_slow,
            frame_error_limit_linear_fast=candidate.frame_limit_linear_fast,
            frame_error_limit_activation_rise_rate=(
                candidate.frame_limit_activation_rise_rate
            ),
            frame_error_limit_activation_fall_rate=(
                candidate.frame_limit_activation_fall_rate
            ),
            lm_damping=0.02,
            damping=0.1,
            posture_cost=0.0,
            dt=CONTROL_DT,
            max_iters=5,
            velocity_limits=_velocity_limits(candidate.caps),
            joint_limit_recovery_velocity_scale=1.0,
            nullspace_cost=12.0,
            nullspace_return_rate=1.6,
            nullspace_max_speed=1.0,
            nullspace_singularity_low=0.02,
            nullspace_singularity_high=0.05,
            nullspace_characteristic_length=0.3,
            speed_scheduled_elbow=candidate.enabled,
            speed_elbow_linear_slow=candidate.linear_slow,
            speed_elbow_linear_fast=candidate.linear_fast,
            speed_elbow_activation_rise_rate=(candidate.activation_rise_rate),
            speed_elbow_activation_fall_rate=(candidate.activation_fall_rate),
            speed_elbow_velocity_cost=candidate.velocity_cost,
            speed_elbow_return_rate=candidate.return_rate,
            speed_elbow_max_return_speed=candidate.max_return_speed,
            speed_elbow_max_return_acceleration=(candidate.max_return_acceleration),
            speed_elbow_corridor_cost=candidate.corridor_cost,
            speed_elbow_projection=candidate.projection,
            elbow_soft_limit_cost=0.0,
            elbow_braking_guard_angle=0.08,
            elbow_braking_profile="distance",
            elbow_braking_acceleration=20.0,
            elbow_braking_slowdown_distance=0.5,
            joint_limit_braking=True,
            joint_limit_braking_slowdown_distance=0.5,
            joint_limit_braking_exponent=2.0,
            joint_limit_braking_reaction_time=0.04,
            joint_limit_braking_distance_buffer=0.01,
            singularity_approach_limit=True,
            singularity_ratio_stop=0.02,
            singularity_ratio_slow=0.08,
            singularity_max_approach_rate=0.25,
            singularity_braking_exponent=2.0,
            kinetic_energy_cost=3e-5,
        ),
    )
    if candidate.near_singular_cost > 0.0:
        solver = kinematics._ik
        assert solver is not None
        solver._speed_elbow_tasks[side] = SpeedScheduledNearSingularTask(
            model=solver._model,
            frame_task=solver._tasks[side],
            dof_indices=solver._arm_dofs_by_side[side],
            home_qpos=solver._posture_task.target_q,
            control_dt=CONTROL_DT,
            substep_dt=solver._substep_dt,
            cost=candidate.near_singular_cost,
            linear_speed_slow=candidate.linear_slow,
            linear_speed_fast=candidate.linear_fast,
            activation_rise_rate=candidate.activation_rise_rate,
            activation_fall_rate=candidate.activation_fall_rate,
            singularity_low=candidate.near_singular_low,
            singularity_high=candidate.near_singular_high,
            characteristic_length=0.3,
            return_rate=candidate.near_singular_return_rate,
            max_return_speed=candidate.near_singular_max_return_speed,
        )
    return kinematics


class ElbowMonitor:
    """Evaluate the home-referenced swivel and world elbow position."""

    def __init__(self, kinematics: Kinematics, side: str) -> None:
        solver = kinematics._ik
        assert solver is not None
        self._setup = kinematics.setup
        self._side = side
        self._base_q = solver._config.q.copy()
        resolved = (
            self._setup.joint_resolver._left
            if side == "left"
            else self._setup.joint_resolver._right
        )
        self._qpos_indices = np.asarray(resolved.arm_qpos, dtype=int)
        self._coordinate = ElbowSwivelCoordinate(
            self._setup.model,
            side,
            solver._tasks[side],
            solver._posture_task.target_q,
            np.asarray(resolved.arm_dof, dtype=int),
            finite_difference_epsilon=1e-5,
        )
        self._data = mujoco.MjData(self._setup.model)
        self._elbow_joint = self._setup.model.joint(f"openarm_{side}_joint4").id

    def evaluate(self, arm_q: np.ndarray) -> tuple[float, np.ndarray]:
        """Return swivel and elbow position at one seven-joint posture."""
        q = self._base_q.copy()
        q[self._qpos_indices] = arm_q
        swivel, _ = self._coordinate.value(q)
        self._data.qpos[:] = q
        mujoco.mj_forward(self._setup.model, self._data)
        return swivel, self._data.xanchor[self._elbow_joint].copy()


def _orientation_errors(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    target_rotation = Rotation.from_quat(target[:, [4, 5, 6, 3]])
    actual_rotation = Rotation.from_quat(actual[:, [4, 5, 6, 3]])
    return (target_rotation.inv() * actual_rotation).magnitude()


def _target_speed(target: np.ndarray) -> np.ndarray:
    speed = np.zeros(target.shape[0], dtype=np.float64)
    speed[1:] = (
        np.linalg.norm(
            np.diff(target[:, :3], axis=0),
            axis=1,
        )
        / CONTROL_DT
    )
    return speed


def simulate(
    side: str,
    candidate: Candidate,
    target: np.ndarray,
    source_q: np.ndarray,
    *,
    settle_duration: float,
) -> Trace:
    """Replay one target stream through Mink and a MuJoCo position plant."""
    kinematics = _make_kinematics(side, candidate)
    dynamics = DynamicSideArm(side, source_q[0])
    dynamics.settle(settle_duration)
    kinematics.sync(dynamics.driver_qpos())
    monitor = ElbowMonitor(kinematics, side)

    count = target.shape[0]
    command_q = np.empty((count, 7), dtype=np.float64)
    command_dq = np.empty((count, 7), dtype=np.float64)
    command_ddq = np.empty((count, 7), dtype=np.float64)
    actual_q = np.empty((count, 7), dtype=np.float64)
    actual_dq = np.empty((count, 7), dtype=np.float64)
    actual_ddq = np.empty((count, 7), dtype=np.float64)
    command_ee = np.empty((count, 7), dtype=np.float64)
    actual_ee = np.empty((count, 7), dtype=np.float64)
    command_elbow = np.empty((count, 3), dtype=np.float64)
    actual_elbow = np.empty((count, 3), dtype=np.float64)
    command_swivel = np.empty(count, dtype=np.float64)
    actual_swivel = np.empty(count, dtype=np.float64)
    activation = np.zeros(count, dtype=np.float64)
    frame_limit_activation = np.zeros(count, dtype=np.float64)
    solve_ms = np.empty(count, dtype=np.float64)
    failed = np.zeros(count, dtype=bool)

    previous_command = dynamics.q()
    previous_command_dq = np.zeros(7, dtype=np.float64)
    previous_actual_dq = dynamics.dq()
    for tick, pose in enumerate(target):
        # Production keeps the integrated IK configuration. Measured state is
        # supplied only to state-aware braking and singularity limits.
        kinematics.update_measured_state(
            dynamics.driver_qpos(),
            dynamics.driver_qvel(),
        )
        kinematics.set_target(side, pose)
        started = time.perf_counter()
        result = kinematics.solve()
        solve_ms[tick] = (time.perf_counter() - started) * 1000.0
        failed[tick] = result is None
        offset = 8 if side == "left" else 0
        current_command = (
            previous_command.copy()
            if result is None
            else result[offset : offset + 7].astype(np.float64)
        )
        current_command_dq = (current_command - previous_command) / CONTROL_DT

        dynamics.set_command(current_command)
        dynamics.step()
        current_actual_q = dynamics.q()
        current_actual_dq = dynamics.dq()

        command_q[tick] = current_command
        command_dq[tick] = current_command_dq
        command_ddq[tick] = (current_command_dq - previous_command_dq) / CONTROL_DT
        actual_q[tick] = current_actual_q
        actual_dq[tick] = current_actual_dq
        actual_ddq[tick] = (current_actual_dq - previous_actual_dq) / CONTROL_DT
        command_ee[tick] = kinematics.fk(
            side,
            np.append(current_command, 0.0),
        )
        actual_ee[tick] = dynamics.pose()
        command_swivel[tick], command_elbow[tick] = monitor.evaluate(current_command)
        actual_swivel[tick], actual_elbow[tick] = monitor.evaluate(current_actual_q)

        solver = kinematics._ik
        assert solver is not None
        task = solver._speed_elbow_tasks.get(side)
        if task is not None and task.last_state is not None:
            activation[tick] = getattr(
                task.last_state,
                "combined_activation",
                getattr(
                    task.last_state,
                    "activation",
                    getattr(task.last_state, "speed_activation", 0.0),
                ),
            )
        frame_task = solver._tasks[side]
        if hasattr(frame_task, "limit_activation"):
            frame_limit_activation[tick] = frame_task.limit_activation

        previous_command = current_command
        previous_command_dq = current_command_dq
        previous_actual_dq = current_actual_dq

    return Trace(
        target=target.copy(),
        command_q=command_q,
        command_dq=command_dq,
        command_ddq=command_ddq,
        actual_q=actual_q,
        actual_dq=actual_dq,
        actual_ddq=actual_ddq,
        command_ee=command_ee,
        actual_ee=actual_ee,
        command_elbow=command_elbow,
        actual_elbow=actual_elbow,
        command_swivel=command_swivel,
        actual_swivel=actual_swivel,
        activation=activation,
        frame_limit_activation=frame_limit_activation,
        target_speed=_target_speed(target),
        solve_ms=solve_ms,
        failed=failed,
    )


def compute_metrics(
    case: str,
    candidate: Candidate,
    trace: Trace,
    raised: Trace,
) -> Metrics:
    """Summarize tracking and how closely the low-cap branch matches raised caps."""
    caps = np.asarray(candidate.caps, dtype=np.float64)
    utilization = np.abs(trace.command_dq) / caps[None, :]
    saturated = utilization >= 0.99
    saturated_by_joint = np.mean(saturated, axis=0)
    elbow_gap = trace.actual_elbow - raised.actual_elbow
    position_gap = trace.actual_ee[:, :3] - raised.actual_ee[:, :3]
    position_error = trace.actual_ee[:, :3] - trace.target[:, :3]
    orientation_error = _orientation_errors(trace.target, trace.actual_ee)
    swivel_gap = np.unwrap(trace.actual_swivel) - np.unwrap(raised.actual_swivel)

    # Ignore startup transients caused by settling at the recorded first pose.
    warmup = min(int(round(0.1 / CONTROL_DT)), trace.target.shape[0] // 4)
    active = slice(warmup, None)
    candidate_position = trace.actual_ee[active, :3]
    raised_position = raised.actual_ee[active, :3]
    phase_distance = np.sum(
        (candidate_position[:, None, :] - raised_position[None, :, :]) ** 2,
        axis=2,
    )
    nearest_raised = np.argmin(phase_distance, axis=1)
    phase_elbow_gap = (
        trace.actual_elbow[active] - raised.actual_elbow[active][nearest_raised]
    )
    return Metrics(
        case=case,
        candidate=candidate.name,
        actual_elbow_path_rmse_to_raised_m=float(
            np.sqrt(np.mean(np.sum(elbow_gap[active] ** 2, axis=1)))
        ),
        actual_elbow_phase_rmse_to_raised_m=float(
            np.sqrt(np.mean(np.sum(phase_elbow_gap**2, axis=1)))
        ),
        actual_position_path_gap_to_raised_m=float(
            np.sqrt(np.mean(np.sum(position_gap[active] ** 2, axis=1)))
        ),
        actual_position_rmse_m=float(
            np.sqrt(np.mean(np.sum(position_error[active] ** 2, axis=1)))
        ),
        actual_orientation_rmse_rad=float(
            np.sqrt(np.mean(orientation_error[active] ** 2))
        ),
        actual_swivel_path_rmse_to_raised_deg=float(
            np.rad2deg(np.sqrt(np.mean(swivel_gap[active] ** 2)))
        ),
        actual_swivel_range_deg=float(
            np.rad2deg(np.ptp(np.unwrap(trace.actual_swivel[active])))
        ),
        target_speed_mean_m_s=float(np.mean(trace.target_speed[active])),
        target_speed_peak_m_s=float(np.max(trace.target_speed[active])),
        activation_mean=float(np.mean(trace.activation[active])),
        activation_high_fraction=float(np.mean(trace.activation[active] >= 0.8)),
        frame_limit_activation_mean=float(
            np.mean(trace.frame_limit_activation[active])
        ),
        frame_limit_activation_high_fraction=float(
            np.mean(trace.frame_limit_activation[active] >= 0.8)
        ),
        any_velocity_limit_fraction=float(np.mean(np.any(saturated[active], axis=1))),
        peak_velocity_utilization=float(np.max(utilization[active])),
        dominant_saturated_joint=int(np.argmax(saturated_by_joint) + 1),
        j1_command_accel_p99_rad_s2=float(
            np.quantile(np.abs(trace.command_ddq[active, 0]), 0.99)
        ),
        j4_command_accel_p99_rad_s2=float(
            np.quantile(np.abs(trace.command_ddq[active, 3]), 0.99)
        ),
        command_accel_p99_rad_s2=float(
            np.quantile(np.abs(trace.command_ddq[active]), 0.99)
        ),
        actual_accel_p99_rad_s2=float(
            np.quantile(np.abs(trace.actual_ddq[active]), 0.99)
        ),
        solve_mean_ms=float(np.mean(trace.solve_ms[active])),
        solve_p95_ms=float(np.quantile(trace.solve_ms[active], 0.95)),
        failure_count=int(np.count_nonzero(trace.failed)),
    )


def _candidates(quick: bool) -> list[Candidate]:
    candidates = [
        Candidate("raised_reference", tuple(RAISED_CAPS)),
        Candidate("low_baseline", tuple(LOW_CAPS)),
        Candidate(
            "low_direct_hold_c0p001",
            tuple(LOW_CAPS),
            enabled=True,
            velocity_cost=0.001,
        ),
        Candidate(
            "low_direct_hold_c0p002",
            tuple(LOW_CAPS),
            enabled=True,
            velocity_cost=0.002,
        ),
        Candidate(
            "low_direct_hold_c0p004",
            tuple(LOW_CAPS),
            enabled=True,
            velocity_cost=0.004,
        ),
        Candidate(
            "low_direct_hold_c0p008",
            tuple(LOW_CAPS),
            enabled=True,
            velocity_cost=0.008,
        ),
        Candidate(
            "low_direct_home_c0p002_r0p1",
            tuple(LOW_CAPS),
            enabled=True,
            velocity_cost=0.002,
            return_rate=0.1,
            max_return_speed=0.05,
        ),
        Candidate(
            "low_projected_hold_c0p008",
            tuple(LOW_CAPS),
            enabled=True,
            projection="exact-nullspace",
            velocity_cost=0.008,
        ),
        Candidate(
            "low_leak_p0p2mm",
            tuple(LOW_CAPS),
            position_error_limit=0.0002,
            orientation_error_limit=0.0048,
        ),
        Candidate(
            "low_leak_p0p3mm",
            tuple(LOW_CAPS),
            position_error_limit=0.0003,
            orientation_error_limit=0.0048,
        ),
        Candidate(
            "low_leak_p0p4mm",
            tuple(LOW_CAPS),
            position_error_limit=0.0004,
            orientation_error_limit=0.0048,
        ),
        Candidate(
            "low_leak_p0p6mm",
            tuple(LOW_CAPS),
            position_error_limit=0.0006,
            orientation_error_limit=0.0048,
        ),
        Candidate(
            "low_leak_position_only_p0p8mm",
            tuple(LOW_CAPS),
            position_error_limit=0.0008,
        ),
        Candidate(
            "low_leak_position_only_p1p0mm",
            tuple(LOW_CAPS),
            position_error_limit=0.0010,
        ),
        Candidate(
            "low_leak_position_only_p1p2mm",
            tuple(LOW_CAPS),
            position_error_limit=0.0012,
        ),
        Candidate(
            "low_leak_position_only_p1p5mm",
            tuple(LOW_CAPS),
            position_error_limit=0.0015,
        ),
        Candidate(
            "low_leak_position_only_p2p0mm",
            tuple(LOW_CAPS),
            position_error_limit=0.0020,
        ),
        Candidate(
            "low_leak_position_only_p2p5mm",
            tuple(LOW_CAPS),
            position_error_limit=0.0025,
        ),
        Candidate(
            "low_leak_position_only_p3p0mm",
            tuple(LOW_CAPS),
            position_error_limit=0.0030,
        ),
        Candidate(
            "low_leak_p3p0mm_window_0p3_0p6",
            tuple(LOW_CAPS),
            position_error_limit=0.0030,
            frame_limit_linear_slow=0.3,
            frame_limit_linear_fast=0.6,
        ),
        Candidate(
            "low_leak_p3p0mm_window_0p4_0p8",
            tuple(LOW_CAPS),
            position_error_limit=0.0030,
            frame_limit_linear_slow=0.4,
            frame_limit_linear_fast=0.8,
        ),
        Candidate(
            "low_leak_p3p0mm_window_0p5_1p0",
            tuple(LOW_CAPS),
            position_error_limit=0.0030,
            frame_limit_linear_slow=0.5,
            frame_limit_linear_fast=1.0,
        ),
        Candidate(
            "low_leak_p8p0mm_window_0p5_1p0",
            tuple(LOW_CAPS),
            position_error_limit=0.0080,
            frame_limit_linear_slow=0.5,
            frame_limit_linear_fast=1.0,
        ),
        Candidate(
            "low_leak_p10mm_window_0p5_1p0",
            tuple(LOW_CAPS),
            position_error_limit=0.0100,
            frame_limit_linear_slow=0.5,
            frame_limit_linear_fast=1.0,
        ),
        Candidate(
            "low_leak_p15mm_window_0p5_1p0",
            tuple(LOW_CAPS),
            position_error_limit=0.0150,
            frame_limit_linear_slow=0.5,
            frame_limit_linear_fast=1.0,
        ),
        Candidate(
            "low_leak_p20mm_window_0p5_1p0",
            tuple(LOW_CAPS),
            position_error_limit=0.0200,
            frame_limit_linear_slow=0.5,
            frame_limit_linear_fast=1.0,
        ),
        Candidate(
            "low_leak_position_only_p4p0mm",
            tuple(LOW_CAPS),
            position_error_limit=0.0040,
        ),
        Candidate(
            "low_leak_position_only_p5p0mm",
            tuple(LOW_CAPS),
            position_error_limit=0.0050,
        ),
        Candidate(
            "low_direct_early_c0p002",
            tuple(LOW_CAPS),
            enabled=True,
            velocity_cost=0.002,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_direct_early_c0p004",
            tuple(LOW_CAPS),
            enabled=True,
            velocity_cost=0.004,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_near_singular_hold_c1",
            tuple(LOW_CAPS),
            near_singular_cost=1.0,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_near_singular_hold_c3",
            tuple(LOW_CAPS),
            near_singular_cost=3.0,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_near_singular_hold_c6",
            tuple(LOW_CAPS),
            near_singular_cost=6.0,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_near_singular_hold_c12",
            tuple(LOW_CAPS),
            near_singular_cost=12.0,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_near_singular_home_c3",
            tuple(LOW_CAPS),
            near_singular_cost=3.0,
            near_singular_return_rate=0.2,
            near_singular_max_return_speed=0.1,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_near_singular_wide_c3",
            tuple(LOW_CAPS),
            near_singular_cost=3.0,
            near_singular_low=0.04,
            near_singular_high=0.2,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_near_singular_wide_c6",
            tuple(LOW_CAPS),
            near_singular_cost=6.0,
            near_singular_low=0.04,
            near_singular_high=0.2,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_near_singular_wide_c12",
            tuple(LOW_CAPS),
            near_singular_cost=12.0,
            near_singular_low=0.04,
            near_singular_high=0.2,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_near_singular_wide_c30",
            tuple(LOW_CAPS),
            near_singular_cost=30.0,
            near_singular_low=0.04,
            near_singular_high=0.2,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_near_singular_wide_c100",
            tuple(LOW_CAPS),
            near_singular_cost=100.0,
            near_singular_low=0.04,
            near_singular_high=0.2,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_corridor_only_early_c0p002",
            tuple(LOW_CAPS),
            enabled=True,
            corridor_cost=0.002,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_corridor_only_early_c0p0002",
            tuple(LOW_CAPS),
            enabled=True,
            corridor_cost=0.0002,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_corridor_only_early_c0p0005",
            tuple(LOW_CAPS),
            enabled=True,
            corridor_cost=0.0005,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_corridor_only_early_c0p001",
            tuple(LOW_CAPS),
            enabled=True,
            corridor_cost=0.001,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_corridor_only_early_c0p004",
            tuple(LOW_CAPS),
            enabled=True,
            corridor_cost=0.004,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_corridor_only_early_c0p008",
            tuple(LOW_CAPS),
            enabled=True,
            corridor_cost=0.008,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_leak_p3p0mm_direct_early_c0p002",
            tuple(LOW_CAPS),
            enabled=True,
            velocity_cost=0.002,
            position_error_limit=0.0030,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_leak_p0p3mm_direct_c0p001",
            tuple(LOW_CAPS),
            enabled=True,
            velocity_cost=0.001,
            position_error_limit=0.0003,
            orientation_error_limit=0.0048,
        ),
    ]
    return candidates[:5] if quick else candidates


def _save_trace(path: Path, trace: Trace) -> None:
    np.savez_compressed(path, **vars(trace))


def _write_rows(path: Path, rows: list[Metrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--episode", type=int, default=202)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--segments", type=int, nargs="+", default=[13])
    parser.add_argument("--include-reverse", action="store_true")
    parser.add_argument("--settle-duration", type=float, default=0.5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--candidates",
        nargs="+",
        help="Run only the named candidate configurations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dev/results/speed_scheduled_elbow_replay"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    timestamp, raw_q = _load_recorded_commands(
        args.run_dir,
        args.episode,
        args.side,
    )
    time_grid, poses, source_q, shoulder, source_elbow = _source_geometry(
        args.side,
        timestamp,
        raw_q,
    )
    segments = detect_retract_segments(
        time_grid,
        poses,
        shoulder,
        source_elbow,
    )
    selected = {segment.index: segment for segment in segments}
    cases: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index in args.segments:
        segment = selected[index]
        window = slice(segment.start, segment.end)
        cases[str(index)] = (poses[window], source_q[window])
        if args.include_reverse:
            cases[f"reverse_{index}"] = (
                poses[window][::-1].copy(),
                source_q[window][::-1].copy(),
            )

    candidates = _candidates(args.quick)
    if args.candidates:
        selected_candidates = set(args.candidates)
        candidates = [
            candidate
            for candidate in candidates
            if candidate.name in selected_candidates
        ]
        missing = selected_candidates - {candidate.name for candidate in candidates}
        if missing:
            parser.error(f"Unknown candidates: {', '.join(sorted(missing))}")
        if "raised_reference" not in selected_candidates:
            parser.error("--candidates must include raised_reference.")
    rows: list[Metrics] = []
    for case, (target, source) in cases.items():
        traces: dict[str, Trace] = {}
        for candidate in candidates:
            print(f"{case}: {candidate.name}", flush=True)
            trace = simulate(
                args.side,
                candidate,
                target,
                source,
                settle_duration=args.settle_duration,
            )
            traces[candidate.name] = trace
            _save_trace(
                args.output / f"trace_{case}_{candidate.name}.npz",
                trace,
            )
        raised = traces["raised_reference"]
        for candidate in candidates:
            metric = compute_metrics(
                case,
                candidate,
                traces[candidate.name],
                raised,
            )
            rows.append(metric)
            print(
                f"  {candidate.name:32s} "
                f"elbow={100.0 * metric.actual_elbow_path_rmse_to_raised_m:5.2f}cm "
                f"phase={100.0 * metric.actual_elbow_phase_rmse_to_raised_m:5.2f}cm "
                f"posgap={100.0 * metric.actual_position_path_gap_to_raised_m:5.2f}cm "
                f"ddq1/4={metric.j1_command_accel_p99_rad_s2:5.1f}/"
                f"{metric.j4_command_accel_p99_rad_s2:5.1f} "
                f"solve95={metric.solve_p95_ms:4.2f}ms"
            )

    _write_rows(args.output / "metrics.csv", rows)
    with (args.output / "parameters.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            {
                "run_dir": str(args.run_dir),
                "episode": args.episode,
                "side": args.side,
                "segments": args.segments,
                "include_reverse": args.include_reverse,
                "candidates": {
                    candidate.name: asdict(candidate) for candidate in candidates
                },
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
