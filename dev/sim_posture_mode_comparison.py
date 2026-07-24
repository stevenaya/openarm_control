#!/usr/bin/env python3
"""Compare home-posture and nullspace regulation during reach and retract.

The desired end-effector pose follows a smooth shoulder-height trajectory:

    hold -> extend beyond reach -> hold -> retract -> hold

Mink runs at 250 Hz and drives a separate MuJoCo position-controlled plant.
The experiment records both command and dynamic state so task behavior can be
separated from actuator lag.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mink
import mujoco
import numpy as np
import numpy.typing as npt

from openarm_control import IKParams, Kinematics
from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S
from openarm_control.kinematics import (
    _arm_qpos_indices,
    _configuration_limit_for_qpos,
    _dof_indices_for_qpos,
)
from openarm_control.nullspace_posture_task import (
    smoothstep_activation,
    structural_nullspace_direction,
)

from sim_reach_braking_experiment import (
    CONTROL_DT,
    INITIAL_RIGHT,
    DynamicArm,
    _setup,
)


CHARACTERISTIC_LENGTH = 0.3
MAX_ITERS = 10
HISTORICAL_SUBSTEP_DT = 0.1
HISTORICAL_OUTER_HZ = 250.0


@dataclass(frozen=True)
class Mode:
    """One solver/task configuration."""

    name: str
    task: str
    posture_cost: float = 0.0
    position_cost: float = 10.0
    orientation_cost: float = 1.0
    lm_damping: float = 0.02
    historical_timing: bool = False
    full_safety: bool = False
    initial_caps: str = "current"
    retract_caps: str = "current"
    nullspace_cost: float = 10.0
    nullspace_return_rate: float = 0.8
    nullspace_max_speed: float = 0.8


@dataclass(frozen=True)
class DirectNullspaceState:
    """Diagnostics for direct nullspace posture regulation."""

    direction: np.ndarray
    singular_values: np.ndarray
    singularity_ratio: float
    activation: float
    effective_cost: float
    posture_error: float
    return_speed: float
    displacement: float
    jacobian_residual: float


class DirectNullspacePostureTask(mink.Task):
    r"""Penalize the full current nullspace displacement from home.

    The objective is

        cost^2 * ||z.T * Delta q + z.T * (q - q_home)||^2

    Unlike ``NullspacePostureTask``, it does not convert the posture error to a
    bounded physical return speed before building the QP objective.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        frame_task: mink.FrameTask,
        dof_indices: npt.ArrayLike,
        home_qpos: npt.ArrayLike,
        *,
        cost: float,
        singularity_low: float,
        singularity_high: float,
        characteristic_length: float,
    ) -> None:
        if cost < 0.0:
            raise ValueError("cost must be non-negative.")
        indices = np.asarray(dof_indices, dtype=int)
        if indices.shape != (7,):
            raise ValueError(f"Expected seven arm DoFs, got {indices.shape}.")
        home = np.asarray(home_qpos, dtype=np.float64)
        if home.shape != (model.nq,):
            raise ValueError(f"Expected home shape ({model.nq},), got {home.shape}.")

        super().__init__(cost=np.array([cost], dtype=np.float64))
        self._base_cost = float(cost)
        self._model = model
        self._frame_task = frame_task
        self._dof_indices = indices.copy()
        self._home_qpos = home.copy()
        self._singularity_low = float(singularity_low)
        self._singularity_high = float(singularity_high)
        self._characteristic_length = float(characteristic_length)
        self._previous_direction: np.ndarray | None = None
        self.last_state: DirectNullspaceState | None = None

    def _compute_terms(
        self,
        configuration: mink.Configuration,
    ) -> tuple[np.ndarray, np.ndarray]:
        frame_jacobian = self._frame_task.compute_jacobian(configuration)
        arm_jacobian = frame_jacobian[:, self._dof_indices]
        normalized = arm_jacobian.copy()
        normalized[:3] /= self._characteristic_length
        direction, singular_values = structural_nullspace_direction(
            normalized,
            self._previous_direction,
        )
        self._previous_direction = direction.copy()

        largest = float(singular_values[0])
        ratio = float(singular_values[-1] / largest) if largest > 0.0 else 0.0
        activation = smoothstep_activation(
            ratio,
            self._singularity_low,
            self._singularity_high,
        )
        self.cost[0] = np.sqrt(activation) * self._base_cost

        configuration_error = np.empty(self._model.nv, dtype=np.float64)
        mujoco.mj_differentiatePos(
            self._model,
            configuration_error,
            1.0,
            self._home_qpos,
            configuration.q,
        )
        posture_error = float(
            direction @ configuration_error[self._dof_indices]
        )
        jacobian = np.zeros((1, self._model.nv), dtype=np.float64)
        jacobian[0, self._dof_indices] = direction
        error = np.array([posture_error], dtype=np.float64)

        self.last_state = DirectNullspaceState(
            direction=direction.copy(),
            singular_values=singular_values.copy(),
            singularity_ratio=ratio,
            activation=activation,
            effective_cost=float(self.cost[0]),
            posture_error=posture_error,
            return_speed=np.nan,
            displacement=-posture_error,
            jacobian_residual=float(np.linalg.norm(arm_jacobian @ direction)),
        )
        return error, jacobian

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        error, _ = self._compute_terms(configuration)
        return error

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        _, jacobian = self._compute_terms(configuration)
        return jacobian

    def compute_qp_objective(
        self,
        configuration: mink.Configuration,
    ) -> mink.Objective:
        error, jacobian = self._compute_terms(configuration)
        return self._assemble_qp(error, jacobian, configuration._eye_nv)


@dataclass
class Trace:
    """Arrays recorded from one simulation."""

    time: np.ndarray
    phase: np.ndarray
    target_pose: np.ndarray
    command_q: np.ndarray
    command_dq: np.ndarray
    actual_q: np.ndarray
    actual_dq: np.ndarray
    actual_ddq: np.ndarray
    command_ee: np.ndarray
    actual_ee: np.ndarray
    command_rho: np.ndarray
    actual_rho: np.ndarray
    task_rho: np.ndarray
    geometric_home_error: np.ndarray
    geometric_nullspace_speed: np.ndarray
    nullspace_activation: np.ndarray
    nullspace_task_error: np.ndarray
    nullspace_return_speed: np.ndarray
    singularity_allowed_rate: np.ndarray
    velocity_caps: np.ndarray
    actuator_force: np.ndarray
    actuator_force_caps: np.ndarray
    bias_force: np.ndarray
    solve_failed: np.ndarray


@dataclass
class Metrics:
    """Summary of one retract segment."""

    mode: str
    retract_peak_speed_m_s: float
    retract_duration_s: float
    min_command_rho: float
    min_actual_rho: float
    retract_command_q4_delta_rad: float
    retract_actual_q4_delta_rad: float
    retract_command_q4_path_excess_rad: float
    retract_actual_q4_path_excess_rad: float
    retract_peak_abs_command_dq1_rad_s: float
    retract_peak_abs_command_dq4_rad_s: float
    retract_peak_abs_actual_dq1_rad_s: float
    retract_peak_abs_actual_dq4_rad_s: float
    retract_peak_abs_actual_ddq4_rad_s2: float
    retract_q1_velocity_limit_fraction: float
    retract_q4_velocity_limit_fraction: float
    retract_any_velocity_limit_fraction: float
    retract_q1_force_limit_fraction: float
    retract_q4_force_limit_fraction: float
    retract_any_force_limit_fraction: float
    retract_peak_abs_actuator_force_q1_nm: float
    retract_peak_abs_actuator_force_q4_nm: float
    retract_peak_abs_bias_force_q1_nm: float
    retract_peak_abs_bias_force_q4_nm: float
    retract_command_tracking_rmse_m: float
    retract_actual_tracking_rmse_m: float
    retract_max_command_actual_q4_error_rad: float
    retract_actual_ee_z_p2p_m: float
    retract_command_q4_reversals: int
    retract_actual_q4_reversals: int
    q4_bend_onset_s: float
    q4_bend_onset_target_gap_m: float
    target_reenters_command_workspace_s: float
    geometric_home_error_initial_abs_rad: float
    geometric_home_error_start_abs_rad: float
    geometric_home_error_retract_end_abs_rad: float
    geometric_home_error_final_abs_rad: float
    min_nullspace_activation: float
    peak_requested_nullspace_return_speed_rad_s: float
    solve_failure_count: int


def _caps(kind: str) -> np.ndarray:
    if kind == "current":
        return np.asarray(ARM_JOINT_VELOCITY_LIMITS_RAD_S, dtype=np.float64)
    if kind == "old":
        return np.ones(7, dtype=np.float64)
    if kind == "high":
        return np.full(7, 50.0, dtype=np.float64)
    raise ValueError(f"Unknown velocity cap profile: {kind}")


def _velocity_mapping(caps: np.ndarray) -> dict[str, float]:
    return {
        f"openarm_{side}_joint{index + 1}": float(cap)
        for side in ("left", "right")
        for index, cap in enumerate(caps)
    }


def _historical_inner_caps(physical_caps: np.ndarray) -> np.ndarray:
    """Reproduce the historical CLI conversion before Mink VelocityLimit."""
    return physical_caps / (
        MAX_ITERS * HISTORICAL_SUBSTEP_DT * HISTORICAL_OUTER_HZ
    )


def _make_kinematics(mode: Mode, cap_kind: str) -> Kinematics:
    physical_caps = _caps(cap_kind)
    historical = mode.historical_timing
    inner_caps = (
        _historical_inner_caps(physical_caps) if historical else physical_caps
    )
    full_safety = mode.full_safety
    task_is_rate = mode.task == "nullspace_rate"
    control_dt = (
        MAX_ITERS * HISTORICAL_SUBSTEP_DT if historical else CONTROL_DT
    )
    kinematics = Kinematics(
        _setup("right"),
        IKParams(
            position_cost=mode.position_cost,
            orientation_cost=mode.orientation_cost,
            lm_damping=mode.lm_damping,
            damping=0.1,
            posture_cost=mode.posture_cost,
            dt=control_dt,
            max_iters=MAX_ITERS,
            velocity_limits=_velocity_mapping(inner_caps),
            joint_limit_recovery_velocity_scale=1.0,
            nullspace_cost=mode.nullspace_cost if task_is_rate else 0.0,
            nullspace_return_rate=mode.nullspace_return_rate,
            nullspace_max_speed=mode.nullspace_max_speed,
            nullspace_singularity_low=0.02,
            nullspace_singularity_high=0.05,
            nullspace_characteristic_length=CHARACTERISTIC_LENGTH,
            elbow_soft_limit_cost=0.0,
            elbow_braking_guard_angle=0.08,
            joint_limit_braking=full_safety,
            joint_limit_braking_slowdown_distance=0.5,
            joint_limit_braking_exponent=2.0,
            joint_limit_braking_reaction_time=0.04,
            joint_limit_braking_distance_buffer=0.01,
            singularity_approach_limit=full_safety,
            singularity_ratio_stop=0.02,
            singularity_ratio_slow=0.08,
            singularity_max_approach_rate=0.25,
            singularity_braking_exponent=2.0,
            kinetic_energy_cost=3e-5 if full_safety else 0.0,
        ),
    )
    solver = kinematics._ik
    assert solver is not None

    if historical:
        active_qpos = {
            int(index)
            for side in solver._sides
            for index in solver._arm_qpos_by_side[side]
        }
        solver._limits = [
            _configuration_limit_for_qpos(solver._model, active_qpos),
            mink.VelocityLimit(solver._model, _velocity_mapping(inner_caps)),
        ]

    if mode.task == "nullspace_direct":
        arm_qpos = _arm_qpos_indices(kinematics.setup, "right")
        home_qpos = solver._posture_task.target_q
        assert home_qpos is not None
        solver._nullspace_tasks["right"] = DirectNullspacePostureTask(
            model=solver._model,
            frame_task=solver._tasks["right"],
            dof_indices=_dof_indices_for_qpos(solver._model, arm_qpos),
            home_qpos=home_qpos,
            cost=mode.nullspace_cost,
            singularity_low=0.02,
            singularity_high=0.05,
            characteristic_length=CHARACTERISTIC_LENGTH,
        )
    return kinematics


def _command_state16(kinematics: Kinematics, right_q: np.ndarray) -> np.ndarray:
    solver = kinematics._ik
    assert solver is not None
    left, left_gripper = kinematics.setup.joint_resolver.get_driver(
        solver._config.q,
        "left",
    )
    return np.concatenate(
        [
            np.append(np.asarray(right_q, dtype=np.float64), 0.0),
            np.append(left, left_gripper),
        ]
    ).astype(np.float32)


def _nullspace_offset_configuration(offset: float) -> np.ndarray:
    """Find an equivalent initial pose displaced along the arm nullspace."""
    if np.isclose(offset, 0.0):
        return INITIAL_RIGHT[:7].copy()

    kinematics = Kinematics(
        _setup("right"),
        IKParams(
            position_cost=1000.0,
            orientation_cost=1000.0,
            lm_damping=0.01,
            damping=1e-4,
            posture_cost=1.0,
            dt=0.02,
            max_iters=20,
            velocity_limits=None,
            nullspace_cost=0.0,
        ),
    )
    target_pose = kinematics.fk("right", INITIAL_RIGHT).astype(np.float64)
    kinematics.sync(_command_state16(kinematics, INITIAL_RIGHT[:7]))
    solver = kinematics._ik
    assert solver is not None

    frame_task = solver._tasks["right"]
    arm_dofs = solver._arm_dofs_by_side["right"]
    arm_qpos = solver._arm_qpos_by_side["right"]
    geometric = solver._config.get_frame_jacobian(
        frame_task.frame_name,
        frame_task.frame_type,
    )[:, arm_dofs]
    geometric[:3] /= CHARACTERISTIC_LENGTH
    direction, _ = structural_nullspace_direction(geometric)

    posture_target = solver._config.q.copy()
    posture_target[arm_qpos] += offset * direction
    solver._posture_task.set_target(posture_target)
    result: np.ndarray | None = None
    for _ in range(300):
        kinematics.set_target("right", target_pose)
        result = kinematics.solve()
        if result is None:
            raise RuntimeError("Failed to construct nullspace-offset initial pose.")
    assert result is not None
    arm_q = result[:7].astype(np.float64)
    achieved = kinematics.fk("right", np.append(arm_q, 0.0)).astype(np.float64)
    position_error = float(np.linalg.norm(achieved[:3] - target_pose[:3]))
    quaternion_alignment = float(abs(achieved[3:] @ target_pose[3:]))
    if position_error > 1e-4 or quaternion_alignment < 1.0 - 1e-6:
        raise RuntimeError(
            "Nullspace-offset pose construction drifted from its frame target: "
            f"position_error={position_error:.3e}, "
            f"quaternion_alignment={quaternion_alignment:.9f}."
        )
    return arm_q


def _set_dynamic_initial_configuration(
    dynamics: DynamicArm,
    arm_q: np.ndarray,
) -> None:
    """Reset the dynamic plant to a supplied right-arm configuration."""
    dynamics.resolver.set_qpos(
        dynamics.data.qpos,
        np.append(np.asarray(arm_q, dtype=np.float64), 0.0),
        "right",
    )
    dynamics.data.qvel[:] = 0.0
    mujoco.mj_forward(dynamics.model, dynamics.data)
    dynamics._held_qpos = dynamics.data.qpos.copy()
    dynamics.set_command(arm_q)


class ConfigurationDiagnostics:
    """Evaluate geometric and task Jacobian diagnostics at command q."""

    def __init__(self, kinematics: Kinematics) -> None:
        solver = kinematics._ik
        assert solver is not None
        self._model = solver._model
        self._resolver = kinematics.setup.joint_resolver
        self._frame_task = solver._tasks["right"]
        self._dofs = solver._arm_dofs_by_side["right"]
        self._home = solver._posture_task.target_q
        assert self._home is not None
        self._base_q = solver._config.q.copy()
        self._command_configuration = mink.Configuration(self._model)
        self._actual_configuration = mink.Configuration(self._model)
        self._previous_direction: np.ndarray | None = None

    def _configuration(
        self,
        arm_q: np.ndarray,
        scratch: mink.Configuration,
    ) -> mink.Configuration:
        qpos = self._base_q.copy()
        self._resolver.set_qpos(
            qpos,
            np.append(np.asarray(arm_q, dtype=np.float64), 0.0),
            "right",
        )
        scratch.update(q=qpos)
        return scratch

    def command(
        self,
        arm_q: np.ndarray,
        arm_dq: np.ndarray,
    ) -> tuple[float, float, float, float]:
        configuration = self._configuration(
            arm_q,
            self._command_configuration,
        )
        geometric = configuration.get_frame_jacobian(
            self._frame_task.frame_name,
            self._frame_task.frame_type,
        )[:, self._dofs]
        normalized_geometric = geometric.copy()
        normalized_geometric[:3] /= CHARACTERISTIC_LENGTH
        direction, singular_values = structural_nullspace_direction(
            normalized_geometric,
            self._previous_direction,
        )
        self._previous_direction = direction.copy()
        geometric_ratio = float(singular_values[-1] / singular_values[0])

        task_jacobian = self._frame_task.compute_jacobian(configuration)[
            :, self._dofs
        ]
        normalized_task = task_jacobian.copy()
        normalized_task[:3] /= CHARACTERISTIC_LENGTH
        task_values = np.linalg.svd(normalized_task, compute_uv=False)
        task_ratio = float(task_values[-1] / task_values[0])

        error = np.empty(self._model.nv, dtype=np.float64)
        mujoco.mj_differentiatePos(
            self._model,
            error,
            1.0,
            self._home,
            configuration.q,
        )
        home_error = float(direction @ error[self._dofs])
        nullspace_speed = float(direction @ np.asarray(arm_dq))
        return geometric_ratio, task_ratio, home_error, nullspace_speed

    def actual_ratio(self, arm_q: np.ndarray) -> float:
        configuration = self._configuration(
            arm_q,
            self._actual_configuration,
        )
        geometric = configuration.get_frame_jacobian(
            self._frame_task.frame_name,
            self._frame_task.frame_type,
        )[:, self._dofs]
        geometric[:3] /= CHARACTERISTIC_LENGTH
        singular_values = np.linalg.svd(geometric, compute_uv=False)
        return float(singular_values[-1] / singular_values[0])


def _quintic_progress(index: int, count: int) -> float:
    u = float(np.clip((index + 1) / count, 0.0, 1.0))
    return u**3 * (10.0 + u * (-15.0 + 6.0 * u))


def _motion_ticks(distance: float, peak_speed: float) -> int:
    if distance <= 0.0 or peak_speed <= 0.0:
        raise ValueError("Distance and peak speed must be positive.")
    # max derivative of the quintic smoothstep is 1.875 at u=0.5.
    duration = 1.875 * distance / peak_speed
    return max(1, int(np.ceil(duration / CONTROL_DT)))


def _target_schedule(
    distance: float,
    extend_peak_speed: float,
    retract_peak_speed: float,
    pre_hold: float,
    far_hold: float,
    post_hold: float,
    near_offset_xyz: npt.ArrayLike | None = None,
    far_offset_xyz: npt.ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    near_offset = (
        np.zeros(3, dtype=np.float64)
        if near_offset_xyz is None
        else np.asarray(near_offset_xyz, dtype=np.float64)
    )
    far_offset = (
        np.array([distance, 0.0, 0.0], dtype=np.float64)
        if far_offset_xyz is None
        else np.asarray(far_offset_xyz, dtype=np.float64)
    )
    if near_offset.shape != (3,) or far_offset.shape != (3,):
        raise ValueError("Near and far target offsets must have shape (3,).")
    path_delta = far_offset - near_offset
    path_distance = float(np.linalg.norm(path_delta))
    if path_distance <= 0.0:
        raise ValueError("Near and far target offsets must be distinct.")

    pre_ticks = int(round(pre_hold / CONTROL_DT))
    extend_ticks = _motion_ticks(path_distance, extend_peak_speed)
    far_ticks = int(round(far_hold / CONTROL_DT))
    retract_ticks = _motion_ticks(path_distance, retract_peak_speed)
    post_ticks = int(round(post_hold / CONTROL_DT))

    phase = np.concatenate(
        [
            np.full(pre_ticks, 0, dtype=np.int8),
            np.full(extend_ticks, 1, dtype=np.int8),
            np.full(far_ticks, 2, dtype=np.int8),
            np.full(retract_ticks, 3, dtype=np.int8),
            np.full(post_ticks, 4, dtype=np.int8),
        ]
    )
    offset = np.empty((phase.size, 3), dtype=np.float64)
    cursor = 0
    offset[cursor : cursor + pre_ticks] = near_offset
    cursor += pre_ticks
    for index in range(extend_ticks):
        progress = _quintic_progress(index, extend_ticks)
        offset[cursor + index] = near_offset + progress * path_delta
    cursor += extend_ticks
    offset[cursor : cursor + far_ticks] = far_offset
    cursor += far_ticks
    for index in range(retract_ticks):
        progress = _quintic_progress(index, retract_ticks)
        offset[cursor + index] = far_offset - progress * path_delta
    cursor += retract_ticks
    offset[cursor : cursor + post_ticks] = near_offset
    return phase, offset


def simulate(
    mode: Mode,
    retract_peak_speed: float,
    *,
    initial_arm_q: np.ndarray,
    distance: float,
    extend_peak_speed: float,
    settle_duration: float,
    pre_hold: float,
    far_hold: float,
    post_hold: float,
    near_offset_xyz: npt.ArrayLike | None = None,
    far_offset_xyz: npt.ArrayLike | None = None,
) -> Trace:
    """Run one command-plus-dynamics reach/retract simulation."""
    phase, target_offset = _target_schedule(
        distance,
        extend_peak_speed,
        retract_peak_speed,
        pre_hold,
        far_hold,
        post_hold,
        near_offset_xyz,
        far_offset_xyz,
    )
    kinematics = _make_kinematics(mode, mode.initial_caps)
    dynamics = DynamicArm()
    _set_dynamic_initial_configuration(dynamics, initial_arm_q)
    dynamics.settle(settle_duration)
    kinematics.sync(dynamics.bimanual_driver_qpos())
    initial_pose = dynamics.right_ee()
    diagnostics = ConfigurationDiagnostics(kinematics)

    count = phase.size
    target_pose_values = np.empty((count, 7))
    command_q_values = np.empty((count, 7))
    command_dq_values = np.empty((count, 7))
    actual_q_values = np.empty((count, 7))
    actual_dq_values = np.empty((count, 7))
    actual_ddq_values = np.empty((count, 7))
    command_ee_values = np.empty((count, 7))
    actual_ee_values = np.empty((count, 7))
    command_rho_values = np.empty(count)
    actual_rho_values = np.empty(count)
    task_rho_values = np.empty(count)
    home_error_values = np.empty(count)
    nullspace_speed_values = np.empty(count)
    activation_values = np.full(count, np.nan)
    task_error_values = np.full(count, np.nan)
    return_speed_values = np.full(count, np.nan)
    singularity_rate_values = np.full(count, np.nan)
    velocity_cap_values = np.empty((count, 7))
    actuator_force_values = np.empty((count, 7))
    actuator_force_cap_values = np.empty((count, 7))
    bias_force_values = np.empty((count, 7))
    failed_values = np.zeros(count, dtype=bool)

    previous_command = dynamics.right_q()
    previous_actual_dq = dynamics.right_dq()
    previous_phase = int(phase[0])
    active_caps = _caps(mode.initial_caps)

    for tick in range(count):
        current_phase = int(phase[tick])
        if (
            current_phase == 3
            and previous_phase != 3
            and mode.retract_caps != mode.initial_caps
        ):
            command_state = _command_state16(kinematics, previous_command)
            kinematics = _make_kinematics(mode, mode.retract_caps)
            kinematics.sync(command_state)
            diagnostics = ConfigurationDiagnostics(kinematics)
            active_caps = _caps(mode.retract_caps)

        target_pose = initial_pose.copy()
        target_pose[:3] += target_offset[tick]
        if mode.full_safety:
            kinematics.update_measured_state(
                dynamics.bimanual_driver_qpos(),
                dynamics.bimanual_driver_qvel(),
            )
        kinematics.set_target("right", target_pose)
        result = kinematics.solve()
        if result is None:
            command_q = previous_command.copy()
            failed_values[tick] = True
        else:
            command_q = result[:7].astype(np.float64)
        command_dq = (command_q - previous_command) / CONTROL_DT

        dynamics.set_command(command_q)
        dynamics.step_control_period()
        actual_q = dynamics.right_q()
        actual_dq = dynamics.right_dq()
        actual_ddq = (actual_dq - previous_actual_dq) / CONTROL_DT
        command_ee = kinematics.fk(
            "right",
            np.append(command_q, 0.0),
        ).astype(np.float64)
        actual_ee = dynamics.right_ee()
        (
            command_rho,
            task_rho,
            home_error,
            nullspace_speed,
        ) = diagnostics.command(command_q, command_dq)

        solver = kinematics._ik
        assert solver is not None
        nullspace_task = solver._nullspace_tasks.get("right")
        if nullspace_task is not None and nullspace_task.last_state is not None:
            state = nullspace_task.last_state
            activation_values[tick] = state.activation
            task_error_values[tick] = state.posture_error
            return_speed_values[tick] = state.return_speed
        singularity_limit = solver._singularity_limits.get("right")
        if singularity_limit is not None and singularity_limit.last_state is not None:
            singularity_rate_values[tick] = (
                singularity_limit.last_state.max_approach_rate
            )

        target_pose_values[tick] = target_pose
        command_q_values[tick] = command_q
        command_dq_values[tick] = command_dq
        actual_q_values[tick] = actual_q
        actual_dq_values[tick] = actual_dq
        actual_ddq_values[tick] = actual_ddq
        command_ee_values[tick] = command_ee
        actual_ee_values[tick] = actual_ee
        command_rho_values[tick] = command_rho
        actual_rho_values[tick] = diagnostics.actual_ratio(actual_q)
        task_rho_values[tick] = task_rho
        home_error_values[tick] = home_error
        nullspace_speed_values[tick] = nullspace_speed
        velocity_cap_values[tick] = active_caps
        actuator_force_values[tick] = dynamics.right_force()
        force_ranges = dynamics.model.actuator_forcerange[
            dynamics._right_actuators
        ]
        actuator_force_cap_values[tick] = np.max(
            np.abs(force_ranges),
            axis=1,
        )
        bias_force_values[tick] = dynamics.right_bias_force()

        previous_command = command_q
        previous_actual_dq = actual_dq
        previous_phase = current_phase

    return Trace(
        time=np.arange(count, dtype=np.float64) * CONTROL_DT,
        phase=phase,
        target_pose=target_pose_values,
        command_q=command_q_values,
        command_dq=command_dq_values,
        actual_q=actual_q_values,
        actual_dq=actual_dq_values,
        actual_ddq=actual_ddq_values,
        command_ee=command_ee_values,
        actual_ee=actual_ee_values,
        command_rho=command_rho_values,
        actual_rho=actual_rho_values,
        task_rho=task_rho_values,
        geometric_home_error=home_error_values,
        geometric_nullspace_speed=nullspace_speed_values,
        nullspace_activation=activation_values,
        nullspace_task_error=task_error_values,
        nullspace_return_speed=return_speed_values,
        singularity_allowed_rate=singularity_rate_values,
        velocity_caps=velocity_cap_values,
        actuator_force=actuator_force_values,
        actuator_force_caps=actuator_force_cap_values,
        bias_force=bias_force_values,
        solve_failed=failed_values,
    )


def _path_excess(values: np.ndarray) -> float:
    path = float(np.sum(np.abs(np.diff(values))))
    net = float(abs(values[-1] - values[0]))
    return max(path - net, 0.0)


def _reversals(values: np.ndarray, threshold: float) -> int:
    velocity = np.diff(values) / CONTROL_DT
    significant = velocity[np.abs(velocity) >= threshold]
    if significant.size < 2:
        return 0
    signs = np.sign(significant)
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def _first_time(mask: np.ndarray, time: np.ndarray, start_time: float) -> float:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return np.nan
    return float(time[indices[0]] - start_time)


def compute_metrics(
    mode: Mode,
    retract_peak_speed: float,
    trace: Trace,
) -> Metrics:
    retract = trace.phase == 3
    final_hold = trace.phase == 4
    retract_indices = np.flatnonzero(retract)
    start = int(retract_indices[0])
    end = int(retract_indices[-1])
    start_before = max(start - 1, 0)
    retract_time = trace.time[retract]
    start_time = float(trace.time[start])
    command_error = np.linalg.norm(
        trace.target_pose[retract, :3] - trace.command_ee[retract, :3],
        axis=1,
    )
    actual_error = np.linalg.norm(
        trace.target_pose[retract, :3] - trace.actual_ee[retract, :3],
        axis=1,
    )
    saturation = (
        np.abs(trace.command_dq[retract])
        >= 0.98 * trace.velocity_caps[retract]
    )
    force_saturation = (
        np.abs(trace.actuator_force[retract])
        >= 0.98 * trace.actuator_force_caps[retract]
    )

    q4_start = float(trace.command_q[start_before, 3])
    bend_mask = retract & (trace.command_q[:, 3] >= q4_start + 0.01)
    bend_indices = np.flatnonzero(bend_mask)
    if bend_indices.size:
        bend_index = int(bend_indices[0])
        bend_time = float(trace.time[bend_index] - start_time)
        bend_gap = float(
            trace.target_pose[bend_index, 0] - trace.command_ee[bend_index, 0]
        )
    else:
        bend_time = np.nan
        bend_gap = np.nan

    workspace_gap = trace.target_pose[:, 0] - trace.command_ee[:, 0]
    reentry_mask = retract & (workspace_gap <= 0.0)
    reentry_time = _first_time(reentry_mask, trace.time, start_time)
    finite_activation = trace.nullspace_activation[retract]
    finite_activation = finite_activation[np.isfinite(finite_activation)]
    finite_return = trace.nullspace_return_speed[retract]
    finite_return = finite_return[np.isfinite(finite_return)]

    return Metrics(
        mode=mode.name,
        retract_peak_speed_m_s=retract_peak_speed,
        retract_duration_s=float(retract_time[-1] - retract_time[0] + CONTROL_DT),
        min_command_rho=float(np.min(trace.command_rho)),
        min_actual_rho=float(np.min(trace.actual_rho)),
        retract_command_q4_delta_rad=float(
            trace.command_q[end, 3] - trace.command_q[start_before, 3]
        ),
        retract_actual_q4_delta_rad=float(
            trace.actual_q[end, 3] - trace.actual_q[start_before, 3]
        ),
        retract_command_q4_path_excess_rad=_path_excess(
            trace.command_q[retract, 3]
        ),
        retract_actual_q4_path_excess_rad=_path_excess(
            trace.actual_q[retract, 3]
        ),
        retract_peak_abs_command_dq1_rad_s=float(
            np.max(np.abs(trace.command_dq[retract, 0]))
        ),
        retract_peak_abs_command_dq4_rad_s=float(
            np.max(np.abs(trace.command_dq[retract, 3]))
        ),
        retract_peak_abs_actual_dq1_rad_s=float(
            np.max(np.abs(trace.actual_dq[retract, 0]))
        ),
        retract_peak_abs_actual_dq4_rad_s=float(
            np.max(np.abs(trace.actual_dq[retract, 3]))
        ),
        retract_peak_abs_actual_ddq4_rad_s2=float(
            np.max(np.abs(trace.actual_ddq[retract, 3]))
        ),
        retract_q1_velocity_limit_fraction=float(np.mean(saturation[:, 0])),
        retract_q4_velocity_limit_fraction=float(np.mean(saturation[:, 3])),
        retract_any_velocity_limit_fraction=float(np.mean(np.any(saturation, axis=1))),
        retract_q1_force_limit_fraction=float(np.mean(force_saturation[:, 0])),
        retract_q4_force_limit_fraction=float(np.mean(force_saturation[:, 3])),
        retract_any_force_limit_fraction=float(
            np.mean(np.any(force_saturation, axis=1))
        ),
        retract_peak_abs_actuator_force_q1_nm=float(
            np.max(np.abs(trace.actuator_force[retract, 0]))
        ),
        retract_peak_abs_actuator_force_q4_nm=float(
            np.max(np.abs(trace.actuator_force[retract, 3]))
        ),
        retract_peak_abs_bias_force_q1_nm=float(
            np.max(np.abs(trace.bias_force[retract, 0]))
        ),
        retract_peak_abs_bias_force_q4_nm=float(
            np.max(np.abs(trace.bias_force[retract, 3]))
        ),
        retract_command_tracking_rmse_m=float(
            np.sqrt(np.mean(command_error**2))
        ),
        retract_actual_tracking_rmse_m=float(np.sqrt(np.mean(actual_error**2))),
        retract_max_command_actual_q4_error_rad=float(
            np.max(
                np.abs(
                    trace.command_q[retract, 3] - trace.actual_q[retract, 3]
                )
            )
        ),
        retract_actual_ee_z_p2p_m=float(np.ptp(trace.actual_ee[retract, 2])),
        retract_command_q4_reversals=_reversals(
            trace.command_q[retract, 3],
            threshold=0.01,
        ),
        retract_actual_q4_reversals=_reversals(
            trace.actual_q[retract, 3],
            threshold=0.01,
        ),
        q4_bend_onset_s=bend_time,
        q4_bend_onset_target_gap_m=bend_gap,
        target_reenters_command_workspace_s=reentry_time,
        geometric_home_error_initial_abs_rad=float(
            abs(trace.geometric_home_error[0])
        ),
        geometric_home_error_start_abs_rad=float(
            abs(trace.geometric_home_error[start_before])
        ),
        geometric_home_error_retract_end_abs_rad=float(
            abs(trace.geometric_home_error[end])
        ),
        geometric_home_error_final_abs_rad=float(
            np.median(np.abs(trace.geometric_home_error[final_hold][-50:]))
        ),
        min_nullspace_activation=(
            float(np.min(finite_activation)) if finite_activation.size else np.nan
        ),
        peak_requested_nullspace_return_speed_rad_s=(
            float(np.max(np.abs(finite_return))) if finite_return.size else np.nan
        ),
        solve_failure_count=int(np.count_nonzero(trace.solve_failed)),
    )


def save_trace(path: Path, trace: Trace) -> None:
    np.savez_compressed(path, **trace.__dict__)


def _phase_boundaries(trace: Trace) -> list[float]:
    changes = np.flatnonzero(trace.phase[1:] != trace.phase[:-1]) + 1
    return [float(trace.time[index]) for index in changes]


def _decorate_axes(axes: np.ndarray, trace: Trace) -> None:
    for axis in axes.flat:
        for boundary in _phase_boundaries(trace):
            axis.axvline(boundary, color="0.75", linewidth=0.7)
        axis.grid(alpha=0.2)


def plot_task_comparison(
    path: Path,
    speed: float,
    traces: dict[str, Trace],
) -> None:
    import matplotlib.pyplot as plt

    names = [
        "historical_home_0p01",
        "historical_home_0p10",
        "home_full_0p01",
        "nullspace_rate_full",
        "nullspace_direct_full",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for name in names:
        trace = traces[name]
        relative_target = trace.target_pose[:, 0] - trace.target_pose[0, 0]
        relative_command = trace.command_ee[:, 0] - trace.target_pose[0, 0]
        axes[0, 0].plot(trace.time, relative_target, "k--", alpha=0.15)
        axes[0, 0].plot(trace.time, relative_command, label=name)
        axes[0, 1].plot(trace.time, trace.command_q[:, 3], label=name)
        axes[1, 0].plot(trace.time, trace.command_dq[:, 3], label=name)
        axes[1, 1].plot(trace.time, trace.command_rho, label=name)
        axes[2, 0].plot(
            trace.time,
            np.abs(trace.geometric_home_error),
            label=name,
        )
        error = np.linalg.norm(
            trace.target_pose[:, :3] - trace.command_ee[:, :3],
            axis=1,
        )
        axes[2, 1].plot(trace.time, error, label=name)

    axes[0, 0].set_ylabel("EE x offset [m]")
    axes[0, 1].set_ylabel("command q4 [rad]")
    axes[1, 0].set_ylabel("command dq4 [rad/s]")
    axes[1, 1].set_ylabel("geometric rho")
    axes[2, 0].set_ylabel("|geometric home error| [rad]")
    axes[2, 1].set_ylabel("target-command error [m]")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 1].set_xlabel("time [s]")
    axes[0, 0].legend(fontsize=8, ncol=2)
    _decorate_axes(axes, next(iter(traces.values())))
    fig.suptitle(f"Task comparison, retract peak speed {speed:.2f} m/s")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_velocity_comparison(
    path: Path,
    speed: float,
    traces: dict[str, Trace],
) -> None:
    import matplotlib.pyplot as plt

    names = [
        "nullspace_rate_full_retract_old_cap",
        "nullspace_rate_full",
        "nullspace_rate_full_retract_high_cap",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for name in names:
        trace = traces[name]
        axes[0, 0].plot(trace.time, trace.command_q[:, 3], label=name)
        axes[0, 1].plot(trace.time, trace.command_dq[:, 3], label=name)
        ratio = np.max(
            np.abs(trace.command_dq) / trace.velocity_caps,
            axis=1,
        )
        axes[1, 0].plot(trace.time, ratio, label=name)
        error = np.linalg.norm(
            trace.target_pose[:, :3] - trace.command_ee[:, :3],
            axis=1,
        )
        axes[1, 1].plot(trace.time, error, label=name)

    axes[0, 0].set_ylabel("command q4 [rad]")
    axes[0, 1].set_ylabel("command dq4 [rad/s]")
    axes[1, 0].set_ylabel("max |dq / cap|")
    axes[1, 1].set_ylabel("target-command error [m]")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 1].set_xlabel("time [s]")
    axes[0, 0].legend(fontsize=8)
    _decorate_axes(axes, next(iter(traces.values())))
    fig.suptitle(f"Retract velocity-cap comparison, peak speed {speed:.2f} m/s")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _default_modes() -> list[Mode]:
    return [
        Mode(
            "historical_home_0p01",
            "home",
            posture_cost=0.01,
            position_cost=1.0,
            lm_damping=0.01,
            historical_timing=True,
            initial_caps="old",
            retract_caps="old",
        ),
        Mode(
            "historical_home_0p10",
            "home",
            posture_cost=0.1,
            position_cost=1.0,
            lm_damping=0.01,
            historical_timing=True,
            initial_caps="old",
            retract_caps="old",
        ),
        Mode(
            "home_full_0p01",
            "home",
            posture_cost=0.01,
            full_safety=True,
        ),
        Mode(
            "home_full_0p10",
            "home",
            posture_cost=0.1,
            full_safety=True,
        ),
        Mode(
            "nullspace_rate_task_only",
            "nullspace_rate",
        ),
        Mode(
            "nullspace_rate_full",
            "nullspace_rate",
            full_safety=True,
        ),
        Mode(
            "nullspace_direct_0p10_full",
            "nullspace_direct",
            full_safety=True,
            nullspace_cost=0.1,
        ),
        Mode(
            "nullspace_direct_1p00_full",
            "nullspace_direct",
            full_safety=True,
            nullspace_cost=1.0,
        ),
        Mode(
            "nullspace_direct_full",
            "nullspace_direct",
            full_safety=True,
        ),
        Mode(
            "nullspace_rate_full_retract_old_cap",
            "nullspace_rate",
            full_safety=True,
            retract_caps="old",
        ),
        Mode(
            "nullspace_rate_full_retract_high_cap",
            "nullspace_rate",
            full_safety=True,
            retract_caps="high",
        ),
    ]


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dev/results/posture_mode_comparison_20260724"),
    )
    parser.add_argument("--retract-speeds", default="0.05,0.1,0.2,0.4")
    parser.add_argument("--distance", type=float, default=0.25)
    parser.add_argument("--extend-peak-speed", type=float, default=0.2)
    parser.add_argument("--settle-duration", type=float, default=2.0)
    parser.add_argument("--pre-hold", type=float, default=1.0)
    parser.add_argument("--far-hold", type=float, default=1.0)
    parser.add_argument("--post-hold", type=float, default=1.5)
    parser.add_argument("--initial-nullspace-offset", type=float, default=0.0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    speeds = _parse_floats(args.retract_speeds)
    modes = _default_modes()
    initial_arm_q = _nullspace_offset_configuration(
        args.initial_nullspace_offset
    )
    metadata = {
        "modes": [asdict(mode) for mode in modes],
        "retract_peak_speeds": speeds,
        "distance": args.distance,
        "extend_peak_speed": args.extend_peak_speed,
        "settle_duration": args.settle_duration,
        "pre_hold": args.pre_hold,
        "far_hold": args.far_hold,
        "post_hold": args.post_hold,
        "initial_nullspace_offset": args.initial_nullspace_offset,
        "initial_arm_q": initial_arm_q.tolist(),
        "control_dt": CONTROL_DT,
        "current_velocity_caps": list(ARM_JOINT_VELOCITY_LIMITS_RAD_S),
        "historical_velocity_caps": _caps("old").tolist(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    metrics: list[Metrics] = []
    for speed in speeds:
        traces: dict[str, Trace] = {}
        for mode in modes:
            print(
                f"Running retract_peak={speed:.3f} mode={mode.name}",
                flush=True,
            )
            trace = simulate(
                mode,
                speed,
                initial_arm_q=initial_arm_q,
                distance=args.distance,
                extend_peak_speed=args.extend_peak_speed,
                settle_duration=args.settle_duration,
                pre_hold=args.pre_hold,
                far_hold=args.far_hold,
                post_hold=args.post_hold,
            )
            traces[mode.name] = trace
            metrics.append(compute_metrics(mode, speed, trace))
            speed_tag = f"{speed:.3f}".replace(".", "p")
            save_trace(
                output_dir / f"trace_{speed_tag}_{mode.name}.npz",
                trace,
            )

        if not args.no_plots:
            speed_tag = f"{speed:.3f}".replace(".", "p")
            plot_task_comparison(
                output_dir / f"tasks_{speed_tag}.png",
                speed,
                traces,
            )
            plot_velocity_comparison(
                output_dir / f"velocity_caps_{speed_tag}.png",
                speed,
                traces,
            )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(metrics[0])))
        writer.writeheader()
        for item in metrics:
            writer.writerow(asdict(item))
    print(f"Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
