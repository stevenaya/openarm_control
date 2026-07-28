#!/usr/bin/env python3
"""Replay a recorded intervention trajectory through MuJoCo and Mink.

The intervention dataset stores joint commands, not the original VR targets.
This experiment converts the recorded commands to full 6D end-effector poses
with FK, resamples them at 250 Hz, and feeds the same pose stream to several IK
configurations. This isolates redundancy resolution from target-path changes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import mink
import mujoco
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation, Slerp

from openarm_control import IKParams, Kinematics
from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S
from openarm_control.nullspace_posture_task import smoothstep_activation
from openarm_control.singularity import normalized_arm_jacobian

from sim_reach_braking_experiment import CONTROL_DT, _setup


DEFAULT_RUN_DIR = Path(
    "/hdd_data/rollout/"
    "pillow_0702_history_with_visual_from_flatten_hist_base_30k_0711_window_20_80k"
)
CHARACTERISTIC_LENGTH = 0.3


@dataclass(frozen=True)
class RetractSegment:
    """One detected motion toward the shoulder."""

    index: int
    start: int
    end: int
    core_start: int
    core_end: int
    duration_s: float
    reach_drop_m: float
    mean_speed_m_s: float
    peak_speed_m_s: float
    elbow_lateral_start_m: float
    elbow_lateral_end_m: float


@dataclass
class ReplayTrace:
    """Signals recorded from one posture-cost replay."""

    time: np.ndarray
    target_pose: np.ndarray
    ik_target_pose: np.ndarray
    source_q: np.ndarray
    source_elbow: np.ndarray
    command_q: np.ndarray
    command_dq: np.ndarray
    actual_q: np.ndarray
    actual_dq: np.ndarray
    command_pose: np.ndarray
    actual_pose: np.ndarray
    command_elbow: np.ndarray
    actual_elbow: np.ndarray
    rho: np.ndarray
    sigma_min: np.ndarray
    exact_null_speed: np.ndarray
    near_null_speed: np.ndarray
    exact_null_error: np.ndarray
    near_null_error: np.ndarray
    velocity_utilization: np.ndarray
    solve_failed: np.ndarray


@dataclass(frozen=True)
class ReplayMetrics:
    """Summary metrics for one segment and posture cost."""

    segment: int
    posture_cost: float
    target_mean_speed_m_s: float
    target_peak_speed_m_s: float
    min_rho: float
    min_sigma: float
    any_velocity_limit_fraction: float
    peak_velocity_utilization: float
    mean_abs_exact_null_speed_rad_s: float
    mean_abs_near_null_speed_rad_s: float
    exact_null_error_delta_rad: float
    near_null_error_delta_rad: float
    command_elbow_lateral_delta_m: float
    actual_elbow_lateral_delta_m: float
    command_elbow_source_rmse_m: float
    command_position_rmse_m: float
    actual_position_rmse_m: float
    command_orientation_rmse_rad: float
    solve_failure_count: int


class NearNullspacePostureTask(mink.Task):
    """Return toward home along the smallest nonzero singular direction."""

    def __init__(
        self,
        model: mujoco.MjModel,
        frame_task: mink.FrameTask,
        dof_indices: np.ndarray,
        home_qpos: np.ndarray,
        *,
        cost: float,
        dt: float,
        return_rate: float,
        max_speed: float,
        singularity_low: float,
        singularity_high: float,
        characteristic_length: float,
    ) -> None:
        super().__init__(cost=np.array([cost], dtype=np.float64))
        self._base_cost = float(cost)
        self._model = model
        self._frame_task = frame_task
        self._dof_indices = np.asarray(dof_indices, dtype=int).copy()
        self._home_qpos = np.asarray(home_qpos, dtype=np.float64).copy()
        self._dt = float(dt)
        self._return_rate = float(return_rate)
        self._max_speed = float(max_speed)
        self._singularity_low = float(singularity_low)
        self._singularity_high = float(singularity_high)
        self._characteristic_length = float(characteristic_length)
        self._previous_direction: np.ndarray | None = None

    def _terms(
        self,
        configuration: mink.Configuration,
    ) -> tuple[np.ndarray, np.ndarray]:
        jacobian = normalized_arm_jacobian(
            self._frame_task,
            configuration,
            self._dof_indices,
            self._characteristic_length,
        )
        _, singular_values, vt = np.linalg.svd(jacobian, full_matrices=True)
        direction = vt[-2].copy()
        if (
            self._previous_direction is not None
            and direction @ self._previous_direction < 0.0
        ):
            direction *= -1.0
        self._previous_direction = direction.copy()

        rho = float(singular_values[-1] / singular_values[0])
        activation = 1.0 - smoothstep_activation(
            rho,
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
        return_speed = float(
            np.clip(
                -self._return_rate * posture_error,
                -self._max_speed,
                self._max_speed,
            )
        )
        displacement = return_speed * self._dt
        objective_jacobian = np.zeros(
            (1, self._model.nv),
            dtype=np.float64,
        )
        objective_jacobian[0, self._dof_indices] = direction
        return (
            np.array([-displacement], dtype=np.float64),
            objective_jacobian,
        )

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        error, _ = self._terms(configuration)
        return error

    def compute_jacobian(
        self,
        configuration: mink.Configuration,
    ) -> np.ndarray:
        _, jacobian = self._terms(configuration)
        return jacobian

    def compute_qp_objective(
        self,
        configuration: mink.Configuration,
    ) -> mink.Objective:
        error, jacobian = self._terms(configuration)
        return self._assemble_qp(error, jacobian, configuration._eye_nv)


class DirectExactNullspacePostureTask(mink.Task):
    """Penalize a fraction of the exact-nullspace home error per substep."""

    def __init__(
        self,
        model: mujoco.MjModel,
        frame_task: mink.FrameTask,
        dof_indices: np.ndarray,
        home_qpos: np.ndarray,
        *,
        cost: float,
        error_gain: float,
        singularity_low: float,
        singularity_high: float,
        characteristic_length: float,
    ) -> None:
        super().__init__(cost=np.array([cost], dtype=np.float64))
        self._base_cost = float(cost)
        self._error_gain = float(error_gain)
        self._model = model
        self._frame_task = frame_task
        self._dof_indices = np.asarray(dof_indices, dtype=int).copy()
        self._home_qpos = np.asarray(home_qpos, dtype=np.float64).copy()
        self._singularity_low = float(singularity_low)
        self._singularity_high = float(singularity_high)
        self._characteristic_length = float(characteristic_length)
        self._previous_direction: np.ndarray | None = None

    def _terms(
        self,
        configuration: mink.Configuration,
    ) -> tuple[np.ndarray, np.ndarray]:
        jacobian = normalized_arm_jacobian(
            self._frame_task,
            configuration,
            self._dof_indices,
            self._characteristic_length,
        )
        _, singular_values, vt = np.linalg.svd(jacobian, full_matrices=True)
        direction = vt[-1].copy()
        if (
            self._previous_direction is not None
            and direction @ self._previous_direction < 0.0
        ):
            direction *= -1.0
        self._previous_direction = direction.copy()

        rho = float(singular_values[-1] / singular_values[0])
        activation = smoothstep_activation(
            rho,
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
        objective_jacobian = np.zeros(
            (1, self._model.nv),
            dtype=np.float64,
        )
        objective_jacobian[0, self._dof_indices] = direction
        return (
            np.array(
                [self._error_gain * posture_error],
                dtype=np.float64,
            ),
            objective_jacobian,
        )

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        error, _ = self._terms(configuration)
        return error

    def compute_jacobian(
        self,
        configuration: mink.Configuration,
    ) -> np.ndarray:
        _, jacobian = self._terms(configuration)
        return jacobian

    def compute_qp_objective(
        self,
        configuration: mink.Configuration,
    ) -> mink.Objective:
        error, jacobian = self._terms(configuration)
        return self._assemble_qp(error, jacobian, configuration._eye_nv)


def _velocity_limits(
    scale: float = 1.0,
    joint_multipliers: np.ndarray | None = None,
) -> dict[str, float]:
    if scale <= 0.0:
        raise ValueError(f"Velocity-limit scale must be positive, got {scale}.")
    multipliers = (
        np.ones(7, dtype=np.float64)
        if joint_multipliers is None
        else np.asarray(joint_multipliers, dtype=np.float64)
    )
    if multipliers.shape != (7,) or np.any(multipliers <= 0.0):
        raise ValueError(
            "Velocity-limit joint multipliers must be seven positive values."
        )
    return {
        f"openarm_{side}_joint{index + 1}": float(
            scale * multipliers[index] * cap
        )
        for side in ("left", "right")
        for index, cap in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S)
    }


def _bimanual_state(
    resolver: object,
    qpos: np.ndarray,
) -> np.ndarray:
    right, right_gripper = resolver.get_driver(qpos, "right")
    left, left_gripper = resolver.get_driver(qpos, "left")
    return np.concatenate(
        [
            np.append(right, right_gripper),
            np.append(left, left_gripper),
        ]
    ).astype(np.float32)


def _bimanual_velocity(
    resolver: object,
    qvel: np.ndarray,
) -> np.ndarray:
    right = np.append(qvel[resolver._right.arm_dof], 0.0)
    left = np.append(qvel[resolver._left.arm_dof], 0.0)
    return np.concatenate([right, left]).astype(np.float32)


class DynamicSideArm:
    """MuJoCo position-controlled plant for one arm."""

    def __init__(self, side: str, initial_q: np.ndarray) -> None:
        self.side = side
        self.setup = _setup(side)
        self.model = self.setup.model
        self.data = self.setup.data
        self.resolver = self.setup.joint_resolver
        self.resolver.set_qpos(
            self.data.qpos,
            np.append(np.asarray(initial_q, dtype=np.float64), 0.0),
            side,
        )
        mujoco.mj_forward(self.model, self.data)
        self._held_qpos = self.data.qpos.copy()
        resolved = (
            self.resolver._left if side == "left" else self.resolver._right
        )
        self._qpos = np.asarray(resolved.arm_qpos, dtype=int)
        self._dofs = np.asarray(resolved.arm_dof, dtype=int)
        self._actuators = np.array(
            [
                self.model.actuator(f"{side}_joint{index}_ctrl").id
                for index in range(1, 8)
            ],
            dtype=int,
        )
        self._shoulder_joint = self.model.joint(
            f"openarm_{side}_joint1"
        ).id
        self._elbow_joint = self.model.joint(f"openarm_{side}_joint4").id
        self.set_command(initial_q)

    def set_command(self, command_q: np.ndarray) -> None:
        target_qpos = self._held_qpos.copy()
        self.resolver.set_qpos(
            target_qpos,
            np.append(np.asarray(command_q, dtype=np.float64), 0.0),
            self.side,
        )
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            qpos_index = int(self.model.jnt_qposadr[joint_id])
            lower, upper = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = np.clip(
                target_qpos[qpos_index],
                lower,
                upper,
            )

    def settle(self, duration: float) -> None:
        for _ in range(int(round(duration / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def step(self) -> None:
        count = int(round(CONTROL_DT / self.model.opt.timestep))
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)

    def q(self) -> np.ndarray:
        return self.data.qpos[self._qpos].copy()

    def dq(self) -> np.ndarray:
        return self.data.qvel[self._dofs].copy()

    def pose(self) -> np.ndarray:
        return self.setup.read_ee_pose(self.side).astype(np.float64)

    def shoulder(self) -> np.ndarray:
        return self.data.xanchor[self._shoulder_joint].copy()

    def elbow(self) -> np.ndarray:
        return self.data.xanchor[self._elbow_joint].copy()

    def driver_qpos(self) -> np.ndarray:
        return _bimanual_state(self.resolver, self.data.qpos)

    def driver_qvel(self) -> np.ndarray:
        return _bimanual_velocity(self.resolver, self.data.qvel)


class CommandDiagnostics:
    """Geometric SVD diagnostics for one arm command."""

    def __init__(self, kinematics: Kinematics, side: str) -> None:
        solver = kinematics._ik
        assert solver is not None
        self.side = side
        self.model = solver._model
        self.resolver = kinematics.setup.joint_resolver
        self.frame_task = solver._tasks[side]
        self.dofs = solver._arm_dofs_by_side[side]
        self.home = solver._posture_task.target_q.copy()
        self.base_q = solver._config.q.copy()
        self.configuration = mink.Configuration(self.model)
        self.previous_z: np.ndarray | None = None
        self.previous_near: np.ndarray | None = None

    def evaluate(
        self,
        q: np.ndarray,
        dq: np.ndarray,
    ) -> tuple[float, float, float, float, float, float]:
        qpos = self.base_q.copy()
        self.resolver.set_qpos(
            qpos,
            np.append(np.asarray(q, dtype=np.float64), 0.0),
            self.side,
        )
        self.configuration.update(q=qpos)
        jacobian = self.configuration.get_frame_jacobian(
            self.frame_task.frame_name,
            self.frame_task.frame_type,
        )[:, self.dofs]
        normalized = jacobian.copy()
        normalized[:3] /= CHARACTERISTIC_LENGTH
        _, singular_values, vt = np.linalg.svd(normalized, full_matrices=True)
        z = vt[-1].copy()
        near = vt[-2].copy()
        if self.previous_z is not None and z @ self.previous_z < 0.0:
            z *= -1.0
        if self.previous_near is not None and near @ self.previous_near < 0.0:
            near *= -1.0
        self.previous_z = z
        self.previous_near = near

        error = np.empty(self.model.nv, dtype=np.float64)
        mujoco.mj_differentiatePos(
            self.model,
            error,
            1.0,
            self.home,
            self.configuration.q,
        )
        arm_error = error[self.dofs]
        rho = float(singular_values[-1] / singular_values[0])
        return (
            rho,
            float(singular_values[-1]),
            float(z @ dq),
            float(near @ dq),
            float(z @ arm_error),
            float(near @ arm_error),
        )


def _make_kinematics(
    side: str,
    posture_cost: float,
    *,
    nullspace_return_rate: float = 0.8,
    nullspace_max_speed: float = 0.8,
    near_nullspace_cost: float = 0.0,
    near_nullspace_return_rate: float = 0.8,
    near_nullspace_max_speed: float = 0.8,
    direct_nullspace_cost: float = 0.0,
    direct_nullspace_error_gain: float = 0.01,
    velocity_limit_scale: float = 1.0,
    velocity_limit_joint_multipliers: np.ndarray | None = None,
) -> Kinematics:
    kinematics = Kinematics(
        _setup(side),
        IKParams(
            position_cost=10.0,
            orientation_cost=1.0,
            lm_damping=0.02,
            damping=0.1,
            posture_cost=posture_cost,
            dt=CONTROL_DT,
            max_iters=10,
            velocity_limits=_velocity_limits(
                velocity_limit_scale,
                velocity_limit_joint_multipliers,
            ),
            joint_limit_recovery_velocity_scale=1.0,
            nullspace_cost=10.0,
            nullspace_return_rate=nullspace_return_rate,
            nullspace_max_speed=nullspace_max_speed,
            nullspace_singularity_low=0.02,
            nullspace_singularity_high=0.05,
            nullspace_characteristic_length=CHARACTERISTIC_LENGTH,
            elbow_soft_limit_cost=0.0,
            elbow_braking_guard_angle=0.08,
            elbow_braking_profile="distance",
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
    if near_nullspace_cost > 0.0:
        solver = kinematics._ik
        assert solver is not None
        solver._nullspace_tasks[f"{side}_near"] = NearNullspacePostureTask(
            model=solver._model,
            frame_task=solver._tasks[side],
            dof_indices=solver._arm_dofs_by_side[side],
            home_qpos=solver._posture_task.target_q,
            cost=near_nullspace_cost,
            dt=solver._substep_dt,
            return_rate=near_nullspace_return_rate,
            max_speed=near_nullspace_max_speed,
            singularity_low=0.02,
            singularity_high=0.08,
            characteristic_length=CHARACTERISTIC_LENGTH,
        )
    if direct_nullspace_cost > 0.0:
        solver = kinematics._ik
        assert solver is not None
        solver._nullspace_tasks[side] = DirectExactNullspacePostureTask(
            model=solver._model,
            frame_task=solver._tasks[side],
            dof_indices=solver._arm_dofs_by_side[side],
            home_qpos=solver._posture_task.target_q,
            cost=direct_nullspace_cost,
            error_gain=direct_nullspace_error_gain,
            singularity_low=0.02,
            singularity_high=0.05,
            characteristic_length=CHARACTERISTIC_LENGTH,
        )
    return kinematics


def _load_recorded_commands(
    run_dir: Path,
    episode: int,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    path = (
        run_dir
        / "dataset"
        / "episodes"
        / str(episode)
        / "action"
        / "arms"
        / side
        / "state.parquet"
    )
    table = pq.read_table(path, columns=["timestamp", "qpos"])
    timestamp = np.asarray(
        table["timestamp"].cast("int64").to_numpy(),
        dtype=np.int64,
    )
    qpos = np.asarray(table["qpos"].to_pylist(), dtype=np.float64)
    keep = np.concatenate([[True], np.diff(timestamp) > 0])
    return timestamp[keep], qpos[keep, :7]


def _source_geometry(
    side: str,
    timestamp_ns: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    setup = _setup(side)
    model = setup.model
    data = setup.data
    resolver = setup.joint_resolver
    shoulder_joint = model.joint(f"openarm_{side}_joint1").id
    elbow_joint = model.joint(f"openarm_{side}_joint4").id
    pose = np.empty((q.shape[0], 7), dtype=np.float64)
    shoulder = np.empty((q.shape[0], 3), dtype=np.float64)
    elbow = np.empty((q.shape[0], 3), dtype=np.float64)
    for index, arm_q in enumerate(q):
        resolver.set_qpos(data.qpos, np.append(arm_q, 0.0), side)
        mujoco.mj_forward(model, data)
        pose[index] = setup.read_ee_pose(side)
        shoulder[index] = data.xanchor[shoulder_joint]
        elbow[index] = data.xanchor[elbow_joint]

    sample_time = (timestamp_ns - timestamp_ns[0]).astype(np.float64) * 1e-9
    grid = np.arange(0.0, sample_time[-1], CONTROL_DT)
    resampled_pose = np.empty((grid.size, 7), dtype=np.float64)
    resampled_q = np.empty((grid.size, 7), dtype=np.float64)
    resampled_shoulder = np.empty((grid.size, 3), dtype=np.float64)
    resampled_elbow = np.empty((grid.size, 3), dtype=np.float64)
    for column in range(3):
        resampled_pose[:, column] = np.interp(
            grid,
            sample_time,
            pose[:, column],
        )
        resampled_shoulder[:, column] = np.interp(
            grid,
            sample_time,
            shoulder[:, column],
        )
        resampled_elbow[:, column] = np.interp(
            grid,
            sample_time,
            elbow[:, column],
        )
    for column in range(7):
        resampled_q[:, column] = np.interp(
            grid,
            sample_time,
            q[:, column],
        )

    source_quaternion = pose[:, [4, 5, 6, 3]]
    rotations = Rotation.from_quat(source_quaternion)
    interpolated = Slerp(sample_time, rotations)(grid)
    resampled_pose[:, 3:] = interpolated.as_quat()[:, [3, 0, 1, 2]]
    return (
        grid,
        resampled_pose,
        resampled_q,
        resampled_shoulder,
        resampled_elbow,
    )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _merge_runs(
    runs: list[tuple[int, int]],
    max_gap_ticks: int,
) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end <= max_gap_ticks:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def detect_retract_segments(
    time: np.ndarray,
    pose: np.ndarray,
    shoulder: np.ndarray,
    elbow: np.ndarray,
) -> list[RetractSegment]:
    """Detect sustained motion that decreases shoulder-to-EEF reach."""
    position = pose[:, :3]
    speed = np.linalg.norm(np.gradient(position, CONTROL_DT, axis=0), axis=1)
    reach = np.linalg.norm(position - shoulder, axis=1)
    reach_rate = np.gradient(reach, CONTROL_DT)
    mask = (reach_rate < -0.03) & (speed > 0.05)
    runs = _merge_runs(_runs(mask), int(round(0.25 / CONTROL_DT)))

    segments: list[RetractSegment] = []
    context = int(round(0.4 / CONTROL_DT))
    for core_start, core_end in runs:
        duration = (core_end - core_start) * CONTROL_DT
        reach_drop = float(reach[core_start] - reach[core_end - 1])
        if duration < 0.25 or reach_drop < 0.06:
            continue
        start = max(0, core_start - context)
        end = min(time.size, core_end + context)
        lateral = elbow[:, 1] - shoulder[:, 1]
        segments.append(
            RetractSegment(
                index=len(segments),
                start=start,
                end=end,
                core_start=core_start,
                core_end=core_end,
                duration_s=duration,
                reach_drop_m=reach_drop,
                mean_speed_m_s=float(np.mean(speed[core_start:core_end])),
                peak_speed_m_s=float(np.max(speed[core_start:core_end])),
                elbow_lateral_start_m=float(lateral[core_start]),
                elbow_lateral_end_m=float(lateral[core_end - 1]),
            )
        )
    return segments


def simulate_replay(
    side: str,
    posture_cost: float,
    target_pose: np.ndarray,
    source_q: np.ndarray,
    source_elbow: np.ndarray,
    *,
    settle_duration: float,
    nullspace_return_rate: float = 0.8,
    nullspace_max_speed: float = 0.8,
    near_nullspace_cost: float = 0.0,
    near_nullspace_return_rate: float = 0.8,
    near_nullspace_max_speed: float = 0.8,
    direct_nullspace_cost: float = 0.0,
    direct_nullspace_error_gain: float = 0.01,
    velocity_limit_scale: float = 1.0,
    velocity_limit_joint_multipliers: np.ndarray | None = None,
    target_governor: (
        Callable[[Kinematics, str, np.ndarray], np.ndarray] | None
    ) = None,
) -> ReplayTrace:
    """Replay one target window through Mink and a dynamic MuJoCo plant."""
    initial_q = source_q[0]
    kinematics = _make_kinematics(
        side,
        posture_cost,
        nullspace_return_rate=nullspace_return_rate,
        nullspace_max_speed=nullspace_max_speed,
        near_nullspace_cost=near_nullspace_cost,
        near_nullspace_return_rate=near_nullspace_return_rate,
        near_nullspace_max_speed=near_nullspace_max_speed,
        direct_nullspace_cost=direct_nullspace_cost,
        direct_nullspace_error_gain=direct_nullspace_error_gain,
        velocity_limit_scale=velocity_limit_scale,
        velocity_limit_joint_multipliers=(
            velocity_limit_joint_multipliers
        ),
    )
    dynamics = DynamicSideArm(side, initial_q)
    dynamics.settle(settle_duration)
    kinematics.sync(dynamics.driver_qpos())
    diagnostics = CommandDiagnostics(kinematics, side)

    count = target_pose.shape[0]
    ik_target_pose = np.empty((count, 7), dtype=np.float64)
    command_q = np.empty((count, 7), dtype=np.float64)
    command_dq = np.empty((count, 7), dtype=np.float64)
    actual_q = np.empty((count, 7), dtype=np.float64)
    actual_dq = np.empty((count, 7), dtype=np.float64)
    command_pose = np.empty((count, 7), dtype=np.float64)
    actual_pose = np.empty((count, 7), dtype=np.float64)
    command_elbow = np.empty((count, 3), dtype=np.float64)
    actual_elbow = np.empty((count, 3), dtype=np.float64)
    rho = np.empty(count, dtype=np.float64)
    sigma_min = np.empty(count, dtype=np.float64)
    exact_null_speed = np.empty(count, dtype=np.float64)
    near_null_speed = np.empty(count, dtype=np.float64)
    exact_null_error = np.empty(count, dtype=np.float64)
    near_null_error = np.empty(count, dtype=np.float64)
    velocity_utilization = np.empty((count, 7), dtype=np.float64)
    solve_failed = np.zeros(count, dtype=bool)

    previous_command = dynamics.q()
    joint_multipliers = (
        np.ones(7, dtype=np.float64)
        if velocity_limit_joint_multipliers is None
        else np.asarray(
            velocity_limit_joint_multipliers,
            dtype=np.float64,
        )
    )
    caps = (
        np.asarray(ARM_JOINT_VELOCITY_LIMITS_RAD_S, dtype=np.float64)
        * velocity_limit_scale
        * joint_multipliers
    )
    for tick in range(count):
        kinematics.update_measured_state(
            dynamics.driver_qpos(),
            dynamics.driver_qvel(),
        )
        governed_target = (
            target_pose[tick]
            if target_governor is None
            else target_governor(kinematics, side, target_pose[tick])
        )
        ik_target_pose[tick] = governed_target
        kinematics.set_target(side, governed_target)
        result = kinematics.solve()
        if result is None:
            current_command = previous_command.copy()
            solve_failed[tick] = True
        else:
            offset = 8 if side == "left" else 0
            current_command = result[offset : offset + 7].astype(np.float64)
        current_dq = (current_command - previous_command) / CONTROL_DT

        dynamics.set_command(current_command)
        dynamics.step()
        current_actual_q = dynamics.q()
        current_actual_dq = dynamics.dq()
        current_command_pose = kinematics.fk(
            side,
            np.append(current_command, 0.0),
        ).astype(np.float64)
        values = diagnostics.evaluate(current_command, current_dq)

        command_q[tick] = current_command
        command_dq[tick] = current_dq
        actual_q[tick] = current_actual_q
        actual_dq[tick] = current_actual_dq
        command_pose[tick] = current_command_pose
        actual_pose[tick] = dynamics.pose()
        command_elbow[tick] = _elbow_at_q(
            kinematics,
            side,
            current_command,
        )
        actual_elbow[tick] = dynamics.elbow()
        (
            rho[tick],
            sigma_min[tick],
            exact_null_speed[tick],
            near_null_speed[tick],
            exact_null_error[tick],
            near_null_error[tick],
        ) = values
        velocity_utilization[tick] = np.abs(current_dq) / caps
        previous_command = current_command

    return ReplayTrace(
        time=np.arange(count, dtype=np.float64) * CONTROL_DT,
        target_pose=target_pose,
        ik_target_pose=ik_target_pose,
        source_q=source_q,
        source_elbow=source_elbow,
        command_q=command_q,
        command_dq=command_dq,
        actual_q=actual_q,
        actual_dq=actual_dq,
        command_pose=command_pose,
        actual_pose=actual_pose,
        command_elbow=command_elbow,
        actual_elbow=actual_elbow,
        rho=rho,
        sigma_min=sigma_min,
        exact_null_speed=exact_null_speed,
        near_null_speed=near_null_speed,
        exact_null_error=exact_null_error,
        near_null_error=near_null_error,
        velocity_utilization=velocity_utilization,
        solve_failed=solve_failed,
    )


def _elbow_at_q(
    kinematics: Kinematics,
    side: str,
    q: np.ndarray,
) -> np.ndarray:
    setup = kinematics.setup
    setup.joint_resolver.set_qpos(
        setup.data.qpos,
        np.append(np.asarray(q, dtype=np.float64), 0.0),
        side,
    )
    mujoco.mj_forward(setup.model, setup.data)
    joint_id = setup.model.joint(f"openarm_{side}_joint4").id
    return setup.data.xanchor[joint_id].copy()


def _orientation_error(
    target: np.ndarray,
    actual: np.ndarray,
) -> np.ndarray:
    target_rotation = Rotation.from_quat(target[:, [4, 5, 6, 3]])
    actual_rotation = Rotation.from_quat(actual[:, [4, 5, 6, 3]])
    return (target_rotation.inv() * actual_rotation).magnitude()


def compute_metrics(
    segment: int,
    posture_cost: float,
    trace: ReplayTrace,
) -> ReplayMetrics:
    target_speed = np.linalg.norm(
        np.gradient(trace.target_pose[:, :3], CONTROL_DT, axis=0),
        axis=1,
    )
    command_position_error = np.linalg.norm(
        trace.command_pose[:, :3] - trace.target_pose[:, :3],
        axis=1,
    )
    actual_position_error = np.linalg.norm(
        trace.actual_pose[:, :3] - trace.target_pose[:, :3],
        axis=1,
    )
    orientation_error = _orientation_error(
        trace.target_pose,
        trace.command_pose,
    )
    lateral_axis = 1
    saturation = np.any(trace.velocity_utilization >= 0.98, axis=1)
    return ReplayMetrics(
        segment=segment,
        posture_cost=posture_cost,
        target_mean_speed_m_s=float(np.mean(target_speed)),
        target_peak_speed_m_s=float(np.max(target_speed)),
        min_rho=float(np.min(trace.rho)),
        min_sigma=float(np.min(trace.sigma_min)),
        any_velocity_limit_fraction=float(np.mean(saturation)),
        peak_velocity_utilization=float(np.max(trace.velocity_utilization)),
        mean_abs_exact_null_speed_rad_s=float(
            np.mean(np.abs(trace.exact_null_speed))
        ),
        mean_abs_near_null_speed_rad_s=float(
            np.mean(np.abs(trace.near_null_speed))
        ),
        exact_null_error_delta_rad=float(
            trace.exact_null_error[-1] - trace.exact_null_error[0]
        ),
        near_null_error_delta_rad=float(
            trace.near_null_error[-1] - trace.near_null_error[0]
        ),
        command_elbow_lateral_delta_m=float(
            trace.command_elbow[-1, lateral_axis]
            - trace.command_elbow[0, lateral_axis]
        ),
        actual_elbow_lateral_delta_m=float(
            trace.actual_elbow[-1, lateral_axis]
            - trace.actual_elbow[0, lateral_axis]
        ),
        command_elbow_source_rmse_m=float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (trace.command_elbow - trace.source_elbow) ** 2,
                        axis=1,
                    )
                )
            )
        ),
        command_position_rmse_m=float(
            np.sqrt(np.mean(command_position_error**2))
        ),
        actual_position_rmse_m=float(
            np.sqrt(np.mean(actual_position_error**2))
        ),
        command_orientation_rmse_rad=float(
            np.sqrt(np.mean(orientation_error**2))
        ),
        solve_failure_count=int(np.count_nonzero(trace.solve_failed)),
    )


def _cost_label(cost: float) -> str:
    return f"{cost:.2f}".replace(".", "p")


def _save_trace(path: Path, trace: ReplayTrace) -> None:
    np.savez_compressed(path, **trace.__dict__)


def _plot_segment(
    path: Path,
    traces: dict[float, ReplayTrace],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is unavailable; skipping plot generation.")
        return

    fig, axes = plt.subplots(4, 2, figsize=(15, 13), sharex=True)
    for cost, trace in traces.items():
        label = f"posture={cost:g}"
        axes[0, 0].plot(trace.time, trace.rho, label=label)
        axes[0, 1].plot(
            trace.time,
            np.max(trace.velocity_utilization, axis=1),
            label=label,
        )
        axes[1, 0].plot(trace.time, trace.exact_null_speed, label=label)
        axes[1, 1].plot(trace.time, trace.near_null_speed, label=label)
        axes[2, 0].plot(trace.time, trace.exact_null_error, label=label)
        axes[2, 1].plot(trace.time, trace.near_null_error, label=label)
        axes[3, 0].plot(
            trace.time,
            trace.command_elbow[:, 1],
            label=label,
        )
        position_error = np.linalg.norm(
            trace.command_pose[:, :3] - trace.target_pose[:, :3],
            axis=1,
        )
        axes[3, 1].plot(trace.time, position_error, label=label)

    axes[0, 0].set_ylabel("rho")
    axes[0, 1].set_ylabel("max |dq / cap|")
    axes[1, 0].set_ylabel("z.T dq [rad/s]")
    axes[1, 1].set_ylabel("v_near.T dq [rad/s]")
    axes[2, 0].set_ylabel("z posture error [rad]")
    axes[2, 1].set_ylabel("v_near posture error [rad]")
    axes[3, 0].set_ylabel("command elbow world y [m]")
    axes[3, 1].set_ylabel("command position error [m]")
    axes[3, 0].set_xlabel("time [s]")
    axes[3, 1].set_xlabel("time [s]")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_segments(path: Path, segments: list[RetractSegment]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(segments[0])))
        writer.writeheader()
        writer.writerows(asdict(segment) for segment in segments)


def _write_metrics(path: Path, metrics: list[ReplayMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(metrics[0])))
        writer.writeheader()
        writer.writerows(asdict(metric) for metric in metrics)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--episode", type=int, default=202)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument(
        "--segments",
        type=int,
        nargs="*",
        default=[],
        help="Detected segment indices to replay; omit to scan only.",
    )
    parser.add_argument(
        "--posture-costs",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.03, 0.1],
    )
    parser.add_argument("--settle-duration", type=float, default=0.5)
    parser.add_argument("--nullspace-return-rate", type=float, default=0.8)
    parser.add_argument("--nullspace-max-speed", type=float, default=0.8)
    parser.add_argument("--near-nullspace-cost", type=float, default=0.0)
    parser.add_argument("--near-nullspace-return-rate", type=float, default=0.8)
    parser.add_argument("--near-nullspace-max-speed", type=float, default=0.8)
    parser.add_argument("--direct-nullspace-cost", type=float, default=0.0)
    parser.add_argument(
        "--direct-nullspace-error-gain",
        type=float,
        default=0.01,
    )
    parser.add_argument("--velocity-limit-scale", type=float, default=1.0)
    parser.add_argument(
        "--velocity-limit-joint-multipliers",
        type=float,
        nargs=7,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dev/results/intervention_posture_replay_ep202"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp, raw_q = _load_recorded_commands(
        args.run_dir,
        args.episode,
        args.side,
    )
    time, target_pose, source_q, shoulder, source_elbow = _source_geometry(
        args.side,
        timestamp,
        raw_q,
    )
    segments = detect_retract_segments(
        time,
        target_pose,
        shoulder,
        source_elbow,
    )
    if not segments:
        raise RuntimeError("No retract segments were detected.")
    _write_segments(args.output_dir / "segments.csv", segments)
    print("Detected retract segments:")
    for segment in segments:
        print(
            f"  {segment.index:2d}: "
            f"t={time[segment.core_start]:6.2f}-"
            f"{time[segment.core_end - 1]:6.2f}s "
            f"drop={segment.reach_drop_m:.3f}m "
            f"speed={segment.mean_speed_m_s:.3f}/"
            f"{segment.peak_speed_m_s:.3f}m/s "
            f"elbow_y_rel={segment.elbow_lateral_start_m:+.3f}->"
            f"{segment.elbow_lateral_end_m:+.3f}m"
        )

    metadata = {
        "run_dir": str(args.run_dir),
        "episode": args.episode,
        "side": args.side,
        "control_dt": CONTROL_DT,
        "posture_costs": args.posture_costs,
        "nullspace_return_rate": args.nullspace_return_rate,
        "nullspace_max_speed": args.nullspace_max_speed,
        "near_nullspace_cost": args.near_nullspace_cost,
        "near_nullspace_return_rate": args.near_nullspace_return_rate,
        "near_nullspace_max_speed": args.near_nullspace_max_speed,
        "direct_nullspace_cost": args.direct_nullspace_cost,
        "direct_nullspace_error_gain": args.direct_nullspace_error_gain,
        "velocity_limit_scale": args.velocity_limit_scale,
        "velocity_limit_joint_multipliers": (
            args.velocity_limit_joint_multipliers
        ),
        "segments": args.segments,
        "source_limitation": (
            "Targets are FK of recorded IK commands because raw VR poses were "
            "not stored in the intervention episode."
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    if not args.segments:
        return

    selected = {segment.index: segment for segment in segments}
    metrics: list[ReplayMetrics] = []
    for segment_index in args.segments:
        segment = selected[segment_index]
        sl = slice(segment.start, segment.end)
        traces: dict[float, ReplayTrace] = {}
        for posture_cost in args.posture_costs:
            print(
                f"Simulating segment {segment_index}, "
                f"posture_cost={posture_cost:g}..."
            )
            trace = simulate_replay(
                args.side,
                posture_cost,
                target_pose[sl],
                source_q[sl],
                source_elbow[sl],
                settle_duration=args.settle_duration,
                nullspace_return_rate=args.nullspace_return_rate,
                nullspace_max_speed=args.nullspace_max_speed,
                near_nullspace_cost=args.near_nullspace_cost,
                near_nullspace_return_rate=(
                    args.near_nullspace_return_rate
                ),
                near_nullspace_max_speed=args.near_nullspace_max_speed,
                direct_nullspace_cost=args.direct_nullspace_cost,
                direct_nullspace_error_gain=(
                    args.direct_nullspace_error_gain
                ),
                velocity_limit_scale=args.velocity_limit_scale,
                velocity_limit_joint_multipliers=(
                    args.velocity_limit_joint_multipliers
                ),
            )
            traces[posture_cost] = trace
            metrics.append(
                compute_metrics(segment_index, posture_cost, trace)
            )
            _save_trace(
                args.output_dir
                / (
                    f"trace_segment_{segment_index:02d}_"
                    f"posture_{_cost_label(posture_cost)}.npz"
                ),
                trace,
            )
        _plot_segment(
            args.output_dir / f"segment_{segment_index:02d}.png",
            traces,
        )
    _write_metrics(args.output_dir / "summary.csv", metrics)


if __name__ == "__main__":
    main()
