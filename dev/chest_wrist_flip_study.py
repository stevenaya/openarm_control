"""Reproduce and diagnose the chest-region wrist-flip excursion.

Run from the openarm_control repository:

    uv run --with pyarrow --with scipy --with matplotlib \
        python dev/chest_wrist_flip_study.py

The script does not modify production code. It extracts the episode-39 event,
replays a fixed-position local-Z wrist rotation through the current IK stack,
and compares the mechanisms that can redirect motion into proximal joints.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import MethodType
from typing import Any

import matplotlib.pyplot as plt
import mink
import mujoco
import numpy as np
import openarm_mujoco.v2 as openarm_mujoco
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from openarm_control import ArmSetup, IKParams, Kinematics
from openarm_control.bounded_frame_task import BoundedFrameTask

CONTROL_DT = 1.0 / 250.0
EPISODE = Path("/hdd_data/rollout/pillow_0702_tune_llm/dataset/episodes/39")
EVENT_START_S = 66.5
EVENT_END_S = 71.0
FLIP_START_Q = np.array(
    [1.02, 0.74, -1.50, 1.34, -1.57, 0.79, -1.09],
    dtype=np.float64,
)
DEFAULT_CAPS = np.array([2.0, 2.0, 3.14, 3.14, 6.3, 6.3, 6.3])


@dataclass(frozen=True)
class Profile:
    name: str
    initial_right: tuple[float, ...] = tuple(float(value) for value in FLIP_START_Q)
    angular_speed: float = 8.0
    angle: float = 4.5
    local_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)
    pre_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pre_translation_speed: float = 1.0
    braking: bool = True
    frame_position_limit: float = 0.003
    frame_error_latch_multiplier: float = 2.0
    forced_activation: float | None = 1.0
    orientation_error_limit: float = 0.0
    independent_orientation_limit: bool = False
    angular_schedule: bool = False
    orientation_latch: bool = True
    angular_speed_slow: float = 2.0
    angular_speed_fast: float = 3.0
    position_cost: float = 10.0
    orientation_cost: float = 1.0
    nullspace_cost: float = 12.0
    singularity_rate: float = 0.25
    kinetic_energy_cost: float = 3e-5
    caps: tuple[float, ...] | None = tuple(DEFAULT_CAPS)
    dynamic_plant: bool = False
    gravity: bool = True


def make_setup(mode: str) -> ArmSetup:
    return ArmSetup.from_args(
        xml=openarm_mujoco.openarm_cell_xml(),
        mode=mode,
        frame_right="right_ee_control_point",
        frame_type_right="site",
        frame_left="left_ee_control_point",
        frame_type_left="site",
    )


def velocity_mapping(caps: tuple[float, ...] | None) -> dict[str, float] | None:
    if caps is None:
        return None
    return {
        f"openarm_{side}_joint{index + 1}": value
        for side in ("right", "left")
        for index, value in enumerate(caps)
    }


def driver_state(setup: ArmSetup) -> np.ndarray:
    values: list[np.ndarray] = []
    for side in ("right", "left"):
        joints, gripper = setup.joint_resolver.get_driver(setup.data.qpos, side)
        values.append(np.append(joints, gripper))
    return np.concatenate(values).astype(np.float64)


def pack_pose(position: np.ndarray, rotation: Rotation) -> np.ndarray:
    quat = rotation.as_quat()
    return np.array(
        [
            position[0],
            position[1],
            position[2],
            quat[3],
            quat[0],
            quat[1],
            quat[2],
        ],
        dtype=np.float64,
    )


def pose_from_configuration(kinematics: Kinematics, side: str) -> np.ndarray:
    solver = kinematics._ik
    assert solver is not None
    kinematics.setup.data.qpos[:] = solver._config.q
    mujoco.mj_forward(kinematics.setup.model, kinematics.setup.data)
    return kinematics.setup.read_ee_pose(side).astype(np.float64)


def joint_ranges(setup: ArmSetup, side: str) -> np.ndarray:
    qpos_indices = setup.joint_resolver.arm_qpos_indices(side)
    joint_ids = [
        int(np.flatnonzero(setup.model.jnt_qposadr == index)[0])
        for index in qpos_indices
    ]
    return setup.model.jnt_range[joint_ids].copy()


class DynamicPlant:
    """Simple MuJoCo position-actuator plant used for representative A/B runs."""

    def __init__(self, initial_right: np.ndarray, *, gravity: bool) -> None:
        self.setup = make_setup("bimanual")
        self.model = self.setup.model
        self.data = self.setup.data
        if not gravity:
            self.model.opt.gravity[:] = 0.0
        self.resolver = self.setup.joint_resolver
        self.qpos_indices = {
            side: self.resolver.arm_qpos_indices(side) for side in ("right", "left")
        }
        self.dof_indices = {
            side: self.resolver.arm_dof_indices(side) for side in ("right", "left")
        }
        self.resolver.set_qpos(
            self.data.qpos,
            np.append(initial_right, 0.0),
            "right",
        )
        mujoco.mj_forward(self.model, self.data)
        self.command = {
            "right": initial_right.copy(),
            "left": self.data.qpos[self.qpos_indices["left"]].copy(),
        }
        self.set_command(initial_right)

    def set_command(self, right: np.ndarray) -> None:
        self.command["right"] = np.asarray(right, dtype=np.float64).copy()
        target_qpos = self.data.qpos.copy()
        for side in ("right", "left"):
            self.resolver.set_qpos(
                target_qpos,
                np.append(self.command[side], 0.0),
                side,
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

    def step(self) -> None:
        count = max(1, round(CONTROL_DT / self.model.opt.timestep))
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)

    def q(self, side: str) -> np.ndarray:
        return self.data.qpos[self.qpos_indices[side]].copy()

    def dq(self, side: str) -> np.ndarray:
        return self.data.qvel[self.dof_indices[side]].copy()

    def pose(self, side: str) -> np.ndarray:
        return self.setup.read_ee_pose(side).astype(np.float64)

    def qpos16(self) -> np.ndarray:
        return driver_state(self.setup)

    def qvel16(self) -> np.ndarray:
        return np.concatenate(
            [
                np.append(self.dq("right"), 0.0),
                np.append(self.dq("left"), 0.0),
            ]
        )


def make_kinematics(profile: Profile) -> Kinematics:
    setup = make_setup("right")
    params = IKParams(
        position_cost=profile.position_cost,
        orientation_cost=profile.orientation_cost,
        lm_damping=0.02,
        damping=0.1,
        dt=CONTROL_DT,
        max_iters=5,
        velocity_limits=velocity_mapping(profile.caps),
        frame_position_error_limit=profile.frame_position_limit,
        frame_error_latch_multiplier=profile.frame_error_latch_multiplier,
        joint_braking=profile.braking,
        joint_braking_distance=0.2,
        nullspace_cost=profile.nullspace_cost,
        nullspace_return_rate=1.6,
        singularity_max_approach_rate=profile.singularity_rate,
        kinetic_energy_cost=profile.kinetic_energy_cost,
    )
    return Kinematics(setup, params)


def simulate(profile: Profile) -> dict[str, np.ndarray]:
    kin = make_kinematics(profile)
    initial_right = np.asarray(profile.initial_right, dtype=np.float64)
    if initial_right.shape != (7,):
        raise ValueError("initial_right must contain seven arm joint positions.")
    state = driver_state(kin.setup)
    state[:7] = initial_right
    state[7] = 0.0
    plant = (
        DynamicPlant(initial_right, gravity=profile.gravity)
        if profile.dynamic_plant
        else None
    )
    if plant is not None:
        state = plant.qpos16()
    kin.sync(state)

    initial_pose = pose_from_configuration(kin, "right")
    initial_rotation = Rotation.from_quat(initial_pose[[4, 5, 6, 3]])
    axis = np.asarray(profile.local_axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    pre_translation = np.asarray(profile.pre_translation, dtype=np.float64)
    translation_distance = float(np.linalg.norm(pre_translation))
    translation_duration = (
        translation_distance / profile.pre_translation_speed
        if translation_distance > 0.0
        else 0.0
    )
    pre_hold = 0.2
    flip_start = pre_hold + translation_duration
    ramp_duration = profile.angle / profile.angular_speed
    post_hold = 1.5
    count = math.ceil((flip_start + ramp_duration + post_hold) / CONTROL_DT)

    arrays = {
        "time": np.arange(count, dtype=np.float64) * CONTROL_DT,
        "target_pose": np.empty((count, 7)),
        "command_pose": np.empty((count, 7)),
        "actual_pose": np.empty((count, 7)),
        "command_q": np.empty((count, 7)),
        "actual_q": np.empty((count, 7)),
        "actual_dq": np.empty((count, 7)),
        "activation": np.zeros(count),
        "command_orientation_error": np.zeros(count),
        "orientation_error": np.zeros(count),
        "singularity_ratio": np.full(count, np.nan),
        "failed": np.zeros(count, dtype=bool),
    }

    solver = kin._ik
    assert solver is not None
    task = solver._tasks["right"]
    if isinstance(task, BoundedFrameTask):
        task.orientation_error_limit = profile.orientation_error_limit
        if profile.independent_orientation_limit:

            def compute_independently_bounded_qp(
                self: BoundedFrameTask,
                configuration: mink.Configuration,
            ) -> mink.Objective:
                full_error = self.compute_full_error(configuration)
                limited_error = self.compute_limited_error(configuration)
                error = full_error.copy()
                error[:3] += self.limit_activation * (
                    limited_error[:3] - full_error[:3]
                )
                error[3:] = limited_error[3:]
                return self._assemble_qp(
                    error,
                    self.compute_jacobian(configuration),
                    configuration._eye_nv,
                )

            task.compute_qp_objective = MethodType(
                compute_independently_bounded_qp,
                task,
            )

    command_q = state[:7].copy()
    previous_command_q = command_q.copy()
    previous_target_rotation: Rotation | None = None
    for tick, elapsed in enumerate(arrays["time"]):
        translation_fraction = (
            np.clip((elapsed - pre_hold) / translation_duration, 0.0, 1.0)
            if translation_duration > 0.0
            else 1.0
        )
        target_position = initial_pose[:3] + translation_fraction * pre_translation
        theta = np.clip(
            (elapsed - flip_start) * profile.angular_speed,
            0.0,
            profile.angle,
        )
        target_rotation = initial_rotation * Rotation.from_rotvec(axis * theta)
        target = pack_pose(target_position, target_rotation)
        previous_activation = (
            task.limit_activation if isinstance(task, BoundedFrameTask) else 0.0
        )
        kin.set_target("right", target)
        if isinstance(task, BoundedFrameTask):
            if (
                profile.angular_schedule
                and profile.orientation_error_limit > 0.0
                and previous_target_rotation is not None
            ):
                angular_speed = (
                    previous_target_rotation.inv() * target_rotation
                ).magnitude() / CONTROL_DT
                u = np.clip(
                    (angular_speed - profile.angular_speed_slow)
                    / (profile.angular_speed_fast - profile.angular_speed_slow),
                    0.0,
                    1.0,
                )
                angular_activation = float(u * u * (3.0 - 2.0 * u))
                orientation_error = np.linalg.norm(
                    task.compute_full_error(solver._config)[3:]
                )
                if (
                    profile.orientation_latch
                    and previous_activation > 0.0
                    and orientation_error
                    > (
                        profile.frame_error_latch_multiplier
                        * profile.orientation_error_limit
                    )
                ):
                    angular_activation = max(
                        angular_activation,
                        previous_activation,
                    )
                task.set_limit_activation(
                    max(task.limit_activation, angular_activation)
                )
            if profile.forced_activation is not None:
                task.set_limit_activation(profile.forced_activation)
        previous_target_rotation = target_rotation

        if plant is None:
            measured_qpos = state.copy()
            measured_qpos[:7] = command_q
            measured_qvel = np.zeros(16)
            measured_qvel[:7] = (command_q - previous_command_q) / CONTROL_DT
        else:
            measured_qpos = plant.qpos16()
            measured_qvel = plant.qvel16()
        kin.update_measured_state(measured_qpos, measured_qvel)
        previous_command_q = command_q.copy()

        result = kin.solve()
        if result is None:
            arrays["failed"][tick] = True
        else:
            command_q = result[:7].astype(np.float64)
        command_pose = pose_from_configuration(kin, "right")

        if plant is None:
            actual_q = command_q.copy()
            actual_dq = (actual_q - previous_command_q) / CONTROL_DT
            actual_pose = command_pose.copy()
        else:
            plant.set_command(command_q)
            plant.step()
            actual_q = plant.q("right")
            actual_dq = plant.dq("right")
            actual_pose = plant.pose("right")

        command_rotation = Rotation.from_quat(command_pose[[4, 5, 6, 3]])
        actual_rotation = Rotation.from_quat(actual_pose[[4, 5, 6, 3]])
        arrays["target_pose"][tick] = target
        arrays["command_pose"][tick] = command_pose
        arrays["actual_pose"][tick] = actual_pose
        arrays["command_q"][tick] = command_q
        arrays["actual_q"][tick] = actual_q
        arrays["actual_dq"][tick] = actual_dq
        arrays["command_orientation_error"][tick] = (
            command_rotation.inv() * target_rotation
        ).magnitude()
        arrays["orientation_error"][tick] = (
            actual_rotation.inv() * target_rotation
        ).magnitude()
        if isinstance(task, BoundedFrameTask):
            arrays["activation"][tick] = task.limit_activation
        singularity = solver._singularity_limits.get("right")
        if singularity is not None and singularity.last_state is not None:
            arrays["singularity_ratio"][tick] = singularity.last_state.effective_ratio
    return arrays


def summarize(profile: Profile, trace: dict[str, np.ndarray]) -> dict[str, Any]:
    target_position = trace["target_pose"][:, :3]
    command_error = np.linalg.norm(
        trace["command_pose"][:, :3] - target_position,
        axis=1,
    )
    actual_error = np.linalg.norm(
        trace["actual_pose"][:, :3] - target_position,
        axis=1,
    )
    command_dq = np.diff(trace["command_q"], axis=0) / CONTROL_DT
    command_ddq = np.diff(command_dq, axis=0) / CONTROL_DT
    actual_ddq = np.diff(trace["actual_dq"], axis=0) / CONTROL_DT
    ranges = joint_ranges(make_setup("right"), "right")
    margins = np.minimum(
        trace["command_q"] - ranges[:, 0],
        ranges[:, 1] - trace["command_q"],
    )
    caps = (
        np.asarray(profile.caps, dtype=np.float64)
        if profile.caps is not None
        else np.full(7, np.nan)
    )
    flip_end = (
        0.2
        + np.linalg.norm(profile.pre_translation) / profile.pre_translation_speed
        + profile.angle / profile.angular_speed
    )
    after_flip = trace["time"] >= flip_end
    settled = np.flatnonzero(after_flip & (actual_error < 0.01))
    settle_time = (
        float(trace["time"][settled[0]] - flip_end) if settled.size else math.nan
    )
    orientation_settled = np.flatnonzero(
        after_flip & (trace["orientation_error"] < 0.05)
    )
    orientation_settle_time = (
        float(trace["time"][orientation_settled[0]] - flip_end)
        if orientation_settled.size
        else math.nan
    )
    finite_singularity_ratios = trace["singularity_ratio"][
        np.isfinite(trace["singularity_ratio"])
    ]
    return {
        "profile": profile.name,
        "dynamic_plant": profile.dynamic_plant,
        "gravity": profile.gravity,
        "angular_speed_rad_s": profile.angular_speed,
        "axis": str(profile.local_axis),
        "pre_translation_m": str(profile.pre_translation),
        "braking": profile.braking,
        "frame_position_limit_m": profile.frame_position_limit,
        "frame_error_latch_multiplier": profile.frame_error_latch_multiplier,
        "forced_activation": profile.forced_activation,
        "orientation_error_limit_rad": profile.orientation_error_limit,
        "independent_orientation_limit": profile.independent_orientation_limit,
        "angular_schedule": profile.angular_schedule,
        "orientation_latch": profile.orientation_latch,
        "angular_speed_slow_rad_s": profile.angular_speed_slow,
        "angular_speed_fast_rad_s": profile.angular_speed_fast,
        "position_cost": profile.position_cost,
        "orientation_cost": profile.orientation_cost,
        "nullspace_cost": profile.nullspace_cost,
        "singularity_rate": profile.singularity_rate,
        "kinetic_energy_cost": profile.kinetic_energy_cost,
        "caps": str(profile.caps),
        "max_command_position_error_m": float(np.max(command_error)),
        "max_actual_position_error_m": float(np.max(actual_error)),
        "final_actual_position_error_m": float(actual_error[-1]),
        "max_command_orientation_error_rad": float(
            np.max(trace["command_orientation_error"])
        ),
        "max_orientation_error_rad": float(np.max(trace["orientation_error"])),
        "final_orientation_error_rad": float(trace["orientation_error"][-1]),
        "settle_time_to_1cm_s": settle_time,
        "orientation_settle_time_to_0p05rad_s": orientation_settle_time,
        "max_command_joint_speed_rad_s": float(np.max(np.abs(command_dq))),
        "max_command_joint_acceleration_rad_s2": float(np.max(np.abs(command_ddq))),
        "actual_joint_acceleration_p99_rad_s2": float(
            np.percentile(np.abs(actual_ddq), 99)
        ),
        "max_proximal_joint_excursion_rad": float(
            np.max(np.abs(trace["actual_q"][:, :4] - trace["actual_q"][0, :4]))
        ),
        "velocity_saturation_fraction": (
            float(np.mean(np.abs(command_dq) >= 0.98 * caps))
            if np.all(np.isfinite(caps))
            else math.nan
        ),
        "minimum_joint_margin_rad": float(np.min(margins)),
        "minimum_singularity_ratio": (
            float(np.min(finite_singularity_ratios))
            if finite_singularity_ratios.size
            else math.nan
        ),
        "mean_frame_limit_activation": float(np.mean(trace["activation"])),
        "solve_failures": int(np.sum(trace["failed"])),
    }


def read_arm_state(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    table = pq.read_table(path)
    timestamp = np.asarray(table["timestamp"].cast(pa.int64()))
    qpos = np.asarray(table["qpos"].to_pylist(), dtype=np.float64)[:, :7]
    qvel = (
        np.asarray(table["qvel"].to_pylist(), dtype=np.float64)[:, :7]
        if "qvel" in table.column_names
        else None
    )
    return timestamp, qpos, qvel


def fk_series(
    setup: ArmSetup,
    side: str,
    qpos: np.ndarray,
) -> np.ndarray:
    base_qpos = setup.data.qpos.copy()
    pose = np.empty((len(qpos), 7))
    full_qpos = base_qpos.copy()
    for index, joints in enumerate(qpos):
        full_qpos[:] = base_qpos
        setup.joint_resolver.set_qpos(
            full_qpos,
            np.append(joints, 0.0),
            side,
        )
        setup.data.qpos[:] = full_qpos
        mujoco.mj_forward(setup.model, setup.data)
        pose[index] = setup.read_ee_pose(side)
    return pose


def extract_record_event() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    side = "right"
    action_ns, action_q, _ = read_arm_state(
        EPISODE / "action" / "arms" / side / "state.parquet"
    )
    relative_time = (action_ns - action_ns[0]) * 1e-9
    selected = np.flatnonzero(
        (relative_time >= EVENT_START_S) & (relative_time <= EVENT_END_S)
    )
    action_ns = action_ns[selected]
    action_q = action_q[selected]
    time_s = relative_time[selected]

    obs_ns, obs_q, obs_dq = read_arm_state(
        EPISODE / "obs" / "arms" / side / "state.parquet"
    )
    obs_indices = np.searchsorted(obs_ns, action_ns)
    obs_indices = np.clip(obs_indices, 1, len(obs_ns) - 1)
    choose_previous = np.abs(obs_ns[obs_indices - 1] - action_ns) < np.abs(
        obs_ns[obs_indices] - action_ns
    )
    obs_indices -= choose_previous
    actual_q = obs_q[obs_indices]
    assert obs_dq is not None
    actual_dq = obs_dq[obs_indices]

    setup = make_setup("bimanual")
    command_pose = fk_series(setup, side, action_q)
    actual_pose = fk_series(setup, side, actual_q)
    command_dq = np.gradient(action_q, time_s, axis=0)
    nominal_command_dq = np.diff(action_q, axis=0) / CONTROL_DT
    command_rotation = Rotation.from_quat(command_pose[:, [4, 5, 6, 3]])
    command_rotation_steps = (
        command_rotation[:-1].inv() * command_rotation[1:]
    ).magnitude()
    command_position_delta = np.linalg.norm(
        command_pose[:, :3] - command_pose[0, :3],
        axis=1,
    )
    actual_position_delta = np.linalg.norm(
        actual_pose[:, :3] - actual_pose[0, :3],
        axis=1,
    )
    tracking_error = np.linalg.norm(
        command_pose[:, :3] - actual_pose[:, :3],
        axis=1,
    )
    ranges = joint_ranges(setup, side)
    margins = np.minimum(
        action_q - ranges[:, 0],
        ranges[:, 1] - action_q,
    )
    command_onset_s: dict[str, dict[str, float | None]] = {}
    for threshold in (0.05, 0.10):
        command_indices = np.flatnonzero(command_position_delta > threshold)
        actual_indices = np.flatnonzero(actual_position_delta > threshold)
        command_onset_s[f"{threshold:.2f}"] = {
            "command": (
                float(time_s[command_indices[0]] - time_s[0])
                if command_indices.size
                else None
            ),
            "actual": (
                float(time_s[actual_indices[0]] - time_s[0])
                if actual_indices.size
                else None
            ),
        }
    metrics = {
        "episode": 39,
        "side": side,
        "event_start_s": EVENT_START_S,
        "event_end_s": EVENT_END_S,
        "max_command_position_excursion_m": float(np.max(command_position_delta)),
        "max_actual_position_excursion_m": float(np.max(actual_position_delta)),
        "max_command_actual_position_error_m": float(np.max(tracking_error)),
        "max_abs_command_joint_speed_rad_s": np.max(
            np.abs(command_dq), axis=0
        ).tolist(),
        "max_abs_actual_joint_speed_rad_s": np.max(np.abs(actual_dq), axis=0).tolist(),
        "minimum_joint_margin_rad": np.min(margins, axis=0).tolist(),
        "near_joint_limit_fraction": np.mean(margins < 0.01, axis=0).tolist(),
        "nominal_velocity_saturation_fraction": np.mean(
            np.abs(nominal_command_dq) >= 0.98 * DEFAULT_CAPS,
            axis=0,
        ).tolist(),
        "max_command_orientation_step_rad": float(np.max(command_rotation_steps)),
        "command_orientation_path_rad": float(np.sum(command_rotation_steps)),
        "position_excursion_onset_s": command_onset_s,
    }
    arrays = {
        "time": time_s,
        "command_q": action_q,
        "actual_q": actual_q,
        "actual_dq": actual_dq,
        "command_dq": command_dq,
        "command_pose": command_pose,
        "actual_pose": actual_pose,
    }
    return arrays, metrics


def profiles() -> list[Profile]:
    base = Profile("latched_current")
    cases = [
        base,
        replace(base, name="latched_no_braking", braking=False),
        replace(base, name="unlatched_current", forced_activation=None),
        replace(
            base,
            name="no_frame_error_limit",
            frame_position_limit=0.0,
            forced_activation=None,
        ),
        replace(base, name="latched_no_nullspace", nullspace_cost=0.0),
        replace(base, name="latched_no_singularity", singularity_rate=0.0),
        replace(base, name="latched_no_kinetic", kinetic_energy_cost=0.0),
        replace(base, name="latched_no_velocity_caps", caps=None),
        replace(
            base,
            name="latched_wrist_caps_12p6",
            caps=(2.0, 2.0, 3.14, 3.14, 12.6, 12.6, 12.6),
        ),
    ]
    cases.extend(
        replace(
            base,
            name=f"activation_{activation:.2f}",
            forced_activation=activation,
        )
        for activation in (0.0, 0.25, 0.5, 0.75)
    )
    cases.extend(
        replace(
            base,
            name=f"natural_latch_{direction_name}",
            forced_activation=None,
            pre_translation=translation,
        )
        for direction_name, translation in (
            ("minus_x_0p05", (-0.05, 0.0, 0.0)),
            ("minus_x_0p10", (-0.10, 0.0, 0.0)),
            ("x_0p05", (0.05, 0.0, 0.0)),
            ("minus_y_0p05", (0.0, -0.05, 0.0)),
            ("z_0p05", (0.0, 0.0, 0.05)),
            ("minus_z_0p05", (0.0, 0.0, -0.05)),
        )
    )
    cases.extend(
        (
            replace(
                base,
                name="natural_no_latch_x_0p05",
                forced_activation=None,
                pre_translation=(0.05, 0.0, 0.0),
                frame_error_latch_multiplier=1e6,
            ),
            replace(base, name="angle_1p57", angle=1.57),
            replace(base, name="angle_3p14", angle=3.14),
            replace(
                base,
                name="natural_latch_x_0p05_angle_3p14",
                forced_activation=None,
                pre_translation=(0.05, 0.0, 0.0),
                angle=3.14,
            ),
        )
    )
    cases.extend(
        replace(
            base,
            name=f"angular_speed_{speed:.1f}",
            angular_speed=speed,
        )
        for speed in (2.0, 4.0, 6.0)
    )
    cases.extend(
        replace(
            base,
            name=f"position_limit_{limit:.3f}",
            frame_position_limit=limit,
        )
        for limit in (0.006, 0.010, 0.020, 0.050)
    )
    cases.extend(
        replace(
            base,
            name=f"orientation_limit_{limit:.3f}",
            orientation_error_limit=limit,
        )
        for limit in (0.010, 0.020, 0.040, 0.080, 0.160)
    )
    cases.extend(
        replace(
            base,
            name=f"orientation_cost_{cost:.2f}",
            orientation_cost=cost,
        )
        for cost in (0.1, 0.25, 0.5)
    )
    cases.extend(
        replace(
            base,
            name=f"position_cost_{cost:.0f}",
            position_cost=cost,
        )
        for cost in (30.0, 100.0)
    )
    cases.extend(
        replace(
            base,
            name=f"axis_{axis_name}",
            local_axis=axis,
        )
        for axis_name, axis in (
            ("x", (1.0, 0.0, 0.0)),
            ("minus_x", (-1.0, 0.0, 0.0)),
            ("y", (0.0, 1.0, 0.0)),
            ("minus_y", (0.0, -1.0, 0.0)),
            ("z", (0.0, 0.0, 1.0)),
        )
    )
    cases.extend(
        replace(case, name=f"{case.name}_dynamic", dynamic_plant=True)
        for case in (
            base,
            replace(base, name="unlatched_current", forced_activation=None),
            replace(base, name="latched_no_braking", braking=False),
            replace(
                base,
                name="natural_latch_x_0p05",
                forced_activation=None,
                pre_translation=(0.05, 0.0, 0.0),
            ),
            replace(
                base,
                name="natural_latch_x_0p05_no_braking",
                forced_activation=None,
                pre_translation=(0.05, 0.0, 0.0),
                braking=False,
            ),
            replace(
                base,
                name="natural_no_latch_x_0p05",
                forced_activation=None,
                pre_translation=(0.05, 0.0, 0.0),
                frame_error_latch_multiplier=1e6,
            ),
            replace(
                base,
                name="latched_current_no_gravity",
                dynamic_plant=True,
                gravity=False,
            ),
            replace(
                base,
                name="natural_latch_x_0p05_no_gravity",
                forced_activation=None,
                pre_translation=(0.05, 0.0, 0.0),
                dynamic_plant=True,
                gravity=False,
            ),
        )
    )
    return cases


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_record(path: Path, trace: dict[str, np.ndarray]) -> None:
    time_s = trace["time"] - trace["time"][0]
    figure, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for dim, label in enumerate(("x", "y", "z")):
        axes[0].plot(
            time_s,
            trace["command_pose"][:, dim],
            label=f"command {label}",
        )
        axes[0].plot(
            time_s,
            trace["actual_pose"][:, dim],
            linestyle="--",
            label=f"actual {label}",
        )
    for joint in range(7):
        axes[1].plot(time_s, trace["command_q"][:, joint], label=f"J{joint + 1}")
        axes[2].plot(
            time_s,
            trace["command_dq"][:, joint],
            label=f"J{joint + 1}",
        )
    axes[0].set_ylabel("EEF position [m]")
    axes[1].set_ylabel("command q [rad]")
    axes[2].set_ylabel("command dq [rad/s]")
    axes[2].set_xlabel("event time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_profiles(
    path: Path,
    traces: dict[str, dict[str, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for name, trace in traces.items():
        target_position = trace["target_pose"][:, :3]
        command_error = np.linalg.norm(
            trace["command_pose"][:, :3] - target_position,
            axis=1,
        )
        actual_error = np.linalg.norm(
            trace["actual_pose"][:, :3] - target_position,
            axis=1,
        )
        axes[0].plot(trace["time"], command_error, label=name)
        axes[1].plot(trace["time"], actual_error, label=name)
        axes[2].plot(
            trace["time"],
            trace["orientation_error"],
            label=name,
        )
    axes[0].set_ylabel("command position error [m]")
    axes[1].set_ylabel("actual position error [m]")
    axes[2].set_ylabel("orientation error [rad]")
    axes[2].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    output = Path("dev/results/chest_wrist_flip_20260729")
    output.mkdir(parents=True, exist_ok=True)

    record_trace, record_metrics = extract_record_event()
    plot_record(output / "record_episode39_event.png", record_trace)
    np.savez_compressed(output / "record_episode39_event.npz", **record_trace)

    rows: list[dict[str, Any]] = []
    selected_traces: dict[str, dict[str, np.ndarray]] = {}
    selected_names = {
        "latched_current",
        "latched_no_braking",
        "unlatched_current",
        "no_frame_error_limit",
        "orientation_limit_0.020",
        "angular_speed_4.0",
        "natural_latch_x_0p05",
        "natural_no_latch_x_0p05",
        "latched_current_dynamic",
        "unlatched_current_dynamic",
        "natural_latch_x_0p05_dynamic",
        "natural_no_latch_x_0p05_dynamic",
        "latched_current_no_gravity_dynamic",
    }
    for profile in profiles():
        print(f"simulate {profile.name}", flush=True)
        trace = simulate(profile)
        rows.append(summarize(profile, trace))
        if profile.name in selected_names:
            selected_traces[profile.name] = trace
            np.savez_compressed(output / f"{profile.name}.npz", **trace)

    write_csv(output / "sweep.csv", rows)
    plot_profiles(output / "representative_profiles.png", selected_traces)
    with open(output / "record_metrics.json", "w", encoding="utf-8") as file:
        json.dump(record_metrics, file, indent=2)
    print(json.dumps(record_metrics, indent=2))


if __name__ == "__main__":
    main()
