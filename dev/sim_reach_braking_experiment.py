#!/usr/bin/env python3
"""Compare OpenArm elbow braking profiles in a dynamic MuJoCo reach test.

The right end-effector starts at shoulder height and moves along world +x until
the target is beyond the reachable workspace. Mink produces position commands
at 250 Hz. A separate MuJoCo model tracks those commands with the same position
actuator gains and force limits used by the current high-PD driver config.
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
import openarm_mujoco.v2 as openarm_mujoco

from openarm_control import ArmSetup, IKParams, Kinematics
from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S
from openarm_control.singularity import (
    normalized_arm_jacobian,
    singularity_ratio,
)


INITIAL_RIGHT = np.array(
    [1.224145, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
CONTROL_DT = 1.0 / 250.0
ELBOW_GUARD = 0.08


@dataclass(frozen=True)
class Profile:
    """One IK safety experiment configuration."""

    name: str
    braking_profile: str
    braking_acceleration: float
    slowdown_distance: float
    joint_limit_braking: bool = False
    joint_limit_braking_exponent: float = 2.0
    joint_limit_braking_reaction_time: float = 0.0
    joint_limit_braking_distance_buffer: float = 0.0
    singularity_approach_limit: bool = False
    singularity_ratio_stop: float = 0.01
    singularity_ratio_slow: float = 0.05
    singularity_max_approach_rate: float = 0.5
    singularity_braking_exponent: float = 2.0
    use_measured_state: bool = False
    kinetic_energy_cost: float = 0.0


@dataclass
class Trace:
    """Time series from one dynamic simulation."""

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
    singularity_allowed_rate: np.ndarray
    actuator_force: np.ndarray
    generalized_actuator_force: np.ndarray
    constraint_force: np.ndarray
    bias_force: np.ndarray


@dataclass
class Metrics:
    """Scalar metrics used to compare one run."""

    profile: str
    target_speed_m_s: float
    min_command_q4_rad: float
    min_actual_q4_rad: float
    min_command_rho: float
    min_actual_rho: float
    actual_q4_guard_undershoot_rad: float
    max_command_actual_error_q1_rad: float
    max_command_actual_error_q2_rad: float
    near_max_command_dq1_rad_s: float
    near_max_command_dq2_rad_s: float
    near_max_actual_dq1_rad_s: float
    near_max_actual_dq2_rad_s: float
    near_max_actual_ddq1_rad_s2: float
    near_max_actual_ddq2_rad_s2: float
    hold_q1_overshoot_rad: float
    hold_q2_overshoot_rad: float
    hold_command_q1_p2p_rad: float
    hold_command_q2_p2p_rad: float
    hold_actual_q1_p2p_rad: float
    hold_actual_q2_p2p_rad: float
    hold_ee_x_overshoot_m: float
    hold_ee_z_excursion_m: float
    ramp_command_ee_z_excursion_m: float
    ramp_actual_ee_upward_excursion_m: float
    ramp_actual_ee_z_p2p_m: float
    tail_ee_x_p2p_m: float
    tail_ee_z_p2p_m: float
    hold_ee_x_zero_crossings: int
    hold_ee_z_zero_crossings: int
    near_q1_torque_saturation_fraction: float
    near_q2_torque_saturation_fraction: float
    near_max_constraint_force_q1_nm: float
    near_max_constraint_force_q4_nm: float
    near_max_bias_force_q1_nm: float
    near_max_bias_force_q4_nm: float
    final_target_position_error_m: float


def _setup(mode: str = "right") -> ArmSetup:
    return ArmSetup.from_args(
        xml=openarm_mujoco.openarm_cell_xml(),
        mode=mode,
        frame_right="right_ee_control_point",
        frame_type_right="site",
        frame_left="left_ee_control_point",
        frame_type_left="site",
        keyframe="home",
    )


def _velocity_limits() -> dict[str, float]:
    return {
        f"openarm_{side}_joint{index + 1}": float(cap)
        for side in ("left", "right")
        for index, cap in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S)
    }


def _make_kinematics(profile: Profile) -> Kinematics:
    return Kinematics(
        _setup("right"),
        IKParams(
            position_cost=10.0,
            orientation_cost=1.0,
            lm_damping=0.02,
            damping=0.1,
            posture_cost=0.0,
            dt=CONTROL_DT,
            max_iters=10,
            velocity_limits=_velocity_limits(),
            joint_limit_recovery_velocity_scale=1.0,
            nullspace_cost=10.0,
            nullspace_return_rate=0.8,
            nullspace_max_speed=0.8,
            nullspace_singularity_low=0.02,
            nullspace_singularity_high=0.05,
            nullspace_characteristic_length=0.3,
            elbow_soft_limit_cost=0.0,
            elbow_braking_guard_angle=ELBOW_GUARD,
            elbow_braking_profile=profile.braking_profile,
            elbow_braking_acceleration=profile.braking_acceleration,
            elbow_braking_slowdown_distance=profile.slowdown_distance,
            joint_limit_braking=profile.joint_limit_braking,
            joint_limit_braking_slowdown_distance=profile.slowdown_distance,
            joint_limit_braking_exponent=profile.joint_limit_braking_exponent,
            joint_limit_braking_reaction_time=(
                profile.joint_limit_braking_reaction_time
            ),
            joint_limit_braking_distance_buffer=(
                profile.joint_limit_braking_distance_buffer
            ),
            singularity_approach_limit=profile.singularity_approach_limit,
            singularity_ratio_stop=profile.singularity_ratio_stop,
            singularity_ratio_slow=profile.singularity_ratio_slow,
            singularity_max_approach_rate=(
                profile.singularity_max_approach_rate
            ),
            singularity_braking_exponent=profile.singularity_braking_exponent,
            kinetic_energy_cost=profile.kinetic_energy_cost,
        ),
    )


class DynamicArm:
    """MuJoCo dynamics driven by the model's position actuators."""

    def __init__(self) -> None:
        self.setup = _setup("right")
        self.model = self.setup.model
        self.data = self.setup.data
        self.resolver = self.setup.joint_resolver
        self.resolver.set_qpos(self.data.qpos, INITIAL_RIGHT, "right")
        mujoco.mj_forward(self.model, self.data)

        self._held_qpos = self.data.qpos.copy()
        self._right_actuators = np.array(
            [
                self.model.actuator(f"right_joint{index}_ctrl").id
                for index in range(1, 8)
            ],
            dtype=int,
        )
        resolved = self.resolver._right
        self._right_qpos = np.asarray(resolved.arm_qpos, dtype=int)
        self._right_dofs = np.asarray(resolved.arm_dof, dtype=int)
        self.set_command(INITIAL_RIGHT[:7])

    def set_command(self, command_q: np.ndarray) -> None:
        target_qpos = self._held_qpos.copy()
        self.resolver.set_qpos(
            target_qpos,
            np.append(np.asarray(command_q, dtype=np.float64), 0.0),
            "right",
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

    def step_control_period(self) -> None:
        steps = int(round(CONTROL_DT / self.model.opt.timestep))
        if not np.isclose(steps * self.model.opt.timestep, CONTROL_DT):
            raise ValueError("MuJoCo timestep must divide the control period.")
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def settle(self, duration: float) -> None:
        steps = int(round(duration / self.model.opt.timestep))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def right_q(self) -> np.ndarray:
        return self.data.qpos[self._right_qpos].copy()

    def right_dq(self) -> np.ndarray:
        return self.data.qvel[self._right_dofs].copy()

    def right_force(self) -> np.ndarray:
        return self.data.actuator_force[self._right_actuators].copy()

    def right_generalized_actuator_force(self) -> np.ndarray:
        return self.data.qfrc_actuator[self._right_dofs].copy()

    def right_constraint_force(self) -> np.ndarray:
        return self.data.qfrc_constraint[self._right_dofs].copy()

    def right_bias_force(self) -> np.ndarray:
        return self.data.qfrc_bias[self._right_dofs].copy()

    def right_ee(self) -> np.ndarray:
        return self.setup.read_ee_pose("right").astype(np.float64)

    def bimanual_driver_qpos(self) -> np.ndarray:
        right, right_gripper = self.resolver.get_driver(self.data.qpos, "right")
        left, left_gripper = self.resolver.get_driver(self.data.qpos, "left")
        return np.concatenate(
            [
                np.append(right, right_gripper),
                np.append(left, left_gripper),
            ]
        ).astype(np.float32)

    def bimanual_driver_qvel(self) -> np.ndarray:
        """Return measured driver-order velocities with zero gripper velocity."""
        right = np.append(
            self.data.qvel[self.resolver._right.arm_dof],
            0.0,
        )
        left = np.append(
            self.data.qvel[self.resolver._left.arm_dof],
            0.0,
        )
        return np.concatenate([right, left]).astype(np.float32)


class SingularityMonitor:
    """Evaluate the same geometric conditioning ratio used by the QP limit."""

    def __init__(self, kinematics: Kinematics) -> None:
        solver = kinematics._ik
        assert solver is not None
        self._setup = kinematics.setup
        self._configuration = mink.Configuration(self._setup.model)
        self._base_qpos = solver._config.q.copy()
        self._frame_task = solver._tasks["right"]
        self._dof_indices = np.asarray(
            self._setup.joint_resolver._right.arm_dof,
            dtype=int,
        )

    def ratio(self, arm_qpos: np.ndarray) -> float:
        qpos = self._base_qpos.copy()
        self._setup.joint_resolver.set_qpos(
            qpos,
            np.append(np.asarray(arm_qpos, dtype=np.float64), 0.0),
            "right",
        )
        self._configuration.update(q=qpos)
        jacobian = normalized_arm_jacobian(
            self._frame_task,
            self._configuration,
            self._dof_indices,
            characteristic_length=0.3,
        )
        ratio, _ = singularity_ratio(jacobian)
        return ratio


def simulate(
    profile: Profile,
    target_speed: float,
    *,
    ramp_distance: float,
    settle_duration: float,
    pre_hold_duration: float,
    post_hold_duration: float,
) -> Trace:
    """Run one command-plus-dynamics reach simulation."""
    if target_speed <= 0.0:
        raise ValueError("Target speed must be positive.")

    kinematics = _make_kinematics(profile)
    initial_pose = kinematics.fk("right", INITIAL_RIGHT)
    singularity_monitor = SingularityMonitor(kinematics)
    dynamics = DynamicArm()
    dynamics.settle(settle_duration)

    # The real node synchronizes while inactive, then runs open-loop in command
    # configuration during intervention.
    kinematics.sync(dynamics.bimanual_driver_qpos())

    pre_ticks = int(round(pre_hold_duration / CONTROL_DT))
    ramp_ticks = int(round(ramp_distance / target_speed / CONTROL_DT))
    hold_ticks = int(round(post_hold_duration / CONTROL_DT))
    total_ticks = pre_ticks + ramp_ticks + hold_ticks

    time_values = np.empty(total_ticks)
    phase_values = np.empty(total_ticks, dtype=np.int8)
    target_values = np.empty((total_ticks, 7))
    command_q_values = np.empty((total_ticks, 7))
    command_dq_values = np.empty((total_ticks, 7))
    actual_q_values = np.empty((total_ticks, 7))
    actual_dq_values = np.empty((total_ticks, 7))
    actual_ddq_values = np.empty((total_ticks, 7))
    command_ee_values = np.empty((total_ticks, 7))
    actual_ee_values = np.empty((total_ticks, 7))
    command_rho_values = np.empty(total_ticks)
    actual_rho_values = np.empty(total_ticks)
    singularity_allowed_rate_values = np.full(total_ticks, np.nan)
    force_values = np.empty((total_ticks, 7))
    generalized_actuator_force_values = np.empty((total_ticks, 7))
    constraint_force_values = np.empty((total_ticks, 7))
    bias_force_values = np.empty((total_ticks, 7))

    previous_command = dynamics.right_q()
    previous_actual_dq = dynamics.right_dq()
    for tick in range(total_ticks):
        if tick < pre_ticks:
            phase = 0
            ramp_progress = 0.0
        elif tick < pre_ticks + ramp_ticks:
            phase = 1
            ramp_step = tick - pre_ticks + 1
            ramp_progress = min(ramp_step * target_speed * CONTROL_DT, ramp_distance)
        else:
            phase = 2
            ramp_progress = ramp_distance

        target_pose = initial_pose.copy()
        target_pose[0] += ramp_progress
        if profile.use_measured_state:
            kinematics.update_measured_state(
                dynamics.bimanual_driver_qpos(),
                dynamics.bimanual_driver_qvel(),
            )
        kinematics.set_target("right", target_pose)
        result = kinematics.solve()
        command_q = (
            previous_command if result is None else result[:7].astype(np.float64)
        )
        command_dq = (command_q - previous_command) / CONTROL_DT

        dynamics.set_command(command_q)
        dynamics.step_control_period()
        actual_q = dynamics.right_q()
        actual_dq = dynamics.right_dq()
        actual_ddq = (actual_dq - previous_actual_dq) / CONTROL_DT

        time_values[tick] = (tick - pre_ticks + 1) * CONTROL_DT
        phase_values[tick] = phase
        target_values[tick] = target_pose
        command_q_values[tick] = command_q
        command_dq_values[tick] = command_dq
        actual_q_values[tick] = actual_q
        actual_dq_values[tick] = actual_dq
        actual_ddq_values[tick] = actual_ddq
        command_ee_values[tick] = kinematics.fk("right", np.append(command_q, 0.0))
        actual_ee_values[tick] = dynamics.right_ee()
        command_rho_values[tick] = singularity_monitor.ratio(command_q)
        actual_rho_values[tick] = singularity_monitor.ratio(actual_q)
        solver = kinematics._ik
        assert solver is not None
        singularity_limit = solver._singularity_limits.get("right")
        if singularity_limit is not None and singularity_limit.last_state is not None:
            singularity_allowed_rate_values[tick] = (
                singularity_limit.last_state.max_approach_rate
            )
        force_values[tick] = dynamics.right_force()
        generalized_actuator_force_values[tick] = (
            dynamics.right_generalized_actuator_force()
        )
        constraint_force_values[tick] = dynamics.right_constraint_force()
        bias_force_values[tick] = dynamics.right_bias_force()

        previous_command = command_q
        previous_actual_dq = actual_dq

    return Trace(
        time=time_values,
        phase=phase_values,
        target_pose=target_values,
        command_q=command_q_values,
        command_dq=command_dq_values,
        actual_q=actual_q_values,
        actual_dq=actual_dq_values,
        actual_ddq=actual_ddq_values,
        command_ee=command_ee_values,
        actual_ee=actual_ee_values,
        command_rho=command_rho_values,
        actual_rho=actual_rho_values,
        singularity_allowed_rate=singularity_allowed_rate_values,
        actuator_force=force_values,
        generalized_actuator_force=generalized_actuator_force_values,
        constraint_force=constraint_force_values,
        bias_force=bias_force_values,
    )


def _overshoot(
    values: np.ndarray, hold_mask: np.ndarray, tail_mask: np.ndarray
) -> float:
    hold = values[hold_mask]
    steady = float(np.median(values[tail_mask]))
    direction = float(np.sign(steady - hold[0]))
    if direction == 0.0:
        return 0.5 * float(np.ptp(hold))
    return max(float(np.max(direction * (hold - steady))), 0.0)


def _zero_crossings(values: np.ndarray, center: float, threshold: float) -> int:
    residual = values - center
    significant = residual[np.abs(residual) >= threshold]
    if significant.size < 2:
        return 0
    signs = np.sign(significant)
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def compute_metrics(
    profile: Profile,
    target_speed: float,
    trace: Trace,
) -> Metrics:
    """Summarize command smoothness and dynamic tracking behavior."""
    active_mask = trace.phase >= 1
    ramp_mask = trace.phase == 1
    ramp_start = int(np.flatnonzero(ramp_mask)[0])
    ramp_baseline = trace.actual_ee[max(ramp_start - 1, 0), 2]
    hold_mask = trace.phase == 2
    tail_start = trace.time[-1] - 0.5
    tail_mask = hold_mask & (trace.time >= tail_start)
    near_mask = active_mask & (trace.command_q[:, 3] < 0.25)
    if not np.any(near_mask):
        near_mask = active_mask

    hold_actual_ee = trace.actual_ee[hold_mask]
    tail_actual_ee = trace.actual_ee[tail_mask]
    steady_ee = np.median(tail_actual_ee[:, :3], axis=0)
    ee_x_direction = float(np.sign(steady_ee[0] - hold_actual_ee[0, 0]))
    if ee_x_direction == 0.0:
        ee_x_overshoot = 0.5 * float(np.ptp(hold_actual_ee[:, 0]))
    else:
        ee_x_overshoot = max(
            float(np.max(ee_x_direction * (hold_actual_ee[:, 0] - steady_ee[0]))),
            0.0,
        )

    return Metrics(
        profile=profile.name,
        target_speed_m_s=target_speed,
        min_command_q4_rad=float(np.min(trace.command_q[active_mask, 3])),
        min_actual_q4_rad=float(np.min(trace.actual_q[active_mask, 3])),
        min_command_rho=float(np.min(trace.command_rho[active_mask])),
        min_actual_rho=float(np.min(trace.actual_rho[active_mask])),
        actual_q4_guard_undershoot_rad=max(
            ELBOW_GUARD - float(np.min(trace.actual_q[active_mask, 3])),
            0.0,
        ),
        max_command_actual_error_q1_rad=float(
            np.max(
                np.abs(trace.command_q[active_mask, 0] - trace.actual_q[active_mask, 0])
            )
        ),
        max_command_actual_error_q2_rad=float(
            np.max(
                np.abs(trace.command_q[active_mask, 1] - trace.actual_q[active_mask, 1])
            )
        ),
        near_max_command_dq1_rad_s=float(
            np.max(np.abs(trace.command_dq[near_mask, 0]))
        ),
        near_max_command_dq2_rad_s=float(
            np.max(np.abs(trace.command_dq[near_mask, 1]))
        ),
        near_max_actual_dq1_rad_s=float(np.max(np.abs(trace.actual_dq[near_mask, 0]))),
        near_max_actual_dq2_rad_s=float(np.max(np.abs(trace.actual_dq[near_mask, 1]))),
        near_max_actual_ddq1_rad_s2=float(
            np.max(np.abs(trace.actual_ddq[near_mask, 0]))
        ),
        near_max_actual_ddq2_rad_s2=float(
            np.max(np.abs(trace.actual_ddq[near_mask, 1]))
        ),
        hold_q1_overshoot_rad=_overshoot(trace.actual_q[:, 0], hold_mask, tail_mask),
        hold_q2_overshoot_rad=_overshoot(trace.actual_q[:, 1], hold_mask, tail_mask),
        hold_command_q1_p2p_rad=float(np.ptp(trace.command_q[hold_mask, 0])),
        hold_command_q2_p2p_rad=float(np.ptp(trace.command_q[hold_mask, 1])),
        hold_actual_q1_p2p_rad=float(np.ptp(trace.actual_q[hold_mask, 0])),
        hold_actual_q2_p2p_rad=float(np.ptp(trace.actual_q[hold_mask, 1])),
        hold_ee_x_overshoot_m=ee_x_overshoot,
        hold_ee_z_excursion_m=float(
            np.max(np.abs(hold_actual_ee[:, 2] - steady_ee[2]))
        ),
        ramp_command_ee_z_excursion_m=float(
            np.max(
                np.abs(
                    trace.command_ee[ramp_mask, 2]
                    - trace.target_pose[ramp_mask, 2]
                )
            )
        ),
        ramp_actual_ee_upward_excursion_m=max(
            float(np.max(trace.actual_ee[ramp_mask, 2] - ramp_baseline)),
            0.0,
        ),
        ramp_actual_ee_z_p2p_m=float(np.ptp(trace.actual_ee[ramp_mask, 2])),
        tail_ee_x_p2p_m=float(np.ptp(tail_actual_ee[:, 0])),
        tail_ee_z_p2p_m=float(np.ptp(tail_actual_ee[:, 2])),
        hold_ee_x_zero_crossings=_zero_crossings(
            hold_actual_ee[:, 0], steady_ee[0], threshold=5e-4
        ),
        hold_ee_z_zero_crossings=_zero_crossings(
            hold_actual_ee[:, 2], steady_ee[2], threshold=5e-4
        ),
        near_q1_torque_saturation_fraction=float(
            np.mean(np.abs(trace.actuator_force[near_mask, 0]) >= 0.99 * 40.0)
        ),
        near_q2_torque_saturation_fraction=float(
            np.mean(np.abs(trace.actuator_force[near_mask, 1]) >= 0.99 * 40.0)
        ),
        near_max_constraint_force_q1_nm=float(
            np.max(np.abs(trace.constraint_force[near_mask, 0]))
        ),
        near_max_constraint_force_q4_nm=float(
            np.max(np.abs(trace.constraint_force[near_mask, 3]))
        ),
        near_max_bias_force_q1_nm=float(np.max(np.abs(trace.bias_force[near_mask, 0]))),
        near_max_bias_force_q4_nm=float(np.max(np.abs(trace.bias_force[near_mask, 3]))),
        final_target_position_error_m=float(
            np.linalg.norm(trace.target_pose[-1, :3] - steady_ee)
        ),
    )


def save_trace(path: Path, trace: Trace) -> None:
    """Write one trace to a flat CSV file."""
    fieldnames = ["time_s", "phase"]
    fieldnames += [f"target_{axis}" for axis in ("x", "y", "z")]
    for prefix in ("command_q", "command_dq", "actual_q", "actual_dq", "actual_ddq"):
        fieldnames += [f"{prefix}{index}" for index in range(1, 8)]
    for prefix in ("command_ee", "actual_ee"):
        fieldnames += [f"{prefix}_{axis}" for axis in ("x", "y", "z")]
    fieldnames += [
        "command_rho",
        "actual_rho",
        "singularity_allowed_rate",
    ]
    fieldnames += [f"actuator_force{index}" for index in range(1, 8)]
    for prefix in (
        "generalized_actuator_force",
        "constraint_force",
        "bias_force",
    ):
        fieldnames += [f"{prefix}{index}" for index in range(1, 8)]

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in range(trace.time.size):
            values: dict[str, float | int] = {
                "time_s": float(trace.time[row]),
                "phase": int(trace.phase[row]),
            }
            values.update(
                {
                    f"target_{axis}": float(trace.target_pose[row, index])
                    for index, axis in enumerate(("x", "y", "z"))
                }
            )
            for prefix, array in (
                ("command_q", trace.command_q),
                ("command_dq", trace.command_dq),
                ("actual_q", trace.actual_q),
                ("actual_dq", trace.actual_dq),
                ("actual_ddq", trace.actual_ddq),
            ):
                values.update(
                    {
                        f"{prefix}{index + 1}": float(array[row, index])
                        for index in range(7)
                    }
                )
            for prefix, array in (
                ("command_ee", trace.command_ee),
                ("actual_ee", trace.actual_ee),
            ):
                values.update(
                    {
                        f"{prefix}_{axis}": float(array[row, index])
                        for index, axis in enumerate(("x", "y", "z"))
                    }
                )
            values["command_rho"] = float(trace.command_rho[row])
            values["actual_rho"] = float(trace.actual_rho[row])
            values["singularity_allowed_rate"] = float(
                trace.singularity_allowed_rate[row]
            )
            values.update(
                {
                    f"actuator_force{index + 1}": float(
                        trace.actuator_force[row, index]
                    )
                    for index in range(7)
                }
            )
            for prefix, array in (
                ("generalized_actuator_force", trace.generalized_actuator_force),
                ("constraint_force", trace.constraint_force),
                ("bias_force", trace.bias_force),
            ):
                values.update(
                    {
                        f"{prefix}{index + 1}": float(array[row, index])
                        for index in range(7)
                    }
                )
            writer.writerow(values)


def plot_speed(
    output_path: Path,
    target_speed: float,
    traces: dict[str, Trace],
) -> None:
    """Plot command and dynamic state comparisons for one target speed."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(5, 2, figsize=(16, 15), sharex=True)
    for profile_name, trace in traces.items():
        time = trace.time
        axes[0, 0].plot(time, trace.command_dq[:, 0], label=profile_name)
        axes[0, 1].plot(time, trace.actual_dq[:, 0], label=profile_name)
        axes[1, 0].plot(time, trace.command_dq[:, 1], label=profile_name)
        axes[1, 1].plot(time, trace.actual_dq[:, 1], label=profile_name)
        axes[2, 0].plot(time, trace.command_q[:, 3], label=profile_name)
        axes[2, 1].plot(time, trace.actual_q[:, 3], label=profile_name)
        axes[3, 0].plot(
            time,
            trace.actual_ee[:, 0] - trace.target_pose[:, 0],
            label=profile_name,
        )
        axes[3, 1].plot(
            time,
            trace.actual_ee[:, 2] - trace.target_pose[:, 2],
            label=profile_name,
        )
        axes[4, 0].plot(time, trace.command_rho, label=profile_name)
        axes[4, 1].plot(time, trace.actual_rho, label=profile_name)

    axes[0, 0].set_title("Command shoulder J1 velocity")
    axes[0, 1].set_title("Actual shoulder J1 velocity")
    axes[1, 0].set_title("Command shoulder J2 velocity")
    axes[1, 1].set_title("Actual shoulder J2 velocity")
    axes[2, 0].set_title("Command elbow J4 position")
    axes[2, 1].set_title("Actual elbow J4 position")
    axes[3, 0].set_title("Actual EE x tracking error")
    axes[3, 1].set_title("Actual EE z tracking error")
    axes[4, 0].set_title("Command singularity ratio")
    axes[4, 1].set_title("Actual singularity ratio")
    for row in range(5):
        unit = "rad/s" if row < 2 else ("rad" if row == 2 else ("m" if row == 3 else ""))
        axes[row, 0].set_ylabel(unit)
        axes[row, 1].set_ylabel(unit)
    for axis in axes[-1]:
        axis.set_xlabel("Time from ramp start [s]")
    for axis in axes.flat:
        axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(ncol=2, fontsize=8)
    figure.suptitle(
        f"Shoulder-height outward reach, target speed {target_speed:.2f} m/s"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _profile_name(distance: float) -> str:
    return f"distance_{distance:.2f}".replace(".", "p")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speeds", default="0.05,0.10,0.20")
    parser.add_argument("--distances", default="0.50,0.70,1.00")
    parser.add_argument("--ramp-distance", type=float, default=0.25)
    parser.add_argument("--settle-duration", type=float, default=2.0)
    parser.add_argument("--pre-hold-duration", type=float, default=1.0)
    parser.add_argument("--post-hold-duration", type=float, default=1.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dev/results/reach_braking"),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles = [
        Profile("none", "acceleration", 0.0, 0.5),
        Profile("acceleration_20", "acceleration", 20.0, 0.5),
        *[
            Profile(_profile_name(distance), "distance", 20.0, distance)
            for distance in _parse_floats(args.distances)
        ],
        Profile(
            "all_joint_command",
            "distance",
            20.0,
            0.5,
            joint_limit_braking=True,
        ),
        Profile(
            "all_joint_state",
            "distance",
            20.0,
            0.5,
            joint_limit_braking=True,
            joint_limit_braking_reaction_time=0.04,
            joint_limit_braking_distance_buffer=0.01,
            use_measured_state=True,
        ),
        Profile(
            "all_joint_state_singularity_energy",
            "distance",
            20.0,
            0.5,
            joint_limit_braking=True,
            joint_limit_braking_reaction_time=0.04,
            joint_limit_braking_distance_buffer=0.01,
            singularity_approach_limit=True,
            singularity_ratio_stop=0.02,
            singularity_ratio_slow=0.08,
            singularity_max_approach_rate=0.25,
            use_measured_state=True,
            kinetic_energy_cost=3e-5,
        ),
    ]
    speeds = _parse_floats(args.speeds)

    all_metrics: list[Metrics] = []
    metadata = {
        "profiles": [asdict(profile) for profile in profiles],
        "speeds": speeds,
        "ramp_distance": args.ramp_distance,
        "settle_duration": args.settle_duration,
        "pre_hold_duration": args.pre_hold_duration,
        "post_hold_duration": args.post_hold_duration,
        "control_dt": CONTROL_DT,
        "initial_right": INITIAL_RIGHT.tolist(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    for speed in speeds:
        traces: dict[str, Trace] = {}
        for profile in profiles:
            print(f"Running speed={speed:.3f} profile={profile.name}", flush=True)
            trace = simulate(
                profile,
                speed,
                ramp_distance=args.ramp_distance,
                settle_duration=args.settle_duration,
                pre_hold_duration=args.pre_hold_duration,
                post_hold_duration=args.post_hold_duration,
            )
            traces[profile.name] = trace
            metrics = compute_metrics(profile, speed, trace)
            all_metrics.append(metrics)
            speed_tag = f"{speed:.3f}".replace(".", "p")
            save_trace(output_dir / f"trace_{speed_tag}_{profile.name}.csv", trace)

        if not args.no_plots:
            speed_tag = f"{speed:.3f}".replace(".", "p")
            plot_speed(
                output_dir / f"comparison_{speed_tag}.png",
                speed,
                traces,
            )

    summary_path = output_dir / "summary.csv"
    fieldnames = list(asdict(all_metrics[0]))
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in all_metrics:
            writer.writerow(asdict(metrics))

    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
