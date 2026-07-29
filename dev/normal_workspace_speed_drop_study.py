#!/usr/bin/env python3
"""Diagnose speed drops during fast motion in the normal arm workspace."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mink
import mujoco
import numpy as np
import openarm_mujoco.v2 as openarm_mujoco

from openarm_control import ArmSetup, IKParams, Kinematics
from openarm_control.arm_joint_limit import ArmJointLimit
from openarm_control.bounded_frame_task import BoundedFrameTask
from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S

CONTROL_DT = 1.0 / 250.0
START_Q_RIGHT = np.array(
    [
        -0.24633651,
        0.12070639,
        0.30554437,
        2.1537668,
        0.44992149,
        -0.02483132,
        0.68082729,
    ]
)
START_Q_LEFT = np.array(
    [0.31404758, -0.30474089, -0.33635407, 2.195766, -0.39094594, 0.1, -0.685755]
)
SIDES = ("right", "left")
OUTPUT_OFFSET = {"right": 0, "left": 8}
PLANE_BASIS = {
    "xy": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    "xz": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    "yz": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
}
LINE_DIRECTIONS = {
    "line_x": np.array([1.0, 0.0, 0.0]),
    "line_y": np.array([0.0, 1.0, 0.0]),
    "line_z": np.array([0.0, 0.0, 1.0]),
    "line_xz": np.array([1.0, 0.0, 1.0]) / math.sqrt(2.0),
}


@dataclass(frozen=True)
class Profile:
    name: str
    braking: bool = True
    braking_distance: float = 0.2
    reaction_time: float = 0.0
    distance_buffer: float = 0.01
    use_measured_state: bool = True
    frame_error_limit: float = 0.003
    frame_error_latch_multiplier: float = 2.0
    frame_activation_override: float | None = None
    singularity_rate: float = 0.25
    nullspace_cost: float = 12.0
    kinetic_energy_cost: float = 3e-5
    split_limits: bool = False


@dataclass(frozen=True)
class Scenario:
    name: str
    side: str
    plane: str
    speed: float
    targets: np.ndarray
    phase: np.ndarray


@dataclass
class SplitBrakingState:
    measured_position: np.ndarray | None
    measured_velocity: np.ndarray | None
    lower_distance: np.ndarray
    upper_distance: np.ndarray
    lower_velocity: np.ndarray
    upper_velocity: np.ndarray


class SplitBrakingLimit(mink.Limit):
    """Benchmark-only old-style braking inequality."""

    def __init__(
        self,
        model: mujoco.MjModel,
        qpos_indices: np.ndarray,
        velocities: dict[str, float],
        *,
        slowdown_distance: float,
        reaction_time: float,
        distance_buffer: float,
        exponent: float = 2.0,
    ) -> None:
        selected = {int(index) for index in qpos_indices}
        names: list[str] = []
        qpos: list[int] = []
        dofs: list[int] = []
        lower: list[float] = []
        upper: list[float] = []
        caps: list[float] = []
        for joint_id in range(model.njnt):
            qpos_index = int(model.jnt_qposadr[joint_id])
            if qpos_index not in selected or not model.jnt_limited[joint_id]:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            assert name is not None
            names.append(name)
            qpos.append(qpos_index)
            dofs.append(int(model.jnt_dofadr[joint_id]))
            lower.append(float(model.jnt_range[joint_id, 0]))
            upper.append(float(model.jnt_range[joint_id, 1]))
            caps.append(float(velocities[name]))
        self.model = model
        self.joint_names = tuple(names)
        self.qpos_indices = np.asarray(qpos, dtype=int)
        self.dof_indices = np.asarray(dofs, dtype=int)
        self.indices = self.dof_indices
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self.max_velocity = np.asarray(caps)
        self.slowdown_distance = float(slowdown_distance)
        self.reaction_time = float(reaction_time)
        self.distance_buffer = float(distance_buffer)
        self.projection_matrix = np.eye(model.nv)[self.dof_indices]
        self._measured_qpos: np.ndarray | None = None
        self._measured_qvel: np.ndarray | None = None
        self.last_state: SplitBrakingState | None = None

    def update_measured_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self._measured_qpos = np.asarray(qpos, dtype=np.float64).copy()
        self._measured_qvel = np.asarray(qvel, dtype=np.float64).copy()

    def compute_qp_inequalities(
        self,
        configuration: mink.Configuration,
        dt: float,
    ) -> mink.Constraint:
        command_q = configuration.q[self.qpos_indices]
        lower_distance = command_q - self.lower
        upper_distance = self.upper - command_q
        measured_q: np.ndarray | None = None
        measured_dq: np.ndarray | None = None
        if self._measured_qpos is not None and self._measured_qvel is not None:
            measured_q = self._measured_qpos[self.qpos_indices]
            measured_dq = self._measured_qvel[self.dof_indices]
            measured_lower = (
                measured_q
                - self.lower
                - self.reaction_time * np.maximum(-measured_dq, 0.0)
                - self.distance_buffer
            )
            measured_upper = (
                self.upper
                - measured_q
                - self.reaction_time * np.maximum(measured_dq, 0.0)
                - self.distance_buffer
            )
            lower_distance = np.minimum(lower_distance, measured_lower)
            upper_distance = np.minimum(upper_distance, measured_upper)
        lower_velocity = distance_velocity_envelope(
            lower_distance,
            self.max_velocity,
            self.slowdown_distance,
        )
        upper_velocity = distance_velocity_envelope(
            upper_distance,
            self.max_velocity,
            self.slowdown_distance,
        )
        self.last_state = SplitBrakingState(
            measured_position=measured_q,
            measured_velocity=measured_dq,
            lower_distance=lower_distance,
            upper_distance=upper_distance,
            lower_velocity=lower_velocity,
            upper_velocity=upper_velocity,
        )
        return mink.Constraint(
            G=np.vstack([self.projection_matrix, -self.projection_matrix]),
            h=dt * np.hstack([upper_velocity, lower_velocity]),
        )


def make_setup(mode: str) -> ArmSetup:
    return ArmSetup.from_args(
        xml=openarm_mujoco.openarm_cell_xml(),
        mode=mode,
        frame_right="right_ee_control_point",
        frame_type_right="site",
        frame_left="left_ee_control_point",
        frame_type_left="site",
        keyframe="home",
    )


def velocity_mapping() -> dict[str, float]:
    return {
        f"openarm_{side}_joint{index + 1}": float(cap)
        for side in SIDES
        for index, cap in enumerate(ARM_JOINT_VELOCITY_LIMITS_RAD_S)
    }


def distance_velocity_envelope(
    distance: np.ndarray,
    max_velocity: np.ndarray,
    slowdown_distance: float,
) -> np.ndarray:
    u = np.clip(np.maximum(distance, 0.0) / slowdown_distance, 0.0, 1.0)
    smoothstep = u * u * (3.0 - 2.0 * u)
    return max_velocity * smoothstep**2


def initial_driver_state(setup: ArmSetup) -> np.ndarray:
    qpos = setup.data.qpos.copy()
    setup.joint_resolver.set_qpos(qpos, np.append(START_Q_RIGHT, 0.0), "right")
    setup.joint_resolver.set_qpos(qpos, np.append(START_Q_LEFT, 0.0), "left")
    values: list[np.ndarray] = []
    for side in SIDES:
        joints, gripper = setup.joint_resolver.get_driver(qpos, side)
        values.append(np.append(joints, gripper))
    return np.concatenate(values)


def initial_pose(side: str) -> np.ndarray:
    setup = make_setup("bimanual")
    qpos = setup.data.qpos.copy()
    setup.joint_resolver.set_qpos(qpos, np.append(START_Q_RIGHT, 0.0), "right")
    setup.joint_resolver.set_qpos(qpos, np.append(START_Q_LEFT, 0.0), "left")
    setup.data.qpos[:] = qpos
    mujoco.mj_forward(setup.model, setup.data)
    return setup.read_ee_pose(side).astype(np.float64)


def circle_angle(elapsed: float, omega: float, ramp: float) -> tuple[float, float]:
    total_angle = 4.0 * math.pi
    ramp_angle = 0.5 * omega * ramp
    cruise = total_angle / omega - ramp
    if elapsed <= ramp:
        angle = omega * (
            0.5 * elapsed - ramp * math.sin(math.pi * elapsed / ramp) / (2.0 * math.pi)
        )
        rate = 0.5 * omega * (1.0 - math.cos(math.pi * elapsed / ramp))
    elif elapsed <= ramp + cruise:
        angle = ramp_angle + omega * (elapsed - ramp)
        rate = omega
    else:
        tail = min(elapsed - ramp - cruise, ramp)
        angle = (
            ramp_angle
            + omega * cruise
            + omega
            * (0.5 * tail + ramp * math.sin(math.pi * tail / ramp) / (2.0 * math.pi))
        )
        rate = 0.5 * omega * (1.0 + math.cos(math.pi * tail / ramp))
    return min(angle, total_angle), max(rate, 0.0)


def make_circle_scenario(side: str, plane: str, speed: float) -> Scenario:
    base = initial_pose(side)
    radius = 0.055
    omega = speed / radius
    ramp = 0.12
    cruise = 4.0 * math.pi / omega - ramp
    motion_duration = 2.0 * ramp + cruise
    hold_before = round(0.30 / CONTROL_DT)
    motion_count = math.ceil(motion_duration / CONTROL_DT)
    hold_after = round(0.40 / CONTROL_DT)
    e1, e2 = PLANE_BASIS[plane]
    mirror = -1.0 if side == "left" and plane in ("xy", "yz") else 1.0
    e1 = e1.copy()
    e2 = e2.copy()
    if abs(e1[1]) > 0.0:
        e1 *= mirror
    if abs(e2[1]) > 0.0:
        e2 *= mirror
    targets: list[np.ndarray] = [base.copy() for _ in range(hold_before)]
    phase: list[int] = [0] * hold_before
    for index in range(1, motion_count + 1):
        angle, _ = circle_angle(index * CONTROL_DT, omega, ramp)
        pose = base.copy()
        pose[:3] += radius * (math.sin(angle) * e1 + (1.0 - math.cos(angle)) * e2)
        targets.append(pose)
        phase.append(1)
    targets.extend(base.copy() for _ in range(hold_after))
    phase.extend([2] * hold_after)
    speed_tag = f"{speed:.1f}".replace(".", "p")
    return Scenario(
        name=f"{side}_{plane}_v{speed_tag}",
        side=side,
        plane=plane,
        speed=speed,
        targets=np.asarray(targets),
        phase=np.asarray(phase, dtype=np.int8),
    )


def make_line_scenario(side: str, direction_name: str, speed: float) -> Scenario:
    base = initial_pose(side)
    direction = LINE_DIRECTIONS[direction_name].copy()
    if side == "left" and abs(direction[1]) > 0.0:
        direction[1] *= -1.0
    distance = 0.18
    start = base.copy()
    start[:3] -= 0.5 * distance * direction
    finish = base.copy()
    finish[:3] += 0.5 * distance * direction

    targets: list[np.ndarray] = []
    phase: list[int] = []
    preposition_count = round(0.50 / CONTROL_DT)
    for index in range(preposition_count):
        u = (index + 1) / preposition_count
        blend = u**3 * (10.0 - 15.0 * u + 6.0 * u**2)
        pose = base.copy()
        pose[:3] += blend * (start[:3] - base[:3])
        targets.append(pose)
        phase.append(0)
    targets.extend(start.copy() for _ in range(round(0.20 / CONTROL_DT)))
    phase.extend([0] * round(0.20 / CONTROL_DT))

    acceleration = 10.0
    acceleration_time = speed / acceleration
    acceleration_distance = 0.5 * acceleration * acceleration_time**2
    cruise_distance = distance - 2.0 * acceleration_distance
    if cruise_distance < 0.0:
        acceleration_time = math.sqrt(distance / acceleration)
        speed = acceleration * acceleration_time
        acceleration_distance = 0.5 * distance
        cruise_distance = 0.0
    cruise_time = cruise_distance / speed
    motion_duration = 2.0 * acceleration_time + cruise_time
    motion_count = math.ceil(motion_duration / CONTROL_DT)
    for index in range(1, motion_count + 1):
        elapsed = min(index * CONTROL_DT, motion_duration)
        if elapsed <= acceleration_time:
            travelled = 0.5 * acceleration * elapsed**2
        elif elapsed <= acceleration_time + cruise_time:
            travelled = acceleration_distance + speed * (elapsed - acceleration_time)
        else:
            remaining = motion_duration - elapsed
            travelled = distance - 0.5 * acceleration * remaining**2
        pose = start.copy()
        pose[:3] += travelled * direction
        targets.append(pose)
        phase.append(1)
    targets.extend(finish.copy() for _ in range(round(0.40 / CONTROL_DT)))
    phase.extend([2] * round(0.40 / CONTROL_DT))
    speed_tag = f"{speed:.1f}".replace(".", "p")
    return Scenario(
        name=f"{side}_{direction_name}_v{speed_tag}",
        side=side,
        plane=direction_name,
        speed=speed,
        targets=np.asarray(targets),
        phase=np.asarray(phase, dtype=np.int8),
    )


class DynamicPlant:
    def __init__(self) -> None:
        self.setup = make_setup("bimanual")
        self.model = self.setup.model
        self.data = self.setup.data
        self.resolver = self.setup.joint_resolver
        self.qpos_indices = {
            side: self.resolver.arm_qpos_indices(side) for side in SIDES
        }
        self.dof_indices = {side: self.resolver.arm_dof_indices(side) for side in SIDES}
        self.actuator_indices = {
            side: np.asarray(
                [
                    self.model.actuator(f"{side}_joint{index}_ctrl").id
                    for index in range(1, 8)
                ],
                dtype=int,
            )
            for side in SIDES
        }
        self.resolver.set_qpos(
            self.data.qpos,
            np.append(START_Q_RIGHT, 0.0),
            "right",
        )
        self.resolver.set_qpos(
            self.data.qpos,
            np.append(START_Q_LEFT, 0.0),
            "left",
        )
        mujoco.mj_forward(self.model, self.data)
        self.command = {
            "right": START_Q_RIGHT.copy(),
            "left": START_Q_LEFT.copy(),
        }
        self.set_command({})

    def set_command(self, updates: dict[str, np.ndarray]) -> None:
        self.command.update(
            {
                side: np.asarray(value, dtype=np.float64).copy()
                for side, value in updates.items()
            }
        )
        target_qpos = self.data.qpos.copy()
        for side in SIDES:
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

    def settle(self, duration: float) -> None:
        for _ in range(round(duration / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

    def step(self) -> None:
        count = round(CONTROL_DT / self.model.opt.timestep)
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)

    def q(self, side: str) -> np.ndarray:
        return self.data.qpos[self.qpos_indices[side]].copy()

    def dq(self, side: str) -> np.ndarray:
        return self.data.qvel[self.dof_indices[side]].copy()

    def pose(self, side: str) -> np.ndarray:
        return self.setup.read_ee_pose(side).astype(np.float64)

    def driver_qpos(self) -> np.ndarray:
        values: list[np.ndarray] = []
        for side in SIDES:
            joints, gripper = self.resolver.get_driver(self.data.qpos, side)
            values.append(np.append(joints, gripper))
        return np.concatenate(values)

    def driver_qvel(self) -> np.ndarray:
        return np.concatenate(
            [
                np.append(self.dq("right"), 0.0),
                np.append(self.dq("left"), 0.0),
            ]
        )


class CommandMonitor:
    def __init__(self) -> None:
        self.setup = make_setup("bimanual")
        self.base_qpos = self.setup.data.qpos.copy()

    def pose(self, side: str, joints: np.ndarray) -> np.ndarray:
        qpos = self.base_qpos.copy()
        self.setup.joint_resolver.set_qpos(
            qpos,
            np.append(joints, 0.0),
            side,
        )
        self.setup.data.qpos[:] = qpos
        mujoco.mj_forward(self.setup.model, self.setup.data)
        return self.setup.read_ee_pose(side).astype(np.float64)


def make_kinematics(
    profile: Profile,
    side: str,
) -> tuple[Kinematics, ArmJointLimit | SplitBrakingLimit]:
    limits = velocity_mapping()
    kinematics = Kinematics(
        make_setup(side),
        IKParams(
            position_cost=10.0,
            orientation_cost=1.0,
            lm_damping=0.01,
            damping=0.1,
            posture_cost=0.0,
            dt=CONTROL_DT,
            max_iters=5,
            velocity_limits=limits,
            frame_position_error_limit=profile.frame_error_limit,
            frame_error_latch_multiplier=profile.frame_error_latch_multiplier,
            joint_braking=profile.braking,
            joint_braking_distance=profile.braking_distance,
            joint_braking_distance_buffer=profile.distance_buffer,
            nullspace_cost=profile.nullspace_cost,
            singularity_max_approach_rate=profile.singularity_rate,
            kinetic_energy_cost=profile.kinetic_energy_cost,
        ),
    )
    solver = kinematics._ik
    assert solver is not None
    assert solver._joint_limit is not None
    diagnostic_limit: ArmJointLimit | SplitBrakingLimit = solver._joint_limit
    if profile.split_limits:
        merged = solver._joint_limit
        recoverable = ArmJointLimit(
            model=solver._config.model,
            qpos_indices=kinematics.setup.joint_resolver.arm_qpos_indices(side),
            velocities=limits,
            position_gain=0.95,
            braking_distance=None,
            braking_exponent=2.0,
            braking_distance_buffer=profile.distance_buffer,
        )
        braking = SplitBrakingLimit(
            model=solver._config.model,
            qpos_indices=kinematics.setup.joint_resolver.arm_qpos_indices(side),
            velocities=limits,
            slowdown_distance=profile.braking_distance,
            reaction_time=profile.reaction_time,
            distance_buffer=profile.distance_buffer,
        )
        index = solver._limits.index(merged)
        solver._limits[index : index + 1] = [recoverable, braking]
        solver._joint_limit = recoverable
        diagnostic_limit = braking
    return kinematics, diagnostic_limit


def limit_velocity_state(
    limit: ArmJointLimit | SplitBrakingLimit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    state = limit.last_state
    assert state is not None
    if isinstance(limit, SplitBrakingLimit):
        return (
            state.lower_velocity.copy(),
            state.upper_velocity.copy(),
            None if state.measured_velocity is None else state.measured_velocity.copy(),
        )
    if state.lower_distance is None or state.upper_distance is None:
        return (
            limit.max_velocity.copy(),
            limit.max_velocity.copy(),
            None,
        )
    return (
        distance_velocity_envelope(
            state.lower_distance,
            limit.max_velocity,
            float(limit.braking_distance),
        ),
        distance_velocity_envelope(
            state.upper_distance,
            limit.max_velocity,
            float(limit.braking_distance),
        ),
        None,
    )


def instrumented_solve(
    kinematics: Kinematics,
    diagnostic_limit: ArmJointLimit | SplitBrakingLimit,
    side: str,
) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    original = mink.solve_ik
    dofs = kinematics.setup.joint_resolver.arm_dof_indices(side)
    velocities: list[np.ndarray] = []
    lower_limits: list[np.ndarray] = []
    upper_limits: list[np.ndarray] = []
    measured_velocities: list[np.ndarray] = []

    def wrapped(*args: Any, **kwargs: Any) -> np.ndarray:
        velocity = original(*args, **kwargs)
        lower, upper, measured = limit_velocity_state(diagnostic_limit)
        velocities.append(velocity[dofs].copy())
        lower_limits.append(lower)
        upper_limits.append(upper)
        measured_velocities.append(np.zeros(7) if measured is None else measured.copy())
        return velocity

    mink.solve_ik = wrapped
    try:
        result = kinematics.solve()
    finally:
        mink.solve_ik = original
    if not velocities:
        empty = np.empty((0, 7))
        return result, {
            "velocity": empty,
            "lower": empty,
            "upper": empty,
            "measured_velocity": empty,
        }
    return result, {
        "velocity": np.asarray(velocities),
        "lower": np.asarray(lower_limits),
        "upper": np.asarray(upper_limits),
        "measured_velocity": np.asarray(measured_velocities),
    }


def simulate(profile: Profile, scenario: Scenario) -> dict[str, np.ndarray]:
    kinematics, diagnostic_limit = make_kinematics(profile, scenario.side)
    solver = kinematics._ik
    assert solver is not None
    plant = DynamicPlant()
    monitor = CommandMonitor()
    plant.settle(0.6)
    kinematics.sync(plant.driver_qpos())
    count = scenario.targets.shape[0]
    arrays = {
        "target_pose": scenario.targets.copy(),
        "command_pose": np.empty((count, 7)),
        "actual_pose": np.empty((count, 7)),
        "command_q": np.empty((count, 7)),
        "actual_q": np.empty((count, 7)),
        "command_dq": np.empty((count, 7)),
        "actual_dq": np.empty((count, 7)),
        "brake_fraction": np.ones((count, 7)),
        "brake_utilization": np.zeros((count, 7)),
        "hard_velocity_utilization": np.zeros((count, 7)),
        "reaction_loss": np.zeros((count, 7)),
        "frame_activation": np.zeros(count),
        "frame_error": np.zeros(count),
        "singularity_activation": np.ones(count),
        "singularity_ratio": np.full(count, np.nan),
        "solve_failed": np.zeros(count, dtype=bool),
        "solve_time": np.zeros(count),
    }
    previous_command = plant.q(scenario.side)
    for tick, target in enumerate(scenario.targets):
        if profile.use_measured_state:
            measured_qpos = plant.driver_qpos()
            measured_qvel = plant.driver_qvel()
            kinematics.update_measured_state(measured_qpos, measured_qvel)
            if isinstance(diagnostic_limit, SplitBrakingLimit):
                mapped_qpos, mapped_qvel = kinematics.setup.driver_state_to_mujoco(
                    measured_qpos,
                    measured_qvel,
                    base_qpos=solver._config.q,
                )
                diagnostic_limit.update_measured_state(mapped_qpos, mapped_qvel)
        kinematics.set_target(scenario.side, target)
        task = solver._tasks[scenario.side]
        if (
            isinstance(task, BoundedFrameTask)
            and profile.frame_activation_override is not None
        ):
            task.set_limit_activation(profile.frame_activation_override)
        started = time.perf_counter()
        result, substeps = instrumented_solve(
            kinematics,
            diagnostic_limit,
            scenario.side,
        )
        arrays["solve_time"][tick] = time.perf_counter() - started
        arrays["solve_failed"][tick] = result is None
        offset = OUTPUT_OFFSET[scenario.side]
        command = (
            previous_command.copy()
            if result is None
            else np.asarray(result[offset : offset + 7], dtype=np.float64)
        )
        command_dq = (command - previous_command) / CONTROL_DT
        arrays["command_q"][tick] = command
        arrays["command_dq"][tick] = command_dq
        arrays["command_pose"][tick] = monitor.pose(scenario.side, command)

        if substeps["velocity"].size:
            direction_limits = np.where(
                substeps["velocity"] >= 0.0,
                substeps["upper"],
                substeps["lower"],
            )
            utilization = np.abs(substeps["velocity"]) / np.maximum(
                direction_limits,
                1e-12,
            )
            fractions = (
                np.minimum(
                    substeps["lower"],
                    substeps["upper"],
                )
                / diagnostic_limit.max_velocity
            )
            arrays["brake_utilization"][tick] = np.max(utilization, axis=0)
            arrays["brake_fraction"][tick] = np.min(fractions, axis=0)
            arrays["hard_velocity_utilization"][tick] = np.max(
                np.abs(substeps["velocity"]) / diagnostic_limit.max_velocity,
                axis=0,
            )
            approach_measured = np.where(
                substeps["velocity"] >= 0.0,
                np.maximum(substeps["measured_velocity"], 0.0),
                np.maximum(-substeps["measured_velocity"], 0.0),
            )
            arrays["reaction_loss"][tick] = np.max(
                profile.reaction_time * approach_measured + profile.distance_buffer,
                axis=0,
            )

        if isinstance(task, BoundedFrameTask):
            arrays["frame_activation"][tick] = task.limit_activation
            arrays["frame_error"][tick] = np.linalg.norm(
                task.compute_full_error(solver._config)[:3]
            )
        singularity = solver._singularity_limits.get(scenario.side)
        if singularity is not None and singularity.last_state is not None:
            arrays["singularity_activation"][tick] = singularity.last_state.activation
            arrays["singularity_ratio"][tick] = singularity.last_state.effective_ratio

        plant.set_command({scenario.side: command})
        plant.step()
        arrays["actual_q"][tick] = plant.q(scenario.side)
        arrays["actual_dq"][tick] = plant.dq(scenario.side)
        arrays["actual_pose"][tick] = plant.pose(scenario.side)
        previous_command = command
    return arrays


def derivative(values: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values)
    output[1:] = np.diff(values, axis=0) / CONTROL_DT
    return output


def smooth(values: np.ndarray, width: int = 5) -> np.ndarray:
    if values.size < width:
        return values.copy()
    kernel = np.ones(width) / width
    return np.convolve(values, kernel, mode="same")


def summarize(
    profile: Profile,
    scenario: Scenario,
    trace: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_velocity = derivative(trace["target_pose"][:, :3])
    command_velocity = derivative(trace["command_pose"][:, :3])
    actual_velocity = derivative(trace["actual_pose"][:, :3])
    target_speed = np.linalg.norm(target_velocity, axis=1)
    command_speed = np.linalg.norm(command_velocity, axis=1)
    actual_speed = np.linalg.norm(actual_velocity, axis=1)
    unit = target_velocity / np.maximum(target_speed[:, None], 1e-12)
    command_along = np.sum(command_velocity * unit, axis=1)
    actual_along = np.sum(actual_velocity * unit, axis=1)
    command_lateral = np.linalg.norm(
        command_velocity - command_along[:, None] * unit,
        axis=1,
    )
    actual_lateral = np.linalg.norm(
        actual_velocity - actual_along[:, None] * unit,
        axis=1,
    )
    active = (scenario.phase == 1) & (target_speed >= 0.95 * scenario.speed)
    active_indices = np.flatnonzero(active)
    smoothed_actual = smooth(actual_speed)
    smoothed_command = smooth(command_speed)
    drop_window = 5
    candidates = active_indices[active_indices >= drop_window]
    candidates = candidates[active[candidates - drop_window]]
    if candidates.size:
        actual_drop_values = (
            smoothed_actual[candidates - drop_window] - smoothed_actual[candidates]
        )
        event_tick = int(candidates[int(np.argmax(actual_drop_values))])
        actual_drop = float(np.max(actual_drop_values))
        command_drop = float(
            np.max(
                smoothed_command[candidates - drop_window]
                - smoothed_command[candidates]
            )
        )
    else:
        event_tick = 0
        actual_drop = 0.0
        command_drop = 0.0
    event_end = min(event_tick + round(0.08 / CONTROL_DT), target_speed.size)
    event_slice = slice(event_tick, max(event_tick + 1, event_end))
    brake_joint = int(np.argmax(trace["brake_utilization"][event_tick]))
    path_error_command = np.linalg.norm(
        trace["target_pose"][:, :3] - trace["command_pose"][:, :3],
        axis=1,
    )
    path_error_actual = np.linalg.norm(
        trace["target_pose"][:, :3] - trace["actual_pose"][:, :3],
        axis=1,
    )
    command_q_gap = trace["command_q"] - trace["actual_q"]
    summary: dict[str, Any] = {
        "profile": profile.name,
        "scenario": scenario.name,
        "side": scenario.side,
        "plane": scenario.plane,
        "target_speed_m_s": scenario.speed,
        "solver_failures": int(np.count_nonzero(trace["solve_failed"])),
        "command_speed_mean_m_s": float(np.mean(command_speed[active])),
        "actual_speed_mean_m_s": float(np.mean(actual_speed[active])),
        "command_along_ratio_p05": float(
            np.percentile(command_along[active] / target_speed[active], 5)
        ),
        "actual_along_ratio_p05": float(
            np.percentile(actual_along[active] / target_speed[active], 5)
        ),
        "command_lateral_ratio_p95": float(
            np.percentile(command_lateral[active] / target_speed[active], 95)
        ),
        "actual_lateral_ratio_p95": float(
            np.percentile(actual_lateral[active] / target_speed[active], 95)
        ),
        "command_speed_drop_20ms_m_s": command_drop,
        "actual_speed_drop_20ms_m_s": actual_drop,
        "command_path_error_p95_m": float(
            np.percentile(path_error_command[active], 95)
        ),
        "actual_path_error_p95_m": float(np.percentile(path_error_actual[active], 95)),
        "q_gap_rms_rad": float(np.sqrt(np.mean(command_q_gap[active] ** 2))),
        "braking_binding_fraction": float(
            np.mean(np.max(trace["brake_utilization"][active], axis=1) >= 0.98)
        ),
        "hard_velocity_binding_fraction": float(
            np.mean(np.max(trace["hard_velocity_utilization"][active], axis=1) >= 0.98)
        ),
        "braking_min_fraction": float(np.min(trace["brake_fraction"][active])),
        "reaction_loss_max_rad": float(np.max(trace["reaction_loss"][active])),
        "frame_activation_mean": float(np.mean(trace["frame_activation"][active])),
        "frame_error_p95_m": float(np.percentile(trace["frame_error"][active], 95)),
        "singularity_active_fraction": float(
            np.mean(trace["singularity_activation"][active] < 0.999)
        ),
        "singularity_ratio_min": float(
            np.nanmin(trace["singularity_ratio"][active])
            if np.any(np.isfinite(trace["singularity_ratio"][active]))
            else math.nan
        ),
        "solve_time_mean_ms": 1e3 * float(np.mean(trace["solve_time"])),
    }
    for joint in range(7):
        summary[f"braking_j{joint + 1}_binding_fraction"] = float(
            np.mean(trace["brake_utilization"][active, joint] >= 0.98)
        )
        summary[f"braking_j{joint + 1}_min_fraction"] = float(
            np.min(trace["brake_fraction"][active, joint])
        )
        summary[f"hard_j{joint + 1}_binding_fraction"] = float(
            np.mean(trace["hard_velocity_utilization"][active, joint] >= 0.98)
        )
    event = {
        "profile": profile.name,
        "scenario": scenario.name,
        "time_s": event_tick * CONTROL_DT,
        "target_speed_m_s": float(target_speed[event_tick]),
        "command_speed_m_s": float(command_speed[event_tick]),
        "actual_speed_m_s": float(actual_speed[event_tick]),
        "actual_speed_drop_20ms_m_s": actual_drop,
        "post_event_lateral_speed_max_m_s": float(np.max(actual_lateral[event_slice])),
        "braking_joint": brake_joint + 1,
        "braking_utilization": float(
            trace["brake_utilization"][event_tick, brake_joint]
        ),
        "braking_fraction": float(trace["brake_fraction"][event_tick, brake_joint]),
        "reaction_loss_rad": float(trace["reaction_loss"][event_tick, brake_joint]),
        "hard_velocity_utilization": float(
            trace["hard_velocity_utilization"][event_tick, brake_joint]
        ),
        "frame_activation": float(trace["frame_activation"][event_tick]),
        "frame_error_m": float(trace["frame_error"][event_tick]),
        "singularity_activation": float(trace["singularity_activation"][event_tick]),
        "singularity_ratio": float(trace["singularity_ratio"][event_tick]),
    }
    return summary, event


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_traces(
    output: Path,
    scenario: Scenario,
    traces: dict[str, dict[str, np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt

    profiles = [
        name
        for name in (
            "current",
            "split_reference",
            "no_braking",
            "measured_position_only",
            "reaction_only",
            "buffer_only",
        )
        if name in traces
    ]
    times = np.arange(scenario.targets.shape[0]) * CONTROL_DT
    target_speed = np.linalg.norm(
        derivative(scenario.targets[:, :3]),
        axis=1,
    )
    figure, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(times, target_speed, "k--", label="target")
    for name in profiles:
        trace = traces[name]
        command_speed = np.linalg.norm(
            derivative(trace["command_pose"][:, :3]),
            axis=1,
        )
        actual_speed = np.linalg.norm(
            derivative(trace["actual_pose"][:, :3]),
            axis=1,
        )
        axes[0].plot(times, command_speed, label=f"{name}: command")
        axes[1].plot(times, actual_speed, label=name)
        axes[2].plot(
            times,
            np.max(trace["brake_utilization"], axis=1),
            label=name,
        )
        axes[3].plot(
            times,
            np.min(trace["brake_fraction"], axis=1),
            label=name,
        )
    axes[0].set_ylabel("command speed [m/s]")
    axes[1].set_ylabel("actual speed [m/s]")
    axes[2].set_ylabel("brake utilization")
    axes[3].set_ylabel("min brake fraction")
    axes[3].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    figure.suptitle(scenario.name)
    figure.tight_layout()
    figure.savefig(output / f"{scenario.name}.png", dpi=150)
    plt.close(figure)


def main() -> None:
    output = Path("dev/results/normal_workspace_speed_drop_20260729")
    output.mkdir(parents=True, exist_ok=True)
    scenarios = [
        make_circle_scenario(side, plane, speed)
        for side in SIDES
        for plane in PLANE_BASIS
        for speed in (0.6, 0.8, 1.0)
    ] + [
        make_line_scenario(side, direction, speed)
        for side in SIDES
        for direction in LINE_DIRECTIONS
        for speed in (0.6, 0.8, 1.0)
    ]
    current = Profile("current")
    summaries: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    trace_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for scenario in scenarios:
        print(f"screen {scenario.name}", flush=True)
        trace = simulate(current, scenario)
        summary, event = summarize(current, scenario, trace)
        summaries.append(summary)
        events.append(event)
        trace_cache[(current.name, scenario.name)] = trace

    current_rows = [row for row in summaries if row["profile"] == "current"]
    current_rows.sort(
        key=lambda row: (
            row["actual_speed_drop_20ms_m_s"]
            + row["actual_lateral_ratio_p95"] * row["target_speed_m_s"]
        ),
        reverse=True,
    )
    selected_names = [
        row["scenario"]
        for is_line in (False, True)
        for row in [
            item
            for item in current_rows
            if item["plane"].startswith("line_") is is_line
        ][:3]
    ]
    selected = [scenario for scenario in scenarios if scenario.name in selected_names]
    profiles = [
        Profile("split_reference", split_limits=True),
        Profile("no_braking", braking=False),
        Profile(
            "measured_position_only",
            reaction_time=0.0,
            distance_buffer=0.0,
        ),
        Profile(
            "reaction_only",
            reaction_time=0.04,
            distance_buffer=0.0,
            split_limits=True,
        ),
        Profile("buffer_only", reaction_time=0.0),
        Profile("reaction_time_0p01", reaction_time=0.01, split_limits=True),
        Profile("reaction_time_0p02", reaction_time=0.02, split_limits=True),
        Profile("reaction_time_0p06", reaction_time=0.06, split_limits=True),
        Profile("reaction_time_0p08", reaction_time=0.08, split_limits=True),
        Profile("buffer_0p005", distance_buffer=0.005),
        Profile("buffer_0p02", distance_buffer=0.02),
        Profile("command_only", use_measured_state=False),
        Profile("braking_distance_0p2", braking_distance=0.2),
        Profile("braking_distance_0p3", braking_distance=0.3),
        Profile("braking_distance_0p4", braking_distance=0.4),
        Profile("braking_distance_0p6", braking_distance=0.6),
        Profile("fixed_frame_error_limit", frame_activation_override=1.0),
        Profile(
            "speed_frame_error_no_latch",
            frame_error_latch_multiplier=1e6,
        ),
        Profile("no_frame_error_limit", frame_error_limit=0.0),
        Profile("no_singularity_limit", singularity_rate=0.0),
        Profile("no_nullspace", nullspace_cost=0.0),
        Profile("no_kinetic_energy", kinetic_energy_cost=0.0),
    ]
    for profile in profiles:
        for scenario in selected:
            print(f"ablate {profile.name} {scenario.name}", flush=True)
            trace = simulate(profile, scenario)
            summary, event = summarize(profile, scenario, trace)
            summaries.append(summary)
            events.append(event)
            trace_cache[(profile.name, scenario.name)] = trace

    write_csv(output / "summary.csv", summaries)
    write_csv(output / "events.csv", events)
    metadata = {
        "velocity_caps_rad_s": list(ARM_JOINT_VELOCITY_LIMITS_RAD_S),
        "control_dt_s": CONTROL_DT,
        "selected_scenarios": selected_names,
        "profiles": [asdict(current), *[asdict(profile) for profile in profiles]],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    split_differences = {}
    for scenario in selected:
        current_trace = trace_cache[("current", scenario.name)]
        split_trace = trace_cache[("split_reference", scenario.name)]
        split_differences[scenario.name] = {
            key: float(np.max(np.abs(current_trace[key] - split_trace[key])))
            for key in (
                "command_q",
                "actual_q",
                "command_pose",
                "actual_pose",
            )
        }
    (output / "split_equivalence.json").write_text(
        json.dumps(split_differences, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for scenario in selected:
        plot_traces(
            output,
            scenario,
            {
                profile: trace
                for (profile, scenario_name), trace in trace_cache.items()
                if scenario_name == scenario.name
            },
        )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
