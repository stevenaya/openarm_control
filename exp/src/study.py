#!/usr/bin/env python3
"""Current-PR MuJoCo study for OpenArm differential-IK behavior.

The benchmark intentionally separates the command-side Mink configuration from
an independently simulated MuJoCo plant.  This matches the production VR path:
Mink integrates position commands open-loop while measured q/dq are supplied
only to state-aware safety limits, and the real arm follows position commands
with finite actuator bandwidth.

Results are cached per (profile, scenario), so interrupted sweeps can resume.

This development-only harness was adapted from the archived 2026-07-29 safety
study. It runs against the current production API, explicitly models the
downstream driver's position-command velocity limiter, and records raw IK,
driver-reference, and dynamic-plant states separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, fields, replace
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import mink
import mujoco
import numpy as np
import yaml
import openarm_mujoco.v2 as openarm_mujoco
from scipy.spatial.transform import Rotation

from openarm_control import ArmSetup, IKParams, Kinematics
from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S
from openarm_control.geometry.jacobian import (
    normalized_arm_jacobian,
    singularity_ratio,
)
from openarm_control.qp.arm_joint_limit import (
    ArmConfigurationLimit,
    ArmJointLimit,
    _distance_velocity_envelope,
)
from openarm_control.qp.bounded_frame_task import BoundedFrameTask
from openarm_control.qp.nullspace_posture_task import (
    smoothstep_activation,
    structural_nullspace_direction,
)

CONTROL_DT = 1.0 / 250.0
SIDES = ("right", "left")
SIDE_OUTPUT_OFFSET = {"right": 0, "left": 8}
CONTROL_CAPS = tuple(float(value) for value in ARM_JOINT_VELOCITY_LIMITS_RAD_S)
# Historical candidate builders below used this name before the PR default
# were sourced directly from openarm_control.config.
CURRENT_CAPS = CONTROL_CAPS
REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER_CONFIG_PATH = (
    Path(__file__).resolve().parent / "inputs" / "experiment_driver_config.yaml"
)


def _load_driver_velocity_caps(path: Path = DRIVER_CONFIG_PATH) -> tuple[float, ...]:
    """Read the seven arm velocity caps from the frozen experiment config."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    configured = payload.get("joint_velocity_limits")
    if not isinstance(configured, list) or len(configured) < 7:
        raise ValueError(f"{path} must define at least seven joint_velocity_limits")
    caps = tuple(float(value) for value in configured[:7])
    if not all(np.isfinite(value) and value > 0.0 for value in caps):
        raise ValueError(f"{path} contains invalid arm joint_velocity_limits")
    return caps


DRIVER_CAPS = _load_driver_velocity_caps()

HOME_Q = np.array([0.0, 0.0, 0.0, math.pi / 2.0, 0.0, 0.0, 0.0])
REACH_Q_RIGHT = np.array([1.224145, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0])
REACH_Q_LEFT = np.array([-1.224145, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0])
EXTENDED_Q_RIGHT = np.array([1.224145, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0])
EXTENDED_Q_LEFT = np.array([-1.224145, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0])
START_Q_RIGHT = np.array(
    [-0.24633651, 0.12070639, 0.30554437, 2.1537668, 0.44992149, -0.02483132, 0.68082729]
)
START_Q_LEFT = np.array(
    [0.31404758, -0.30474089, -0.33635407, 2.195766, -0.39094594, 0.1, -0.685755]
)
CHEST_Q_RIGHT = np.array(
    [1.02, 0.74, -1.50, 1.34, -1.57, 0.79, -1.09],
    dtype=np.float64,
)


@lru_cache(maxsize=1)
def _git_revision() -> str:
    source_file = Path(inspect.getfile(ArmSetup)).resolve()
    package_root = source_file.parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=package_root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@lru_cache(maxsize=1)
def _git_worktree_state() -> dict[str, Any]:
    source_file = Path(inspect.getfile(ArmSetup)).resolve()
    package_root = source_file.parents[2]
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=package_root,
            text=True,
        )
        patch = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"],
            cwd=package_root,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"dirty": None, "tracked_diff_sha256": "unknown"}
    return {
        "dirty": bool(status),
        "tracked_diff_sha256": hashlib.sha256(patch).hexdigest(),
    }


@lru_cache(maxsize=1)
def _study_revision() -> str:
    source = Path(__file__).read_bytes()
    return hashlib.sha256(source).hexdigest()[:16]


@dataclass(frozen=True)
class Profile:
    """One command-side controller configuration."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    velocity_caps: tuple[float, ...] | None = CONTROL_CAPS
    limit_style: str = "recoverable"
    driver_velocity_caps: tuple[float, ...] | None = DRIVER_CAPS
    use_measured_state: bool = True
    gravity_compensation: bool = False
    state_delay_s: float = 0.0
    command_delay_s: float = 0.0
    state_rate_hz: float = 250.0
    state_dropout_start_s: float = -1.0
    state_dropout_duration_s: float = 0.0
    actuator_kp_scale: float = 1.0
    actuator_kv_scale: float = 1.0
    description: str = ""


@dataclass(frozen=True)
class Scenario:
    """Fully sampled target trajectory and fixed initial joint state."""

    name: str
    family: str
    mode: str
    speed: float
    times: np.ndarray
    phase: np.ndarray
    target_right: np.ndarray
    target_left: np.ndarray
    initial_right: np.ndarray
    initial_left: np.ndarray
    description: str


@dataclass
class SideTrace:
    """Time-series diagnostics for one arm."""

    target_pose: np.ndarray
    command_pose: np.ndarray
    driver_pose: np.ndarray
    actual_pose: np.ndarray
    command_q: np.ndarray
    driver_q: np.ndarray
    actual_q: np.ndarray
    command_dq: np.ndarray
    driver_dq: np.ndarray
    actual_dq: np.ndarray
    command_ddq: np.ndarray
    driver_ddq: np.ndarray
    actual_ddq: np.ndarray
    command_elbow: np.ndarray
    driver_elbow: np.ndarray
    actual_elbow: np.ndarray
    command_rho: np.ndarray
    driver_rho: np.ndarray
    actual_rho: np.ndarray
    frame_full_position_error: np.ndarray
    frame_used_position_error: np.ndarray
    frame_full_orientation_error: np.ndarray
    frame_used_orientation_error: np.ndarray
    frame_limit_activation: np.ndarray
    nullspace_activation: np.ndarray
    nullspace_error: np.ndarray
    nullspace_return_speed: np.ndarray
    singularity_activation: np.ndarray
    singularity_allowed_rate: np.ndarray
    braking_min_fraction: np.ndarray
    braking_utilization: np.ndarray
    braking_fraction_by_joint: np.ndarray
    braking_utilization_by_joint: np.ndarray
    driver_limit_active_by_joint: np.ndarray
    actuator_force: np.ndarray


@dataclass
class RunTrace:
    """All time series from one command-plus-plant simulation."""

    times: np.ndarray
    phase: np.ndarray
    solve_time: np.ndarray
    solver_failed: np.ndarray
    sides: dict[str, SideTrace]


def current_parameter_values() -> dict[str, Any]:
    """Return the production defaults used by the experiment profile."""
    values = asdict(IKParams())
    values.pop("solver")
    values.pop("velocity_limits")
    return values


def driver_config_values() -> dict[str, Any]:
    """Return the frozen driver velocity envelope and its provenance."""
    return {
        "path": DRIVER_CONFIG_PATH.relative_to(REPO_ROOT).as_posix(),
        "sha256": hashlib.sha256(DRIVER_CONFIG_PATH.read_bytes()).hexdigest(),
        "arm_joint_velocity_caps_rad_s": list(DRIVER_CAPS),
    }


def velocity_mapping(caps: tuple[float, ...] | None) -> dict[str, float] | None:
    """Build a Mink joint-name velocity mapping."""
    if caps is None:
        return None
    if len(caps) != 7:
        raise ValueError("Expected seven arm velocity caps.")
    return {
        f"openarm_{side}_joint{index + 1}": float(cap)
        for side in SIDES
        for index, cap in enumerate(caps)
    }


def make_setup(mode: str, *, relative: bool = True) -> ArmSetup:
    """Build a current-model setup in relative or world coordinates."""
    return ArmSetup.from_args(
        xml=openarm_mujoco.openarm_cell_xml(),
        mode=mode,
        frame_right="right_ee_control_point",
        frame_type_right="site",
        frame_left="left_ee_control_point",
        frame_type_left="site",
        keyframe="home",
        origin_frame="arm_origin" if relative else "world",
        origin_frame_type="site",
    )


def make_kinematics(profile: Profile, mode: str) -> Kinematics:
    """Construct Kinematics and apply benchmark-only limit variants."""
    values = current_parameter_values()
    values.update(profile.overrides)
    values["velocity_limits"] = (
        velocity_mapping(profile.velocity_caps)
        if profile.limit_style in {"recoverable", "standard"}
        else None
    )

    supported = {item.name for item in fields(IKParams)}
    params = IKParams(**{key: value for key, value in values.items() if key in supported})
    kinematics = Kinematics(make_setup(mode), params)

    if profile.limit_style == "standard":
        solver = kinematics._ik
        assert solver is not None
        active_qpos = {
            int(index)
            for side in solver._sides
            for index in solver._joint_resolver.arm_qpos_indices(side)
        }
        configuration_limit = ArmConfigurationLimit(
            solver._setup.model,
            active_qpos,
            gain=values["joint_limit_gain"],
        )
        velocity_limit = mink.VelocityLimit(
            solver._setup.model,
            velocity_mapping(profile.velocity_caps),
        )
        extra_limits = [
            limit
            for limit in solver._limits
            if not isinstance(limit, (ArmJointLimit, ArmConfigurationLimit))
        ]
        solver._limits = [
            configuration_limit,
            velocity_limit,
            *extra_limits,
        ]
        solver._joint_limit = None
    elif profile.limit_style not in ("recoverable", "configuration_only"):
        raise ValueError(f"Unknown limit style {profile.limit_style!r}.")
    return kinematics


def _quat_wxyz_to_rotation(quaternion: np.ndarray) -> Rotation:
    return Rotation.from_quat(
        [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    )


def _rotation_to_quat_wxyz(rotation: Rotation) -> np.ndarray:
    q = rotation.as_quat()
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quaternion_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Return the shortest SO(3) angle between two wxyz quaternions."""
    relative = _quat_wxyz_to_rotation(q1).inv() * _quat_wxyz_to_rotation(q2)
    return float(np.linalg.norm(relative.as_rotvec()))


def interpolate_pose(start: np.ndarray, end: np.ndarray, amount: float) -> np.ndarray:
    """Interpolate position linearly and orientation on SO(3)."""
    output = np.empty(7, dtype=np.float64)
    output[:3] = (1.0 - amount) * start[:3] + amount * end[:3]
    start_rotation = _quat_wxyz_to_rotation(start[3:])
    delta = (start_rotation.inv() * _quat_wxyz_to_rotation(end[3:])).as_rotvec()
    output[3:] = _rotation_to_quat_wxyz(
        start_rotation * Rotation.from_rotvec(amount * delta)
    )
    return output


def rotate_pose_local(pose: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """Apply a local-frame rotation to a pose."""
    output = pose.copy()
    output[3:] = _rotation_to_quat_wxyz(
        _quat_wxyz_to_rotation(pose[3:]) * Rotation.from_rotvec(rotvec)
    )
    return output


def _continuous_quaternions(poses: np.ndarray) -> np.ndarray:
    output = poses.copy()
    for index in range(1, output.shape[0]):
        if float(output[index - 1, 3:] @ output[index, 3:]) < 0.0:
            output[index, 3:] *= -1.0
    return output


class PathBuilder:
    """Build piecewise-smooth sampled pose paths for one or two arms."""

    def __init__(self, initial: dict[str, np.ndarray]) -> None:
        self.current = {side: pose.copy() for side, pose in initial.items()}
        self._poses = {side: [] for side in initial}
        self._phase: list[int] = []

    def _append(self, poses: dict[str, np.ndarray], phase: int) -> None:
        for side, pose in poses.items():
            self._poses[side].append(pose.copy())
            self.current[side] = pose.copy()
        self._phase.append(phase)

    def hold(self, duration: float, phase: int = 0) -> None:
        count = max(1, round(duration / CONTROL_DT))
        for _ in range(count):
            self._append(self.current, phase)

    def move(
        self,
        targets: dict[str, np.ndarray],
        *,
        linear_speed: float,
        angular_speed: float,
        phase: int,
    ) -> None:
        duration = 0.0
        starts = {side: self.current[side].copy() for side in targets}
        for side, target in targets.items():
            distance = float(np.linalg.norm(target[:3] - starts[side][:3]))
            angle = quaternion_angle(starts[side][3:], target[3:])
            if distance > 0.0:
                duration = max(duration, 1.5 * distance / linear_speed)
            if angle > 0.0:
                duration = max(duration, 1.5 * angle / angular_speed)
        count = max(1, math.ceil(duration / CONTROL_DT))
        for index in range(1, count + 1):
            u = index / count
            amount = u * u * (3.0 - 2.0 * u)
            self._append(
                {
                    side: interpolate_pose(starts[side], target, amount)
                    for side, target in targets.items()
                },
                phase,
            )

    def sample(
        self,
        duration: float,
        function: Any,
        *,
        phase: int,
    ) -> None:
        count = max(1, math.ceil(duration / CONTROL_DT))
        for index in range(1, count + 1):
            self._append(function(index * CONTROL_DT), phase)

    def arrays(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        phase = np.asarray(self._phase, dtype=np.int16)
        poses = {
            side: _continuous_quaternions(np.asarray(values, dtype=np.float64))
            for side, values in self._poses.items()
        }
        return phase, poses


class PoseFactory:
    """Read initial poses from one relative-frame reference model."""

    def __init__(self) -> None:
        self.setup = make_setup("bimanual")
        self.kinematics = Kinematics(self.setup)

    def bimanual(
        self,
        right: np.ndarray,
        left: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.kinematics.fk_bimanual(
            np.append(right, 0.0),
            np.append(left, 0.0),
        )


def _scenario_from_builder(
    *,
    name: str,
    family: str,
    mode: str,
    speed: float,
    builder: PathBuilder,
    initial_right: np.ndarray,
    initial_left: np.ndarray,
    description: str,
) -> Scenario:
    phase, poses = builder.arrays()
    count = phase.size
    target_right = poses.get("right")
    target_left = poses.get("left")
    if target_right is None:
        target_right = np.full((count, 7), np.nan)
    if target_left is None:
        target_left = np.full((count, 7), np.nan)
    return Scenario(
        name=name,
        family=family,
        mode=mode,
        speed=float(speed),
        times=np.arange(count, dtype=np.float64) * CONTROL_DT,
        phase=phase,
        target_right=target_right,
        target_left=target_left,
        initial_right=initial_right.copy(),
        initial_left=initial_left.copy(),
        description=description,
    )


def make_reach_scenario(
    factory: PoseFactory,
    *,
    speed: float,
    lateral: float = 0.0,
    side: str = "right",
) -> Scenario:
    """Reach beyond the workspace, hold, then retract."""
    initial_right = REACH_Q_RIGHT.copy()
    initial_left = REACH_Q_LEFT.copy()
    right_pose, left_pose = factory.bimanual(initial_right, initial_left)
    base = right_pose if side == "right" else left_pose
    builder = PathBuilder({side: base})
    builder.hold(0.25)
    target = base.copy()
    target[:3] += np.array([0.30, lateral, 0.0])
    builder.move(
        {side: target},
        linear_speed=speed,
        angular_speed=8.0,
        phase=1,
    )
    builder.hold(0.65, phase=2)
    builder.move(
        {side: base},
        linear_speed=speed,
        angular_speed=8.0,
        phase=3,
    )
    builder.hold(0.55, phase=4)
    lateral_tag = f"{lateral:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
    return _scenario_from_builder(
        name=f"reach_{side}_{lateral_tag}_v{speed:.2f}".replace(".", "p"),
        family="reach",
        mode=side,
        speed=speed,
        builder=builder,
        initial_right=initial_right,
        initial_left=initial_left,
        description="Shoulder-height reach beyond workspace, hold, and retract.",
    )


def make_diagonal_retract_scenario(
    factory: PoseFactory,
    *,
    speed: float,
    lateral: float,
) -> Scenario:
    """Retract from a weak, extended pose toward a low lateral target."""
    initial_right = EXTENDED_Q_RIGHT.copy()
    initial_left = HOME_Q.copy()
    right_pose, _ = factory.bimanual(initial_right, initial_left)
    builder = PathBuilder({"right": right_pose})
    builder.hold(0.3)
    target = right_pose.copy()
    target[:3] += np.array([-0.22, lateral, -0.12])
    builder.move(
        {"right": target},
        linear_speed=speed,
        angular_speed=8.0,
        phase=1,
    )
    builder.hold(0.65, phase=2)
    lateral_tag = f"{lateral:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
    return _scenario_from_builder(
        name=f"retract_diag_{lateral_tag}_v{speed:.2f}".replace(".", "p"),
        family="retract",
        mode="right",
        speed=speed,
        builder=builder,
        initial_right=initial_right,
        initial_left=initial_left,
        description="Fast diagonal retraction from an extended weak configuration.",
    )


def make_extended_circle_scenario(
    factory: PoseFactory,
    *,
    speed: float,
    side: str = "right",
) -> Scenario:
    """Trace two yz circles while the arm remains nearly extended."""
    initial_right = EXTENDED_Q_RIGHT.copy()
    initial_left = EXTENDED_Q_LEFT.copy()
    right_pose, left_pose = factory.bimanual(initial_right, initial_left)
    base = right_pose if side == "right" else left_pose
    builder = PathBuilder({side: base})
    builder.hold(0.25)
    radius = 0.045
    angular_rate = speed / radius
    duration = 2.0 * math.pi / angular_rate
    lateral_mirror = -1.0 if side == "left" else 1.0

    def target_at(elapsed: float) -> dict[str, np.ndarray]:
        angle = angular_rate * elapsed
        pose = base.copy()
        pose[1] += lateral_mirror * radius * math.sin(angle)
        pose[2] += radius * (1.0 - math.cos(angle))
        return {side: pose}

    builder.sample(duration, target_at, phase=1)
    builder.hold(0.5, phase=2)
    return _scenario_from_builder(
        name=f"extended_circle_{side}_v{speed:.2f}".replace(".", "p"),
        family="extended_translation",
        mode=side,
        speed=speed,
        builder=builder,
        initial_right=initial_right,
        initial_left=initial_left,
        description="Two vertical-lateral circles from rho approximately 0.033.",
    )


def make_extended_axis_scenario(
    factory: PoseFactory,
    *,
    speed: float,
    axis: str,
    side: str = "right",
) -> Scenario:
    """Move out and back along one Cartesian axis near arm extension."""
    initial_right = EXTENDED_Q_RIGHT.copy()
    initial_left = EXTENDED_Q_LEFT.copy()
    right_pose, left_pose = factory.bimanual(initial_right, initial_left)
    base = right_pose if side == "right" else left_pose
    displacement_by_axis = {
        "up": np.array([0.0, 0.0, 0.08]),
        "down": np.array([0.0, 0.0, -0.08]),
        "left": np.array([0.0, 0.08, 0.0]),
        "right": np.array([0.0, -0.08, 0.0]),
    }
    displacement = displacement_by_axis[axis].copy()
    if side == "left":
        displacement[1] *= -1.0
    target = base.copy()
    target[:3] += displacement

    builder = PathBuilder({side: base})
    builder.hold(0.25)
    builder.move(
        {side: target},
        linear_speed=speed,
        angular_speed=8.0,
        phase=1,
    )
    builder.hold(0.25, phase=2)
    builder.move(
        {side: base},
        linear_speed=speed,
        angular_speed=8.0,
        phase=3,
    )
    builder.hold(0.45, phase=4)
    return _scenario_from_builder(
        name=f"extended_{axis}_{side}_v{speed:.2f}".replace(".", "p"),
        family="extended_axis",
        mode=side,
        speed=speed,
        builder=builder,
        initial_right=initial_right,
        initial_left=initial_left,
        description=f"Nearly extended {axis} translation and return.",
    )


def make_wrist_flip_scenario(
    factory: PoseFactory,
    *,
    angular_speed: float,
    extended: bool,
    side: str = "right",
    initial_right_override: np.ndarray | None = None,
    initial_left_override: np.ndarray | None = None,
    name_suffix: str = "",
    rotation_sign: float = 1.0,
) -> Scenario:
    """Flip the wrist back and forth at fixed end-effector position."""
    if extended:
        initial_right = EXTENDED_Q_RIGHT.copy()
        initial_left = EXTENDED_Q_LEFT.copy()
        family = "extended_wrist"
    else:
        initial_right = START_Q_RIGHT.copy()
        initial_left = START_Q_LEFT.copy()
        family = "normal_wrist"
    if initial_right_override is not None:
        initial_right = np.asarray(initial_right_override, dtype=np.float64).copy()
    if initial_left_override is not None:
        initial_left = np.asarray(initial_left_override, dtype=np.float64).copy()
    right_pose, left_pose = factory.bimanual(initial_right, initial_left)
    base = right_pose if side == "right" else left_pose
    builder = PathBuilder({side: base})
    builder.hold(0.25)
    positive = rotate_pose_local(
        base,
        np.array([1.2 * rotation_sign, 0.0, 0.0]),
    )
    negative = rotate_pose_local(
        base,
        np.array([-1.2 * rotation_sign, 0.0, 0.0]),
    )
    builder.move(
        {side: positive},
        linear_speed=1.0,
        angular_speed=angular_speed,
        phase=1,
    )
    builder.move(
        {side: negative},
        linear_speed=1.0,
        angular_speed=angular_speed,
        phase=1,
    )
    builder.move(
        {side: base},
        linear_speed=1.0,
        angular_speed=angular_speed,
        phase=1,
    )
    builder.hold(0.5, phase=2)
    return _scenario_from_builder(
        name=f"{family}_{side}{name_suffix}_w{angular_speed:.1f}".replace(
            ".",
            "p",
        ),
        family=family,
        mode=side,
        speed=angular_speed,
        builder=builder,
        initial_right=initial_right,
        initial_left=initial_left,
        description="Local wrist roll with fixed end-effector position.",
    )


def make_chest_wrist_outward_scenario(
    factory: PoseFactory,
    *,
    linear_speed: float,
    angular_speed: float,
    displacement: tuple[float, float, float] = (0.05, -0.05, -0.02),
    side: str = "right",
) -> Scenario:
    """Rotate the wrist while the chest-region target moves abruptly outward."""
    initial_right = CHEST_Q_RIGHT.copy()
    initial_left = START_Q_LEFT.copy()
    right_pose, left_pose = factory.bimanual(initial_right, initial_left)
    base = right_pose if side == "right" else left_pose
    builder = PathBuilder({side: base})
    builder.hold(0.25)
    target = rotate_pose_local(base, np.array([0.0, 0.0, -math.pi]))
    translation = np.asarray(displacement, dtype=np.float64)
    if side == "left":
        translation[1] *= -1.0
    target[:3] += translation
    distance = float(np.linalg.norm(translation))
    angle = quaternion_angle(base[3:], target[3:])
    translation_duration = 1.5 * distance / linear_speed
    rotation_duration = 1.5 * angle / angular_speed
    duration = max(translation_duration, rotation_duration)
    base_rotation = _quat_wxyz_to_rotation(base[3:])
    rotation_delta = (
        base_rotation.inv() * _quat_wxyz_to_rotation(target[3:])
    ).as_rotvec()

    def target_at(elapsed: float) -> dict[str, np.ndarray]:
        position_u = float(np.clip(elapsed / translation_duration, 0.0, 1.0))
        rotation_u = float(np.clip(elapsed / rotation_duration, 0.0, 1.0))
        position_amount = position_u * position_u * (3.0 - 2.0 * position_u)
        rotation_amount = rotation_u * rotation_u * (3.0 - 2.0 * rotation_u)
        pose = base.copy()
        pose[:3] += position_amount * translation
        pose[3:] = _rotation_to_quat_wxyz(
            base_rotation
            * Rotation.from_rotvec(rotation_amount * rotation_delta)
        )
        return {side: pose}

    builder.sample(duration, target_at, phase=1)
    builder.hold(0.55, phase=2)
    return _scenario_from_builder(
        name=(
            f"chest_outward_flip_{side}_v{linear_speed:.2f}_w{angular_speed:.1f}"
        ).replace(".", "p"),
        family="chest_wrist_outward",
        mode=side,
        speed=max(linear_speed, angular_speed),
        builder=builder,
        initial_right=initial_right,
        initial_left=initial_left,
        description=(
            "Chest-region pi wrist roll combined with an outward/lateral position step."
        ),
    )


def make_normal_lissajous_scenario(
    factory: PoseFactory,
    *,
    speed: float,
    mode: str,
) -> Scenario:
    """Exercise normal workspace translation and orientation."""
    initial_right = START_Q_RIGHT.copy()
    initial_left = START_Q_LEFT.copy()
    right_pose, left_pose = factory.bimanual(initial_right, initial_left)
    active_sides = SIDES if mode == "bimanual" else (mode,)
    bases = {"right": right_pose, "left": left_pose}
    builder = PathBuilder({side: bases[side] for side in active_sides})
    builder.hold(0.25)
    amplitudes = np.array([0.075, 0.055, 0.045])
    harmonic_velocity_bound = np.array(
        [amplitudes[0], 2.0 * amplitudes[1], 3.0 * amplitudes[2]]
    )
    angular_rate = speed / float(np.linalg.norm(harmonic_velocity_bound))
    duration = 2.0 * math.pi / angular_rate

    def target_at(elapsed: float) -> dict[str, np.ndarray]:
        angle = angular_rate * elapsed
        targets: dict[str, np.ndarray] = {}
        for side in active_sides:
            mirror = -1.0 if side == "left" else 1.0
            pose = bases[side].copy()
            pose[:3] += np.array(
                [
                    amplitudes[0] * math.sin(angle),
                    mirror
                    * amplitudes[1]
                    * (math.sin(2.0 * angle + 0.4) - math.sin(0.4)),
                    amplitudes[2]
                    * (math.sin(3.0 * angle - 0.2) - math.sin(-0.2)),
                ]
            )
            pose = rotate_pose_local(
                pose,
                np.array(
                    [
                        0.25 * math.sin(1.3 * angle),
                        mirror
                        * 0.20
                        * (math.sin(0.7 * angle + 0.2) - math.sin(0.2)),
                        mirror
                        * 0.30
                        * (math.sin(angle - 0.3) - math.sin(-0.3)),
                    ]
                ),
            )
            targets[side] = pose
        return targets

    builder.sample(duration, target_at, phase=1)
    builder.hold(0.5, phase=2)
    return _scenario_from_builder(
        name=f"normal_{mode}_v{speed:.2f}".replace(".", "p"),
        family="normal_workspace",
        mode=mode,
        speed=speed,
        builder=builder,
        initial_right=initial_right,
        initial_left=initial_left,
        description="Normal-workspace Lissajous translation and moderate orientation.",
    )


def screening_scenarios() -> list[Scenario]:
    """Return the broad trajectory set used for feature ablations."""
    factory = PoseFactory()
    scenarios: list[Scenario] = []
    for speed in (0.1, 0.4, 0.8):
        scenarios.append(make_reach_scenario(factory, speed=speed))
    for lateral in (-0.10, 0.10):
        scenarios.append(make_reach_scenario(factory, speed=0.4, lateral=lateral))
        for speed in (0.1, 0.4, 0.8):
            scenarios.append(
                make_diagonal_retract_scenario(
                    factory,
                    speed=speed,
                    lateral=lateral,
                )
            )
    for speed in (0.15, 0.4, 0.8):
        scenarios.append(make_extended_circle_scenario(factory, speed=speed))
    for axis in ("up", "down", "left", "right"):
        for speed in (0.2, 0.8):
            scenarios.append(
                make_extended_axis_scenario(factory, speed=speed, axis=axis)
            )
    for angular_speed in (2.0, 6.0, 10.0):
        scenarios.append(
            make_wrist_flip_scenario(
                factory,
                angular_speed=angular_speed,
                extended=True,
            )
        )
        scenarios.append(
            make_wrist_flip_scenario(
                factory,
                angular_speed=angular_speed,
                extended=False,
            )
        )
    for speed in (0.2, 0.5, 0.8):
        scenarios.append(
            make_normal_lissajous_scenario(factory, speed=speed, mode="right")
        )
    for speed in (0.3, 0.6):
        scenarios.append(
            make_normal_lissajous_scenario(factory, speed=speed, mode="bimanual")
        )
    for linear_speed in (0.3, 0.8, 1.2):
        for angular_speed in (4.0, 8.0, 12.0):
            scenarios.append(
                make_chest_wrist_outward_scenario(
                    factory,
                    linear_speed=linear_speed,
                    angular_speed=angular_speed,
                )
            )
    return scenarios


def focused_scenarios() -> list[Scenario]:
    """Return a smaller but adversarial set for parameter sweeps."""
    factory = PoseFactory()
    return [
        make_reach_scenario(factory, speed=0.4),
        make_reach_scenario(factory, speed=0.8),
        make_reach_scenario(factory, speed=0.4, lateral=-0.10),
        make_diagonal_retract_scenario(factory, speed=0.4, lateral=-0.10),
        make_diagonal_retract_scenario(factory, speed=0.8, lateral=0.10),
        make_extended_circle_scenario(factory, speed=0.4),
        make_extended_circle_scenario(factory, speed=0.8),
        make_wrist_flip_scenario(factory, angular_speed=6.0, extended=True),
        make_wrist_flip_scenario(factory, angular_speed=6.0, extended=False),
        make_chest_wrist_outward_scenario(
            factory,
            linear_speed=0.8,
            angular_speed=8.0,
        ),
        make_normal_lissajous_scenario(factory, speed=0.6, mode="bimanual"),
    ]


def symmetry_scenarios() -> list[Scenario]:
    """Return mirrored right/left trajectories for side-equivalence checks."""
    factory = PoseFactory()
    scenarios: list[Scenario] = []
    for speed in (0.4, 0.8):
        scenarios.extend(
            [
                make_reach_scenario(factory, speed=speed, side="right"),
                make_reach_scenario(factory, speed=speed, side="left"),
                make_reach_scenario(
                    factory,
                    speed=speed,
                    lateral=0.10,
                    side="right",
                ),
                make_reach_scenario(
                    factory,
                    speed=speed,
                    lateral=-0.10,
                    side="left",
                ),
                make_extended_circle_scenario(
                    factory,
                    speed=speed,
                    side="right",
                ),
                make_extended_circle_scenario(
                    factory,
                    speed=speed,
                    side="left",
                ),
            ]
        )
    for angular_speed in (6.0, 10.0):
        for extended in (False, True):
            scenarios.extend(
                [
                    make_wrist_flip_scenario(
                        factory,
                        angular_speed=angular_speed,
                        extended=extended,
                        side="right",
                    ),
                    make_wrist_flip_scenario(
                        factory,
                        angular_speed=angular_speed,
                        extended=extended,
                        side="left",
                    ),
                ]
            )
    mirror_signs = np.array([-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
    exact_left = START_Q_RIGHT * mirror_signs
    scenarios.extend(
        [
            make_wrist_flip_scenario(
                factory,
                angular_speed=10.0,
                extended=False,
                side=side,
                initial_right_override=START_Q_RIGHT,
                initial_left_override=exact_left,
                name_suffix="_exact_mirror",
                rotation_sign=-1.0 if side == "left" else 1.0,
            )
            for side in SIDES
        ]
    )
    scenarios.extend(
        [
            make_normal_lissajous_scenario(factory, speed=0.6, mode="right"),
            make_normal_lissajous_scenario(factory, speed=0.6, mode="left"),
            make_normal_lissajous_scenario(factory, speed=0.6, mode="bimanual"),
        ]
    )
    return scenarios


def symmetry_profiles() -> list[Profile]:
    """Compare the active profile with the compact conservative candidate."""
    return [
        current_profile("current"),
        current_profile(
            "compact_candidate",
            velocity_caps=(2.5, *CURRENT_CAPS[1:]),
            braking_distances=(0.25, 0.12, 0.25, 0.50, 0.20, 0.15, 0.20),
            kinetic_energy_cost=5e-5,
            frame_error_limit_linear_slow=0.6,
            frame_error_limit_linear_fast=0.9,
            frame_error_limit_activation_rise_rate=1e6,
            frame_error_limit_activation_fall_rate=1e6,
            nullspace_singularity_high=0.08,
        ),
    ]


def current_profile(name: str = "current", **changes: Any) -> Profile:
    """Return the experiment default profile with selected metadata changes."""
    profile_fields = {
        "velocity_caps",
        "limit_style",
        "error_schedule_override",
        "use_measured_state",
        "gravity_compensation",
        "braking_distances",
        "state_delay_s",
        "command_delay_s",
        "state_rate_hz",
        "state_dropout_start_s",
        "state_dropout_duration_s",
        "actuator_kp_scale",
        "actuator_kv_scale",
        "description",
    }
    metadata = {key: changes.pop(key) for key in list(changes) if key in profile_fields}
    return Profile(name=name, overrides=changes, **metadata)


def screening_profiles() -> list[Profile]:
    """Return one-feature ablations plus historical posture baselines."""
    return [
        current_profile("current", description="Current evaluation-ui settings."),
        current_profile(
            "no_nullspace",
            nullspace_cost=0.0,
            description="Current controller without nullspace home return.",
        ),
        current_profile(
            "no_singularity_limit",
            singularity_approach_limit=False,
            description="Current controller without rho approach-rate limiting.",
        ),
        current_profile(
            "no_joint_braking",
            joint_limit_braking=False,
            description="Current controller without position-dependent speed envelopes.",
        ),
        current_profile(
            "no_error_limit",
            frame_position_error_limit=0.0,
            description="Current controller with native full FrameTask error.",
        ),
        current_profile(
            "no_kinetic_energy",
            kinetic_energy_cost=0.0,
            description="Current controller without inertia-weighted damping.",
        ),
        current_profile(
            "velocity_only",
            nullspace_cost=0.0,
            singularity_approach_limit=False,
            joint_limit_braking=False,
            frame_position_error_limit=0.0,
            kinetic_energy_cost=0.0,
            description="Recoverable position and velocity limits only.",
        ),
        current_profile(
            "standard_position_velocity",
            limit_style="standard",
            nullspace_cost=0.0,
            singularity_approach_limit=False,
            joint_limit_braking=False,
            frame_position_error_limit=0.0,
            kinetic_energy_cost=0.0,
            description="Native intersected position and velocity limits.",
        ),
        current_profile(
            "no_velocity_limit",
            limit_style="no_velocity",
            nullspace_cost=0.0,
            singularity_approach_limit=False,
            joint_limit_braking=False,
            frame_position_error_limit=0.0,
            kinetic_energy_cost=0.0,
            description="Configuration limits only, diagnostic unsafe reference.",
        ),
        current_profile(
            "home_posture_0p01",
            posture_cost=0.01,
            nullspace_cost=0.0,
            description="Historical weak full-joint home posture task.",
        ),
        current_profile(
            "home_posture_0p1",
            posture_cost=0.1,
            nullspace_cost=0.0,
            description="Historical stronger full-joint home posture task.",
        ),
        current_profile(
            "command_only_safety",
            use_measured_state=False,
            description="Current features without measured q/dq limit feedback.",
        ),
    ]


def parameter_profiles() -> list[Profile]:
    """Return one-factor and semantic simplification candidates."""
    profiles: list[Profile] = [current_profile("current")]

    for value in (0.0, 0.0015, 0.003, 0.006, 0.012):
        profiles.append(
            current_profile(
                f"error_limit_{value:g}".replace(".", "p"),
                frame_position_error_limit=value,
            )
        )
    profiles.extend(
        [
            current_profile(
                "error_always_3mm",
                error_schedule_override="always",
            ),
            current_profile(
                "error_instant_schedule",
                frame_error_limit_activation_rise_rate=1e6,
                frame_error_limit_activation_fall_rate=1e6,
            ),
            current_profile(
                "error_tied_threshold",
                frame_error_limit_linear_slow=0.6,
                frame_error_limit_linear_fast=0.9,
                frame_error_limit_activation_rise_rate=1e6,
                frame_error_limit_activation_fall_rate=1e6,
            ),
        ]
    )

    for cost in (0.0, 3.0, 6.0, 12.0, 24.0):
        profiles.append(
            current_profile(
                f"null_cost_{cost:g}".replace(".", "p"),
                nullspace_cost=cost,
            )
        )
    for rate in (0.4, 0.8, 1.6, 3.2):
        profiles.append(
            current_profile(
                f"null_rate_{rate:g}".replace(".", "p"),
                nullspace_return_rate=rate,
            )
        )
    for max_speed in (0.35, 0.6, 1.0, 1.6):
        profiles.append(
            current_profile(
                f"null_max_{max_speed:g}".replace(".", "p"),
                nullspace_max_speed=max_speed,
            )
        )
    profiles.extend(
        [
            current_profile(
                "null_share_rho_window",
                nullspace_singularity_low=0.02,
                nullspace_singularity_high=0.08,
            ),
            current_profile(
                "null_no_activation_window",
                nullspace_singularity_low=0.0,
                nullspace_singularity_high=1e-6,
            ),
        ]
    )

    for slow in (0.05, 0.08, 0.12):
        profiles.append(
            current_profile(
                f"sing_slow_{slow:g}".replace(".", "p"),
                singularity_ratio_slow=slow,
            )
        )
    for max_rate in (0.10, 0.25, 0.50):
        profiles.append(
            current_profile(
                f"sing_rate_{max_rate:g}".replace(".", "p"),
                singularity_max_approach_rate=max_rate,
            )
        )
    for exponent in (1.0, 2.0, 3.0):
        profiles.append(
            current_profile(
                f"sing_exp_{exponent:g}".replace(".", "p"),
                singularity_braking_exponent=exponent,
            )
        )

    for distance in (0.20, 0.35, 0.50, 0.70):
        profiles.append(
            current_profile(
                f"brake_distance_{distance:g}".replace(".", "p"),
                joint_limit_braking_slowdown_distance=distance,
            )
        )
    for exponent in (1.0, 2.0, 3.0):
        profiles.append(
            current_profile(
                f"brake_exp_{exponent:g}".replace(".", "p"),
                joint_limit_braking_exponent=exponent,
            )
        )
    for reaction_time in (0.0, 0.02, 0.04, 0.08):
        profiles.append(
            current_profile(
                f"brake_reaction_{reaction_time:g}".replace(".", "p"),
                joint_limit_braking_reaction_time=reaction_time,
            )
        )
    for buffer in (0.0, 0.01, 0.02):
        profiles.append(
            current_profile(
                f"brake_buffer_{buffer:g}".replace(".", "p"),
                joint_limit_braking_distance_buffer=buffer,
            )
        )

    for cost in (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
        profiles.append(
            current_profile(
                f"energy_{cost:g}".replace(".", "p").replace("-", "m"),
                kinetic_energy_cost=cost,
            )
        )
    for scale in (0.75, 1.0, 1.25, 1.5):
        profiles.append(
            current_profile(
                f"velocity_scale_{scale:g}".replace(".", "p"),
                velocity_caps=tuple(scale * value for value in CURRENT_CAPS),
            )
        )
    for recovery in (1.0, 1.1, 1.3):
        profiles.append(
            current_profile(
                f"recovery_scale_{recovery:g}".replace(".", "p"),
                joint_limit_recovery_velocity_scale=recovery,
            )
        )

    # Remove duplicate current-equivalent profiles while preserving order.
    unique: dict[str, Profile] = {}
    for profile in profiles:
        unique.setdefault(profile.name, profile)
    return list(unique.values())


def simplified_profiles() -> list[Profile]:
    """Return compact parameterizations selected for direct comparison."""
    return [
        current_profile("current"),
        current_profile(
            "simple_fixed_internal",
            frame_error_limit_activation_rise_rate=1e6,
            frame_error_limit_activation_fall_rate=1e6,
            nullspace_singularity_high=0.08,
            singularity_braking_exponent=2.0,
            singularity_gradient_epsilon=1e-4,
            joint_limit_braking_exponent=2.0,
            joint_limit_recovery_velocity_scale=1.0,
            description=(
                "Share rho window and treat schedule rates, exponents, gradient "
                "epsilon, and recovery scale as fixed implementation constants."
            ),
        ),
        current_profile(
            "simple_no_energy",
            frame_error_limit_activation_rise_rate=1e6,
            frame_error_limit_activation_fall_rate=1e6,
            nullspace_singularity_high=0.08,
            kinetic_energy_cost=0.0,
        ),
        current_profile(
            "simple_no_buffer",
            frame_error_limit_activation_rise_rate=1e6,
            frame_error_limit_activation_fall_rate=1e6,
            nullspace_singularity_high=0.08,
            joint_limit_braking_distance_buffer=0.0,
        ),
        current_profile(
            "simple_error_only_threshold",
            frame_position_error_limit=0.003,
            frame_error_limit_linear_slow=0.6,
            frame_error_limit_linear_fast=0.9,
            frame_error_limit_activation_rise_rate=1e6,
            frame_error_limit_activation_fall_rate=1e6,
            nullspace_singularity_high=0.08,
        ),
    ]


def targeted_profiles() -> list[Profile]:
    """Return follow-up profiles for parameter reduction and model A/B tests."""
    compact_distances = (0.25, 0.12, 0.25, 0.50, 0.20, 0.15, 0.20)
    return [
        current_profile("current"),
        current_profile("error_fixed_3mm", error_schedule_override="always"),
        current_profile(
            "error_fixed_6mm",
            frame_position_error_limit=0.006,
            error_schedule_override="always",
        ),
        current_profile(
            "error_fixed_12mm",
            frame_position_error_limit=0.012,
            error_schedule_override="always",
        ),
        current_profile(
            "error_instant_0p5_1p0",
            frame_error_limit_activation_rise_rate=1e6,
            frame_error_limit_activation_fall_rate=1e6,
        ),
        current_profile(
            "error_instant_0p6_0p9",
            frame_error_limit_linear_slow=0.6,
            frame_error_limit_linear_fast=0.9,
            frame_error_limit_activation_rise_rate=1e6,
            frame_error_limit_activation_fall_rate=1e6,
        ),
        current_profile(
            "brake_distance_0p10",
            joint_limit_braking_slowdown_distance=0.10,
        ),
        current_profile(
            "brake_distance_0p15",
            joint_limit_braking_slowdown_distance=0.15,
        ),
        current_profile(
            "brake_per_joint_j2_only",
            braking_distances=(0.50, 0.12, 0.50, 0.50, 0.50, 0.50, 0.50),
        ),
        current_profile(
            "brake_per_joint_compact",
            braking_distances=compact_distances,
        ),
        current_profile(
            "brake_per_joint_short",
            braking_distances=(0.20, 0.10, 0.20, 0.35, 0.15, 0.10, 0.15),
        ),
        current_profile("gravity_compensation", gravity_compensation=True),
        current_profile("energy_0p0001", kinetic_energy_cost=1e-4),
        current_profile(
            "candidate_compact",
            braking_distances=compact_distances,
            frame_error_limit_linear_slow=0.6,
            frame_error_limit_linear_fast=0.9,
            frame_error_limit_activation_rise_rate=1e6,
            frame_error_limit_activation_fall_rate=1e6,
            nullspace_singularity_high=0.08,
            kinetic_energy_cost=1e-4,
        ),
        current_profile(
            "candidate_fixed_6mm",
            braking_distances=compact_distances,
            frame_position_error_limit=0.006,
            error_schedule_override="always",
            nullspace_singularity_high=0.08,
            kinetic_energy_cost=1e-4,
        ),
    ]


def branch_profiles() -> list[Profile]:
    """Return profiles that isolate velocity-induced IK branch changes."""
    compact_distances = (0.25, 0.12, 0.25, 0.50, 0.20, 0.15, 0.20)
    return [
        current_profile("current"),
        current_profile(
            "no_joint_braking_current_tasks",
            joint_limit_braking=False,
        ),
        current_profile(
            "no_velocity_current_tasks",
            limit_style="no_velocity",
            joint_limit_braking=False,
        ),
        current_profile(
            "unlimited_reference",
            limit_style="no_velocity",
            joint_limit_braking=False,
            singularity_approach_limit=False,
        ),
        current_profile(
            "velocity_scale_1p25_current_tasks",
            velocity_caps=tuple(1.25 * value for value in CURRENT_CAPS),
        ),
        current_profile(
            "j1_cap_2p5",
            velocity_caps=(2.5, *CURRENT_CAPS[1:]),
        ),
        current_profile(
            "j1_cap_3p0",
            velocity_caps=(3.0, *CURRENT_CAPS[1:]),
        ),
        current_profile(
            "j1_cap_4p0",
            velocity_caps=(4.0, *CURRENT_CAPS[1:]),
        ),
        current_profile(
            "j1_j4_caps_2p5_3p8",
            velocity_caps=(
                2.5,
                CURRENT_CAPS[1],
                CURRENT_CAPS[2],
                3.8,
                *CURRENT_CAPS[4:],
            ),
        ),
        current_profile(
            "j1_cap_2p5_no_braking",
            velocity_caps=(2.5, *CURRENT_CAPS[1:]),
            joint_limit_braking=False,
        ),
        current_profile(
            "brake_distance_0p10",
            joint_limit_braking_slowdown_distance=0.10,
        ),
        current_profile(
            "brake_per_joint_compact",
            braking_distances=compact_distances,
        ),
        current_profile(
            "candidate_compact",
            braking_distances=compact_distances,
            frame_error_limit_linear_slow=0.6,
            frame_error_limit_linear_fast=0.9,
            frame_error_limit_activation_rise_rate=1e6,
            frame_error_limit_activation_fall_rate=1e6,
            nullspace_singularity_high=0.08,
            kinetic_energy_cost=1e-4,
        ),
    ]


def branch_scenarios() -> list[Scenario]:
    """Return paths that expose branch changes without omitting normal motion."""
    factory = PoseFactory()
    scenarios: list[Scenario] = []
    for lateral in (-0.10, 0.10):
        for speed in (0.1, 0.4, 0.8):
            scenarios.append(
                make_diagonal_retract_scenario(
                    factory,
                    speed=speed,
                    lateral=lateral,
                )
            )
    for speed in (0.4, 0.8):
        scenarios.append(make_reach_scenario(factory, speed=speed))
    for lateral in (-0.10, 0.10):
        scenarios.append(make_reach_scenario(factory, speed=0.4, lateral=lateral))
    for speed in (0.15, 0.4, 0.8):
        scenarios.append(make_extended_circle_scenario(factory, speed=speed))
    for angular_speed in (6.0, 10.0):
        scenarios.append(
            make_wrist_flip_scenario(
                factory,
                angular_speed=angular_speed,
                extended=True,
            )
        )
    for speed in (0.5, 0.8):
        scenarios.append(
            make_normal_lissajous_scenario(factory, speed=speed, mode="right")
        )
    return scenarios


def candidate_profiles() -> list[Profile]:
    """Return compact combinations around the best branch/smoothness tradeoff."""
    compact_distances = (0.25, 0.12, 0.25, 0.50, 0.20, 0.15, 0.20)

    def candidate(
        name: str,
        *,
        j1_cap: float,
        energy_cost: float,
        simplify_error_schedule: bool = False,
    ) -> Profile:
        changes: dict[str, Any] = {
            "velocity_caps": (j1_cap, *CURRENT_CAPS[1:]),
            "braking_distances": compact_distances,
            "kinetic_energy_cost": energy_cost,
        }
        if simplify_error_schedule:
            changes.update(
                frame_error_limit_linear_slow=0.6,
                frame_error_limit_linear_fast=0.9,
                frame_error_limit_activation_rise_rate=1e6,
                frame_error_limit_activation_fall_rate=1e6,
                nullspace_singularity_high=0.08,
            )
        return current_profile(name, **changes)

    return [
        current_profile("current"),
        current_profile(
            "no_velocity_current_tasks",
            limit_style="no_velocity",
            joint_limit_braking=False,
        ),
        candidate("j1_2p5_compact", j1_cap=2.5, energy_cost=3e-5),
        candidate("j1_3p0_compact", j1_cap=3.0, energy_cost=3e-5),
        candidate("j1_2p5_compact_energy5e5", j1_cap=2.5, energy_cost=5e-5),
        candidate("j1_3p0_compact_energy5e5", j1_cap=3.0, energy_cost=5e-5),
        candidate("j1_3p0_compact_energy1e4", j1_cap=3.0, energy_cost=1e-4),
        candidate(
            "j1_2p5_compact_simple",
            j1_cap=2.5,
            energy_cost=5e-5,
            simplify_error_schedule=True,
        ),
    ]


def finalist_profiles() -> list[Profile]:
    """Return final safety/branch candidates with controlled brake variants."""
    simple = {
        "kinetic_energy_cost": 5e-5,
        "frame_error_limit_linear_slow": 0.6,
        "frame_error_limit_linear_fast": 0.9,
        "frame_error_limit_activation_rise_rate": 1e6,
        "frame_error_limit_activation_fall_rate": 1e6,
        "nullspace_singularity_high": 0.08,
    }
    compact = (0.25, 0.12, 0.25, 0.50, 0.20, 0.15, 0.20)
    balanced = (0.25, 0.35, 0.25, 0.50, 0.20, 0.15, 0.20)
    return [
        current_profile("current"),
        current_profile(
            "no_velocity_current_tasks",
            limit_style="no_velocity",
            joint_limit_braking=False,
        ),
        current_profile(
            "j1_2p5_global_brake_simple",
            velocity_caps=(2.5, *CURRENT_CAPS[1:]),
            **simple,
        ),
        current_profile(
            "j1_2p5_global_brake_minimal",
            velocity_caps=(2.5, *CURRENT_CAPS[1:]),
            **{
                **simple,
                "frame_error_limit_linear_slow": 0.45,
                "joint_limit_braking_distance_buffer": 0.0,
            },
        ),
        current_profile(
            "j1_2p5_balanced_brake_simple",
            velocity_caps=(2.5, *CURRENT_CAPS[1:]),
            braking_distances=balanced,
            **simple,
        ),
        current_profile(
            "j1_2p5_compact_brake_simple",
            velocity_caps=(2.5, *CURRENT_CAPS[1:]),
            braking_distances=compact,
            **simple,
        ),
        current_profile(
            "j1_2p0_balanced_brake_simple",
            braking_distances=balanced,
            **simple,
        ),
    ]


def iteration_profiles() -> list[Profile]:
    """Expose the substep dependence of bounded FrameTask errors."""
    profiles = [current_profile("current")]
    for max_iters in (1, 2, 10):
        profiles.append(
            current_profile(
                f"native_3mm_iters_{max_iters}",
                max_iters=max_iters,
            )
        )
    for max_iters in (1, 2, 5, 10):
        profiles.append(
            current_profile(
                f"always_3mm_iters_{max_iters}",
                max_iters=max_iters,
                error_schedule_override="always",
            )
        )
    for max_iters in (1, 2, 5, 10):
        profiles.append(
            current_profile(
                f"fixed_15mm_total_iters_{max_iters}",
                max_iters=max_iters,
                frame_position_error_limit=0.015 / max_iters,
                error_schedule_override="always",
            )
        )
    for max_iters in (1, 2, 5, 10):
        profiles.append(
            current_profile(
                f"no_error_limit_iters_{max_iters}",
                max_iters=max_iters,
                frame_position_error_limit=0.0,
            )
        )
    return profiles


def iteration_scenarios() -> list[Scenario]:
    """Return a compact set spanning lag, singularity, and wrist rotation."""
    factory = PoseFactory()
    return [
        make_reach_scenario(factory, speed=0.8),
        make_reach_scenario(factory, speed=0.4, lateral=-0.10),
        make_diagonal_retract_scenario(factory, speed=0.8, lateral=-0.10),
        make_diagonal_retract_scenario(factory, speed=0.8, lateral=0.10),
        make_extended_circle_scenario(factory, speed=0.8),
        make_wrist_flip_scenario(
            factory,
            angular_speed=10.0,
            extended=True,
        ),
        make_normal_lissajous_scenario(factory, speed=0.8, mode="right"),
    ]


def make_profile(
    name: str,
    *,
    limit_style: str = "recoverable",
    velocity_caps: tuple[float, ...] | None = CONTROL_CAPS,
    driver_velocity_caps: tuple[float, ...] | None = DRIVER_CAPS,
    use_measured_state: bool = True,
    gravity_compensation: bool = False,
    state_delay_s: float = 0.0,
    command_delay_s: float = 0.0,
    state_rate_hz: float = 250.0,
    state_dropout_start_s: float = -1.0,
    state_dropout_duration_s: float = 0.0,
    actuator_kp_scale: float = 1.0,
    actuator_kv_scale: float = 1.0,
    description: str = "",
    **overrides: Any,
) -> Profile:
    """Build one current-API benchmark profile."""
    return Profile(
        name=name,
        overrides=overrides,
        velocity_caps=velocity_caps,
        limit_style=limit_style,
        driver_velocity_caps=driver_velocity_caps,
        use_measured_state=use_measured_state,
        gravity_compensation=gravity_compensation,
        state_delay_s=state_delay_s,
        command_delay_s=command_delay_s,
        state_rate_hz=state_rate_hz,
        state_dropout_start_s=state_dropout_start_s,
        state_dropout_duration_s=state_dropout_duration_s,
        actuator_kp_scale=actuator_kp_scale,
        actuator_kv_scale=actuator_kv_scale,
        description=description,
    )


def deployment_profile(
    name: str = "current_deployment",
    **overrides: Any,
) -> Profile:
    """Return the PR default profile with optional isolated changes."""
    description = overrides.pop(
        "description",
        (
            "PR default experiment profile: "
            "bounded 6D frame error, recoverable joint limits and braking, "
            "exact-nullspace home regulation, singularity-approach limiting, "
            "and weak kinetic-energy regularization."
        ),
    )
    return make_profile(
        name,
        description=description,
        **overrides,
    )


def strict_mainline_profile(name: str = "strict_mainline") -> Profile:
    """Use mainline baseline tasks with the shared joint envelope and substep timing."""
    return make_profile(
        name,
        use_measured_state=False,
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=0.01,
        damping=0.25,
        posture_cost=0.01,
        frame_position_error_limit=0.0,
        frame_orientation_error_limit=0.0,
        joint_braking=False,
        nullspace_cost=0.0,
        singularity_max_approach_rate=0.0,
        kinetic_energy_cost=0.0,
        description=(
            "Mainline baseline task parameters, evaluated with the PR "
            "recoverable position/velocity envelope and common 0.8 ms substeps."
        ),
    )


def no_branch_regulation_profile(
    name: str = "no_branch_regulation",
) -> Profile:
    """Disable both exact-nullspace and full-home branch regularization."""
    return deployment_profile(
        name,
        nullspace_cost=0.0,
        posture_cost=0.0,
        description=(
            "PR default profile with both exact-nullspace and "
            "full-home branch regulation disabled."
        ),
    )


def full_home_replacement_profile(
    name: str = "full_home_replacement_0p01",
    *,
    posture_cost: float = 0.01,
) -> Profile:
    """Replace exact-nullspace regulation with a full-joint home task."""
    return deployment_profile(
        name,
        nullspace_cost=0.0,
        posture_cost=posture_cost,
        description=(
            "PR default mechanisms with exact-nullspace regulation "
            f"replaced by a full-home PostureTask at cost {posture_cost:g}."
        ),
    )


def upstream_style_profile(name: str = "upstream_style") -> Profile:
    """Emulate upstream main's task set with corrected 4 ms outer timing."""
    return make_profile(
        name,
        limit_style="configuration_only",
        velocity_caps=None,
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=0.01,
        damping=0.25,
        posture_cost=0.01,
        frame_position_error_limit=0.0,
        frame_orientation_error_limit=0.0,
        nullspace_cost=0.0,
        singularity_max_approach_rate=0.0,
        kinetic_energy_cost=0.0,
        description=(
            "Upstream task set: full-home posture, scalar damping, and "
            "configuration position limits. Uses corrected outer-period timing."
        ),
    )


def upstream_style_ik_velocity_profile(
    name: str = "upstream_style_ik_velocity",
) -> Profile:
    """Add only the current IK velocity envelope to the main-style tasks."""
    base = upstream_style_profile(name)
    return replace(
        base,
        limit_style="recoverable",
        velocity_caps=CONTROL_CAPS,
        overrides={**base.overrides, "joint_braking": False},
        description=(
            "Upstream task set with the current recoverable IK velocity "
            "envelope. Joint braking remains disabled to isolate velocity caps."
        ),
    )


def final_screening_profiles() -> list[Profile]:
    """Profiles for broad current-revision feature attribution."""
    return [
        deployment_profile(),
        strict_mainline_profile(),
        no_branch_regulation_profile(),
        full_home_replacement_profile(),
        deployment_profile(
            "no_singularity_limit",
            singularity_max_approach_rate=0.0,
        ),
        deployment_profile("no_joint_braking", joint_braking=False),
        deployment_profile(
            "no_frame_error_bound",
            frame_position_error_limit=0.0,
            frame_orientation_error_limit=0.0,
        ),
        deployment_profile(
            "no_position_error_bound",
            frame_position_error_limit=0.0,
        ),
        deployment_profile(
            "no_orientation_error_bound",
            frame_orientation_error_limit=0.0,
        ),
        deployment_profile(
            "no_kinetic_regularization",
            kinetic_energy_cost=0.0,
        ),
        deployment_profile(
            "standard_position_velocity",
            limit_style="standard",
            joint_braking=False,
            description="Native configuration plus velocity limits.",
        ),
        deployment_profile(
            "driver_only_velocity",
            limit_style="configuration_only",
            velocity_caps=None,
            description="PR tasks with velocity limiting only in the driver.",
        ),
        deployment_profile(
            "ik_only_velocity",
            driver_velocity_caps=None,
            description="IK velocity/braking without downstream driver clipping.",
        ),
        deployment_profile(
            "no_velocity_limits",
            limit_style="configuration_only",
            velocity_caps=None,
            driver_velocity_caps=None,
            description="Unsafe diagnostic reference with no velocity envelope.",
        ),
    ]


def final_parameter_profiles() -> list[Profile]:
    """One-factor sweeps around the PR default."""
    profiles: list[Profile] = [deployment_profile()]
    for position_budget in (0.0, 0.010, 0.015, 0.020, 0.025, 0.030):
        profiles.append(
            deployment_profile(
                f"position_budget_{position_budget:g}".replace(".", "p"),
                frame_position_error_limit=position_budget,
            )
        )
    for orientation_budget in (0.0, 0.15, 0.20, 0.25, 0.30, 0.40):
        profiles.append(
            deployment_profile(
                f"orientation_budget_{orientation_budget:g}".replace(".", "p"),
                frame_orientation_error_limit=orientation_budget,
            )
        )
    for cost in (0.0, 3.0, 7.0, 8.5, 12.0, 18.0):
        profiles.append(
            deployment_profile(
                f"null_cost_{cost:g}".replace(".", "p"),
                nullspace_cost=cost,
            )
        )
    for rate in (0.6, 1.0, 1.6, 2.4, 3.2):
        profiles.append(
            deployment_profile(
                f"null_rate_{rate:g}".replace(".", "p"),
                nullspace_return_rate=rate,
            )
        )
    for rate in (0.0, 0.10, 0.18, 0.25, 0.35, 0.50):
        profiles.append(
            deployment_profile(
                f"sing_rate_{rate:g}".replace(".", "p"),
                singularity_max_approach_rate=rate,
            )
        )
    for distance in (0.08, 0.12, 0.20, 0.30, 0.50):
        profiles.append(
            deployment_profile(
                f"brake_distance_{distance:g}".replace(".", "p"),
                joint_braking_distance=distance,
            )
        )
    for cost in (0.0, 1e-5, 2e-5, 3e-5, 5e-5, 1e-4):
        profiles.append(
            deployment_profile(
                f"energy_{cost:g}".replace(".", "p").replace("-", "m"),
                kinetic_energy_cost=cost,
            )
        )
    for scale in (0.75, 1.0, 1.25, 1.5):
        profiles.append(
            deployment_profile(
                f"control_caps_x{scale:g}".replace(".", "p"),
                velocity_caps=tuple(scale * value for value in CONTROL_CAPS),
            )
        )
    for position_cost, orientation_cost in (
        (8.0, 1.0),
        (10.0, 1.0),
        (12.0, 1.5),
        (15.0, 1.5),
        (12.0, 2.0),
    ):
        profiles.append(
            deployment_profile(
                f"task_cost_{position_cost:g}_{orientation_cost:g}".replace(
                    ".",
                    "p",
                ),
                position_cost=position_cost,
                orientation_cost=orientation_cost,
            )
        )
    profiles.append(no_branch_regulation_profile())
    for posture_cost in (0.003, 0.01, 0.03):
        profiles.append(
            full_home_replacement_profile(
                f"full_home_replacement_{posture_cost:g}".replace(".", "p"),
                posture_cost=posture_cost,
            )
        )
    return list({profile.name: profile for profile in profiles}.values())


def driver_coupling_profiles() -> list[Profile]:
    """Compare every practical IK/driver velocity-limit placement."""
    return [
        deployment_profile(),
        deployment_profile(
            "driver_only_velocity",
            limit_style="configuration_only",
            velocity_caps=None,
        ),
        deployment_profile(
            "ik_only_velocity",
            driver_velocity_caps=None,
        ),
        deployment_profile(
            "no_velocity_limits",
            limit_style="configuration_only",
            velocity_caps=None,
            driver_velocity_caps=None,
        ),
    ]


def current_symmetry_profiles() -> list[Profile]:
    return [
        deployment_profile(),
        strict_mainline_profile(),
    ]


class DynamicPlant:
    """Independent MuJoCo plant driven by position actuator references."""

    def __init__(
        self,
        initial_right: np.ndarray,
        initial_left: np.ndarray,
        *,
        gravity_compensation: bool,
        actuator_kp_scale: float = 1.0,
        actuator_kv_scale: float = 1.0,
    ) -> None:
        for name, value in (
            ("actuator_kp_scale", actuator_kp_scale),
            ("actuator_kv_scale", actuator_kv_scale),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        self.setup = make_setup("bimanual")
        self.model = self.setup.model
        self.data = self.setup.data
        self.resolver = self.setup.joint_resolver
        self.gravity_compensation = gravity_compensation
        self._gravity_data = mujoco.MjData(self.model)
        self._qpos_by_side = {
            side: np.asarray(
                (self.resolver._right if side == "right" else self.resolver._left).arm_qpos,
                dtype=int,
            )
            for side in SIDES
        }
        self._dofs_by_side = {
            side: np.asarray(
                (self.resolver._right if side == "right" else self.resolver._left).arm_dof,
                dtype=int,
            )
            for side in SIDES
        }
        self._actuators_by_side = {
            side: np.asarray(
                [
                    self.model.actuator(f"{side}_joint{index}_ctrl").id
                    for index in range(1, 8)
                ],
                dtype=int,
            )
            for side in SIDES
        }
        for actuator_ids in self._actuators_by_side.values():
            self.model.actuator_gainprm[actuator_ids, 0] *= actuator_kp_scale
            self.model.actuator_biasprm[actuator_ids, 1] *= actuator_kp_scale
            self.model.actuator_biasprm[actuator_ids, 2] *= actuator_kv_scale
        self._elbow_body = {
            side: mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"openarm_{side}_link4",
            )
            for side in SIDES
        }
        self.resolver.set_qpos(
            self.data.qpos,
            np.append(initial_right, 0.0),
            "right",
        )
        self.resolver.set_qpos(
            self.data.qpos,
            np.append(initial_left, 0.0),
            "left",
        )
        mujoco.mj_forward(self.model, self.data)
        self.set_command({"right": initial_right, "left": initial_left})

    def set_command(self, commands: dict[str, np.ndarray]) -> None:
        target_qpos = self.data.qpos.copy()
        for side, command in commands.items():
            self.resolver.set_qpos(
                target_qpos,
                np.append(np.asarray(command, dtype=np.float64), 0.0),
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

    def _physics_step(self) -> None:
        if self.gravity_compensation:
            self._gravity_data.qpos[:] = self.data.qpos
            self._gravity_data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self._gravity_data)
            self.data.qfrc_applied[:] = 0.0
            for side in SIDES:
                indices = self._dofs_by_side[side]
                self.data.qfrc_applied[indices] = self._gravity_data.qfrc_bias[indices]
        mujoco.mj_step(self.model, self.data)

    def step_control_period(self) -> None:
        count = round(CONTROL_DT / self.model.opt.timestep)
        if not np.isclose(count * self.model.opt.timestep, CONTROL_DT):
            raise ValueError("MuJoCo timestep must divide the control period.")
        for _ in range(count):
            self._physics_step()

    def settle(self, duration: float) -> None:
        for _ in range(round(duration / self.model.opt.timestep)):
            self._physics_step()

    def q(self, side: str) -> np.ndarray:
        return self.data.qpos[self._qpos_by_side[side]].copy()

    def dq(self, side: str) -> np.ndarray:
        return self.data.qvel[self._dofs_by_side[side]].copy()

    def ee(self, side: str) -> np.ndarray:
        return self.setup.read_ee_pose(side).astype(np.float64)

    def elbow(self, side: str) -> np.ndarray:
        return self.data.xpos[self._elbow_body[side]].copy()

    def actuator_force(self, side: str) -> np.ndarray:
        return self.data.actuator_force[self._actuators_by_side[side]].copy()

    def driver_qpos(self) -> np.ndarray:
        values: list[np.ndarray] = []
        for side in SIDES:
            q, gripper = self.resolver.get_driver(self.data.qpos, side)
            values.append(np.append(q, gripper))
        return np.concatenate(values).astype(np.float32)

    def driver_qvel(self) -> np.ndarray:
        return np.concatenate(
            [
                np.append(self.dq("right"), 0.0),
                np.append(self.dq("left"), 0.0),
            ]
        ).astype(np.float32)


class ConfigurationMonitor:
    """Evaluate FK, elbow position, and geometric rho for arbitrary q."""

    def __init__(self, kinematics: Kinematics) -> None:
        solver = kinematics._ik
        assert solver is not None
        self.setup = kinematics.setup
        self.configuration = mink.Configuration(self.setup.model)
        self.base_q = solver._config.q.copy()
        self.frame_tasks = solver._tasks
        self.dof_indices = {
            side: self.setup.joint_resolver.arm_dof_indices(side)
            for side in solver._sides
        }
        self.elbow_body = {
            side: mujoco.mj_name2id(
                self.setup.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"openarm_{side}_link4",
            )
            for side in solver._sides
        }

    def evaluate(
        self,
        commands: dict[str, np.ndarray],
    ) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
        qpos = self.base_q.copy()
        for side, command in commands.items():
            self.setup.joint_resolver.set_qpos(
                qpos,
                np.append(command, 0.0),
                side,
            )
        self.configuration.update(q=qpos)
        self.setup.data.qpos[:] = qpos
        mujoco.mj_forward(self.setup.model, self.setup.data)
        output: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for side in commands:
            jacobian = normalized_arm_jacobian(
                self.frame_tasks[side],
                self.configuration,
                self.dof_indices[side],
                characteristic_length=0.3,
            )
            ratio, _ = singularity_ratio(jacobian)
            output[side] = (
                self.setup.read_ee_pose(side).astype(np.float64),
                self.setup.data.xpos[self.elbow_body[side]].copy(),
                ratio,
            )
        return output

    def ratio(self, side: str, q: np.ndarray) -> float:
        return self.evaluate({side: q})[side][2]


def _limit_driver_command(
    command: np.ndarray,
    previous: np.ndarray,
    caps: tuple[float, ...] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the driver's per-command velocity clamp at the nominal control period."""
    if caps is None:
        return command.copy(), np.zeros(7, dtype=bool)
    limits = np.asarray(caps, dtype=np.float64)
    if limits.shape != (7,):
        raise ValueError("Driver velocity caps must contain seven values.")
    requested_delta = command - previous
    max_delta = limits * CONTROL_DT
    limited_delta = np.clip(requested_delta, -max_delta, max_delta)
    return previous + limited_delta, np.abs(requested_delta) > max_delta + 1e-12


def _frame_task_diagnostics(
    task: mink.Task,
    configuration: mink.Configuration,
) -> tuple[float, float, float, float, float]:
    """Return full/used position and orientation errors plus activation."""
    if not isinstance(task, BoundedFrameTask):
        error = task.compute_error(configuration)
        return (
            float(np.linalg.norm(error[:3])),
            float(np.linalg.norm(error[:3])),
            float(np.linalg.norm(error[3:])),
            float(np.linalg.norm(error[3:])),
            0.0,
        )

    full = task.compute_full_error(configuration)
    limited = task._clip_error(full)
    used = full.copy()
    if task.position_error_limit > 0.0 and task.limit_activation > 0.0:
        used[:3] += task.limit_activation * (limited[:3] - full[:3])
    if task.orientation_error_limit > 0.0:
        used[3:] = limited[3:]
    return (
        float(np.linalg.norm(full[:3])),
        float(np.linalg.norm(used[:3])),
        float(np.linalg.norm(full[3:])),
        float(np.linalg.norm(used[3:])),
        task.limit_activation,
    )


def _nullspace_diagnostics(
    task: mink.Task | None,
    configuration: mink.Configuration,
) -> tuple[float, float, float]:
    """Recompute current nullspace activation, home error, and return speed."""
    if task is None:
        return 0.0, 0.0, 0.0
    jacobian = normalized_arm_jacobian(
        task._frame_task,
        configuration,
        task._dof_indices,
        task._characteristic_length,
        root_is_independent_of_dofs=task._root_is_independent_of_dofs,
    )
    direction, singular_values = structural_nullspace_direction(jacobian)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    ratio = float(singular_values[-1] / largest) if largest > 0.0 else 0.0
    activation = smoothstep_activation(
        ratio,
        task._singularity_low,
        task._singularity_high,
    )
    configuration_error = np.empty(task._model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(
        m=task._model,
        qvel=configuration_error,
        dt=1.0,
        qpos1=task._home_qpos,
        qpos2=configuration.q,
    )
    posture_error = float(
        direction @ configuration_error[task._dof_indices]
    )
    return_speed = float(
        np.clip(
            -task._return_rate * posture_error,
            -task._max_speed,
            task._max_speed,
        )
    )
    return activation, posture_error, return_speed


def _joint_limit_diagnostics(
    limit: ArmJointLimit | None,
    configuration: mink.Configuration,
    side_qpos: np.ndarray,
    command_dq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return braking-envelope fractions and utilization for one arm."""
    fractions = np.ones(7, dtype=np.float64)
    utilization = np.zeros(7, dtype=np.float64)
    if limit is None or limit.braking_distance is None:
        return fractions, utilization

    index_by_qpos = {
        int(qpos): index for index, qpos in enumerate(limit.qpos_indices)
    }
    selected = np.asarray(
        [index_by_qpos[int(qpos)] for qpos in side_qpos],
        dtype=int,
    )
    command_q = configuration.q[limit.qpos_indices]
    lower_distance = command_q - limit.lower
    upper_distance = limit.upper - command_q
    if limit._measured_qpos is not None:
        measured_q = limit._measured_qpos[limit.qpos_indices]
        measured_lower = (
            measured_q - limit.lower - limit.braking_distance_buffer
        )
        measured_upper = (
            limit.upper - measured_q - limit.braking_distance_buffer
        )
        lower_distance = np.minimum(lower_distance, measured_lower)
        upper_distance = np.minimum(upper_distance, measured_upper)
    lower_velocity = _distance_velocity_envelope(
        lower_distance,
        limit.max_velocity,
        limit.braking_distance,
        limit.braking_exponent,
    )
    upper_velocity = _distance_velocity_envelope(
        upper_distance,
        limit.max_velocity,
        limit.braking_distance,
        limit.braking_exponent,
    )
    max_velocity = limit.max_velocity[selected]
    fractions = (
        np.minimum(lower_velocity[selected], upper_velocity[selected])
        / max_velocity
    )
    approach_limit = np.where(
        command_dq >= 0.0,
        upper_velocity[selected],
        lower_velocity[selected],
    )
    utilization = np.abs(command_dq) / np.maximum(approach_limit, 1e-12)
    return fractions, utilization


def _active_sides(mode: str) -> tuple[str, ...]:
    return SIDES if mode == "bimanual" else (mode,)


def simulate(profile: Profile, scenario: Scenario) -> RunTrace:
    """Run one sampled trajectory through Mink and the dynamic plant."""
    kinematics = make_kinematics(profile, scenario.mode)
    solver = kinematics._ik
    assert solver is not None
    monitor = ConfigurationMonitor(kinematics)
    plant = DynamicPlant(
        scenario.initial_right,
        scenario.initial_left,
        gravity_compensation=profile.gravity_compensation,
        actuator_kp_scale=profile.actuator_kp_scale,
        actuator_kv_scale=profile.actuator_kv_scale,
    )
    initial_commands = {
        "right": scenario.initial_right.copy(),
        "left": scenario.initial_left.copy(),
    }
    plant.set_command(initial_commands)
    plant.settle(0.6)
    kinematics.sync(plant.driver_qpos())

    if profile.state_delay_s < 0.0 or profile.command_delay_s < 0.0:
        raise ValueError("Transport delays must be non-negative.")
    if not np.isfinite(profile.state_rate_hz) or profile.state_rate_hz <= 0.0:
        raise ValueError("state_rate_hz must be finite and positive.")
    if profile.state_dropout_duration_s < 0.0:
        raise ValueError("state_dropout_duration_s must be non-negative.")
    state_delay_ticks = round(profile.state_delay_s / CONTROL_DT)
    command_delay_ticks = round(profile.command_delay_s / CONTROL_DT)
    state_period_ticks = max(1, round(1.0 / (profile.state_rate_hz * CONTROL_DT)))
    initial_measured_qpos = plant.driver_qpos()
    measured_history: deque[np.ndarray] = deque(
        (
            initial_measured_qpos.copy()
            for _ in range(state_delay_ticks + 1)
        ),
        maxlen=state_delay_ticks + 1,
    )
    command_history: deque[dict[str, np.ndarray]] = deque()

    active_sides = _active_sides(scenario.mode)
    count = scenario.times.size
    solve_time = np.empty(count)
    solver_failed = np.zeros(count, dtype=bool)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for side in active_sides:
        arrays[side] = {
            "target_pose": np.empty((count, 7)),
            "command_pose": np.empty((count, 7)),
            "driver_pose": np.empty((count, 7)),
            "actual_pose": np.empty((count, 7)),
            "command_q": np.empty((count, 7)),
            "driver_q": np.empty((count, 7)),
            "actual_q": np.empty((count, 7)),
            "command_dq": np.empty((count, 7)),
            "driver_dq": np.empty((count, 7)),
            "actual_dq": np.empty((count, 7)),
            "command_ddq": np.empty((count, 7)),
            "driver_ddq": np.empty((count, 7)),
            "actual_ddq": np.empty((count, 7)),
            "command_elbow": np.empty((count, 3)),
            "driver_elbow": np.empty((count, 3)),
            "actual_elbow": np.empty((count, 3)),
            "command_rho": np.empty(count),
            "driver_rho": np.empty(count),
            "actual_rho": np.empty(count),
            "frame_full_position_error": np.empty(count),
            "frame_used_position_error": np.empty(count),
            "frame_full_orientation_error": np.empty(count),
            "frame_used_orientation_error": np.empty(count),
            "frame_limit_activation": np.zeros(count),
            "nullspace_activation": np.zeros(count),
            "nullspace_error": np.zeros(count),
            "nullspace_return_speed": np.zeros(count),
            "singularity_activation": np.ones(count),
            "singularity_allowed_rate": np.full(count, np.nan),
            "braking_min_fraction": np.ones(count),
            "braking_utilization": np.zeros(count),
            "braking_fraction_by_joint": np.ones((count, 7)),
            "braking_utilization_by_joint": np.zeros((count, 7)),
            "driver_limit_active_by_joint": np.zeros((count, 7), dtype=bool),
            "actuator_force": np.empty((count, 7)),
        }

    previous_command = {side: plant.q(side) for side in active_sides}
    previous_driver = {side: plant.q(side) for side in active_sides}
    previous_command_dq = {side: np.zeros(7) for side in active_sides}
    previous_driver_dq = {side: np.zeros(7) for side in active_sides}
    previous_actual_dq = {side: plant.dq(side) for side in active_sides}
    held_commands = {
        side: (
            previous_driver[side].copy()
            if side in previous_driver
            else initial_commands[side].copy()
        )
        for side in SIDES
    }

    for tick in range(count):
        measured_history.append(plant.driver_qpos())
        delayed_qpos = measured_history[0]
        sim_time = tick * CONTROL_DT
        dropout_active = (
            profile.state_dropout_start_s >= 0.0
            and profile.state_dropout_start_s
            <= sim_time
            < (
                profile.state_dropout_start_s
                + profile.state_dropout_duration_s
            )
        )
        if profile.use_measured_state:
            if dropout_active:
                kinematics.clear_measured_state()
            elif tick % state_period_ticks == 0:
                kinematics.update_measured_state(delayed_qpos)
        for side in active_sides:
            target = (
                scenario.target_right[tick]
                if side == "right"
                else scenario.target_left[tick]
            )
            kinematics.set_target(side, target)
            arrays[side]["target_pose"][tick] = target

        started = time.perf_counter()
        result = kinematics.solve()
        solve_time[tick] = time.perf_counter() - started
        solver_failed[tick] = result is None

        commands: dict[str, np.ndarray] = {}
        driver_commands: dict[str, np.ndarray] = {}
        for side in active_sides:
            offset = SIDE_OUTPUT_OFFSET[side]
            command = (
                previous_command[side].copy()
                if result is None
                else result[offset : offset + 7].astype(np.float64)
            )
            commands[side] = command
            held_commands[side] = command
            command_dq = (command - previous_command[side]) / CONTROL_DT
            arrays[side]["command_q"][tick] = command
            arrays[side]["command_dq"][tick] = command_dq
            arrays[side]["command_ddq"][tick] = (
                command_dq - previous_command_dq[side]
            ) / CONTROL_DT
            driver_command, driver_limited = _limit_driver_command(
                command,
                previous_driver[side],
                profile.driver_velocity_caps,
            )
            driver_dq = (
                driver_command - previous_driver[side]
            ) / CONTROL_DT
            driver_commands[side] = driver_command
            held_commands[side] = driver_command
            arrays[side]["driver_q"][tick] = driver_command
            arrays[side]["driver_dq"][tick] = driver_dq
            arrays[side]["driver_ddq"][tick] = (
                driver_dq - previous_driver_dq[side]
            ) / CONTROL_DT
            arrays[side]["driver_limit_active_by_joint"][tick] = driver_limited

        command_diagnostics = monitor.evaluate(commands)
        driver_diagnostics = monitor.evaluate(driver_commands)
        command_history.append(
            {
                side: command.copy()
                for side, command in held_commands.items()
            }
        )
        if len(command_history) > command_delay_ticks:
            applied_commands = command_history.popleft()
        else:
            applied_commands = initial_commands
        plant.set_command(applied_commands)
        plant.step_control_period()

        for side in active_sides:
            command_pose, command_elbow, command_rho = command_diagnostics[side]
            driver_pose, driver_elbow, driver_rho = driver_diagnostics[side]
            actual_q = plant.q(side)
            actual_dq = plant.dq(side)
            actual_pose = plant.ee(side)
            actual_rho = monitor.ratio(side, actual_q)
            arrays[side]["command_pose"][tick] = command_pose
            arrays[side]["driver_pose"][tick] = driver_pose
            arrays[side]["actual_pose"][tick] = actual_pose
            arrays[side]["actual_q"][tick] = actual_q
            arrays[side]["actual_dq"][tick] = actual_dq
            arrays[side]["actual_ddq"][tick] = (
                actual_dq - previous_actual_dq[side]
            ) / CONTROL_DT
            arrays[side]["command_elbow"][tick] = command_elbow
            arrays[side]["driver_elbow"][tick] = driver_elbow
            arrays[side]["actual_elbow"][tick] = plant.elbow(side)
            arrays[side]["command_rho"][tick] = command_rho
            arrays[side]["driver_rho"][tick] = driver_rho
            arrays[side]["actual_rho"][tick] = actual_rho
            arrays[side]["actuator_force"][tick] = plant.actuator_force(side)

            task = solver._tasks[side]
            (
                arrays[side]["frame_full_position_error"][tick],
                arrays[side]["frame_used_position_error"][tick],
                arrays[side]["frame_full_orientation_error"][tick],
                arrays[side]["frame_used_orientation_error"][tick],
                arrays[side]["frame_limit_activation"][tick],
            ) = _frame_task_diagnostics(task, solver._config)

            (
                arrays[side]["nullspace_activation"][tick],
                arrays[side]["nullspace_error"][tick],
                arrays[side]["nullspace_return_speed"][tick],
            ) = _nullspace_diagnostics(
                solver._nullspace_tasks.get(side),
                solver._config,
            )

            singularity_limit = solver._singularity_limits.get(side)
            if singularity_limit is not None:
                arrays[side]["singularity_activation"][tick] = (
                    singularity_limit._allowed_rate
                    / singularity_limit.max_rate
                )
                arrays[side]["singularity_allowed_rate"][tick] = (
                    singularity_limit._allowed_rate
                )

            fractions, utilization = _joint_limit_diagnostics(
                solver._joint_limit,
                solver._config,
                solver._joint_resolver.arm_qpos_indices(side),
                arrays[side]["command_dq"][tick],
            )
            arrays[side]["braking_fraction_by_joint"][tick] = fractions
            arrays[side]["braking_utilization_by_joint"][tick] = utilization
            arrays[side]["braking_min_fraction"][tick] = float(
                np.min(fractions)
            )
            arrays[side]["braking_utilization"][tick] = float(
                np.max(utilization)
            )

            previous_command[side] = commands[side]
            previous_driver[side] = driver_commands[side]
            previous_command_dq[side] = arrays[side]["command_dq"][tick]
            previous_driver_dq[side] = arrays[side]["driver_dq"][tick]
            previous_actual_dq[side] = actual_dq

    side_traces = {
        side: SideTrace(**side_arrays) for side, side_arrays in arrays.items()
    }
    return RunTrace(
        times=scenario.times.copy(),
        phase=scenario.phase.copy(),
        solve_time=solve_time,
        solver_failed=solver_failed,
        sides=side_traces,
    )


def _orientation_errors(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            quaternion_angle(first[index, 3:], second[index, 3:])
            for index in range(first.shape[0])
        ]
    )


def _derivative(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    result[0] = 0.0
    result[1:] = np.diff(values, axis=0) / CONTROL_DT
    return result


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def _percentile_abs(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.abs(values), percentile)) if values.size else 0.0


def _joint_margins(
    setup: ArmSetup,
    side: str,
    q_values: np.ndarray,
) -> np.ndarray:
    qpos_indices = setup.joint_resolver.arm_qpos_indices(side)
    margins = np.empty_like(q_values)
    for index, qpos_index in enumerate(qpos_indices):
        joint_ids = np.flatnonzero(setup.model.jnt_qposadr == int(qpos_index))
        joint_id = int(joint_ids[0])
        lower, upper = setup.model.jnt_range[joint_id]
        margins[:, index] = np.minimum(
            q_values[:, index] - lower,
            upper - q_values[:, index],
        )
    return margins


def compute_metrics(
    profile: Profile,
    scenario: Scenario,
    trace: RunTrace,
) -> list[dict[str, Any]]:
    """Compute one scalar metric row per active arm."""
    setup = make_setup(scenario.mode)
    active = trace.phase > 0
    if not np.any(active):
        active = np.ones_like(trace.phase, dtype=bool)
    tail_count = max(10, round(0.35 / CONTROL_DT))
    tail = np.zeros_like(active)
    tail[-min(tail_count, tail.size) :] = True
    rows: list[dict[str, Any]] = []

    for side, values in trace.sides.items():
        poses = {
            "command": values.command_pose,
            "driver": values.driver_pose,
            "actual": values.actual_pose,
        }
        position_errors = {
            layer: np.linalg.norm(
                values.target_pose[:, :3] - pose[:, :3],
                axis=1,
            )
            for layer, pose in poses.items()
        }
        orientation_errors = {
            layer: _orientation_errors(values.target_pose, pose)
            for layer, pose in poses.items()
        }
        elbow_velocities = {
            "command": _derivative(values.command_elbow),
            "driver": _derivative(values.driver_elbow),
            "actual": _derivative(values.actual_elbow),
        }
        elbow_accelerations = {
            layer: _derivative(velocity)
            for layer, velocity in elbow_velocities.items()
        }
        command_margin = _joint_margins(setup, side, values.command_q)
        driver_margin = _joint_margins(setup, side, values.driver_q)
        actual_margin = _joint_margins(setup, side, values.actual_q)
        command_utilization = (
            np.abs(values.command_dq)
            / np.asarray(profile.velocity_caps, dtype=np.float64)
            if profile.velocity_caps is not None
            else np.zeros_like(values.command_dq)
        )
        driver_utilization = (
            np.abs(values.driver_dq)
            / np.asarray(profile.driver_velocity_caps, dtype=np.float64)
            if profile.driver_velocity_caps is not None
            else np.zeros_like(values.driver_dq)
        )
        physical_utilization = np.abs(values.actual_dq) / np.asarray(DRIVER_CAPS)
        command_vs_driver_caps = (
            np.abs(values.command_dq)
            / np.asarray(profile.driver_velocity_caps, dtype=np.float64)
            if profile.driver_velocity_caps is not None
            else np.zeros_like(values.command_dq)
        )
        force_limits = np.array([40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0])
        tail_residuals = {
            layer: pose[tail, :3] - np.mean(pose[tail, :3], axis=0)
            for layer, pose in poses.items()
        }

        row: dict[str, Any] = {
            "profile": profile.name,
            "scenario": scenario.name,
            "family": scenario.family,
            "mode": scenario.mode,
            "side": side,
            "speed": scenario.speed,
            "duration_s": float(trace.times[-1] + CONTROL_DT),
            "solver_failures": int(np.count_nonzero(trace.solver_failed)),
            "solve_time_mean_ms": 1e3 * float(np.mean(trace.solve_time)),
            "solve_time_p95_ms": 1e3 * float(np.percentile(trace.solve_time, 95)),
            "ik_velocity_limit_enabled": profile.velocity_caps is not None,
            "driver_velocity_limit_enabled": (
                profile.driver_velocity_caps is not None
            ),
            "command_position_rmse_m": _rms(position_errors["command"][active]),
            "command_position_p95_m": float(
                np.percentile(position_errors["command"][active], 95)
            ),
            "command_position_max_m": float(
                np.max(position_errors["command"][active])
            ),
            "driver_position_rmse_m": _rms(position_errors["driver"][active]),
            "driver_position_p95_m": float(
                np.percentile(position_errors["driver"][active], 95)
            ),
            "driver_position_max_m": float(
                np.max(position_errors["driver"][active])
            ),
            "actual_position_rmse_m": _rms(position_errors["actual"][active]),
            "actual_position_p95_m": float(
                np.percentile(position_errors["actual"][active], 95)
            ),
            "actual_position_max_m": float(
                np.max(position_errors["actual"][active])
            ),
            "command_orientation_rmse_rad": _rms(
                orientation_errors["command"][active]
            ),
            "driver_orientation_rmse_rad": _rms(
                orientation_errors["driver"][active]
            ),
            "actual_orientation_rmse_rad": _rms(
                orientation_errors["actual"][active]
            ),
            "q_gap_rms_rad": _rms(
                (values.command_q - values.actual_q)[active]
            ),
            "q_gap_max_rad": float(
                np.max(np.abs(values.command_q - values.actual_q)[active])
            ),
            "command_driver_q_gap_rms_rad": _rms(
                (values.command_q - values.driver_q)[active]
            ),
            "command_driver_q_gap_max_rad": float(
                np.max(np.abs(values.command_q - values.driver_q)[active])
            ),
            "driver_actual_q_gap_rms_rad": _rms(
                (values.driver_q - values.actual_q)[active]
            ),
            "driver_actual_q_gap_max_rad": float(
                np.max(np.abs(values.driver_q - values.actual_q)[active])
            ),
            "command_dq_p99_rad_s": _percentile_abs(
                values.command_dq[active],
                99,
            ),
            "command_dq_max_rad_s": float(
                np.max(np.abs(values.command_dq[active]))
            ),
            "driver_dq_p99_rad_s": _percentile_abs(values.driver_dq[active], 99),
            "driver_dq_max_rad_s": float(
                np.max(np.abs(values.driver_dq[active]))
            ),
            "actual_dq_p99_rad_s": _percentile_abs(values.actual_dq[active], 99),
            "actual_dq_max_rad_s": float(np.max(np.abs(values.actual_dq[active]))),
            "command_ddq_p99_rad_s2": _percentile_abs(
                values.command_ddq[active],
                99,
            ),
            "command_ddq_max_rad_s2": float(
                np.max(np.abs(values.command_ddq[active]))
            ),
            "driver_ddq_p99_rad_s2": _percentile_abs(
                values.driver_ddq[active],
                99,
            ),
            "driver_ddq_max_rad_s2": float(
                np.max(np.abs(values.driver_ddq[active]))
            ),
            "actual_ddq_p99_rad_s2": _percentile_abs(
                values.actual_ddq[active],
                99,
            ),
            "actual_ddq_max_rad_s2": float(
                np.max(np.abs(values.actual_ddq[active]))
            ),
            "command_j4_dq_max_rad_s": float(
                np.max(np.abs(values.command_dq[active, 3]))
            ),
            "driver_j4_dq_max_rad_s": float(
                np.max(np.abs(values.driver_dq[active, 3]))
            ),
            "actual_j4_dq_max_rad_s": float(
                np.max(np.abs(values.actual_dq[active, 3]))
            ),
            "command_j4_ddq_p99_rad_s2": _percentile_abs(
                values.command_ddq[active, 3],
                99,
            ),
            "driver_j4_ddq_p99_rad_s2": _percentile_abs(
                values.driver_ddq[active, 3],
                99,
            ),
            "actual_j4_ddq_p99_rad_s2": _percentile_abs(
                values.actual_ddq[active, 3],
                99,
            ),
            "command_elbow_speed_max_m_s": float(
                np.max(
                    np.linalg.norm(elbow_velocities["command"][active], axis=1)
                )
            ),
            "driver_elbow_speed_max_m_s": float(
                np.max(
                    np.linalg.norm(elbow_velocities["driver"][active], axis=1)
                )
            ),
            "actual_elbow_speed_max_m_s": float(
                np.max(
                    np.linalg.norm(elbow_velocities["actual"][active], axis=1)
                )
            ),
            "command_elbow_accel_p99_m_s2": float(
                np.percentile(
                    np.linalg.norm(
                        elbow_accelerations["command"][active],
                        axis=1,
                    ),
                    99,
                )
            ),
            "driver_elbow_accel_p99_m_s2": float(
                np.percentile(
                    np.linalg.norm(
                        elbow_accelerations["driver"][active],
                        axis=1,
                    ),
                    99,
                )
            ),
            "actual_elbow_accel_p99_m_s2": float(
                np.percentile(
                    np.linalg.norm(elbow_accelerations["actual"][active], axis=1),
                    99,
                )
            ),
            "command_elbow_lateral_range_m": float(
                np.ptp(values.command_elbow[active, 1])
            ),
            "driver_elbow_lateral_range_m": float(
                np.ptp(values.driver_elbow[active, 1])
            ),
            "actual_elbow_lateral_range_m": float(
                np.ptp(values.actual_elbow[active, 1])
            ),
            "command_min_rho": float(np.min(values.command_rho[active])),
            "driver_min_rho": float(np.min(values.driver_rho[active])),
            "actual_min_rho": float(np.min(values.actual_rho[active])),
            "command_min_joint_margin_rad": float(np.min(command_margin[active])),
            "driver_min_joint_margin_rad": float(np.min(driver_margin[active])),
            "actual_min_joint_margin_rad": float(np.min(actual_margin[active])),
            "command_velocity_saturation_fraction": float(
                np.mean(np.any(command_utilization[active] >= 0.995, axis=1))
            ),
            "command_above_driver_cap_fraction": float(
                np.mean(
                    np.any(command_vs_driver_caps[active] > 1.0 + 1e-9, axis=1)
                )
            ),
            "driver_velocity_saturation_fraction": float(
                np.mean(np.any(driver_utilization[active] >= 0.995, axis=1))
            ),
            "driver_limit_active_fraction": float(
                np.mean(
                    np.any(
                        values.driver_limit_active_by_joint[active],
                        axis=1,
                    )
                )
            ),
            "actual_physical_velocity_exceed_fraction": float(
                np.mean(np.any(physical_utilization[active] > 1.0, axis=1))
            ),
            "command_j1_saturation_fraction": float(
                np.mean(command_utilization[active, 0] >= 0.995)
            ),
            "command_j4_saturation_fraction": float(
                np.mean(command_utilization[active, 3] >= 0.995)
            ),
            "torque_saturation_fraction": float(
                np.mean(
                    np.any(
                        np.abs(values.actuator_force[active])
                        >= 0.99 * force_limits,
                        axis=1,
                    )
                )
            ),
            "frame_full_error_p95_m": float(
                np.percentile(values.frame_full_position_error[active], 95)
            ),
            "frame_used_error_p95_m": float(
                np.percentile(values.frame_used_position_error[active], 95)
            ),
            "frame_full_orientation_error_p95_rad": float(
                np.percentile(
                    values.frame_full_orientation_error[active],
                    95,
                )
            ),
            "frame_used_orientation_error_p95_rad": float(
                np.percentile(
                    values.frame_used_orientation_error[active],
                    95,
                )
            ),
            "frame_limit_activation_mean": float(
                np.mean(values.frame_limit_activation[active])
            ),
            "nullspace_activation_mean": float(
                np.mean(values.nullspace_activation[active])
            ),
            "nullspace_error_abs_p95_rad": float(
                np.percentile(np.abs(values.nullspace_error[active]), 95)
            ),
            "nullspace_return_speed_abs_p95_rad_s": float(
                np.percentile(np.abs(values.nullspace_return_speed[active]), 95)
            ),
            "singularity_limit_active_fraction": float(
                np.mean(values.singularity_activation[active] < 0.999)
            ),
            "braking_active_fraction": float(
                np.mean(values.braking_min_fraction[active] < 0.999)
            ),
            "braking_binding_fraction": float(
                np.mean(values.braking_utilization[active] >= 0.98)
            ),
            "tail_actual_ee_p2p_m": float(
                np.max(np.ptp(values.actual_pose[tail, :3], axis=0))
            ),
            "tail_driver_ee_p2p_m": float(
                np.max(np.ptp(values.driver_pose[tail, :3], axis=0))
            ),
            "tail_command_ee_p2p_m": float(
                np.max(np.ptp(values.command_pose[tail, :3], axis=0))
            ),
            "tail_actual_ee_vibration_rms_m": _rms(
                tail_residuals["actual"]
            ),
            "tail_driver_ee_vibration_rms_m": _rms(
                tail_residuals["driver"]
            ),
            "tail_command_ee_vibration_rms_m": _rms(
                tail_residuals["command"]
            ),
            "tail_actual_ddq_rms_rad_s2": _rms(values.actual_ddq[tail]),
        }
        for index in range(7):
            row[f"command_j{index + 1}_dq_max_rad_s"] = float(
                np.max(np.abs(values.command_dq[active, index]))
            )
            row[f"command_j{index + 1}_sat_fraction"] = float(
                np.mean(command_utilization[active, index] >= 0.995)
            )
            row[f"driver_j{index + 1}_dq_max_rad_s"] = float(
                np.max(np.abs(values.driver_dq[active, index]))
            )
            row[f"driver_j{index + 1}_limit_active_fraction"] = float(
                np.mean(values.driver_limit_active_by_joint[active, index])
            )
            row[f"actual_j{index + 1}_dq_max_rad_s"] = float(
                np.max(np.abs(values.actual_dq[active, index]))
            )
            row[f"actual_j{index + 1}_ddq_p99_rad_s2"] = _percentile_abs(
                values.actual_ddq[active, index],
                99,
            )
            row[f"braking_j{index + 1}_active_fraction"] = float(
                np.mean(values.braking_fraction_by_joint[active, index] < 0.999)
            )
            row[f"braking_j{index + 1}_binding_fraction"] = float(
                np.mean(values.braking_utilization_by_joint[active, index] >= 0.98)
            )
        rows.append(row)
    return rows


def _jsonable_profile(profile: Profile) -> dict[str, Any]:
    values = asdict(profile)
    values["velocity_caps"] = (
        None if profile.velocity_caps is None else list(profile.velocity_caps)
    )
    values["driver_velocity_caps"] = (
        None
        if profile.driver_velocity_caps is None
        else list(profile.driver_velocity_caps)
    )
    return values


def _scenario_metadata(scenario: Scenario) -> dict[str, Any]:
    digest = hashlib.sha256()
    for array in (
        scenario.times,
        scenario.phase,
        scenario.target_right,
        scenario.target_left,
        scenario.initial_right,
        scenario.initial_left,
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return {
        "name": scenario.name,
        "family": scenario.family,
        "mode": scenario.mode,
        "speed": scenario.speed,
        "sample_count": int(scenario.times.size),
        "description": scenario.description,
        "initial_right": scenario.initial_right.tolist(),
        "initial_left": scenario.initial_left.tolist(),
        "trajectory_sha256": digest.hexdigest(),
    }


def _package_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _scenario_inventory(scenarios: list[Scenario]) -> dict[str, Any]:
    family_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    for scenario in scenarios:
        family_counts[scenario.family] = family_counts.get(scenario.family, 0) + 1
        mode_counts[scenario.mode] = mode_counts.get(scenario.mode, 0) + 1
    return {
        "independent_target_trajectories": len(scenarios),
        "family_counts": dict(sorted(family_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "speed_values": sorted({float(scenario.speed) for scenario in scenarios}),
    }


def _run_key(profile: Profile, scenario: Scenario) -> str:
    encoded = json.dumps(
        {
            "profile": _jsonable_profile(profile),
            "scenario": _scenario_metadata(scenario),
            "revision": _git_revision(),
            "worktree": _git_worktree_state(),
            "study_revision": _study_revision(),
            "model": str(openarm_mujoco.openarm_cell_xml()),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def save_trace(path: Path, scenario: Scenario, trace: RunTrace) -> None:
    """Save a compressed representative trace."""
    payload: dict[str, np.ndarray] = {
        "times": trace.times,
        "phase": trace.phase,
        "solve_time": trace.solve_time,
        "solver_failed": trace.solver_failed,
    }
    for side, values in trace.sides.items():
        for key, array in asdict(values).items():
            payload[f"{side}_{key}"] = array
        payload[f"{side}_target"] = (
            scenario.target_right if side == "right" else scenario.target_left
        )
    np.savez_compressed(path, **payload)


def run_cached(
    profile: Profile,
    scenario: Scenario,
    output_dir: Path,
    *,
    save_full_trace: bool,
) -> list[dict[str, Any]]:
    """Run or load one cached profile/scenario pair."""
    run_dir = output_dir / "runs"
    trace_dir = output_dir / "traces"
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    key = _run_key(profile, scenario)
    result_path = run_dir / f"{key}.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))["metrics"]

    trace = simulate(profile, scenario)
    metrics = compute_metrics(profile, scenario, trace)
    result = {
        "key": key,
        "revision": _git_revision(),
        "study_revision": _study_revision(),
        "profile": _jsonable_profile(profile),
        "scenario": _scenario_metadata(scenario),
        "metrics": metrics,
    }
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary, result_path)
    if save_full_trace:
        save_trace(trace_dir / f"{key}_{profile.name}_{scenario.name}.npz", scenario, trace)
    return metrics


def _worker(payload: tuple[Profile, Scenario, str, bool]) -> list[dict[str, Any]]:
    profile, scenario, output_dir, save_full_trace = payload
    return run_cached(
        profile,
        scenario,
        Path(output_dir),
        save_full_trace=save_full_trace,
    )


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_matrix(
    profiles: list[Profile],
    scenarios: list[Scenario],
    output_dir: Path,
    *,
    workers: int,
    save_all_traces: bool = False,
) -> list[dict[str, Any]]:
    """Run a resumable experiment matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_xml = Path(openarm_mujoco.openarm_cell_xml())
    metadata = {
        "revision": _git_revision(),
        "worktree": _git_worktree_state(),
        "study_revision": _study_revision(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": ["python", *sys.argv],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "mujoco": _package_version("mujoco"),
            "mink": _package_version("mink"),
            "daqp": _package_version("daqp"),
            "openarm_mujoco": _package_version("openarm-mujoco"),
        },
        "sampling": {
            "random_sampling": False,
            "random_seed": 0,
        },
        "control_dt": CONTROL_DT,
        "model_xml": f"openarm_mujoco/{model_xml.parent.name}/{model_xml.name}",
        "model_xml_sha256": hashlib.sha256(model_xml.read_bytes()).hexdigest(),
        "current_deployment_parameters": current_parameter_values(),
        "driver_config": driver_config_values(),
        "matrix": {
            "profile_count": len(profiles),
            "scenario_count": len(scenarios),
            "controller_trajectory_runs": len(profiles) * len(scenarios),
        },
        "scenario_inventory": _scenario_inventory(scenarios),
        "profiles": [_jsonable_profile(profile) for profile in profiles],
        "scenarios": [_scenario_metadata(scenario) for scenario in scenarios],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    representative_profiles = {
        "current_deployment",
        "strict_mainline",
        "no_branch_regulation",
        "full_home_replacement_0p01",
        "no_singularity_limit",
        "no_joint_braking",
        "no_frame_error_bound",
        "no_orientation_error_bound",
        "no_kinetic_regularization",
        "driver_only_velocity",
        "standard_position_velocity",
    }
    representative_families = {
        "reach",
        "retract",
        "extended_translation",
        "extended_axis",
        "extended_wrist",
        "normal_wrist",
        "normal_workspace",
        "chest_wrist_outward",
    }
    jobs = [
        (
            profile,
            scenario,
            str(output_dir),
            save_all_traces
            or (
                profile.name in representative_profiles
                and scenario.family in representative_families
                and scenario.speed >= 0.4
            ),
        )
        for profile in profiles
        for scenario in scenarios
    ]
    rows: list[dict[str, Any]] = []
    total = len(jobs)
    if workers <= 1:
        for index, payload in enumerate(jobs, start=1):
            profile, scenario, _, _ = payload
            print(
                f"[{index}/{total}] {profile.name} :: {scenario.name}",
                flush=True,
            )
            rows.extend(_worker(payload))
            write_summary(output_dir / "summary.csv", rows)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(_worker, payload): (payload[0], payload[1])
                for payload in jobs
            }
            for index, future in enumerate(as_completed(future_map), start=1):
                profile, scenario = future_map[future]
                rows.extend(future.result())
                print(
                    f"[{index}/{total}] {profile.name} :: {scenario.name}",
                    flush=True,
                )
                if index % max(1, workers) == 0:
                    write_summary(output_dir / "summary.csv", rows)
    write_summary(output_dir / "summary.csv", rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=(
            "screening",
            "parameters",
            "driver",
            "symmetry",
            "all",
        ),
        default="screening",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/results/development"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2) // 2)),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only two profiles and two scenarios.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_dir.resolve()
    if args.smoke:
        scenarios = focused_scenarios()[:2]
        profiles = final_screening_profiles()[:2]
        run_matrix(
            profiles,
            scenarios,
            output_root / "smoke",
            workers=args.workers,
        )
        return

    if args.suite in ("screening", "all"):
        run_matrix(
            final_screening_profiles(),
            screening_scenarios(),
            output_root / "screening",
            workers=args.workers,
        )
    if args.suite in ("parameters", "all"):
        run_matrix(
            final_parameter_profiles(),
            focused_scenarios(),
            output_root / "parameters",
            workers=args.workers,
        )
    if args.suite in ("driver", "all"):
        run_matrix(
            driver_coupling_profiles(),
            focused_scenarios(),
            output_root / "driver",
            workers=args.workers,
            save_all_traces=True,
        )
    if args.suite in ("symmetry", "all"):
        run_matrix(
            current_symmetry_profiles(),
            symmetry_scenarios(),
            output_root / "symmetry",
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
