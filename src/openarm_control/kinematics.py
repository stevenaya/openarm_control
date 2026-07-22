# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""High-level FK + IK interface for OpenArm.

Usage:
    # FK only
    kin = Kinematics(setup)
    pose = kin.fk("right", joints)          # float32[7]
    pose_r, pose_l = kin.fk_bimanual(r, l)  # single mj_forward

    # FK + IK
    kin = Kinematics(setup, IKParams(dt=1.0 / 250.0, max_iters=10))
    kin.set_target("right", pose)
    kin.set_target("left", pose)
    result = kin.solve()                    # float32[16] or None
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass

import mink
import mink.exceptions
import mujoco
import numpy as np
import yaml

from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S, ArmSetup
from openarm_control.lower_bound_braking_limit import LowerBoundBrakingLimit
from openarm_control.nullspace_posture_task import NullspacePostureTask
from openarm_control.poses import pose_to_se3
from openarm_control.recoverable_configuration_limit import (
    RecoverableConfigurationLimit,
)
from openarm_control.soft_limit_task import SoftLimitTask


@dataclass
class IKParams:
    """Configuration for the mink QP-based IK solver."""

    position_cost: float = 1.0
    orientation_cost: float = 1.0
    lm_damping: float = 0.01
    damping: float = 0.25
    solver: str = "daqp"
    posture_cost: float = 0.01
    diag_reg: float = 0.0
    dt: float = 1.0 / 250.0
    max_iters: int = 10
    velocity_limits: dict[str, float] | None = None
    joint_limit_recovery_velocity_scale: float = 1.1
    nullspace_cost: float = 0.3
    nullspace_return_rate: float = 0.5
    nullspace_max_speed: float = 0.5
    nullspace_singularity_low: float = 0.02
    nullspace_singularity_high: float = 0.05
    nullspace_characteristic_length: float = 0.3
    elbow_soft_limit_cost: float = 0.0
    elbow_soft_limit_angle: float = 0.08
    elbow_soft_limit_max_speed: float = 0.2
    elbow_braking_guard_angle: float = 0.08
    elbow_braking_acceleration: float = 0.0


class Kinematics:
    """Unified FK + IK for OpenArm, backed by MuJoCo + mink.

    FK is always available. IK is enabled by passing ``IKParams``.
    Both share the same ``ArmSetup`` context (model, resolver, frame IDs).
    """

    def __init__(self, setup: ArmSetup, ik_params: IKParams | None = None) -> None:
        """Initialize."""
        self.setup = setup
        self._ik: _IKSolver | None = (
            _IKSolver(setup, ik_params) if ik_params is not None else None
        )

    # ── FK ───────────────────────────────────────────────────────────────────

    def fk(self, side: str, joints: np.ndarray) -> np.ndarray:
        """Set qpos for one arm, run mj_forward, return float32[7] EE pose."""
        self.setup.joint_resolver.set_qpos(self.setup.data.qpos, joints, side)
        mujoco.mj_forward(self.setup.model, self.setup.data)
        return self.setup.read_ee_pose(side)

    def fk_bimanual(
        self, right: np.ndarray, left: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Set both arms and run a single mj_forward. Returns (pose_right, pose_left)."""
        self.setup.joint_resolver.set_qpos(self.setup.data.qpos, right, "right")
        self.setup.joint_resolver.set_qpos(self.setup.data.qpos, left, "left")
        mujoco.mj_forward(self.setup.model, self.setup.data)
        return self.setup.read_ee_pose("right"), self.setup.read_ee_pose("left")

    # ── IK ───────────────────────────────────────────────────────────────────

    def set_target(self, side: str, pose: np.ndarray) -> None:
        """Set EE target for one arm. pose: float32[7] = [px, py, pz, qw, qx, qy, qz]."""
        self._require_ik().set_target(side, pose)

    def sync(self, values16: np.ndarray) -> None:
        """Sync IK internal config from float32[16] driver state (right[8]+left[8])."""
        self._require_ik().sync(values16)

    def blend_state(
        self,
        qpos16: np.ndarray,
        qvel16: np.ndarray,
        blend: float,
        prediction_dt: float = 0.0,
    ) -> None:
        """Blend predicted measured arm state into the persistent IK state."""
        self._require_ik().blend_state(qpos16, qvel16, blend, prediction_dt)

    def ready(self) -> bool:
        """Return True once all active arms have received at least one target this cycle."""
        return self._require_ik().ready()

    def solve(self) -> np.ndarray | None:
        """Run IK. Returns float32[16] (right[8]+left[8]) or None on failure."""
        return self._require_ik().solve()

    def set_gripper(self, side: str, value: float) -> None:
        """Pass through a gripper value; IK does not solve for it."""
        idx = 0 if side == "right" else 1
        self._require_ik()._gripper[idx] = value

    def _require_ik(self) -> _IKSolver:
        if self._ik is None:
            raise RuntimeError("Kinematics was not initialized with IKParams.")
        return self._ik


# ── internal IK implementation ────────────────────────────────────────────────


class _IKSolver:
    """mink QP-based differential IK. Managed by Kinematics; not public API."""

    def __init__(self, setup: ArmSetup, params: IKParams) -> None:
        if params.dt <= 0.0:
            raise ValueError("IK control timestep must be positive.")
        if params.max_iters <= 0:
            raise ValueError("IK max_iters must be positive.")

        self._sides = setup.sides
        self._solver_name = params.solver
        self._posture_cost = params.posture_cost
        self._joint_resolver = setup.joint_resolver
        self._arm_dofs = {
            side: _dof_indices_for_qpos(
                setup.model, _arm_qpos_indices(setup, side)
            )
            for side in setup.sides
        }
        self._substep_dt = params.dt / params.max_iters
        self._max_iters = params.max_iters

        self._config = mink.Configuration(setup.model)
        self._config.update(q=setup.data.qpos.copy())
        home_qpos = self._config.data.qpos.copy()

        task_kwargs = dict(
            position_cost=params.position_cost,
            orientation_cost=params.orientation_cost,
            lm_damping=params.lm_damping,
        )
        self._tasks: dict[str, mink.FrameTask] = {
            side: mink.FrameTask(
                frame_name=_frame_name(setup, side),
                frame_type=setup.frame_types[side],
                **task_kwargs,
            )
            for side in setup.sides
        }

        active_qpos: set[int] = set(
            setup.joint_resolver._right.arm_qpos.tolist()
        ) | set(setup.joint_resolver._left.arm_qpos.tolist())
        freeze_dofs = [
            int(setup.model.jnt_dofadr[j])
            for j in range(setup.model.njnt)
            if setup.model.jnt_qposadr[j] not in active_qpos
        ]
        self._freeze_task: mink.DofFreezingTask | None = (
            mink.DofFreezingTask(model=setup.model, dof_indices=freeze_dofs)
            if freeze_dofs
            else None
        )

        # Only constrain the DoFs that IK can move. Frozen DoFs such as the
        # gripper and lifter may be valid in driver space but outside the MuJoCo
        # model range; constraining and freezing them at the same time makes the
        # QP infeasible.
        if params.velocity_limits is not None:
            self._limits = [
                RecoverableConfigurationLimit(
                    model=setup.model,
                    qpos_indices=active_qpos,
                    velocities=params.velocity_limits,
                    recovery_velocity_scale=(
                        params.joint_limit_recovery_velocity_scale
                    ),
                )
            ]
        else:
            self._limits = [_configuration_limit_for_qpos(setup.model, active_qpos)]

        self._elbow_braking_limits: dict[str, LowerBoundBrakingLimit] = {}
        if params.elbow_braking_acceleration < 0.0:
            raise ValueError("Elbow braking acceleration must be non-negative.")
        if params.elbow_braking_acceleration > 0.0:
            for side in setup.sides:
                elbow_qpos, elbow_dof = _elbow_joint_indices(setup, side)
                joint_name = f"openarm_{side}_joint4"
                max_velocity = _joint_velocity_cap(
                    params.velocity_limits,
                    joint_name,
                    ARM_JOINT_VELOCITY_LIMITS_RAD_S[3],
                )
                braking_limit = LowerBoundBrakingLimit(
                    model=setup.model,
                    joint_qpos_index=elbow_qpos,
                    joint_dof_index=elbow_dof,
                    guard_position=params.elbow_braking_guard_angle,
                    max_deceleration=params.elbow_braking_acceleration,
                    max_velocity=max_velocity,
                )
                self._elbow_braking_limits[side] = braking_limit
                self._limits.append(braking_limit)

        self._posture_task = mink.PostureTask(setup.model, cost=params.posture_cost)
        self._posture_task.set_target(home_qpos)

        self._nullspace_tasks: dict[str, NullspacePostureTask] = {}
        if params.nullspace_cost > 0.0:
            for side in setup.sides:
                arm_qpos = _arm_qpos_indices(setup, side)
                self._nullspace_tasks[side] = NullspacePostureTask(
                    model=setup.model,
                    frame_task=self._tasks[side],
                    dof_indices=_dof_indices_for_qpos(setup.model, arm_qpos),
                    home_qpos=home_qpos,
                    cost=params.nullspace_cost,
                    dt=self._substep_dt,
                    return_rate=params.nullspace_return_rate,
                    max_speed=params.nullspace_max_speed,
                    singularity_low=params.nullspace_singularity_low,
                    singularity_high=params.nullspace_singularity_high,
                    characteristic_length=params.nullspace_characteristic_length,
                )

        self._elbow_soft_limit_tasks: dict[str, SoftLimitTask] = {}
        if params.elbow_soft_limit_cost > 0.0:
            for side in setup.sides:
                elbow_qpos, elbow_dof = _elbow_joint_indices(setup, side)
                self._elbow_soft_limit_tasks[side] = SoftLimitTask(
                    model=setup.model,
                    joint_qpos_index=elbow_qpos,
                    joint_dof_index=elbow_dof,
                    cost=params.elbow_soft_limit_cost,
                    dt=self._substep_dt,
                    limit=params.elbow_soft_limit_angle,
                    max_speed=params.elbow_soft_limit_max_speed,
                )

        self._solver_params: dict = {"damping": params.damping}
        if params.diag_reg > 0.0:
            self._solver_params["diag_reg"] = params.diag_reg

        self._pending: set[str] = set(setup.sides)
        self._gripper = np.zeros(2, dtype=np.float32)

    def set_target(self, side: str, pose: np.ndarray) -> None:
        self._tasks[side].set_target(pose_to_se3(pose))
        self._pending.discard(side)

    def sync(self, values16: np.ndarray) -> None:
        qpos = self._config.data.qpos.copy()
        self._joint_resolver.set_qpos(qpos, values16[:8], "right")
        self._joint_resolver.set_qpos(qpos, values16[8:16], "left")
        self._config.update(q=qpos)
        # Gripper is intentionally NOT synced here. set_gripper() is the sole
        # writer of self._gripper ("IK does not solve for it"); syncing it
        # from the raw driver state here would race with set_gripper() calls
        # (e.g. from a VR trigger) arriving on a similar cadence, causing the
        # commanded gripper to flicker between the real motor position and
        # the trigger-commanded value depending on event arrival order.

    def blend_state(
        self,
        qpos16: np.ndarray,
        qvel16: np.ndarray,
        blend: float,
        prediction_dt: float,
    ) -> None:
        """Correct the IK state toward a short-horizon measured-state estimate."""
        qpos16 = np.asarray(qpos16, dtype=float)
        qvel16 = np.asarray(qvel16, dtype=float)
        if qpos16.shape != (16,) or qvel16.shape != (16,):
            raise ValueError("Measured qpos and qvel must each contain 16 values.")
        if not 0.0 <= blend <= 1.0:
            raise ValueError("State feedback blend must be between 0 and 1.")
        if prediction_dt < 0.0:
            raise ValueError("State prediction timestep must be non-negative.")
        if blend == 0.0:
            return

        model = self._config.model
        theoretical_qpos = self._config.q.copy()
        measured_qpos = theoretical_qpos.copy()
        measured_qvel = np.zeros(model.nv)

        for side in self._sides:
            offset = 0 if side == "right" else 8
            self._joint_resolver.set_qpos(
                measured_qpos, qpos16[offset : offset + 8], side
            )
            measured_qvel[self._arm_dofs[side]] = qvel16[offset : offset + 7]

        if prediction_dt > 0.0:
            mujoco.mj_integratePos(
                model, measured_qpos, measured_qvel, prediction_dt
            )

        correction = np.zeros(model.nv)
        mujoco.mj_differentiatePos(
            model,
            correction,
            1.0,
            theoretical_qpos,
            measured_qpos,
        )
        blended_qpos = theoretical_qpos.copy()
        mujoco.mj_integratePos(model, blended_qpos, correction, blend)
        self._config.update(q=blended_qpos)

    def ready(self) -> bool:
        return len(self._pending) == 0

    def solve(self) -> np.ndarray | None:
        tasks = list(self._tasks.values())
        if self._posture_cost > 0.0:
            tasks.append(self._posture_task)
        tasks.extend(self._nullspace_tasks.values())
        tasks.extend(self._elbow_soft_limit_tasks.values())
        constraints = [self._freeze_task] if self._freeze_task else []

        q_before = self._config.q.copy()
        # This solve attempt consumes the current target pair. Even on failure,
        # wait for a fresh target from every active side before trying again.
        self._pending = set(self._sides)

        for _ in range(self._max_iters):
            try:
                vel = mink.solve_ik(
                    self._config,
                    tasks,
                    self._substep_dt,
                    self._solver_name,
                    limits=self._limits,
                    constraints=constraints,
                    safety_break=False,
                    **self._solver_params,
                )
            except mink.exceptions.NoSolutionFound:
                # Earlier substeps may already have advanced the internal model,
                # while no command from this failed solve reaches the real arm.
                self._config.update(q=q_before)
                print("Warning: constrained IK solver failed. Skipping step.")
                return None
            self._config.integrate_inplace(vel, self._substep_dt)

        qpos = self._config.data.qpos
        right_joints, _ = self._joint_resolver.get_driver(qpos, "right")
        left_joints, _ = self._joint_resolver.get_driver(qpos, "left")
        return np.concatenate(
            [
                np.append(right_joints, self._gripper[0]),
                np.append(left_joints, self._gripper[1]),
            ]
        ).astype(np.float32)


def _frame_name(setup: ArmSetup, side: str) -> str:
    ftype = setup.frame_types[side]
    fid = setup.frame_ids[side]
    obj = {
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "site": mujoco.mjtObj.mjOBJ_SITE,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
    }[ftype]
    return mujoco.mj_id2name(setup.model, obj, fid)


def _arm_qpos_indices(setup: ArmSetup, side: str) -> np.ndarray:
    resolved = (
        setup.joint_resolver._right if side == "right" else setup.joint_resolver._left
    )
    return np.asarray(resolved.arm_qpos, dtype=int)


def _dof_indices_for_qpos(
    model: mujoco.MjModel, qpos_indices: np.ndarray
) -> np.ndarray:
    """Map scalar arm-joint qpos addresses to tangent-space DoF addresses."""
    dof_indices: list[int] = []
    for qpos_index in qpos_indices:
        joint_ids = np.flatnonzero(model.jnt_qposadr == int(qpos_index))
        if joint_ids.size != 1:
            raise ValueError(
                f"Expected qpos index {qpos_index} to start exactly one joint."
            )
        joint_id = int(joint_ids[0])
        joint_type = model.jnt_type[joint_id]
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(
                "Nullspace arm joints must be scalar hinge or slide joints."
            )
        dof_indices.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(dof_indices, dtype=int)


def _elbow_joint_indices(setup: ArmSetup, side: str) -> tuple[int, int]:
    """Return the scalar qpos and DoF addresses for one arm's joint4."""
    joint_name = f"openarm_{side}_joint4"
    joint_id = mujoco.mj_name2id(setup.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"MuJoCo model has no joint named {joint_name!r}.")
    return (
        int(setup.model.jnt_qposadr[joint_id]),
        int(setup.model.jnt_dofadr[joint_id]),
    )


def _joint_velocity_cap(
    velocity_limits: dict[str, float] | None,
    joint_name: str,
    default: float,
) -> float:
    """Return a scalar configured joint velocity cap or its built-in default."""
    if velocity_limits is None:
        return default
    value = np.asarray(velocity_limits[joint_name], dtype=np.float64)
    if value.size != 1:
        raise ValueError(f"Velocity limit for {joint_name!r} must be scalar.")
    return float(value.reshape(-1)[0])


def _configuration_limit_for_qpos(
    model: mujoco.MjModel, qpos_indices: set[int]
) -> mink.ConfigurationLimit:
    limit = mink.ConfigurationLimit(model)
    active_qpos = {int(index) for index in qpos_indices}
    active_dofs = [
        int(model.jnt_dofadr[j])
        for j in range(model.njnt)
        if model.jnt_limited[j] and int(model.jnt_qposadr[j]) in active_qpos
    ]
    indices = np.asarray(active_dofs, dtype=int)
    indices.setflags(write=False)
    limit.indices = indices
    limit.projection_matrix = np.eye(model.nv)[indices] if indices.size else None
    return limit


def _load_velocity_caps(config_path: pathlib.Path | None) -> list[float]:
    """Return per-joint velocity caps in rad/s.

    With no config path, returns the built-in ARM_JOINT_VELOCITY_LIMITS_RAD_S.
    When a path is given, reads the legacy IK-specific
    'arm_velocity_limits' key from the YAML.
    """
    if config_path is None:
        return ARM_JOINT_VELOCITY_LIMITS_RAD_S

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must contain a YAML mapping.")

    key = "arm_velocity_limits"
    if key not in data:
        raise ValueError(f"Config file {config_path} has no top-level '{key}' list.")

    expected = len(ARM_JOINT_VELOCITY_LIMITS_RAD_S)
    raw_caps = data[key]
    caps = [float(v) for v in raw_caps]
    if len(caps) != expected:
        raise ValueError(
            f"{key} in {config_path} has {len(caps)} arm entries; expected {expected}."
        )
    return caps


# ── CLI helpers ───────────────────────────────────────────────────────────────


def register_ik_args(parser: argparse.ArgumentParser) -> None:
    """Register IK-specific CLI flags. Call after register_common_args."""
    parser.add_argument(
        "--pos-cost", type=float, default=1.0, help="Position task cost (default: 1.0)"
    )
    parser.add_argument(
        "--ori-cost",
        type=float,
        default=1.0,
        help="Orientation task cost (default: 1.0)",
    )
    parser.add_argument(
        "--lm-damping",
        type=float,
        default=0.01,
        help="Per-task LM damping (default: 0.01)",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.25,
        help="Global Tikhonov regularization (default: 0.25)",
    )
    parser.add_argument("--solver", default="daqp", help="QP backend (default: daqp)")
    parser.add_argument(
        "--max-iters", type=int, default=10, help="IK substeps per event (default: 10)"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help=(
            "Outer control period in seconds. Defaults to 1 / --tick-hz; "
            "each IK substep uses this value divided by --max-iters."
        ),
    )
    parser.add_argument(
        "--posture-cost",
        type=float,
        default=0.01,
        help="Posture task weight, 0=disabled (default: 0.01)",
    )
    parser.add_argument(
        "--diag-reg",
        type=float,
        default=0.0,
        help="QP diagonal regularization (default: 0.0)",
    )
    parser.add_argument(
        "--limit-velocity",
        action="store_true",
        help="Enable per-joint IK velocity limits.",
    )
    parser.add_argument(
        "--joint-limit-recovery-velocity-scale",
        type=float,
        default=1.1,
        help=(
            "Velocity-limit multiplier used only while a joint is outside its "
            "position range (default: 1.1)."
        ),
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional YAML file with 'arm_velocity_limits: [rad/s, ...]'. "
            "Without it, built-in limits are used. Used only with --limit-velocity."
        ),
    )
    parser.add_argument(
        "--tick-hz",
        type=float,
        default=500.0,
        help=(
            "Dora tick rate used when --dt is omitted; must match the dataflow "
            "timer (default: 500.0)."
        ),
    )
    parser.add_argument(
        "--nullspace-cost",
        type=float,
        default=0.3,
        help="One-dimensional nullspace posture cost (default: 0.3).",
    )
    parser.add_argument(
        "--nullspace-return-rate",
        type=float,
        default=0.5,
        help="Nullspace home return rate in 1/s (default: 0.5).",
    )
    parser.add_argument(
        "--nullspace-max-speed",
        type=float,
        default=0.5,
        help="Maximum nullspace-coordinate speed in rad/s (default: 0.5).",
    )
    parser.add_argument(
        "--nullspace-singularity-low",
        type=float,
        default=0.02,
        help="Singularity ratio where nullspace return is disabled (default: 0.02).",
    )
    parser.add_argument(
        "--nullspace-singularity-high",
        type=float,
        default=0.05,
        help="Singularity ratio where nullspace return is fully active (default: 0.05).",
    )
    parser.add_argument(
        "--nullspace-characteristic-length",
        type=float,
        default=0.3,
        help="Length in meters used to normalize translational Jacobian rows.",
    )
    parser.add_argument(
        "--elbow-soft-limit-cost",
        type=float,
        default=0.0,
        help="Joint4 lower soft-limit cost, 0=disabled (default: 0.0).",
    )
    parser.add_argument(
        "--elbow-soft-limit-angle",
        type=float,
        default=0.08,
        help="Joint4 lower soft limit in radians (default: 0.08).",
    )
    parser.add_argument(
        "--elbow-soft-limit-max-speed",
        type=float,
        default=0.2,
        help="Maximum joint4 soft-limit return speed in rad/s (default: 0.2).",
    )
    parser.add_argument(
        "--elbow-braking-guard-angle",
        type=float,
        default=0.08,
        help="Joint4 lower braking guard position in radians (default: 0.08).",
    )
    parser.add_argument(
        "--elbow-braking-acceleration",
        type=float,
        default=0.0,
        help=(
            "Maximum joint4 deceleration toward its lower guard in rad/s^2; "
            "0 disables the braking limit (default: 0.0)."
        ),
    )


def ik_params_from_args(args: argparse.Namespace) -> IKParams:
    """Build IKParams from parsed args (requires register_ik_args to have been called)."""
    control_dt = args.dt
    if control_dt is None:
        if args.tick_hz <= 0.0:
            raise ValueError("--tick-hz must be positive when --dt is omitted.")
        control_dt = 1.0 / args.tick_hz

    velocity_limits: dict[str, float] | None = None
    if args.limit_velocity:
        caps = _load_velocity_caps(getattr(args, "config", None))
        velocity_limits = {
            f"openarm_{side}_joint{i + 1}": v
            for side in ("left", "right")
            for i, v in enumerate(caps)
        }

    return IKParams(
        position_cost=args.pos_cost,
        orientation_cost=args.ori_cost,
        lm_damping=args.lm_damping,
        damping=args.damping,
        solver=args.solver,
        posture_cost=args.posture_cost,
        diag_reg=args.diag_reg,
        dt=control_dt,
        max_iters=args.max_iters,
        velocity_limits=velocity_limits,
        joint_limit_recovery_velocity_scale=(args.joint_limit_recovery_velocity_scale),
        nullspace_cost=args.nullspace_cost,
        nullspace_return_rate=args.nullspace_return_rate,
        nullspace_max_speed=args.nullspace_max_speed,
        nullspace_singularity_low=args.nullspace_singularity_low,
        nullspace_singularity_high=args.nullspace_singularity_high,
        nullspace_characteristic_length=args.nullspace_characteristic_length,
        elbow_soft_limit_cost=args.elbow_soft_limit_cost,
        elbow_soft_limit_angle=args.elbow_soft_limit_angle,
        elbow_soft_limit_max_speed=args.elbow_soft_limit_max_speed,
        elbow_braking_guard_angle=args.elbow_braking_guard_angle,
        elbow_braking_acceleration=args.elbow_braking_acceleration,
    )
