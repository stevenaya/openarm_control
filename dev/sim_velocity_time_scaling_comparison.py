#!/usr/bin/env python3
"""Compare hard limits, preview scaling, soft overflow, and augmented QP."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import mink
import mujoco
import numpy as np
import qpsolvers
from scipy.spatial import cKDTree

from openarm_control.braking import distance_velocity_envelope
from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S
from openarm_control.kinematics import _configuration_limit_for_qpos
from openarm_control.singularity import normalized_arm_jacobian

from sim_intervention_posture_replay import (
    CHARACTERISTIC_LENGTH,
    CONTROL_DT,
    DEFAULT_RUN_DIR,
    CommandDiagnostics,
    DynamicSideArm,
    ReplayTrace,
    _elbow_at_q,
    _load_recorded_commands,
    _make_kinematics,
    _orientation_error,
    _source_geometry,
    detect_retract_segments,
)
from sim_modal_shoulder_limit_comparison import (
    _extended_circle_inputs,
)


@dataclass(frozen=True)
class Strategy:
    """One velocity-allocation strategy."""

    name: str
    mode: str
    soft_weight: float = 0.0
    joint_leak_limit: float = 0.0
    exact_null_leak_limit: float = 0.0
    near_null_leak_limit: float = 0.0
    progress_weight: float = 1_000.0


@dataclass
class PolicyDiagnostics:
    """Per-tick diagnostics from a velocity policy."""

    reference_velocity: np.ndarray
    flexible_velocity: np.ndarray
    final_velocity: np.ndarray
    scale: np.ndarray
    augmented_progress: np.ndarray
    preview_peak_utilization: np.ndarray
    flexible_peak_utilization: np.ndarray
    exact_null_leak: np.ndarray
    near_null_leak: np.ndarray
    residual_norm: np.ndarray
    soft_slack_norm: np.ndarray
    policy_failed: np.ndarray


@dataclass(frozen=True)
class PolicyStep:
    """One final command and policy diagnostics."""

    command: np.ndarray | None
    reference_velocity: np.ndarray
    flexible_velocity: np.ndarray
    final_velocity: np.ndarray
    scale: float
    augmented_progress: float
    preview_peak_utilization: float
    flexible_peak_utilization: float
    exact_null_leak: float
    near_null_leak: float
    residual_norm: float
    soft_slack_norm: float
    failed: bool


def _strategies() -> list[Strategy]:
    return [
        Strategy("A_current_hard_limits", "baseline"),
        Strategy("N_guard_reference", "reference"),
        Strategy("P_preview_pure_scale", "pure_scale"),
        Strategy("S_soft_w0p3_then_scale", "soft_scale", soft_weight=0.3),
        Strategy("S_soft_w3_then_scale", "soft_scale", soft_weight=3.0),
        Strategy("S_soft_w30_then_scale", "soft_scale", soft_weight=30.0),
        Strategy("S_soft_w100_then_scale", "soft_scale", soft_weight=100.0),
        Strategy("S_soft_w300_then_scale", "soft_scale", soft_weight=300.0),
        Strategy("Q_augmented_strict", "augmented_strict"),
        Strategy(
            "Q_augmented_flex_tight",
            "augmented_flex",
            soft_weight=1.0,
            joint_leak_limit=0.15,
            exact_null_leak_limit=0.02,
            near_null_leak_limit=0.05,
        ),
        Strategy(
            "Q_augmented_flex_medium",
            "augmented_flex",
            soft_weight=1.0,
            joint_leak_limit=0.3,
            exact_null_leak_limit=0.05,
            near_null_leak_limit=0.1,
        ),
        Strategy(
            "Q_augmented_flex",
            "augmented_flex",
            soft_weight=1.0,
            joint_leak_limit=0.5,
            exact_null_leak_limit=0.1,
            near_null_leak_limit=0.2,
        ),
        Strategy(
            "M_mink_soft_w3_then_scale",
            "mink_soft_scale",
            soft_weight=3.0,
        ),
        Strategy(
            "M_mink_soft_w30_then_scale",
            "mink_soft_scale",
            soft_weight=30.0,
        ),
        Strategy(
            "M_mink_soft_w100_then_scale",
            "mink_soft_scale",
            soft_weight=100.0,
        ),
        Strategy(
            "M_mink_soft_w300_then_scale",
            "mink_soft_scale",
            soft_weight=300.0,
        ),
        Strategy(
            "M_mink_augmented_strict",
            "mink_augmented_strict",
            soft_weight=3.0,
        ),
        Strategy(
            "M_mink_augmented_strict_p0p1",
            "mink_augmented_strict",
            soft_weight=30.0,
            progress_weight=0.1,
        ),
        Strategy(
            "M_mink_augmented_strict_p1",
            "mink_augmented_strict",
            soft_weight=30.0,
            progress_weight=1.0,
        ),
        Strategy(
            "M_mink_augmented_strict_p10",
            "mink_augmented_strict",
            soft_weight=30.0,
            progress_weight=10.0,
        ),
        Strategy(
            "M_mink_augmented_flex_tight",
            "mink_augmented_flex",
            soft_weight=3.0,
            joint_leak_limit=0.15,
            exact_null_leak_limit=0.02,
            near_null_leak_limit=0.05,
        ),
        Strategy(
            "M_mink_augmented_flex_tight_p1",
            "mink_augmented_flex",
            soft_weight=30.0,
            joint_leak_limit=0.15,
            exact_null_leak_limit=0.02,
            near_null_leak_limit=0.05,
            progress_weight=1.0,
        ),
        Strategy(
            "M_mink_augmented_hard_strict",
            "mink_augmented_hard_strict",
            soft_weight=3.0,
        ),
        Strategy(
            "M_mink_augmented_hard_flex_tight",
            "mink_augmented_hard_flex",
            soft_weight=3.0,
            joint_leak_limit=0.15,
            exact_null_leak_limit=0.02,
            near_null_leak_limit=0.05,
        ),
    ]


def _velocity_mapping(
    side: str,
    caps: np.ndarray,
) -> dict[str, float]:
    return {
        f"openarm_{side}_joint{index + 1}": float(caps[index])
        for index in range(7)
    }


class PreviewVelocityPolicy:
    """Run a transactional preview solve and commit only the final velocity."""

    def __init__(
        self,
        kinematics,
        side: str,
        strategy: Strategy,
        *,
        guard_scale: float,
        soft_ratio: float,
    ) -> None:
        if guard_scale <= 1.0:
            raise ValueError("guard_scale must exceed one.")
        if not 0.0 < soft_ratio <= 1.0:
            raise ValueError("soft_ratio must be in (0, 1].")
        self.kinematics = kinematics
        self.side = side
        self.strategy = strategy
        self.soft_ratio = float(soft_ratio)
        self.physical_caps = np.asarray(
            ARM_JOINT_VELOCITY_LIMITS_RAD_S,
            dtype=np.float64,
        )
        self.guard_caps = self.physical_caps * float(guard_scale)

        solver = kinematics._ik
        assert solver is not None
        self.solver = solver
        self.model = solver._model
        self.qpos_indices = np.asarray(
            solver._arm_qpos_by_side[side],
            dtype=int,
        )
        self.dof_indices = np.asarray(
            solver._arm_dofs_by_side[side],
            dtype=int,
        )
        self.frame_task = solver._tasks[side]
        self.position_limit = _configuration_limit_for_qpos(
            self.model,
            set(self.qpos_indices.tolist()),
        )
        self.guard_limit = mink.VelocityLimit(
            self.model,
            velocities=_velocity_mapping(side, self.guard_caps),
        )
        self.solver._limits = [
            self.position_limit,
            self.guard_limit,
            *self.solver._singularity_limits.values(),
        ]

    def solve(
        self,
        measured_q: np.ndarray,
        measured_dq: np.ndarray,
    ) -> PolicyStep:
        q_start = self.solver._config.q.copy()
        lower_allowed, upper_allowed = self._physical_velocity_envelope(
            q_start,
            measured_q,
            measured_dq,
        )
        if self.strategy.mode.startswith("mink_"):
            return self._solve_mink_augmented(
                q_start,
                measured_q,
                measured_dq,
                lower_allowed,
                upper_allowed,
            )

        preview_result = self.solver.solve()
        if preview_result is None:
            return self._failed_step(q_start)

        q_preview = self.solver._config.q.copy()
        full_displacement = np.empty(self.model.nv, dtype=np.float64)
        mujoco.mj_differentiatePos(
            self.model,
            full_displacement,
            CONTROL_DT,
            q_start,
            q_preview,
        )
        reference_velocity = full_displacement[self.dof_indices]

        start_configuration = mink.Configuration(self.model, q=q_start)
        jacobian = normalized_arm_jacobian(
            self.frame_task,
            start_configuration,
            self.dof_indices,
            CHARACTERISTIC_LENGTH,
        )
        _, _, vt = np.linalg.svd(jacobian, full_matrices=True)
        exact_null = vt[-1]
        near_null = vt[-2]

        flexible_velocity = reference_velocity.copy()
        final_velocity = reference_velocity.copy()
        scale = 1.0
        progress = 1.0
        slack = np.zeros(7, dtype=np.float64)
        policy_failed = False

        if self.strategy.mode == "pure_scale":
            scale = _max_directional_scale(
                reference_velocity,
                lower_allowed,
                upper_allowed,
            )
            final_velocity = scale * reference_velocity
        elif self.strategy.mode == "soft_scale":
            soft_result = _solve_soft_overflow_qp(
                reference_velocity,
                jacobian,
                exact_null,
                near_null,
                lower_allowed,
                upper_allowed,
                self.guard_caps,
                soft_weight=self.strategy.soft_weight,
            )
            if soft_result is None:
                policy_failed = True
            else:
                flexible_velocity, slack = soft_result
            scale = _max_directional_scale(
                flexible_velocity,
                lower_allowed,
                upper_allowed,
            )
            final_velocity = scale * flexible_velocity
        elif self.strategy.mode in {
            "augmented_strict",
            "augmented_flex",
        }:
            augmented_result = _solve_augmented_qp(
                reference_velocity,
                jacobian,
                exact_null,
                near_null,
                lower_allowed,
                upper_allowed,
                lower_allowed * self.soft_ratio,
                upper_allowed * self.soft_ratio,
                self.strategy,
            )
            if augmented_result is None:
                policy_failed = True
                scale = _max_directional_scale(
                    reference_velocity,
                    lower_allowed,
                    upper_allowed,
                )
                flexible_velocity = reference_velocity.copy()
                final_velocity = scale * reference_velocity
                progress = scale
            else:
                flexible_velocity, progress, slack = augmented_result
                final_velocity = flexible_velocity.copy()
                scale = _max_directional_scale(
                    final_velocity,
                    lower_allowed,
                    upper_allowed,
                )
                if scale < 1.0 - 1e-7:
                    policy_failed = True
                    final_velocity *= scale
        elif self.strategy.mode != "reference":
            raise ValueError(f"Unsupported mode: {self.strategy.mode}")

        branch_scale = (
            progress * scale
            if self.strategy.mode.startswith("augmented")
            else scale
        )
        residual = final_velocity - branch_scale * reference_velocity
        exact_null_leak = float(exact_null @ residual)
        near_null_leak = float(near_null @ residual)

        q_final = q_start.copy()
        full_velocity = np.zeros(self.model.nv, dtype=np.float64)
        full_velocity[self.dof_indices] = final_velocity
        mujoco.mj_integratePos(
            self.model,
            q_final,
            full_velocity,
            CONTROL_DT,
        )
        self.solver._config.update(q=q_final)
        command, _ = self.solver._joint_resolver.get_driver(q_final, self.side)
        return PolicyStep(
            command=np.asarray(command, dtype=np.float64),
            reference_velocity=reference_velocity,
            flexible_velocity=flexible_velocity,
            final_velocity=final_velocity,
            scale=float(scale),
            augmented_progress=float(progress),
            preview_peak_utilization=float(
                np.max(np.abs(reference_velocity) / self.physical_caps)
            ),
            flexible_peak_utilization=float(
                np.max(np.abs(flexible_velocity) / self.physical_caps)
            ),
            exact_null_leak=exact_null_leak,
            near_null_leak=near_null_leak,
            residual_norm=float(np.linalg.norm(residual)),
            soft_slack_norm=float(np.linalg.norm(slack)),
            failed=policy_failed,
        )

    def _solve_mink_augmented(
        self,
        q_start: np.ndarray,
        measured_q: np.ndarray,
        measured_dq: np.ndarray,
        lower_allowed: np.ndarray,
        upper_allowed: np.ndarray,
    ) -> PolicyStep:
        reference_result = self.solver.solve()
        if reference_result is None:
            return self._failed_step(q_start)
        q_reference = self.solver._config.q.copy()
        reference_velocity = _configuration_velocity(
            self.model,
            q_start,
            q_reference,
            self.dof_indices,
            CONTROL_DT,
        )
        self.solver._config.update(q=q_start)

        tasks = list(self.solver._tasks.values())
        if self.solver._posture_cost > 0.0:
            tasks.append(self.solver._posture_task)
        if self.solver._kinetic_energy_task is not None:
            tasks.append(self.solver._kinetic_energy_task)
        tasks.extend(self.solver._nullspace_tasks.values())
        tasks.extend(self.solver._elbow_soft_limit_tasks.values())
        constraints = (
            [self.solver._freeze_task]
            if self.solver._freeze_task is not None
            else []
        )
        for limit in self.solver._singularity_limits.values():
            limit.prepare(self.solver._config)

        progress_values: list[float] = []
        slack_values: list[np.ndarray] = []
        for _ in range(self.solver._max_iters):
            sub_lower, sub_upper = self._physical_velocity_envelope(
                self.solver._config.q,
                measured_q,
                measured_dq,
            )
            base_problem = mink.build_ik(
                self.solver._config,
                tasks,
                self.solver._substep_dt,
                damping=float(self.solver._solver_params["damping"]),
                limits=self.solver._limits,
                constraints=constraints,
            )
            base_solution = _solve_qpsolvers_problem(base_problem)
            if base_solution is None:
                return self._failed_step(q_start)

            jacobian = normalized_arm_jacobian(
                self.frame_task,
                self.solver._config,
                self.dof_indices,
                CHARACTERISTIC_LENGTH,
            )
            _, _, vt = np.linalg.svd(jacobian, full_matrices=True)
            exact_null = vt[-1]
            near_null = vt[-2]

            if self.strategy.mode == "mink_soft_scale":
                augmented = _augment_mink_soft_problem(
                    base_problem,
                    self.dof_indices,
                    sub_lower,
                    sub_upper,
                    self.solver._substep_dt,
                    self.strategy.soft_weight,
                )
                solution = _solve_qpsolvers_problem(augmented)
                alpha = 1.0
                slack_start = self.model.nv
            else:
                feedback_norm_sq = _task_feedback_norm_sq(
                    tasks,
                    self.solver._config,
                )
                augmented = _augment_mink_progress_problem(
                    base_problem,
                    base_solution,
                    self.dof_indices,
                    exact_null,
                    near_null,
                    sub_lower * self.soft_ratio,
                    sub_upper * self.soft_ratio,
                    self.solver._substep_dt,
                    feedback_norm_sq,
                    self.strategy,
                    hard_velocity=(
                        self.strategy.mode.startswith(
                            "mink_augmented_hard_"
                        )
                    ),
                    hard_lower_caps=sub_lower,
                    hard_upper_caps=sub_upper,
                )
                solution = _solve_qpsolvers_problem(augmented)
                alpha = (
                    0.0
                    if solution is None
                    else float(solution[self.model.nv])
                )
                slack_start = self.model.nv + 1
            if solution is None:
                return self._failed_step(q_start)

            displacement = solution[: self.model.nv]
            self.solver._config.integrate_inplace(
                displacement / self.solver._substep_dt,
                self.solver._substep_dt,
            )
            progress_values.append(alpha)
            slack_values.append(
                solution[slack_start : slack_start + 7]
                / self.solver._substep_dt
            )

        q_flexible = self.solver._config.q.copy()
        flexible_velocity = _configuration_velocity(
            self.model,
            q_start,
            q_flexible,
            self.dof_indices,
            CONTROL_DT,
        )
        scale = _max_directional_scale(
            flexible_velocity,
            lower_allowed,
            upper_allowed,
        )
        final_velocity = scale * flexible_velocity
        progress = float(np.mean(progress_values))
        branch_scale = scale * progress
        residual = final_velocity - branch_scale * reference_velocity

        start_configuration = mink.Configuration(self.model, q=q_start)
        start_jacobian = normalized_arm_jacobian(
            self.frame_task,
            start_configuration,
            self.dof_indices,
            CHARACTERISTIC_LENGTH,
        )
        _, _, start_vt = np.linalg.svd(
            start_jacobian,
            full_matrices=True,
        )
        exact_null = start_vt[-1]
        near_null = start_vt[-2]

        q_final = q_start.copy()
        full_velocity = np.zeros(self.model.nv, dtype=np.float64)
        full_velocity[self.dof_indices] = final_velocity
        mujoco.mj_integratePos(
            self.model,
            q_final,
            full_velocity,
            CONTROL_DT,
        )
        self.solver._config.update(q=q_final)
        command, _ = self.solver._joint_resolver.get_driver(
            q_final,
            self.side,
        )
        slack_array = np.vstack(slack_values)
        return PolicyStep(
            command=np.asarray(command, dtype=np.float64),
            reference_velocity=reference_velocity,
            flexible_velocity=flexible_velocity,
            final_velocity=final_velocity,
            scale=float(scale),
            augmented_progress=progress,
            preview_peak_utilization=float(
                np.max(np.abs(reference_velocity) / self.physical_caps)
            ),
            flexible_peak_utilization=float(
                np.max(np.abs(flexible_velocity) / self.physical_caps)
            ),
            exact_null_leak=float(exact_null @ residual),
            near_null_leak=float(near_null @ residual),
            residual_norm=float(np.linalg.norm(residual)),
            soft_slack_norm=float(
                np.sqrt(np.mean(np.square(slack_array)))
            ),
            failed=False,
        )

    def _physical_velocity_envelope(
        self,
        q_start: np.ndarray,
        measured_q: np.ndarray,
        measured_dq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        command_q = q_start[self.qpos_indices]
        lower_guard = np.empty(7, dtype=np.float64)
        upper_guard = np.empty(7, dtype=np.float64)
        for index in range(7):
            joint = self.model.joint(
                f"openarm_{self.side}_joint{index + 1}"
            )
            lower_guard[index] = self.model.jnt_range[joint.id, 0]
            upper_guard[index] = self.model.jnt_range[joint.id, 1]
        lower_guard[3] = 0.08

        lower_distance = np.minimum(
            command_q - lower_guard,
            measured_q
            - lower_guard
            - 0.04 * np.maximum(-measured_dq, 0.0)
            - 0.01,
        )
        upper_distance = np.minimum(
            upper_guard - command_q,
            upper_guard
            - measured_q
            - 0.04 * np.maximum(measured_dq, 0.0)
            - 0.01,
        )
        slowdown = np.full(7, 0.5, dtype=np.float64)
        lower_allowed = distance_velocity_envelope(
            lower_distance,
            self.physical_caps,
            slowdown,
            exponent=2.0,
        )
        upper_allowed = distance_velocity_envelope(
            upper_distance,
            self.physical_caps,
            slowdown,
            exponent=2.0,
        )
        return lower_allowed, upper_allowed

    def _failed_step(self, q_start: np.ndarray) -> PolicyStep:
        self.solver._config.update(q=q_start)
        zeros = np.zeros(7, dtype=np.float64)
        return PolicyStep(
            command=None,
            reference_velocity=zeros,
            flexible_velocity=zeros,
            final_velocity=zeros,
            scale=0.0,
            augmented_progress=0.0,
            preview_peak_utilization=0.0,
            flexible_peak_utilization=0.0,
            exact_null_leak=0.0,
            near_null_leak=0.0,
            residual_norm=0.0,
            soft_slack_norm=0.0,
            failed=True,
        )


def _max_directional_scale(
    velocity: np.ndarray,
    lower_allowed: np.ndarray,
    upper_allowed: np.ndarray,
) -> float:
    scale = 1.0
    for value, lower, upper in zip(
        velocity,
        lower_allowed,
        upper_allowed,
        strict=True,
    ):
        magnitude = abs(float(value))
        if magnitude <= 1e-12:
            continue
        allowed = float(upper if value > 0.0 else lower)
        scale = min(scale, max(allowed, 0.0) / magnitude)
    return float(np.clip(scale, 0.0, 1.0))


def _configuration_velocity(
    model: mujoco.MjModel,
    q_start: np.ndarray,
    q_end: np.ndarray,
    dof_indices: np.ndarray,
    dt: float,
) -> np.ndarray:
    full_velocity = np.empty(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(
        model,
        full_velocity,
        dt,
        q_start,
        q_end,
    )
    return full_velocity[dof_indices]


def _solve_qpsolvers_problem(
    problem: qpsolvers.Problem,
) -> np.ndarray | None:
    return _solve_problem(
        np.asarray(problem.P, dtype=np.float64),
        np.asarray(problem.q, dtype=np.float64),
        G=(
            None
            if problem.G is None
            else np.asarray(problem.G, dtype=np.float64)
        ),
        h=(
            None
            if problem.h is None
            else np.asarray(problem.h, dtype=np.float64)
        ),
        A=(
            None
            if problem.A is None
            else np.asarray(problem.A, dtype=np.float64)
        ),
        b=(
            None
            if problem.b is None
            else np.asarray(problem.b, dtype=np.float64)
        ),
    )


def _extended_rows(
    matrix: np.ndarray | None,
    variable_count: int,
) -> np.ndarray | None:
    if matrix is None:
        return None
    extended = np.zeros(
        (matrix.shape[0], variable_count),
        dtype=np.float64,
    )
    extended[:, : matrix.shape[1]] = matrix
    return extended


def _stack_optional_rows(
    base: np.ndarray | None,
    extra: np.ndarray,
) -> np.ndarray:
    return extra if base is None else np.vstack([base, extra])


def _stack_optional_values(
    base: np.ndarray | None,
    extra: np.ndarray,
) -> np.ndarray:
    return extra if base is None else np.hstack([base, extra])


def _soft_velocity_rows(
    dof_indices: np.ndarray,
    variable_count: int,
    slack_start: int,
    lower_caps: np.ndarray,
    upper_caps: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    joint_count = len(dof_indices)
    rows = np.zeros(
        (3 * joint_count, variable_count),
        dtype=np.float64,
    )
    values = np.empty(3 * joint_count, dtype=np.float64)
    for local_index, dof_index in enumerate(dof_indices):
        slack_index = slack_start + local_index
        rows[local_index, dof_index] = 1.0
        rows[local_index, slack_index] = -1.0
        values[local_index] = dt * upper_caps[local_index]

        lower_row = joint_count + local_index
        rows[lower_row, dof_index] = -1.0
        rows[lower_row, slack_index] = -1.0
        values[lower_row] = dt * lower_caps[local_index]

        slack_row = 2 * joint_count + local_index
        rows[slack_row, slack_index] = -1.0
        values[slack_row] = 0.0
    return rows, values


def _hard_velocity_rows(
    dof_indices: np.ndarray,
    variable_count: int,
    lower_caps: np.ndarray,
    upper_caps: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    joint_count = len(dof_indices)
    rows = np.zeros(
        (2 * joint_count, variable_count),
        dtype=np.float64,
    )
    values = np.empty(2 * joint_count, dtype=np.float64)
    for local_index, dof_index in enumerate(dof_indices):
        rows[local_index, dof_index] = 1.0
        values[local_index] = dt * upper_caps[local_index]
        lower_row = joint_count + local_index
        rows[lower_row, dof_index] = -1.0
        values[lower_row] = dt * lower_caps[local_index]
    return rows, values


def _augment_mink_soft_problem(
    base: qpsolvers.Problem,
    dof_indices: np.ndarray,
    lower_caps: np.ndarray,
    upper_caps: np.ndarray,
    dt: float,
    soft_weight: float,
) -> qpsolvers.Problem:
    nv = base.q.shape[0]
    joint_count = len(dof_indices)
    variable_count = nv + joint_count
    slack_start = nv
    hessian = np.zeros(
        (variable_count, variable_count),
        dtype=np.float64,
    )
    hessian[:nv, :nv] = np.asarray(base.P, dtype=np.float64)
    hessian[
        slack_start:,
        slack_start:,
    ] = 2.0 * soft_weight * np.eye(joint_count)
    linear = np.zeros(variable_count, dtype=np.float64)
    linear[:nv] = np.asarray(base.q, dtype=np.float64)

    G = _extended_rows(base.G, variable_count)
    h = None if base.h is None else np.asarray(base.h, dtype=np.float64)
    soft_G, soft_h = _soft_velocity_rows(
        dof_indices,
        variable_count,
        slack_start,
        lower_caps,
        upper_caps,
        dt,
    )
    G = _stack_optional_rows(G, soft_G)
    h = _stack_optional_values(h, soft_h)
    return qpsolvers.Problem(
        hessian,
        linear,
        G,
        h,
        _extended_rows(base.A, variable_count),
        None if base.b is None else np.asarray(base.b, dtype=np.float64),
    )


def _task_feedback_norm_sq(
    tasks: list[mink.BaseTask],
    configuration: mink.Configuration,
) -> float:
    norm_sq = 0.0
    for task in tasks:
        if not isinstance(task, mink.Task):
            continue
        error = task.compute_error(configuration)
        weighted_feedback = np.asarray(task.cost, dtype=np.float64) * (
            -task.gain * error
        )
        norm_sq += float(weighted_feedback @ weighted_feedback)
    return norm_sq


def _augment_mink_progress_problem(
    base: qpsolvers.Problem,
    reference_displacement: np.ndarray,
    dof_indices: np.ndarray,
    exact_null: np.ndarray,
    near_null: np.ndarray,
    lower_soft_caps: np.ndarray,
    upper_soft_caps: np.ndarray,
    dt: float,
    feedback_norm_sq: float,
    strategy: Strategy,
    *,
    hard_velocity: bool,
    hard_lower_caps: np.ndarray,
    hard_upper_caps: np.ndarray,
) -> qpsolvers.Problem:
    nv = base.q.shape[0]
    joint_count = len(dof_indices)
    alpha_index = nv
    slack_start = alpha_index + 1
    variable_count = slack_start + joint_count
    hessian = np.zeros(
        (variable_count, variable_count),
        dtype=np.float64,
    )
    base_hessian = np.asarray(base.P, dtype=np.float64)
    base_linear = np.asarray(base.q, dtype=np.float64)
    hessian[:nv, :nv] = base_hessian
    hessian[:nv, alpha_index] = base_linear
    hessian[alpha_index, :nv] = base_linear
    hessian[alpha_index, alpha_index] = (
        feedback_norm_sq + 2.0 * strategy.progress_weight
    )
    hessian[
        slack_start:,
        slack_start:,
    ] = 2.0 * strategy.soft_weight * np.eye(joint_count)
    linear = np.zeros(variable_count, dtype=np.float64)
    linear[alpha_index] = -2.0 * strategy.progress_weight

    G = _extended_rows(base.G, variable_count)
    h = None if base.h is None else np.asarray(base.h, dtype=np.float64)
    soft_G, soft_h = _soft_velocity_rows(
        dof_indices,
        variable_count,
        slack_start,
        lower_soft_caps,
        upper_soft_caps,
        dt,
    )
    alpha_G = np.zeros((2, variable_count), dtype=np.float64)
    alpha_G[0, alpha_index] = 1.0
    alpha_G[1, alpha_index] = -1.0
    G = _stack_optional_rows(G, soft_G)
    h = _stack_optional_values(h, soft_h)
    G = _stack_optional_rows(G, alpha_G)
    h = _stack_optional_values(
        h,
        np.array([1.0, 0.0], dtype=np.float64),
    )
    if hard_velocity:
        hard_G, hard_h = _hard_velocity_rows(
            dof_indices,
            variable_count,
            hard_lower_caps,
            hard_upper_caps,
            dt,
        )
        G = _stack_optional_rows(G, hard_G)
        h = _stack_optional_values(h, hard_h)

    residual = np.zeros(
        (joint_count, variable_count),
        dtype=np.float64,
    )
    for local_index, dof_index in enumerate(dof_indices):
        residual[local_index, dof_index] = 1.0
        residual[local_index, alpha_index] = -reference_displacement[
            dof_index
        ]

    A = _extended_rows(base.A, variable_count)
    b = None if base.b is None else np.asarray(base.b, dtype=np.float64)
    if strategy.mode.endswith("_strict"):
        A = _stack_optional_rows(A, residual)
        b = _stack_optional_values(
            b,
            np.zeros(joint_count, dtype=np.float64),
        )
    else:
        joint_bound = (
            strategy.joint_leak_limit
            * dt
            * np.ones(joint_count, dtype=np.float64)
        )
        branch = np.vstack([exact_null, near_null]) @ residual
        branch_bound = dt * np.array(
            [
                strategy.exact_null_leak_limit,
                strategy.near_null_leak_limit,
            ],
            dtype=np.float64,
        )
        G = _stack_optional_rows(G, residual)
        h = _stack_optional_values(h, joint_bound)
        G = _stack_optional_rows(G, -residual)
        h = _stack_optional_values(h, joint_bound)
        G = _stack_optional_rows(G, branch)
        h = _stack_optional_values(h, branch_bound)
        G = _stack_optional_rows(G, -branch)
        h = _stack_optional_values(h, branch_bound)

    return qpsolvers.Problem(hessian, linear, G, h, A, b)


def _add_least_squares(
    hessian: np.ndarray,
    linear: np.ndarray,
    matrix: np.ndarray,
    target: np.ndarray,
    weight: float,
) -> None:
    if weight <= 0.0:
        return
    hessian += 2.0 * weight * (matrix.T @ matrix)
    linear -= 2.0 * weight * (matrix.T @ target)


def _solve_problem(
    hessian: np.ndarray,
    linear: np.ndarray,
    *,
    G: np.ndarray | None = None,
    h: np.ndarray | None = None,
    A: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> np.ndarray | None:
    hessian = 0.5 * (hessian + hessian.T)
    hessian += np.eye(hessian.shape[0], dtype=np.float64) * 1e-9
    result = qpsolvers.solve_problem(
        qpsolvers.Problem(hessian, linear, G, h, A, b),
        solver="daqp",
    )
    if not result.found or result.x is None:
        return None
    return np.asarray(result.x, dtype=np.float64)


def _solve_soft_overflow_qp(
    reference_velocity: np.ndarray,
    jacobian: np.ndarray,
    exact_null: np.ndarray,
    near_null: np.ndarray,
    lower_soft_caps: np.ndarray,
    upper_soft_caps: np.ndarray,
    guard_caps: np.ndarray,
    *,
    soft_weight: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    joint_count = 7
    variable_count = 14
    velocity_slice = slice(0, joint_count)
    slack_slice = slice(joint_count, variable_count)
    hessian = np.zeros((variable_count, variable_count), dtype=np.float64)
    linear = np.zeros(variable_count, dtype=np.float64)

    shape = np.zeros((joint_count, variable_count), dtype=np.float64)
    shape[:, velocity_slice] = np.eye(joint_count)
    _add_least_squares(
        hessian,
        linear,
        shape,
        reference_velocity,
        1.0,
    )
    task = np.zeros((6, variable_count), dtype=np.float64)
    task[:, velocity_slice] = jacobian
    _add_least_squares(
        hessian,
        linear,
        task,
        jacobian @ reference_velocity,
        50.0,
    )
    branch = np.zeros((2, variable_count), dtype=np.float64)
    branch[:, velocity_slice] = np.vstack([exact_null, near_null])
    _add_least_squares(
        hessian,
        linear,
        branch,
        branch[:, velocity_slice] @ reference_velocity,
        50.0,
    )
    slack = np.zeros((joint_count, variable_count), dtype=np.float64)
    slack[:, slack_slice] = np.eye(joint_count)
    _add_least_squares(
        hessian,
        linear,
        slack,
        np.zeros(joint_count, dtype=np.float64),
        soft_weight,
    )

    identity = np.eye(joint_count, dtype=np.float64)
    zero = np.zeros((joint_count, joint_count), dtype=np.float64)
    G = np.vstack(
        [
            np.hstack([identity, -identity]),
            np.hstack([-identity, -identity]),
            np.hstack([zero, -identity]),
            np.hstack([identity, zero]),
            np.hstack([-identity, zero]),
        ]
    )
    h = np.hstack(
        [
            upper_soft_caps,
            lower_soft_caps,
            np.zeros(joint_count),
            guard_caps,
            guard_caps,
        ]
    )
    solution = _solve_problem(hessian, linear, G=G, h=h)
    if solution is None:
        return None
    return solution[velocity_slice], solution[slack_slice]


def _solve_augmented_qp(
    reference_velocity: np.ndarray,
    jacobian: np.ndarray,
    exact_null: np.ndarray,
    near_null: np.ndarray,
    lower_allowed: np.ndarray,
    upper_allowed: np.ndarray,
    lower_soft_caps: np.ndarray,
    upper_soft_caps: np.ndarray,
    strategy: Strategy,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    joint_count = 7
    alpha_index = joint_count
    slack_start = alpha_index + 1
    variable_count = slack_start + joint_count
    velocity_slice = slice(0, joint_count)
    slack_slice = slice(slack_start, variable_count)
    hessian = np.zeros((variable_count, variable_count), dtype=np.float64)
    linear = np.zeros(variable_count, dtype=np.float64)

    residual = np.zeros((joint_count, variable_count), dtype=np.float64)
    residual[:, velocity_slice] = np.eye(joint_count)
    residual[:, alpha_index] = -reference_velocity
    _add_least_squares(
        hessian,
        linear,
        residual,
        np.zeros(joint_count),
        1.0,
    )
    task_residual = np.zeros((6, variable_count), dtype=np.float64)
    task_residual[:, velocity_slice] = jacobian
    task_residual[:, alpha_index] = -(jacobian @ reference_velocity)
    _add_least_squares(
        hessian,
        linear,
        task_residual,
        np.zeros(6),
        50.0,
    )
    branch_residual = np.vstack([exact_null, near_null]) @ residual
    _add_least_squares(
        hessian,
        linear,
        branch_residual,
        np.zeros(2),
        50.0,
    )
    progress = np.zeros((1, variable_count), dtype=np.float64)
    progress[0, alpha_index] = 1.0
    _add_least_squares(
        hessian,
        linear,
        progress,
        np.ones(1),
        strategy.progress_weight,
    )
    slack = np.zeros((joint_count, variable_count), dtype=np.float64)
    slack[:, slack_slice] = np.eye(joint_count)
    _add_least_squares(
        hessian,
        linear,
        slack,
        np.zeros(joint_count),
        strategy.soft_weight,
    )

    identity = np.eye(joint_count, dtype=np.float64)
    velocity_rows = np.zeros(
        (2 * joint_count, variable_count),
        dtype=np.float64,
    )
    velocity_rows[:joint_count, velocity_slice] = identity
    velocity_rows[joint_count:, velocity_slice] = -identity
    soft_rows = np.zeros(
        (3 * joint_count, variable_count),
        dtype=np.float64,
    )
    soft_rows[:joint_count, velocity_slice] = identity
    soft_rows[:joint_count, slack_slice] = -identity
    soft_rows[joint_count : 2 * joint_count, velocity_slice] = -identity
    soft_rows[joint_count : 2 * joint_count, slack_slice] = -identity
    soft_rows[2 * joint_count :, slack_slice] = -identity
    alpha_rows = np.zeros((2, variable_count), dtype=np.float64)
    alpha_rows[0, alpha_index] = 1.0
    alpha_rows[1, alpha_index] = -1.0

    G_parts = [velocity_rows, soft_rows, alpha_rows]
    h_parts = [
        np.hstack([upper_allowed, lower_allowed]),
        np.hstack(
            [
                upper_soft_caps,
                lower_soft_caps,
                np.zeros(joint_count),
            ]
        ),
        np.array([1.0, 0.0], dtype=np.float64),
    ]
    A: np.ndarray | None = None
    b: np.ndarray | None = None
    if strategy.mode == "augmented_strict":
        A = residual
        b = np.zeros(joint_count, dtype=np.float64)
    else:
        joint_limit = np.full(
            joint_count,
            strategy.joint_leak_limit,
            dtype=np.float64,
        )
        G_parts.extend([residual, -residual])
        h_parts.extend([joint_limit, joint_limit])
        branch_limit = np.array(
            [
                strategy.exact_null_leak_limit,
                strategy.near_null_leak_limit,
            ],
            dtype=np.float64,
        )
        G_parts.extend([branch_residual, -branch_residual])
        h_parts.extend([branch_limit, branch_limit])

    G = np.vstack(G_parts)
    h = np.hstack(h_parts)
    solution = _solve_problem(
        hessian,
        linear,
        G=G,
        h=h,
        A=A,
        b=b,
    )
    if solution is None:
        return None
    return (
        solution[velocity_slice],
        float(solution[alpha_index]),
        solution[slack_slice],
    )


def _baseline_step(kinematics, side: str) -> PolicyStep:
    solver = kinematics._ik
    assert solver is not None
    q_start = solver._config.q.copy()
    result = solver.solve()
    zeros = np.zeros(7, dtype=np.float64)
    if result is None:
        return PolicyStep(
            None,
            zeros,
            zeros,
            zeros,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            True,
        )
    full_displacement = np.empty(solver._model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(
        solver._model,
        full_displacement,
        CONTROL_DT,
        q_start,
        solver._config.q,
    )
    velocity = full_displacement[solver._arm_dofs_by_side[side]]
    command, _ = solver._joint_resolver.get_driver(solver._config.q, side)
    utilization = float(
        np.max(
            np.abs(velocity)
            / np.asarray(
                ARM_JOINT_VELOCITY_LIMITS_RAD_S,
                dtype=np.float64,
            )
        )
    )
    return PolicyStep(
        np.asarray(command, dtype=np.float64),
        velocity,
        velocity,
        velocity,
        1.0,
        1.0,
        utilization,
        utilization,
        0.0,
        0.0,
        0.0,
        0.0,
        False,
    )


def _simulate(
    side: str,
    strategy: Strategy,
    target_pose: np.ndarray,
    source_q: np.ndarray,
    source_elbow: np.ndarray,
    *,
    settle_duration: float,
    guard_scale: float,
    soft_ratio: float,
) -> tuple[ReplayTrace, PolicyDiagnostics]:
    kinematics = _make_kinematics(side, 0.0)
    dynamics = DynamicSideArm(side, source_q[0])
    dynamics.settle(settle_duration)
    kinematics.sync(dynamics.driver_qpos())
    diagnostics = CommandDiagnostics(kinematics, side)
    policy = (
        None
        if strategy.mode == "baseline"
        else PreviewVelocityPolicy(
            kinematics,
            side,
            strategy,
            guard_scale=guard_scale,
            soft_ratio=soft_ratio,
        )
    )

    count = target_pose.shape[0]
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
    reference_velocity = np.empty((count, 7), dtype=np.float64)
    flexible_velocity = np.empty((count, 7), dtype=np.float64)
    final_velocity = np.empty((count, 7), dtype=np.float64)
    scale = np.empty(count, dtype=np.float64)
    augmented_progress = np.empty(count, dtype=np.float64)
    preview_peak_utilization = np.empty(count, dtype=np.float64)
    flexible_peak_utilization = np.empty(count, dtype=np.float64)
    exact_null_leak = np.empty(count, dtype=np.float64)
    near_null_leak = np.empty(count, dtype=np.float64)
    residual_norm = np.empty(count, dtype=np.float64)
    soft_slack_norm = np.empty(count, dtype=np.float64)
    policy_failed = np.zeros(count, dtype=bool)

    previous_command = dynamics.q()
    caps = np.asarray(
        ARM_JOINT_VELOCITY_LIMITS_RAD_S,
        dtype=np.float64,
    )
    for tick in range(count):
        measured_q = dynamics.q()
        measured_dq = dynamics.dq()
        kinematics.update_measured_state(
            dynamics.driver_qpos(),
            dynamics.driver_qvel(),
        )
        kinematics.set_target(side, target_pose[tick])
        step = (
            _baseline_step(kinematics, side)
            if policy is None
            else policy.solve(measured_q, measured_dq)
        )
        if step.command is None:
            current_command = previous_command.copy()
            solve_failed[tick] = True
        else:
            current_command = step.command
        current_dq = (current_command - previous_command) / CONTROL_DT

        dynamics.set_command(current_command)
        dynamics.step()
        current_command_pose = kinematics.fk(
            side,
            np.append(current_command, 0.0),
        ).astype(np.float64)
        values = diagnostics.evaluate(current_command, current_dq)

        command_q[tick] = current_command
        command_dq[tick] = current_dq
        actual_q[tick] = dynamics.q()
        actual_dq[tick] = dynamics.dq()
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
        reference_velocity[tick] = step.reference_velocity
        flexible_velocity[tick] = step.flexible_velocity
        final_velocity[tick] = step.final_velocity
        scale[tick] = step.scale
        augmented_progress[tick] = step.augmented_progress
        preview_peak_utilization[tick] = step.preview_peak_utilization
        flexible_peak_utilization[tick] = step.flexible_peak_utilization
        exact_null_leak[tick] = step.exact_null_leak
        near_null_leak[tick] = step.near_null_leak
        residual_norm[tick] = step.residual_norm
        soft_slack_norm[tick] = step.soft_slack_norm
        policy_failed[tick] = step.failed
        previous_command = current_command

    trace = ReplayTrace(
        time=np.arange(count, dtype=np.float64) * CONTROL_DT,
        target_pose=target_pose,
        ik_target_pose=target_pose.copy(),
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
    policy_diagnostics = PolicyDiagnostics(
        reference_velocity=reference_velocity,
        flexible_velocity=flexible_velocity,
        final_velocity=final_velocity,
        scale=scale,
        augmented_progress=augmented_progress,
        preview_peak_utilization=preview_peak_utilization,
        flexible_peak_utilization=flexible_peak_utilization,
        exact_null_leak=exact_null_leak,
        near_null_leak=near_null_leak,
        residual_norm=residual_norm,
        soft_slack_norm=soft_slack_norm,
        policy_failed=policy_failed,
    )
    return trace, policy_diagnostics


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _nearest_path_rmse(
    pose: np.ndarray,
    elbow: np.ndarray,
    reference_pose: np.ndarray,
    reference_elbow: np.ndarray,
) -> tuple[float, float]:
    tree = cKDTree(reference_pose[:, :3])
    distance, indices = tree.query(pose[:, :3])
    elbow_error = elbow - reference_elbow[indices]
    return (
        _rms(np.linalg.norm(elbow_error, axis=1)),
        _rms(distance),
    )


def _summarize(
    case_name: int | str,
    strategy: Strategy,
    trace: ReplayTrace,
    diagnostics: PolicyDiagnostics,
    reference: ReplayTrace,
    core: slice,
) -> dict[str, float | int | str]:
    position_error = np.linalg.norm(
        trace.actual_pose[:, :3] - trace.target_pose[:, :3],
        axis=1,
    )
    orientation_error = _orientation_error(
        trace.target_pose,
        trace.actual_pose,
    )
    elbow_difference = trace.actual_elbow - reference.actual_elbow
    q_difference = trace.actual_q - reference.actual_q
    command_path_elbow_rmse, command_path_position_gap = _nearest_path_rmse(
        trace.command_pose[core],
        trace.command_elbow[core],
        reference.command_pose[core],
        reference.command_elbow[core],
    )
    actual_path_elbow_rmse, actual_path_position_gap = _nearest_path_rmse(
        trace.actual_pose[core],
        trace.actual_elbow[core],
        reference.actual_pose[core],
        reference.actual_elbow[core],
    )
    command_acceleration = np.gradient(
        trace.command_dq,
        CONTROL_DT,
        axis=0,
    )
    actual_acceleration = np.gradient(
        trace.actual_dq,
        CONTROL_DT,
        axis=0,
    )
    return {
        "case": case_name,
        "strategy": strategy.name,
        "mode": strategy.mode,
        "soft_weight": strategy.soft_weight,
        "scale_mean": float(np.mean(diagnostics.scale[core])),
        "scale_min": float(np.min(diagnostics.scale[core])),
        "scale_active_fraction": float(
            np.mean(diagnostics.scale[core] < 1.0 - 1e-6)
        ),
        "augmented_progress_mean": float(
            np.mean(diagnostics.augmented_progress[core])
        ),
        "preview_peak_utilization": float(
            np.max(diagnostics.preview_peak_utilization[core])
        ),
        "flexible_peak_utilization": float(
            np.max(diagnostics.flexible_peak_utilization[core])
        ),
        "final_peak_utilization": float(
            np.max(trace.velocity_utilization[core])
        ),
        "exact_null_leak_rms_rad_s": _rms(
            diagnostics.exact_null_leak[core]
        ),
        "near_null_leak_rms_rad_s": _rms(
            diagnostics.near_null_leak[core]
        ),
        "residual_rms_rad_s": _rms(diagnostics.residual_norm[core]),
        "soft_slack_rms_rad_s": _rms(
            diagnostics.soft_slack_norm[core]
        ),
        "elbow_y_rmse_to_reference_m": _rms(
            elbow_difference[core, 1]
        ),
        "elbow_xyz_rmse_to_reference_m": _rms(
            np.linalg.norm(elbow_difference[core], axis=1)
        ),
        "q_rmse_to_reference_rad": _rms(q_difference[core]),
        "command_path_elbow_rmse_m": command_path_elbow_rmse,
        "command_path_position_gap_m": command_path_position_gap,
        "actual_path_elbow_rmse_m": actual_path_elbow_rmse,
        "actual_path_position_gap_m": actual_path_position_gap,
        "position_rmse_m": _rms(position_error[core]),
        "position_peak_m": float(np.max(position_error[core])),
        "orientation_rmse_rad": _rms(orientation_error[core]),
        "j1_command_peak_rad_s": float(
            np.max(np.abs(trace.command_dq[core, 0]))
        ),
        "j4_command_peak_rad_s": float(
            np.max(np.abs(trace.command_dq[core, 3]))
        ),
        "j1_command_accel_p99_rad_s2": float(
            np.percentile(np.abs(command_acceleration[core, 0]), 99.0)
        ),
        "j1_actual_accel_p99_rad_s2": float(
            np.percentile(np.abs(actual_acceleration[core, 0]), 99.0)
        ),
        "solve_failures": int(np.count_nonzero(trace.solve_failed)),
        "policy_failures": int(
            np.count_nonzero(diagnostics.policy_failed)
        ),
    }


def _save_policy_trace(
    path: Path,
    trace: ReplayTrace,
    diagnostics: PolicyDiagnostics,
) -> None:
    values = dict(trace.__dict__)
    values.update(
        {
            f"policy_{name}": value
            for name, value in diagnostics.__dict__.items()
        }
    )
    np.savez_compressed(path, **values)


def _plot_case(
    path: Path,
    traces: dict[str, ReplayTrace],
    diagnostics: dict[str, PolicyDiagnostics],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    figure, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    for name, trace in traces.items():
        axes[0].plot(trace.time, trace.command_dq[:, 0], label=name)
        axes[1].plot(trace.time, trace.command_dq[:, 3], label=name)
        axes[2].plot(trace.time, trace.actual_elbow[:, 1], label=name)
        axes[3].plot(trace.time, diagnostics[name].scale, label=name)
        axes[4].plot(
            trace.time,
            np.linalg.norm(
                trace.actual_pose[:, :3] - trace.target_pose[:, :3],
                axis=1,
            ),
            label=name,
        )
    axes[0].set_ylabel("J1 command [rad/s]")
    axes[1].set_ylabel("J4 command [rad/s]")
    axes[2].set_ylabel("Elbow y [m]")
    axes[3].set_ylabel("Outer scale")
    axes[4].set_ylabel("EEF pos error [m]")
    axes[4].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _number_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--episode", type=int, default=202)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--segments", type=int, nargs="*", default=[13])
    parser.add_argument("--reverse-segments", type=int, nargs="*", default=[])
    parser.add_argument("--circle-speeds", type=float, nargs="*", default=[])
    parser.add_argument("--circle-radius", type=float, default=0.06)
    parser.add_argument("--extension-speed", type=float, default=0.15)
    parser.add_argument("--circle-ramp-duration", type=float, default=0.3)
    parser.add_argument("--circle-seed-segment", type=int, default=13)
    parser.add_argument("--settle-duration", type=float, default=0.5)
    parser.add_argument("--guard-scale", type=float, default=4.0)
    parser.add_argument("--soft-ratio", type=float, default=0.8)
    parser.add_argument("--strategies", nargs="*", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "dev/results/velocity_time_scaling_comparison_20260724"
        ),
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
    selected = {segment.index: segment for segment in segments}
    strategies = [
        strategy
        for strategy in _strategies()
        if not args.strategies or strategy.name in set(args.strategies)
    ]
    if not strategies:
        raise ValueError("No strategies selected.")

    cases: list[
        tuple[int | str, np.ndarray, np.ndarray, np.ndarray, slice]
    ] = []
    for segment_index in args.segments:
        segment = selected[segment_index]
        window = slice(segment.start, segment.end)
        core = slice(
            segment.core_start - segment.start,
            segment.core_end - segment.start,
        )
        cases.append(
            (
                segment_index,
                target_pose[window],
                source_q[window],
                source_elbow[window],
                core,
            )
        )
    for segment_index in args.reverse_segments:
        segment = selected[segment_index]
        window = slice(segment.start, segment.end)
        count = segment.end - segment.start
        original_core_start = segment.core_start - segment.start
        original_core_end = segment.core_end - segment.start
        core = slice(
            count - original_core_end,
            count - original_core_start,
        )
        cases.append(
            (
                f"reverse_{segment_index}",
                target_pose[window][::-1].copy(),
                source_q[window][::-1].copy(),
                source_elbow[window][::-1].copy(),
                core,
            )
        )
    if args.circle_speeds:
        seed = selected[args.circle_seed_segment]
        for speed in args.circle_speeds:
            circle_target, circle_q, circle_elbow, core = (
                _extended_circle_inputs(
                    speed,
                    args.circle_radius,
                    args.extension_speed,
                    args.circle_ramp_duration,
                    seed,
                    target_pose,
                    source_q,
                    shoulder,
                    source_elbow,
                )
            )
            cases.append(
                (
                    f"circle_{_number_label(speed)}mps",
                    circle_target,
                    circle_q,
                    circle_elbow,
                    core,
                )
            )

    rows: list[dict[str, float | int | str]] = []
    for case_name, case_target, case_q, case_elbow, core in cases:
        traces: dict[str, ReplayTrace] = {}
        policy_traces: dict[str, PolicyDiagnostics] = {}
        for strategy in strategies:
            print(f"Simulating case={case_name}, strategy={strategy.name}...")
            trace, policy_trace = _simulate(
                args.side,
                strategy,
                case_target,
                case_q,
                case_elbow,
                settle_duration=args.settle_duration,
                guard_scale=args.guard_scale,
                soft_ratio=args.soft_ratio,
            )
            traces[strategy.name] = trace
            policy_traces[strategy.name] = policy_trace
            _save_policy_trace(
                args.output_dir
                / f"trace_{case_name}_{strategy.name}.npz",
                trace,
                policy_trace,
            )
        reference_name = "N_guard_reference"
        if reference_name not in traces:
            raise ValueError("N_guard_reference is required.")
        reference = traces[reference_name]
        for strategy in strategies:
            rows.append(
                _summarize(
                    case_name,
                    strategy,
                    traces[strategy.name],
                    policy_traces[strategy.name],
                    reference,
                    core,
                )
            )
        _plot_case(
            args.output_dir / f"{case_name}.png",
            traces,
            policy_traces,
        )

    with (args.output_dir / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
