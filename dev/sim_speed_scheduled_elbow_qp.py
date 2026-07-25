#!/usr/bin/env python3
"""Evaluate a speed-scheduled elbow corridor in an augmented Mink QP.

This is an offline development experiment. It deliberately does not modify the
runtime IK path. The augmented decision vector is

    x = [dq, task_scale, cartesian_slack, elbow_corridor_slack]

where ``dq`` is Mink's tangent displacement for one IK substep.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mink
import mujoco
import numpy as np
import qpsolvers
from scipy.spatial.transform import Rotation

from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S

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
from sim_modal_shoulder_limit_comparison import _extended_circle_inputs
from sim_velocity_time_scaling_comparison import (
    Strategy,
    _nearest_path_rmse,
    _simulate,
)


@dataclass(frozen=True)
class ElbowQpParams:
    """Physical and normalized parameters for one augmented-QP candidate."""

    name: str
    linear_fast_m_s: float = 0.6
    angular_fast_rad_s: float = 4.0
    speed_ratio_slow: float = 0.65
    speed_ratio_fast: float = 1.0
    activation_rise_rate_s_inv: float = 8.0
    activation_fall_rate_s_inv: float = 4.0
    task_feedback_gain_s_inv: float = 10.0
    task_command_linear_limit_m_s: float = 1.0
    task_command_angular_limit_rad_s: float = 8.0
    task_command_mode: str = "twist"
    task_error_position_leak_m: float = 0.0004
    task_error_orientation_leak_rad: float = 0.0032
    reference_mode: str = "initial"
    corridor_mode: str = "symmetric"
    corridor_anchor_mode: str = "current"
    corridor_latch_alpha: float = 0.8
    corridor_release_alpha: float = 0.05
    corridor_margin_power: float = 2.0
    corridor_slow_rad: float = np.deg2rad(25.0)
    corridor_fast_rad: float = np.deg2rad(5.0)
    corridor_tighten_rad_s: float = np.deg2rad(120.0)
    corridor_relax_rad_s: float = np.deg2rad(60.0)
    elbow_return_gain_s_inv: float = 0.5
    elbow_return_max_rad_s: float = 0.3
    elbow_return_scale_fast: float = 0.0
    elbow_velocity_scale_rad_s: float = 0.25
    elbow_velocity_weight_slow: float = 0.0
    elbow_velocity_weight_fast: float = 20.0
    nullspace_cost_scale_fast: float = 1.0
    task_scale_weight: float = 10.0
    cartesian_slack_position_scale_m: float = 0.002
    cartesian_slack_orientation_scale_rad: float = 0.01
    cartesian_slack_weight_slow: float = 0.0
    cartesian_slack_weight_fast: float = 1e6
    corridor_slack_scale_rad: float = np.deg2rad(1.0)
    corridor_slack_linear_slow: float = 0.0
    corridor_slack_linear_fast: float = 1.0
    corridor_slack_quadratic_slow: float = 0.0
    corridor_slack_quadratic_fast: float = 10.0
    swivel_min_radius_m: float = 0.02
    swivel_fd_epsilon_rad: float = 1e-5


@dataclass
class ElbowQpDiagnostics:
    """Signals specific to the augmented elbow QP."""

    alpha_v: np.ndarray
    nominal_linear_speed: np.ndarray
    nominal_angular_speed: np.ndarray
    corridor_width: np.ndarray
    task_scale_mean: np.ndarray
    task_scale_min: np.ndarray
    cartesian_position_slack: np.ndarray
    cartesian_orientation_slack: np.ndarray
    elbow_corridor_slack: np.ndarray
    command_swivel: np.ndarray
    actual_swivel: np.ndarray
    swivel_radius: np.ndarray
    qp_failed: np.ndarray


@dataclass(frozen=True)
class AugmentedStep:
    """One augmented-QP command and diagnostics."""

    command: np.ndarray | None
    velocity: np.ndarray
    alpha_v: float
    nominal_linear_speed: float
    nominal_angular_speed: float
    corridor_width: float
    task_scale_mean: float
    task_scale_min: float
    cartesian_position_slack: float
    cartesian_orientation_slack: float
    elbow_corridor_slack: float
    swivel: float
    swivel_radius: float
    failed: bool


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _lerp(low: float, high: float, alpha: float) -> float:
    return (1.0 - alpha) * low + alpha * high


def _wrapped_difference(lhs: float, rhs: float) -> float:
    return float(np.arctan2(np.sin(lhs - rhs), np.cos(lhs - rhs)))


def _limit_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    output = np.asarray(vector, dtype=np.float64).copy()
    norm = float(np.linalg.norm(output))
    if norm > limit:
        output *= limit / norm
    return output


class ElbowSwivelCoordinate:
    """Elbow swivel around the shoulder-to-end-effector axis."""

    def __init__(
        self,
        model: mujoco.MjModel,
        side: str,
        frame_task: mink.FrameTask,
        reference_q: np.ndarray,
        dof_indices: np.ndarray,
        *,
        finite_difference_epsilon: float,
    ) -> None:
        self.model = model
        self.side = side
        self.frame_task = frame_task
        self.dof_indices = np.asarray(dof_indices, dtype=int)
        self.epsilon = float(finite_difference_epsilon)
        self.data = mujoco.MjData(model)
        self.shoulder_joint = model.joint(
            f"openarm_{side}_joint1"
        ).id
        self.elbow_joint = model.joint(f"openarm_{side}_joint4").id
        object_type = {
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "site": mujoco.mjtObj.mjOBJ_SITE,
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
        }[frame_task.frame_type]
        self.frame_id = mujoco.mj_name2id(
            model,
            object_type,
            frame_task.frame_name,
        )
        _, _, _, reference_normal, radius = self._geometry(reference_q)
        if radius <= 1e-8:
            raise ValueError("Elbow swivel reference is degenerate.")
        self.reference_normal = reference_normal

    def _frame_position(self) -> np.ndarray:
        if self.frame_task.frame_type == "body":
            return self.data.xpos[self.frame_id]
        if self.frame_task.frame_type == "site":
            return self.data.site_xpos[self.frame_id]
        return self.data.geom_xpos[self.frame_id]

    def _geometry(
        self,
        q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        shoulder = self.data.xanchor[self.shoulder_joint].copy()
        elbow = self.data.xanchor[self.elbow_joint].copy()
        wrist = self._frame_position().copy()
        axis = wrist - shoulder
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-8:
            raise ValueError("Shoulder-to-wrist axis is degenerate.")
        axis /= axis_norm
        elbow_vector = elbow - shoulder
        elbow_normal = elbow_vector - axis * float(axis @ elbow_vector)
        radius = float(np.linalg.norm(elbow_normal))
        if radius > 1e-10:
            elbow_normal /= radius
        return shoulder, elbow, wrist, elbow_normal, radius

    def value(self, q: np.ndarray) -> tuple[float, float]:
        _, _, _, elbow_normal, radius = self._geometry(q)
        reference = self.reference_normal.copy()
        # Transport the clutch-time normal by projecting it onto the current
        # shoulder-wrist normal plane.
        shoulder = self.data.xanchor[self.shoulder_joint]
        wrist = self._frame_position()
        axis = wrist - shoulder
        axis /= np.linalg.norm(axis)
        reference = reference - axis * float(axis @ reference)
        reference_norm = float(np.linalg.norm(reference))
        if reference_norm <= 1e-10 or radius <= 1e-10:
            return 0.0, radius
        reference /= reference_norm
        sine = float(axis @ np.cross(reference, elbow_normal))
        cosine = float(reference @ elbow_normal)
        return float(np.arctan2(sine, cosine)), radius

    def linearize(
        self,
        q: np.ndarray,
    ) -> tuple[float, np.ndarray, float]:
        value, radius = self.value(q)
        jacobian = np.zeros(self.model.nv, dtype=np.float64)
        tangent = np.zeros(self.model.nv, dtype=np.float64)
        for dof in self.dof_indices:
            tangent.fill(0.0)
            tangent[dof] = 1.0
            q_plus = q.copy()
            q_minus = q.copy()
            mujoco.mj_integratePos(
                self.model,
                q_plus,
                tangent,
                self.epsilon,
            )
            mujoco.mj_integratePos(
                self.model,
                q_minus,
                tangent,
                -self.epsilon,
            )
            plus, _ = self.value(q_plus)
            minus, _ = self.value(q_minus)
            jacobian[dof] = (
                _wrapped_difference(plus, minus)
                / (2.0 * self.epsilon)
            )
        return value, jacobian, radius


def _extend_rows(
    matrix: np.ndarray | None,
    variable_count: int,
    displacement_scale: np.ndarray,
) -> np.ndarray | None:
    if matrix is None:
        return None
    output = np.zeros((matrix.shape[0], variable_count), dtype=np.float64)
    output[:, : matrix.shape[1]] = (
        matrix * displacement_scale[None, :]
    )
    return output


def _stack_rows(
    base: np.ndarray | None,
    extra: np.ndarray,
) -> np.ndarray:
    return extra if base is None else np.vstack([base, extra])


def _stack_values(
    base: np.ndarray | None,
    extra: np.ndarray,
) -> np.ndarray:
    return extra if base is None else np.hstack([base, extra])


def _solve_problem(problem: qpsolvers.Problem) -> np.ndarray | None:
    hessian = np.asarray(problem.P, dtype=np.float64)
    hessian = 0.5 * (hessian + hessian.T)
    hessian += np.eye(hessian.shape[0], dtype=np.float64) * 1e-9
    result = qpsolvers.solve_problem(
        qpsolvers.Problem(
            hessian,
            np.asarray(problem.q, dtype=np.float64),
            None if problem.G is None else np.asarray(problem.G),
            None if problem.h is None else np.asarray(problem.h),
            None if problem.A is None else np.asarray(problem.A),
            None if problem.b is None else np.asarray(problem.b),
        ),
        solver="daqp",
    )
    if not result.found or result.x is None:
        return None
    return np.asarray(result.x, dtype=np.float64)


def _add_squared_residual(
    hessian: np.ndarray,
    linear: np.ndarray,
    row: np.ndarray,
    target: float,
    weight: float,
) -> None:
    if weight <= 0.0:
        return
    hessian += weight * np.outer(row, row)
    linear -= weight * target * row


def _augment_problem(
    base: qpsolvers.Problem,
    frame_task: mink.FrameTask,
    configuration: mink.Configuration,
    elbow_jacobian: np.ndarray,
    elbow_error: float,
    elbow_desired_displacement: float,
    corridor_lower: float,
    corridor_upper: float,
    alpha_v: float,
    nominal_task_velocity: np.ndarray,
    displacement_scale: np.ndarray,
    dt: float,
    params: ElbowQpParams,
    *,
    elbow_active: bool,
) -> tuple[qpsolvers.Problem, dict[str, int | slice | np.ndarray]]:
    nv = base.q.shape[0]
    scale_index = nv
    cartesian_slice = slice(scale_index + 1, scale_index + 7)
    elbow_slack_index = cartesian_slice.stop
    variable_count = elbow_slack_index + 1

    hessian = np.zeros((variable_count, variable_count), dtype=np.float64)
    linear = np.zeros(variable_count, dtype=np.float64)
    base_hessian = np.asarray(base.P, dtype=np.float64)
    hessian[:nv, :nv] = (
        displacement_scale[:, None]
        * base_hessian
        * displacement_scale[None, :]
    )
    linear[:nv] = (
        displacement_scale * np.asarray(base.q, dtype=np.float64)
    )

    hessian[scale_index, scale_index] += 2.0 * params.task_scale_weight
    linear[scale_index] -= 2.0 * params.task_scale_weight

    cartesian_slack_weight = _lerp(
        params.cartesian_slack_weight_slow,
        params.cartesian_slack_weight_fast,
        alpha_v,
    )
    hessian[
        cartesian_slice,
        cartesian_slice,
    ] += cartesian_slack_weight * np.eye(6)

    # At alpha_v=0 this reproduces the original Mink FrameTask objective.
    # At alpha_v=1 the explicit task equality and scale variable take over.
    frame_objective = frame_task.compute_qp_objective(configuration)
    frame_weight = 1.0 - alpha_v
    frame_hessian = np.asarray(frame_objective.H, dtype=np.float64)
    hessian[:nv, :nv] += frame_weight * (
        displacement_scale[:, None]
        * frame_hessian
        * displacement_scale[None, :]
    )
    linear[:nv] += frame_weight * displacement_scale * np.asarray(
        frame_objective.c,
        dtype=np.float64,
    )

    velocity_weight = _lerp(
        params.elbow_velocity_weight_slow,
        params.elbow_velocity_weight_fast,
        alpha_v,
    )
    velocity_normalizer = params.elbow_velocity_scale_rad_s * dt
    if elbow_active and velocity_normalizer > 0.0:
        velocity_row = np.zeros(variable_count, dtype=np.float64)
        velocity_row[:nv] = (
            elbow_jacobian * displacement_scale / velocity_normalizer
        )
        _add_squared_residual(
            hessian,
            linear,
            velocity_row,
            elbow_desired_displacement / velocity_normalizer,
            velocity_weight,
        )

    slack_linear = _lerp(
        params.corridor_slack_linear_slow,
        params.corridor_slack_linear_fast,
        alpha_v,
    )
    slack_quadratic = _lerp(
        params.corridor_slack_quadratic_slow,
        params.corridor_slack_quadratic_fast,
        alpha_v,
    )
    hessian[elbow_slack_index, elbow_slack_index] += slack_quadratic
    linear[elbow_slack_index] += slack_linear

    frame_jacobian = frame_task.compute_jacobian(configuration)
    task_error = frame_task.compute_error(configuration)
    if params.task_command_mode == "twist":
        task_velocity = (
            nominal_task_velocity
            + params.task_feedback_gain_s_inv * task_error
        )
        task_velocity[:3] = _limit_norm(
            task_velocity[:3],
            params.task_command_linear_limit_m_s,
        )
        task_velocity[3:] = _limit_norm(
            task_velocity[3:],
            params.task_command_angular_limit_rad_s,
        )
        # FrameTask's error Jacobian has the opposite sign of the geometric
        # frame Jacobian, hence the minus sign for a desired physical twist.
        task_displacement = -task_velocity * dt
    elif params.task_command_mode == "error_leak":
        task_displacement = -task_error.copy()
        task_displacement[:3] = _limit_norm(
            task_displacement[:3],
            params.task_error_position_leak_m,
        )
        task_displacement[3:] = _limit_norm(
            task_displacement[3:],
            params.task_error_orientation_leak_rad,
        )
    else:
        raise ValueError(
            f"Unsupported task command mode: {params.task_command_mode}"
        )
    row_scale = np.array(
        [
            1.0 / CHARACTERISTIC_LENGTH,
            1.0 / CHARACTERISTIC_LENGTH,
            1.0 / CHARACTERISTIC_LENGTH,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float64,
    )
    normalized_jacobian = row_scale[:, None] * frame_jacobian
    normalized_task = row_scale * task_displacement
    cartesian_variable_scale = row_scale * np.array(
        [
            params.cartesian_slack_position_scale_m,
            params.cartesian_slack_position_scale_m,
            params.cartesian_slack_position_scale_m,
            params.cartesian_slack_orientation_scale_rad,
            params.cartesian_slack_orientation_scale_rad,
            params.cartesian_slack_orientation_scale_rad,
        ],
        dtype=np.float64,
    )

    equality = _extend_rows(
        base.A,
        variable_count,
        displacement_scale,
    )
    equality_values = (
        None if base.b is None else np.asarray(base.b, dtype=np.float64)
    )
    task_rows = np.zeros((6, variable_count), dtype=np.float64)
    task_rows[:, :nv] = (
        normalized_jacobian * displacement_scale[None, :]
    )
    task_rows[:, scale_index] = -normalized_task
    task_rows[:, cartesian_slice] = -np.diag(cartesian_variable_scale)
    equality = _stack_rows(equality, task_rows)
    equality_values = _stack_values(
        equality_values,
        np.zeros(6, dtype=np.float64),
    )

    inequality = _extend_rows(
        base.G,
        variable_count,
        displacement_scale,
    )
    inequality_values = (
        None if base.h is None else np.asarray(base.h, dtype=np.float64)
    )
    bounds = np.zeros((3, variable_count), dtype=np.float64)
    bounds[0, scale_index] = 1.0
    bounds[1, scale_index] = -1.0
    bounds[2, elbow_slack_index] = -1.0
    inequality = _stack_rows(inequality, bounds)
    inequality_values = _stack_values(
        inequality_values,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )

    if elbow_active:
        corridor_scale = params.corridor_slack_scale_rad
        corridor_rows = np.zeros((2, variable_count), dtype=np.float64)
        corridor_rows[0, :nv] = elbow_jacobian * displacement_scale
        corridor_rows[0, elbow_slack_index] = -corridor_scale
        corridor_rows[1, :nv] = -elbow_jacobian * displacement_scale
        corridor_rows[1, elbow_slack_index] = -corridor_scale
        corridor_values = np.array(
            [
                corridor_upper - elbow_error,
                elbow_error - corridor_lower,
            ],
            dtype=np.float64,
        )
        inequality = _stack_rows(inequality, corridor_rows)
        inequality_values = _stack_values(
            inequality_values,
            corridor_values,
        )

    problem = qpsolvers.Problem(
        hessian,
        linear,
        inequality,
        inequality_values,
        equality,
        equality_values,
    )
    indices: dict[str, int | slice | np.ndarray] = {
        "scale": scale_index,
        "cartesian": cartesian_slice,
        "elbow_slack": elbow_slack_index,
        "cartesian_variable_scale": cartesian_variable_scale,
    }
    return problem, indices


class SpeedScheduledElbowPolicy:
    """Solve the explicit task-scale/slack/elbow augmented QP."""

    def __init__(
        self,
        kinematics,
        side: str,
        params: ElbowQpParams,
    ) -> None:
        self.kinematics = kinematics
        self.side = side
        self.params = params
        solver = kinematics._ik
        assert solver is not None
        self.solver = solver
        self.model = solver._model
        self.frame_task = solver._tasks[side]
        self.dof_indices = np.asarray(
            solver._arm_dofs_by_side[side],
            dtype=int,
        )
        self.displacement_scale = np.full(
            self.model.nv,
            solver._substep_dt,
            dtype=np.float64,
        )
        self.displacement_scale[self.dof_indices] = (
            np.asarray(
                ARM_JOINT_VELOCITY_LIMITS_RAD_S,
                dtype=np.float64,
            )
            * solver._substep_dt
        )
        if params.reference_mode == "initial":
            reference_q = solver._config.q.copy()
        elif params.reference_mode == "home":
            reference_q = solver._posture_task.target_q.copy()
        else:
            raise ValueError(
                f"Unsupported elbow reference mode: {params.reference_mode}"
            )
        self.coordinate = ElbowSwivelCoordinate(
            self.model,
            side,
            self.frame_task,
            reference_q,
            self.dof_indices,
            finite_difference_epsilon=params.swivel_fd_epsilon_rad,
        )
        self.corridor_width = params.corridor_slow_rad
        self.corridor_lower = -self.corridor_width
        self.corridor_upper = self.corridor_width
        self.nullspace_task = solver._nullspace_tasks.get(side)
        self.nullspace_base_cost = (
            None
            if self.nullspace_task is None
            else float(self.nullspace_task._base_cost)
        )
        self.previous_target: np.ndarray | None = None
        self.alpha_v = 0.0
        self.pre_fast_swivel: float | None = None
        self.fast_anchor_swivel: float | None = None

    def _target_speed(
        self,
        target: np.ndarray,
    ) -> tuple[float, float, float, np.ndarray]:
        if self.previous_target is None:
            self.previous_target = target.copy()
            return 0.0, 0.0, 0.0, np.zeros(6, dtype=np.float64)
        previous_rotation = Rotation.from_quat(
            self.previous_target[[4, 5, 6, 3]]
        )
        current_rotation = Rotation.from_quat(target[[4, 5, 6, 3]])
        linear_world = (
            target[:3] - self.previous_target[:3]
        ) / CONTROL_DT
        linear_local = current_rotation.inv().apply(linear_world)
        angular_local = (
            previous_rotation.inv() * current_rotation
        ).as_rotvec() / CONTROL_DT
        nominal_task_velocity = np.concatenate(
            [linear_local, angular_local]
        )
        linear_speed = float(np.linalg.norm(linear_local))
        angular_speed = float(np.linalg.norm(angular_local))
        self.previous_target = target.copy()
        speed_ratio = max(
            linear_speed / self.params.linear_fast_m_s,
            angular_speed / self.params.angular_fast_rad_s,
        )
        denominator = (
            self.params.speed_ratio_fast
            - self.params.speed_ratio_slow
        )
        activation_input = (
            speed_ratio - self.params.speed_ratio_slow
        ) / denominator
        target_alpha = _smoothstep(activation_input)
        difference = target_alpha - self.alpha_v
        rate = (
            self.params.activation_rise_rate_s_inv
            if difference > 0.0
            else self.params.activation_fall_rate_s_inv
        )
        self.alpha_v += float(
            np.clip(
                difference,
                -rate * CONTROL_DT,
                rate * CONTROL_DT,
            )
        )
        return (
            linear_speed,
            angular_speed,
            self.alpha_v,
            nominal_task_velocity,
        )

    def _update_corridor(
        self,
        alpha_v: float,
        current_swivel: float,
    ) -> None:
        if self.params.corridor_mode == "home_interval":
            if self.params.corridor_anchor_mode == "latched":
                if (
                    self.alpha_v >= self.params.corridor_latch_alpha
                    and self.fast_anchor_swivel is None
                ):
                    self.fast_anchor_swivel = (
                        current_swivel
                        if self.pre_fast_swivel is None
                        else self.pre_fast_swivel
                    )
                elif (
                    self.fast_anchor_swivel is not None
                    and self.alpha_v <= self.params.corridor_release_alpha
                ):
                    self.fast_anchor_swivel = None
                if self.fast_anchor_swivel is None:
                    self.pre_fast_swivel = current_swivel
                anchor_swivel = (
                    current_swivel
                    if self.fast_anchor_swivel is None
                    else self.fast_anchor_swivel
                )
            elif self.params.corridor_anchor_mode == "current":
                anchor_swivel = current_swivel
            else:
                raise ValueError(
                    "Unsupported corridor anchor mode: "
                    f"{self.params.corridor_anchor_mode}"
                )
            residual = (1.0 - alpha_v) ** self.params.corridor_margin_power
            self.corridor_width = _lerp(
                self.params.corridor_fast_rad,
                self.params.corridor_slow_rad,
                residual,
            )
            self.corridor_lower = (
                min(0.0, anchor_swivel) - self.corridor_width
            )
            self.corridor_upper = (
                max(0.0, anchor_swivel) + self.corridor_width
            )
            return
        if self.params.corridor_mode != "symmetric":
            raise ValueError(
                f"Unsupported corridor mode: {self.params.corridor_mode}"
            )
        target = _lerp(
            self.params.corridor_slow_rad,
            self.params.corridor_fast_rad,
            alpha_v,
        )
        difference = target - self.corridor_width
        rate = (
            self.params.corridor_tighten_rad_s
            if difference < 0.0
            else self.params.corridor_relax_rad_s
        )
        self.corridor_width += float(
            np.clip(
                difference,
                -rate * CONTROL_DT,
                rate * CONTROL_DT,
            )
        )
        self.corridor_lower = -self.corridor_width
        self.corridor_upper = self.corridor_width

    def _update_nullspace_cost(self, alpha_v: float) -> None:
        if (
            self.nullspace_task is None
            or self.nullspace_base_cost is None
        ):
            return
        scale = _lerp(
            1.0,
            self.params.nullspace_cost_scale_fast,
            alpha_v,
        )
        self.nullspace_task._base_cost = self.nullspace_base_cost * scale

    def _secondary_tasks(self) -> list[mink.BaseTask]:
        tasks: list[mink.BaseTask] = []
        if self.solver._posture_cost > 0.0:
            tasks.append(self.solver._posture_task)
        if self.solver._kinetic_energy_task is not None:
            tasks.append(self.solver._kinetic_energy_task)
        tasks.extend(self.solver._nullspace_tasks.values())
        tasks.extend(self.solver._elbow_soft_limit_tasks.values())
        return tasks

    def solve(
        self,
        target: np.ndarray,
    ) -> AugmentedStep:
        q_start = self.solver._config.q.copy()
        (
            linear_speed,
            angular_speed,
            alpha_v,
            nominal_task_velocity,
        ) = self._target_speed(target)
        swivel, swivel_jacobian, radius = self.coordinate.linearize(q_start)
        self._update_corridor(alpha_v, swivel)
        self._update_nullspace_cost(alpha_v)
        elbow_active = radius >= self.params.swivel_min_radius_m

        return_velocity = float(
            np.clip(
                -self.params.elbow_return_gain_s_inv * swivel,
                -self.params.elbow_return_max_rad_s,
                self.params.elbow_return_max_rad_s,
            )
        )
        return_scale = _lerp(
            1.0,
            self.params.elbow_return_scale_fast,
            alpha_v,
        )
        desired_elbow_velocity = return_scale * return_velocity
        desired_elbow_displacement = (
            desired_elbow_velocity * self.solver._substep_dt
        )
        constraints = (
            [self.solver._freeze_task]
            if self.solver._freeze_task is not None
            else []
        )
        secondary_tasks = self._secondary_tasks()
        for limit in self.solver._singularity_limits.values():
            limit.prepare(self.solver._config)

        task_scales: list[float] = []
        position_slacks: list[float] = []
        orientation_slacks: list[float] = []
        elbow_slacks: list[float] = []
        for _ in range(self.solver._max_iters):
            base = mink.build_ik(
                self.solver._config,
                secondary_tasks,
                self.solver._substep_dt,
                damping=float(self.solver._solver_params["damping"]),
                limits=self.solver._limits,
                constraints=constraints,
            )
            current_swivel, current_radius = self.coordinate.value(
                self.solver._config.q
            )
            active = elbow_active and (
                current_radius >= self.params.swivel_min_radius_m
            )
            problem, indices = _augment_problem(
                base,
                self.frame_task,
                self.solver._config,
                swivel_jacobian,
                current_swivel,
                desired_elbow_displacement,
                self.corridor_lower,
                self.corridor_upper,
                alpha_v,
                nominal_task_velocity,
                self.displacement_scale,
                self.solver._substep_dt,
                self.params,
                elbow_active=active,
            )
            solution = _solve_problem(problem)
            if solution is None:
                self.solver._config.update(q=q_start)
                return self._failed_step(
                    alpha_v,
                    linear_speed,
                    angular_speed,
                    swivel,
                    radius,
                )
            displacement = (
                solution[: self.model.nv] * self.displacement_scale
            )
            self.solver._config.integrate_inplace(
                displacement / self.solver._substep_dt,
                self.solver._substep_dt,
            )
            scale_index = int(indices["scale"])
            cartesian_slice = indices["cartesian"]
            assert isinstance(cartesian_slice, slice)
            elbow_slack_index = int(indices["elbow_slack"])
            cartesian_scale = indices["cartesian_variable_scale"]
            assert isinstance(cartesian_scale, np.ndarray)
            cartesian_slack = (
                solution[cartesian_slice] * cartesian_scale
            )
            task_scales.append(float(solution[scale_index]))
            position_slacks.append(
                float(np.linalg.norm(cartesian_slack[:3]))
            )
            orientation_slacks.append(
                float(np.linalg.norm(cartesian_slack[3:]))
            )
            elbow_slacks.append(
                float(
                    solution[elbow_slack_index]
                    * self.params.corridor_slack_scale_rad
                )
            )

        q_end = self.solver._config.q.copy()
        full_velocity = np.empty(self.model.nv, dtype=np.float64)
        mujoco.mj_differentiatePos(
            self.model,
            full_velocity,
            CONTROL_DT,
            q_start,
            q_end,
        )
        command, _ = self.solver._joint_resolver.get_driver(q_end, self.side)
        return AugmentedStep(
            command=np.asarray(command, dtype=np.float64),
            velocity=full_velocity[self.dof_indices],
            alpha_v=alpha_v,
            nominal_linear_speed=linear_speed,
            nominal_angular_speed=angular_speed,
            corridor_width=self.corridor_width,
            task_scale_mean=float(np.mean(task_scales)),
            task_scale_min=float(np.min(task_scales)),
            cartesian_position_slack=float(np.max(position_slacks)),
            cartesian_orientation_slack=float(np.max(orientation_slacks)),
            elbow_corridor_slack=float(np.max(elbow_slacks)),
            swivel=swivel,
            swivel_radius=radius,
            failed=False,
        )

    def _failed_step(
        self,
        alpha_v: float,
        linear_speed: float,
        angular_speed: float,
        swivel: float,
        radius: float,
    ) -> AugmentedStep:
        return AugmentedStep(
            command=None,
            velocity=np.zeros(7, dtype=np.float64),
            alpha_v=alpha_v,
            nominal_linear_speed=linear_speed,
            nominal_angular_speed=angular_speed,
            corridor_width=self.corridor_width,
            task_scale_mean=0.0,
            task_scale_min=0.0,
            cartesian_position_slack=0.0,
            cartesian_orientation_slack=0.0,
            elbow_corridor_slack=0.0,
            swivel=swivel,
            swivel_radius=radius,
            failed=True,
        )


def _full_q_from_arm(
    kinematics,
    side: str,
    base_q: np.ndarray,
    arm_q: np.ndarray,
) -> np.ndarray:
    output = base_q.copy()
    kinematics.setup.joint_resolver.set_qpos(
        output,
        np.append(np.asarray(arm_q, dtype=np.float64), 0.0),
        side,
    )
    return output


def _simulate_augmented(
    side: str,
    params: ElbowQpParams,
    target_pose: np.ndarray,
    source_q: np.ndarray,
    source_elbow: np.ndarray,
    *,
    settle_duration: float,
) -> tuple[ReplayTrace, ElbowQpDiagnostics]:
    kinematics = _make_kinematics(side, 0.0)
    dynamics = DynamicSideArm(side, source_q[0])
    dynamics.settle(settle_duration)
    kinematics.sync(dynamics.driver_qpos())
    solver = kinematics._ik
    assert solver is not None
    base_q = solver._config.q.copy()
    policy = SpeedScheduledElbowPolicy(kinematics, side, params)
    diagnostics = CommandDiagnostics(kinematics, side)

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

    alpha_v = np.empty(count, dtype=np.float64)
    nominal_linear_speed = np.empty(count, dtype=np.float64)
    nominal_angular_speed = np.empty(count, dtype=np.float64)
    corridor_width = np.empty(count, dtype=np.float64)
    task_scale_mean = np.empty(count, dtype=np.float64)
    task_scale_min = np.empty(count, dtype=np.float64)
    cartesian_position_slack = np.empty(count, dtype=np.float64)
    cartesian_orientation_slack = np.empty(count, dtype=np.float64)
    elbow_corridor_slack = np.empty(count, dtype=np.float64)
    command_swivel = np.empty(count, dtype=np.float64)
    actual_swivel = np.empty(count, dtype=np.float64)
    swivel_radius = np.empty(count, dtype=np.float64)
    qp_failed = np.zeros(count, dtype=bool)

    previous_command = dynamics.q()
    caps = np.asarray(
        ARM_JOINT_VELOCITY_LIMITS_RAD_S,
        dtype=np.float64,
    )
    for tick in range(count):
        kinematics.update_measured_state(
            dynamics.driver_qpos(),
            dynamics.driver_qvel(),
        )
        kinematics.set_target(side, target_pose[tick])
        step = policy.solve(target_pose[tick])
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

        alpha_v[tick] = step.alpha_v
        nominal_linear_speed[tick] = step.nominal_linear_speed
        nominal_angular_speed[tick] = step.nominal_angular_speed
        corridor_width[tick] = step.corridor_width
        task_scale_mean[tick] = step.task_scale_mean
        task_scale_min[tick] = step.task_scale_min
        cartesian_position_slack[tick] = step.cartesian_position_slack
        cartesian_orientation_slack[
            tick
        ] = step.cartesian_orientation_slack
        elbow_corridor_slack[tick] = step.elbow_corridor_slack
        command_full = _full_q_from_arm(
            kinematics,
            side,
            base_q,
            current_command,
        )
        actual_full = _full_q_from_arm(
            kinematics,
            side,
            base_q,
            dynamics.q(),
        )
        command_swivel[tick], _ = policy.coordinate.value(command_full)
        actual_swivel[tick], _ = policy.coordinate.value(actual_full)
        swivel_radius[tick] = step.swivel_radius
        qp_failed[tick] = step.failed
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
    extra = ElbowQpDiagnostics(
        alpha_v=alpha_v,
        nominal_linear_speed=nominal_linear_speed,
        nominal_angular_speed=nominal_angular_speed,
        corridor_width=corridor_width,
        task_scale_mean=task_scale_mean,
        task_scale_min=task_scale_min,
        cartesian_position_slack=cartesian_position_slack,
        cartesian_orientation_slack=cartesian_orientation_slack,
        elbow_corridor_slack=elbow_corridor_slack,
        command_swivel=command_swivel,
        actual_swivel=actual_swivel,
        swivel_radius=swivel_radius,
        qp_failed=qp_failed,
    )
    return trace, extra


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _p99_acceleration(values: np.ndarray) -> float:
    acceleration = np.gradient(values, CONTROL_DT)
    return float(np.quantile(np.abs(acceleration), 0.99))


def _summarize_augmented(
    case: int | str,
    params: ElbowQpParams,
    trace: ReplayTrace,
    diagnostics: ElbowQpDiagnostics,
    reference: ReplayTrace,
    core: slice,
) -> dict[str, float | int | str]:
    actual_path_elbow_rmse, actual_path_position_gap = _nearest_path_rmse(
        trace.actual_pose[core, :3],
        trace.actual_elbow[core],
        reference.actual_pose[core, :3],
        reference.actual_elbow[core],
    )
    position_error = np.linalg.norm(
        trace.actual_pose[core, :3] - trace.target_pose[core, :3],
        axis=1,
    )
    orientation_error = _orientation_error(
        trace.target_pose[core],
        trace.actual_pose[core],
    )
    swivel = diagnostics.actual_swivel[core]
    return {
        "case": case,
        "strategy": params.name,
        "actual_path_elbow_rmse_m": actual_path_elbow_rmse,
        "actual_path_position_gap_m": actual_path_position_gap,
        "actual_position_rmse_m": _rms(position_error),
        "actual_orientation_rmse_rad": _rms(orientation_error),
        "actual_swivel_delta_deg": float(
            np.rad2deg(swivel[-1] - swivel[0])
        ),
        "actual_swivel_range_deg": float(np.rad2deg(np.ptp(swivel))),
        "alpha_mean": float(np.mean(diagnostics.alpha_v[core])),
        "alpha_high_fraction": float(
            np.mean(diagnostics.alpha_v[core] > 0.8)
        ),
        "task_scale_mean": float(
            np.mean(diagnostics.task_scale_mean[core])
        ),
        "task_scale_p05": float(
            np.quantile(diagnostics.task_scale_min[core], 0.05)
        ),
        "position_slack_p99_m": float(
            np.quantile(
                diagnostics.cartesian_position_slack[core],
                0.99,
            )
        ),
        "orientation_slack_p99_rad": float(
            np.quantile(
                diagnostics.cartesian_orientation_slack[core],
                0.99,
            )
        ),
        "elbow_slack_p99_deg": float(
            np.rad2deg(
                np.quantile(
                    diagnostics.elbow_corridor_slack[core],
                    0.99,
                )
            )
        ),
        "corridor_width_mean_deg": float(
            np.rad2deg(np.mean(diagnostics.corridor_width[core]))
        ),
        "min_swivel_radius_m": float(
            np.min(diagnostics.swivel_radius[core])
        ),
        "peak_velocity_utilization": float(
            np.max(trace.velocity_utilization[core])
        ),
        "j1_command_accel_p99_rad_s2": _p99_acceleration(
            trace.command_dq[core, 0]
        ),
        "j4_command_accel_p99_rad_s2": _p99_acceleration(
            trace.command_dq[core, 3]
        ),
        "qp_failure_count": int(np.sum(diagnostics.qp_failed)),
    }


def _save_augmented_trace(
    path: Path,
    trace: ReplayTrace,
    diagnostics: ElbowQpDiagnostics,
) -> None:
    np.savez_compressed(
        path,
        **trace.__dict__,
        **{
            f"augmented_{key}": value
            for key, value in diagnostics.__dict__.items()
        },
    )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _number_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _candidate_params() -> list[ElbowQpParams]:
    base = ElbowQpParams(name="base")

    def scaled(
        source: ElbowQpParams,
        name: str,
        factor: float,
    ) -> ElbowQpParams:
        return replace(
            source,
            name=name,
            elbow_velocity_weight_slow=(
                source.elbow_velocity_weight_slow * factor
            ),
            elbow_velocity_weight_fast=(
                source.elbow_velocity_weight_fast * factor
            ),
            task_scale_weight=source.task_scale_weight * factor,
            cartesian_slack_weight_slow=(
                source.cartesian_slack_weight_slow * factor
            ),
            cartesian_slack_weight_fast=(
                source.cartesian_slack_weight_fast * factor
            ),
            corridor_slack_linear_slow=(
                source.corridor_slack_linear_slow * factor
            ),
            corridor_slack_linear_fast=(
                source.corridor_slack_linear_fast * factor
            ),
            corridor_slack_quadratic_slow=(
                source.corridor_slack_quadratic_slow * factor
            ),
            corridor_slack_quadratic_fast=(
                source.corridor_slack_quadratic_fast * factor
            ),
        )

    task_scale_only = replace(
        base,
        name="task_scale_only",
        corridor_slow_rad=np.deg2rad(180.0),
        corridor_fast_rad=np.deg2rad(180.0),
        elbow_velocity_weight_slow=0.0,
        elbow_velocity_weight_fast=0.0,
        corridor_slack_linear_fast=0.0,
        corridor_slack_quadratic_fast=0.0,
    )
    home_leak_base = replace(
        task_scale_only,
        name="home_leak_base",
        speed_ratio_slow=0.75,
        task_command_mode="error_leak",
        reference_mode="home",
        corridor_mode="home_interval",
        corridor_margin_power=2.0,
        corridor_slow_rad=np.pi,
        corridor_fast_rad=0.0,
        elbow_return_scale_fast=1.0,
    )
    candidates = [
        task_scale_only,
        scaled(task_scale_only, "task_scale_only_x100", 100.0),
        replace(
            task_scale_only,
            name="task_scale_only_d1e6",
            cartesian_slack_weight_fast=1e6,
        ),
        replace(
            task_scale_only,
            name="task_scale_only_d1e8",
            cartesian_slack_weight_fast=1e8,
        ),
        replace(
            base,
            name="corridor_d1e6",
            cartesian_slack_weight_fast=1e6,
        ),
        replace(
            base,
            name="corridor_d1e8",
            cartesian_slack_weight_fast=1e8,
        ),
        replace(
            base,
            name="corridor_progress_100",
            task_scale_weight=100.0,
            elbow_velocity_weight_fast=5.0,
        ),
        replace(
            base,
            name="corridor_progress_300",
            task_scale_weight=300.0,
            elbow_velocity_weight_fast=5.0,
        ),
        replace(
            base,
            name="corridor_progress_1000",
            task_scale_weight=1_000.0,
            elbow_velocity_weight_fast=5.0,
        ),
        replace(
            base,
            name="corridor_progress_3000",
            task_scale_weight=3_000.0,
            elbow_velocity_weight_fast=5.0,
        ),
        replace(
            base,
            name="corridor_balanced",
            task_scale_weight=300.0,
            elbow_velocity_weight_fast=5.0,
            corridor_slack_linear_fast=0.2,
            corridor_slack_quadratic_fast=2.0,
        ),
        replace(
            base,
            name="corridor_flexible",
            task_scale_weight=300.0,
            elbow_velocity_weight_fast=1.0,
            corridor_slack_linear_fast=0.1,
            corridor_slack_quadratic_fast=1.0,
        ),
        replace(
            base,
            name="corridor_10deg_e0p5",
            corridor_fast_rad=np.deg2rad(10.0),
            elbow_velocity_weight_fast=0.5,
            corridor_slack_linear_fast=0.1,
            corridor_slack_quadratic_fast=1.0,
        ),
        replace(
            base,
            name="corridor_15deg_e0p5",
            corridor_fast_rad=np.deg2rad(15.0),
            elbow_velocity_weight_fast=0.5,
            corridor_slack_linear_fast=0.1,
            corridor_slack_quadratic_fast=1.0,
        ),
        replace(
            base,
            name="corridor_20deg_e0p5",
            corridor_fast_rad=np.deg2rad(20.0),
            elbow_velocity_weight_fast=0.5,
            corridor_slack_linear_fast=0.1,
            corridor_slack_quadratic_fast=1.0,
        ),
        replace(
            base,
            name="corridor_30deg_e0p5",
            corridor_fast_rad=np.deg2rad(30.0),
            elbow_velocity_weight_fast=0.5,
            corridor_slack_linear_fast=0.1,
            corridor_slack_quadratic_fast=1.0,
        ),
        replace(
            base,
            name="corridor_15deg_e2",
            corridor_fast_rad=np.deg2rad(15.0),
            elbow_velocity_weight_fast=2.0,
            corridor_slack_linear_fast=0.1,
            corridor_slack_quadratic_fast=1.0,
        ),
        replace(
            base,
            name="corridor_20deg_e2",
            corridor_fast_rad=np.deg2rad(20.0),
            elbow_velocity_weight_fast=2.0,
            corridor_slack_linear_fast=0.1,
            corridor_slack_quadratic_fast=1.0,
        ),
        replace(
            task_scale_only,
            name="damping_only_e20",
            elbow_velocity_weight_fast=20.0,
        ),
        replace(
            task_scale_only,
            name="damping_only_e0p5",
            elbow_velocity_weight_fast=0.5,
        ),
        replace(
            task_scale_only,
            name="damping_only_e0p05",
            elbow_velocity_weight_fast=0.05,
        ),
        replace(
            task_scale_only,
            name="damping_only_e0p01",
            elbow_velocity_weight_fast=0.01,
        ),
        replace(
            task_scale_only,
            name="damping_only_e0p02",
            elbow_velocity_weight_fast=0.02,
        ),
        replace(
            task_scale_only,
            name="damping_only_e0p1",
            elbow_velocity_weight_fast=0.1,
        ),
        replace(
            task_scale_only,
            name="damping_only_e0p2",
            elbow_velocity_weight_fast=0.2,
        ),
        replace(
            task_scale_only,
            name="damping_e0p05_r0p75",
            speed_ratio_slow=0.75,
            elbow_velocity_weight_fast=0.05,
        ),
        replace(
            task_scale_only,
            name="damping_e0p075_r0p75",
            speed_ratio_slow=0.75,
            elbow_velocity_weight_fast=0.075,
        ),
        replace(
            task_scale_only,
            name="damping_e0p06_r0p75",
            speed_ratio_slow=0.75,
            elbow_velocity_weight_fast=0.06,
        ),
        replace(
            task_scale_only,
            name="damping_e0p065_r0p75",
            speed_ratio_slow=0.75,
            elbow_velocity_weight_fast=0.065,
        ),
        replace(
            task_scale_only,
            name="damping_e0p07_r0p75",
            speed_ratio_slow=0.75,
            elbow_velocity_weight_fast=0.07,
        ),
        replace(
            task_scale_only,
            name="damping_e0p08_r0p75",
            speed_ratio_slow=0.75,
            elbow_velocity_weight_fast=0.08,
        ),
        replace(
            task_scale_only,
            name="damping_e0p05_r0p85",
            speed_ratio_slow=0.85,
            elbow_velocity_weight_fast=0.05,
        ),
        replace(
            base,
            name="guard_25deg_e0p02",
            corridor_fast_rad=np.deg2rad(25.0),
            elbow_velocity_weight_fast=0.02,
        ),
        replace(
            base,
            name="guard_30deg_e0p02",
            corridor_fast_rad=np.deg2rad(30.0),
            elbow_velocity_weight_fast=0.02,
        ),
        replace(
            base,
            name="guard_30deg_e0p05",
            corridor_fast_rad=np.deg2rad(30.0),
            elbow_velocity_weight_fast=0.05,
        ),
        replace(
            base,
            name="guard45_30_e0p05_r0p75",
            speed_ratio_slow=0.75,
            corridor_slow_rad=np.deg2rad(45.0),
            corridor_fast_rad=np.deg2rad(30.0),
            elbow_velocity_weight_fast=0.05,
        ),
        replace(
            base,
            name="guard45_25_e0p05_r0p75",
            speed_ratio_slow=0.75,
            corridor_slow_rad=np.deg2rad(45.0),
            corridor_fast_rad=np.deg2rad(25.0),
            elbow_velocity_weight_fast=0.05,
        ),
        replace(
            base,
            name="corridor_only_10deg",
            corridor_fast_rad=np.deg2rad(10.0),
            elbow_velocity_weight_fast=0.0,
            corridor_slack_linear_fast=0.1,
            corridor_slack_quadratic_fast=1.0,
        ),
        replace(
            base,
            name="corridor_only_20deg",
            corridor_fast_rad=np.deg2rad(20.0),
            elbow_velocity_weight_fast=0.0,
            corridor_slack_linear_fast=0.1,
            corridor_slack_quadratic_fast=1.0,
        ),
        replace(
            base,
            name="corridor_only_30deg",
            corridor_fast_rad=np.deg2rad(30.0),
            elbow_velocity_weight_fast=0.0,
            corridor_slack_linear_fast=0.1,
            corridor_slack_quadratic_fast=1.0,
        ),
        scaled(base, "corridor_x10", 10.0),
        scaled(base, "corridor_x100", 100.0),
        scaled(base, "corridor_x300", 300.0),
    ]
    for task_weight in (3.0, 10.0, 30.0):
        for elbow_weight in (5.0, 20.0, 80.0):
            candidates.append(
                replace(
                    base,
                    name=(
                        f"corridor_s{_number_label(task_weight)}"
                        f"_e{_number_label(elbow_weight)}"
                    ),
                    task_scale_weight=task_weight,
                    elbow_velocity_weight_fast=elbow_weight,
                )
            )
    candidates.extend(
        [
            replace(
                base,
                name="corridor_wide_fast",
                corridor_fast_rad=np.deg2rad(8.0),
            ),
            replace(
                base,
                name="corridor_narrow_fast",
                corridor_fast_rad=np.deg2rad(3.0),
            ),
            replace(
                base,
                name="corridor_soft_slack",
                corridor_slack_linear_fast=0.2,
                corridor_slack_quadratic_fast=2.0,
            ),
            replace(
                base,
                name="corridor_hard_slack",
                corridor_slack_linear_fast=3.0,
                corridor_slack_quadratic_fast=30.0,
            ),
            home_leak_base,
            replace(
                home_leak_base,
                name="home_leak_0p2mm",
                task_error_position_leak_m=0.0002,
                task_error_orientation_leak_rad=0.0016,
            ),
            replace(
                home_leak_base,
                name="home_leak_0p8mm",
                task_error_position_leak_m=0.0008,
                task_error_orientation_leak_rad=0.0064,
            ),
            replace(
                home_leak_base,
                name="home_cost_x1p5",
                nullspace_cost_scale_fast=1.5,
            ),
            replace(
                home_leak_base,
                name="home_cost_x2",
                nullspace_cost_scale_fast=2.0,
            ),
            replace(
                home_leak_base,
                name="home_cost_x3",
                nullspace_cost_scale_fast=3.0,
            ),
            replace(
                home_leak_base,
                name="home_damping_e0p02",
                elbow_velocity_weight_fast=0.02,
            ),
            replace(
                home_leak_base,
                name="home_damping_e0p06",
                elbow_velocity_weight_fast=0.06,
            ),
            replace(
                home_leak_base,
                name="home_damping_e0p1",
                elbow_velocity_weight_fast=0.1,
            ),
            replace(
                home_leak_base,
                name="home_corr_l0p01_q0p1",
                corridor_slack_linear_fast=0.01,
                corridor_slack_quadratic_fast=0.1,
            ),
            replace(
                home_leak_base,
                name="home_corr_l0p05_q0p5",
                corridor_slack_linear_fast=0.05,
                corridor_slack_quadratic_fast=0.5,
            ),
            replace(
                home_leak_base,
                name="home_corr_l0p1_q1",
                corridor_slack_linear_fast=0.1,
                corridor_slack_quadratic_fast=1.0,
            ),
            replace(
                home_leak_base,
                name="home_corr_l0p5_q5",
                corridor_slack_linear_fast=0.5,
                corridor_slack_quadratic_fast=5.0,
            ),
            replace(
                home_leak_base,
                name="home_combo_h1p5_e0p02_c0p05",
                nullspace_cost_scale_fast=1.5,
                elbow_velocity_weight_fast=0.02,
                corridor_slack_linear_fast=0.05,
                corridor_slack_quadratic_fast=0.5,
            ),
            replace(
                home_leak_base,
                name="home_combo_h2_e0p02_c0p1",
                nullspace_cost_scale_fast=2.0,
                elbow_velocity_weight_fast=0.02,
                corridor_slack_linear_fast=0.1,
                corridor_slack_quadratic_fast=1.0,
            ),
            replace(
                home_leak_base,
                name="home_combo_h2_e0p06_c0p05",
                nullspace_cost_scale_fast=2.0,
                elbow_velocity_weight_fast=0.06,
                corridor_slack_linear_fast=0.05,
                corridor_slack_quadratic_fast=0.5,
            ),
            replace(
                home_leak_base,
                name="home_leak_0p3mm",
                task_error_position_leak_m=0.0003,
                task_error_orientation_leak_rad=0.0024,
            ),
            replace(
                home_leak_base,
                name="home_corr_l1_q10",
                corridor_slack_linear_fast=1.0,
                corridor_slack_quadratic_fast=10.0,
            ),
            replace(
                home_leak_base,
                name="home_corr_l5_q50",
                corridor_slack_linear_fast=5.0,
                corridor_slack_quadratic_fast=50.0,
            ),
            replace(
                home_leak_base,
                name="home_corr_l20_q200",
                corridor_slack_linear_fast=20.0,
                corridor_slack_quadratic_fast=200.0,
            ),
            replace(
                home_leak_base,
                name="home_corr_l100_q1000",
                corridor_slack_linear_fast=100.0,
                corridor_slack_quadratic_fast=1000.0,
            ),
            replace(
                home_leak_base,
                name="home_corr_p3_l5_q50",
                corridor_margin_power=3.0,
                corridor_slack_linear_fast=5.0,
                corridor_slack_quadratic_fast=50.0,
            ),
            replace(
                home_leak_base,
                name="home_damping_e0p03_v0p1",
                elbow_return_max_rad_s=0.1,
                elbow_velocity_weight_fast=0.03,
            ),
            replace(
                home_leak_base,
                name="home_damping_e0p04_v0p1",
                elbow_return_max_rad_s=0.1,
                elbow_velocity_weight_fast=0.04,
            ),
            replace(
                home_leak_base,
                name="home_damping_e0p04_v0p2",
                elbow_return_max_rad_s=0.2,
                elbow_velocity_weight_fast=0.04,
            ),
            replace(
                home_leak_base,
                name="home_combo_0p3mm_e0p03_c5",
                task_error_position_leak_m=0.0003,
                task_error_orientation_leak_rad=0.0024,
                elbow_return_max_rad_s=0.1,
                elbow_velocity_weight_fast=0.03,
                corridor_slack_linear_fast=5.0,
                corridor_slack_quadratic_fast=50.0,
            ),
            replace(
                home_leak_base,
                name="home_combo_0p3mm_e0p04_c20",
                task_error_position_leak_m=0.0003,
                task_error_orientation_leak_rad=0.0024,
                elbow_return_max_rad_s=0.1,
                elbow_velocity_weight_fast=0.04,
                corridor_slack_linear_fast=20.0,
                corridor_slack_quadratic_fast=200.0,
            ),
            replace(
                home_leak_base,
                name="home_combo_slowalpha_0p3mm_e0p03_c5",
                activation_rise_rate_s_inv=4.0,
                activation_fall_rate_s_inv=2.0,
                task_error_position_leak_m=0.0003,
                task_error_orientation_leak_rad=0.0024,
                elbow_return_max_rad_s=0.1,
                elbow_velocity_weight_fast=0.03,
                corridor_slack_linear_fast=5.0,
                corridor_slack_quadratic_fast=50.0,
            ),
            replace(
                home_leak_base,
                name="home_0p3mm_damping_e0p03",
                task_error_position_leak_m=0.0003,
                task_error_orientation_leak_rad=0.0024,
                elbow_return_max_rad_s=0.1,
                elbow_velocity_weight_fast=0.03,
            ),
            replace(
                home_leak_base,
                name="home_0p3mm_corr_c5",
                task_error_position_leak_m=0.0003,
                task_error_orientation_leak_rad=0.0024,
                corridor_slack_linear_fast=5.0,
                corridor_slack_quadratic_fast=50.0,
            ),
            replace(
                home_leak_base,
                name="home_slowalpha_0p3mm_damping_e0p03",
                activation_rise_rate_s_inv=4.0,
                activation_fall_rate_s_inv=2.0,
                task_error_position_leak_m=0.0003,
                task_error_orientation_leak_rad=0.0024,
                elbow_return_max_rad_s=0.1,
                elbow_velocity_weight_fast=0.03,
            ),
            replace(
                home_leak_base,
                name="home_final_h1p5_0p3mm_e0p03_c5",
                activation_rise_rate_s_inv=4.0,
                activation_fall_rate_s_inv=2.0,
                task_error_position_leak_m=0.0003,
                task_error_orientation_leak_rad=0.0024,
                nullspace_cost_scale_fast=1.5,
                elbow_return_max_rad_s=0.1,
                elbow_velocity_weight_fast=0.03,
                corridor_slack_linear_fast=5.0,
                corridor_slack_quadratic_fast=50.0,
            ),
            replace(
                home_leak_base,
                name="home_final_latched_h1p5_0p3mm_e0p03_c5",
                corridor_anchor_mode="latched",
                activation_rise_rate_s_inv=4.0,
                activation_fall_rate_s_inv=2.0,
                task_error_position_leak_m=0.0003,
                task_error_orientation_leak_rad=0.0024,
                nullspace_cost_scale_fast=1.5,
                elbow_return_max_rad_s=0.1,
                elbow_velocity_weight_fast=0.03,
                corridor_slack_linear_fast=5.0,
                corridor_slack_quadratic_fast=50.0,
            ),
            replace(
                home_leak_base,
                name="home_final_0p4mm_e0p02_c1",
                activation_rise_rate_s_inv=4.0,
                activation_fall_rate_s_inv=2.0,
                nullspace_cost_scale_fast=1.5,
                elbow_return_max_rad_s=0.1,
                elbow_velocity_weight_fast=0.02,
                corridor_slack_linear_fast=1.0,
                corridor_slack_quadratic_fast=10.0,
            ),
            replace(
                home_leak_base,
                name="home_final_0p4mm_e0p02_c5",
                activation_rise_rate_s_inv=4.0,
                activation_fall_rate_s_inv=2.0,
                nullspace_cost_scale_fast=1.5,
                elbow_return_max_rad_s=0.1,
                elbow_velocity_weight_fast=0.02,
                corridor_slack_linear_fast=5.0,
                corridor_slack_quadratic_fast=50.0,
            ),
            replace(
                home_leak_base,
                name="home_final_0p4mm_e0p03_c1",
                activation_rise_rate_s_inv=4.0,
                activation_fall_rate_s_inv=2.0,
                nullspace_cost_scale_fast=1.5,
                elbow_return_max_rad_s=0.1,
                elbow_velocity_weight_fast=0.03,
                corridor_slack_linear_fast=1.0,
                corridor_slack_quadratic_fast=10.0,
            ),
            replace(
                home_leak_base,
                name="home_final_0p4mm_e0p03_c5",
                activation_rise_rate_s_inv=4.0,
                activation_fall_rate_s_inv=2.0,
                nullspace_cost_scale_fast=1.5,
                elbow_return_max_rad_s=0.1,
                elbow_velocity_weight_fast=0.03,
                corridor_slack_linear_fast=5.0,
                corridor_slack_quadratic_fast=50.0,
            ),
            replace(
                home_leak_base,
                name="home_final_0p5mm_e0p02_c5",
                activation_rise_rate_s_inv=4.0,
                activation_fall_rate_s_inv=2.0,
                task_error_position_leak_m=0.0005,
                task_error_orientation_leak_rad=0.004,
                nullspace_cost_scale_fast=1.5,
                elbow_return_max_rad_s=0.1,
                elbow_velocity_weight_fast=0.02,
                corridor_slack_linear_fast=5.0,
                corridor_slack_quadratic_fast=50.0,
            ),
        ]
    )
    return candidates


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
    parser.add_argument("--candidates", nargs="*", default=[])
    parser.add_argument(
        "--include-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "dev/results/speed_scheduled_elbow_qp_20260724"
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
    candidates = [
        candidate
        for candidate in _candidate_params()
        if not args.candidates or candidate.name in set(args.candidates)
    ]
    if not candidates:
        raise ValueError("No augmented-QP candidates selected.")

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
        core_start = segment.core_start - segment.start
        core_end = segment.core_end - segment.start
        cases.append(
            (
                f"reverse_{segment_index}",
                target_pose[window][::-1].copy(),
                source_q[window][::-1].copy(),
                source_elbow[window][::-1].copy(),
                slice(count - core_end, count - core_start),
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

    (args.output_dir / "parameters.json").write_text(
        json.dumps([asdict(candidate) for candidate in candidates], indent=2),
        encoding="utf-8",
    )
    rows: list[dict[str, object]] = []
    for case_name, case_target, case_q, case_elbow, core in cases:
        if args.include_reference:
            print(f"Simulating case={case_name}, guard reference...")
            reference, _ = _simulate(
                args.side,
                Strategy("N_guard_reference", "reference"),
                case_target,
                case_q,
                case_elbow,
                settle_duration=args.settle_duration,
                guard_scale=4.0,
                soft_ratio=0.8,
            )
        else:
            reference = ReplayTrace(
                time=np.arange(case_target.shape[0]) * CONTROL_DT,
                target_pose=case_target,
                ik_target_pose=case_target,
                source_q=case_q,
                source_elbow=case_elbow,
                command_q=case_q,
                command_dq=np.gradient(case_q, CONTROL_DT, axis=0),
                actual_q=case_q,
                actual_dq=np.gradient(case_q, CONTROL_DT, axis=0),
                command_pose=case_target,
                actual_pose=case_target,
                command_elbow=case_elbow,
                actual_elbow=case_elbow,
                rho=np.zeros(case_target.shape[0]),
                sigma_min=np.zeros(case_target.shape[0]),
                exact_null_speed=np.zeros(case_target.shape[0]),
                near_null_speed=np.zeros(case_target.shape[0]),
                exact_null_error=np.zeros(case_target.shape[0]),
                near_null_error=np.zeros(case_target.shape[0]),
                velocity_utilization=np.zeros((case_target.shape[0], 7)),
                solve_failed=np.zeros(case_target.shape[0], dtype=bool),
            )
        for candidate in candidates:
            print(
                f"Simulating case={case_name}, candidate={candidate.name}..."
            )
            trace, diagnostics = _simulate_augmented(
                args.side,
                candidate,
                case_target,
                case_q,
                case_elbow,
                settle_duration=args.settle_duration,
            )
            rows.append(
                _summarize_augmented(
                    case_name,
                    candidate,
                    trace,
                    diagnostics,
                    reference,
                    core,
                )
            )
            _save_augmented_trace(
                args.output_dir
                / f"trace_{case_name}_{candidate.name}.npz",
                trace,
                diagnostics,
            )
    _write_rows(args.output_dir / "metrics.csv", rows)
    print(f"Wrote {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
