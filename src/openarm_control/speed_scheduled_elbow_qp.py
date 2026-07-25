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

"""Speed-scheduled task scaling and home-referenced elbow regulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mink
import mujoco
import numpy as np
import numpy.typing as npt
import qpsolvers


@dataclass(frozen=True)
class SpeedScheduledElbowParams:
    """Parameters shared by the per-side speed scheduler and augmented QP."""

    control_dt: float
    substep_dt: float
    linear_fast: float = 0.6
    angular_fast: float = 4.0
    speed_ratio_slow: float = 0.75
    speed_ratio_fast: float = 1.0
    activation_rise_rate: float = 4.0
    activation_fall_rate: float = 2.0
    position_error_leak: float = 0.0003
    orientation_error_leak: float = 0.0024
    task_scale_weight: float = 10.0
    cartesian_position_slack_scale: float = 0.002
    cartesian_orientation_slack_scale: float = 0.01
    cartesian_slack_weight_fast: float = 1e6
    nullspace_cost_scale_fast: float = 1.5
    swivel_return_rate: float = 0.5
    swivel_return_max_speed: float = 0.1
    swivel_velocity_scale: float = 0.25
    swivel_velocity_weight_fast: float = 0.03
    corridor_margin_slow: float = np.pi
    corridor_margin_fast: float = 0.0
    corridor_margin_power: float = 2.0
    corridor_slack_scale: float = np.deg2rad(1.0)
    corridor_slack_linear_fast: float = 5.0
    corridor_slack_quadratic_fast: float = 50.0
    swivel_min_radius: float = 0.02
    swivel_fd_epsilon: float = 1e-5
    latch_fast_reference: bool = False
    latch_alpha: float = 0.8
    release_alpha: float = 0.05

    def __post_init__(self) -> None:
        """Validate physical scales and interpolation ranges."""
        positive = {
            "control_dt": self.control_dt,
            "substep_dt": self.substep_dt,
            "linear_fast": self.linear_fast,
            "angular_fast": self.angular_fast,
            "activation_rise_rate": self.activation_rise_rate,
            "activation_fall_rate": self.activation_fall_rate,
            "position_error_leak": self.position_error_leak,
            "orientation_error_leak": self.orientation_error_leak,
            "task_scale_weight": self.task_scale_weight,
            "cartesian_position_slack_scale": (self.cartesian_position_slack_scale),
            "cartesian_orientation_slack_scale": (
                self.cartesian_orientation_slack_scale
            ),
            "cartesian_slack_weight_fast": self.cartesian_slack_weight_fast,
            "swivel_velocity_scale": self.swivel_velocity_scale,
            "corridor_slack_scale": self.corridor_slack_scale,
            "swivel_min_radius": self.swivel_min_radius,
            "swivel_fd_epsilon": self.swivel_fd_epsilon,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        nonnegative = {
            "nullspace_cost_scale_fast": self.nullspace_cost_scale_fast,
            "swivel_return_rate": self.swivel_return_rate,
            "swivel_return_max_speed": self.swivel_return_max_speed,
            "swivel_velocity_weight_fast": self.swivel_velocity_weight_fast,
            "corridor_margin_slow": self.corridor_margin_slow,
            "corridor_margin_fast": self.corridor_margin_fast,
            "corridor_slack_linear_fast": self.corridor_slack_linear_fast,
            "corridor_slack_quadratic_fast": self.corridor_slack_quadratic_fast,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not 0.0 <= self.speed_ratio_slow < self.speed_ratio_fast:
            raise ValueError("Expected 0 <= speed_ratio_slow < speed_ratio_fast.")
        if not np.isfinite(self.corridor_margin_power) or (
            self.corridor_margin_power <= 0.0
        ):
            raise ValueError("corridor_margin_power must be finite and positive.")
        if self.corridor_margin_fast > self.corridor_margin_slow:
            raise ValueError(
                "corridor_margin_fast must not exceed corridor_margin_slow."
            )
        if not 0.0 <= self.release_alpha < self.latch_alpha <= 1.0:
            raise ValueError("Expected 0 <= release_alpha < latch_alpha <= 1.")


@dataclass(frozen=True)
class SpeedScheduledElbowState:
    """Outer-cycle diagnostics for one arm."""

    linear_speed: float
    angular_speed: float
    target_activation: float
    activation: float
    swivel: float
    swivel_radius: float
    anchor_swivel: float
    corridor_lower: float
    corridor_upper: float
    corridor_margin: float
    reference_latched: bool
    nullspace_cost_scale: float


@dataclass(frozen=True)
class SpeedScheduledQpState:
    """Diagnostics from the latest augmented QP substep."""

    task_scale: float
    cartesian_position_slack: float
    cartesian_orientation_slack: float
    corridor_slack: float


@dataclass(frozen=True)
class _SideCycle:
    side: str
    frame_task: mink.FrameTask
    coordinate: ElbowSwivelCoordinate
    activation: float
    swivel_jacobian: np.ndarray
    desired_swivel_displacement: float
    corridor_lower: float
    corridor_upper: float
    elbow_active: bool


@dataclass(frozen=True)
class _VariableLayout:
    scale: int
    corridor_slack: int
    frame_jacobian: np.ndarray
    task_displacement: np.ndarray


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _lerp(low: float, high: float, alpha: float) -> float:
    return (1.0 - alpha) * low + alpha * high


def _wrapped_difference(lhs: float, rhs: float) -> float:
    return float(np.arctan2(np.sin(lhs - rhs), np.cos(lhs - rhs)))


def _limit_norm(vector: npt.ArrayLike, limit: float) -> np.ndarray:
    output = np.asarray(vector, dtype=np.float64).copy()
    norm = float(np.linalg.norm(output))
    if norm > limit:
        output *= limit / norm
    return output


def _quaternion_angle(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        raise ValueError("Target quaternion must have non-zero norm.")
    cosine = abs(float((lhs / lhs_norm) @ (rhs / rhs_norm)))
    return float(2.0 * np.arccos(np.clip(cosine, 0.0, 1.0)))


class ElbowSwivelCoordinate:
    """Measure elbow swivel around the shoulder-to-end-effector axis."""

    def __init__(
        self,
        model: mujoco.MjModel,
        side: str,
        frame_task: mink.FrameTask,
        reference_q: npt.ArrayLike,
        dof_indices: npt.ArrayLike,
        *,
        finite_difference_epsilon: float,
    ) -> None:
        """Build a home-referenced geometric elbow coordinate."""
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported arm side: {side!r}.")
        if finite_difference_epsilon <= 0.0:
            raise ValueError("finite_difference_epsilon must be positive.")

        self.model = model
        self.side = side
        self.frame_task = frame_task
        self.dof_indices = np.asarray(dof_indices, dtype=int)
        if self.dof_indices.shape != (7,):
            raise ValueError("Elbow swivel requires seven arm DoF indices.")
        self.epsilon = float(finite_difference_epsilon)
        self.data = mujoco.MjData(model)
        self.shoulder_joint = model.joint(f"openarm_{side}_joint1").id
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
        reference_q = np.asarray(reference_q, dtype=np.float64)
        if reference_q.shape != (model.nq,):
            raise ValueError(
                f"Expected reference_q shape ({model.nq},), got {reference_q.shape}."
            )
        _, _, reference_normal, radius = self._geometry(reference_q)
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
        return shoulder, wrist, elbow_normal, radius

    def value(self, q: npt.ArrayLike) -> tuple[float, float]:
        """Return home-referenced swivel angle and geometric radius."""
        q = np.asarray(q, dtype=np.float64)
        shoulder, wrist, elbow_normal, radius = self._geometry(q)
        axis = wrist - shoulder
        axis /= np.linalg.norm(axis)
        reference = self.reference_normal.copy()
        reference -= axis * float(axis @ reference)
        reference_norm = float(np.linalg.norm(reference))
        if reference_norm <= 1e-10 or radius <= 1e-10:
            return 0.0, radius
        reference /= reference_norm
        sine = float(axis @ np.cross(reference, elbow_normal))
        cosine = float(reference @ elbow_normal)
        return float(np.arctan2(sine, cosine)), radius

    def linearize(
        self,
        q: npt.ArrayLike,
    ) -> tuple[float, np.ndarray, float]:
        """Return swivel, its tangent-space Jacobian, and swivel radius."""
        q = np.asarray(q, dtype=np.float64)
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
            jacobian[dof] = _wrapped_difference(plus, minus) / (2.0 * self.epsilon)
        return value, jacobian, radius


class _SideScheduler:
    """Maintain target-speed and optional fast-entry anchor state for one arm."""

    def __init__(
        self,
        model: mujoco.MjModel,
        side: str,
        frame_task: mink.FrameTask,
        home_qpos: np.ndarray,
        dof_indices: np.ndarray,
        params: SpeedScheduledElbowParams,
    ) -> None:
        self.side = side
        self.frame_task = frame_task
        self.params = params
        self.coordinate = ElbowSwivelCoordinate(
            model,
            side,
            frame_task,
            home_qpos,
            dof_indices,
            finite_difference_epsilon=params.swivel_fd_epsilon,
        )
        self.previous_target: np.ndarray | None = None
        self.activation = 0.0
        self.pre_fast_swivel: float | None = None
        self.fast_anchor_swivel: float | None = None
        self.last_state: SpeedScheduledElbowState | None = None
        self.last_qp_state: SpeedScheduledQpState | None = None

    def reset(self) -> None:
        """Clear target history, activation, and any latched reference."""
        self.previous_target = None
        self.activation = 0.0
        self.pre_fast_swivel = None
        self.fast_anchor_swivel = None
        self.last_state = None
        self.last_qp_state = None

    def prepare(
        self,
        configuration: mink.Configuration,
        target: npt.ArrayLike,
    ) -> _SideCycle | None:
        """Consume one outer-cycle target and prepare fixed substep terms."""
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError("Target pose must contain seven finite values.")
        linear_speed, angular_speed, target_activation = self._target_motion(target)
        difference = target_activation - self.activation
        rate = (
            self.params.activation_rise_rate
            if difference > 0.0
            else self.params.activation_fall_rate
        )
        self.activation += float(
            np.clip(
                difference,
                -rate * self.params.control_dt,
                rate * self.params.control_dt,
            )
        )
        if self.activation <= 1e-12:
            self.activation = 0.0
            if self.fast_anchor_swivel is not None:
                self.fast_anchor_swivel = None
            self.last_state = SpeedScheduledElbowState(
                linear_speed=linear_speed,
                angular_speed=angular_speed,
                target_activation=target_activation,
                activation=0.0,
                swivel=0.0,
                swivel_radius=0.0,
                anchor_swivel=0.0,
                corridor_lower=-self.params.corridor_margin_slow,
                corridor_upper=self.params.corridor_margin_slow,
                corridor_margin=self.params.corridor_margin_slow,
                reference_latched=False,
                nullspace_cost_scale=1.0,
            )
            self.last_qp_state = None
            return None

        swivel, swivel_jacobian, radius = self.coordinate.linearize(configuration.q)
        anchor = self._update_anchor(swivel)
        residual = (1.0 - self.activation) ** self.params.corridor_margin_power
        margin = _lerp(
            self.params.corridor_margin_fast,
            self.params.corridor_margin_slow,
            residual,
        )
        corridor_lower = min(0.0, anchor) - margin
        corridor_upper = max(0.0, anchor) + margin
        nullspace_cost_scale = _lerp(
            1.0,
            self.params.nullspace_cost_scale_fast,
            self.activation,
        )
        self.last_state = SpeedScheduledElbowState(
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            target_activation=target_activation,
            activation=self.activation,
            swivel=swivel,
            swivel_radius=radius,
            anchor_swivel=anchor,
            corridor_lower=corridor_lower,
            corridor_upper=corridor_upper,
            corridor_margin=margin,
            reference_latched=self.fast_anchor_swivel is not None,
            nullspace_cost_scale=nullspace_cost_scale,
        )
        return_speed = float(
            np.clip(
                -self.params.swivel_return_rate * swivel,
                -self.params.swivel_return_max_speed,
                self.params.swivel_return_max_speed,
            )
        )
        return _SideCycle(
            side=self.side,
            frame_task=self.frame_task,
            coordinate=self.coordinate,
            activation=self.activation,
            swivel_jacobian=swivel_jacobian,
            desired_swivel_displacement=(return_speed * self.params.substep_dt),
            corridor_lower=corridor_lower,
            corridor_upper=corridor_upper,
            elbow_active=radius >= self.params.swivel_min_radius,
        )

    def _target_motion(self, target: np.ndarray) -> tuple[float, float, float]:
        if self.previous_target is None:
            self.previous_target = target.copy()
            return 0.0, 0.0, 0.0
        linear_speed = float(
            np.linalg.norm(target[:3] - self.previous_target[:3])
            / self.params.control_dt
        )
        angular_speed = (
            _quaternion_angle(target[3:], self.previous_target[3:])
            / self.params.control_dt
        )
        self.previous_target = target.copy()
        speed_ratio = max(
            linear_speed / self.params.linear_fast,
            angular_speed / self.params.angular_fast,
        )
        activation_input = (speed_ratio - self.params.speed_ratio_slow) / (
            self.params.speed_ratio_fast - self.params.speed_ratio_slow
        )
        return linear_speed, angular_speed, _smoothstep(activation_input)

    def _update_anchor(self, current_swivel: float) -> float:
        if not self.params.latch_fast_reference:
            return current_swivel
        if (
            self.activation >= self.params.latch_alpha
            and self.fast_anchor_swivel is None
        ):
            self.fast_anchor_swivel = (
                current_swivel if self.pre_fast_swivel is None else self.pre_fast_swivel
            )
        elif (
            self.fast_anchor_swivel is not None
            and self.activation <= self.params.release_alpha
        ):
            self.fast_anchor_swivel = None
        if self.fast_anchor_swivel is None:
            self.pre_fast_swivel = current_swivel
            return current_swivel
        return self.fast_anchor_swivel


class SpeedScheduledElbowQP:
    """Coordinate speed scheduling and augmented QP assembly for all arms."""

    def __init__(
        self,
        model: mujoco.MjModel,
        frame_tasks: Mapping[str, mink.FrameTask],
        arm_dofs: Mapping[str, npt.ArrayLike],
        home_qpos: npt.ArrayLike,
        displacement_scale: npt.ArrayLike,
        params: SpeedScheduledElbowParams,
    ) -> None:
        """Initialize one scheduler per active arm."""
        home_qpos = np.asarray(home_qpos, dtype=np.float64)
        self.params = params
        self.displacement_scale = np.asarray(
            displacement_scale,
            dtype=np.float64,
        )
        if self.displacement_scale.shape != (model.nv,) or np.any(
            self.displacement_scale <= 0.0
        ):
            raise ValueError(f"Expected {model.nv} positive displacement scales.")
        self._schedulers = {
            side: _SideScheduler(
                model,
                side,
                frame_tasks[side],
                home_qpos,
                np.asarray(arm_dofs[side], dtype=int),
                params,
            )
            for side in frame_tasks
        }
        self._cycles: dict[str, _SideCycle] = {}

    @property
    def states(self) -> dict[str, SpeedScheduledElbowState | None]:
        """Return the latest outer-cycle state by side."""
        return {
            side: scheduler.last_state for side, scheduler in self._schedulers.items()
        }

    @property
    def qp_states(self) -> dict[str, SpeedScheduledQpState | None]:
        """Return latest augmented-QP diagnostics by side."""
        return {
            side: scheduler.last_qp_state
            for side, scheduler in self._schedulers.items()
        }

    def reset(self) -> None:
        """Reset every side's target and fast-entry state."""
        for scheduler in self._schedulers.values():
            scheduler.reset()
        self._cycles.clear()

    def prepare(
        self,
        configuration: mink.Configuration,
        targets: Mapping[str, npt.ArrayLike],
    ) -> dict[str, float]:
        """Prepare one outer control cycle and return nullspace cost scales."""
        self._cycles.clear()
        cost_scales: dict[str, float] = {}
        for side, scheduler in self._schedulers.items():
            target = targets.get(side)
            if target is None:
                continue
            cycle = scheduler.prepare(configuration, target)
            state = scheduler.last_state
            cost_scales[side] = 1.0 if state is None else state.nullspace_cost_scale
            if cycle is not None:
                self._cycles[side] = cycle
        return cost_scales

    def active(self) -> bool:
        """Return whether at least one arm currently uses the augmented QP."""
        return bool(self._cycles)

    def solve(
        self,
        configuration: mink.Configuration,
        secondary_tasks: Sequence[mink.BaseTask],
        *,
        dt: float,
        solver: str,
        damping: float,
        limits: Sequence[mink.Limit],
        constraints: Sequence[mink.Task],
        solver_kwargs: Mapping[str, object] | None = None,
    ) -> np.ndarray:
        """Solve one IK substep with independent per-arm task scales."""
        configuration.check_limits(safety_break=False)
        inactive_frame_tasks = [
            scheduler.frame_task
            for side, scheduler in self._schedulers.items()
            if side not in self._cycles
        ]
        base = mink.build_ik(
            configuration,
            [*inactive_frame_tasks, *secondary_tasks],
            dt,
            damping=damping,
            limits=limits,
            constraints=constraints,
        )
        problem, layouts = _augment_problem(
            base,
            tuple(self._cycles.values()),
            configuration,
            self.displacement_scale,
            dt,
            self.params,
        )
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
            solver=solver,
            **dict(solver_kwargs or {}),
        )
        if not result.found or result.x is None:
            raise mink.exceptions.NoSolutionFound(solver)
        solution = np.asarray(result.x, dtype=np.float64)
        displacement = solution[: configuration.model.nv] * self.displacement_scale
        for side, layout in layouts.items():
            cartesian_slack = (
                layout.frame_jacobian @ displacement
                - solution[layout.scale] * layout.task_displacement
            )
            self._schedulers[side].last_qp_state = SpeedScheduledQpState(
                task_scale=float(solution[layout.scale]),
                cartesian_position_slack=float(np.linalg.norm(cartesian_slack[:3])),
                cartesian_orientation_slack=float(np.linalg.norm(cartesian_slack[3:])),
                corridor_slack=float(
                    solution[layout.corridor_slack] * self.params.corridor_slack_scale
                ),
            )
        return displacement / dt


def _extend_rows(
    matrix: np.ndarray | None,
    variable_count: int,
    displacement_scale: np.ndarray,
) -> np.ndarray | None:
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=np.float64)
    output = np.zeros((matrix.shape[0], variable_count), dtype=np.float64)
    output[:, : matrix.shape[1]] = matrix * displacement_scale[None, :]
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
    cycles: tuple[_SideCycle, ...],
    configuration: mink.Configuration,
    displacement_scale: np.ndarray,
    dt: float,
    params: SpeedScheduledElbowParams,
) -> tuple[qpsolvers.Problem, dict[str, _VariableLayout]]:
    nv = int(base.q.shape[0])
    variable_count = nv + 2 * len(cycles)
    hessian = np.zeros((variable_count, variable_count), dtype=np.float64)
    linear = np.zeros(variable_count, dtype=np.float64)
    base_hessian = np.asarray(base.P, dtype=np.float64)
    hessian[:nv, :nv] = (
        displacement_scale[:, None] * base_hessian * displacement_scale[None, :]
    )
    linear[:nv] = displacement_scale * np.asarray(base.q, dtype=np.float64)

    equality = _extend_rows(
        base.A,
        variable_count,
        displacement_scale,
    )
    equality_values = None if base.b is None else np.asarray(base.b, dtype=np.float64)
    inequality = _extend_rows(
        base.G,
        variable_count,
        displacement_scale,
    )
    inequality_values = None if base.h is None else np.asarray(base.h, dtype=np.float64)
    layouts: dict[str, _VariableLayout] = {}
    cartesian_slack_scale = np.array(
        [
            params.cartesian_position_slack_scale,
            params.cartesian_position_slack_scale,
            params.cartesian_position_slack_scale,
            params.cartesian_orientation_slack_scale,
            params.cartesian_orientation_slack_scale,
            params.cartesian_orientation_slack_scale,
        ],
        dtype=np.float64,
    )

    offset = nv
    for cycle in cycles:
        scale_index = offset
        corridor_slack_index = offset + 1
        offset += 2

        hessian[scale_index, scale_index] += 2.0 * params.task_scale_weight
        linear[scale_index] -= 2.0 * params.task_scale_weight

        frame_objective = cycle.frame_task.compute_qp_objective(configuration)
        frame_weight = 1.0 - cycle.activation
        hessian[:nv, :nv] += frame_weight * (
            displacement_scale[:, None]
            * np.asarray(frame_objective.H, dtype=np.float64)
            * displacement_scale[None, :]
        )
        linear[:nv] += (
            frame_weight
            * displacement_scale
            * np.asarray(frame_objective.c, dtype=np.float64)
        )

        if cycle.elbow_active:
            velocity_normalizer = params.swivel_velocity_scale * dt
            velocity_row = np.zeros(variable_count, dtype=np.float64)
            velocity_row[:nv] = (
                cycle.swivel_jacobian * displacement_scale / velocity_normalizer
            )
            _add_squared_residual(
                hessian,
                linear,
                velocity_row,
                cycle.desired_swivel_displacement / velocity_normalizer,
                cycle.activation * params.swivel_velocity_weight_fast,
            )

        hessian[corridor_slack_index, corridor_slack_index] += (
            cycle.activation * params.corridor_slack_quadratic_fast
        )
        linear[corridor_slack_index] += (
            cycle.activation * params.corridor_slack_linear_fast
        )

        frame_jacobian = cycle.frame_task.compute_jacobian(configuration)
        task_displacement = -cycle.frame_task.compute_error(configuration).copy()
        task_displacement[:3] = _limit_norm(
            task_displacement[:3],
            params.position_error_leak,
        )
        task_displacement[3:] = _limit_norm(
            task_displacement[3:],
            params.orientation_error_leak,
        )
        layouts[cycle.side] = _VariableLayout(
            scale=scale_index,
            corridor_slack=corridor_slack_index,
            frame_jacobian=frame_jacobian.copy(),
            task_displacement=task_displacement.copy(),
        )

        # Cartesian slack is unconstrained and only has a quadratic cost. Eliminate
        # it analytically: delta_x = J Delta_q - s Delta_x_cmd. This is exactly the
        # same augmented QP with six fewer variables and equalities per arm.
        task_rows = np.zeros((6, variable_count), dtype=np.float64)
        task_rows[:, :nv] = (
            frame_jacobian
            * displacement_scale[None, :]
            / cartesian_slack_scale[:, None]
        )
        task_rows[:, scale_index] = -(task_displacement / cartesian_slack_scale)
        cartesian_slack_weight = cycle.activation * params.cartesian_slack_weight_fast
        hessian += cartesian_slack_weight * (task_rows.T @ task_rows)

        bounds = np.zeros((3, variable_count), dtype=np.float64)
        bounds[0, scale_index] = 1.0
        bounds[1, scale_index] = -1.0
        bounds[2, corridor_slack_index] = -1.0
        inequality = _stack_rows(inequality, bounds)
        inequality_values = _stack_values(
            inequality_values,
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )

        if cycle.elbow_active:
            current_swivel, current_radius = cycle.coordinate.value(configuration.q)
            if current_radius >= params.swivel_min_radius:
                corridor_rows = np.zeros(
                    (2, variable_count),
                    dtype=np.float64,
                )
                corridor_rows[0, :nv] = cycle.swivel_jacobian * displacement_scale
                corridor_rows[0, corridor_slack_index] = -params.corridor_slack_scale
                corridor_rows[1, :nv] = -cycle.swivel_jacobian * displacement_scale
                corridor_rows[1, corridor_slack_index] = -params.corridor_slack_scale
                corridor_values = np.array(
                    [
                        cycle.corridor_upper - current_swivel,
                        current_swivel - cycle.corridor_lower,
                    ],
                    dtype=np.float64,
                )
                inequality = _stack_rows(inequality, corridor_rows)
                inequality_values = _stack_values(
                    inequality_values,
                    corridor_values,
                )

    return (
        qpsolvers.Problem(
            hessian,
            linear,
            inequality,
            inequality_values,
            equality,
            equality_values,
        ),
        layouts,
    )
