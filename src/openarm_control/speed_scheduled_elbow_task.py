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

"""Translation-speed-scheduled spatial elbow regularization."""

from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np
import numpy.typing as npt

from .nullspace_posture_task import structural_nullspace_direction
from .singularity import normalized_arm_jacobian


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _wrapped_difference(lhs: float, rhs: float) -> float:
    return float(np.arctan2(np.sin(lhs - rhs), np.cos(lhs - rhs)))


@dataclass(frozen=True)
class SpeedScheduledElbowState:
    """Diagnostics for the most recently prepared outer control cycle."""

    linear_speed: float
    target_activation: float
    activation: float
    swivel: float
    swivel_radius: float
    anchor_swivel: float
    corridor_lower: float
    corridor_upper: float
    desired_swivel_speed: float
    corridor_error: float
    projection_mode: str
    task_row_norm: float
    task_row_nullspace_residual: float


class ElbowSwivelCoordinate:
    """Measure elbow swivel around the shoulder-to-end-effector axis."""

    def __init__(
        self,
        model: mujoco.MjModel,
        side: str,
        frame_task: mink.FrameTask,
        home_qpos: npt.ArrayLike,
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

        reference_q = np.asarray(home_qpos, dtype=np.float64)
        if reference_q.shape != (model.nq,):
            raise ValueError(
                f"Expected home_qpos shape ({model.nq},), got {reference_q.shape}."
            )
        _, _, reference_normal, radius = self._geometry(reference_q)
        if radius <= 1e-8:
            raise ValueError("Elbow swivel home reference is degenerate.")
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
        """Return home-referenced swivel angle and elbow-plane radius."""
        q = np.asarray(q, dtype=np.float64)
        shoulder, wrist, elbow_normal, radius = self._geometry(q)
        axis = wrist - shoulder
        axis /= np.linalg.norm(axis)

        # Projecting the fixed home normal into the current shoulder-wrist
        # normal plane transports the reference without tying it to a world axis.
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


class SpeedScheduledElbowTask(mink.Task):
    """Regularize spatial elbow motion more strongly at high translation speed.

    The task is a normal weighted Mink objective. It introduces no additional
    QP variables or equalities, so Cartesian tracking may yield softly when the
    elbow objective conflicts with the frame task.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        side: str,
        frame_task: mink.FrameTask,
        dof_indices: npt.ArrayLike,
        home_qpos: npt.ArrayLike,
        *,
        control_dt: float,
        substep_dt: float,
        linear_speed_slow: float,
        linear_speed_fast: float,
        activation_rise_rate: float,
        activation_fall_rate: float,
        velocity_cost_fast: float,
        velocity_scale: float,
        return_rate: float,
        max_return_speed: float,
        max_return_acceleration: float,
        corridor_cost_fast: float,
        corridor_margin_slow: float,
        corridor_margin_fast: float,
        corridor_return_rate: float,
        corridor_max_speed: float,
        projection_mode: str,
        characteristic_length: float,
        min_swivel_radius: float,
        finite_difference_epsilon: float,
    ) -> None:
        """Initialize scheduling, geometry, and physical-rate parameters."""
        if control_dt <= 0.0 or substep_dt <= 0.0:
            raise ValueError("Elbow task timesteps must be positive.")
        if not 0.0 <= linear_speed_slow < linear_speed_fast:
            raise ValueError("Expected 0 <= linear_speed_slow < linear_speed_fast.")
        if activation_rise_rate <= 0.0 or activation_fall_rate <= 0.0:
            raise ValueError("Elbow activation rates must be positive.")
        if velocity_cost_fast < 0.0 or corridor_cost_fast < 0.0:
            raise ValueError("Elbow task costs must be non-negative.")
        if velocity_scale <= 0.0:
            raise ValueError("Elbow velocity_scale must be positive.")
        if return_rate < 0.0 or max_return_speed < 0.0:
            raise ValueError("Elbow return parameters must be non-negative.")
        if max_return_acceleration < 0.0:
            raise ValueError("Elbow return acceleration must be non-negative.")
        if not 0.0 <= corridor_margin_fast <= corridor_margin_slow:
            raise ValueError(
                "Expected 0 <= corridor_margin_fast <= corridor_margin_slow."
            )
        if corridor_return_rate < 0.0 or corridor_max_speed < 0.0:
            raise ValueError("Elbow corridor return parameters must be non-negative.")
        if projection_mode not in {"direct", "exact-nullspace"}:
            raise ValueError(
                "Elbow projection_mode must be 'direct' or 'exact-nullspace'."
            )
        if characteristic_length <= 0.0:
            raise ValueError("Elbow characteristic_length must be positive.")
        if min_swivel_radius < 0.0:
            raise ValueError("Elbow min_swivel_radius must be non-negative.")

        super().__init__(cost=np.zeros(2, dtype=np.float64))
        self._model = model
        self._frame_task = frame_task
        self._dof_indices = np.asarray(dof_indices, dtype=int)
        self._control_dt = float(control_dt)
        self._substep_dt = float(substep_dt)
        self._linear_speed_slow = float(linear_speed_slow)
        self._linear_speed_fast = float(linear_speed_fast)
        self._activation_rise_rate = float(activation_rise_rate)
        self._activation_fall_rate = float(activation_fall_rate)
        self._velocity_cost_fast = float(velocity_cost_fast)
        self._velocity_scale = float(velocity_scale)
        self._return_rate = float(return_rate)
        self._max_return_speed = float(max_return_speed)
        self._max_return_acceleration = float(max_return_acceleration)
        self._corridor_cost_fast = float(corridor_cost_fast)
        self._corridor_margin_slow = float(corridor_margin_slow)
        self._corridor_margin_fast = float(corridor_margin_fast)
        self._corridor_return_rate = float(corridor_return_rate)
        self._corridor_max_speed = float(corridor_max_speed)
        self._projection_mode = projection_mode
        self._characteristic_length = float(characteristic_length)
        self._min_swivel_radius = float(min_swivel_radius)
        self._coordinate = ElbowSwivelCoordinate(
            model,
            side,
            frame_task,
            home_qpos,
            self._dof_indices,
            finite_difference_epsilon=finite_difference_epsilon,
        )

        self._previous_target_position: np.ndarray | None = None
        self._activation = 0.0
        self._pre_fast_swivel: float | None = None
        self._fast_anchor_swivel: float | None = None
        self._desired_swivel_speed = 0.0
        self._error = np.zeros(2, dtype=np.float64)
        self._jacobian = np.zeros((2, model.nv), dtype=np.float64)
        self.last_state: SpeedScheduledElbowState | None = None

    def reset(self) -> None:
        """Clear target history and all speed-scheduling state."""
        self._previous_target_position = None
        self._activation = 0.0
        self._pre_fast_swivel = None
        self._fast_anchor_swivel = None
        self._desired_swivel_speed = 0.0
        self.cost.fill(0.0)
        self._error.fill(0.0)
        self._jacobian.fill(0.0)
        self.last_state = None

    def prepare(
        self,
        configuration: mink.Configuration,
        target_pose: npt.ArrayLike,
    ) -> None:
        """Prepare a fixed task linearization for one outer control cycle."""
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (7,) or not np.all(np.isfinite(target_pose)):
            raise ValueError("Target pose must contain seven finite values.")

        linear_speed = self._target_linear_speed(target_pose[:3])
        target_activation = _smoothstep(
            (linear_speed - self._linear_speed_slow)
            / (self._linear_speed_fast - self._linear_speed_slow)
        )
        difference = target_activation - self._activation
        rate = (
            self._activation_rise_rate
            if difference > 0.0
            else self._activation_fall_rate
        )
        self._activation += float(
            np.clip(
                difference,
                -rate * self._control_dt,
                rate * self._control_dt,
            )
        )
        if self._activation <= 1e-12:
            self._activation = 0.0
            self._fast_anchor_swivel = None
            self._desired_swivel_speed = 0.0
            self.cost.fill(0.0)
            self._error.fill(0.0)
            self._jacobian.fill(0.0)
            self.last_state = SpeedScheduledElbowState(
                linear_speed=linear_speed,
                target_activation=target_activation,
                activation=0.0,
                swivel=0.0,
                swivel_radius=0.0,
                anchor_swivel=0.0,
                corridor_lower=-self._corridor_margin_slow,
                corridor_upper=self._corridor_margin_slow,
                desired_swivel_speed=0.0,
                corridor_error=0.0,
                projection_mode=self._projection_mode,
                task_row_norm=0.0,
                task_row_nullspace_residual=0.0,
            )
            return

        swivel, direct_row, radius = self._coordinate.linearize(configuration.q)
        anchor = self._update_anchor(swivel)
        margin_blend = (1.0 - self._activation) ** 2
        corridor_margin = (
            self._corridor_margin_fast
            + (self._corridor_margin_slow - self._corridor_margin_fast) * margin_blend
        )
        corridor_lower = min(0.0, anchor) - corridor_margin
        corridor_upper = max(0.0, anchor) + corridor_margin

        task_row = self._project_task_row(configuration, direct_row)
        row_residual = self._row_nullspace_residual(configuration, task_row)
        if radius < self._min_swivel_radius:
            task_row.fill(0.0)

        raw_return_speed = float(
            np.clip(
                -self._return_rate * swivel,
                -self._max_return_speed,
                self._max_return_speed,
            )
        )
        if self._max_return_acceleration > 0.0:
            max_delta = self._max_return_acceleration * self._control_dt
            self._desired_swivel_speed += float(
                np.clip(
                    raw_return_speed - self._desired_swivel_speed,
                    -max_delta,
                    max_delta,
                )
            )
        else:
            self._desired_swivel_speed = raw_return_speed

        sqrt_activation = float(np.sqrt(self._activation))
        self.cost[0] = sqrt_activation * self._velocity_cost_fast
        velocity_denominator = self._velocity_scale * self._substep_dt
        self._jacobian[0] = task_row / velocity_denominator
        desired_displacement = self._desired_swivel_speed * self._substep_dt
        self._error[0] = -desired_displacement / velocity_denominator

        corridor_error = 0.0
        if swivel < corridor_lower:
            corridor_error = swivel - corridor_lower
        elif swivel > corridor_upper:
            corridor_error = swivel - corridor_upper
        corridor_return_speed = float(
            np.clip(
                -self._corridor_return_rate * corridor_error,
                -self._corridor_max_speed,
                self._corridor_max_speed,
            )
        )
        self.cost[1] = sqrt_activation * self._corridor_cost_fast
        self._jacobian[1] = task_row / velocity_denominator
        self._error[1] = -(
            corridor_return_speed * self._substep_dt / velocity_denominator
        )
        if corridor_error == 0.0 or radius < self._min_swivel_radius:
            self._jacobian[1].fill(0.0)

        self.last_state = SpeedScheduledElbowState(
            linear_speed=linear_speed,
            target_activation=target_activation,
            activation=self._activation,
            swivel=swivel,
            swivel_radius=radius,
            anchor_swivel=anchor,
            corridor_lower=corridor_lower,
            corridor_upper=corridor_upper,
            desired_swivel_speed=self._desired_swivel_speed,
            corridor_error=corridor_error,
            projection_mode=self._projection_mode,
            task_row_norm=float(np.linalg.norm(task_row)),
            task_row_nullspace_residual=row_residual,
        )

    def _target_linear_speed(self, target_position: np.ndarray) -> float:
        if self._previous_target_position is None:
            self._previous_target_position = target_position.copy()
            return 0.0
        speed = float(
            np.linalg.norm(target_position - self._previous_target_position)
            / self._control_dt
        )
        self._previous_target_position = target_position.copy()
        return speed

    def _update_anchor(self, swivel: float) -> float:
        if self._activation >= 0.8 and self._fast_anchor_swivel is None:
            self._fast_anchor_swivel = (
                swivel if self._pre_fast_swivel is None else self._pre_fast_swivel
            )
        elif self._fast_anchor_swivel is not None and self._activation <= 0.05:
            self._fast_anchor_swivel = None
        if self._fast_anchor_swivel is None:
            self._pre_fast_swivel = swivel
            return swivel
        return self._fast_anchor_swivel

    def _project_task_row(
        self,
        configuration: mink.Configuration,
        direct_row: np.ndarray,
    ) -> np.ndarray:
        if self._projection_mode == "direct":
            return direct_row.copy()
        arm_jacobian = normalized_arm_jacobian(
            self._frame_task,
            configuration,
            self._dof_indices,
            self._characteristic_length,
        )
        direction, _ = structural_nullspace_direction(arm_jacobian)
        projected = np.zeros(self._model.nv, dtype=np.float64)
        arm_row = direct_row[self._dof_indices]
        projected[self._dof_indices] = float(arm_row @ direction) * direction
        return projected

    def _row_nullspace_residual(
        self,
        configuration: mink.Configuration,
        task_row: np.ndarray,
    ) -> float:
        arm_jacobian = normalized_arm_jacobian(
            self._frame_task,
            configuration,
            self._dof_indices,
            self._characteristic_length,
        )
        return float(np.linalg.norm(arm_jacobian @ task_row[self._dof_indices]))

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        """Return the prepared normalized displacement errors."""
        del configuration
        return self._error

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        """Return the prepared spatial-elbow task rows."""
        del configuration
        return self._jacobian

    def compute_qp_objective(self, configuration: mink.Configuration) -> mink.Objective:
        """Assemble the cached outer-cycle terms without another finite difference."""
        return self._assemble_qp(self._error, self._jacobian, configuration._eye_nv)
