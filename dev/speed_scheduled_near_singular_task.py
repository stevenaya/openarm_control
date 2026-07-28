#!/usr/bin/env python3
"""Experimental ordinary-Mink task for the first near-null joint direction."""

from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np

from openarm_control.nullspace_posture_task import smoothstep_activation
from openarm_control.singularity import normalized_arm_jacobian


def _speed_activation(speed: float, slow: float, fast: float) -> float:
    unit = float(np.clip((speed - slow) / (fast - slow), 0.0, 1.0))
    return unit * unit * (3.0 - 2.0 * unit)


@dataclass(frozen=True)
class NearSingularState:
    """Diagnostics from the most recent QP linearization."""

    linear_speed: float
    speed_activation: float
    singularity_ratio: float
    singularity_activation: float
    combined_activation: float
    desired_speed: float


class SpeedScheduledNearSingularTask(mink.Task):
    """Penalize motion along ``V[:, -2]`` only near a singular configuration.

    For a full-row-rank 6x7 geometric Jacobian, ``V[:, -1]`` is the exact
    one-dimensional nullspace while ``V[:, -2]`` is the joint direction mapped
    to the weakest Cartesian direction. Penalizing only the latter allows the
    QP to yield that weak Cartesian component instead of rerouting a saturated
    solution through a different shoulder/elbow branch.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        frame_task: mink.FrameTask,
        dof_indices: np.ndarray,
        home_qpos: np.ndarray,
        control_dt: float,
        substep_dt: float,
        cost: float,
        linear_speed_slow: float,
        linear_speed_fast: float,
        activation_rise_rate: float,
        activation_fall_rate: float,
        singularity_low: float,
        singularity_high: float,
        characteristic_length: float,
        return_rate: float = 0.0,
        max_return_speed: float = 0.0,
    ) -> None:
        if cost < 0.0:
            raise ValueError("Near-singular task cost must be non-negative.")
        if not 0.0 <= linear_speed_slow < linear_speed_fast:
            raise ValueError("Expected 0 <= linear_speed_slow < linear_speed_fast.")
        if activation_rise_rate <= 0.0 or activation_fall_rate <= 0.0:
            raise ValueError("Near-singular activation rates must be positive.")
        if not 0.0 <= singularity_low < singularity_high:
            raise ValueError("Expected 0 <= singularity_low < singularity_high.")
        if control_dt <= 0.0 or substep_dt <= 0.0:
            raise ValueError("Near-singular task timesteps must be positive.")
        if return_rate < 0.0 or max_return_speed < 0.0:
            raise ValueError("Near-singular return parameters must be non-negative.")

        super().__init__(cost=np.zeros(1, dtype=np.float64))
        self._model = model
        self._frame_task = frame_task
        self._dof_indices = np.asarray(dof_indices, dtype=int)
        self._home_qpos = np.asarray(home_qpos, dtype=np.float64).copy()
        self._control_dt = float(control_dt)
        self._substep_dt = float(substep_dt)
        self._base_cost = float(cost)
        self._linear_speed_slow = float(linear_speed_slow)
        self._linear_speed_fast = float(linear_speed_fast)
        self._activation_rise_rate = float(activation_rise_rate)
        self._activation_fall_rate = float(activation_fall_rate)
        self._singularity_low = float(singularity_low)
        self._singularity_high = float(singularity_high)
        self._characteristic_length = float(characteristic_length)
        self._return_rate = float(return_rate)
        self._max_return_speed = float(max_return_speed)

        self._previous_target_position: np.ndarray | None = None
        self._linear_speed = 0.0
        self._speed_activation = 0.0
        self._previous_direction: np.ndarray | None = None
        self._error = np.zeros(1, dtype=np.float64)
        self._jacobian = np.zeros((1, model.nv), dtype=np.float64)
        self.last_state: NearSingularState | None = None

    def prepare(
        self,
        configuration: mink.Configuration,
        target_pose: np.ndarray,
    ) -> None:
        """Update only the desired-translation speed schedule once per tick."""
        del configuration
        target_position = np.asarray(target_pose, dtype=np.float64)[:3]
        if self._previous_target_position is None:
            self._linear_speed = 0.0
        else:
            self._linear_speed = float(
                np.linalg.norm(target_position - self._previous_target_position)
                / self._control_dt
            )
        self._previous_target_position = target_position.copy()

        target_activation = _speed_activation(
            self._linear_speed,
            self._linear_speed_slow,
            self._linear_speed_fast,
        )
        difference = target_activation - self._speed_activation
        rate = (
            self._activation_rise_rate
            if difference > 0.0
            else self._activation_fall_rate
        )
        self._speed_activation += float(
            np.clip(
                difference,
                -rate * self._control_dt,
                rate * self._control_dt,
            )
        )

    def _terms(
        self,
        configuration: mink.Configuration,
    ) -> tuple[np.ndarray, np.ndarray]:
        arm_jacobian = normalized_arm_jacobian(
            self._frame_task,
            configuration,
            self._dof_indices,
            self._characteristic_length,
        )
        _, singular_values, vt = np.linalg.svd(
            arm_jacobian,
            full_matrices=True,
        )
        direction = vt[-2].copy()
        if (
            self._previous_direction is not None
            and float(direction @ self._previous_direction) < 0.0
        ):
            direction *= -1.0
        self._previous_direction = direction.copy()

        ratio = float(singular_values[-1] / singular_values[0])
        singularity_activation = 1.0 - smoothstep_activation(
            ratio,
            self._singularity_low,
            self._singularity_high,
        )
        combined_activation = self._speed_activation * singularity_activation
        self.cost[0] = np.sqrt(combined_activation) * self._base_cost

        configuration_error = np.empty(self._model.nv, dtype=np.float64)
        mujoco.mj_differentiatePos(
            self._model,
            configuration_error,
            1.0,
            self._home_qpos,
            configuration.q,
        )
        home_error = float(direction @ configuration_error[self._dof_indices])
        desired_speed = float(
            np.clip(
                -self._return_rate * home_error,
                -self._max_return_speed,
                self._max_return_speed,
            )
        )
        self._error[0] = -desired_speed * self._substep_dt
        self._jacobian.fill(0.0)
        self._jacobian[0, self._dof_indices] = direction
        self.last_state = NearSingularState(
            linear_speed=self._linear_speed,
            speed_activation=self._speed_activation,
            singularity_ratio=ratio,
            singularity_activation=singularity_activation,
            combined_activation=combined_activation,
            desired_speed=desired_speed,
        )
        return self._error, self._jacobian

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
