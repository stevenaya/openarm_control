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

"""One-sided scalar-joint soft limit."""

from __future__ import annotations

import mink
import mujoco
import numpy as np


def _smoothstep01(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class SoftLimitTask(mink.Task):
    """Slowly return a scalar joint above a positive lower soft limit."""

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_qpos_index: int,
        joint_dof_index: int,
        *,
        cost: float,
        dt: float,
        limit: float,
        max_speed: float,
    ) -> None:
        """Initialize the lower soft limit with a physical speed cap."""
        if cost < 0.0:
            raise ValueError("cost must be non-negative.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if limit <= 0.0:
            raise ValueError("limit must be positive.")
        if max_speed < 0.0:
            raise ValueError("max_speed must be non-negative.")
        if joint_qpos_index < 0 or joint_qpos_index >= model.nq:
            raise ValueError("Joint qpos index is outside the model configuration.")
        if joint_dof_index < 0 or joint_dof_index >= model.nv:
            raise ValueError("Joint DoF index is outside the model tangent space.")

        super().__init__(cost=np.array([cost], dtype=np.float64))
        self._base_cost = cost
        self._model = model
        self._joint_qpos_index = joint_qpos_index
        self._joint_dof_index = joint_dof_index
        self._dt = dt
        self._limit = limit
        self._max_speed = max_speed

    def _compute_terms(
        self, configuration: mink.Configuration
    ) -> tuple[np.ndarray, np.ndarray]:
        joint_position = float(configuration.q[self._joint_qpos_index])
        position_error = max(self._limit - joint_position, 0.0)
        activation = _smoothstep01(position_error / self._limit)
        displacement = min(position_error, self._max_speed * self._dt)
        self.cost[0] = np.sqrt(activation) * self._base_cost

        jacobian = np.zeros((1, self._model.nv), dtype=np.float64)
        jacobian[0, self._joint_dof_index] = 1.0
        error = np.array([-displacement], dtype=np.float64)
        return error, jacobian

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        """Return the one-step positive-joint displacement error."""
        error, _ = self._compute_terms(configuration)
        return error

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        """Return a Jacobian that selects only the configured joint."""
        _, jacobian = self._compute_terms(configuration)
        return jacobian

    def compute_qp_objective(self, configuration: mink.Configuration) -> mink.Objective:
        """Assemble the objective from one shared activation calculation."""
        error, jacobian = self._compute_terms(configuration)
        return self._assemble_qp(error, jacobian, configuration._eye_nv)
