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

"""Prevent a scalar joint from reaching its lower guard at excessive speed."""

from __future__ import annotations

import mink
import mujoco
import numpy as np


class LowerBoundBrakingLimit(mink.Limit):
    """Apply a velocity envelope above a lower joint guard.

    Two profiles are supported for margin ``m = q - guard_position``:

    - ``acceleration``: ``min(v_max, sqrt(2 * a * max(m, 0)))``.
    - ``distance``: ``v_max * smoothstep(clamp(m / m_slow, 0, 1))``.

    The QP inequality is expressed in configuration displacement because that
    is Mink's decision variable: ``delta_q >= -v_approach * dt``.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_qpos_index: int,
        joint_dof_index: int,
        *,
        guard_position: float,
        max_velocity: float,
        profile: str = "acceleration",
        max_deceleration: float | None = None,
        slowdown_distance: float | None = None,
    ) -> None:
        """Initialize the lower-bound braking limit for one scalar joint."""
        joint_ids = np.flatnonzero(model.jnt_qposadr == joint_qpos_index)
        if joint_ids.size != 1:
            raise ValueError(
                f"Expected qpos index {joint_qpos_index} to start one joint."
            )
        joint_id = int(joint_ids[0])
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError("Braking limit requires a scalar hinge or slide joint.")
        if int(model.jnt_dofadr[joint_id]) != joint_dof_index:
            raise ValueError(
                "Joint qpos and DoF indices do not refer to the same joint."
            )
        if not np.isfinite(guard_position):
            raise ValueError("Guard position must be finite.")
        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            if guard_position < lower or guard_position > upper:
                raise ValueError(
                    "Guard position must lie within the physical joint range."
                )
        if not np.isfinite(max_velocity) or max_velocity <= 0.0:
            raise ValueError("Maximum velocity must be finite and positive.")
        if profile not in {"acceleration", "distance"}:
            raise ValueError(
                "Braking profile must be either 'acceleration' or 'distance'."
            )
        if profile == "acceleration" and (
            max_deceleration is None
            or not np.isfinite(max_deceleration)
            or max_deceleration <= 0.0
        ):
            raise ValueError(
                "Maximum deceleration must be finite and positive for the "
                "acceleration profile."
            )
        if profile == "distance" and (
            slowdown_distance is None
            or not np.isfinite(slowdown_distance)
            or slowdown_distance <= 0.0
        ):
            raise ValueError(
                "Slowdown distance must be finite and positive for the "
                "distance profile."
            )

        self.model = model
        self.joint_qpos_index = joint_qpos_index
        self.joint_dof_index = joint_dof_index
        self.guard_position = float(guard_position)
        self.max_velocity = float(max_velocity)
        self.profile = profile
        self.max_deceleration = (
            float(max_deceleration) if max_deceleration is not None else None
        )
        self.slowdown_distance = (
            float(slowdown_distance) if slowdown_distance is not None else None
        )
        self._G = np.zeros((1, model.nv), dtype=np.float64)
        self._G[0, joint_dof_index] = -1.0

    def compute_qp_inequalities(
        self,
        configuration: mink.Configuration,
        dt: float,
    ) -> mink.Constraint:
        """Return the one-sided approach displacement inequality."""
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        position = float(configuration.q[self.joint_qpos_index])
        margin = max(position - self.guard_position, 0.0)
        if self.profile == "acceleration":
            assert self.max_deceleration is not None
            braking_velocity = np.sqrt(2.0 * self.max_deceleration * margin)
            approach_velocity = min(self.max_velocity, braking_velocity)
        else:
            assert self.slowdown_distance is not None
            u = np.clip(margin / self.slowdown_distance, 0.0, 1.0)
            activation = u * u * (3.0 - 2.0 * u)
            approach_velocity = self.max_velocity * activation
        return mink.Constraint(
            G=self._G,
            h=np.array([dt * approach_velocity], dtype=np.float64),
        )
