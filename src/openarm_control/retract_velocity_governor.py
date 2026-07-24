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

"""Context-gated J1/J4 velocity caps for arm-retraction motions."""

from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class RetractVelocityState:
    """Diagnostics from the most recent governor update."""

    perpendicular_radius: float
    reach: float
    perpendicular_retract_speed: float
    reach_retract_speed: float
    retract_speed: float
    activation: float
    joint1_cap: float
    joint4_cap: float


class RetractVelocityGovernor:
    """Temporarily relax J1/J4 caps only during genuine shoulder retraction."""

    def __init__(
        self,
        model: mujoco.MjModel,
        side: str,
        *,
        dt: float,
        joint1_base_speed: float,
        joint4_base_speed: float,
        joint1_high_speed: float,
        joint4_high_speed: float,
        deadband: float,
        full_speed: float,
        cap_slew_rate: float,
    ) -> None:
        """Initialize one side's geometric gate and cap state."""
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported arm side: {side!r}.")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive.")
        if not 0.0 <= deadband < full_speed:
            raise ValueError("Expected 0 <= deadband < full_speed.")
        if not np.isfinite(cap_slew_rate) or cap_slew_rate <= 0.0:
            raise ValueError("cap_slew_rate must be finite and positive.")

        base_caps = np.array(
            [joint1_base_speed, joint4_base_speed],
            dtype=np.float64,
        )
        high_caps = np.array(
            [joint1_high_speed, joint4_high_speed],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(base_caps)) or np.any(base_caps <= 0.0):
            raise ValueError("Base joint speeds must be finite and positive.")
        if np.any(~np.isfinite(high_caps)) or np.any(high_caps < base_caps):
            raise ValueError(
                "High joint speeds must be finite and at least their base speeds."
            )

        shoulder_joint_name = f"openarm_{side}_joint1"
        shoulder_joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            shoulder_joint_name,
        )
        if shoulder_joint_id < 0:
            raise ValueError(
                f"MuJoCo model has no joint named {shoulder_joint_name!r}."
            )

        self.model = model
        self.side = side
        self.joint_names = (
            shoulder_joint_name,
            f"openarm_{side}_joint4",
        )
        self._shoulder_joint_id = shoulder_joint_id
        self._dt = float(dt)
        self._base_caps = base_caps
        self._high_caps = high_caps
        self._deadband = float(deadband)
        self._full_speed = float(full_speed)
        self._cap_slew_rate = float(cap_slew_rate)
        self._caps = base_caps.copy()
        self._previous_perpendicular_radius: float | None = None
        self._previous_reach: float | None = None
        self.last_state: RetractVelocityState | None = None

    def update(
        self,
        configuration: mink.Configuration,
        target_position: npt.ArrayLike,
    ) -> dict[str, float]:
        """Update and return this side's effective J1/J4 velocity caps."""
        target = np.asarray(target_position, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_position must contain three finite values.")

        axis = np.asarray(
            configuration.data.xaxis[self._shoulder_joint_id],
            dtype=np.float64,
        )
        axis_norm = float(np.linalg.norm(axis))
        if not np.isfinite(axis_norm) or axis_norm <= 0.0:
            raise ValueError("Shoulder joint axis is invalid.")
        axis = axis / axis_norm
        shoulder = np.asarray(
            configuration.data.xanchor[self._shoulder_joint_id],
            dtype=np.float64,
        )
        displacement = target - shoulder
        perpendicular = displacement - axis * float(axis @ displacement)
        perpendicular_radius = float(np.linalg.norm(perpendicular))
        reach = float(np.linalg.norm(displacement))

        perpendicular_retract_speed = self._retract_speed(
            perpendicular_radius,
            self._previous_perpendicular_radius,
        )
        reach_retract_speed = self._retract_speed(
            reach,
            self._previous_reach,
        )
        retract_speed = min(
            perpendicular_retract_speed,
            reach_retract_speed,
        )
        normalized = (retract_speed - self._deadband) / (
            self._full_speed - self._deadband
        )
        u = float(np.clip(normalized, 0.0, 1.0))
        activation = u * u * (3.0 - 2.0 * u)
        desired_caps = self._base_caps + activation * (
            self._high_caps - self._base_caps
        )
        max_cap_step = self._cap_slew_rate * self._dt
        self._caps += np.clip(
            desired_caps - self._caps,
            -max_cap_step,
            max_cap_step,
        )

        self._previous_perpendicular_radius = perpendicular_radius
        self._previous_reach = reach
        self.last_state = RetractVelocityState(
            perpendicular_radius=perpendicular_radius,
            reach=reach,
            perpendicular_retract_speed=perpendicular_retract_speed,
            reach_retract_speed=reach_retract_speed,
            retract_speed=retract_speed,
            activation=activation,
            joint1_cap=float(self._caps[0]),
            joint4_cap=float(self._caps[1]),
        )
        return self.velocity_limits

    @property
    def velocity_limits(self) -> dict[str, float]:
        """Return the current J1/J4 caps by MuJoCo joint name."""
        return {
            joint_name: float(cap)
            for joint_name, cap in zip(
                self.joint_names,
                self._caps,
                strict=True,
            )
        }

    def reset(self) -> None:
        """Reset target history and restore the base caps."""
        self._caps = self._base_caps.copy()
        self._previous_perpendicular_radius = None
        self._previous_reach = None
        self.last_state = None

    def _retract_speed(
        self,
        value: float,
        previous: float | None,
    ) -> float:
        if previous is None:
            return 0.0
        return max(-(value - previous) / self._dt, 0.0)
