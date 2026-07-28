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

"""Frame task with a bounded first-order error request."""

from __future__ import annotations

import mink
import numpy as np


def _limit_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    output = np.asarray(vector, dtype=np.float64).copy()
    norm = float(np.linalg.norm(output))
    if limit > 0.0 and norm > limit:
        output *= limit / norm
    return output


class ErrorLimitedFrameTask(mink.FrameTask):
    """Limit the pose error presented to one ordinary Mink task.

    The full target remains stored by :class:`mink.FrameTask`. Only the
    first-order error in ``J Delta q = -gain * error`` is norm-limited, so
    later control cycles continue closing any tracking lag. A zero limit keeps
    that part of the error unchanged.
    """

    def __init__(
        self,
        *,
        frame_name: str,
        frame_type: str,
        position_cost: np.ndarray | float,
        orientation_cost: np.ndarray | float,
        position_error_limit: float,
        orientation_error_limit: float,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ) -> None:
        """Configure independent translational and rotational error bounds."""
        if not np.isfinite(position_error_limit) or position_error_limit < 0.0:
            raise ValueError(
                "Frame position_error_limit must be finite and non-negative."
            )
        if not np.isfinite(orientation_error_limit) or orientation_error_limit < 0.0:
            raise ValueError(
                "Frame orientation_error_limit must be finite and non-negative."
            )
        super().__init__(
            frame_name=frame_name,
            frame_type=frame_type,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            gain=gain,
            lm_damping=lm_damping,
        )
        self.position_error_limit = float(position_error_limit)
        self.orientation_error_limit = float(orientation_error_limit)
        self.limit_activation = 1.0

    def set_limit_activation(self, activation: float) -> None:
        """Blend the bounded request with the native full FrameTask error."""
        if not np.isfinite(activation) or not 0.0 <= activation <= 1.0:
            raise ValueError("Frame error-limit activation must be in [0, 1].")
        self.limit_activation = float(activation)

    def compute_limited_error(
        self,
        configuration: mink.Configuration,
    ) -> np.ndarray:
        """Return the frame error after independent SE(3) norm limits."""
        error = super().compute_error(configuration)
        limited_position = _limit_norm(error[:3], self.position_error_limit)
        limited_orientation = _limit_norm(
            error[3:],
            self.orientation_error_limit,
        )
        error[:3] += self.limit_activation * (limited_position - error[:3])
        error[3:] += self.limit_activation * (limited_orientation - error[3:])
        return error

    def compute_qp_objective(
        self,
        configuration: mink.Configuration,
    ) -> mink.Objective:
        """Assemble the normal Mink objective with the bounded error request."""
        if self.limit_activation == 0.0 or (
            self.position_error_limit == 0.0 and self.orientation_error_limit == 0.0
        ):
            return super().compute_qp_objective(configuration)
        error = self.compute_limited_error(configuration)
        jacobian = super().compute_jacobian(configuration)
        return self._assemble_qp(error, jacobian, configuration._eye_nv)
