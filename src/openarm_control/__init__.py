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

"""Kinematics and control utilities for OpenArm."""

from .config import ArmSetup, register_common_args, setup_from_args
from .kinematics import (
    IKParams,
    Kinematics,
    register_ik_args,
    ik_params_from_args,
)
from .joint_braking_limit import JointBrakingLimit
from .lower_bound_braking_limit import LowerBoundBrakingLimit
from .nullspace_posture_task import NullspacePostureTask
from .poses import read_ee_pose, pose_to_se3, se3_to_pose
from .recoverable_configuration_limit import RecoverableConfigurationLimit
from .retract_velocity_governor import RetractVelocityGovernor
from .soft_limit_task import SoftLimitTask
from .speed_scheduled_elbow_qp import (
    ElbowSwivelCoordinate,
    SpeedScheduledElbowParams,
    SpeedScheduledElbowQP,
)
from .singularity_approach_limit import SingularityApproachLimit

__all__ = [
    # context
    "ArmSetup",
    # high-level interface
    "Kinematics",
    "IKParams",
    "JointBrakingLimit",
    "LowerBoundBrakingLimit",
    "SoftLimitTask",
    "NullspacePostureTask",
    "RecoverableConfigurationLimit",
    "RetractVelocityGovernor",
    "ElbowSwivelCoordinate",
    "SpeedScheduledElbowParams",
    "SpeedScheduledElbowQP",
    "SingularityApproachLimit",
    # CLI helpers
    "register_common_args",
    "register_ik_args",
    "setup_from_args",
    "ik_params_from_args",
    # pose utilities
    "read_ee_pose",
    "pose_to_se3",
    "se3_to_pose",
]
