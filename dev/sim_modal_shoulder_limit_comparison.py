#!/usr/bin/env python3
"""Compare geometric and SVD-modal J1 velocity budgets in replay."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import mink
import numpy as np
from scipy.spatial.transform import Rotation

from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S
from openarm_control.kinematics import _configuration_limit_for_qpos
from openarm_control.singularity import normalized_arm_jacobian

from sim_dynamic_shoulder_limit_comparison import (
    DirectionalLimitController,
    Variant as DirectionalVariant,
)
from sim_intervention_posture_replay import (
    CHARACTERISTIC_LENGTH,
    CONTROL_DT,
    DEFAULT_RUN_DIR,
    Kinematics,
    RetractSegment,
    ReplayTrace,
    _load_recorded_commands,
    _make_kinematics,
    _save_trace,
    _source_geometry,
    detect_retract_segments,
    simulate_replay,
)
from sim_intervention_weak_direction_governor import (
    WeakDirectionTargetGovernor,
)


@dataclass(frozen=True)
class Variant:
    """One J1 velocity-budget strategy."""

    name: str
    mode: str
    total_j1_speed: float
    governor_margin: float | None = None
    modal_projector: str = "single"
    nonweak_j1_speed: float | None = None
    retract_deadband: float = 0.02
    retract_full_speed: float = 0.15
    retract_cap_slew_rate: float | None = None


@dataclass(frozen=True)
class ModalSignals:
    """SVD-modal decomposition of a solved command trajectory."""

    weak_j1_velocity: np.ndarray
    strong_j1_velocity: np.ndarray
    null_j1_velocity: np.ndarray
    nonweak_j1_velocity: np.ndarray
    weak_modal_velocity: np.ndarray
    weak_j1_participation: np.ndarray
    singular_gap_ratio: np.ndarray
    projector_change: np.ndarray
    actual_rho: np.ndarray


class ModalShoulderLimit(mink.Limit):
    """Reserve extra total J1 speed for the weakest task-space mode."""

    def __init__(
        self,
        kinematics: Kinematics,
        side: str,
        *,
        nonweak_speed: float,
        projector_mode: str,
        characteristic_length: float = CHARACTERISTIC_LENGTH,
    ) -> None:
        if nonweak_speed <= 0.0:
            raise ValueError("Nonweak J1 speed must be positive.")
        solver = kinematics._ik
        assert solver is not None
        self._model = solver._model
        self._frame_task = solver._tasks[side]
        self._arm_dofs = np.asarray(
            solver._arm_dofs_by_side[side],
            dtype=int,
        )
        self._nonweak_speed = float(nonweak_speed)
        self._projector_mode = projector_mode
        self._characteristic_length = float(characteristic_length)
        self._G: np.ndarray | None = None
        self._previous_projector: np.ndarray | None = None

        self.ratio: list[float] = []
        self.singular_gap_ratio: list[float] = []
        self.weak_j1_participation: list[float] = []
        self.projector_change: list[float] = []
        self.nonweak_row_norm: list[float] = []

    def begin_outer_step(self, configuration: mink.Configuration) -> None:
        """Freeze the current modal projector for one outer control tick."""
        jacobian = normalized_arm_jacobian(
            self._frame_task,
            configuration,
            self._arm_dofs,
            self._characteristic_length,
        )
        _, singular_values, vt = np.linalg.svd(
            jacobian,
            full_matrices=True,
        )
        weak_projector = _weak_projector(
            singular_values,
            vt,
            self._projector_mode,
        )
        nonweak_row = np.eye(self._arm_dofs.size)[0] - weak_projector[0]

        G = np.zeros((2, self._model.nv), dtype=np.float64)
        G[0, self._arm_dofs] = nonweak_row
        G[1, self._arm_dofs] = -nonweak_row
        self._G = G

        largest = float(singular_values[0])
        smallest = float(singular_values[-1])
        next_smallest = float(singular_values[-2])
        self.ratio.append(smallest / largest if largest > 0.0 else 0.0)
        self.singular_gap_ratio.append(
            smallest / next_smallest if next_smallest > 0.0 else 0.0
        )
        self.weak_j1_participation.append(
            float(np.sqrt(max(weak_projector[0, 0], 0.0)))
        )
        self.projector_change.append(
            0.0
            if self._previous_projector is None
            else float(
                np.linalg.norm(
                    weak_projector - self._previous_projector,
                    ord="fro",
                )
            )
        )
        self.nonweak_row_norm.append(float(np.linalg.norm(nonweak_row)))
        self._previous_projector = weak_projector

    def compute_qp_inequalities(
        self,
        configuration: mink.Configuration,
        dt: float,
    ) -> mink.Constraint:
        """Limit only the nonweak modal contribution to J1 velocity."""
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self._G is None:
            self.begin_outer_step(configuration)
        assert self._G is not None
        bound = self._nonweak_speed * dt
        return mink.Constraint(
            G=self._G,
            h=np.array([bound, bound], dtype=np.float64),
        )


class ModalLimitController:
    """Install a modal J1 limit and optionally govern the target stream."""

    def __init__(
        self,
        side: str,
        *,
        nonweak_speed: float,
        total_j1_speed: float,
        governor_margin: float | None,
        projector_mode: str,
    ) -> None:
        self._side = side
        self._nonweak_speed = float(nonweak_speed)
        self._projector_mode = projector_mode
        self.limit: ModalShoulderLimit | None = None
        self.governor = (
            None
            if governor_margin is None
            else WeakDirectionTargetGovernor(
                velocity_margin=governor_margin,
            )
        )
        if self.governor is not None:
            self.governor.caps[0] = total_j1_speed

    def __call__(
        self,
        kinematics: Kinematics,
        side: str,
        desired_pose: np.ndarray,
    ) -> np.ndarray:
        if side != self._side:
            raise ValueError(f"Expected side {self._side}, got {side}.")
        solver = kinematics._ik
        assert solver is not None
        if self.limit is None:
            self.limit = ModalShoulderLimit(
                kinematics,
                side,
                nonweak_speed=self._nonweak_speed,
                projector_mode=self._projector_mode,
            )
            solver._limits.append(self.limit)
        self.limit.begin_outer_step(solver._config)
        if self.governor is None:
            return np.asarray(desired_pose, dtype=np.float64)
        return self.governor(kinematics, side, desired_pose)


class HorizontalRetractShoulderLimit(mink.Limit):
    """Raise the symmetric J1 cap while target radius about J1 decreases."""

    def __init__(
        self,
        kinematics: Kinematics,
        side: str,
        *,
        base_speed: float,
        high_speed: float,
        deadband: float,
        full_speed: float,
        cap_slew_rate: float | None,
    ) -> None:
        if not 0.0 < base_speed <= high_speed:
            raise ValueError("Expected 0 < base J1 speed <= high J1 speed.")
        if not 0.0 <= deadband < full_speed:
            raise ValueError(
                "Expected 0 <= retract deadband < full activation speed."
            )
        if cap_slew_rate is not None and cap_slew_rate <= 0.0:
            raise ValueError("Cap slew rate must be positive when provided.")
        solver = kinematics._ik
        assert solver is not None
        self._model = solver._model
        self._shoulder_dof = int(solver._arm_dofs_by_side[side][0])
        self._shoulder_joint = self._model.joint(
            f"openarm_{side}_joint1"
        ).id
        self._base_speed = float(base_speed)
        self._high_speed = float(high_speed)
        self._deadband = float(deadband)
        self._full_speed = float(full_speed)
        self._cap_slew_rate = cap_slew_rate
        self._previous_radius: float | None = None
        self._cap = self._base_speed
        self._G = np.zeros((2, self._model.nv), dtype=np.float64)
        self._G[0, self._shoulder_dof] = 1.0
        self._G[1, self._shoulder_dof] = -1.0

        self.radius: list[float] = []
        self.radial_velocity: list[float] = []
        self.activation: list[float] = []
        self.cap: list[float] = []

    def begin_outer_step(
        self,
        configuration: mink.Configuration,
        desired_pose: np.ndarray,
    ) -> None:
        """Update the cap from target radius in the plane normal to J1."""
        axis = np.asarray(
            configuration.data.xaxis[self._shoulder_joint],
            dtype=np.float64,
        )
        axis /= np.linalg.norm(axis)
        shoulder = np.asarray(
            configuration.data.xanchor[self._shoulder_joint],
            dtype=np.float64,
        )
        displacement = np.asarray(desired_pose[:3], dtype=np.float64) - shoulder
        perpendicular = displacement - axis * float(axis @ displacement)
        radius = float(np.linalg.norm(perpendicular))
        radial_velocity = (
            0.0
            if self._previous_radius is None
            else (radius - self._previous_radius) / CONTROL_DT
        )
        retract_speed = max(-radial_velocity, 0.0)
        normalized = (
            (retract_speed - self._deadband)
            / (self._full_speed - self._deadband)
        )
        activation = float(_smoothstep(np.asarray(normalized)))
        desired_cap = self._base_speed + activation * (
            self._high_speed - self._base_speed
        )
        if self._cap_slew_rate is None:
            self._cap = desired_cap
        else:
            cap_step = self._cap_slew_rate * CONTROL_DT
            self._cap += float(
                np.clip(
                    desired_cap - self._cap,
                    -cap_step,
                    cap_step,
                )
            )
        self._previous_radius = radius

        self.radius.append(radius)
        self.radial_velocity.append(radial_velocity)
        self.activation.append(activation)
        self.cap.append(self._cap)

    def compute_qp_inequalities(
        self,
        configuration: mink.Configuration,
        dt: float,
    ) -> mink.Constraint:
        del configuration
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        bound = self._cap * dt
        return mink.Constraint(
            G=self._G,
            h=np.array([bound, bound], dtype=np.float64),
        )


class HorizontalRetractLimitController:
    """Install and update target-radius-gated symmetric J1 limits."""

    def __init__(
        self,
        side: str,
        *,
        base_speed: float,
        high_speed: float,
        deadband: float,
        full_speed: float,
        cap_slew_rate: float | None,
    ) -> None:
        self._side = side
        self._base_speed = float(base_speed)
        self._high_speed = float(high_speed)
        self._deadband = float(deadband)
        self._full_speed = float(full_speed)
        self._cap_slew_rate = cap_slew_rate
        self.limit: HorizontalRetractShoulderLimit | None = None

    def __call__(
        self,
        kinematics: Kinematics,
        side: str,
        desired_pose: np.ndarray,
    ) -> np.ndarray:
        if side != self._side:
            raise ValueError(f"Expected side {self._side}, got {side}.")
        solver = kinematics._ik
        assert solver is not None
        if self.limit is None:
            self.limit = HorizontalRetractShoulderLimit(
                kinematics,
                side,
                base_speed=self._base_speed,
                high_speed=self._high_speed,
                deadband=self._deadband,
                full_speed=self._full_speed,
                cap_slew_rate=self._cap_slew_rate,
            )
            solver._limits.append(self.limit)
        self.limit.begin_outer_step(solver._config, desired_pose)
        return np.asarray(desired_pose, dtype=np.float64)


class NoVelocityLimitController:
    """Remove joint-speed inequalities, optionally retaining singular braking."""

    def __init__(
        self,
        *,
        keep_joint_braking: bool,
        keep_singularity_limit: bool,
    ) -> None:
        self._installed = False
        self._keep_joint_braking = keep_joint_braking
        self._keep_singularity_limit = keep_singularity_limit

    def __call__(
        self,
        kinematics: Kinematics,
        side: str,
        desired_pose: np.ndarray,
    ) -> np.ndarray:
        del side
        if not self._installed:
            solver = kinematics._ik
            assert solver is not None
            active_qpos = {
                int(qpos)
                for arm_qpos in solver._arm_qpos_by_side.values()
                for qpos in arm_qpos
            }
            limits: list[mink.Limit] = [
                _configuration_limit_for_qpos(
                    solver._model,
                    active_qpos,
                )
            ]
            if (
                self._keep_joint_braking
                and solver._joint_braking_limit is not None
            ):
                limits.append(solver._joint_braking_limit)
            else:
                solver._joint_braking_limit = None
                solver._elbow_braking_limits.clear()
            if self._keep_singularity_limit:
                limits.extend(solver._singularity_limits.values())
            else:
                solver._singularity_limits.clear()
            solver._limits = limits
            self._installed = True
        return np.asarray(desired_pose, dtype=np.float64)


def _variants(
    base_speed: float,
    total_speed: float,
    governor_margins: list[float],
    *,
    pair_probe_nonweak_speed: float | None,
    pair_probe_total_speed: float | None,
    horizontal_retract_speeds: list[float],
    retract_deadband: float,
    retract_full_speed: float,
    retract_cap_slew_rate: float | None,
) -> list[Variant]:
    variants = [
        Variant("A_baseline_total2", "baseline", base_speed),
        Variant("H_no_velocity_box", "no_velocity_box", float("inf")),
        Variant(
            "F_no_joint_velocity_limits",
            "unlimited_joints",
            float("inf"),
        ),
        Variant("G_no_speed_limits", "unlimited", float("inf")),
        Variant("E_fixed_total3", "fixed", total_speed),
        Variant("B_reach_3_2", "reach", total_speed),
        Variant(
            "C_modal_single_3_2",
            "modal",
            total_speed,
            modal_projector="single",
        ),
        Variant(
            "C_modal_pair_3_2",
            "modal",
            total_speed,
            modal_projector="pair",
        ),
        Variant(
            "C_modal_spectral_3_2",
            "modal",
            total_speed,
            modal_projector="spectral",
        ),
    ]
    variants.extend(
        Variant(
            f"D_modal_governor_m{margin:g}",
            "modal_governor",
            total_speed,
            governor_margin=margin,
            modal_projector="single",
        )
        for margin in governor_margins
    )
    if (pair_probe_nonweak_speed is None) != (
        pair_probe_total_speed is None
    ):
        raise ValueError(
            "Pair probe requires both nonweak and total J1 speeds."
        )
    if pair_probe_nonweak_speed is not None:
        assert pair_probe_total_speed is not None
        if pair_probe_nonweak_speed > pair_probe_total_speed:
            raise ValueError(
                "Pair-probe nonweak speed cannot exceed total speed."
            )
        low = _number_label(pair_probe_nonweak_speed)
        high = _number_label(pair_probe_total_speed)
        variants.extend(
            [
                Variant(
                    f"P_fixed_low_{low}",
                    "fixed",
                    pair_probe_nonweak_speed,
                ),
                Variant(
                    f"P_modal_pair_{high}_{low}",
                    "modal",
                    pair_probe_total_speed,
                    modal_projector="pair",
                    nonweak_j1_speed=pair_probe_nonweak_speed,
                ),
                Variant(
                    f"P_fixed_high_{high}",
                    "fixed",
                    pair_probe_total_speed,
                ),
            ]
        )
    variants.extend(
        Variant(
            f"R_horizontal_retract_{_number_label(high_speed)}",
            "horizontal_retract",
            high_speed,
            retract_deadband=retract_deadband,
            retract_full_speed=retract_full_speed,
            retract_cap_slew_rate=retract_cap_slew_rate,
        )
        for high_speed in horizontal_retract_speeds
    )
    return variants


def _number_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _controller_for(
    side: str,
    variant: Variant,
    *,
    base_speed: float,
    nonweak_speed: float,
) -> (
    DirectionalLimitController
    | HorizontalRetractLimitController
    | ModalLimitController
    | NoVelocityLimitController
    | WeakDirectionTargetGovernor
    | None
):
    if variant.mode in {"baseline", "fixed"}:
        return None
    if variant.mode in {
        "no_velocity_box",
        "unlimited",
        "unlimited_joints",
    }:
        return NoVelocityLimitController(
            keep_joint_braking=variant.mode == "no_velocity_box",
            keep_singularity_limit=variant.mode != "unlimited",
        )
    if variant.mode == "reach":
        return DirectionalLimitController(
            side,
            DirectionalVariant(
                name=variant.name,
                high_speed=variant.total_j1_speed,
                direction_mode="reach",
            ),
            base_speed=base_speed,
        )
    if variant.mode == "horizontal_retract":
        return HorizontalRetractLimitController(
            side,
            base_speed=base_speed,
            high_speed=variant.total_j1_speed,
            deadband=variant.retract_deadband,
            full_speed=variant.retract_full_speed,
            cap_slew_rate=variant.retract_cap_slew_rate,
        )
    if variant.mode in {"modal", "modal_governor"}:
        return ModalLimitController(
            side,
            nonweak_speed=(
                nonweak_speed
                if variant.nonweak_j1_speed is None
                else variant.nonweak_j1_speed
            ),
            total_j1_speed=variant.total_j1_speed,
            governor_margin=variant.governor_margin,
            projector_mode=variant.modal_projector,
        )
    raise ValueError(f"Unknown variant mode: {variant.mode}")


def _simulate_variant(
    side: str,
    variant: Variant,
    target_pose: np.ndarray,
    source_q: np.ndarray,
    source_elbow: np.ndarray,
    *,
    settle_duration: float,
    base_speed: float,
    nonweak_speed: float,
) -> tuple[
    ReplayTrace,
    DirectionalLimitController
    | HorizontalRetractLimitController
    | ModalLimitController
    | NoVelocityLimitController
    | WeakDirectionTargetGovernor
    | None,
]:
    controller = _controller_for(
        side,
        variant,
        base_speed=base_speed,
        nonweak_speed=nonweak_speed,
    )
    if variant.mode in {"unlimited", "unlimited_joints"}:
        multiplier = 100.0
    elif variant.mode == "no_velocity_box":
        multiplier = 1.0
    else:
        multiplier = (
            variant.total_j1_speed
            / float(ARM_JOINT_VELOCITY_LIMITS_RAD_S[0])
        )
    trace = simulate_replay(
        side,
        0.0,
        target_pose,
        source_q,
        source_elbow,
        settle_duration=settle_duration,
        velocity_limit_joint_multipliers=np.array(
            [multiplier, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float64,
        ),
        target_governor=controller,
    )
    return trace, controller


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _weak_projector(
    singular_values: np.ndarray,
    vt: np.ndarray,
    mode: str,
) -> np.ndarray:
    task_vt = vt[: singular_values.size]
    if mode == "single":
        weak_direction = task_vt[-1]
        return np.outer(weak_direction, weak_direction)
    if mode == "pair":
        weak_basis = task_vt[-2:]
        return weak_basis.T @ weak_basis
    if mode == "spectral":
        ratios = singular_values / singular_values[0]
        normalized = (ratios - 0.02) / (0.08 - 0.02)
        weights = 1.0 - _smoothstep(normalized)
        return task_vt.T @ np.diag(weights) @ task_vt
    raise ValueError(f"Unknown modal projector mode: {mode}")


def _modal_signals(
    side: str,
    trace: ReplayTrace,
    projector_mode: str,
) -> ModalSignals:
    kinematics = _make_kinematics(side, 0.0)
    solver = kinematics._ik
    assert solver is not None
    configuration = solver._config
    frame_task = solver._tasks[side]
    dofs = np.asarray(solver._arm_dofs_by_side[side], dtype=int)

    count = trace.time.size
    weak_j1 = np.empty(count, dtype=np.float64)
    strong_j1 = np.empty(count, dtype=np.float64)
    null_j1 = np.empty(count, dtype=np.float64)
    nonweak_j1 = np.empty(count, dtype=np.float64)
    weak_modal = np.empty(count, dtype=np.float64)
    participation = np.empty(count, dtype=np.float64)
    gap = np.empty(count, dtype=np.float64)
    projector_change = np.empty(count, dtype=np.float64)
    actual_rho = np.empty(count, dtype=np.float64)
    previous_projector: np.ndarray | None = None

    pre_solve_q = trace.command_q - CONTROL_DT * trace.command_dq
    for index, (q, dq) in enumerate(zip(pre_solve_q, trace.command_dq)):
        qpos = configuration.q.copy()
        solver._joint_resolver.set_qpos(
            qpos,
            np.append(q, 0.0),
            side,
        )
        configuration.update(q=qpos)
        jacobian = normalized_arm_jacobian(
            frame_task,
            configuration,
            dofs,
            CHARACTERISTIC_LENGTH,
        )
        _, singular_values, vt = np.linalg.svd(
            jacobian,
            full_matrices=True,
        )
        weak_projector = _weak_projector(
            singular_values,
            vt,
            projector_mode,
        )
        null_direction = vt[-1]
        null_coefficient = float(null_direction @ dq)
        weak_velocity = weak_projector @ dq
        weak_j1[index] = weak_velocity[0]
        null_j1[index] = null_direction[0] * null_coefficient
        strong_j1[index] = dq[0] - weak_j1[index] - null_j1[index]
        nonweak_j1[index] = dq[0] - weak_j1[index]
        weak_modal[index] = float(np.linalg.norm(weak_velocity))
        participation[index] = float(
            np.sqrt(max(weak_projector[0, 0], 0.0))
        )
        gap[index] = singular_values[-1] / singular_values[-2]
        projector = weak_projector
        projector_change[index] = (
            0.0
            if previous_projector is None
            else float(
                np.linalg.norm(
                    projector - previous_projector,
                    ord="fro",
                )
            )
        )
        previous_projector = projector

        qpos = configuration.q.copy()
        solver._joint_resolver.set_qpos(
            qpos,
            np.append(trace.actual_q[index], 0.0),
            side,
        )
        configuration.update(q=qpos)
        actual_jacobian = normalized_arm_jacobian(
            frame_task,
            configuration,
            dofs,
            CHARACTERISTIC_LENGTH,
        )
        actual_singular_values = np.linalg.svd(
            actual_jacobian,
            compute_uv=False,
        )
        actual_rho[index] = (
            actual_singular_values[-1] / actual_singular_values[0]
        )

    return ModalSignals(
        weak_j1_velocity=weak_j1,
        strong_j1_velocity=strong_j1,
        null_j1_velocity=null_j1,
        nonweak_j1_velocity=nonweak_j1,
        weak_modal_velocity=weak_modal,
        weak_j1_participation=participation,
        singular_gap_ratio=gap,
        projector_change=projector_change,
        actual_rho=actual_rho,
    )


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _orientation_error(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    target_rotation = Rotation.from_quat(target[:, [4, 5, 6, 3]])
    actual_rotation = Rotation.from_quat(actual[:, [4, 5, 6, 3]])
    return (target_rotation.inv() * actual_rotation).magnitude()


def _summary_row(
    segment: int | str,
    variant: Variant,
    trace: ReplayTrace,
    signals: ModalSignals,
    controller: (
        DirectionalLimitController
        | HorizontalRetractLimitController
        | ModalLimitController
        | NoVelocityLimitController
        | WeakDirectionTargetGovernor
        | None
    ),
    core: slice,
    *,
    nonweak_speed: float,
) -> dict[str, float | int | str]:
    target_velocity = np.linalg.norm(
        np.gradient(trace.target_pose[:, :3], CONTROL_DT, axis=0),
        axis=1,
    )
    position_error = np.linalg.norm(
        trace.actual_pose[:, :3] - trace.target_pose[:, :3],
        axis=1,
    )
    governed_position_error = np.linalg.norm(
        trace.ik_target_pose[:, :3] - trace.target_pose[:, :3],
        axis=1,
    )
    orientation_error = _orientation_error(
        trace.target_pose,
        trace.actual_pose,
    )
    governed_orientation_error = _orientation_error(
        trace.target_pose,
        trace.ik_target_pose,
    )
    command_acceleration = np.gradient(
        trace.command_dq[:, 0],
        CONTROL_DT,
    )
    actual_acceleration = np.gradient(
        trace.actual_dq[:, 0],
        CONTROL_DT,
    )
    actual_ee_velocity = np.gradient(
        trace.actual_pose[:, :3],
        CONTROL_DT,
        axis=0,
    )
    actual_ee_acceleration = np.linalg.norm(
        np.gradient(actual_ee_velocity, CONTROL_DT, axis=0),
        axis=1,
    )
    elbow_y = trace.actual_elbow[:, 1]
    base_caps = np.asarray(
        ARM_JOINT_VELOCITY_LIMITS_RAD_S,
        dtype=np.float64,
    )
    caps = (
        np.full(7, np.inf, dtype=np.float64)
        if variant.mode
        in {"no_velocity_box", "unlimited", "unlimited_joints"}
        else base_caps.copy()
    )
    if variant.mode not in {
        "no_velocity_box",
        "unlimited",
        "unlimited_joints",
    }:
        caps[0] = variant.total_j1_speed
    utilization = np.abs(trace.command_dq) / caps
    horizontal_limit = (
        controller.limit
        if isinstance(controller, HorizontalRetractLimitController)
        else None
    )
    if horizontal_limit is None:
        j1_cap = np.full(trace.time.shape, caps[0], dtype=np.float64)
        retract_activation = np.zeros(trace.time.shape, dtype=np.float64)
        retract_radial_velocity = np.zeros(
            trace.time.shape,
            dtype=np.float64,
        )
    else:
        j1_cap = np.asarray(horizontal_limit.cap, dtype=np.float64)
        retract_activation = np.asarray(
            horizontal_limit.activation,
            dtype=np.float64,
        )
        retract_radial_velocity = np.asarray(
            horizontal_limit.radial_velocity,
            dtype=np.float64,
        )
        utilization[:, 0] = np.abs(trace.command_dq[:, 0]) / j1_cap
    nominal_utilization = np.abs(trace.command_dq) / base_caps
    effective_rho = np.minimum(trace.rho, signals.actual_rho)
    singularity_activation = np.square(
        _smoothstep((effective_rho - 0.02) / (0.08 - 0.02))
    )
    rho_approach_rate = np.maximum(
        -np.gradient(trace.rho, CONTROL_DT),
        0.0,
    )
    weak_abs = np.abs(signals.weak_j1_velocity)
    nonweak_abs = np.abs(signals.nonweak_j1_velocity)
    modal_denominator = weak_abs + nonweak_abs
    modal_share = np.divide(
        weak_abs,
        modal_denominator,
        out=np.zeros_like(weak_abs),
        where=modal_denominator > 1e-9,
    )
    effective_nonweak_speed = (
        variant.nonweak_j1_speed
        if variant.nonweak_j1_speed is not None
        else nonweak_speed
    )

    governor = (
        controller.governor
        if isinstance(controller, ModalLimitController)
        else None
    )
    if governor is None:
        governor_active = 0.0
        mean_weak_scale = 1.0
        predicted_before = 0.0
        predicted_after = 0.0
    else:
        weak_scale = np.asarray(governor.weak_scale)
        governor_active = float(np.mean(weak_scale[core] < 1.0 - 1e-9))
        mean_weak_scale = float(np.mean(weak_scale[core]))
        predicted_before = float(
            np.max(
                np.asarray(governor.predicted_peak_utilization_before)[core]
            )
        )
        predicted_after = float(
            np.max(
                np.asarray(governor.predicted_peak_utilization_after)[core]
            )
        )

    row: dict[str, float | int | str] = {
        "segment": segment,
        "variant": variant.name,
        "modal_projector": variant.modal_projector,
        "target_mean_speed_m_s": float(np.mean(target_velocity[core])),
        "target_peak_speed_m_s": float(np.max(target_velocity[core])),
        "min_rho": float(np.min(trace.rho[core])),
        "total_j1_cap_rad_s": variant.total_j1_speed,
        "nonweak_j1_cap_rad_s": effective_nonweak_speed,
        "j1_total_limit_fraction": float(
            np.mean(utilization[core, 0] >= 0.98)
        ),
        "j1_nonweak_limit_fraction": float(
            0.0
            if variant.mode not in {"modal", "modal_governor"}
            else np.mean(
                np.abs(signals.nonweak_j1_velocity[core])
                >= 0.98 * effective_nonweak_speed
            )
        ),
        "horizontal_retract_active_fraction": float(
            np.mean(retract_activation[core] > 1e-9)
        ),
        "horizontal_retract_full_fraction": float(
            np.mean(retract_activation[core] >= 1.0 - 1e-9)
        ),
        "horizontal_retract_mean_activation": float(
            np.mean(retract_activation[core])
        ),
        "horizontal_retract_mean_j1_cap_rad_s": float(
            np.mean(j1_cap[core])
        ),
        "horizontal_retract_speed_p90_m_s": float(
            np.percentile(
                np.maximum(-retract_radial_velocity[core], 0.0),
                90.0,
            )
        ),
        "other_joint_limit_fraction": float(
            np.mean(np.any(utilization[core, 1:] >= 0.98, axis=1))
        ),
        "nominal_any_velocity_limit_fraction": float(
            np.mean(np.any(nominal_utilization[core] >= 0.98, axis=1))
        ),
        "nominal_peak_velocity_utilization": float(
            np.max(nominal_utilization[core])
        ),
        "actual_min_rho": float(np.min(signals.actual_rho[core])),
        "effective_min_rho": float(np.min(effective_rho[core])),
        "singularity_limit_enabled": int(
            variant.mode != "unlimited"
        ),
        "singularity_envelope_region_fraction": float(
            np.mean(singularity_activation[core] < 1.0 - 1e-9)
        ),
        "singularity_stop_region_fraction": float(
            np.mean(effective_rho[core] <= 0.0205)
        ),
        "command_rho_approach_rate_p99_s": float(
            np.percentile(rho_approach_rate[core], 99.0)
        ),
        "j1_peak_command_speed_rad_s": float(
            np.max(np.abs(trace.command_dq[core, 0]))
        ),
        "j1_peak_actual_speed_rad_s": float(
            np.max(np.abs(trace.actual_dq[core, 0]))
        ),
        "j1_weak_contribution_p99_rad_s": float(
            np.percentile(weak_abs[core], 99.0)
        ),
        "j1_nonweak_contribution_p99_rad_s": float(
            np.percentile(nonweak_abs[core], 99.0)
        ),
        "j1_strong_contribution_p99_rad_s": float(
            np.percentile(np.abs(signals.strong_j1_velocity[core]), 99.0)
        ),
        "j1_null_contribution_p99_rad_s": float(
            np.percentile(np.abs(signals.null_j1_velocity[core]), 99.0)
        ),
        "j1_weak_share_mean": float(np.mean(modal_share[core])),
        "weak_direction_j1_participation_mean": float(
            np.mean(signals.weak_j1_participation[core])
        ),
        "weak_direction_j1_participation_max": float(
            np.max(signals.weak_j1_participation[core])
        ),
        "singular_gap_ratio_max": float(
            np.max(signals.singular_gap_ratio[core])
        ),
        "weak_projector_change_p99": float(
            np.percentile(signals.projector_change[core], 99.0)
        ),
        "actual_elbow_lateral_delta_m": float(
            elbow_y[core.stop - 1] - elbow_y[core.start]
        ),
        "actual_elbow_lateral_range_m": float(np.ptp(elbow_y[core])),
        "actual_position_rmse_m": _rms(position_error[core]),
        "actual_position_peak_m": float(np.max(position_error[core])),
        "actual_orientation_rmse_rad": _rms(orientation_error[core]),
        "governed_target_position_rmse_m": _rms(
            governed_position_error[core]
        ),
        "governed_target_orientation_rmse_rad": _rms(
            governed_orientation_error[core]
        ),
        "j1_command_acceleration_p99_rad_s2": float(
            np.percentile(np.abs(command_acceleration[core]), 99.0)
        ),
        "j1_actual_acceleration_p99_rad_s2": float(
            np.percentile(np.abs(actual_acceleration[core]), 99.0)
        ),
        "actual_ee_acceleration_p99_m_s2": float(
            np.percentile(actual_ee_acceleration[core], 99.0)
        ),
        "governor_active_fraction": governor_active,
        "governor_mean_weak_scale": mean_weak_scale,
        "governor_peak_predicted_utilization_before": predicted_before,
        "governor_peak_predicted_utilization_after": predicted_after,
        "solve_failures": int(np.count_nonzero(trace.solve_failed)),
    }
    for joint in range(7):
        row[f"j{joint + 1}_limit_fraction"] = float(
            np.mean(utilization[core, joint] >= 0.98)
        )
        row[f"j{joint + 1}_nominal_limit_fraction"] = float(
            np.mean(nominal_utilization[core, joint] >= 0.98)
        )
        row[f"j{joint + 1}_peak_command_speed_rad_s"] = float(
            np.max(np.abs(trace.command_dq[core, joint]))
        )
    return row


def _plot(
    path: Path,
    traces: dict[str, ReplayTrace],
    signals: dict[str, ModalSignals],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping plot generation.")
        return

    figure, axes = plt.subplots(7, 1, figsize=(14, 20), sharex=True)
    for name, trace in traces.items():
        time = trace.time
        axes[0].plot(time, trace.command_dq[:, 0], label=name)
        axes[1].plot(
            time,
            signals[name].weak_j1_velocity,
            label=name,
        )
        axes[2].plot(
            time,
            signals[name].nonweak_j1_velocity,
            label=name,
        )
        axes[3].plot(
            time,
            trace.actual_elbow[:, 1] - trace.actual_elbow[0, 1],
            label=name,
        )
        axes[4].plot(
            time,
            np.linalg.norm(
                trace.actual_pose[:, :3] - trace.target_pose[:, :3],
                axis=1,
            ),
            label=name,
        )
        axes[5].plot(
            time,
            np.linalg.norm(
                trace.ik_target_pose[:, :3] - trace.target_pose[:, :3],
                axis=1,
            ),
            label=name,
        )
        axes[6].plot(time, trace.rho, label=name)

    axes[0].set_ylabel("J1 cmd dq [rad/s]")
    axes[1].set_ylabel("J1 weak [rad/s]")
    axes[2].set_ylabel("J1 nonweak [rad/s]")
    axes[3].set_ylabel("Elbow lateral delta [m]")
    axes[4].set_ylabel("Actual EEF error [m]")
    axes[5].set_ylabel("Governed target error [m]")
    axes[6].set_ylabel("rho")
    axes[6].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _payload(
    trace: ReplayTrace,
    signals: ModalSignals,
    controller: (
        DirectionalLimitController
        | HorizontalRetractLimitController
        | ModalLimitController
        | NoVelocityLimitController
        | WeakDirectionTargetGovernor
        | None
    ),
) -> dict[str, np.ndarray]:
    payload = dict(trace.__dict__)
    payload.update(
        {
            "j1_weak_velocity": signals.weak_j1_velocity,
            "j1_strong_velocity": signals.strong_j1_velocity,
            "j1_null_velocity": signals.null_j1_velocity,
            "j1_nonweak_velocity": signals.nonweak_j1_velocity,
            "weak_modal_velocity": signals.weak_modal_velocity,
            "weak_j1_participation": signals.weak_j1_participation,
            "singular_gap_ratio": signals.singular_gap_ratio,
            "weak_projector_change": signals.projector_change,
            "actual_rho": signals.actual_rho,
        }
    )
    if isinstance(controller, ModalLimitController):
        modal_limit = controller.limit
        assert modal_limit is not None
        payload["limit_rho"] = np.asarray(modal_limit.ratio)
        payload["limit_singular_gap_ratio"] = np.asarray(
            modal_limit.singular_gap_ratio
        )
        payload["limit_weak_j1_participation"] = np.asarray(
            modal_limit.weak_j1_participation
        )
        payload["limit_projector_change"] = np.asarray(
            modal_limit.projector_change
        )
        if controller.governor is not None:
            payload["governor_weak_scale"] = np.asarray(
                controller.governor.weak_scale
            )
    if isinstance(controller, HorizontalRetractLimitController):
        retract_limit = controller.limit
        assert retract_limit is not None
        payload["horizontal_retract_radius"] = np.asarray(
            retract_limit.radius
        )
        payload["horizontal_retract_radial_velocity"] = np.asarray(
            retract_limit.radial_velocity
        )
        payload["horizontal_retract_activation"] = np.asarray(
            retract_limit.activation
        )
        payload["horizontal_retract_j1_cap"] = np.asarray(
            retract_limit.cap
        )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--episode", type=int, default=202)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--segments", type=int, nargs="*", default=[12, 13])
    parser.add_argument(
        "--reverse-segments",
        type=int,
        nargs="*",
        default=[13],
    )
    parser.add_argument("--settle-duration", type=float, default=0.5)
    parser.add_argument("--base-j1-speed", type=float, default=2.0)
    parser.add_argument("--total-j1-speed", type=float, default=3.0)
    parser.add_argument("--nonweak-j1-speed", type=float, default=2.0)
    parser.add_argument("--pair-probe-nonweak-speed", type=float)
    parser.add_argument("--pair-probe-total-speed", type=float)
    parser.add_argument(
        "--horizontal-retract-speeds",
        type=float,
        nargs="*",
        default=[],
        help="High J1 caps gated by shrinking target radius about J1.",
    )
    parser.add_argument(
        "--retract-deadband",
        type=float,
        default=0.02,
        help="Target radial retract speed before J1 cap starts rising.",
    )
    parser.add_argument(
        "--retract-full-speed",
        type=float,
        default=0.15,
        help="Target radial retract speed for full high J1 cap.",
    )
    parser.add_argument(
        "--retract-cap-slew-rate",
        type=float,
        help="Optional maximum J1-cap change rate in rad/s^2.",
    )
    parser.add_argument(
        "--governor-margins",
        type=float,
        nargs="*",
        default=[1.0, 0.9],
    )
    parser.add_argument(
        "--circle-speeds",
        type=float,
        nargs="*",
        default=[],
        help="Run near-extended shoulder-sphere circles at these speeds.",
    )
    parser.add_argument("--circle-radius", type=float, default=0.06)
    parser.add_argument("--extension-speed", type=float, default=0.15)
    parser.add_argument("--circle-ramp-duration", type=float, default=0.3)
    parser.add_argument("--circle-seed-segment", type=int, default=13)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[],
        help="Optional variant-name filter.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "dev/results/modal_shoulder_limit_comparison_ep202"
        ),
    )
    return parser.parse_args()


def _interpolate_pose_path(
    start: np.ndarray,
    end: np.ndarray,
    count: int,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 7), dtype=np.float64)
    fraction = np.linspace(0.0, 1.0, count, endpoint=False)[:, None]
    positions = start[:3] + fraction * (end[:3] - start[:3])
    rotations = Rotation.from_quat(
        np.vstack([start[[4, 5, 6, 3]], end[[4, 5, 6, 3]]])
    )
    rotation_vectors = (
        rotations[0].inv() * rotations[1]
    ).as_rotvec()
    interpolated = rotations[0] * Rotation.from_rotvec(
        fraction * rotation_vectors
    )
    poses = np.empty((count, 7), dtype=np.float64)
    poses[:, :3] = positions
    poses[:, 3:] = interpolated.as_quat()[:, [3, 0, 1, 2]]
    return poses


def _extended_circle_inputs(
    speed: float,
    circle_radius: float,
    extension_speed: float,
    ramp_duration: float,
    segment: RetractSegment,
    target_pose: np.ndarray,
    source_q: np.ndarray,
    shoulder: np.ndarray,
    source_elbow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, slice]:
    """Extend from a bent pose, then trace a constant-reach spherical circle."""
    if speed <= 0.0:
        raise ValueError("Circle speed must be positive.")
    if circle_radius <= 0.0:
        raise ValueError("Circle radius must be positive.")
    if extension_speed <= 0.0:
        raise ValueError("Extension speed must be positive.")
    if ramp_duration <= 0.0:
        raise ValueError("Circle ramp duration must be positive.")

    bent_index = segment.core_end - 1
    extended_index = segment.core_start
    bent_pose = target_pose[bent_index].copy()
    extended_pose = target_pose[extended_index].copy()
    shoulder_position = shoulder[extended_index]
    radial = extended_pose[:3] - shoulder_position
    reach = float(np.linalg.norm(radial))
    radial /= reach
    if circle_radius >= reach:
        raise ValueError("Circle radius must be smaller than arm reach.")

    tangent_a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    tangent_a -= radial * float(tangent_a @ radial)
    if np.linalg.norm(tangent_a) < 1e-6:
        tangent_a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tangent_a -= radial * float(tangent_a @ radial)
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(radial, tangent_a)
    tangent_b /= np.linalg.norm(tangent_b)

    angular_radius = float(np.arcsin(circle_radius / reach))
    circle_center_scale = np.cos(angular_radius)
    circle_tangent_scale = np.sin(angular_radius)
    first_direction = (
        circle_center_scale * radial
        + circle_tangent_scale * tangent_a
    )
    first_pose = extended_pose.copy()
    first_pose[:3] = shoulder_position + reach * first_direction

    pre_ticks = int(round(0.2 / CONTROL_DT))
    extension_distance = float(
        np.linalg.norm(extended_pose[:3] - bent_pose[:3])
    )
    extension_ticks = max(
        1,
        int(round(extension_distance / extension_speed / CONTROL_DT)),
    )
    transition_ticks = max(
        1,
        int(round(circle_radius / extension_speed / CONTROL_DT)),
    )
    circumference = 2.0 * np.pi * circle_radius
    circle_ticks = max(
        4,
        int(round(circumference / speed / CONTROL_DT)),
    )
    post_ticks = int(round(0.4 / CONTROL_DT))

    pre = np.repeat(bent_pose[None, :], pre_ticks, axis=0)
    extension = _interpolate_pose_path(
        bent_pose,
        extended_pose,
        extension_ticks,
    )
    transition = _interpolate_pose_path(
        extended_pose,
        first_pose,
        transition_ticks,
    )
    ramp_ticks = max(2, int(round(ramp_duration / CONTROL_DT)))
    angular_speed = speed / circle_radius
    ramp_fraction = _smoothstep(
        (np.arange(ramp_ticks, dtype=np.float64) + 0.5) / ramp_ticks
    )
    ramp_up_step = angular_speed * ramp_fraction * CONTROL_DT
    ramp_up_phase = np.cumsum(
        np.hstack([0.0, ramp_up_step[:-1]])
    )
    circle_start_phase = float(np.sum(ramp_up_step))
    circle_phase = circle_start_phase + angular_speed * CONTROL_DT * np.arange(
        circle_ticks,
        dtype=np.float64,
    )
    ramp_down_start = circle_start_phase + angular_speed * (
        circle_ticks * CONTROL_DT
    )
    ramp_down_step = (
        angular_speed * ramp_fraction[::-1] * CONTROL_DT
    )
    ramp_down_phase = ramp_down_start + np.cumsum(
        np.hstack([0.0, ramp_down_step[:-1]])
    )

    def poses_at_phase(phase: np.ndarray) -> np.ndarray:
        poses = np.repeat(extended_pose[None, :], phase.size, axis=0)
        directions = (
            circle_center_scale * radial[None, :]
            + circle_tangent_scale
            * (
                np.cos(phase)[:, None] * tangent_a[None, :]
                + np.sin(phase)[:, None] * tangent_b[None, :]
            )
        )
        poses[:, :3] = shoulder_position[None, :] + reach * directions
        return poses

    ramp_up = poses_at_phase(ramp_up_phase)
    circle = poses_at_phase(circle_phase)
    ramp_down = poses_at_phase(ramp_down_phase)
    post = np.repeat(ramp_down[-1][None, :], post_ticks, axis=0)
    poses = np.vstack(
        [
            pre,
            extension,
            transition,
            ramp_up,
            circle,
            ramp_down,
            post,
        ]
    )

    initial_q = source_q[bent_index]
    initial_elbow = source_elbow[bent_index]
    q_stream = np.repeat(initial_q[None, :], poses.shape[0], axis=0)
    elbow_stream = np.repeat(initial_elbow[None, :], poses.shape[0], axis=0)
    circle_start = (
        pre_ticks + extension_ticks + transition_ticks + ramp_ticks
    )
    return (
        poses,
        q_stream,
        elbow_stream,
        slice(circle_start, circle_start + circle_ticks),
    )


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
    variants = [
        variant
        for variant in _variants(
            args.base_j1_speed,
            args.total_j1_speed,
            args.governor_margins,
            pair_probe_nonweak_speed=args.pair_probe_nonweak_speed,
            pair_probe_total_speed=args.pair_probe_total_speed,
            horizontal_retract_speeds=args.horizontal_retract_speeds,
            retract_deadband=args.retract_deadband,
            retract_full_speed=args.retract_full_speed,
            retract_cap_slew_rate=args.retract_cap_slew_rate,
        )
        if not args.variants or variant.name in set(args.variants)
    ]
    if not variants:
        raise ValueError("No modal shoulder variants were selected.")

    rows: list[dict[str, float | int | str]] = []
    cases: list[
        tuple[
            int | str,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            slice,
        ]
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
        seed_segment = selected[args.circle_seed_segment]
        for speed in args.circle_speeds:
            (
                circle_target,
                circle_q,
                circle_elbow,
                circle_core,
            ) = _extended_circle_inputs(
                speed,
                args.circle_radius,
                args.extension_speed,
                args.circle_ramp_duration,
                seed_segment,
                target_pose,
                source_q,
                shoulder,
                source_elbow,
            )
            speed_label = f"{speed:g}".replace(".", "p")
            cases.append(
                (
                    f"circle_{speed_label}mps",
                    circle_target,
                    circle_q,
                    circle_elbow,
                    circle_core,
                )
            )

    for case_name, case_target, case_q, case_elbow, core in cases:
        traces: dict[str, ReplayTrace] = {}
        modal_signals: dict[str, ModalSignals] = {}
        for variant in variants:
            print(f"Simulating case={case_name}, variant={variant.name}...")
            trace, controller = _simulate_variant(
                args.side,
                variant,
                case_target,
                case_q,
                case_elbow,
                settle_duration=args.settle_duration,
                base_speed=args.base_j1_speed,
                nonweak_speed=args.nonweak_j1_speed,
            )
            signals = _modal_signals(
                args.side,
                trace,
                variant.modal_projector,
            )
            rows.append(
                _summary_row(
                    case_name,
                    variant,
                    trace,
                    signals,
                    controller,
                    core,
                    nonweak_speed=args.nonweak_j1_speed,
                )
            )
            traces[variant.name] = trace
            modal_signals[variant.name] = signals
            _save_trace(
                args.output_dir
                / f"trace_{case_name}_{variant.name}.npz",
                trace,
            )
            np.savez_compressed(
                args.output_dir
                / f"modal_{case_name}_{variant.name}.npz",
                **_payload(trace, signals, controller),
            )
        _plot(
            args.output_dir / f"{case_name}.png",
            traces,
            modal_signals,
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
