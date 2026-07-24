#!/usr/bin/env python3
"""Test a geometric weak-direction target governor on recorded IK paths."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import mink
import numpy as np
import qpsolvers
from scipy.spatial.transform import Rotation

from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S
from openarm_control.poses import pose_to_se3, se3_to_pose
from openarm_control.singularity import normalized_arm_jacobian

from sim_intervention_posture_replay import (
    CHARACTERISTIC_LENGTH,
    CONTROL_DT,
    DEFAULT_RUN_DIR,
    Kinematics,
    ReplayTrace,
    _load_recorded_commands,
    _save_trace,
    _source_geometry,
    compute_metrics,
    detect_retract_segments,
    simulate_replay,
)


class WeakDirectionTargetGovernor:
    """Filter raw target increments along the weakest geometric mode."""

    name = "weak_first_hybrid"

    def __init__(
        self,
        *,
        velocity_margin: float,
        characteristic_length: float = CHARACTERISTIC_LENGTH,
        dt: float = CONTROL_DT,
        catchup_trigger_rate: float = 0.05,
    ) -> None:
        if not 0.0 < velocity_margin <= 1.0:
            raise ValueError("Velocity margin must be in the range (0, 1].")
        self.velocity_margin = float(velocity_margin)
        self.characteristic_length = float(characteristic_length)
        self.dt = float(dt)
        self.catchup_trigger_rate = float(catchup_trigger_rate)
        self.caps = np.asarray(
            ARM_JOINT_VELOCITY_LIMITS_RAD_S,
            dtype=np.float64,
        )
        self.requested_weak_speed: list[float] = []
        self.allowed_weak_speed: list[float] = []
        self.weak_scale: list[float] = []
        self.predicted_peak_utilization_before: list[float] = []
        self.predicted_peak_utilization_after: list[float] = []
        self.sigma_min: list[float] = []
        self.infeasible: list[bool] = []
        self._previous_desired: mink.SE3 | None = None
        self._limited_target: mink.SE3 | None = None

    def __call__(
        self,
        kinematics: Kinematics,
        side: str,
        desired_pose: np.ndarray,
    ) -> np.ndarray:
        solver = kinematics._ik
        assert solver is not None
        configuration = solver._config
        frame_task = solver._tasks[side]
        dof_indices = solver._arm_dofs_by_side[side]

        desired = pose_to_se3(desired_pose)
        if self._previous_desired is None or self._limited_target is None:
            self._previous_desired = desired
            self._limited_target = desired
            self.requested_weak_speed.append(0.0)
            self.allowed_weak_speed.append(0.0)
            self.weak_scale.append(1.0)
            self.predicted_peak_utilization_before.append(0.0)
            self.predicted_peak_utilization_after.append(0.0)
            self.sigma_min.append(float("nan"))
            self.infeasible.append(False)
            return np.asarray(desired_pose, dtype=np.float64)

        target_increment = desired.minus(self._previous_desired)
        normalized_error = target_increment.copy()
        normalized_error[:3] /= self.characteristic_length
        raw_target_rate = float(np.linalg.norm(normalized_error) / self.dt)

        jacobian = normalized_arm_jacobian(
            frame_task,
            configuration,
            dof_indices,
            self.characteristic_length,
        )
        u, singular_values, vt = np.linalg.svd(
            jacobian,
            full_matrices=True,
        )
        task_coefficients = u.T @ normalized_error
        modal_displacements = task_coefficients / singular_values
        weak_index = singular_values.size - 1
        weak_direction = vt[weak_index]
        weak_displacement = float(modal_displacements[weak_index])
        nonweak_displacement = (
            vt[:weak_index].T @ modal_displacements[:weak_index]
        )
        requested_displacement = (
            nonweak_displacement + weak_displacement * weak_direction
        )

        displacement_limits = (
            self.caps * self.dt * self.velocity_margin
        )
        weak_delta = weak_displacement * weak_direction
        gamma_low = 0.0
        gamma_high = 1.0
        feasible = True
        epsilon = 1e-12
        for joint in range(7):
            slope = weak_delta[joint]
            base = nonweak_displacement[joint]
            limit = displacement_limits[joint]
            if abs(slope) <= epsilon:
                if abs(base) > limit:
                    feasible = False
                continue
            bound_a = (-limit - base) / slope
            bound_b = (limit - base) / slope
            gamma_low = max(gamma_low, min(bound_a, bound_b))
            gamma_high = min(gamma_high, max(bound_a, bound_b))

        feasible = feasible and gamma_low <= gamma_high
        if feasible:
            gamma = float(np.clip(gamma_high, 0.0, 1.0))
            guarded_modal_displacements = modal_displacements.copy()
            guarded_modal_displacements[weak_index] *= gamma
        else:
            utilization = np.abs(requested_displacement) / displacement_limits
            peak_utilization = float(np.max(utilization))
            gamma = min(1.0, 1.0 / peak_utilization)
            guarded_modal_displacements = modal_displacements * gamma
        guarded_coefficients = (
            guarded_modal_displacements * singular_values
        )
        guarded_normalized_error = u @ guarded_coefficients
        guarded_error = guarded_normalized_error.copy()
        guarded_error[:3] *= self.characteristic_length
        guarded_target = self._limited_target.plus(guarded_error)

        guarded_displacement = (
            vt[: singular_values.size].T @ guarded_modal_displacements
        )
        backlog = desired.minus(guarded_target)
        normalized_backlog = backlog.copy()
        normalized_backlog[:3] /= self.characteristic_length
        backlog_coefficients = u.T @ normalized_backlog
        catchup_displacement = float(
            backlog_coefficients[weak_index] / singular_values[weak_index]
        )
        catchup_delta = catchup_displacement * weak_direction
        catchup_low = 0.0
        catchup_high = 1.0
        catchup_feasible = True
        for joint in range(7):
            slope = catchup_delta[joint]
            base = guarded_displacement[joint]
            limit = displacement_limits[joint]
            if abs(slope) <= epsilon:
                if abs(base) > limit:
                    catchup_feasible = False
                continue
            bound_a = (-limit - base) / slope
            bound_b = (limit - base) / slope
            catchup_low = max(catchup_low, min(bound_a, bound_b))
            catchup_high = min(catchup_high, max(bound_a, bound_b))
        catchup_feasible = (
            catchup_feasible and catchup_low <= catchup_high
        )
        catchup_scale = (
            float(np.clip(catchup_high, 0.0, 1.0))
            if catchup_feasible
            else 0.0
        )
        applied_catchup = catchup_scale * catchup_displacement
        if abs(applied_catchup) > epsilon:
            normalized_catchup = (
                u[:, weak_index]
                * singular_values[weak_index]
                * applied_catchup
            )
            catchup_error = normalized_catchup.copy()
            catchup_error[:3] *= self.characteristic_length
            guarded_target = guarded_target.plus(catchup_error)
            guarded_displacement += (
                applied_catchup * weak_direction
            )

        if raw_target_rate <= self.catchup_trigger_rate:
            full_backlog = desired.minus(guarded_target)
            normalized_full_backlog = full_backlog.copy()
            normalized_full_backlog[:3] /= self.characteristic_length
            full_backlog_modes = (
                u.T @ normalized_full_backlog
            ) / singular_values
            full_backlog_displacement = (
                vt[: singular_values.size].T @ full_backlog_modes
            )
            recovery_low = 0.0
            recovery_high = 1.0
            recovery_feasible = True
            for joint in range(7):
                slope = full_backlog_displacement[joint]
                base = guarded_displacement[joint]
                limit = displacement_limits[joint]
                if abs(slope) <= epsilon:
                    if abs(base) > limit:
                        recovery_feasible = False
                    continue
                bound_a = (-limit - base) / slope
                bound_b = (limit - base) / slope
                recovery_low = max(recovery_low, min(bound_a, bound_b))
                recovery_high = min(recovery_high, max(bound_a, bound_b))
            recovery_feasible = (
                recovery_feasible and recovery_low <= recovery_high
            )
            recovery_scale = (
                float(np.clip(recovery_high, 0.0, 1.0))
                if recovery_feasible
                else 0.0
            )
            if recovery_scale > epsilon:
                recovery_error = recovery_scale * full_backlog
                guarded_target = guarded_target.plus(recovery_error)
                guarded_displacement += (
                    recovery_scale * full_backlog_displacement
                )

        outer_limits = self.caps * self.dt
        self.requested_weak_speed.append(weak_displacement / self.dt)
        self.allowed_weak_speed.append(
            (
                guarded_modal_displacements[weak_index]
                + applied_catchup
            )
            / self.dt
        )
        self.weak_scale.append(gamma)
        self.predicted_peak_utilization_before.append(
            float(np.max(np.abs(requested_displacement) / outer_limits))
        )
        self.predicted_peak_utilization_after.append(
            float(np.max(np.abs(guarded_displacement) / outer_limits))
        )
        self.sigma_min.append(float(singular_values[-1]))
        self.infeasible.append(not feasible)
        self._previous_desired = desired
        self._limited_target = guarded_target
        return se3_to_pose(guarded_target).astype(np.float64)


class FeasibleTwistTargetGovernor:
    """Project current-to-desired error into the feasible row-space motion."""

    name = "feasible_projection"

    def __init__(
        self,
        *,
        velocity_margin: float,
        nullspace_speed_allowance: float,
        characteristic_length: float = CHARACTERISTIC_LENGTH,
        dt: float = CONTROL_DT,
    ) -> None:
        if not 0.0 < velocity_margin <= 1.0:
            raise ValueError("Velocity margin must be in the range (0, 1].")
        self.velocity_margin = float(velocity_margin)
        if nullspace_speed_allowance < 0.0:
            raise ValueError("Nullspace speed allowance must be non-negative.")
        self.nullspace_speed_allowance = float(
            nullspace_speed_allowance
        )
        self.name = (
            "feasible_projection_z"
            f"{self.nullspace_speed_allowance:g}"
        )
        self.characteristic_length = float(characteristic_length)
        self.dt = float(dt)
        self.caps = np.asarray(
            ARM_JOINT_VELOCITY_LIMITS_RAD_S,
            dtype=np.float64,
        )
        self.requested_weak_speed: list[float] = []
        self.allowed_weak_speed: list[float] = []
        self.weak_scale: list[float] = []
        self.predicted_peak_utilization_before: list[float] = []
        self.predicted_peak_utilization_after: list[float] = []
        self.sigma_min: list[float] = []
        self.infeasible: list[bool] = []

    def __call__(
        self,
        kinematics: Kinematics,
        side: str,
        desired_pose: np.ndarray,
    ) -> np.ndarray:
        solver = kinematics._ik
        assert solver is not None
        configuration = solver._config
        frame_task = solver._tasks[side]
        dof_indices = solver._arm_dofs_by_side[side]

        current_parameters = (
            configuration._get_transform_frame_to_world_wxyz_xyz(
                frame_task.frame_name,
                frame_task.frame_type,
            )
        )
        current = mink.SE3(wxyz_xyz=current_parameters)
        desired = pose_to_se3(desired_pose)
        error = desired.minus(current)
        normalized_error = error.copy()
        normalized_error[:3] /= self.characteristic_length

        jacobian = normalized_arm_jacobian(
            frame_task,
            configuration,
            dof_indices,
            self.characteristic_length,
        )
        u, singular_values, vt = np.linalg.svd(
            jacobian,
            full_matrices=True,
        )
        task_coefficients = u.T @ normalized_error
        requested_modes = task_coefficients / singular_values
        modal_directions = vt.T
        rowspace = modal_directions[:, : singular_values.size]
        displacement_limits = (
            self.caps * self.dt * self.velocity_margin
        )
        requested_with_nullspace = np.append(requested_modes, 0.0)
        qp_hessian = np.diag(
            np.append(singular_values**2, 1e-6)
        )
        qp_gradient = -qp_hessian @ requested_with_nullspace
        nullspace_row = np.zeros((1, 7), dtype=np.float64)
        nullspace_row[0, -1] = 1.0
        inequalities = np.vstack(
            [
                modal_directions,
                -modal_directions,
                nullspace_row,
                -nullspace_row,
            ]
        )
        nullspace_displacement = (
            self.nullspace_speed_allowance * self.dt
        )
        inequality_bounds = np.concatenate(
            [
                displacement_limits,
                displacement_limits,
                [nullspace_displacement, nullspace_displacement],
            ]
        )
        projected_with_nullspace = qpsolvers.solve_qp(
            qp_hessian,
            qp_gradient,
            inequalities,
            inequality_bounds,
            solver="daqp",
        )
        failed = projected_with_nullspace is None
        if failed:
            projected_with_nullspace = np.zeros(7, dtype=np.float64)
        projected_modes = projected_with_nullspace[:6]

        projected_coefficients = singular_values * projected_modes
        projected_normalized_error = u @ projected_coefficients
        projected_error = projected_normalized_error.copy()
        projected_error[:3] *= self.characteristic_length
        governed_target = current.plus(projected_error)

        requested_displacement = rowspace @ requested_modes
        projected_displacement = (
            modal_directions @ projected_with_nullspace
        )
        weak_requested = float(requested_modes[-1])
        weak_projected = float(projected_modes[-1])
        weak_scale = (
            weak_projected / weak_requested
            if abs(weak_requested) > 1e-12
            else 1.0
        )
        outer_limits = self.caps * self.dt
        self.requested_weak_speed.append(weak_requested / self.dt)
        self.allowed_weak_speed.append(weak_projected / self.dt)
        self.weak_scale.append(weak_scale)
        self.predicted_peak_utilization_before.append(
            float(np.max(np.abs(requested_displacement) / outer_limits))
        )
        self.predicted_peak_utilization_after.append(
            float(np.max(np.abs(projected_displacement) / outer_limits))
        )
        self.sigma_min.append(float(singular_values[-1]))
        self.infeasible.append(failed)
        return se3_to_pose(governed_target).astype(np.float64)


def _orientation_rmse(target: np.ndarray, actual: np.ndarray) -> float:
    target_rotation = Rotation.from_quat(target[:, [4, 5, 6, 3]])
    actual_rotation = Rotation.from_quat(actual[:, [4, 5, 6, 3]])
    error = (target_rotation.inv() * actual_rotation).magnitude()
    return float(np.sqrt(np.mean(error**2)))


def _governor_row(
    segment_index: int,
    margin: float | None,
    segment_start: int,
    core_start: int,
    core_end: int,
    trace: ReplayTrace,
    governor: (
        WeakDirectionTargetGovernor | FeasibleTwistTargetGovernor | None
    ),
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "variant": "baseline" if governor is None else governor.name,
        "velocity_margin": 0.0 if margin is None else margin,
        "nullspace_speed_allowance": (
            governor.nullspace_speed_allowance
            if isinstance(governor, FeasibleTwistTargetGovernor)
            else 0.0
        ),
        **asdict(compute_metrics(segment_index, 0.0, trace)),
    }
    relative_start = core_start - segment_start
    relative_end = core_end - segment_start - 1
    core = slice(relative_start, relative_end + 1)
    row["core_elbow_delta_m"] = float(
        trace.command_elbow[relative_end, 1]
        - trace.command_elbow[relative_start, 1]
    )
    row["core_actual_elbow_delta_m"] = float(
        trace.actual_elbow[relative_end, 1]
        - trace.actual_elbow[relative_start, 1]
    )
    row["core_exact_z_delta_rad"] = float(
        trace.exact_null_error[relative_end]
        - trace.exact_null_error[relative_start]
    )
    for joint in range(7):
        row[f"j{joint + 1}_core_limit_fraction"] = float(
            np.mean(trace.velocity_utilization[core, joint] >= 0.98)
        )

    target_position_error = np.linalg.norm(
        trace.ik_target_pose[:, :3] - trace.target_pose[:, :3],
        axis=1,
    )
    row["governed_target_position_rmse_m"] = float(
        np.sqrt(np.mean(target_position_error**2))
    )
    row["governed_target_orientation_rmse_rad"] = _orientation_rmse(
        trace.target_pose,
        trace.ik_target_pose,
    )
    if governor is None:
        row["governor_active_fraction"] = 0.0
        row["governor_infeasible_fraction"] = 0.0
        row["mean_weak_scale"] = 1.0
        row["peak_requested_weak_speed_rad_s"] = 0.0
        row["peak_allowed_weak_speed_rad_s"] = 0.0
        row["peak_predicted_utilization_before"] = 0.0
        row["peak_predicted_utilization_after"] = 0.0
    else:
        weak_scale = np.asarray(governor.weak_scale)
        row["governor_active_fraction"] = float(
            np.mean(weak_scale < 1.0 - 1e-9)
        )
        row["governor_infeasible_fraction"] = float(
            np.mean(governor.infeasible)
        )
        row["mean_weak_scale"] = float(np.mean(weak_scale))
        row["peak_requested_weak_speed_rad_s"] = float(
            np.max(np.abs(governor.requested_weak_speed))
        )
        row["peak_allowed_weak_speed_rad_s"] = float(
            np.max(np.abs(governor.allowed_weak_speed))
        )
        row["peak_predicted_utilization_before"] = float(
            np.max(governor.predicted_peak_utilization_before)
        )
        row["peak_predicted_utilization_after"] = float(
            np.max(governor.predicted_peak_utilization_after)
        )
    return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--episode", type=int, default=202)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--segments", type=int, nargs="+", default=[12, 13])
    parser.add_argument(
        "--velocity-margins",
        type=float,
        nargs="+",
        default=[1.0, 0.9, 0.8, 0.7, 0.5],
    )
    parser.add_argument(
        "--nullspace-speed-allowances",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.5, 0.8],
    )
    parser.add_argument("--settle-duration", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "dev/results/intervention_weak_direction_governor_ep202"
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

    rows: list[dict[str, float | int | str]] = []
    for segment_index in args.segments:
        segment = segments[segment_index]
        window = slice(segment.start, segment.end)
        variants: list[
            tuple[
                float | None,
                WeakDirectionTargetGovernor
                | FeasibleTwistTargetGovernor
                | None,
            ]
        ]
        variants = [(None, None)]
        variants.extend(
            (
                margin,
                WeakDirectionTargetGovernor(velocity_margin=margin),
            )
            for margin in args.velocity_margins
        )
        variants.extend(
            (
                margin,
                FeasibleTwistTargetGovernor(
                    velocity_margin=margin,
                    nullspace_speed_allowance=nullspace_speed,
                ),
            )
            for margin in args.velocity_margins
            for nullspace_speed in args.nullspace_speed_allowances
        )
        for margin, governor in variants:
            label = (
                "baseline"
                if governor is None
                else f"{governor.name}_margin_{margin:g}"
            )
            print(f"Simulating segment={segment_index}, {label}...")
            trace = simulate_replay(
                args.side,
                0.0,
                target_pose[window],
                source_q[window],
                source_elbow[window],
                settle_duration=args.settle_duration,
                target_governor=governor,
            )
            rows.append(
                _governor_row(
                    segment_index,
                    margin,
                    segment.start,
                    segment.core_start,
                    segment.core_end,
                    trace,
                    governor,
                )
            )
            _save_trace(
                args.output_dir
                / f"trace_segment_{segment_index:02d}_{label}.npz",
                trace,
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
