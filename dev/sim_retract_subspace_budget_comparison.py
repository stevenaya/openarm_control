#!/usr/bin/env python3
"""Compare retract-gated total and near-null joint-speed budgets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import mink
import numpy as np

from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S
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
    detect_retract_segments,
    simulate_replay,
)
from sim_modal_shoulder_limit_comparison import (
    NoVelocityLimitController,
    _extended_circle_inputs,
    _orientation_error,
    _smoothstep,
)


@dataclass(frozen=True)
class Variant:
    """One retract-gated velocity-budget strategy."""

    name: str
    mode: str
    j1_high: float
    j4_high: float
    cap_slew_rate: float
    gate_metric: str = "perpendicular"


class RetractSubspaceLimit(mink.Limit):
    """Gate larger J1/J4 caps and optionally reserve them for weak modes."""

    _selected_local = np.array([0, 3], dtype=int)

    def __init__(
        self,
        kinematics: Kinematics,
        side: str,
        variant: Variant,
        *,
        deadband: float,
        full_speed: float,
    ) -> None:
        solver = kinematics._ik
        assert solver is not None
        if variant.mode not in {"total", "vnear", "near2d"}:
            raise ValueError(f"Unsupported gated mode: {variant.mode}")
        if variant.gate_metric not in {
            "perpendicular",
            "reach",
            "combined",
        }:
            raise ValueError(
                f"Unsupported gate metric: {variant.gate_metric}"
            )
        if not 0.0 <= deadband < full_speed:
            raise ValueError("Expected 0 <= deadband < full speed.")
        if variant.cap_slew_rate <= 0.0:
            raise ValueError("Cap slew rate must be positive.")

        self._model = solver._model
        self._frame_task = solver._tasks[side]
        self._arm_dofs = np.asarray(
            solver._arm_dofs_by_side[side],
            dtype=int,
        )
        self._shoulder_joint = self._model.joint(
            f"openarm_{side}_joint1"
        ).id
        self._mode = variant.mode
        self._gate_metric = variant.gate_metric
        self._deadband = float(deadband)
        self._full_speed = float(full_speed)
        self._cap_slew_rate = float(variant.cap_slew_rate)
        all_base_caps = np.asarray(
            ARM_JOINT_VELOCITY_LIMITS_RAD_S,
            dtype=np.float64,
        )
        self._base_caps = all_base_caps[self._selected_local]
        self._high_caps = np.array(
            [variant.j1_high, variant.j4_high],
            dtype=np.float64,
        )
        if np.any(self._high_caps < self._base_caps):
            raise ValueError("High caps cannot be below base caps.")
        self._caps = self._base_caps.copy()
        self._previous_perpendicular_radius: float | None = None
        self._previous_reach: float | None = None
        self._G: np.ndarray | None = None
        self._h_speed: np.ndarray | None = None

        self.radius: list[float] = []
        self.radial_velocity: list[float] = []
        self.activation: list[float] = []
        self.caps: list[np.ndarray] = []
        self.rho: list[float] = []
        self.vnear_j1: list[float] = []
        self.vnear_j4: list[float] = []
        self.z_j1: list[float] = []
        self.z_j4: list[float] = []

    def begin_outer_step(
        self,
        configuration: mink.Configuration,
        desired_pose: np.ndarray,
    ) -> None:
        """Freeze geometric gate, caps, and SVD projector for one tick."""
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
        perpendicular_radius = float(np.linalg.norm(perpendicular))
        reach = float(np.linalg.norm(displacement))
        perpendicular_velocity = (
            0.0
            if self._previous_perpendicular_radius is None
            else (
                perpendicular_radius - self._previous_perpendicular_radius
            )
            / CONTROL_DT
        )
        reach_velocity = (
            0.0
            if self._previous_reach is None
            else (reach - self._previous_reach) / CONTROL_DT
        )
        perpendicular_retract_speed = max(-perpendicular_velocity, 0.0)
        reach_retract_speed = max(-reach_velocity, 0.0)
        if self._gate_metric == "perpendicular":
            retract_speed = perpendicular_retract_speed
        elif self._gate_metric == "reach":
            retract_speed = reach_retract_speed
        else:
            retract_speed = min(
                perpendicular_retract_speed,
                reach_retract_speed,
            )
        normalized = (
            (retract_speed - self._deadband)
            / (self._full_speed - self._deadband)
        )
        activation = float(_smoothstep(np.asarray(normalized)))
        desired_caps = self._base_caps + activation * (
            self._high_caps - self._base_caps
        )
        cap_step = self._cap_slew_rate * CONTROL_DT
        self._caps += np.clip(
            desired_caps - self._caps,
            -cap_step,
            cap_step,
        )
        self._previous_perpendicular_radius = perpendicular_radius
        self._previous_reach = reach

        jacobian = normalized_arm_jacobian(
            self._frame_task,
            configuration,
            self._arm_dofs,
            CHARACTERISTIC_LENGTH,
        )
        _, singular_values, vt = np.linalg.svd(
            jacobian,
            full_matrices=True,
        )
        vnear = vt[-2]
        exact_null = vt[-1]
        projector = np.zeros((7, 7), dtype=np.float64)
        if self._mode in {"vnear", "near2d"}:
            projector += np.outer(vnear, vnear)
        if self._mode == "near2d":
            projector += np.outer(exact_null, exact_null)
        self._prepare_inequalities(projector)

        self.radius.append(perpendicular_radius)
        self.radial_velocity.append(-retract_speed)
        self.activation.append(activation)
        self.caps.append(self._caps.copy())
        self.rho.append(float(singular_values[-1] / singular_values[0]))
        self.vnear_j1.append(float(vnear[0]))
        self.vnear_j4.append(float(vnear[3]))
        self.z_j1.append(float(exact_null[0]))
        self.z_j4.append(float(exact_null[3]))

    def _prepare_inequalities(self, projector: np.ndarray) -> None:
        rows: list[np.ndarray] = []
        speed_bounds: list[float] = []
        identity = np.eye(7, dtype=np.float64)
        nonweak = identity - projector
        for selected_index, local_joint in enumerate(self._selected_local):
            total_row = np.zeros(self._model.nv, dtype=np.float64)
            total_row[self._arm_dofs[local_joint]] = 1.0
            rows.extend([total_row, -total_row])
            speed_bounds.extend(
                [self._caps[selected_index], self._caps[selected_index]]
            )
            if self._mode != "total":
                nonweak_row = np.zeros(self._model.nv, dtype=np.float64)
                nonweak_row[self._arm_dofs] = nonweak[local_joint]
                rows.extend([nonweak_row, -nonweak_row])
                speed_bounds.extend(
                    [
                        self._base_caps[selected_index],
                        self._base_caps[selected_index],
                    ]
                )
        self._G = np.asarray(rows, dtype=np.float64)
        self._h_speed = np.asarray(speed_bounds, dtype=np.float64)

    def compute_qp_inequalities(
        self,
        configuration: mink.Configuration,
        dt: float,
    ) -> mink.Constraint:
        del configuration
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self._G is None or self._h_speed is None:
            raise RuntimeError("begin_outer_step() must run before solve.")
        return mink.Constraint(G=self._G, h=self._h_speed * dt)


class RetractSubspaceController:
    """Install and update one retract-gated subspace limit."""

    def __init__(
        self,
        side: str,
        variant: Variant,
        *,
        deadband: float,
        full_speed: float,
    ) -> None:
        self._side = side
        self._variant = variant
        self._deadband = float(deadband)
        self._full_speed = float(full_speed)
        self.limit: RetractSubspaceLimit | None = None

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
            self.limit = RetractSubspaceLimit(
                kinematics,
                side,
                self._variant,
                deadband=self._deadband,
                full_speed=self._full_speed,
            )
            solver._limits.append(self.limit)
        self.limit.begin_outer_step(solver._config, desired_pose)
        return np.asarray(desired_pose, dtype=np.float64)


def _number_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _variants() -> list[Variant]:
    variants = [
        Variant("A_baseline", "baseline", 2.0, 3.8, 5.0),
        Variant("N_no_joint_velocity_limits", "no_speed", 100.0, 100.0, 5.0),
    ]
    for j1_high in (4.0, 5.0, 6.0):
        variants.append(
            Variant(
                f"T_total_j1_{_number_label(j1_high)}_j4_4p4_s10",
                "total",
                j1_high,
                4.4,
                10.0,
            )
        )
    variants.extend(
        [
            Variant("T_total_j1_5_j4_4p4_s5", "total", 5.0, 4.4, 5.0),
            Variant("T_total_j1_5_j4_4p4_s7p5", "total", 5.0, 4.4, 7.5),
            Variant("T_total_j1_5_j4_4p4_s15", "total", 5.0, 4.4, 15.0),
            Variant("T_total_j1_5p5_j4_4p4_s10", "total", 5.5, 4.4, 10.0),
            Variant(
                "R_reach_j1_5p5_j4_4p4_s10",
                "total",
                5.5,
                4.4,
                10.0,
                "reach",
            ),
            Variant(
                "C_combined_j1_5p5_j4_4p4_s10",
                "total",
                5.5,
                4.4,
                10.0,
                "combined",
            ),
            Variant("T_total_j1_6_j4_4p4_s5", "total", 6.0, 4.4, 5.0),
            Variant("T_total_j1_6_j4_4p4_s20", "total", 6.0, 4.4, 20.0),
            Variant("W_vnear_j1_6_j4_3p8_s10", "vnear", 6.0, 3.8, 10.0),
            Variant("W_vnear_j1_6_j4_4p4_s10", "vnear", 6.0, 4.4, 10.0),
            Variant("D_near2d_j1_6_j4_4p4_s10", "near2d", 6.0, 4.4, 10.0),
        ]
    )
    return variants


def _simulate(
    side: str,
    variant: Variant,
    target_pose: np.ndarray,
    source_q: np.ndarray,
    source_elbow: np.ndarray,
    *,
    settle_duration: float,
    deadband: float,
    full_speed: float,
) -> tuple[
    ReplayTrace,
    RetractSubspaceController | NoVelocityLimitController | None,
]:
    base_caps = np.asarray(
        ARM_JOINT_VELOCITY_LIMITS_RAD_S,
        dtype=np.float64,
    )
    multipliers = np.ones(7, dtype=np.float64)
    controller: (
        RetractSubspaceController | NoVelocityLimitController | None
    )
    if variant.mode == "baseline":
        controller = None
    elif variant.mode == "no_speed":
        multipliers[:] = 100.0
        controller = NoVelocityLimitController(
            keep_joint_braking=False,
            keep_singularity_limit=True,
        )
    else:
        multipliers[0] = variant.j1_high / base_caps[0]
        multipliers[3] = variant.j4_high / base_caps[3]
        controller = RetractSubspaceController(
            side,
            variant,
            deadband=deadband,
            full_speed=full_speed,
        )
    trace = simulate_replay(
        side,
        0.0,
        target_pose,
        source_q,
        source_elbow,
        settle_duration=settle_duration,
        velocity_limit_joint_multipliers=multipliers,
        target_governor=controller,
    )
    return trace, controller


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _summary(
    case_name: int | str,
    variant: Variant,
    trace: ReplayTrace,
    controller: (
        RetractSubspaceController | NoVelocityLimitController | None
    ),
    reference: ReplayTrace,
    core: slice,
) -> dict[str, float | int | str]:
    position_error = np.linalg.norm(
        trace.actual_pose[:, :3] - trace.target_pose[:, :3],
        axis=1,
    )
    orientation_error = _orientation_error(
        trace.target_pose,
        trace.actual_pose,
    )
    command_acceleration = np.gradient(
        trace.command_dq,
        CONTROL_DT,
        axis=0,
    )
    actual_acceleration = np.gradient(
        trace.actual_dq,
        CONTROL_DT,
        axis=0,
    )
    elbow_difference = trace.actual_elbow - reference.actual_elbow
    q_difference = trace.actual_q - reference.actual_q
    active_limit = (
        controller.limit
        if isinstance(controller, RetractSubspaceController)
        else None
    )
    if active_limit is None:
        activation = np.zeros(trace.time.shape, dtype=np.float64)
        caps = np.tile(
            np.array([variant.j1_high, variant.j4_high], dtype=np.float64),
            (trace.time.size, 1),
        )
    else:
        activation = np.asarray(active_limit.activation)
        caps = np.asarray(active_limit.caps)

    row: dict[str, float | int | str] = {
        "case": case_name,
        "variant": variant.name,
        "mode": variant.mode,
        "gate_metric": variant.gate_metric,
        "j1_high_rad_s": variant.j1_high,
        "j4_high_rad_s": variant.j4_high,
        "cap_slew_rad_s2": variant.cap_slew_rate,
        "gate_active_fraction": float(
            np.mean(activation[core] > 1e-9)
        ),
        "gate_mean_activation": float(np.mean(activation[core])),
        "j1_mean_cap_rad_s": float(np.mean(caps[core, 0])),
        "j4_mean_cap_rad_s": float(np.mean(caps[core, 1])),
        "elbow_y_range_m": float(np.ptp(trace.actual_elbow[core, 1])),
        "elbow_y_rmse_to_no_speed_m": _rms(elbow_difference[core, 1]),
        "elbow_xyz_rmse_to_no_speed_m": _rms(
            np.linalg.norm(elbow_difference[core], axis=1)
        ),
        "q_rmse_to_no_speed_rad": _rms(q_difference[core]),
        "position_rmse_m": _rms(position_error[core]),
        "position_peak_m": float(np.max(position_error[core])),
        "orientation_rmse_rad": _rms(orientation_error[core]),
        "j1_command_accel_p99_rad_s2": float(
            np.percentile(np.abs(command_acceleration[core, 0]), 99.0)
        ),
        "j4_command_accel_p99_rad_s2": float(
            np.percentile(np.abs(command_acceleration[core, 3]), 99.0)
        ),
        "j1_actual_accel_p99_rad_s2": float(
            np.percentile(np.abs(actual_acceleration[core, 0]), 99.0)
        ),
        "j4_actual_accel_p99_rad_s2": float(
            np.percentile(np.abs(actual_acceleration[core, 3]), 99.0)
        ),
        "solve_failures": int(np.count_nonzero(trace.solve_failed)),
    }
    for joint in range(7):
        row[f"j{joint + 1}_command_peak_rad_s"] = float(
            np.max(np.abs(trace.command_dq[core, joint]))
        )
        row[f"j{joint + 1}_actual_peak_rad_s"] = float(
            np.max(np.abs(trace.actual_dq[core, joint]))
        )
    return row


def _plot(
    path: Path,
    traces: dict[str, ReplayTrace],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    for name, trace in traces.items():
        axes[0].plot(trace.time, trace.command_dq[:, 0], label=name)
        axes[1].plot(trace.time, trace.command_dq[:, 3], label=name)
        axes[2].plot(
            trace.time,
            trace.actual_elbow[:, 1] - trace.actual_elbow[0, 1],
            label=name,
        )
        axes[3].plot(
            trace.time,
            np.linalg.norm(
                trace.actual_pose[:, :3] - trace.target_pose[:, :3],
                axis=1,
            ),
            label=name,
        )
    axes[0].set_ylabel("J1 command [rad/s]")
    axes[1].set_ylabel("J4 command [rad/s]")
    axes[2].set_ylabel("Elbow lateral delta [m]")
    axes[3].set_ylabel("EEF position error [m]")
    axes[3].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--episode", type=int, default=202)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--segments", type=int, nargs="*", default=[13])
    parser.add_argument("--reverse-segments", type=int, nargs="*", default=[])
    parser.add_argument("--circle-speeds", type=float, nargs="*", default=[])
    parser.add_argument("--circle-radius", type=float, default=0.06)
    parser.add_argument("--extension-speed", type=float, default=0.15)
    parser.add_argument("--circle-ramp-duration", type=float, default=0.3)
    parser.add_argument("--circle-seed-segment", type=int, default=13)
    parser.add_argument("--settle-duration", type=float, default=0.5)
    parser.add_argument("--retract-deadband", type=float, default=0.02)
    parser.add_argument("--retract-full-speed", type=float, default=0.15)
    parser.add_argument("--variants", nargs="*", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "dev/results/retract_subspace_budget_comparison_20260724"
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
    selected = {segment.index: segment for segment in segments}
    variants = [
        variant
        for variant in _variants()
        if not args.variants or variant.name in set(args.variants)
    ]
    if not variants:
        raise ValueError("No variants selected.")

    cases: list[
        tuple[int | str, np.ndarray, np.ndarray, np.ndarray, slice]
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
        seed = selected[args.circle_seed_segment]
        for speed in args.circle_speeds:
            circle_target, circle_q, circle_elbow, core = (
                _extended_circle_inputs(
                    speed,
                    args.circle_radius,
                    args.extension_speed,
                    args.circle_ramp_duration,
                    seed,
                    target_pose,
                    source_q,
                    shoulder,
                    source_elbow,
                )
            )
            cases.append(
                (
                    f"circle_{_number_label(speed)}mps",
                    circle_target,
                    circle_q,
                    circle_elbow,
                    core,
                )
            )

    rows: list[dict[str, float | int | str]] = []
    for case_name, case_target, case_q, case_elbow, core in cases:
        traces: dict[str, ReplayTrace] = {}
        controllers: dict[
            str,
            RetractSubspaceController | NoVelocityLimitController | None,
        ] = {}
        for variant in variants:
            print(f"Simulating case={case_name}, variant={variant.name}...")
            trace, controller = _simulate(
                args.side,
                variant,
                case_target,
                case_q,
                case_elbow,
                settle_duration=args.settle_duration,
                deadband=args.retract_deadband,
                full_speed=args.retract_full_speed,
            )
            traces[variant.name] = trace
            controllers[variant.name] = controller
            _save_trace(
                args.output_dir / f"trace_{case_name}_{variant.name}.npz",
                trace,
            )
        reference_name = "N_no_joint_velocity_limits"
        if reference_name not in traces:
            raise ValueError(
                "N_no_joint_velocity_limits is required as the reference."
            )
        reference = traces[reference_name]
        for variant in variants:
            rows.append(
                _summary(
                    case_name,
                    variant,
                    traces[variant.name],
                    controllers[variant.name],
                    reference,
                    core,
                )
            )
        _plot(args.output_dir / f"{case_name}.png", traces)

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
