#!/usr/bin/env python3
"""Compare fixed and direction-dependent shoulder velocity limits in replay."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import mink
import mujoco
import numpy as np

from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S

from sim_intervention_posture_replay import (
    CONTROL_DT,
    DEFAULT_RUN_DIR,
    Kinematics,
    ReplayTrace,
    DynamicSideArm,
    _load_recorded_commands,
    _setup,
    _source_geometry,
    detect_retract_segments,
    simulate_replay,
)
from sim_reach_braking_experiment import INITIAL_RIGHT


@dataclass(frozen=True)
class Variant:
    """One shoulder-limit strategy."""

    name: str
    high_speed: float
    direction_mode: str | None = None
    near_singularity_only: bool = False
    cap_slew_rate: float | None = None
    acceleration_limit: float | None = None
    excess_acceleration_only: bool = False


class DirectionalShoulderLimit(mink.Limit):
    """Allow extra J1 speed only along a selected beneficial direction."""

    def __init__(
        self,
        kinematics: Kinematics,
        side: str,
        variant: Variant,
        *,
        base_speed: float,
        ratio_stop: float = 0.02,
        ratio_slow: float = 0.08,
    ) -> None:
        solver = kinematics._ik
        assert solver is not None
        if variant.direction_mode is None:
            raise ValueError("Directional limit requires a direction mode.")
        self._solver = solver
        self._model = solver._model
        self._side = side
        self._variant = variant
        self._base_speed = float(base_speed)
        self._high_speed = float(variant.high_speed)
        self._ratio_stop = float(ratio_stop)
        self._ratio_slow = float(ratio_slow)
        self._frame_task = solver._tasks[side]
        self._arm_dofs = np.asarray(
            solver._arm_dofs_by_side[side],
            dtype=int,
        )
        self._shoulder_dof = int(self._arm_dofs[0])
        self._shoulder_joint = self._model.joint(
            f"openarm_{side}_joint1"
        ).id
        self._singularity_limit = solver._singularity_limits[side]
        self._scratch = mink.Configuration(self._model)
        self._outer_step = -1
        self._prepared_step = -2
        self._previous_configuration: np.ndarray | None = None
        self._previous_velocity = 0.0
        self._positive_cap = self._base_speed
        self._negative_cap = self._base_speed
        self._G = np.zeros((2, self._model.nv), dtype=np.float64)
        self._G[0, self._shoulder_dof] = 1.0
        self._G[1, self._shoulder_dof] = -1.0

        self.ratio: list[float] = []
        self.direction_metric: list[float] = []
        self.activation: list[float] = []
        self.positive_cap: list[float] = []
        self.negative_cap: list[float] = []
        self.lower_bound: list[float] = []
        self.upper_bound: list[float] = []
        self.previous_velocity: list[float] = []

    def begin_outer_step(self, configuration: mink.Configuration) -> None:
        """Capture the previous outer command velocity for this control tick."""
        current = configuration.q.copy()
        if self._previous_configuration is None:
            self._previous_velocity = 0.0
        else:
            difference = np.empty(self._model.nv, dtype=np.float64)
            mujoco.mj_differentiatePos(
                self._model,
                difference,
                CONTROL_DT,
                self._previous_configuration,
                current,
            )
            self._previous_velocity = float(
                difference[self._shoulder_dof]
            )
        self._previous_configuration = current
        self._outer_step += 1

    def compute_qp_inequalities(
        self,
        configuration: mink.Configuration,
        dt: float,
    ) -> mink.Constraint:
        """Return asymmetric J1 velocity bounds for the current outer step."""
        if self._prepared_step != self._outer_step:
            self._prepare_bounds(configuration)
            self._prepared_step = self._outer_step
        lower, upper = self.lower_bound[-1], self.upper_bound[-1]
        return mink.Constraint(
            G=self._G,
            h=np.array([upper * dt, -lower * dt], dtype=np.float64),
        )

    def _prepare_bounds(self, configuration: mink.Configuration) -> None:
        state = self._singularity_limit.last_state
        if state is None:
            raise RuntimeError("Singularity limit was not prepared first.")
        ratio = float(state.effective_ratio)
        metric = self._direction_metric(configuration, state.gradient)
        direction_sign = 0.0 if abs(metric) < 1e-10 else float(np.sign(metric))
        activation = (
            self._near_singularity_activation(ratio)
            if self._variant.near_singularity_only
            else 1.0
        )
        extra = activation * (self._high_speed - self._base_speed)
        desired_positive = self._base_speed + extra * max(direction_sign, 0.0)
        desired_negative = self._base_speed + extra * max(-direction_sign, 0.0)

        if self._variant.cap_slew_rate is None:
            self._positive_cap = desired_positive
            self._negative_cap = desired_negative
        else:
            cap_step = self._variant.cap_slew_rate * CONTROL_DT
            self._positive_cap += float(
                np.clip(
                    desired_positive - self._positive_cap,
                    -cap_step,
                    cap_step,
                )
            )
            self._negative_cap += float(
                np.clip(
                    desired_negative - self._negative_cap,
                    -cap_step,
                    cap_step,
                )
            )

        lower = -self._negative_cap
        upper = self._positive_cap
        if self._variant.acceleration_limit is not None:
            velocity_step = self._variant.acceleration_limit * CONTROL_DT
            if self._variant.excess_acceleration_only:
                lower, upper = self._apply_excess_acceleration_window(
                    lower,
                    upper,
                    velocity_step,
                )
            else:
                acceleration_lower = self._previous_velocity - velocity_step
                acceleration_upper = self._previous_velocity + velocity_step
                if acceleration_lower > upper:
                    lower = acceleration_lower
                    upper = acceleration_lower
                elif acceleration_upper < lower:
                    lower = acceleration_upper
                    upper = acceleration_upper
                else:
                    lower = max(lower, acceleration_lower)
                    upper = min(upper, acceleration_upper)
        lower = max(lower, -self._high_speed)
        upper = min(upper, self._high_speed)

        self.ratio.append(ratio)
        self.direction_metric.append(metric)
        self.activation.append(activation)
        self.positive_cap.append(self._positive_cap)
        self.negative_cap.append(self._negative_cap)
        self.lower_bound.append(lower)
        self.upper_bound.append(upper)
        self.previous_velocity.append(self._previous_velocity)

    def _apply_excess_acceleration_window(
        self,
        lower: float,
        upper: float,
        velocity_step: float,
    ) -> tuple[float, float]:
        """Limit acceleration only outside the normal J1 speed envelope."""
        previous = self._previous_velocity
        if self._positive_cap > self._base_speed or previous > self._base_speed:
            upper = min(
                upper,
                max(self._base_speed, previous + velocity_step),
            )
            if previous > self._base_speed:
                lower = max(lower, previous - velocity_step)
        if self._negative_cap > self._base_speed or previous < -self._base_speed:
            lower = max(
                lower,
                min(-self._base_speed, previous - velocity_step),
            )
            if previous < -self._base_speed:
                upper = min(upper, previous + velocity_step)
        if lower > upper:
            transition = float(
                np.clip(
                    previous,
                    -self._high_speed,
                    self._high_speed,
                )
            )
            return transition, transition
        return lower, upper

    def _direction_metric(
        self,
        configuration: mink.Configuration,
        singularity_gradient: np.ndarray,
    ) -> float:
        mode = self._variant.direction_mode
        if mode == "singularity":
            return float(singularity_gradient[0])
        if mode == "reach":
            transform = configuration.get_transform_frame_to_world(
                self._frame_task.frame_name,
                self._frame_task.frame_type,
            )
            end_effector = transform.wxyz_xyz[4:]
            shoulder = configuration.data.xanchor[self._shoulder_joint]
            radial = end_effector - shoulder
            norm = float(np.linalg.norm(radial))
            if norm < 1e-9:
                return 0.0
            radial /= norm
            jacobian = configuration.get_frame_jacobian(
                self._frame_task.frame_name,
                self._frame_task.frame_type,
            )
            reach_gradient = float(
                radial @ jacobian[:3, self._shoulder_dof]
            )
            return -reach_gradient
        if mode == "inertia":
            return -self._shoulder_inertia_gradient(configuration)
        raise ValueError(f"Unknown direction mode: {mode}")

    def _shoulder_inertia_gradient(
        self,
        configuration: mink.Configuration,
    ) -> float:
        epsilon = 1e-4
        tangent = np.zeros(self._model.nv, dtype=np.float64)
        tangent[self._shoulder_dof] = 1.0
        q_plus = configuration.q.copy()
        q_minus = configuration.q.copy()
        mujoco.mj_integratePos(self._model, q_plus, tangent, epsilon)
        mujoco.mj_integratePos(self._model, q_minus, tangent, -epsilon)
        self._scratch.update(q=q_plus)
        plus = self._shoulder_inertia(self._scratch)
        self._scratch.update(q=q_minus)
        minus = self._shoulder_inertia(self._scratch)
        return (plus - minus) / (2.0 * epsilon)

    def _shoulder_inertia(self, configuration: mink.Configuration) -> float:
        mass = np.empty((self._model.nv, self._model.nv), dtype=np.float64)
        mujoco.mj_fullM(self._model, configuration.data, mass)
        return float(mass[self._shoulder_dof, self._shoulder_dof])

    def _near_singularity_activation(self, ratio: float) -> float:
        u = np.clip(
            (ratio - self._ratio_stop)
            / (self._ratio_slow - self._ratio_stop),
            0.0,
            1.0,
        )
        return float(1.0 - (3.0 * u**2 - 2.0 * u**3))


class DirectionalLimitController:
    """Install and update a directional shoulder limit during replay."""

    def __init__(
        self,
        side: str,
        variant: Variant,
        *,
        base_speed: float,
    ) -> None:
        self._side = side
        self._variant = variant
        self._base_speed = base_speed
        self.limit: DirectionalShoulderLimit | None = None

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
            self.limit = DirectionalShoulderLimit(
                kinematics,
                side,
                self._variant,
                base_speed=self._base_speed,
            )
            solver._limits.append(self.limit)
        self.limit.begin_outer_step(solver._config)
        return np.asarray(desired_pose, dtype=np.float64)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--episode", type=int, default=202)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--segments", type=int, nargs="+", default=[12, 13])
    parser.add_argument(
        "--reverse-segments",
        type=int,
        nargs="*",
        default=[],
        help="Also replay selected recorded windows in reverse.",
    )
    parser.add_argument("--settle-duration", type=float, default=0.5)
    parser.add_argument("--base-j1-speed", type=float, default=2.0)
    parser.add_argument("--high-j1-speed", type=float, default=4.0)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[],
        help="Optional variant-name filter.",
    )
    parser.add_argument(
        "--outward-speeds",
        type=float,
        nargs="*",
        default=[],
        help="Also run right-arm shoulder-height outward reaches.",
    )
    parser.add_argument("--outward-distance", type=float, default=0.45)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "dev/results/dynamic_shoulder_limit_comparison_ep202"
        ),
    )
    return parser.parse_args()


def _variants(high_speed: float) -> list[Variant]:
    return [
        Variant("baseline_j1_2", high_speed=2.0),
        Variant("fixed_j1_4", high_speed=high_speed),
        Variant(
            "singularity_escape_always",
            high_speed=high_speed,
            direction_mode="singularity",
        ),
        Variant(
            "singularity_escape_near",
            high_speed=high_speed,
            direction_mode="singularity",
            near_singularity_only=True,
        ),
        Variant(
            "singularity_escape_near_cap_slew50",
            high_speed=high_speed,
            direction_mode="singularity",
            near_singularity_only=True,
            cap_slew_rate=50.0,
        ),
        Variant(
            "singularity_escape_near_accel100",
            high_speed=high_speed,
            direction_mode="singularity",
            near_singularity_only=True,
            acceleration_limit=100.0,
        ),
        Variant(
            "reach_retract_always",
            high_speed=high_speed,
            direction_mode="reach",
        ),
        Variant(
            "reach_retract_near",
            high_speed=high_speed,
            direction_mode="reach",
            near_singularity_only=True,
        ),
        Variant(
            "reach_retract_always_cap_slew50",
            high_speed=high_speed,
            direction_mode="reach",
            cap_slew_rate=50.0,
        ),
        Variant(
            "reach_retract_always_accel50",
            high_speed=high_speed,
            direction_mode="reach",
            acceleration_limit=50.0,
        ),
        Variant(
            "reach_retract_always_accel20",
            high_speed=high_speed,
            direction_mode="reach",
            acceleration_limit=20.0,
        ),
        Variant(
            "reach_retract_always_accel30",
            high_speed=high_speed,
            direction_mode="reach",
            acceleration_limit=30.0,
        ),
        Variant(
            "reach_retract_always_accel100",
            high_speed=high_speed,
            direction_mode="reach",
            acceleration_limit=100.0,
        ),
        Variant(
            "reach_retract_excess_accel20",
            high_speed=high_speed,
            direction_mode="reach",
            acceleration_limit=20.0,
            excess_acceleration_only=True,
        ),
        Variant(
            "reach_retract_excess_accel30",
            high_speed=high_speed,
            direction_mode="reach",
            acceleration_limit=30.0,
            excess_acceleration_only=True,
        ),
        Variant(
            "inertia_decrease_always",
            high_speed=high_speed,
            direction_mode="inertia",
        ),
    ]


def _simulate_variant(
    side: str,
    variant: Variant,
    target_pose: np.ndarray,
    source_q: np.ndarray,
    source_elbow: np.ndarray,
    *,
    settle_duration: float,
    base_speed: float,
) -> tuple[ReplayTrace, DirectionalShoulderLimit | None]:
    controller = (
        None
        if variant.direction_mode is None
        else DirectionalLimitController(
            side,
            variant,
            base_speed=base_speed,
        )
    )
    multiplier = variant.high_speed / base_speed
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
    return trace, None if controller is None else controller.limit


def _mass_matrix_signal(
    side: str,
    q_trajectory: np.ndarray,
) -> np.ndarray:
    setup = _setup(side)
    model, data = setup.model, setup.data
    resolved = (
        setup.joint_resolver._left
        if side == "left"
        else setup.joint_resolver._right
    )
    shoulder_dof = int(resolved.arm_dof[0])
    mass = np.empty((model.nv, model.nv), dtype=np.float64)
    signal = np.empty(q_trajectory.shape[0], dtype=np.float64)
    for index, q in enumerate(q_trajectory):
        setup.joint_resolver.set_qpos(
            data.qpos,
            np.append(np.asarray(q, dtype=np.float64), 0.0),
            side,
        )
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, data, mass)
        signal[index] = mass[shoulder_dof, shoulder_dof]
    return signal


def _synthetic_outward_inputs(
    speed: float,
    distance: float,
    *,
    settle_duration: float,
    pre_hold_duration: float = 0.2,
    post_hold_duration: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, slice]:
    """Build a shoulder-height outward target stream for the right arm."""
    if speed <= 0.0:
        raise ValueError("Outward speed must be positive.")
    plant = DynamicSideArm("right", INITIAL_RIGHT[:7])
    plant.settle(settle_duration)
    initial_pose = plant.pose()
    initial_elbow = plant.elbow()
    pre_ticks = int(round(pre_hold_duration / CONTROL_DT))
    ramp_ticks = int(round(distance / speed / CONTROL_DT))
    post_ticks = int(round(post_hold_duration / CONTROL_DT))
    count = pre_ticks + ramp_ticks + post_ticks
    target_pose = np.repeat(initial_pose[None, :], count, axis=0)
    progress = np.clip(
        (np.arange(count) - pre_ticks + 1) * speed * CONTROL_DT,
        0.0,
        distance,
    )
    target_pose[:, 0] += progress
    source_q = np.repeat(INITIAL_RIGHT[None, :7], count, axis=0)
    source_elbow = np.repeat(initial_elbow[None, :], count, axis=0)
    return (
        target_pose,
        source_q,
        source_elbow,
        slice(pre_ticks, pre_ticks + ramp_ticks),
    )


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _summary_row(
    segment: int | str,
    variant: Variant,
    trace: ReplayTrace,
    dynamic_limit: DirectionalShoulderLimit | None,
    command_inertia: np.ndarray,
    actual_inertia: np.ndarray,
    core: slice,
    *,
    base_speed: float,
) -> dict[str, float | int | str]:
    command_acceleration = np.diff(trace.command_dq, axis=0) / CONTROL_DT
    actual_acceleration = np.diff(trace.actual_dq, axis=0) / CONTROL_DT
    actual_velocity = np.gradient(
        trace.actual_pose[:, :3],
        CONTROL_DT,
        axis=0,
    )
    actual_ee_acceleration = np.linalg.norm(
        np.gradient(actual_velocity, CONTROL_DT, axis=0),
        axis=1,
    )
    position_error = np.linalg.norm(
        trace.actual_pose[:, :3] - trace.target_pose[:, :3],
        axis=1,
    )
    command_lag = trace.command_q[:, 0] - trace.actual_q[:, 0]
    elbow_y = trace.actual_elbow[:, 1]
    source_elbow_error = np.linalg.norm(
        trace.command_elbow - trace.source_elbow,
        axis=1,
    )

    if dynamic_limit is None:
        positive_cap = np.full(trace.time.shape, variant.high_speed)
        negative_cap = positive_cap
        direction_metric = np.full(trace.time.shape, np.nan)
        activation = np.zeros(trace.time.shape)
    else:
        positive_cap = np.asarray(dynamic_limit.positive_cap)
        negative_cap = np.asarray(dynamic_limit.negative_cap)
        direction_metric = np.asarray(dynamic_limit.direction_metric)
        activation = np.asarray(dynamic_limit.activation)
    j1_utilization = np.where(
        trace.command_dq[:, 0] >= 0.0,
        trace.command_dq[:, 0] / positive_cap,
        -trace.command_dq[:, 0] / negative_cap,
    )
    base_caps = np.asarray(
        ARM_JOINT_VELOCITY_LIMITS_RAD_S,
        dtype=np.float64,
    )
    other_utilization = np.abs(trace.command_dq[:, 1:]) / base_caps[1:]
    core_indices = np.arange(trace.time.size)[core]
    acceleration_core = core_indices[core_indices < command_acceleration.shape[0]]

    return {
        "segment": segment,
        "variant": variant.name,
        "base_j1_speed_rad_s": base_speed,
        "high_j1_speed_rad_s": variant.high_speed,
        "target_mean_speed_m_s": float(
            np.mean(
                np.linalg.norm(
                    np.gradient(
                        trace.target_pose[:, :3],
                        CONTROL_DT,
                        axis=0,
                    ),
                    axis=1,
                )[core]
            )
        ),
        "min_rho": float(np.min(trace.rho[core])),
        "j1_limit_fraction_core": float(
            np.mean(j1_utilization[core] >= 0.98)
        ),
        "other_limit_fraction_core": float(
            np.mean(np.any(other_utilization[core] >= 0.98, axis=1))
        ),
        "j1_peak_command_speed_rad_s": float(
            np.max(np.abs(trace.command_dq[core, 0]))
        ),
        "j1_peak_actual_speed_rad_s": float(
            np.max(np.abs(trace.actual_dq[core, 0]))
        ),
        "j1_p99_command_acceleration_rad_s2": float(
            np.percentile(
                np.abs(command_acceleration[acceleration_core, 0]),
                99.0,
            )
        ),
        "j1_p99_actual_acceleration_rad_s2": float(
            np.percentile(
                np.abs(actual_acceleration[acceleration_core, 0]),
                99.0,
            )
        ),
        "j1_command_actual_lag_rmse_rad": _rms(command_lag[core]),
        "j1_command_actual_lag_max_rad": float(
            np.max(np.abs(command_lag[core]))
        ),
        "actual_elbow_lateral_delta_m": float(
            elbow_y[core.stop - 1] - elbow_y[core.start]
        ),
        "actual_elbow_lateral_range_m": float(
            np.ptp(elbow_y[core])
        ),
        "command_elbow_source_rmse_m": _rms(source_elbow_error[core]),
        "actual_position_rmse_m": _rms(position_error[core]),
        "actual_position_peak_m": float(np.max(position_error[core])),
        "actual_ee_acceleration_p99_m_s2": float(
            np.percentile(actual_ee_acceleration[core], 99.0)
        ),
        "actual_ee_acceleration_peak_m_s2": float(
            np.max(actual_ee_acceleration[core])
        ),
        "command_shoulder_inertia_start": float(command_inertia[core.start]),
        "command_shoulder_inertia_end": float(command_inertia[core.stop - 1]),
        "actual_shoulder_inertia_start": float(actual_inertia[core.start]),
        "actual_shoulder_inertia_end": float(actual_inertia[core.stop - 1]),
        "dynamic_activation_mean": float(np.mean(activation[core])),
        "direction_metric_sign_changes": int(
            np.count_nonzero(
                np.diff(np.signbit(direction_metric[core]))
            )
            if dynamic_limit is not None
            else 0
        ),
        "solve_failures": int(np.count_nonzero(trace.solve_failed)),
    }


def _plot_segment(
    path: Path,
    traces: dict[str, ReplayTrace],
    limits: dict[str, DirectionalShoulderLimit | None],
    inertias: dict[str, np.ndarray],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping plot generation.")
        return

    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)
    for name, trace in traces.items():
        time = trace.time
        axes[0].plot(time, trace.command_dq[:, 0], label=f"{name} cmd")
        axes[1].plot(time, trace.actual_dq[:, 0], label=name)
        axes[2].plot(
            time,
            trace.actual_elbow[:, 1] - trace.actual_elbow[0, 1],
            label=name,
        )
        axes[3].plot(time, trace.rho, label=name)
        axes[4].plot(
            time,
            np.linalg.norm(
                trace.actual_pose[:, :3] - trace.target_pose[:, :3],
                axis=1,
            ),
            label=name,
        )
        axes[5].plot(time, inertias[name], label=name)
        dynamic_limit = limits[name]
        if dynamic_limit is not None:
            axes[0].plot(
                time,
                dynamic_limit.positive_cap,
                linestyle=":",
                alpha=0.6,
            )
            axes[0].plot(
                time,
                -np.asarray(dynamic_limit.negative_cap),
                linestyle=":",
                alpha=0.6,
            )
    axes[0].set_ylabel("J1 cmd dq [rad/s]")
    axes[1].set_ylabel("J1 actual dq [rad/s]")
    axes[2].set_ylabel("Elbow lateral delta [m]")
    axes[3].set_ylabel("rho")
    axes[4].set_ylabel("Actual EEF error [m]")
    axes[5].set_ylabel("Shoulder M11")
    axes[5].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


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
    rows: list[dict[str, float | int | str]] = []
    variants = [
        variant
        for variant in _variants(args.high_j1_speed)
        if not args.variants or variant.name in set(args.variants)
    ]
    if not variants:
        raise ValueError("No shoulder-limit variants were selected.")

    for segment_index in args.segments:
        segment = selected[segment_index]
        window = slice(segment.start, segment.end)
        core = slice(
            segment.core_start - segment.start,
            segment.core_end - segment.start,
        )
        traces: dict[str, ReplayTrace] = {}
        limits: dict[str, DirectionalShoulderLimit | None] = {}
        inertias: dict[str, np.ndarray] = {}
        for variant in variants:
            print(f"Simulating segment={segment_index}, variant={variant.name}...")
            trace, dynamic_limit = _simulate_variant(
                args.side,
                variant,
                target_pose[window],
                source_q[window],
                source_elbow[window],
                settle_duration=args.settle_duration,
                base_speed=args.base_j1_speed,
            )
            command_inertia = _mass_matrix_signal(
                args.side,
                trace.command_q,
            )
            actual_inertia = _mass_matrix_signal(
                args.side,
                trace.actual_q,
            )
            rows.append(
                _summary_row(
                    segment_index,
                    variant,
                    trace,
                    dynamic_limit,
                    command_inertia,
                    actual_inertia,
                    core,
                    base_speed=args.base_j1_speed,
                )
            )
            traces[variant.name] = trace
            limits[variant.name] = dynamic_limit
            inertias[variant.name] = actual_inertia
            payload = dict(trace.__dict__)
            payload["command_shoulder_inertia"] = command_inertia
            payload["actual_shoulder_inertia"] = actual_inertia
            if dynamic_limit is not None:
                payload["j1_positive_cap"] = np.asarray(
                    dynamic_limit.positive_cap
                )
                payload["j1_negative_cap"] = np.asarray(
                    dynamic_limit.negative_cap
                )
                payload["j1_direction_metric"] = np.asarray(
                    dynamic_limit.direction_metric
                )
                payload["j1_dynamic_activation"] = np.asarray(
                    dynamic_limit.activation
                )
            np.savez_compressed(
                args.output_dir
                / f"trace_segment_{segment_index:02d}_{variant.name}.npz",
                **payload,
            )
        _plot_segment(
            args.output_dir / f"segment_{segment_index:02d}.png",
            traces,
            limits,
            inertias,
        )

    outward_names = {
        "baseline_j1_2",
        "fixed_j1_4",
        "reach_retract_always",
        "reach_retract_always_accel30",
        "reach_retract_always_accel50",
    }
    reverse_names = outward_names
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
        reverse_target = target_pose[window][::-1].copy()
        reverse_q = source_q[window][::-1].copy()
        reverse_elbow = source_elbow[window][::-1].copy()
        traces = {}
        limits = {}
        inertias = {}
        for variant in variants:
            if variant.name not in reverse_names:
                continue
            print(
                f"Simulating reverse={segment_index}, "
                f"variant={variant.name}..."
            )
            trace, dynamic_limit = _simulate_variant(
                args.side,
                variant,
                reverse_target,
                reverse_q,
                reverse_elbow,
                settle_duration=args.settle_duration,
                base_speed=args.base_j1_speed,
            )
            command_inertia = _mass_matrix_signal(
                args.side,
                trace.command_q,
            )
            actual_inertia = _mass_matrix_signal(args.side, trace.actual_q)
            rows.append(
                _summary_row(
                    f"reverse_{segment_index}",
                    variant,
                    trace,
                    dynamic_limit,
                    command_inertia,
                    actual_inertia,
                    core,
                    base_speed=args.base_j1_speed,
                )
            )
            traces[variant.name] = trace
            limits[variant.name] = dynamic_limit
            inertias[variant.name] = actual_inertia
            payload = dict(trace.__dict__)
            payload["command_shoulder_inertia"] = command_inertia
            payload["actual_shoulder_inertia"] = actual_inertia
            if dynamic_limit is not None:
                payload["j1_positive_cap"] = np.asarray(
                    dynamic_limit.positive_cap
                )
                payload["j1_negative_cap"] = np.asarray(
                    dynamic_limit.negative_cap
                )
                payload["j1_direction_metric"] = np.asarray(
                    dynamic_limit.direction_metric
                )
            np.savez_compressed(
                args.output_dir
                / f"trace_reverse_{segment_index}_{variant.name}.npz",
                **payload,
            )
        _plot_segment(
            args.output_dir / f"reverse_{segment_index}.png",
            traces,
            limits,
            inertias,
        )

    for speed in args.outward_speeds:
        outward_target, outward_q, outward_elbow, core = (
            _synthetic_outward_inputs(
                speed,
                args.outward_distance,
                settle_duration=args.settle_duration,
            )
        )
        traces = {}
        limits = {}
        inertias = {}
        for variant in variants:
            if variant.name not in outward_names:
                continue
            print(f"Simulating outward={speed:g}, variant={variant.name}...")
            trace, dynamic_limit = _simulate_variant(
                "right",
                variant,
                outward_target,
                outward_q,
                outward_elbow,
                settle_duration=args.settle_duration,
                base_speed=args.base_j1_speed,
            )
            command_inertia = _mass_matrix_signal("right", trace.command_q)
            actual_inertia = _mass_matrix_signal("right", trace.actual_q)
            rows.append(
                _summary_row(
                    f"outward_{speed:g}",
                    variant,
                    trace,
                    dynamic_limit,
                    command_inertia,
                    actual_inertia,
                    core,
                    base_speed=args.base_j1_speed,
                )
            )
            traces[variant.name] = trace
            limits[variant.name] = dynamic_limit
            inertias[variant.name] = actual_inertia
            payload = dict(trace.__dict__)
            payload["command_shoulder_inertia"] = command_inertia
            payload["actual_shoulder_inertia"] = actual_inertia
            if dynamic_limit is not None:
                payload["j1_positive_cap"] = np.asarray(
                    dynamic_limit.positive_cap
                )
                payload["j1_negative_cap"] = np.asarray(
                    dynamic_limit.negative_cap
                )
                payload["j1_direction_metric"] = np.asarray(
                    dynamic_limit.direction_metric
                )
            speed_label = str(speed).replace(".", "p")
            np.savez_compressed(
                args.output_dir
                / f"trace_outward_{speed_label}_{variant.name}.npz",
                **payload,
            )
        speed_label = str(speed).replace(".", "p")
        _plot_segment(
            args.output_dir / f"outward_{speed_label}.png",
            traces,
            limits,
            inertias,
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
