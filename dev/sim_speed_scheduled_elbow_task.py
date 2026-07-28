#!/usr/bin/env python3
"""Compare a non-augmented speed-scheduled elbow task in MuJoCo.

The experiment keeps the production Mink solve structure. It compares the old
physical joint velocity limits against the locally raised-limit reference and
tests both direct ``J_psi Delta q`` and exact-nullspace ``J_psi N Delta q``
spatial-elbow objectives.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from openarm_control import IKParams, Kinematics
from openarm_control.speed_scheduled_elbow_task import ElbowSwivelCoordinate

from sim_reach_braking_experiment import (
    CONTROL_DT,
    INITIAL_RIGHT,
    DynamicArm,
    _setup,
)
from speed_scheduled_near_singular_task import (
    SpeedScheduledNearSingularTask,
)


LOW_CAPS = np.array(
    [1.57, 1.57, 3.14, 3.14, 12.6, 12.6, 12.6],
    dtype=np.float64,
)
RAISED_CAPS = np.array(
    [3.0, 3.0, 4.14, 4.14, 12.6, 12.6, 12.6],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Candidate:
    """One velocity-envelope and elbow-task configuration."""

    name: str
    caps: tuple[float, ...]
    enabled: bool = False
    projection: str = "direct"
    velocity_cost: float = 0.0
    return_rate: float = 0.5
    max_return_speed: float = 0.1
    max_return_acceleration: float = 2.0
    corridor_cost: float = 0.0
    position_error_limit: float = 0.0
    orientation_error_limit: float = 0.0
    frame_limit_linear_slow: float = 0.2
    frame_limit_linear_fast: float = 0.5
    frame_limit_activation_rise_rate: float = 4.0
    frame_limit_activation_fall_rate: float = 2.0
    near_singular_cost: float = 0.0
    near_singular_return_rate: float = 0.0
    near_singular_max_return_speed: float = 0.0
    near_singular_low: float = 0.02
    near_singular_high: float = 0.08
    linear_slow: float = 0.45
    linear_fast: float = 0.6
    activation_rise_rate: float = 4.0
    activation_fall_rate: float = 2.0


@dataclass(frozen=True)
class Scenario:
    """A target-pose sequence sampled at the production control rate."""

    name: str
    poses: np.ndarray


@dataclass
class Trace:
    """Signals used to compare command kinematics and MuJoCo dynamics."""

    target: np.ndarray
    command_q: np.ndarray
    command_dq: np.ndarray
    command_ddq: np.ndarray
    actual_q: np.ndarray
    actual_dq: np.ndarray
    actual_ddq: np.ndarray
    command_ee: np.ndarray
    actual_ee: np.ndarray
    command_elbow: np.ndarray
    actual_elbow: np.ndarray
    command_swivel: np.ndarray
    actual_swivel: np.ndarray
    activation: np.ndarray
    solve_ms: np.ndarray
    failed: np.ndarray


@dataclass
class Metrics:
    """Scalar comparison metrics for one candidate and scenario."""

    scenario: str
    candidate: str
    actual_position_rmse_m: float
    actual_orientation_rmse_rad: float
    command_position_rmse_m: float
    actual_elbow_path_rmse_to_raised_m: float
    actual_swivel_range_deg: float
    actual_swivel_delta_deg: float
    peak_velocity_utilization: float
    saturated_joint_fraction: float
    command_accel_p99_rad_s2: float
    command_accel_max_rad_s2: float
    actual_accel_p99_rad_s2: float
    actual_accel_max_rad_s2: float
    command_step_max_rad: float
    activation_mean: float
    activation_max: float
    solve_mean_ms: float
    solve_p95_ms: float
    failure_count: int


def _velocity_limits(caps: tuple[float, ...]) -> dict[str, float]:
    return {
        f"openarm_{side}_joint{index + 1}": float(cap)
        for side in ("left", "right")
        for index, cap in enumerate(caps)
    }


def _make_kinematics(candidate: Candidate) -> Kinematics:
    kinematics = Kinematics(
        _setup("right"),
        IKParams(
            position_cost=10.0,
            orientation_cost=1.0,
            frame_position_error_limit=candidate.position_error_limit,
            frame_orientation_error_limit=candidate.orientation_error_limit,
            frame_error_limit_linear_slow=candidate.frame_limit_linear_slow,
            frame_error_limit_linear_fast=candidate.frame_limit_linear_fast,
            frame_error_limit_activation_rise_rate=(
                candidate.frame_limit_activation_rise_rate
            ),
            frame_error_limit_activation_fall_rate=(
                candidate.frame_limit_activation_fall_rate
            ),
            lm_damping=0.02,
            damping=0.1,
            posture_cost=0.0,
            dt=CONTROL_DT,
            max_iters=5,
            velocity_limits=_velocity_limits(candidate.caps),
            joint_limit_recovery_velocity_scale=1.0,
            nullspace_cost=12.0,
            nullspace_return_rate=1.6,
            nullspace_max_speed=1.0,
            nullspace_singularity_low=0.02,
            nullspace_singularity_high=0.05,
            nullspace_characteristic_length=0.3,
            speed_scheduled_elbow=candidate.enabled,
            speed_elbow_linear_slow=candidate.linear_slow,
            speed_elbow_linear_fast=candidate.linear_fast,
            speed_elbow_activation_rise_rate=(candidate.activation_rise_rate),
            speed_elbow_activation_fall_rate=(candidate.activation_fall_rate),
            speed_elbow_velocity_cost=candidate.velocity_cost,
            speed_elbow_return_rate=candidate.return_rate,
            speed_elbow_max_return_speed=candidate.max_return_speed,
            speed_elbow_max_return_acceleration=(candidate.max_return_acceleration),
            speed_elbow_corridor_cost=candidate.corridor_cost,
            speed_elbow_projection=candidate.projection,
            elbow_soft_limit_cost=0.0,
            elbow_braking_guard_angle=0.08,
            elbow_braking_profile="distance",
            elbow_braking_acceleration=20.0,
            elbow_braking_slowdown_distance=0.5,
            joint_limit_braking=True,
            joint_limit_braking_slowdown_distance=0.5,
            joint_limit_braking_exponent=2.0,
            joint_limit_braking_reaction_time=0.04,
            joint_limit_braking_distance_buffer=0.01,
            singularity_approach_limit=True,
            singularity_ratio_stop=0.02,
            singularity_ratio_slow=0.08,
            singularity_max_approach_rate=0.25,
            singularity_braking_exponent=2.0,
            kinetic_energy_cost=3e-5,
        ),
    )
    if candidate.near_singular_cost > 0.0:
        solver = kinematics._ik
        assert solver is not None
        solver._speed_elbow_tasks["right"] = SpeedScheduledNearSingularTask(
            model=solver._model,
            frame_task=solver._tasks["right"],
            dof_indices=solver._arm_dofs_by_side["right"],
            home_qpos=solver._posture_task.target_q,
            control_dt=CONTROL_DT,
            substep_dt=solver._substep_dt,
            cost=candidate.near_singular_cost,
            linear_speed_slow=candidate.linear_slow,
            linear_speed_fast=candidate.linear_fast,
            activation_rise_rate=candidate.activation_rise_rate,
            activation_fall_rate=candidate.activation_fall_rate,
            singularity_low=candidate.near_singular_low,
            singularity_high=candidate.near_singular_high,
            characteristic_length=0.3,
            return_rate=candidate.near_singular_return_rate,
            max_return_speed=candidate.near_singular_max_return_speed,
        )
    return kinematics


def _hold(pose: np.ndarray, duration: float) -> np.ndarray:
    count = max(int(round(duration / CONTROL_DT)), 1)
    return np.repeat(pose[None, :], count, axis=0)


def _linear_segment(
    start: np.ndarray,
    end: np.ndarray,
    speed: float,
) -> np.ndarray:
    distance = float(np.linalg.norm(end[:3] - start[:3]))
    count = max(int(np.ceil(distance / (speed * CONTROL_DT))), 1)
    alpha = np.linspace(1.0 / count, 1.0, count)
    output = np.repeat(start[None, :], count, axis=0)
    output[:, :3] = start[None, :3] + alpha[:, None] * (end[None, :3] - start[None, :3])
    output[:, 3:] = end[3:]
    return output


def _reach_retract(initial: np.ndarray, speed: float) -> Scenario:
    outward = initial.copy()
    outward[:3] += np.array([0.34, 0.0, 0.0])
    retracted = initial.copy()
    retracted[:3] += np.array([-0.03, 0.13, -0.11])
    poses = np.vstack(
        [
            _hold(initial, 0.3),
            _linear_segment(initial, outward, speed),
            _hold(outward, 0.2),
            _linear_segment(outward, retracted, speed),
            _hold(retracted, 0.4),
        ]
    )
    return Scenario(f"reach_retract_{speed:.2f}", poses)


def _circle(initial: np.ndarray, speed: float) -> Scenario:
    radius = 0.08
    duration = 2.0 * np.pi * radius / speed
    count = max(int(np.ceil(duration / CONTROL_DT)), 1)
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=True)
    poses = np.repeat(initial[None, :], count, axis=0)
    poses[:, 1] += radius * (np.cos(theta) - 1.0)
    poses[:, 2] += radius * np.sin(theta)
    return Scenario(
        f"circle_{speed:.2f}",
        np.vstack([_hold(initial, 0.3), poses, _hold(initial, 0.3)]),
    )


def _wrist_flip(initial: np.ndarray) -> Scenario:
    duration = 1.6
    count = int(round(duration / CONTROL_DT))
    time_values = np.arange(count) * CONTROL_DT
    roll = 2.2 * np.sin(2.0 * np.pi * 1.25 * time_values)
    base = Rotation.from_quat(
        np.array([initial[4], initial[5], initial[6], initial[3]])
    )
    poses = np.repeat(initial[None, :], count, axis=0)
    for index, angle in enumerate(roll):
        quaternion = (base * Rotation.from_rotvec([angle, 0.0, 0.0])).as_quat()
        poses[index, 3:] = quaternion[[3, 0, 1, 2]]
    return Scenario(
        "wrist_flip",
        np.vstack([_hold(initial, 0.3), poses, _hold(initial, 0.3)]),
    )


class ElbowMonitor:
    """Evaluate home-referenced swivel and elbow position for arbitrary q."""

    def __init__(self, kinematics: Kinematics) -> None:
        solver = kinematics._ik
        assert solver is not None
        self._setup = kinematics.setup
        self._base_q = solver._config.q.copy()
        self._qpos_indices = np.asarray(
            self._setup.joint_resolver._right.arm_qpos,
            dtype=int,
        )
        self._coordinate = ElbowSwivelCoordinate(
            self._setup.model,
            "right",
            solver._tasks["right"],
            solver._posture_task.target_q,
            np.asarray(self._setup.joint_resolver._right.arm_dof, dtype=int),
            finite_difference_epsilon=1e-5,
        )
        self._data = mujoco.MjData(self._setup.model)
        self._elbow_joint = self._setup.model.joint("openarm_right_joint4").id

    def evaluate(self, arm_q: np.ndarray) -> tuple[float, np.ndarray]:
        q = self._base_q.copy()
        q[self._qpos_indices] = arm_q
        swivel, _ = self._coordinate.value(q)
        self._data.qpos[:] = q
        mujoco.mj_forward(self._setup.model, self._data)
        return swivel, self._data.xanchor[self._elbow_joint].copy()


def _orientation_error(target: np.ndarray, actual: np.ndarray) -> float:
    target_rotation = Rotation.from_quat(target[[4, 5, 6, 3]])
    actual_rotation = Rotation.from_quat(actual[[4, 5, 6, 3]])
    return float((target_rotation.inv() * actual_rotation).magnitude())


def simulate(candidate: Candidate, scenario: Scenario) -> Trace:
    """Run one production-shaped IK and MuJoCo dynamics simulation."""
    kinematics = _make_kinematics(candidate)
    dynamics = DynamicArm()
    dynamics.settle(0.5)
    kinematics.sync(dynamics.bimanual_driver_qpos())
    monitor = ElbowMonitor(kinematics)

    count = scenario.poses.shape[0]
    command_q = np.empty((count, 7))
    command_dq = np.empty((count, 7))
    command_ddq = np.empty((count, 7))
    actual_q = np.empty((count, 7))
    actual_dq = np.empty((count, 7))
    actual_ddq = np.empty((count, 7))
    command_ee = np.empty((count, 7))
    actual_ee = np.empty((count, 7))
    command_elbow = np.empty((count, 3))
    actual_elbow = np.empty((count, 3))
    command_swivel = np.empty(count)
    actual_swivel = np.empty(count)
    activation = np.zeros(count)
    solve_ms = np.empty(count)
    failed = np.zeros(count, dtype=bool)

    previous_command = dynamics.right_q()
    previous_command_dq = np.zeros(7)
    previous_actual_dq = dynamics.right_dq()
    for tick, target in enumerate(scenario.poses):
        kinematics.set_target("right", target)
        started = time.perf_counter()
        result = kinematics.solve()
        solve_ms[tick] = (time.perf_counter() - started) * 1000.0
        failed[tick] = result is None
        current_command = (
            previous_command.copy() if result is None else result[:7].astype(np.float64)
        )
        current_command_dq = (current_command - previous_command) / CONTROL_DT
        command_q[tick] = current_command
        command_dq[tick] = current_command_dq
        command_ddq[tick] = (current_command_dq - previous_command_dq) / CONTROL_DT

        dynamics.set_command(current_command)
        dynamics.step_control_period()
        current_actual_q = dynamics.right_q()
        current_actual_dq = dynamics.right_dq()
        actual_q[tick] = current_actual_q
        actual_dq[tick] = current_actual_dq
        actual_ddq[tick] = (current_actual_dq - previous_actual_dq) / CONTROL_DT
        command_ee[tick] = kinematics.fk(
            "right",
            np.append(current_command, 0.0),
        )
        actual_ee[tick] = dynamics.right_ee()
        command_swivel[tick], command_elbow[tick] = monitor.evaluate(current_command)
        actual_swivel[tick], actual_elbow[tick] = monitor.evaluate(current_actual_q)

        solver = kinematics._ik
        assert solver is not None
        elbow_task = solver._speed_elbow_tasks.get("right")
        if elbow_task is not None and elbow_task.last_state is not None:
            activation[tick] = getattr(
                elbow_task.last_state,
                "combined_activation",
                getattr(
                    elbow_task.last_state,
                    "activation",
                    getattr(elbow_task.last_state, "speed_activation", 0.0),
                ),
            )

        previous_command = current_command
        previous_command_dq = current_command_dq
        previous_actual_dq = current_actual_dq

    return Trace(
        target=scenario.poses.copy(),
        command_q=command_q,
        command_dq=command_dq,
        command_ddq=command_ddq,
        actual_q=actual_q,
        actual_dq=actual_dq,
        actual_ddq=actual_ddq,
        command_ee=command_ee,
        actual_ee=actual_ee,
        command_elbow=command_elbow,
        actual_elbow=actual_elbow,
        command_swivel=command_swivel,
        actual_swivel=actual_swivel,
        activation=activation,
        solve_ms=solve_ms,
        failed=failed,
    )


def compute_metrics(
    candidate: Candidate,
    scenario: Scenario,
    trace: Trace,
    raised_trace: Trace,
) -> Metrics:
    """Summarize tracking, branch similarity, smoothness, and runtime."""
    position_error = trace.actual_ee[:, :3] - trace.target[:, :3]
    command_position_error = trace.command_ee[:, :3] - trace.target[:, :3]
    orientation_error = np.array(
        [
            _orientation_error(target, actual)
            for target, actual in zip(trace.target, trace.actual_ee, strict=True)
        ]
    )
    elbow_difference = trace.actual_elbow - raised_trace.actual_elbow
    caps = np.asarray(candidate.caps)
    velocity_utilization = np.abs(trace.command_dq) / caps[None, :]
    saturated = velocity_utilization >= 0.99
    return Metrics(
        scenario=scenario.name,
        candidate=candidate.name,
        actual_position_rmse_m=float(
            np.sqrt(np.mean(np.sum(position_error * position_error, axis=1)))
        ),
        actual_orientation_rmse_rad=float(
            np.sqrt(np.mean(orientation_error * orientation_error))
        ),
        command_position_rmse_m=float(
            np.sqrt(
                np.mean(
                    np.sum(
                        command_position_error * command_position_error,
                        axis=1,
                    )
                )
            )
        ),
        actual_elbow_path_rmse_to_raised_m=float(
            np.sqrt(np.mean(np.sum(elbow_difference * elbow_difference, axis=1)))
        ),
        actual_swivel_range_deg=float(np.rad2deg(np.ptp(trace.actual_swivel))),
        actual_swivel_delta_deg=float(
            np.rad2deg(trace.actual_swivel[-1] - trace.actual_swivel[0])
        ),
        peak_velocity_utilization=float(np.max(velocity_utilization)),
        saturated_joint_fraction=float(np.mean(saturated)),
        command_accel_p99_rad_s2=float(np.quantile(np.abs(trace.command_ddq), 0.99)),
        command_accel_max_rad_s2=float(np.max(np.abs(trace.command_ddq))),
        actual_accel_p99_rad_s2=float(np.quantile(np.abs(trace.actual_ddq), 0.99)),
        actual_accel_max_rad_s2=float(np.max(np.abs(trace.actual_ddq))),
        command_step_max_rad=float(np.max(np.abs(np.diff(trace.command_q, axis=0)))),
        activation_mean=float(np.mean(trace.activation)),
        activation_max=float(np.max(trace.activation)),
        solve_mean_ms=float(np.mean(trace.solve_ms)),
        solve_p95_ms=float(np.quantile(trace.solve_ms, 0.95)),
        failure_count=int(np.count_nonzero(trace.failed)),
    )


def _save_trace(path: Path, trace: Trace) -> None:
    np.savez_compressed(
        path,
        **{name: value for name, value in vars(trace).items()},
    )


def _candidates(quick: bool) -> list[Candidate]:
    base = [
        Candidate("raised_reference", tuple(RAISED_CAPS)),
        Candidate("low_baseline", tuple(LOW_CAPS)),
        Candidate(
            "low_near_singular_hold_c1",
            tuple(LOW_CAPS),
            near_singular_cost=1.0,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_near_singular_hold_c3",
            tuple(LOW_CAPS),
            near_singular_cost=3.0,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_near_singular_hold_c6",
            tuple(LOW_CAPS),
            near_singular_cost=6.0,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_near_singular_hold_c12",
            tuple(LOW_CAPS),
            near_singular_cost=12.0,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_near_singular_wide_c6",
            tuple(LOW_CAPS),
            near_singular_cost=6.0,
            near_singular_low=0.04,
            near_singular_high=0.2,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_near_singular_wide_c30",
            tuple(LOW_CAPS),
            near_singular_cost=30.0,
            near_singular_low=0.04,
            near_singular_high=0.2,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_leak_position_only_p3p0mm",
            tuple(LOW_CAPS),
            position_error_limit=0.003,
        ),
        Candidate(
            "low_leak_p3p0mm_window_0p5_1p0",
            tuple(LOW_CAPS),
            position_error_limit=0.003,
            frame_limit_linear_slow=0.5,
            frame_limit_linear_fast=1.0,
        ),
        Candidate(
            "low_leak_p20mm_window_0p5_1p0",
            tuple(LOW_CAPS),
            position_error_limit=0.02,
            frame_limit_linear_slow=0.5,
            frame_limit_linear_fast=1.0,
        ),
        Candidate(
            "low_direct_early_c0p002",
            tuple(LOW_CAPS),
            enabled=True,
            velocity_cost=0.002,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_corridor_only_early_c0p004",
            tuple(LOW_CAPS),
            enabled=True,
            corridor_cost=0.004,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_corridor_only_early_c0p0005",
            tuple(LOW_CAPS),
            enabled=True,
            corridor_cost=0.0005,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_leak_p3p0mm_direct_early_c0p002",
            tuple(LOW_CAPS),
            enabled=True,
            velocity_cost=0.002,
            position_error_limit=0.003,
            linear_slow=0.2,
            linear_fast=0.5,
        ),
        Candidate(
            "low_direct_c0p04",
            tuple(LOW_CAPS),
            enabled=True,
            velocity_cost=0.04,
        ),
        Candidate(
            "low_direct_c0p08",
            tuple(LOW_CAPS),
            enabled=True,
            velocity_cost=0.08,
        ),
        Candidate(
            "low_direct_c0p12",
            tuple(LOW_CAPS),
            enabled=True,
            velocity_cost=0.12,
        ),
        Candidate(
            "low_projected_c0p12",
            tuple(LOW_CAPS),
            enabled=True,
            projection="exact-nullspace",
            velocity_cost=0.12,
        ),
        Candidate(
            "low_direct_c0p08_corr0p04",
            tuple(LOW_CAPS),
            enabled=True,
            velocity_cost=0.08,
            corridor_cost=0.04,
        ),
    ]
    return base[:7] if quick else base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dev/results/speed_scheduled_elbow_task"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the three shortest scenarios and a reduced candidate set.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        help="Run only the named candidate configurations.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    initial_kinematics = _make_kinematics(Candidate("initial", tuple(LOW_CAPS)))
    initial_pose = initial_kinematics.fk("right", INITIAL_RIGHT)
    scenarios = [
        _reach_retract(initial_pose, 0.3),
        _reach_retract(initial_pose, 0.6),
        _circle(initial_pose, 0.4),
        _circle(initial_pose, 0.8),
        _wrist_flip(initial_pose),
    ]
    if args.quick:
        scenarios = [scenarios[1], scenarios[3], scenarios[4]]

    candidates = _candidates(args.quick)
    if args.candidates:
        selected_candidates = set(args.candidates)
        candidates = [
            candidate
            for candidate in _candidates(False)
            if candidate.name in selected_candidates
        ]
        missing = selected_candidates - {candidate.name for candidate in candidates}
        if missing:
            parser.error(f"Unknown candidates: {', '.join(sorted(missing))}")
        if "raised_reference" not in selected_candidates:
            parser.error("--candidates must include raised_reference.")
    rows: list[Metrics] = []
    parameters = {candidate.name: asdict(candidate) for candidate in candidates}
    for scenario in scenarios:
        traces: dict[str, Trace] = {}
        for candidate in candidates:
            print(f"{scenario.name}: {candidate.name}", flush=True)
            trace = simulate(candidate, scenario)
            traces[candidate.name] = trace
            _save_trace(
                args.output / f"trace_{scenario.name}_{candidate.name}.npz",
                trace,
            )
        raised = traces["raised_reference"]
        rows.extend(
            compute_metrics(candidate, scenario, traces[candidate.name], raised)
            for candidate in candidates
        )

    with (args.output / "metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    with (args.output / "parameters.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(parameters, stream, indent=2)

    for row in rows:
        print(
            f"{row.scenario:22s} {row.candidate:30s} "
            f"elbow={100.0 * row.actual_elbow_path_rmse_to_raised_m:5.2f}cm "
            f"pos={100.0 * row.actual_position_rmse_m:5.2f}cm "
            f"ddq99={row.command_accel_p99_rad_s2:6.1f} "
            f"solve95={row.solve_p95_ms:5.2f}ms "
            f"fail={row.failure_count}"
        )


if __name__ == "__main__":
    main()
