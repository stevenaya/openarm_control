#!/usr/bin/env python3
"""Plot saved outputs from sim_posture_mode_comparison.py.

This script intentionally depends only on NumPy and Matplotlib, so plotting can
run in a different environment from the MuJoCo/Mink simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASK_MODES = (
    "home_full_0p01",
    "home_full_0p10",
    "nullspace_rate_full",
    "nullspace_direct_0p10_full",
    "nullspace_direct_1p00_full",
    "nullspace_direct_full",
)
TASK_LABELS = {
    "home_full_0p01": "home posture 0.01",
    "home_full_0p10": "home posture 0.10",
    "nullspace_rate_full": "nullspace rate 0.8 / max 0.8",
    "nullspace_direct_0p10_full": "direct nullspace cost 0.10",
    "nullspace_direct_1p00_full": "direct nullspace cost 1.0",
    "nullspace_direct_full": "direct nullspace cost 10",
}
SPEEDS = (0.05, 0.1, 0.2, 0.4)
RATE_TUNING = (
    "nullspace_rate_0p8_max0p8_full",
    "nullspace_rate_1p6_max0p8_full",
    "nullspace_rate_2p0_max1p2_full",
    "nullspace_rate_3p0_max1p2_full",
)


def _speed_tag(speed: float) -> str:
    return f"{speed:.3f}".replace(".", "p")


def _load(root: Path, speed: float, mode: str) -> np.lib.npyio.NpzFile:
    return np.load(root / f"trace_{_speed_tag(speed)}_{mode}.npz")


def _load_tuning(
    root: Path,
    speed: float,
    mode: str,
) -> np.lib.npyio.NpzFile:
    tag = str(speed).replace(".", "p")
    return np.load(root / f"trace_tuning_{tag}_{mode}.npz")


def _orientation_error(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    alignment = np.abs(np.sum(target * actual, axis=1))
    return 2.0 * np.arccos(np.clip(alignment, -1.0, 1.0))


def _phase_lines(axes: np.ndarray, trace: np.lib.npyio.NpzFile) -> None:
    changes = np.flatnonzero(trace["phase"][1:] != trace["phase"][:-1]) + 1
    for axis in axes.flat:
        for index in changes:
            axis.axvline(trace["time"][index], color="0.8", linewidth=0.7)
        axis.grid(alpha=0.2)


def plot_task_modes(root: Path, speed: float) -> None:
    traces = {mode: _load(root, speed, mode) for mode in TASK_MODES}
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for mode, trace in traces.items():
        label = TASK_LABELS[mode]
        position_error = np.linalg.norm(
            trace["target_pose"][:, :3] - trace["command_ee"][:, :3],
            axis=1,
        )
        orientation_error = _orientation_error(
            trace["target_pose"][:, 3:],
            trace["command_ee"][:, 3:],
        )
        axes[0, 0].plot(
            trace["time"],
            np.abs(trace["geometric_home_error"]),
            label=label,
        )
        axes[0, 1].plot(
            trace["time"],
            trace["geometric_nullspace_speed"],
            label=label,
        )
        axes[1, 0].plot(trace["time"], position_error, label=label)
        axes[1, 1].plot(trace["time"], orientation_error, label=label)
        axes[2, 0].plot(trace["time"], trace["command_q"][:, 3], label=label)
        axes[2, 1].plot(trace["time"], trace["command_rho"], label=label)

    axes[0, 0].set_ylabel("|nullspace home error| [rad]")
    axes[0, 1].set_ylabel("nullspace speed [rad/s]")
    axes[1, 0].set_ylabel("command position error [m]")
    axes[1, 1].set_ylabel("command orientation error [rad]")
    axes[2, 0].set_ylabel("command q4 [rad]")
    axes[2, 1].set_ylabel("geometric singularity ratio")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 1].set_xlabel("time [s]")
    axes[0, 0].legend(fontsize=8, ncol=2)
    _phase_lines(axes, next(iter(traces.values())))
    fig.suptitle(
        "Posture objective comparison from a 0.6 rad nullspace offset\n"
        f"retract peak speed = {speed:.2f} m/s"
    )
    fig.tight_layout()
    fig.savefig(root / f"task_modes_{_speed_tag(speed)}.png", dpi=170)
    plt.close(fig)


def _retract_progress(trace: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    mask = trace["phase"] == 3
    target_x = trace["target_pose"][mask, 0]
    progress = (np.max(target_x) - target_x) / (
        np.max(target_x) - np.min(target_x)
    )
    return mask, progress


def plot_retract_speeds(root: Path) -> None:
    traces = {
        speed: np.load(
            root
            / (
                f"trace_force_{str(speed).replace('.', 'p')}_"
                "nullspace_rate_full.npz"
            )
        )
        for speed in SPEEDS
    }
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for speed, trace in traces.items():
        mask, progress = _retract_progress(trace)
        label = f"{speed:.2f} m/s"
        axes[0, 0].plot(
            progress,
            trace["command_q"][mask, 3],
            label=f"{label} command",
        )
        axes[0, 0].plot(
            progress,
            trace["actual_q"][mask, 3],
            linestyle="--",
            label=f"{label} actual",
        )
        axes[0, 1].plot(
            progress,
            trace["command_q"][mask, 0],
            label=f"{label} command",
        )
        axes[0, 1].plot(
            progress,
            trace["actual_q"][mask, 0],
            linestyle="--",
            label=f"{label} actual",
        )
        axes[1, 0].plot(
            progress,
            trace["command_dq"][mask, 3],
            label=label,
        )
        axes[1, 1].plot(
            progress,
            trace["command_q"][mask, 3] - trace["actual_q"][mask, 3],
            label=label,
        )
        axes[2, 0].plot(
            progress,
            trace["actuator_force"][mask, 3],
            label=label,
        )
        axes[2, 1].plot(
            progress,
            trace["actual_ee"][mask, 2] - trace["actual_ee"][mask, 2][0],
            label=label,
        )

    axes[0, 0].set_ylabel("q4 [rad]")
    axes[0, 1].set_ylabel("q1 [rad]")
    axes[1, 0].set_ylabel("command dq4 [rad/s]")
    axes[1, 1].set_ylabel("command - actual q4 [rad]")
    axes[2, 0].set_ylabel("q4 actuator force [Nm]")
    axes[2, 1].set_ylabel("actual EE z change [m]")
    axes[2, 0].set_xlabel("retract handle progress")
    axes[2, 1].set_xlabel("retract handle progress")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("Current nullspace mode: retract-speed comparison")
    fig.tight_layout()
    fig.savefig(root / "retract_speed_comparison.png", dpi=170)
    plt.close(fig)


def plot_rate_tuning(root: Path, speed: float) -> None:
    traces = {
        mode: _load_tuning(root, speed, mode) for mode in RATE_TUNING
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for mode, trace in traces.items():
        label = mode.removeprefix("nullspace_rate_").removesuffix("_full")
        position_error = np.linalg.norm(
            trace["target_pose"][:, :3] - trace["command_ee"][:, :3],
            axis=1,
        )
        axes[0, 0].plot(
            trace["time"],
            np.abs(trace["geometric_home_error"]),
            label=label,
        )
        axes[0, 1].plot(
            trace["time"],
            trace["geometric_nullspace_speed"],
            label=label,
        )
        axes[1, 0].plot(trace["time"], position_error, label=label)
        axes[1, 1].plot(
            trace["time"],
            np.max(np.abs(trace["command_dq"]), axis=1),
            label=label,
        )

    axes[0, 0].set_ylabel("|nullspace home error| [rad]")
    axes[0, 1].set_ylabel("nullspace speed [rad/s]")
    axes[1, 0].set_ylabel("command position error [m]")
    axes[1, 1].set_ylabel("max joint |dq| [rad/s]")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 1].set_xlabel("time [s]")
    axes[0, 0].legend(fontsize=8)
    _phase_lines(axes, next(iter(traces.values())))
    fig.suptitle(
        "Rate-limited nullspace regulation tuning\n"
        f"retract peak speed = {speed:.2f} m/s"
    )
    fig.tight_layout()
    fig.savefig(root / f"rate_tuning_{_speed_tag(speed)}.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--task-speed", type=float, default=0.1)
    args = parser.parse_args()

    plot_task_modes(args.results, args.task_speed)
    plot_retract_speeds(args.results)
    plot_rate_tuning(args.results, args.task_speed)
    print(f"Wrote plots to {args.results}")


if __name__ == "__main__":
    main()
