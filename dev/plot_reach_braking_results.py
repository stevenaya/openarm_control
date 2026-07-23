#!/usr/bin/env python3
"""Plot the saved reach-braking experiment without loading the robot model."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROFILES = (
    ("none", "none"),
    ("acceleration_20", "acceleration, a=20"),
    ("distance_0p50", "distance, d=0.50"),
    ("distance_0p70", "distance, d=0.70"),
    ("distance_1p00", "distance, d=1.00"),
)
COLORS = {
    "none": "#c23b22",
    "acceleration_20": "#df8f00",
    "distance_0p50": "#2878b5",
    "distance_0p70": "#38a169",
    "distance_1p00": "#775da6",
}


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty trace: {path}")
    columns = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        try:
            columns[key] = np.asarray(values, dtype=float)
        except ValueError:
            columns[key] = np.asarray(values, dtype=str)
    return columns


def trace_path(result_dir: Path, profile: str, speed: float) -> Path:
    speed_key = f"{speed:.3f}".replace(".", "p")
    return result_dir / f"trace_{speed_key}_{profile}.csv"


def plot_lines(
    axes: Iterable[plt.Axes],
    result_dir: Path,
    speed: float,
) -> None:
    ax_q1, ax_dq1, ax_q4, ax_ee = axes
    for profile, label in PROFILES:
        data = load_csv(trace_path(result_dir, profile, speed))
        t = data["time_s"]
        color = COLORS[profile]
        ax_q1.plot(t, data["command_q1"], color=color, linestyle="--", alpha=0.7)
        ax_q1.plot(t, data["actual_q1"], color=color, label=label)
        ax_dq1.plot(
            t,
            data["command_dq1"],
            color=color,
            linestyle="--",
            alpha=0.7,
        )
        ax_dq1.plot(t, data["actual_dq1"], color=color)
        ax_q4.plot(t, data["command_q4"], color=color, linestyle="--", alpha=0.7)
        ax_q4.plot(t, data["actual_q4"], color=color)
        ax_ee.plot(
            t,
            1e3 * (data["command_ee_z"] - data["target_z"]),
            color=color,
            linestyle="--",
            alpha=0.7,
        )
        ax_ee.plot(
            t,
            1e3 * (data["actual_ee_z"] - data["target_z"]),
            color=color,
        )

    ax_q1.set_ylabel("shoulder q1 [rad]")
    ax_dq1.set_ylabel("shoulder dq1 [rad/s]")
    ax_q4.set_ylabel("elbow q4 [rad]")
    ax_ee.set_ylabel("EE vertical error [mm]")
    ax_ee.set_xlabel("time [s]")
    ax_q4.axhline(0.08, color="#333333", linewidth=0.8, linestyle=":")
    ax_q4.axhline(0.0, color="#333333", linewidth=0.8)
    ax_q1.legend(ncol=3, fontsize=8, loc="best")
    for axis in (ax_q1, ax_dq1, ax_q4, ax_ee):
        axis.grid(alpha=0.25)


def make_trace_plot(result_dir: Path, speed: float) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    plot_lines(axes, result_dir, speed)
    figure.suptitle(
        f"Shoulder-height reach beyond workspace, target speed={speed:.2f} m/s\n"
        "solid: simulated actual, dashed: command"
    )
    figure.tight_layout()
    speed_key = f"{speed:.2f}".replace(".", "p")
    figure.savefig(
        result_dir / f"comparison_speed_{speed_key}.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def make_summary_plot(result_dir: Path) -> None:
    summary = load_csv(result_dir / "summary.csv")
    speeds = sorted(set(summary["target_speed_m_s"]))
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = (
        ("near_max_actual_dq1_rad_s", "peak actual |dq1| [rad/s]"),
        ("near_max_actual_ddq1_rad_s2", "peak actual |ddq1| [rad/s²]"),
        ("min_actual_q4_rad", "minimum actual q4 [rad]"),
        ("tail_ee_position_p2p_m", "tail EE position p-p [m]"),
    )
    profile_names = np.asarray(summary["profile"], dtype=str)
    speed_values = summary["target_speed_m_s"]
    for axis, (metric, ylabel) in zip(axes.flat, metrics, strict=True):
        for profile, label in PROFILES:
            values = []
            for speed in speeds:
                index = np.flatnonzero(
                    (profile_names == profile) & np.isclose(speed_values, speed)
                )[0]
                if metric == "tail_ee_position_p2p_m":
                    value = np.hypot(
                        summary["tail_ee_x_p2p_m"][index],
                        summary["tail_ee_z_p2p_m"][index],
                    )
                else:
                    value = summary[metric][index]
                values.append(value)
            axis.plot(
                speeds,
                values,
                marker="o",
                color=COLORS[profile],
                label=label,
            )
        axis.set_xlabel("target speed [m/s]")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Reach-braking profile comparison")
    figure.tight_layout()
    figure.savefig(
        result_dir / "summary_metrics.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    for speed in (0.05, 0.10, 0.20):
        make_trace_plot(args.result_dir, speed)
    make_summary_plot(args.result_dir)


if __name__ == "__main__":
    main()
