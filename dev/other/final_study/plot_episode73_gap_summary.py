#!/usr/bin/env python3
"""Plot the episode-73 tracking, plant-replay, and weak-direction summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULT_ROOT = HERE / "results"
GAP_DIR = RESULT_ROOT / "episode73_tracking_gap_20260731"
REPLAY_DIR = RESULT_ROOT / "current_pr_20260730" / "recorded_replay"
OUTPUT = GAP_DIR / "episode73_gap_summary.png"


def _tracking_percentiles(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    percentiles = (50, 95, 99, 100)
    return {
        "same-time": 100.0
        * np.percentile(data["temporal_error_m"], percentiles),
        "cross-track": 100.0
        * np.percentile(data["cross_track_error_m"], percentiles),
    }


def main() -> None:
    right = _tracking_percentiles(GAP_DIR / "episode73_right_tracking.npz")
    left = _tracking_percentiles(GAP_DIR / "episode73_left_tracking.npz")
    plant = pd.read_csv(GAP_DIR / "plant" / "profile_metrics.csv")
    weak = pd.read_csv(REPLAY_DIR / "weak_direction_decomposition.csv")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    labels = ("p50", "p95", "p99", "max")
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.19
    series = (
        ("right same-time", right["same-time"], "#d1495b"),
        ("right cross-track", right["cross-track"], "#f79256"),
        ("left same-time", left["same-time"], "#00798c"),
        ("left cross-track", left["cross-track"], "#76b7b2"),
    )
    for index, (name, values, color) in enumerate(series):
        axes[0].bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=name,
            color=color,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("position error [cm]")
    axes[0].set_title("A. Lag is not the same as leaving the path")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.22)

    profiles = (
        "nominal_no_gc",
        "no_driver_cap",
        "delay_20ms",
        "delay_40ms",
        "gravity_compensation",
    )
    profile_labels = (
        "nominal",
        "no driver\ncap",
        "20 ms\ndelay",
        "40 ms\ndelay",
        "gravity\ncomp.",
    )
    right_plant = (
        plant[(plant["side"] == "right")]
        .set_index("name")
        .loc[list(profiles)]
    )
    px = np.arange(len(profiles), dtype=np.float64)
    axes[1].bar(
        px - 0.18,
        1000.0 * right_plant["position_rmse_m"],
        0.36,
        color="#4e79a7",
        label="same-time",
    )
    axes[1].bar(
        px + 0.18,
        1000.0 * right_plant["position_rmse_aligned_m"],
        0.36,
        color="#a0cbe8",
        label="time-aligned",
    )
    axes[1].set_xticks(px, profile_labels)
    axes[1].set_ylabel("real-vs-sim EEF RMSE [mm]")
    axes[1].set_title("B. Same command: MuJoCo matches the real plant")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.22)

    scenarios = (
        "recorded_ep73_right_chest40_position_smoothed",
        "recorded_ep73_right_chest50_position_smoothed",
        "recorded_ep73_right_chest85_position_smoothed",
    )
    weak_recorded = (
        weak[weak["profile"] == "recorded_hardware"]
        .set_index("scenario")
        .loc[list(scenarios)]
    )
    sx = np.arange(len(scenarios), dtype=np.float64)
    exact = 100.0 * weak_recorded["exact_z_energy_fraction"].to_numpy()
    near = 100.0 * weak_recorded["near_weak_energy_fraction"].to_numpy()
    other = 100.0 * weak_recorded["other_energy_fraction"].to_numpy()
    axes[2].bar(sx, exact, color="#59a14f", label="exact nullspace z")
    axes[2].bar(
        sx,
        near,
        bottom=exact,
        color="#f28e2b",
        label="near-weak V[-2]",
    )
    axes[2].bar(
        sx,
        other,
        bottom=exact + near,
        color="#bab0ac",
        label="other task directions",
    )
    axes[2].set_xticks(sx, ("event 40", "event 50", "event 85"))
    axes[2].set_ylim(0.0, 100.0)
    axes[2].set_ylabel("joint-velocity energy [%]")
    axes[2].set_title("C. Large chest motion is not exact-z drift")
    axes[2].legend(fontsize=8, loc="lower right")
    axes[2].grid(axis="y", alpha=0.22)

    fig.suptitle(
        "Episode 73: command-side excursion, plant lag, and weak directions",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=190)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
