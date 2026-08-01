#!/usr/bin/env python3
"""Create final-report figures for targeted nullspace parameter tuning."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RESULTS = (
    Path(__file__).resolve().parent
    / "results"
    / "current_pr_20260730"
)
REPORT = ROOT / "note" / "openarm_control" / "final_report"
ASSETS = REPORT / "assets"
TABLES = REPORT / "tables"


def decode(value: str) -> float:
    return float(value.replace("p", "."))


def tuning_figure(summary: pd.DataFrame) -> None:
    cost = summary[summary.profile.str.match(r"ns_cost_[0-9]")].copy()
    cost["cost"] = cost.profile.str.removeprefix("ns_cost_").map(decode)
    cost = cost.sort_values("cost")
    baseline = cost[cost.cost == 7.0].iloc[0]

    cap = summary[summary.profile.str.startswith("ns_cap_")].copy()
    parsed = cap.profile.str.extract(r"ns_cap_c([0-9p]+)_v([0-9p]+)")
    cap["cost"] = parsed[0].map(decode)
    cap["vmax"] = parsed[1].map(decode)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.7), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(
        cost.cost,
        cost.actual_swivel_max_rad,
        marker="o",
        linewidth=2,
        color="#1f77b4",
    )
    axis.axvline(7.0, color="#2ca02c", linestyle="--", label="current cost = 7")
    axis.set(
        title="Higher cost does not monotonically stabilize the elbow",
        xlabel="Nullspace cost",
        ylabel="Max elbow swivel departure [rad]",
    )
    axis.legend()
    axis.grid(alpha=0.25)

    axis = axes[0, 1]
    position_line = axis.plot(
        cost.cost,
        100.0 * cost.actual_position_rmse_m,
        marker="o",
        color="#1f77b4",
        label="position RMSE [cm]",
    )[0]
    orientation_axis = axis.twinx()
    orientation_line = orientation_axis.plot(
        cost.cost,
        cost.actual_orientation_rmse_rad,
        marker="s",
        color="#ff7f0e",
        label="orientation RMSE [rad]",
    )[0]
    axis.axvline(7.0, color="#2ca02c", linestyle="--")
    axis.set(
        title="Tracking trade-off on deep retraction",
        xlabel="Nullspace cost",
        ylabel="Position RMSE [cm]",
    )
    orientation_axis.set_ylabel("Orientation RMSE [rad]")
    axis.legend(
        [position_line, orientation_line],
        [position_line.get_label(), orientation_line.get_label()],
        loc="center right",
    )
    axis.grid(alpha=0.25)

    axis = axes[1, 0]
    for metric, label, marker in (
        ("actual_ddq_p99_rad_s2", "joint acceleration p99", "o"),
        ("actual_elbow_accel_p99_m_s2", "elbow acceleration p99", "s"),
        (
            "command_velocity_saturation_fraction",
            "IK velocity saturation",
            "^",
        ),
    ):
        axis.plot(
            cost.cost,
            100.0 * cost[metric] / baseline[metric],
            marker=marker,
            label=label,
        )
    axis.axhline(100.0, color="#2ca02c", linestyle="--", label="current")
    axis.set(
        title="Dynamic cost relative to current cost = 7",
        xlabel="Nullspace cost",
        ylabel="Relative metric [%]",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)

    axis = axes[1, 1]
    for cost_value, group in cap.groupby("cost"):
        group = group.sort_values("vmax")
        axis.plot(
            group.vmax,
            group.actual_swivel_max_rad,
            marker="o",
            label=f"cost {cost_value:g}",
        )
    axis.scatter(
        [1.0],
        [baseline.actual_swivel_max_rad],
        marker="*",
        s=150,
        color="#2ca02c",
        label="current: 7 / 1.0",
        zorder=5,
    )
    axis.set(
        title="Capping aggressive nullspace return is more effective",
        xlabel="Nullspace max speed [rad/s]",
        ylabel="Max elbow swivel departure [rad]",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)

    fig.suptitle("Targeted nullspace tuning: deep diagonal retract", fontsize=15)
    fig.savefig(ASSETS / "17_nullspace_targeted_tuning.png", dpi=190)
    plt.close(fig)

    cost.to_csv(TABLES / "nullspace_deep_cost_sweep.csv", index=False)
    cap.to_csv(TABLES / "nullspace_deep_speed_cap_sweep.csv", index=False)


def validation_figure(summary: pd.DataFrame) -> None:
    keys = ["scenario", "side"]
    baseline = summary[summary.profile == "current_c7_r1p6"].set_index(keys)
    profiles = (
        ("cost_only_c7p5_v1p0", "cost 7.5 only"),
        ("candidate_c7p5_v0p6", "cost 7.5 / cap 0.6"),
        ("candidate_c7p5_v0p7", "cost 7.5 / cap 0.7"),
        ("candidate_c7p5_v0p8", "cost 7.5 / cap 0.8"),
        ("aggressive_c9_r0p8", "cost 9 / return 0.8"),
    )
    metrics = (
        ("actual_swivel_max_rad", "elbow swivel"),
        ("actual_position_rmse_m", "position RMSE"),
        ("actual_orientation_rmse_rad", "orientation RMSE"),
        ("actual_ddq_p99_rad_s2", "joint accel p99"),
        ("actual_elbow_accel_p99_m_s2", "elbow accel p99"),
        ("tail_actual_ee_vibration_rms_m", "tail vibration"),
    )
    aggregate_rows: list[dict[str, float | str]] = []
    scenario_rows: list[dict[str, float | str]] = []
    for profile, label in profiles:
        values = summary[summary.profile == profile].set_index(keys).loc[
            baseline.index
        ]
        aggregate: dict[str, float | str] = {
            "profile": profile,
            "label": label,
        }
        for metric, _ in metrics:
            aggregate[metric] = float(
                100.0 * np.mean(values[metric] / baseline[metric] - 1.0)
            )
        aggregate_rows.append(aggregate)
        for index in baseline.index:
            scenario_rows.append(
                {
                    "profile": profile,
                    "label": label,
                    "scenario": index[0],
                    "side": index[1],
                    "swivel_delta_percent": float(
                        100.0
                        * (
                            values.loc[index, "actual_swivel_max_rad"]
                            / baseline.loc[index, "actual_swivel_max_rad"]
                            - 1.0
                        )
                    ),
                    "elbow_accel_delta_percent": float(
                        100.0
                        * (
                            values.loc[index, "actual_elbow_accel_p99_m_s2"]
                            / baseline.loc[
                                index,
                                "actual_elbow_accel_p99_m_s2",
                            ]
                            - 1.0
                        )
                    ),
                }
            )

    aggregate_frame = pd.DataFrame(aggregate_rows)
    matrix = aggregate_frame[[metric for metric, _ in metrics]].to_numpy()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.5, 10.0),
        constrained_layout=True,
        height_ratios=(1.0, 1.35),
    )
    image = axes[0].imshow(
        matrix,
        cmap="RdYlGn_r",
        vmin=-20.0,
        vmax=20.0,
        aspect="auto",
    )
    axes[0].set_xticks(
        range(len(metrics)),
        [label for _, label in metrics],
        rotation=20,
        ha="right",
    )
    axes[0].set_yticks(
        range(len(profiles)),
        [label for _, label in profiles],
    )
    axes[0].set_title("Mean change across 12 cross-validation trajectories")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[0].text(
                column,
                row,
                f"{matrix[row, column]:+.1f}%",
                ha="center",
                va="center",
                fontsize=9,
            )
    fig.colorbar(image, ax=axes[0], label="Change from current defaults [%]")

    scenario_frame = pd.DataFrame(scenario_rows)
    selected = scenario_frame[
        scenario_frame.profile == "candidate_c7p5_v0p6"
    ].copy()
    selected["case"] = selected.scenario.str.replace(
        "_right",
        "",
        regex=False,
    )
    selected.loc[selected.side == "left", "case"] += " [left]"
    y = np.arange(len(selected))
    axes[1].barh(
        y - 0.18,
        selected.swivel_delta_percent,
        height=0.35,
        label="elbow swivel",
        color="#4c78a8",
    )
    axes[1].barh(
        y + 0.18,
        selected.elbow_accel_delta_percent,
        height=0.35,
        label="elbow acceleration p99",
        color="#f58518",
    )
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_yticks(y, selected.case)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Change from current defaults [%]")
    axes[1].set_title(
        "The best stress candidate is not uniformly better "
        "(cost 7.5 / cap 0.6)"
    )
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("Nullspace parameter cross-validation", fontsize=15)
    fig.savefig(ASSETS / "18_nullspace_cross_validation.png", dpi=190)
    plt.close(fig)

    aggregate_frame.to_csv(
        TABLES / "nullspace_cross_validation_means.csv",
        index=False,
    )
    scenario_frame.to_csv(
        TABLES / "nullspace_cross_validation_by_scenario.csv",
        index=False,
    )


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    tuning = pd.read_csv(
        RESULTS / "nullspace_tuning" / "summary_with_swivel.csv"
    )
    validation = pd.read_csv(
        RESULTS
        / "nullspace_candidate_validation"
        / "summary_with_swivel.csv"
    )
    tuning_figure(tuning)
    validation_figure(validation)


if __name__ == "__main__":
    main()
