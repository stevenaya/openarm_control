#!/usr/bin/env python3
"""Plot focused comparisons for the latest recorded intervention study."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recorded_intervention_study as recorded

METRICS_PATH = recorded.REPLAY_DIR / "metrics.csv"
ASSET_DIR = (
    HERE.parents[2]
    / "note"
    / "openarm_control"
    / "final_report"
    / "assets"
)

COLORS = {
    "pr_full_recorded": "#2878B5",
    "mainline_ik_velocity_recorded": "#E07A2D",
    "mainline_recorded": "#7A7A7A",
    "pr_driver_velocity_only_recorded": "#52A675",
    "pr_nullspace_cost_10_recorded": "#8D67B5",
    "pr_nullspace_cost_12_recorded": "#C65B7C",
    "pr_no_frame_error_recorded": "#F0A34A",
    "pr_no_nullspace_recorded": "#D14B4B",
}
LABELS = {
    "pr_full_recorded": "Current PR",
    "mainline_ik_velocity_recorded": "Mainline + IK cap",
    "mainline_recorded": "Mainline",
    "pr_driver_velocity_only_recorded": "PR, IK cap off",
    "pr_nullspace_cost_10_recorded": "Nullspace cost 10",
    "pr_nullspace_cost_12_recorded": "Nullspace cost 12",
    "pr_no_frame_error_recorded": "No frame-error bound",
    "pr_no_nullspace_recorded": "No nullspace task",
}
SCENARIO_LABELS = {
    "recorded_ep73_right_chest62_position_smoothed": "Chest 62",
    "recorded_ep73_right_chest44_position_smoothed": "Chest 44",
    "recorded_ep75_right_retract05_straight_retract": "Retract 5",
    "recorded_ep75_right_retract41_straight_retract": "Retract 41",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _grouped_bars(
    axis: plt.Axes,
    data: pd.DataFrame,
    *,
    profiles: list[str],
    scenarios: list[str],
    metric: str,
    scale: float,
    ylabel: str,
    title: str,
) -> None:
    x = np.arange(len(scenarios), dtype=np.float64)
    width = 0.78 / len(profiles)
    offsets = (np.arange(len(profiles)) - (len(profiles) - 1) / 2) * width
    for offset, profile in zip(offsets, profiles, strict=True):
        values = np.asarray(
            [
                data.loc[
                    (data["profile"] == profile)
                    & (data["scenario"] == scenario),
                    metric,
                ].iloc[0]
                * scale
                for scenario in scenarios
            ]
        )
        bars = axis.bar(
            x + offset,
            values,
            width=width * 0.92,
            color=COLORS[profile],
            label=LABELS[profile],
        )
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.2)
    axis.set_xticks(x, [SCENARIO_LABELS[item] for item in scenarios])
    axis.set_ylabel(ylabel)
    axis.set_title(title)


def plot_chest_comparison(data: pd.DataFrame) -> None:
    profiles = [
        "pr_full_recorded",
        "mainline_ik_velocity_recorded",
        "mainline_recorded",
    ]
    scenarios = [
        "recorded_ep73_right_chest62_position_smoothed",
        "recorded_ep73_right_chest44_position_smoothed",
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))
    _grouped_bars(
        axes[0],
        data,
        profiles=profiles,
        scenarios=scenarios,
        metric="actual_position_max_m",
        scale=100.0,
        ylabel="Max actual position error (cm)",
        title="Position preservation",
    )
    _grouped_bars(
        axes[1],
        data,
        profiles=profiles,
        scenarios=scenarios,
        metric="actual_elbow_lateral_range_m",
        scale=100.0,
        ylabel="Actual elbow lateral range (cm)",
        title="Elbow branch motion",
    )
    _grouped_bars(
        axes[2],
        data,
        profiles=profiles,
        scenarios=scenarios,
        metric="actual_elbow_accel_p99_m_s2",
        scale=1.0,
        ylabel="Actual elbow acceleration p99 (m/s²)",
        title="Elbow transient",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    figure.suptitle(
        "Recorded chest flip intent proxies: current PR vs mainline",
        y=1.12,
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.02,
        "Raw VR targets were not recorded. Position-smoothed FK(action q) is "
        "used as the shared intent proxy; recorded rapid orientation is retained.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    figure.tight_layout()
    figure.savefig(
        ASSET_DIR / "19_recorded_chest_controller_comparison.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_retract_tradeoff(data: pd.DataFrame) -> None:
    profiles = [
        "pr_full_recorded",
        "mainline_ik_velocity_recorded",
        "mainline_recorded",
        "pr_driver_velocity_only_recorded",
    ]
    scenarios = [
        "recorded_ep75_right_retract05_straight_retract",
        "recorded_ep75_right_retract41_straight_retract",
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.0))
    _grouped_bars(
        axes[0, 0],
        data,
        profiles=profiles,
        scenarios=scenarios,
        metric="actual_position_max_m",
        scale=100.0,
        ylabel="Max actual position error (cm)",
        title="Driver-following error",
    )
    _grouped_bars(
        axes[0, 1],
        data,
        profiles=profiles,
        scenarios=scenarios,
        metric="actual_elbow_lateral_range_m",
        scale=100.0,
        ylabel="Actual elbow lateral range (cm)",
        title="Redundancy branch motion",
    )
    _grouped_bars(
        axes[1, 0],
        data,
        profiles=profiles,
        scenarios=scenarios,
        metric="command_j1_dq_max_rad_s",
        scale=1.0,
        ylabel="Command |J1 velocity| max (rad/s)",
        title="J1 demand before driver",
    )
    axes[1, 0].axhline(
        2.0,
        color="#B00020",
        linestyle="--",
        linewidth=1.2,
        label="2 rad/s physical cap",
    )
    _grouped_bars(
        axes[1, 1],
        data,
        profiles=profiles,
        scenarios=scenarios,
        metric="command_j1_sat_fraction",
        scale=100.0,
        ylabel="J1 IK-cap occupancy (%)",
        title="Hard-cap activity",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
    )
    figure.suptitle(
        "Recorded retract proxies: IK hard cap trades elbow branch for lag",
        y=1.07,
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(
        ASSET_DIR / "20_recorded_retract_velocity_tradeoff.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_retract_clearance() -> None:
    lookup = recorded._event_lookup(recorded.ANALYSIS_DIR)
    event = lookup[("retract", "right", 5)]
    record = recorded.load_record(
        event.episode,
        event.side,
        recorded.ANALYSIS_DIR / "cache",
    )
    clearance = np.load(
        recorded.ANALYSIS_DIR
        / "episode_75_right_table_clearance.npz",
        allow_pickle=False,
    )
    selected = slice(event.core_start, event.core_end)
    time = record.time[selected] - record.time[event.core_start]
    count = event.core_end - event.core_start
    chord = np.linspace(
        record.action_world_eef[event.core_start, 2],
        record.action_world_eef[event.core_end - 1, 2],
        count,
    )
    action_relative_z = (
        record.action_world_eef[selected, 2] - chord
    ) * 100.0
    obs_relative_z = (record.obs_world_eef[selected, 2] - chord) * 100.0
    action_dq = recorded._derivative(record.action_q)

    figure, axes = plt.subplots(3, 1, figsize=(11.2, 8.7), sharex=True)
    axes[0].plot(
        time,
        action_relative_z,
        color="#2878B5",
        linewidth=2.0,
        label="Recorded IK command",
    )
    axes[0].plot(
        time,
        obs_relative_z,
        color="#E07A2D",
        linewidth=1.8,
        label="Measured state",
    )
    axes[0].axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("EEF z vs endpoint chord (cm)")
    axes[0].set_title(
        "The downward path is already present in the recorded command"
    )
    axes[0].legend(frameon=False, loc="lower left")

    action_clearance = (
        clearance["action_vertical_clearance_m"][selected] * 100.0
    )
    obs_clearance = clearance["obs_vertical_clearance_m"][selected] * 100.0
    axes[1].plot(
        time,
        action_clearance,
        color="#2878B5",
        linewidth=2.0,
        label="Command collision mesh",
    )
    axes[1].plot(
        time,
        obs_clearance,
        color="#E07A2D",
        linewidth=1.8,
        label="Measured collision mesh",
    )
    axes[1].axhline(
        0.0,
        color="#B00020",
        linestyle="--",
        linewidth=1.2,
        label="Tabletop contact",
    )
    axes[1].fill_between(
        time,
        obs_clearance,
        0.0,
        where=obs_clearance < 0.0,
        color="#D43F3A",
        alpha=0.18,
    )
    axes[1].set_ylabel("Gripper vertical clearance (cm)")
    axes[1].set_title(
        "Command reaches 0.91 cm clearance; measured mesh penetrates 1.38 cm"
    )
    axes[1].legend(frameon=False, loc="upper right")

    for joint, color in ((0, "#2878B5"), (1, "#52A675")):
        axes[2].plot(
            time,
            action_dq[selected, joint],
            color=color,
            linewidth=1.8,
            label=f"Command J{joint + 1}",
        )
        axes[2].plot(
            time,
            record.obs_dq[selected, joint],
            color=color,
            linewidth=1.0,
            linestyle=":",
            alpha=0.9,
            label=f"Measured J{joint + 1}",
        )
    axes[2].axhline(2.0, color="#B00020", linestyle="--", linewidth=1.0)
    axes[2].axhline(-2.0, color="#B00020", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("Joint velocity (rad/s)")
    axes[2].set_xlabel("Time from retract onset (s)")
    axes[2].set_title("J1/J2 velocity demand during the retract")
    axes[2].legend(frameon=False, ncol=4, loc="upper center")

    figure.suptitle(
        "Recorded episode 75, retract case 5: downward collision risk",
        y=1.01,
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(
        ASSET_DIR / "21_recorded_retract_table_clearance.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_nullspace_cost(data: pd.DataFrame) -> None:
    profiles = [
        "pr_full_recorded",
        "pr_nullspace_cost_10_recorded",
        "pr_nullspace_cost_12_recorded",
    ]
    scenarios = list(SCENARIO_LABELS)
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.0))
    metrics = (
        (
            "actual_elbow_lateral_range_m",
            100.0,
            "Elbow lateral range (cm)",
            "Elbow branch",
        ),
        (
            "actual_elbow_accel_p99_m_s2",
            1.0,
            "Elbow acceleration p99 (m/s²)",
            "Elbow transient",
        ),
        (
            "actual_position_max_m",
            100.0,
            "Max position error (cm)",
            "Position tracking",
        ),
        (
            "actual_orientation_rmse_rad",
            1.0,
            "Orientation RMSE (rad)",
            "Orientation lag",
        ),
    )
    for axis, (metric, scale, ylabel, title) in zip(
        axes.flat,
        metrics,
        strict=True,
    ):
        _grouped_bars(
            axis,
            data,
            profiles=profiles,
            scenarios=scenarios,
            metric=metric,
            scale=scale,
            ylabel=ylabel,
            title=title,
        )
        axis.tick_params(axis="x", rotation=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
    )
    figure.suptitle(
        "Higher nullspace cost does not materially reduce recorded-case elbow range",
        y=1.07,
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(
        ASSET_DIR / "22_recorded_nullspace_cost_sweep.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_weak_direction_decomposition() -> None:
    data = pd.read_csv(
        recorded.REPLAY_DIR / "weak_direction_decomposition.csv"
    )
    scenarios = list(SCENARIO_LABELS)
    current = data[data["profile"] == "pr_full_recorded"].set_index("scenario")
    x = np.arange(len(scenarios), dtype=np.float64)

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.5))
    z_fraction = (
        current.loc[scenarios, "exact_z_energy_fraction"].to_numpy() * 100.0
    )
    near_fraction = (
        current.loc[scenarios, "near_weak_energy_fraction"].to_numpy()
        * 100.0
    )
    other_fraction = (
        current.loc[scenarios, "other_energy_fraction"].to_numpy() * 100.0
    )
    axes[0].bar(
        x,
        z_fraction,
        color="#2878B5",
        label="Exact nullspace z",
    )
    axes[0].bar(
        x,
        near_fraction,
        bottom=z_fraction,
        color="#E07A2D",
        label="Near-weak direction",
    )
    axes[0].bar(
        x,
        other_fraction,
        bottom=z_fraction + near_fraction,
        color="#B7B7B7",
        label="Other directions",
    )
    axes[0].set_xticks(
        x,
        [SCENARIO_LABELS[item] for item in scenarios],
        rotation=10,
    )
    axes[0].set_ylabel("Command velocity energy (%)")
    axes[0].set_title("Current PR SVD decomposition")
    axes[0].legend(frameon=False, loc="lower left")

    width = 0.22
    cost_profiles = [
        "pr_full_recorded",
        "pr_nullspace_cost_10_recorded",
        "pr_nullspace_cost_12_recorded",
    ]
    for offset_index, profile in enumerate(cost_profiles):
        subset = data[data["profile"] == profile].set_index("scenario")
        ratio = (
            subset.loc[scenarios, "near_weak_abs_mean_rad_s"].to_numpy()
            / subset.loc[scenarios, "exact_z_abs_mean_rad_s"].to_numpy()
        )
        bars = axes[1].bar(
            x + (offset_index - 1) * width,
            ratio,
            width=width * 0.9,
            color=COLORS[profile],
            label=LABELS[profile],
        )
        axes[1].bar_label(bars, fmt="%.1f", padding=2, fontsize=7.5)
    axes[1].axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(
        x,
        [SCENARIO_LABELS[item] for item in scenarios],
        rotation=10,
    )
    axes[1].set_ylabel("Mean |v_near projection| / mean |z projection|")
    axes[1].set_title("Higher exact-z cost leaves near-weak motion")
    axes[1].legend(frameon=False)

    figure.suptitle(
        "Most branch-changing speed is not the exact 1D nullspace coordinate",
        y=1.03,
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(
        ASSET_DIR / "23_recorded_weak_direction_decomposition.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_recorded_feature_ablation(data: pd.DataFrame) -> None:
    profiles = [
        "pr_full_recorded",
        "pr_no_frame_error_recorded",
        "pr_no_nullspace_recorded",
    ]
    scenarios = list(SCENARIO_LABELS)
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))
    metrics = (
        (
            "actual_position_max_m",
            100.0,
            "Max actual position error (cm)",
            "Position error",
        ),
        (
            "actual_elbow_lateral_range_m",
            100.0,
            "Actual elbow lateral range (cm)",
            "Elbow branch",
        ),
        (
            "actual_elbow_accel_p99_m_s2",
            1.0,
            "Actual elbow acceleration p99 (m/s²)",
            "Elbow transient",
        ),
    )
    for axis, (metric, scale, ylabel, title) in zip(
        axes,
        metrics,
        strict=True,
    ):
        _grouped_bars(
            axis,
            data,
            profiles=profiles,
            scenarios=scenarios,
            metric=metric,
            scale=scale,
            ylabel=ylabel,
            title=title,
        )
        axis.tick_params(axis="x", rotation=12)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.suptitle(
        "Recorded proxies isolate nullspace benefit, but not frame-error activation",
        y=1.08,
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(
        ASSET_DIR / "24_recorded_feature_ablation.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def write_summary_table(data: pd.DataFrame) -> None:
    selected = data[
        data["profile"].isin(
            [
                "pr_full_recorded",
                "mainline_ik_velocity_recorded",
                "mainline_recorded",
                "pr_driver_velocity_only_recorded",
                "pr_nullspace_cost_10_recorded",
                "pr_nullspace_cost_12_recorded",
                "pr_no_frame_error_recorded",
                "pr_no_nullspace_recorded",
            ]
        )
    ].copy()
    selected["profile_label"] = selected["profile"].map(LABELS)
    selected["scenario_label"] = selected["scenario"].map(SCENARIO_LABELS)
    columns = [
        "profile",
        "profile_label",
        "scenario",
        "scenario_label",
        "actual_position_max_m",
        "actual_orientation_rmse_rad",
        "actual_elbow_lateral_range_m",
        "actual_ddq_p99_rad_s2",
        "actual_elbow_accel_p99_m_s2",
        "command_j1_dq_max_rad_s",
        "command_j1_sat_fraction",
        "command_j2_sat_fraction",
        "singularity_limit_active_fraction",
    ]
    table_dir = ASSET_DIR.parent / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    selected[columns].to_csv(
        table_dir / "recorded_intervention_controller_comparison.csv",
        index=False,
    )


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    data = pd.read_csv(METRICS_PATH)
    plot_chest_comparison(data)
    plot_retract_tradeoff(data)
    plot_retract_clearance()
    plot_nullspace_cost(data)
    plot_weak_direction_decomposition()
    plot_recorded_feature_ablation(data)
    write_summary_table(data)
    metadata = {
        "metrics": str(METRICS_PATH),
        "selected_events": json.loads(
            (recorded.REPLAY_DIR / "selected_scenarios.json").read_text(
                encoding="utf-8"
            )
        ),
    }
    (recorded.REPLAY_DIR / "plot_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote recorded-intervention plots to {ASSET_DIR}")


if __name__ == "__main__":
    main()
