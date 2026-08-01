#!/usr/bin/env python3
"""Generate publication-style figures and tables from the final IK study."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "current_pr_20260730"
REPORT = Path(
    "/home/hilab/workdir.evaluation/note/openarm_control/final_report"
)
ASSETS = REPORT / "assets"
TABLES = REPORT / "tables"

COLORS = {
    "pr_full": "#1261A0",
    "driver_only_ablation": "#7C3AED",
    "upstream_style": "#D14343",
    "upstream_tasks_tuned_costs": "#E58B28",
    "no_frame_error": "#D14343",
    "no_nullspace": "#E58B28",
    "no_singularity": "#6B7280",
    "no_kinetic": "#64748B",
    "driver_only_velocity": "#7C3AED",
    "no_velocity_diagnostic": "#D14343",
    "orientation_budget_0p15": "#159A78",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
        }
    )


def read_summary(suite: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / suite / "summary.csv")


def load_trace(suite: str, profile: str, scenario: str) -> np.lib.npyio.NpzFile:
    matches = list(
        (RESULTS / suite / "traces").glob(
            f"*_{profile}_{scenario}.npz"
        )
    )
    if not matches:
        raise RuntimeError(
            f"No trace found for {suite}/{profile}/{scenario}."
        )
    latest = max(matches, key=lambda path: path.stat().st_mtime_ns)
    return np.load(latest, allow_pickle=False)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ASSETS / name)
    plt.close(fig)


def pose_position_error(trace: np.lib.npyio.NpzFile, side: str = "right") -> np.ndarray:
    target = trace[f"{side}_target_pose"][:, :3]
    actual = trace[f"{side}_actual_pose"][:, :3]
    return np.linalg.norm(target - actual, axis=1)


def quaternion_error(
    trace: np.lib.npyio.NpzFile,
    side: str = "right",
) -> np.ndarray:
    target = trace[f"{side}_target_pose"][:, 3:]
    actual = trace[f"{side}_actual_pose"][:, 3:]
    dot = np.abs(np.sum(target * actual, axis=1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def max_abs(values: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.max(np.abs(values), axis=axis)


def profile_means(
    frame: pd.DataFrame,
    metrics: Iterable[str],
) -> pd.DataFrame:
    return frame.groupby("profile")[list(metrics)].mean()


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 4.4))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    boxes = [
        (
            0.2,
            1.15,
            2.25,
            2.1,
            "#DCEEFE",
            "Pose target",
            "VR mapping / test path\n6D target pose",
        ),
        (
            2.85,
            0.55,
            4.0,
            3.25,
            "#E7F6EE",
            "Mink QP (5 substeps)",
            (
                "Bounded 6D error\nNullspace home task\n"
                "Singularity approach limit\nRecoverable joint envelope\n"
                "Kinetic tie-breaker"
            ),
        ),
        (
            7.25,
            1.15,
            1.65,
            2.1,
            "#F1EAFE",
            "Raw IK",
            "Integrated q command",
        ),
        (
            9.3,
            1.15,
            1.65,
            2.1,
            "#FFF0DB",
            "Driver",
            "Position reference\nvelocity cap",
        ),
        (
            11.35,
            1.15,
            1.85,
            2.1,
            "#FDE3E3",
            "MuJoCo plant",
            "PD actuator\nactual q / dq",
        ),
    ]
    for x, y, width, height, color, title, body in boxes:
        box = patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color,
            edgecolor="#334155",
            linewidth=1.4,
        )
        ax.add_patch(box)
        ax.text(
            x + width / 2,
            y + height - 0.35,
            title,
            ha="center",
            va="center",
            weight="bold",
            fontsize=12,
        )
        ax.text(
            x + width / 2,
            y + height / 2 - 0.18,
            body,
            ha="center",
            va="center",
            fontsize=9.5,
            linespacing=1.35,
        )
    for start, end in ((2.45, 2.85), (6.85, 7.25), (8.9, 9.3), (10.95, 11.35)):
        ax.annotate(
            "",
            xy=(end, 2.2),
            xytext=(start, 2.2),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "#334155"},
        )
    ax.annotate(
        "measured q (selective use)",
        xy=(5.4, 0.7),
        xytext=(12.15, 0.7),
        ha="center",
        va="center",
        fontsize=9.5,
        color="#475569",
        arrowprops={
            "arrowstyle": "->",
            "lw": 1.5,
            "linestyle": "--",
            "color": "#64748B",
            "connectionstyle": "arc3,rad=-0.10",
        },
    )
    ax.set_title(
        "Evaluation separates IK command, driver reference, and physical state",
        pad=8,
    )
    save(fig, "01_control_layers.png")


def plot_trajectory_catalog() -> None:
    paths = [
        ("reach_right_p0p00_v0p80", "Reach beyond workspace"),
        ("retract_diag_p0p10_v0p80", "Fast diagonal retract"),
        ("extended_circle_right_v0p80", "Extended circle"),
        ("extended_wrist_right_w10p0", "Extended wrist roll"),
        ("chest_outward_flip_right_v1p20_w8p0", "Chest roll + translation"),
        ("normal_right_v0p80", "Normal workspace"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (scenario, title) in zip(axes.flat, paths, strict=True):
        trace = load_trace("candidates", "pr_full", scenario)
        target = trace["right_target_pose"][:, :3]
        actual = trace["right_actual_pose"][:, :3]
        ax.plot(target[:, 0], target[:, 2], color="#111827", lw=2.2, label="target")
        ax.plot(actual[:, 0], actual[:, 2], color=COLORS["pr_full"], lw=1.8, label="actual")
        ax.scatter(target[0, 0], target[0, 2], s=35, color="#159A78", zorder=4)
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Representative trajectory suite (side view)", y=1.01)
    fig.tight_layout()
    save(fig, "02_trajectory_catalog.png")


def plot_headline_comparison() -> None:
    frame = read_summary("screening")
    order = [
        "pr_full",
        "driver_only_ablation",
        "upstream_style",
        "upstream_tasks_tuned_costs",
    ]
    labels = ["Full PR", "Driver-only A/B", "Main-style", "Full-home tuned"]
    specs = [
        ("actual_position_rmse_m", 100.0, "Position RMSE [cm]"),
        (
            "actual_orientation_rmse_rad",
            180.0 / np.pi,
            "Orientation RMSE [deg]",
        ),
        ("actual_ddq_p99_rad_s2", 1.0, "Joint acceleration p99 [rad/s2]"),
        (
            "actual_elbow_accel_p99_m_s2",
            1.0,
            "Elbow acceleration p99 [m/s2]",
        ),
        ("actual_elbow_lateral_range_m", 100.0, "Elbow lateral range [cm]"),
        ("tail_actual_ee_p2p_m", 100.0, "Tail EEF p2p [cm]"),
    ]
    means = frame.groupby("profile").mean(numeric_only=True)
    table = pd.DataFrame(
        {
            label: [means.loc[profile, metric] * scale for metric, scale, _ in specs]
            for profile, label in zip(order, labels, strict=True)
        },
        index=[title for _, _, title in specs],
    )
    table.to_csv(TABLES / "headline_baseline_means.csv")

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4))
    x = np.arange(len(order))
    for ax, (metric, scale, title) in zip(axes.flat, specs, strict=True):
        values = [means.loc[p, metric] * scale for p in order]
        bars = ax.bar(
            x,
            values,
            color=[COLORS.get(p, "#64748B") for p in order],
            width=0.72,
        )
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Current PR versus upstream-style IK across the broad suite",
        y=1.01,
    )
    fig.tight_layout()
    save(fig, "03_headline_baseline_comparison.png")


def paired_effects(
    frame: pd.DataFrame,
    profiles: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    keys = ["scenario", "side"]
    base = frame[frame.profile == "pr_full"].set_index(keys)
    rows: list[pd.Series] = []
    for profile in profiles:
        other = frame[frame.profile == profile].set_index(keys)
        common = base.index.intersection(other.index)
        baseline = base.loc[common, metrics].mean()
        delta = other.loc[common, metrics].mean() - baseline
        relative = 100.0 * delta / baseline.abs().clip(lower=1.0e-6)
        relative.name = profile
        rows.append(relative)
    return pd.DataFrame(rows)


def plot_ablation_heatmaps() -> None:
    frame = read_summary("screening")
    profiles = [
        "no_position_error_bound",
        "no_orientation_error_bound",
        "no_nullspace",
        "no_singularity",
        "no_braking",
        "no_kinetic",
        "driver_only_ablation",
        "upstream_style",
    ]
    profile_labels = {
        "no_position_error_bound": "No position cap",
        "no_orientation_error_bound": "No orientation cap",
        "no_nullspace": "No nullspace task",
        "no_singularity": "No singularity limit",
        "no_braking": "No braking",
        "no_kinetic": "No kinetic task",
        "driver_only_ablation": "Driver-only velocity",
        "upstream_style": "Main-style baseline",
    }
    metrics = [
        "actual_position_rmse_m",
        "actual_orientation_rmse_rad",
        "actual_ddq_p99_rad_s2",
        "actual_elbow_accel_p99_m_s2",
        "actual_elbow_lateral_range_m",
        "tail_actual_ee_p2p_m",
        "driver_limit_active_fraction",
    ]
    metric_labels = [
        "Position\nRMSE",
        "Orientation\nRMSE",
        "Joint\naccel",
        "Elbow\naccel",
        "Elbow\nrange",
        "Tail\nmotion",
        "Driver cap\noccupancy",
    ]
    effects = paired_effects(frame, profiles, metrics)
    effects.index = [profile_labels[p] for p in profiles]
    effects.columns = metric_labels
    effects.to_csv(TABLES / "ablation_relative_effects_percent.csv")

    fig, ax = plt.subplots(figsize=(12.7, 6.6))
    sns.heatmap(
        effects.clip(-150.0, 150.0),
        annot=effects,
        fmt=".0f",
        cmap="RdYlGn_r",
        center=0.0,
        vmin=-100.0,
        vmax=150.0,
        linewidths=0.5,
        cbar_kws={"label": "Change from full PR [%]; positive is more"},
        ax=ax,
    )
    ax.set_title("Feature removal is not interchangeable: paired broad-suite effects")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    save(fig, "04_feature_ablation_heatmap.png")

    families = sorted(frame.family.unique())
    selected = [
        "no_position_error_bound",
        "no_orientation_error_bound",
        "no_nullspace",
        "no_singularity",
        "no_kinetic",
        "driver_only_ablation",
    ]
    base = frame[frame.profile == "pr_full"].set_index(["scenario", "side"])
    ddq = pd.DataFrame(index=[profile_labels[p] for p in selected], columns=families)
    elbow = ddq.copy()
    for profile in selected:
        other = frame[frame.profile == profile].set_index(["scenario", "side"])
        common = base.index.intersection(other.index)
        joined = other.loc[common].copy()
        joined["ddq_delta"] = (
            other.loc[common, "actual_ddq_p99_rad_s2"]
            - base.loc[common, "actual_ddq_p99_rad_s2"]
        )
        joined["elbow_delta"] = (
            other.loc[common, "actual_elbow_lateral_range_m"]
            - base.loc[common, "actual_elbow_lateral_range_m"]
        ) * 100.0
        ddq.loc[profile_labels[profile]] = joined.groupby("family")["ddq_delta"].mean()
        elbow.loc[profile_labels[profile]] = joined.groupby("family")[
            "elbow_delta"
        ].mean()
    ddq = ddq.astype(float)
    elbow = elbow.astype(float)
    ddq.to_csv(TABLES / "ablation_ddq_delta_by_family.csv")
    elbow.to_csv(TABLES / "ablation_elbow_delta_by_family_cm.csv")
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    sns.heatmap(
        ddq,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Actual ddq p99 delta [rad/s2]"},
        ax=axes[0],
    )
    axes[0].set_title("Where each mechanism acts: acceleration effect")
    sns.heatmap(
        elbow,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Elbow lateral-range delta [cm]"},
        ax=axes[1],
    )
    axes[1].set_title("Where each mechanism acts: branch-motion effect")
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.tight_layout()
    save(fig, "05_feature_effect_by_trajectory_family.png")


def plot_chest_case() -> None:
    chest_summary = read_summary("chest")
    chest_metrics = [
        "actual_position_rmse_m",
        "actual_orientation_rmse_rad",
        "actual_ddq_p99_rad_s2",
        "actual_elbow_accel_p99_m_s2",
        "actual_elbow_lateral_range_m",
        "driver_limit_active_fraction",
        "tail_actual_ee_p2p_m",
    ]
    profile_means(chest_summary, chest_metrics).to_csv(
        TABLES / "chest_profile_means.csv"
    )
    scenario = "chest_outward_flip_right_diagonal_v1p20_w8p0"
    profiles = [
        "pr_full",
        "orientation_budget_0p15",
        "no_nullspace",
        "no_frame_error",
    ]
    labels = {
        "pr_full": "Full PR (0.25 rad)",
        "orientation_budget_0p15": "0.15 rad orientation cap",
        "no_nullspace": "No nullspace regulation",
        "no_frame_error": "No 6D error modulation",
    }
    traces = {p: load_trace("chest", p, scenario) for p in profiles}
    fig, axes = plt.subplots(3, 2, figsize=(14.2, 11), sharex="col")
    for profile, trace in traces.items():
        t = trace["times"]
        color = COLORS.get(profile)
        axes[0, 0].plot(
            t,
            pose_position_error(trace) * 100.0,
            label=labels[profile],
            color=color,
        )
        axes[1, 0].plot(
            t,
            quaternion_error(trace) * 180.0 / np.pi,
            color=color,
        )
        elbow = trace["right_actual_elbow"]
        axes[2, 0].plot(
            t,
            (elbow[:, 1] - elbow[0, 1]) * 100.0,
            color=color,
        )
        axes[0, 1].plot(
            t,
            max_abs(trace["right_actual_dq"]),
            color=color,
        )
        axes[1, 1].plot(
            t,
            max_abs(trace["right_actual_ddq"]),
            color=color,
        )
        active = np.any(trace["right_driver_limit_active_by_joint"], axis=1)
        axes[2, 1].plot(t, active.astype(float), color=color)
    axes[0, 0].set_ylabel("Position error [cm]")
    axes[1, 0].set_ylabel("Orientation error [deg]")
    axes[2, 0].set_ylabel("Elbow lateral delta [cm]")
    axes[2, 0].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("max |actual dq| [rad/s]")
    axes[1, 1].set_ylabel("max |actual ddq| [rad/s2]")
    axes[2, 1].set_ylabel("Driver cap active")
    axes[2, 1].set_xlabel("Time [s]")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[2, 1].set_ylim(-0.05, 1.05)
    fig.suptitle(
        "Chest wrist roll with fast diagonal translation: stability comes from "
        "spending less joint authority on instantaneous orientation",
        y=1.005,
    )
    fig.tight_layout()
    save(fig, "06_chest_wrist_error_modulation_timeseries.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    target = traces["pr_full"]["right_target_pose"][:, :3]
    axes[0].plot(target[:, 0], target[:, 2], "k--", lw=2.2, label="target")
    axes[1].plot(target[:, 1], target[:, 2], "k--", lw=2.2, label="target")
    for profile, trace in traces.items():
        actual = trace["right_actual_pose"][:, :3]
        axes[0].plot(
            actual[:, 0],
            actual[:, 2],
            color=COLORS.get(profile),
            label=labels[profile],
        )
        axes[1].plot(actual[:, 1], actual[:, 2], color=COLORS.get(profile))
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("z [m]")
    axes[0].set_title("Side view")
    axes[1].set_xlabel("y [m]")
    axes[1].set_ylabel("z [m]")
    axes[1].set_title("Front view")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Actual end-effector path during the chest stress case")
    fig.tight_layout()
    save(fig, "07_chest_wrist_eef_paths.png")


def plot_nullspace_branch() -> None:
    scenario = "chest_outward_flip_right_lateral_v1p20_w8p0"
    profiles = ["pr_full", "no_nullspace"]
    labels = ["Full PR", "No nullspace task"]
    traces = {
        profile: load_trace("chest", profile, scenario) for profile in profiles
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for profile, label in zip(profiles, labels, strict=True):
        trace = traces[profile]
        color = COLORS[profile]
        elbow = trace["right_actual_elbow"]
        axes[0, 0].plot(
            elbow[:, 1],
            elbow[:, 2],
            color=color,
            label=label,
        )
        axes[0, 1].plot(
            trace["times"],
            (elbow[:, 1] - elbow[0, 1]) * 100.0,
            color=color,
            label=label,
        )
        axes[1, 0].plot(
            trace["times"],
            trace["right_actual_q"][:, 0],
            color=color,
            label=f"{label}: J1",
        )
        axes[1, 0].plot(
            trace["times"],
            trace["right_actual_q"][:, 3],
            color=color,
            linestyle="--",
            label=f"{label}: J4",
        )
        axes[1, 1].plot(
            trace["times"],
            max_abs(trace["right_actual_dq"]),
            color=color,
        )
    axes[0, 0].set_xlabel("Elbow y [m]")
    axes[0, 0].set_ylabel("Elbow z [m]")
    axes[0, 0].set_title("Elbow branch path")
    axes[0, 0].legend()
    axes[0, 1].set_ylabel("Elbow lateral delta [cm]")
    axes[0, 1].set_title("Lateral branch excursion")
    axes[1, 0].set_ylabel("Actual joint position [rad]")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_title("Shoulder/elbow branch coordinates")
    axes[1, 0].legend(fontsize=7, ncol=2)
    axes[1, 1].set_ylabel("max |actual dq| [rad/s]")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_title("Physical joint speed")
    fig.suptitle("Nullspace regulation controls branch motion, not Cartesian pose")
    fig.tight_layout()
    save(fig, "08_nullspace_branch_control.png")


def plot_singularity_case() -> None:
    scenario = "reach_right_p0p00_v0p80"
    profiles = ["pr_full", "no_singularity"]
    labels = ["Full PR", "No singularity approach limit"]
    traces = {
        p: load_trace("screening", p, scenario) for p in profiles
    }
    fig, axes = plt.subplots(4, 1, figsize=(12.8, 10), sharex=True)
    for profile, label in zip(profiles, labels, strict=True):
        trace = traces[profile]
        color = COLORS[profile]
        t = trace["times"]
        axes[0].plot(t, trace["right_actual_rho"], color=color, label=label)
        axes[1].plot(t, trace["right_actual_dq"][:, 0], color=color)
        axes[2].plot(t, trace["right_actual_dq"][:, 3], color=color)
        axes[3].plot(t, max_abs(trace["right_actual_ddq"]), color=color)
    axes[0].axhspan(0.02, 0.08, color="#FDE68A", alpha=0.25)
    axes[0].axhline(0.02, color="#D14343", lw=1, ls="--")
    axes[0].set_ylabel("Geometric rho")
    axes[0].legend()
    axes[1].set_ylabel("Actual J1 dq [rad/s]")
    axes[2].set_ylabel("Actual J4 dq [rad/s]")
    axes[3].set_ylabel("max |actual ddq| [rad/s2]")
    axes[3].set_xlabel("Time [s]")
    fig.suptitle(
        "Straight reach: one-sided singularity limiting reduces the final "
        "shoulder/elbow velocity tail"
    )
    fig.tight_layout()
    save(fig, "09_singularity_reach_timeseries.png")


def plot_driver_coupling() -> None:
    driver_summary = read_summary("driver")
    driver_metrics = [
        "actual_position_rmse_m",
        "actual_orientation_rmse_rad",
        "command_dq_max_rad_s",
        "driver_dq_max_rad_s",
        "actual_dq_max_rad_s",
        "actual_ddq_p99_rad_s2",
        "actual_elbow_accel_p99_m_s2",
        "command_driver_q_gap_rms_rad",
        "driver_actual_q_gap_rms_rad",
    ]
    profile_means(driver_summary, driver_metrics).to_csv(
        TABLES / "driver_coupling_profile_means.csv"
    )
    scenario = "retract_diag_p0p10_v0p80"
    profiles = [
        "pr_full",
        "driver_only_velocity",
        "no_velocity_diagnostic",
    ]
    labels = ["IK + driver caps", "Driver cap only", "No velocity caps"]
    traces = {
        p: load_trace("driver", p, scenario) for p in profiles
    }
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="col")
    for profile, label in zip(profiles, labels, strict=True):
        trace = traces[profile]
        color = COLORS.get(profile)
        t = trace["times"]
        axes[0, 0].plot(
            t,
            pose_position_error(trace) * 100.0,
            color=color,
            label=label,
        )
        axes[1, 0].plot(t, trace["right_command_dq"][:, 0], color=color)
        axes[2, 0].plot(t, trace["right_actual_dq"][:, 0], color=color)
        gap = np.max(
            np.abs(trace["right_command_q"] - trace["right_driver_q"]),
            axis=1,
        )
        axes[0, 1].plot(t, gap, color=color)
        axes[1, 1].plot(
            t,
            max_abs(trace["right_actual_dq"]),
            color=color,
        )
        axes[2, 1].plot(
            t,
            max_abs(trace["right_actual_ddq"]),
            color=color,
        )
    axes[0, 0].set_ylabel("Position error [cm]")
    axes[0, 0].legend()
    axes[1, 0].set_ylabel("Command J1 dq [rad/s]")
    axes[2, 0].set_ylabel("Actual J1 dq [rad/s]")
    axes[2, 0].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Raw-driver q gap [rad]")
    axes[1, 1].set_ylabel("max |actual dq| [rad/s]")
    axes[2, 1].set_ylabel("max |actual ddq| [rad/s2]")
    axes[2, 1].set_xlabel("Time [s]")
    fig.suptitle(
        "Fast retract: downstream clipping cannot reproduce an IK-feasible "
        "velocity-limited branch"
    )
    fig.tight_layout()
    save(fig, "10_driver_limit_coupling_timeseries.png")


def plot_boundary_recovery() -> None:
    frame = read_summary("boundary")
    labels = {
        "recoverable_joint_limit": "Recoverable pos + velocity",
        "native_position_plus_velocity": "Native pos + velocity",
        "configuration_only": "Configuration only",
    }
    solved = frame.groupby("profile")["solved"].agg(["sum", "count"])
    solved = solved.reindex(labels)
    solved.to_csv(TABLES / "boundary_recovery_solved_counts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x = np.arange(len(solved))
    bars = axes[0].bar(
        x,
        solved["sum"],
        color=["#159A78", "#D14343", "#E58B28"],
    )
    axes[0].set_xticks(x, [labels[p] for p in solved.index], rotation=15)
    axes[0].set_ylabel("Solved static cases")
    axes[0].set_ylim(0, solved["count"].max() * 1.12)
    axes[0].bar_label(
        bars,
        labels=[f"{int(a)}/{int(b)}" for a, b in solved.to_numpy()],
    )
    axes[0].set_title("Feasibility when synchronized outside a bound")

    subset = frame[
        (frame.joint == 4)
        & (frame.boundary == "lower")
    ]
    for profile, group in subset.groupby("profile"):
        axes[1].plot(
            group.offset_rad * 1000.0,
            group.remaining_violation_rad * 1000.0,
            marker="o",
            label=labels.get(profile, profile),
        )
    axes[1].axhline(0.0, color="#111827", lw=1)
    axes[1].set_xlabel("Initial offset outside J4 bound [mrad]")
    axes[1].set_ylabel("Remaining violation [mrad]")
    axes[1].set_title("J4 recovery behavior")
    axes[1].legend(fontsize=8)
    fig.suptitle("Recoverable joint envelope keeps restoration feasible")
    fig.tight_layout()
    save(fig, "11_recoverable_joint_limit.png")


def braking_envelope(distance: np.ndarray, slowdown: float, exponent: float = 2.0) -> np.ndarray:
    u = np.clip(np.maximum(distance, 0.0) / slowdown, 0.0, 1.0)
    smoothstep = u * u * (3.0 - 2.0 * u)
    return np.power(smoothstep, exponent)


def plot_braking() -> None:
    frame = read_summary("braking")
    braking_metrics = [
        "actual_min_joint_margin_rad",
        "actual_j6_dq_max_rad_s",
        "actual_ddq_p99_rad_s2",
        "actual_orientation_rmse_rad",
        "braking_active_fraction",
    ]
    profile_means(frame, braking_metrics).to_csv(
        TABLES / "braking_profile_means.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    distance = np.linspace(0.0, 0.32, 300)
    for value, color in zip(
        (0.08, 0.12, 0.20, 0.30),
        ("#159A78", "#1261A0", "#E58B28", "#D14343"),
        strict=True,
    ):
        axes[0].plot(
            distance,
            braking_envelope(distance, value),
            lw=2.2,
            color=color,
            label=f"{value:.2f} rad",
        )
    axes[0].set_xlabel("Remaining joint margin [rad]")
    axes[0].set_ylabel("Allowed approach speed / physical cap")
    axes[0].set_title("Preventive distance-speed envelope")
    axes[0].legend(title="Slowdown distance")

    profiles = [
        "no_braking",
        "brake_distance_0p08",
        "brake_distance_0p12",
        "brake_distance_0p2",
        "brake_distance_0p3",
        "configuration_only",
    ]
    labels = [
        "No braking",
        "0.08",
        "0.12",
        "0.20",
        "0.30",
        "No IK velocity",
    ]
    label_offsets = [(5, 8), (5, -12), (5, 8), (5, 8), (5, 8), (5, 8)]
    fast = frame[frame.speed == frame.speed.max()].groupby("profile").mean(
        numeric_only=True
    )
    margin = [fast.loc[p, "actual_min_joint_margin_rad"] * 1000.0 for p in profiles]
    speed = [fast.loc[p, "actual_j6_dq_max_rad_s"] for p in profiles]
    axes[1].scatter(margin, speed, s=95, color=sns.color_palette("tab10", len(profiles)))
    for x, y, label, offset in zip(margin, speed, labels, label_offsets, strict=True):
        axes[1].annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_xlabel("Actual minimum joint margin [mrad]")
    axes[1].set_ylabel("Actual max |J6 dq| [rad/s]")
    axes[1].set_title("12 rad/s target sweep: margin versus speed")
    fig.suptitle("Joint braking trades boundary margin for a gradual slowdown")
    fig.tight_layout()
    save(fig, "12_joint_braking_envelope.png")


def plot_parameter_sweeps() -> None:
    frame = read_summary("parameters")
    means = frame.groupby("profile").mean(numeric_only=True)
    means.to_csv(TABLES / "parameter_profile_means.csv")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2))

    def series(mapping: list[tuple[float, str]], metric: str) -> tuple[list[float], list[float]]:
        return (
            [value for value, _ in mapping],
            [means.loc[name, metric] for _, name in mapping],
        )

    orientation = [
        (0.0, "orientation_budget_0"),
        (0.15, "orientation_budget_0p15"),
        (0.20, "orientation_budget_0p2"),
        (0.25, "orientation_budget_0p25"),
        (0.30, "orientation_budget_0p3"),
        (0.40, "orientation_budget_0p4"),
    ]
    x, y = series(orientation, "actual_position_rmse_m")
    axes[0, 0].plot(x, np.asarray(y) * 100.0, marker="o", label="position RMSE [cm]")
    x, y = series(orientation, "actual_elbow_lateral_range_m")
    axes[0, 0].plot(x, np.asarray(y) * 100.0, marker="s", label="elbow range [cm]")
    axes[0, 0].axvline(0.25, color="#111827", ls="--", lw=1)
    axes[0, 0].set_xlabel("Orientation total budget [rad]")
    axes[0, 0].set_title("Orientation error budget")
    axes[0, 0].legend(fontsize=8)

    null_cost = [
        (0.0, "null_cost_0"),
        (3.0, "null_cost_3"),
        (7.0, "null_cost_7"),
        (12.0, "null_cost_12"),
        (18.0, "null_cost_18"),
    ]
    x, y = series(null_cost, "actual_elbow_lateral_range_m")
    axes[0, 1].plot(x, np.asarray(y) * 100.0, marker="o", label="elbow range [cm]")
    x, y = series(null_cost, "actual_ddq_p99_rad_s2")
    axes[0, 1].plot(x, np.asarray(y) / 5.0, marker="s", label="ddq p99 / 5")
    axes[0, 1].axvline(7.0, color="#111827", ls="--", lw=1)
    axes[0, 1].set_xlabel("Nullspace cost")
    axes[0, 1].set_title("Nullspace home regulation")
    axes[0, 1].legend(fontsize=8)

    singular = [
        (0.0, "sing_rate_0"),
        (0.10, "sing_rate_0p1"),
        (0.18, "sing_rate_0p18"),
        (0.25, "sing_rate_0p25"),
        (0.35, "sing_rate_0p35"),
        (0.50, "sing_rate_0p5"),
    ]
    x, y = series(singular, "actual_min_rho")
    axes[0, 2].plot(x, y, marker="o", label="minimum rho")
    ax2 = axes[0, 2].twinx()
    _, y2 = series(singular, "actual_ddq_p99_rad_s2")
    ax2.plot(x, y2, marker="s", color="#D14343", label="ddq p99")
    axes[0, 2].axvline(0.25, color="#111827", ls="--", lw=1)
    axes[0, 2].set_xlabel("Maximum singularity approach rate")
    axes[0, 2].set_title("Singularity approach limit")
    axes[0, 2].set_ylabel("minimum rho")
    ax2.set_ylabel("ddq p99 [rad/s2]", color="#D14343")

    braking = [
        (0.08, "brake_distance_0p08"),
        (0.12, "brake_distance_0p12"),
        (0.20, "brake_distance_0p2"),
        (0.30, "brake_distance_0p3"),
        (0.50, "brake_distance_0p5"),
    ]
    x, y = series(braking, "actual_min_joint_margin_rad")
    axes[1, 0].plot(x, np.asarray(y) * 1000.0, marker="o", label="min margin [mrad]")
    x, y = series(braking, "actual_position_rmse_m")
    axes[1, 0].plot(x, np.asarray(y) * 100.0, marker="s", label="position RMSE [cm]")
    axes[1, 0].axvline(0.20, color="#111827", ls="--", lw=1)
    axes[1, 0].set_xlabel("Braking distance [rad]")
    axes[1, 0].set_title("Joint braking distance")
    axes[1, 0].legend(fontsize=8)

    energy = [
        (0.0, "energy_0"),
        (1.0e-5, "energy_1em05"),
        (2.0e-5, "energy_2em05"),
        (3.0e-5, "energy_3em05"),
        (5.0e-5, "energy_5em05"),
        (1.0e-4, "energy_0p0001"),
    ]
    x, y = series(energy, "actual_ddq_p99_rad_s2")
    axes[1, 1].plot(np.asarray(x) * 1.0e5, y, marker="o", label="ddq p99")
    x, y = series(energy, "actual_elbow_accel_p99_m_s2")
    axes[1, 1].plot(np.asarray(x) * 1.0e5, y, marker="s", label="elbow accel p99")
    axes[1, 1].axvline(2.0, color="#111827", ls="--", lw=1)
    axes[1, 1].set_xlabel("Kinetic cost [x 1e-5]")
    axes[1, 1].set_title("Kinetic-energy tie-breaker")
    axes[1, 1].legend(fontsize=8)

    caps = [
        (0.75, "control_caps_x0p75"),
        (1.0, "control_caps_x1"),
        (1.25, "control_caps_x1p25"),
        (1.5, "control_caps_x1p5"),
    ]
    x, y = series(caps, "actual_position_rmse_m")
    axes[1, 2].plot(x, np.asarray(y) * 100.0, marker="o", label="position RMSE [cm]")
    x, y = series(caps, "actual_ddq_p99_rad_s2")
    axes[1, 2].plot(x, np.asarray(y) / 5.0, marker="s", label="ddq p99 / 5")
    axes[1, 2].axvline(1.0, color="#111827", ls="--", lw=1)
    axes[1, 2].set_xlabel("IK velocity-cap scale")
    axes[1, 2].set_title("IK velocity envelope")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("One-factor tuning: useful plateaus are wider than a single optimum", y=1.01)
    fig.tight_layout()
    save(fig, "13_parameter_sweep_summary.png")


def plot_candidates() -> None:
    frame = read_summary("candidates")
    metrics = [
        "actual_position_rmse_m",
        "actual_orientation_rmse_rad",
        "actual_ddq_p99_rad_s2",
        "actual_elbow_accel_p99_m_s2",
        "actual_elbow_lateral_range_m",
        "driver_limit_active_fraction",
    ]
    means = profile_means(frame, metrics)
    means.to_csv(TABLES / "combined_candidate_means.csv")
    order = [
        "pr_full",
        "candidate_minimal",
        "candidate_balanced",
        "candidate_smooth",
        "candidate_tracking",
    ]
    labels = ["Current", "Minimal", "Balanced", "Smooth", "Tracking"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    for profile, label in zip(order, labels, strict=True):
        row = means.loc[profile]
        axes[0].scatter(
            row.actual_position_rmse_m * 100.0,
            row.actual_ddq_p99_rad_s2,
            s=80 + 1200 * row.actual_elbow_lateral_range_m,
            label=label,
        )
        axes[1].scatter(
            row.actual_orientation_rmse_rad * 180.0 / np.pi,
            row.actual_elbow_accel_p99_m_s2,
            s=80 + 800 * row.driver_limit_active_fraction,
            label=label,
        )
    axes[0].set_xlabel("Position RMSE [cm]")
    axes[0].set_ylabel("Joint acceleration p99 [rad/s2]")
    axes[0].set_title("Bubble size: elbow lateral range")
    axes[1].set_xlabel("Orientation RMSE [deg]")
    axes[1].set_ylabel("Elbow acceleration p99 [m/s2]")
    axes[1].set_title("Bubble size: driver-cap occupancy")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(
        "Combined one-factor winners do not dominate the current balanced defaults"
    )
    fig.tight_layout()
    save(fig, "14_combined_tuning_candidates.png")


def plot_symmetry_and_robustness() -> None:
    symmetry = read_summary("symmetry")
    exact = symmetry[
        symmetry.scenario.str.contains("exact_mirror")
        & (symmetry.profile == "pr_full")
    ]
    metrics = [
        "actual_position_rmse_m",
        "actual_orientation_rmse_rad",
        "actual_ddq_p99_rad_s2",
        "actual_elbow_accel_p99_m_s2",
        "actual_elbow_lateral_range_m",
    ]
    by_side = exact.groupby("side")[metrics].mean()
    by_side.to_csv(TABLES / "exact_mirror_side_means.csv")

    robustness = read_summary("robustness")
    robust_metrics = [
        "actual_position_rmse_m",
        "actual_orientation_rmse_rad",
        "actual_ddq_p99_rad_s2",
        "driver_actual_q_gap_rms_rad",
    ]
    robust = robustness.groupby("profile")[robust_metrics].mean()
    robust.to_csv(TABLES / "robustness_profile_means.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    scaled = by_side.copy()
    scaled["actual_position_rmse_m"] *= 100.0
    scaled["actual_orientation_rmse_rad"] *= 180.0 / np.pi
    scaled["actual_elbow_lateral_range_m"] *= 100.0
    scaled = scaled.rename(
        columns={
            "actual_position_rmse_m": "pos RMSE [cm]",
            "actual_orientation_rmse_rad": "ori RMSE [deg]",
            "actual_ddq_p99_rad_s2": "ddq p99",
            "actual_elbow_accel_p99_m_s2": "elbow acc p99",
            "actual_elbow_lateral_range_m": "elbow range [cm]",
        }
    )
    scaled.T.plot.bar(ax=axes[0], color=["#1261A0", "#D14343"])
    axes[0].set_title("Exact mirrored paths")
    axes[0].set_ylabel("Metric value")
    axes[0].tick_params(axis="x", rotation=20)

    order = [
        "pr_full",
        "gravity_compensation",
        "state_delay_20ms",
        "command_delay_20ms",
        "actuator_gain_x0p7",
        "actuator_gain_x1p3",
        "driver_only_ablation",
    ]
    base = robust.loc["pr_full"]
    delta = pd.DataFrame(
        {
            p: 100.0 * (robust.loc[p] - base) / base.abs().clip(lower=1.0e-6)
            for p in order[1:]
        }
    ).T
    delta.columns = ["position", "orientation", "ddq", "q gap"]
    sns.heatmap(
        delta,
        cmap="RdYlGn_r",
        center=0,
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "Change from PR [%]"},
        ax=axes[1],
    )
    axes[1].set_title("Plant and delay sensitivity")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    fig.suptitle("Side equivalence and dynamic-plant robustness")
    fig.tight_layout()
    save(fig, "15_symmetry_and_robustness.png")


def plot_solver_timing() -> None:
    frame = read_summary("screening")
    selected = frame[
        frame.profile.isin(
            [
                "pr_full",
                "driver_only_ablation",
                "upstream_style",
                "no_frame_error",
            ]
        )
    ]
    means = selected.groupby("profile")[
        ["solve_time_mean_ms", "solve_time_p95_ms"]
    ].mean()
    means.to_csv(TABLES / "solver_timing_means.csv")
    order = ["pr_full", "driver_only_ablation", "upstream_style", "no_frame_error"]
    labels = ["Full PR", "Driver-only A/B", "Main-style", "No error bound"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(
        x - 0.18,
        means.loc[order, "solve_time_mean_ms"],
        width=0.36,
        label="mean",
        color="#1261A0",
    )
    ax.bar(
        x + 0.18,
        means.loc[order, "solve_time_p95_ms"],
        width=0.36,
        label="p95",
        color="#8BBCE5",
    )
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("Outer solve time [ms]")
    ax.set_title("Five-substep bimanual solve timing in the simulation harness")
    ax.legend()
    fig.tight_layout()
    save(fig, "16_solver_timing.png")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_architecture()
    plot_trajectory_catalog()
    plot_headline_comparison()
    plot_ablation_heatmaps()
    plot_chest_case()
    plot_nullspace_branch()
    plot_singularity_case()
    plot_driver_coupling()
    plot_boundary_recovery()
    plot_braking()
    plot_parameter_sweeps()
    plot_candidates()
    plot_symmetry_and_robustness()
    plot_solver_timing()
    print(f"Wrote figures to {ASSETS}")
    print(f"Wrote tables to {TABLES}")


if __name__ == "__main__":
    main()
