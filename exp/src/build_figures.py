#!/usr/bin/env python3
"""Generate the public figures and traceable metric tables."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from matplotlib.colors import TwoSlopeNorm

import study


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / "results" / "final_report_20260801"
DEFAULT_REPORT = HERE.parent

RESULTS = DEFAULT_RESULTS
REPORT = DEFAULT_REPORT
ASSETS = REPORT / "assets"
TABLES = REPORT / "tables"

COLORS = {
    "current_deployment": "#13795B",
    "strict_mainline": "#C44536",
    "full_home_replacement_0p003": "#9A6A1F",
    "full_home_replacement_0p01": "#D49324",
    "full_home_replacement_0p03": "#F2B84B",
    "no_branch_regulation": "#7C3AED",
    "no_frame_error_bound": "#D49324",
    "no_position_error_bound": "#4C78A8",
    "no_orientation_error_bound": "#E45756",
    "no_singularity_limit": "#6B7280",
    "no_joint_braking": "#8B5E3C",
    "no_kinetic_regularization": "#64748B",
    "driver_only_velocity": "#7C3AED",
    "no_velocity_limits": "#111827",
    "orientation_budget_0p15": "#159A78",
}

LABELS = {
    "current_deployment": "PR default",
    "strict_mainline": "Mainline baseline",
    "full_home_replacement_0p003": "PR: full-home posture 0.003",
    "full_home_replacement_0p01": "PR: full-home posture 0.01",
    "full_home_replacement_0p03": "PR: full-home posture 0.03",
    "no_branch_regulation": "PR w/o posture regulation",
    "no_frame_error_bound": "PR w/o 6D error bound",
    "no_position_error_bound": "PR w/o position bound",
    "no_orientation_error_bound": "PR w/o orientation bound",
    "no_singularity_limit": "PR w/o singularity limit",
    "no_joint_braking": "PR w/o braking",
    "no_kinetic_regularization": "PR w/o kinetic regularizer",
    "driver_only_velocity": "PR w/o IK velocity limits",
    "no_velocity_limits": "PR w/o velocity limits",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11.5,
            "legend.title_fontsize": 11.5,
            "font.family": "DejaVu Sans",
        }
    )


def read_summary(suite: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / suite / "summary.csv")


def load_trace(suite: str, profile: str, scenario: str) -> dict[str, np.ndarray]:
    matches = list(
        (RESULTS / suite / "traces").glob(f"*_{profile}_{scenario}.npz")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one trace for {suite}/{profile}/{scenario}, got {len(matches)}."
        )
    with np.load(matches[0], allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def save(fig: plt.Figure, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / name)
    plt.close(fig)


def position_error(trace: dict[str, np.ndarray], side: str = "right") -> np.ndarray:
    return np.linalg.norm(
        trace[f"{side}_target_pose"][:, :3]
        - trace[f"{side}_actual_pose"][:, :3],
        axis=1,
    )


def orientation_error(
    trace: dict[str, np.ndarray],
    side: str = "right",
) -> np.ndarray:
    target = trace[f"{side}_target_pose"][:, 3:]
    actual = trace[f"{side}_actual_pose"][:, 3:]
    dot = np.abs(np.sum(target * actual, axis=1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def max_abs(values: np.ndarray) -> np.ndarray:
    return np.max(np.abs(values), axis=1)


def profile_means(frame: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    return frame.groupby("profile")[list(metrics)].mean()


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(14.2, 4.6))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    boxes = (
        (0.2, 1.2, 2.25, 2.1, "#DCEEFE", "Pose target", "VR mapping / test path\n6D target pose"),
        (
            2.85,
            0.55,
            4.05,
            3.35,
            "#E7F6EE",
            "Mink QP (5 substeps)",
            "Bounded 6D error\nExact-nullspace home task\nSingularity approach limit\nRecoverable joint envelope\nKinetic tie-breaker",
        ),
        (7.22, 1.2, 2.25, 2.1, "#F1EAFE", "Raw IK command", "Integrated joint-position\ncommand"),
        (9.82, 1.2, 1.88, 2.1, "#FFF0DB", "Driver", "Position reference\nvelocity limits"),
        (12.05, 1.2, 1.95, 2.1, "#FDE3E3", "MuJoCo plant", "PD actuator\nactual q / dq"),
    )
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
        ax.text(x + width / 2, y + height - 0.36, title, ha="center", va="center", weight="bold", fontsize=12.0)
        ax.text(x + width / 2, y + height / 2 - 0.18, body, ha="center", va="center", fontsize=10.2, linespacing=1.32)
    for start, end in ((2.45, 2.85), (6.90, 7.22), (9.47, 9.82), (11.70, 12.05)):
        ax.annotate("", xy=(end, 2.25), xytext=(start, 2.25), arrowprops={"arrowstyle": "->", "lw": 2, "color": "#334155"})
    ax.annotate(
        "measured q (state-aware limits only)",
        xy=(5.45, 0.72),
        xytext=(12.9, 0.72),
        ha="center",
        va="center",
        fontsize=10.2,
        color="#475569",
        arrowprops={"arrowstyle": "->", "lw": 1.4, "linestyle": "--", "color": "#64748B", "connectionstyle": "arc3,rad=-0.10"},
    )
    ax.set_title("Command, driver-reference, and physical-state layers", pad=8)
    save(fig, "01_control_layers.png")


def plot_trajectory_catalog() -> None:
    paths = (
        ("screening", "reach_right_p0p00_v0p80", "Arm extension beyond reach"),
        ("screening", "retract_diag_p0p10_v0p80", "Fast diagonal retraction"),
        ("screening", "extended_circle_right_v0p80", "Extended-arm circle"),
        ("screening", "extended_up_right_v0p80", "Extended vertical motion"),
        ("screening", "extended_wrist_right_w10p0", "Extended wrist roll"),
        ("screening", "normal_right_v0p80", "Normal-workspace motion"),
        ("frozen", "near_chest_fast_wrist_roll", "Recorded near-chest wrist roll"),
        ("frozen", "fast_retract_elbow_branch", "Recorded elbow-branch retract"),
    )
    fig = plt.figure(figsize=(17.8, 10.8))
    axes = [fig.add_subplot(2, 4, index + 1, projection="3d") for index in range(8)]
    for axis, (suite, scenario, title) in zip(axes, paths, strict=True):
        trace = load_trace(suite, "current_deployment", scenario)
        target = trace["right_target_pose"][:, :3]
        actual = trace["right_actual_pose"][:, :3]
        axis.plot(
            target[:, 0],
            target[:, 1],
            target[:, 2],
            color="#111827",
            lw=2.0,
            ls="--",
            label="target",
        )
        axis.plot(
            actual[:, 0],
            actual[:, 1],
            actual[:, 2],
            color=COLORS["current_deployment"],
            lw=1.7,
            label="state",
        )
        axis.scatter(
            target[0, 0],
            target[0, 1],
            target[0, 2],
            s=34,
            color="#2563EB",
            depthshade=False,
            label="start",
            zorder=4,
        )
        points = np.vstack((target, actual))
        center = 0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
        side = max(float(np.max(np.ptp(points, axis=0))), 0.04) * 1.14
        half_side = 0.5 * side
        axis.set_xlim(center[0] - half_side, center[0] + half_side)
        axis.set_ylim(center[1] - half_side, center[1] + half_side)
        axis.set_zlim(center[2] - half_side, center[2] + half_side)
        axis.set_box_aspect((1.0, 1.0, 1.0))
        axis.view_init(elev=24, azim=-58)
        axis.set_title(title, fontsize=12.5, pad=7)
        axis.set_xlabel("x [m]", labelpad=2)
        axis.set_ylabel("y [m]", labelpad=2)
        axis.set_zlabel("z [m]", labelpad=2)
        axis.tick_params(axis="both", labelsize=9.5, pad=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        fontsize=12.0,
    )
    fig.suptitle("Representative 3D Target and State Paths in arm_origin", y=0.995)
    fig.tight_layout(rect=(0.0, 0.01, 1.0, 0.91), h_pad=5.8, w_pad=1.2)
    save(fig, "02_trajectory_catalog.png")


def plot_headline_comparison() -> None:
    frame = read_summary("screening")
    order = (
        "current_deployment",
        "driver_only_velocity",
        "strict_mainline",
        "full_home_replacement_0p01",
    )
    specs = (
        ("actual_position_rmse_m", 100.0, "Position RMSE [cm]"),
        ("actual_orientation_rmse_rad", 180.0 / np.pi, "Orientation RMSE [deg]"),
        ("actual_ddq_p99_rad_s2", 1.0, "Joint acceleration p99 [rad/s²]"),
        ("actual_elbow_accel_p99_m_s2", 1.0, "Elbow acceleration p99 [m/s²]"),
        ("actual_elbow_lateral_range_m", 100.0, "Elbow lateral range [cm]"),
        ("tail_actual_ee_p2p_m", 100.0, "Tail EEF p2p [cm]"),
    )
    means = frame.groupby("profile").mean(numeric_only=True)
    table = pd.DataFrame(
        {
            LABELS[profile]: [means.loc[profile, metric] * scale for metric, scale, _ in specs]
            for profile in order
        },
        index=[title for _, _, title in specs],
    )
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "headline_baseline_means.csv")
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.3))
    x = np.arange(len(order))
    labels = [LABELS[p] for p in order]
    for axis, (metric, scale, title) in zip(axes.flat, specs, strict=True):
        values = [means.loc[profile, metric] * scale for profile in order]
        bars = axis.bar(x, values, color=[COLORS[p] for p in order], width=0.72)
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=11)
        axis.grid(axis="x", visible=False)
    fig.suptitle("IK Control Strategy Comparison", y=1.01)
    fig.tight_layout()
    save(fig, "03_headline_baseline_comparison.png")


def paired_effects(
    frame: pd.DataFrame,
    profiles: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    keys = ["scenario", "side"]
    base = frame[frame.profile == "current_deployment"].set_index(keys)
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
        "no_branch_regulation",
        "no_singularity_limit",
        "no_joint_braking",
        "no_kinetic_regularization",
        "driver_only_velocity",
    ]
    profile_labels = {profile: LABELS[profile] for profile in profiles}
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
        "Joint\nacceleration",
        "Elbow\nacceleration",
        "Elbow lateral\nrange",
        "Tail EEF\nmotion",
        "Driver-cap\noccupancy",
    ]
    effects = paired_effects(frame, profiles, metrics)
    effects.index = [profile_labels[p] for p in profiles]
    effects.columns = metric_labels
    effects.to_csv(TABLES / "ablation_relative_effects_percent.csv")
    display_effects = effects.mask(effects.abs() < 0.5, 0.0)
    fig, axis = plt.subplots(figsize=(12.7, 6.5))
    sns.heatmap(
        effects.clip(-150.0, 150.0),
        annot=display_effects,
        fmt=".0f",
        cmap="RdYlGn_r",
        center=0.0,
        vmin=-100.0,
        vmax=150.0,
        linewidths=0.5,
        cbar_kws={"label": "Change after removing mechanism [%]"},
        ax=axis,
    )
    axis.set_title("Single-Feature Ablation")
    axis.set_xlabel("")
    axis.set_ylabel("")
    fig.tight_layout()
    save(fig, "04_feature_ablation_heatmap.png")

    families = sorted(frame.family.unique())
    base = frame[frame.profile == "current_deployment"].set_index(["scenario", "side"])
    ddq = pd.DataFrame(index=[profile_labels[p] for p in profiles], columns=families)
    elbow = ddq.copy()
    for profile in profiles:
        other = frame[frame.profile == profile].set_index(["scenario", "side"])
        common = base.index.intersection(other.index)
        joined = other.loc[common].copy()
        joined["ddq_delta"] = other.loc[common, "actual_ddq_p99_rad_s2"] - base.loc[common, "actual_ddq_p99_rad_s2"]
        joined["elbow_delta"] = 100.0 * (other.loc[common, "actual_elbow_lateral_range_m"] - base.loc[common, "actual_elbow_lateral_range_m"])
        ddq.loc[profile_labels[profile]] = joined.groupby("family")["ddq_delta"].mean()
        elbow.loc[profile_labels[profile]] = joined.groupby("family")["elbow_delta"].mean()
    ddq = ddq.astype(float)
    elbow = elbow.astype(float)
    ddq.to_csv(TABLES / "ablation_ddq_delta_by_family.csv")
    elbow.to_csv(TABLES / "ablation_elbow_delta_by_family_cm.csv")
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 9.2), sharex=True)
    sns.heatmap(ddq, cmap="RdBu_r", center=0, annot=True, fmt=".1f", cbar_kws={"label": "Joint ddq p99 change [rad/s²]"}, ax=axes[0])
    axes[0].set_title("Joint-acceleration effect by trajectory family")
    sns.heatmap(elbow, cmap="RdBu_r", center=0, annot=True, fmt=".2f", cbar_kws={"label": "Elbow lateral-range change [cm]"}, ax=axes[1])
    axes[1].set_title("Elbow-branch effect by trajectory family")
    for axis in axes:
        axis.set_xlabel("")
        axis.set_ylabel("")
    fig.tight_layout()
    save(fig, "05_feature_effect_by_trajectory_family.png")


def plot_chest_suite() -> None:
    frame = read_summary("chest")
    metrics = [
        "actual_position_rmse_m",
        "actual_orientation_rmse_rad",
        "actual_ddq_p99_rad_s2",
        "actual_elbow_accel_p99_m_s2",
        "actual_elbow_lateral_range_m",
        "driver_limit_active_fraction",
        "tail_actual_ee_p2p_m",
    ]
    profile_means(frame, metrics).to_csv(TABLES / "chest_profile_means.csv")
    scenario = "chest_outward_flip_right_diagonal_v1p20_w8p0"
    profiles = (
        "current_deployment",
        "orientation_budget_0p15",
        "no_branch_regulation",
        "no_frame_error_bound",
    )
    labels = {
        "current_deployment": "PR default",
        "orientation_budget_0p15": "PR: orientation budget 0.15 rad",
        "no_branch_regulation": "PR w/o posture regulation",
        "no_frame_error_bound": "PR w/o 6D error bound",
    }
    traces = {profile: load_trace("chest", profile, scenario) for profile in profiles}
    fig, axes = plt.subplots(3, 2, figsize=(14.2, 10.8), sharex="col")
    for profile, trace in traces.items():
        times = trace["times"] - trace["times"][0]
        color = COLORS.get(profile, "#159A78")
        axes[0, 0].plot(times, 100.0 * position_error(trace), color=color, label=labels[profile])
        axes[1, 0].plot(times, np.rad2deg(orientation_error(trace)), color=color)
        elbow = trace["right_actual_elbow"]
        axes[2, 0].plot(times, 100.0 * (elbow[:, 1] - elbow[0, 1]), color=color)
        axes[0, 1].plot(times, max_abs(trace["right_actual_dq"]), color=color)
        axes[1, 1].plot(times, max_abs(trace["right_actual_ddq"]), color=color)
        active = np.any(trace["right_driver_limit_active_by_joint"], axis=1)
        axes[2, 1].plot(times, active.astype(float), color=color)
    axes[0, 0].set_ylabel("Position error [cm]")
    axes[1, 0].set_ylabel("Orientation error [deg]")
    axes[2, 0].set_ylabel(r"Elbow $y-y_0$ [cm]")
    axes[2, 0].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel(r"$\max_i |\dot q_i|$ [rad/s]")
    axes[1, 1].set_ylabel(r"$\max_i |\ddot q_i|$ [rad/s²]")
    axes[2, 1].set_ylabel("Any driver velocity cap active")
    axes[2, 1].set_xlabel("Time [s]")
    axes[0, 0].legend(loc="upper left", fontsize=11)
    axes[2, 1].set_ylim(-0.05, 1.05)
    fig.suptitle("Controller Response to Fast Wrist Roll and Translation", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    save(fig, "06_chest_wrist_error_modulation_timeseries.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    target = traces["current_deployment"]["right_target_pose"][:, :3]
    axes[0].plot(target[:, 0], target[:, 2], "k--", lw=2.2, label="target")
    axes[1].plot(target[:, 1], target[:, 2], "k--", lw=2.2, label="target")
    for profile, trace in traces.items():
        actual = trace["right_actual_pose"][:, :3]
        axes[0].plot(actual[:, 0], actual[:, 2], color=COLORS.get(profile), label=labels[profile])
        axes[1].plot(actual[:, 1], actual[:, 2], color=COLORS.get(profile))
    axes[0].set(xlabel="arm-origin x [m]", ylabel="arm-origin z [m]", title="Side view")
    axes[1].set(xlabel="arm-origin y [m]", ylabel="arm-origin z [m]", title="Front view")
    axes[0].legend(fontsize=11)
    for axis in axes:
        axis.set_aspect("equal", adjustable="datalim")
    fig.suptitle("End-Effector Paths in the Near-Chest Stress Case")
    fig.tight_layout()
    save(fig, "07_chest_wrist_eef_paths.png")


def _frozen_metric_table(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    columns = [
        "actual_position_rmse_m",
        "actual_position_max_m",
        "actual_orientation_rmse_rad",
        "actual_elbow_lateral_range_m",
        "actual_ddq_p99_rad_s2",
        "actual_elbow_accel_p99_m_s2",
        "driver_limit_active_fraction",
    ]
    return frame[(frame.scenario == scenario) & (frame.side == "right")].set_index("profile")[columns]


def plot_branch_regulation() -> None:
    scenario = "fast_retract_elbow_branch"
    profiles = (
        "current_deployment",
        "no_branch_regulation",
        "full_home_replacement_0p01",
    )
    traces = {profile: load_trace("frozen", profile, scenario) for profile in profiles}
    metrics = _frozen_metric_table(read_summary("frozen"), scenario).loc[list(profiles)]
    metrics.to_csv(TABLES / "branch_regulation_frozen_metrics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))
    for profile in profiles:
        trace = traces[profile]
        times = trace["times"] - trace["times"][0]
        elbow = trace["right_actual_elbow"]
        color = COLORS[profile]
        label = LABELS[profile]
        axes[0, 0].plot(elbow[:, 1], elbow[:, 2], color=color, label=label)
        axes[0, 1].plot(times, 100.0 * (elbow[:, 1] - elbow[0, 1]), color=color, label=label)
        axes[1, 0].plot(times, 100.0 * position_error(trace), color=color)
        axes[1, 1].plot(times, max_abs(trace["right_actual_ddq"]), color=color)
    axes[0, 0].set(xlabel="arm-origin y [m]", ylabel="arm-origin z [m]", title="Actual elbow path in the Y-Z plane")
    axes[0, 0].legend(fontsize=11)
    axes[0, 1].set(xlabel="Time [s]", ylabel=r"Elbow $y-y_0$ [cm]", title="Lateral displacement from the initial elbow")
    axes[1, 0].set(xlabel="Time [s]", ylabel="Position error [cm]", title="End-effector position error")
    axes[1, 1].set(xlabel="Time [s]", ylabel=r"$\max_i |\ddot q_i|$ [rad/s²]", title="Largest physical joint acceleration")
    fig.suptitle("Posture Regulation During Fast Retraction", y=1.01)
    fig.tight_layout()
    save(fig, "08_nullspace_branch_control.png")


def plot_singularity_case() -> None:
    scenario = "reach_right_p0p00_v0p80"
    profiles = ("current_deployment", "no_singularity_limit")
    traces = {profile: load_trace("screening", profile, scenario) for profile in profiles}
    fig, axes = plt.subplots(4, 1, figsize=(12.8, 10.2), sharex=True)
    reference = traces["current_deployment"]
    reference_times = reference["times"] - reference["times"][0]
    extension_indices = np.flatnonzero(reference["phase"] == 1)
    extension_end = (
        float(reference_times[extension_indices[-1]])
        if extension_indices.size
        else float(reference_times[-1])
    )
    for profile in profiles:
        trace = traces[profile]
        times = trace["times"] - trace["times"][0]
        color = COLORS[profile]
        focused = times <= extension_end
        faded = times >= extension_end
        values = (
            trace["right_actual_rho"],
            trace["right_actual_dq"][:, 0],
            trace["right_actual_dq"][:, 3],
            max_abs(trace["right_actual_ddq"]),
        )
        for axis_index, (axis, series) in enumerate(
            zip(axes, values, strict=True)
        ):
            if axis_index == 0:
                axis.plot(
                    times,
                    series,
                    color=color,
                    label=LABELS[profile],
                )
                continue
            axis.plot(times[focused], series[focused], color=color)
            axis.plot(times[faded], series[faded], color=color, alpha=0.25)
    times = reference["times"] - reference["times"][0]
    phase = reference["phase"]
    extension = times[phase == 1]
    retract = times[phase >= 3]
    for axis in axes:
        if extension.size:
            axis.axvspan(extension[0], extension[-1], color="#BFDBFE", alpha=0.22, label="extension" if axis is axes[0] else None)
        if retract.size:
            axis.axvspan(retract[0], retract[-1], color="#CBD5E1", alpha=0.35, label="retraction (secondary)" if axis is axes[0] else None)
    axes[0].axhspan(0.02, 0.08, color="#FDE68A", alpha=0.24)
    axes[0].axhline(0.02, color="#D14343", lw=1, ls="--")
    axes[0].set_ylabel(r"Geometric $\rho$")
    axes[0].legend(ncol=2, fontsize=11)
    axes[1].set_ylabel(r"Actual $\dot q_1$ [rad/s]")
    axes[2].set_ylabel(r"Actual $\dot q_4$ [rad/s]")
    axes[3].set_ylabel(r"$\max_i |\ddot q_i|$ [rad/s²]")
    axes[3].set_xlabel("Time [s]")
    fig.suptitle("Singularity Approach During Arm Extension", y=1.005)
    fig.tight_layout()
    save(fig, "09_singularity_reach_timeseries.png")


def command_path_error_components(
    trace: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Split actual position error into command-path tangent and normal parts."""
    times = trace["times"]
    command = trace["right_command_pose"][:, :3]
    actual = trace["right_actual_pose"][:, :3]
    command_velocity = np.gradient(command, times, axis=0)
    speed = np.linalg.norm(command_velocity, axis=1)
    valid = speed > 1.0e-5
    if not np.any(valid):
        return np.zeros_like(speed), np.linalg.norm(command - actual, axis=1)

    sample_indices = np.arange(len(times))
    tangent = np.empty_like(command_velocity)
    for axis in range(3):
        tangent[:, axis] = np.interp(
            sample_indices,
            sample_indices[valid],
            command_velocity[valid, axis],
        )
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True).clip(min=1.0e-12)

    command_error = command - actual
    along_track_lag = np.einsum("ij,ij->i", command_error, tangent)
    cross_track_error = command_error - along_track_lag[:, None] * tangent
    return along_track_lag, np.linalg.norm(cross_track_error, axis=1)


def plot_driver_coupling() -> None:
    frame = read_summary("driver")
    metrics = [
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
    profile_means(frame, metrics).to_csv(TABLES / "driver_coupling_profile_means.csv")
    scenario = "retract_diag_p0p10_v0p80"
    profiles = ("current_deployment", "driver_only_velocity", "no_velocity_limits")
    traces = {profile: load_trace("driver", profile, scenario) for profile in profiles}
    path_rows: list[dict[str, float | str]] = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), sharex="col")
    for profile in profiles:
        trace = traces[profile]
        times = trace["times"] - trace["times"][0]
        color = COLORS[profile]
        along_track_lag, cross_track_gap = command_path_error_components(trace)
        axes[0, 0].plot(
            times,
            100.0 * along_track_lag,
            color=color,
            label=LABELS[profile],
        )
        axes[0, 1].plot(times, 100.0 * cross_track_gap, color=color)
        axes[1, 0].plot(times, trace["right_command_dq"][:, 0], color=color)
        axes[1, 1].plot(times, trace["right_actual_dq"][:, 0], color=color)
        path_rows.append(
            {
                "profile": profile,
                "scenario": scenario,
                "along_track_lag_abs_max_cm": 100.0
                * float(np.max(np.abs(along_track_lag))),
                "cross_track_gap_max_cm": 100.0 * float(np.max(cross_track_gap)),
            }
        )
    pd.DataFrame(path_rows).to_csv(
        TABLES / "driver_path_tracking_components.csv", index=False
    )
    axes[0, 0].axhline(0.0, color="#64748B", lw=1.0, ls="--")
    axes[0, 0].set_ylabel("Along-track lag [cm]")
    axes[0, 0].legend(fontsize=11)
    axes[0, 1].set_ylabel("Cross-track gap [cm]")
    axes[1, 0].set_ylabel(r"Raw IK $\dot q_1$ [rad/s]")
    axes[1, 1].set_ylabel(r"Actual $\dot q_1$ [rad/s]")
    for axis in axes[1]:
        axis.axhline(2.0, color="#94A3B8", lw=1.0, ls=":")
        axis.axhline(-2.0, color="#94A3B8", lw=1.0, ls=":")
        axis.set_xlabel("Time [s]")
    fig.suptitle("Effect of Moving Velocity Limits Out of IK", y=1.005)
    fig.tight_layout()
    save(fig, "10_driver_limit_coupling_timeseries.png")


def plot_boundary_recovery() -> None:
    frame = read_summary("boundary")
    labels = {
        "recoverable_joint_limit": "Recoverable position + velocity",
        "native_position_plus_velocity": "Independent position + velocity",
        "configuration_only": "Position only",
    }
    solved = frame.groupby("profile")["solved"].agg(["sum", "count"]).reindex(labels)
    solved.to_csv(TABLES / "boundary_recovery_solved_counts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.4))
    x = np.arange(len(solved))
    bars = axes[0].bar(x, solved["sum"], color=["#159A78", "#D14343", "#E58B28"])
    axes[0].set_xticks(x, [labels[p] for p in solved.index], rotation=14, ha="right")
    axes[0].set_ylabel("Solved static conditions")
    axes[0].set_ylim(0, solved["count"].max() * 1.12)
    axes[0].bar_label(bars, labels=[f"{int(a)}/{int(b)}" for a, b in solved.to_numpy()])
    axes[0].set_title("Feasibility from an out-of-bound initial state")
    subset = frame[(frame.joint == 4) & (frame.boundary == "lower")]
    for profile, group in subset.groupby("profile"):
        axes[1].plot(group.offset_rad * 1000.0, group.remaining_violation_rad * 1000.0, marker="o", label=labels.get(profile, profile))
    axes[1].axhline(0.0, color="#111827", lw=1)
    axes[1].set_xlabel("Initial distance outside J4 limit [mrad]")
    axes[1].set_ylabel("Distance still outside position limit [mrad]")
    axes[1].set_title("J4 residual after one 4 ms outer update")
    axes[1].legend(fontsize=11)
    fig.suptitle("Joint-Limit Recovery Feasibility and Response", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.42)
    save(fig, "11_recoverable_joint_limit.png")


def braking_envelope(distance: np.ndarray, slowdown: float, exponent: float = 2.0) -> np.ndarray:
    u = np.clip(np.maximum(distance, 0.0) / slowdown, 0.0, 1.0)
    return np.power(u * u * (3.0 - 2.0 * u), exponent)


def plot_braking() -> None:
    frame = read_summary("braking")
    metrics = [
        "actual_min_joint_margin_rad",
        "actual_j6_dq_max_rad_s",
        "actual_ddq_p99_rad_s2",
        "actual_orientation_rmse_rad",
        "braking_active_fraction",
    ]
    profile_means(frame, metrics).to_csv(TABLES / "braking_profile_means.csv")
    distances = (0.08, 0.12, 0.20, 0.30)
    colors = dict(zip(distances, ("#159A78", "#1261A0", "#E58B28", "#D14343"), strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    margin_axis = np.linspace(0.0, 0.32, 300)
    for distance in distances:
        axes[0].plot(margin_axis, braking_envelope(margin_axis, distance), lw=2.2, color=colors[distance], label=f"{distance:.2f} rad")
    axes[0].plot(
        margin_axis,
        np.ones_like(margin_axis),
        color="#6B7280",
        ls="--",
        lw=1.8,
        label="PR w/o braking",
    )
    axes[0].set_xlabel("Distance to position limit [rad]")
    axes[0].set_ylabel("Allowed approach speed / joint speed cap")
    axes[0].set_title("Allowed speed while approaching a joint limit")
    axes[0].legend(title="Braking distance")
    fast = frame[frame.speed == frame.speed.max()].groupby("profile").mean(numeric_only=True)
    for distance in distances:
        profile = f"brake_distance_{distance:g}".replace(".", "p")
        axes[1].scatter(1000.0 * fast.loc[profile, "actual_min_joint_margin_rad"], fast.loc[profile, "actual_j6_dq_max_rad_s"], s=95, color=colors[distance], label=f"{distance:.2f} rad")
    neutral = (
        ("no_joint_braking", "PR w/o braking", "#6B7280", "o"),
        (
            "no_ik_velocity_or_braking",
            "PR w/o IK velocity limits / braking",
            "#111827",
            "X",
        ),
    )
    for profile, label, color, marker in neutral:
        axes[1].scatter(1000.0 * fast.loc[profile, "actual_min_joint_margin_rad"], fast.loc[profile, "actual_j6_dq_max_rad_s"], s=100, color=color, marker=marker, label=label)
    axes[1].set_xlabel("Minimum physical joint margin [mrad]")
    axes[1].set_ylabel(r"Maximum physical $|\dot q_6|$ [rad/s]")
    axes[1].set_title("12 rad/s wrist target: margin-speed trade-off")
    axes[1].legend(fontsize=10.5)
    fig.suptitle("Distance-Dependent Joint Braking", y=1.02)
    fig.tight_layout()
    save(fig, "12_joint_braking_envelope.png")


def _series(
    means: pd.DataFrame,
    mapping: list[tuple[float, str]],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([value for value, _ in mapping], dtype=float),
        np.asarray([means.loc[name, metric] for _, name in mapping], dtype=float),
    )


def _plot_sweep_metric(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    best: str = "min",
    **plot_kwargs: object,
) -> None:
    """Plot one sweep metric and mark its best observed sample."""
    best_index = int(np.nanargmax(y) if best == "max" else np.nanargmin(y))
    marker = plot_kwargs.get("marker")
    if marker is not None:
        plot_kwargs["markevery"] = [index for index in range(len(x)) if index != best_index]
        plot_kwargs.setdefault("markersize", 9.0)
    plot_kwargs.setdefault("linewidth", 2.4)
    line = axis.plot(x, y, **plot_kwargs)[0]
    axis.scatter(
        x[best_index],
        y[best_index],
        marker="*",
        s=275,
        color=line.get_color(),
        edgecolor="#111827",
        linewidth=1.25,
        zorder=6,
    )


def _mark_deployment_value(
    axis: plt.Axes,
    value: float,
    *,
    show_label: bool = False,
) -> None:
    """Mark PR default without adding a legend entry."""
    axis.axvline(value, color="#111827", ls="--", lw=1.2)
    if not show_label:
        return
    axis.text(
        value,
        0.035,
        "PR default",
        transform=axis.get_xaxis_transform(),
        rotation=0,
        ha="right",
        va="bottom",
        fontsize=16.0,
        color="#111827",
    )


def plot_parameter_sweeps() -> None:
    frame = read_summary("parameters")
    means = frame.groupby("profile").mean(numeric_only=True)
    means.to_csv(TABLES / "parameter_profile_means.csv")
    fig, axes = plt.subplots(2, 4, figsize=(21.0, 11.8))

    position = [(v, f"position_budget_{v:g}".replace(".", "p")) for v in (0.0, 0.010, 0.015, 0.020, 0.025, 0.030)]
    x, y = _series(means, position, "actual_position_rmse_m")
    _plot_sweep_metric(axes[0, 0], x, 100.0 * y, marker="o", label="position RMSE [cm]")
    _, y = _series(means, position, "actual_ddq_p99_rad_s2")
    _plot_sweep_metric(axes[0, 0], x, y / 5.0, marker="s", label="joint ddq p99 / 5")
    _mark_deployment_value(axes[0, 0], 0.020, show_label=True)
    axes[0, 0].set(xlabel="Position total budget [m]", title="Position error budget")
    axes[0, 0].margins(y=0.36)
    axes[0, 0].legend(loc="upper right", fontsize=15.0)

    orientation = [(v, f"orientation_budget_{v:g}".replace(".", "p")) for v in (0.0, 0.15, 0.20, 0.25, 0.30, 0.40)]
    x, y = _series(means, orientation, "actual_position_rmse_m")
    _plot_sweep_metric(axes[0, 1], x, 100.0 * y, marker="o", label="position RMSE [cm]")
    _, y = _series(means, orientation, "actual_orientation_rmse_rad")
    _plot_sweep_metric(axes[0, 1], x, np.rad2deg(y), marker="s", label="orientation RMSE [deg]")
    _mark_deployment_value(axes[0, 1], 0.25)
    axes[0, 1].set(xlabel="Orientation total budget [rad]", title="Orientation error budget")
    axes[0, 1].legend(fontsize=15.0)

    null_cost = [(v, f"null_cost_{v:g}".replace(".", "p")) for v in (0.0, 3.0, 7.0, 8.5, 12.0, 18.0)]
    x, y = _series(means, null_cost, "actual_elbow_lateral_range_m")
    _plot_sweep_metric(axes[0, 2], x, 100.0 * y, marker="o", label="elbow range [cm]")
    _, y = _series(means, null_cost, "actual_ddq_p99_rad_s2")
    _plot_sweep_metric(axes[0, 2], x, y / 5.0, marker="s", label="joint ddq p99 / 5")
    _mark_deployment_value(axes[0, 2], 8.5)
    axes[0, 2].set(xlabel="Exact-nullspace task cost", title="Nullspace regulation cost")
    axes[0, 2].legend(fontsize=15.0)

    singular = [(v, f"sing_rate_{v:g}".replace(".", "p")) for v in (0.0, 0.10, 0.18, 0.25, 0.35, 0.50)]
    x, y = _series(means, singular, "actual_min_rho")
    _plot_sweep_metric(axes[0, 3], x, y, marker="o", label=r"minimum $\rho$", best="max")
    axis2 = axes[0, 3].twinx()
    _, y2 = _series(means, singular, "actual_ddq_p99_rad_s2")
    _plot_sweep_metric(axis2, x, y2, marker="s", color="#D14343", label="joint ddq p99")
    _mark_deployment_value(axes[0, 3], 0.25)
    axes[0, 3].set(xlabel=r"Maximum allowed $-\dot\rho$", ylabel=r"minimum $\rho$", title="Singularity approach rate")
    axis2.set_ylabel("joint ddq p99 [rad/s²]", color="#D14343")

    braking = [(v, f"brake_distance_{v:g}".replace(".", "p")) for v in (0.08, 0.12, 0.20, 0.30, 0.50)]
    x, y = _series(means, braking, "actual_min_joint_margin_rad")
    _plot_sweep_metric(axes[1, 0], x, 1000.0 * y, marker="o", color="#1261A0", label="minimum margin", best="max")
    _, y = _series(means, braking, "actual_position_rmse_m")
    braking_rmse_axis = axes[1, 0].twinx()
    _plot_sweep_metric(braking_rmse_axis, x, 100.0 * y, marker="s", color="#D14343", label="position RMSE")
    _mark_deployment_value(axes[1, 0], 0.20)
    axes[1, 0].set(
        xlabel="Braking distance [rad]",
        ylabel="Minimum physical joint margin [mrad]",
        title="Joint braking distance",
    )
    braking_rmse_axis.set_ylabel("Position RMSE [cm]", color="#D14343")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    handles2, labels2 = braking_rmse_axis.get_legend_handles_labels()
    axes[1, 0].legend(handles + handles2, labels + labels2, fontsize=15.0)

    energy = [(0.0, "energy_0"), (1e-5, "energy_1em05"), (2e-5, "energy_2em05"), (3e-5, "energy_3em05"), (5e-5, "energy_5em05"), (1e-4, "energy_0p0001")]
    x, y = _series(means, energy, "actual_ddq_p99_rad_s2")
    scaled_x = 1e5 * x
    _plot_sweep_metric(axes[1, 1], scaled_x, y, marker="o", label="joint ddq p99")
    _, y = _series(means, energy, "actual_elbow_accel_p99_m_s2")
    _plot_sweep_metric(axes[1, 1], scaled_x, y, marker="s", label="elbow acceleration p99")
    _mark_deployment_value(axes[1, 1], 2.0)
    axes[1, 1].set(xlabel="Kinetic cost [×1e-5]", title="Kinetic-energy tie-breaker")
    axes[1, 1].legend(fontsize=15.0)

    caps = [(v, f"control_caps_x{v:g}".replace(".", "p")) for v in (0.75, 1.0, 1.25, 1.5)]
    x, y = _series(means, caps, "actual_position_rmse_m")
    _plot_sweep_metric(axes[1, 2], x, 100.0 * y, marker="o", label="position RMSE [cm]")
    _, y = _series(means, caps, "actual_ddq_p99_rad_s2")
    _plot_sweep_metric(axes[1, 2], x, y / 5.0, marker="s", label="joint ddq p99 / 5")
    _mark_deployment_value(axes[1, 2], 1.0)
    axes[1, 2].set(xlabel="Multiplier on all seven IK speed caps", title="IK velocity-envelope scale")
    axes[1, 2].legend(fontsize=15.0)

    task_names = ("task_cost_8_1", "task_cost_10_1", "task_cost_12_1p5", "task_cost_15_1p5", "task_cost_12_2")
    task_labels = ("8 / 1", "10 / 1", "12 / 1.5", "15 / 1.5", "12 / 2")
    x = np.arange(len(task_names))
    _plot_sweep_metric(axes[1, 3], x, 100.0 * means.loc[list(task_names), "actual_position_rmse_m"].to_numpy(), marker="o", label="position RMSE [cm]")
    _plot_sweep_metric(axes[1, 3], x, np.rad2deg(means.loc[list(task_names), "actual_orientation_rmse_rad"].to_numpy()), marker="s", label="orientation RMSE [deg]")
    _mark_deployment_value(axes[1, 3], 2)
    axes[1, 3].set_xticks(x, task_labels, rotation=15)
    axes[1, 3].set(xlabel="Position / orientation cost", title="Cartesian task weights")
    axes[1, 3].legend(fontsize=15.0)

    for axis in axes.flat:
        axis.tick_params(axis="both", labelsize=16.5)
        axis.xaxis.label.set_size(18.5)
        axis.yaxis.label.set_size(18.5)
    for axis in (axis2, braking_rmse_axis):
        axis.tick_params(axis="y", labelsize=16.5)
        axis.yaxis.label.set_size(18.5)

    fig.suptitle("Single-Parameter Sensitivity", y=1.01)
    fig.text(
        0.5,
        0.012,
        "Dashed line: PR default.  Star: best observed sample for that metric (not a joint multi-objective optimum).",
        ha="center",
        va="bottom",
        fontsize=16.5,
        color="#111827",
    )
    fig.tight_layout(rect=(0.0, 0.065, 1.0, 1.0), h_pad=1.7, w_pad=1.5)
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
    order = (
        "current_deployment",
        "ori0p20_sing0p18",
        "ori0p20_null10_sing0p18_brake0p12_energy3em5",
        "ori0p15_null12_sing0p18_brake0p12_energy5em5",
        "ori0p25_null10_sing0p18_brake0p12_energy3em5",
    )
    codes = ("PR", "A", "B", "C", "D")
    descriptions = (
        "PR default",
        "orientation=0.20 rad, singularity rate=0.18",
        "orientation=0.20, nullspace=10, singularity=0.18, braking=0.12, kinetic=3e-5",
        "orientation=0.15, nullspace=12, singularity=0.18, braking=0.12, kinetic=5e-5",
        "orientation=0.25, nullspace=10, singularity=0.18, braking=0.12, kinetic=3e-5",
    )
    mapping = pd.DataFrame(
        {"code": codes, "profile": order, "parameter_changes": descriptions}
    )
    mapping.to_csv(TABLES / "combined_candidate_definitions.csv", index=False)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14.5, 14.0),
        gridspec_kw={"height_ratios": (1.12, 1.0)},
    )
    palette = sns.color_palette("colorblind", len(order))
    offsets = ((8, -3), (7, -13), (7, 7), (-18, 7), (-18, -14))
    for profile, code, color, offset in zip(
        order, codes, palette, offsets, strict=True
    ):
        row = means.loc[profile]
        axes[1].scatter(100.0 * row.actual_position_rmse_m, row.actual_ddq_p99_rad_s2, s=90, color=color)
        axes[1].annotate(
            code,
            (100.0 * row.actual_position_rmse_m, row.actual_ddq_p99_rad_s2),
            xytext=offset,
            textcoords="offset points",
            fontsize=13,
            weight="bold",
        )
    axes[1].set_xlabel("Position RMSE [cm]")
    axes[1].set_ylabel("Joint acceleration p99 [rad/s²]")
    axes[1].set_title("Pareto view: tracking versus joint dynamics")
    axes[1].margins(x=0.08, y=0.14)
    base = means.loc["current_deployment", metrics]
    relative = 100.0 * (means.loc[list(order), metrics] - base) / base.abs().clip(lower=1e-9)
    relative.index = codes
    relative.columns = (
        "Position\nRMSE",
        "Orientation\nRMSE",
        "Joint ddq\np99",
        "Elbow\naccel. p99",
        "Elbow\nlateral range",
        "Driver velocity\ncap occupancy",
    )
    relative.to_csv(TABLES / "combined_candidate_relative_percent.csv")
    sns.heatmap(
        relative,
        annot=True,
        annot_kws={"fontsize": 13.5, "fontweight": "semibold"},
        fmt=".1f",
        cmap="RdYlGn_r",
        center=0.0,
        cbar_kws={"label": "Change from deployment [%]\n(green = lower / better)"},
        ax=axes[0],
    )
    axes[0].set_title("Metric changes relative to deployment")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="x", rotation=0)
    for label in axes[0].get_xticklabels():
        label.set_horizontalalignment("center")
    for axis in axes:
        axis.tick_params(axis="both", labelsize=13.5)
        axis.xaxis.label.set_size(15)
        axis.yaxis.label.set_size(15)
    colorbar = axes[0].collections[0].colorbar
    colorbar.ax.tick_params(labelsize=13)
    colorbar.set_label(
        "Change from deployment [%]\n(green = lower / better)", fontsize=14
    )
    definitions = "\n".join(
        f"{code}: {description}"
        for code, description in zip(codes, descriptions, strict=True)
    )
    fig.text(
        0.08,
        0.014,
        definitions,
        ha="left",
        va="bottom",
        fontsize=12.5,
        color="#111827",
        linespacing=1.3,
    )
    fig.suptitle("Combined Parameter Trade-offs", y=0.995)
    fig.tight_layout(rect=(0.0, 0.155, 1.0, 0.965), h_pad=1.5)
    save(fig, "14_combined_tuning_candidates.png")


def plot_symmetry_and_robustness() -> None:
    symmetry = read_summary("symmetry")
    exact = symmetry[symmetry.scenario.str.contains("exact_mirror") & (symmetry.profile == "current_deployment")]
    metrics = ["actual_position_rmse_m", "actual_orientation_rmse_rad", "actual_ddq_p99_rad_s2", "actual_elbow_accel_p99_m_s2", "actual_elbow_lateral_range_m"]
    by_side = exact.groupby("side")[metrics].mean()
    by_side.to_csv(TABLES / "exact_mirror_side_means.csv")
    robustness = read_summary("robustness")
    robust_metrics = ["actual_position_rmse_m", "actual_orientation_rmse_rad", "actual_ddq_p99_rad_s2", "driver_actual_q_gap_rms_rad"]
    robust = robustness.groupby("profile")[robust_metrics].mean()
    robust.to_csv(TABLES / "robustness_profile_means.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6))
    scaled = by_side.copy()
    scaled["actual_position_rmse_m"] *= 100.0
    scaled["actual_orientation_rmse_rad"] *= 180.0 / np.pi
    scaled["actual_elbow_lateral_range_m"] *= 100.0
    scaled.columns = ("pos RMSE [cm]", "ori RMSE [deg]", "ddq p99", "elbow accel p99", "elbow range [cm]")
    scaled.T.plot.bar(ax=axes[0], color=["#1261A0", "#D14343"])
    axes[0].set_title("Exactly mirrored right/left targets")
    axes[0].set_ylabel("Metric value")
    axes[0].tick_params(axis="x", rotation=18)
    order = ("current_deployment", "gravity_compensation", "state_delay_20ms", "command_delay_20ms", "actuator_gain_x0p7", "actuator_gain_x1p3", "driver_only_velocity")
    base = robust.loc["current_deployment"]
    delta = pd.DataFrame({profile: 100.0 * (robust.loc[profile] - base) / base.abs().clip(lower=1e-6) for profile in order[1:]}).T
    delta.columns = ("position", "orientation", "joint ddq", "q gap")
    sns.heatmap(delta, cmap="RdYlGn_r", center=0, annot=True, fmt=".0f", cbar_kws={"label": "Change from deployment [%]"}, ax=axes[1])
    axes[1].set_title("Plant and delay sensitivity")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    fig.suptitle("Side Equivalence and Plant Sensitivity", y=1.02)
    fig.tight_layout()
    save(fig, "15_symmetry_and_robustness.png")


def plot_solver_timing() -> None:
    frame = read_summary("screening")
    order = ("current_deployment", "driver_only_velocity", "strict_mainline", "no_frame_error_bound")
    means = frame[frame.profile.isin(order)].groupby("profile")[["solve_time_mean_ms", "solve_time_p95_ms"]].mean()
    means.to_csv(TABLES / "solver_timing_means.csv")
    x = np.arange(len(order))
    fig, axis = plt.subplots(figsize=(9.4, 4.8))
    axis.bar(x - 0.18, means.loc[list(order), "solve_time_mean_ms"], width=0.36, label="mean", color="#1261A0")
    axis.bar(x + 0.18, means.loc[list(order), "solve_time_p95_ms"], width=0.36, label="p95", color="#8BBCE5")
    axis.set_xticks(x, [LABELS[p] for p in order], rotation=15, ha="right")
    axis.set_ylabel("Outer solve time [ms]")
    axis.set_title("Five-Substep Outer-Solve Timing Across the Screening Suite")
    axis.legend()
    fig.tight_layout()
    save(fig, "16_solver_timing.png")


class SwivelMonitor:
    """Measure elbow rotation around the shoulder-to-wrist axis."""

    def __init__(self) -> None:
        self.setup = study.make_setup("bimanual")
        self.body_ids = np.array(
            [
                mujoco.mj_name2id(self.setup.model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_right_{link}")
                for link in ("link2", "link4", "link6")
            ],
            dtype=int,
        )

    def departure(self, arm_q: np.ndarray) -> float:
        positions = np.empty((len(arm_q), 3, 3), dtype=float)
        qpos = self.setup.data.qpos.copy()
        for index, q in enumerate(arm_q):
            self.setup.joint_resolver.set_qpos(qpos, np.append(q, 0.0), "right")
            self.setup.data.qpos[:] = qpos
            mujoco.mj_forward(self.setup.model, self.setup.data)
            positions[index] = self.setup.data.xpos[self.body_ids]
        shoulder, elbow, wrist = positions[:, 0], positions[:, 1], positions[:, 2]
        axis = wrist - shoulder
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        up = np.array([0.0, 0.0, 1.0])
        reference = up - np.sum(up * axis, axis=1, keepdims=True) * axis
        reference /= np.linalg.norm(reference, axis=1, keepdims=True)
        radial = elbow - shoulder
        radial -= np.sum(radial * axis, axis=1, keepdims=True) * axis
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        angle = np.unwrap(np.arctan2(np.einsum("ij,ij->i", axis, np.cross(reference, radial)), np.einsum("ij,ij->i", reference, radial)))
        return float(np.max(np.abs(angle - angle[0])))


NULL_GRID = re.compile(r"null_c(?P<cost>[0-9p]+)_r(?P<rate>[0-9p]+)_v(?P<speed>[0-9p]+)")


def _decode(value: str) -> float:
    return float(value.replace("p", "."))


def plot_nullspace_sweep() -> None:
    frame = read_summary("nullspace_sweep")
    monitor = SwivelMonitor()
    swivel: list[float] = []
    for row in frame.itertuples(index=False):
        trace = load_trace("nullspace_sweep", row.profile, row.scenario)
        swivel.append(monitor.departure(trace["right_actual_q"]))
    frame = frame.assign(actual_swivel_max_rad=swivel)
    frame.to_csv(TABLES / "nullspace_frozen_sweep_metrics.csv", index=False)
    rows: list[dict[str, float]] = []
    for row in frame.itertuples(index=False):
        match = NULL_GRID.fullmatch(row.profile)
        if match is None or not np.isclose(_decode(match.group("speed")), 1.0):
            continue
        values = row._asdict()
        values["cost"] = _decode(match.group("cost"))
        values["return_rate"] = _decode(match.group("rate"))
        rows.append(values)
    grid = pd.DataFrame(rows)
    grid.to_csv(TABLES / "nullspace_cost_return_grid.csv", index=False)
    specs = (
        ("actual_swivel_max_rad", "Elbow swivel departure [rad]", 0.75),
        ("actual_position_rmse_m", "EEF position RMSE [m]", 0.002),
        ("actual_orientation_rmse_rad", "EEF orientation RMSE [rad]", 0.010),
        ("actual_elbow_lateral_range_m", "Elbow lateral range [m]", 0.10),
        ("actual_ddq_p99_rad_s2", "Joint acceleration p99 [rad/s²]", 2.0),
        ("driver_limit_active_fraction", "Driver velocity-cap occupancy", 0.05),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.3), constrained_layout=True)
    for axis, (metric, title, color_floor) in zip(axes.flat, specs, strict=True):
        table = grid.pivot(index="return_rate", columns="cost", values=metric).sort_index(ascending=False)
        current_value = float(table.loc[1.6, 8.5])
        delta = table - current_value
        max_abs_delta = float(np.nanmax(np.abs(delta.to_numpy(dtype=float))))
        color_limit = max(color_floor, 1.15 * max_abs_delta)
        normalization = TwoSlopeNorm(
            vmin=-color_limit,
            vcenter=0.0,
            vmax=color_limit,
        )
        image = axis.imshow(
            delta,
            aspect="auto",
            cmap="RdYlGn_r",
            norm=normalization,
        )
        axis.set_title(f"{title}\nPR default = {current_value:.3g}", fontsize=11.5)
        axis.set_xlabel("Exact-nullspace posture cost")
        axis.set_ylabel("Home return rate [s⁻¹]")
        axis.set_xticks(range(len(table.columns)), table.columns)
        axis.set_yticks(range(len(table.index)), table.index)
        axis.grid(False)
        colormap = plt.get_cmap("RdYlGn_r")
        for row_index in range(len(table.index)):
            for column_index in range(len(table.columns)):
                value = float(delta.iloc[row_index, column_index])
                red, green, blue, _ = colormap(normalization(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                is_current = bool(
                    np.isclose(float(table.index[row_index]), 1.6)
                    and np.isclose(float(table.columns[column_index]), 8.5)
                )
                axis.text(
                    column_index,
                    row_index,
                    f"PR default\n{current_value:.3g}"
                    if is_current
                    else f"{value:+.3g}",
                    ha="center",
                    va="center",
                    color="black" if luminance > 0.52 else "white",
                    fontsize=10.5 if is_current else 11.5,
                    weight="semibold",
                )
                if is_current:
                    axis.add_patch(
                        patches.Rectangle(
                            (column_index - 0.5, row_index - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            edgecolor="#2563EB",
                            linewidth=2.2,
                        )
                    )
        colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
        colorbar.set_label(
            "Change from PR default\n(lower is better)", fontsize=10.0
        )
    fig.suptitle("Nullspace Regulation Parameter Sweep", fontsize=15)
    save(fig, "17_nullspace_targeted_tuning.png")


def plot_nullspace_validation() -> None:
    frame = read_summary("nullspace_validation")
    profiles = (
        "no_branch_regulation",
        "null_c5_r1p6_v1p0",
        "null_c8p5_r0p8_v0p6",
        "null_c12_r1p6_v1p0",
        "null_c8p5_r2p4_v1p0",
    )
    labels = (
        "PR w/o posture regulation",
        "cost 5 / return 1.6 / max 1.0",
        "cost 8.5 / return 0.8 / max 0.6",
        "cost 12 / return 1.6 / max 1.0",
        "cost 8.5 / return 2.4 / max 1.0",
    )
    metrics = (
        ("actual_position_rmse_m", 100.0, "Position RMSE change [cm]"),
        ("actual_orientation_rmse_rad", 180.0 / np.pi, "Orientation RMSE change [deg]"),
        ("actual_elbow_lateral_range_m", 100.0, "Elbow lateral-range change [cm]"),
        ("actual_ddq_p99_rad_s2", 1.0, "Joint acceleration p99 change [rad/s²]"),
    )
    base = frame[frame.profile == "current_deployment"].groupby("family").mean(numeric_only=True)
    family_labels = {
        "chest_wrist_outward": "Chest\nwrist roll",
        "extended_translation": "Extended\ntranslation",
        "extended_wrist": "Extended\nwrist roll",
        "normal_workspace": "Normal\nworkspace",
        "normal_wrist": "Normal\nwrist roll",
        "reach": "Arm\nextension",
        "retract": "Fast\nretraction",
    }
    fig, axes = plt.subplots(4, 1, figsize=(18.5, 15.5))
    exported: list[pd.DataFrame] = []
    for axis_index, (axis, (metric, scale, title)) in enumerate(
        zip(axes.flat, metrics, strict=True)
    ):
        table = pd.DataFrame(index=labels, columns=base.index, dtype=float)
        for profile, label in zip(profiles, labels, strict=True):
            other = frame[frame.profile == profile].groupby("family").mean(numeric_only=True)
            table.loc[label] = scale * (other.loc[base.index, metric] - base[metric])
        display_table = table.mask(table.abs() < 0.005, 0.0)
        table = table.rename(columns=family_labels)
        display_table = display_table.rename(columns=family_labels)
        sns.heatmap(
            table,
            annot=display_table,
            annot_kws={"fontsize": 13.5, "fontweight": "semibold"},
            fmt=".2f",
            cmap="RdYlGn_r",
            center=0.0,
            cbar_kws={"label": title},
            ax=axis,
        )
        axis.set_title(title.replace(" change", ""))
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="both", labelsize=13.5)
        axis.tick_params(axis="x", rotation=0)
        axis.tick_params(axis="y", rotation=0)
        colorbar = axis.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=13)
        colorbar.set_label(f"{title}\n(green = improvement)", fontsize=13.5)
        if axis_index < len(metrics) - 1:
            axis.set_xticklabels([])
        exported.append(table.assign(metric=metric))
    pd.concat(exported).to_csv(TABLES / "nullspace_cross_validation_deltas.csv")
    fig.suptitle("Posture-Regulation Candidate Cross-Validation", y=1.005)
    fig.tight_layout(rect=(0.015, 0.0, 1.0, 0.985))
    save(fig, "18_nullspace_cross_validation.png")


def plot_frozen_showcases() -> None:
    frame = read_summary("frozen")
    chest_scenario = "near_chest_fast_wrist_roll"
    retract_scenario = "fast_retract_elbow_branch"
    chest_profiles = ("current_deployment", "strict_mainline", "no_frame_error_bound")
    retract_profiles = ("current_deployment", "strict_mainline", "no_branch_regulation")
    chest_metrics = _frozen_metric_table(frame, chest_scenario).loc[list(chest_profiles)]
    retract_metrics = _frozen_metric_table(frame, retract_scenario).loc[list(retract_profiles)]
    chest_metrics.to_csv(TABLES / "near_chest_frozen_controller_metrics.csv")
    retract_metrics.to_csv(TABLES / "fast_retract_frozen_controller_metrics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    panels = (
        (axes[0, 0], chest_metrics.actual_position_max_m * 100.0, "Near-chest: maximum position error", "cm"),
        (axes[0, 1], chest_metrics.actual_ddq_p99_rad_s2, "Near-chest: joint acceleration p99", "rad/s²"),
        (axes[1, 0], retract_metrics.actual_elbow_lateral_range_m * 100.0, "Fast retract: elbow lateral range", "cm"),
        (axes[1, 1], retract_metrics.actual_position_max_m * 100.0, "Fast retract: maximum position error", "cm"),
    )
    for axis, values, title, unit in panels:
        profiles = tuple(values.index)
        bars = axis.bar([LABELS[p] for p in profiles], values.to_numpy(), color=[COLORS[p] for p in profiles], width=0.68)
        axis.set_title(title, fontsize=12)
        axis.set_ylabel(unit)
        axis.tick_params(axis="x", rotation=12)
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=11)
    fig.suptitle("Recorded Reference-Command Controller Comparisons", y=1.01)
    fig.tight_layout()
    save(fig, "33_final_showcase_controller_comparison.png")

    chest = {profile: load_trace("frozen", profile, chest_scenario) for profile in chest_profiles}
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.5))
    for profile in chest_profiles:
        trace = chest[profile]
        times = trace["times"] - trace["times"][0]
        color = COLORS[profile]
        axes[0, 0].plot(times, 100.0 * position_error(trace), color=color, label=LABELS[profile])
        axes[0, 1].plot(times, np.rad2deg(orientation_error(trace)), color=color)
        axes[1, 0].plot(times, max_abs(trace["right_actual_ddq"]), color=color)
        actual = trace["right_actual_pose"][:, :3]
        axes[1, 1].plot(100.0 * actual[:, 1], 100.0 * actual[:, 2], color=color, label=LABELS[profile])
    target = chest["current_deployment"]["right_target_pose"][:, :3]
    axes[1, 1].plot(100.0 * target[:, 1], 100.0 * target[:, 2], color="#111827", ls="--", lw=2.0, label="Target")
    axes[0, 0].set(xlabel="Time [s]", ylabel="Position error [cm]", title="End-effector position error")
    axes[0, 0].legend(loc="upper left", fontsize=11)
    axes[0, 1].set(xlabel="Time [s]", ylabel="Orientation error [deg]", title="End-effector orientation error")
    axes[1, 0].set(xlabel="Time [s]", ylabel=r"$\max_i |\ddot q_i|$ [rad/s²]", title="Largest physical joint acceleration")
    axes[1, 1].set(xlabel="arm-origin y [cm]", ylabel="arm-origin z [cm]", title="Actual end-effector path in the Y-Z plane")
    axes[1, 1].set_aspect("equal", adjustable="datalim")
    axes[1, 1].legend(fontsize=11)
    fig.suptitle("Near-Chest Fast Wrist-Roll Benchmark", y=1.01)
    fig.tight_layout()
    save(fig, "34_chest_flip_benchmark_timeseries_and_path.png")

    retract = {profile: load_trace("frozen", profile, retract_scenario) for profile in retract_profiles}
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.5))
    for profile in retract_profiles:
        trace = retract[profile]
        times = trace["times"] - trace["times"][0]
        color = COLORS[profile]
        elbow = trace["right_actual_elbow"]
        axes[0, 0].plot(elbow[:, 1], elbow[:, 2], color=color, label=LABELS[profile])
        axes[0, 1].plot(times, 100.0 * (elbow[:, 1] - elbow[0, 1]), color=color)
        axes[1, 0].plot(times, 100.0 * position_error(trace), color=color)
        axes[1, 1].plot(times, max_abs(trace["right_actual_ddq"]), color=color)
    axes[0, 0].set(xlabel="arm-origin y [m]", ylabel="arm-origin z [m]", title="Actual elbow path in the Y-Z plane")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend(fontsize=11)
    axes[0, 1].set(xlabel="Time [s]", ylabel=r"Elbow $y-y_0$ [cm]", title="Lateral displacement from the initial elbow")
    axes[1, 0].set(xlabel="Time [s]", ylabel="Position error [cm]", title="End-effector position error")
    axes[1, 1].set(xlabel="Time [s]", ylabel=r"$\max_i |\ddot q_i|$ [rad/s²]", title="Largest physical joint acceleration")
    fig.suptitle("Fast-Retract Elbow-Branch Benchmark", y=1.01)
    fig.tight_layout()
    save(fig, "35_fast_retract_benchmark_timeseries_and_path.png")


def main() -> None:
    global RESULTS, REPORT, ASSETS, TABLES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    RESULTS = args.results.resolve()
    REPORT = args.report_dir.resolve()
    ASSETS = REPORT / "assets"
    TABLES = REPORT / "tables"
    ASSETS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_architecture()
    plot_trajectory_catalog()
    plot_headline_comparison()
    plot_ablation_heatmaps()
    plot_chest_suite()
    plot_branch_regulation()
    plot_singularity_case()
    plot_driver_coupling()
    plot_boundary_recovery()
    plot_braking()
    plot_parameter_sweeps()
    plot_candidates()
    plot_symmetry_and_robustness()
    plot_solver_timing()
    plot_nullspace_sweep()
    plot_nullspace_validation()
    plot_frozen_showcases()
    print(f"Wrote figures to {ASSETS}")
    print(f"Wrote tables to {TABLES}")


if __name__ == "__main__":
    main()
