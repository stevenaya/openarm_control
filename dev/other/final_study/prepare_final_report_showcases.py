#!/usr/bin/env python3
"""Rerun and render the two frozen showcase trajectories for the final report."""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pr_showcase_selection_study as selection
import render_videos as video
import study


ROOT = HERE / "results" / "final_report_showcases_20260731"
BROAD_ROOT = HERE / "results" / "final_report_mainline_broad_20260731"
REPLICATION_ROOT = (
    HERE / "results" / "final_report_retract_replication_20260731"
)
SENSITIVITY_ROOT = (
    HERE / "results" / "final_report_showcase_sensitivity_20260731"
)
SCREENING_ROOT = HERE / "results" / "current_pr_20260730" / "screening"
SUITE = "final_report_showcases"
video.SUITE_ROOTS[SUITE] = ROOT

CHEST_SOURCE = (
    HERE
    / "results"
    / "pr_showcase_selection_20260731"
    / "selected"
    / "chest_ep73_triphasic98_v1.npz"
)
RETRACT_SOURCE = (
    HERE
    / "results"
    / "corrected_retract_showcase_screen_20260731"
    / "frozen_scenarios"
    / "recorded_ep75_right_retract30_straight_retract.npz"
)

CHEST_SCENARIO = "episode73_triphasic_98s_straight_position"
RETRACT_SCENARIO = "recorded_ep75_right_retract30_straight_retract"
MAINLINE_PROFILE = "ori_main_defaults_with_pr_joint_limits"
REPORT_ASSET = Path(
    "/home/hilab/workdir.evaluation/note/openarm_control/final_report/assets/"
    "33_final_showcase_controller_comparison.png"
)
HEADLINE_ASSET = Path(
    "/home/hilab/workdir.evaluation/note/openarm_control/final_report/assets/"
    "03_headline_baseline_comparison.png"
)
HEADLINE_TABLE = Path(
    "/home/hilab/workdir.evaluation/note/openarm_control/final_report/tables/"
    "headline_baseline_means.csv"
)
CHEST_TRACE_ASSET = Path(
    "/home/hilab/workdir.evaluation/note/openarm_control/final_report/assets/"
    "34_chest_flip_benchmark_timeseries_and_path.png"
)
RETRACT_TRACE_ASSET = Path(
    "/home/hilab/workdir.evaluation/note/openarm_control/final_report/assets/"
    "35_fast_retract_benchmark_timeseries_and_path.png"
)


def _load_scenario(
    path: Path,
    *,
    name: str,
    family: str,
    description: str,
) -> study.Scenario:
    with np.load(path, allow_pickle=False) as payload:
        target_right = payload["target_right"].copy()
        times = payload["times"].copy()
        sample_dt = np.diff(times)
        speed = float(
            np.max(
                np.linalg.norm(
                    np.diff(target_right[:, :3], axis=0),
                    axis=1,
                )
                / sample_dt[:, None]
            )
        )
        return study.Scenario(
            name=name,
            family=family,
            mode="right",
            speed=speed,
            times=times,
            phase=payload["phase"].copy(),
            target_right=target_right,
            target_left=payload["target_left"].copy(),
            initial_right=payload["initial_right"].copy(),
            initial_left=payload["initial_left"].copy(),
            description=description,
        )


def _peak_angular_speed(poses: np.ndarray, times: np.ndarray) -> float:
    angles = np.asarray(
        [
            study.quaternion_angle(previous, current)
            for previous, current in zip(
                poses[:-1, 3:],
                poses[1:, 3:],
                strict=True,
            )
        ]
    )
    return float(np.max(angles / np.diff(times)))


def _ori_main_profile() -> study.Profile:
    """Use mainline baseline tasks with the PR recoverable limit and no braking."""
    return study.make_profile(
        MAINLINE_PROFILE,
        limit_style="recoverable",
        velocity_caps=study.CONTROL_CAPS,
        use_measured_state=False,
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=0.01,
        damping=0.25,
        posture_cost=0.01,
        frame_position_error_limit=0.0,
        frame_orientation_error_limit=0.0,
        nullspace_cost=0.0,
        joint_braking=False,
        singularity_max_approach_rate=0.0,
        kinetic_energy_cost=0.0,
        description=(
            "Mainline baseline task defaults using the PR recoverable position/velocity "
            "limit and current physical velocity caps. Joint braking and "
            "measured-state safety are disabled."
        ),
    )


def _profiles() -> tuple[
    study.Profile,
    study.Profile,
    study.Profile,
    study.Profile,
]:
    current = selection._current_dataflow_profile()
    no_frame = replace(
        current,
        name="pr_current_no_frame_bound",
        overrides={
            **current.overrides,
            "frame_position_error_limit": 0.0,
            "frame_orientation_error_limit": 0.0,
        },
        description="Current PR dataflow settings without the 6D frame bound.",
    )
    no_nullspace = replace(
        current,
        name="pr_current_no_nullspace",
        overrides={**current.overrides, "nullspace_cost": 0.0},
        description=(
            "Current PR dataflow settings without nullspace home regulation."
        ),
    )
    return current, _ori_main_profile(), no_frame, no_nullspace


def _write_summary(rows: list[dict[str, object]]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (ROOT / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run() -> None:
    chest = _load_scenario(
        CHEST_SOURCE,
        name=CHEST_SCENARIO,
        family="recorded_chest_counterfactual",
        description=(
            "Episode-73 recorded wrist orientation with a smooth start-to-end "
            "position path."
        ),
    )
    retract = _load_scenario(
        RETRACT_SOURCE,
        name=RETRACT_SCENARIO,
        family="recorded_retract",
        description=(
            "Episode-75 event-30 smooth retract chord with recorded wrist "
            "orientation."
        ),
    )
    current, mainline, no_frame, no_nullspace = _profiles()
    matrix = (
        (current, chest),
        (mainline, chest),
        (no_frame, chest),
        (current, retract),
        (mainline, retract),
        (no_nullspace, retract),
    )
    rows: list[dict[str, object]] = []
    for profile, scenario in matrix:
        rows.extend(
            study.run_cached(
                profile,
                scenario,
                ROOT,
                save_full_trace=True,
            )
        )
    _write_summary(rows)
    manifest = {
        "control_period_s": study.CONTROL_DT,
        "qp_substeps": 5,
        "qp_substep_s": study.CONTROL_DT / 5.0,
        "driver_velocity_caps_rad_s": list(study.DRIVER_CAPS),
        "ik_velocity_caps_rad_s": list(study.CONTROL_CAPS),
        "mainline_definition": asdict(mainline),
        "scenarios": {
            chest.name: {
                "source": str(CHEST_SOURCE),
                "description": chest.description,
                "samples": int(chest.times.size),
                "duration_s": float(chest.times[-1] - chest.times[0]),
                "peak_target_linear_speed_m_s": chest.speed,
                "peak_target_angular_speed_rad_s": _peak_angular_speed(
                    chest.target_right,
                    chest.times,
                ),
            },
            retract.name: {
                "source": str(RETRACT_SOURCE),
                "description": retract.description,
                "samples": int(retract.times.size),
                "duration_s": float(retract.times[-1] - retract.times[0]),
                "peak_target_linear_speed_m_s": retract.speed,
                "peak_target_angular_speed_rad_s": _peak_angular_speed(
                    retract.target_right,
                    retract.times,
                ),
            },
        },
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def run_broad_mainline() -> None:
    study.run_matrix(
        [_ori_main_profile()],
        study.screening_scenarios(),
        BROAD_ROOT,
        workers=1,
        save_all_traces=False,
    )


def run_retract_replication() -> None:
    current = selection._current_dataflow_profile()
    mainline = _ori_main_profile()
    candidates = ((75, 31), (75, 57), (79, 38), (85, 58))
    rows: list[dict[str, object]] = []
    for episode, event_index in candidates:
        name = (
            f"recorded_ep{episode}_right_retract{event_index:02d}_"
            "straight_retract"
        )
        scenario = _load_scenario(
            (
                HERE
                / "results"
                / "corrected_retract_showcase_screen_20260731"
                / "frozen_scenarios"
                / f"{name}.npz"
            ),
            name=name,
            family="recorded_retract",
            description=(
                f"Episode-{episode} event-{event_index} smooth retract chord "
                "with recorded wrist orientation."
            ),
        )
        for profile in (current, mainline):
            rows.extend(
                study.run_cached(
                    profile,
                    scenario,
                    REPLICATION_ROOT,
                    save_full_trace=False,
                )
            )
    fields = sorted({key for row in rows for key in row})
    with (REPLICATION_ROOT / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_showcase_sensitivity() -> None:
    chest = _load_scenario(
        CHEST_SOURCE,
        name=CHEST_SCENARIO,
        family="recorded_chest_counterfactual",
        description="Frozen chest target",
    )
    retract = _load_scenario(
        RETRACT_SOURCE,
        name=RETRACT_SCENARIO,
        family="recorded_retract",
        description="Frozen retract target",
    )
    current = selection._current_dataflow_profile()

    def variant(
        name: str,
        *,
        use_measured_state: bool = True,
        **overrides: object,
    ) -> study.Profile:
        return replace(
            current,
            name=name,
            overrides={**current.overrides, **overrides},
            use_measured_state=use_measured_state,
            description=name,
        )

    no_braking = variant("pr_no_braking", joint_braking=False)
    no_braking_no_frame = variant(
        "pr_no_braking_no_frame",
        joint_braking=False,
        frame_position_error_limit=0.0,
        frame_orientation_error_limit=0.0,
    )
    no_braking_no_nullspace = variant(
        "pr_no_braking_no_nullspace",
        joint_braking=False,
        nullspace_cost=0.0,
    )
    no_braking_no_measured = variant(
        "pr_no_braking_no_measured",
        use_measured_state=False,
        joint_braking=False,
    )
    current_no_measured = variant(
        "pr_current_no_measured",
        use_measured_state=False,
    )
    current_no_measured = replace(
        current_no_measured,
        description="Current PR dataflow profile without measured-state safety",
    )
    mainline = _ori_main_profile()
    mainline_with_braking = replace(
        mainline,
        name="ori_main_strict_with_pr_braking",
        overrides={
            **mainline.overrides,
            "joint_braking": True,
            "joint_braking_distance": 0.2,
        },
        use_measured_state=True,
        description=(
            "Mainline baseline task defaults with PR braking and "
            "measured-state safety"
        ),
    )
    matrix = (
        (no_braking, chest),
        (no_braking_no_frame, chest),
        (no_braking_no_measured, chest),
        (current_no_measured, chest),
        (mainline_with_braking, chest),
        (no_braking, retract),
        (no_braking_no_nullspace, retract),
        (no_braking_no_measured, retract),
        (current_no_measured, retract),
        (mainline_with_braking, retract),
    )
    rows: list[dict[str, object]] = []
    for profile, scenario in matrix:
        rows.extend(
            study.run_cached(
                profile,
                scenario,
                SENSITIVITY_ROOT,
                save_full_trace=True,
            )
        )
    study.write_summary(SENSITIVITY_ROOT / "summary.csv", rows)


def plot_headline() -> Path:
    with (SCREENING_ROOT / "summary.csv").open(encoding="utf-8") as stream:
        screening_rows = list(csv.DictReader(stream))
    with (BROAD_ROOT / "summary.csv").open(encoding="utf-8") as stream:
        mainline_rows = list(csv.DictReader(stream))
    rows = screening_rows + mainline_rows
    profile_order = (
        "pr_full",
        "driver_only_ablation",
        MAINLINE_PROFILE,
        "upstream_tasks_tuned_costs",
    )
    labels = (
        "PR default",
        "PR w/o IK velocity limits",
        "Mainline baseline",
        "PR: tuned full-home posture",
    )
    specs = (
        ("actual_position_rmse_m", 100.0, "Position RMSE [cm]"),
        (
            "actual_orientation_rmse_rad",
            180.0 / np.pi,
            "Orientation RMSE [deg]",
        ),
        ("actual_ddq_p99_rad_s2", 1.0, "Joint acceleration p99 [rad/s²]"),
        (
            "actual_elbow_accel_p99_m_s2",
            1.0,
            "Elbow acceleration p99 [m/s²]",
        ),
        (
            "actual_elbow_lateral_range_m",
            100.0,
            "Elbow lateral range [cm]",
        ),
        ("tail_actual_ee_p2p_m", 100.0, "Tail EEF p2p [cm]"),
    )

    means: dict[tuple[str, str], float] = {}
    for profile in profile_order:
        profile_rows = [row for row in rows if row["profile"] == profile]
        for metric, scale, _ in specs:
            means[(profile, metric)] = scale * float(
                np.mean([float(row[metric]) for row in profile_rows])
            )

    HEADLINE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with HEADLINE_TABLE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("", *labels))
        for metric, _, title in specs:
            writer.writerow(
                (
                    title,
                    *[means[(profile, metric)] for profile in profile_order],
                )
            )

    colors = ("#148f77", "#4c78a8", "#c44536", "#d99a16")
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4))
    positions = np.arange(len(profile_order))
    for axis, (metric, _, title) in zip(axes.flat, specs, strict=True):
        metric_values = [
            means[(profile, metric)]
            for profile in profile_order
        ]
        bars = axis.bar(positions, metric_values, color=colors, width=0.72)
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        axis.grid(axis="x", visible=False)
    fig.suptitle(
        "PR default versus Mainline baseline across the broad suite",
        y=1.01,
    )
    fig.tight_layout()
    HEADLINE_ASSET.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(HEADLINE_ASSET, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return HEADLINE_ASSET


def plot() -> Path:
    with (ROOT / "summary.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {
        (row["scenario"], row["profile"]): row
        for row in rows
    }
    profiles = (
        "pr_current_dataflow",
        MAINLINE_PROFILE,
    )
    chest_profiles = (*profiles, "pr_current_no_frame_bound")
    retract_profiles = (*profiles, "pr_current_no_nullspace")
    chest_labels = (
        "PR default",
        "Mainline baseline",
        "PR w/o 6D error bound",
    )
    retract_labels = (
        "PR default",
        "Mainline baseline",
        "PR w/o posture regulation",
    )
    colors = ("#148f77", "#c44536", "#d99a16")

    def values(
        scenario: str,
        profile_names: tuple[str, ...],
        key: str,
        scale: float,
    ) -> list[float]:
        return [
            scale * float(lookup[(scenario, profile)][key])
            for profile in profile_names
        ]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0), constrained_layout=True)
    panels = (
        (
            axes[0, 0],
            values(
                CHEST_SCENARIO,
                chest_profiles,
                "actual_position_max_m",
                100.0,
            ),
            "Chest wrist flip: max position error",
            "cm",
            chest_labels,
        ),
        (
            axes[0, 1],
            values(
                CHEST_SCENARIO,
                chest_profiles,
                "actual_ddq_p99_rad_s2",
                1.0,
            ),
            "Chest wrist flip: joint acceleration p99",
            "rad/s²",
            chest_labels,
        ),
        (
            axes[1, 0],
            values(
                RETRACT_SCENARIO,
                retract_profiles,
                "actual_elbow_lateral_range_m",
                100.0,
            ),
            "Fast retract: elbow lateral range",
            "cm",
            retract_labels,
        ),
        (
            axes[1, 1],
            values(
                RETRACT_SCENARIO,
                retract_profiles,
                "actual_position_max_m",
                100.0,
            ),
            "Fast retract: max position error",
            "cm",
            retract_labels,
        ),
    )
    for axis, panel_values, title, unit, panel_labels in panels:
        bars = axis.bar(panel_labels, panel_values, color=colors, width=0.68)
        axis.set_title(title, fontsize=12)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(axis="x", rotation=12)
        for bar, value in zip(bars, panel_values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    fig.suptitle(
        "Frozen showcase trajectories under one shared plant and driver model",
        fontsize=14,
    )
    REPORT_ASSET.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_ASSET, dpi=180)
    plt.close(fig)
    return REPORT_ASSET


def _load_trace(profile: str, scenario: str) -> dict[str, np.ndarray]:
    matches = list(
        (ROOT / "traces").glob(f"*_{profile}_{scenario}.npz")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one trace for {profile}/{scenario}, got {len(matches)}."
        )
    with np.load(matches[0], allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def _position_error(trace: dict[str, np.ndarray]) -> np.ndarray:
    return np.linalg.norm(
        trace["right_target_pose"][:, :3]
        - trace["right_actual_pose"][:, :3],
        axis=1,
    )


def _orientation_error(trace: dict[str, np.ndarray]) -> np.ndarray:
    target = trace["right_target_pose"][:, 3:]
    actual = trace["right_actual_pose"][:, 3:]
    dot = np.abs(np.sum(target * actual, axis=1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def plot_showcase_traces() -> tuple[Path, Path]:
    profile_colors = {
        "pr_current_dataflow": "#148f77",
        MAINLINE_PROFILE: "#c44536",
        "pr_current_no_frame_bound": "#d99a16",
        "pr_current_no_nullspace": "#d99a16",
    }

    def traces_for(
        scenario: str,
        profiles: tuple[str, ...],
    ) -> dict[str, dict[str, np.ndarray]]:
        return {
            profile: _load_trace(profile, scenario)
            for profile in profiles
        }

    chest_profiles = (
        "pr_current_dataflow",
        MAINLINE_PROFILE,
        "pr_current_no_frame_bound",
    )
    chest_labels = {
        "pr_current_dataflow": "PR default",
        MAINLINE_PROFILE: "Mainline baseline",
        "pr_current_no_frame_bound": "PR w/o 6D error bound",
    }
    chest = traces_for(CHEST_SCENARIO, chest_profiles)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6))
    for profile in chest_profiles:
        trace = chest[profile]
        times = trace["times"] - trace["times"][0]
        color = profile_colors[profile]
        label = chest_labels[profile]
        axes[0, 0].plot(
            times,
            100.0 * _position_error(trace),
            color=color,
            label=label,
        )
        axes[0, 1].plot(
            times,
            np.rad2deg(_orientation_error(trace)),
            color=color,
            label=label,
        )
        axes[1, 0].plot(
            times,
            np.max(np.abs(trace["right_actual_dq"]), axis=1),
            color=color,
            label=label,
        )
        actual = trace["right_actual_pose"][:, :3]
        axes[1, 1].plot(
            100.0 * actual[:, 1],
            100.0 * actual[:, 2],
            color=color,
            label=label,
        )
    target = chest["pr_current_dataflow"]["right_target_pose"][:, :3]
    axes[1, 1].plot(
        100.0 * target[:, 1],
        100.0 * target[:, 2],
        color="#111827",
        linestyle="--",
        linewidth=2.0,
        label="Target",
    )
    axes[0, 0].set_title("Position tracking error")
    axes[0, 0].set_ylabel("Error [cm]")
    axes[0, 1].set_title("Orientation tracking error")
    axes[0, 1].set_ylabel("Error [deg]")
    axes[1, 0].set_title("Physical joint speed")
    axes[1, 0].set_ylabel(r"max $|\dot q|$ [rad/s]")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 1].set_title("Actual EEF path (front view)")
    axes[1, 1].set_xlabel("y [cm]")
    axes[1, 1].set_ylabel("z [cm]")
    axes[1, 1].set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[1, 1].legend(loc="best", fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.22)
    fig.suptitle(
        "Chest Flip Benchmark",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    CHEST_TRACE_ASSET.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHEST_TRACE_ASSET, dpi=180)
    plt.close(fig)

    retract_profiles = (
        "pr_current_dataflow",
        MAINLINE_PROFILE,
        "pr_current_no_nullspace",
    )
    retract_labels = {
        "pr_current_dataflow": "PR default",
        MAINLINE_PROFILE: "Mainline baseline",
        "pr_current_no_nullspace": "PR w/o posture regulation",
    }
    retract = traces_for(RETRACT_SCENARIO, retract_profiles)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6))
    for profile in retract_profiles:
        trace = retract[profile]
        times = trace["times"] - trace["times"][0]
        color = profile_colors[profile]
        label = retract_labels[profile]
        axes[0, 0].plot(
            times,
            100.0 * _position_error(trace),
            color=color,
            label=label,
        )
        elbow = trace["right_actual_elbow"]
        axes[0, 1].plot(
            times,
            100.0 * (elbow[:, 1] - elbow[0, 1]),
            color=color,
            label=label,
        )
        axes[1, 0].plot(
            times,
            np.max(np.abs(trace["right_actual_dq"]), axis=1),
            color=color,
            label=label,
        )
        axes[1, 1].plot(
            100.0 * elbow[:, 1],
            100.0 * elbow[:, 2],
            color=color,
            label=label,
        )
        axes[1, 1].scatter(
            100.0 * elbow[0, 1],
            100.0 * elbow[0, 2],
            color=color,
            edgecolor="white",
            linewidth=0.6,
            s=36,
            zorder=3,
        )
    axes[0, 0].set_title("Position tracking error")
    axes[0, 0].set_ylabel("Error [cm]")
    axes[0, 1].set_title("Signed elbow lateral displacement")
    axes[0, 1].set_ylabel(r"$y-y_0$ [cm]")
    axes[1, 0].set_title("Physical joint speed")
    axes[1, 0].set_ylabel(r"max $|\dot q|$ [rad/s]")
    axes[1, 0].text(
        0.5,
        -0.12,
        "Time [s]",
        ha="center",
        va="top",
        transform=axes[1, 0].transAxes,
    )
    axes[1, 1].set_title("Actual elbow branch path")
    axes[1, 1].set_ylabel("z [cm]")
    axes[1, 1].text(
        0.5,
        -0.12,
        "y [cm]",
        ha="center",
        va="top",
        transform=axes[1, 1].transAxes,
    )
    axes[1, 1].set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[1, 1].legend(loc="best", fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.22)
    fig.suptitle(
        "Fast Retract Benchmark",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(RETRACT_TRACE_ASSET, dpi=180)
    plt.close(fig)
    return CHEST_TRACE_ASSET, RETRACT_TRACE_ASSET


def render() -> tuple[Path, Path]:
    chest = video.Comparison(
        name="chest_flip_benchmark_controller_comparison",
        suite=SUITE,
        scenario=CHEST_SCENARIO,
        panels=(
            video.Panel("pr_current_dataflow", "PR default"),
            video.Panel(MAINLINE_PROFILE, "Mainline baseline"),
            video.Panel(
                "pr_current_no_frame_bound",
                "PR w/o 6D error bound",
            ),
        ),
        caption=(
            "Chest Flip Benchmark. All panels use the same frozen target and "
            "downstream driver limits."
        ),
        slowdown=2.0,
        show_elbow=True,
        show_max_error=True,
        columns=3,
    )
    retract = video.Comparison(
        name="fast_retract_benchmark_controller_comparison",
        suite=SUITE,
        scenario=RETRACT_SCENARIO,
        panels=(
            video.Panel("pr_current_dataflow", "PR default"),
            video.Panel(MAINLINE_PROFILE, "Mainline baseline"),
            video.Panel(
                "pr_current_no_nullspace",
                "PR w/o posture regulation",
            ),
        ),
        caption=(
            "Fast Retract Benchmark. All panels use the same frozen target and "
            "downstream driver limits."
        ),
        slowdown=2.0,
        show_elbow=True,
        show_max_error=True,
        replay_rear_view=True,
        columns=3,
    )
    return video.comparison_video(chest), video.comparison_video(retract)


def main() -> None:
    run_broad_mainline()
    run_retract_replication()
    run_showcase_sensitivity()
    run()
    headline_asset = plot_headline()
    asset = plot()
    chest_trace_asset, retract_trace_asset = plot_showcase_traces()
    chest_video, retract_video = render()
    catalog_video = video.ideal_reference_video()
    print(f"Wrote {ROOT / 'summary.csv'}")
    print(f"Wrote {ROOT / 'manifest.json'}")
    print(f"Wrote {HEADLINE_TABLE}")
    print(f"Wrote {headline_asset}")
    print(f"Wrote {asset}")
    print(f"Wrote {chest_trace_asset}")
    print(f"Wrote {retract_trace_asset}")
    print(f"Wrote {chest_video}")
    print(f"Wrote {retract_video}")
    print(f"Wrote {catalog_video}")


if __name__ == "__main__":
    main()
