#!/usr/bin/env python3
"""Freeze the two selected PR showcase trajectories and their provenance."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import episode73_retract_pause_extend_study as episode73
import pr_showcase_selection_study as selection
import study


RESULT_ROOT = HERE / "results" / "pr_showcase_selection_20260731"
EPISODE73_ROOT = HERE / "results" / "episode73_triphasic_20260731"
SELECTED_DIR = RESULT_ROOT / "selected"
ASSET_DIR = (
    HERE.parents[2]
    / "note"
    / "openarm_control"
    / "final_report"
    / "assets"
)
VIDEO_DIR = ASSET_DIR.parent / "videos"

CHEST_SCENARIO = "episode73_triphasic_98s_straight_position"
RETRACT_SCENARIO = "recorded_ep75_right_retract32_straight_retract"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _metric_subset(
    rows: list[dict[str, str]],
    *,
    scenario_key: str,
    scenario_value: str,
    profiles: tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    keys = (
        "actual_elbow_lateral_range_m",
        "actual_position_max_m",
        "actual_position_rmse_m",
        "actual_orientation_rmse_rad",
        "actual_elbow_accel_p99_m_s2",
        "command_velocity_saturation_fraction",
        "solver_failures",
    )
    result: dict[str, dict[str, float | int]] = {}
    for row in rows:
        profile = row["profile"]
        if row[scenario_key] != scenario_value or profile not in profiles:
            continue
        result[profile] = {
            key: (
                int(float(row[key]))
                if key == "solver_failures"
                else float(row[key])
            )
            for key in keys
        }
    return result


def _freeze_chest() -> Path:
    record = episode73._record()
    pattern = next(
        item
        for item in episode73.detect_patterns()
        if item.name == "triphasic_98s"
    )
    scenario = episode73._scenario(record, pattern)
    path = SELECTED_DIR / "chest_ep73_triphasic98_v1.npz"
    window = slice(pattern.start, pattern.end)
    np.savez_compressed(
        path,
        times=scenario.times,
        phase=scenario.phase,
        target_right=scenario.target_right,
        target_left=scenario.target_left,
        initial_right=scenario.initial_right,
        initial_left=scenario.initial_left,
        source_time=record.time[window],
        source_action_q=record.action_q[window],
        source_obs_q=record.obs_q[window],
        source_obs_dq=record.obs_dq[window],
        source_action_pose=record.action_pose[window],
        source_obs_pose=record.obs_pose[window],
        source_action_elbow=record.action_elbow[window],
        source_obs_elbow=record.obs_elbow[window],
    )
    return path


def _freeze_retract() -> Path:
    source = RESULT_ROOT / "frozen_scenarios" / f"{RETRACT_SCENARIO}.npz"
    path = SELECTED_DIR / "retract_ep75_event32_v1.npz"
    shutil.copy2(source, path)
    return path


def _plot_metrics(
    chest: dict[str, dict[str, float | int]],
    retract: dict[str, dict[str, float | int]],
) -> Path:
    colors = ("#159d8c", "#e9948c", "#8a7d6d")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    chest_profiles = (
        "pr_current_dataflow",
        "pr_current_no_frame_bound",
        "mainline_with_ik_velocity",
    )
    chest_labels = ("Current PR", "PR, frame bound off", "Mainline + IK cap")
    chest_values = [
        100.0 * float(chest[profile]["actual_position_max_m"])
        for profile in chest_profiles
    ]
    axes[0].bar(chest_labels, chest_values, color=colors)
    axes[0].set_title("Chest flip: maximum position error")
    axes[0].set_ylabel("maximum error [cm]")
    axes[0].tick_params(axis="x", rotation=16)
    axes[0].grid(axis="y", alpha=0.25)
    for index, value in enumerate(chest_values):
        axes[0].text(index, value + 0.35, f"{value:.1f}", ha="center")

    retract_profiles = (
        "pr_current_dataflow",
        "pr_current_no_nullspace",
        "mainline_with_ik_velocity",
    )
    retract_labels = ("Current PR", "PR, nullspace off", "Mainline + IK cap")
    retract_values = [
        100.0 * float(retract[profile]["actual_elbow_lateral_range_m"])
        for profile in retract_profiles
    ]
    axes[1].bar(retract_labels, retract_values, color=colors)
    axes[1].set_title("Fast retract: elbow lateral range")
    axes[1].set_ylabel("lateral range [cm]")
    axes[1].tick_params(axis="x", rotation=16)
    axes[1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(retract_values):
        axes[1].text(index, value + 0.35, f"{value:.1f}", ha="center")

    fig.suptitle("Selected PR showcase trajectories (active dataflow profile)")
    fig.tight_layout()
    result_path = SELECTED_DIR / "selected_showcase_metrics.png"
    asset_path = ASSET_DIR / "32_selected_showcase_metrics.png"
    fig.savefig(result_path, dpi=180, bbox_inches="tight")
    fig.savefig(asset_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return result_path


def main() -> None:
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    chest_path = _freeze_chest()
    retract_path = _freeze_retract()

    chest_rows = _read_csv(
        EPISODE73_ROOT / "current_dataflow_metrics.csv"
    )
    retract_rows = _read_csv(RESULT_ROOT / "metrics.csv")
    chest_metrics = _metric_subset(
        chest_rows,
        scenario_key="name",
        scenario_value="triphasic_98s",
        profiles=(
            "pr_current_dataflow",
            "pr_current_no_frame_bound",
            "mainline_with_ik_velocity",
        ),
    )
    retract_metrics = _metric_subset(
        retract_rows,
        scenario_key="scenario",
        scenario_value=RETRACT_SCENARIO,
        profiles=(
            "pr_current_dataflow",
            "pr_current_no_nullspace",
            "mainline_with_ik_velocity",
        ),
    )
    plot_path = _plot_metrics(chest_metrics, retract_metrics)

    profiles = {
        profile.name: asdict(profile)
        for profile in selection._profiles()
        if profile.name
        in {
            "pr_current_dataflow",
            "mainline_with_ik_velocity",
            "pr_current_no_frame_bound",
            "pr_current_no_nullspace",
        }
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "openarm_control_revision": study._git_revision(),
        "dataflow_args": selection.DATAFLOW_ARGS,
        "profiles": profiles,
        "selected": {
            "chest": {
                "source_episode": 73,
                "source_window_s": [97.70274209976196, 100.34270811080933],
                "source_label": "triphasic_98s",
                "scenario": CHEST_SCENARIO,
                "artifact": str(chest_path),
                "target_semantics": (
                    "Recorded FK(action) orientation with a smooth "
                    "start-to-end position path; raw VR target unavailable."
                ),
                "selection_reason": (
                    "Clearly reproduces mainline retract-pause-recover "
                    "behavior while the exact active PR profile remains near "
                    "the smooth position command."
                ),
                "metrics": chest_metrics,
                "videos": [
                    str(
                        VIDEO_DIR
                        / "showcase_candidate_v1_chest_ep73_triphasic98_"
                        "pr_vs_mainline.mp4"
                    ),
                    str(
                        VIDEO_DIR
                        / "showcase_selected_v1_chest_ep73_triphasic98_"
                        "controller_comparison.mp4"
                    ),
                ],
            },
            "retract": {
                "source_episode": 75,
                "source_event": 32,
                "source_window_s": [90.3268370628357, 92.33481121063232],
                "scenario": RETRACT_SCENARIO,
                "artifact": str(retract_path),
                "target_semantics": (
                    "Smooth start-to-end position chord with the recorded "
                    "FK(action) orientation; raw VR target unavailable."
                ),
                "selection_reason": (
                    "Current PR preserves a compact elbow branch while "
                    "mainline with the same IK limits and PR without "
                    "nullspace regulation both swing outward."
                ),
                "metrics": retract_metrics,
                "videos": [
                    str(
                        VIDEO_DIR
                        / "showcase_candidate_v1_retract_ep75_event32_"
                        "pr_vs_mainline.mp4"
                    ),
                    str(
                        VIDEO_DIR
                        / "showcase_selected_v1_retract_ep75_event32_"
                        "controller_comparison.mp4"
                    ),
                ],
            },
        },
        "summary_figure": str(plot_path),
        "deferred": (
            "Root-cause work on residual chest wrist-flip failures is "
            "deferred until raw/filtered/final VR pose logging is available."
        ),
    }
    (SELECTED_DIR / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {chest_path}")
    print(f"Wrote {retract_path}")
    print(f"Wrote {SELECTED_DIR / 'selection_manifest.json'}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
