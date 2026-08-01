#!/usr/bin/env python3
"""Rerun the selected showcases against a clean ori/main velocity baseline."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pr_showcase_selection_study as selection
import render_pr_showcase_candidates as showcase_render
import render_videos as video
import study


RETRACT_ROOT = HERE / "results" / "pr_showcase_selection_20260731"
CHEST_ROOT = HERE / "results" / "episode73_triphasic_20260731"
SELECTED_ROOT = RETRACT_ROOT / "selected"


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
        speed = float(
            np.max(
                np.linalg.norm(
                    np.diff(target_right[:, :3], axis=0),
                    axis=1,
                )
                / study.CONTROL_DT
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


def _run_baseline() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    profile = selection._ori_main_velocity_profile()
    chest = _load_scenario(
        SELECTED_ROOT / "chest_ep73_triphasic98_v1.npz",
        name="episode73_triphasic_98s_straight_position",
        family="recorded_chest_counterfactual",
        description=(
            "Episode-73 recorded wrist orientation with a smooth "
            "start-to-end position path."
        ),
    )
    retract = _load_scenario(
        SELECTED_ROOT / "retract_ep75_event32_v1.npz",
        name="recorded_ep75_right_retract32_straight_retract",
        family="recorded_retract",
        description=(
            "Episode-75 smooth retract chord with recorded wrist orientation."
        ),
    )
    chest_rows = study.run_cached(
        profile,
        chest,
        CHEST_ROOT,
        save_full_trace=True,
    )
    retract_rows = study.run_cached(
        profile,
        retract,
        RETRACT_ROOT,
        save_full_trace=True,
    )
    metadata = {
        "profile": asdict(profile),
        "timing_equivalence": (
            "ori/main converts v_phys by max_iters * solver_dt * tick_hz; "
            "the current solver uses outer_dt / max_iters. Both produce "
            "the same per-substep displacement cap v_phys / "
            "(max_iters * tick_hz)."
        ),
        "chest_metrics": chest_rows,
        "retract_metrics": retract_rows,
    }
    output = SELECTED_ROOT / "corrected_ori_main_baseline_v2.json"
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return chest_rows, retract_rows


def _render() -> tuple[Path, Path]:
    chest = video.Comparison(
        name=(
            "showcase_selected_v2_chest_ep73_triphasic98_"
            "corrected_ori_main_comparison"
        ),
        suite="episode73_triphasic",
        scenario="episode73_triphasic_98s_straight_position",
        panels=(
            video.Panel(
                "pr_current_dataflow",
                "Current PR: exact active dataflow settings",
            ),
            video.Panel(
                "ori_main_defaults_with_current_velocity",
                "ori/main defaults + current IK velocity limits",
            ),
            video.Panel(
                "recorded_hardware",
                "Recorded source: original command + hardware",
            ),
            video.Panel(
                "pr_current_no_frame_bound",
                "Current PR: 6D frame bound disabled",
            ),
        ),
        caption=(
            "Episode-73 chest wrist flip. The ori/main baseline uses standard "
            "position/velocity limits without PR braking or measured-state safety."
        ),
        slowdown=2.0,
        show_elbow=True,
        show_max_error=True,
        columns=2,
    )
    retract = video.Comparison(
        name=(
            "showcase_selected_v2_retract_ep75_event32_"
            "corrected_ori_main_comparison"
        ),
        suite=showcase_render.SUITE,
        scenario="recorded_ep75_right_retract32_straight_retract",
        panels=(
            video.Panel(
                "pr_current_dataflow",
                "Current PR: exact active dataflow settings",
            ),
            video.Panel(
                "ori_main_defaults_with_current_velocity",
                "ori/main defaults + current IK velocity limits",
            ),
            video.Panel(
                "recorded_mainline_hardware",
                "Recorded hardware source (episode 75)",
            ),
            video.Panel(
                "pr_current_no_nullspace",
                "Current PR: nullspace home disabled",
            ),
        ),
        caption=(
            "Episode-75 fast retract. The ori/main baseline uses standard "
            "position/velocity limits without PR braking or measured-state safety."
        ),
        slowdown=2.0,
        show_elbow=True,
        show_max_error=True,
        replay_rear_view=True,
        columns=2,
    )
    return video.comparison_video(chest), video.comparison_video(retract)


def main() -> None:
    chest_rows, retract_rows = _run_baseline()
    chest_video, retract_video = _render()
    print(f"Chest metrics: {chest_rows}")
    print(f"Retract metrics: {retract_rows}")
    print(f"Wrote {chest_video}")
    print(f"Wrote {retract_video}")


if __name__ == "__main__":
    main()
