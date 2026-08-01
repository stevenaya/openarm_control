#!/usr/bin/env python3
"""Render the strongest corrected-baseline right-arm retract candidates."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pr_showcase_selection_study as selection
import render_videos as video
import study


ROOT = HERE / "results" / "corrected_retract_showcase_screen_20260731"
SUITE = "corrected_retract_showcase_screen"
video.SUITE_ROOTS[SUITE] = ROOT

CANDIDATES = (
    (75, 30),
    (75, 31),
    (75, 57),
    (79, 38),
    (85, 58),
)


def _scenario(episode: int, event_index: int) -> study.Scenario:
    name = (
        f"recorded_ep{episode}_right_retract{event_index:02d}_"
        "straight_retract"
    )
    path = ROOT / "frozen_scenarios" / f"{name}.npz"
    with np.load(path, allow_pickle=False) as payload:
        target_right = payload["target_right"].copy()
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
            family="recorded_retract",
            mode="right",
            speed=speed,
            times=payload["times"].copy(),
            phase=payload["phase"].copy(),
            target_right=target_right,
            target_left=payload["target_left"].copy(),
            initial_right=payload["initial_right"].copy(),
            initial_left=payload["initial_left"].copy(),
            description=(
                f"Episode {episode} right-arm retract event {event_index}."
            ),
        )


def _no_nullspace_profile() -> study.Profile:
    current = selection._current_dataflow_profile()
    return replace(
        current,
        name="pr_current_no_nullspace",
        overrides={**current.overrides, "nullspace_cost": 0.0},
        description=(
            "Current PR dataflow settings with nullspace home regulation disabled."
        ),
    )


def main() -> None:
    no_nullspace = _no_nullspace_profile()
    video.OUTPUT.mkdir(parents=True, exist_ok=True)
    for episode, event_index in CANDIDATES:
        scenario = _scenario(episode, event_index)
        study.run_cached(
            no_nullspace,
            scenario,
            ROOT,
            save_full_trace=True,
        )
        comparison = video.Comparison(
            name=(
                f"showcase_retract_screen_v1_ep{episode}_event"
                f"{event_index:02d}_corrected_ori_main"
            ),
            suite=SUITE,
            scenario=scenario.name,
            panels=(
                video.Panel(
                    "pr_current_dataflow",
                    "Current PR: active dataflow settings",
                ),
                video.Panel(
                    "ori_main_defaults_with_current_velocity",
                    "ori/main defaults + current IK velocity limits",
                ),
                video.Panel(
                    "hardware",
                    f"Recorded hardware: episode {episode}",
                ),
                video.Panel(
                    "pr_current_no_nullspace",
                    "Current PR: nullspace home disabled",
                ),
            ),
            caption=(
                "Right-arm retract using a smooth position chord and recorded "
                "wrist orientation. Mainline braking and measured-state safety "
                "are disabled."
            ),
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            replay_rear_view=True,
            columns=2,
        )
        output = video.comparison_video(comparison)
        print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
