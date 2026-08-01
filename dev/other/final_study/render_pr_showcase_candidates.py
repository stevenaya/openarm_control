#!/usr/bin/env python3
"""Render uniquely named videos for the final PR showcase selection."""

from __future__ import annotations

from pathlib import Path

import render_videos as video


SUITE = "pr_showcase_20260731"
video.SUITE_ROOTS[SUITE] = (
    Path(__file__).resolve().parent
    / "results"
    / "pr_showcase_selection_20260731"
)


def screening_comparisons() -> tuple[video.Comparison, ...]:
    """Return candidate A/B videos without replacing existing report videos."""
    candidates = (
        (
            "chest_ep77_event176",
            "recorded_ep77_right_chest176_position_smoothed",
            "Recorded chest flip, episode 77 event 176.",
            False,
        ),
        (
            "chest_ep77_event182",
            "recorded_ep77_right_chest182_position_smoothed",
            "Recorded chest flip, episode 77 event 182.",
            False,
        ),
        (
            "chest_ep83_event02",
            "recorded_ep83_right_chest02_position_smoothed",
            "Recent recorded chest flip, episode 83 event 2.",
            False,
        ),
        (
            "retract_ep77_event03",
            "recorded_ep77_right_retract03_straight_retract",
            "Recorded fast retract, episode 77 event 3.",
            True,
        ),
        (
            "retract_ep83_event20",
            "recorded_ep83_right_retract20_straight_retract",
            "Recent recorded fast retract, episode 83 event 20.",
            True,
        ),
        (
            "retract_ep83_event21",
            "recorded_ep83_right_retract21_straight_retract",
            "Recent recorded fast retract, episode 83 event 21.",
            True,
        ),
    )
    return tuple(
        video.Comparison(
            name=f"showcase_candidate_v1_{name}_pr_vs_mainline",
            suite=SUITE,
            scenario=scenario,
            panels=(
                video.Panel(
                    "pr_current_dataflow",
                    "Current PR: active dataflow settings",
                ),
                video.Panel(
                    "mainline_with_ik_velocity",
                    "Mainline + the same IK velocity limits",
                ),
            ),
            caption=caption,
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            replay_rear_view=replay_rear,
            columns=2,
        )
        for name, scenario, caption, replay_rear in candidates
    )


def legacy_comparisons() -> tuple[video.Comparison, ...]:
    """Return episode-73/75 candidates from the previous study round."""
    episode73 = tuple(
        video.Comparison(
            name=(
                "showcase_candidate_v1_chest_ep73_"
                f"triphasic{timestamp}_pr_vs_mainline"
            ),
            suite="episode73_triphasic",
            scenario=(
                f"episode73_triphasic_{timestamp}s_straight_position"
            ),
            panels=(
                video.Panel(
                    "pr_current_dataflow",
                    "Current PR: active dataflow settings",
                ),
                video.Panel(
                    "mainline_with_ik_velocity",
                    "Mainline + the same IK velocity limits",
                ),
            ),
            caption=(
                "Episode-73 recorded wrist orientation with a smooth "
                "start-to-end position path."
            ),
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            columns=2,
        )
        for timestamp in ("34", "98", "132")
    )
    episode75_scenarios = (
        (
            "retract_ep75_event32",
            "recorded_ep75_right_retract32_straight_retract",
        ),
        (
            "retract_ep75_event57",
            "recorded_ep75_right_retract57_straight_retract",
        ),
        (
            "retract_ep75_event41",
            "recorded_ep75_right_retract41_straight_retract",
        ),
        (
            "chest_ep75_event33",
            "recorded_ep75_right_chest33_position_smoothed",
        ),
    )
    episode75 = tuple(
        video.Comparison(
            name=f"showcase_candidate_v1_{name}_pr_vs_mainline",
            suite=SUITE,
            scenario=scenario,
            panels=(
                video.Panel(
                    "pr_current_dataflow",
                    "Current PR: active dataflow settings",
                ),
                video.Panel(
                    "mainline_with_ik_velocity",
                    "Mainline + the same IK velocity limits",
                ),
            ),
            caption="Episode-75 recorded-motion counterfactual.",
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            replay_rear_view=name.startswith("retract"),
            columns=2,
        )
        for name, scenario in episode75_scenarios
    )
    return episode73 + episode75


def selected_comparisons() -> tuple[video.Comparison, ...]:
    """Return the frozen final showcase comparisons."""
    chest_scenario = "episode73_triphasic_98s_straight_position"
    retract_scenario = "recorded_ep75_right_retract32_straight_retract"
    return (
        video.Comparison(
            name=(
                "showcase_selected_v1_chest_ep73_triphasic98_"
                "controller_comparison"
            ),
            suite="episode73_triphasic",
            scenario=chest_scenario,
            panels=(
                video.Panel(
                    "pr_current_dataflow",
                    "Current PR: exact active dataflow settings",
                ),
                video.Panel(
                    "mainline_with_ik_velocity",
                    "Mainline + the same IK velocity limits",
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
                "Episode-73 chest wrist flip. Counterfactual panels retain "
                "the recorded orientation and use a smooth position path."
            ),
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            columns=2,
        ),
        video.Comparison(
            name=(
                "showcase_selected_v1_retract_ep75_event32_"
                "controller_comparison"
            ),
            suite=SUITE,
            scenario=retract_scenario,
            panels=(
                video.Panel(
                    "pr_current_dataflow",
                    "Current PR: exact active dataflow settings",
                ),
                video.Panel(
                    "mainline_with_ik_velocity",
                    "Mainline + the same IK velocity limits",
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
                "Episode-75 fast retract using a smooth start-to-end "
                "position chord and the recorded orientation."
            ),
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            replay_rear_view=True,
            columns=2,
        ),
    )


def main() -> None:
    video.OUTPUT.mkdir(parents=True, exist_ok=True)
    for comparison in screening_comparisons():
        path = video.comparison_video(comparison)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
