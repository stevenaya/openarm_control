#!/usr/bin/env python3
"""Render selected real-intervention traces beside controller replays."""

from __future__ import annotations

from render_videos import Comparison, Panel, comparison_video


def comparisons() -> tuple[Comparison, ...]:
    return (
        Comparison(
            name="recorded_chest_ep73_case62_controller_comparison",
            suite="recorded_replay",
            scenario="recorded_ep73_right_chest62_position_smoothed",
            panels=(
                Panel("recorded_hardware", "Recorded action / hardware"),
                Panel("pr_full_recorded", "Current PR"),
                Panel(
                    "mainline_ik_velocity_recorded",
                    "Mainline + IK velocity cap",
                ),
                Panel("mainline_recorded", "Mainline"),
            ),
            caption=(
                "Episode 73 chest flip case 62. Red target uses a "
                "position-smoothed intent proxy; recorded ghost is action q."
            ),
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            columns=2,
        ),
        Comparison(
            name="recorded_chest_ep73_case44_controller_comparison",
            suite="recorded_replay",
            scenario="recorded_ep73_right_chest44_position_smoothed",
            panels=(
                Panel("recorded_hardware", "Recorded action / hardware"),
                Panel("pr_full_recorded", "Current PR"),
                Panel(
                    "mainline_ik_velocity_recorded",
                    "Mainline + IK velocity cap",
                ),
                Panel("mainline_recorded", "Mainline"),
            ),
            caption=(
                "Episode 73 chest flip case 44. Red target uses a "
                "position-smoothed intent proxy; recorded ghost is action q."
            ),
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            columns=2,
        ),
        Comparison(
            name="recorded_chest_ep73_case40_controller_comparison",
            suite="recorded_replay",
            scenario="recorded_ep73_right_chest40_position_smoothed",
            panels=(
                Panel("recorded_hardware", "Recorded action / hardware"),
                Panel(
                    "pr_dataflow_recorded",
                    "Current PR: recorded config",
                ),
                Panel(
                    "mainline_ik_velocity_recorded",
                    "Mainline + IK velocity cap",
                ),
                Panel("mainline_recorded", "Mainline"),
            ),
            caption=(
                "Episode 73 case 40 has the largest same-time measured "
                "position error and orientation lag. Red target is a "
                "position-smoothed proxy because raw VR target was not "
                "recorded."
            ),
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            columns=2,
        ),
        Comparison(
            name="recorded_chest_ep73_case50_controller_comparison",
            suite="recorded_replay",
            scenario="recorded_ep73_right_chest50_position_smoothed",
            panels=(
                Panel("recorded_hardware", "Recorded action / hardware"),
                Panel(
                    "pr_dataflow_recorded",
                    "Current PR: recorded config",
                ),
                Panel(
                    "mainline_ik_velocity_recorded",
                    "Mainline + IK velocity cap",
                ),
                Panel("mainline_recorded", "Mainline"),
            ),
            caption=(
                "Episode 73 case 50 has the largest command-side "
                "short-horizon reversal during a fast wrist flip. Red target "
                "is a position-smoothed proxy."
            ),
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            columns=2,
        ),
        Comparison(
            name="recorded_chest_ep73_case85_controller_comparison",
            suite="recorded_replay",
            scenario="recorded_ep73_right_chest85_position_smoothed",
            panels=(
                Panel("recorded_hardware", "Recorded action / hardware"),
                Panel(
                    "pr_dataflow_recorded",
                    "Current PR: recorded config",
                ),
                Panel(
                    "mainline_ik_velocity_recorded",
                    "Mainline + IK velocity cap",
                ),
                Panel("mainline_recorded", "Mainline"),
            ),
            caption=(
                "Episode 73 case 85 has the largest measured cross-track "
                "departure from the recent command path. Red target is a "
                "position-smoothed proxy."
            ),
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            columns=2,
        ),
        Comparison(
            name="recorded_retract_ep75_case05_controller_comparison",
            suite="recorded_replay",
            scenario="recorded_ep75_right_retract05_straight_retract",
            panels=(
                Panel("recorded_hardware", "Recorded action / hardware"),
                Panel("pr_full_recorded", "Current PR"),
                Panel(
                    "mainline_ik_velocity_recorded",
                    "Mainline + IK velocity cap",
                ),
                Panel("mainline_recorded", "Mainline"),
            ),
            caption=(
                "Episode 75 retract case 5. Red target connects the recorded "
                "start/end poses without the recorded vertical dip."
            ),
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            replay_rear_view=True,
            columns=2,
        ),
        Comparison(
            name="recorded_retract_ep75_case41_controller_comparison",
            suite="recorded_replay",
            scenario="recorded_ep75_right_retract41_straight_retract",
            panels=(
                Panel("recorded_hardware", "Recorded action / hardware"),
                Panel("pr_full_recorded", "Current PR"),
                Panel(
                    "mainline_ik_velocity_recorded",
                    "Mainline + IK velocity cap",
                ),
                Panel("mainline_recorded", "Mainline"),
            ),
            caption=(
                "Episode 75 retract case 41. Red target connects the recorded "
                "start/end poses without the recorded vertical dip."
            ),
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            replay_rear_view=True,
            columns=2,
        ),
        Comparison(
            name="recorded_retract_ep75_case05_ik_velocity_ablation",
            suite="recorded_replay",
            scenario="recorded_ep75_right_retract05_straight_retract",
            panels=(
                Panel("pr_full_recorded", "Current PR: IK cap ON"),
                Panel(
                    "pr_driver_velocity_only_recorded",
                    "Current PR: IK cap OFF",
                ),
            ),
            caption=(
                "Same current-PR tasks; only the IK velocity envelope changes. "
                "The downstream driver cap remains enabled in both panels."
            ),
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            replay_rear_view=True,
        ),
        Comparison(
            name="recorded_chest_ep73_case40_direct_plant_validation",
            suite="episode73_direct_plant",
            scenario="episode73_event40_direct_command",
            panels=(
                Panel("recorded_real", "Recorded hardware state"),
                Panel("mujoco_nominal", "Direct MuJoCo replay"),
            ),
            caption=(
                "Both panels receive the exact recorded joint command. No IK "
                "target proxy or controller replay is used."
            ),
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
        ),
        Comparison(
            name="recorded_chest_ep73_case50_direct_plant_validation",
            suite="episode73_direct_plant",
            scenario="episode73_event50_direct_command",
            panels=(
                Panel("recorded_real", "Recorded hardware state"),
                Panel("mujoco_nominal", "Direct MuJoCo replay"),
            ),
            caption=(
                "Both panels receive the exact recorded joint command. This "
                "case contains the largest short-horizon command reversal."
            ),
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
        ),
    )


def main() -> None:
    for comparison in comparisons():
        path = comparison_video(comparison)
        print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
