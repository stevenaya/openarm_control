#!/usr/bin/env python3
"""Render corrected residual-failure and direct-plant comparisons."""

from __future__ import annotations

from render_videos import Comparison, Panel, comparison_video


def comparisons() -> tuple[Comparison, ...]:
    return (
        Comparison(
            name="episode73_triphasic_98s_corrected_analysis",
            suite="episode73_triphasic",
            scenario="episode73_triphasic_98s_straight_position",
            panels=(
                Panel(
                    "recorded_hardware",
                    "Recorded IK command / hardware",
                ),
                Panel(
                    "pr_recorded_config",
                    "Current PR, smooth position",
                ),
                Panel(
                    "pr_orientation_budget_010",
                    "Current PR, 0.10 rad orientation budget",
                ),
                Panel(
                    "mainline_ik_velocity",
                    "Mainline + IK velocity cap",
                ),
            ),
            caption=(
                "Episode 73 at 98 s. The first panel shows the actual "
                "retract-pause-extend command. Counterfactual panels preserve "
                "its rotation but use a smooth non-reversing position target."
            ),
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            columns=2,
        ),
        Comparison(
            name="episode75_event05_same_command_plant_validation",
            suite="episode75_direct_plant",
            scenario="episode75_event05_direct_command",
            panels=(
                Panel("kinematic_command", "Recorded IK command"),
                Panel("recorded_hardware", "Physical arm"),
                Panel("mujoco_nominal", "MuJoCo nominal"),
                Panel(
                    "mujoco_gravity_comp",
                    "MuJoCo with gravity compensation",
                ),
            ),
            caption=(
                "All panels use the exact same recorded IK joint-position "
                "command. Cyan is the command ghost; the solid arm is the "
                "executed state."
            ),
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            replay_rear_view=True,
            columns=2,
        ),
    )


def main() -> None:
    for comparison in comparisons():
        print(f"Wrote {comparison_video(comparison)}", flush=True)


if __name__ == "__main__":
    main()
