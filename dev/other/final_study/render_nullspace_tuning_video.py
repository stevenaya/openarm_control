#!/usr/bin/env python3
"""Render the targeted nullspace parameter comparison only."""

from render_videos import Comparison, Panel, comparison_video


def main() -> None:
    output = comparison_video(
        Comparison(
            name="retract_nullspace_parameter_comparison",
            suite="nullspace_candidate_validation",
            scenario="retract_deep_p0p10_v0p80",
            panels=(
                Panel("current_c7_r1p6", "Current: cost 7 / cap 1.0"),
                Panel(
                    "cost_only_c7p5_v1p0",
                    "Cost only: 7.5 / cap 1.0",
                ),
                Panel(
                    "candidate_c7p5_v0p6",
                    "Stress candidate: 7.5 / cap 0.6",
                ),
                Panel(
                    "aggressive_c9_r0p8",
                    "Aggressive: cost 9 / return 0.8",
                ),
            ),
            caption=(
                "Nullspace cost and return-speed tuning during deep retraction."
            ),
            show_elbow=True,
            show_max_error=True,
            replay_rear_view=True,
            columns=2,
        )
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
