#!/usr/bin/env python3
"""Generate the extended trajectories used by the final comparison videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import study


REACH_DEEP_START_Q_RIGHT = np.array(
    [
        0.798584,
        0.000184,
        -0.000189,
        1.562622,
        0.000273,
        0.437062,
        -0.000207,
    ],
    dtype=np.float64,
)


def make_deep_start_reach(factory: study.PoseFactory) -> study.Scenario:
    """Reach from 10 cm farther back to the original far target."""
    original_start, _ = factory.bimanual(
        study.REACH_Q_RIGHT,
        study.REACH_Q_LEFT,
    )
    start, _ = factory.bimanual(
        REACH_DEEP_START_Q_RIGHT,
        study.REACH_Q_LEFT,
    )
    far_target = original_start.copy()
    far_target[0] += 0.30

    builder = study.PathBuilder({"right": start})
    builder.hold(0.25)
    builder.move(
        {"right": far_target},
        linear_speed=0.8,
        angular_speed=8.0,
        phase=1,
    )
    builder.hold(0.65, phase=2)
    builder.move(
        {"right": start},
        linear_speed=0.8,
        angular_speed=8.0,
        phase=3,
    )
    builder.hold(0.55, phase=4)
    return study._scenario_from_builder(
        name="reach_deep_start_right_p0p00_v0p80",
        family="reach",
        mode="right",
        speed=0.8,
        builder=builder,
        initial_right=REACH_DEEP_START_Q_RIGHT,
        initial_left=study.REACH_Q_LEFT,
        description=(
            "Shoulder-height reach starting 10 cm farther back while retaining "
            "the original far target."
        ),
    )


def make_deep_retract(
    factory: study.PoseFactory,
    *,
    lateral: float,
) -> study.Scenario:
    """Retract 10 cm closer to the base-link rear plane."""
    initial_right = study.EXTENDED_Q_RIGHT.copy()
    initial_left = study.HOME_Q.copy()
    start, _ = factory.bimanual(initial_right, initial_left)
    target = start.copy()
    target[:3] += np.array([-0.32, lateral, -0.12])

    builder = study.PathBuilder({"right": start})
    builder.hold(0.30)
    builder.move(
        {"right": target},
        linear_speed=0.8,
        angular_speed=8.0,
        phase=1,
    )
    builder.hold(0.65, phase=2)
    lateral_tag = (
        f"{lateral:+.2f}"
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
    )
    return study._scenario_from_builder(
        name=f"retract_deep_{lateral_tag}_v0p80",
        family="retract",
        mode="right",
        speed=0.8,
        builder=builder,
        initial_right=initial_right,
        initial_left=initial_left,
        description=(
            "Fast diagonal retract ending 10 cm closer to the base-link rear "
            f"plane with {lateral:+.2f} m lateral motion."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/results/development/video_trajectories"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    factory = study.PoseFactory()
    reach = make_deep_start_reach(factory)
    retract_positive = make_deep_retract(factory, lateral=0.10)
    retract_negative = make_deep_retract(factory, lateral=-0.10)
    pairs = (
        (study.make_profile("pr_full"), reach),
        (
            study.make_profile(
                "no_singularity",
                singularity_max_approach_rate=0.0,
            ),
            reach,
        ),
        (study.make_profile("pr_full"), retract_positive),
        (
            study.make_profile(
                "no_frame_error",
                frame_position_error_limit=0.0,
                frame_orientation_error_limit=0.0,
            ),
            retract_positive,
        ),
        (
            study.make_profile(
                "driver_only_ablation",
                limit_style="configuration_only",
                velocity_caps=None,
            ),
            retract_positive,
        ),
        (
            study.make_profile("no_nullspace", nullspace_cost=0.0),
            retract_positive,
        ),
        (study.upstream_style_profile(), retract_positive),
        (
            study.upstream_style_ik_velocity_profile(),
            retract_positive,
        ),
        (study.make_profile("pr_full"), retract_negative),
    )

    rows: list[dict[str, object]] = []
    for profile, scenario in pairs:
        print(f"{profile.name} :: {scenario.name}")
        rows.extend(
            study.run_cached(
                profile,
                scenario,
                output_dir,
                save_full_trace=True,
            )
        )
    study.write_summary(output_dir / "summary.csv", rows)


if __name__ == "__main__":
    main()
