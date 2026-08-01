#!/usr/bin/env python3
"""Tune nullspace regulation on the deep-retract stress trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import study
from video_trajectory_study import make_deep_retract


def tuning_profiles() -> list[study.Profile]:
    """Return cost, return-rate, and speed-cap sweeps."""
    profiles = [
        study.make_profile(
            f"ns_cost_{cost:g}".replace(".", "p"),
            nullspace_cost=cost,
        )
        for cost in (0.0, 3.0, 7.0, 10.0, 12.0, 18.0, 25.0)
    ]
    profiles.extend(
        study.make_profile(
            f"ns_c12_v{speed:g}".replace(".", "p"),
            nullspace_cost=12.0,
            nullspace_max_speed=speed,
        )
        for speed in (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
    )
    profiles.extend(
        study.make_profile(
            f"ns_c12_v1p5_r{rate:g}".replace(".", "p"),
            nullspace_cost=12.0,
            nullspace_return_rate=rate,
            nullspace_max_speed=1.50,
        )
        for rate in (0.8, 1.2, 1.6, 2.4, 3.2)
    )
    profiles.extend(
        study.make_profile(
            f"ns_local_c{cost:g}_r{rate:g}".replace(".", "p"),
            nullspace_cost=cost,
            nullspace_return_rate=rate,
        )
        for cost in (5.0, 6.0, 7.0, 8.0, 9.0)
        for rate in (0.4, 0.6, 0.8, 1.0, 1.2, 1.6)
    )
    profiles.extend(
        study.make_profile(
            f"ns_cap_c{cost:g}_v{speed:g}".replace(".", "p"),
            nullspace_cost=cost,
            nullspace_max_speed=speed,
        )
        for cost in (7.5, 8.0, 8.5)
        for speed in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00)
    )
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "dev/final_study/results/current_pr_20260730/nullspace_tuning"
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    scenario = make_deep_retract(study.PoseFactory(), lateral=0.10)
    study.run_matrix(
        tuning_profiles(),
        [scenario],
        args.output_dir.resolve(),
        workers=args.workers,
        save_all_traces=True,
    )


if __name__ == "__main__":
    main()
