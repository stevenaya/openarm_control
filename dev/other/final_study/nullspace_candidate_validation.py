#!/usr/bin/env python3
"""Cross-validate promising nullspace parameters beyond deep retraction."""

from __future__ import annotations

import argparse
from pathlib import Path

import study
from video_trajectory_study import make_deep_retract


def validation_profiles() -> list[study.Profile]:
    """Return current, conservative candidates, and an aggressive comparison."""
    return [
        study.make_profile("current_c7_r1p6"),
        study.make_profile(
            "candidate_c8_r0p4",
            nullspace_cost=8.0,
            nullspace_return_rate=0.4,
        ),
        study.make_profile(
            "candidate_c8_r0p6",
            nullspace_cost=8.0,
            nullspace_return_rate=0.6,
        ),
        study.make_profile(
            "aggressive_c9_r0p8",
            nullspace_cost=9.0,
            nullspace_return_rate=0.8,
        ),
        study.make_profile(
            "cost_only_c7p5_v1p0",
            nullspace_cost=7.5,
        ),
        study.make_profile(
            "candidate_c7p5_v0p6",
            nullspace_cost=7.5,
            nullspace_max_speed=0.6,
        ),
        study.make_profile(
            "candidate_c7p5_v0p7",
            nullspace_cost=7.5,
            nullspace_max_speed=0.7,
        ),
        study.make_profile(
            "candidate_c7p5_v0p8",
            nullspace_cost=7.5,
            nullspace_max_speed=0.8,
        ),
    ]


def validation_scenarios() -> list[study.Scenario]:
    """Cover mirrored retraction, weak poses, normal motion, and wrist flips."""
    factory = study.PoseFactory()
    return [
        make_deep_retract(factory, lateral=0.10),
        make_deep_retract(factory, lateral=-0.10),
        study.make_diagonal_retract_scenario(
            factory,
            speed=0.4,
            lateral=0.10,
        ),
        study.make_diagonal_retract_scenario(
            factory,
            speed=0.8,
            lateral=0.10,
        ),
        study.make_diagonal_retract_scenario(
            factory,
            speed=0.8,
            lateral=-0.10,
        ),
        study.make_extended_circle_scenario(factory, speed=0.8),
        study.make_wrist_flip_scenario(
            factory,
            angular_speed=10.0,
            extended=True,
        ),
        study.make_wrist_flip_scenario(
            factory,
            angular_speed=10.0,
            extended=False,
        ),
        study.make_chest_wrist_outward_scenario(
            factory,
            linear_speed=1.2,
            angular_speed=8.0,
        ),
        study.make_chest_wrist_outward_scenario(
            factory,
            linear_speed=0.8,
            angular_speed=12.0,
        ),
        study.make_normal_lissajous_scenario(
            factory,
            speed=0.8,
            mode="right",
        ),
        study.make_normal_lissajous_scenario(
            factory,
            speed=0.6,
            mode="bimanual",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "dev/final_study/results/current_pr_20260730/"
            "nullspace_candidate_validation"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    study.run_matrix(
        validation_profiles(),
        validation_scenarios(),
        args.output_dir.resolve(),
        workers=args.workers,
        save_all_traces=True,
    )


if __name__ == "__main__":
    main()
