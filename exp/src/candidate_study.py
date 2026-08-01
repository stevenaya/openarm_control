#!/usr/bin/env python3
"""Validate combined tuning candidates across the broad trajectory suite."""

from __future__ import annotations

import argparse
from pathlib import Path

import study


def candidate_profiles() -> list[study.Profile]:
    """Return current defaults and candidates suggested by one-factor sweeps."""
    return [
        study.deployment_profile(),
        study.deployment_profile(
            "ori0p20_sing0p18",
            frame_orientation_error_limit=0.20,
            singularity_max_approach_rate=0.18,
            description=(
                "Only tighten the orientation budget and singularity approach "
                "rate; retain all other PR default settings."
            ),
        ),
        study.deployment_profile(
            "ori0p20_null10_sing0p18_brake0p12_energy3em5",
            frame_orientation_error_limit=0.20,
            nullspace_cost=10.0,
            singularity_max_approach_rate=0.18,
            joint_braking_distance=0.12,
            kinetic_energy_cost=3.0e-5,
            description=(
                "Balanced combination selected from the one-factor plateaus."
            ),
        ),
        study.deployment_profile(
            "ori0p15_null12_sing0p18_brake0p12_energy5em5",
            frame_orientation_error_limit=0.15,
            nullspace_cost=12.0,
            singularity_max_approach_rate=0.18,
            joint_braking_distance=0.12,
            kinetic_energy_cost=5.0e-5,
            description=(
                "More conservative branch and acceleration tuning with "
                "additional permitted orientation lag."
            ),
        ),
        study.deployment_profile(
            "ori0p25_null10_sing0p18_brake0p12_energy3em5",
            frame_orientation_error_limit=0.25,
            nullspace_cost=10.0,
            singularity_max_approach_rate=0.18,
            joint_braking_distance=0.12,
            kinetic_energy_cost=3.0e-5,
            description=(
                "Retain the current orientation budget while tuning the "
                "secondary mechanisms."
            ),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/results/development/candidates"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    study.run_matrix(
        candidate_profiles(),
        study.screening_scenarios(),
        args.output_dir.resolve(),
        workers=args.workers,
        save_all_traces=True,
    )


if __name__ == "__main__":
    main()
