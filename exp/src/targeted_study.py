#!/usr/bin/env python3
"""Targeted current-PR studies for chest wrist motion, braking, and plant lag."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import study


def make_joint6_limit_scenario(
    factory: study.PoseFactory,
    *,
    angular_speed: float,
) -> study.Scenario:
    """Drive J6 toward both bounds through reachable orientation targets."""
    initial_right = study.START_Q_RIGHT.copy()
    initial_left = study.START_Q_LEFT.copy()
    initial_right[5] = 0.0
    positive_q = initial_right.copy()
    negative_q = initial_right.copy()
    positive_q[5] = 0.782
    negative_q[5] = -0.782
    base, _ = factory.bimanual(initial_right, initial_left)
    positive, _ = factory.bimanual(positive_q, initial_left)
    negative, _ = factory.bimanual(negative_q, initial_left)
    builder = study.PathBuilder({"right": base})
    builder.hold(0.3)
    builder.move(
        {"right": positive},
        linear_speed=1.0,
        angular_speed=angular_speed,
        phase=1,
    )
    builder.hold(0.3, phase=2)
    builder.move(
        {"right": negative},
        linear_speed=1.0,
        angular_speed=angular_speed,
        phase=3,
    )
    builder.hold(0.3, phase=4)
    builder.move(
        {"right": base},
        linear_speed=1.0,
        angular_speed=angular_speed,
        phase=5,
    )
    builder.hold(0.45, phase=6)
    return study._scenario_from_builder(
        name=f"joint6_bounds_w{angular_speed:.1f}".replace(".", "p"),
        family="joint_braking",
        mode="right",
        speed=angular_speed,
        builder=builder,
        initial_right=initial_right,
        initial_left=initial_left,
        description="Orientation sweep whose nominal branch approaches both J6 bounds.",
    )


def chest_scenarios() -> list[study.Scenario]:
    factory = study.PoseFactory()
    scenarios = [
        study.make_chest_wrist_outward_scenario(
            factory,
            linear_speed=linear_speed,
            angular_speed=angular_speed,
        )
        for linear_speed in (0.3, 0.8, 1.2)
        for angular_speed in (4.0, 8.0, 12.0)
    ]
    for label, displacement in (
        ("forward", (0.07, 0.0, 0.0)),
        ("lateral", (0.0, -0.07, 0.0)),
        ("down", (0.0, 0.0, -0.07)),
        ("diagonal", (0.07, -0.07, -0.03)),
    ):
        scenario = study.make_chest_wrist_outward_scenario(
            factory,
            linear_speed=1.2,
            angular_speed=8.0,
            displacement=displacement,
        )
        scenarios.append(
            replace(
                scenario,
                name=f"chest_outward_flip_right_{label}_v1p20_w8p0",
                description=(
                    scenario.description
                    + f" Translation variant: {label}."
                ),
            )
        )
    scenarios.append(
        study.make_wrist_flip_scenario(
            factory,
            angular_speed=10.0,
            extended=False,
            initial_right_override=study.CHEST_Q_RIGHT,
            name_suffix="_chest",
        )
    )
    return scenarios


def chest_profiles() -> list[study.Profile]:
    profiles = [
        study.deployment_profile(),
        study.strict_mainline_profile(),
        study.deployment_profile(
            "no_frame_error_bound",
            frame_position_error_limit=0.0,
            frame_orientation_error_limit=0.0,
        ),
        study.no_branch_regulation_profile(),
        study.deployment_profile(
            "no_kinetic_regularization",
            kinetic_energy_cost=0.0,
        ),
    ]
    for value in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        profiles.append(
            study.deployment_profile(
                f"orientation_budget_{value:g}".replace(".", "p"),
                frame_orientation_error_limit=value,
            )
        )
    for value in (0.0, 0.010, 0.015, 0.020, 0.025, 0.030):
        profiles.append(
            study.deployment_profile(
                f"position_budget_{value:g}".replace(".", "p"),
                frame_position_error_limit=value,
            )
        )
    return profiles


def braking_profiles() -> list[study.Profile]:
    profiles: list[study.Profile] = []
    for distance in (0.08, 0.12, 0.20, 0.30):
        profiles.append(
            study.deployment_profile(
                f"brake_distance_{distance:g}".replace(".", "p"),
                joint_braking_distance=distance,
                singularity_max_approach_rate=0.0,
            )
        )
    profiles.extend(
        [
            study.deployment_profile(
                "no_joint_braking",
                joint_braking=False,
                singularity_max_approach_rate=0.0,
            ),
            study.deployment_profile(
                "no_ik_velocity_or_braking",
                limit_style="configuration_only",
                velocity_caps=None,
                joint_braking=False,
                singularity_max_approach_rate=0.0,
            ),
        ]
    )
    return profiles


def robustness_profiles() -> list[study.Profile]:
    return [
        study.deployment_profile(),
        study.deployment_profile(
            "gravity_compensation",
            gravity_compensation=True,
        ),
        study.deployment_profile("state_delay_8ms", state_delay_s=0.008),
        study.deployment_profile("state_delay_20ms", state_delay_s=0.020),
        study.deployment_profile("command_delay_8ms", command_delay_s=0.008),
        study.deployment_profile("command_delay_20ms", command_delay_s=0.020),
        study.deployment_profile("actuator_gain_x0p7", actuator_kp_scale=0.7),
        study.deployment_profile("actuator_gain_x1p3", actuator_kp_scale=1.3),
        study.deployment_profile(
            "driver_only_velocity",
            limit_style="configuration_only",
            velocity_caps=None,
        ),
    ]


def robustness_scenarios() -> list[study.Scenario]:
    factory = study.PoseFactory()
    return [
        study.make_reach_scenario(factory, speed=0.8),
        study.make_diagonal_retract_scenario(
            factory,
            speed=0.8,
            lateral=0.10,
        ),
        study.make_extended_circle_scenario(factory, speed=0.8),
        study.make_chest_wrist_outward_scenario(
            factory,
            linear_speed=1.2,
            angular_speed=12.0,
        ),
        study.make_normal_lissajous_scenario(
            factory,
            speed=0.8,
            mode="right",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("chest", "braking", "robustness", "all"),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/results/development"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if args.suite in ("chest", "all"):
        study.run_matrix(
            chest_profiles(),
            chest_scenarios(),
            root / "chest",
            workers=args.workers,
            save_all_traces=True,
        )
    if args.suite in ("braking", "all"):
        factory = study.PoseFactory()
        study.run_matrix(
            braking_profiles(),
            [
                make_joint6_limit_scenario(factory, angular_speed=speed)
                for speed in (2.0, 4.0, 8.0, 12.0)
            ],
            root / "braking",
            workers=args.workers,
            save_all_traces=True,
        )
    if args.suite in ("robustness", "all"):
        study.run_matrix(
            robustness_profiles(),
            robustness_scenarios(),
            root / "robustness",
            workers=args.workers,
            save_all_traces=True,
        )


if __name__ == "__main__":
    main()
