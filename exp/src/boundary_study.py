#!/usr/bin/env python3
"""Isolate recoverability when the synchronized arm starts outside a joint bound."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np
import study


def _profile(name: str, limit_style: str) -> study.Profile:
    return study.make_profile(
        name,
        limit_style=limit_style,
        velocity_caps=(
            None if limit_style == "configuration_only" else study.CONTROL_CAPS
        ),
        driver_velocity_caps=None,
        joint_braking=False,
        frame_position_error_limit=0.0,
        frame_orientation_error_limit=0.0,
        posture_cost=0.0,
        nullspace_cost=0.0,
        singularity_max_approach_rate=0.0,
        kinetic_energy_cost=0.0,
    )


def _joint_ranges(setup: study.ArmSetup, side: str) -> np.ndarray:
    ranges = np.empty((7, 2), dtype=np.float64)
    for index, qpos_index in enumerate(
        setup.joint_resolver.arm_qpos_indices(side)
    ):
        joint_id = int(
            np.flatnonzero(setup.model.jnt_qposadr == int(qpos_index))[0]
        )
        ranges[index] = setup.model.jnt_range[joint_id]
    return ranges


def _target_pose(q: np.ndarray) -> np.ndarray:
    setup = study.make_setup("right")
    setup.joint_resolver.set_qpos(
        setup.data.qpos,
        np.append(q, 0.0),
        "right",
    )
    mujoco.mj_forward(setup.model, setup.data)
    return setup.read_ee_pose("right").astype(np.float64)


def run() -> list[dict[str, float | int | str | bool]]:
    setup = study.make_setup("right")
    ranges = _joint_ranges(setup, "right")
    offsets = (0.0, 1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2)
    profiles = (
        _profile("recoverable_joint_limit", "recoverable"),
        _profile("native_position_plus_velocity", "standard"),
        _profile("configuration_only", "configuration_only"),
    )
    rows: list[dict[str, float | int | str | bool]] = []

    for joint in range(7):
        for boundary_name, direction in (("lower", -1.0), ("upper", 1.0)):
            boundary = ranges[joint, 0 if direction < 0.0 else 1]
            for offset in offsets:
                initial = study.HOME_Q.copy()
                initial[joint] = boundary + direction * offset
                target = _target_pose(initial)
                values16 = np.concatenate(
                    [
                        np.append(initial, 0.0),
                        np.append(study.HOME_Q, 0.0),
                    ]
                ).astype(np.float32)

                for profile in profiles:
                    kinematics = study.make_kinematics(profile, "right")
                    kinematics.sync(values16)
                    kinematics.update_measured_state(values16)
                    kinematics.set_target("right", target)
                    result = kinematics.solve()
                    solved = result is not None
                    command = (
                        initial.copy()
                        if result is None
                        else result[:7].astype(np.float64)
                    )
                    dq = (command - initial) / study.CONTROL_DT
                    toward_interior = (
                        dq[joint] if direction < 0.0 else -dq[joint]
                    )
                    rows.append(
                        {
                            "profile": profile.name,
                            "joint": joint + 1,
                            "boundary": boundary_name,
                            "offset_rad": offset,
                            "solved": solved,
                            "recovery_speed_rad_s": toward_interior,
                            "max_abs_dq_rad_s": float(np.max(np.abs(dq))),
                            "remaining_violation_rad": max(
                                0.0,
                                (
                                    ranges[joint, 0] - command[joint]
                                    if direction < 0.0
                                    else command[joint] - ranges[joint, 1]
                                ),
                            ),
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/results/development/boundary"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run()
    path = args.output_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
