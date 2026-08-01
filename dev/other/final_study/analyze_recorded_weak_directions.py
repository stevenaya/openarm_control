#!/usr/bin/env python3
"""Decompose recorded-case joint speed into exact and near-weak directions."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import mink
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recorded_intervention_study as recorded
import study

from openarm_control.geometry.jacobian import normalized_arm_jacobian

OUTPUT_PATH = recorded.REPLAY_DIR / "weak_direction_decomposition.csv"


def _trace_path(profile: str, scenario: str) -> Path:
    matches = list(
        (recorded.REPLAY_DIR / "traces").glob(
            f"*_{profile}_{scenario}.npz"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one trace for {profile}/{scenario}, got {matches}."
        )
    return matches[0]


def _decompose(profile: str, scenario: str) -> dict[str, Any]:
    trace = np.load(_trace_path(profile, scenario), allow_pickle=False)
    active = trace["phase"] == 1
    q = trace["right_command_q"][active]
    dq = trace["right_command_dq"][active]

    kinematics = study.make_kinematics(
        study.make_profile("jacobian_analysis"),
        "right",
    )
    solver = kinematics._ik
    assert solver is not None
    setup = solver._setup
    configuration = mink.Configuration(setup.model)
    base_qpos = solver._config.q.copy()
    dof_indices = setup.joint_resolver.arm_dof_indices("right")
    frame_task = solver._tasks["right"]

    z_projection = np.empty(q.shape[0], dtype=np.float64)
    near_projection = np.empty(q.shape[0], dtype=np.float64)
    rho = np.empty(q.shape[0], dtype=np.float64)
    for index, (arm_q, arm_dq) in enumerate(zip(q, dq, strict=True)):
        qpos = base_qpos.copy()
        setup.joint_resolver.set_qpos(
            qpos,
            np.append(arm_q, 0.0),
            "right",
        )
        configuration.update(q=qpos)
        jacobian = normalized_arm_jacobian(
            frame_task,
            configuration,
            dof_indices,
            characteristic_length=0.3,
        )
        _, singular_values, vh = np.linalg.svd(
            jacobian,
            full_matrices=True,
        )
        z_projection[index] = float(vh[-1] @ arm_dq)
        near_projection[index] = float(vh[-2] @ arm_dq)
        rho[index] = float(singular_values[-1] / singular_values[0])

    total_energy = float(np.sum(np.square(dq)))
    z_energy = float(np.sum(np.square(z_projection)))
    near_energy = float(np.sum(np.square(near_projection)))
    return {
        "profile": profile,
        "scenario": scenario,
        "sample_count": int(q.shape[0]),
        "rho_min": float(np.min(rho)),
        "rho_p05": float(np.quantile(rho, 0.05)),
        "exact_z_abs_mean_rad_s": float(np.mean(np.abs(z_projection))),
        "near_weak_abs_mean_rad_s": float(
            np.mean(np.abs(near_projection))
        ),
        "exact_z_energy_fraction": z_energy / total_energy,
        "near_weak_energy_fraction": near_energy / total_energy,
        "other_energy_fraction": max(
            0.0,
            1.0 - (z_energy + near_energy) / total_energy,
        ),
    }


def main() -> None:
    metrics = pd.read_csv(recorded.REPLAY_DIR / "metrics.csv")
    profiles = (
        "recorded_hardware",
        "pr_full_recorded",
        "mainline_ik_velocity_recorded",
        "mainline_recorded",
        "pr_driver_velocity_only_recorded",
        "pr_nullspace_cost_10_recorded",
        "pr_nullspace_cost_12_recorded",
    )
    rows = [
        _decompose(profile, scenario)
        for scenario in metrics["scenario"].drop_duplicates()
        for profile in profiles
    ]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
