#!/usr/bin/env python3
"""Run the reproducible experiment set used by the public report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import boundary_study
import candidate_study
import study
import targeted_study
import video_trajectory_study


HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
REPO_ROOT = EXP_ROOT.parent
DEFAULT_ROOT = EXP_ROOT / "results" / "final_report_20260801"
CHEST_SOURCE = HERE / "inputs" / "near_chest_fast_wrist_roll.npz"
RETRACT_SOURCE = HERE / "inputs" / "fast_retract_elbow_branch.npz"
SUITES = (
    "screening",
    "parameters",
    "driver",
    "symmetry",
    "chest",
    "braking",
    "robustness",
    "candidates",
    "boundary",
    "frozen",
    "fast_retract_error_bound",
    "singularity_video",
    "nullspace_sweep",
    "nullspace_validation",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_scenario(
    path: Path,
    *,
    name: str,
    family: str,
    description: str,
) -> study.Scenario:
    """Load a frozen Cartesian target without retaining private episode labels."""
    with np.load(path, allow_pickle=False) as payload:
        times = payload["times"].astype(np.float64, copy=True)
        target_right = payload["target_right"].astype(np.float64, copy=True)
        sample_dt = np.diff(times)
        linear_speed = np.linalg.norm(
            np.diff(target_right[:, :3], axis=0),
            axis=1,
        ) / sample_dt
        return study.Scenario(
            name=name,
            family=family,
            mode="right",
            speed=float(np.max(linear_speed)),
            times=times,
            phase=payload["phase"].copy(),
            target_right=target_right,
            target_left=payload["target_left"].copy(),
            initial_right=payload["initial_right"].copy(),
            initial_left=payload["initial_left"].copy(),
            description=description,
        )


def frozen_benchmarks() -> tuple[study.Scenario, study.Scenario]:
    """Return the two public, immutable controller-level benchmarks."""
    chest = load_frozen_scenario(
        CHEST_SOURCE,
        name="near_chest_fast_wrist_roll",
        family="frozen_near_chest_wrist_roll",
        description=(
            "Frozen near-chest target with fast wrist roll and simultaneous "
            "Cartesian translation; used to evaluate tracking and vibration."
        ),
    )
    retract = load_frozen_scenario(
        RETRACT_SOURCE,
        name="fast_retract_elbow_branch",
        family="frozen_fast_retract",
        description=(
            "Frozen fast right-arm retract target; used to evaluate elbow "
            "branch consistency, Cartesian error, and joint dynamics."
        ),
    )
    return chest, retract


def accelerated_fast_retract_benchmark() -> study.Scenario:
    """Return the frozen fast-retract command path at twice its original rate."""
    _, source = frozen_benchmarks()
    indices = np.arange(0, source.times.size, 2)
    if indices[-1] != source.times.size - 1:
        indices = np.append(indices, source.times.size - 1)
    target_right = source.target_right[indices].copy()
    linear_speed = np.linalg.norm(
        np.diff(target_right[:, :3], axis=0),
        axis=1,
    ) / study.CONTROL_DT
    return study.Scenario(
        name="fast_retract_elbow_branch_2x",
        family="frozen_fast_retract_time_compressed",
        mode=source.mode,
        speed=float(np.max(linear_speed)),
        times=np.arange(indices.size, dtype=np.float64) * study.CONTROL_DT,
        phase=source.phase[indices].copy(),
        target_right=target_right,
        target_left=source.target_left[indices].copy(),
        initial_right=source.initial_right.copy(),
        initial_left=source.initial_left.copy(),
        description=(
            "Frozen recorded fast-retract reference command, replayed at "
            "twice its original rate to exercise the 6D frame-error bound."
        ),
    )


def frozen_profiles() -> list[study.Profile]:
    """Profiles needed by Figures 08, 34, and 35."""
    return [
        study.deployment_profile(),
        study.strict_mainline_profile(),
        study.deployment_profile(
            "no_frame_error_bound",
            frame_position_error_limit=0.0,
            frame_orientation_error_limit=0.0,
        ),
        study.no_branch_regulation_profile(),
        study.full_home_replacement_profile(
            "full_home_replacement_0p003",
            posture_cost=0.003,
        ),
        study.full_home_replacement_profile(),
        study.full_home_replacement_profile(
            "full_home_replacement_0p03",
            posture_cost=0.03,
        ),
    ]


def nullspace_sweep_profiles() -> list[study.Profile]:
    """Compact structured sweep around the PR default nullspace task."""
    profiles: list[study.Profile] = [
        study.deployment_profile(),
        study.no_branch_regulation_profile(),
    ]
    for cost in (3.0, 5.0, 8.5, 12.0, 18.0):
        profiles.append(
            study.deployment_profile(
                f"null_cost_{cost:g}".replace(".", "p"),
                nullspace_cost=cost,
            )
        )
    for rate in (0.6, 1.0, 1.6, 2.4, 3.2):
        profiles.append(
            study.deployment_profile(
                f"null_return_{rate:g}".replace(".", "p"),
                nullspace_return_rate=rate,
            )
        )
    for max_speed in (0.4, 0.7, 1.0, 1.3, 1.6):
        profiles.append(
            study.deployment_profile(
                f"null_max_speed_{max_speed:g}".replace(".", "p"),
                nullspace_max_speed=max_speed,
            )
        )
    for cost in (5.0, 8.5, 12.0):
        for rate in (0.8, 1.6, 2.4):
            for max_speed in (0.6, 1.0, 1.4):
                profiles.append(
                    study.deployment_profile(
                        (
                            f"null_c{cost:g}_r{rate:g}_v{max_speed:g}"
                        ).replace(".", "p"),
                        nullspace_cost=cost,
                        nullspace_return_rate=rate,
                        nullspace_max_speed=max_speed,
                    )
                )
    return list({profile.name: profile for profile in profiles}.values())


def nullspace_validation_profiles() -> list[study.Profile]:
    """Small candidate set for cross-scenario nullspace validation."""
    return [
        study.deployment_profile(),
        study.no_branch_regulation_profile(),
        study.deployment_profile(
            "null_c5_r1p6_v1p0",
            nullspace_cost=5.0,
        ),
        study.deployment_profile(
            "null_c8p5_r0p8_v0p6",
            nullspace_return_rate=0.8,
            nullspace_max_speed=0.6,
        ),
        study.deployment_profile(
            "null_c12_r1p6_v1p0",
            nullspace_cost=12.0,
        ),
        study.deployment_profile(
            "null_c8p5_r2p4_v1p0",
            nullspace_return_rate=2.4,
        ),
    ]


def suite_scenarios(name: str) -> list[study.Scenario]:
    """Build the exact target trajectories used by one dynamic suite."""
    if name in {"screening", "candidates"}:
        return study.screening_scenarios()
    if name in {"parameters", "driver", "nullspace_validation"}:
        return study.focused_scenarios()
    if name == "symmetry":
        return study.symmetry_scenarios()
    if name == "chest":
        return targeted_study.chest_scenarios()
    if name == "braking":
        factory = study.PoseFactory()
        return [
            targeted_study.make_joint6_limit_scenario(
                factory,
                angular_speed=speed,
            )
            for speed in (2.0, 4.0, 8.0, 12.0)
        ]
    if name == "robustness":
        return targeted_study.robustness_scenarios()
    if name == "frozen":
        return list(frozen_benchmarks())
    if name == "fast_retract_error_bound":
        return [accelerated_fast_retract_benchmark()]
    if name == "singularity_video":
        return [
            video_trajectory_study.make_deep_start_reach(
                study.PoseFactory()
            )
        ]
    if name == "nullspace_sweep":
        _, retract = frozen_benchmarks()
        return [retract]
    if name == "boundary":
        return []
    raise ValueError(f"Unknown suite {name!r}")


def _write_boundary(root: Path) -> None:
    output = root / "boundary"
    output.mkdir(parents=True, exist_ok=True)
    rows = boundary_study.run()
    with (output / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "generated_by": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
        "condition_count": len(rows),
        "joint_count": 7,
        "bound_sides": 2,
        "offset_count": 9,
        "profile_count": 3,
        "random_sampling": False,
        "random_seed": 0,
        "position_limit_excess_field": "remaining_violation_rad",
        "position_limit_excess_definition": (
            "Distance in radians still outside the position bound after one "
            "4 ms outer solve containing five 0.8 ms QP substeps."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _write_top_manifest(root: Path) -> None:
    children: dict[str, dict[str, object]] = {}
    dynamic_runs = 0
    unique_trajectory_hashes: set[str] = set()
    for path in sorted(root.glob("*/metadata.json")):
        if path.parent.name == "smoke":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        relative = path.parent.relative_to(root).as_posix()
        matrix = payload.get("matrix", {})
        run_count = int(matrix.get("controller_trajectory_runs", 0))
        dynamic_runs += run_count
        for scenario in payload.get("scenarios", []):
            trajectory_hash = scenario.get("trajectory_sha256")
            if trajectory_hash:
                unique_trajectory_hashes.add(str(trajectory_hash))
        children[relative] = {
            "metadata": str(path.relative_to(root)),
            "controller_trajectory_runs": run_count,
            "scenario_count": int(matrix.get("scenario_count", 0)),
            "profile_count": int(matrix.get("profile_count", 0)),
        }
    manifest = {
        "purpose": "Refreshed OpenArm IK public report experiment set",
        "reproduction_environment": {
            "python": (HERE / ".python-version").read_text(encoding="utf-8").strip(),
            "lock_path": (HERE / "uv.lock").relative_to(REPO_ROOT).as_posix(),
            "lock_sha256": _sha256(HERE / "uv.lock"),
        },
        "current_deployment_parameters": study.current_parameter_values(),
        "driver_config": study.driver_config_values(),
        "frozen_sources": {
            "near_chest_fast_wrist_roll": {
                "path": CHEST_SOURCE.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(CHEST_SOURCE),
            },
            "fast_retract_elbow_branch": {
                "path": RETRACT_SOURCE.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(RETRACT_SOURCE),
            },
        },
        "dynamic_controller_trajectory_runs": dynamic_runs,
        "unique_target_trajectory_hashes": len(unique_trajectory_hashes),
        "static_joint_limit_conditions": 378,
        "suites": children,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def run_suite(name: str, root: Path, workers: int) -> None:
    if name == "boundary":
        _write_boundary(root)
        return

    scenarios = suite_scenarios(name)
    if name == "screening":
        study.run_matrix(
            study.final_screening_profiles(),
            scenarios,
            root / "screening",
            workers=workers,
        )
    elif name == "parameters":
        study.run_matrix(
            study.final_parameter_profiles(),
            scenarios,
            root / "parameters",
            workers=workers,
        )
    elif name == "driver":
        study.run_matrix(
            study.driver_coupling_profiles(),
            scenarios,
            root / "driver",
            workers=workers,
            save_all_traces=True,
        )
    elif name == "symmetry":
        study.run_matrix(
            study.current_symmetry_profiles(),
            scenarios,
            root / "symmetry",
            workers=workers,
        )
    elif name == "chest":
        study.run_matrix(
            targeted_study.chest_profiles(),
            scenarios,
            root / "chest",
            workers=workers,
            save_all_traces=True,
        )
    elif name == "braking":
        study.run_matrix(
            targeted_study.braking_profiles(),
            scenarios,
            root / "braking",
            workers=workers,
            save_all_traces=True,
        )
    elif name == "robustness":
        study.run_matrix(
            targeted_study.robustness_profiles(),
            scenarios,
            root / "robustness",
            workers=workers,
            save_all_traces=True,
        )
    elif name == "candidates":
        study.run_matrix(
            candidate_study.candidate_profiles(),
            scenarios,
            root / "candidates",
            workers=workers,
            save_all_traces=True,
        )
    elif name == "frozen":
        study.run_matrix(
            frozen_profiles(),
            scenarios,
            root / "frozen",
            workers=workers,
            save_all_traces=True,
        )
    elif name == "fast_retract_error_bound":
        study.run_matrix(
            [
                study.deployment_profile(),
                study.deployment_profile(
                    "no_frame_error_bound",
                    frame_position_error_limit=0.0,
                    frame_orientation_error_limit=0.0,
                ),
            ],
            scenarios,
            root / "fast_retract_error_bound",
            workers=workers,
            save_all_traces=True,
        )
    elif name == "singularity_video":
        study.run_matrix(
            [
                study.deployment_profile(),
                study.deployment_profile(
                    "no_singularity_limit",
                    singularity_max_approach_rate=0.0,
                ),
            ],
            scenarios,
            root / "singularity_video",
            workers=workers,
            save_all_traces=True,
        )
    elif name == "nullspace_sweep":
        study.run_matrix(
            nullspace_sweep_profiles(),
            scenarios,
            root / "nullspace_sweep",
            workers=workers,
            save_all_traces=True,
        )
    elif name == "nullspace_validation":
        study.run_matrix(
            nullspace_validation_profiles(),
            scenarios,
            root / "nullspace_validation",
            workers=workers,
        )
    else:
        raise ValueError(f"Unknown suite {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = SUITES if args.suite == "all" else (args.suite,)
    for suite_name in selected:
        print(f"\n=== {suite_name} ===", flush=True)
        run_suite(suite_name, root, args.workers)
        _write_top_manifest(root)


if __name__ == "__main__":
    main()
