#!/usr/bin/env python3
"""Select and freeze representative PR showcase trajectories.

The recorded dataset contains IK joint commands and measured robot state, but
not the original VR pose target.  Candidate targets in this study are therefore
documented FK(action) proxies.  The selected scenarios are frozen so later
controller revisions can replay exactly the same target and initial state.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recorded_intervention_study as recorded
import study

OUTPUT_DIR = HERE / "results" / "pr_showcase_selection_20260731"
RECENT_EPISODES = (75, 77, 79, 81, 83)

# These candidates cover the largest recent elbow excursions while retaining
# several different wrist speeds and retract directions.
CANDIDATES = {
    "chest": (
        (75, 48),
        (75, 32),
        (75, 35),
        (75, 33),
        (75, 43),
        (77, 176),
        (77, 182),
        (77, 147),
        (77, 150),
        (77, 82),
        (79, 107),
        (79, 30),
        (79, 37),
        (79, 24),
        (81, 2),
        (81, 3),
        (83, 6),
        (83, 2),
        (83, 1),
        (83, 3),
    ),
    "retract": (
        (75, 5),
        (75, 28),
        (75, 32),
        (75, 41),
        (75, 44),
        (75, 57),
        (77, 2),
        (77, 3),
        (77, 93),
        (79, 0),
        (79, 9),
        (79, 13),
        (79, 21),
        (79, 42),
        (81, 1),
        (81, 3),
        (81, 4),
        (83, 20),
        (83, 21),
        (83, 28),
        (83, 32),
    ),
}

DATAFLOW_ARGS = (
    "--tick-hz 250 --mode bimanual --origin-frame arm_origin "
    "--origin-frame-type site --max-iters 5 --pos-cost 12 --ori-cost 1.5 "
    "--damping 0.1 --posture-cost 0 --lm-damping 0.01 "
    "--frame-position-error-limit 0.02 "
    "--frame-orientation-error-limit 0.25 --limit-velocity "
    "--nullspace-cost 8.5 --nullspace-return-rate 1.6 "
    "--joint-braking-distance 0.2 --singularity-max-approach-rate 0.25 "
    "--kinetic-energy-cost 2e-5 --measured-state-timeout 0.1"
)


@dataclass(frozen=True)
class Candidate:
    """A recorded event and the target proxy used for replay."""

    family: str
    episode: int
    event_index: int
    target_style: str
    scenario: str
    source_start_s: float
    source_end_s: float
    peak_linear_speed_m_s: float
    peak_angular_speed_rad_s: float
    orientation_travel_rad: float
    reach_drop_m: float
    recorded_elbow_lateral_range_m: float
    recorded_command_elbow_lateral_range_m: float


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _events_for_episode(
    episode: int,
    family: str,
    cache_dir: Path,
) -> tuple[recorded.ArmRecord, dict[int, recorded.Event]]:
    arm = recorded.load_record(episode, "right", cache_dir)
    events = (
        recorded.detect_chest_events(episode, "right", arm)
        if family == "chest"
        else recorded.detect_retract_events(episode, "right", arm)
    )
    return arm, {event.index: event for event in events}


def _current_dataflow_profile() -> study.Profile:
    return study.make_profile(
        "pr_current_dataflow",
        position_cost=12.0,
        orientation_cost=1.5,
        lm_damping=0.01,
        damping=0.1,
        posture_cost=0.0,
        frame_position_error_limit=0.02,
        frame_orientation_error_limit=0.25,
        nullspace_cost=8.5,
        nullspace_return_rate=1.6,
        joint_braking=True,
        joint_braking_distance=0.2,
        singularity_max_approach_rate=0.25,
        kinetic_energy_cost=2e-5,
        description="Exact active dataflow arguments on 2026-07-31.",
    )


def _mainline_velocity_profile() -> study.Profile:
    return study.make_profile(
        "mainline_with_ik_velocity",
        limit_style="recoverable",
        velocity_caps=study.CONTROL_CAPS,
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=0.01,
        damping=0.1,
        posture_cost=0.01,
        frame_position_error_limit=0.0,
        frame_orientation_error_limit=0.0,
        nullspace_cost=0.0,
        joint_braking=True,
        joint_braking_distance=0.2,
        singularity_max_approach_rate=0.0,
        kinetic_energy_cost=0.0,
        description=(
            "Mainline task set with the same IK velocity/braking envelope."
        ),
    )


def _ori_main_velocity_profile() -> study.Profile:
    return study.make_profile(
        "ori_main_defaults_with_current_velocity",
        limit_style="standard",
        velocity_caps=study.CONTROL_CAPS,
        use_measured_state=False,
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=0.01,
        damping=0.25,
        posture_cost=0.01,
        frame_position_error_limit=0.0,
        frame_orientation_error_limit=0.0,
        nullspace_cost=0.0,
        joint_braking=False,
        singularity_max_approach_rate=0.0,
        kinetic_energy_cost=0.0,
        description=(
            "Exact ori/main task defaults with standard ConfigurationLimit "
            "and VelocityLimit using the current physical velocity caps. "
            "Joint braking and measured-state safety are disabled."
        ),
    )


def _mainline_profile() -> study.Profile:
    return study.make_profile(
        "mainline_no_ik_velocity",
        limit_style="configuration_only",
        velocity_caps=None,
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=0.01,
        damping=0.1,
        posture_cost=0.01,
        frame_position_error_limit=0.0,
        frame_orientation_error_limit=0.0,
        nullspace_cost=0.0,
        joint_braking=False,
        singularity_max_approach_rate=0.0,
        kinetic_energy_cost=0.0,
        description="Mainline task set without an IK velocity envelope.",
    )


def _profiles() -> tuple[study.Profile, ...]:
    current = _current_dataflow_profile()
    return (
        current,
        _mainline_velocity_profile(),
        _mainline_profile(),
        study.make_profile(
            "pr_current_no_frame_bound",
            **{
                **current.overrides,
                "frame_position_error_limit": 0.0,
                "frame_orientation_error_limit": 0.0,
            },
            description="Current dataflow profile without the 6D frame bound.",
        ),
        study.make_profile(
            "pr_current_no_nullspace",
            **{**current.overrides, "nullspace_cost": 0.0},
            description="Current dataflow profile without nullspace regulation.",
        ),
    )


def _freeze_scenario(
    path: Path,
    scenario: study.Scenario,
    source: dict[str, np.ndarray],
) -> None:
    np.savez_compressed(
        path,
        times=scenario.times,
        phase=scenario.phase,
        target_right=scenario.target_right,
        target_left=scenario.target_left,
        initial_right=scenario.initial_right,
        initial_left=scenario.initial_left,
        source_action_q=source["action_q"],
        source_obs_q=source["obs_q"],
        source_obs_dq=source["obs_dq"],
        source_action_pose=source["action_pose"],
        source_obs_pose=source["obs_pose"],
        source_action_elbow=source["action_elbow"],
        source_obs_elbow=source["obs_elbow"],
    )


def replay(output_dir: Path) -> None:
    cache_dir = output_dir / "cache"
    trace_dir = output_dir / "traces"
    frozen_dir = output_dir / "frozen_scenarios"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    frozen_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[tuple[int, str], tuple[recorded.ArmRecord, dict[int, recorded.Event]]] = {}
    candidates: list[Candidate] = []
    metric_rows: list[dict[str, Any]] = []
    profiles = _profiles()

    for family, entries in CANDIDATES.items():
        target_style = (
            "position_smoothed" if family == "chest" else "straight_retract"
        )
        for episode, event_index in entries:
            key = (episode, family)
            if key not in loaded:
                loaded[key] = _events_for_episode(
                    episode,
                    family,
                    cache_dir,
                )
            arm, event_lookup = loaded[key]
            if event_index not in event_lookup:
                raise RuntimeError(
                    f"Missing {family} event {event_index} in episode {episode}."
                )
            event = event_lookup[event_index]
            scenario, source = recorded._scenario_from_event(
                event,
                arm,
                target_style,
            )
            candidate = Candidate(
                family=family,
                episode=episode,
                event_index=event_index,
                target_style=target_style,
                scenario=scenario.name,
                source_start_s=event.start_s,
                source_end_s=event.end_s,
                peak_linear_speed_m_s=event.peak_linear_speed_m_s,
                peak_angular_speed_rad_s=event.peak_angular_speed_rad_s,
                orientation_travel_rad=event.orientation_travel_rad,
                reach_drop_m=event.reach_drop_m,
                recorded_elbow_lateral_range_m=(
                    event.actual_elbow_lateral_range_m
                ),
                recorded_command_elbow_lateral_range_m=(
                    event.command_elbow_lateral_range_m
                ),
            )
            candidates.append(candidate)

            hardware_path = (
                trace_dir
                / f"recorded_recorded_mainline_hardware_{scenario.name}.npz"
            )
            if not hardware_path.exists():
                recorded._save_recorded_trace(
                    hardware_path,
                    scenario,
                    "right",
                    source,
                )
            frozen_path = frozen_dir / f"{scenario.name}.npz"
            if not frozen_path.exists():
                _freeze_scenario(frozen_path, scenario, source)

            print(
                f"\n{family} episode {episode} event {event_index}: "
                f"{scenario.name}",
                flush=True,
            )
            for profile in profiles:
                print(f"  {profile.name}", flush=True)
                rows = study.run_cached(
                    profile,
                    scenario,
                    output_dir,
                    save_full_trace=True,
                )
                for row in rows:
                    metric_rows.append(
                        {
                            **row,
                            **asdict(candidate),
                        }
                    )
                _write_csv(output_dir / "metrics.csv", metric_rows)

    candidate_rows = [asdict(candidate) for candidate in candidates]
    _write_csv(output_dir / "candidates.csv", candidate_rows)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": str(recorded.RUN_DIR),
                "episodes_scanned": RECENT_EPISODES,
                "dataflow_args": DATAFLOW_ARGS,
                "target_semantics": (
                    "FK(action) proxy; chest position is smoothed over 0.6 s; "
                    "retract position follows a smooth start-to-end chord."
                ),
                "profiles": [
                    {
                        "name": profile.name,
                        "overrides": profile.overrides,
                        "velocity_caps": profile.velocity_caps,
                        "limit_style": profile.limit_style,
                    }
                    for profile in profiles
                ],
                "candidates": candidate_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _score_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row["side"] != "right":
            continue
        grouped.setdefault(row["scenario"], {})[row["profile"]] = row

    rankings: list[dict[str, Any]] = []
    for scenario, profiles in grouped.items():
        required = {"pr_current_dataflow", "mainline_with_ik_velocity"}
        if not required.issubset(profiles):
            continue
        current = profiles["pr_current_dataflow"]
        mainline = profiles["mainline_with_ik_velocity"]
        family = current["family"].removeprefix("recorded_")
        elbow_delta = float(
            mainline["actual_elbow_lateral_range_m"]
        ) - float(current["actual_elbow_lateral_range_m"])
        accel_delta = float(
            mainline["actual_elbow_accel_p99_m_s2"]
        ) - float(current["actual_elbow_accel_p99_m_s2"])
        position_delta = float(
            mainline["actual_position_max_m"]
        ) - float(current["actual_position_max_m"])
        recorded_elbow = float(current["recorded_elbow_lateral_range_m"])
        if family == "chest":
            score = (
                100.0 * elbow_delta
                + 0.20 * accel_delta
                + 25.0 * position_delta
                + 10.0 * recorded_elbow
            )
        else:
            score = (
                140.0 * elbow_delta
                + 0.15 * accel_delta
                + 15.0 * position_delta
                + 12.0 * recorded_elbow
            )
        rankings.append(
            {
                "family": family,
                "scenario": scenario,
                "episode": int(current["episode"]),
                "event_index": int(current["event_index"]),
                "score": score,
                "recorded_elbow_range_m": recorded_elbow,
                "current_elbow_range_m": float(
                    current["actual_elbow_lateral_range_m"]
                ),
                "mainline_elbow_range_m": float(
                    mainline["actual_elbow_lateral_range_m"]
                ),
                "elbow_range_improvement_m": elbow_delta,
                "current_elbow_accel_p99_m_s2": float(
                    current["actual_elbow_accel_p99_m_s2"]
                ),
                "mainline_elbow_accel_p99_m_s2": float(
                    mainline["actual_elbow_accel_p99_m_s2"]
                ),
                "elbow_accel_improvement_m_s2": accel_delta,
                "current_position_max_m": float(
                    current["actual_position_max_m"]
                ),
                "mainline_position_max_m": float(
                    mainline["actual_position_max_m"]
                ),
                "position_max_improvement_m": position_delta,
                "peak_linear_speed_m_s": float(
                    current["peak_linear_speed_m_s"]
                ),
                "peak_angular_speed_rad_s": float(
                    current["peak_angular_speed_rad_s"]
                ),
                "orientation_travel_rad": float(
                    current["orientation_travel_rad"]
                ),
                "reach_drop_m": float(current["reach_drop_m"]),
            }
        )
    return sorted(
        rankings,
        key=lambda row: (row["family"], -row["score"]),
    )


def _plot_rankings(rankings: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for row_index, family in enumerate(("chest", "retract")):
        values = [row for row in rankings if row["family"] == family][:10]
        labels = [
            f"ep{row['episode']}/e{row['event_index']}" for row in values
        ]
        y = np.arange(len(values))
        axes[row_index, 0].barh(
            y,
            [100.0 * row["current_elbow_range_m"] for row in values],
            color="#0f9d8a",
            label="Current PR",
        )
        axes[row_index, 0].barh(
            y,
            [100.0 * row["mainline_elbow_range_m"] for row in values],
            color="#dc493a",
            alpha=0.55,
            label="Mainline + IK cap",
        )
        axes[row_index, 0].set_yticks(y, labels)
        axes[row_index, 0].invert_yaxis()
        axes[row_index, 0].set_xlabel("simulated actual elbow lateral range [cm]")
        axes[row_index, 0].set_title(f"{family}: elbow branch motion")
        axes[row_index, 0].legend()

        axes[row_index, 1].scatter(
            [
                100.0 * row["elbow_range_improvement_m"]
                for row in values
            ],
            [
                100.0 * row["position_max_improvement_m"]
                for row in values
            ],
            s=[
                100.0 + 600.0 * row["recorded_elbow_range_m"]
                for row in values
            ],
            color="#3d5a80",
        )
        for row in values:
            axes[row_index, 1].annotate(
                f"ep{row['episode']}/e{row['event_index']}",
                (
                    100.0 * row["elbow_range_improvement_m"],
                    100.0 * row["position_max_improvement_m"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axes[row_index, 1].axvline(0.0, color="#777777", linewidth=1)
        axes[row_index, 1].axhline(0.0, color="#777777", linewidth=1)
        axes[row_index, 1].set_xlabel("PR elbow-range improvement [cm]")
        axes[row_index, 1].set_ylabel("PR max-position-error improvement [cm]")
        axes[row_index, 1].set_title(f"{family}: improvement trade-off")
        axes[row_index, 1].grid(alpha=0.25)
    figure.suptitle(
        "PR showcase candidate selection using the active dataflow profile",
        fontsize=18,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def report(output_dir: Path) -> None:
    rankings = _score_rows(_read_metrics(output_dir / "metrics.csv"))
    _write_csv(output_dir / "rankings.csv", rankings)
    (output_dir / "rankings.json").write_text(
        json.dumps(rankings, indent=2),
        encoding="utf-8",
    )
    _plot_rankings(rankings, output_dir / "candidate_rankings.png")
    for family in ("chest", "retract"):
        print(f"\nTop {family} candidates:", flush=True)
        for row in (
            value for value in rankings if value["family"] == family
        ):
            print(
                f"  ep{row['episode']} event{row['event_index']} "
                f"score={row['score']:.2f} "
                f"elbow PR/main={100*row['current_elbow_range_m']:.1f}/"
                f"{100*row['mainline_elbow_range_m']:.1f}cm "
                f"recorded={100*row['recorded_elbow_range_m']:.1f}cm "
                f"pos PR/main={100*row['current_position_max_m']:.1f}/"
                f"{100*row['mainline_position_max_m']:.1f}cm "
                f"w={row['peak_angular_speed_rad_s']:.1f}rad/s",
                flush=True,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("replay", "report", "all"),
        default="all",
        nargs="?",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    if args.command in {"replay", "all"}:
        replay(output_dir)
    if args.command in {"report", "all"}:
        report(output_dir)


if __name__ == "__main__":
    main()
