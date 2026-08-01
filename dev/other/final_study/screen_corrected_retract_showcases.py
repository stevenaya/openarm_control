#!/usr/bin/env python3
"""Screen recorded retracts for a representative ori/main elbow excursion."""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pr_showcase_selection_study as selection
import recorded_intervention_study as recorded
import study


OUTPUT = HERE / "results" / "corrected_retract_showcase_screen_20260731"
EPISODES = (73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93)
SIDES = ("right",)
TOP_PER_SIGNAL = 8


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _freeze(
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


def _select_events(
    episode: int,
    events: list[recorded.Event],
) -> list[recorded.Event]:
    selected: dict[int, recorded.Event] = {}
    signals = (
        lambda event: event.score,
        lambda event: event.actual_elbow_lateral_range_m,
        lambda event: event.command_elbow_lateral_range_m,
        lambda event: event.peak_command_velocity_utilization,
        lambda event: event.reach_drop_m,
        lambda event: event.peak_angular_speed_rad_s,
    )
    for signal in signals:
        for event in sorted(events, key=signal, reverse=True)[:TOP_PER_SIGNAL]:
            selected[event.index] = event

    previous = {
        event_index
        for candidate_episode, event_index in selection.CANDIDATES["retract"]
        if candidate_episode == episode
    }
    for event in events:
        if event.index in previous:
            selected[event.index] = event
    return [selected[index] for index in sorted(selected)]


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["scenario"], {})[row["profile"]] = row

    ranked: list[dict[str, Any]] = []
    for scenario, profiles in grouped.items():
        if {
            "pr_current_dataflow",
            "ori_main_defaults_with_current_velocity",
        } - profiles.keys():
            continue
        pr = profiles["pr_current_dataflow"]
        main = profiles["ori_main_defaults_with_current_velocity"]
        pr_elbow = float(pr["actual_elbow_lateral_range_m"])
        main_elbow = float(main["actual_elbow_lateral_range_m"])
        delta = main_elbow - pr_elbow
        main_error = float(main["actual_position_max_m"])
        pr_error = float(pr["actual_position_max_m"])
        failures = int(main["solver_failures"]) + int(pr["solver_failures"])
        score = (
            180.0 * delta
            + 35.0 * max(main_elbow - 0.10, 0.0)
            - 25.0 * max(main_error - 0.10, 0.0)
            - 100.0 * failures
        )
        ranked.append(
            {
                "scenario": scenario,
                "episode": int(pr["episode"]),
                "side": pr["event_side"],
                "event_index": int(pr["event_index"]),
                "score": score,
                "peak_linear_speed_m_s": float(pr["peak_linear_speed_m_s"]),
                "peak_angular_speed_rad_s": float(
                    pr["peak_angular_speed_rad_s"]
                ),
                "reach_drop_m": float(pr["reach_drop_m"]),
                "recorded_actual_elbow_range_m": float(
                    pr["recorded_actual_elbow_range_m"]
                ),
                "pr_elbow_range_m": pr_elbow,
                "main_elbow_range_m": main_elbow,
                "elbow_improvement_m": delta,
                "pr_position_max_m": pr_error,
                "main_position_max_m": main_error,
                "pr_orientation_rmse_rad": float(
                    pr["actual_orientation_rmse_rad"]
                ),
                "main_orientation_rmse_rad": float(
                    main["actual_orientation_rmse_rad"]
                ),
                "pr_velocity_saturation_fraction": float(
                    pr["command_velocity_saturation_fraction"]
                ),
                "main_velocity_saturation_fraction": float(
                    main["command_velocity_saturation_fraction"]
                ),
                "solver_failures": failures,
            }
        )
    return sorted(ranked, key=lambda row: float(row["score"]), reverse=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT / "cache"
    frozen = OUTPUT / "frozen_scenarios"
    traces = OUTPUT / "traces"
    frozen.mkdir(exist_ok=True)
    traces.mkdir(exist_ok=True)

    profiles = (
        selection._current_dataflow_profile(),
        selection._ori_main_velocity_profile(),
    )
    rows: list[dict[str, Any]] = []
    event_count = 0
    for episode in EPISODES:
        for side in SIDES:
            try:
                arm = recorded.load_record(episode, side, cache)
            except (FileNotFoundError, RuntimeError) as error:
                print(f"Skip episode {episode} {side}: {error}", flush=True)
                continue
            detected = recorded.detect_retract_events(episode, side, arm)
            events = _select_events(episode, detected)
            print(
                f"Episode {episode} {side}: {len(events)} selected from "
                f"{len(detected)} retract events",
                flush=True,
            )
            for event in events:
                event_count += 1
                scenario, source = recorded._scenario_from_event(
                    event,
                    arm,
                    "straight_retract",
                )
                _freeze(frozen / f"{scenario.name}.npz", scenario, source)
                recorded_path = (
                    traces
                    / f"recorded_hardware_{scenario.name}.npz"
                )
                if not recorded_path.exists():
                    recorded._save_recorded_trace(
                        recorded_path,
                        scenario,
                        side,
                        source,
                    )
                print(f"  {scenario.name}", flush=True)
                for profile in profiles:
                    print(f"    {profile.name}", flush=True)
                    metrics = study.run_cached(
                        profile,
                        scenario,
                        OUTPUT,
                        save_full_trace=True,
                    )
                    for metric in metrics:
                        rows.append(
                            {
                                **metric,
                                "episode": episode,
                                "event_side": side,
                                "event_index": event.index,
                                "source_start_s": event.start_s,
                                "source_end_s": event.end_s,
                                "peak_linear_speed_m_s": (
                                    event.peak_linear_speed_m_s
                                ),
                                "peak_angular_speed_rad_s": (
                                    event.peak_angular_speed_rad_s
                                ),
                                "reach_drop_m": event.reach_drop_m,
                                "recorded_actual_elbow_range_m": (
                                    event.actual_elbow_lateral_range_m
                                ),
                            }
                        )
                _write_csv(OUTPUT / "metrics.csv", rows)

    ranked = _rank(rows)
    _write_csv(OUTPUT / "ranking.csv", ranked)
    (OUTPUT / "summary.json").write_text(
        json.dumps(
            {
                "episodes": EPISODES,
                "sides": SIDES,
                "event_count": event_count,
                "profiles": [asdict(profile) for profile in profiles],
                "top_candidates": ranked[:20],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nTop corrected-baseline retract candidates:", flush=True)
    for index, row in enumerate(ranked[:20], start=1):
        print(
            f"{index:2d}. {row['scenario']} "
            f"main={100.0 * float(row['main_elbow_range_m']):.1f} cm "
            f"pr={100.0 * float(row['pr_elbow_range_m']):.1f} cm "
            f"delta={100.0 * float(row['elbow_improvement_m']):+.1f} cm "
            f"main_error={100.0 * float(row['main_position_max_m']):.1f} cm",
            flush=True,
        )


if __name__ == "__main__":
    main()
