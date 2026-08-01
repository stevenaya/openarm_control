#!/usr/bin/env python3
"""Find and replay recorded bimanual circle trajectories.

The intervention dataset does not contain the original VR controller poses.
As in the other recorded-replay studies, this script uses FK of the recorded
IK command as a documented end-effector target proxy.  Both arms are replayed
from the same recorded window so controller variants see identical targets and
initial joint configurations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pr_showcase_selection_study as showcase
import recorded_intervention_study as recorded
import study

OUTPUT_DIR = HERE / "results" / "bimanual_circle_selection_20260731"
CACHE_DIR = OUTPUT_DIR / "cache"
RECENT_INTERVENTIONS = (73, 75, 77, 79, 81, 83)
DT = study.CONTROL_DT
ORI_MAIN_VELOCITY_CAPS = (
    1.57,
    1.57,
    3.14,
    3.14,
    12.6,
    12.6,
    12.6,
)

# Windows selected from the automatic multi-duration scan.  Padding includes
# the onset and completion of each repeated motion rather than cutting exactly
# at the highest-scoring circular subsection.
REPLAY_WINDOWS = (
    ("ep77_long", 77, 184.00, 216.00),
    ("ep77_skyward", 77, 207.30, 214.30),
    ("ep81_all", 81, 0.00, 13.50),
    ("ep81_dense", 81, 3.80, 12.80),
    ("ep83_early_long", 83, 0.00, 22.00),
    ("ep83_early", 83, 3.90, 16.90),
    ("ep83_late", 83, 26.80, 39.45),
    ("ep79_regular", 79, 303.50, 310.55),
    ("ep79_mid", 79, 250.90, 262.05),
)

# Fixed-duration matches to episode 77's 205.60-215.60 s bimanual
# trajectory. Each candidate is initialized directly from its own first frame.
SIMILAR_WINDOWS = (
    ("scan_ep83_late", 83, 27.28, 39.28),
    ("scan_ep79_late", 79, 304.05, 310.05),
    ("scan_ep83_early", 83, 4.46, 16.46),
    ("scan_ep81", 81, 4.22, 12.22),
    ("scan_ep77_tail", 77, 207.82, 213.82),
    ("scan_ep83_middle", 83, 13.89, 21.89),
    ("scan_ep75", 75, 162.44, 172.44),
    ("ep77_tail_direct", 77, 205.60, 215.60),
    ("ep81_match", 81, 0.00, 10.00),
    ("ep79_match_late", 79, 306.53, 316.53),
    ("ep79_match_early", 79, 297.60, 307.60),
    ("ep83_match_early", 83, 0.25, 10.25),
    ("ep83_match_late", 83, 25.79, 35.79),
    ("ep79_match_high_elbow", 79, 342.98, 352.98),
    ("ep77_match_early", 77, 187.74, 197.74),
    ("ep83_match_middle", 83, 17.11, 27.11),
    ("ep77_match_middle", 77, 195.42, 205.42),
    ("ep79_match_middle", 79, 250.23, 260.23),
    ("ep77_match_after", 77, 214.27, 224.27),
)


@dataclass(frozen=True)
class CircleMetrics:
    """Geometric descriptors for one hand path in a candidate window."""

    path_length_m: float
    closure_ratio: float
    plane_axis_ratio: float
    planarity: float
    net_turn_rad: float
    total_turn_rad: float
    turn_directionality: float
    active_fraction: float
    elbow_rise_m: float
    elbow_range_m: float


@dataclass(frozen=True)
class CircleCandidate:
    """One non-overlapping bimanual window from the automatic scan."""

    rank: int
    episode: int
    start_s: float
    end_s: float
    duration_s: float
    score: float
    circle_score: float
    right: CircleMetrics
    left: CircleMetrics


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _smooth(values: np.ndarray, window_s: float = 0.08) -> np.ndarray:
    window = max(5, round(window_s / DT))
    if window % 2 == 0:
        window += 1
    if window >= values.shape[0]:
        return values.copy()
    return savgol_filter(
        values,
        window_length=window,
        polyorder=2,
        axis=0,
        mode="interp",
    )


def _circle_metrics(
    positions: np.ndarray,
    elbows: np.ndarray,
    start: int,
    end: int,
) -> CircleMetrics:
    path = positions[start:end]
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    path_length = float(np.sum(segments))
    chord = float(np.linalg.norm(path[-1] - path[0]))

    centered = path - np.mean(path, axis=0)
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    plane_axis_ratio = float(
        singular_values[1] / max(singular_values[0], 1.0e-9)
    )
    planarity = float(
        1.0 - singular_values[2] / max(singular_values[1], 1.0e-9)
    )
    plane = centered @ axes[:2].T
    plane /= np.maximum(np.std(plane, axis=0), 1.0e-6)
    angle = np.unwrap(np.arctan2(plane[:, 1], plane[:, 0]))
    angle_steps = np.diff(angle)
    total_turn = float(np.sum(np.abs(angle_steps)))
    net_turn = float(abs(angle[-1] - angle[0]))

    elbow_z = elbows[start:end, 2]
    baseline_count = max(10, round(0.4 / DT))
    elbow_baseline = float(np.median(elbow_z[:baseline_count]))
    return CircleMetrics(
        path_length_m=path_length,
        closure_ratio=chord / max(path_length, 1.0e-9),
        plane_axis_ratio=plane_axis_ratio,
        planarity=planarity,
        net_turn_rad=net_turn,
        total_turn_rad=total_turn,
        turn_directionality=net_turn / max(total_turn, 1.0e-9),
        active_fraction=float(np.mean(segments / DT > 0.035)),
        elbow_rise_m=float(np.max(elbow_z) - elbow_baseline),
        elbow_range_m=float(np.ptp(elbow_z)),
    )


def _candidate_score(
    right: CircleMetrics,
    left: CircleMetrics,
) -> tuple[float, float]:
    loops = min(right.net_turn_rad, left.net_turn_rad) / (2.0 * np.pi)
    shape = min(right.plane_axis_ratio, left.plane_axis_ratio)
    planarity = min(right.planarity, left.planarity)
    active = min(right.active_fraction, left.active_fraction)
    closure = max(right.closure_ratio, left.closure_ratio)
    directionality = min(
        right.turn_directionality,
        left.turn_directionality,
    )
    minimum_path = min(right.path_length_m, left.path_length_m)
    circle_score = (
        loops
        * min(shape / 0.45, 1.0)
        * min(planarity / 0.75, 1.0)
        * min(active / 0.65, 1.0)
        * min(directionality / 0.65, 1.0)
        * np.exp(-1.8 * closure)
        * min(minimum_path / 0.7, 1.0)
    )
    elbow_rise = max(right.elbow_rise_m, left.elbow_rise_m)
    return float(circle_score * (1.0 + 3.0 * max(elbow_rise, 0.0))), float(
        circle_score
    )


def scan_candidates() -> list[CircleCandidate]:
    """Scan recent intervention records for bimanual closed-loop motion."""
    windows: list[
        tuple[
            float,
            float,
            int,
            float,
            float,
            CircleMetrics,
            CircleMetrics,
        ]
    ] = []
    stride = round(0.25 / DT)
    for episode in RECENT_INTERVENTIONS:
        right_record = recorded.load_record(episode, "right", CACHE_DIR)
        left_record = recorded.load_record(episode, "left", CACHE_DIR)
        count = min(right_record.time.size, left_record.time.size)
        right_positions = _smooth(right_record.action_world_eef[:count])
        left_positions = _smooth(left_record.action_world_eef[:count])
        for duration_s in (6.0, 8.0, 10.0, 12.0):
            width = round(duration_s / DT)
            if width >= count:
                continue
            for start in range(0, count - width, stride):
                end = start + width
                right = _circle_metrics(
                    right_positions,
                    right_record.action_elbow,
                    start,
                    end,
                )
                left = _circle_metrics(
                    left_positions,
                    left_record.action_elbow,
                    start,
                    end,
                )
                score, circle_score = _candidate_score(right, left)
                windows.append(
                    (
                        score,
                        circle_score,
                        episode,
                        start * DT,
                        end * DT,
                        right,
                        left,
                    )
                )

    windows.sort(key=lambda item: item[0], reverse=True)
    selected: list[CircleCandidate] = []
    for score, circle_score, episode, start_s, end_s, right, left in windows:
        if circle_score < 0.25:
            continue
        overlap = any(
            episode == previous.episode
            and min(end_s, previous.end_s) - max(start_s, previous.start_s)
            > 0.35
            * min(end_s - start_s, previous.end_s - previous.start_s)
            for previous in selected
        )
        if overlap:
            continue
        selected.append(
            CircleCandidate(
                rank=len(selected),
                episode=episode,
                start_s=start_s,
                end_s=end_s,
                duration_s=end_s - start_s,
                score=score,
                circle_score=circle_score,
                right=right,
                left=left,
            )
        )
        if len(selected) >= 40:
            break

    rows: list[dict[str, Any]] = []
    for candidate in selected:
        row = {
            key: value
            for key, value in asdict(candidate).items()
            if key not in {"right", "left"}
        }
        for side in ("right", "left"):
            for key, value in asdict(getattr(candidate, side)).items():
                row[f"{side}_{key}"] = value
        rows.append(row)
    _write_csv(OUTPUT_DIR / "candidate_scan.csv", rows)
    return selected


def _scenario_from_window(
    label: str,
    episode: int,
    start_s: float,
    end_s: float,
) -> tuple[study.Scenario, dict[str, np.ndarray]]:
    right = recorded.load_record(episode, "right", CACHE_DIR)
    left = recorded.load_record(episode, "left", CACHE_DIR)
    start = max(0, round(start_s / DT))
    end = min(right.time.size, left.time.size, round(end_s / DT))
    if end - start < 2:
        raise ValueError(f"Empty replay window {label}.")

    hold_before = round(0.4 / DT)
    hold_after = round(0.6 / DT)

    def padded(values: np.ndarray) -> np.ndarray:
        selected = values[start:end]
        return np.concatenate(
            (
                np.repeat(selected[:1], hold_before, axis=0),
                selected,
                np.repeat(selected[-1:], hold_after, axis=0),
            )
        )

    target_right = padded(right.action_pose)
    target_left = padded(left.action_pose)
    count = target_right.shape[0]
    phase = np.concatenate(
        (
            np.zeros(hold_before, dtype=np.int32),
            np.ones(end - start, dtype=np.int32),
            np.full(hold_after, 2, dtype=np.int32),
        )
    )
    right_speed = np.linalg.norm(
        np.diff(right.action_world_eef[start:end], axis=0),
        axis=1,
    ) / DT
    left_speed = np.linalg.norm(
        np.diff(left.action_world_eef[start:end], axis=0),
        axis=1,
    ) / DT
    peak_speed = float(
        max(
            np.quantile(right_speed, 0.99),
            np.quantile(left_speed, 0.99),
        )
    )
    scenario = study.Scenario(
        name=f"recorded_bimanual_circle_{label}",
        family="recorded_bimanual_circle",
        mode="bimanual",
        speed=peak_speed,
        times=np.arange(count, dtype=np.float64) * DT,
        phase=phase,
        target_right=target_right,
        target_left=target_left,
        initial_right=right.action_q[start].copy(),
        initial_left=left.action_q[start].copy(),
        description=(
            f"Episode {episode}, {start_s:.2f}-{end_s:.2f} s; "
            "bimanual FK(action) target proxy with 0.4 s pre-hold."
        ),
    )
    source = {
        "episode": np.asarray(episode),
        "source_start_s": np.asarray(start_s),
        "source_end_s": np.asarray(end_s),
        "source_right_action_q": right.action_q[start:end],
        "source_left_action_q": left.action_q[start:end],
        "source_right_obs_q": right.obs_q[start:end],
        "source_left_obs_q": left.obs_q[start:end],
        "source_right_action_pose": right.action_pose[start:end],
        "source_left_action_pose": left.action_pose[start:end],
        "source_right_action_elbow": right.action_elbow[start:end],
        "source_left_action_elbow": left.action_elbow[start:end],
        "source_right_obs_elbow": right.obs_elbow[start:end],
        "source_left_obs_elbow": left.obs_elbow[start:end],
    }
    return scenario, source


def _profiles() -> tuple[study.Profile, ...]:
    current = showcase._current_dataflow_profile()
    return (
        current,
        study.make_profile(
            "pr_current_no_nullspace",
            **{**current.overrides, "nullspace_cost": 0.0},
            description="Current dataflow profile without nullspace regulation.",
        ),
        showcase._mainline_velocity_profile(),
        showcase._mainline_profile(),
    )


def _ori_main_velocity_profile() -> study.Profile:
    """Represent ori/main with the same numeric IK velocity caps as the PR."""
    return study.make_profile(
        "ori_main_same_ik_velocity",
        limit_style="standard",
        velocity_caps=study.CONTROL_CAPS,
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=0.01,
        damping=0.1,
        posture_cost=0.01,
        # Current Kinematics divides dt across iterations. Multiplying the
        # outer period here reproduces ori/main's full 4 ms per iteration.
        dt=DT * 5,
        max_iters=5,
        frame_position_error_limit=0.0,
        frame_orientation_error_limit=0.0,
        nullspace_cost=0.0,
        joint_braking=False,
        singularity_max_approach_rate=0.0,
        kinetic_energy_cost=0.0,
        use_measured_state=False,
        description=(
            "ori/main FrameTask, full-home PostureTask, standard Mink "
            "configuration/velocity limits, and the PR's numeric velocity "
            "caps. Each of five iterations retains ori/main's full 4 ms "
            "integration period."
        ),
    )


def _trace_path(profile: str, scenario: str) -> Path:
    matches = list(
        (OUTPUT_DIR / "traces").glob(f"*_{profile}_{scenario}.npz")
    )
    if not matches:
        raise RuntimeError(f"Missing trace for {profile}/{scenario}.")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _side_trace_metrics(
    trace: np.lib.npyio.NpzFile,
    side: str,
) -> dict[str, float]:
    phase = trace["phase"]
    active = phase == 1
    target = trace[f"{side}_target_pose"][active]
    actual = trace[f"{side}_actual_pose"][active]
    command_elbow = trace[f"{side}_command_elbow"][active]
    actual_elbow = trace[f"{side}_actual_elbow"][active]
    baseline_count = max(10, round(0.4 / DT))
    command_baseline = float(
        np.median(command_elbow[:baseline_count, 2])
    )
    actual_baseline = float(np.median(actual_elbow[:baseline_count, 2]))
    position_error = np.linalg.norm(target[:, :3] - actual[:, :3], axis=1)
    return {
        "command_elbow_rise_m": float(
            np.max(command_elbow[:, 2]) - command_baseline
        ),
        "actual_elbow_rise_m": float(
            np.max(actual_elbow[:, 2]) - actual_baseline
        ),
        "command_elbow_z_range_m": float(np.ptp(command_elbow[:, 2])),
        "actual_elbow_z_range_m": float(np.ptp(actual_elbow[:, 2])),
        "actual_elbow_lateral_range_m": float(
            np.ptp(actual_elbow[:, 1])
        ),
        "position_error_max_m": float(np.max(position_error)),
        "position_error_p95_m": float(np.quantile(position_error, 0.95)),
        "actual_joint_speed_max_rad_s": float(
            np.max(np.abs(trace[f"{side}_actual_dq"][active]))
        ),
        "actual_joint_accel_p99_rad_s2": float(
            np.quantile(
                np.max(
                    np.abs(trace[f"{side}_actual_ddq"][active]),
                    axis=1,
                ),
                0.99,
            )
        ),
    }


def replay_windows() -> list[dict[str, Any]]:
    """Replay fixed candidate windows and collect vertical-elbow metrics."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen_dir = OUTPUT_DIR / "frozen_scenarios"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    profiles = _profiles()
    rows: list[dict[str, Any]] = []

    for label, episode, start_s, end_s in REPLAY_WINDOWS:
        scenario, source = _scenario_from_window(
            label,
            episode,
            start_s,
            end_s,
        )
        np.savez_compressed(
            frozen_dir / f"{scenario.name}.npz",
            times=scenario.times,
            phase=scenario.phase,
            target_right=scenario.target_right,
            target_left=scenario.target_left,
            initial_right=scenario.initial_right,
            initial_left=scenario.initial_left,
            **source,
        )
        for profile in profiles:
            print(f"{scenario.name}: {profile.name}", flush=True)
            study.run_cached(
                profile,
                scenario,
                OUTPUT_DIR,
                save_full_trace=True,
            )
            with np.load(
                _trace_path(profile.name, scenario.name),
                allow_pickle=False,
            ) as trace:
                per_side = {
                    side: _side_trace_metrics(trace, side)
                    for side in study.SIDES
                }
                row: dict[str, Any] = {
                    "label": label,
                    "episode": episode,
                    "source_start_s": start_s,
                    "source_end_s": end_s,
                    "scenario": scenario.name,
                    "profile": profile.name,
                }
                for side, values in per_side.items():
                    row.update(
                        {
                            f"{side}_{key}": value
                            for key, value in values.items()
                        }
                    )
                row["max_actual_elbow_rise_m"] = max(
                    values["actual_elbow_rise_m"]
                    for values in per_side.values()
                )
                row["max_position_error_m"] = max(
                    values["position_error_max_m"]
                    for values in per_side.values()
                )
                rows.append(row)
                _write_csv(OUTPUT_DIR / "replay_metrics.csv", rows)

    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": str(recorded.RUN_DIR),
                "episodes_scanned": RECENT_INTERVENTIONS,
                "replay_windows": REPLAY_WINDOWS,
                "target_semantics": (
                    "Bimanual FK(action) proxy from the recorded command; "
                    "no extra path smoothing or geometric replacement."
                ),
                "profiles": [
                    {
                        "name": profile.name,
                        "overrides": profile.overrides,
                        "velocity_caps": profile.velocity_caps,
                        "driver_velocity_caps": profile.driver_velocity_caps,
                        "limit_style": profile.limit_style,
                    }
                    for profile in profiles
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def replay_similar_windows() -> list[dict[str, Any]]:
    """Replay trajectory-shape matches from their own initial states."""
    similar_dir = OUTPUT_DIR / "similar"
    frozen_dir = similar_dir / "frozen_scenarios"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    profiles = (
        showcase._current_dataflow_profile(),
        showcase._mainline_velocity_profile(),
    )
    rows: list[dict[str, Any]] = []

    for label, episode, start_s, end_s in SIMILAR_WINDOWS:
        scenario, source = _scenario_from_window(
            label,
            episode,
            start_s,
            end_s,
        )
        np.savez_compressed(
            frozen_dir / f"{scenario.name}.npz",
            times=scenario.times,
            phase=scenario.phase,
            target_right=scenario.target_right,
            target_left=scenario.target_left,
            initial_right=scenario.initial_right,
            initial_left=scenario.initial_left,
            **source,
        )
        traces: dict[str, np.lib.npyio.NpzFile] = {}
        try:
            for profile in profiles:
                print(f"{scenario.name}: {profile.name}", flush=True)
                study.run_cached(
                    profile,
                    scenario,
                    similar_dir,
                    save_full_trace=True,
                )
                matches = list(
                    (similar_dir / "traces").glob(
                        f"*_{profile.name}_{scenario.name}.npz"
                    )
                )
                traces[profile.name] = np.load(
                    max(matches, key=lambda path: path.stat().st_mtime_ns),
                    allow_pickle=False,
                )

            current = traces["pr_current_dataflow"]
            mainline = traces["mainline_with_ik_velocity"]
            active = current["phase"] == 1
            active_indices = np.flatnonzero(active)
            row: dict[str, Any] = {
                "label": label,
                "episode": episode,
                "source_start_s": start_s,
                "source_end_s": end_s,
                "scenario": scenario.name,
                "duration_s": end_s - start_s,
            }
            peak_difference = -np.inf
            for side in study.SIDES:
                difference = (
                    mainline[f"{side}_actual_elbow"][active, 2]
                    - current[f"{side}_actual_elbow"][active, 2]
                )
                local_index = int(np.argmax(difference))
                trace_index = int(active_indices[local_index])
                side_peak = float(difference[local_index])
                row[f"{side}_peak_mainline_minus_pr_elbow_height_m"] = (
                    side_peak
                )
                row[f"{side}_peak_difference_replay_time_s"] = float(
                    current["times"][trace_index]
                )
                for profile_name, trace in (
                    ("pr", current),
                    ("mainline", mainline),
                ):
                    position_error = np.linalg.norm(
                        trace[f"{side}_target_pose"][active, :3]
                        - trace[f"{side}_actual_pose"][active, :3],
                        axis=1,
                    )
                    row[f"{side}_{profile_name}_position_error_max_m"] = (
                        float(np.max(position_error))
                    )
                    elbow_z = trace[f"{side}_actual_elbow"][active, 2]
                    row[f"{side}_{profile_name}_elbow_z_range_m"] = float(
                        np.ptp(elbow_z)
                    )
                peak_difference = max(peak_difference, side_peak)
            row["max_peak_mainline_minus_pr_elbow_height_m"] = float(
                peak_difference
            )
            row["pr_position_error_max_m"] = max(
                row[f"{side}_pr_position_error_max_m"]
                for side in study.SIDES
            )
            row["mainline_position_error_max_m"] = max(
                row[f"{side}_mainline_position_error_max_m"]
                for side in study.SIDES
            )
            row["score"] = (
                100.0 * row["max_peak_mainline_minus_pr_elbow_height_m"]
                + 10.0
                * (
                    row["mainline_position_error_max_m"]
                    - row["pr_position_error_max_m"]
                )
            )
            rows.append(row)
            _write_csv(similar_dir / "similar_ranking.csv", rows)
        finally:
            for trace in traces.values():
                trace.close()

    rows.sort(key=lambda row: row["score"], reverse=True)
    _write_csv(similar_dir / "similar_ranking.csv", rows)
    return rows


def rank_replays(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank windows by mainline-versus-PR elbow-rise separation."""
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["scenario"], {})[row["profile"]] = row
    ranking: list[dict[str, Any]] = []
    for scenario, profiles in grouped.items():
        required = {"pr_current_dataflow", "mainline_with_ik_velocity"}
        if not required.issubset(profiles):
            continue
        current = profiles["pr_current_dataflow"]
        mainline = profiles["mainline_with_ik_velocity"]
        elbow_delta = (
            mainline["max_actual_elbow_rise_m"]
            - current["max_actual_elbow_rise_m"]
        )
        error_delta = (
            mainline["max_position_error_m"]
            - current["max_position_error_m"]
        )
        ranking.append(
            {
                "scenario": scenario,
                "label": current["label"],
                "episode": current["episode"],
                "source_start_s": current["source_start_s"],
                "source_end_s": current["source_end_s"],
                "score": 120.0 * elbow_delta + 15.0 * error_delta,
                "pr_elbow_rise_m": current["max_actual_elbow_rise_m"],
                "mainline_velocity_elbow_rise_m": mainline[
                    "max_actual_elbow_rise_m"
                ],
                "elbow_rise_improvement_m": elbow_delta,
                "pr_position_error_max_m": current["max_position_error_m"],
                "mainline_velocity_position_error_max_m": mainline[
                    "max_position_error_m"
                ],
                "position_error_improvement_m": error_delta,
            }
        )
    ranking.sort(key=lambda row: row["score"], reverse=True)
    _write_csv(OUTPUT_DIR / "replay_ranking.csv", ranking)
    return ranking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("scan", "replay", "similar", "rank", "all"),
        default="all",
        nargs="?",
    )
    args = parser.parse_args()

    if args.command in {"scan", "all"}:
        candidates = scan_candidates()
        for candidate in candidates[:15]:
            print(
                f"{candidate.rank:02d} ep{candidate.episode} "
                f"{candidate.start_s:.2f}-{candidate.end_s:.2f}s "
                f"score={candidate.score:.3f} "
                f"circle={candidate.circle_score:.3f}",
                flush=True,
            )
    rows: list[dict[str, Any]] = []
    if args.command in {"replay", "all"}:
        rows = replay_windows()
    if args.command in {"similar", "all"}:
        similar = replay_similar_windows()
        print("\nSimilar-trajectory replay ranking:")
        for rank, row in enumerate(similar):
            print(
                f"{rank:02d} {row['label']}: "
                "aligned elbow delta="
                f"{row['max_peak_mainline_minus_pr_elbow_height_m']:.3f}m, "
                f"PR/mainline max error={row['pr_position_error_max_m']:.3f}/"
                f"{row['mainline_position_error_max_m']:.3f}m",
                flush=True,
            )
    if args.command in {"rank", "all"}:
        if not rows:
            metrics_path = OUTPUT_DIR / "replay_metrics.csv"
            with metrics_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                for key, value in tuple(row.items()):
                    if key not in {
                        "label",
                        "scenario",
                        "profile",
                    }:
                        row[key] = float(value)
        rankings = rank_replays(rows)
        print("\nReplay ranking:")
        for rank, row in enumerate(rankings):
            print(
                f"{rank:02d} {row['label']}: "
                f"PR={row['pr_elbow_rise_m']:.3f}m, "
                "mainline+velocity="
                f"{row['mainline_velocity_elbow_rise_m']:.3f}m, "
                f"delta={row['elbow_rise_improvement_m']:+.3f}m, "
                f"error delta={row['position_error_improvement_m']:+.3f}m",
                flush=True,
            )


if __name__ == "__main__":
    main()
