#!/usr/bin/env python3
"""Analyze and replay the latest recorded VR intervention trajectories.

The dataset stores the IK joint command and measured driver state, but not the
raw VR target.  FK of the action stream is therefore a command-pose proxy, not
the original operator target.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import study

RUN_DIR = Path("/hdd_data/rollout/pillow_0702_tune_llm")
EPISODES = {"chest": 73, "retract": 75}
DT = study.CONTROL_DT
CAPS = np.asarray(study.CONTROL_CAPS, dtype=np.float64)
ANALYSIS_DIR = HERE / "results" / "recorded_interventions_20260731"
REPLAY_DIR = (
    HERE
    / "results"
    / "current_pr_20260730"
    / "recorded_replay"
)
SELECTED_EVENTS = (
    ("chest", "right", 62, "position_smoothed"),
    ("chest", "right", 44, "position_smoothed"),
    # Additional episode-73 cases selected by the path-relative follow-up:
    # 40 has the largest same-time hardware error, 50 has the largest
    # command-side short-horizon reversal, and 85 has the largest measured
    # cross-track departure from the recent command path.
    ("chest", "right", 40, "position_smoothed"),
    ("chest", "right", 50, "position_smoothed"),
    ("chest", "right", 85, "position_smoothed"),
    ("retract", "right", 5, "straight_retract"),
    ("retract", "right", 41, "straight_retract"),
)


@dataclass
class ArmRecord:
    """Aligned action and measured state for one recorded arm."""

    time: np.ndarray
    timestamp_s: np.ndarray
    action_q: np.ndarray
    obs_q: np.ndarray
    obs_dq: np.ndarray
    action_pose: np.ndarray
    obs_pose: np.ndarray
    action_world_eef: np.ndarray
    obs_world_eef: np.ndarray
    action_elbow: np.ndarray
    obs_elbow: np.ndarray
    shoulder: np.ndarray


@dataclass(frozen=True)
class Event:
    """One automatically detected motion window."""

    family: str
    episode: int
    side: str
    index: int
    start: int
    end: int
    core_start: int
    core_end: int
    start_s: float
    end_s: float
    duration_s: float
    score: float
    peak_linear_speed_m_s: float
    peak_angular_speed_rad_s: float
    orientation_travel_rad: float
    reach_start_m: float
    reach_end_m: float
    reach_drop_m: float
    command_chord_dip_m: float
    actual_chord_dip_m: float
    actual_below_command_m: float
    command_elbow_lateral_range_m: float
    actual_elbow_lateral_range_m: float
    peak_command_velocity_utilization: float


def _load_stream(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    schema = pq.read_schema(path)
    columns = ["timestamp", "qpos"]
    if "qvel" in schema.names:
        columns.append("qvel")
    table = pq.read_table(path, columns=columns)
    timestamp = np.asarray(
        table["timestamp"].cast("int64").to_numpy(),
        dtype=np.int64,
    ).astype(np.float64) * 1.0e-9
    qpos = np.asarray(table["qpos"].to_pylist(), dtype=np.float64)[:, :7]
    qvel = (
        np.asarray(table["qvel"].to_pylist(), dtype=np.float64)[:, :7]
        if "qvel" in columns
        else None
    )
    keep = np.concatenate([[True], np.diff(timestamp) > 1.0e-6])
    return timestamp[keep], qpos[keep], None if qvel is None else qvel[keep]


def _interpolate(
    timestamp: np.ndarray,
    values: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(target_time, timestamp, values[:, column])
            for column in range(values.shape[1])
        ]
    )


def _geometry(side: str, q_values: np.ndarray) -> dict[str, np.ndarray]:
    setup = study.make_setup("bimanual")
    model = setup.model
    data = setup.data
    resolver = setup.joint_resolver
    shoulder_id = model.joint(f"openarm_{side}_joint1").id
    elbow_id = model.joint(f"openarm_{side}_joint4").id
    frame_id = model.site(f"{side}_ee_control_point").id

    pose = np.empty((q_values.shape[0], 7), dtype=np.float64)
    world_eef = np.empty((q_values.shape[0], 3), dtype=np.float64)
    elbow = np.empty((q_values.shape[0], 3), dtype=np.float64)
    shoulder = np.empty((q_values.shape[0], 3), dtype=np.float64)
    for index, arm_q in enumerate(q_values):
        resolver.set_qpos(
            data.qpos,
            np.append(np.asarray(arm_q, dtype=np.float64), 0.0),
            side,
        )
        mujoco.mj_forward(model, data)
        pose[index] = setup.read_ee_pose(side)
        world_eef[index] = data.site_xpos[frame_id]
        elbow[index] = data.xanchor[elbow_id]
        shoulder[index] = data.xanchor[shoulder_id]
    return {
        "pose": pose,
        "world_eef": world_eef,
        "elbow": elbow,
        "shoulder": shoulder,
    }


def load_record(
    episode: int,
    side: str,
    cache_dir: Path,
) -> ArmRecord:
    """Load, align, and cache one arm at the 250 Hz controller grid."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"episode_{episode}_{side}.npz"
    if cache_path.exists():
        values = np.load(cache_path, allow_pickle=False)
        return ArmRecord(
            **{name: values[name] for name in ArmRecord.__annotations__}
        )

    root = RUN_DIR / "dataset" / "episodes" / str(episode)
    action_time, action_q_raw, _ = _load_stream(
        root / "action" / "arms" / side / "state.parquet"
    )
    obs_time, obs_q_raw, obs_dq_raw = _load_stream(
        root / "obs" / "arms" / side / "state.parquet"
    )
    if obs_dq_raw is None:
        raise RuntimeError(f"Episode {episode} {side} has no measured qvel.")

    start = max(action_time[0], obs_time[0])
    stop = min(action_time[-1], obs_time[-1])
    timestamp_s = np.arange(start, stop, DT)
    action_q = _interpolate(action_time, action_q_raw, timestamp_s)
    obs_q = _interpolate(obs_time, obs_q_raw, timestamp_s)
    obs_dq = _interpolate(obs_time, obs_dq_raw, timestamp_s)
    action_geometry = _geometry(side, action_q)
    obs_geometry = _geometry(side, obs_q)
    record = ArmRecord(
        time=timestamp_s - timestamp_s[0],
        timestamp_s=timestamp_s,
        action_q=action_q,
        obs_q=obs_q,
        obs_dq=obs_dq,
        action_pose=action_geometry["pose"],
        obs_pose=obs_geometry["pose"],
        action_world_eef=action_geometry["world_eef"],
        obs_world_eef=obs_geometry["world_eef"],
        action_elbow=action_geometry["elbow"],
        obs_elbow=obs_geometry["elbow"],
        shoulder=action_geometry["shoulder"],
    )
    np.savez_compressed(cache_path, **asdict(record))
    return record


def _smooth(values: np.ndarray, window_s: float = 0.044) -> np.ndarray:
    count = values.shape[0]
    window = max(5, round(window_s / DT))
    if window % 2 == 0:
        window += 1
    window = min(window, count - (1 - count % 2))
    if window < 5:
        return values.copy()
    return savgol_filter(
        values,
        window_length=window,
        polyorder=2,
        axis=0,
        mode="interp",
    )


def _derivative(values: np.ndarray, window_s: float = 0.044) -> np.ndarray:
    smoothed = _smooth(values, window_s)
    return np.gradient(smoothed, DT, axis=0)


def _angular_speed(pose: np.ndarray) -> np.ndarray:
    quaternion_xyzw = pose[:, [4, 5, 6, 3]]
    rotation = Rotation.from_quat(quaternion_xyzw)
    relative = rotation[:-1].inv() * rotation[1:]
    raw = np.zeros(pose.shape[0], dtype=np.float64)
    raw[1:] = relative.magnitude() / DT
    return _smooth(raw, 0.044)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
    ]


def _merge_runs(
    runs: list[tuple[int, int]],
    max_gap_s: float,
) -> list[tuple[int, int]]:
    if not runs:
        return []
    max_gap = round(max_gap_s / DT)
    merged = [runs[0]]
    for start, end in runs[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end <= max_gap:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def _window_chord_dip(
    z: np.ndarray,
    start: int,
    end: int,
) -> float:
    count = end - start
    if count < 2:
        return 0.0
    chord = np.linspace(z[start], z[end - 1], count)
    return float(max(0.0, np.max(chord - z[start:end])))


def _event(
    *,
    family: str,
    episode: int,
    side: str,
    index: int,
    record: ArmRecord,
    start: int,
    end: int,
    core_start: int,
    core_end: int,
    linear_speed: np.ndarray,
    angular_speed: np.ndarray,
    reach: np.ndarray,
    score: float,
) -> Event:
    action_lateral = (
        record.action_elbow[start:end, 1] - record.shoulder[start:end, 1]
    )
    obs_lateral = (
        record.obs_elbow[start:end, 1] - record.shoulder[start:end, 1]
    )
    action_dq = _derivative(record.action_q[start:end])
    return Event(
        family=family,
        episode=episode,
        side=side,
        index=index,
        start=start,
        end=end,
        core_start=core_start,
        core_end=core_end,
        start_s=float(record.time[start]),
        end_s=float(record.time[end - 1]),
        duration_s=float((core_end - core_start) * DT),
        score=float(score),
        peak_linear_speed_m_s=float(
            np.max(linear_speed[core_start:core_end])
        ),
        peak_angular_speed_rad_s=float(
            np.max(angular_speed[core_start:core_end])
        ),
        orientation_travel_rad=float(
            np.sum(angular_speed[core_start:core_end]) * DT
        ),
        reach_start_m=float(reach[core_start]),
        reach_end_m=float(reach[core_end - 1]),
        reach_drop_m=float(reach[core_start] - reach[core_end - 1]),
        command_chord_dip_m=_window_chord_dip(
            record.action_world_eef[:, 2],
            core_start,
            core_end,
        ),
        actual_chord_dip_m=_window_chord_dip(
            record.obs_world_eef[:, 2],
            core_start,
            core_end,
        ),
        actual_below_command_m=float(
            max(
                0.0,
                np.max(
                    record.action_world_eef[core_start:core_end, 2]
                    - record.obs_world_eef[core_start:core_end, 2]
                ),
            )
        ),
        command_elbow_lateral_range_m=float(np.ptp(action_lateral)),
        actual_elbow_lateral_range_m=float(np.ptp(obs_lateral)),
        peak_command_velocity_utilization=float(
            np.max(np.abs(action_dq) / CAPS)
        ),
    )


def detect_chest_events(
    episode: int,
    side: str,
    record: ArmRecord,
) -> list[Event]:
    position = record.action_world_eef
    linear_speed = np.linalg.norm(_derivative(position), axis=1)
    angular_speed = _angular_speed(record.action_pose)
    reach = np.linalg.norm(position - record.shoulder, axis=1)
    mask = (
        (angular_speed > 1.5)
        & (linear_speed > 0.025)
        & (reach < 0.62)
    )
    runs = _merge_runs(_runs(mask), 0.20)
    context = round(0.65 / DT)
    events: list[Event] = []
    for core_start, core_end in runs:
        duration = (core_end - core_start) * DT
        orientation_travel = (
            np.sum(angular_speed[core_start:core_end]) * DT
        )
        if duration < 0.08 or orientation_travel < 0.35:
            continue
        start = max(0, core_start - context)
        end = min(record.time.size, core_end + context)
        score = (
            orientation_travel
            * (1.0 + 2.0 * np.max(linear_speed[core_start:core_end]))
            / max(float(np.mean(reach[core_start:core_end])), 0.1)
        )
        events.append(
            _event(
                family="chest",
                episode=episode,
                side=side,
                index=len(events),
                record=record,
                start=start,
                end=end,
                core_start=core_start,
                core_end=core_end,
                linear_speed=linear_speed,
                angular_speed=angular_speed,
                reach=reach,
                score=score,
            )
        )
    return events


def detect_retract_events(
    episode: int,
    side: str,
    record: ArmRecord,
) -> list[Event]:
    position = record.action_world_eef
    linear_speed = np.linalg.norm(_derivative(position), axis=1)
    angular_speed = _angular_speed(record.action_pose)
    reach = np.linalg.norm(position - record.shoulder, axis=1)
    reach_rate = _derivative(reach)
    mask = (reach_rate < -0.035) & (linear_speed > 0.045)
    runs = _merge_runs(_runs(mask), 0.25)
    context = round(0.55 / DT)
    events: list[Event] = []
    for core_start, core_end in runs:
        duration = (core_end - core_start) * DT
        reach_drop = float(reach[core_start] - reach[core_end - 1])
        if duration < 0.20 or reach_drop < 0.05:
            continue
        start = max(0, core_start - context)
        end = min(record.time.size, core_end + context)
        actual_dip = _window_chord_dip(
            record.obs_world_eef[:, 2],
            core_start,
            core_end,
        )
        score = (
            5.0 * reach_drop
            + 2.0 * actual_dip
            + np.max(linear_speed[core_start:core_end])
        )
        events.append(
            _event(
                family="retract",
                episode=episode,
                side=side,
                index=len(events),
                record=record,
                start=start,
                end=end,
                core_start=core_start,
                core_end=core_end,
                linear_speed=linear_speed,
                angular_speed=angular_speed,
                reach=reach,
                score=score,
            )
        )
    return events


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare(output_dir: Path) -> None:
    cache_dir = output_dir / "cache"
    all_events: list[Event] = []
    records: dict[str, dict[str, Any]] = {}
    for family, episode in EPISODES.items():
        records[family] = {}
        for side in study.SIDES:
            print(f"Loading episode {episode} {side}...", flush=True)
            record = load_record(episode, side, cache_dir)
            records[family][side] = {
                "duration_s": float(record.time[-1]),
                "sample_count": int(record.time.size),
            }
            detected = (
                detect_chest_events(episode, side, record)
                if family == "chest"
                else detect_retract_events(episode, side, record)
            )
            all_events.extend(detected)
            print(f"  detected {len(detected)} {family} events", flush=True)

    rows = [asdict(event) for event in all_events]
    _write_csv(output_dir / "events.csv", rows)
    ranked = sorted(
        all_events,
        key=lambda event: (event.family, -event.score),
    )
    (output_dir / "events.json").write_text(
        json.dumps([asdict(event) for event in ranked], indent=2),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(RUN_DIR),
                "episodes": EPISODES,
                "dt": DT,
                "records": records,
                "target_semantics": (
                    "FK(action q): IK command-pose proxy, not raw VR target"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for family in EPISODES:
        print(f"\nTop {family} events:")
        candidates = sorted(
            (event for event in all_events if event.family == family),
            key=lambda event: event.score,
            reverse=True,
        )
        for event in candidates[:12]:
            print(
                f"  {event.side} #{event.index} "
                f"{event.start_s:.2f}-{event.end_s:.2f}s "
                f"score={event.score:.2f} "
                f"v={event.peak_linear_speed_m_s:.2f}m/s "
                f"w={event.peak_angular_speed_rad_s:.1f}rad/s "
                f"reach_drop={event.reach_drop_m:.3f}m "
                f"actual_dip={event.actual_chord_dip_m:.3f}m "
                f"elbow={event.actual_elbow_lateral_range_m:.3f}m",
                flush=True,
            )


def _event_lookup(output_dir: Path) -> dict[tuple[str, str, int], Event]:
    rows = json.loads((output_dir / "events.json").read_text(encoding="utf-8"))
    result: dict[tuple[str, str, int], Event] = {}
    for row in rows:
        event = Event(**row)
        result[(event.family, event.side, event.index)] = event
    return result


def _pad(values: np.ndarray, prefix: int, suffix: int) -> np.ndarray:
    return np.concatenate(
        [
            np.repeat(values[:1], prefix, axis=0),
            values,
            np.repeat(values[-1:], suffix, axis=0),
        ],
        axis=0,
    )


def _intent_target(
    event: Event,
    record: ArmRecord,
    target_style: str,
) -> np.ndarray:
    """Build a documented intent proxy from the recorded command pose."""
    target = record.action_pose[event.start : event.end].copy()
    local_core_start = event.core_start - event.start
    local_core_end = event.core_end - event.start
    if target_style == "position_smoothed":
        target[:, :3] = _smooth(target[:, :3], 0.60)
    elif target_style == "straight_retract":
        count = local_core_end - local_core_start
        u = np.linspace(0.0, 1.0, count)
        progress = 3.0 * np.square(u) - 2.0 * np.power(u, 3)
        start = target[local_core_start, :3]
        end = target[local_core_end - 1, :3]
        target[local_core_start:local_core_end, :3] = (
            start[None, :]
            + progress[:, None] * (end - start)[None, :]
        )
    elif target_style != "recorded_command":
        raise ValueError(f"Unknown target style: {target_style}")
    return target


def _scenario_from_event(
    event: Event,
    record: ArmRecord,
    target_style: str,
) -> tuple[study.Scenario, dict[str, np.ndarray]]:
    prefix = round(0.40 / DT)
    suffix = round(0.55 / DT)
    target_unpadded = _intent_target(event, record, target_style)
    target = _pad(target_unpadded, prefix, suffix)
    action_q = _pad(
        record.action_q[event.start : event.end],
        prefix,
        suffix,
    )
    obs_q = _pad(
        record.obs_q[event.start : event.end],
        prefix,
        suffix,
    )
    obs_dq = _pad(
        record.obs_dq[event.start : event.end],
        prefix,
        suffix,
    )
    action_pose = _pad(
        record.action_pose[event.start : event.end],
        prefix,
        suffix,
    )
    obs_pose = _pad(
        record.obs_pose[event.start : event.end],
        prefix,
        suffix,
    )
    action_elbow = _pad(
        record.action_elbow[event.start : event.end],
        prefix,
        suffix,
    )
    obs_elbow = _pad(
        record.obs_elbow[event.start : event.end],
        prefix,
        suffix,
    )
    phase = np.zeros(target.shape[0], dtype=np.int8)
    phase[prefix : prefix + target_unpadded.shape[0]] = 1
    phase[prefix + target_unpadded.shape[0] :] = 2
    times = np.arange(target.shape[0], dtype=np.float64) * DT
    name = (
        f"recorded_ep{event.episode}_{event.side}_"
        f"{event.family}{event.index:02d}_{target_style}"
    )
    initial = record.action_q[event.start].copy()
    initial_right = (
        initial.copy() if event.side == "right" else study.HOME_Q.copy()
    )
    initial_left = (
        initial.copy() if event.side == "left" else study.HOME_Q.copy()
    )
    inactive_target = np.repeat(target[:1], target.shape[0], axis=0)
    scenario = study.Scenario(
        name=name,
        family=f"recorded_{event.family}",
        mode=event.side,
        speed=event.peak_linear_speed_m_s,
        times=times,
        phase=phase,
        target_right=(
            target.copy() if event.side == "right" else inactive_target
        ),
        target_left=(
            target.copy() if event.side == "left" else inactive_target
        ),
        initial_right=initial_right,
        initial_left=initial_left,
        description=(
            f"Episode {event.episode} {event.side} event {event.index}; "
            f"{target_style} intent proxy derived from FK(action q)."
        ),
    )
    source = {
        "action_q": action_q,
        "obs_q": obs_q,
        "obs_dq": obs_dq,
        "action_pose": action_pose,
        "obs_pose": obs_pose,
        "action_elbow": action_elbow,
        "obs_elbow": obs_elbow,
        "target_pose": target,
    }
    return scenario, source


def _save_recorded_trace(
    path: Path,
    scenario: study.Scenario,
    side: str,
    source: dict[str, np.ndarray],
) -> None:
    action_q = source["action_q"]
    obs_q = source["obs_q"]
    action_dq = _derivative(action_q)
    action_ddq = _derivative(action_dq)
    obs_dq = source["obs_dq"]
    obs_ddq = _derivative(obs_dq)
    payload: dict[str, np.ndarray] = {
        "times": scenario.times,
        "phase": scenario.phase,
        "solve_time": np.zeros_like(scenario.times),
        "solver_failed": np.zeros_like(scenario.times, dtype=bool),
        f"{side}_target_pose": source["target_pose"],
        f"{side}_target": source["target_pose"],
        f"{side}_command_q": action_q,
        f"{side}_driver_q": action_q,
        f"{side}_actual_q": obs_q,
        f"{side}_command_dq": action_dq,
        f"{side}_driver_dq": action_dq,
        f"{side}_actual_dq": obs_dq,
        f"{side}_command_ddq": action_ddq,
        f"{side}_driver_ddq": action_ddq,
        f"{side}_actual_ddq": obs_ddq,
        f"{side}_command_pose": source["action_pose"],
        f"{side}_driver_pose": source["action_pose"],
        f"{side}_actual_pose": source["obs_pose"],
        f"{side}_command_elbow": source["action_elbow"],
        f"{side}_driver_elbow": source["action_elbow"],
        f"{side}_actual_elbow": source["obs_elbow"],
    }
    np.savez_compressed(path, **payload)


def _replay_profiles() -> list[study.Profile]:
    return [
        study.make_profile(
            "pr_full_recorded",
            description="PR default.",
        ),
        study.make_profile(
            "pr_dataflow_recorded",
            nullspace_cost=8.0,
            description=(
                "Current PR with the nullspace cost used by the recorded "
                "evaluation dataflow."
            ),
        ),
        study.upstream_style_ik_velocity_profile(
            "mainline_ik_velocity_recorded"
        ),
        study.upstream_style_profile("mainline_recorded"),
        study.make_profile(
            "pr_driver_velocity_only_recorded",
            limit_style="configuration_only",
            velocity_caps=None,
            description=(
                "PR w/o IK velocity limits; downstream "
                "driver velocity cap remains enabled."
            ),
        ),
        study.make_profile(
            "pr_nullspace_cost_10_recorded",
            nullspace_cost=10.0,
            description="Current PR with nullspace cost increased from 7 to 10.",
        ),
        study.make_profile(
            "pr_nullspace_cost_12_recorded",
            nullspace_cost=12.0,
            description="Current PR with nullspace cost increased from 7 to 12.",
        ),
        study.make_profile(
            "pr_no_frame_error_recorded",
            frame_position_error_limit=0.0,
            frame_orientation_error_limit=0.0,
            description="Current PR with both frame-error bounds disabled.",
        ),
        study.make_profile(
            "pr_no_nullspace_recorded",
            nullspace_cost=0.0,
            description="Current PR with exact-nullspace regulation disabled.",
        ),
    ]


def replay(output_dir: Path) -> None:
    lookup = _event_lookup(output_dir)
    cache_dir = output_dir / "cache"
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    (REPLAY_DIR / "traces").mkdir(parents=True, exist_ok=True)
    profiles = _replay_profiles()
    metric_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []

    for family, side, index, target_style in SELECTED_EVENTS:
        event = lookup[(family, side, index)]
        record = load_record(event.episode, side, cache_dir)
        scenario, source = _scenario_from_event(
            event,
            record,
            target_style,
        )
        scenario_rows.append(
            {
                **asdict(event),
                "scenario": scenario.name,
                "target_style": target_style,
                "target_semantics": (
                    "intent proxy derived from FK(recorded action q)"
                ),
            }
        )
        hardware_path = (
            REPLAY_DIR
            / "traces"
            / f"recorded_recorded_hardware_{scenario.name}.npz"
        )
        _save_recorded_trace(hardware_path, scenario, side, source)
        print(f"\nScenario {scenario.name}", flush=True)
        for profile in profiles:
            print(f"  simulating {profile.name}...", flush=True)
            rows = study.run_cached(
                profile,
                scenario,
                REPLAY_DIR,
                save_full_trace=True,
            )
            metric_rows.extend(rows)

    _write_csv(REPLAY_DIR / "metrics.csv", metric_rows)
    (REPLAY_DIR / "selected_scenarios.json").write_text(
        json.dumps(scenario_rows, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS_DIR,
    )
    parser.add_argument(
        "command",
        choices=("prepare", "replay"),
        default="prepare",
        nargs="?",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare(output_dir)
    elif args.command == "replay":
        replay(output_dir)


if __name__ == "__main__":
    main()
