#!/usr/bin/env python3
"""Replay one recorded 6D intervention path at several temporal scales."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from sim_intervention_posture_replay import (
    CONTROL_DT,
    DEFAULT_RUN_DIR,
    _load_recorded_commands,
    _save_trace,
    _source_geometry,
    compute_metrics,
    detect_retract_segments,
    simulate_replay,
)


def _resample_columns(
    values: np.ndarray,
    source_time: np.ndarray,
    query_time: np.ndarray,
) -> np.ndarray:
    output = np.empty((query_time.size, values.shape[1]), dtype=np.float64)
    for column in range(values.shape[1]):
        output[:, column] = np.interp(
            query_time,
            source_time,
            values[:, column],
        )
    return output


def time_scale_window(
    target_pose: np.ndarray,
    source_q: np.ndarray,
    source_elbow: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the same geometric path with duration multiplied by ``scale``."""
    if scale <= 0.0:
        raise ValueError("Time scale must be positive.")
    source_time = np.arange(target_pose.shape[0], dtype=np.float64) * CONTROL_DT
    duration = source_time[-1] * scale
    replay_time = np.arange(0.0, duration + 0.5 * CONTROL_DT, CONTROL_DT)
    query_time = np.minimum(replay_time / scale, source_time[-1])

    replay_pose = np.empty((replay_time.size, 7), dtype=np.float64)
    replay_pose[:, :3] = _resample_columns(
        target_pose[:, :3],
        source_time,
        query_time,
    )
    rotation = Rotation.from_quat(target_pose[:, [4, 5, 6, 3]])
    replay_rotation = Slerp(source_time, rotation)(query_time)
    replay_pose[:, 3:] = replay_rotation.as_quat()[:, [3, 0, 1, 2]]
    return (
        replay_pose,
        _resample_columns(source_q, source_time, query_time),
        _resample_columns(source_elbow, source_time, query_time),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--episode", type=int, default=202)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--segment", type=int, default=13)
    parser.add_argument(
        "--time-scales",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0],
        help="Duration multipliers; larger values replay more slowly.",
    )
    parser.add_argument(
        "--posture-costs",
        type=float,
        nargs="+",
        default=[0.0, 0.3],
    )
    parser.add_argument("--settle-duration", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "dev/results/intervention_speed_scaled_replay_ep202_seg13"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp, raw_q = _load_recorded_commands(
        args.run_dir,
        args.episode,
        args.side,
    )
    time, target_pose, source_q, shoulder, source_elbow = _source_geometry(
        args.side,
        timestamp,
        raw_q,
    )
    segments = detect_retract_segments(
        time,
        target_pose,
        shoulder,
        source_elbow,
    )
    segment = segments[args.segment]
    window = slice(segment.start, segment.end)

    rows: list[dict[str, float | int]] = []
    for scale in args.time_scales:
        replay_pose, replay_q, replay_elbow = time_scale_window(
            target_pose[window],
            source_q[window],
            source_elbow[window],
            scale,
        )
        for posture_cost in args.posture_costs:
            print(
                f"Simulating time_scale={scale:g}, "
                f"posture_cost={posture_cost:g}..."
            )
            trace = simulate_replay(
                args.side,
                posture_cost,
                replay_pose,
                replay_q,
                replay_elbow,
                settle_duration=args.settle_duration,
            )
            row = asdict(
                compute_metrics(args.segment, posture_cost, trace)
            )
            row["time_scale"] = scale
            rows.append(row)
            scale_label = str(scale).replace(".", "p")
            cost_label = str(posture_cost).replace(".", "p")
            _save_trace(
                args.output_dir
                / f"trace_scale_{scale_label}_posture_{cost_label}.npz",
                trace,
            )

    fieldnames = ["time_scale", *[key for key in rows[0] if key != "time_scale"]]
    with (args.output_dir / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
