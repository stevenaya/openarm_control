#!/usr/bin/env python3
"""Sweep arm velocity limits on one recorded intervention replay."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sim_intervention_posture_replay import (
    DEFAULT_RUN_DIR,
    _load_recorded_commands,
    _save_trace,
    _source_geometry,
    compute_metrics,
    detect_retract_segments,
    simulate_replay,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--episode", type=int, default=202)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--segment", type=int, default=13)
    parser.add_argument(
        "--velocity-limit-scales",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 4.0],
    )
    parser.add_argument(
        "--direct-error-gains",
        type=float,
        nargs="+",
        default=[0.0, 0.003],
        help=(
            "Zero uses the current rate-limited nullspace task. Positive "
            "values replace it with the experimental direct exact-z task."
        ),
    )
    parser.add_argument("--direct-nullspace-cost", type=float, default=10.0)
    parser.add_argument("--settle-duration", type=float, default=0.5)
    parser.add_argument(
        "--joint-ablation",
        action="store_true",
        help="Also relax selected joints by 4x while retaining all other caps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "dev/results/intervention_velocity_limit_sweep_ep202_seg13"
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

    rows: list[dict[str, float | int | str]] = []
    for error_gain in args.direct_error_gains:
        variant = "rate_limited_z" if error_gain == 0.0 else "direct_exact_z"
        direct_cost = 0.0 if error_gain == 0.0 else args.direct_nullspace_cost
        for limit_scale in args.velocity_limit_scales:
            print(
                f"Simulating variant={variant}, "
                f"beta={error_gain:g}, velocity_scale={limit_scale:g}..."
            )
            trace = simulate_replay(
                args.side,
                0.0,
                target_pose[window],
                source_q[window],
                source_elbow[window],
                settle_duration=args.settle_duration,
                direct_nullspace_cost=direct_cost,
                direct_nullspace_error_gain=error_gain,
                velocity_limit_scale=limit_scale,
            )
            row: dict[str, float | int | str] = {
                "variant": variant,
                "velocity_profile": f"all_x{limit_scale:g}",
                "direct_error_gain": error_gain,
                "velocity_limit_scale": limit_scale,
                **asdict(compute_metrics(args.segment, 0.0, trace)),
            }
            for joint in range(7):
                utilization = trace.velocity_utilization[:, joint]
                row[f"j{joint + 1}_limit_fraction"] = float(
                    np.mean(utilization >= 0.98)
                )
                row[f"j{joint + 1}_peak_utilization"] = float(
                    np.max(utilization)
                )
                row[f"j{joint + 1}_peak_abs_dq_rad_s"] = float(
                    np.max(np.abs(trace.command_dq[:, joint]))
                )
            rows.append(row)

            scale_label = str(limit_scale).replace(".", "p")
            gain_label = str(error_gain).replace(".", "p")
            _save_trace(
                args.output_dir
                / (
                    f"trace_{variant}_beta_{gain_label}_"
                    f"velocity_scale_{scale_label}.npz"
                ),
                trace,
            )

    if args.joint_ablation:
        profiles = {
            "j1_x4": [4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "j2_x4": [1.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "j1_j2_x4": [4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "j4_x4": [1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0],
            "j1_j4_x4": [4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0],
        }
        for profile, multipliers in profiles.items():
            print(f"Simulating joint-ablation profile={profile}...")
            trace = simulate_replay(
                args.side,
                0.0,
                target_pose[window],
                source_q[window],
                source_elbow[window],
                settle_duration=args.settle_duration,
                velocity_limit_joint_multipliers=np.asarray(multipliers),
            )
            row = {
                "variant": "rate_limited_z",
                "velocity_profile": profile,
                "direct_error_gain": 0.0,
                "velocity_limit_scale": 1.0,
                **asdict(compute_metrics(args.segment, 0.0, trace)),
            }
            for joint in range(7):
                utilization = trace.velocity_utilization[:, joint]
                row[f"j{joint + 1}_limit_fraction"] = float(
                    np.mean(utilization >= 0.98)
                )
                row[f"j{joint + 1}_peak_utilization"] = float(
                    np.max(utilization)
                )
                row[f"j{joint + 1}_peak_abs_dq_rad_s"] = float(
                    np.max(np.abs(trace.command_dq[:, joint]))
                )
            rows.append(row)
            _save_trace(
                args.output_dir / f"trace_ablation_{profile}.npz",
                trace,
            )

    with (args.output_dir / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
