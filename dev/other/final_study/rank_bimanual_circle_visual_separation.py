#!/usr/bin/env python3
"""Rank bimanual-circle replays by persistent branch separation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

PR_PROFILE = "pr_current_dataflow"
MAINLINE_PROFILE = "mainline_with_ik_velocity"
DEFAULT_RESULTS = (
    Path(__file__).resolve().parent
    / "results"
    / "bimanual_circle_latest_ep89_20260731"
)


def _trace_path(results: Path, profile: str, scenario: str) -> Path:
    matches = list(
        results.glob(f"**/traces/*_{profile}_{scenario}.npz")
    )
    if not matches:
        raise RuntimeError(f"Missing trace for {profile}/{scenario}.")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def _side_metrics(
    pr: np.lib.npyio.NpzFile,
    mainline: np.lib.npyio.NpzFile,
    side: str,
    active: np.ndarray,
) -> dict[str, float]:
    times = pr["times"][active]
    elbow_delta = (
        mainline[f"{side}_actual_elbow"][active]
        - pr[f"{side}_actual_elbow"][active]
    )
    elbow_separation = np.linalg.norm(elbow_delta, axis=1)
    eef_separation = np.linalg.norm(
        mainline[f"{side}_actual_pose"][active, :3]
        - pr[f"{side}_actual_pose"][active, :3],
        axis=1,
    )
    joint_separation = np.linalg.norm(
        mainline[f"{side}_actual_q"][active]
        - pr[f"{side}_actual_q"][active],
        axis=1,
    )
    same_eef = eef_separation <= 0.03
    same_eef_elbow = (
        elbow_separation[same_eef]
        if np.any(same_eef)
        else np.asarray([np.nan])
    )
    elbow_p95 = _quantile(elbow_separation, 0.95)
    elbow_max_index = int(np.argmax(elbow_separation))
    elbow_p95_index = int(np.argmin(np.abs(elbow_separation - elbow_p95)))
    return {
        f"{side}_elbow_separation_mean_m": float(np.mean(elbow_separation)),
        f"{side}_elbow_separation_p95_m": elbow_p95,
        f"{side}_elbow_separation_p95_time_s": float(
            times[elbow_p95_index]
        ),
        f"{side}_elbow_separation_max_m": float(
            np.max(elbow_separation)
        ),
        f"{side}_elbow_separation_max_time_s": float(
            times[elbow_max_index]
        ),
        f"{side}_elbow_separation_over_5cm_fraction": float(
            np.mean(elbow_separation >= 0.05)
        ),
        f"{side}_same_eef_fraction": float(np.mean(same_eef)),
        f"{side}_same_eef_elbow_separation_p95_m": _quantile(
            same_eef_elbow, 0.95
        ),
        f"{side}_eef_separation_p95_m": _quantile(eef_separation, 0.95),
        f"{side}_joint_separation_p95_rad": _quantile(
            joint_separation, 0.95
        ),
    }


def rank(results: Path) -> list[dict[str, float | str]]:
    ranking_path = results / "similar" / "similar_ranking.csv"
    with ranking_path.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))

    rows: list[dict[str, float | str]] = []
    for source in source_rows:
        scenario = source["scenario"]
        with (
            np.load(
                _trace_path(results, PR_PROFILE, scenario),
                allow_pickle=False,
            ) as pr,
            np.load(
                _trace_path(results, MAINLINE_PROFILE, scenario),
                allow_pickle=False,
            ) as mainline,
        ):
            active = pr["phase"] == 1
            row: dict[str, float | str] = {
                "label": source["label"],
                "scenario": scenario,
                "source_start_s": float(source["source_start_s"]),
                "source_end_s": float(source["source_end_s"]),
                "pr_position_error_max_m": float(
                    source["pr_position_error_max_m"]
                ),
                "mainline_position_error_max_m": float(
                    source["mainline_position_error_max_m"]
                ),
            }
            for side in ("right", "left"):
                row.update(_side_metrics(pr, mainline, side, active))

        elbow_p95 = max(
            float(row[f"{side}_elbow_separation_p95_m"])
            for side in ("right", "left")
        )
        elbow_mean = max(
            float(row[f"{side}_elbow_separation_mean_m"])
            for side in ("right", "left")
        )
        persistent_fraction = max(
            float(row[f"{side}_elbow_separation_over_5cm_fraction"])
            for side in ("right", "left")
        )
        eef_p95 = max(
            float(row[f"{side}_eef_separation_p95_m"])
            for side in ("right", "left")
        )
        row["persistent_branch_score"] = (
            100.0 * elbow_p95
            + 40.0 * elbow_mean
            + 3.0 * persistent_fraction
            - 30.0 * eef_p95
        )
        rows.append(row)

    rows.sort(
        key=lambda row: float(row["persistent_branch_score"]),
        reverse=True,
    )
    output = results / "persistent_branch_ranking.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    rows = rank(args.results_dir)
    for index, row in enumerate(rows):
        print(
            f"{index:02d} {row['label']} "
            f"elbow p95 R/L="
            f"{100 * float(row['right_elbow_separation_p95_m']):.1f}/"
            f"{100 * float(row['left_elbow_separation_p95_m']):.1f}cm "
            f">5cm R/L="
            f"{100 * float(row['right_elbow_separation_over_5cm_fraction']):.0f}/"
            f"{100 * float(row['left_elbow_separation_over_5cm_fraction']):.0f}% "
            f"eef p95 R/L="
            f"{100 * float(row['right_eef_separation_p95_m']):.1f}/"
            f"{100 * float(row['left_eef_separation_p95_m']):.1f}cm",
            flush=True,
        )


if __name__ == "__main__":
    main()
