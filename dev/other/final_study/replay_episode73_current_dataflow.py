#!/usr/bin/env python3
"""Replay the episode-73 triphasic cases with the active dataflow profile."""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import episode73_retract_pause_extend_study as episode73
import pr_showcase_selection_study as selection
import study


OUTPUT_DIR = HERE / "results" / "episode73_triphasic_20260731"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    record = episode73._record()
    rows: list[dict[str, Any]] = []
    for pattern in episode73.detect_patterns():
        scenario = episode73._scenario(record, pattern)
        print(f"\n{scenario.name}", flush=True)
        for profile in selection._profiles():
            print(f"  {profile.name}", flush=True)
            metrics = study.run_cached(
                profile,
                scenario,
                OUTPUT_DIR,
                save_full_trace=True,
            )
            rows.extend(
                {
                    **metric,
                    **asdict(pattern),
                    "source_episode": 73,
                    "dataflow_args": selection.DATAFLOW_ARGS,
                }
                for metric in metrics
            )
    _write_csv(OUTPUT_DIR / "current_dataflow_metrics.csv", rows)


if __name__ == "__main__":
    main()
