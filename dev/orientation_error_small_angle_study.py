"""Check independent orientation limiting on ordinary wrist motions."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chest_wrist_flip_study import simulate, summarize
from orientation_error_modulation_study import (
    NORMAL_START_Q_RIGHT,
    Scenario,
    Strategy,
    make_profile,
)

OUTPUT = Path("dev/results/orientation_error_small_angle_20260729")


def scenarios() -> list[Scenario]:
    motions = (
        (0.25, 0.5),
        (0.50, 1.0),
        (1.00, 2.0),
        (1.57, 4.0),
    )
    return [
        Scenario(
            name=(
                f"normal_{axis_name}_a{str(angle).replace('.', 'p')}"
                f"_v{str(speed).replace('.', 'p')}"
            ),
            group="small_angle",
            initial_right=NORMAL_START_Q_RIGHT,
            angular_speed=speed,
            angle=angle,
            local_axis=axis,
        )
        for axis_name, axis in (
            ("x", (1.0, 0.0, 0.0)),
            ("y", (0.0, 1.0, 0.0)),
            ("z", (0.0, 0.0, 1.0)),
        )
        for angle, speed in motions
    ]


def strategies() -> list[Strategy]:
    return [
        Strategy("current", 0.0, False),
        *[
            Strategy(
                f"independent_{str(limit).replace('.', 'p')}",
                limit,
                False,
                independent_orientation_limit=True,
            )
            for limit in (0.02, 0.03, 0.04)
        ],
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenario_values = scenarios()
    strategy_values = strategies()
    rows: list[dict[str, Any]] = []
    for scenario in scenario_values:
        for strategy in strategy_values:
            print(f"simulate {scenario.name} {strategy.name}", flush=True)
            profile = make_profile(scenario, strategy)
            row = summarize(profile, simulate(profile))
            row["scenario"] = scenario.name
            row["strategy"] = strategy.name
            rows.append(row)
    write_csv(OUTPUT / "runs.csv", rows)
    metadata = {
        "scenarios": [asdict(value) for value in scenario_values],
        "strategies": [asdict(value) for value in strategy_values],
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
