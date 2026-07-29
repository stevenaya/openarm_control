"""Tune balanced position/orientation error modulation for wrist flips.

This is an experiment-only driver around ``chest_wrist_flip_study``. It keeps
the production position limiter enabled, sweeps an analogous orientation
limit, and checks the result in the recorded chest pose and in a normal
workspace pose.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from chest_wrist_flip_study import Profile, simulate, summarize

OUTPUT = Path("dev/results/orientation_error_modulation_20260729")
NORMAL_START_Q_RIGHT = (
    -0.24633651,
    0.12070639,
    0.30554437,
    2.1537668,
    0.44992149,
    -0.02483132,
    0.68082729,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    group: str
    initial_right: tuple[float, ...]
    angular_speed: float
    angle: float
    local_axis: tuple[float, float, float]
    pre_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pre_translation_speed: float = 1.0
    forced_activation: float | None = None


@dataclass(frozen=True)
class Strategy:
    name: str
    orientation_error_limit: float
    angular_schedule: bool
    angular_speed_slow: float = 2.0
    angular_speed_fast: float = 3.0
    orientation_latch: bool = True
    independent_orientation_limit: bool = False


def scenarios() -> list[Scenario]:
    chest = tuple(float(value) for value in Profile("base").initial_right)
    return [
        Scenario(
            "chest_forced_z_4p5_v8",
            "chest_forced",
            chest,
            8.0,
            4.5,
            (0.0, 0.0, -1.0),
            forced_activation=1.0,
        ),
        Scenario(
            "chest_natural_pre_x_z_4p5_v8",
            "chest_natural",
            chest,
            8.0,
            4.5,
            (0.0, 0.0, -1.0),
            pre_translation=(0.05, 0.0, 0.0),
        ),
        Scenario(
            "chest_pure_z_4p5_v8",
            "chest_pure",
            chest,
            8.0,
            4.5,
            (0.0, 0.0, -1.0),
        ),
        Scenario(
            "chest_pure_y_pi_v4",
            "chest_pure",
            chest,
            4.0,
            math.pi,
            (0.0, 1.0, 0.0),
        ),
        *[
            Scenario(
                f"normal_pure_{axis_name}_pi_v{speed:.0f}",
                "normal_rotation",
                NORMAL_START_Q_RIGHT,
                speed,
                math.pi,
                axis,
            )
            for axis_name, axis in (
                ("x", (1.0, 0.0, 0.0)),
                ("y", (0.0, 1.0, 0.0)),
                ("z", (0.0, 0.0, 1.0)),
            )
            for speed in (4.0, 8.0)
        ],
        *[
            Scenario(
                f"normal_speed_sweep_{axis_name}_pi_v{str(speed).replace('.', 'p')}",
                "speed_transition",
                NORMAL_START_Q_RIGHT,
                speed,
                math.pi,
                axis,
            )
            for axis_name, axis in (
                ("y", (0.0, 1.0, 0.0)),
                ("z", (0.0, 0.0, 1.0)),
            )
            for speed in (1.0, 2.0, 2.5, 3.0, 6.0)
        ],
        Scenario(
            "normal_pre_x_z_pi_v8",
            "normal_translated_rotation",
            NORMAL_START_Q_RIGHT,
            8.0,
            math.pi,
            (0.0, 0.0, 1.0),
            pre_translation=(0.05, 0.0, 0.0),
        ),
        Scenario(
            "normal_pre_z_x_pi_v8",
            "normal_translated_rotation",
            NORMAL_START_Q_RIGHT,
            8.0,
            math.pi,
            (1.0, 0.0, 0.0),
            pre_translation=(0.0, 0.0, 0.05),
        ),
        Scenario(
            "normal_translate_x_0p12_v1",
            "translation_only",
            NORMAL_START_Q_RIGHT,
            8.0,
            0.0,
            (0.0, 0.0, 1.0),
            pre_translation=(0.12, 0.0, 0.0),
        ),
        Scenario(
            "normal_translate_z_0p12_v1",
            "translation_only",
            NORMAL_START_Q_RIGHT,
            8.0,
            0.0,
            (0.0, 0.0, 1.0),
            pre_translation=(0.0, 0.0, 0.12),
        ),
    ]


def strategies() -> list[Strategy]:
    return [
        Strategy("current", 0.0, False),
        Strategy("shared_only_0p02", 0.02, False),
        *[
            Strategy(
                f"independent_{str(limit).replace('.', 'p')}",
                limit,
                False,
                independent_orientation_limit=True,
            )
            for limit in (0.01, 0.02, 0.03, 0.04, 0.06, 0.08)
        ],
        *[
            Strategy(
                f"angular_{str(limit).replace('.', 'p')}",
                limit,
                True,
            )
            for limit in (0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10)
        ],
        Strategy("angular_0p02_slow_4_6", 0.02, True, 4.0, 6.0),
        Strategy("angular_0p04_slow_4_6", 0.04, True, 4.0, 6.0),
        Strategy("angular_0p03_no_latch", 0.03, True, orientation_latch=False),
    ]


def make_profile(scenario: Scenario, strategy: Strategy) -> Profile:
    return Profile(
        name=f"{scenario.name}__{strategy.name}",
        initial_right=scenario.initial_right,
        angular_speed=scenario.angular_speed,
        angle=scenario.angle,
        local_axis=scenario.local_axis,
        pre_translation=scenario.pre_translation,
        pre_translation_speed=scenario.pre_translation_speed,
        forced_activation=scenario.forced_activation,
        orientation_error_limit=strategy.orientation_error_limit,
        independent_orientation_limit=strategy.independent_orientation_limit,
        angular_schedule=strategy.angular_schedule,
        orientation_latch=strategy.orientation_latch,
        angular_speed_slow=strategy.angular_speed_slow,
        angular_speed_fast=strategy.angular_speed_fast,
        dynamic_plant=True,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    rows: list[dict[str, Any]],
    strategy_values: list[Strategy],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for strategy in strategy_values:
        selected = [row for row in rows if row["strategy"] == strategy.name]
        rotation = [row for row in selected if row["group"] != "translation_only"]
        translation = [row for row in selected if row["group"] == "translation_only"]
        output.append(
            {
                "strategy": strategy.name,
                "orientation_error_limit_rad": strategy.orientation_error_limit,
                "independent_orientation_limit": (
                    strategy.independent_orientation_limit
                ),
                "angular_schedule": strategy.angular_schedule,
                "orientation_latch": strategy.orientation_latch,
                "angular_speed_slow_rad_s": strategy.angular_speed_slow,
                "angular_speed_fast_rad_s": strategy.angular_speed_fast,
                "rotation_worst_actual_position_error_m": max(
                    row["max_actual_position_error_m"] for row in rotation
                ),
                "rotation_p95_actual_position_error_m": float(
                    np.percentile(
                        [row["max_actual_position_error_m"] for row in rotation],
                        95,
                    )
                ),
                "rotation_worst_proximal_excursion_rad": max(
                    row["max_proximal_joint_excursion_rad"] for row in rotation
                ),
                "rotation_worst_actual_joint_accel_p99_rad_s2": max(
                    row["actual_joint_acceleration_p99_rad_s2"] for row in rotation
                ),
                "rotation_worst_orientation_error_rad": max(
                    row["max_orientation_error_rad"] for row in rotation
                ),
                "rotation_worst_final_orientation_error_rad": max(
                    row["final_orientation_error_rad"] for row in rotation
                ),
                "translation_worst_actual_position_error_m": max(
                    row["max_actual_position_error_m"] for row in translation
                ),
                "translation_worst_actual_joint_accel_p99_rad_s2": max(
                    row["actual_joint_acceleration_p99_rad_s2"] for row in translation
                ),
                "solve_failures": sum(row["solve_failures"] for row in selected),
            }
        )
    return output


def plot_aggregate(rows: list[dict[str, Any]]) -> None:
    labels = [row["strategy"] for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    axes[0].bar(
        x,
        [row["rotation_worst_actual_position_error_m"] for row in rows],
    )
    axes[0].set_ylabel("worst rotation position error [m]")
    axes[1].bar(
        x,
        [row["rotation_worst_actual_joint_accel_p99_rad_s2"] for row in rows],
    )
    axes[1].set_ylabel("worst actual joint accel p99 [rad/s^2]")
    axes[2].bar(
        x,
        [row["rotation_worst_orientation_error_rad"] for row in rows],
    )
    axes[2].set_ylabel("worst orientation lag [rad]")
    axes[2].set_xticks(x, labels, rotation=45, ha="right")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT / "aggregate.png", dpi=160)
    plt.close(figure)


def plot_representative(
    traces: dict[tuple[str, str], dict[str, np.ndarray]],
    scenario_name: str,
) -> None:
    selected = (
        "current",
        "shared_only_0p02",
        "angular_0p01",
        "angular_0p02",
        "angular_0p04",
        "angular_0p08",
        "angular_0p03_no_latch",
        "independent_0p02",
        "independent_0p03",
    )
    figure, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
    for strategy in selected:
        trace = traces[(scenario_name, strategy)]
        target_position = trace["target_pose"][:, :3]
        position_error = np.linalg.norm(
            trace["actual_pose"][:, :3] - target_position,
            axis=1,
        )
        proximal_excursion = np.max(
            np.abs(trace["actual_q"][:, :4] - trace["actual_q"][0, :4]),
            axis=1,
        )
        axes[0].plot(trace["time"], position_error, label=strategy)
        axes[1].plot(trace["time"], trace["orientation_error"], label=strategy)
        axes[2].plot(trace["time"], proximal_excursion, label=strategy)
        axes[3].plot(trace["time"], trace["activation"], label=strategy)
    axes[0].set_ylabel("actual position error [m]")
    axes[1].set_ylabel("orientation lag [rad]")
    axes[2].set_ylabel("max proximal excursion [rad]")
    axes[3].set_ylabel("limit activation")
    axes[3].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
    figure.suptitle(scenario_name)
    figure.tight_layout()
    figure.savefig(OUTPUT / f"{scenario_name}.png", dpi=160)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenario_values = scenarios()
    strategy_values = strategies()
    rows: list[dict[str, Any]] = []
    traces: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    representative = {
        "chest_natural_pre_x_z_4p5_v8",
        "chest_pure_z_4p5_v8",
        "normal_pure_z_pi_v8",
        "normal_translate_x_0p12_v1",
    }

    for scenario in scenario_values:
        for strategy in strategy_values:
            print(f"simulate {scenario.name} {strategy.name}", flush=True)
            profile = make_profile(scenario, strategy)
            trace = simulate(profile)
            row = summarize(profile, trace)
            row["scenario"] = scenario.name
            row["group"] = scenario.group
            row["strategy"] = strategy.name
            rows.append(row)
            if scenario.name in representative:
                traces[(scenario.name, strategy.name)] = trace

    aggregate_rows = aggregate(rows, strategy_values)
    write_csv(OUTPUT / "runs.csv", rows)
    write_csv(OUTPUT / "aggregate.csv", aggregate_rows)
    plot_aggregate(aggregate_rows)
    for scenario_name in representative:
        plot_representative(traces, scenario_name)
    metadata = {
        "scenarios": [asdict(value) for value in scenario_values],
        "strategies": [asdict(value) for value in strategy_values],
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
