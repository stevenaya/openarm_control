#!/usr/bin/env python3
"""Analyze nullspace tuning sweeps with a geometric elbow-swivel metric."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd

import study


LOCAL_PROFILE = re.compile(
    r"ns_local_c(?P<cost>[0-9p]+)_r(?P<rate>[0-9p]+)"
)


class SwivelMonitor:
    """Recover shoulder, elbow, and wrist geometry from saved joint traces."""

    def __init__(self) -> None:
        self.setup = study.make_setup("bimanual")
        self.body_ids = {
            side: np.array(
                [
                    mujoco.mj_name2id(
                        self.setup.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        f"openarm_{side}_{link}",
                    )
                    for link in ("link2", "link4", "link6")
                ],
                dtype=int,
            )
            for side in study.SIDES
        }
        self.qpos = self.setup.data.qpos.copy()

    def max_departure(self, arm_q: np.ndarray, side: str) -> float:
        """Return maximum elbow swivel departure from the initial arm plane."""
        positions = np.empty((len(arm_q), 3, 3), dtype=np.float64)
        for index, q in enumerate(arm_q):
            self.setup.joint_resolver.set_qpos(
                self.qpos,
                np.append(q, 0.0),
                side,
            )
            self.setup.data.qpos[:] = self.qpos
            mujoco.mj_forward(self.setup.model, self.setup.data)
            positions[index] = self.setup.data.xpos[self.body_ids[side]]

        shoulder = positions[:, 0]
        elbow = positions[:, 1]
        wrist = positions[:, 2]
        axis = wrist - shoulder
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)

        world_up = np.array([0.0, 0.0, 1.0])
        reference = world_up - np.sum(world_up * axis, axis=1, keepdims=True) * axis
        reference /= np.linalg.norm(reference, axis=1, keepdims=True)

        elbow_radius = elbow - shoulder
        elbow_radius -= (
            np.sum(elbow_radius * axis, axis=1, keepdims=True) * axis
        )
        elbow_radius /= np.linalg.norm(elbow_radius, axis=1, keepdims=True)

        angle = np.unwrap(
            np.arctan2(
                np.einsum(
                    "ij,ij->i",
                    axis,
                    np.cross(reference, elbow_radius),
                ),
                np.einsum("ij,ij->i", reference, elbow_radius),
            )
        )
        return float(np.max(np.abs(angle - angle[0])))


def _decode_number(value: str) -> float:
    return float(value.replace("p", "."))


def add_swivel_metrics(input_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(input_dir / "summary.csv")
    monitor = SwivelMonitor()
    actual_swivel: list[float] = []
    command_swivel: list[float] = []
    for row in summary.itertuples(index=False):
        matches = list(
            (input_dir / "traces").glob(
                f"*_{row.profile}_{row.scenario}.npz"
            )
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one trace for {row.profile} / {row.scenario}, "
                f"found {len(matches)}."
            )
        with np.load(matches[0]) as trace:
            actual_swivel.append(
                monitor.max_departure(
                    trace[f"{row.side}_actual_q"],
                    row.side,
                )
            )
            command_swivel.append(
                monitor.max_departure(
                    trace[f"{row.side}_command_q"],
                    row.side,
                )
            )
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                {
                    "actual_swivel_max_rad": actual_swivel,
                    "command_swivel_max_rad": command_swivel,
                }
            ),
        ],
        axis=1,
    )
    summary.to_csv(input_dir / "summary_with_swivel.csv", index=False)
    return summary


def local_grid(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for row in summary.itertuples(index=False):
        match = LOCAL_PROFILE.fullmatch(row.profile)
        if match is None:
            continue
        values = row._asdict()
        values["cost"] = _decode_number(match.group("cost"))
        values["return_rate"] = _decode_number(match.group("rate"))
        rows.append(values)
    return pd.DataFrame(rows)


def plot_heatmaps(grid: pd.DataFrame, output_path: Path) -> None:
    metrics = (
        ("actual_swivel_max_rad", "Elbow swivel departure [rad]"),
        ("actual_position_rmse_m", "EEF position RMSE [m]"),
        ("actual_orientation_rmse_rad", "EEF orientation RMSE [rad]"),
        ("actual_elbow_accel_p99_m_s2", "Elbow acceleration p99 [m/s²]"),
        ("actual_ddq_p99_rad_s2", "Joint acceleration p99 [rad/s²]"),
        ("command_velocity_saturation_fraction", "IK velocity saturation"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
        table = grid.pivot(
            index="return_rate",
            columns="cost",
            values=metric,
        ).sort_index(ascending=False)
        image = axis.imshow(table, aspect="auto", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("Nullspace cost")
        axis.set_ylabel("Return rate [s⁻¹]")
        axis.set_xticks(range(len(table.columns)), table.columns)
        axis.set_yticks(range(len(table.index)), table.index)
        for row_index in range(len(table.index)):
            for column_index in range(len(table.columns)):
                value = table.iloc[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3g}",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if value > np.nanmean(table.to_numpy())
                        else "black"
                    ),
                    fontsize=8,
                )
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.suptitle(
        "Deep retract: nullspace cost × return-rate tuning",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "dev/final_study/results/current_pr_20260730/nullspace_tuning"
        ),
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    summary = add_swivel_metrics(input_dir)
    grid = local_grid(summary)
    grid.to_csv(input_dir / "local_grid.csv", index=False)
    if not grid.empty:
        plot_heatmaps(grid, input_dir / "local_grid_heatmaps.png")


if __name__ == "__main__":
    main()
