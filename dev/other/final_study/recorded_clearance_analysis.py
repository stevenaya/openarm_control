#!/usr/bin/env python3
"""Measure recorded gripper-to-table clearance with MuJoCo collision geoms."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recorded_intervention_study as recorded
import study

OUTPUT_DIR = recorded.ANALYSIS_DIR
TABLE_GEOM = "cell_table_col"


@dataclass(frozen=True)
class ClearanceTrace:
    """Clearance samples for one recorded command or observed state stream."""

    geom_distance_m: np.ndarray
    closest_distance_geom: np.ndarray
    vertical_clearance_m: np.ndarray
    closest_vertical_geom: np.ndarray
    eef_z_m: np.ndarray


def _load_full_qpos(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["timestamp", "qpos"])
    timestamp = np.asarray(
        table["timestamp"].cast("int64").to_numpy(),
        dtype=np.int64,
    ).astype(np.float64) * 1.0e-9
    qpos = np.asarray(table["qpos"].to_pylist(), dtype=np.float64)
    if qpos.shape[1] != 8:
        raise ValueError(f"Expected 8 qpos values in {path}, got {qpos.shape}.")
    keep = np.concatenate([[True], np.diff(timestamp) > 1.0e-6])
    return timestamp[keep], qpos[keep]


def _gripper_collision_geoms(
    model: mujoco.MjModel,
    side: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    prefixes = (
        f"ee_base_link_{side}_collision",
        f"finger_inner_{side}_collision",
        f"finger_outer_{side}_collision",
    )
    ids: list[int] = []
    names: list[str] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is not None and name.startswith(prefixes):
            ids.append(geom_id)
            names.append(name)
    if not ids:
        raise RuntimeError(f"No gripper collision geoms found for {side}.")
    return np.asarray(ids, dtype=np.int32), tuple(names)


def _clearance_trace(
    side: str,
    qpos: np.ndarray,
) -> tuple[ClearanceTrace, tuple[str, ...]]:
    setup = study.make_setup("bimanual")
    model = setup.model
    data = setup.data
    table_id = model.geom(TABLE_GEOM).id
    gripper_ids, gripper_names = _gripper_collision_geoms(model, side)
    frame_id = model.site(f"{side}_ee_control_point").id

    geom_distance = np.empty(qpos.shape[0], dtype=np.float64)
    closest_distance_geom = np.empty(qpos.shape[0], dtype=np.int16)
    vertical_clearance = np.empty(qpos.shape[0], dtype=np.float64)
    closest_vertical_geom = np.empty(qpos.shape[0], dtype=np.int16)
    eef_z = np.empty(qpos.shape[0], dtype=np.float64)
    fromto = np.empty(6, dtype=np.float64)
    for sample, arm_qpos in enumerate(qpos):
        setup.joint_resolver.set_qpos(data.qpos, arm_qpos, side)
        mujoco.mj_forward(model, data)
        distances = np.asarray(
            [
                mujoco.mj_geomDistance(
                    model,
                    data,
                    table_id,
                    int(geom_id),
                    2.0,
                    fromto,
                )
                for geom_id in gripper_ids
            ],
            dtype=np.float64,
        )
        closest = int(np.argmin(distances))
        geom_distance[sample] = distances[closest]
        closest_distance_geom[sample] = closest

        table_rotation = data.geom_xmat[table_id].reshape(3, 3)
        table_center = data.geom_xpos[table_id]
        table_size = model.geom_size[table_id]
        vertical = np.full(gripper_ids.size, np.inf, dtype=np.float64)
        for local_index, geom_id in enumerate(gripper_ids):
            mesh_id = model.geom_dataid[geom_id]
            vertex_start = model.mesh_vertadr[mesh_id]
            vertex_count = model.mesh_vertnum[mesh_id]
            vertices = model.mesh_vert[
                vertex_start : vertex_start + vertex_count
            ]
            geom_rotation = data.geom_xmat[geom_id].reshape(3, 3)
            world_vertices = (
                data.geom_xpos[geom_id]
                + vertices @ geom_rotation.T
            )
            table_vertices = (world_vertices - table_center) @ table_rotation
            overlaps_x = (
                np.max(table_vertices[:, 0]) >= -table_size[0]
                and np.min(table_vertices[:, 0]) <= table_size[0]
            )
            overlaps_y = (
                np.max(table_vertices[:, 1]) >= -table_size[1]
                and np.min(table_vertices[:, 1]) <= table_size[1]
            )
            if overlaps_x and overlaps_y:
                vertical[local_index] = (
                    np.min(table_vertices[:, 2]) - table_size[2]
                )
        vertical_closest = int(np.argmin(vertical))
        vertical_clearance[sample] = vertical[vertical_closest]
        closest_vertical_geom[sample] = vertical_closest
        eef_z[sample] = data.site_xpos[frame_id, 2]

    return ClearanceTrace(
        geom_distance_m=geom_distance,
        closest_distance_geom=closest_distance_geom,
        vertical_clearance_m=vertical_clearance,
        closest_vertical_geom=closest_vertical_geom,
        eef_z_m=eef_z,
    ), gripper_names


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output_dir = OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = recorded._event_lookup(output_dir)
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "table_geom": TABLE_GEOM,
        "distance_semantics": (
            "Primary clearance is the lowest collision-mesh vertex above the "
            "tabletop, after transforming vertices into the table frame. "
            "MuJoCo mj_geomDistance is retained only as a penetration check "
            "because separated box-mesh pairs return zero in this build. "
            "Full 8D recorded qpos is used."
        ),
        "episodes": {},
    }

    for episode in sorted(set(recorded.EPISODES.values())):
        root = recorded.RUN_DIR / "dataset" / "episodes" / str(episode)
        record = recorded.load_record(
            episode,
            "right",
            output_dir / "cache",
        )
        traces: dict[str, ClearanceTrace] = {}
        geom_names: tuple[str, ...] | None = None
        for stream in ("action", "obs"):
            timestamp, qpos = _load_full_qpos(
                root / stream / "arms" / "right" / "state.parquet"
            )
            aligned_qpos = recorded._interpolate(
                timestamp,
                qpos,
                record.timestamp_s,
            )
            trace, names = _clearance_trace("right", aligned_qpos)
            traces[stream] = trace
            geom_names = names

        assert geom_names is not None
        metadata["episodes"][str(episode)] = {
            "samples": int(record.time.size),
            "duration_s": float(record.time[-1]),
            "gripper_collision_geoms": list(geom_names),
            "action_min_clearance_m": float(
                np.min(traces["action"].geom_distance_m)
            ),
            "obs_min_clearance_m": float(
                np.min(traces["obs"].geom_distance_m)
            ),
            "action_min_vertical_clearance_m": float(
                np.min(traces["action"].vertical_clearance_m)
            ),
            "obs_min_vertical_clearance_m": float(
                np.min(traces["obs"].vertical_clearance_m)
            ),
        }

        np.savez_compressed(
            output_dir / f"episode_{episode}_right_table_clearance.npz",
            time=record.time,
            action_geom_distance_m=traces["action"].geom_distance_m,
            obs_geom_distance_m=traces["obs"].geom_distance_m,
            action_closest_distance_geom=(
                traces["action"].closest_distance_geom
            ),
            obs_closest_distance_geom=traces["obs"].closest_distance_geom,
            action_vertical_clearance_m=(
                traces["action"].vertical_clearance_m
            ),
            obs_vertical_clearance_m=traces["obs"].vertical_clearance_m,
            action_closest_vertical_geom=(
                traces["action"].closest_vertical_geom
            ),
            obs_closest_vertical_geom=traces["obs"].closest_vertical_geom,
            action_eef_z_m=traces["action"].eef_z_m,
            obs_eef_z_m=traces["obs"].eef_z_m,
            geom_names=np.asarray(geom_names),
        )

        selected = [
            event
            for (family, side, _), event in lookup.items()
            if event.episode == episode
            and side == "right"
            and (event.family, event.side, event.index)
            in {
                (family, selected_side, index)
                for family, selected_side, index, _ in recorded.SELECTED_EVENTS
            }
        ]
        for event in selected:
            for stream, trace in traces.items():
                values = trace.geom_distance_m[
                    event.core_start : event.core_end
                ]
                vertical = trace.vertical_clearance_m[
                    event.core_start : event.core_end
                ]
                closest_distance = trace.closest_distance_geom[
                    event.core_start : event.core_end
                ]
                closest_vertical = trace.closest_vertical_geom[
                    event.core_start : event.core_end
                ]
                min_local = int(np.argmin(values))
                min_vertical_local = int(np.argmin(vertical))
                distance_geom_index = int(closest_distance[min_local])
                vertical_geom_index = int(
                    closest_vertical[min_vertical_local]
                )
                rows.append(
                    {
                        "episode": event.episode,
                        "family": event.family,
                        "case": event.index,
                        "side": event.side,
                        "stream": stream,
                        "start_s": event.start_s,
                        "end_s": event.end_s,
                        "min_geom_distance_m": float(np.min(values)),
                        "closest_distance_geom": geom_names[
                            distance_geom_index
                        ],
                        "min_vertical_clearance_m": float(
                            np.min(vertical)
                        ),
                        "p05_vertical_clearance_m": float(
                            np.quantile(vertical, 0.05)
                        ),
                        "fraction_vertical_below_10mm": float(
                            np.mean(vertical < 0.010)
                        ),
                        "fraction_vertical_at_or_below_zero": float(
                            np.mean(vertical <= 0.0)
                        ),
                        "closest_vertical_geom": geom_names[
                            vertical_geom_index
                        ],
                        "eef_z_at_min_vertical_clearance_m": float(
                            trace.eef_z_m[
                                event.core_start + min_vertical_local
                            ]
                        ),
                    }
                )

    _write_csv(output_dir / "selected_table_clearance.csv", rows)
    (output_dir / "table_clearance_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))
    for row in rows:
        print(
            f"ep{row['episode']} {row['family']}#{row['case']} "
            f"{row['stream']}: "
            f"vertical={1000 * row['min_vertical_clearance_m']:.1f} mm "
            f"({row['closest_vertical_geom']})"
        )


if __name__ == "__main__":
    main()
