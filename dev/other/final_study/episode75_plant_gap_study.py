#!/usr/bin/env python3
"""Replay episode 75 IK commands through MuJoCo and compare with hardware.

The action stream is the joint-position command emitted by IK.  Replaying that
exact stream through the dynamic plant separates command-generation behavior
from differences between the simulated and physical position servos.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import episode73_tracking_gap_study as tracking_gap
import recorded_intervention_study as recorded
import study

EPISODE = 75
DT = study.CONTROL_DT
SIDES = study.SIDES
OUTPUT_DIR = HERE / "results" / "episode75_plant_gap_20260731"
FINAL_ASSET_DIR = (
    HERE.parents[2]
    / "note"
    / "openarm_control"
    / "final_report"
    / "assets"
)
RECORDED_DIR = (
    HERE / "results" / "recorded_interventions_20260731"
)
EVENT_IDS = (5, 41)
VIDEO_DIR = OUTPUT_DIR / "video" / "traces"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _profiles() -> list[tracking_gap.PlantProfile]:
    return [
        tracking_gap.PlantProfile("nominal_no_gc"),
        tracking_gap.PlantProfile(
            "no_driver_cap",
            driver_velocity_limit=False,
        ),
        tracking_gap.PlantProfile("delay_8ms", command_delay_s=0.008),
        tracking_gap.PlantProfile("delay_20ms", command_delay_s=0.020),
        tracking_gap.PlantProfile("delay_40ms", command_delay_s=0.040),
        tracking_gap.PlantProfile(
            "gravity_compensation",
            gravity_compensation=True,
        ),
        tracking_gap.PlantProfile(
            "kp_050_kv_100",
            actuator_kp_scale=0.50,
        ),
        tracking_gap.PlantProfile(
            "kp_075_kv_100",
            actuator_kp_scale=0.75,
        ),
        tracking_gap.PlantProfile(
            "kp_125_kv_100",
            actuator_kp_scale=1.25,
        ),
        tracking_gap.PlantProfile(
            "kp_100_kv_050",
            actuator_kv_scale=0.50,
        ),
        tracking_gap.PlantProfile(
            "kp_100_kv_150",
            actuator_kv_scale=1.50,
        ),
        tracking_gap.PlantProfile(
            "kp_100_kv_200",
            actuator_kv_scale=2.00,
        ),
    ]


def _records() -> dict[str, recorded.ArmRecord]:
    cache_dir = RECORDED_DIR / "cache"
    return {
        side: recorded.load_record(EPISODE, side, cache_dir)
        for side in SIDES
    }


def _events() -> dict[int, recorded.Event]:
    rows = json.loads(
        (RECORDED_DIR / "events.json").read_text(encoding="utf-8")
    )
    result: dict[int, recorded.Event] = {}
    for row in rows:
        event = recorded.Event(**row)
        if (
            event.episode == EPISODE
            and event.side == "right"
            and event.index in EVENT_IDS
        ):
            result[event.index] = event
    missing = set(EVENT_IDS) - set(result)
    if missing:
        raise RuntimeError(f"Missing episode-75 events: {sorted(missing)}")
    return result


def _save_trace(
    path: Path,
    profile: tracking_gap.PlantProfile,
    output: dict[str, np.ndarray],
) -> None:
    payload: dict[str, np.ndarray] = {
        key: value for key, value in output.items()
    }
    payload["profile_json"] = np.asarray(
        json.dumps(asdict(profile), sort_keys=True)
    )
    np.savez_compressed(path, **payload)


def run_profiles() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trace_dir = OUTPUT_DIR / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    records = _records()
    events = _events()
    full_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for profile in _profiles():
        print(f"Running {profile.name}...", flush=True)
        output = tracking_gap._run_plant(records, profile)
        for side in SIDES:
            geometry = recorded._geometry(side, output[f"{side}_q"])
            output[f"{side}_world_eef"] = geometry["world_eef"]
        _save_trace(trace_dir / f"{profile.name}.npz", profile, output)
        full_rows.extend(
            tracking_gap._plant_metrics(profile, records, output)
        )
        for event_id, event in events.items():
            window = slice(event.start, event.end)
            core = slice(event.core_start, event.core_end)
            record = records["right"]
            command_z = record.action_world_eef[:, 2]
            actual_z = record.obs_world_eef[:, 2]
            sim_world_eef = output["right_world_eef"]
            sim_z = sim_world_eef[:, 2]
            real_below_command = command_z[core] - actual_z[core]
            sim_below_command = command_z[core] - sim_z[core]
            actual_position_error = np.linalg.norm(
                record.obs_world_eef[window]
                - record.action_world_eef[window],
                axis=1,
            )
            sim_position_error = np.linalg.norm(
                sim_world_eef[window]
                - record.action_world_eef[window],
                axis=1,
            )
            event_rows.append(
                {
                    **asdict(profile),
                    "event": event_id,
                    "start_s": event.start_s,
                    "end_s": event.end_s,
                    "hardware_below_command_max_m": float(
                        np.max(real_below_command)
                    ),
                    "simulation_below_command_max_m": float(
                        np.max(sim_below_command)
                    ),
                    "hardware_below_simulation_max_m": float(
                        np.max(sim_z[core] - actual_z[core])
                    ),
                    "hardware_position_error_rmse_m": float(
                        np.sqrt(np.mean(np.square(actual_position_error)))
                    ),
                    "simulation_position_error_rmse_m": float(
                        np.sqrt(np.mean(np.square(sim_position_error)))
                    ),
                    "hardware_position_error_max_m": float(
                        np.max(actual_position_error)
                    ),
                    "simulation_position_error_max_m": float(
                        np.max(sim_position_error)
                    ),
                    "driver_limit_active_pct": float(
                        100.0
                        * np.mean(
                            np.any(output["right_limited"][core], axis=1)
                        )
                    ),
                    "driver_limit_j1_pct": float(
                        100.0 * np.mean(output["right_limited"][core, 0])
                    ),
                    "driver_limit_j4_pct": float(
                        100.0 * np.mean(output["right_limited"][core, 3])
                    ),
                }
            )
    _write_csv(OUTPUT_DIR / "profile_metrics.csv", full_rows)
    _write_csv(OUTPUT_DIR / "event_metrics.csv", event_rows)


def _load_driver_state(
    side: str,
    target_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = (
        recorded.RUN_DIR
        / "dataset"
        / "episodes"
        / str(EPISODE)
        / "obs"
        / "arms"
        / side
        / "state.parquet"
    )
    table = pq.read_table(path, columns=["timestamp", "qpos", "qvel", "qtorque"])
    timestamp = np.asarray(
        table["timestamp"].cast("int64").to_numpy(),
        dtype=np.int64,
    ).astype(np.float64) * 1.0e-9
    keep = np.concatenate([[True], np.diff(timestamp) > 1.0e-6])
    timestamp = timestamp[keep]
    values = []
    for name in ("qpos", "qvel", "qtorque"):
        raw = np.asarray(table[name].to_pylist(), dtype=np.float64)[keep, :7]
        values.append(recorded._interpolate(timestamp, raw, target_time))
    return values[0], values[1], values[2]


def inverse_dynamics_residual() -> None:
    """Compare recorded motor torque with model inverse dynamics."""
    records = _records()
    driver = {
        side: _load_driver_state(side, records[side].timestamp_s)
        for side in SIDES
    }
    setup = study.make_setup("bimanual")
    model = setup.model
    data = setup.data
    dofs = {
        side: setup.joint_resolver.arm_dof_indices(side)
        for side in SIDES
    }
    qacc = {
        side: recorded._derivative(
            recorded._smooth(driver[side][1], 0.060),
            0.060,
        )
        for side in SIDES
    }
    count = min(record.time.shape[0] for record in records.values())
    inverse = {
        side: np.empty((count, 7), dtype=np.float64)
        for side in SIDES
    }
    bias = {
        side: np.empty((count, 7), dtype=np.float64)
        for side in SIDES
    }
    inverse_full = np.empty(model.nv, dtype=np.float64)
    bias_full = np.empty(model.nv, dtype=np.float64)
    for index in range(count):
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        for side in SIDES:
            setup.joint_resolver.set_qpos(
                data.qpos,
                np.append(driver[side][0][index], 0.0),
                side,
            )
            data.qvel[dofs[side]] = driver[side][1][index]
            data.qacc[dofs[side]] = qacc[side][index]
        mujoco.mj_forward(model, data)
        # Recursive Newton-Euler excludes contact-constraint forces.  Those
        # become enormous when a recorded configuration intersects the
        # simplified scene and are not part of the motor torque model here.
        mujoco.mj_rne(model, data, 1, inverse_full)
        mujoco.mj_rne(model, data, 0, bias_full)
        for side in SIDES:
            inverse[side][index] = inverse_full[dofs[side]]
            bias[side][index] = bias_full[dofs[side]]

    rows: list[dict[str, Any]] = []
    npz_payload: dict[str, np.ndarray] = {}
    for side in SIDES:
        torque = driver[side][2][:count]
        npz_payload[f"{side}_recorded_torque"] = torque
        npz_payload[f"{side}_inverse_torque"] = inverse[side]
        npz_payload[f"{side}_bias_torque"] = bias[side]
        for joint in range(7):
            measured = torque[:, joint]
            expected = inverse[side][:, joint]
            design = np.column_stack([expected, np.ones_like(expected)])
            slope, offset = np.linalg.lstsq(
                design,
                measured,
                rcond=None,
            )[0]
            correlation = float(np.corrcoef(measured, expected)[0, 1])
            gravity = bias[side][:, joint]
            gravity_design = np.column_stack(
                [gravity, np.ones_like(gravity)]
            )
            gravity_slope, gravity_offset = np.linalg.lstsq(
                gravity_design,
                measured,
                rcond=None,
            )[0]
            rows.append(
                {
                    "side": side,
                    "joint": joint + 1,
                    "correlation": correlation,
                    "fit_slope": float(slope),
                    "fit_offset_nm": float(offset),
                    "direct_rmse_nm": float(
                        np.sqrt(np.mean(np.square(measured - expected)))
                    ),
                    "fit_rmse_nm": float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    measured - (slope * expected + offset)
                                )
                            )
                        )
                    ),
                    "gravity_correlation": float(
                        np.corrcoef(measured, gravity)[0, 1]
                    ),
                    "gravity_fit_slope": float(gravity_slope),
                    "gravity_fit_offset_nm": float(gravity_offset),
                    "gravity_fit_rmse_nm": float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    measured
                                    - (
                                        gravity_slope * gravity
                                        + gravity_offset
                                    )
                                )
                            )
                        )
                    ),
                    "recorded_p95_abs_nm": float(
                        np.percentile(np.abs(measured), 95)
                    ),
                    "inverse_p95_abs_nm": float(
                        np.percentile(np.abs(expected), 95)
                    ),
                    "bias_p95_abs_nm": float(
                        np.percentile(np.abs(bias[side][:, joint]), 95)
                    ),
                }
            )
    _write_csv(OUTPUT_DIR / "inverse_dynamics_torque.csv", rows)
    np.savez_compressed(OUTPUT_DIR / "inverse_dynamics_torque.npz", **npz_payload)


def plot_summary() -> None:
    records = _records()
    events = _events()
    profiles = {
        profile.name: profile
        for profile in _profiles()
    }
    selected = (
        "nominal_no_gc",
        "no_driver_cap",
        "delay_20ms",
        "kp_050_kv_100",
        "gravity_compensation",
    )
    traces = {
        name: np.load(OUTPUT_DIR / "traces" / f"{name}.npz")
        for name in selected
    }
    colors = {
        "nominal_no_gc": "#1f77b4",
        "no_driver_cap": "#9467bd",
        "delay_20ms": "#d62728",
        "kp_050_kv_100": "#ff7f0e",
        "gravity_compensation": "#2ca02c",
    }
    labels = {
        "nominal_no_gc": "MuJoCo nominal",
        "no_driver_cap": "MuJoCo no driver cap",
        "delay_20ms": "MuJoCo +20 ms",
        "kp_050_kv_100": "MuJoCo 0.5x Kp",
        "gravity_compensation": "MuJoCo gravity comp",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    record = records["right"]
    for column, event_id in enumerate(EVENT_IDS):
        event = events[event_id]
        window = slice(event.start, event.end)
        time = record.time[window] - record.time[event.start]
        axis = axes[0, column]
        axis.plot(
            time,
            record.action_world_eef[window, 2],
            color="#111111",
            linewidth=2.0,
            label="IK command",
        )
        axis.plot(
            time,
            record.obs_world_eef[window, 2],
            color="#00a6a6",
            linewidth=2.2,
            label="hardware",
        )
        for name in selected:
            axis.plot(
                time,
                traces[name]["right_world_eef"][window, 2],
                color=colors[name],
                linewidth=1.2,
                alpha=0.9,
                label=labels[name],
            )
        axis.set_title(f"Episode 75 retract event {event_id}: vertical motion")
        axis.set_ylabel("world z [m]")
        axis.grid(alpha=0.25)

        axis = axes[1, column]
        hardware_error = np.linalg.norm(
            record.obs_world_eef[window]
            - record.action_world_eef[window],
            axis=1,
        )
        axis.plot(
            time,
            100.0 * hardware_error,
            color="#00a6a6",
            linewidth=2.2,
            label="hardware",
        )
        for name in selected:
            error = np.linalg.norm(
                traces[name]["right_world_eef"][window]
                - record.action_world_eef[window],
                axis=1,
            )
            axis.plot(
                time,
                100.0 * error,
                color=colors[name],
                linewidth=1.2,
                alpha=0.9,
                label=labels[name],
            )
        axis.set_title("Distance from identical IK command")
        axis.set_xlabel("event time [s]")
        axis.set_ylabel("position error [cm]")
        axis.grid(alpha=0.25)
    handles, labels_out = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_out,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=4,
        fontsize=9,
    )
    fig.suptitle(
        "Same episode-75 IK command: physical arm vs MuJoCo plant",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "episode75_direct_plant_summary.png", dpi=180)
    FINAL_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FINAL_ASSET_DIR / "29_episode75_direct_plant_summary.png",
        dpi=180,
    )
    plt.close(fig)

    metadata = {
        "episode": EPISODE,
        "events": EVENT_IDS,
        "profiles": [asdict(profiles[name]) for name in selected],
        "command_semantics": "recorded IK joint-position action",
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    _save_video_traces(records, events, traces)


def _save_video_traces(
    records: dict[str, recorded.ArmRecord],
    events: dict[int, recorded.Event],
    traces: dict[str, np.lib.npyio.NpzFile],
) -> None:
    """Save renderer-compatible windows for direct-command validation."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    record = records["right"]
    for event_id, event in events.items():
        window = slice(event.start, event.end)
        count = event.end - event.start
        scenario = f"episode75_event{event_id:02d}_direct_command"
        command_q = record.action_q[window]
        command_pose = record.action_pose[window]
        command_elbow = record.action_elbow[window]
        common = {
            "times": np.arange(count, dtype=np.float64) * DT,
            "phase": np.ones(count, dtype=np.int8),
            "right_target_pose": command_pose,
            "right_target": command_pose,
            "right_command_q": command_q,
            "right_driver_q": command_q,
            "right_command_pose": command_pose,
            "right_driver_pose": command_pose,
            "right_command_elbow": command_elbow,
            "right_driver_elbow": command_elbow,
        }

        variants = {
            "kinematic_command": (
                command_q,
                command_pose,
                command_elbow,
            ),
            "recorded_hardware": (
                record.obs_q[window],
                record.obs_pose[window],
                record.obs_elbow[window],
            ),
            "mujoco_nominal": (
                traces["nominal_no_gc"]["right_q"][window],
                traces["nominal_no_gc"]["right_pose"][window],
                traces["nominal_no_gc"]["right_elbow"][window],
            ),
            "mujoco_gravity_comp": (
                traces["gravity_compensation"]["right_q"][window],
                traces["gravity_compensation"]["right_pose"][window],
                traces["gravity_compensation"]["right_elbow"][window],
            ),
        }
        for name, (actual_q, actual_pose, actual_elbow) in variants.items():
            np.savez_compressed(
                VIDEO_DIR / f"direct_{name}_{scenario}.npz",
                **common,
                right_actual_q=actual_q,
                right_actual_pose=actual_pose,
                right_actual_elbow=actual_elbow,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("profiles", "inverse", "plot", "all"),
        default="all",
        nargs="?",
    )
    args = parser.parse_args()
    if args.stage in {"profiles", "all"}:
        run_profiles()
    if args.stage in {"inverse", "all"}:
        inverse_dynamics_residual()
    if args.stage in {"plot", "all"}:
        plot_summary()


if __name__ == "__main__":
    main()
