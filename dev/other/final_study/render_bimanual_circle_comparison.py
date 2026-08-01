#!/usr/bin/env python3
"""Render bimanual circle replays with both elbow trajectories visible."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_videos as video
import study

RESULTS = HERE / "results" / "bimanual_circle_selection_20260731"
OUTPUT = Path(
    "/home/hilab/workdir.evaluation/note/openarm_control/final_report/videos"
)
PANEL_WIDTH = 640
PANEL_HEIGHT = 480
DEFAULT_FPS = 20
PATH_TRAIL_SECONDS = 1.0


@dataclass(frozen=True)
class Panel:
    """Trace profile and user-facing label for one panel."""

    profile: str
    label: str


DIAGNOSTIC_PANELS = (
    Panel("pr_current_dataflow", "PR: current dataflow"),
    Panel(
        "mainline_with_ik_velocity",
        "Mainline + same IK velocity limits",
    ),
    Panel("pr_current_no_nullspace", "PR without nullspace task"),
    Panel("mainline_no_ik_velocity", "Mainline: driver cap only"),
)

FINAL_PANELS = DIAGNOSTIC_PANELS[:2]

ORI_MAIN_PANELS = (
    Panel("pr_current_dataflow", "PR: current dataflow"),
    Panel(
        "ori_main_same_ik_velocity",
        "ori/main + same numeric IK velocity limits",
    ),
)


def _trace_path(
    profile: str,
    scenario: str,
    results_dir: Path,
) -> Path:
    matches = list(
        results_dir.glob(f"**/traces/*_{profile}_{scenario}.npz")
    )
    if not matches:
        raise RuntimeError(f"Missing trace for {profile}/{scenario}.")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _load_trace(
    profile: str,
    scenario: str,
    results_dir: Path,
) -> np.lib.npyio.NpzFile:
    return np.load(
        _trace_path(profile, scenario, results_dir),
        allow_pickle=False,
    )


class BimanualDiagnosticRenderer(video.BimanualRenderer):
    """Composite actual and command configurations for both arms."""

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        self.camera.lookat[:] = [0.25, 0.0, 1.25]
        self.camera.distance = 1.08
        self.camera.azimuth = 180.0
        self.camera.elevation = -12.0
        self._background = self._render_background()

    def _set_bimanual_q(
        self,
        right_q: np.ndarray,
        left_q: np.ndarray,
    ) -> None:
        self.setup.joint_resolver.set_qpos(
            self.data.qpos,
            np.append(np.asarray(right_q, dtype=np.float64), 0.0),
            "right",
        )
        self.setup.joint_resolver.set_qpos(
            self.data.qpos,
            np.append(np.asarray(left_q, dtype=np.float64), 0.0),
            "left",
        )
        mujoco.mj_forward(self.model, self.data)

    def _render_background(self) -> np.ndarray:
        self._set_arm_visible(False)
        self._set_bimanual_q(study.START_Q_RIGHT, study.START_Q_LEFT)
        self.renderer.update_scene(
            self.data,
            camera=self.camera,
            scene_option=self.option,
        )
        background = self.renderer.render().copy()
        self._set_arm_visible(True)
        return background

    def _render_actual(
        self,
        right_q: np.ndarray,
        left_q: np.ndarray,
        right_target: np.ndarray,
        left_target: np.ndarray,
    ) -> np.ndarray:
        self._set_arm_visible(True)
        self.model.geom_matid[self._right_visual_geoms] = self._original_matid[
            self._right_visual_geoms
        ]
        self.model.geom_rgba[self._right_visual_geoms] = self._original_rgba[
            self._right_visual_geoms
        ]
        self._set_bimanual_q(right_q, left_q)
        self.renderer.update_scene(
            self.data,
            camera=self.camera,
            scene_option=self.option,
        )
        self._add_target(
            right_target,
            right_target[None, :],
            draw_trail=False,
        )
        self._add_target(
            left_target,
            left_target[None, :],
            draw_trail=False,
        )
        return self.renderer.render().copy()

    def _render_command(
        self,
        right_q: np.ndarray,
        left_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._set_arm_visible(True)
        self.model.geom_matid[self._right_visual_geoms] = -1
        self.model.geom_rgba[self._right_visual_geoms] = np.array(
            [0.05, 0.82, 0.95, 1.0]
        )
        self._set_bimanual_q(right_q, left_q)
        self.renderer.update_scene(
            self.data,
            camera=self.camera,
            scene_option=self.option,
        )
        command = self.renderer.render().copy()
        difference = np.max(
            np.abs(
                command.astype(np.int16)
                - self._background.astype(np.int16)
            ),
            axis=2,
        )
        return command, difference > 6

    def _draw_path(
        self,
        draw: ImageDraw.ImageDraw,
        points: np.ndarray,
        *,
        relative: bool,
        color: tuple[int, int, int],
        width: int,
        fade_exponent: float | None,
        constant_alpha: int = 235,
        maximum_points: int = 100,
    ) -> np.ndarray:
        if points.shape[0] == 0:
            return np.empty((0, 2))
        indices = np.linspace(
            0,
            points.shape[0] - 1,
            min(points.shape[0], maximum_points),
            dtype=int,
        )
        selected = points[indices]
        pixels = (
            self._project_relative_positions(selected)
            if relative
            else self._project_world_positions(selected)
        )
        self._draw_fading_path(
            draw,
            pixels,
            color=color,
            width=width,
            fade_exponent=fade_exponent,
            constant_alpha=constant_alpha,
        )
        return pixels

    def _overlay_paths(
        self,
        frame: np.ndarray,
        trace: np.lib.npyio.NpzFile,
        index: int,
    ) -> np.ndarray:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image, "RGBA")
        trail_count = max(2, round(PATH_TRAIL_SECONDS / study.CONTROL_DT))
        elbow_colors = {
            "right": (255, 224, 138),
            "left": (142, 226, 255),
        }
        for side in study.SIDES:
            self._draw_path(
                draw,
                trace[f"{side}_target_pose"][
                    max(0, index + 1 - trail_count) : index + 1
                ],
                relative=True,
                color=(255, 52, 42),
                width=3,
                fade_exponent=2.0,
                maximum_points=90,
            )
            self._draw_path(
                draw,
                trace[f"{side}_actual_pose"][
                    max(0, index + 1 - trail_count) : index + 1
                ],
                relative=True,
                color=(255, 184, 5),
                width=3,
                fade_exponent=2.0,
                maximum_points=90,
            )
        for side in study.SIDES:
            elbow_pixels = self._draw_path(
                draw,
                trace[f"{side}_actual_elbow"][
                    max(0, index + 1 - trail_count) : index + 1
                ],
                relative=False,
                color=elbow_colors[side],
                width=2,
                fade_exponent=2.0,
                maximum_points=90,
            )
            if elbow_pixels.size and np.all(np.isfinite(elbow_pixels[-1])):
                x, y = np.rint(elbow_pixels[-1]).astype(int)
                draw.ellipse(
                    (x - 3, y - 3, x + 3, y + 3),
                    fill=(*elbow_colors[side], 190),
                )
        return np.asarray(image)

    def composite(
        self,
        trace: np.lib.npyio.NpzFile,
        index: int,
    ) -> np.ndarray:
        actual = self._render_actual(
            trace["right_actual_q"][index],
            trace["left_actual_q"][index],
            trace["right_target_pose"][index],
            trace["left_target_pose"][index],
        ).astype(np.float32)
        command, mask = self._render_command(
            trace["right_command_q"][index],
            trace["left_command_q"][index],
        )
        alpha = (video.GHOST_ALPHA * mask.astype(np.float32))[..., None]
        combined = actual * (1.0 - alpha) + command.astype(np.float32) * alpha
        return self._overlay_paths(
            np.clip(combined, 0, 255).astype(np.uint8),
            trace,
            index,
        )


def _position_error(trace: np.lib.npyio.NpzFile, index: int) -> float:
    return max(
        float(
            np.linalg.norm(
                trace[f"{side}_target_pose"][index, :3]
                - trace[f"{side}_actual_pose"][index, :3]
            )
        )
        for side in study.SIDES
    )


def _maximum_position_error(trace: np.lib.npyio.NpzFile) -> float:
    return max(
        float(
            np.max(
                np.linalg.norm(
                    trace[f"{side}_target_pose"][:, :3]
                    - trace[f"{side}_actual_pose"][:, :3],
                    axis=1,
                )
            )
        )
        for side in study.SIDES
    )


def _elbow_baselines(
    trace: np.lib.npyio.NpzFile,
) -> dict[str, float]:
    count = max(10, round(0.4 / study.CONTROL_DT))
    return {
        side: float(np.median(trace[f"{side}_actual_elbow"][:count, 2]))
        for side in study.SIDES
    }


def _annotate_elbow_height(
    panel: np.ndarray,
    trace: np.lib.npyio.NpzFile,
    index: int,
    baseline: dict[str, float],
) -> np.ndarray:
    image = Image.fromarray(panel)
    draw = ImageDraw.Draw(image, "RGBA")
    rises = {
        side: 100.0
        * (
            float(trace[f"{side}_actual_elbow"][index, 2])
            - baseline[side]
        )
        for side in study.SIDES
    }
    label = f"elbow height from start  R {rises['right']:+.1f} / L {rises['left']:+.1f} cm"
    label_font = video.font(13)
    bounds = draw.textbbox((0, 0), label, font=label_font)
    width = bounds[2] - bounds[0]
    draw.rounded_rectangle(
        (PANEL_WIDTH - width - 27, 77, PANEL_WIDTH - 10, 103),
        radius=4,
        fill=(8, 15, 28, 175),
    )
    draw.text(
        (PANEL_WIDTH - width - 18, 80),
        label,
        fill=(255, 237, 173, 255),
        font=label_font,
    )
    return np.asarray(image)


def _render_frame(
    renderer: BimanualDiagnosticRenderer,
    traces: list[np.lib.npyio.NpzFile],
    panels: tuple[Panel, ...],
    max_errors: list[float],
    baselines: list[dict[str, float]],
    time_s: float,
    columns: int,
    playback_rate: float,
) -> np.ndarray:
    rendered: list[np.ndarray] = []
    for trace, panel, max_error, baseline in zip(
        traces,
        panels,
        max_errors,
        baselines,
        strict=True,
    ):
        index = int(np.argmin(np.abs(trace["times"] - time_s)))
        frame = renderer.composite(trace, index)
        annotated = video.annotated_panel(
            frame,
            panel.label,
            time_s=time_s,
            position_error_m=_position_error(trace, index),
            max_position_error_m=max_error,
            playback_rate=playback_rate,
            show_elbow=True,
        )
        rendered.append(
            _annotate_elbow_height(annotated, trace, index, baseline)
        )
    rows = [
        np.concatenate(rendered[start : start + columns], axis=1)
        for start in range(0, len(rendered), columns)
    ]
    return np.concatenate(rows, axis=0)


def render(
    *,
    scenario: str,
    panels: tuple[Panel, ...],
    name: str,
    preview_time_s: float | None,
    fps: int,
    results_dir: Path = RESULTS,
    start_time_s: float = 0.0,
    end_time_s: float | None = None,
    playback_rate: float = 1.0,
) -> Path:
    if playback_rate <= 0.0:
        raise ValueError("playback_rate must be positive.")
    traces = [
        _load_trace(panel.profile, scenario, results_dir)
        for panel in panels
    ]
    max_errors = [_maximum_position_error(trace) for trace in traces]
    baselines = [_elbow_baselines(trace) for trace in traces]
    columns = 2
    duration = min(float(trace["times"][-1]) for trace in traces)
    start_time_s = float(np.clip(start_time_s, 0.0, duration))
    end_time_s = (
        duration
        if end_time_s is None
        else float(np.clip(end_time_s, start_time_s, duration))
    )
    renderer = BimanualDiagnosticRenderer(PANEL_WIDTH, PANEL_HEIGHT)
    try:
        if preview_time_s is not None:
            frame = _render_frame(
                renderer,
                traces,
                panels,
                max_errors,
                baselines,
                min(preview_time_s, duration),
                columns,
                playback_rate,
            )
            path = OUTPUT / f"{name}_preview_{preview_time_s:.2f}s.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(path)
            return path

        output = OUTPUT / f"{name}.mp4"
        temporary = output.with_name(f".{output.stem}.partial.mp4")
        writer = video.VideoWriter(
            temporary,
            PANEL_WIDTH * columns,
            PANEL_HEIGHT * (len(panels) // columns),
            fps,
        )
        try:
            for time_s in np.arange(
                start_time_s,
                end_time_s + 1.0e-9,
                playback_rate / fps,
            ):
                writer.write(
                    _render_frame(
                        renderer,
                        traces,
                        panels,
                        max_errors,
                        baselines,
                        time_s,
                        columns,
                        playback_rate,
                    )
                )
        finally:
            writer.close()
        os.replace(temporary, output)
        return output
    finally:
        renderer.close()
        for trace in traces:
            trace.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default="recorded_bimanual_circle_ep77_long",
    )
    parser.add_argument(
        "--layout",
        choices=("diagnostic", "final", "ori-main"),
        default="diagnostic",
    )
    parser.add_argument("--name")
    parser.add_argument("--preview-time", type=float)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--playback-rate", type=float, default=1.0)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.layout == "diagnostic":
        panels = DIAGNOSTIC_PANELS
    elif args.layout == "ori-main":
        panels = ORI_MAIN_PANELS
    else:
        panels = FINAL_PANELS
    name = args.name or f"{args.scenario}_{args.layout}_v1"
    print(
        render(
            scenario=args.scenario,
            panels=panels,
            name=name,
            preview_time_s=args.preview_time,
            fps=args.fps,
            results_dir=args.results_dir,
            start_time_s=args.start_time,
            end_time_s=args.end_time,
            playback_rate=args.playback_rate,
        )
    )


if __name__ == "__main__":
    main()
