#!/usr/bin/env python3
"""Render diagnostic videos with raw-command ghosts and physical arm states."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "final_report_20260801"
SUITE_ROOTS: dict[str, Path] = {}
OUTPUT = HERE.parent / "videos"

PANEL_WIDTH = 640
PANEL_HEIGHT = 480
FPS = 30
GHOST_ALPHA = 0.38
COMPARISON_SLOWDOWN = 2.0
TARGET_AXIS_LENGTH = 0.192
TARGET_AXIS_BACK_LENGTH = 0.048
TARGET_AXIS_WIDTH = 0.0078
TARGET_TRAIL_POINTS = 70
EEF_TRAIL_POINTS = 90
STATE_TRAIL_SAMPLES = 250
ELBOW_TRAIL_POINTS = 90
CURRENT_CAMERA_AZIMUTH = 145.0
REAR_CAMERA_AZIMUTH = 55.0


@dataclass(frozen=True)
class Panel:
    """One profile and label in a comparison video."""

    profile: str
    label: str
    suite: str | None = None
    scenario: str | None = None


@dataclass(frozen=True)
class Comparison:
    """One multi-panel diagnostic video definition."""

    name: str
    suite: str
    scenario: str
    panels: tuple[Panel, ...]
    caption: str
    slowdown: float = COMPARISON_SLOWDOWN
    show_elbow: bool = False
    show_max_error: bool = False
    replay_rear_view: bool = False
    columns: int | None = None
    camera_azimuth: float | None = None
    camera_lookat: tuple[float, float, float] | None = None
    camera_distance: float | None = None
    show_j1_acceleration: bool = False


def load_trace(suite: str, profile: str, scenario: str) -> np.lib.npyio.NpzFile:
    suite_root = SUITE_ROOTS.get(suite, RESULTS / suite)
    matches = list(
        (suite_root / "traces").glob(
            f"*_{profile}_{scenario}.npz"
        )
    )
    if not matches:
        raise RuntimeError(
            f"No trace for {suite}/{profile}/{scenario}"
        )
    selected = max(matches, key=lambda path: path.stat().st_mtime_ns)
    if len(matches) > 1:
        print(
            f"Using newest of {len(matches)} traces for "
            f"{suite}/{profile}/{scenario}: {selected.name}"
        )
    return np.load(selected, allow_pickle=False)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


class ArmRenderer:
    """Render one right arm with optional target and command coloring."""

    def __init__(self, width: int, height: int) -> None:
        self.setup = study.make_setup("bimanual")
        self.model = self.setup.model
        self.data = self.setup.data
        self.width = width
        self.height = height
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height
        self._original_rgba = self.model.geom_rgba.copy()
        self._original_matid = self.model.geom_matid.copy()
        self._right_visual_geoms = self._select_right_visual_geoms()
        self._hide_non_right_geometry()

        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.292, 0.001, 1.18]
        self.camera.distance = 1.12
        self.camera.azimuth = CURRENT_CAMERA_AZIMUTH
        self.camera.elevation = -15.0

        self.option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.option)
        self.option.geomgroup[:] = 0
        self.option.geomgroup[2] = 1
        self.option.sitegroup[:] = 0

        self.renderer = mujoco.Renderer(
            self.model,
            height=height,
            width=width,
        )
        self.origin_site = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "arm_origin",
        )
        self._background = self._render_background()

    def set_camera_view(self, view: str) -> None:
        """Select a diagnostic camera without changing its framing."""
        if view == "current":
            self.camera.azimuth = CURRENT_CAMERA_AZIMUTH
        elif view == "rear":
            self.camera.azimuth = REAR_CAMERA_AZIMUTH
        else:
            raise ValueError(f"Unknown camera view: {view}")
        self._background = self._render_background()

    def set_camera(
        self,
        *,
        azimuth: float,
        lookat: tuple[float, float, float] | None = None,
        distance: float | None = None,
    ) -> None:
        """Apply a comparison-specific camera framing."""
        self.camera.azimuth = azimuth
        if lookat is not None:
            self.camera.lookat[:] = lookat
        if distance is not None:
            self.camera.distance = distance
        self._background = self._render_background()

    def _select_right_visual_geoms(self) -> np.ndarray:
        selected: list[int] = []
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = (
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_id,
                )
                or ""
            )
            if (
                body_name.startswith("openarm_right")
                and int(self.model.geom_group[geom_id]) == 2
            ):
                selected.append(geom_id)
        return np.asarray(selected, dtype=int)

    def _hide_non_right_geometry(self) -> None:
        self.model.geom_rgba[:, 3] = 0.0
        self.model.geom_rgba[self._right_visual_geoms] = self._original_rgba[
            self._right_visual_geoms
        ]

    def _set_arm_visible(self, visible: bool) -> None:
        self.model.geom_rgba[self._right_visual_geoms, 3] = (
            self._original_rgba[self._right_visual_geoms, 3]
            if visible
            else 0.0
        )

    def _set_q(self, right_q: np.ndarray) -> None:
        self.setup.joint_resolver.set_qpos(
            self.data.qpos,
            np.append(np.asarray(right_q, dtype=np.float64), 0.0),
            "right",
        )
        self.setup.joint_resolver.set_qpos(
            self.data.qpos,
            np.append(study.START_Q_LEFT, 0.0),
            "left",
        )
        mujoco.mj_forward(self.model, self.data)

    def _render_background(self) -> np.ndarray:
        self._set_arm_visible(False)
        self._set_q(study.START_Q_RIGHT)
        self.renderer.update_scene(
            self.data,
            camera=self.camera,
            scene_option=self.option,
        )
        background = self.renderer.render().copy()
        self._set_arm_visible(True)
        return background

    def _target_world_pose(
        self,
        target_relative: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        origin_rotation = self.data.site_xmat[self.origin_site].reshape(3, 3)
        relative_rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(relative_rotation, target_relative[3:])
        return (
            self.data.site_xpos[self.origin_site]
            + origin_rotation @ target_relative[:3],
            origin_rotation @ relative_rotation.reshape(3, 3),
        )

    def _target_world(self, target_relative: np.ndarray) -> np.ndarray:
        return self._target_world_pose(target_relative)[0]

    def _project_relative_positions(
        self,
        poses_relative: np.ndarray,
    ) -> np.ndarray:
        world_points = np.asarray(
            [self._target_world(pose) for pose in poses_relative]
        )
        return self._project_world_positions(world_points)

    def _project_world_positions(
        self,
        world_points: np.ndarray,
    ) -> np.ndarray:
        cameras = self.renderer.scene.camera
        camera_position = 0.5 * (cameras[0].pos + cameras[1].pos)
        forward = np.asarray(cameras[0].forward, dtype=np.float64)
        forward /= np.linalg.norm(forward)
        up = np.asarray(cameras[0].up, dtype=np.float64)
        up /= np.linalg.norm(up)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        relative = world_points - camera_position
        depth = relative @ forward
        near = float(self.model.vis.map.znear * self.model.stat.extent)
        half_height = 0.5 * (
            float(cameras[0].frustum_top)
            - float(cameras[0].frustum_bottom)
        )
        half_width = half_height * self.width / self.height
        valid = depth > near
        pixels = np.full((world_points.shape[0], 2), np.nan)
        horizontal = (relative[valid] @ right) * near / depth[valid]
        vertical = (relative[valid] @ up) * near / depth[valid]
        pixels[valid, 0] = 0.5 * self.width * (
            1.0 + horizontal / half_width
        )
        pixels[valid, 1] = 0.5 * self.height * (
            1.0 - vertical / half_height
        )
        return pixels

    @staticmethod
    def _draw_fading_path(
        draw: ImageDraw.ImageDraw,
        points: np.ndarray,
        *,
        color: tuple[int, int, int],
        width: int,
        fade_exponent: float | None,
        constant_alpha: int = 235,
    ) -> None:
        if points.shape[0] < 2:
            return
        for index, (start, end) in enumerate(
            zip(points[:-1], points[1:], strict=True),
            start=1,
        ):
            if not np.all(np.isfinite((start, end))):
                continue
            if fade_exponent is None:
                alpha = constant_alpha
            else:
                age = index / max(points.shape[0] - 1, 1)
                alpha = round(
                    6 + 239 * (1.0 - (1.0 - age) ** fade_exponent)
                )
            draw.line(
                (
                    tuple(np.rint(start).astype(int)),
                    tuple(np.rint(end).astype(int)),
                ),
                fill=(*color, alpha),
                width=width,
            )

    def _overlay_eef_paths(
        self,
        frame: np.ndarray,
        command_history_relative: np.ndarray,
        state_history_relative: np.ndarray,
        elbow_history_world: np.ndarray | None,
    ) -> np.ndarray:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image, "RGBA")

        def sampled_pixels(history: np.ndarray) -> np.ndarray:
            indices = np.linspace(
                0,
                history.shape[0] - 1,
                min(history.shape[0], EEF_TRAIL_POINTS),
                dtype=int,
            )
            return self._project_relative_positions(history[indices])

        command_pixels = sampled_pixels(command_history_relative)
        state_pixels = sampled_pixels(
            state_history_relative[-STATE_TRAIL_SAMPLES:]
        )
        self._draw_fading_path(
            draw,
            command_pixels,
            color=(255, 52, 42),
            width=4,
            fade_exponent=None,
        )
        self._draw_fading_path(
            draw,
            state_pixels,
            color=(255, 184, 5),
            width=4,
            fade_exponent=2.0,
        )
        if elbow_history_world is not None:
            elbow_indices = np.linspace(
                0,
                elbow_history_world.shape[0] - 1,
                min(elbow_history_world.shape[0], ELBOW_TRAIL_POINTS),
                dtype=int,
            )
            elbow_pixels = self._project_world_positions(
                elbow_history_world[elbow_indices]
            )
            self._draw_fading_path(
                draw,
                elbow_pixels,
                color=(255, 224, 138),
                width=2,
                fade_exponent=None,
                constant_alpha=135,
            )
            if np.all(np.isfinite(elbow_pixels[-1])):
                x, y = np.rint(elbow_pixels[-1]).astype(int)
                draw.ellipse(
                    (x - 3, y - 3, x + 3, y + 3),
                    fill=(255, 237, 173, 185),
                )
        return np.asarray(image)

    def _add_connector(
        self,
        *,
        geom_type: mujoco.mjtGeom,
        width: float,
        start: np.ndarray,
        end: np.ndarray,
        rgba: np.ndarray,
    ) -> None:
        if self.renderer.scene.ngeom >= self.renderer.scene.maxgeom:
            return
        geom = self.renderer.scene.geoms[self.renderer.scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=geom_type,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).ravel(),
            rgba=np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            geom_type,
            width,
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        self.renderer.scene.ngeom += 1

    def _add_target(
        self,
        target_relative: np.ndarray,
        target_history_relative: np.ndarray,
        *,
        draw_trail: bool = True,
    ) -> None:
        position, rotation = self._target_world_pose(target_relative)
        colors = (
            np.array([0.95, 0.08, 0.08, 1.0]),
            np.array([0.08, 0.85, 0.20, 1.0]),
            np.array([0.10, 0.42, 1.00, 1.0]),
        )
        for axis, color in zip(rotation.T, colors, strict=True):
            self._add_connector(
                geom_type=mujoco.mjtGeom.mjGEOM_ARROW,
                width=TARGET_AXIS_WIDTH,
                start=position - TARGET_AXIS_BACK_LENGTH * axis,
                end=position + TARGET_AXIS_LENGTH * axis,
                rgba=color,
            )

        history = np.asarray(target_history_relative)
        if not draw_trail:
            return
        if history.shape[0] < 2:
            return
        indices = np.linspace(
            0,
            history.shape[0] - 1,
            min(history.shape[0], TARGET_TRAIL_POINTS),
            dtype=int,
        )
        world_points = np.asarray(
            [self._target_world(history[index]) for index in indices]
        )
        for start, end in zip(world_points[:-1], world_points[1:], strict=True):
            if np.linalg.norm(end - start) < 1.0e-6:
                continue
            self._add_connector(
                geom_type=mujoco.mjtGeom.mjGEOM_LINE,
                width=3.0,
                start=start,
                end=end,
                rgba=np.array([1.0, 0.18, 0.08, 0.72]),
            )

    def render_actual(
        self,
        q: np.ndarray,
        target_relative: np.ndarray,
        target_history_relative: np.ndarray,
        *,
        draw_target_trail: bool = True,
    ) -> np.ndarray:
        self._set_arm_visible(True)
        self.model.geom_matid[self._right_visual_geoms] = self._original_matid[
            self._right_visual_geoms
        ]
        self.model.geom_rgba[self._right_visual_geoms] = self._original_rgba[
            self._right_visual_geoms
        ]
        self._set_q(q)
        self.renderer.update_scene(
            self.data,
            camera=self.camera,
            scene_option=self.option,
        )
        self._add_target(
            target_relative,
            target_history_relative,
            draw_trail=draw_target_trail,
        )
        return self.renderer.render().copy()

    def render_command(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._set_arm_visible(True)
        self.model.geom_matid[self._right_visual_geoms] = -1
        self.model.geom_rgba[self._right_visual_geoms] = np.array(
            [0.05, 0.82, 0.95, 1.0]
        )
        self._set_q(q)
        self.renderer.update_scene(
            self.data,
            camera=self.camera,
            scene_option=self.option,
        )
        command = self.renderer.render().copy()
        difference = np.max(
            np.abs(command.astype(np.int16) - self._background.astype(np.int16)),
            axis=2,
        )
        mask = difference > 6
        return command, mask

    def composite(
        self,
        actual_q: np.ndarray,
        command_q: np.ndarray,
        target_relative: np.ndarray,
        command_history_relative: np.ndarray,
        state_history_relative: np.ndarray,
        elbow_history_world: np.ndarray | None = None,
    ) -> np.ndarray:
        actual = self.render_actual(
            actual_q,
            target_relative,
            command_history_relative,
            draw_target_trail=False,
        ).astype(np.float32)
        command, mask = self.render_command(command_q)
        alpha = (GHOST_ALPHA * mask.astype(np.float32))[..., None]
        output = actual * (1.0 - alpha) + command.astype(np.float32) * alpha
        return self._overlay_eef_paths(
            np.clip(output, 0, 255).astype(np.uint8),
            command_history_relative,
            state_history_relative,
            elbow_history_world,
        )

    def render_ideal(
        self,
        command_q: np.ndarray,
        target_relative: np.ndarray,
        target_history_relative: np.ndarray,
    ) -> np.ndarray:
        return self.render_actual(
            command_q,
            target_relative,
            target_history_relative,
        )

    def close(self) -> None:
        self.renderer.close()


class BimanualRenderer(ArmRenderer):
    """Render both arms for the bimanual catalog segment."""

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        left_visual_geoms: list[int] = []
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = (
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_id,
                )
                or ""
            )
            if (
                body_name.startswith("openarm_left")
                and int(self.model.geom_group[geom_id]) == 2
            ):
                left_visual_geoms.append(geom_id)
        self._right_visual_geoms = np.concatenate(
            (
                self._right_visual_geoms,
                np.asarray(left_visual_geoms, dtype=int),
            )
        )
        self._hide_non_right_geometry()
        self.camera.lookat[:] = [0.25, 0.0, 1.27]
        self.camera.distance = 0.95
        self.camera.azimuth = 180.0
        self._background = self._render_background()

    def render_ideal_bimanual(
        self,
        right_q: np.ndarray,
        left_q: np.ndarray,
        right_target: np.ndarray,
        left_target: np.ndarray,
        right_history: np.ndarray,
        left_history: np.ndarray,
    ) -> np.ndarray:
        self._set_arm_visible(True)
        self.model.geom_matid[self._right_visual_geoms] = self._original_matid[
            self._right_visual_geoms
        ]
        self.model.geom_rgba[self._right_visual_geoms] = self._original_rgba[
            self._right_visual_geoms
        ]
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
        self.renderer.update_scene(
            self.data,
            camera=self.camera,
            scene_option=self.option,
        )
        self._add_target(right_target, right_history)
        self._add_target(left_target, left_history)
        return self.renderer.render().copy()


class VideoWriter:
    """Stream RGB frames directly to ffmpeg."""

    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with status {return_code}")


def annotated_panel(
    frame: np.ndarray,
    title: str,
    *,
    time_s: float,
    position_error_m: float | None,
    max_position_error_m: float | None = None,
    ideal: bool = False,
    playback_rate: float = 1.0,
    show_elbow: bool = False,
    view_label: str | None = None,
    caption: str | None = None,
    j1_acceleration_history_rad_s2: np.ndarray | None = None,
    j1_total_samples: int | None = None,
    j1_acceleration_limit_rad_s2: float | None = None,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    header_bottom = 82 if caption is not None else 67
    draw.rectangle(
        (0, 0, image.width, header_bottom), fill=(8, 15, 28, 205)
    )
    draw.text((14, 7), title, fill="white", font=font(20))
    status_parts: list[tuple[str, tuple[int, int, int, int]]] = [
        (f"t={time_s:4.2f}s", (226, 232, 240, 255))
    ]
    if position_error_m is not None:
        status_parts.append(
            (
                f"  |  error={100.0 * position_error_m:4.1f} cm",
                (226, 232, 240, 255),
            )
        )
    if max_position_error_m is not None:
        status_parts.append(
            (
                f"  |  max error={100.0 * max_position_error_m:4.1f} cm",
                (255, 232, 153, 255),
            )
        )
    if view_label is not None:
        status_parts.append(
            (f"  |  {view_label}", (226, 232, 240, 255))
        )
    status_x = 14.0
    status_font = font(14)
    for text, color in status_parts:
        draw.text((status_x, 38), text, fill=color, font=status_font)
        status_x += draw.textlength(text, font=status_font)
    if caption is not None:
        draw.text(
            (14, 59),
            caption,
            fill=(166, 180, 199, 255),
            font=font(11),
        )

    if not ideal or not np.isclose(playback_rate, 1.0):
        speed_value = f"{playback_rate:.2f}".rstrip("0").rstrip(".")
        speed_text = f"{speed_value}x"
        speed_font = font(14)
        speed_box = draw.textbbox((0, 0), speed_text, font=speed_font)
        speed_width = speed_box[2] - speed_box[0]
        speed_top = 92 if caption is not None else 77
        draw.rounded_rectangle(
            (10, speed_top, 26 + speed_width, speed_top + 26),
            radius=4,
            fill=(8, 15, 28, 175),
        )
        draw.text(
            (18, speed_top + 3),
            speed_text,
            fill=(238, 242, 247, 255),
            font=speed_font,
        )

    if ideal:
        draw.rectangle(
            (10, image.height - 34, 415, image.height - 8),
            fill=(8, 15, 28, 175),
        )
        origin = (24, image.height - 20)
        draw.line(
            (origin, (37, image.height - 20)),
            fill=(242, 25, 25),
            width=3,
        )
        draw.line(
            (origin, (24, image.height - 31)),
            fill=(20, 220, 70),
            width=3,
        )
        draw.line(
            (origin, (15, image.height - 13)),
            fill=(25, 105, 255),
            width=3,
        )
        draw.text(
            (44, image.height - 30),
            "target frame + trail",
            fill="white",
            font=font(14),
        )
        draw.rectangle(
            (205, image.height - 26, 220, image.height - 14),
            fill=(225, 225, 225, 255),
        )
        draw.text(
            (227, image.height - 30),
            "solid raw IK solution",
            fill="white",
            font=font(14),
        )
    else:
        legend_font = font(12)
        draw.rectangle(
            (
                10,
                image.height - 34,
                570 if show_elbow else 455,
                image.height - 8,
            ),
            fill=(8, 15, 28, 175),
        )
        origin = (20, image.height - 20)
        draw.line(
            (origin, (30, image.height - 20)),
            fill=(242, 25, 25),
            width=3,
        )
        draw.line(
            (origin, (20, image.height - 29)),
            fill=(20, 220, 70),
            width=3,
        )
        draw.line(
            (origin, (13, image.height - 14)),
            fill=(25, 105, 255),
            width=3,
        )
        draw.text(
            (34, image.height - 28),
            "target",
            fill="white",
            font=legend_font,
        )
        draw.line(
            ((84, image.height - 20), (100, image.height - 20)),
            fill=(255, 52, 42, 255),
            width=4,
        )
        draw.text(
            (105, image.height - 28),
            "command",
            fill="white",
            font=legend_font,
        )
        draw.line(
            ((174, image.height - 20), (190, image.height - 20)),
            fill=(255, 184, 5, 255),
            width=4,
        )
        draw.text(
            (195, image.height - 28),
            "state",
            fill="white",
            font=legend_font,
        )
        draw.rectangle(
            (244, image.height - 26, 259, image.height - 14),
            fill=(13, 209, 242, 155),
        )
        draw.text(
            (265, image.height - 28),
            "IK ghost",
            fill="white",
            font=legend_font,
        )
        draw.rectangle(
            (333, image.height - 26, 348, image.height - 14),
            fill=(225, 225, 225, 255),
        )
        draw.text(
            (354, image.height - 28),
            "actual",
            fill="white",
            font=legend_font,
        )
        if show_elbow:
            draw.line(
                ((410, image.height - 20), (426, image.height - 20)),
                fill=(255, 224, 138, 255),
                width=2,
            )
            draw.ellipse(
                (422, image.height - 23, 428, image.height - 17),
                fill=(255, 237, 173, 255),
            )
            draw.text(
                (436, image.height - 28),
                "elbow",
                fill="white",
                font=legend_font,
            )

    if (
        j1_acceleration_history_rad_s2 is not None
        and j1_acceleration_history_rad_s2.size > 0
        and j1_total_samples is not None
        and j1_acceleration_limit_rad_s2 is not None
    ):
        chart_left = image.width - 320
        chart_top = 92 if caption is not None else 77
        chart_right = image.width - 10
        chart_bottom = chart_top + 94
        draw.rounded_rectangle(
            (chart_left, chart_top, chart_right, chart_bottom),
            radius=5,
            fill=(8, 15, 28, 205),
            outline=(91, 109, 132, 210),
            width=1,
        )
        actual_acceleration = float(j1_acceleration_history_rad_s2[-1])
        draw.text(
            (chart_left + 10, chart_top + 6),
            f"J1 actual acceleration {actual_acceleration:+6.1f} rad/s^2",
            fill=(238, 242, 247, 255),
            font=font(11),
        )

        plot_left = chart_left + 10
        plot_top = chart_top + 28
        plot_right = chart_right - 10
        plot_bottom = chart_bottom - 10
        limit = j1_acceleration_limit_rad_s2

        def acceleration_y(acceleration: float) -> int:
            fraction = np.clip(
                (acceleration + limit) / (2.0 * limit),
                0.0,
                1.0,
            )
            return round(plot_bottom - fraction * (plot_bottom - plot_top))

        zero_y = acceleration_y(0.0)
        for start_x in range(plot_left, plot_right, 10):
            draw.line(
                (
                    (start_x, zero_y),
                    (min(start_x + 5, plot_right), zero_y),
                ),
                fill=(160, 174, 193, 210),
                width=1,
            )
        scale_font = font(9)
        draw.text(
            (plot_left, plot_top - 1),
            f"+{limit:.0f}",
            fill=(166, 180, 199, 230),
            font=scale_font,
        )
        draw.text(
            (plot_left, plot_bottom - 10),
            f"-{limit:.0f}",
            fill=(166, 180, 199, 230),
            font=scale_font,
        )

        denominator = max(j1_total_samples - 1, 1)
        history_points = [
            (
                round(
                    plot_left
                    + sample_index / denominator * (plot_right - plot_left)
                ),
                acceleration_y(float(acceleration)),
            )
            for sample_index, acceleration in enumerate(
                j1_acceleration_history_rad_s2
            )
        ]
        if len(history_points) >= 2:
            draw.line(
                history_points,
                fill=(255, 224, 138, 255),
                width=3,
                joint="curve",
            )
        current_x, current_y = history_points[-1]
        draw.ellipse(
            (current_x - 3, current_y - 3, current_x + 3, current_y + 3),
            fill=(255, 237, 173, 255),
        )
    return np.asarray(image)


def comparison_video(spec: Comparison) -> Path:
    traces = [
        (
            load_trace(
                panel.suite or spec.suite,
                panel.profile,
                panel.scenario or spec.scenario,
            ),
            panel.label,
        )
        for panel in spec.panels
    ]
    max_position_errors = [
        float(
            np.max(
                np.linalg.norm(
                    trace["right_target_pose"][:, :3]
                    - trace["right_actual_pose"][:, :3],
                    axis=1,
                )
            )
        )
        for trace, _ in traces
    ]
    j1_acceleration_limit: float | None = None
    if spec.show_j1_acceleration:
        all_j1_acceleration = np.concatenate(
            [trace["right_actual_ddq"][:, 0] for trace, _ in traces]
        )
        j1_acceleration_limit = max(
            5.0,
            1.05 * float(np.max(np.abs(all_j1_acceleration))),
        )
    duration = min(float(trace["times"][-1]) for trace, _ in traces)
    frame_times = np.arange(
        0.0,
        duration + 1.0e-9,
        1.0 / (FPS * spec.slowdown),
    )
    output = OUTPUT / f"{spec.name}.mp4"
    columns = spec.columns or len(traces)
    if len(traces) % columns != 0:
        raise ValueError(
            f"{spec.name} has {len(traces)} panels, not divisible by {columns}"
        )
    rows = len(traces) // columns
    writer = VideoWriter(
        output,
        PANEL_WIDTH * columns,
        PANEL_HEIGHT * rows,
        FPS,
    )
    renderer = ArmRenderer(PANEL_WIDTH, PANEL_HEIGHT)
    try:
        views = ("current", "rear") if spec.replay_rear_view else ("current",)
        for view in views:
            if view == "current" and spec.camera_azimuth is not None:
                renderer.set_camera(
                    azimuth=spec.camera_azimuth,
                    lookat=spec.camera_lookat,
                    distance=spec.camera_distance,
                )
            else:
                renderer.set_camera_view(view)
            view_label = (
                "current view" if view == "current" else "right-rear view"
            )
            for time_s in frame_times:
                panels: list[np.ndarray] = []
                for (trace, label), max_position_error in zip(
                    traces,
                    max_position_errors,
                    strict=True,
                ):
                    index = int(np.argmin(np.abs(trace["times"] - time_s)))
                    frame = renderer.composite(
                        trace["right_actual_q"][index],
                        trace["right_command_q"][index],
                        trace["right_target_pose"][index],
                        trace["right_target_pose"][: index + 1],
                        trace["right_actual_pose"][: index + 1],
                        (
                            trace["right_actual_elbow"][: index + 1]
                            if spec.show_elbow
                            else None
                        ),
                    )
                    position_error = float(
                        np.linalg.norm(
                            trace["right_target_pose"][index, :3]
                            - trace["right_actual_pose"][index, :3]
                        )
                    )
                    panels.append(
                        annotated_panel(
                            frame,
                            label,
                            time_s=time_s,
                            position_error_m=position_error,
                            max_position_error_m=(
                                max_position_error
                                if spec.show_max_error
                                else None
                            ),
                            playback_rate=1.0 / spec.slowdown,
                            show_elbow=spec.show_elbow,
                            view_label=(
                                view_label if spec.replay_rear_view else None
                            ),
                            caption=spec.caption,
                            j1_acceleration_history_rad_s2=(
                                trace["right_actual_ddq"][: index + 1, 0]
                                if spec.show_j1_acceleration
                                else None
                            ),
                            j1_total_samples=(
                                trace["right_actual_ddq"].shape[0]
                                if spec.show_j1_acceleration
                                else None
                            ),
                            j1_acceleration_limit_rad_s2=(
                                j1_acceleration_limit
                            ),
                        )
                    )
                panel_rows = [
                    np.concatenate(panels[start : start + columns], axis=1)
                    for start in range(0, len(panels), columns)
                ]
                writer.write(np.concatenate(panel_rows, axis=0))
    finally:
        renderer.close()
        writer.close()
    return output


def ideal_reference_video() -> Path:
    definitions = [
        (
            "candidates",
            "reach_right_p0p00_v0p80",
            "Reach beyond workspace: center",
            False,
        ),
        (
            "candidates",
            "reach_right_p0p10_v0p40",
            "Reach beyond workspace: lateral +",
            False,
        ),
        (
            "candidates",
            "reach_right_m0p10_v0p40",
            "Reach beyond workspace: lateral -",
            False,
        ),
        (
            "video_trajectories",
            "retract_deep_p0p10_v0p80",
            "Fast diagonal retract: lateral +",
            False,
        ),
        (
            "video_trajectories",
            "retract_deep_m0p10_v0p80",
            "Fast diagonal retract: lateral -",
            False,
        ),
        (
            "candidates",
            "extended_circle_right_v0p80",
            "Extended vertical-lateral circle",
            False,
        ),
        (
            "candidates",
            "extended_up_right_v0p80",
            "Extended: up and return",
            False,
        ),
        (
            "candidates",
            "extended_down_right_v0p80",
            "Extended: down and return",
            False,
        ),
        (
            "candidates",
            "extended_left_right_v0p80",
            "Extended: left and return",
            False,
        ),
        (
            "candidates",
            "extended_right_right_v0p80",
            "Extended: right and return",
            False,
        ),
        (
            "candidates",
            "extended_wrist_right_w10p0",
            "Extended wrist roll",
            False,
        ),
        (
            "candidates",
            "normal_wrist_right_w10p0",
            "Normal-workspace wrist roll",
            False,
        ),
        (
            "chest",
            "normal_wrist_right_chest_w10p0",
            "Chest pose: wrist roll only",
            False,
        ),
        (
            "chest",
            "chest_outward_flip_right_forward_v1p20_w8p0",
            "Chest roll + forward translation",
            False,
        ),
        (
            "chest",
            "chest_outward_flip_right_lateral_v1p20_w8p0",
            "Chest roll + lateral translation",
            False,
        ),
        (
            "chest",
            "chest_outward_flip_right_down_v1p20_w8p0",
            "Chest roll + downward translation",
            False,
        ),
        (
            "chest",
            "chest_outward_flip_right_diagonal_v1p20_w8p0",
            "Chest roll + diagonal translation",
            False,
        ),
        (
            "candidates",
            "normal_right_v0p80",
            "Normal workspace",
            False,
        ),
        (
            "candidates",
            "normal_bimanual_v0p60",
            "Bimanual workspace",
            True,
        ),
    ]
    traces = [
        (
            load_trace(suite, "pr_full", scenario),
            label,
            bimanual,
        )
        for suite, scenario, label, bimanual in definitions
    ]
    traces.extend(
        (
            load_trace(
                "final_report_showcases",
                "pr_current_dataflow",
                scenario,
            ),
            label,
            False,
        )
        for scenario, label in (
            (
                "episode73_triphasic_98s_straight_position",
                "Recorded-derived chest wrist flip",
            ),
            (
                "recorded_ep75_right_retract30_straight_retract",
                "Recorded-derived fast retract",
            ),
        )
    )
    segment_duration = 2.5
    output = OUTPUT / "ideal_reference_trajectory_catalog.mp4"
    writer = VideoWriter(output, PANEL_WIDTH, PANEL_HEIGHT, FPS)
    renderer = ArmRenderer(PANEL_WIDTH, PANEL_HEIGHT)
    bimanual_renderer = BimanualRenderer(PANEL_WIDTH, PANEL_HEIGHT)
    try:
        segment_frames = round(segment_duration * FPS)
        for trace, label, bimanual in traces:
            playback_rate = float(
                (trace["times"][-1] - trace["times"][0])
                / segment_duration
            )
            for frame_index in range(segment_frames):
                progress = frame_index / max(segment_frames - 1, 1)
                index = round(progress * (len(trace["times"]) - 1))
                if bimanual:
                    frame = bimanual_renderer.render_ideal_bimanual(
                        trace["right_command_q"][index],
                        trace["left_command_q"][index],
                        trace["right_target_pose"][index],
                        trace["left_target_pose"][index],
                        trace["right_target_pose"][: index + 1],
                        trace["left_target_pose"][: index + 1],
                    )
                else:
                    frame = renderer.render_ideal(
                        trace["right_command_q"][index],
                        trace["right_target_pose"][index],
                        trace["right_target_pose"][: index + 1],
                    )
                writer.write(
                    annotated_panel(
                        frame,
                        label,
                        time_s=float(trace["times"][index]),
                        position_error_m=None,
                        ideal=True,
                        playback_rate=playback_rate,
                    )
                )
    finally:
        renderer.close()
        bimanual_renderer.close()
        writer.close()
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparisons = [
        Comparison(
            name="chest_full_vs_unbounded_error",
            suite="chest",
            scenario="chest_outward_flip_right_diagonal_v1p20_w8p0",
            panels=(
                Panel("pr_full", "PR: 6D error modulation ON"),
                Panel("no_frame_error", "PR: 6D error modulation OFF"),
            ),
            caption="Chest wrist rotation with simultaneous diagonal motion.",
        ),
        Comparison(
            name="reach_full_vs_no_singularity_limit",
            suite="video_trajectories",
            scenario="reach_deep_start_right_p0p00_v0p80",
            panels=(
                Panel("pr_full", "PR: singularity limit ON"),
                Panel("no_singularity", "PR: singularity limit OFF"),
            ),
            caption="Shoulder-height reach through the reachable boundary.",
            slowdown=2.0,
            camera_azimuth=90.0,
            camera_lookat=(0.52, 0.001, 1.18),
            camera_distance=1.05,
        ),
        Comparison(
            name="retract_ik_and_driver_vs_driver_only",
            suite="video_trajectories",
            scenario="retract_deep_p0p10_v0p80",
            panels=(
                Panel("pr_full", "PR: IK velocity cap ON"),
                Panel(
                    "driver_only_ablation",
                    "PR: IK cap OFF, driver cap ON",
                ),
            ),
            caption="Fast diagonal retract from a weak extended configuration.",
            show_elbow=True,
            show_max_error=True,
            replay_rear_view=True,
        ),
        Comparison(
            name="retract_nullspace_and_mainline_comparison",
            suite="video_trajectories",
            scenario="retract_deep_p0p10_v0p80",
            panels=(
                Panel("pr_full", "PR: nullspace home ON"),
                Panel("no_nullspace", "PR: nullspace home OFF"),
                Panel(
                    "upstream_style",
                    "Mainline baseline (IK cap OFF)",
                ),
                Panel(
                    "upstream_style_ik_velocity",
                    "Mainline + --velocity-limit",
                ),
            ),
            caption="Elbow branch behavior during fast diagonal retract.",
            show_elbow=True,
            show_max_error=True,
            replay_rear_view=True,
            columns=2,
        ),
        Comparison(
            name="retract_full_vs_no_frame_error",
            suite="video_trajectories",
            scenario="retract_deep_p0p10_v0p80",
            panels=(
                Panel("pr_full", "PR: 6D error modulation ON"),
                Panel("no_frame_error", "PR: 6D error modulation OFF"),
            ),
            caption="Frame-error modulation during fast diagonal retract.",
            show_elbow=True,
            show_max_error=True,
            replay_rear_view=True,
        ),
    ]
    for comparison in comparisons:
        path = comparison_video(comparison)
        print(f"Wrote {path}")
    print(f"Wrote {ideal_reference_video()}")


if __name__ == "__main__":
    main()
