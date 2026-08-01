#!/usr/bin/env python3
"""Render the public videos from the final-report experiment traces."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import render_videos as video


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / "results" / "final_report_20260801"
DEFAULT_OUTPUT = HERE.parent / "videos"
DEFAULT_PREVIEW_OUTPUT = HERE.parent / "assets" / "video_previews"


def animated_previews(video_dir: Path, output: Path) -> tuple[Path, ...]:
    """Generate compact GitHub-renderable previews for the public MP4 files."""
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for source in sorted(video_dir.glob("*.mp4")):
        catalog = source.name == "ideal_reference_trajectory_catalog.mp4"
        fps = 4 if catalog else 6
        width = 600 if catalog else 720
        colors = 96 if catalog else 128
        target = output / f"{source.stem}.gif"
        video_filter = (
            f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];"
            f"[s0]palettegen=max_colors={colors}[p];"
            "[s1][p]paletteuse=dither=bayer:bayer_scale=5"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                video_filter,
                "-loop",
                "0",
                str(target),
            ],
            check=True,
        )
        paths.append(target)
    return tuple(paths)


def ideal_reference_video(output: Path) -> Path:
    """Render all 21 public reference patterns from PR-default traces."""
    family_counts = {
        "reach": ("reach family", 5, 11),
        "retract": ("retract family", 6, 6),
        "extended_translation": ("extended translation", 3, 5),
        "extended_axis": ("extended axis", 8, 8),
        "extended_wrist": ("extended wrist", 3, 5),
        "normal_wrist": ("normal wrist", 3, 8),
        "chest_wrist_outward": ("chest wrist + translation", 9, 13),
        "normal_workspace": ("normal workspace", 5, 7),
        "recorded_near_chest": ("recorded near-chest", 0, 1),
        "recorded_retract": ("recorded retract variants", 0, 2),
    }
    definitions = (
        ("refresh_screening", "reach_right_p0p00_v0p80", "Arm extension beyond reach: center", False, "reach"),
        ("refresh_screening", "reach_right_p0p10_v0p40", "Arm extension beyond reach: lateral +", False, "reach"),
        ("refresh_screening", "reach_right_m0p10_v0p40", "Arm extension beyond reach: lateral -", False, "reach"),
        ("refresh_screening", "retract_diag_p0p10_v0p80", "Fast diagonal retract: lateral +", False, "retract"),
        ("refresh_screening", "retract_diag_m0p10_v0p80", "Fast diagonal retract: lateral -", False, "retract"),
        ("refresh_screening", "extended_circle_right_v0p80", "Extended-arm circle", False, "extended_translation"),
        ("refresh_screening", "extended_up_right_v0p80", "Extended: up and return", False, "extended_axis"),
        ("refresh_screening", "extended_down_right_v0p80", "Extended: down and return", False, "extended_axis"),
        ("refresh_screening", "extended_left_right_v0p80", "Extended: left and return", False, "extended_axis"),
        ("refresh_screening", "extended_right_right_v0p80", "Extended: right and return", False, "extended_axis"),
        ("refresh_screening", "extended_wrist_right_w10p0", "Extended-arm wrist roll", False, "extended_wrist"),
        ("refresh_screening", "normal_wrist_right_w10p0", "Normal-workspace wrist roll", False, "normal_wrist"),
        ("refresh_chest", "normal_wrist_right_chest_w10p0", "Near-chest wrist roll only", False, "normal_wrist"),
        ("refresh_chest", "chest_outward_flip_right_forward_v1p20_w8p0", "Near-chest roll + forward translation", False, "chest_wrist_outward"),
        ("refresh_chest", "chest_outward_flip_right_lateral_v1p20_w8p0", "Near-chest roll + lateral translation", False, "chest_wrist_outward"),
        ("refresh_chest", "chest_outward_flip_right_down_v1p20_w8p0", "Near-chest roll + downward translation", False, "chest_wrist_outward"),
        ("refresh_chest", "chest_outward_flip_right_diagonal_v1p20_w8p0", "Near-chest roll + diagonal translation", False, "chest_wrist_outward"),
        ("refresh_screening", "normal_right_v0p80", "Normal-workspace motion", False, "normal_workspace"),
        ("refresh_screening", "normal_bimanual_v0p60", "Bimanual workspace motion", True, "normal_workspace"),
        ("refresh_frozen", "near_chest_fast_wrist_roll", "Recorded near-chest fast wrist-roll benchmark", False, "recorded_near_chest"),
        ("refresh_frozen", "fast_retract_elbow_branch", "Recorded fast-retract elbow-posture benchmark", False, "recorded_retract"),
    )
    traces = [
        (
            video.load_trace(suite, "current_deployment", scenario),
            label,
            bimanual,
            family_counts[family],
        )
        for suite, scenario, label, bimanual, family in definitions
    ]
    segment_duration = 2.5
    path = output / "ideal_reference_trajectory_catalog.mp4"
    writer = video.VideoWriter(path, video.PANEL_WIDTH, video.PANEL_HEIGHT, video.FPS)
    renderer = video.ArmRenderer(video.PANEL_WIDTH, video.PANEL_HEIGHT)
    bimanual_renderer = video.BimanualRenderer(video.PANEL_WIDTH, video.PANEL_HEIGHT)
    try:
        segment_frames = round(segment_duration * video.FPS)
        for trace, label, bimanual, family_count in traces:
            family_label, screening_count, all_unique_count = family_count
            caption = (
                f"{family_label} | screening-42: {screening_count} "
                f"| all-70: {all_unique_count}"
            )
            playback_rate = float((trace["times"][-1] - trace["times"][0]) / segment_duration)
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
                    video.annotated_panel(
                        frame,
                        label,
                        time_s=float(trace["times"][index]),
                        position_error_m=None,
                        ideal=True,
                        playback_rate=playback_rate,
                        caption=caption,
                    )
                )
    finally:
        renderer.close()
        bimanual_renderer.close()
        writer.close()
        for trace, _, _, _ in traces:
            trace.close()
    return path


def comparisons() -> tuple[video.Comparison, ...]:
    return (
        video.Comparison(
            name="near_chest_fast_wrist_roll_controller_comparison",
            suite="refresh_frozen",
            scenario="near_chest_fast_wrist_roll",
            panels=(
                video.Panel("current_deployment", "PR default"),
                video.Panel("strict_mainline", "Mainline baseline"),
                video.Panel("no_frame_error_bound", "PR w/o 6D error bound"),
            ),
            caption="Recorded fast wrist roll | arm_origin | identical driver velocity caps",
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            columns=3,
        ),
        video.Comparison(
            name="fast_retract_frame_error_bound_comparison",
            suite="refresh_fast_retract_error_bound",
            scenario="fast_retract_elbow_branch_2x",
            panels=(
                video.Panel(
                    "current_deployment",
                    "PR default",
                ),
                video.Panel(
                    "no_frame_error_bound",
                    "PR w/o 6D error bound",
                ),
            ),
            caption="Recorded fast-retract command path at 2x speed | arm_origin | only 6D error modulation changes",
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            columns=2,
        ),
        video.Comparison(
            name="fast_retract_controller_comparison",
            suite="refresh_frozen",
            scenario="fast_retract_elbow_branch",
            panels=(
                video.Panel("current_deployment", "PR default"),
                video.Panel("strict_mainline", "Mainline baseline"),
                video.Panel("no_branch_regulation", "PR w/o posture regulation"),
            ),
            caption="Recorded fast retract | arm_origin | identical driver velocity caps",
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            replay_rear_view=True,
            columns=3,
        ),
        video.Comparison(
            name="fast_retract_posture_regulation_comparison",
            suite="refresh_frozen",
            scenario="fast_retract_elbow_branch",
            panels=(
                video.Panel("current_deployment", "PR default"),
                video.Panel("no_branch_regulation", "PR w/o posture regulation"),
                video.Panel(
                    "full_home_replacement_0p01",
                    "PR: full-home posture 0.01",
                ),
                video.Panel(
                    "full_home_replacement_0p03",
                    "PR: full-home posture 0.03",
                ),
            ),
            caption="Recorded fast retract | arm_origin | only the secondary posture task changes",
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            replay_rear_view=True,
            columns=2,
        ),
        video.Comparison(
            name="fast_retract_nullspace_parameter_comparison",
            suite="refresh_nullspace",
            scenario="fast_retract_elbow_branch",
            panels=(
                video.Panel("current_deployment", "PR default"),
                video.Panel("null_c5_r1p6_v1", "PR: cost 5 / return 1.6 / max 1.0"),
                video.Panel(
                    "null_c8p5_r0p8_v0p6",
                    "PR: cost 8.5 / return 0.8 / max 0.6",
                ),
                video.Panel(
                    "null_c12_r1p6_v1",
                    "PR: cost 12 / return 1.6 / max 1.0",
                ),
            ),
            caption="Recorded fast retract | arm_origin | posture cost / return rate / maximum return speed",
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            replay_rear_view=True,
            columns=2,
        ),
        video.Comparison(
            name="straight_reach_singularity_limit_comparison",
            suite="refresh_singularity_video",
            scenario="reach_deep_start_right_p0p00_v0p80",
            panels=(
                video.Panel("current_deployment", "PR default"),
                video.Panel("no_singularity_limit", "PR w/o singularity limit"),
            ),
            caption="Shoulder-height extension | x=0.310 to 0.710 m in arm_origin",
            slowdown=2.0,
            show_max_error=True,
            columns=2,
            camera_azimuth=90.0,
            camera_lookat=(0.52, 0.001, 1.30),
            camera_distance=1.0,
            show_j1_acceleration=True,
        ),
        video.Comparison(
            name="fast_retract_ik_velocity_limit_comparison",
            suite="refresh_driver",
            scenario="retract_diag_p0p10_v0p80",
            panels=(
                video.Panel("current_deployment", "PR default"),
                video.Panel(
                    "driver_only_velocity",
                    "PR w/o IK velocity limits",
                ),
            ),
            caption="Synthetic retract | arm_origin | same downstream driver velocity limits",
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            replay_rear_view=True,
            columns=2,
        ),
        video.Comparison(
            name="near_chest_roll_translation_error_bound_comparison",
            suite="refresh_chest",
            scenario="chest_outward_flip_right_diagonal_v1p20_w8p0",
            panels=(
                video.Panel("current_deployment", "PR default"),
                video.Panel("no_frame_error_bound", "PR w/o 6D error bound"),
            ),
            caption="Synthetic roll + diagonal translation | arm_origin",
            slowdown=2.0,
            show_elbow=True,
            show_max_error=True,
            columns=2,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--skip-catalog",
        action="store_true",
        help="Render comparison videos without regenerating the 21-mode catalog.",
    )
    output_mode.add_argument(
        "--catalog-only",
        action="store_true",
        help="Regenerate only the 21-mode reference catalog.",
    )
    output_mode.add_argument(
        "--previews-only",
        action="store_true",
        help="Regenerate animated previews from existing MP4 files.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=DEFAULT_PREVIEW_OUTPUT,
        help="Output directory for GitHub-renderable GIF previews.",
    )
    args = parser.parse_args()
    root = args.results.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    video.OUTPUT = output
    video.SUITE_ROOTS.update(
        {
            "refresh_screening": root / "screening",
            "refresh_driver": root / "driver",
            "refresh_chest": root / "chest",
            "refresh_frozen": root / "frozen",
            "refresh_fast_retract_error_bound": (
                root / "fast_retract_error_bound"
            ),
            "refresh_singularity_video": root / "singularity_video",
            "refresh_nullspace": root / "nullspace_sweep",
        }
    )
    if not args.previews_only:
        if not args.catalog_only:
            for spec in comparisons():
                path = video.comparison_video(spec)
                print(f"Wrote {path}")
        if not args.skip_catalog:
            print(f"Wrote {ideal_reference_video(output)}")
    for path in animated_previews(output, args.preview_dir.resolve()):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
