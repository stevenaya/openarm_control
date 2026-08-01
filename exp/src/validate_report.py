#!/usr/bin/env python3
"""Validate final-report links, math delimiters, inventory, and parameters."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
from pathlib import Path

import yaml

from openarm_control.config import ARM_JOINT_VELOCITY_LIMITS_RAD_S
from openarm_control.ik_params import IKParams


REPO_ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
PUBLIC_DIRS = ("assets", "videos", "tables", "manifests")
DEFAULT_OMISSIONS = {"solver", "velocity_limits"}
UNSAFE_GITHUB_MATH = (r"\begin{bmatrix}", r"\begin{cases}")
IK_VELOCITY_LIMIT_SOURCE = (
    "openarm_control.config.ARM_JOINT_VELOCITY_LIMITS_RAD_S"
)


def _resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _markdown_targets(path: Path) -> set[Path]:
    targets: set[Path] = set()
    for match in MARKDOWN_LINK.finditer(path.read_text(encoding="utf-8")):
        raw = match.group(1).strip().split(" ", 1)[0].strip("<>")
        if raw.startswith(("http://", "https://", "#", "mailto:")):
            continue
        raw = raw.partition("#")[0].partition("?")[0]
        if not raw:
            continue
        targets.add((path.parent / raw).resolve())
    return targets


def _validate_math(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    if text.count("$") % 2:
        errors.append(f"{path.name}: unbalanced dollar math delimiters")
    if text.count("$$") % 2:
        errors.append(f"{path.name}: unbalanced $$ delimiters")
    if text.count(r"\[") != text.count(r"\]"):
        errors.append(f"{path.name}: unbalanced \\[ / \\] delimiters")
    if "\\\\[" in text or "\\\\]" in text:
        errors.append(f"{path.name}: double-escaped display-math delimiter")
    if r"\operatorname" in text:
        errors.append(f"{path.name}: GitHub math does not allow \\operatorname")
    for expression in UNSAFE_GITHUB_MATH:
        if expression in text:
            errors.append(
                f"{path.name}: GitHub may consume row separators in {expression}"
            )
    if "<video" in text:
        errors.append(f"{path.name}: GitHub does not render repository video tags")
    return errors


def _manifest_targets(report_dir: Path) -> set[Path]:
    manifest_path = report_dir / "manifests" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        (manifest_path.parent / suite["metadata"]).resolve()
        for suite in manifest["suites"].values()
    }


def _validate_parameters(report_dir: Path) -> list[str]:
    manifest_path = report_dir / "manifests" / "manifest.json"
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))[
        "current_deployment_parameters"
    ]
    defaults = dataclasses.asdict(IKParams())
    errors: list[str] = []
    for key, expected in recorded.items():
        if key in DEFAULT_OMISSIONS:
            continue
        actual = defaults.get(key)
        if actual != expected:
            errors.append(
                f"IKParams.{key}={actual!r}, report manifest records {expected!r}"
            )
    return errors


def _validate_driver_config(report_dir: Path) -> list[str]:
    manifest_path = report_dir / "manifests" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest["driver_config"]
    path = _resolve_recorded_path(recorded["path"])
    errors: list[str] = []
    if not path.is_file():
        return [f"driver config does not exist: {path}"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != recorded["sha256"]:
        errors.append("driver config hash differs from report manifest")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = payload.get("joint_velocity_limits") if isinstance(payload, dict) else None
    if not isinstance(values, list) or len(values) < 7:
        errors.append("driver config must define at least seven joint velocity limits")
    else:
        actual = [float(value) for value in values[:7]]
        expected = [
            float(value)
            for value in recorded["arm_joint_velocity_caps_rad_s"]
        ]
        if actual != expected:
            errors.append(
                f"driver config arm caps {actual!r} differ from manifest {expected!r}"
            )
    return errors


def _validate_velocity_envelopes(report_dir: Path) -> list[str]:
    manifest_path = report_dir / "manifests" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("velocity_envelopes")
    if not isinstance(recorded, dict):
        return ["top manifest does not define velocity_envelopes"]

    ik = recorded.get("ik")
    driver = recorded.get("driver")
    if not isinstance(ik, dict) or not isinstance(driver, dict):
        return ["velocity_envelopes must define independent ik and driver entries"]

    errors: list[str] = []
    expected_ik_caps = [
        float(value) for value in ARM_JOINT_VELOCITY_LIMITS_RAD_S
    ]
    recorded_ik_caps = ik.get("arm_joint_caps_rad_s")
    if ik.get("enabled") is not True:
        errors.append("current deployment must enable the IK velocity envelope")
    if ik.get("deployment_flag") != "--limit-velocity":
        errors.append("current deployment IK envelope must record --limit-velocity")
    if ik.get("config_override") is not None:
        errors.append("current deployment IK envelope must use the built-in caps")
    if ik.get("limit_style") != "recoverable":
        errors.append("current deployment IK limit_style must be 'recoverable'")
    if ik.get("source") != IK_VELOCITY_LIMIT_SOURCE:
        errors.append("IK velocity-envelope source differs from the expected symbol")
    if recorded_ik_caps != expected_ik_caps:
        errors.append(
            f"IK caps {recorded_ik_caps!r} differ from controller caps "
            f"{expected_ik_caps!r}"
        )

    driver_config = manifest["driver_config"]
    expected_driver_caps = driver_config["arm_joint_velocity_caps_rad_s"]
    if driver.get("enabled") is not True:
        errors.append("current deployment must enable the driver velocity envelope")
    if driver.get("source") != driver_config["path"]:
        errors.append("driver velocity-envelope source differs from driver_config")
    if driver.get("config_sha256") != driver_config["sha256"]:
        errors.append("driver velocity-envelope hash differs from driver_config")
    if driver.get("arm_joint_caps_rad_s") != expected_driver_caps:
        errors.append("driver velocity-envelope caps differ from driver_config")

    deployment_profiles = 0
    for suite in manifest["suites"].values():
        metadata_path = manifest_path.parent / suite["metadata"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for profile in metadata.get("profiles", []):
            if profile.get("name") != "current_deployment":
                continue
            deployment_profiles += 1
            if profile.get("velocity_caps") != expected_ik_caps:
                errors.append(
                    f"{metadata_path.parent.name}: current_deployment IK caps "
                    "differ from the top manifest"
                )
            if profile.get("limit_style") != ik.get("limit_style"):
                errors.append(
                    f"{metadata_path.parent.name}: current_deployment limit style "
                    "differs from the top manifest"
                )
            if profile.get("driver_velocity_caps") != expected_driver_caps:
                errors.append(
                    f"{metadata_path.parent.name}: current_deployment driver caps "
                    "differ from the top manifest"
                )
    if deployment_profiles == 0:
        errors.append("suite manifests contain no current_deployment profile")
    return errors


def _validate_manifests(report_dir: Path) -> list[str]:
    manifest_path = report_dir / "manifests" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_params = manifest["current_deployment_parameters"]
    run_total = 0
    errors: list[str] = []
    for name, suite in manifest["suites"].items():
        run_total += suite["controller_trajectory_runs"]
        metadata_path = manifest_path.parent / suite["metadata"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        suite_params = metadata.get("current_deployment_parameters")
        if suite_params is not None and suite_params != expected_params:
            errors.append(f"{name}: experiment parameters differ from top manifest")
        matrix = metadata.get("matrix")
        if matrix is not None:
            recorded = matrix.get("controller_trajectory_runs")
            if recorded != suite["controller_trajectory_runs"]:
                errors.append(
                    f"{name}: suite run count {suite['controller_trajectory_runs']} "
                    f"!= metadata {recorded}"
                )
        if suite["controller_trajectory_runs"]:
            suite_driver = metadata.get("driver_config")
            if suite_driver != manifest["driver_config"]:
                errors.append(f"{name}: driver config differs from top manifest")
    expected_total = manifest["dynamic_controller_trajectory_runs"]
    if run_total != expected_total:
        errors.append(f"suite run sum {run_total} != top-level total {expected_total}")
    return errors


def _validate_frozen_inputs(report_dir: Path) -> list[str]:
    manifest_path = report_dir / "manifests" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for name, recorded in manifest["frozen_sources"].items():
        path = _resolve_recorded_path(recorded["path"])
        if not path.is_file():
            errors.append(f"{name}: frozen input does not exist: {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != recorded["sha256"]:
            errors.append(f"{name}: frozen input hash differs from report manifest")
    return errors


def _validate_reproduction_environment(report_dir: Path) -> list[str]:
    manifest_path = report_dir / "manifests" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest["reproduction_environment"]
    lock_path = _resolve_recorded_path(recorded["lock_path"])
    errors: list[str] = []
    if not lock_path.is_file():
        errors.append(f"reproduction lock does not exist: {lock_path}")
    elif hashlib.sha256(lock_path.read_bytes()).hexdigest() != recorded["lock_sha256"]:
        errors.append("reproduction lock hash differs from report manifest")

    python_version_path = lock_path.parent / ".python-version"
    if not python_version_path.is_file():
        errors.append(f"Python version file does not exist: {python_version_path}")
    elif python_version_path.read_text(encoding="utf-8").strip() != recorded["python"]:
        errors.append("Python version differs from report manifest")
    return errors


def _write_checksums(report_dir: Path) -> None:
    checksum_path = report_dir / "tables" / "asset_checksums.sha256"
    paths: list[Path] = []
    for directory in PUBLIC_DIRS:
        paths.extend(
            path
            for path in (report_dir / directory).rglob("*")
            if path.is_file() and path != checksum_path
        )
    with checksum_path.open("w", encoding="utf-8") as output:
        for path in sorted(paths):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            output.write(f"{digest}  {path.relative_to(report_dir)}\n")


def validate(report_dir: Path, *, write_checksums: bool) -> list[str]:
    if write_checksums:
        _write_checksums(report_dir)

    markdown = sorted(report_dir.glob("*.md"))
    errors: list[str] = []
    referenced: set[Path] = set()
    for path in markdown:
        referenced.update(_markdown_targets(path))
        errors.extend(_validate_math(path))

    referenced.update(_manifest_targets(report_dir))
    for target in sorted(referenced):
        if not target.exists():
            errors.append(f"missing link target: {target}")

    public_files: set[Path] = set()
    for directory in PUBLIC_DIRS:
        public_files.update(
            path.resolve()
            for path in (report_dir / directory).rglob("*")
            if path.is_file()
        )
    unreferenced = sorted(public_files - referenced)
    errors.extend(
        f"unreferenced public asset: {path.relative_to(report_dir)}"
        for path in unreferenced
    )
    errors.extend(_validate_parameters(report_dir))
    errors.extend(_validate_driver_config(report_dir))
    errors.extend(_validate_velocity_envelopes(report_dir))
    errors.extend(_validate_manifests(report_dir))
    errors.extend(_validate_frozen_inputs(report_dir))
    errors.extend(_validate_reproduction_environment(report_dir))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--write-checksums", action="store_true")
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    errors = validate(report_dir, write_checksums=args.write_checksums)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated {report_dir}")


if __name__ == "__main__":
    main()
