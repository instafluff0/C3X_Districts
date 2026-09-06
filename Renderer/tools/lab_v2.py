#!/usr/bin/env python3
"""Inspect, validate, and print prompts for the Renderer Lab v2 campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any


RENDERER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = RENDERER_ROOT / "terrain_lab/v2/campaigns/Q1/campaign.json"
CAMPAIGN_SCHEMA = "c3x.renderer_lab_v2_campaign.v0"
PACKAGE_SCHEMA = "c3x.renderer_lab_v2_work_package.v0"
STATUS_SCHEMA = "c3x.renderer_lab_v2_track_status.v0"
REFERENCE_SCHEMA = "c3x.renderer_lab_v2_reference_catalog.v0"
TRACK_STATES = {
    "blocked_by_dependency",
    "ready",
    "active",
    "technical_pass",
    "review_pending",
    "accepted",
    "blocked",
}
REQUIRED_PACKAGE_FIELDS = {
    "id",
    "campaign",
    "title",
    "status_file",
    "dependencies",
    "owns_paths",
    "reads_paths",
    "forbidden_paths",
    "references",
    "scope",
    "acceptance",
    "prompt",
}
PROTECTED_OWNERSHIP_PREFIXES = (
    "Renderer/native/",
    "Renderer/handoffs/",
    "Renderer/project_status.json",
    "Renderer/ROADMAP.md",
    "C3X.h",
    "injected_code.c",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _portable_path(value: Any, *, directory: bool = False) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        return None
    if directory and not value.endswith("/"):
        return None
    return value


def _overlaps(left: str, right: str) -> bool:
    return left.startswith(right) or right.startswith(left)


def load_campaign(campaign_path: Path = DEFAULT_CAMPAIGN) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    campaign = _load(campaign_path)
    packages: dict[str, dict[str, Any]] = {}
    statuses: dict[str, dict[str, Any]] = {}
    for entry in campaign.get("tracks", []):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        track_id = entry["id"]
        manifest_path = campaign_path.parent / str(entry.get("manifest", ""))
        status_path = campaign_path.parent / str(entry.get("status", ""))
        if manifest_path.is_file():
            package = _load(manifest_path)
            package["_path"] = manifest_path
            packages[track_id] = package
        if status_path.is_file():
            status = _load(status_path)
            status["_path"] = status_path
            statuses[track_id] = status
    return campaign, packages, statuses


def validate_campaign(campaign_path: Path = DEFAULT_CAMPAIGN) -> list[str]:
    errors: list[str] = []
    try:
        campaign, packages, statuses = load_campaign(campaign_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"Could not load Lab v2 campaign: {exc}"]

    if campaign.get("schema") != CAMPAIGN_SCHEMA:
        errors.append("Unsupported or missing Lab v2 campaign schema.")
    campaign_id = campaign.get("id")
    entries = campaign.get("tracks")
    if not isinstance(entries, list) or not entries:
        errors.append("Campaign must contain at least one track.")
        entries = []
    entry_ids = [entry.get("id") for entry in entries if isinstance(entry, dict)]
    if len(entry_ids) != len(set(entry_ids)):
        errors.append("Campaign track IDs must be unique.")

    reference_path = campaign_path.parent / str(campaign.get("reference_catalog", ""))
    reference_ids: set[str] = set()
    try:
        references = _load(reference_path)
        if references.get("schema") != REFERENCE_SCHEMA:
            errors.append("Unsupported Lab v2 reference catalog schema.")
        for reference in references.get("references", []):
            reference_id = reference.get("id") if isinstance(reference, dict) else None
            if not isinstance(reference_id, str) or not reference_id:
                errors.append("Every reference must have a nonempty ID.")
            elif reference_id in reference_ids:
                errors.append(f"Duplicate reference ID: {reference_id}")
            else:
                reference_ids.add(reference_id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"Could not load Lab v2 reference catalog: {exc}")

    ownership: list[tuple[str, str]] = []
    for track_id in entry_ids:
        if track_id not in packages:
            errors.append(f"Missing work-package manifest for {track_id}.")
            continue
        package = packages[track_id]
        package_path = package["_path"]
        if package.get("schema") != PACKAGE_SCHEMA:
            errors.append(f"Unsupported work-package schema for {track_id}.")
        missing = sorted(REQUIRED_PACKAGE_FIELDS - set(package))
        if missing:
            errors.append(f"Work package {track_id} is missing fields: {', '.join(missing)}")
        if package.get("id") != track_id or package.get("campaign") != campaign_id:
            errors.append(f"Work package identity mismatch for {track_id}.")
        for field in ("dependencies", "owns_paths", "reads_paths", "forbidden_paths", "references", "scope", "acceptance"):
            if not isinstance(package.get(field), list):
                errors.append(f"Work package {track_id}.{field} must be a list.")
        for owned in package.get("owns_paths", []):
            normalized = _portable_path(owned, directory=True)
            if normalized is None:
                errors.append(f"Work package {track_id} has unsafe owned path: {owned!r}")
                continue
            if any(_overlaps(normalized, protected) for protected in PROTECTED_OWNERSHIP_PREFIXES):
                errors.append(f"Work package {track_id} claims protected path: {normalized}")
            for other_track, other_path in ownership:
                if _overlaps(normalized, other_path):
                    errors.append(
                        f"Owned paths overlap: {track_id}:{normalized} and {other_track}:{other_path}"
                    )
            ownership.append((track_id, normalized))
        for field in ("reads_paths", "forbidden_paths"):
            for value in package.get(field, []):
                if _portable_path(value) is None:
                    errors.append(f"Work package {track_id} has unsafe {field} entry: {value!r}")
        for dependency in package.get("dependencies", []):
            if dependency not in entry_ids:
                errors.append(f"Work package {track_id} has unknown dependency {dependency!r}.")
        for reference_id in package.get("references", []):
            if reference_id not in reference_ids:
                errors.append(f"Work package {track_id} has unknown reference {reference_id!r}.")
        prompt_path = package_path.parent / str(package.get("prompt", ""))
        if not prompt_path.is_file():
            errors.append(f"Work package {track_id} prompt is missing: {prompt_path}")
        status_path = package_path.parent / str(package.get("status_file", ""))
        entry_status_path = statuses.get(track_id, {}).get("_path")
        if entry_status_path is None or status_path.resolve() != Path(entry_status_path).resolve():
            errors.append(f"Work package {track_id} status path does not match the campaign entry.")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(track_id: str) -> None:
        if track_id in visited or track_id not in packages:
            return
        if track_id in visiting:
            errors.append(f"Lab v2 dependency cycle includes {track_id}.")
            return
        visiting.add(track_id)
        for dependency in packages[track_id].get("dependencies", []):
            visit(dependency)
        visiting.remove(track_id)
        visited.add(track_id)

    for track_id in entry_ids:
        visit(track_id)

    for track_id in entry_ids:
        status = statuses.get(track_id)
        if status is None:
            errors.append(f"Missing status file for {track_id}.")
            continue
        if status.get("schema") != STATUS_SCHEMA:
            errors.append(f"Unsupported track-status schema for {track_id}.")
        if status.get("id") != track_id or status.get("campaign") != campaign_id:
            errors.append(f"Track-status identity mismatch for {track_id}.")
        state = status.get("state")
        if state not in TRACK_STATES:
            errors.append(f"Invalid Lab v2 state for {track_id}: {state!r}")
        if state == "accepted" and not (
            status.get("candidate") and status.get("evidence") and status.get("approval")
        ):
            errors.append(f"Accepted track {track_id} must record candidate, evidence, and approval.")
        dependencies = packages.get(track_id, {}).get("dependencies", [])
        blockers = status.get("blockers")
        if state == "blocked_by_dependency" and set(blockers or []) != set(dependencies):
            errors.append(f"Blocked track {track_id} must name all package dependencies as blockers.")

    common_prompt = campaign_path.parent / str(campaign.get("common_prompt", ""))
    if not common_prompt.is_file():
        errors.append(f"Campaign common prompt is missing: {common_prompt}")
    baseline = campaign.get("baseline", {}).get("handoff")
    if _portable_path(baseline) is None or not (RENDERER_ROOT.parent / str(baseline)).is_file():
        errors.append(f"Campaign baseline handoff is missing or unsafe: {baseline!r}")
    return errors


def render_prompt(track_id: str, campaign_path: Path = DEFAULT_CAMPAIGN) -> str:
    errors = validate_campaign(campaign_path)
    if errors:
        raise ValueError("\n".join(errors))
    campaign, packages, _statuses = load_campaign(campaign_path)
    if track_id not in packages:
        raise ValueError(f"Unknown Lab v2 track: {track_id}")
    package = packages[track_id]
    common_path = campaign_path.parent / campaign["common_prompt"]
    role_path = package["_path"].parent / package["prompt"]
    header = (
        f"TRACK_ID: {track_id}\n"
        f"WORK_PACKAGE: {package['_path'].relative_to(RENDERER_ROOT.parent).as_posix()}\n\n"
    )
    common = common_path.read_text(encoding="utf-8").replace("<TRACK_ID>", track_id).rstrip()
    role = role_path.read_text(encoding="utf-8").rstrip()
    return header + common + "\n\n" + role + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("status")
    prompt_parser = subparsers.add_parser("prompt")
    prompt_parser.add_argument("track")
    args = parser.parse_args()

    errors = validate_campaign(args.campaign)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    if args.command == "validate":
        print("Renderer Lab v2 campaign is valid.")
        return 0
    campaign, packages, statuses = load_campaign(args.campaign)
    if args.command == "status":
        print(f"{campaign['id']}: {campaign['title']}")
        for entry in campaign["tracks"]:
            track_id = entry["id"]
            print(f"{track_id:17} {statuses[track_id]['state']:23} {packages[track_id]['title']}")
        return 0
    try:
        print(render_prompt(args.track, args.campaign), end="")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

