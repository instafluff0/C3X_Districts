#!/usr/bin/env python3
"""Compare stable logical assets across two source-independent C3X packs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


SCHEMA = "c3x.pack_equivalence.v0"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_file(root: Path, value: str) -> Path | None:
    if not value or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return None
    candidate = root / Path(value.replace("\\", "/"))
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Pack reference escapes its root: {value}") from exc
    return candidate if candidate.is_file() else None


def _collect_files(root: Path, value: Any, files: dict[str, str], visiting: set[str]) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _collect_files(root, nested, files, visiting)
        return
    if isinstance(value, list):
        for nested in value:
            _collect_files(root, nested, files, visiting)
        return
    if not isinstance(value, str):
        return
    path = _safe_file(root, value)
    if path is None:
        return
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    if relative in files:
        return
    data = path.read_bytes()
    files[relative] = _sha256(data)
    if path.suffix.lower() == ".json" and relative not in visiting:
        visiting.add(relative)
        try:
            _collect_files(root, json.loads(data.decode("utf-8")), files, visiting)
        finally:
            visiting.remove(relative)


def _load_pack(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read pack manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema") != "c3x.asset_pack.v0" or not isinstance(manifest.get("assets"), dict):
        raise ValueError(f"Unsupported or incomplete C3X pack manifest: {manifest_path}")
    fingerprints: dict[str, dict[str, Any]] = {}
    for logical_id, record in sorted(manifest["assets"].items()):
        if not isinstance(logical_id, str) or not logical_id or not isinstance(record, dict):
            raise ValueError(f"Invalid logical asset record in {manifest_path}")
        files: dict[str, str] = {}
        _collect_files(root, record, files, set())
        fingerprint_input = {"record": record, "files": files}
        fingerprints[logical_id] = {
            "fingerprint": _sha256(_canonical(fingerprint_input)),
            "files": files,
        }
    return manifest, fingerprints


def compare_packs(baseline_root: Path, alternate_root: Path) -> dict[str, Any]:
    baseline_manifest, baseline = _load_pack(baseline_root)
    alternate_manifest, alternate = _load_pack(alternate_root)
    baseline_ids = set(baseline)
    alternate_ids = set(alternate)
    shared = sorted(baseline_ids & alternate_ids)
    replaced = sorted(
        logical_id for logical_id in shared
        if baseline[logical_id]["fingerprint"] != alternate[logical_id]["fingerprint"]
    )
    unchanged = sorted(set(shared) - set(replaced))
    missing = sorted(baseline_ids - alternate_ids)
    added = sorted(alternate_ids - baseline_ids)
    assets = {}
    for logical_id in sorted(baseline_ids | alternate_ids):
        status = (
            "missing" if logical_id in missing else
            "added" if logical_id in added else
            "replaced" if logical_id in replaced else
            "unchanged"
        )
        assets[logical_id] = {
            "status": status,
            "baseline_fingerprint": baseline.get(logical_id, {}).get("fingerprint"),
            "alternate_fingerprint": alternate.get(logical_id, {}).get("fingerprint"),
        }
    result = {
        "schema": SCHEMA,
        "baseline": {
            "name": baseline_manifest.get("name"),
            "manifest_sha256": _sha256(_canonical(baseline_manifest)),
            "logical_id_count": len(baseline),
        },
        "alternate": {
            "name": alternate_manifest.get("name"),
            "manifest_sha256": _sha256(_canonical(alternate_manifest)),
            "logical_id_count": len(alternate),
        },
        "logical_ids": {
            "shared": shared,
            "replaced": replaced,
            "inherited": unchanged,
            "missing": missing,
            "added": added,
        },
        "counts": {
            "shared": len(shared),
            "replaced": len(replaced),
            "inherited": len(unchanged),
            "missing": len(missing),
            "added": len(added),
        },
        "assets": assets,
    }
    result["report_sha256"] = _sha256(_canonical(result))
    return result


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-pack", type=Path, required=True)
    parser.add_argument("--alternate-pack", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = compare_packs(args.baseline_pack, args.alternate_pack)
        write_report(args.report, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 1
    print(
        f"Compared {report['counts']['shared']} shared logical IDs: "
        f"{report['counts']['replaced']} replaced, "
        f"{report['counts']['inherited']} inherited, "
        f"{report['counts']['missing']} missing"
    )
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
