#!/usr/bin/env python3
"""Reference loader for the generic content-addressed runtime pack ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler.pack_content_index import validate_bundle


SPEC_PATH = Path(__file__).with_name("pack_loader_abi.json")
INDEX_SCHEMA = "c3x.content_addressed_pack.v0"


def normalize_logical_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/"):
        raise ValueError("logical pack path must be nonempty relative POSIX text")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("logical pack path contains an unsafe component")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError("logical pack path is not canonical")
    return normalized


@dataclass(frozen=True)
class ResolvedObject:
    mount: str
    logical_path: str
    digest: str
    offset: int
    size: int


class BundleMount:
    def __init__(self, directory: Path, name: str | None = None) -> None:
        self.directory = directory.resolve()
        self.name = name or self.directory.parent.name
        validate_bundle(self.directory)
        self.index = json.loads((self.directory / "index.json").read_text(encoding="utf-8"))
        if self.index.get("schema") != INDEX_SCHEMA:
            raise ValueError("unsupported runtime pack index")
        self.blob_path = self.directory / self.index["objects_blob"]
        canonical_paths = {normalize_logical_path(path): digest for path, digest in self.index["paths"].items()}
        if len(canonical_paths) != len(self.index["paths"]):
            raise ValueError("logical pack paths collide after normalization")
        self.paths = canonical_paths

    def resolve(self, logical_path: str) -> ResolvedObject | None:
        logical_path = normalize_logical_path(logical_path)
        digest = self.paths.get(logical_path)
        if digest is None:
            return None
        record = self.index["objects"][digest]
        return ResolvedObject(self.name, logical_path, digest, record["offset"], record["size"])

    def read(self, resolved: ResolvedObject) -> bytes:
        if resolved.mount != self.name:
            raise ValueError("resolved object belongs to another mount")
        with self.blob_path.open("rb") as stream:
            stream.seek(resolved.offset)
            data = stream.read(resolved.size)
        if len(data) != resolved.size or hashlib.sha256(data).hexdigest() != resolved.digest:
            raise ValueError(f"content-addressed object failed verification: {resolved.logical_path}")
        return data


class RuntimePackSet:
    """Ordered mount set; later mounts override complete logical paths."""

    def __init__(self, mounts: Iterable[BundleMount]) -> None:
        self.mounts = list(mounts)
        if not self.mounts:
            raise ValueError("runtime pack set needs at least one mount")
        names = [mount.name for mount in self.mounts]
        if len(names) != len(set(names)):
            raise ValueError("runtime pack mounts need unique names")

    def resolve(self, logical_path: str) -> ResolvedObject:
        for mount in reversed(self.mounts):
            resolved = mount.resolve(logical_path)
            if resolved is not None:
                return resolved
        raise ValueError(f"enabled custom pack path is missing: {logical_path}")

    def read(self, logical_path: str) -> bytes:
        resolved = self.resolve(logical_path)
        mount = next(mount for mount in self.mounts if mount.name == resolved.mount)
        return mount.read(resolved)

    def visible_paths(self) -> dict[str, ResolvedObject]:
        paths = sorted({path for mount in self.mounts for path in mount.paths})
        return {path: self.resolve(path) for path in paths}

    def plan(self, logical_paths: Iterable[str], budget_bytes: int) -> dict[str, Any]:
        if not isinstance(budget_bytes, int) or budget_bytes < 0:
            raise ValueError("resident budget must be a nonnegative integer")
        resolved = [self.resolve(path) for path in logical_paths]
        objects: dict[str, ResolvedObject] = {}
        for item in resolved:
            existing = objects.get(item.digest)
            if existing is not None and existing.size != item.size:
                raise ValueError("same content digest has inconsistent sizes")
            objects.setdefault(item.digest, item)
        required = sum(item.size for item in objects.values())
        if required > budget_bytes:
            raise ValueError(f"resident set requires {required} bytes, exceeds budget {budget_bytes}")
        return {
            "schema": "c3x.runtime_resident_plan.v0",
            "paths": [item.logical_path for item in resolved],
            "objects": [
                {"sha256": digest, "size": item.size, "source_mount": item.mount}
                for digest, item in sorted(objects.items())
            ],
            "required_bytes": required,
            "budget_bytes": budget_bytes,
            "deduplicated_path_references": len(resolved) - len(objects),
            "allocation_allowed": True,
        }

    def load(self, plan: dict[str, Any]) -> dict[str, bytes]:
        if plan.get("schema") != "c3x.runtime_resident_plan.v0" or plan.get("allocation_allowed") is not True:
            raise ValueError("invalid resident plan")
        loaded = {}
        for item in plan["objects"]:
            mount = next((value for value in self.mounts if value.name == item["source_mount"]), None)
            if mount is None:
                raise ValueError("resident plan references an unmounted pack")
            record = mount.index["objects"].get(item["sha256"])
            if not isinstance(record, dict) or record["size"] != item["size"]:
                raise ValueError("resident plan object changed after planning")
            resolved = ResolvedObject(mount.name, record["aliases"][0], item["sha256"], record["offset"], record["size"])
            loaded[item["sha256"]] = mount.read(resolved)
        return loaded


def validate_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "c3x.runtime_pack_loader_abi.v0",
        "index_schema": INDEX_SCHEMA,
        "mount_precedence": "last_mount_wins_per_complete_logical_path",
        "partial_override": "path_granularity_only_unmentioned_base_paths_remain_visible",
        "missing_or_corrupt": "visible_hard_failure_for_enabled_custom_category_no_native_replay",
        "runtime_activation": "not_enabled",
    }
    if any(spec.get(key) != value for key, value in required.items()):
        raise ValueError("runtime pack loader ABI policy drifted")
    if spec.get("runtime_source_format_dependency") is not None:
        raise ValueError("runtime pack loader cannot depend on a source format")
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", type=Path, required=True)
    parser.add_argument("--request", action="append", default=[])
    parser.add_argument("--budget-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        validate_spec()
        packs = RuntimePackSet(BundleMount(path, f"mount_{index}") for index, path in enumerate(args.bundle))
        requested = args.request or sorted(packs.visible_paths())
        plan = packs.plan(requested, args.budget_bytes)
        loaded = packs.load(plan)
        report = {
            "schema": "c3x.runtime_pack_loader_preflight.v0",
            "mounts": [str(path) for path in args.bundle],
            "plan": plan,
            "loaded_objects": len(loaded),
            "status": "offline_reference_loader_passed",
            "runtime_activation": "not_enabled",
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Loaded {len(loaded)} verified objects / {plan['required_bytes']} bytes from {len(args.bundle)} mounts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
