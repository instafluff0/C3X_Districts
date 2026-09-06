#!/usr/bin/env python3
"""Build a deterministic content-addressed, byte-deduplicated pack bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


OUTPUT_DIRECTORY = "content_addressed"


def _files(pack: Path, output: Path) -> list[Path]:
    result = []
    for path in pack.rglob("*"):
        if not path.is_file():
            continue
        try:
            path.relative_to(output)
        except ValueError:
            result.append(path)
    return sorted(result, key=lambda value: value.relative_to(pack).as_posix())


def build_content_addressed_bundle(pack: Path, output: Path | None = None) -> dict[str, Any]:
    if not (pack / "manifest.json").is_file():
        raise ValueError("content-addressed bundle source has no manifest.json")
    output = output or pack / OUTPUT_DIRECTORY
    paths: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    aliases: defaultdict[str, list[str]] = defaultdict(list)
    total_bytes = 0
    for path in _files(pack, output):
        relative = path.relative_to(pack).as_posix()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        total_bytes += len(data)
        paths[relative] = digest
        aliases[digest].append(relative)
        if digest in payloads and payloads[digest] != data:
            raise ValueError("SHA-256 collision while indexing pack")
        payloads[digest] = data

    blob = bytearray()
    objects = {}
    for digest in sorted(payloads):
        data = payloads[digest]
        objects[digest] = {
            "offset": len(blob),
            "size": len(data),
            "aliases": sorted(aliases[digest]),
        }
        blob.extend(data)
    resident_sets: defaultdict[str, set[str]] = defaultdict(set)
    for relative, digest in paths.items():
        resident_sets[relative.split("/", 1)[0]].add(digest)
    index = {
        "schema": "c3x.content_addressed_pack.v0",
        "source_manifest": "manifest.json",
        "objects_blob": "objects.bin",
        "objects": objects,
        "paths": paths,
        "resident_sets": {
            key: {
                "objects": sorted(values),
                "unique_bytes": sum(len(payloads[digest]) for digest in values),
            }
            for key, values in sorted(resident_sets.items())
        },
        "summary": {
            "logical_files": len(paths),
            "unique_objects": len(payloads),
            "logical_bytes": total_bytes,
            "unique_bytes": len(blob),
            "deduplicated_bytes": total_bytes - len(blob),
        },
        "runtime_activation": "not_enabled",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "objects.bin").write_bytes(blob)
    (output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def validate_bundle(output: Path) -> dict[str, int]:
    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    if index.get("schema") != "c3x.content_addressed_pack.v0":
        raise ValueError("unsupported content-addressed pack index")
    blob = (output / index["objects_blob"]).read_bytes()
    expected_offset = 0
    for digest, record in sorted(index["objects"].items()):
        if record["offset"] != expected_offset or record["size"] < 0:
            raise ValueError("content-addressed object ranges are not canonical")
        data = blob[record["offset"] : record["offset"] + record["size"]]
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("content-addressed object hash mismatch")
        expected_offset += record["size"]
    if expected_offset != len(blob):
        raise ValueError("content-addressed object blob has trailing or missing bytes")
    if set(index["paths"].values()) - set(index["objects"]):
        raise ValueError("content-addressed path references a missing object")
    return index["summary"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = build_content_addressed_bundle(args.pack, args.output)
        output = args.output or args.pack / OUTPUT_DIRECTORY
        validate_bundle(output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    summary = result["summary"]
    print(
        f"Indexed {summary['logical_files']} files as {summary['unique_objects']} objects; "
        f"saved {summary['deduplicated_bytes']} duplicate bytes at {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
