#!/usr/bin/env python3
"""Read-only structural probe for Civ VI CIVBLP material packages.

This is intentionally not a generic CIVBLP decoder.  It reads only the file
header and package-data region, discovers the serialized allocation table, and
follows enough typed pointers to report one material record and its candidate
texture records.  Any field whose meaning is not established structurally is
reported as unknown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CIV6_BASE = Path(
    r"Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI"
    r"\Civ6.app\Contents\Assets\Base"
)
DEFAULT_PACKAGE = (
    DEFAULT_CIV6_BASE
    / "Platforms"
    / "Windows"
    / "BLPs"
    / "terrain"
    / "TerrainMaterialSet_Base.blp"
)
DEFAULT_TARGET = "ART_DEF_TERRAIN_MATERIAL_GRASSLAND"
DEFAULT_RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = DEFAULT_RENDERER_ROOT / "docs" / "civ6_grassland_material_probe.json"

FILE_HEADER_SIZE = 28
ALLOCATION_SIZE = 40
ENTRY_MAP_TYPE = b"BLP::Package::EntryMap\x00"
MATERIAL_TYPE = "TerrainMaterialPackageEntry"
TEXTURE_TYPE = "BLP::TextureEntry"
CHAR_TYPE = "char"


def offset_fields(value: int) -> dict[str, int | str]:
    return {"offset": value, "offset_hex": f"0x{value:08x}"}


def parse_file_header(
    header: bytes,
    actual_size: int,
    *,
    allow_declared_size_mismatch: bool = False,
) -> dict[str, Any]:
    if len(header) < FILE_HEADER_SIZE:
        raise ValueError(f"CIVBLP file is shorter than its {FILE_HEADER_SIZE}-byte header")
    if header[:6] != b"CIVBLP":
        raise ValueError("File does not start with the CIVBLP signature")
    version = struct.unpack_from("<H", header, 6)[0]
    package_offset, package_size, big_data_offset, big_data_count, declared_size = struct.unpack_from(
        "<5I", header, 8
    )
    if not package_size:
        raise ValueError("CIVBLP package-data size is zero")
    if package_offset + package_size != big_data_offset:
        raise ValueError("CIVBLP package-data range does not end at the declared big-data offset")
    if big_data_offset > actual_size:
        raise ValueError("CIVBLP package-data range extends past the end of the file")
    if declared_size != actual_size and not allow_declared_size_mismatch:
        raise ValueError(
            f"CIVBLP declared file size {declared_size} does not match actual size {actual_size}"
        )
    parsed = {
        "magic": "CIVBLP",
        "version": version,
        "package_data": {**offset_fields(package_offset), "bytes": package_size},
        "big_data": {**offset_fields(big_data_offset), "entry_count": big_data_count},
        "declared_file_bytes": declared_size,
    }
    if declared_size != actual_size:
        parsed["actual_file_bytes"] = actual_size
        parsed["declared_size_mismatch_accepted"] = True
    return parsed


def unpack_allocation(data: bytes, offset: int) -> dict[str, Any]:
    stripe, allocation_type, padding, parent, target_offset, size, count, user_data, type_pointer = (
        struct.unpack_from("<BB4sHIII4xQQ", data, offset)
    )
    return {
        "stripe": stripe,
        "allocation_type": allocation_type,
        "padding_hex": padding.hex(),
        "parent_pointer": parent,
        "target_offset": target_offset,
        "size": size,
        "element_count": count,
        "user_data": user_data,
        "type_pointer": type_pointer,
    }


def parse_allocation_candidate(package: bytes, table_offset: int) -> list[dict[str, Any]] | None:
    allocations: list[dict[str, Any]] = []
    cursor = table_offset
    while cursor + ALLOCATION_SIZE <= len(package):
        chunk = package[cursor : cursor + ALLOCATION_SIZE]
        if not any(chunk):
            if any(package[cursor:]):
                return None
            break
        item = unpack_allocation(package, cursor)
        if item["stripe"] not in (0, 1) or item["padding_hex"] != "00000000":
            return None
        if item["parent_pointer"] > len(package) or item["target_offset"] > len(package):
            return None
        allocations.append(item)
        cursor += ALLOCATION_SIZE
    if not allocations or any(package[cursor:]):
        return None
    first = allocations[0]
    if (
        first["stripe"] != 0
        or first["parent_pointer"] != 0
        or first["target_offset"] != 0
        or not first["size"]
        or not first["element_count"]
        or first["size"] % first["element_count"]
    ):
        return None
    if any(item["type_pointer"] > len(allocations) for item in allocations):
        return None
    return allocations


def find_allocation_table(package: bytes) -> tuple[int, list[dict[str, Any]]]:
    candidates: list[tuple[int, list[dict[str, Any]]]] = []
    cursor = 0
    while True:
        marker = package.find(ENTRY_MAP_TYPE, cursor)
        if marker < 0:
            break
        table_offset = marker + len(ENTRY_MAP_TYPE)
        parsed = parse_allocation_candidate(package, table_offset)
        if parsed is not None:
            candidates.append((table_offset, parsed))
        cursor = marker + 1
    if len(candidates) != 1:
        raise ValueError(f"Expected one structural allocation-table candidate, found {len(candidates)}")
    return candidates[0]


def raw_occurrences(data: bytes, value: bytes) -> list[int]:
    found: list[int] = []
    cursor = 0
    while True:
        cursor = data.find(value, cursor)
        if cursor < 0:
            return found
        found.append(cursor)
        cursor += 1


def infer_temp_stripe_base(
    package: bytes,
    allocations: list[dict[str, Any]],
    minimum_support: int = 3,
) -> tuple[int, list[str]]:
    if minimum_support < 2:
        raise ValueError("Temp-stripe inference requires at least two supporting type names")
    type_names = (MATERIAL_TYPE, TEXTURE_TYPE, CHAR_TYPE, ENTRY_MAP_TYPE[:-1].decode("ascii"))
    votes: Counter[int] = Counter()
    evidence: dict[int, list[str]] = {}
    for type_name in type_names:
        encoded = type_name.encode("ascii") + b"\x00"
        for raw_offset in raw_occurrences(package, encoded):
            for pointer, allocation in enumerate(allocations, 1):
                if (
                    allocation["stripe"] == 1
                    and allocation["parent_pointer"] == 0
                    and allocation["size"] == len(encoded)
                    and allocation["element_count"] == len(encoded)
                ):
                    candidate = raw_offset - allocation["target_offset"]
                    votes[candidate] += 1
                    evidence.setdefault(candidate, []).append(
                        f"pointer {pointer} + stripe offset 0x{allocation['target_offset']:x} -> {type_name}"
                    )
    if not votes:
        raise ValueError("Could not infer the temp-stripe base from reflected allocation type names")
    stripe_base, score = votes.most_common(1)[0]
    if score < minimum_support or list(votes.values()).count(score) != 1:
        raise ValueError("Temp-stripe base inference was not uniquely supported by reflected type names")
    return stripe_base, evidence[stripe_base]


def resolve_direct_file_offset(
    allocation: dict[str, Any], stripe_bases: dict[int, int]
) -> int | None:
    if allocation["parent_pointer"] != 0 or allocation["stripe"] not in stripe_bases:
        return None
    return stripe_bases[allocation["stripe"]] + allocation["target_offset"]


def read_allocated_string(data: bytes, offset: int, size: int) -> tuple[str, int] | None:
    if offset < 0 or size < 1 or offset + size > len(data):
        return None
    raw = data[offset : offset + size]
    if raw[-1:] == b"\x00" and all(0x20 <= byte < 0x7F for byte in raw[:-1]):
        return raw[:-1].decode("ascii"), offset
    if size >= 9:
        capacity, length = struct.unpack_from("<II", raw)
        if capacity == length + 1 and size == 8 + capacity:
            encoded = raw[8 : 8 + length]
            if raw[8 + length] == 0 and all(0x20 <= byte < 0x7F for byte in encoded):
                return encoded.decode("ascii"), offset + 8
    return None


def resolve_type_name(
    pointer: int,
    package: bytes,
    allocations: list[dict[str, Any]],
    stripe_bases: dict[int, int],
) -> str | None:
    if pointer < 1 or pointer > len(allocations):
        return None
    allocation = allocations[pointer - 1]
    target = resolve_direct_file_offset(allocation, stripe_bases)
    if target is None:
        return None
    resolved = read_allocated_string(package, target, allocation["size"])
    return None if resolved is None else resolved[0]


def infer_package_stripe_base(
    package: bytes,
    allocations: list[dict[str, Any]],
    temp_base: int,
    target: str,
) -> tuple[int, int, list[str]]:
    target_bytes = target.encode("ascii") + b"\x00"
    target_offsets = raw_occurrences(package, target_bytes)
    if len(target_offsets) != 1:
        raise ValueError(f"Expected target name once in package data, found {len(target_offsets)}")
    target_offset = target_offsets[0]
    temp_bases = {1: temp_base}
    candidate_bases: set[int] = set()
    char_allocations: list[tuple[int, dict[str, Any]]] = []
    for pointer, allocation in enumerate(allocations, 1):
        if (
            allocation["stripe"] == 0
            and allocation["parent_pointer"] == 0
            and resolve_type_name(allocation["type_pointer"], package, allocations, temp_bases) == CHAR_TYPE
        ):
            char_allocations.append((pointer, allocation))
            layouts = []
            if allocation["size"] == len(target_bytes):
                layouts.append(0)
            if allocation["size"] == len(target_bytes) + 8:
                layouts.append(8)
            for text_delta in layouts:
                candidate = target_offset - allocation["target_offset"] - text_delta
                resolved = read_allocated_string(
                    package,
                    candidate + allocation["target_offset"],
                    allocation["size"],
                )
                if resolved is not None and resolved[0] == target:
                    candidate_bases.add(candidate)
    if not candidate_bases:
        raise ValueError("Could not derive a package-stripe base candidate from the target string")

    scores: Counter[int] = Counter()
    for candidate in candidate_bases:
        for _pointer, allocation in char_allocations:
            resolved = read_allocated_string(
                package,
                candidate + allocation["target_offset"],
                allocation["size"],
            )
            if resolved is not None:
                scores[candidate] += 1
    package_base, score = scores.most_common(1)[0]
    if score < 2 or list(scores.values()).count(score) != 1:
        raise ValueError("Package-stripe base inference was not uniquely supported by string allocations")

    target_pointers = []
    for pointer, allocation in char_allocations:
        resolved = read_allocated_string(
            package,
            package_base + allocation["target_offset"],
            allocation["size"],
        )
        if resolved is not None and resolved[0] == target:
            target_pointers.append(pointer)
    if len(target_pointers) != 1:
        raise ValueError(f"Expected one target-name allocation, found {len(target_pointers)}")
    target_pointer = target_pointers[0]

    evidence = [
        f"target bytes at package offset 0x{target_offset:x}",
        f"allocation pointer {target_pointer} is a typed char allocation at stripe offset "
        f"0x{allocations[target_pointer - 1]['target_offset']:x}",
        f"the selected base resolves {score} typed char allocations, more than any competing target-derived base",
    ]
    return package_base, target_pointer, evidence


def resolve_allocation_target(
    pointer: int,
    allocations: list[dict[str, Any]],
    stripe_bases: dict[int, int],
) -> tuple[int, int] | None:
    if pointer < 1 or pointer > len(allocations):
        return None
    allocation = allocations[pointer - 1]
    direct = resolve_direct_file_offset(allocation, stripe_bases)
    if direct is not None:
        return direct, allocation["size"]
    parent_pointer = allocation["parent_pointer"]
    if parent_pointer < 1 or parent_pointer > len(allocations):
        return None
    parent = allocations[parent_pointer - 1]
    parent_target = resolve_direct_file_offset(parent, stripe_bases)
    if parent_target is None or not parent["element_count"] or parent["size"] % parent["element_count"]:
        return None
    element_size = parent["size"] // parent["element_count"]
    element_index = allocation["target_offset"]
    if element_index >= parent["element_count"]:
        return None
    return parent_target + element_index * element_size, element_size


def strings_in_record(
    package: bytes,
    record_offset: int,
    record_size: int,
    allocations: list[dict[str, Any]],
    stripe_bases: dict[int, int],
    package_file_offset: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for field_offset in range(0, record_size - 7, 8):
        pointer = struct.unpack_from("<Q", package, record_offset + field_offset)[0]
        if pointer < 1 or pointer > len(allocations):
            continue
        allocation = allocations[pointer - 1]
        if resolve_type_name(allocation["type_pointer"], package, allocations, stripe_bases) != CHAR_TYPE:
            continue
        target = resolve_allocation_target(pointer, allocations, stripe_bases)
        if target is None:
            continue
        text = read_allocated_string(package, target[0], allocation["size"])
        if text is not None:
            result.append(
                {
                    "field_offset": field_offset,
                    "field_offset_hex": f"0x{field_offset:02x}",
                    "pointer": pointer,
                    "value": text[0],
                    "string_file_offset": package_file_offset + text[1],
                    "string_file_offset_hex": f"0x{package_file_offset + text[1]:08x}",
                }
            )
    return result


def probe_package_bytes(
    data: bytes,
    source: str,
    target: str = DEFAULT_TARGET,
    actual_file_size: int | None = None,
) -> dict[str, Any]:
    header = parse_file_header(
        data[:FILE_HEADER_SIZE], len(data) if actual_file_size is None else actual_file_size
    )
    package_file_offset = header["package_data"]["offset"]
    package_end = header["big_data"]["offset"]
    package = data[package_file_offset:package_end]

    table_offset, allocations = find_allocation_table(package)
    temp_base, temp_evidence = infer_temp_stripe_base(package, allocations)
    package_base, target_pointer, package_evidence = infer_package_stripe_base(
        package, allocations, temp_base, target
    )
    stripe_bases = {0: package_base, 1: temp_base}

    material_candidates: list[tuple[int, dict[str, Any], int, int]] = []
    for pointer, allocation in enumerate(allocations, 1):
        if resolve_type_name(allocation["type_pointer"], package, allocations, stripe_bases) != MATERIAL_TYPE:
            continue
        resolved = resolve_allocation_target(pointer, allocations, stripe_bases)
        if resolved is None:
            continue
        record_offset, record_size = resolved
        for field_offset in range(0, record_size - 7, 8):
            if struct.unpack_from("<Q", package, record_offset + field_offset)[0] == target_pointer:
                material_candidates.append((pointer, allocation, record_offset, field_offset))
    if len(material_candidates) != 1:
        raise ValueError(f"Expected one typed material record referencing the target, found {len(material_candidates)}")
    material_pointer, material_allocation, material_offset, name_field_offset = material_candidates[0]
    material_size = material_allocation["size"]

    texture_pointers: list[dict[str, Any]] = []
    recognized_offsets = {name_field_offset}
    for field_offset in range(0, material_size - 7, 8):
        pointer = struct.unpack_from("<Q", package, material_offset + field_offset)[0]
        if pointer < 1 or pointer > len(allocations):
            continue
        allocation = allocations[pointer - 1]
        type_name = resolve_type_name(allocation["type_pointer"], package, allocations, stripe_bases)
        if type_name != TEXTURE_TYPE:
            continue
        resolved = resolve_allocation_target(pointer, allocations, stripe_bases)
        if resolved is None:
            continue
        texture_offset, texture_size = resolved
        record_strings = strings_in_record(
            package,
            texture_offset,
            texture_size,
            allocations,
            stripe_bases,
            package_file_offset,
        )
        texture_pointers.append(
            {
                "material_field_offset": field_offset,
                "material_field_offset_hex": f"0x{field_offset:02x}",
                "pointer": pointer,
                "allocation_index": pointer - 1,
                "candidate_record": {
                    "start": package_file_offset + texture_offset,
                    "start_hex": f"0x{package_file_offset + texture_offset:08x}",
                    "end_exclusive": package_file_offset + texture_offset + texture_size,
                    "end_exclusive_hex": f"0x{package_file_offset + texture_offset + texture_size:08x}",
                    "bytes": texture_size,
                    "strings": record_strings,
                },
                "confidence": "high",
                "evidence": [
                    f"material qword resolves through allocation pointer {pointer}",
                    f"allocation type resolves to {TEXTURE_TYPE}",
                    "child allocation selects a bounded element of the typed texture array",
                ],
            }
        )
        recognized_offsets.add(field_offset)

    unknown_qwords = []
    for field_offset in range(0, material_size - 7, 8):
        if field_offset in recognized_offsets:
            continue
        value = struct.unpack_from("<Q", package, material_offset + field_offset)[0]
        unknown_qwords.append(
            {
                "name": f"unknown_0x{field_offset:02x}",
                "field_offset": field_offset,
                "raw_u64": value,
                "raw_hex": f"0x{value:016x}",
            }
        )

    type_string_names = (
        MATERIAL_TYPE,
        TEXTURE_TYPE,
        "m_pBaseColorTexture",
        "m_pHeightTexture",
        "m_pSpecTexture",
        "m_pFOWColorTexture",
        "m_pFuzzTexture",
    )
    reflected_strings = []
    for text in type_string_names:
        for offset in raw_occurrences(package, text.encode("ascii") + b"\x00"):
            reflected_strings.append(
                {
                    "value": text,
                    "file_offset": package_file_offset + offset,
                    "file_offset_hex": f"0x{package_file_offset + offset:08x}",
                }
            )

    material_start = package_file_offset + material_offset
    return {
        "schema": "c3x.civblp_material_probe.v0",
        "source": source,
        "target": target,
        "read_policy": "header and package-data region only; big-data payload was not read by the file probe",
        "file_header": header,
        "package_metadata_sha256": hashlib.sha256(package).hexdigest(),
        "allocation_table": {
            "file_offset": package_file_offset + table_offset,
            "file_offset_hex": f"0x{package_file_offset + table_offset:08x}",
            "entry_bytes": ALLOCATION_SIZE,
            "entry_count": len(allocations),
            "confidence": "high",
            "evidence": [
                f"table follows the reflected {ENTRY_MAP_TYPE[:-1].decode('ascii')} type string",
                "every decoded allocation has a known stripe, zero padding, and an in-range type pointer",
                "the first allocation is a divisible non-empty element array",
                "the remaining package-data bytes after the table are zero padding",
            ],
        },
        "stripe_bases": {
            "package_block": {
                "file_offset": package_file_offset + package_base,
                "file_offset_hex": f"0x{package_file_offset + package_base:08x}",
                "confidence": "high",
                "evidence": package_evidence,
            },
            "temp_data": {
                "file_offset": package_file_offset + temp_base,
                "file_offset_hex": f"0x{package_file_offset + temp_base:08x}",
                "confidence": "high",
                "evidence": temp_evidence,
            },
        },
        "reflected_type_and_field_strings": reflected_strings,
        "material_record": {
            "candidate_type": MATERIAL_TYPE,
            "allocation_pointer": material_pointer,
            "allocation_index": material_pointer - 1,
            "start": material_start,
            "start_hex": f"0x{material_start:08x}",
            "end_exclusive": material_start + material_size,
            "end_exclusive_hex": f"0x{material_start + material_size:08x}",
            "bytes": material_size,
            "entry_name_pointer": target_pointer,
            "entry_name_field_offset": name_field_offset,
            "entry_name_field_offset_hex": f"0x{name_field_offset:02x}",
            "confidence": "high",
            "evidence": [
                f"allocation type resolves to {MATERIAL_TYPE}",
                f"record contains the pointer to the unique target string allocation at +0x{name_field_offset:x}",
            ],
            "candidate_texture_pointers": texture_pointers,
            "unknown_qwords": unknown_qwords,
        },
        "limits": [
            "This report identifies typed records and pointer associations, not texture-role semantics.",
            "The reflected material field-name strings are evidence that roles exist, but M1.3 must map field offsets to those roles.",
            "A null texture slot is not named because this probe does not yet prove its field offset or role.",
            "The allocation-table discovery rule is scoped to the observed Civ VI package layout and is not a generic CIVBLP decoder.",
        ],
    }


def probe_file(path: Path, target: str = DEFAULT_TARGET) -> dict[str, Any]:
    with path.open("rb") as source_file:
        header_bytes = source_file.read(FILE_HEADER_SIZE)
        actual_size = path.stat().st_size
        header = parse_file_header(header_bytes, actual_size)
        package_offset = header["package_data"]["offset"]
        package_size = header["package_data"]["bytes"]
        source_file.seek(package_offset)
        package = source_file.read(package_size)
    if len(package) != package_size:
        raise ValueError("Could not read the complete CIVBLP package-data region")

    synthetic = bytearray(header_bytes)
    if len(synthetic) < package_offset:
        synthetic.extend(b"\x00" * (package_offset - len(synthetic)))
    synthetic.extend(package)
    report = probe_package_bytes(bytes(synthetic), str(path), target, actual_file_size=actual_size)
    report["file_header"]["declared_file_bytes"] = actual_size
    report["read_policy"] = "read exactly 28 header bytes plus the declared package-data region; skipped big-data payload"
    return report


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe one Civ VI CIVBLP material binding")
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        report = probe_file(args.source, args.target)
        write_json(args.output, report)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
