#!/usr/bin/env python3
"""Resolve typed material roles and embedded texture metadata in a CIVBLP package.

The resolver deliberately reads only the fixed header and package-data region.
It validates resource ranges against the file length but does not read or copy
the embedded texture payloads.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import civblp_probe


DEFAULT_REPORT = (
    civblp_probe.DEFAULT_RENDERER_ROOT / "docs" / "civ6_grassland_material_binding.json"
)

ROLE_BY_TEXTURE_CLASS = {
    "Terrain_BaseColor": "base_color",
    "Terrain_Heightmap": "height",
    "Terrain_Spec": "specular",
    "Terrain_FOWColor": "fow_color",
}

FORMAT_INFO = {
    71: ("BC1_UNORM", "linear", 8),
    72: ("BC1_UNORM_SRGB", "srgb", 8),
    74: ("BC2_UNORM", "linear", 16),
    75: ("BC2_UNORM_SRGB", "srgb", 16),
    77: ("BC3_UNORM", "linear", 16),
    78: ("BC3_UNORM_SRGB", "srgb", 16),
    80: ("BC4_UNORM", "linear", 8),
    81: ("BC4_SNORM", "linear", 8),
    83: ("BC5_UNORM", "linear", 16),
    84: ("BC5_SNORM", "linear", 16),
    95: ("BC6H_UF16", "linear", 16),
    96: ("BC6H_SF16", "linear", 16),
    98: ("BC7_UNORM", "linear", 16),
    99: ("BC7_UNORM_SRGB", "srgb", 16),
}


def expected_bc_bytes(width: int, height: int, mip_count: int, block_bytes: int) -> int:
    total = 0
    for _ in range(mip_count):
        total += max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * block_bytes
        width = max(1, width // 2)
        height = max(1, height // 2)
    return total


def resolve_string_pointer(
    pointer: int,
    package: bytes,
    allocations: list[dict[str, Any]],
    stripe_bases: dict[int, int],
) -> str | None:
    if pointer < 1 or pointer > len(allocations):
        return None
    allocation = allocations[pointer - 1]
    if (
        civblp_probe.resolve_type_name(
            allocation["type_pointer"], package, allocations, stripe_bases
        )
        != civblp_probe.CHAR_TYPE
    ):
        return None
    target = civblp_probe.resolve_allocation_target(pointer, allocations, stripe_bases)
    if target is None:
        return None
    decoded = civblp_probe.read_allocated_string(package, target[0], allocation["size"])
    return None if decoded is None else decoded[0]


def infer_texture_string_offsets(
    package: bytes,
    record_offsets: list[int],
    record_size: int,
    allocations: list[dict[str, Any]],
    stripe_bases: dict[int, int],
) -> tuple[int, int, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for field_offset in range(0, record_size - 7, 8):
        values = []
        for record_offset in record_offsets:
            pointer = struct.unpack_from("<Q", package, record_offset + field_offset)[0]
            value = resolve_string_pointer(pointer, package, allocations, stripe_bases)
            if value is None:
                break
            values.append(value)
        if len(values) == len(record_offsets):
            candidates.append(
                {
                    "field_offset": field_offset,
                    "field_offset_hex": f"0x{field_offset:02x}",
                    "resolved_records": len(values),
                    "unique_values": len(set(values)),
                    "values": sorted(set(values)),
                }
            )
    name_candidates = [item for item in candidates if item["unique_values"] == len(record_offsets)]
    class_candidates = [
        item
        for item in candidates
        if set(item["values"]) == set(ROLE_BY_TEXTURE_CLASS)
    ]
    if len(name_candidates) != 1 or len(class_candidates) != 1:
        raise ValueError("Could not uniquely infer texture logical-name and class fields")
    return name_candidates[0]["field_offset"], class_candidates[0]["field_offset"], candidates


def decode_texture_metadata(record: bytes, field_offset: int) -> dict[str, Any]:
    dxgi_format, height, width, depth, unknown, mip_count = struct.unpack_from(
        "<6H", record, field_offset
    )
    if dxgi_format not in FORMAT_INFO:
        raise ValueError(f"Unsupported texture format {dxgi_format}")
    format_name, color_space, block_bytes = FORMAT_INFO[dxgi_format]
    if min(width, height, depth, mip_count) < 1:
        raise ValueError("Texture metadata has a zero dimension or mip count")
    return {
        "field_offset": field_offset,
        "field_offset_hex": f"0x{field_offset:02x}",
        "format": {
            "dxgi": dxgi_format,
            "name": format_name,
            "color_space": color_space,
            "block_bytes": block_bytes,
        },
        "width": width,
        "height": height,
        "depth": depth,
        "unknown_u16_after_depth": unknown,
        "mip_count": mip_count,
    }


def infer_texture_metadata_offset(records: list[bytes]) -> tuple[int, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for field_offset in range(0, len(records[0]) - 11, 2):
        decoded = []
        try:
            for record in records:
                item = decode_texture_metadata(record, field_offset)
                if (
                    item["depth"] != 1
                    or item["unknown_u16_after_depth"] != 1
                    or item["mip_count"] > 16
                    or item["width"] & (item["width"] - 1)
                    or item["height"] & (item["height"] - 1)
                ):
                    raise ValueError("implausible texture metadata")
                decoded.append(item)
        except (ValueError, struct.error):
            continue
        candidates.append(
            {
                "field_offset": field_offset,
                "field_offset_hex": f"0x{field_offset:02x}",
                "validated_records": len(decoded),
            }
        )
    if len(candidates) != 1:
        raise ValueError(f"Expected one texture metadata layout candidate, found {len(candidates)}")
    return candidates[0]["field_offset"], candidates


def infer_embedded_resource_fields(
    records: list[bytes], metadata_offset: int, big_data_bytes: int
) -> tuple[int, int, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for offset_field in range(0, len(records[0]) - 15, 8):
        size_field = offset_field + 8
        valid = True
        for record in records:
            relative_offset, byte_count = struct.unpack_from("<QQ", record, offset_field)
            metadata = decode_texture_metadata(record, metadata_offset)
            expected = expected_bc_bytes(
                metadata["width"],
                metadata["height"],
                metadata["mip_count"],
                metadata["format"]["block_bytes"],
            )
            if byte_count != expected or relative_offset + byte_count > big_data_bytes:
                valid = False
                break
        if valid:
            candidates.append(
                {
                    "offset_field": offset_field,
                    "offset_field_hex": f"0x{offset_field:02x}",
                    "size_field": size_field,
                    "size_field_hex": f"0x{size_field:02x}",
                    "validated_records": len(records),
                }
            )
    if len(candidates) != 1:
        raise ValueError(f"Expected one embedded-resource layout candidate, found {len(candidates)}")
    item = candidates[0]
    return item["offset_field"], item["size_field"], candidates


def infer_material_role_offsets(
    package: bytes,
    material_offsets: list[int],
    material_size: int,
    texture_class_offset: int,
    allocations: list[dict[str, Any]],
    stripe_bases: dict[int, int],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    roles: dict[str, int] = {}
    for field_offset in range(0, material_size - 7, 8):
        classes = []
        for material_offset in material_offsets:
            pointer = struct.unpack_from("<Q", package, material_offset + field_offset)[0]
            if pointer < 1 or pointer > len(allocations):
                break
            allocation = allocations[pointer - 1]
            if (
                civblp_probe.resolve_type_name(
                    allocation["type_pointer"], package, allocations, stripe_bases
                )
                != civblp_probe.TEXTURE_TYPE
            ):
                break
            target = civblp_probe.resolve_allocation_target(pointer, allocations, stripe_bases)
            if target is None:
                break
            class_pointer = struct.unpack_from(
                "<Q", package, target[0] + texture_class_offset
            )[0]
            texture_class = resolve_string_pointer(
                class_pointer, package, allocations, stripe_bases
            )
            if texture_class is None:
                break
            classes.append(texture_class)
        if len(classes) != len(material_offsets) or len(set(classes)) != 1:
            continue
        texture_class = classes[0]
        if texture_class not in ROLE_BY_TEXTURE_CLASS:
            continue
        role = ROLE_BY_TEXTURE_CLASS[texture_class]
        if role in roles:
            raise ValueError(f"Material role {role} has more than one structural candidate")
        roles[role] = field_offset
        evidence.append(
            {
                "role": role,
                "texture_class": texture_class,
                "field_offset": field_offset,
                "field_offset_hex": f"0x{field_offset:02x}",
                "consistent_records": len(classes),
            }
        )
    if set(roles) != set(ROLE_BY_TEXTURE_CLASS.values()):
        raise ValueError("Could not infer all four non-null material texture roles")
    return roles, evidence


def _typed_parent(
    type_name: str,
    package: bytes,
    allocations: list[dict[str, Any]],
    stripe_bases: dict[int, int],
) -> tuple[dict[str, Any], int, int]:
    candidates = []
    for allocation in allocations:
        if allocation["parent_pointer"] != 0:
            continue
        if (
            civblp_probe.resolve_type_name(
                allocation["type_pointer"], package, allocations, stripe_bases
            )
            != type_name
        ):
            continue
        direct = civblp_probe.resolve_direct_file_offset(allocation, stripe_bases)
        if direct is not None and allocation["element_count"] and allocation["size"] % allocation["element_count"] == 0:
            candidates.append((allocation, direct, allocation["size"] // allocation["element_count"]))
    if len(candidates) != 1:
        raise ValueError(f"Expected one parent allocation for {type_name}, found {len(candidates)}")
    return candidates[0]


def _typed_direct_records(
    type_name: str,
    package: bytes,
    allocations: list[dict[str, Any]],
    stripe_bases: dict[int, int],
) -> tuple[list[int], int]:
    records: list[int] = []
    sizes: set[int] = set()
    for allocation in allocations:
        if allocation["parent_pointer"] != 0:
            continue
        if (
            civblp_probe.resolve_type_name(
                allocation["type_pointer"], package, allocations, stripe_bases
            )
            != type_name
        ):
            continue
        direct = civblp_probe.resolve_direct_file_offset(allocation, stripe_bases)
        if direct is None or allocation["element_count"] != 1:
            continue
        records.append(direct)
        sizes.add(allocation["size"])
    if not records or len(sizes) != 1:
        raise ValueError(f"Expected consistent direct records for {type_name}")
    return records, sizes.pop()


def resolve_file(
    path: Path, target: str = civblp_probe.DEFAULT_TARGET, occurrence: int | None = None
) -> dict[str, Any]:
    actual_size = path.stat().st_size
    with path.open("rb") as source_file:
        header_bytes = source_file.read(civblp_probe.FILE_HEADER_SIZE)
        header = civblp_probe.parse_file_header(header_bytes, actual_size)
        source_file.seek(header["package_data"]["offset"])
        package = source_file.read(header["package_data"]["bytes"])
    if len(package) != header["package_data"]["bytes"]:
        raise ValueError("Could not read the complete CIVBLP package-data region")

    table_offset, allocations = civblp_probe.find_allocation_table(package)
    temp_base, _ = civblp_probe.infer_temp_stripe_base(package, allocations)
    package_base, target_pointer, _ = civblp_probe.infer_package_stripe_base(
        package, allocations, temp_base, target
    )
    stripe_bases = {0: package_base, 1: temp_base}

    texture_parent, texture_base, texture_size = _typed_parent(
        civblp_probe.TEXTURE_TYPE, package, allocations, stripe_bases
    )
    if texture_parent["element_count"] != header["big_data"]["entry_count"]:
        raise ValueError("Texture record count does not match the header big-data entry count")
    texture_offsets = [
        texture_base + index * texture_size for index in range(texture_parent["element_count"])
    ]
    texture_records = [package[offset : offset + texture_size] for offset in texture_offsets]
    name_offset, class_offset, string_candidates = infer_texture_string_offsets(
        package, texture_offsets, texture_size, allocations, stripe_bases
    )
    metadata_offset, metadata_candidates = infer_texture_metadata_offset(texture_records)
    big_data_bytes = actual_size - header["big_data"]["offset"]
    resource_offset_field, resource_size_field, resource_candidates = infer_embedded_resource_fields(
        texture_records, metadata_offset, big_data_bytes
    )

    material_offsets, material_size = _typed_direct_records(
        civblp_probe.MATERIAL_TYPE, package, allocations, stripe_bases
    )
    role_offsets, role_evidence = infer_material_role_offsets(
        package,
        material_offsets,
        material_size,
        class_offset,
        allocations,
        stripe_bases,
    )
    target_materials = []
    for material_offset in material_offsets:
        if any(
            struct.unpack_from("<Q", package, material_offset + field_offset)[0] == target_pointer
            for field_offset in range(0, material_size - 7, 8)
        ):
            target_materials.append(material_offset)
    if len(target_materials) != 1 and occurrence is None:
        raise ValueError(f"Expected one target material, found {len(target_materials)}")
    if not target_materials or occurrence is not None and not 0 <= occurrence < len(target_materials):
        raise ValueError(f"Target material occurrence {occurrence} is out of range for {len(target_materials)} records")
    material_offset = target_materials[0 if occurrence is None else occurrence]

    roles = []
    for role in ("base_color", "height", "specular", "fow_color"):
        field_offset = role_offsets[role]
        pointer = struct.unpack_from("<Q", package, material_offset + field_offset)[0]
        resolved = civblp_probe.resolve_allocation_target(pointer, allocations, stripe_bases)
        if resolved is None:
            raise ValueError(f"Could not resolve target material role {role}")
        texture_offset, resolved_size = resolved
        if resolved_size != texture_size:
            raise ValueError("Texture child allocation has an unexpected record size")
        record = package[texture_offset : texture_offset + texture_size]
        name_pointer = struct.unpack_from("<Q", record, name_offset)[0]
        class_pointer = struct.unpack_from("<Q", record, class_offset)[0]
        logical_name = resolve_string_pointer(name_pointer, package, allocations, stripe_bases)
        texture_class = resolve_string_pointer(class_pointer, package, allocations, stripe_bases)
        if logical_name is None or ROLE_BY_TEXTURE_CLASS.get(texture_class) != role:
            raise ValueError(f"Target material role {role} failed its typed-class validation")
        metadata = decode_texture_metadata(record, metadata_offset)
        relative_offset = struct.unpack_from("<Q", record, resource_offset_field)[0]
        byte_count = struct.unpack_from("<Q", record, resource_size_field)[0]
        absolute_offset = header["big_data"]["offset"] + relative_offset
        roles.append(
            {
                "role": role,
                "status": "resolved",
                "confidence": "high",
                "material_field_offset": field_offset,
                "material_field_offset_hex": f"0x{field_offset:02x}",
                "allocation_pointer": pointer,
                "logical_name": logical_name,
                "texture_class": texture_class,
                "texture_record": {
                    "file_offset": header["package_data"]["offset"] + texture_offset,
                    "bytes": texture_size,
                },
                "metadata": metadata,
                "storage": {
                    "mode": "embedded_blp_big_data",
                    "relative_offset": relative_offset,
                    "absolute_file_offset": absolute_offset,
                    "bytes": byte_count,
                    "end_exclusive": absolute_offset + byte_count,
                    "bounds_valid": absolute_offset + byte_count <= actual_size,
                },
                "evidence": [
                    f"the same material offset resolves to class {texture_class} in all {len(material_offsets)} material records",
                    "logical name and class come from independently inferred typed string fields",
                    "embedded byte count exactly matches the declared BC mip chain",
                ],
            }
        )

    fuzz_offset = max(role_offsets.values()) + 8
    fuzz_values = [
        struct.unpack_from("<Q", package, offset + fuzz_offset)[0] for offset in material_offsets
    ]
    roles.append(
        {
            "role": "fuzz",
            "status": "null",
            "confidence": "medium",
            "material_field_offset": fuzz_offset,
            "material_field_offset_hex": f"0x{fuzz_offset:02x}",
            "logical_name": None,
            "texture_class": None,
            "storage": None,
            "evidence": [
                "m_pFuzzTexture follows m_pFOWColorTexture in the package's reflected field-name sequence",
                f"the next qword after the four proven role fields is null in all {len(fuzz_values)} material records",
                "the exact fuzz offset remains inferred because the package does not expose a non-null typed instance",
            ],
        }
    )

    return {
        "schema": "c3x.civblp_material_binding.v0",
        "source": str(path),
        "target": target,
        **({
            "target_occurrence": occurrence,
            "target_candidate_count": len(target_materials),
        } if occurrence is not None else {}),
        "read_policy": "read exactly the 28-byte header and package-data region; validated but did not read embedded big-data payloads",
        "file_header": header,
        "layout_analysis": {
            "allocation_table_file_offset": header["package_data"]["offset"] + table_offset,
            "material_record_count": len(material_offsets),
            "material_record_bytes": material_size,
            "texture_record_count": len(texture_offsets),
            "texture_record_bytes": texture_size,
            "texture_string_field_candidates": string_candidates,
            "texture_metadata_candidates": metadata_candidates,
            "embedded_resource_candidates": resource_candidates,
            "material_role_fields": role_evidence,
            "confidence": "high",
            "evidence": [
                "all texture layouts were inferred across every typed texture record",
                "all non-null material roles were inferred across every typed material record",
                "all embedded texture byte ranges are bounded by the CIVBLP big-data region",
            ],
        },
        "resource_resolution": {
            "rule": "follow the material's typed texture pointer, then use that TextureEntry's embedded resource offset and byte count relative to the CIVBLP big-data region",
            "mode": "embedded_blp_big_data",
            "standalone_shared_data_required": False,
            "correction": "The probed material textures are embedded in TerrainMaterialSet_Base.blp; matching standalone SHARED_DATA files are neither present nor required.",
        },
        "roles": roles,
        "limitations": [
            "This resolves material-to-texture bindings and metadata only; it does not decode payloads.",
            "The unknown u16 following depth is preserved without assigning a semantic name.",
            "Fuzz is inferred from reflection order and consistent null values rather than a live typed pointer.",
        ],
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve CIVBLP material texture roles")
    parser.add_argument("source", nargs="?", type=Path, default=civblp_probe.DEFAULT_PACKAGE)
    parser.add_argument("--target", default=civblp_probe.DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        report = resolve_file(args.source, args.target)
        write_json(args.output, report)
    except (OSError, ValueError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
