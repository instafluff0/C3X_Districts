#!/usr/bin/env python3
"""Extract typed terrain-element height fields into generic R8 DDS resources.

The runtime never reads CIVBLP.  This local compiler follows reflected allocation
metadata, validates every byte range and name hash, and emits ordinary DDS files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import civblp_probe


MAC_DEFAULT_PACKAGE = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common"
    / "Sid Meier's Civilization VI/Civ6.app/Contents/Assets/Base/Platforms/Windows/BLPs"
    / "terrain/TerrainElementSet_Base.blp"
)
DEFAULT_PACKAGE = (
    MAC_DEFAULT_PACKAGE
    if MAC_DEFAULT_PACKAGE.is_file()
    else civblp_probe.DEFAULT_PACKAGE.with_name("TerrainElementSet_Base.blp")
)
DEFAULT_OUTPUT = civblp_probe.DEFAULT_RENDERER_ROOT / "preview" / "out" / "terrain_relief"
ELEMENT_TYPE = "TerrainElementPackageEntry"
BLOB_TYPE = "BLP::BlobEntry"
ENTRY_NAME_PATTERN = re.compile(rb"ART_DEF_TERRAIN_ELEMENT_[A-Z0-9_]+\0")
HEIGHT_RESOURCES = {
    "surface_detail": "TER_Hills_Standard_Element_0",
}
MOUNTAIN_FAMILIES = {
    "standard": tuple(f"Mountain_Single_0{index}" for index in range(1, 6)),
    "desert": tuple(f"MountainDesert_Single_0{index}" for index in range(1, 5)),
}
MOUNTAIN_CHANNELS = ("HM", "HBLEND", "ID")
MOUNTAIN_RESOURCES = tuple(
    f"{stem}_HM_0" for stems in MOUNTAIN_FAMILIES.values() for stem in stems
)
HILL_FAMILIES = {
    "standard": "TER_Hills_Standard_Element",
    "continental": "TER_Hills_Continental_Element",
    "continental_plains": "TER_Hills_Continental_Element_Plains",
    "continental_snow": "TER_Hills_Continental_Element_Snow",
}
RELIEF_OUTPUTS = ("surface_detail", "mountain_atlas")
WATER_RELIEF_FAMILIES = {
    "oasis": {
        "height": "Oasis_HM",
        "blend": "Oasis_HBLEND",
        "region_ids": "Oasis_ID",
    },
    "river_bank_noise": {"height": "RiverBank_Noise_HM"},
    "river_origin_flat": {
        "height": "RiverOrigin_Generic_HM",
        "blend": "RiverOrigin_Generic_H_BLEND",
    },
    "river_origin_hill": {
        "height": "RiverOrigin_Hill_HM",
        "blend": "RiverOrigin_Hill_H_BLEND",
    },
    "river_origin_mountain": {
        "height": "RiverOrigin_Mountain_HM",
        "blend": "RiverOrigin_Mountain_H_BLEND",
    },
}


def fnv1a32(text: str) -> int:
    value = 0x811C9DC5
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def make_r8_dds(width: int, height: int, pixels: bytes, dxgi_format: int = 61) -> bytes:
    if width < 1 or height < 1 or len(pixels) != width * height:
        raise ValueError("R8 dimensions do not match the payload")
    if dxgi_format not in (61, 62):
        raise ValueError("Relief DDS must use R8_UNORM or R8_UINT")
    header = bytearray(148)
    header[:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    struct.pack_into("<I", header, 8, 0x0002100F)  # caps, size, pitch, pixel format, mip count
    struct.pack_into("<II", header, 12, height, width)
    struct.pack_into("<I", header, 20, width)
    struct.pack_into("<I", header, 28, 1)
    struct.pack_into("<I", header, 76, 32)
    struct.pack_into("<I", header, 80, 0x4)
    header[84:88] = b"DX10"
    struct.pack_into("<I", header, 108, 0x1000)
    struct.pack_into("<IIIII", header, 128, dxgi_format, 3, 0, 1, 0)
    return bytes(header) + pixels


def summarize_channel(pixels: bytes, width: int, height: int) -> dict[str, Any]:
    if width < 1 or height < 1 or len(pixels) != width * height:
        raise ValueError("Channel dimensions do not match the payload")
    counts = Counter(pixels)
    edge = (
        pixels[:width]
        + pixels[-width:]
        + pixels[0::width]
        + pixels[width - 1 :: width]
    )
    return {
        "width": width,
        "height": height,
        "minimum": min(pixels),
        "maximum": max(pixels),
        "mean": sum(pixels) / len(pixels),
        "unique_values": len(counts),
        "values": sorted(counts) if len(counts) <= 16 else None,
        "edge_mean": sum(edge) / len(edge),
        "edge_nonzero_fraction": sum(value != 0 for value in edge) / len(edge),
        "sha256": hashlib.sha256(pixels).hexdigest(),
    }


def compare_lod_pair(high: bytes, high_width: int, low: bytes, low_width: int) -> dict[str, Any]:
    if high_width != low_width * 2 or len(high) != high_width * high_width or len(low) != low_width * low_width:
        raise ValueError("Expected a square 2:1 relief LOD pair")
    downsampled = []
    for y in range(low_width):
        for x in range(low_width):
            offset = (2 * y) * high_width + 2 * x
            downsampled.append((
                high[offset]
                + high[offset + 1]
                + high[offset + high_width]
                + high[offset + high_width + 1]
            ) / 4.0)
    high_mean = sum(downsampled) / len(downsampled)
    low_mean = sum(low) / len(low)
    covariance = sum((a - high_mean) * (b - low_mean) for a, b in zip(downsampled, low))
    high_variance = sum((value - high_mean) ** 2 for value in downsampled)
    low_variance = sum((value - low_mean) ** 2 for value in low)
    denominator = math.sqrt(high_variance * low_variance)
    correlation = covariance / denominator if denominator else 1.0
    mean_absolute_error = sum(abs(a - b) for a, b in zip(downsampled, low)) / len(low)
    return {
        "resolution_ratio": 2,
        "box_downsample_correlation": correlation,
        "box_downsample_mean_absolute_error": mean_absolute_error,
        "interpretation": "lower_resolution_lod" if correlation >= 0.98 else "relationship_unconfirmed",
        "confidence": "high" if correlation >= 0.98 else "low",
    }


def _infer_temp_base(package: bytes, allocations: list[dict[str, Any]]) -> int:
    type_names = (ELEMENT_TYPE, BLOB_TYPE, civblp_probe.CHAR_TYPE, "BLP::Package::EntryMap")
    votes: Counter[int] = Counter()
    for name in type_names:
        encoded = name.encode("ascii") + b"\0"
        for raw_offset in civblp_probe.raw_occurrences(package, encoded):
            for allocation in allocations:
                if (allocation["stripe"] == 1 and allocation["parent_pointer"] == 0 and
                        allocation["size"] == len(encoded) and allocation["element_count"] == len(encoded)):
                    votes[raw_offset - allocation["target_offset"]] += 1
    if not votes:
        raise ValueError("Could not infer terrain-element temp stripe")
    base, score = votes.most_common(1)[0]
    if score < 3 or list(votes.values()).count(score) != 1:
        raise ValueError("Terrain-element temp stripe was not uniquely supported")
    return base


def _infer_terrain_package_base(
    package: bytes,
    allocations: list[dict[str, Any]],
    temp_base: int,
) -> tuple[int, dict[str, Any]]:
    """Infer stripe zero from all unique terrain-element entry names.

    The old reader used one Base-only entry as a bootstrap string. Expansion
    packages have the same reflected layout but a disjoint entry catalog. Every
    independently resolved entry must now vote for the same package base.
    """
    targets = sorted({match.group()[:-1].decode("ascii") for match in ENTRY_NAME_PATTERN.finditer(package)})
    if not targets:
        raise ValueError("Terrain-element package has no entry-name bootstrap candidates")
    confirmations = []
    failures = []
    for target in targets:
        try:
            package_base, pointer, evidence = civblp_probe.infer_package_stripe_base(
                package, allocations, temp_base, target
            )
        except ValueError as exc:
            failures.append({"target": target, "error": str(exc)})
            continue
        confirmations.append({
            "target": target,
            "package_base": package_base,
            "target_pointer": pointer,
            "evidence": evidence,
        })
    bases = {item["package_base"] for item in confirmations}
    if not confirmations or len(bases) != 1:
        raise ValueError(
            "Terrain-element package-base inference did not converge on one stripe base"
        )
    return next(iter(bases)), {
        "candidate_count": len(targets),
        "confirmation_count": len(confirmations),
        "failure_count": len(failures),
        "confirmations": confirmations,
        "failures": failures,
    }


def decode_terrain_element_record(
    raw: bytes,
    entry_name: str,
    resources_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if len(raw) != 144:
        raise ValueError(f"Terrain-element record must be 144 bytes, found {len(raw)}")
    stored_hash = struct.unpack_from("<I", raw, 0x40)[0]
    if stored_hash != fnv1a32(entry_name):
        raise ValueError(f"Terrain-element entry-name hash mismatch: {entry_name}")
    dimensions = struct.unpack_from("<2I", raw, 0x84)
    if dimensions[0] < 4 or dimensions[1] < 4 or any(value % 4 for value in dimensions):
        raise ValueError(f"Terrain-element grid dimensions are invalid: {entry_name}")
    parameters = {
        "noise_scale": struct.unpack_from("<f", raw, 0x78)[0],
        "base_height": struct.unpack_from("<f", raw, 0x7C)[0],
        "height_scale": struct.unpack_from("<f", raw, 0x80)[0],
    }
    if not all(math.isfinite(value) for value in parameters.values()):
        raise ValueError(f"Terrain-element calibration contains a non-finite value: {entry_name}")
    if parameters["height_scale"] <= 0.0:
        raise ValueError(f"Terrain-element height scale is not positive: {entry_name}")

    channels = {}
    for role, offset in (("height", 0x48), ("blend", 0x54), ("region_ids", 0x60), ("noise_2d", 0x6C)):
        sentinel, lod0_index, lod1_index = struct.unpack_from("<3I", raw, offset)
        if sentinel != 0 or bool(lod0_index) != bool(lod1_index):
            raise ValueError(f"Terrain-element {role} LOD vector is malformed: {entry_name}")
        if not lod0_index:
            continue
        lods = []
        for level, index in enumerate((lod0_index, lod1_index)):
            resource = resources_by_index.get(index)
            if resource is None:
                raise ValueError(
                    f"Terrain-element {role} references missing blob {index}: {entry_name}"
                )
            expected_width = dimensions[0] // (2 ** (level + 1))
            expected_height = dimensions[1] // (2 ** (level + 1))
            if resource["width"] != expected_width or resource["height"] != expected_height:
                raise ValueError(
                    f"Terrain-element {role} LOD dimensions disagree with its grid: {entry_name}"
                )
            lods.append({"level": level, "resource_index": index, **resource})
        channels[role] = lods
    if "height" not in channels:
        raise ValueError(f"Terrain-element entry has no height channel: {entry_name}")
    return {
        "entry_name": entry_name,
        "entry_name_hash": f"0x{stored_hash:08x}",
        "grid_dimensions": list(dimensions),
        "parameters": parameters,
        "channels": channels,
    }


def inspect_terrain_element_package(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    actual_size = path.stat().st_size
    with path.open("rb") as source:
        header_bytes = source.read(civblp_probe.FILE_HEADER_SIZE)
        header = civblp_probe.parse_file_header(header_bytes, actual_size)
        source.seek(header["package_data"]["offset"])
        package = source.read(header["package_data"]["bytes"])
    table_offset, allocations = civblp_probe.find_allocation_table(package)
    temp_base = _infer_temp_base(package, allocations)
    package_base, package_base_evidence = _infer_terrain_package_base(
        package, allocations, temp_base
    )
    stripe_bases = {0: package_base, 1: temp_base}
    parents = []
    for allocation in allocations:
        if allocation["parent_pointer"] != 0:
            continue
        if civblp_probe.resolve_type_name(
                allocation["type_pointer"], package, allocations, stripe_bases) != BLOB_TYPE:
            continue
        direct = civblp_probe.resolve_direct_file_offset(allocation, stripe_bases)
        if direct is not None and allocation["element_count"] and allocation["size"] % allocation["element_count"] == 0:
            parents.append((allocation, direct, allocation["size"] // allocation["element_count"]))
    if len(parents) != 1:
        raise ValueError(f"Expected one typed blob array, found {len(parents)}")
    parent, record_base, record_size = parents[0]
    if parent["element_count"] != header["big_data"]["entry_count"] or record_size < 56:
        raise ValueError("Typed blob count/record size disagrees with the file header")

    resources: dict[str, dict[str, Any]] = {}
    big_bytes = actual_size - header["big_data"]["offset"]
    for index in range(parent["element_count"]):
        offset = record_base + index * record_size
        strings = civblp_probe.strings_in_record(
            package, offset, record_size, allocations, stripe_bases,
            header["package_data"]["offset"],
        )
        names = [item["value"] for item in strings if item["field_offset"] == 8]
        if len(names) != 1:
            continue
        name = names[0]
        relative, byte_count = struct.unpack_from("<QQ", package, offset + 0x20)
        name_hash = struct.unpack_from("<I", package, offset + 0x30)[0]
        side = math.isqrt(byte_count)
        if name_hash != fnv1a32(name) or side * side != byte_count or relative + byte_count > big_bytes:
            raise ValueError(f"Invalid typed terrain-element resource: {name}")
        resources[name] = {
            "index": index,
            "name": name,
            "relative_offset": relative,
            "bytes": byte_count,
            "width": side,
            "height": side,
            "fnv1a32": f"0x{name_hash:08x}",
        }
    resources_by_index = {item["index"]: item for item in resources.values()}
    elements: dict[str, dict[str, Any]] = {}
    for allocation in allocations:
        if allocation["parent_pointer"] != 0:
            continue
        if civblp_probe.resolve_type_name(
            allocation["type_pointer"], package, allocations, stripe_bases
        ) != ELEMENT_TYPE:
            continue
        direct = civblp_probe.resolve_direct_file_offset(allocation, stripe_bases)
        if direct is None or not allocation["element_count"] or allocation["size"] % allocation["element_count"]:
            raise ValueError("Terrain-element entry allocation is not a bounded regular array")
        record_size = allocation["size"] // allocation["element_count"]
        for index in range(allocation["element_count"]):
            offset = direct + index * record_size
            strings = civblp_probe.strings_in_record(
                package,
                offset,
                record_size,
                allocations,
                stripe_bases,
                header["package_data"]["offset"],
            )
            names = [item["value"] for item in strings if item["field_offset"] == 0x38]
            if len(names) != 1:
                raise ValueError("Terrain-element record does not have one entry name")
            name = names[0]
            if name in elements:
                raise ValueError(f"Duplicate terrain-element entry: {name}")
            elements[name] = decode_terrain_element_record(
                package[offset : offset + record_size], name, resources_by_index
            )
    report = {
        "schema": "c3x.terrain_relief_probe.v0",
        "source": str(path),
        "package_metadata_sha256": hashlib.sha256(package).hexdigest(),
        "allocation_table_offset": table_offset,
        "typed_blob_count": len(resources),
        "terrain_element_count": len(elements),
        "big_data_offset": header["big_data"]["offset"],
        "package_base_inference": package_base_evidence,
    }
    if len(elements) != package_base_evidence["confirmation_count"]:
        raise ValueError("Terrain-element entry count disagrees with package-base confirmations")
    return resources, elements, report


def inspect_relief_package(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    resources, _elements, report = inspect_terrain_element_package(path)
    return resources, report


def extract_water_relief_resources(path: Path, output_root: Path) -> dict[str, Any]:
    """Compile river-bank, river-origin, and oasis channels to generic R8 DDS."""
    resources, package_report = inspect_relief_package(path)
    compiled: dict[str, Any] = {}
    with path.open("rb") as source:
        for family, channels in WATER_RELIEF_FAMILIES.items():
            family_record: dict[str, Any] = {}
            for role, stem in channels.items():
                lod_records = []
                for lod in (0, 1):
                    name = f"{stem}_{lod}"
                    item = resources.get(name)
                    if item is None:
                        raise ValueError(f"Missing authored water relief channel: {name}")
                    source.seek(package_report["big_data_offset"] + item["relative_offset"])
                    payload = source.read(item["bytes"])
                    if len(payload) != item["bytes"]:
                        raise ValueError(f"Short read for authored water relief channel: {name}")
                    target = output_root / family / f"{role}_lod{lod}.dds"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(make_r8_dds(
                        item["width"], item["height"], payload,
                        62 if role == "region_ids" else 61,
                    ))
                    lod_records.append({
                        "lod": lod,
                        "resource": name,
                        "output": str(target),
                        **summarize_channel(payload, item["width"], item["height"]),
                    })
                family_record[role] = lod_records
            compiled[family] = family_record
    return {
        "schema": "c3x.water_relief_compile.v0",
        "source": str(path),
        "compiled": compiled,
        "compiled_texture_count": sum(
            len(lods) for family in compiled.values() for lods in family.values()
        ),
    }


def analyze_relief_package(path: Path) -> dict[str, Any]:
    resources, package_report = inspect_relief_package(path)

    def payload(source, name: str) -> bytes:
        if name not in resources:
            raise ValueError(f"Missing authored relief channel: {name}")
        item = resources[name]
        source.seek(package_report["big_data_offset"] + item["relative_offset"])
        data = source.read(item["bytes"])
        if len(data) != item["bytes"]:
            raise ValueError(f"Short read for authored relief channel: {name}")
        return data

    mountain_variants = []
    hill_families = []
    with path.open("rb") as source:
        for family, stems in MOUNTAIN_FAMILIES.items():
            for variant_index, stem in enumerate(stems, 1):
                channels = {}
                for channel in MOUNTAIN_CHANNELS:
                    high_name = f"{stem}_{channel}_0"
                    low_name = f"{stem}_{channel}_1"
                    high_item = resources.get(high_name)
                    low_item = resources.get(low_name)
                    if high_item is None or low_item is None:
                        raise ValueError(f"Incomplete authored relief LOD pair: {stem}_{channel}")
                    high = payload(source, high_name)
                    low = payload(source, low_name)
                    channels[channel.casefold()] = {
                        "semantic_status": "confirmed_height" if channel == "HM" else "inferred",
                        "semantic_interpretation": {
                            "HM": "authored_macro_height",
                            "HBLEND": "continuous_footprint_or_height_blend",
                            "ID": "discrete_material_or_region_identifier",
                        }[channel],
                        "lod0": {"resource": high_name, **summarize_channel(
                            high, high_item["width"], high_item["height"]
                        )},
                        "lod1": {"resource": low_name, **summarize_channel(
                            low, low_item["width"], low_item["height"]
                        )},
                        "lod_relationship": compare_lod_pair(
                            high, high_item["width"], low, low_item["width"]
                        ),
                    }
                mountain_variants.append({
                    "family": family,
                    "variant": variant_index,
                    "source_stem": stem,
                    "channels": channels,
                })

        for family, stem in HILL_FAMILIES.items():
            high_name, low_name = f"{stem}_0", f"{stem}_1"
            high_item = resources.get(high_name)
            low_item = resources.get(low_name)
            if high_item is None or low_item is None:
                raise ValueError(f"Incomplete authored hill LOD pair: {stem}")
            high = payload(source, high_name)
            low = payload(source, low_name)
            hill_families.append({
                "family": family,
                "semantic_status": "confirmed_height",
                "semantic_interpretation": "authored_macro_height",
                "lod0": {"resource": high_name, **summarize_channel(
                    high, high_item["width"], high_item["height"]
                )},
                "lod1": {"resource": low_name, **summarize_channel(
                    low, low_item["width"], low_item["height"]
                )},
                "lod_relationship": compare_lod_pair(
                    high, high_item["width"], low, low_item["width"]
                ),
            })

    return {
        "schema": "c3x.authored_relief_analysis.v0",
        "source": str(path),
        "source_metadata_sha256": package_report["package_metadata_sha256"],
        "confirmed": [
            "HM resources are authored macro-height fields",
            "mountain HM, HBLEND, and ID channels have aligned 2:1 resolution pairs",
            "hill element resources have aligned 2:1 resolution pairs",
            "standard and desert mountains are distinct authored variant families",
        ],
        "inferred": [
            "the strongly correlated _1 resources are lower-resolution LODs of _0",
            "HBLEND is a continuous footprint or height-blending field",
            "ID stores discrete material or region identifiers",
        ],
        "unresolved": [
            "exact HBLEND equation and whether it modifies height, footprint, or both",
            "mapping from each nonzero ID value to a material or geometric region",
            "selection roles of standard versus continental hill families",
        ],
        "mountain_variants": mountain_variants,
        "hill_families": hill_families,
    }


def render_relief_contact_sheets(path: Path, output: Path) -> list[dict[str, Any]]:
    from Renderer.preview.render_iso import Canvas
    from Renderer.preview.render_textured_patch import write_png

    resources, package_report = inspect_relief_package(path)

    def read(name: str) -> tuple[bytes, int]:
        item = resources[name]
        with path.open("rb") as source:
            source.seek(package_report["big_data_offset"] + item["relative_offset"])
            return source.read(item["bytes"]), item["width"]

    def draw_cell(canvas: Canvas, left: int, top: int, pixels: bytes, width: int, is_id: bool) -> None:
        cell = 128
        minimum, maximum = min(pixels), max(pixels)
        span = max(1, maximum - minimum)
        palette = (
            (24, 27, 30), (72, 129, 190), (237, 177, 32), (81, 168, 98),
            (194, 82, 111), (139, 103, 184), (211, 111, 45), (83, 178, 171),
        )
        ids = {value: index for index, value in enumerate(sorted(set(pixels)))}
        for y in range(cell):
            source_y = min(width - 1, y * width // cell)
            for x in range(cell):
                source_x = min(width - 1, x * width // cell)
                value = pixels[source_y * width + source_x]
                if is_id:
                    color = palette[ids[value] % len(palette)]
                else:
                    level = int((value - minimum) * 255 / span)
                    color = (level, level, level)
                canvas.set_pixel(left + x, top + y, color)

    output.mkdir(parents=True, exist_ok=True)
    gap, cell = 4, 128
    mountain_rows = [(family, stem) for family, stems in MOUNTAIN_FAMILIES.items() for stem in stems]
    mountain = Canvas(3 * cell + 4 * gap, len(mountain_rows) * cell + (len(mountain_rows) + 1) * gap,
                      (28, 31, 34))
    mountain_layout = []
    for row, (family, stem) in enumerate(mountain_rows):
        cells = []
        for column, channel in enumerate(MOUNTAIN_CHANNELS):
            name = f"{stem}_{channel}_0"
            pixels, width = read(name)
            left, top = gap + column * (cell + gap), gap + row * (cell + gap)
            draw_cell(mountain, left, top, pixels, width, channel == "ID")
            cells.append({"column": column, "channel": channel, "resource": name})
        mountain_layout.append({"row": row, "family": family, "source_stem": stem, "cells": cells})
    mountain_path = output / "mountain_authored_channels.png"
    write_png(mountain, mountain_path)

    hill = Canvas(2 * cell + 3 * gap, len(HILL_FAMILIES) * cell + (len(HILL_FAMILIES) + 1) * gap,
                  (28, 31, 34))
    hill_layout = []
    for row, (family, stem) in enumerate(HILL_FAMILIES.items()):
        cells = []
        for column, lod in enumerate((0, 1)):
            name = f"{stem}_{lod}"
            pixels, width = read(name)
            left, top = gap + column * (cell + gap), gap + row * (cell + gap)
            draw_cell(hill, left, top, pixels, width, False)
            cells.append({"column": column, "lod": lod, "resource": name})
        hill_layout.append({"row": row, "family": family, "cells": cells})
    hill_path = output / "hill_authored_lods.png"
    write_png(hill, hill_path)
    return [
        {"kind": "mountain_channels", "path": str(mountain_path), "layout": mountain_layout},
        {"kind": "hill_lods", "path": str(hill_path), "layout": hill_layout},
    ]


def compile_authored_relief_sets(path: Path, pack: Path) -> dict[str, Any]:
    resources, package_report = inspect_relief_package(path)
    relief_root = pack / "relief"

    def write_json(target: Path, value: Any) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def compile_resource(source, name: str, relative: Path, discrete: bool = False) -> dict[str, Any]:
        if name not in resources:
            raise ValueError(f"Missing authored relief resource: {name}")
        item = resources[name]
        source.seek(package_report["big_data_offset"] + item["relative_offset"])
        pixels = source.read(item["bytes"])
        if len(pixels) != item["bytes"]:
            raise ValueError(f"Short read for authored relief resource: {name}")
        target = pack / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(make_r8_dds(item["width"], item["height"], pixels, 62 if discrete else 61))
        return {
            "texture": relative.as_posix(),
            "format": "R8_UINT" if discrete else "R8_UNORM",
            "width": item["width"],
            "height": item["height"],
            "source_resource": name,
            "source_sha256": hashlib.sha256(pixels).hexdigest(),
        }

    runtime_sets = {
        "hills": "relief/hills.json",
        "mountains": "relief/mountains.json",
    }
    source_evidence = []
    with path.open("rb") as source:
        hill_records = []
        for family, stem in HILL_FAMILIES.items():
            lods = []
            for level in (0, 1):
                source_name = f"{stem}_{level}"
                relative = Path("textures") / "relief" / "hills" / family / f"height_lod{level}.dds"
                compiled = compile_resource(source, source_name, relative)
                lods.append({key: value for key, value in compiled.items() if not key.startswith("source_")})
                source_evidence.append({"runtime_texture": relative.as_posix(), **compiled})
            hill_records.append({
                "id": family,
                "macro_height_authority": "authored",
                "lods": lods,
            })
        hills = {
            "schema": "c3x.relief_set.v0",
            "kind": "hills",
            "surface_mode": "shared_height_field",
            "families": hill_records,
            "selection": {
                "default_family": "standard",
                "family_mapping_status": "authored_mapping_required",
                "lod0_min_projected_width_px": 96,
            },
            "material_thresholds": {
                "units": "normalized_source_height",
                "grassland_high": 3.75 / 16.0,
                "plains_high": 3.0 / 16.0,
                "desert_high": 3.75 / 16.0,
                "tundra_high": 3.75 / 16.0,
                "snow_high": 1.5 / 16.0,
            },
        }
        write_json(relief_root / "hills.json", hills)

        mountain_families = []
        for family, stems in MOUNTAIN_FAMILIES.items():
            variants = []
            for variant_index, stem in enumerate(stems, 1):
                lods = []
                for level in (0, 1):
                    channels = {}
                    for source_channel, runtime_channel in (
                        ("HM", "height"), ("HBLEND", "blend"), ("ID", "region_ids")
                    ):
                        source_name = f"{stem}_{source_channel}_{level}"
                        relative = Path("textures") / "relief" / "mountains" / family / \
                            f"variant_{variant_index:02d}" / f"{runtime_channel}_lod{level}.dds"
                        compiled = compile_resource(source, source_name, relative, source_channel == "ID")
                        channels[runtime_channel] = {
                            key: value for key, value in compiled.items() if not key.startswith("source_")
                        }
                        source_evidence.append({"runtime_texture": relative.as_posix(), **compiled})
                    lods.append({"level": level, "channels": channels})
                variants.append({"id": f"variant_{variant_index:02d}", "lods": lods})
            if family == "standard":
                style = {
                    "height_to_width": 32.0 / 42.399979,
                    "snow_low": 24.0 / 32.0,
                    "snow_high": 26.0 / 32.0,
                }
                biome_tags = ["grassland", "plains", "tundra", "snow"]
            else:
                style = {"height_to_width": 24.0 / 54.0}
                biome_tags = ["desert"]
            mountain_families.append({
                "id": family,
                "biome_tags": biome_tags,
                "macro_height_authority": "authored",
                "style": style,
                "variants": variants,
            })
        mountains = {
            "schema": "c3x.relief_set.v0",
            "kind": "mountains",
            "surface_mode": "composed_authored_contributions",
            "families": mountain_families,
            "selection": {"lod0_min_projected_width_px": 96},
            "channel_semantics": {
                "height": "confirmed_macro_height",
                "blend": "continuous_footprint_or_height_blend_inferred",
                "region_ids": "discrete_material_or_region_identifier_inferred",
            },
        }
        write_json(relief_root / "mountains.json", mountains)

    validation_errors = validate_authored_relief_sets(pack)
    if validation_errors:
        raise ValueError("Compiled authored relief sets are invalid: " + "; ".join(validation_errors))
    return {
        "schema": "c3x.authored_relief_compile.v0",
        "runtime_sets": runtime_sets,
        "source_evidence": source_evidence,
        "compiled_texture_count": len(source_evidence),
        "validation": "passed",
    }


def validate_authored_relief_sets(pack: Path) -> list[str]:
    errors = []
    expected = {"hills": (4, 8), "mountains": (2, 54)}
    root = pack.resolve()
    for name, (family_count, texture_count) in expected.items():
        document_path = pack / "relief" / f"{name}.json"
        try:
            document = json.loads(document_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{name} relief set is unreadable: {exc}")
            continue
        if document.get("schema") != "c3x.relief_set.v0" or document.get("kind") != name:
            errors.append(f"{name} relief set has an invalid schema or kind")
        families = document.get("families")
        if not isinstance(families, list) or len(families) != family_count:
            errors.append(f"{name} relief set must contain {family_count} families")
        stack = [document]
        textures = []
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                if "texture" in value:
                    textures.append(value)
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        if len(textures) != texture_count:
            errors.append(f"{name} relief set must reference {texture_count} textures, found {len(textures)}")
        for texture in textures:
            relative = texture.get("texture")
            if not isinstance(relative, str):
                errors.append(f"{name} relief set has a non-string texture path")
                continue
            target = (pack / relative).resolve()
            if target != root and root not in target.parents:
                errors.append(f"{name} relief texture escapes the pack: {relative}")
                continue
            try:
                data = target.read_bytes()
            except OSError as exc:
                errors.append(f"{name} relief texture is unreadable: {relative}: {exc}")
                continue
            if len(data) < 148 or data[:4] != b"DDS " or data[84:88] != b"DX10":
                errors.append(f"{name} relief texture is not bounded DX10 DDS: {relative}")
                continue
            width = struct.unpack_from("<I", data, 16)[0]
            height = struct.unpack_from("<I", data, 12)[0]
            format_id = struct.unpack_from("<I", data, 128)[0]
            expected_format = 62 if texture.get("format") == "R8_UINT" else 61
            if width != texture.get("width") or height != texture.get("height") or format_id != expected_format:
                errors.append(f"{name} relief texture metadata disagrees with DDS: {relative}")
            if len(data) != 148 + width * height:
                errors.append(f"{name} relief texture payload is not tightly bounded: {relative}")
    return errors


def extract_relief_resources(path: Path, output: Path) -> dict[str, Any]:
    resources, report = inspect_relief_package(path)
    output.mkdir(parents=True, exist_ok=True)
    extracted = []
    with path.open("rb") as source:
        big_data_offset = report["big_data_offset"]
        for role, name in HEIGHT_RESOURCES.items():
            if name not in resources:
                raise ValueError(f"Missing required typed relief resource: {name}")
            item = resources[name]
            source.seek(big_data_offset + item["relative_offset"])
            pixels = source.read(item["bytes"])
            if len(pixels) != item["bytes"]:
                raise ValueError(f"Short read for relief resource: {name}")
            target = output / f"{role}.dds"
            target.write_bytes(make_r8_dds(item["width"], item["height"], pixels))
            extracted.append({
                "role": role,
                "source_resource": name,
                "output": str(target),
                "width": item["width"],
                "height": item["height"],
                "source_sha256": hashlib.sha256(pixels).hexdigest(),
            })
        mountain_fields = []
        for name in MOUNTAIN_RESOURCES:
            if name not in resources or resources[name]["width"] != 256 or resources[name]["height"] != 256:
                raise ValueError(f"Missing or incompatible mountain height resource: {name}")
            item = resources[name]
            source.seek(big_data_offset + item["relative_offset"])
            pixels = source.read(item["bytes"])
            if len(pixels) != 256 * 256:
                raise ValueError(f"Short read for mountain height resource: {name}")
            mountain_fields.append((name, pixels))
        atlas = bytearray(768 * 768)
        for index, (_name, pixels) in enumerate(mountain_fields):
            cell_x, cell_y = index % 3, index // 3
            for row in range(256):
                source_start = row * 256
                target_start = (cell_y * 256 + row) * 768 + cell_x * 256
                atlas[target_start:target_start + 256] = pixels[source_start:source_start + 256]
        target = output / "mountain_atlas.dds"
        target.write_bytes(make_r8_dds(768, 768, bytes(atlas)))
        extracted.append({
            "role": "mountain_atlas",
            "source_resources": [name for name, _pixels in mountain_fields],
            "output": str(target),
            "width": 768,
            "height": 768,
            "source_sha256": hashlib.sha256(atlas).hexdigest(),
        })
    report["extracted"] = extracted
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = extract_relief_resources(args.package, args.output)
        authored_analysis = analyze_relief_package(args.package)
        authored_analysis["contact_sheets"] = render_relief_contact_sheets(args.package, args.output)
        report["authored_relief_analysis"] = authored_analysis
        report_path = args.report or args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Extracted {len(report['extracted'])} runtime relief resources and analyzed "
        f"{len(authored_analysis['mountain_variants'])} mountain variants plus "
        f"{len(authored_analysis['hill_families'])} hill families to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
