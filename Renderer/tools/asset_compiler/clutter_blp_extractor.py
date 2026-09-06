#!/usr/bin/env python3
"""Extract verified Civ VI vegetation models into a source-agnostic C3X pack.

This importer is intentionally fail-closed.  It supports only the static CIVBLP
layout and the two vegetation vertex profiles proven in the installed Base
environment/clutter.blp package.  Source-specific names and offsets are written
to the ignored build report, never to runtime-facing pack JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import civblp_probe
from Renderer.tools.asset_compiler.c3x_asset_compiler import (
    DXGI_FORMAT_NAMES,
    make_dds_dx10_header,
    parse_civbig_header,
)
from Renderer.tools.asset_compiler.grassland_pack_builder import validate_runtime_independence


RENDERER_ROOT = Path(__file__).resolve().parents[2]
MAC_BLP_ROOT = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common"
    / "Sid Meier's Civilization VI/Civ6.app/Contents/Assets/Base/Platforms/Windows/BLPs"
)
WINDOWS_BLP_ROOT = Path(
    r"Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI"
    r"\Civ6.app\Contents\Assets\Base\Platforms\Windows\BLPs"
)
DEFAULT_PACK = RENDERER_ROOT / "packs" / "VegetationNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "clutter" / "vegetation_build.json"
RUNTIME_BUNDLE = "vegetation_runtime.bin"

ALLOCATION_RECORD_BYTES = 40
ALLOCATION_SEARCH_BYTES = 8192
POSITION_OFFSETS = (0, 2, 4)
UV0_OFFSET = 8
SOURCE_UNITS_PER_TILE = 12.0
VERTEX_PROFILES = {
    0x6679B170: {"stride": 32, "uv0_encoding": "half2"},
    0x315CFCD9: {"stride": 24, "uv0_encoding": "half2"},
}

FEATURE_SPECS = tuple(
    {
        "source_name": source_name,
        "asset_id": asset_id,
        "manifest_key": manifest_key,
        "stem": stem,
        "group": group,
    }
    for source_name, asset_id, manifest_key, stem, group in (
        ("Tree_Pine_01", "feature.forest.pine_01", "feature/forest/pine_01", "forest_pine_01", "forest"),
        ("Tree_Pine_02", "feature.forest.pine_02", "feature/forest/pine_02", "forest_pine_02", "forest"),
        ("Tree_Pine_03", "feature.forest.pine_03", "feature/forest/pine_03", "forest_pine_03", "forest"),
        ("Tree_Pine_Clump_01", "feature.forest.pine_clump_01", "feature/forest/pine_clump_01", "forest_pine_clump_01", "forest"),
        ("Tree_Pine_Clump_02", "feature.forest.pine_clump_02", "feature/forest/pine_clump_02", "forest_pine_clump_02", "forest"),
        ("Shrub_01", "feature.forest.shrub_01", "feature/forest/shrub_01", "forest_shrub_01", "forest"),
        ("Shrub_02", "feature.forest.shrub_02", "feature/forest/shrub_02", "forest_shrub_02", "forest"),
        ("Tree_Pine_Snow_01", "feature.forest_snow.pine_01", "feature/forest_snow/pine_01", "forest_snow_pine_01", "forest_snow"),
        ("Tree_Pine_Snow_02", "feature.forest_snow.pine_02", "feature/forest_snow/pine_02", "forest_snow_pine_02", "forest_snow"),
        ("Tree_Pine_Snow_03", "feature.forest_snow.pine_03", "feature/forest_snow/pine_03", "forest_snow_pine_03", "forest_snow"),
        ("Tree_Pine_Clump_Snow_01", "feature.forest_snow.pine_clump_01", "feature/forest_snow/pine_clump_01", "forest_snow_pine_clump_01", "forest_snow"),
        ("Tree_Pine_Clump_Snow_02", "feature.forest_snow.pine_clump_02", "feature/forest_snow/pine_clump_02", "forest_snow_pine_clump_02", "forest_snow"),
        ("Jungle_Grass_01", "feature.jungle.grass_01", "feature/jungle/grass_01", "jungle_grass_01", "jungle"),
        ("Jungle_Grass_02", "feature.jungle.grass_02", "feature/jungle/grass_02", "jungle_grass_02", "jungle"),
        ("Jungle_Grass_03", "feature.jungle.grass_03", "feature/jungle/grass_03", "jungle_grass_03", "jungle"),
        ("Jungle_Grass_04", "feature.jungle.grass_04", "feature/jungle/grass_04", "jungle_grass_04", "jungle"),
        ("Jungle_Palm_01", "feature.jungle.palm_01", "feature/jungle/palm_01", "jungle_palm_01", "jungle"),
        ("Jungle_Palm_02", "feature.jungle.palm_02", "feature/jungle/palm_02", "jungle_palm_02", "jungle"),
        ("Jungle_Palm_03", "feature.jungle.palm_03", "feature/jungle/palm_03", "jungle_palm_03", "jungle"),
        ("Jungle_Plant_01", "feature.jungle.plant_01", "feature/jungle/plant_01", "jungle_plant_01", "jungle"),
        ("Jungle_Plant_02", "feature.jungle.plant_02", "feature/jungle/plant_02", "jungle_plant_02", "jungle"),
        ("Jungle_Plant_03", "feature.jungle.plant_03", "feature/jungle/plant_03", "jungle_plant_03", "jungle"),
    )
)

# The Civilization V Environment Skin keeps the generic forest/jungle groups
# but adds its own leafy bodies. Include these only when the selected ArtDef
# actually references them, so the baseline Civ VI build remains unchanged.
ALTERNATE_FEATURE_SPECS = tuple(
    {
        "source_name": source_name,
        "asset_id": asset_id,
        "manifest_key": manifest_key,
        "stem": stem,
        "group": group,
    }
    for source_name, asset_id, manifest_key, stem, group in (
        ("Trees_Leafy_v1_01", "feature.forest.leafy_v1_01", "feature/forest/leafy_v1_01", "forest_leafy_v1_01", "forest"),
        ("Trees_Leafy_v1_02", "feature.forest.leafy_v1_02", "feature/forest/leafy_v1_02", "forest_leafy_v1_02", "forest"),
        ("Trees_Leafy_v1_03", "feature.forest.leafy_v1_03", "feature/forest/leafy_v1_03", "forest_leafy_v1_03", "forest"),
        ("Tree_Leafy_v2_01", "feature.forest.leafy_v2_01", "feature/forest/leafy_v2_01", "forest_leafy_v2_01", "forest"),
        ("Tree_Leafy_v2_02", "feature.forest.leafy_v2_02", "feature/forest/leafy_v2_02", "forest_leafy_v2_02", "forest"),
        ("Tree_Leafy_v2_03", "feature.forest.leafy_v2_03", "feature/forest/leafy_v2_03", "forest_leafy_v2_03", "forest"),
        ("Tree_Leafy_v3_01", "feature.forest.leafy_v3_01", "feature/forest/leafy_v3_01", "forest_leafy_v3_01", "forest"),
        ("Tree_Leafy_v3_02", "feature.forest.leafy_v3_02", "feature/forest/leafy_v3_02", "forest_leafy_v3_02", "forest"),
        ("Tree_Leafy_v3_03", "feature.forest.leafy_v3_03", "feature/forest/leafy_v3_03", "forest_leafy_v3_03", "forest"),
        ("Tree_Leafy_v4_01", "feature.forest.leafy_v4_01", "feature/forest/leafy_v4_01", "forest_leafy_v4_01", "forest"),
        ("Tree_Leafy_v4_02", "feature.forest.leafy_v4_02", "feature/forest/leafy_v4_02", "forest_leafy_v4_02", "forest"),
        ("Tree_Leafy_v4_03", "feature.forest.leafy_v4_03", "feature/forest/leafy_v4_03", "forest_leafy_v4_03", "forest"),
        ("Trees_Leafy_Forest_01", "feature.forest.leafy_clump_01", "feature/forest/leafy_clump_01", "forest_leafy_clump_01", "forest"),
        ("Trees_Leafy_Forest_02", "feature.forest.leafy_clump_02", "feature/forest/leafy_clump_02", "forest_leafy_clump_02", "forest"),
        ("Trees_Leafy_Forest_03", "feature.forest.leafy_clump_03", "feature/forest/leafy_clump_03", "forest_leafy_clump_03", "forest"),
        ("CivV_Jungle_Leafy_v2_01", "feature.jungle.leafy_01", "feature/jungle/leafy_01", "jungle_leafy_01", "jungle"),
        ("CivV_Jungle_Leafy_v2_02", "feature.jungle.leafy_02", "feature/jungle/leafy_02", "jungle_leafy_02", "jungle"),
    )
)

EXCLUDED_ARTDEF_ENTRIES = (
    {"entries": ["Boulder_01", "Boulder_02", "Boulder_Clump_01", "Boulder_Clump_02"], "reason": "rock bodies are routed to the authored-rock intake, not the vegetation sheet"},
    {"entries": ["Boulder_Clump_Snow_01", "Boulder_Clump_Snow_02"], "reason": "snow rock bodies are routed to the authored-rock intake"},
    {"entries": ["Tree_Dirt_Decal01", "Tree_Dirt_Decal02", "Jungle_Decal_01", "Jungle_Decal_02"], "reason": "decal entries have no static feature Model container and are not vegetation bodies"},
    {"entries": ["Jungle_Clump_IOS_01"], "reason": "explicit low-end iOS fallback, not the desktop production set"},
)

ARTDEF_GROUPS = {
    "forest": "CLUTTER_FOREST",
    "forest_snow": "CLUTTER_FOREST_SNOW",
    "jungle": "CLUTTER_JUNGLE",
}

TYPE_STRING = (
    "String::BasicT<Serialization::StaticPackageAllocator<Platform::StaticHeapAllocator<5, 0>>, "
    "String::ASCII>"
)
TYPE_PACKAGE_MODEL = "Granny::PackageModel"
TYPE_PACKAGE_MODEL_POINTER = "PackagePtr64<Granny::PackageModel>"
TYPE_BASE_MODEL = "ModelPackageEntry::BaseModelData_Entry"
TYPE_MODEL = "FGXModel::ContainerDesc::Model"
TYPE_MESH = "FGXModel::ContainerDesc::Mesh"
TYPE_PRIM_GROUP = "FGXModel::ContainerDesc::PrimGroup"
TYPE_MATERIAL = "FGXModel::ContainerDesc::Material"
TYPE_USER_DATA_POINTER = "BLP::BLPPtr<FGXModel::IUserData>"
TYPE_PRIM_DATA = "ModelPrimGroupData"
TYPE_MATERIAL_DATA = "ModelMaterialData"
TYPE_VERTEX_BUFFER = "BLP::VertexBufferEntry"
TYPE_INDEX_BUFFER = "BLP::IndexBufferEntry"
TYPE_TEXTURE = "BLP::TextureEntry"
TYPE_LANDMARK = "LandmarkPackageEntry"
TYPE_CITY_BLOCK = "CityBlockPackageEntry"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_allocation_candidate(package: bytes, start: int) -> list[dict[str, Any]] | None:
    allocations: list[dict[str, Any]] = []
    cursor = start
    while cursor + ALLOCATION_RECORD_BYTES <= len(package):
        chunk = package[cursor : cursor + ALLOCATION_RECORD_BYTES]
        if not any(chunk):
            if any(package[cursor:]):
                return None
            break
        allocation = civblp_probe.unpack_allocation(package, cursor)
        if (
            allocation["stripe"] not in (0, 1)
            or allocation["allocation_type"] != 0
            or allocation["padding_hex"] != "00000000"
            or allocation["target_offset"] > len(package)
            or allocation["size"] > len(package) * 8
            or allocation["element_count"] > 1_000_000
        ):
            return None
        allocations.append(allocation)
        cursor += ALLOCATION_RECORD_BYTES
    if len(allocations) < 3 or any(package[cursor:]):
        return None
    first = allocations[0]
    if (
        first["stripe"] != 0
        or first["parent_pointer"] != 0
        or first["target_offset"] != 0
        or first["size"] == 0
        or first["element_count"] == 0
        or first["size"] % first["element_count"]
    ):
        return None
    count = len(allocations)
    if any(
        allocation["type_pointer"] < 1
        or allocation["type_pointer"] > count
        or allocation["parent_pointer"] > count
        for allocation in allocations
    ):
        return None
    return allocations


def find_static_allocation_table(package: bytes) -> tuple[int, list[dict[str, Any]]]:
    marker = package.rfind(civblp_probe.ENTRY_MAP_TYPE)
    if marker < 0:
        raise ValueError("Static package has no reflected entry-map type marker")
    search_start = marker + len(civblp_probe.ENTRY_MAP_TYPE)
    search_end = min(len(package) - ALLOCATION_RECORD_BYTES, search_start + ALLOCATION_SEARCH_BYTES)
    candidates: list[tuple[int, list[dict[str, Any]]]] = []
    for start in range(search_start, search_end + 1):
        if package[start] not in (0, 1) or package[start + 2 : start + 6] != b"\x00" * 4:
            continue
        parsed = _parse_allocation_candidate(package, start)
        if parsed is not None:
            candidates.append((start, parsed))
    if not candidates:
        raise ValueError("Could not locate the static package allocation table")
    largest = max(len(candidate[1]) for candidate in candidates)
    winners = [candidate for candidate in candidates if len(candidate[1]) == largest]
    if len(winners) != 1:
        raise ValueError("Static package allocation-table selection is ambiguous")
    return winners[0]


class StaticPackage:
    def __init__(self, source: Path, target_string: str) -> None:
        self.source = source
        self.data = source.read_bytes()
        self.header = civblp_probe.parse_file_header(self.data[:28], len(self.data))
        self.package_file_offset = self.header["package_data"]["offset"]
        self.big_data_file_offset = self.header["big_data"]["offset"]
        self.package = self.data[self.package_file_offset : self.big_data_file_offset]
        self.table_offset, self.allocations = find_static_allocation_table(self.package)
        temp_base, self.temp_evidence = civblp_probe.infer_temp_stripe_base(
            self.package, self.allocations
        )
        package_base, self.target_char_pointer, self.package_evidence = (
            civblp_probe.infer_package_stripe_base(
                self.package, self.allocations, temp_base, target_string
            )
        )
        self.stripe_bases = {0: package_base, 1: temp_base}

    def type_name(self, pointer: int) -> str | None:
        if pointer < 1 or pointer > len(self.allocations):
            return None
        return civblp_probe.resolve_type_name(
            self.allocations[pointer - 1]["type_pointer"],
            self.package,
            self.allocations,
            self.stripe_bases,
        )

    def resolve(self, pointer: int) -> tuple[int, int]:
        result = civblp_probe.resolve_allocation_target(
            pointer, self.allocations, self.stripe_bases
        )
        if result is None:
            raise ValueError(f"Allocation pointer {pointer} does not resolve")
        offset, size = result
        if offset < 0 or size < 0 or offset + size > len(self.package):
            raise ValueError(f"Allocation pointer {pointer} resolves outside package data")
        return offset, size

    def bytes_for(self, pointer: int) -> bytes:
        offset, size = self.resolve(pointer)
        return self.package[offset : offset + size]

    def direct_string(self, pointer: int) -> str | None:
        if self.type_name(pointer) != civblp_probe.CHAR_TYPE:
            return None
        allocation = self.allocations[pointer - 1]
        offset, _size = self.resolve(pointer)
        value = civblp_probe.read_allocated_string(
            self.package, offset, allocation["size"]
        )
        return None if value is None else value[0]

    def select_direct_string(self, value: str) -> int:
        pointers = [
            pointer
            for pointer in range(1, len(self.allocations) + 1)
            if self.direct_string(pointer) == value
        ]
        if len(pointers) != 1:
            raise ValueError(f"Expected source asset name once, found {len(pointers)}: {value}")
        self.target_char_pointer = pointers[0]
        return pointers[0]

    def string_value(self, pointer: int) -> str | None:
        direct = self.direct_string(pointer)
        if direct is not None:
            return direct
        if self.type_name(pointer) != TYPE_STRING:
            return None
        raw = self.bytes_for(pointer)
        strings = []
        for offset in range(0, len(raw) - 7, 8):
            child = struct.unpack_from("<Q", raw, offset)[0]
            value = self.direct_string(child)
            if value is not None:
                strings.append(value)
        return strings[0] if len(strings) == 1 else None

    def pointer_fields(self, owner: int, target_type: str) -> list[tuple[int, int]]:
        raw = self.bytes_for(owner)
        result = []
        for offset in range(0, len(raw) - 7, 8):
            pointer = struct.unpack_from("<Q", raw, offset)[0]
            if self.type_name(pointer) == target_type:
                result.append((offset, pointer))
        return result

    def references_to(self, target: int, expected_type: str) -> list[tuple[int, int]]:
        result = []
        for pointer in range(1, len(self.allocations) + 1):
            if self.type_name(pointer) != expected_type:
                continue
            raw = self.bytes_for(pointer)
            for offset in range(0, len(raw) - 7, 8):
                if struct.unpack_from("<Q", raw, offset)[0] == target:
                    result.append((pointer, offset))
        return result

    def unique_reference(self, target: int, expected_type: str) -> int:
        references = self.references_to(target, expected_type)
        if len(references) != 1:
            raise ValueError(
                f"Expected one {expected_type} reference to pointer {target}, found {len(references)}"
            )
        return references[0][0]

    def unique_pointer_field(self, owner: int, target_type: str) -> int:
        fields = self.pointer_fields(owner, target_type)
        if len(fields) != 1:
            raise ValueError(
                f"Expected one {target_type} pointer in allocation {owner}, found {len(fields)}"
            )
        return fields[0][1]

    def unique_allocation(self, target_type: str) -> int:
        pointers = [
            pointer
            for pointer in range(1, len(self.allocations) + 1)
            if self.type_name(pointer) == target_type
        ]
        if len(pointers) != 1:
            raise ValueError(f"Expected one {target_type} allocation, found {len(pointers)}")
        return pointers[0]

    def array_element(self, pointer: int, index: int) -> bytes:
        allocation = self.allocations[pointer - 1]
        count = allocation["element_count"]
        raw = self.bytes_for(pointer)
        if count < 1 or len(raw) % count:
            raise ValueError(f"Allocation {pointer} is not a regular typed array")
        if index < 0 or index >= count:
            raise ValueError(f"Array index {index} is outside allocation {pointer}")
        stride = len(raw) // count
        return raw[index * stride : (index + 1) * stride]

    def big_data(self, offset: int, size: int) -> bytes:
        start = self.big_data_file_offset + offset
        end = start + size
        if offset < 0 or size < 0 or end > len(self.data):
            raise ValueError("Embedded big-data range is outside the package")
        return self.data[start:end]


def decode_buffer_entry(package: StaticPackage, pointer: int, index: int, vertex: bool) -> dict[str, Any]:
    raw = package.array_element(pointer, index)
    if len(raw) != 72:
        raise ValueError("Expected a 72-byte Civ VI GPU buffer entry")
    name_pointer = struct.unpack_from("<Q", raw, 0x08)[0]
    name = package.string_value(name_pointer)
    if name is None:
        raise ValueError("GPU buffer entry name does not resolve")
    result = {
        "index": index,
        "name": name,
        "offset": struct.unpack_from("<Q", raw, 0x20)[0],
        "bytes": struct.unpack_from("<I", raw, 0x28)[0],
        "flags": struct.unpack_from("<I", raw, 0x2C)[0],
        "name_hash": struct.unpack_from("<I", raw, 0x30)[0],
    }
    if vertex:
        result["format"] = struct.unpack_from("<I", raw, 0x40)[0]
        result["count"] = struct.unpack_from("<I", raw, 0x44)[0]
        if not result["count"] or result["bytes"] % result["count"]:
            raise ValueError("Vertex buffer byte count is not divisible by its vertex count")
        result["stride"] = result["bytes"] // result["count"]
    else:
        result["bytes_per_index"] = struct.unpack_from("<I", raw, 0x40)[0]
        result["count"] = struct.unpack_from("<I", raw, 0x44)[0]
        if result["bytes"] != result["bytes_per_index"] * result["count"]:
            raise ValueError("Index buffer byte count does not match its declared layout")
    return result


def decode_texture_entry(package: StaticPackage, pointer: int, index: int) -> dict[str, Any]:
    raw = package.array_element(pointer, index)
    if len(raw) != 104:
        raise ValueError("Expected a 104-byte Civ VI texture entry")
    name = package.string_value(struct.unpack_from("<Q", raw, 0x08)[0])
    texture_class = package.string_value(struct.unpack_from("<Q", raw, 0x40)[0])
    if name is None or texture_class is None:
        raise ValueError("Texture name or class does not resolve")
    return {"index": index, "name": name, "class": texture_class}


def normalize_mesh(
    vertex_bytes: bytes,
    index_bytes: bytes,
    vertex_entry: dict[str, Any],
    index_entry: dict[str, Any],
    primitive: dict[str, int],
    asset_id: str,
    allow_wrapping_uvs: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = VERTEX_PROFILES.get(vertex_entry["format"])
    if profile is None or vertex_entry["stride"] != profile["stride"]:
        raise ValueError(
            f"Unsupported vertex profile format=0x{vertex_entry['format']:08x} "
            f"stride={vertex_entry['stride']}"
        )
    if index_entry["bytes_per_index"] != 2:
        raise ValueError("The proven clutter profile requires 16-bit indices")
    vertex_count = vertex_entry["count"]
    stride = profile["stride"]
    if len(vertex_bytes) != vertex_count * stride:
        raise ValueError("Vertex payload does not match the proven profile")
    all_indices = list(struct.unpack(f"<{index_entry['count']}H", index_bytes))
    first = primitive["first_index"]
    count = primitive["index_count"]
    if count < 3 or count % 3 or first + count > len(all_indices):
        raise ValueError("Primitive group has an invalid triangle index range")
    indices = [index + primitive["base_vertex"] for index in all_indices[first : first + count]]
    if min(indices) < 0 or max(indices) >= vertex_count:
        raise ValueError("Primitive group contains an out-of-range vertex index")
    if primitive["vertex_count"] != vertex_count or len(set(indices)) != vertex_count:
        raise ValueError("Primitive group does not reference its complete declared vertex range")

    source_positions = [
        tuple(
            struct.unpack_from("<e", vertex_bytes, vertex * stride + offset)[0]
            for offset in POSITION_OFFSETS
        )
        for vertex in range(vertex_count)
    ]
    source_uvs = [
        tuple(struct.unpack_from("<ee", vertex_bytes, vertex * stride + UV0_OFFSET))
        for vertex in range(vertex_count)
    ]
    if any(not all(math.isfinite(value) for value in position) for position in source_positions):
        raise ValueError("Vertex payload contains a non-finite position")
    if any(not all(math.isfinite(value) for value in uv) for uv in source_uvs):
        raise ValueError("Vertex payload contains a non-finite UV coordinate")
    wraps_uv0 = any(
        not (0.0 <= uv[0] <= 1.0 and 0.0 <= uv[1] <= 1.0) for uv in source_uvs
    )
    if wraps_uv0 and not allow_wrapping_uvs:
        raise ValueError("Proven packed UV profile produced a coordinate outside 0..1")

    source_min = [min(position[axis] for position in source_positions) for axis in range(3)]
    source_max = [max(position[axis] for position in source_positions) for axis in range(3)]
    source_height = source_max[2] - source_min[2]
    if source_height <= 0.0:
        raise ValueError("Source mesh has no positive height")
    center_x = (source_min[0] + source_max[0]) * 0.5
    center_y = (source_min[1] + source_max[1]) * 0.5
    positions = [
        (
            (position[0] - center_x) / SOURCE_UNITS_PER_TILE,
            (position[1] - center_y) / SOURCE_UNITS_PER_TILE,
            (position[2] - source_min[2]) / SOURCE_UNITS_PER_TILE,
        )
        for position in source_positions
    ]

    normal_sums = [[0.0, 0.0, 0.0] for _ in positions]
    triangle_areas = []
    for start in range(0, len(indices), 3):
        ia, ib, ic = indices[start : start + 3]
        a, b, c = positions[ia], positions[ib], positions[ic]
        ab = tuple(b[axis] - a[axis] for axis in range(3))
        ac = tuple(c[axis] - a[axis] for axis in range(3))
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        area2 = math.sqrt(sum(component * component for component in cross))
        if area2 <= 1.0e-10:
            raise ValueError(f"Primitive group contains a degenerate triangle at {start // 3}")
        triangle_areas.append(area2 * 0.5)
        for index in (ia, ib, ic):
            for axis in range(3):
                normal_sums[index][axis] += cross[axis]
    normals = []
    for index, value in enumerate(normal_sums):
        length = math.sqrt(sum(component * component for component in value))
        if length <= 1.0e-10:
            raise ValueError(f"Vertex {index} has no computable geometric normal")
        normals.append(tuple(component / length for component in value))

    def rounded(values: tuple[float, ...] | list[float]) -> list[float]:
        return [round(value, 8) for value in values]

    normalized_min = [min(position[axis] for position in positions) for axis in range(3)]
    normalized_max = [max(position[axis] for position in positions) for axis in range(3)]
    mesh = {
        "schema": "c3x.normalized_mesh.v0",
        "asset_id": asset_id,
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Z",
            "horizontal_axes": ["X", "Y"],
            "position_unit": "tile",
            "uv0_origin": "upper_left",
            "uv0_address_mode": "wrap" if wraps_uv0 else "clamp",
        },
        "topology": {
            "primitive": "triangles",
            "front_face": "counter_clockwise",
            "indices": indices,
        },
        "vertices": [
            {"position": rounded(position), "normal": rounded(normal), "uv0": rounded(uv)}
            for position, normal, uv in zip(positions, normals, source_uvs)
        ],
        "bounds": {"minimum": rounded(normalized_min), "maximum": rounded(normalized_max)},
        "material_slots": [
            {
                "slot": 0,
                "name": "feature_surface",
                "triangle_start": 0,
                "triangle_count": len(indices) // 3,
            }
        ],
        "provenance": {
            "kind": "local_normalized_import",
            "adapter": "c3x.feature_mesh.v0",
            "source_format_dependency": None,
        },
    }
    evidence = {
        "source_bounds": {"minimum": rounded(source_min), "maximum": rounded(source_max)},
        "normalization": {
            "horizontal_center": rounded((center_x, center_y)),
            "ground_z": source_min[2],
            "uniform_scale": 1.0 / SOURCE_UNITS_PER_TILE,
            "source_units_per_tile": SOURCE_UNITS_PER_TILE,
        },
        "vertices": vertex_count,
        "indices": len(indices),
        "triangles": len(indices) // 3,
        "unique_uv0": len(set(source_uvs)),
        "uv0_address_mode": "wrap" if wraps_uv0 else "clamp",
        "triangle_area_range": [min(triangle_areas), max(triangle_areas)],
    }
    return mesh, evidence


def extract_civbig_texture(source: Path, target: Path) -> dict[str, Any]:
    data = source.read_bytes()
    info = parse_civbig_header(data)
    payload = data[48 : 48 + info["payload_bytes"]]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(make_dds_dx10_header(info) + payload)
    format_name = DXGI_FORMAT_NAMES[info["dxgi_format"]]
    return {
        **info,
        "format_name": format_name,
        "color_space": "srgb" if format_name.endswith("_SRGB") else "linear",
        "source_sha256": sha256_bytes(data),
        "dds_sha256": sha256_bytes(target.read_bytes()),
    }


def _bundle_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if not encoded or len(encoded) > 4096:
        raise ValueError("Runtime-bundle string has an invalid length")
    return struct.pack("<I", len(encoded)) + encoded


def write_runtime_bundle(pack: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    asset_items = list(manifest["assets"].items())
    asset_indices = {asset_id: index for index, (asset_id, _asset) in enumerate(asset_items)}
    texture_indices: dict[str, int] = {}
    texture_paths: list[str] = []
    encoded_assets = []
    for asset_id, asset in asset_items:
        mesh = json.loads((pack / asset["mesh"]).read_text(encoding="utf-8"))
        material = json.loads((pack / asset["material"]).read_text(encoding="utf-8"))
        texture_path = material["base_color"]["texture"]
        texture_hash = sha256_bytes((pack / texture_path).read_bytes())
        if texture_hash not in texture_indices:
            texture_indices[texture_hash] = len(texture_paths)
            texture_paths.append(texture_path)
        vertices = mesh["vertices"]
        indices = mesh["topology"]["indices"]
        payload = bytearray()
        payload.extend(_bundle_string(asset_id))
        payload.extend(
            struct.pack(
                "<III",
                texture_indices[texture_hash],
                len(vertices),
                len(indices),
            )
        )
        for vertex in vertices:
            payload.extend(
                struct.pack(
                    "<8f", *(vertex["position"] + vertex["normal"] + vertex["uv0"])
                )
            )
        payload.extend(struct.pack(f"<{len(indices)}I", *indices))
        encoded_assets.append(bytes(payload))
    if len(texture_paths) > 8:
        raise ValueError("Terrain Lab runtime bundle supports at most eight base-color textures")

    encoded_groups = []
    for group_name, feature_group in manifest["features"].items():
        payload = bytearray(_bundle_string(group_name))
        placements = feature_group["placements"]
        payload.extend(struct.pack("<I", len(placements)))
        for placement in placements:
            flags = (
                (1 if placement["allow_overlap"] else 0)
                | (2 if placement["show_decal"] else 0)
                | (4 if placement["is_center_model"] else 0)
            )
            payload.extend(
                struct.pack(
                    "<IffIIIIff",
                    asset_indices[placement["asset"]],
                    placement["scale"],
                    placement["scale_variation"],
                    placement["count"],
                    placement["min_count"],
                    placement["priority"],
                    flags,
                    placement["width"],
                    placement["low_end_reduction"],
                )
            )
        encoded_groups.append(bytes(payload))

    output = bytearray(b"C3XVEG1\0")
    output.extend(
        struct.pack("<IIII", 1, len(texture_paths), len(encoded_assets), len(encoded_groups))
    )
    for texture_path in texture_paths:
        output.extend(_bundle_string(texture_path))
    for asset in encoded_assets:
        output.extend(asset)
    for group in encoded_groups:
        output.extend(group)
    target = pack / RUNTIME_BUNDLE
    target.write_bytes(output)
    lowered = bytes(output).lower()
    for forbidden in (b"civ6", b"civilization vi", b"environment/clutter", b"landmarkpackage"):
        if forbidden in lowered:
            raise ValueError("Runtime bundle contains a source-specific identifier")
    return {
        "path": RUNTIME_BUNDLE,
        "bytes": len(output),
        "sha256": sha256_bytes(output),
        "textures": len(texture_paths),
        "assets": len(encoded_assets),
        "groups": len(encoded_groups),
    }


def landmark_base_model(package: StaticPackage) -> tuple[int, int, int]:
    landmarks = []
    for pointer in range(1, len(package.allocations) + 1):
        if package.type_name(pointer) not in (TYPE_LANDMARK, TYPE_CITY_BLOCK):
            continue
        raw = package.bytes_for(pointer)
        if len(raw) >= 0x40 and struct.unpack_from("<Q", raw, 0x38)[0] == package.target_char_pointer:
            landmarks.append(pointer)
    if len(landmarks) != 1:
        raise ValueError(
            f"Expected one landmark/city-block entry for selected asset, found {len(landmarks)}"
        )
    landmark = landmarks[0]
    landmark_raw = package.bytes_for(landmark)
    user_data = struct.unpack_from("<Q", landmark_raw, 0x20)[0]
    if package.type_name(user_data) != TYPE_USER_DATA_POINTER:
        raise ValueError("Landmark user-data pointer has an unexpected type")
    user_raw = package.bytes_for(user_data)
    if len(user_raw) < 8:
        raise ValueError("Landmark user-data record is truncated")
    base_model = struct.unpack_from("<Q", user_raw, 0)[0]
    if package.type_name(base_model) != TYPE_BASE_MODEL:
        raise ValueError("Landmark does not resolve to a base-model record")
    return landmark, user_data, base_model


def build_feature(
    package: StaticPackage,
    shared_data: Path,
    pack: Path,
    spec: dict[str, str],
    *,
    allow_wrapping_uvs: bool = False,
    allow_optional_maps: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    asset_name = spec["source_name"]
    package.select_direct_string(asset_name)
    landmark, landmark_user_data, base_model = landmark_base_model(package)

    container_pointers = {
        "models": package.unique_pointer_field(base_model, TYPE_MODEL),
        "meshes": package.unique_pointer_field(base_model, TYPE_MESH),
        "primitive_groups": package.unique_pointer_field(base_model, TYPE_PRIM_GROUP),
        "materials": package.unique_pointer_field(base_model, TYPE_MATERIAL),
    }
    for role, pointer in container_pointers.items():
        count = package.allocations[pointer - 1]["element_count"]
        if count != 1:
            raise ValueError(f"Feature extraction requires exactly one {role} record, found {count}")

    prim_user_data = package.unique_pointer_field(
        container_pointers["primitive_groups"], TYPE_USER_DATA_POINTER
    )
    prim_data_candidates = [
        struct.unpack_from("<Q", package.bytes_for(prim_user_data), offset)[0]
        for offset in range(0, len(package.bytes_for(prim_user_data)) - 7, 8)
    ]
    prim_data_candidates = [
        pointer for pointer in prim_data_candidates if package.type_name(pointer) == TYPE_PRIM_DATA
    ]
    if len(prim_data_candidates) != 1:
        raise ValueError("Feature primitive group does not have one ModelPrimGroupData record")
    prim_data_pointer = prim_data_candidates[0]
    prim_raw = package.bytes_for(prim_data_pointer)
    if len(prim_raw) != 32:
        raise ValueError("Expected a 32-byte ModelPrimGroupData record")
    values = struct.unpack_from("<6I", prim_raw, 8)
    primitive = {
        "vertex_buffer": values[0],
        "index_buffer": values[1],
        "first_index": values[2],
        "index_count": values[3],
        "base_vertex": values[4],
        "vertex_count": values[5],
    }

    vertex_array = package.unique_allocation(TYPE_VERTEX_BUFFER)
    index_array = package.unique_allocation(TYPE_INDEX_BUFFER)
    vertex_entry = decode_buffer_entry(package, vertex_array, primitive["vertex_buffer"], True)
    index_entry = decode_buffer_entry(package, index_array, primitive["index_buffer"], False)
    if vertex_entry["name"] != index_entry["name"]:
        raise ValueError("Selected vertex and index buffer names do not match")
    if primitive["index_count"] != index_entry["count"]:
        raise ValueError("Primitive index range does not cover the selected index buffer")
    vertex_payload = package.big_data(vertex_entry["offset"], vertex_entry["bytes"])
    index_payload = package.big_data(index_entry["offset"], index_entry["bytes"])
    mesh, mesh_evidence = normalize_mesh(
        vertex_payload,
        index_payload,
        vertex_entry,
        index_entry,
        primitive,
        spec["asset_id"],
        allow_wrapping_uvs,
    )

    material_user_data = package.unique_pointer_field(
        container_pointers["materials"], TYPE_USER_DATA_POINTER
    )
    material_data_candidates = [
        struct.unpack_from("<Q", package.bytes_for(material_user_data), offset)[0]
        for offset in range(0, len(package.bytes_for(material_user_data)) - 7, 8)
    ]
    material_data_candidates = [
        pointer
        for pointer in material_data_candidates
        if package.type_name(pointer) == TYPE_MATERIAL_DATA
    ]
    if len(material_data_candidates) != 1:
        raise ValueError("Feature material does not have one ModelMaterialData record")
    material_raw = package.bytes_for(material_data_candidates[0])
    if len(material_raw) != 64:
        raise ValueError("Expected a 64-byte ModelMaterialData record")
    texture_indices = {
        "normal_0": struct.unpack_from("<I", material_raw, 0x1C)[0],
        "normal_1": struct.unpack_from("<I", material_raw, 0x20)[0],
        "base_color": struct.unpack_from("<I", material_raw, 0x24)[0],
        "gloss": struct.unpack_from("<I", material_raw, 0x28)[0],
        "emissive": struct.unpack_from("<I", material_raw, 0x3C)[0],
    }
    texture_array = package.unique_allocation(TYPE_TEXTURE)
    texture_entries = {}
    for role, index in texture_indices.items():
        if index == 0xFFFFFFFF:
            if role == "emissive" or (role != "base_color" and allow_optional_maps):
                continue
            raise ValueError(f"Feature is missing required {role} texture")
        texture_entries[role] = decode_texture_entry(package, texture_array, index)
    expected_classes = {
        "normal_0": "LEAN",
        "normal_1": "LEAN",
        "base_color": "Generic_BaseColor",
        "gloss": "Generic_Gloss",
        "emissive": "Generic_Emissive",
    }
    for role, entry in texture_entries.items():
        expected = expected_classes[role]
        if entry["class"] != expected:
            raise ValueError(
                f"Feature {role} texture class is {texture_entries[role]['class']}, expected {expected}"
            )

    texture_outputs = {
        role: f"textures/features/{spec['stem']}_{role}.dds"
        for role in texture_entries
    }
    texture_evidence = {}
    for role, relative in texture_outputs.items():
        source = shared_data / texture_entries[role]["name"]
        if not source.is_file():
            raise ValueError(f"Missing selected standalone texture: {source}")
        texture_evidence[role] = {
            **texture_entries[role],
            **extract_civbig_texture(source, pack / relative),
            "source": str(source),
        }

    mesh_relative = f"meshes/features/{spec['stem']}.json"
    material_relative = f"materials/features/{spec['stem']}.json"
    write_json(pack / mesh_relative, mesh)
    base_info = texture_evidence["base_color"]
    address_mode = mesh["coordinate_system"]["uv0_address_mode"]
    material = {
        "schema": "c3x.material.v0",
        "name": spec["stem"],
        "base_color": {
            "texture": texture_outputs["base_color"],
            "format": base_info["format_name"],
            "color_space": base_info["color_space"],
            "uv_channel": "uv0",
            "address_mode_u": address_mode,
            "address_mode_v": address_mode,
        },
        "alpha_mode": "opaque",
        "status": "normalized_local_import",
    }
    if "gloss" in texture_outputs:
        material["gloss"] = {
            "texture": texture_outputs["gloss"],
            "uv_channel": "uv0",
            "address_mode_u": address_mode,
            "address_mode_v": address_mode,
        }
    if "normal_0" in texture_outputs and "normal_1" in texture_outputs:
        material["lean_normal"] = {
            "texture_0": texture_outputs["normal_0"],
            "texture_1": texture_outputs["normal_1"],
            "uv_channel": "uv0",
            "address_mode_u": address_mode,
            "address_mode_v": address_mode,
        }
    if "emissive" in texture_outputs:
        material["emissive"] = {
            "mask": texture_outputs["emissive"],
            "color": [1.0, 1.0, 1.0],
            "intensity": 1.0,
            "activation": "night",
            "missing_policy": "non-emissive",
        }
    write_json(pack / material_relative, material)
    manifest_asset = {
        "type": "feature",
        "mesh": mesh_relative,
        "material": material_relative,
    }
    report = {
        "selected_asset": asset_name,
        "normalized_asset_id": spec["asset_id"],
        "pointer_chain": {
            "name_char": package.target_char_pointer,
            "landmark": landmark,
            "landmark_user_data": landmark_user_data,
            "base_model": base_model,
            **container_pointers,
            "primitive_data": prim_data_pointer,
            "material_data": material_data_candidates[0],
        },
        "primitive": primitive,
        "vertex_buffer": {**vertex_entry, "sha256": sha256_bytes(vertex_payload)},
        "index_buffer": {**index_entry, "sha256": sha256_bytes(index_payload)},
        "mesh": mesh_evidence,
        "textures": texture_evidence,
    }
    return manifest_asset, report


def _artdef_value(value: ET.Element) -> str | None:
    child = next((child for child in value if child.tag != "m_ParamName"), None)
    if child is None:
        return None
    return child.attrib.get("text", child.text)


def read_artdef_placements(path: Path) -> dict[str, list[dict[str, Any]]]:
    root = ET.parse(path).getroot()
    result: dict[str, list[dict[str, Any]]] = {}
    for group, set_name in ARTDEF_GROUPS.items():
        matches = [
            element
            for element in root.iter("Element")
            if element.find("m_Name") is not None
            and element.find("m_Name").attrib.get("text") == set_name
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one ArtDef clutter set {set_name}, found {len(matches)}")
        plant_collections = [
            collection
            for collection in matches[0].findall("./m_ChildCollections/Element")
            if collection.find("m_CollectionName") is not None
            and collection.find("m_CollectionName").attrib.get("text") == "Plants"
        ]
        if len(plant_collections) != 1:
            raise ValueError(f"Expected one Plants collection in {set_name}")
        placements = []
        for item in plant_collections[0].findall("Element"):
            values = {}
            for value in item.findall("./m_Fields/m_Values/Element"):
                parameter = value.find("m_ParamName")
                if parameter is not None:
                    values[parameter.attrib["text"]] = _artdef_value(value)
            required = ("Asset", "Scale", "Count", "ScaleVariation")
            if any(values.get(key) is None for key in required):
                raise ValueError(f"Incomplete placement record in {set_name}")
            placements.append(
                {
                    "source_asset": values["Asset"],
                    "scale": float(values["Scale"]),
                    "count": int(values["Count"]),
                    "scale_variation": float(values["ScaleVariation"]),
                    "low_end_reduction": float(values.get("LowendReduction", 0.0)),
                    "show_decal": values.get("ShowDecal", "false").lower() == "true",
                    "priority": int(values.get("Priority", 0)),
                    "width": float(values.get("Width", 0.0)),
                    "rotate_mode": values.get("RotateMode", "RotateZ"),
                    "is_center_model": values.get("IsCenterModel", "false").lower() == "true",
                    "allow_overlap": values.get("AllowOverlap", "false").lower() == "true",
                    "min_count": int(values.get("MinCount", 0)),
                }
            )
        result[group] = placements
    return result


def build_vegetation_pack(
    package_path: Path,
    shared_data: Path,
    artdef_path: Path,
    pack: Path,
    report_path: Path,
) -> dict[str, Any]:
    artdef_placements = read_artdef_placements(artdef_path)
    artdef_names = {
        placement["source_asset"]
        for placements in artdef_placements.values()
        for placement in placements
    }
    specs = FEATURE_SPECS + tuple(
        spec for spec in ALTERNATE_FEATURE_SPECS if spec["source_name"] in artdef_names
    )
    assets = {}
    reports = []
    feature_groups: dict[str, list[str]] = {}
    package = StaticPackage(package_path, specs[0]["source_name"])
    alternate_source_names = {
        spec["source_name"] for spec in ALTERNATE_FEATURE_SPECS
    }
    for spec in specs:
        manifest_asset, report = build_feature(
            package,
            shared_data,
            pack,
            spec,
            allow_wrapping_uvs=spec["source_name"] in alternate_source_names,
        )
        assets[spec["manifest_key"]] = manifest_asset
        feature_groups.setdefault(spec["group"], []).append(spec["manifest_key"])
        reports.append(report)

    source_to_key = {spec["source_name"]: spec["manifest_key"] for spec in specs}
    excluded_names = {
        name for exclusion in EXCLUDED_ARTDEF_ENTRIES for name in exclusion["entries"]
    }
    unaccounted = sorted(artdef_names - set(source_to_key) - excluded_names)
    if unaccounted:
        raise ValueError("Unaccounted ArtDef clutter entries: " + ", ".join(unaccounted))
    normalized_placements: dict[str, list[dict[str, Any]]] = {}
    preview_scales: dict[str, float] = {}
    for group, placements in artdef_placements.items():
        normalized_placements[group] = []
        for placement in placements:
            asset_key = source_to_key.get(placement["source_asset"])
            if asset_key is None or asset_key not in feature_groups[group]:
                continue
            runtime_placement = {
                key: value for key, value in placement.items() if key != "source_asset"
            }
            runtime_placement["asset"] = asset_key
            normalized_placements[group].append(runtime_placement)
            if placement["scale"] > 0.0 and asset_key not in preview_scales:
                preview_scales[asset_key] = placement["scale"]
    for asset_key, asset in assets.items():
        if asset_key not in preview_scales:
            raise ValueError(f"No positive ArtDef scale resolves for {asset_key}")
        asset["preview_scale"] = preview_scales[asset_key]

    manifest = {
        "schema": "c3x.asset_pack.v0",
        "name": "VegetationNormalized",
        "display_name": "Normalized Vegetation",
        "source_policy": "Local licensed-source import; derived art is not redistributable.",
        "projection": {
            "tile_width_px": 128,
            "tile_height_px": 64,
            "height_scale_px": 96,
            "basis": {"x": [64, 32], "y": [-64, 32], "z": [0, -96]},
        },
        "assets": assets,
        "features": {
            group: {
                "variants": variants,
                "placements": normalized_placements[group],
                "status": "normalized_verified_set",
            }
            for group, variants in feature_groups.items()
        },
    }
    write_json(pack / "manifest.json", manifest)
    bundle_evidence = write_runtime_bundle(pack, manifest)
    independence_errors = validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError("Runtime pack is source-dependent: " + "; ".join(independence_errors))

    report = {
        "schema": "c3x.civ6_clutter_extract.v0",
        "source": str(package_path),
        "source_sha256": sha256_bytes(package.data),
        "artdef": {"source": str(artdef_path), "source_sha256": sha256_bytes(artdef_path.read_bytes())},
        "allocation_table": {
            "package_offset": package.table_offset,
            "allocation_count": len(package.allocations),
            "stripe_bases": package.stripe_bases,
        },
        "assets": reports,
        "excluded_artdef_entries": EXCLUDED_ARTDEF_ENTRIES,
        "artdef_coverage": {
            "unique_entries": len(artdef_names),
            "normalized_bodies": len(set(source_to_key) & artdef_names),
            "explicitly_excluded_or_routed": len(excluded_names & artdef_names),
            "unaccounted": [],
        },
        "pack": str(pack),
        "runtime_independence": "passed",
        "runtime_bundle": bundle_evidence,
    }
    write_json(report_path, report)
    return report


def default_blp_root() -> Path:
    return MAC_BLP_ROOT if MAC_BLP_ROOT.is_dir() else WINDOWS_BLP_ROOT


def main(argv: list[str] | None = None) -> int:
    root = default_blp_root()
    parser = argparse.ArgumentParser(description="Extract verified vegetation into a C3X pack")
    parser.add_argument("--package", type=Path, default=root / "environment" / "clutter.blp")
    parser.add_argument("--shared-data", type=Path, default=root / "SHARED_DATA")
    parser.add_argument("--artdef", type=Path, default=root.parents[2] / "ArtDefs" / "Clutter.artdef")
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        report = build_vegetation_pack(
            args.package, args.shared_data, args.artdef, args.pack, args.report
        )
    except (OSError, ValueError, struct.error, ET.ParseError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for asset in report["assets"]:
        print(
            f"Extracted {asset['selected_asset']}: {asset['mesh']['vertices']} vertices, "
            f"{asset['mesh']['triangles']} triangles"
        )
    print(f"Pack: {args.pack}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
