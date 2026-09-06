#!/usr/bin/env python3
"""Compile compound Landmark/TileBase assets into generic component records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler.clutter_blp_extractor import (
    TYPE_INDEX_BUFFER,
    TYPE_MATERIAL,
    TYPE_MATERIAL_DATA,
    TYPE_MESH,
    TYPE_MODEL,
    TYPE_PRIM_DATA,
    TYPE_PRIM_GROUP,
    TYPE_TEXTURE,
    TYPE_USER_DATA_POINTER,
    TYPE_VERTEX_BUFFER,
    decode_buffer_entry,
    decode_texture_entry,
    extract_civbig_texture,
    landmark_base_model,
)
from Renderer.tools.asset_compiler.generic_decal_compiler import (
    TYPE_DECAL,
    TYPE_DECAL_VECTOR,
    TYPE_TERRAIN_EDIT_VECTOR,
    decode_decal_descriptor,
)
from Renderer.tools.asset_compiler.grassland_pack_builder import validate_runtime_independence
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = Path(__file__).with_name("compound_landmark_sets.json")
DEFAULT_PACK = RENDERER_ROOT / "packs" / "CompoundLandmarksNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "compound_landmarks" / "build.json"
MAC_ASSETS_ROOT = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common"
    / "Sid Meier's Civilization VI/Civ6.app/Contents/Assets"
)
WINDOWS_ASSETS_ROOT = Path(
    r"Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI"
    r"\Civ6.app\Contents\Assets"
)
SAFE_ID = re.compile(r"^[a-z0-9]+(?:[._-]?[a-z0-9]+)*(?:/[a-z0-9]+(?:[._-]?[a-z0-9]+)*)*$")
TYPE_STATE_DATA = "ModelPackageEntry::ModelStateData_Entry"
TYPE_STATE_STRINGS = (
    "BLP::BLPPtr<String::BasicT<Serialization::StaticPackageAllocator<"
    "Platform::StaticHeapAllocator<5, 0>>, String::ASCII>>"
)
TYPE_PACKAGE_MODEL_POINTER = "PackagePtr64<Granny::PackageModel>"
TYPE_PACKAGE_MODEL = "Granny::PackageModel"
TYPE_GRANNY_BONE = "granny_bone"
TYPE_ATTACHMENT_LIST = "AttachmentPointList"
TYPE_ATTACHMENT_DATA = "AttachmentPointCookData"
TYPE_BLP_VALUE_POINTER = "BLP::BLPPtr<BLP::Value>"
TYPE_BLP_ENTRY_VALUE = "BLP::BLPEntryValue"
TYPE_STRING_VALUE = "BLP::StringValue"
TYPE_ARTDEF_REFERENCE_VALUE = "BLP::ArtDefReferenceValue"
TYPE_BOOL_VALUE = "BLP::BoolValue"
TYPE_STRING = (
    "String::BasicT<Serialization::StaticPackageAllocator<Platform::StaticHeapAllocator<5, 0>>, "
    "String::ASCII>"
)
TYPE_TERRAIN_EDIT = "TerrainEditDesc3"
STATIC_VERTEX_PROFILES = {0x6679B170: 32, 0x315CFCD9: 24}
SKINNED_VERTEX_PROFILE = (0x6679B170, 32)
SOURCE_POSITION_OFFSETS = (0, 2, 4)
UV0_OFFSET = 8
SKIN_INDEX_OFFSET = 12
SKIN_WEIGHT_OFFSET = 16
MATERIAL_TEXTURE_SLOTS = {
    "normal_0": (0x1C, "LEAN", False),
    "normal_1": (0x20, "LEAN", False),
    "base_color": (0x24, "Generic_BaseColor", True),
    "gloss": (0x28, "Generic_Gloss", False),
    "ambient_occlusion": (0x34, "Generic_AO", False),
    "emissive": (0x3C, "Generic_Emissive", False),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and "\\" not in value


def load_mapping(path: Path) -> dict[str, Any]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping.get("schema") != "c3x.source_compound_landmark_mapping.v0":
        raise ValueError("Unsupported compound-landmark mapping schema")
    packages = mapping.get("packages")
    if not isinstance(packages, list) or not packages:
        raise ValueError("Compound-landmark mapping has no packages")
    source_entries: set[tuple[str, str]] = set()
    asset_ids: set[str] = set()
    for package in packages:
        source_package = package.get("source_package")
        if not isinstance(source_package, str) or not _safe_relative(source_package):
            raise ValueError("Compound-landmark source_package must be a safe relative path")
        shared_data = package.get("shared_data")
        shared_roots = [shared_data] if isinstance(shared_data, str) else shared_data
        if (
            not isinstance(shared_roots, list)
            or not shared_roots
            or not all(isinstance(value, str) and _safe_relative(value) for value in shared_roots)
            or len(shared_roots) != len(set(shared_roots))
        ):
            raise ValueError(
                "Compound-landmark shared_data must be a safe path or ordered unique path list"
            )
        scale = package.get("source_units_per_tile")
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("source_units_per_tile must be positive and finite")
        assets = package.get("assets")
        if not isinstance(assets, list) or not assets:
            raise ValueError("Compound-landmark source package has no assets")
        for asset in assets:
            source_entry = asset.get("source_entry")
            asset_id = asset.get("asset_id")
            if not isinstance(source_entry, str) or not source_entry:
                raise ValueError("Compound-landmark asset has no source entry")
            if not isinstance(asset_id, str) or not SAFE_ID.fullmatch(asset_id):
                raise ValueError(f"Invalid normalized compound-landmark ID: {asset_id!r}")
            source_key = (package["source_package"], source_entry)
            if source_key in source_entries:
                raise ValueError(f"Duplicate compound-landmark source entry: {source_entry}")
            if asset_id in asset_ids:
                raise ValueError(f"Duplicate normalized compound-landmark ID: {asset_id}")
            source_entries.add(source_key)
            asset_ids.add(asset_id)
    return mapping


def decode_state_mask(mask: int, states: list[str]) -> list[str]:
    if mask <= 0 or mask & ~((1 << len(states)) - 1):
        raise ValueError(f"Compound draw-group state mask 0x{mask:x} is outside the state table")
    return [state for index, state in enumerate(states) if mask & (1 << index)]


def decode_granny_bone(
    raw: bytes,
    name: str,
    index: int,
    bone_count: int,
    source_units_per_tile: float,
) -> dict[str, Any]:
    if len(raw) != 164 or not name:
        raise ValueError("Compound skeleton requires a named 164-byte bone record")
    parent = struct.unpack_from("<i", raw, 0x08)[0]
    if parent < -1 or parent >= bone_count or parent >= index:
        raise ValueError(f"Bone {name} has an invalid or non-topological parent {parent}")
    flags = struct.unpack_from("<I", raw, 0x0C)[0]
    position = list(struct.unpack_from("<3f", raw, 0x10))
    orientation = list(struct.unpack_from("<4f", raw, 0x1C))
    scale_shear = list(struct.unpack_from("<9f", raw, 0x2C))
    inverse_bind = list(struct.unpack_from("<16f", raw, 0x50))
    lod_error = struct.unpack_from("<f", raw, 0x90)[0]
    numeric = position + orientation + scale_shear + inverse_bind + [lod_error]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError(f"Bone {name} contains a non-finite transform")
    orientation_length = math.sqrt(sum(value * value for value in orientation))
    if orientation_length < 0.999 or orientation_length > 1.001:
        raise ValueError(f"Bone {name} has a non-unit rest orientation")
    unit_scale = 1.0 / source_units_per_tile
    position = [round(value * unit_scale, 8) for value in position]
    for component in (12, 13, 14):
        inverse_bind[component] *= unit_scale
    return {
        "name": name,
        "parent": parent,
        "transform_flags": flags,
        "rest": {
            "position": position,
            "orientation": [round(value, 8) for value in orientation],
            "scale_shear": [round(value, 8) for value in scale_shear],
        },
        "inverse_bind_matrix": [round(value, 8) for value in inverse_bind],
        "lod_error": round(lod_error * unit_scale, 8),
    }


def _matrix_multiply(a: list[float], b: list[float]) -> list[float]:
    return [
        sum(a[row * 4 + inner] * b[inner * 4 + column] for inner in range(4))
        for row in range(4)
        for column in range(4)
    ]


def measure_bind_pose(bones: list[dict[str, Any]]) -> tuple[float, str]:
    """Measure normalized rest-world times inverse-bind error.

    Granny's matrices use row vectors. Quaternion rotation is transposed from
    the common column-vector form, local world composition is local * parent,
    and translation occupies the final row.
    """
    worlds: list[list[float]] = []
    maximum_error = 0.0
    maximum_error_bone = ""
    identity = [1.0 if row == column else 0.0 for row in range(4) for column in range(4)]
    for bone in bones:
        x, y, z, w = bone["rest"]["orientation"]
        rotation_column = [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ]
        rotation_row = [rotation_column[column * 3 + row] for row in range(3) for column in range(3)]
        scale = bone["rest"]["scale_shear"]
        # Match normalized_skin._local_matrix: Granny's row-vector local is
        # scale/shear * transposed quaternion rotation. Uniform-scale assets
        # conceal the order; mounted rigs with non-uniform ankle transforms do
        # not.
        linear = [
            sum(scale[row * 3 + inner] * rotation_row[inner * 3 + column] for inner in range(3))
            for row in range(3)
            for column in range(3)
        ]
        local = [
            linear[0], linear[1], linear[2], 0.0,
            linear[3], linear[4], linear[5], 0.0,
            linear[6], linear[7], linear[8], 0.0,
            *bone["rest"]["position"], 1.0,
        ]
        parent = bone["parent"]
        world = local if parent < 0 else _matrix_multiply(local, worlds[parent])
        worlds.append(world)
        product = _matrix_multiply(world, bone["inverse_bind_matrix"])
        bone_error = max(abs(actual - expected) for actual, expected in zip(product, identity))
        if bone_error > maximum_error:
            maximum_error = bone_error
            maximum_error_bone = bone["name"]
    return maximum_error, maximum_error_bone


def validate_bind_pose(bones: list[dict[str, Any]]) -> float:
    """Prove normalized local transforms still invert against inverse-bind matrices."""
    maximum_error, maximum_error_bone = measure_bind_pose(bones)
    if maximum_error > 2.0e-5:
        raise ValueError(
            "Compound skeleton bind-pose/inverse-bind error is too large: "
            f"{maximum_error} at bone {maximum_error_bone}"
        )
    return maximum_error


def decode_skin_influences(
    raw: bytes, offset: int, bone_count: int, normalize_nonzero_weights: bool = False
) -> dict[str, Any]:
    indices = list(raw[offset + SKIN_INDEX_OFFSET : offset + SKIN_INDEX_OFFSET + 4])
    weights_u8 = list(raw[offset + SKIN_WEIGHT_OFFSET : offset + SKIN_WEIGHT_OFFSET + 4])
    weight_sum = sum(weights_u8)
    if weight_sum != 255 and not (normalize_nonzero_weights and weight_sum > 0):
        raise ValueError("Skinned compound vertex weights do not sum to 255")
    influences = [
        (index, weight)
        for index, weight in zip(indices, weights_u8)
        if weight
    ]
    if not influences or any(index >= bone_count for index, _weight in influences):
        raise ValueError("Skinned compound vertex references an invalid bone")
    return {
        "bone_indices": [index for index, _weight in influences],
        "bone_weights": [round(weight / weight_sum, 8) for _index, weight in influences],
        "source_weight_sum": weight_sum,
    }


def _decode_states(package: IndexedStaticPackage, landmark_user_data: int) -> tuple[list[str], dict[str, Any]]:
    fields = package.pointer_fields(landmark_user_data, TYPE_STATE_DATA)
    if len(fields) != 1:
        raise ValueError("Compound landmark must have exactly one model-state record")
    state_pointer = fields[0][1]
    arrays = package.pointer_fields(state_pointer, TYPE_STATE_STRINGS)
    if len(arrays) != 1:
        raise ValueError("Compound landmark model-state record has no unique state table")
    array_pointer = arrays[0][1]
    states = []
    for index in range(package.allocations[array_pointer - 1]["element_count"]):
        record = package.array_element(array_pointer, index)
        if len(record) != 8:
            raise ValueError("Compound landmark state table does not use pointer records")
        value = package.string_value(struct.unpack_from("<Q", record, 0)[0])
        if not value:
            raise ValueError("Compound landmark state name does not resolve")
        normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        if not normalized or normalized in states:
            raise ValueError("Compound landmark state names are empty or duplicated")
        states.append(normalized)
    return states, {"state_record": state_pointer, "state_array": array_pointer}


def _decode_skeletons(
    package: IndexedStaticPackage,
    base_model: int,
    source_units_per_tile: float,
    allow_unvalidated: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wrapper_fields = package.pointer_fields(base_model, TYPE_PACKAGE_MODEL_POINTER)
    if not wrapper_fields:
        return [], {"wrapper": None, "package_models": []}
    if len(wrapper_fields) != 1:
        raise ValueError("Compound landmark has ambiguous package-model wrappers")
    wrapper = wrapper_fields[0][1]
    package_models = []
    for offset in range(0, len(package.bytes_for(wrapper)), 8):
        pointer = struct.unpack_from("<Q", package.bytes_for(wrapper), offset)[0]
        if package.type_name(pointer) != TYPE_PACKAGE_MODEL:
            raise ValueError("Compound package-model wrapper has an unexpected target")
        bone_fields = package.pointer_fields(pointer, TYPE_GRANNY_BONE)
        name_fields = package.pointer_fields(pointer, TYPE_STRING)
        if len(bone_fields) != 1 or len(name_fields) != 1:
            raise ValueError("Compound package model has no unique bone/name arrays")
        bone_array = bone_fields[0][1]
        name_array = name_fields[0][1]
        bone_count = package.allocations[bone_array - 1]["element_count"]
        if package.allocations[name_array - 1]["element_count"] != bone_count or bone_count < 1:
            raise ValueError("Compound skeleton bone and name counts disagree")
        names = []
        bones = []
        for index in range(bone_count):
            name_record = package.array_element(name_array, index)
            name = package.string_value(struct.unpack_from("<Q", name_record, 0)[0])
            if not name:
                raise ValueError("Compound skeleton bone name does not resolve")
            names.append(name)
            bones.append(
                decode_granny_bone(
                    package.array_element(bone_array, index),
                    name,
                    index,
                    bone_count,
                    source_units_per_tile,
                )
            )
        if len(names) != len(set(names)):
            raise ValueError("Compound skeleton contains duplicate bone names")
        bind_pose_max_error, bind_pose_max_error_bone = measure_bind_pose(bones)
        if bind_pose_max_error > 2.0e-5 and not allow_unvalidated:
            raise ValueError(
                "Compound skeleton bind-pose/inverse-bind error is too large: "
                f"{bind_pose_max_error} at bone {bind_pose_max_error_bone}"
            )
        model_raw = package.bytes_for(pointer)
        track_group = package.string_value(struct.unpack_from("<Q", model_raw, 0x98)[0])
        if not track_group:
            raise ValueError("Compound package model has no animation track-group name")
        package_models.append(
            {
                "track_group": track_group,
                "bones": bones,
                "bind_pose_max_error": bind_pose_max_error,
                "bind_pose_max_error_bone": bind_pose_max_error_bone,
                "bind_pose_status": (
                    "passed" if bind_pose_max_error <= 2.0e-5 else "failed_retained_for_static_bake_only"
                ),
                "source": {
                    "package_model": pointer,
                    "bone_array": bone_array,
                    "name_array": name_array,
                },
            }
        )
    return package_models, {
        "wrapper": wrapper,
        "package_models": [item["source"] for item in package_models],
        "bind_pose_validation": [
            {
                "track_group": item["track_group"],
                "maximum_error": item["bind_pose_max_error"],
                "maximum_error_bone": item["bind_pose_max_error_bone"],
                "status": item["bind_pose_status"],
            }
            for item in package_models
        ],
    }


def _attachment_semantic(source_name: str) -> str:
    lowered = source_name.lower()
    if "smoke" in lowered or "chimney" in lowered:
        return "smoke"
    if "fire" in lowered or "torch" in lowered or "brazier" in lowered:
        return "flame"
    if "lantern" in lowered or "light" in lowered or "glow" in lowered:
        return "night_light"
    return "unresolved"


def _decode_terrain_edit(
    package: IndexedStaticPackage,
    vector: int,
    policy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if policy not in ("reject", "preserve_unresolved"):
        raise ValueError(f"Unsupported compound terrain-edit policy: {policy}")
    raw = package.bytes_for(vector)
    if not any(raw):
        return {"status": "absent"}, {"vector": vector, "status": "absent"}
    if policy == "reject":
        raise ValueError("Compound landmark terrain-edit payload needs a separate normalized profile")
    if len(raw) != 32:
        raise ValueError("Compound terrain-edit vector has an unsupported layout")
    reserved, data_pointer, count, capacity = struct.unpack("<4Q", raw)
    if reserved or not data_pointer or not count or count != capacity:
        raise ValueError("Compound terrain-edit vector has inconsistent fields")
    if package.type_name(data_pointer) != TYPE_TERRAIN_EDIT:
        raise ValueError("Compound terrain-edit vector has an unsupported record type")
    allocation = package.allocations[data_pointer - 1]
    records = package.bytes_for(data_pointer)
    if allocation["element_count"] != count or len(records) % count:
        raise ValueError("Compound terrain-edit record count is inconsistent")
    return (
        {
            "status": "unresolved",
            "application": "disabled",
            "missing_policy": "renderer_terrain_surface",
        },
        {
            "vector": vector,
            "status": "preserved_unresolved",
            "record_type": TYPE_TERRAIN_EDIT,
            "record_count": count,
            "record_bytes": len(records) // count,
            "records_sha256": _sha256(records),
        },
    )


def _decode_component_attachment_values(
    package: IndexedStaticPackage, pointer: int, expected_count: int
) -> dict[str, Any]:
    if package.type_name(pointer) != TYPE_BLP_VALUE_POINTER:
        raise ValueError("Compound component attachment has an unsupported value table")
    allocation = package.allocations[pointer - 1]
    raw = package.bytes_for(pointer)
    if allocation["element_count"] != expected_count or len(raw) != expected_count * 8:
        raise ValueError("Compound component attachment value count is inconsistent")
    values: dict[str, Any] = {}
    terminal = None
    for index in range(expected_count):
        value_pointer = struct.unpack_from("<Q", raw, index * 8)[0]
        value_type = package.type_name(value_pointer)
        value_raw = package.bytes_for(value_pointer)
        if value_type == TYPE_BLP_ENTRY_VALUE:
            if len(value_raw) != 64:
                raise ValueError("Compound component BLP entry has an unsupported layout")
            parameter = package.string_value(struct.unpack_from("<Q", value_raw, 0x10)[0])
            source_class = package.string_value(struct.unpack_from("<Q", value_raw, 0x18)[0])
            library = package.string_value(struct.unpack_from("<Q", value_raw, 0x20)[0])
            candidate_terminal = {
                "parameter": parameter,
                "class": source_class or library,
                "library": library,
                "entry": package.string_value(struct.unpack_from("<Q", value_raw, 0x28)[0]),
                "package": package.string_value(struct.unpack_from("<Q", value_raw, 0x30)[0]),
            }
            if candidate_terminal["parameter"] != "Asset":
                raise ValueError("Compound component attachment has an invalid asset terminal")
            if source_class and library and source_class != library:
                raise ValueError("Compound component attachment class/library disagree")
            if candidate_terminal["entry"] or candidate_terminal["package"]:
                if not all(candidate_terminal[field] for field in ("class", "entry", "package")):
                    raise ValueError("Compound component attachment has an incomplete asset terminal")
                terminal = candidate_terminal
        elif value_type == TYPE_STRING_VALUE:
            if len(value_raw) != 32:
                raise ValueError("Compound component string value has an unsupported layout")
            parameter = package.string_value(struct.unpack_from("<Q", value_raw, 0x10)[0])
            value = package.string_value(struct.unpack_from("<Q", value_raw, 0x18)[0])
            if not parameter or parameter in values:
                raise ValueError("Compound component attachment repeats a string parameter")
            values[parameter] = value
        elif value_type == TYPE_ARTDEF_REFERENCE_VALUE:
            if len(value_raw) != 48:
                raise ValueError("Compound component reference value has an unsupported layout")
            parameter = package.string_value(struct.unpack_from("<Q", value_raw, 0x10)[0])
            value = package.string_value(struct.unpack_from("<Q", value_raw, 0x18)[0])
            root = package.string_value(struct.unpack_from("<Q", value_raw, 0x20)[0])
            if not parameter or parameter in values:
                raise ValueError("Compound component attachment repeats a reference parameter")
            values[parameter] = {"value": value, "root": root}
        elif value_type == TYPE_BOOL_VALUE:
            if len(value_raw) != 32:
                raise ValueError("Compound component Boolean value has an unsupported layout")
            parameter = package.string_value(struct.unpack_from("<Q", value_raw, 0x10)[0])
            value = struct.unpack_from("<Q", value_raw, 0x18)[0]
            if not parameter or parameter in values or value not in (0, 1):
                raise ValueError("Compound component attachment has an invalid Boolean parameter")
            values[parameter] = bool(value)
        else:
            raise ValueError(f"Unsupported compound component value type {value_type}")
    required = {
        "ConnectionType",
        "ResourceType",
        "TerrainFollowMode",
        "Cull Mode",
        "RandomizeAnims",
    }
    if set(values) != required:
        raise ValueError(
            "Compound component attachment has unexpected parameters: "
            + ", ".join(sorted(set(values) ^ required))
        )
    resource = values["ResourceType"]
    if (
        values["ConnectionType"] not in ("NONE", "ROAD")
        or not isinstance(resource, dict)
        or resource.get("root") not in ("Resource", "ResourceTags")
        or not resource.get("value")
        or values["TerrainFollowMode"] != "Pivot Height"
        or values["Cull Mode"] not in ("OPTIONAL", "REQUIRED", "PERMANENT")
    ):
        raise ValueError(
            "Compound component attachment uses an unsupported selection policy: "
            + json.dumps(values, sort_keys=True)
        )
    return {
        "source_terminal": terminal,
        "source_values": values,
        "runtime_selection": {
            "connection_type": values["ConnectionType"].lower(),
            "resource_filter": (
                "any" if resource["value"] == "DON'T CARE" else "mapped_resource_required"
            ),
            "terrain_follow": "pivot_height",
            "cull": values["Cull Mode"].lower(),
            "randomize_animation": values["RandomizeAnims"],
        },
    }


def _decode_attachment_points(
    package: IndexedStaticPackage,
    landmark_user_data: int,
    skeletons: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fields = package.pointer_fields(landmark_user_data, TYPE_ATTACHMENT_LIST)
    if len(fields) != 1:
        raise ValueError("Compound landmark must have exactly one attachment-point list")
    list_pointer = fields[0][1]
    data_fields = package.pointer_fields(list_pointer, TYPE_ATTACHMENT_DATA)
    if not data_fields:
        if any(package.bytes_for(list_pointer)):
            raise ValueError("Compound attachment list has data but no typed attachment array")
        return [], {"list": list_pointer, "data": None, "count": 0, "points": []}
    if len(data_fields) != 1:
        raise ValueError("Compound attachment list has ambiguous attachment arrays")
    data_pointer = data_fields[0][1]
    count = package.allocations[data_pointer - 1]["element_count"]
    points = []
    evidence = []
    for index in range(count):
        raw = package.array_element(data_pointer, index)
        if len(raw) != 32:
            raise ValueError("Compound attachment record has an unsupported layout")
        source_name = package.string_value(struct.unpack_from("<Q", raw, 0)[0])
        if not source_name:
            raise ValueError("Compound attachment name does not resolve")
        value_pointer, first_count, second_count = struct.unpack_from("<3Q", raw, 8)
        component = None
        if value_pointer:
            if not first_count or first_count != second_count:
                raise ValueError("Compound component attachment has inconsistent value counts")
            component = _decode_component_attachment_values(
                package, value_pointer, first_count
            )
        elif first_count or second_count:
            raise ValueError("Compound attachment has counts without a value table")
        matches = [
            (skeleton_index, bone_index, bone)
            for skeleton_index, skeleton in enumerate(skeletons)
            for bone_index, bone in enumerate(skeleton["bones"])
            if bone["name"] == source_name
        ]
        has_component = component is not None and component["source_terminal"] is not None
        source_condition_unmapped = (
            has_component
            and component["runtime_selection"]["resource_filter"]
            == "mapped_resource_required"
        )
        component_transform_unresolved = has_component and len(matches) != 1
        source_asset_absent = (
            component is not None
            and component["source_terminal"] is None
            and component["runtime_selection"]["cull"] == "optional"
            and len(matches) != 1
        )
        if len(matches) != 1 and not (
            source_condition_unmapped or component_transform_unresolved or source_asset_absent
        ):
            raise ValueError(
                f"Compound attachment {source_name!r} resolves to {len(matches)} skeleton bones"
            )
        skeleton_index, bone_index, bone = (
            matches[0] if len(matches) == 1 else (None, None, None)
        )
        semantic = (
            "component"
            if has_component
            else "source_placeholder"
            if source_asset_absent
            else _attachment_semantic(source_name)
        )
        point = {
            "id": f"attachment_{index:02d}",
            "skeleton": skeleton_index,
            "bone": bone_index,
            "semantic": semantic,
            "state_hint": "pillaged" if "pil" in source_name.lower() else "operational",
            "binding_status": (
                "source_condition_unmapped"
                if source_condition_unmapped
                else "component_transform_unresolved"
                if component_transform_unresolved
                else "source_asset_absent"
                if source_asset_absent
                else "component_unresolved"
                if has_component
                else "resource_unresolved"
            ),
        }
        if component is not None:
            point["selection"] = component["runtime_selection"]
        points.append(point)
        evidence.append(
            {
                **point,
                "source_name": source_name,
                "bone_local_transform": None if bone is None else bone["rest"],
                **(
                    {
                        "component_source": component["source_terminal"],
                        "component_source_values": component["source_values"],
                    }
                    if component is not None
                    else {}
                ),
            }
        )
    return points, {
        "list": list_pointer,
        "data": data_pointer,
        "count": count,
        "points": evidence,
    }


def _decode_primitive(
    package: IndexedStaticPackage,
    record: bytes,
    states: list[str],
    material_count: int,
) -> dict[str, Any]:
    if len(record) != 32:
        raise ValueError("Compound primitive-group record must be 32 bytes")
    state_mask, material_index = struct.unpack_from("<2I", record, 0)
    if material_index >= material_count:
        raise ValueError("Compound primitive group references an invalid material")
    user_data = struct.unpack_from("<Q", record, 8)[0]
    if package.type_name(user_data) != TYPE_USER_DATA_POINTER:
        raise ValueError("Compound primitive group has no user-data pointer")
    data_fields = package.pointer_fields(user_data, TYPE_PRIM_DATA)
    if len(data_fields) != 1:
        raise ValueError("Compound primitive group has no unique geometry record")
    data_pointer = data_fields[0][1]
    raw = package.bytes_for(data_pointer)
    if len(raw) != 32:
        raise ValueError("Compound primitive geometry record must be 32 bytes")
    values = struct.unpack_from("<6I", raw, 8)
    return {
        "states": decode_state_mask(state_mask, states),
        "state_mask": state_mask,
        "material_index": material_index,
        "vertex_buffer": values[0],
        "index_buffer": values[1],
        "first_index": values[2],
        "index_count": values[3],
        "base_vertex": values[4],
        "vertex_count": values[5],
        "source": {"user_data": user_data, "primitive_data": data_pointer},
    }


def _normalize_geometry(
    vertex_bytes: bytes,
    index_bytes: bytes,
    vertex_entry: dict[str, Any],
    index_entry: dict[str, Any],
    primitive: dict[str, Any],
    source_units_per_tile: float,
    bone_count: int | None,
    normalize_skin_weights: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_stride = STATIC_VERTEX_PROFILES.get(vertex_entry["format"])
    if expected_stride != vertex_entry["stride"]:
        raise ValueError(
            f"Unsupported compound vertex profile 0x{vertex_entry['format']:08x}/"
            f"{vertex_entry['stride']}"
        )
    if bone_count is not None and (
        vertex_entry["format"], vertex_entry["stride"]
    ) != SKINNED_VERTEX_PROFILE:
        raise ValueError("Skinned compound mesh does not use the proven skin vertex profile")
    if index_entry["bytes_per_index"] != 2:
        raise ValueError("Compound mesh requires 16-bit source indices")
    stride = vertex_entry["stride"]
    count = vertex_entry["count"]
    if len(vertex_bytes) != count * stride:
        raise ValueError("Compound vertex payload length disagrees with its record")
    source_indices = struct.unpack(f"<{index_entry['count']}H", index_bytes)
    first = primitive["first_index"]
    index_count = primitive["index_count"]
    if index_count < 3 or index_count % 3 or first + index_count > len(source_indices):
        raise ValueError("Compound primitive index range is invalid")
    indices = [index + primitive["base_vertex"] for index in source_indices[first : first + index_count]]
    if min(indices) < 0 or max(indices) >= count:
        raise ValueError("Compound primitive contains an out-of-range vertex index")
    vertex_start = primitive["base_vertex"]
    vertex_end = vertex_start + primitive["vertex_count"]
    if (
        primitive["vertex_count"] < 1
        or vertex_end > count
        or any(index < vertex_start or index >= vertex_end for index in indices)
    ):
        raise ValueError("Compound primitive indices escape its declared vertex range")
    positions = [
        tuple(
            struct.unpack_from("<e", vertex_bytes, vertex * stride + offset)[0]
            / source_units_per_tile
            for offset in SOURCE_POSITION_OFFSETS
        )
        for vertex in range(count)
    ]
    uvs = [
        tuple(struct.unpack_from("<ee", vertex_bytes, vertex * stride + UV0_OFFSET))
        for vertex in range(count)
    ]
    if any(not all(math.isfinite(value) for value in position) for position in positions):
        raise ValueError("Compound mesh contains a non-finite position")
    if any(not all(math.isfinite(value) for value in uv) for uv in uvs):
        raise ValueError("Compound mesh contains a non-finite UV")

    normal_sums = [[0.0, 0.0, 0.0] for _ in positions]
    kept_indices = []
    omitted_degenerate_triangles = 0
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
        length = math.sqrt(sum(value * value for value in cross))
        if length <= 1.0e-12:
            omitted_degenerate_triangles += 1
            continue
        kept_indices.extend((ia, ib, ic))
        for vertex in (ia, ib, ic):
            for axis in range(3):
                normal_sums[vertex][axis] += cross[axis]
    if not kept_indices:
        raise ValueError("Compound mesh contains no non-degenerate triangles")
    vertices = []
    referenced = set(kept_indices)
    for index, (position, uv, normal_sum) in enumerate(zip(positions, uvs, normal_sums)):
        if index not in referenced:
            continue
        length = math.sqrt(sum(value * value for value in normal_sum))
        if length <= 1.0e-12:
            raise ValueError("Compound mesh referenced vertex has no geometric normal")
        vertex = {
            "source_index": index,
            "position": [round(value, 8) for value in position],
            "normal": [round(value / length, 8) for value in normal_sum],
            "uv0": [round(value, 8) for value in uv],
        }
        if bone_count is not None:
            vertex["skin"] = decode_skin_influences(
                vertex_bytes,
                index * stride,
                bone_count,
                normalize_skin_weights,
            )
        vertices.append(vertex)
    remap = {vertex["source_index"]: index for index, vertex in enumerate(vertices)}
    for vertex in vertices:
        del vertex["source_index"]
    normalized_indices = [remap[index] for index in kept_indices]
    wrap = any(value < 0.0 or value > 1.0 for uv in uvs for value in uv)
    source_weight_sums = [
        vertex["skin"].pop("source_weight_sum")
        for vertex in vertices
        if "skin" in vertex
    ]
    return {
        "schema": "c3x.skinned_mesh.v0" if bone_count is not None else "c3x.normalized_mesh.v0",
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Z",
            "position_unit": "tile",
            "uv0_origin": "upper_left",
        },
        "vertices": vertices,
        "topology": {"primitive": "triangles", "indices": normalized_indices},
        "skin": None if bone_count is None else {"skeleton_index": 0, "max_influences": 4},
    }, {
        "vertices": len(vertices),
        "triangles": len(kept_indices) // 3,
        "source_triangles": len(indices) // 3,
        "omitted_degenerate_triangles": omitted_degenerate_triangles,
        "skinned": bone_count is not None,
        "uv_address": "repeat" if wrap else "clamp",
        "source_vertex_buffer": vertex_entry,
        "source_index_buffer": index_entry,
        "skin_weight_normalization": (
            None
            if not source_weight_sums
            else {
                "policy": "normalize_nonzero_u8_sum" if normalize_skin_weights else "require_255",
                "source_sum_min": min(source_weight_sums),
                "source_sum_max": max(source_weight_sums),
                "normalized_vertices": sum(value != 255 for value in source_weight_sums),
            }
        ),
        "vertex_sha256": _sha256(vertex_bytes),
        "index_sha256": _sha256(index_bytes),
    }


def _decode_materials(
    package: IndexedStaticPackage,
    material_array: int,
    shared_data: Path | list[Path],
    pack: Path,
    asset_stem: str,
    uv_addresses: list[str] | list[tuple[int, str]],
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]],
    artifact_family: str = "compound",
    adapter: str = "c3x.compound_landmark.v0",
) -> tuple[list[str], list[dict[str, Any]]]:
    texture_array = package.unique_allocation(TYPE_TEXTURE)
    outputs = []
    evidence = []
    source_material_count = package.allocations[material_array - 1]["element_count"]
    if uv_addresses and isinstance(uv_addresses[0], tuple):
        material_variants = uv_addresses
    else:
        if len(uv_addresses) != source_material_count:
            raise ValueError("Compound material addressing does not match material count")
        material_variants = list(enumerate(uv_addresses))
    for output_index, (material_index, uv_address) in enumerate(material_variants):
        if not 0 <= material_index < source_material_count or uv_address not in ("clamp", "repeat"):
            raise ValueError("Compound material variant is invalid")
        record = package.array_element(material_array, material_index)
        if len(record) != 24:
            raise ValueError("Compound material container must use 24-byte records")
        user_data = struct.unpack_from("<Q", record, 0)[0]
        fields = package.pointer_fields(user_data, TYPE_MATERIAL_DATA)
        if len(fields) != 1:
            raise ValueError("Compound material has no unique material-data record")
        data_pointer = fields[0][1]
        raw = package.bytes_for(data_pointer)
        if len(raw) != 64:
            raise ValueError("Compound material-data record must be 64 bytes")
        channels = {}
        slots = {}
        for role, (offset, expected_class, required) in MATERIAL_TEXTURE_SLOTS.items():
            texture_index = struct.unpack_from("<I", raw, offset)[0]
            if texture_index == 0xFFFFFFFF:
                if required:
                    raise ValueError(f"Compound material is missing required {role}")
                slots[role] = {"status": "absent"}
                continue
            entry = decode_texture_entry(package, texture_array, texture_index)
            if entry["class"] != expected_class:
                if required:
                    raise ValueError(
                        f"Compound material {role} class is {entry['class']}, expected {expected_class}"
                    )
                slots[role] = {"status": "class_mismatch", "source": entry}
                continue
            shared_roots = [shared_data] if isinstance(shared_data, Path) else shared_data
            source = next(
                (root / entry["name"] for root in shared_roots if (root / entry["name"]).is_file()),
                None,
            )
            if source is None:
                raise ValueError(
                    f"Missing compound material texture {entry['name']} in "
                    + ", ".join(str(root) for root in shared_roots)
                )
            key = (str(source), entry["class"])
            digest = _sha256(source.read_bytes())[:16]
            relative = f"textures/{artifact_family}/{role}_{digest}.dds"
            if key not in texture_cache:
                texture_cache[key] = (relative, extract_civbig_texture(source, pack / relative))
            cached_relative, info = texture_cache[key]
            if cached_relative != relative:
                raise ValueError("Compound texture cache resolved one source inconsistently")
            channels[role] = {
                "texture": relative,
                "format": info["format_name"],
                "color_space": info["color_space"],
                "address_u": uv_address,
                "address_v": uv_address,
            }
            slots[role] = {"status": "accepted", "source": entry, **info}
        if ("normal_0" in channels) != ("normal_1" in channels):
            raise ValueError("Compound material has an incomplete LEAN normal pair")
        relative_document = f"materials/{artifact_family}/{asset_stem}_{output_index:02d}.json"
        material_document = {
            "schema": "c3x.material.v0",
            "name": f"material_{output_index:02d}",
            "channels": channels,
            "alpha_mode": "opaque",
            "provenance": {
                "kind": "local_normalized_import",
                "adapter": adapter,
                "source_format_dependency": None,
            },
        }
        if "emissive" in channels:
            material_document["emissive"] = {
                "mask": channels["emissive"]["texture"],
                "color": [1.0, 1.0, 1.0],
                "intensity": 1.0,
                "activation": "night",
                "missing_policy": "non-emissive",
            }
        _write_json(pack / relative_document, material_document)
        outputs.append(relative_document)
        evidence.append(
            {
                "material_index": material_index,
                "normalized_material_index": output_index,
                "uv_address": uv_address,
                "user_data": user_data,
                "material_data": data_pointer,
                "texture_slots": slots,
            }
        )
    return outputs, evidence


def _extract_decal_component(
    package: IndexedStaticPackage,
    landmark_user_data: int,
    shared_data: Path | list[Path],
    pack: Path,
    asset_id: str,
    asset_stem: str,
    source_units_per_tile: float,
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]],
) -> tuple[str | None, dict[str, Any]]:
    vector_fields = package.pointer_fields(landmark_user_data, TYPE_DECAL_VECTOR)
    if len(vector_fields) != 1:
        raise ValueError("Compound landmark must have exactly one decal vector")
    vector = vector_fields[0][1]
    if not any(package.bytes_for(vector)):
        return None, {"vector": vector, "status": "empty"}
    decal_fields = package.pointer_fields(vector, TYPE_DECAL)
    if len(decal_fields) != 1:
        raise ValueError("Compound landmark supports one decal descriptor per asset")
    decal_pointer = decal_fields[0][1]
    decal_count = package.allocations[decal_pointer - 1]["element_count"]
    if decal_count < 1:
        raise ValueError("Compound landmark decal array is empty")
    texture_array = package.unique_allocation(TYPE_TEXTURE)
    shared_roots = [shared_data] if isinstance(shared_data, Path) else shared_data
    relative_documents = []
    descriptor_evidence = []
    for descriptor_index in range(decal_count):
        descriptor = decode_decal_descriptor(
            package.array_element(decal_pointer, descriptor_index),
            lambda index: decode_texture_entry(package, texture_array, index),
            source_units_per_tile,
            required_roles=("base_color",),
        )
        channels = {}
        texture_evidence = {}
        for role, entry in descriptor["textures"].items():
            source = next(
                (root / entry["name"] for root in shared_roots if (root / entry["name"]).is_file()),
                None,
            )
            if source is None:
                raise ValueError(
                    f"Missing compound decal texture {entry['name']} in "
                    + ", ".join(str(root) for root in shared_roots)
                )
            key = (str(source), entry["class"])
            digest = _sha256(source.read_bytes())[:16]
            relative = f"textures/compound/{role}_{digest}.dds"
            if key not in texture_cache:
                texture_cache[key] = (relative, extract_civbig_texture(source, pack / relative))
            cached_relative, info = texture_cache[key]
            if cached_relative != relative:
                raise ValueError("Compound decal texture resolved inconsistently")
            channels[role] = {
                "texture": relative,
                "format": info["format_name"],
                "color_space": info["color_space"],
                "address_u": "clamp",
                "address_v": "clamp",
            }
            texture_evidence[role] = {**entry, **info}
        suffix = "" if decal_count == 1 else f"_{descriptor_index:02d}"
        relative_document = f"decals/compound/{asset_stem}{suffix}.json"
        _write_json(
            pack / relative_document,
            {
                "schema": "c3x.decal.v0",
                "asset_id": asset_id + (
                    "/decal" if decal_count == 1 else f"/decal_{descriptor_index:02d}"
                ),
                "footprint": {
                    "bounds_xy": descriptor["footprint_bounds"],
                    "content_bounds_xy": descriptor["content_bounds"],
                },
                "uv_rect": [0.0, 0.0, 1.0, 1.0],
                "channels": channels,
                "render": {
                    "projection": "terrain_surface",
                    "blend_mode": "alpha",
                    "depth_bias_policy": "terrain_decal",
                },
                "provenance": {
                    "kind": "local_normalized_import",
                    "adapter": "c3x.compound_landmark.v0",
                    "source_format_dependency": None,
                },
            },
        )
        relative_documents.append(relative_document)
        descriptor_evidence.append(
            {
                "index": descriptor_index,
                "source_bounds": descriptor["source_bounds"],
                "texture_slots": descriptor["texture_slots"],
                "textures": texture_evidence,
            }
        )
    if decal_count == 1:
        component_path = relative_documents[0]
    else:
        component_path = f"decals/compound/{asset_stem}.json"
        _write_json(
            pack / component_path,
            {
                "schema": "c3x.decal_set.v0",
                "asset_id": asset_id + "/decals",
                "decals": relative_documents,
                "provenance": {
                    "kind": "local_normalized_import",
                    "adapter": "c3x.compound_landmark.v0",
                    "source_format_dependency": None,
                },
            },
        )
    return component_path, {
        "vector": vector,
        "decal": decal_pointer,
        "count": decal_count,
        "status": "normalized",
        "descriptors": descriptor_evidence,
    }


def _compile_asset(
    package: IndexedStaticPackage,
    shared_data: Path | list[Path],
    pack: Path,
    source_entry: str,
    asset_id: str,
    source_units_per_tile: float,
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]],
    terrain_edit_policy: str = "reject",
    static_bake: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    package.select_direct_string(source_entry)
    landmark, landmark_user_data, base_model = landmark_base_model(package)
    states, state_evidence = _decode_states(package, landmark_user_data)
    skeletons, skeleton_evidence = _decode_skeletons(
        package, base_model, source_units_per_tile, allow_unvalidated=static_bake
    )
    attachment_points, attachment_evidence = _decode_attachment_points(
        package, landmark_user_data, skeletons
    )
    if static_bake:
        source_points = attachment_evidence.get("points", [])
        attachment_points = []
        omitted_points = []
        for point in source_points:
            if point.get("semantic") == "component" and point.get("binding_status") == "component_unresolved":
                baked = dict(point)
                baked["skeleton"] = None
                baked["bone"] = None
                baked["transform_binding"] = "static_root_local_from_source_rest_transform"
                attachment_points.append(baked)
            else:
                omitted_points.append(point)
        attachment_evidence = {
            **attachment_evidence,
            "points": attachment_points,
            "static_bake_omitted_points": omitted_points,
            "policy": "preserve_component_rest_transforms_as_explicit_static_root_local_joins_and_omit_noncomponent_rig_effects",
        }
    asset_stem = asset_id.replace("/", "_").replace(".", "_")
    skeleton_paths = []
    for index, skeleton in enumerate([] if static_bake else skeletons):
        relative = f"skeletons/compound/{asset_stem}_{index:02d}.json"
        _write_json(
            pack / relative,
            {
                "schema": "c3x.skeleton.v0",
                "track_group": skeleton["track_group"],
                "bones": skeleton["bones"],
                "validation": {
                    "bind_pose_max_error": skeleton["bind_pose_max_error"],
                    "inverse_bind_status": "passed",
                },
                "provenance": {
                    "kind": "local_normalized_import",
                    "adapter": "c3x.compound_landmark.v0",
                    "source_format_dependency": None,
                },
            },
        )
        skeleton_paths.append(relative)

    geometry_types = (TYPE_MODEL, TYPE_MESH, TYPE_PRIM_GROUP, TYPE_MATERIAL)
    geometry_fields = {target: package.pointer_fields(base_model, target) for target in geometry_types}
    present = [bool(geometry_fields[target]) for target in geometry_types]
    skeleton_only_model = present == [True, False, False, False] and bool(skeletons)
    if any(present) and not all(present) and not skeleton_only_model:
        raise ValueError("Compound landmark has an incomplete geometry container set")
    geometry_paths = []
    draw_bindings = []
    material_paths = []
    geometry_evidence = []
    material_evidence = []
    if all(present):
        if any(len(geometry_fields[target]) != 1 for target in geometry_types):
            raise ValueError("Compound landmark has ambiguous geometry container arrays")
        model_array = geometry_fields[TYPE_MODEL][0][1]
        mesh_array = geometry_fields[TYPE_MESH][0][1]
        primitive_array = geometry_fields[TYPE_PRIM_GROUP][0][1]
        material_array = geometry_fields[TYPE_MATERIAL][0][1]
        model_count = package.allocations[model_array - 1]["element_count"]
        mesh_count = package.allocations[mesh_array - 1]["element_count"]
        primitive_count = package.allocations[primitive_array - 1]["element_count"]
        material_count = package.allocations[material_array - 1]["element_count"]
        if skeletons and len(skeletons) not in (1, model_count):
            raise ValueError("Compound skeleton count cannot be associated with model count")
        models = []
        covered_meshes = []
        for model_index in range(model_count):
            record = package.array_element(model_array, model_index)
            if len(record) != 32:
                raise ValueError("Compound model container must use 32-byte records")
            first_mesh, count = struct.unpack_from("<2I", record, 0x18)
            if count < 1 or first_mesh + count > mesh_count:
                raise ValueError("Compound model has an invalid mesh range")
            covered_meshes.extend(range(first_mesh, first_mesh + count))
            models.append((model_index, first_mesh, count))
        if sorted(covered_meshes) != list(range(mesh_count)):
            raise ValueError("Compound model ranges do not cover meshes exactly once")

        primitives = [
            _decode_primitive(
                package,
                package.array_element(primitive_array, index),
                states,
                material_count,
            )
            for index in range(primitive_count)
        ]
        vertex_array = package.unique_allocation(TYPE_VERTEX_BUFFER)
        index_array = package.unique_allocation(TYPE_INDEX_BUFFER)
        geometry_by_key: dict[tuple[int, ...], int] = {}
        geometry_is_skinned: dict[tuple[int, ...], bool] = {}
        geometry_uv_addresses: dict[tuple[int, ...], str] = {}
        geometry_address_by_index: dict[int, str] = {}
        material_uv_addresses: dict[int, set[str]] = {
            index: set() for index in range(material_count)
        }
        mesh_to_model = {
            mesh: model_index
            for model_index, first_mesh, count in models
            for mesh in range(first_mesh, first_mesh + count)
        }
        covered_primitives = []
        for mesh_index in range(mesh_count):
            record = package.array_element(mesh_array, mesh_index)
            if len(record) != 40:
                raise ValueError("Compound mesh container must use 40-byte records")
            first_primitive, count = struct.unpack_from("<2I", record, 0x18)
            if count < 1 or first_primitive + count > primitive_count:
                raise ValueError("Compound mesh has an invalid primitive range")
            covered_primitives.extend(range(first_primitive, first_primitive + count))
            model_index = mesh_to_model[mesh_index]
            skeleton_index = 0 if len(skeletons) == 1 else model_index
            model_bone_count = len(skeletons[skeleton_index]["bones"]) if skeletons else None
            for primitive_index in range(first_primitive, first_primitive + count):
                primitive = primitives[primitive_index]
                key = tuple(
                    primitive[field]
                    for field in (
                        "vertex_buffer",
                        "index_buffer",
                        "first_index",
                        "index_count",
                        "base_vertex",
                        "vertex_count",
                    )
                )
                if key not in geometry_by_key:
                    vertex_entry = decode_buffer_entry(
                        package, vertex_array, primitive["vertex_buffer"], True
                    )
                    index_entry = decode_buffer_entry(
                        package, index_array, primitive["index_buffer"], False
                    )
                    bone_count = (
                        model_bone_count
                        if (vertex_entry["format"], vertex_entry["stride"])
                        == SKINNED_VERTEX_PROFILE
                        else None
                    )
                    geometry_is_skinned[key] = bone_count is not None
                    vertex_bytes = package.big_data(vertex_entry["offset"], vertex_entry["bytes"])
                    index_bytes = package.big_data(index_entry["offset"], index_entry["bytes"])
                    mesh, evidence = _normalize_geometry(
                        vertex_bytes,
                        index_bytes,
                        vertex_entry,
                        index_entry,
                        primitive,
                        source_units_per_tile,
                        bone_count,
                    )
                    if static_bake:
                        if bone_count is not None:
                            mesh["schema"] = "c3x.normalized_mesh.v0"
                            mesh["skin"] = None
                            for vertex in mesh["vertices"]:
                                vertex.pop("skin", None)
                        evidence["static_bake"] = {
                            "policy": "preserve_source_vertex_positions_and_strip_skin_influences",
                            "source_bind_pose_max_error": skeletons[skeleton_index]["bind_pose_max_error"],
                            "source_bind_pose_max_error_bone": skeletons[skeleton_index]["bind_pose_max_error_bone"],
                            "source_geometry_was_skinned": bone_count is not None,
                            "runtime_skeleton_dependency": None,
                        }
                    geometry_index = len(geometry_paths)
                    relative = f"meshes/compound/{asset_stem}_{geometry_index:02d}.json"
                    mesh["asset_id"] = f"{asset_id}/geometry_{geometry_index:02d}"
                    _write_json(pack / relative, mesh)
                    geometry_paths.append(relative)
                    geometry_evidence.append(evidence)
                    geometry_by_key[key] = geometry_index
                    geometry_uv_addresses[key] = evidence["uv_address"]
                    geometry_address_by_index[geometry_index] = evidence["uv_address"]
                material_uv_addresses[primitive["material_index"]].add(
                    geometry_uv_addresses[key]
                )
                draw_bindings.append(
                    {
                        "model": model_index,
                        "mesh": mesh_index,
                        "geometry": geometry_by_key[key],
                        "material": primitive["material_index"],
                        "states": primitive["states"],
                        "skeleton": None if static_bake else skeleton_index if skeletons else None,
                        "binding_mode": (
                            "static_source_vertex_bake"
                            if static_bake and geometry_is_skinned[key]
                            else "vertex_skin"
                            if geometry_is_skinned[key]
                            else "rigid_model"
                        ),
                    }
                )
        if sorted(covered_primitives) != list(range(primitive_count)):
            raise ValueError("Compound mesh ranges do not cover primitive groups exactly once")
        material_variants = []
        for material_index in range(material_count):
            addresses = material_uv_addresses[material_index]
            if not addresses:
                raise ValueError("Compound material is not referenced by any primitive")
            material_variants.extend(
                (material_index, address) for address in sorted(addresses)
            )
        variant_indices = {
            variant: index for index, variant in enumerate(material_variants)
        }
        for binding in draw_bindings:
            binding["material"] = variant_indices[
                (binding["material"], geometry_address_by_index[binding["geometry"]])
            ]
        material_paths, material_evidence = _decode_materials(
            package,
            material_array,
            shared_data,
            pack,
            asset_stem,
            material_variants,
            texture_cache,
        )

    terrain_fields = package.pointer_fields(landmark_user_data, TYPE_TERRAIN_EDIT_VECTOR)
    if len(terrain_fields) != 1:
        raise ValueError("Compound landmark must have exactly one terrain-edit vector")
    terrain_vector = terrain_fields[0][1]
    terrain_edit, terrain_edit_evidence = _decode_terrain_edit(
        package, terrain_vector, terrain_edit_policy
    )
    decal_path, decal_evidence = _extract_decal_component(
        package,
        landmark_user_data,
        shared_data,
        pack,
        asset_id,
        asset_stem,
        source_units_per_tile,
        texture_cache,
    )
    if (
        not geometry_paths
        and decal_path is None
        and not any(point["semantic"] == "component" for point in attachment_points)
    ):
        raise ValueError("Compound landmark has no normalized visual component")
    relative_document = f"compound_landmarks/{asset_stem}.json"
    document = {
        "schema": "c3x.compound_landmark.v0",
        "asset_id": asset_id,
        "states": states,
        "components": {
            "geometry": geometry_paths,
            "materials": material_paths,
            "skeletons": skeleton_paths,
            "decal": decal_path,
        },
        "draw_bindings": draw_bindings,
        "animation_binding": [
            {"skeleton": index, "track_group": skeleton["track_group"]}
            for index, skeleton in enumerate([] if static_bake else skeletons)
        ],
        "attachment_points": attachment_points,
        "terrain_edit": terrain_edit,
        "provenance": {
            "kind": "local_normalized_import",
            "adapter": (
                "c3x.compound_landmark.static_source_vertex_bake.v0"
                if static_bake
                else "c3x.compound_landmark.v0"
            ),
            "source_format_dependency": None,
            "static_bake": static_bake,
        },
    }
    _write_json(pack / relative_document, document)
    return {"type": "compound_landmark", "landmark": relative_document}, {
        "source_entry": source_entry,
        "asset_id": asset_id,
        "pointer_chain": {
            "landmark": landmark,
            "landmark_user_data": landmark_user_data,
            "base_model": base_model,
            **state_evidence,
            **skeleton_evidence,
            "terrain_edit_vector": terrain_vector,
        },
        "geometry_container_mode": (
            "complete" if all(present) else "skeleton_only_decal" if skeleton_only_model else "absent"
        ),
        "states": states,
        "geometry": geometry_evidence,
        "materials": material_evidence,
        "decal": decal_evidence,
        "attachments": attachment_evidence,
        "terrain_edit": terrain_edit_evidence,
        "static_bake": {
            "enabled": static_bake,
            "policy": (
                "source_vertex_positions_preserved_skin_and_skeleton_runtime_dependency_removed"
                if static_bake
                else "none"
            ),
        },
    }


def compile_compound_landmarks(
    assets_root: Path,
    mapping_path: Path,
    pack: Path,
    report_path: Path,
) -> dict[str, Any]:
    mapping = load_mapping(mapping_path)
    try:
        report_path.resolve().relative_to(pack.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("Compound-landmark source report must be outside the runtime pack")
    assets = {}
    reports = []
    packages = []
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for package_mapping in mapping["packages"]:
        source_path = assets_root / package_mapping["source_package"]
        shared_values = package_mapping["shared_data"]
        shared_data = (
            [assets_root / value for value in shared_values]
            if isinstance(shared_values, list)
            else assets_root / shared_values
        )
        shared_roots = shared_data if isinstance(shared_data, list) else [shared_data]
        missing_shared = next((path for path in shared_roots if not path.is_dir()), None)
        if not source_path.is_file() or missing_shared is not None:
            raise FileNotFoundError(source_path if not source_path.is_file() else missing_shared)
        first_entry = package_mapping["assets"][0]["source_entry"]
        package = IndexedStaticPackage(source_path, first_entry)
        packages.append(
            {
                "source": str(source_path),
                "source_sha256": _sha256(source_path.read_bytes()),
                "allocation_count": len(package.allocations),
            }
        )
        for asset_mapping in package_mapping["assets"]:
            asset_id = asset_mapping["asset_id"]
            try:
                manifest_asset, evidence = _compile_asset(
                    package,
                    shared_data,
                    pack,
                    asset_mapping["source_entry"],
                    asset_id,
                    float(package_mapping["source_units_per_tile"]),
                    texture_cache,
                )
            except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
                raise ValueError(
                    f"Failed compound asset {asset_id} from {asset_mapping['source_entry']}: {exc}"
                ) from exc
            assets[asset_id] = manifest_asset
            reports.append(evidence)
    manifest = {
        "schema": "c3x.asset_pack.v0",
        "name": "CompoundLandmarksNormalized",
        "display_name": "Normalized Compound Landmarks",
        "source_policy": "Local licensed-source import; derived art is not redistributable.",
        "assets": assets,
    }
    _write_json(pack / "manifest.json", manifest)
    independence_errors = validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError("Runtime pack is source-dependent: " + "; ".join(independence_errors))
    report = {
        "schema": "c3x.source_compound_landmark_build.v0",
        "mapping": {"path": str(mapping_path), "sha256": _sha256(mapping_path.read_bytes())},
        "packages": packages,
        "assets": reports,
        "outputs": {
            "pack": str(pack),
            "assets": len(assets),
            "textures": len(texture_cache),
            "geometry_assets": sum(bool(asset["geometry"]) for asset in reports),
            "skinned_assets": sum(
                any(geometry["skinned"] for geometry in asset["geometry"])
                for asset in reports
            ),
            "decal_assets": sum(asset["decal"]["status"] == "normalized" for asset in reports),
        },
        "runtime_independence": "passed",
        "runtime_integration": "not_enabled",
    }
    _write_json(report_path, report)
    return report


def default_assets_root() -> Path:
    return MAC_ASSETS_ROOT if MAC_ASSETS_ROOT.is_dir() else WINDOWS_ASSETS_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=default_assets_root())
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        report = compile_compound_landmarks(
            args.assets_root, args.mapping, args.pack, args.report
        )
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Compiled {report['outputs']['assets']} compound landmarks: "
        f"{report['outputs']['skinned_assets']} skinned, "
        f"{report['outputs']['decal_assets']} decal-bearing"
    )
    print(f"Pack: {args.pack}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
