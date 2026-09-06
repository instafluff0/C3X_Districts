#!/usr/bin/env python3
"""Normalize skinned tile-base landmarks into generic C3X JSON assets."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

from Renderer.tools.asset_compiler import clutter_blp_extractor as static
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage


TYPE_ANIMATION_BINDING = "int32"
BONE_RECORD_BYTES = 164
SKIN_VERTEX_FORMAT = 0x6679B170
SKIN_VERTEX_BYTES = 32
SOURCE_UNITS_PER_TILE = 100.0
POSITION_OFFSETS = (0, 2, 4)
UV_OFFSET = 8
JOINT_OFFSET = 12
WEIGHT_OFFSET = 16


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rounded(values: tuple[float, ...] | list[float]) -> list[float]:
    return [0.0 if abs(value) < 1.0e-12 else round(value, 8) for value in values]


def _one_typed_child(package: IndexedStaticPackage, owner: int, type_name: str) -> int:
    candidates = []
    raw = package.bytes_for(owner)
    for offset in range(0, len(raw) - 7, 8):
        pointer = struct.unpack_from("<Q", raw, offset)[0]
        if package.type_name(pointer) == type_name:
            candidates.append(pointer)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one {type_name} child of allocation {owner}, found {len(candidates)}"
        )
    return candidates[0]


def _package_model(package: IndexedStaticPackage, base_model: int) -> int:
    wrapper = package.unique_pointer_field(base_model, static.TYPE_PACKAGE_MODEL_POINTER)
    raw = package.bytes_for(wrapper)
    if len(raw) != 8:
        raise ValueError("Package-model pointer wrapper is not eight bytes")
    model = struct.unpack_from("<Q", raw)[0]
    if package.type_name(model) != static.TYPE_PACKAGE_MODEL:
        raise ValueError("Package-model pointer does not resolve to Granny::PackageModel")
    return model


def _decode_skeleton(
    package: IndexedStaticPackage, package_model: int, asset_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = package.bytes_for(package_model)
    if len(raw) < 0xC0:
        raise ValueError("Granny package-model record is truncated")
    bones_pointer = struct.unpack_from("<Q", raw, 0xA0)[0]
    names_pointer = struct.unpack_from("<Q", raw, 0xB8)[0]
    if package.type_name(bones_pointer) != "granny_bone":
        raise ValueError("Package model has no typed Granny bone array")
    if package.type_name(names_pointer) != static.TYPE_STRING:
        raise ValueError("Package model has no typed bone-name array")
    bone_count = package.allocations[bones_pointer - 1]["element_count"]
    if package.allocations[names_pointer - 1]["element_count"] != bone_count:
        raise ValueError("Bone and bone-name array counts differ")
    if len(package.bytes_for(bones_pointer)) != bone_count * BONE_RECORD_BYTES:
        raise ValueError("Granny bone array does not use 164-byte records")

    bones = []
    names = []
    for index in range(bone_count):
        bone = package.array_element(bones_pointer, index)
        name_record = package.array_element(names_pointer, index)
        if len(name_record) != 8:
            raise ValueError("Bone-name wrapper is not eight bytes")
        name = package.string_value(struct.unpack_from("<Q", name_record)[0])
        if not name or name in names:
            raise ValueError("Skeleton contains an empty or duplicate bone name")
        names.append(name)
        parent, flags = struct.unpack_from("<iI", bone, 8)
        if parent >= index or parent < -1:
            raise ValueError(f"Bone {index} has a non-canonical parent index {parent}")
        position = list(struct.unpack_from("<3f", bone, 16))
        position = [value / SOURCE_UNITS_PER_TILE for value in position]
        orientation = list(struct.unpack_from("<4f", bone, 28))
        scale_shear = list(struct.unpack_from("<9f", bone, 44))
        inverse_bind = list(struct.unpack_from("<16f", bone, 80))
        for component in (12, 13, 14):
            inverse_bind[component] /= SOURCE_UNITS_PER_TILE
        all_values = position + orientation + scale_shear + inverse_bind
        if not all(math.isfinite(value) for value in all_values):
            raise ValueError(f"Bone {index} contains a non-finite transform")
        bones.append(
            {
                "name": name,
                "parent": parent,
                "local": {
                    "position": _rounded(position),
                    "orientation": _rounded(orientation),
                    "scale_shear": _rounded(scale_shear),
                },
                "inverse_bind_matrix": _rounded(inverse_bind),
            }
        )
    return (
        {
            "schema": "c3x.normalized_skeleton.v0",
            "asset_id": asset_id,
            "matrix_convention": "row_major_row_vector",
            "position_unit": "tile",
            "bones": bones,
        },
        {
            "package_model": package_model,
            "bones_pointer": bones_pointer,
            "names_pointer": names_pointer,
            "bone_count": bone_count,
            "bone_flags": [struct.unpack_from("<I", package.array_element(bones_pointer, i), 12)[0] for i in range(bone_count)],
        },
    )


def _primitive(
    package: IndexedStaticPackage, base_model: int
) -> tuple[dict[str, int], dict[str, Any]]:
    primitive_groups = package.unique_pointer_field(base_model, static.TYPE_PRIM_GROUP)
    decoded = []
    pointers = []
    for index in range(package.allocations[primitive_groups - 1]["element_count"]):
        record = package.array_element(primitive_groups, index)
        user_data = struct.unpack_from("<Q", record, 8)[0]
        primitive_data = _one_typed_child(package, user_data, static.TYPE_PRIM_DATA)
        raw = package.bytes_for(primitive_data)
        if len(raw) != 32:
            raise ValueError("Skinned primitive-data record is not 32 bytes")
        values = struct.unpack_from("<6I", raw, 8)
        decoded.append(
            {
                "vertex_buffer": values[0],
                "index_buffer": values[1],
                "first_index": values[2],
                "index_count": values[3],
                "base_vertex": values[4],
                "vertex_count": values[5],
            }
        )
        pointers.append(primitive_data)
    if not decoded or any(value != decoded[0] for value in decoded[1:]):
        raise ValueError("Tile-base primitive groups do not share one proven geometry range")
    return decoded[0], {
        "primitive_groups": primitive_groups,
        "primitive_data": pointers,
        "deduplicated_equivalent_groups": len(decoded),
    }


def _bone_palette(package: IndexedStaticPackage, base_model: int, bone_count: int) -> list[int]:
    fields = package.pointer_fields(base_model, TYPE_ANIMATION_BINDING)
    if len(fields) != 1:
        raise ValueError(f"Expected one skinned-mesh bone palette, found {len(fields)}")
    raw = package.bytes_for(fields[0][1])
    if len(raw) % 4:
        raise ValueError("Bone palette is not an int32 array")
    palette = list(struct.unpack(f"<{len(raw) // 4}i", raw))
    if not palette or len(set(palette)) != len(palette):
        raise ValueError("Bone palette is empty or contains duplicate skeleton indices")
    if min(palette) < 0 or max(palette) >= bone_count:
        raise ValueError("Bone palette references outside the skeleton")
    return palette


def _decode_mesh(
    package: IndexedStaticPackage,
    base_model: int,
    skeleton_id: str,
    bone_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    primitive, primitive_evidence = _primitive(package, base_model)
    vertex_array = package.unique_allocation(static.TYPE_VERTEX_BUFFER)
    index_array = package.unique_allocation(static.TYPE_INDEX_BUFFER)
    vertex_entry = static.decode_buffer_entry(
        package, vertex_array, primitive["vertex_buffer"], True
    )
    index_entry = static.decode_buffer_entry(
        package, index_array, primitive["index_buffer"], False
    )
    if vertex_entry["format"] != SKIN_VERTEX_FORMAT or vertex_entry["stride"] != SKIN_VERTEX_BYTES:
        raise ValueError("Landmark does not use the proven 32-byte skinned vertex profile")
    if index_entry["bytes_per_index"] != 2:
        raise ValueError("Landmark does not use 16-bit indices")
    vertex_bytes = package.big_data(vertex_entry["offset"], vertex_entry["bytes"])
    index_bytes = package.big_data(index_entry["offset"], index_entry["bytes"])
    source_indices = list(struct.unpack(f"<{index_entry['count']}H", index_bytes))
    first = primitive["first_index"]
    count = primitive["index_count"]
    if count < 3 or count % 3 or first + count > len(source_indices):
        raise ValueError("Skinned primitive has an invalid triangle range")
    indices = [value + primitive["base_vertex"] for value in source_indices[first : first + count]]
    if min(indices) < 0 or max(indices) >= vertex_entry["count"]:
        raise ValueError("Skinned primitive has an out-of-range vertex index")
    palette = _bone_palette(package, base_model, bone_count)

    positions = []
    uvs = []
    joints = []
    weights = []
    for vertex in range(vertex_entry["count"]):
        offset = vertex * SKIN_VERTEX_BYTES
        position = [
            struct.unpack_from("<e", vertex_bytes, offset + component)[0]
            / SOURCE_UNITS_PER_TILE
            for component in POSITION_OFFSETS
        ]
        uv = list(struct.unpack_from("<2e", vertex_bytes, offset + UV_OFFSET))
        source_joints = list(vertex_bytes[offset + JOINT_OFFSET : offset + JOINT_OFFSET + 4])
        source_weights = list(vertex_bytes[offset + WEIGHT_OFFSET : offset + WEIGHT_OFFSET + 4])
        if sum(source_weights) != 255:
            raise ValueError(f"Vertex {vertex} skin weights do not sum to 255")
        if any(joint >= len(palette) for joint, weight in zip(source_joints, source_weights) if weight):
            raise ValueError(f"Vertex {vertex} references outside the bone palette")
        active = next(joint for joint, weight in zip(source_joints, source_weights) if weight)
        joints.append([palette[joint if weight else active] for joint, weight in zip(source_joints, source_weights)])
        weights.append([weight / 255.0 for weight in source_weights])
        positions.append(position)
        uvs.append(uv)
    if not all(math.isfinite(value) for values in positions + uvs for value in values):
        raise ValueError("Skinned vertex payload contains a non-finite value")

    normal_sums = [[0.0, 0.0, 0.0] for _ in positions]
    for start in range(0, len(indices), 3):
        ia, ib, ic = indices[start : start + 3]
        a, b, c = positions[ia], positions[ib], positions[ic]
        ab = [b[axis] - a[axis] for axis in range(3)]
        ac = [c[axis] - a[axis] for axis in range(3)]
        cross = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ]
        if sum(value * value for value in cross) <= 1.0e-20:
            raise ValueError(f"Skinned primitive contains a degenerate triangle at {start // 3}")
        for index in (ia, ib, ic):
            for axis in range(3):
                normal_sums[index][axis] += cross[axis]
    normals = []
    for index, value in enumerate(normal_sums):
        length = math.sqrt(sum(component * component for component in value))
        if length <= 1.0e-10:
            raise ValueError(f"Skinned vertex {index} has no geometric normal")
        normals.append([component / length for component in value])

    minimum = [min(position[axis] for position in positions) for axis in range(3)]
    maximum = [max(position[axis] for position in positions) for axis in range(3)]
    mesh = {
        "schema": "c3x.normalized_skinned_mesh.v0",
        "asset_id": skeleton_id.removesuffix(".skeleton") + ".mesh",
        "skeleton": skeleton_id,
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Z",
            "position_unit": "tile",
            "uv0_origin": "upper_left",
        },
        "topology": {"primitive": "triangles", "front_face": "counter_clockwise", "indices": indices},
        "vertices": [
            {
                "position": _rounded(position),
                "normal": _rounded(normal),
                "uv0": _rounded(uv),
                "joints": joint,
                "weights": _rounded(weight),
            }
            for position, normal, uv, joint, weight in zip(positions, normals, uvs, joints, weights)
        ],
        "bounds": {"minimum": _rounded(minimum), "maximum": _rounded(maximum)},
        "material_slots": [{"slot": 0, "name": "resource_surface", "triangle_start": 0, "triangle_count": len(indices) // 3}],
    }
    evidence = {
        **primitive_evidence,
        "primitive": primitive,
        "vertex_buffer": vertex_entry,
        "index_buffer": index_entry,
        "bone_palette": palette,
        "vertices": len(positions),
        "triangles": len(indices) // 3,
        "source_vertex_sha256": hashlib.sha256(vertex_bytes).hexdigest(),
        "source_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "uv_range": [
            [min(uv[0] for uv in uvs), min(uv[1] for uv in uvs)],
            [max(uv[0] for uv in uvs), max(uv[1] for uv in uvs)],
        ],
    }
    return mesh, evidence


def _decode_material(
    package: IndexedStaticPackage, base_model: int, shared_data: Path, pack: Path, stem: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    materials = package.unique_pointer_field(base_model, static.TYPE_MATERIAL)
    decoded = []
    data_pointers = []
    for index in range(package.allocations[materials - 1]["element_count"]):
        record = package.array_element(materials, index)
        user_data = struct.unpack_from("<Q", record)[0]
        data_pointer = _one_typed_child(package, user_data, static.TYPE_MATERIAL_DATA)
        raw = package.bytes_for(data_pointer)
        if len(raw) != 64:
            raise ValueError("Skinned material-data record is not 64 bytes")
        decoded.append(
            {
                "normal_0": struct.unpack_from("<I", raw, 0x1C)[0],
                "normal_1": struct.unpack_from("<I", raw, 0x20)[0],
                "base_color": struct.unpack_from("<I", raw, 0x24)[0],
                "gloss": struct.unpack_from("<I", raw, 0x28)[0],
                "emissive": struct.unpack_from("<I", raw, 0x3C)[0],
            }
        )
        data_pointers.append(data_pointer)
    if not decoded or any(value != decoded[0] for value in decoded[1:]):
        raise ValueError("Tile-base material slots do not share one proven material")
    texture_array = package.unique_allocation(static.TYPE_TEXTURE)
    expected_classes = {
        "normal_0": "LEAN",
        "normal_1": "LEAN",
        "base_color": "Generic_BaseColor",
        "gloss": "Generic_Gloss",
        "emissive": "Generic_Emissive",
    }
    material: dict[str, Any] = {
        "schema": "c3x.material.v0",
        "name": stem,
        "alpha_mode": "opaque",
        "status": "normalized_local_import",
    }
    evidence: dict[str, Any] = {"material_data": data_pointers, "textures": {}}
    for role, texture_index in decoded[0].items():
        if texture_index == 0xFFFFFFFF:
            continue
        entry = static.decode_texture_entry(package, texture_array, texture_index)
        if entry["class"] != expected_classes[role]:
            raise ValueError(f"Skinned {role} texture has unexpected class {entry['class']}")
        relative = f"textures/resources/{stem}_{role}.dds"
        source = shared_data / entry["name"]
        if not source.is_file():
            raise ValueError(f"Missing skinned landmark texture: {source}")
        texture_evidence = static.extract_civbig_texture(source, pack / relative)
        evidence["textures"][role] = {**entry, **texture_evidence, "source": str(source)}
        if role == "base_color":
            material[role] = {
                "texture": relative,
                "format": texture_evidence["format_name"],
                "color_space": texture_evidence["color_space"],
                "uv_channel": "uv0",
            }
        else:
            material[role] = {"texture": relative, "uv_channel": "uv0"}
    if "base_color" not in material:
        raise ValueError("Skinned landmark has no base-color texture")
    if "emissive" in material:
        material["emissive"] = {
            "mask": material["emissive"]["texture"],
            "color": [1.0, 1.0, 1.0],
            "intensity": 1.0,
            "activation": "night",
            "missing_policy": "non-emissive",
        }
    return material, evidence


def extract_skins(
    inventory: dict[str, Any], assets_root: Path, pack: Path
) -> dict[str, Any]:
    candidates = [
        (resource["resource_id"], landmark["asset"])
        for resource in inventory["resources"]
        for landmark in resource["landmarks"]
        if landmark["asset"]["package"] == "landmarks/tilebases"
    ]
    package_path = assets_root / "Base" / "Platforms" / "Windows" / "BLPs" / "landmarks" / "tilebases.blp"
    package = IndexedStaticPackage(package_path, candidates[0][1]["entry"])
    shared_data = assets_root / "Base" / "Platforms" / "Windows" / "BLPs" / "SHARED_DATA"
    results = []
    for resource_id, source_asset in candidates:
        stem = resource_id.removeprefix("resource/").replace("/", "_")
        try:
            package.select_direct_string(source_asset["entry"])
            _landmark, _user_data, base_model = static.landmark_base_model(package)
            package_model = _package_model(package, base_model)
            skeleton_id = f"resource.{stem}.skeleton"
            skeleton, skeleton_evidence = _decode_skeleton(package, package_model, skeleton_id)
            mesh, mesh_evidence = _decode_mesh(package, base_model, skeleton_id, len(skeleton["bones"]))
            material, material_evidence = _decode_material(package, base_model, shared_data, pack, stem)
            skeleton_relative = f"skeletons/resources/{stem}.json"
            mesh_relative = f"meshes/resources/{stem}.json"
            material_relative = f"materials/resources/{stem}.json"
            _write_json(pack / skeleton_relative, skeleton)
            _write_json(pack / mesh_relative, mesh)
            _write_json(pack / material_relative, material)
            results.append(
                {
                    "resource_id": resource_id,
                    "source": source_asset,
                    "status": "normalized",
                    "asset": {
                        "type": "skinned_resource",
                        "mesh": mesh_relative,
                        "skeleton": skeleton_relative,
                        "material": material_relative,
                    },
                    "evidence": {
                        "base_model": base_model,
                        "skeleton": skeleton_evidence,
                        "mesh": mesh_evidence,
                        "material": material_evidence,
                    },
                }
            )
        except (OSError, ValueError, struct.error) as error:
            results.append(
                {"resource_id": resource_id, "source": source_asset, "status": "unsupported", "reason": str(error)}
            )
    return {
        "schema": "c3x.resource_skin_extract.v0",
        "source_package": str(package_path),
        "assets": results,
        "summary": {
            "candidates": len(results),
            "normalized": sum(item["status"] == "normalized" for item in results),
            "unsupported": sum(item["status"] == "unsupported" for item in results),
        },
    }
