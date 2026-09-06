#!/usr/bin/env python3
"""Compile resolved Civ VI unit members into a source-independent lab pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler.clutter_blp_extractor import (
    TYPE_INDEX_BUFFER,
    TYPE_MATERIAL,
    TYPE_MESH,
    TYPE_MODEL,
    TYPE_PRIM_GROUP,
    TYPE_VERTEX_BUFFER,
    decode_buffer_entry,
)
from Renderer.tools.asset_compiler.compound_landmark_importer import (
    SKINNED_VERTEX_PROFILE,
    _decode_materials,
    _decode_primitive,
    _decode_skeletons,
    _decode_states,
    _normalize_geometry,
)
from Renderer.tools.asset_compiler.grassland_pack_builder import validate_runtime_independence
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage
from Renderer.tools.asset_compiler import normalized_skin
from Renderer.tools.asset_compiler.unit_member_resolver import ASSETS_ROOT, resolve_unit


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK = RENDERER_ROOT / "packs/UnitWarriorLab"
DEFAULT_REPORT = RENDERER_ROOT / "preview/out/units/warrior_build.json"
TYPE_MODEL_ENTRY = "ModelPackageEntry"
TYPE_USER_DATA = "BLP::BLPPtr<FGXModel::IUserData>"
TYPE_BASE_MODEL = "ModelPackageEntry::BaseModelData_Entry"
SOURCE_UNITS_PER_TILE = 100.0


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_base(package: IndexedStaticPackage, source_entry: str) -> tuple[int, int, int]:
    target_pointers = {
        pointer
        for pointer in range(1, len(package.allocations) + 1)
        if package.direct_string(pointer) == source_entry
    }
    if not target_pointers:
        raise ValueError(f"Model package does not contain source entry {source_entry}")
    matches = []
    for pointer in range(1, len(package.allocations) + 1):
        if package.type_name(pointer) != TYPE_MODEL_ENTRY:
            continue
        raw = package.bytes_for(pointer)
        if len(raw) == 72 and struct.unpack_from("<Q", raw, 0x38)[0] in target_pointers:
            matches.append(pointer)
    if len(matches) != 1:
        raise ValueError(f"Expected one ModelPackageEntry for {source_entry}, found {len(matches)}")
    entry = matches[0]
    user_data = struct.unpack_from("<Q", package.bytes_for(entry), 0x20)[0]
    if package.type_name(user_data) != TYPE_USER_DATA or len(package.bytes_for(user_data)) != 24:
        raise ValueError(f"{source_entry} has an unexpected model user-data record")
    fields = package.pointer_fields(user_data, TYPE_BASE_MODEL)
    if len(fields) != 1:
        raise ValueError(f"{source_entry} has no unique base-model record")
    return entry, user_data, fields[0][1]


def _normalized_skeleton(
    source: dict[str, Any], asset_id: str, adapter: str = "c3x.unit_component.v0"
) -> dict[str, Any]:
    return {
        "schema": "c3x.normalized_skeleton.v0",
        "asset_id": asset_id + "/skeleton",
        "track_group": source["track_group"],
        "matrix_convention": "row_major_row_vector",
        "position_unit": "tile",
        "bones": [
            {
                "name": bone["name"],
                "parent": bone["parent"],
                "local": bone["rest"],
                "inverse_bind_matrix": bone["inverse_bind_matrix"],
            }
            for bone in source["bones"]
        ],
        "validation": {
            "bind_pose_max_error": source["bind_pose_max_error"],
            "inverse_bind_status": "passed",
        },
        "provenance": {
            "kind": "local_normalized_import",
            "adapter": adapter,
            "source_format_dependency": None,
        },
    }


def _normalized_mesh(
    mesh: dict[str, Any], asset_id: str, adapter: str = "c3x.unit_component.v0"
) -> dict[str, Any]:
    if mesh["schema"] == "c3x.skinned_mesh.v0":
        mesh["schema"] = "c3x.normalized_skinned_mesh.v0"
        for vertex in mesh["vertices"]:
            skin = vertex.pop("skin")
            pairs = list(zip(skin["bone_indices"], skin["bone_weights"]))
            while len(pairs) < 4:
                pairs.append((0, 0.0))
            vertex["joints"] = [item[0] for item in pairs]
            vertex["weights"] = [item[1] for item in pairs]
    mesh["asset_id"] = asset_id + "/mesh"
    mesh["provenance"] = {
        "kind": "local_normalized_import",
        "adapter": adapter,
        "source_format_dependency": None,
    }
    return mesh


def _remap_skin_palette(mesh: dict[str, Any], palette: list[int], bone_count: int) -> None:
    if not palette or any(index < 0 or index >= bone_count for index in palette):
        raise ValueError("unit skin palette contains an invalid skeleton index")
    for vertex in mesh["vertices"]:
        skin = vertex.get("skin")
        if skin is None or any(index < 0 or index >= len(palette) for index in skin["bone_indices"]):
            raise ValueError("unit vertex references outside its local skin palette")
        skin["bone_indices"] = [palette[index] for index in skin["bone_indices"]]


def _compile_component(
    package: IndexedStaticPackage,
    shared_data: Path,
    pack: Path,
    component: dict[str, Any],
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]],
    unit_slug: str = "warrior",
    component_key: str | None = None,
    artifact_family: str = "unit",
    component_schema: str = "c3x.unit_component.v0",
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_entry = component["source_entry"]
    role_id = component_key or component["role"].lower()
    if artifact_family not in {"unit", "resource"}:
        raise ValueError("unsupported normalized component family")
    asset_id = f"{artifact_family}/{unit_slug}/{role_id}"
    source_pointers = component.get("resolved_source_pointers")
    if source_pointers is None:
        entry, user_data, base_model = _model_base(package, source_entry)
    else:
        entry = source_pointers.get("entry")
        user_data = source_pointers.get("user_data")
        base_model = source_pointers.get("base_model")
        if not all(isinstance(value, int) and value > 0 for value in (entry, user_data, base_model)):
            raise ValueError(f"{source_entry} has invalid pre-resolved source pointers")
    states, state_evidence = _decode_states(package, user_data)
    skeletons, skeleton_evidence = _decode_skeletons(package, base_model, SOURCE_UNITS_PER_TILE)
    source_model_index = component.get("source_model_index")
    skeleton_index = component.get("source_skeleton_index", source_model_index or 0)
    if source_model_index is None and len(skeletons) != 1:
        raise ValueError(f"First unit component profile requires one skeleton: {source_entry}")
    if not isinstance(skeleton_index, int) or not 0 <= skeleton_index < len(skeletons):
        raise ValueError(f"Component selects an invalid source skeleton: {source_entry}")
    selected_skeleton = skeletons[skeleton_index]
    geometry_fields = {
        target: package.pointer_fields(base_model, target)
        for target in (TYPE_MODEL, TYPE_MESH, TYPE_PRIM_GROUP, TYPE_MATERIAL)
    }
    if any(len(value) != 1 for value in geometry_fields.values()):
        raise ValueError(f"First unit component profile requires complete unique geometry: {source_entry}")
    model_array = geometry_fields[TYPE_MODEL][0][1]
    mesh_array = geometry_fields[TYPE_MESH][0][1]
    primitive_array = geometry_fields[TYPE_PRIM_GROUP][0][1]
    material_array = geometry_fields[TYPE_MATERIAL][0][1]
    counts = {
        "models": package.allocations[model_array - 1]["element_count"],
        "meshes": package.allocations[mesh_array - 1]["element_count"],
        "primitives": package.allocations[primitive_array - 1]["element_count"],
        "materials": package.allocations[material_array - 1]["element_count"],
    }
    if min(counts.values()) < 1:
        raise ValueError(f"Unit component has unsupported geometry counts: {source_entry} {counts}")
    if source_model_index is None:
        if counts["models"] != 1:
            raise ValueError(f"First unit component profile requires one model: {source_entry}")
        source_model_index = 0
    if not isinstance(source_model_index, int) or not 0 <= source_model_index < counts["models"]:
        raise ValueError(f"Component selects an invalid source model: {source_entry}")
    model_record = package.array_element(model_array, source_model_index)
    first_mesh, model_mesh_count = struct.unpack_from("<2I", model_record, 0x18)
    if model_mesh_count < 1 or first_mesh + model_mesh_count > counts["meshes"]:
        raise ValueError(f"Unit component model has an invalid mesh range: {source_entry}")
    primitives = [
        _decode_primitive(
            package,
            package.array_element(primitive_array, index),
            states,
            counts["materials"],
        )
        for index in range(counts["primitives"])
    ]
    vertex_array = package.unique_allocation(TYPE_VERTEX_BUFFER)
    index_array = package.unique_allocation(TYPE_INDEX_BUFFER)
    mesh_records = []
    covered_primitives = []
    total_palette_count = 0
    for mesh_index in range(counts["meshes"]):
        mesh_record = package.array_element(mesh_array, mesh_index)
        first_primitive, mesh_primitive_count = struct.unpack_from("<2I", mesh_record, 0x18)
        if mesh_primitive_count < 1 or first_primitive + mesh_primitive_count > counts["primitives"]:
            raise ValueError(f"Unit component mesh has an invalid primitive range: {source_entry}")
        palette_count = struct.unpack_from("<I", mesh_record, 0x20)[0]
        covered_primitives.extend(range(first_primitive, first_primitive + mesh_primitive_count))
        mesh_records.append(
            (mesh_index, first_primitive, mesh_primitive_count, palette_count, total_palette_count)
        )
        total_palette_count += palette_count
    if sorted(covered_primitives) != list(range(counts["primitives"])):
        raise ValueError(f"Unit component meshes do not cover primitives exactly once: {source_entry}")
    mesh_records = [
        record
        for record in mesh_records
        if first_mesh <= record[0] < first_mesh + model_mesh_count
    ]

    source_palette: list[int] = []
    if total_palette_count:
        base_raw = package.bytes_for(base_model)
        palette_pointer = struct.unpack_from("<Q", base_raw, 0x128)[0]
        if package.type_name(palette_pointer) != "int32":
            raise ValueError(f"{source_entry} has no typed local skin palette")
        palette_raw = package.bytes_for(palette_pointer)
        if len(palette_raw) != total_palette_count * 4 or package.allocations[palette_pointer - 1]["element_count"] != total_palette_count:
            raise ValueError(f"{source_entry} skin-palette count disagrees with its mesh")
        source_palette = list(struct.unpack(f"<{total_palette_count}i", palette_raw))

    stem = f"{unit_slug}_{role_id}"
    skeleton_path = f"skeletons/{artifact_family}/{stem}.json"
    skeleton = _normalized_skeleton(selected_skeleton, asset_id, component_schema)
    _write_json(pack / skeleton_path, skeleton)
    mesh_paths = []
    draw_bindings = []
    geometry_parts = []
    material_addresses: dict[int, set[str]] = {}
    all_skinned = True
    all_rigid = True
    for mesh_index, first_primitive, mesh_primitive_count, palette_count, palette_offset in mesh_records:
        mesh_palette = source_palette[palette_offset : palette_offset + palette_count]
        for primitive_index in range(first_primitive, first_primitive + mesh_primitive_count):
            primitive = primitives[primitive_index]
            vertex_entry = decode_buffer_entry(
                package, vertex_array, primitive["vertex_buffer"], True
            )
            index_entry = decode_buffer_entry(
                package, index_array, primitive["index_buffer"], False
            )
            is_skinned = (
                vertex_entry["format"], vertex_entry["stride"]
            ) == SKINNED_VERTEX_PROFILE
            all_skinned = all_skinned and is_skinned
            all_rigid = all_rigid and not is_skinned
            if is_skinned and not mesh_palette:
                raise ValueError(f"{source_entry} skinned mesh has no local palette")
            mesh, geometry_evidence = _normalize_geometry(
                package.big_data(vertex_entry["offset"], vertex_entry["bytes"]),
                package.big_data(index_entry["offset"], index_entry["bytes"]),
                vertex_entry,
                index_entry,
                primitive,
                SOURCE_UNITS_PER_TILE,
                len(selected_skeleton["bones"]) if is_skinned else None,
                normalize_skin_weights=True,
            )
            if is_skinned:
                _remap_skin_palette(mesh, mesh_palette, len(selected_skeleton["bones"]))
            geometry_index = len(mesh_paths)
            part_asset_id = asset_id if counts["primitives"] == 1 else f"{asset_id}/part_{geometry_index:02d}"
            mesh = _normalized_mesh(mesh, part_asset_id, component_schema)
            mesh_path = (
                f"meshes/{artifact_family}/{stem}.json"
                if counts["primitives"] == 1
                else f"meshes/{artifact_family}/{stem}_{geometry_index:02d}.json"
            )
            _write_json(pack / mesh_path, mesh)
            mesh_paths.append(mesh_path)
            material_addresses.setdefault(primitive["material_index"], set()).add(
                geometry_evidence["uv_address"]
            )
            draw_bindings.append(
                {
                    "mesh": geometry_index,
                    "material": primitive["material_index"],
                    "states": primitive["states"],
                    "binding_mode": "vertex_skin" if is_skinned else "rigid_attachment",
                }
            )
            rest_validation = None
            if is_skinned:
                rest_validation = normalized_skin.validate_rest_pose(
                    normalized_skin.load_mesh(pack / mesh_path, len(skeleton["bones"])),
                    normalized_skin.load_skeleton(pack / skeleton_path),
                )
            geometry_parts.append(
                {
                    **geometry_evidence,
                    "mesh_index": mesh_index,
                    "primitive_index": primitive_index,
                    "skin_palette": mesh_palette or None,
                    "rest_pose": rest_validation,
                }
            )
    material_variants = [
        (material_index, address)
        for material_index in sorted(material_addresses)
        for address in sorted(material_addresses[material_index])
    ]
    variant_indices = {
        variant: index for index, variant in enumerate(material_variants)
    }
    for binding in draw_bindings:
        address = geometry_parts[binding["mesh"]]["uv_address"]
        binding["material"] = variant_indices[(binding["material"], address)]
    material_paths, material_evidence = _decode_materials(
        package,
        material_array,
        shared_data,
        pack,
        stem,
        material_variants,
        texture_cache,
        artifact_family=artifact_family,
        adapter=component_schema,
    )
    document_path = f"{artifact_family}s/components/{stem}.json"
    binding_mode = (
        "vertex_skin" if all_skinned else "rigid_attachment" if all_rigid else "mixed"
    )
    document = {
        "schema": component_schema,
        "asset_id": asset_id,
        "role": component["role"],
        "attachment_point": component["point"],
        "model_scale": component["scale"],
        "tint": component["tint"],
        "states": states,
        "meshes": mesh_paths,
        "materials": material_paths,
        "draw_bindings": draw_bindings,
        "skeleton": skeleton_path,
        "binding_mode": binding_mode,
        "provenance": {
            "kind": "local_normalized_import",
            "adapter": component_schema,
            "source_format_dependency": None,
        },
    }
    if len(mesh_paths) == 1 and len(material_paths) == 1:
        document["mesh"] = mesh_paths[0]
        document["material"] = material_paths[0]
    _write_json(pack / document_path, document)
    return {"type": f"{artifact_family}_component", "component": document_path}, {
        "source_entry": source_entry,
        "role": component["role"],
        "attachment_point": component["point"],
        "tint": component["tint"],
        "pointer_chain": {
            "model_entry": entry,
            "user_data": user_data,
            "base_model": base_model,
            **state_evidence,
            **skeleton_evidence,
        },
        "geometry": {
            "skinned": not all_rigid,
            "binding_mode": binding_mode,
            "parts": geometry_parts,
            "mesh_count": counts["meshes"],
            "primitive_count": counts["primitives"],
        },
        "skin_palette": source_palette or None,
        "materials": material_evidence,
        "rest_pose": [part["rest_pose"] for part in geometry_parts if part["rest_pose"]],
        "source_model_index": source_model_index,
        "source_skeleton_index": skeleton_index,
        "track_group": selected_skeleton["track_group"],
        "bones": len(selected_skeleton["bones"]),
    }


def compile_unit_warrior(
    assets_root: Path, pack: Path = DEFAULT_PACK, report_path: Path = DEFAULT_REPORT
) -> dict[str, Any]:
    recipe = resolve_unit(assets_root, "UNIT_WARRIOR", "Any")
    source_package = assets_root / "Base/Platforms/Windows/BLPs/units/units.blp"
    shared_data = assets_root / "Base/Platforms/Windows/BLPs/SHARED_DATA"
    if not source_package.is_file() or not shared_data.is_dir():
        raise FileNotFoundError(source_package if not source_package.is_file() else shared_data)
    package = IndexedStaticPackage(source_package, recipe["selected_components"][0]["source_entry"])
    assets = {}
    evidence = []
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for component in recipe["selected_components"]:
        asset, report = _compile_component(package, shared_data, pack, component, texture_cache)
        assets[f"unit/warrior/{component['role'].lower()}"] = asset
        evidence.append(report)
    recipe_path = "units/warrior_recipe.json"
    runtime_recipe = {
        "schema": "c3x.unit_recipe.v0",
        "unit_id": "unit/warrior",
        "member": {
            "count": recipe["member"]["count"],
            "member_scale": recipe["member"]["member_scale"],
            "variation_scale": recipe["member"]["variation_scale"],
        },
        "components": [
            {
                "asset": f"unit/warrior/{item['role'].lower()}",
                "role": item["role"],
                "attachment_point": item["point"],
                "scale": item["scale"],
                "tint": item["tint"],
            }
            for item in recipe["selected_components"]
        ],
        "formation": recipe["formation"],
        "movement": recipe["movement"],
        "actions": {
            action: f"animation/unit/warrior/{action}" for action in recipe["actions"]
        },
        "runtime_integration": "not_enabled",
    }
    _write_json(pack / recipe_path, runtime_recipe)
    manifest = {
        "schema": "c3x.unit_pack.v0",
        "name": "UnitWarriorLab",
        "source_policy": "Local licensed-source import; derived art is not redistributable.",
        "units": {"unit/warrior": {"recipe": recipe_path}},
        "assets": assets,
        "animations": {},
        "runtime_integration": "not_enabled",
    }
    _write_json(pack / "manifest.json", manifest)
    independence_errors = validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError("Runtime pack is source-dependent: " + "; ".join(independence_errors))
    report = {
        "schema": "c3x.source_unit_build.v0",
        "unit": "UNIT_WARRIOR",
        "source_package": {"path": str(source_package), "sha256": _sha256(source_package.read_bytes())},
        "source_units_per_tile": SOURCE_UNITS_PER_TILE,
        "recipe": recipe,
        "components": evidence,
        "outputs": {
            "pack": str(pack),
            "components": len(evidence),
            "skinned_components": sum(item["geometry"]["skinned"] for item in evidence),
            "rigid_components": sum(not item["geometry"]["skinned"] for item in evidence),
            "textures": len(texture_cache),
        },
        "runtime_independence": "passed",
        "runtime_integration": "not_enabled",
    }
    _write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=ASSETS_ROOT)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        report = compile_unit_warrior(args.assets_root, args.pack, args.report)
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Compiled UNIT_WARRIOR: {report['outputs']['components']} components "
        f"({report['outputs']['skinned_components']} skinned, {report['outputs']['rigid_components']} rigid)"
    )
    print(f"Pack: {args.pack}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
