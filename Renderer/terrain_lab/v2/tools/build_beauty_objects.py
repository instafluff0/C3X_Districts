#!/usr/bin/env python3
"""Bake source-neutral tree, city, and Warrior inputs for beauty studies."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from Renderer.preview.render_unit_turntable import _rigid_mesh, _skinned_mesh
from Renderer.tools.asset_compiler import normalized_pose_cache, normalized_skin


OUTPUT = ROOT / "Renderer/packs/BeautyStudies"
VEGETATION = ROOT / "Renderer/packs/Civ5EnvironmentVegetation"
CITY = ROOT / "Renderer/packs/CityComponentsNormalized"
WARRIOR = ROOT / "Renderer/packs/UnitWarriorLab"
TREE_IDS = (
    "feature/forest/leafy_clump_01",
    "feature/forest/leafy_v2_01",
    "feature/forest/leafy_v3_02",
    "feature/forest/leafy_v4_03",
    "feature/forest/pine_01",
    "feature/forest/pine_03",
)
CITY_POOL = "city/pool/european/medieval"
WARRIOR_ROLES = ("body", "head", "armor", "hair", "weapon")
CHANNELS = ("base_color", "normal_0", "normal_1", "ambient_occlusion", "gloss", "emissive")
KIND = {"tree": 1, "city": 2, "warrior": 3}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def material_record(pack: Path, path: str, owner_tint: bool = False) -> dict[str, Any]:
    source = read_json(pack / path)
    if "channels" in source:
        channels = source["channels"]
        textures = {
            name: channels.get(name, {}).get("texture", "") for name in CHANNELS
        }
    else:
        textures = {
            "base_color": source["base_color"]["texture"],
            "normal_0": source.get("lean_normal", {}).get("texture_0", ""),
            "normal_1": source.get("lean_normal", {}).get("texture_1", ""),
            "ambient_occlusion": "",
            "gloss": source.get("gloss", {}).get("texture", ""),
            "emissive": "",
        }
    return {
        "textures": {
            name: str((pack / texture).relative_to(ROOT)) if texture else ""
            for name, texture in textures.items()
        },
        "owner_tint": owner_tint,
    }


def expanded_vertices(mesh: dict[str, Any]) -> list[dict[str, Any]]:
    vertices = mesh["vertices"]
    return [vertices[index] for index in mesh["topology"]["indices"]]


def add_object(
    objects: list[dict[str, Any]], materials: list[dict[str, Any]], object_id: str,
    kind: str, mesh: dict[str, Any], material: dict[str, Any],
) -> None:
    try:
        material_index = materials.index(material)
    except ValueError:
        material_index = len(materials)
        materials.append(material)
    objects.append({
        "id": object_id,
        "kind": KIND[kind],
        "material": material_index,
        "vertices": expanded_vertices(mesh),
    })


def add_trees(objects: list[dict[str, Any]], materials: list[dict[str, Any]]) -> None:
    manifest = read_json(VEGETATION / "manifest.json")
    for asset_id in TREE_IDS:
        asset = manifest["assets"][asset_id]
        add_object(
            objects, materials, asset_id, "tree",
            read_json(VEGETATION / asset["mesh"]),
            material_record(VEGETATION, asset["material"]),
        )


def add_city(objects: list[dict[str, Any]], materials: list[dict[str, Any]]) -> None:
    manifest = read_json(CITY / "manifest.json")
    catalog = read_json(CITY / manifest["city_catalog"])
    for asset_id in catalog["pools"][CITY_POOL]["components"]:
        landmark = read_json(CITY / manifest["assets"][asset_id]["landmark"])
        mesh_path = landmark["components"]["geometry"][0]
        material_path = landmark["components"]["materials"][0]
        add_object(
            objects, materials, asset_id, "city", read_json(CITY / mesh_path),
            material_record(CITY, material_path),
        )


def posed_warrior() -> list[tuple[str, dict[str, Any], str, bool]]:
    manifest = read_json(WARRIOR / "manifest.json")
    action = manifest["animations"]["animation/unit/warrior/idle"]
    recipe = read_json(WARRIOR / manifest["units"]["unit/warrior"]["recipe"])
    components: dict[str, dict[str, Any]] = {}
    worlds: dict[str, list[list[float]]] = {}
    sample_time = action["duration"] * 0.28
    for role in WARRIOR_ROLES:
        asset_id = f"unit/warrior/{role}"
        document = read_json(WARRIOR / manifest["assets"][asset_id]["component"])
        skeleton = normalized_skin.load_skeleton(WARRIOR / document["skeleton"])
        mesh_path = WARRIOR / document["mesh"]
        mesh = (
            normalized_skin.load_mesh(mesh_path, len(skeleton["bones"]))
            if document["binding_mode"] == "vertex_skin"
            else read_json(mesh_path)
        )
        components[role] = {"document": document, "skeleton": skeleton, "mesh": mesh}
        if document["binding_mode"] == "vertex_skin":
            pose = normalized_pose_cache.load_pose_cache(
                WARRIOR / action["pose_caches"][asset_id]["path"]
            )
            normalized_pose_cache.validate_skeleton_binding(pose, skeleton)
            worlds[role] = pose.sample(sample_time, action["loop"])

    body = components["body"]
    body_names = [bone["name"] for bone in body["skeleton"]["bones"]]
    result = []
    for role in WARRIOR_ROLES:
        component = components[role]
        document = component["document"]
        if document["binding_mode"] == "vertex_skin":
            mesh = _skinned_mesh(component["mesh"], component["skeleton"], worlds[role])
        else:
            socket = manifest["unit_binding"]["sockets"][document["attachment_point"]]
            mesh = _rigid_mesh(
                component["mesh"], worlds["body"][body_names.index(socket["bone"])],
                document["model_scale"],
            )
        result.append((role, mesh, document["material"], document.get("tint") == "USE_CIV_COLOR"))

    positions = [
        vertex["position"] for _role, mesh, _material, _owner in result
        for vertex in mesh["vertices"]
    ]
    center_x = (min(p[0] for p in positions) + max(p[0] for p in positions)) * 0.5
    center_y = (min(p[1] for p in positions) + max(p[1] for p in positions)) * 0.5
    floor = min(p[2] for p in positions)
    for _role, mesh, _material, _owner in result:
        for vertex in mesh["vertices"]:
            p = vertex["position"]
            vertex["position"] = [p[0] - center_x, p[1] - center_y, p[2] - floor]
    return result


def add_warrior(objects: list[dict[str, Any]], materials: list[dict[str, Any]]) -> None:
    for role, mesh, material_path, owner in posed_warrior():
        add_object(
            objects, materials, f"unit/warrior/{role}", "warrior", mesh,
            material_record(WARRIOR, material_path, owner),
        )


def serialize(materials: list[dict[str, Any]], objects: list[dict[str, Any]]) -> bytes:
    output = bytearray(b"C3XBTO1\0")
    output.extend(struct.pack("<III", 1, len(materials), len(objects)))
    for material in materials:
        for channel in CHANNELS:
            output.extend(string(material["textures"][channel]))
        output.extend(struct.pack("<I", int(material["owner_tint"])))
    for obj in objects:
        output.extend(string(obj["id"]))
        output.extend(struct.pack("<III", obj["kind"], obj["material"], len(obj["vertices"])))
        for vertex in obj["vertices"]:
            output.extend(struct.pack(
                "<8f", *(vertex["position"] + vertex["normal"] + vertex["uv0"])
            ))
    return bytes(output)


def main() -> int:
    materials: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    add_trees(objects, materials)
    add_city(objects, materials)
    add_warrior(objects, materials)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bundle = OUTPUT / "beauty_objects.bin"
    bundle.write_bytes(serialize(materials, objects))
    manifest = {
        "schema": "c3x.beauty_studies.v1",
        "runtime": "beauty_objects.bin",
        "source_packs": [
            "Renderer/packs/Civ5EnvironmentVegetation",
            "Renderer/packs/CityComponentsNormalized",
            "Renderer/packs/UnitWarriorLab",
        ],
        "tree_ids": list(TREE_IDS),
        "city_pool": CITY_POOL,
        "warrior_action": {"name": "idle", "phase": 0.28},
        "material_count": len(materials),
        "object_count": len(objects),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {bundle.relative_to(ROOT)}: {len(objects)} objects, "
        f"{len(materials)} materials, {bundle.stat().st_size} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
