#!/usr/bin/env python3
"""Bake deterministic source-backed L20 unit action samples for Terrain Lab."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Renderer.preview.render_unit_turntable import _rigid_mesh, _skinned_mesh
from Renderer.tools.asset_compiler import normalized_animation, normalized_skin
from Renderer.tools.asset_compiler.build_mine_runtime import (
    MAGIC, bundle_string, group_payload, merged_asset,
)
from Renderer.tools.asset_compiler.unit_family_action_validator import SOCKET_PROFILE, _best_group


UNITS = ("archer", "swordsman", "infantry", "fighter", "galley")
OWNER_COLOR_OVERRIDES = json.loads(
    Path(__file__).with_name("unit_family_owner_color_overrides.json").read_text(
        encoding="utf-8"
    )
)["overrides"]
SOURCE_TINT_CODES = {
    "BaseMale_SkinColor_Caucasian": 1,
    "GreatPeople_Military": 2,
    "Horse_Default": 3,
    "Horse_Secondary": 4,
    "Infantry_European": 5,
    "Vehicle_Woodland": 6,
    "Wood": 7,
}
ACTIONS = {
    "idle": (0.37,),
    "fidget": (0.61,),
    "move": (0.12, 0.37, 0.62, 0.87),
    "fortify": (1.0,),
    "attack": (0.08, 0.36, 0.64, 0.92),
    "defend": (0.61,),
    "victory": (0.72,),
    "death": (0.08, 0.36, 0.64, 0.88),
}
WORKER_ACTIONS = {
    "idle": ("family", "idle", (0.37,)),
    "fidget": ("family", "fidget", (0.61,)),
    "move": ("family", "move", (0.12, 0.37, 0.62, 0.87)),
    "fortify": ("worker", "work_ground", (1.0,)),
    "attack": ("worker", "work_heavy", (0.08, 0.36, 0.64, 0.92)),
    "defend": ("family", "defend", (0.61,)),
    "victory": ("family", "victory", (0.72,)),
    "death": ("family", "death", (0.08, 0.36, 0.64, 0.88)),
    "work_ground": ("worker", "work_ground", (0.12, 0.37, 0.62, 0.87)),
    "work_heavy": ("worker", "work_heavy", (0.12, 0.37, 0.62, 0.87)),
    "work_cut": ("worker", "work_cut", (0.12, 0.37, 0.62, 0.87)),
    "capture": ("worker", "captured_1", (0.12, 0.37, 0.62, 0.87)),
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def component_document(pack: Path, manifest: dict, asset_id: str):
    return load_json(pack / manifest["assets"][asset_id]["component"])


def owner_color_style(asset_id: str, document: dict) -> int:
    """Encode mask source and authored strength in the Lab material marker."""
    contract = OWNER_COLOR_OVERRIDES.get(asset_id, document.get("owner_color", {}))
    if contract.get("mode") == "none" or float(contract.get("strength", 0.0)) <= 0.0:
        return 0
    source_code = 1 if contract.get("mask_source") == "base_color_alpha_inverse" else 2
    strength = float(contract.get("strength", 0.0))
    tier_offset = 0 if strength >= 0.70 else (2 if strength >= 0.46 else 4)
    return source_code + tier_offset


def source_tint_style(document: dict) -> int:
    """Encode the named Civ VI material tint retained by the generic component."""
    return SOURCE_TINT_CODES.get(document.get("tint"), 0)


def posed_parts(pack: Path, manifest: dict, recipe: dict, action: str, phase: float):
    animation = manifest["animations"][recipe["actions"][action]]
    clip = normalized_animation.load_clip(pack / animation["clip"])
    sample_time = clip.duration * phase
    documents = {
        record["asset"]: component_document(pack, manifest, record["asset"])
        for record in recipe["components"]
    }
    skeletons = {
        asset_id: normalized_skin.load_skeleton(pack / document["skeleton"])
        for asset_id, document in documents.items()
    }
    worlds = {}
    for asset_id, document in documents.items():
        if document["binding_mode"] != "vertex_skin":
            continue
        skeleton = skeletons[asset_id]
        group_index, _common = _best_group(
            clip, {bone["name"] for bone in skeleton["bones"]})
        pose = normalized_skin.sample_pose(
            skeleton, clip, group_index, sample_time, animation["loop"])
        worlds[asset_id] = normalized_skin.world_matrices(skeleton, pose)

    driver_id = recipe["animation_driver"]
    driver_skeleton = skeletons[driver_id]
    driver_names = [bone["name"] for bone in driver_skeleton["bones"]]
    driver_worlds = worlds[driver_id]
    parts = []
    for record in recipe["components"]:
        asset_id = record["asset"]
        document = documents[asset_id]
        skeleton = skeletons[asset_id]
        owner = owner_color_style(asset_id, document)
        source_tint = source_tint_style(document)
        for binding in document["draw_bindings"]:
            mesh_paths = document.get("meshes")
            if mesh_paths is None:
                mesh_paths = [document["mesh"]]
            mesh_path = mesh_paths[binding["mesh"]]
            if document["binding_mode"] == "vertex_skin":
                mesh = normalized_skin.load_mesh(pack / mesh_path, len(skeleton["bones"]))
                posed = _skinned_mesh(mesh, skeleton, worlds[asset_id])
            else:
                mesh = load_json(pack / mesh_path)
                profile = SOCKET_PROFILE[document["attachment_point"]]
                posed = _rigid_mesh(
                    mesh,
                    driver_worlds[driver_names.index(profile["bone"])],
                    document["model_scale"],
                )
            material_paths = document.get("materials")
            if material_paths is None:
                material_paths = [document["material"]]
            material_path = material_paths[binding["material"]]
            material = load_json(pack / material_path)
            texture = material["channels"]["base_color"]["texture"]
            parts.append((posed, texture, owner, source_tint))
    return parts


def write_unit_bundle(pack: Path, manifest: dict, slug: str):
    recipe = load_json(pack / manifest["units"][f"unit/{slug}"]["recipe"])
    cached = {}
    textures = []
    for action, phases in ACTIONS.items():
        for phase in phases:
            parts = posed_parts(pack, manifest, recipe, action, phase)
            cached[(action, phase)] = parts
            for _mesh, texture, _owner, _source_tint in parts:
                if texture not in textures:
                    textures.append(texture)
    if len(textures) > 8:
        raise ValueError(f"{slug} needs {len(textures)} base textures; Terrain Lab ABI permits 8")

    assets = []
    groups = []
    member_scale = recipe["member"]["member_scale"] * recipe["member"]["variation_scale"]
    bounds = [1.0e30, -1.0e30, 1.0e30, -1.0e30, 1.0e30, -1.0e30]
    for action, phases in ACTIONS.items():
        for phase_index, phase in enumerate(phases):
            placements = []
            group_parts = cached[(action, phase)]
            positions = [vertex["position"] for mesh, _texture, _owner, _source_tint in group_parts
                         for vertex in mesh["vertices"]]
            minimum = [min(position[axis] for position in positions) for axis in range(3)]
            maximum = [max(position[axis] for position in positions) for axis in range(3)]
            center_x = (minimum[0] + maximum[0]) * 0.5
            center_y = (minimum[1] + maximum[1]) * 0.5
            for part_index, (mesh, texture, owner, source_tint) in enumerate(group_parts):
                for vertex in mesh["vertices"]:
                    position = vertex["position"]
                    vertex["position"] = [
                        (position[0] - center_x) * member_scale,
                        (position[1] - center_y) * member_scale,
                        (position[2] - minimum[2]) * member_scale,
                    ]
                    for axis in range(3):
                        bounds[axis * 2] = min(bounds[axis * 2], vertex["position"][axis])
                        bounds[axis * 2 + 1] = max(bounds[axis * 2 + 1], vertex["position"][axis])
                asset_index = len(assets)
                assets.append(merged_asset(
                    f"{slug}:{action}:{phase_index}:part_{part_index}:t{owner}:n{source_tint}",
                    textures.index(texture), 0, [mesh]))
                placements.append((asset_index, 1.0))
            groups.append(group_payload(f"{action}_{phase_index}", placements))

    target = pack / f"unit_{slug}_runtime.bin"
    output = bytearray(MAGIC)
    output.extend(struct.pack("<IIII", 1, len(textures), len(assets), len(groups)))
    for texture in textures:
        output.extend(bundle_string(texture))
    for asset in assets:
        output.extend(asset)
    for group in groups:
        output.extend(group)
    target.write_bytes(output)
    return target, bounds, len(textures), len(groups)


def worker_parts(pack: Path, manifest: dict, recipe: dict, clip_path: Path,
                 loop: bool, phase: float):
    clip = normalized_animation.load_clip(clip_path)
    parts = []
    for record in recipe["components"]:
        if record["role"] not in ("Body", "Head", "Armor"):
            continue
        asset_id = record["asset"]
        document = component_document(pack, manifest, asset_id)
        skeleton = normalized_skin.load_skeleton(pack / document["skeleton"])
        group_index, _common = _best_group(
            clip, {bone["name"] for bone in skeleton["bones"]})
        pose = normalized_skin.sample_pose(
            skeleton, clip, group_index, clip.duration * phase, loop)
        worlds = normalized_skin.world_matrices(skeleton, pose)
        owner = owner_color_style(asset_id, document)
        source_tint = source_tint_style(document)
        for binding in document["draw_bindings"]:
            mesh_paths = document.get("meshes", [document.get("mesh")])
            mesh = normalized_skin.load_mesh(
                pack / mesh_paths[binding["mesh"]], len(skeleton["bones"]))
            material_paths = document.get("materials", [document.get("material")])
            material = load_json(pack / material_paths[binding["material"]])
            texture = material["channels"]["base_color"]["texture"]
            parts.append((_skinned_mesh(mesh, skeleton, worlds), texture, owner, source_tint))
    return parts


def write_worker_bundle(pack: Path, manifest: dict, worker_pack: Path):
    recipe = load_json(pack / manifest["units"]["unit/swordsman"]["recipe"])
    cached = {}
    textures = []
    for action, (source, clip_name, phases) in WORKER_ACTIONS.items():
        for phase in phases:
            if source == "family":
                parts = posed_parts(pack, manifest, recipe, clip_name, phase)
            else:
                parts = worker_parts(
                    pack, manifest, recipe,
                    worker_pack / "animations" / "unit" / "worker" /
                    f"{clip_name}.c3anim", action in ("work_ground", "work_heavy", "work_cut"),
                    phase)
            cached[(action, phase)] = parts
            for _mesh, texture, _owner, _source_tint in parts:
                if texture not in textures:
                    textures.append(texture)
    assets, groups = [], []
    bounds = [1.0e30, -1.0e30, 1.0e30, -1.0e30, 1.0e30, -1.0e30]
    for action, (_source, _clip_name, phases) in WORKER_ACTIONS.items():
        for phase_index, phase in enumerate(phases):
            parts = cached[(action, phase)]
            positions = [vertex["position"] for mesh, _texture, _owner, _source_tint in parts
                         for vertex in mesh["vertices"]]
            minimum = [min(position[axis] for position in positions) for axis in range(3)]
            maximum = [max(position[axis] for position in positions) for axis in range(3)]
            center_x = (minimum[0] + maximum[0]) * 0.5
            center_y = (minimum[1] + maximum[1]) * 0.5
            placements = []
            for part_index, (mesh, texture, owner, source_tint) in enumerate(parts):
                for vertex in mesh["vertices"]:
                    position = vertex["position"]
                    vertex["position"] = [position[0] - center_x,
                                          position[1] - center_y,
                                          position[2] - minimum[2]]
                    for axis in range(3):
                        bounds[axis * 2] = min(bounds[axis * 2], vertex["position"][axis])
                        bounds[axis * 2 + 1] = max(bounds[axis * 2 + 1], vertex["position"][axis])
                asset_index = len(assets)
                assets.append(merged_asset(
                    f"worker:{action}:{phase_index}:part_{part_index}:t{owner}:n{source_tint}",
                    textures.index(texture), 0, [mesh]))
                placements.append((asset_index, 1.0))
            groups.append(group_payload(f"{action}_{phase_index}", placements))
    target = pack / "unit_worker_runtime.bin"
    output = bytearray(MAGIC)
    output.extend(struct.pack("<IIII", 1, len(textures), len(assets), len(groups)))
    for texture in textures:
        output.extend(bundle_string(texture))
    for asset in assets:
        output.extend(asset)
    for group in groups:
        output.extend(group)
    target.write_bytes(output)
    return target, bounds, len(textures), len(groups)


def build(pack: Path):
    manifest = load_json(pack / "manifest.json")
    results = []
    for slug in UNITS:
        results.append((slug, *write_unit_bundle(pack, manifest, slug)))
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=Path("Renderer/packs/UnitFamilyLab"))
    parser.add_argument("--worker-pack", type=Path,
                        default=Path("Renderer/packs/WorkerBuilderLab"))
    args = parser.parse_args()
    for slug, path, bounds, textures, groups in build(args.pack):
        print(f"{slug}: {path} textures={textures} groups={groups} bounds={bounds}")
    manifest = load_json(args.pack / "manifest.json")
    path, bounds, textures, groups = write_worker_bundle(
        args.pack, manifest, args.worker_pack)
    print(f"worker: {path} textures={textures} groups={groups} bounds={bounds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
