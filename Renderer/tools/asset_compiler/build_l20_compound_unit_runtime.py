#!/usr/bin/env python3
"""Bake deterministic posed compound-unit samples for the L20 Terrain Lab."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Renderer.preview.render_unit_turntable import (
    _multiply, _rigid_mesh, _skinned_mesh, _transform_point, _transform_vector,
)
from Renderer.tools.asset_compiler import normalized_animation, normalized_pose_cache, normalized_skin
from Renderer.tools.asset_compiler.build_l20_unit_runtime import ACTIONS
from Renderer.tools.asset_compiler.build_mine_runtime import (
    MAGIC, bundle_string, group_payload, merged_asset,
)
from Renderer.tools.asset_compiler.unit_family_action_validator import SOCKET_PROFILE, _best_group


UNITS = ("horseman", "catapult", "tank", "great_general_classical")
IDENTITY = (1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def scale_matrix(value: float):
    return (value, 0.0, 0.0, 0.0,
            0.0, value, 0.0, 0.0,
            0.0, 0.0, value, 0.0,
            0.0, 0.0, 0.0, 1.0)


def transformed(mesh: dict, matrix):
    return {
        "topology": mesh["topology"],
        "vertices": [
            {
                "position": list(_transform_point(vertex["position"], matrix)),
                "normal": list(_transform_vector(vertex["normal"], matrix)),
                "uv0": vertex["uv0"],
            }
            for vertex in mesh["vertices"]
        ],
    }


def posed_parts(pack: Path, manifest: dict, recipe: dict, action: str, phase: float):
    binding = recipe["actions"].get(action, recipe["actions"]["idle"])
    node_worlds = {}
    node_skeletons = {}
    for node_id, node in recipe["nodes"].items():
        skeleton = normalized_skin.load_skeleton(pack / node["skeleton"])
        cache = normalized_pose_cache.load_pose_cache(
            pack / binding["node_pose_caches"][node_id])
        normalized_pose_cache.validate_skeleton_binding(cache, skeleton)
        node_worlds[node_id] = cache.sample(cache.duration * phase, binding["loop"])
        node_skeletons[node_id] = skeleton

    composition = {
        recipe["root_node"]: scale_matrix(
            recipe["nodes"][recipe["root_node"]]["variation_scale"])
    }
    pending = list(recipe["joints"])
    while pending:
        progressed = False
        for joint in list(pending):
            if joint["parent"] not in composition:
                continue
            parent_skeleton = node_skeletons[joint["parent"]]
            parent_names = [bone["name"] for bone in parent_skeleton["bones"]]
            parent_bone = joint["parent_bone"]
            # The extracted Tank GunnerAttach helper is at vehicle origin; the
            # source declaration also names hatchPivot, whose authored transform
            # is the usable exposed-crew socket.
            if recipe["unit_id"] == "unit/tank" and "hatchPivot" in parent_names:
                parent_bone = "hatchPivot"
            parent_index = parent_names.index(parent_bone)
            socket = node_worlds[joint["parent"]][parent_index]
            child_scale_value = recipe["nodes"][joint["child"]]["variation_scale"]
            if recipe["unit_id"] == "unit/tank":
                child_scale_value *= 0.45
            child_scale = scale_matrix(child_scale_value)
            composition[joint["child"]] = _multiply(
                _multiply(_multiply(child_scale, joint.get("local_transform", IDENTITY)), socket),
                composition[joint["parent"]],
            )
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError(f"{recipe['unit_id']} has an unresolved joint graph")

    parts = []
    for node_id, node in recipe["nodes"].items():
        driver_skeleton = node_skeletons[node_id]
        driver_names = [bone["name"] for bone in driver_skeleton["bones"]]
        for record in node["components"]:
            document = load_json(pack / manifest["assets"][record["asset"]]["component"])
            skeleton = normalized_skin.load_skeleton(pack / document["skeleton"])
            owner_contract = document.get("owner_color", {})
            owner = (0 if owner_contract.get("mode") == "none" else
                     (1 if owner_contract.get("mask_source") == "base_color_alpha_inverse" else 2))
            for draw in document["draw_bindings"]:
                mesh_paths = document.get("meshes", [document.get("mesh")])
                if document["binding_mode"] == "vertex_skin":
                    animation = manifest["animations"][binding["node_clips"][node_id]]
                    clip = normalized_animation.load_clip(pack / animation["clip"])
                    group_index, _common = _best_group(
                        clip, {bone["name"] for bone in skeleton["bones"]})
                    pose = normalized_skin.sample_pose(
                        skeleton, clip, group_index, clip.duration * phase,
                        binding["loop"])
                    component_worlds = normalized_skin.world_matrices(skeleton, pose)
                    mesh = normalized_skin.load_mesh(
                        pack / mesh_paths[draw["mesh"]], len(skeleton["bones"]))
                    local_mesh = _skinned_mesh(
                        mesh, skeleton, component_worlds)
                else:
                    mesh = load_json(pack / mesh_paths[draw["mesh"]])
                    attachment = document["attachment_point"]
                    profile = SOCKET_PROFILE.get(attachment, {"bone": "Root"})
                    requested_bone = profile["bone"]
                    if attachment == "ArmBand" and requested_bone not in driver_names:
                        requested_bone = "LForearm"
                    socket_names = driver_names
                    socket_worlds = node_worlds[node_id]
                    if requested_bone not in socket_names:
                        socket_names = [bone["name"] for bone in skeleton["bones"]]
                        socket_worlds = normalized_skin.world_matrices(skeleton)
                    if requested_bone not in socket_names:
                        requested_bone = socket_names[0]
                    local_mesh = _rigid_mesh(
                        mesh,
                        socket_worlds[socket_names.index(requested_bone)],
                        document["model_scale"],
                    )
                material_paths = document.get("materials", [document.get("material")])
                material = load_json(pack / material_paths[draw["material"]])
                texture = material["channels"]["base_color"]["texture"]
                parts.append((transformed(local_mesh, composition[node_id]), texture, owner))
    return parts


def write_bundle(pack: Path, manifest: dict, slug: str):
    recipe = load_json(pack / manifest["units"][f"unit/{slug}"]["recipe"])
    cached = {}
    textures = []
    for action, phases in ACTIONS.items():
        for phase in phases:
            parts = posed_parts(pack, manifest, recipe, action, phase)
            cached[(action, phase)] = parts
            for _mesh, texture, _owner in parts:
                if texture not in textures:
                    textures.append(texture)
    if len(textures) > 8:
        raise ValueError(f"{slug} needs {len(textures)} textures; Lab ABI permits 8")

    assets, groups = [], []
    bounds = [1.0e30, -1.0e30, 1.0e30, -1.0e30, 1.0e30, -1.0e30]
    for action, phases in ACTIONS.items():
        for phase_index, phase in enumerate(phases):
            parts = cached[(action, phase)]
            positions = [vertex["position"] for mesh, _texture, _owner in parts
                         for vertex in mesh["vertices"]]
            minimum = [min(position[axis] for position in positions) for axis in range(3)]
            maximum = [max(position[axis] for position in positions) for axis in range(3)]
            center_x = (minimum[0] + maximum[0]) * 0.5
            center_y = (minimum[1] + maximum[1]) * 0.5
            placements = []
            for part_index, (mesh, texture, owner) in enumerate(parts):
                for vertex in mesh["vertices"]:
                    position = vertex["position"]
                    vertex["position"] = [
                        position[0] - center_x,
                        position[1] - center_y,
                        position[2] - minimum[2],
                    ]
                    for axis in range(3):
                        bounds[axis * 2] = min(bounds[axis * 2], vertex["position"][axis])
                        bounds[axis * 2 + 1] = max(bounds[axis * 2 + 1], vertex["position"][axis])
                asset_index = len(assets)
                assets.append(merged_asset(
                    f"{slug}:{action}:{phase_index}:part_{part_index}:t{owner}",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=Path("Renderer/packs/CompoundUnitLab"))
    args = parser.parse_args()
    manifest = load_json(args.pack / "manifest.json")
    for slug in UNITS:
        target, bounds, textures, groups = write_bundle(args.pack, manifest, slug)
        print(f"{slug}: {target} textures={textures} groups={groups} bounds={bounds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
