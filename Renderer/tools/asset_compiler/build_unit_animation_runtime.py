#!/usr/bin/env python3
"""Compile complete unit kits/clips into the generic DLL animation payload."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import normalized_animation, normalized_pose_cache, normalized_skin
from Renderer.tools.asset_compiler.build_resource_animation_runtime import encode, pack_path
from Renderer.tools.asset_compiler.build_l20_unit_runtime import OWNER_COLOR_OVERRIDES
from Renderer.tools.asset_compiler.unit_family_action_validator import SOCKET_PROFILE, _best_group


IDENTITY = (1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.)
# Retained L20 source ArtDef colors, compiled to ordinary linear material data.
SOURCE_TINTS = {"BaseMale_SkinColor_Caucasian": (.878, .765, .647),
    "GreatPeople_Military": (.631, .067, .059), "Horse_Default": (.529, .286, .059),
    "Horse_Secondary": (.404, .345, .239), "Infantry_European": (.651, .514, .239),
    "Vehicle_Woodland": (.431, .596, .290), "Wood": (.784, .580, .302),
    None: (1., 1., 1.), "USE_CIV_COLOR": (1., 1., 1.)}


def document(root: Path, relative: str) -> dict:
    return json.loads(pack_path(root, relative).read_text(encoding="utf-8"))


def checked_cache(pack: Path, record: dict, skeleton: dict):
    path = pack_path(pack, record["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("stale unit pose cache")
    cache = normalized_pose_cache.load_pose_cache(path)
    normalized_pose_cache.validate_skeleton_binding(cache, skeleton)
    return cache


def socket_payload(mesh: dict, driver, bone_name: str, model_scale: float):
    """An attachment is a one-bone palette, evaluated by the same DLL skinner.

    Preserve its model-space origin. Recentring a weapon on each sampled pose
    disconnects it from the hand and introduces false motion at action changes.
    """
    index = driver.bone_names.index(bone_name)
    matrices = tuple(value for frame in range(driver.frame_count)
                     for value in driver.matrices[(frame*len(driver.bone_names)+index)*16:
                                                  (frame*len(driver.bone_names)+index+1)*16])
    cache = normalized_pose_cache.PoseCache(driver.duration, driver.sample_rate,
        driver.frame_count, (bone_name,), matrices)
    skeleton = {"bones": [{"name": bone_name, "parent": -1,
        "local": {"position": [0., 0., 0.], "orientation": [0., 0., 0., 1.],
                  "scale_shear": [1., 0., 0., 0., 1., 0., 0., 0., 1.]},
        "inverse_bind_matrix": IDENTITY}]}
    bound = {"topology": mesh["topology"], "vertices": [
        {**vertex, "position": [v*model_scale for v in vertex["position"]],
         "joints": [0, 0, 0, 0], "weights": [1., 0., 0., 0.]}
        for vertex in mesh["vertices"]]}
    return encode(bound, skeleton, cache)


def native_anchor_caches(caches: dict, driver_id: str, skeleton: dict) -> dict:
    """Strip planar root travel once for the whole kit, retaining joint/Z motion."""
    roots = [i for i, bone in enumerate(skeleton["bones"]) if bone["parent"] == -1]
    if len(roots) != 1:
        raise ValueError("unit animation driver requires one unambiguous root")
    root = roots[0]
    driver = caches[driver_id]
    rest = normalized_skin.world_matrices(skeleton)[root]
    result = {}
    for asset, cache in caches.items():
        matrices = list(cache.matrices)
        for frame in range(cache.frame_count):
            origin = (frame*len(driver.bone_names)+root)*16
            for bone in range(len(cache.bone_names)):
                for axis in (0, 1):
                    matrices[(frame*len(cache.bone_names)+bone)*16+12+axis] -= driver.matrices[origin+12+axis]-rest[12+axis]
        result[asset] = normalized_pose_cache.PoseCache(cache.duration, cache.sample_rate,
            cache.frame_count, cache.bone_names, tuple(matrices))
    return result


def fortify_transition(skeleton: dict, idle, target):
    """Bake a transition for pose-only clips; native cursor still owns duration."""
    names = {b["name"] for b in skeleton["bones"]}
    start = normalized_skin.sample_pose(skeleton, idle, _best_group(idle, names)[0], 0., False)
    end = normalized_skin.sample_pose(skeleton, target, _best_group(target, names)[0], target.duration, False)
    frames = []
    for frame in range(16):
        t = frame/15
        t = t*t*(3-2*t)
        pose = []
        for a, b in zip(start, end):
            sign = -1 if sum(x*y for x,y in zip(a.orientation,b.orientation)) < 0 else 1
            q = [(1-t)*x+t*sign*y for x,y in zip(a.orientation,b.orientation)]
            length = math.sqrt(sum(x*x for x in q))
            pose.append(normalized_animation.Transform(
                tuple((1-t)*x+t*y for x,y in zip(a.position,b.position)),
                tuple(x/length for x in q),
                tuple((1-t)*x+t*y for x,y in zip(a.scale_shear,b.scale_shear))))
        frames.extend(v for matrix in normalized_skin.world_matrices(skeleton,pose) for v in matrix)
    return normalized_pose_cache.PoseCache(.5,30.,16,tuple(b["name"] for b in skeleton["bones"]),tuple(frames))


def build(packs: list[Path], output: Path) -> dict:
    previous = output/"manifest.json"
    old_payloads = set()
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        if old.get("schema") != "c3x.unit_animation_runtime.v1":
            raise ValueError("output belongs to a different pack compiler")
        old_payloads = {part["mesh"] for unit in old["units"].values()
                        for action in unit["actions"].values() for part in action["parts"]}
    result = {"schema": "c3x.unit_animation_runtime.v1", "units": {},
              "runtime_enabled": False,
              "clock": "native_action_cursor; no independent gameplay clock",
              "placement": "native_body_anchor; whole-kit planar root travel removed; joint and vertical motion retained"}
    for name in ("clips", "textures"):
        (output/name).mkdir(parents=True, exist_ok=True)
    payloads = {}

    def publish(payload: bytes, directory: str, extension: str) -> str:
        digest = hashlib.sha256(payload).hexdigest()
        relative = f"{directory}/{digest}.{extension}"
        target = output/relative
        if not target.exists() or target.read_bytes() != payload:
            target.write_bytes(payload)
        if directory == "clips":
            payloads[digest] = len(payload)
        return relative

    for pack in packs:
        manifest = document(pack, "manifest.json")
        sockets = manifest.get("unit_binding", {}).get("sockets", SOCKET_PROFILE)
        for unit_id, entry in manifest["units"].items():
            if unit_id in result["units"]:
                raise ValueError(f"duplicate unit binding: {unit_id}")
            recipe = document(pack, entry["recipe"])
            # The original Warrior proof predates the explicit driver field.
            driver_id = recipe.get("animation_driver")
            if driver_id is None and unit_id == "unit/warrior":
                driver_id = "unit/warrior/body"
            if driver_id is None:
                raise ValueError(f"unit has no authored animation driver: {unit_id}")
            components = {record["asset"]: document(pack, manifest["assets"][record["asset"]]["component"])
                          for record in recipe["components"]}
            skeletons = {asset: normalized_skin.load_skeleton(pack_path(pack, item["skeleton"]))
                         for asset, item in components.items() if item["binding_mode"] == "vertex_skin"}
            unit = {"actions": {}, "civ3_ids": recipe.get("civ3_ids", []),
                    "member_scale": recipe["member"]["member_scale"]*recipe["member"]["variation_scale"],
                    "source_pack": pack.name, "source_recipe": entry["recipe"]}
            for action, animation_id in recipe["actions"].items():
                animation = manifest["animations"][animation_id]
                clip = normalized_animation.load_clip(pack_path(pack, animation["clip"]))
                caches = {asset: checked_cache(pack, animation["pose_caches"][asset], skeleton)
                          for asset, skeleton in skeletons.items()}
                if any(c.frame_count != clip.frame_count or abs(c.duration-clip.duration)>1e-5
                       for c in caches.values()):
                    raise ValueError(f"unit parts disagree on action timing: {unit_id}/{action}")
                transition = action == "fortify" and clip.frame_count <= 2
                if transition:
                    idle_record = manifest["animations"][recipe["actions"]["idle"]]
                    idle_clip = normalized_animation.load_clip(pack_path(pack, idle_record["clip"]))
                    caches = {asset: fortify_transition(skeleton, idle_clip, clip) for asset,skeleton in skeletons.items()}
                caches = native_anchor_caches(caches, driver_id, skeletons[driver_id])
                parts = []
                selected = set(recipe.get("action_components", {}).get(action, components))
                if not selected or selected - components.keys() or driver_id not in selected:
                    raise ValueError(f"invalid action component selection: {unit_id}/{action}")
                for asset, component in components.items():
                    if asset not in selected:
                        continue
                    bindings = component.get("draw_bindings", [{"mesh": 0, "material": 0}])
                    for binding in bindings:
                        meshes = component["meshes"] if "meshes" in component else [component["mesh"]]
                        materials = component["materials"] if "materials" in component else [component["material"]]
                        mesh_relative = meshes[binding["mesh"]]
                        material_relative = materials[binding["material"]]
                        if component["binding_mode"] == "vertex_skin":
                            mesh = normalized_skin.load_mesh(pack_path(pack, mesh_relative), len(skeletons[asset]["bones"]))
                            payload = encode(mesh, skeletons[asset], caches[asset])
                        elif component["binding_mode"] == "rigid_attachment":
                            mesh = document(pack, mesh_relative)
                            payload = socket_payload(mesh, caches[driver_id],
                                sockets[component["attachment_point"]]["bone"], component["model_scale"])
                        else:
                            raise ValueError(f"unsupported complete-kit binding: {asset}")
                        material = document(pack, material_relative)
                        channels = {}
                        for channel, data in material["channels"].items():
                            channels[channel] = {**data, "texture": publish(
                                pack_path(pack, data["texture"]).read_bytes(), "textures", "dds")}
                        parts.append({"asset": asset, "mesh": publish(payload, "clips", "bin"),
                            "bytes": len(payload), "material": {"alpha_mode": material.get("alpha_mode", "opaque"),
                                "channels": channels, "source_tint": component.get("tint"),
                                "owner_color": OWNER_COLOR_OVERRIDES.get(asset, component.get("owner_color"))},
                            "source_mesh": mesh_relative, "source_material": material_relative})
                unit["actions"][action] = {"duration": caches[driver_id].duration, "frames": caches[driver_id].frame_count,
                    "presentation": "idle_to_static_fortify" if transition else "source_clip",
                    "loop": animation["loop"], "parts": parts}
            result["units"][unit_id] = unit
    result["unique_payloads"] = len(payloads)
    result["payload_bytes"] = sum(payloads.values())
    # A small direct-member table avoids runtime interpretation of source
    # recipes/material names. Unknown owner contracts reject the complete kit.
    bindings = {"unit_count": len(result["units"])}
    for index, (unit_id, unit) in enumerate(result["units"].items()):
        keys = unit["civ3_ids"] or (["PRTO_Warrior"] if unit_id == "unit/warrior" else [])
        # Fit one authored idle stance once. Never refit moving/death poses.
        stance = []
        for part in unit["actions"]["idle"]["parts"]:
            payload = (output/part["mesh"]).read_bytes()
            _version, vertices, indices, _bones, _frames = struct.unpack_from("<5I", payload, 8)
            palette_offset = 32+vertices*64+indices*4
            for vertex in range(vertices):
                values = struct.unpack_from("<8f4I4f", payload, 32+vertex*64)
                position = [0., 0., 0.]
                for joint, weight in zip(values[8:12], values[12:16]):
                    if not weight:
                        continue
                    matrix = struct.unpack_from("<16f", payload, palette_offset+joint*64)
                    for axis in range(3):
                        position[axis] += weight*(sum(values[a]*matrix[a*4+axis] for a in range(3))+matrix[12+axis])
                stance.append(position)
        low_z = min(v[2] for v in stance)
        span = 0.
        for direction in range(8):
            cosine, sine = math.cos(direction*math.pi/4), math.sin(direction*math.pi/4)
            points = [((v[0]*cosine-v[1]*sine-v[0]*sine-v[1]*cosine)*64,
                       (v[0]*cosine-v[1]*sine+v[0]*sine+v[1]*cosine)*32-(v[2]-low_z)*(150*128/224)) for v in stance]
            span = max(span, *(max(v[a] for v in points)-min(v[a] for v in points) for a in range(2)))
        if span<=0:
            raise ValueError("empty projected unit stance")
        record = {"key_count": len(keys), "scale": 56/span, "offset_z": -low_z, "yaw_offset": 225.0,
                  "fit_policy": "fixed_idle_stance_56_pixels_at_normal_zoom",
                  **{f"key{i}": key for i, key in enumerate(keys)}}
        complete = bool(keys)
        for action, data in unit["actions"].items():
            target = {"part_count": len(data["parts"]), "loop": int(data["loop"])}
            for i, part in enumerate(data["parts"]):
                material = part["material"]
                tint = SOURCE_TINTS[material["source_tint"]]
                owner = material["owner_color"]
                if owner is None and material["source_tint"] == "USE_CIV_COLOR":
                    complete = False
                mask = 0 if not owner or owner["mode"] == "none" else (1 if owner["mask_source"] == "base_color_alpha_inverse" else 2)
                target[f"part{i}"] = {"mesh": part["mesh"],
                    "texture": material["channels"]["base_color"]["texture"],
                    "tint_r": tint[0], "tint_g": tint[1], "tint_b": tint[2],
                    "owner_mask": mask, "owner_strength": owner["strength"] if owner else 0,
                    "cutout": int(material["alpha_mode"] == "mask")}
            record[action] = target
        record["complete"] = int(complete)
        bindings[f"unit{index}"] = record
    (output/"bindings.json").write_text(json.dumps(bindings, indent=2, sort_keys=True)+"\n")
    (output/"manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    # Only retire blobs owned by the previous successful manifest. Interrupted
    # builds or unrelated files are never treated as permission to delete data.
    live = {f"clips/{digest}.bin" for digest in payloads}
    for relative in old_payloads-live:
        path = pack_path(output, relative)
        if path.exists() and path.stem == hashlib.sha256(path.read_bytes()).hexdigest():
            path.unlink()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=Path("Renderer/packs/UnitAnimationRuntime"))
    args = parser.parse_args()
    result = build(args.pack or [Path("Renderer/packs/UnitFamilyLab"), Path("Renderer/packs/UnitEarlyLab")], args.output)
    print(json.dumps({"units": len(result["units"]), "actions": sum(len(u["actions"]) for u in result["units"].values()),
                      "unique_payloads": result["unique_payloads"], "bytes": result["payload_bytes"]}))


if __name__ == "__main__":
    main()
