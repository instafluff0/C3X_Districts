#!/usr/bin/env python3
"""Compile complete unit kits/clips into the generic DLL animation payload."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import normalized_animation, normalized_pose_cache, normalized_skin
from Renderer.tools.asset_compiler.build_resource_animation_runtime import encode, pack_path
from Renderer.tools.asset_compiler.build_l20_unit_runtime import OWNER_COLOR_OVERRIDES
from Renderer.tools.asset_compiler.unit_family_action_validator import SOCKET_PROFILE


IDENTITY = (1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.)


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
                caches = native_anchor_caches(caches, driver_id, skeletons[driver_id])
                parts = []
                for asset, component in components.items():
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
                unit["actions"][action] = {"duration": clip.duration, "frames": clip.frame_count,
                    "loop": animation["loop"], "parts": parts}
            result["units"][unit_id] = unit
    result["unique_payloads"] = len(payloads)
    result["payload_bytes"] = sum(payloads.values())
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
    result = build(args.pack or [Path("Renderer/packs/UnitWarriorLab"), Path("Renderer/packs/UnitFamilyLab")], args.output)
    print(json.dumps({"units": len(result["units"]), "actions": sum(len(u["actions"]) for u in result["units"].values()),
                      "unique_payloads": result["unique_payloads"], "bytes": result["payload_bytes"]}))


if __name__ == "__main__":
    main()
