#!/usr/bin/env python3
"""Bind validated resource poses into generic DLL skin-palette payloads offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import normalized_animation, normalized_pose_cache, normalized_skin
from Renderer.tools.asset_compiler.school_orientation import align_school_payload
from Renderer.tools.asset_compiler.unit_model_extractor import SOURCE_UNITS_PER_TILE

ROOT = Path(__file__).resolve().parents[3]
CLIP_UNITS = ROOT / "Renderer/lab/shared/resources/clip_units.json"


def pack_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("animation input escapes its pack")
    return path


def encode(mesh: dict, skeleton: dict, cache: normalized_pose_cache.PoseCache) -> bytes:
    normalized_skin.validate_rest_pose(mesh, skeleton)
    normalized_pose_cache.validate_skeleton_binding(cache, skeleton)
    vertices, indices = mesh["vertices"], mesh["topology"]["indices"]
    bones = skeleton["bones"]
    if not (0 < len(vertices) <= 65536 and 0 < len(indices) <= 393216 and
            0 < len(bones) <= 256 and 2 <= cache.frame_count <= 4096 and 0 < cache.duration <= 3600):
        raise ValueError("animation exceeds DLL dimensions")
    size = 32 + 64 * len(vertices) + 4 * len(indices) + 64 * len(bones) * cache.frame_count
    if size > 64 * 1024 * 1024:
        raise ValueError("animation exceeds DLL byte budget")
    output = bytearray(struct.pack("<8s5If", b"C3XANM1\0", 1, len(vertices), len(indices),
                                   len(bones), cache.frame_count, cache.duration))
    for vertex in vertices:
        output.extend(struct.pack("<8f4I4f", *(vertex["position"] + vertex["normal"] +
            vertex["uv0"] + vertex["joints"] + vertex["weights"])))
    output.extend(struct.pack(f"<{len(indices)}I", *indices))
    for frame in range(cache.frame_count):
        for index, bone in enumerate(bones):
            offset = (frame * len(bones) + index) * 16
            palette = normalized_skin._multiply(bone["inverse_bind_matrix"], cache.matrices[offset:offset+16])
            if not all(math.isfinite(value) for value in palette):
                raise ValueError("non-finite animation palette")
            output.extend(struct.pack("<16f", *palette))
    if len(output) != size:
        raise AssertionError("animation serialization length mismatch")
    return bytes(output)


def calibrated_resource_poses(skeleton: dict, clip, group: int, translation_ratio: float, target: Path):
    """Normalize authored local translations, then remove only constant root placement.

    Clutter clips were exported at 1/12 while component skeletons use 1/100.
    Root placement belongs to the source scene, not the Civ III resource anchor.
    Keep animated root deltas and every authored rotation/scale channel intact.
    """
    tracks = {track.name: track for track in clip.groups[group].tracks}
    frames = []
    root_positions = {}
    for frame in range(clip.frame_count):
        sampled = normalized_skin.sample_pose(skeleton, clip, group,
            clip.duration*frame/(clip.frame_count-1), False)
        local = []
        for index, (bone, transform) in enumerate(zip(skeleton["bones"], sampled)):
            position = tuple(transform.position)
            track = tracks.get(bone["name"])
            if any(not math.isfinite(v) or abs(v)>1024 for v in position):
                position = tuple(bone["local"]["position"])
            elif track and track.position.mode != normalized_animation.IDENTITY:
                position = tuple(v*translation_ratio for v in position)
            scale = transform.scale_shear
            if any(not math.isfinite(v) or abs(v)>1024 for v in scale):
                scale = tuple(bone["local"]["scale_shear"])
            if bone["parent"] == -1:
                if frame == 0:
                    root_positions[index] = position
                position = tuple(position[a]-root_positions[index][a]+bone["local"]["position"][a] for a in range(3))
            local.append(normalized_animation.Transform(position, transform.orientation, scale))
        frames.append(normalized_skin.world_matrices(skeleton, local))
    return normalized_pose_cache.write_pose_cache(target, clip.duration, clip.sample_rate,
        [bone["name"] for bone in skeleton["bones"]], frames)


def clip_translation_scales(landmarks, calibration=CLIP_UNITS, *, read_bytes=lambda p: p.read_bytes()):
    recipe = json.loads(read_bytes(calibration))
    if recipe.get("schema") != "c3x.resource_clip_units.v1" or not recipe.get("clips"):
        raise ValueError("Missing current resource clip-unit recipe")
    scales = {}
    for entry in recipe["clips"]:
        scale = entry["translation_scale"]
        if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("Invalid resource clip translation scale")
        key = hashlib.sha256(read_bytes(pack_path(landmarks, entry["path"]))).hexdigest()
        if key in scales and scales[key] != scale:
            raise ValueError("Conflicting units for the same resource clip")
        scales[key] = scale
    return scales


def build(animated: Path, landmarks: Path, output: Path, *, consumed=None) -> dict:
    animated, landmarks, output = (Path(p).resolve() for p in (animated, landmarks, output))
    for source in (animated, landmarks):
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Resource output must not overlap normalized source packs")
    def read_bytes(path):
        path = Path(path).resolve()
        data = path.read_bytes()
        if consumed is not None:
            key = path.relative_to(ROOT).as_posix()
            value = hashlib.sha256(data).hexdigest()
            if key in consumed and consumed[key] != value:
                raise ValueError("Resource input changed during build: " + key)
            consumed[key] = value
        return data
    def read(path):return json.loads(read_bytes(path))
    def tracked(root, relative):
        path = pack_path(root, relative)
        read_bytes(path)
        return path
    result = {"schema": "c3x.resource_animation_runtime.v1", "resources": {},
              "presentation": {"facing": "SE", "phase": "absolute_time_plus_stable_instance_seed"}}
    (output / "clips").mkdir(parents=True, exist_ok=True)
    (output / "textures").mkdir(exist_ok=True)
    total = 0
    clip_scales = clip_translation_scales(landmarks, read_bytes=read_bytes)
    (output / "poses").mkdir(exist_ok=True)

    def compile_subject(root: Path, subject: str, meshes: list[str], materials: list[str],
                        skeleton_path: str, animation: dict, bindings: list[dict]) -> dict:
        nonlocal total
        status = animation.get("binding_status", animation.get("pose_status"))
        if status != "validated_model_aware_pose_cache":
            raise ValueError(f"{subject}: unvalidated model-aware animation")
        skeleton = normalized_skin.load_skeleton(tracked(root, skeleton_path))
        pose_path = tracked(root, animation["pose_cache"])
        if animation.get("pose_cache_sha256") and hashlib.sha256(read_bytes(pose_path)).hexdigest() != animation["pose_cache_sha256"]:
            raise ValueError(f"{subject}: stale pose-cache hash")
        cache = normalized_pose_cache.load_pose_cache(pose_path)
        calibration = None
        if root == animated:
            clip_path = tracked(root, animation["clip"])
            clip_hash = hashlib.sha256(read_bytes(clip_path)).hexdigest()
            ratio = (1.0/SOURCE_UNITS_PER_TILE)/clip_scales[clip_hash]
            clip = normalized_animation.load_clip(clip_path)
            pose_relative = f"poses/{subject}.c3pose"
            cache = calibrated_resource_poses(skeleton, clip, animation["group_index"], ratio, output/pose_relative)
            calibration = {"translation_ratio": ratio, "root_policy": "remove_constant_source_scene_placement",
                           "pose_cache": pose_relative}
        parts = []
        for ordinal, binding in enumerate(bindings):
            if binding.get("binding_mode", "vertex_skin") != "vertex_skin":
                raise ValueError(f"{subject}: unsupported resource binding")
            mesh = normalized_skin.load_mesh(tracked(root, meshes[binding["mesh"]]), len(skeleton["bones"]))
            material = read(pack_path(root, materials[binding["material"]]))
            channel = material.get("base_color", material.get("channels", {}).get("base_color"))
            texture = pack_path(root, channel["texture"])
            texture_name = "textures/" + hashlib.sha256(read_bytes(texture)).hexdigest() + ".dds"
            if not (output / texture_name).exists():
                shutil.copyfile(texture, output / texture_name)
            payload = encode(mesh, skeleton, cache)
            facing = None
            if root == landmarks and subject in ("fish", "whales"):
                pairs = ([(f"Fish{i:02}_Head_Bone", f"Fish{i:02}_Tail_Bone") for i in range(1, 13)]
                         if subject == "fish" else
                         [(f"Whale_{name}_Bone_01", f"Whale_{name}_Bone_02") for name in "ABC"])
                payload, facing = align_school_payload(payload, mesh, skeleton, pairs)
            filename = f"clips/{subject}_{ordinal}.bin"
            (output / filename).write_bytes(payload)
            total += len(payload)
            parts.append({"mesh": filename, "texture": texture_name,
                          "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                          "vertices": len(mesh["vertices"]), "bones": struct.unpack_from("<I", payload, 20)[0],
                          "facing": facing})
        return {"parts": parts, "duration": cache.duration, "frames": cache.frame_count, "calibration": calibration}

    manifest = read(animated / "manifest.json")
    for resource, record in manifest["resources"].items():
        subjects = []
        for ordinal, candidate in enumerate(record["subject_candidates"]):
            component = read(pack_path(animated, manifest["assets"][candidate["asset"]]["component"]))
            subjects.append(compile_subject(animated, resource.split("/")[-1]+f"_{ordinal}",
                component.get("meshes", [component["mesh"]]),
                component.get("materials", [component["material"]]), component["skeleton"],
                manifest["animations"][candidate["animation"]], component["draw_bindings"]))
        result["resources"][resource] = subjects
    manifest = read(landmarks / "manifest.json")
    for resource in ("resource/fish", "resource/whales"):
        asset = manifest["assets"][manifest["resources"][resource]["landmark_asset"]]
        animation = dict(manifest["animations"][resource])
        animation["binding_status"] = animation["pose_status"]
        result["resources"][resource] = [compile_subject(landmarks, resource.split("/")[-1],
            [asset["mesh"]], [asset["material"]], asset["skeleton"], animation,
            [{"mesh": 0, "material": 0}])]
    # One current primary subject per resource.
    # Source pose coordinates stay unchanged; presentation calibration is generic data.
    from Renderer.tools.asset_compiler.build_resource_runtime import SELECTIONS
    static_selections = {name: (asset, scale, count) for name, asset, scale, count in SELECTIONS}
    original = read(landmarks / "manifest.json")
    bindings = {}
    forward_y = {"horses": -1, "cattle": 1, "game": -1, "furs": -1, "ivory": -1,
                 "whales": 0, "fish": 0, "wheat": -1, "bananas": -1, "rubber": -1}
    for resource, subjects in result["resources"].items():
        name = resource.split("/")[-1]
        selected = 2 if name == "cattle" else 0
        subject = subjects[selected]
        if len(subject["parts"]) != 1:
            raise ValueError("primary resource requires one material part")
        part = subject["parts"][0]
        data = (output / part["mesh"]).read_bytes()
        count = struct.unpack_from("<I", data, 12)[0]
        vertices = [struct.unpack_from("<3f", data, 32+i*64) for i in range(count)]
        low = [min(v[a] for v in vertices) for a in range(3)]
        high = [max(v[a] for v in vertices) for a in range(3)]
        span = [high[a]-low[a] for a in range(3)]
        offset = [-(low[0]+high[0])*.5, -(low[1]+high[1])*.5, -low[2]]
        instances = 1
        if name in static_selections:
            asset_id, old_scale, instances = static_selections[name]
            old = read(pack_path(landmarks, original["assets"][asset_id]["mesh"]))
            old_height = max(v["position"][2] for v in old["vertices"])-min(v["position"][2] for v in old["vertices"])
            # Match the restrained gameplay footprint used by static resource
            # bodies while preserving uniform XYZ scale and source proportions.
            scale = old_scale*.72*old_height/span[2]
        else:
            scale = min(.65/max(span[0],span[1]), .65/span[2])*(.72/.78)
        if name == "fish":
            offset = [0, 0, .060]  # Preserve the approved school surface offset.
        bindings[name] = {"mesh": part["mesh"], "texture": part["texture"],
            "scale": scale, "count": instances, "offset_x": offset[0], "offset_y": offset[1],
            "offset_z": offset[2], "yaw": -math.atan2(forward_y[name], 0 if forward_y[name] else 1),
            "orientation_evidence": ("each_school_body_head_minus_tail_aligned_to_SE" if part.get("facing") else
                                     "source_rest_bones; visual checkpoint pending")}
    (output / "bindings.json").write_text(json.dumps({"schema": "c3x.resource_animation_bindings.v1",
        "bindings": bindings}, indent=2, sort_keys=True)+"\n")
    result["payload_bytes"] = total
    (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animated", type=Path, default=Path("Renderer/packs/ResourceAnimatedLab"))
    parser.add_argument("--landmarks", type=Path, default=Path("Renderer/packs/ResourceNormalized"))
    parser.add_argument("--output", type=Path, default=Path("Renderer/packs/ResourceAnimationRuntime"))
    args = parser.parse_args()
    result = build(args.animated, args.landmarks, args.output)
    print(json.dumps({"resources": len(result["resources"]), "bytes": result["payload_bytes"]}))


if __name__ == "__main__":
    main()
