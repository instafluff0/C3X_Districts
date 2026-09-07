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
from Renderer.tools.asset_compiler import normalized_pose_cache, normalized_skin


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


def build(animated: Path, landmarks: Path, output: Path) -> dict:
    result = {"schema": "c3x.resource_animation_runtime.v1", "resources": {},
              "presentation": {"facing": "SE", "phase": "absolute_time_plus_stable_instance_seed"},
              "runtime_enabled": False, "calibration": "pending_dynamic_layer_visual_verification"}
    (output / "clips").mkdir(parents=True, exist_ok=True)
    (output / "textures").mkdir(exist_ok=True)
    total = 0

    def compile_subject(root: Path, subject: str, meshes: list[str], materials: list[str],
                        skeleton_path: str, animation: dict, bindings: list[dict]) -> dict:
        nonlocal total
        status = animation.get("binding_status", animation.get("pose_status"))
        if status != "validated_model_aware_pose_cache":
            raise ValueError(f"{subject}: unvalidated model-aware animation")
        skeleton = normalized_skin.load_skeleton(pack_path(root, skeleton_path))
        pose_path = pack_path(root, animation["pose_cache"])
        if animation.get("pose_cache_sha256") and hashlib.sha256(pose_path.read_bytes()).hexdigest() != animation["pose_cache_sha256"]:
            raise ValueError(f"{subject}: stale pose-cache hash")
        cache = normalized_pose_cache.load_pose_cache(pose_path)
        parts = []
        for ordinal, binding in enumerate(bindings):
            if binding.get("binding_mode", "vertex_skin") != "vertex_skin":
                raise ValueError(f"{subject}: unsupported resource binding")
            mesh = normalized_skin.load_mesh(pack_path(root, meshes[binding["mesh"]]), len(skeleton["bones"]))
            material = json.loads(pack_path(root, materials[binding["material"]]).read_text())
            channel = material.get("base_color", material.get("channels", {}).get("base_color"))
            texture = pack_path(root, channel["texture"])
            texture_name = "textures/" + hashlib.sha256(texture.read_bytes()).hexdigest() + ".dds"
            if not (output / texture_name).exists():
                shutil.copyfile(texture, output / texture_name)
            payload = encode(mesh, skeleton, cache)
            filename = f"clips/{subject}_{ordinal}.bin"
            (output / filename).write_bytes(payload)
            total += len(payload)
            parts.append({"mesh": filename, "texture": texture_name,
                          "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                          "vertices": len(mesh["vertices"]), "bones": len(skeleton["bones"])})
        return {"parts": parts, "duration": cache.duration, "frames": cache.frame_count}

    manifest = json.loads((animated / "manifest.json").read_text())
    for resource, record in manifest["resources"].items():
        subjects = []
        for ordinal, candidate in enumerate(record["subject_candidates"]):
            component = json.loads(pack_path(animated, manifest["assets"][candidate["asset"]]["component"]).read_text())
            subjects.append(compile_subject(animated, resource.split("/")[-1]+f"_{ordinal}",
                component.get("meshes", [component["mesh"]]),
                component.get("materials", [component["material"]]), component["skeleton"],
                manifest["animations"][candidate["animation"]], component["draw_bindings"]))
        result["resources"][resource] = subjects
    manifest = json.loads((landmarks / "manifest.json").read_text())
    for resource in ("resource/fish", "resource/whales"):
        asset = manifest["assets"][manifest["resources"][resource]["landmark_asset"]]
        animation = dict(manifest["animations"][resource])
        animation["binding_status"] = animation["pose_status"]
        result["resources"][resource] = [compile_subject(landmarks, resource.split("/")[-1],
            [asset["mesh"]], [asset["material"]], asset["skeleton"], animation,
            [{"mesh": 0, "material": 0}])]
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
