#!/usr/bin/env python3
"""Compile animated resource bodies and model-aware pose caches for future L16 Lab use."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import normalized_animation, normalized_pose_cache, normalized_skin
from Renderer.tools.asset_compiler.clutter_blp_extractor import landmark_base_model
from Renderer.tools.asset_compiler.compound_landmark_importer import _decode_skeletons
from Renderer.tools.asset_compiler.grassland_pack_builder import validate_runtime_independence
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage
from Renderer.tools.asset_compiler.resource_animation_converter import load_extract_report
from Renderer.tools.asset_compiler.unit_family_action_validator import _best_group
from Renderer.tools.asset_compiler.unit_member_resolver import ASSETS_ROOT
from Renderer.tools.asset_compiler.unit_model_extractor import _compile_component


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTRACT_REPORT = RENDERER_ROOT / "preview/out/resources/resource_animation_extract.json"
DEFAULT_SOURCE_PACK = RENDERER_ROOT / "packs/ResourceNormalized"
DEFAULT_PACK = RENDERER_ROOT / "packs/ResourceAnimatedLab"
DEFAULT_REPORT = RENDERER_ROOT / "preview/out/resources/resource_body_profiles.json"
COMPONENT_SCHEMA = "c3x.animated_resource_component.v0"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resource_slug(resource_id: str) -> str:
    if not resource_id.startswith("resource/"):
        raise ValueError(f"unsupported resource id {resource_id!r}")
    slug = resource_id.split("/", 1)[1]
    if not slug or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in slug):
        raise ValueError(f"resource id is not canonical: {resource_id!r}")
    return slug


def _semantic_skeleton_hash(skeleton: dict[str, Any]) -> str:
    semantic = {
        "track_group": skeleton["track_group"],
        "matrix_convention": skeleton["matrix_convention"],
        "position_unit": skeleton["position_unit"],
        "bones": skeleton["bones"],
    }
    return hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _pose_frames(
    skeleton: dict[str, Any], clip: normalized_animation.AnimationClip, group_index: int
) -> tuple[list[tuple[tuple[float, ...], ...]], int]:
    frames = []
    repaired_channels = 0
    for frame in range(clip.frame_count):
        time = clip.duration * frame / (clip.frame_count - 1)
        sampled = normalized_skin.sample_pose(skeleton, clip, group_index, time, False)
        pose = []
        for bone, transform in zip(skeleton["bones"], sampled):
            local = bone["local"]
            position = transform.position
            scale_shear = transform.scale_shear
            # Some shipped Granny tracks contain a single enormous sentinel in
            # an otherwise ordinary sampled channel. A tile-normalized resource
            # can never legitimately use it; preserve the authored rest channel
            # for that sample and record the repair in source evidence.
            if any(not math.isfinite(value) or abs(value) > 1024.0 for value in position):
                position = tuple(local["position"])
                repaired_channels += 1
            if any(not math.isfinite(value) or abs(value) > 1024.0 for value in scale_shear):
                scale_shear = tuple(local["scale_shear"])
                repaired_channels += 1
            pose.append(
                normalized_animation.Transform(position, transform.orientation, scale_shear)
            )
        worlds = normalized_skin.world_matrices(skeleton, pose)
        values = [value for matrix in worlds for value in matrix]
        if any(not math.isfinite(value) for value in values) or max(abs(value) for value in values) > 1024.0:
            raise ValueError(
                "resource animation produces an invalid world matrix "
                f"(maximum absolute component {max(abs(value) for value in values):.9g})"
            )
        frames.append(worlds)
    return frames, repaired_channels


def validate_animated_resource_pack(pack: Path) -> dict[str, int]:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "c3x.animated_resource_pack.v0":
        raise ValueError("unsupported animated-resource pack schema")
    assets = manifest.get("assets")
    animations = manifest.get("animations")
    resources = manifest.get("resources")
    if not all(isinstance(value, dict) and value for value in (assets, animations, resources)):
        raise ValueError("animated-resource pack is incomplete")
    referenced = set()
    for resource_id, resource in resources.items():
        _resource_slug(resource_id)
        candidates = resource.get("subject_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{resource_id} has no animated subject candidates")
        for candidate in candidates:
            asset_id = candidate.get("asset")
            animation_id = candidate.get("animation")
            if asset_id not in assets or animation_id not in animations:
                raise ValueError(f"{resource_id} references a missing body or animation")
            referenced.add(asset_id)
            animation = animations[animation_id]
            document = json.loads((pack / assets[asset_id]["component"]).read_text(encoding="utf-8"))
            if document.get("schema") != COMPONENT_SCHEMA or document.get("binding_mode") != "vertex_skin":
                raise ValueError(f"{asset_id} is not a normalized skinned resource body")
            skeleton = normalized_skin.load_skeleton(pack / document["skeleton"])
            clip = normalized_animation.load_clip(pack / animation["clip"])
            cache = normalized_pose_cache.load_pose_cache(pack / animation["pose_cache"])
            normalized_pose_cache.validate_skeleton_binding(cache, skeleton)
            if animation.get("group_index", -1) >= len(clip.groups):
                raise ValueError(f"{animation_id} has an invalid group binding")
            if _sha256(pack / animation["clip"]) != animation.get("sha256"):
                raise ValueError(f"{animation_id} clip hash is stale")
            if _sha256(pack / animation["pose_cache"]) != animation.get("pose_cache_sha256"):
                raise ValueError(f"{animation_id} pose-cache hash is stale")
    if referenced != set(assets):
        raise ValueError("animated-resource pack contains orphan bodies")
    independence = validate_runtime_independence(pack)
    if independence:
        raise ValueError("runtime pack is source-dependent: " + "; ".join(independence))
    return {
        "resources": len(resources),
        "subjects": len(assets),
        "animations": len(animations),
    }


def compile_resource_body_profiles(
    assets_root: Path,
    extract_report: Path = DEFAULT_EXTRACT_REPORT,
    source_pack: Path = DEFAULT_SOURCE_PACK,
    pack: Path = DEFAULT_PACK,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    extraction = load_extract_report(extract_report)
    source_package = assets_root / "Base/Platforms/Windows/BLPs/environment/clutter.blp"
    shared_data = assets_root / "Base/Platforms/Windows/BLPs/SHARED_DATA"
    pending = [
        asset
        for package in extraction["packages"]
        if package["source_package"] == "environment/clutter"
        for asset in package["assets"]
        if asset["binding_status"] == "raw_clips_extracted_body_profile_pending"
    ]
    if not pending:
        raise ValueError("resource extraction report contains no pending clutter body profiles")
    if not source_package.is_file() or not shared_data.is_dir():
        raise FileNotFoundError(source_package if not source_package.is_file() else shared_data)
    for asset in pending:
        if len(asset["resource_ids"]) != 1 or len(asset["clips"]) != 1:
            raise ValueError(f"resource body is not a one-resource/one-clip profile: {asset}")

    package = IndexedStaticPackage(source_package, pending[0]["source_entry"])
    resources: dict[str, dict[str, Any]] = {}
    assets: dict[str, dict[str, Any]] = {}
    animations: dict[str, dict[str, Any]] = {}
    evidence = []
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    resource_ordinals: defaultdict[str, int] = defaultdict(int)
    copied_clips: dict[str, str] = {}
    pose_paths: dict[str, str] = {}

    for item in sorted(pending, key=lambda value: (value["resource_ids"][0], value["source_entry"])):
        resource_id = item["resource_ids"][0]
        slug = _resource_slug(resource_id)
        ordinal = resource_ordinals[resource_id]
        resource_ordinals[resource_id] += 1
        subject_slug = f"{slug}_subject_{ordinal:02d}"
        asset_id = f"resource/{subject_slug}/body"
        animation_id = f"animation/resource/{subject_slug}/ambient"

        package.select_direct_string(item["source_entry"])
        entry, user_data, base_model = landmark_base_model(package)
        source_skeletons, _source_skeleton_evidence = _decode_skeletons(
            package, base_model, 100.0
        )
        source_model_index = max(
            range(len(source_skeletons)), key=lambda index: len(source_skeletons[index]["bones"])
        )
        asset, body_report = _compile_component(
            package,
            shared_data,
            pack,
            {
                "source_entry": item["source_entry"],
                "role": "Body",
                "point": "Root",
                "scale": 1.0,
                "tint": None,
                "source_model_index": source_model_index,
                "source_skeleton_index": source_model_index,
                "resolved_source_pointers": {
                    "entry": entry,
                    "user_data": user_data,
                    "base_model": base_model,
                },
            },
            texture_cache,
            unit_slug=subject_slug,
            component_key="body",
            artifact_family="resource",
            component_schema=COMPONENT_SCHEMA,
        )
        if body_report["geometry"]["binding_mode"] != "vertex_skin":
            raise ValueError(f"animated resource body is not wholly skinned: {item['source_entry']}")
        document = json.loads((pack / asset["component"]).read_text(encoding="utf-8"))
        skeleton = normalized_skin.load_skeleton(pack / document["skeleton"])

        source_clip = source_pack / item["clips"][0]["normalized_clip"]
        clip_hash = _sha256(source_clip)
        clip_relative = copied_clips.get(clip_hash)
        if clip_relative is None:
            clip_relative = f"animations/clips/{clip_hash}.c3anim"
            (pack / clip_relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_clip, pack / clip_relative)
            copied_clips[clip_hash] = clip_relative
        clip = normalized_animation.load_clip(pack / clip_relative)
        group_index, matching = _best_group(clip, {bone["name"] for bone in skeleton["bones"]})
        mesh_paths = document.get("meshes") or [document["mesh"]]
        bindings = [
            normalized_skin.bind_clip(
                normalized_skin.load_mesh(pack / mesh_path, len(skeleton["bones"])),
                skeleton,
                clip,
                group_index,
            )
            for mesh_path in mesh_paths
        ]

        pose_key = hashlib.sha256(
            (clip_hash + _semantic_skeleton_hash(skeleton) + str(group_index)).encode("ascii")
        ).hexdigest()
        pose_relative = pose_paths.get(pose_key)
        if pose_relative is None:
            pose_relative = f"animations/poses/{pose_key}.c3pose"
            frames, repaired_channels = _pose_frames(skeleton, clip, group_index)
            cache = normalized_pose_cache.write_pose_cache(
                pack / pose_relative,
                clip.duration,
                clip.sample_rate,
                [bone["name"] for bone in skeleton["bones"]],
                frames,
            )
            normalized_pose_cache.validate_skeleton_binding(cache, skeleton)
            pose_paths[pose_key] = pose_relative
        else:
            repaired_channels = 0

        assets[asset_id] = asset
        animations[animation_id] = {
            "clip": clip_relative,
            "sha256": clip_hash,
            "pose_cache": pose_relative,
            "pose_cache_sha256": _sha256(pack / pose_relative),
            "group_index": group_index,
            "loop": True,
            "duration": clip.duration,
            "sample_rate": clip.sample_rate,
            "frame_count": clip.frame_count,
            "binding_status": "validated_model_aware_pose_cache",
        }
        resources.setdefault(
            resource_id,
            {
                "presentation_profile": (
                    "single_primary_subject"
                    if resource_id in {"resource/horses", "resource/cattle", "resource/game", "resource/ivory", "resource/furs"}
                    else "source_authored_cluster"
                ),
                "subject_candidates": [],
            },
        )["subject_candidates"].append(
            {"asset": asset_id, "animation": animation_id, "weight": 1}
        )
        evidence.append(
            {
                "resource_id": resource_id,
                "source_entry": item["source_entry"],
                "source_clip": item["clips"][0]["name"],
                "asset": asset_id,
                "animation": animation_id,
                "clip_sha256": clip_hash,
                "pose_cache_sha256": animations[animation_id]["pose_cache_sha256"],
                "matched_tracks": len(matching),
                "repaired_outlier_channels": repaired_channels,
                "bindings": bindings,
                "body": body_report,
            }
        )

    manifest = {
        "schema": "c3x.animated_resource_pack.v0",
        "name": "ResourceAnimatedLab",
        "source_policy": "Local licensed-source import; derived art is not redistributable.",
        "resources": resources,
        "assets": assets,
        "animations": animations,
        "runtime_integration": "not_enabled",
    }
    _write_json(pack / "manifest.json", manifest)
    validated = validate_animated_resource_pack(pack)
    report = {
        "schema": "c3x.resource_body_profile_build.v0",
        "source_package": {"path": str(source_package), "sha256": _sha256(source_package)},
        "source_units_per_tile": 100.0,
        "profiles": evidence,
        "summary": {
            **validated,
            "unique_clips": len(copied_clips),
            "unique_pose_caches": len(pose_paths),
            "textures": len(texture_cache),
            "source_profiles_pending": 0,
        },
        "runtime_independence": "passed",
        "runtime_integration": "not_enabled",
    }
    _write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=ASSETS_ROOT)
    parser.add_argument("--extract-report", type=Path, default=DEFAULT_EXTRACT_REPORT)
    parser.add_argument("--source-pack", type=Path, default=DEFAULT_SOURCE_PACK)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        report = compile_resource_body_profiles(
            args.assets_root, args.extract_report, args.source_pack, args.pack, args.report
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summary = report["summary"]
    print(
        f"Compiled {summary['subjects']} animated resource bodies across "
        f"{summary['resources']} resources; {summary['source_profiles_pending']} pending"
    )
    print(f"Pack: {args.pack}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
