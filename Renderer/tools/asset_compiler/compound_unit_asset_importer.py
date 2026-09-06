#!/usr/bin/env python3
"""Compile mounted and crewed source members into generic socket recipes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import normalized_animation, normalized_skin
from Renderer.tools.asset_compiler.grassland_pack_builder import validate_runtime_independence
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage
from Renderer.tools.asset_compiler.unit_family_action_validator import _best_group
from Renderer.tools.asset_compiler.unit_family_asset_importer import (
    _initial_entry,
    _physical_package,
    load_owner_color_contract,
)
from Renderer.tools.asset_compiler.unit_member_resolver import ASSETS_ROOT, resolve_unit
from Renderer.tools.asset_compiler.unit_model_extractor import _compile_component


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_SETS = Path(__file__).with_name("compound_unit_source_sets.json")
DEFAULT_PACK = RENDERER_ROOT / "packs/CompoundUnitLab"
DEFAULT_REPORT = RENDERER_ROOT / "preview/out/units/compound_unit_build.json"
SAFE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
IDENTITY = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_source_sets(path: Path = DEFAULT_SOURCE_SETS) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "c3x.source_compound_unit_sets.v0":
        raise ValueError("unsupported compound-unit source-set schema")
    if document.get("runtime_integration") != "not_enabled":
        raise ValueError("compound-unit source intake must remain offline-only")
    if document.get("source_content") != "Base":
        raise ValueError("first compound-unit source slice must use Base content")
    compositions = document.get("compositions")
    if not isinstance(compositions, list) or not compositions:
        raise ValueError("compound-unit source set contains no compositions")
    seen_slugs = set()
    for composition in compositions:
        slug = composition.get("slug")
        if not isinstance(slug, str) or not SAFE_ID.fullmatch(slug) or slug in seen_slugs:
            raise ValueError("compound-unit composition has an invalid or duplicate slug")
        seen_slugs.add(slug)
        if not str(composition.get("source_artdef", "")).startswith("UNIT_"):
            raise ValueError(f"{slug} has an invalid source ArtDef")
        if not isinstance(composition.get("member_index"), int):
            raise ValueError(f"{slug} has no member recipe index")
        parent = composition.get("parent")
        children = composition.get("children")
        if not isinstance(parent, dict) or not isinstance(children, list) or not children:
            raise ValueError(f"{slug} must have one parent and at least one child")
        node_ids = [parent.get("id")] + [child.get("id") for child in children]
        if any(not isinstance(node, str) or not SAFE_ID.fullmatch(node) for node in node_ids):
            raise ValueError(f"{slug} contains an invalid node ID")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError(f"{slug} contains duplicate node IDs")
        for node in [parent, *children]:
            if not isinstance(node.get("variation"), str) or not node["variation"]:
                raise ValueError(f"{slug}/{node.get('id')} has no source variation")
        available = {parent["id"]}
        for child in children:
            if child.get("parent") not in available:
                raise ValueError(f"{slug}/{child['id']} references a later or unknown parent")
            if not isinstance(child.get("virtual_member"), str) or not child["virtual_member"]:
                raise ValueError(f"{slug}/{child['id']} has no virtual-member binding")
            for key in ("socket_bone_candidates", "child_root_bone_candidates"):
                values = child.get(key)
                if not isinstance(values, list) or not values or any(
                    not isinstance(value, str) or not value for value in values
                ):
                    raise ValueError(f"{slug}/{child['id']} has invalid {key}")
            available.add(child["id"])
        actions = composition.get("actions")
        if not isinstance(actions, dict) or "idle" not in actions:
            raise ValueError(f"{slug} must define at least idle animation")
        for action, record in actions.items():
            if not SAFE_ID.fullmatch(action) or not isinstance(record, dict):
                raise ValueError(f"{slug} has an invalid action")
            if not isinstance(record.get("loop"), bool):
                raise ValueError(f"{slug}/{action} must declare loop policy")
            if "alias" in record:
                if set(record) != {"loop", "alias"} or record["alias"] not in actions or "nodes" not in actions[record["alias"]]:
                    raise ValueError(f"{slug}/{action} has an invalid direct action alias")
                continue
            if set(record.get("nodes", {})) != set(node_ids):
                raise ValueError(f"{slug}/{action} must bind every node and declare loop policy")
            if any(
                not isinstance(source, str) or not source.startswith("ANIMATION_")
                for source in record["nodes"].values()
            ):
                raise ValueError(f"{slug}/{action} has an invalid animation source")
    contract = document.get("instance_contract", {})
    if (
        contract.get("hud") != "one_retained_native_parent_hud"
        or contract.get("failure") != "atomic_complete_unit_body"
        or contract.get("runtime_dispatch") != "recipe_data_only_no_unit_name_branches"
    ):
        raise ValueError("compound-unit instance ownership contract is invalid")
    return document


def _select_bone(
    candidates: list[str], skeleton: dict[str, Any], label: str
) -> str:
    names = {bone["name"] for bone in skeleton["bones"]}
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise ValueError(f"{label} has none of the required bones: {', '.join(candidates)}")


def _validate_generic_recipe(recipe: dict[str, Any], manifest: dict[str, Any], pack: Path) -> None:
    if recipe.get("schema") != "c3x.unit_composition.v0":
        raise ValueError("unsupported generic compound-unit recipe")
    nodes = recipe.get("nodes")
    if not isinstance(nodes, dict) or recipe.get("root_node") not in nodes:
        raise ValueError("compound-unit recipe has no valid root node")
    root = recipe["root_node"]
    joints = recipe.get("joints")
    if not isinstance(joints, list) or len(joints) != len(nodes) - 1:
        raise ValueError("compound-unit recipe must have exactly one joint per non-root node")
    parents = {}
    for joint in joints:
        parent, child = joint.get("parent"), joint.get("child")
        if parent not in nodes or child not in nodes or child == root or child in parents:
            raise ValueError("compound-unit recipe has an invalid parent/child joint")
        transform = joint.get("local_transform")
        if (
            not isinstance(transform, list)
            or len(transform) != 16
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in transform)
        ):
            raise ValueError("compound-unit joint has an invalid local transform")
        parent_skeleton = normalized_skin.load_skeleton(pack / nodes[parent]["skeleton"])
        child_skeleton = normalized_skin.load_skeleton(pack / nodes[child]["skeleton"])
        _select_bone([joint["parent_bone"]], parent_skeleton, f"joint {parent}->{child} parent")
        _select_bone([joint["child_root_bone"]], child_skeleton, f"joint {parent}->{child} child")
        parents[child] = parent
    for node in nodes:
        visited = set()
        current = node
        while current != root:
            if current in visited or current not in parents:
                raise ValueError("compound-unit node graph is cyclic or disconnected")
            visited.add(current)
            current = parents[current]
    for node_id, node in nodes.items():
        if node.get("animation_driver") not in manifest["assets"]:
            raise ValueError(f"compound-unit node {node_id} has no valid animation driver")
        if any(component["asset"] not in manifest["assets"] for component in node["components"]):
            raise ValueError(f"compound-unit node {node_id} references an unknown component")
    if not isinstance(recipe.get("actions"), dict) or not recipe["actions"]:
        raise ValueError("compound-unit recipe has no action bindings")
    for action, binding in recipe["actions"].items():
        if binding.get("timeline") != "shared_normalized_phase" or set(binding.get("node_clips", {})) != set(nodes):
            raise ValueError(f"compound-unit action {action} does not bind every node")
        if any(animation not in manifest["animations"] for animation in binding["node_clips"].values()):
            raise ValueError(f"compound-unit action {action} references an unknown animation")
    if recipe.get("instance_contract", {}).get("hud") != "one_retained_native_parent_hud":
        raise ValueError("compound-unit recipe would duplicate native unit HUD")


def compile_compound_units(
    assets_root: Path,
    source_sets_path: Path = DEFAULT_SOURCE_SETS,
    pack: Path = DEFAULT_PACK,
    report_path: Path = DEFAULT_REPORT,
    require_animations: bool = False,
    only_slugs: set[str] | None = None,
) -> dict[str, Any]:
    source_sets = load_source_sets(source_sets_path)
    if only_slugs is not None:
        known = {item["slug"] for item in source_sets["compositions"]}
        unknown = sorted(only_slugs - known)
        if unknown:
            raise ValueError("unknown compound-unit slug: " + ", ".join(unknown))
        source_sets = {
            **source_sets,
            "compositions": [
                item for item in source_sets["compositions"] if item["slug"] in only_slugs
            ],
        }
    owner_color = load_owner_color_contract()
    tint_strength = owner_color["shader"]["lab_calibration"]["strength"]
    content = source_sets["source_content"]
    shared_data = assets_root / content / "Platforms/Windows/BLPs/SHARED_DATA"
    if not shared_data.is_dir():
        raise FileNotFoundError(shared_data)

    resolved: list[tuple[dict[str, Any], dict[str, dict[str, Any]]]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for composition in source_sets["compositions"]:
        nodes = {}
        for node in [composition["parent"], *composition["children"]]:
            recipe = resolve_unit(
                assets_root,
                composition["source_artdef"],
                "Any",
                node["variation"],
                composition["member_index"],
            )
            nodes[node["id"]] = recipe
            for component in recipe["selected_components"]:
                grouped[component["source_package"]].append(
                    {"composition": composition, "node": node, "component": component}
                )
        parent_recipe = nodes[composition["parent"]["id"]]
        virtuals = {
            (item["member"], item["point"]): item for item in parent_recipe["virtual_attachments"]
        }
        for child in composition["children"]:
            if (child["virtual_member"], child["socket"]) not in virtuals:
                raise ValueError(
                    f"{composition['slug']}/{child['id']} does not match a declared virtual attachment"
                )
            if nodes[child["id"]]["member"]["is_attachment"] is not True:
                raise ValueError(f"{composition['slug']}/{child['id']} is not an attachment variation")
        resolved.append((composition, nodes))

    packages = {}
    package_reports = {}
    for logical, items in grouped.items():
        path = _physical_package(assets_root, content, logical)
        packages[logical] = IndexedStaticPackage(
            path,
            _initial_entry(path, [item["component"]["source_entry"] for item in items]),
        )
        package_reports[logical] = {"path": str(path), "sha256": _sha256(path)}

    assets: dict[str, Any] = {}
    units: dict[str, Any] = {}
    animations: dict[str, Any] = {}
    component_evidence = []
    animation_evidence = []
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for composition, source_nodes in resolved:
        slug = composition["slug"]
        compiled_nodes = {}
        source_node_records = [composition["parent"], *composition["children"]]
        for node in source_node_records:
            node_id = node["id"]
            source_recipe = source_nodes[node_id]
            role_counts: dict[str, int] = defaultdict(int)
            component_records = []
            driver_candidates = []
            for component in source_recipe["selected_components"]:
                role = re.sub(r"[^a-z0-9]+", "_", component["role"].lower()).strip("_")
                role_counts[role] += 1
                key = role if role_counts[role] == 1 else f"{role}_{role_counts[role]}"
                compile_slug = f"{slug}_{node_id}"
                try:
                    asset, evidence = _compile_component(
                        packages[component["source_package"]],
                        shared_data,
                        pack,
                        component,
                        texture_cache,
                        compile_slug,
                        key,
                    )
                except (ValueError, KeyError, struct.error) as exc:
                    raise ValueError(
                        f"{slug}/{node_id}/{component['source_entry']}: {exc}"
                    ) from exc
                asset_id = f"unit/{compile_slug}/{key}"
                document_path = pack / asset["component"]
                document = json.loads(document_path.read_text(encoding="utf-8"))
                document["owner_color"] = (
                    {
                        "mode": "source_mask",
                        "mask_source": "base_color_alpha_inverse",
                        "strength": tint_strength,
                        "representative_palette_index": 6,
                    }
                    if component["tint"] == "USE_CIV_COLOR"
                    else {
                        "mode": "none",
                        "mask_source": "constant_one",
                        "strength": 0.0,
                        "representative_palette_index": 6,
                    }
                )
                _write_json(document_path, document)
                assets[asset_id] = asset
                component_records.append(
                    {
                        "asset": asset_id,
                        "role": component["role"],
                        "attachment_point": component["point"],
                        "scale": component["scale"],
                        "tint": component["tint"],
                    }
                )
                skeleton = normalized_skin.load_skeleton(pack / document["skeleton"])
                driver_candidates.append((len(skeleton["bones"]), asset_id, document["skeleton"], skeleton))
                component_evidence.append(
                    {
                        "composition": slug,
                        "node": node_id,
                        "source_package": component["source_package"],
                        **evidence,
                    }
                )
            driver_candidates.sort(key=lambda item: (-item[0], item[1]))
            _, driver_id, skeleton_path, skeleton = driver_candidates[0]
            compiled_nodes[node_id] = {
                "components": component_records,
                "animation_driver": driver_id,
                "skeleton": skeleton_path,
                "member_scale": source_recipe["member"]["member_scale"],
                "variation_scale": source_recipe["member"]["variation_scale"],
                "source_track_group": skeleton["track_group"],
            }

        joints = []
        for child in composition["children"]:
            parent_node = compiled_nodes[child["parent"]]
            child_node = compiled_nodes[child["id"]]
            parent_skeleton = normalized_skin.load_skeleton(pack / parent_node["skeleton"])
            child_skeleton = normalized_skin.load_skeleton(pack / child_node["skeleton"])
            parent_bone = _select_bone(
                child["socket_bone_candidates"], parent_skeleton, f"{slug}/{child['id']} parent socket"
            )
            child_root = _select_bone(
                child["child_root_bone_candidates"], child_skeleton, f"{slug}/{child['id']} child root"
            )
            joints.append(
                {
                    "parent": child["parent"],
                    "child": child["id"],
                    "socket": child["socket"],
                    "parent_bone": parent_bone,
                    "child_root_bone": child_root,
                    "local_transform": IDENTITY,
                    "transform_convention": "row_major_row_vector_child_local_then_parent_socket",
                    "scale_policy": "node_member_times_variation_then_parent_instance",
                }
            )

        action_bindings = {}
        for action, source_action in composition["actions"].items():
            if "alias" in source_action:
                continue
            node_clips = {}
            for node_id, source_name in source_action["nodes"].items():
                animation_id = f"animation/unit/{slug}/{node_id}/{action}"
                relative = Path("animations/unit") / slug / node_id / f"{action}.c3anim"
                output = pack / relative
                source = shared_data / source_name
                if not source.is_file():
                    raise FileNotFoundError(source)
                record: dict[str, Any] = {
                    "clip": relative.as_posix(),
                    "loop": source_action["loop"],
                    "binding_status": "pending_offline_conversion",
                }
                evidence = {
                    "composition": slug,
                    "node": node_id,
                    "action": action,
                    "source": source_name,
                    "source_path": str(source),
                    "source_sha256": _sha256(source),
                    "output": str(output),
                    "converted": output.is_file(),
                }
                if output.is_file():
                    clip = normalized_animation.load_clip(output)
                    skeleton = normalized_skin.load_skeleton(pack / compiled_nodes[node_id]["skeleton"])
                    group_index, common = _best_group(
                        clip, {bone["name"] for bone in skeleton["bones"]}
                    )
                    record.update(
                        {
                            "binding_status": "validated_node_local_raw_clip",
                            "group_index": group_index,
                            "matched_tracks": len(common),
                            "duration": clip.duration,
                            "sample_rate": clip.sample_rate,
                            "frame_count": clip.frame_count,
                            "sha256": _sha256(output),
                        }
                    )
                    evidence["matched_tracks"] = len(common)
                elif require_animations:
                    raise ValueError(f"compound-unit animation has not been converted: {relative}")
                animations[animation_id] = record
                node_clips[node_id] = animation_id
                animation_evidence.append(evidence)
            action_bindings[action] = {
                "loop": source_action["loop"],
                "timeline": "shared_normalized_phase",
                "node_clips": node_clips,
                "completion": "authoritative_civ3_action_not_child_clip_duration",
            }
        for action, source_action in composition["actions"].items():
            if "alias" not in source_action:
                continue
            target = source_action["alias"]
            action_bindings[action] = {
                "loop": source_action["loop"],
                "timeline": "shared_normalized_phase",
                "node_clips": dict(action_bindings[target]["node_clips"]),
                "alias_of": target,
                "completion": "authoritative_civ3_action_not_child_clip_duration",
            }

        recipe_path = Path("units") / f"{slug}_composition.json"
        recipe = {
            "schema": "c3x.unit_composition.v0",
            "unit_id": f"unit/{slug}",
            "civ3_ids": composition["civ3_ids"],
            "root_node": composition["parent"]["id"],
            "nodes": compiled_nodes,
            "joints": joints,
            "actions": action_bindings,
            "instance_contract": source_sets["instance_contract"],
            "runtime_integration": "not_enabled",
        }
        _write_json(pack / recipe_path, recipe)
        units[f"unit/{slug}"] = {"recipe": recipe_path.as_posix(), "type": "compound"}

    manifest = {
        "schema": "c3x.unit_pack.v0",
        "name": "CompoundUnitLab",
        "source_policy": "Local licensed-source import; derived art is not redistributable.",
        "units": units,
        "assets": assets,
        "animations": animations,
        "owner_color_contract": "units/owner_color_runtime.json",
        "composition_contract": {
            "schema": "c3x.unit_composition.v0",
            "node_graph": "arbitrary_acyclic_parent_child_tree",
            "animation": "node_local_clips_on_one_authoritative_action_clock",
            "runtime_dispatch": "recipe_data_only_no_unit_name_branches",
        },
        "runtime_integration": "not_enabled",
    }
    _write_json(pack / manifest["owner_color_contract"], owner_color)
    _write_json(pack / "manifest.json", manifest)
    for unit in units.values():
        recipe = json.loads((pack / unit["recipe"]).read_text(encoding="utf-8"))
        _validate_generic_recipe(recipe, manifest, pack)
    independence_errors = validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError("Runtime compound-unit pack is source-dependent: " + "; ".join(independence_errors))

    report = {
        "schema": "c3x.source_compound_unit_build.v0",
        "source_sets": {"path": str(source_sets_path), "sha256": _sha256(source_sets_path)},
        "packages": package_reports,
        "components": component_evidence,
        "animations": animation_evidence,
        "resolved_compositions": [
            {
                "slug": composition["slug"],
                "source_artdef": composition["source_artdef"],
                "nodes": source_nodes,
            }
            for composition, source_nodes in resolved
        ],
        "outputs": {
            "pack": str(pack),
            "compositions": len(units),
            "nodes": sum(
                len(json.loads((pack / unit["recipe"]).read_text(encoding="utf-8"))["nodes"])
                for unit in units.values()
            ),
            "joints": sum(len(composition["children"]) for composition in source_sets["compositions"]),
            "components": len(assets),
            "textures": len(texture_cache),
            "animation_bindings": len(animations),
            "converted_animation_bindings": sum(
                record["binding_status"] == "validated_node_local_raw_clip"
                for record in animations.values()
            ),
        },
        "runtime_independence": "passed",
        "runtime_integration": "not_enabled",
    }
    _write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=ASSETS_ROOT)
    parser.add_argument("--source-sets", type=Path, default=DEFAULT_SOURCE_SETS)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--require-animations", action="store_true")
    parser.add_argument("--only", action="append", dest="only_slugs")
    args = parser.parse_args(argv)
    try:
        report = compile_compound_units(
            args.assets_root,
            args.source_sets,
            args.pack,
            args.report,
            args.require_animations,
            None if args.only_slugs is None else set(args.only_slugs),
        )
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
