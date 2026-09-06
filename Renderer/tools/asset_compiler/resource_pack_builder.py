#!/usr/bin/env python3
"""Inventory and compile Civ III-mapped Civ VI resource art into a generic pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import clutter_blp_extractor
from Renderer.tools.asset_compiler import normalized_animation
from Renderer.tools.asset_compiler import normalized_pose_cache
from Renderer.tools.asset_compiler import normalized_skin
from Renderer.tools.asset_compiler import resource_skin_extractor
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage
from Renderer.tools.asset_compiler.terrain_relief_builder import fnv1a32


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_ROOT = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common"
    / "Sid Meier's Civilization VI/Civ6.app/Contents/Assets"
)
DEFAULT_MAPPING = RENDERER_ROOT / "inventory" / "vanilla_conquests_to_civ6_resources.json"
DEFAULT_PRESENTATION = RENDERER_ROOT / "inventory" / "resource_presentation_profiles.json"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "resources" / "resource_inventory.json"
DEFAULT_PROBE_PACK = RENDERER_ROOT / "preview" / "out" / "resources" / "static_probe_pack"
DEFAULT_ANIMATION_STAGE = RENDERER_ROOT / "preview" / "out" / "resources" / "raw_animations"
DEFAULT_PACK = RENDERER_ROOT / "packs" / "ResourceNormalized"
COMPOUND_MAPPING = Path(__file__).with_name("compound_landmark_sets.json")
DECAL_MAPPING = Path(__file__).with_name("decal_sets.json")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_special_asset_routes() -> dict[str, dict[str, str]]:
    """Map source entries to normalized assets owned by specialized pack adapters."""
    routes: dict[str, dict[str, str]] = {}
    compound = json.loads(COMPOUND_MAPPING.read_text(encoding="utf-8"))
    for package in compound["packages"]:
        if not package["source_package"].endswith("/environment/clutter.blp"):
            continue
        for asset in package["assets"]:
            routes[asset["source_entry"]] = {
                "pack": "CompoundLandmarksNormalized",
                "asset": asset["asset_id"],
            }
    decals = json.loads(DECAL_MAPPING.read_text(encoding="utf-8"))
    for group in decals["groups"]:
        if not group["group_id"].startswith("resource/"):
            continue
        for asset in group["assets"]:
            source = asset["source_asset"]
            if source in routes:
                raise ValueError(f"Source resource asset has multiple specialized routes: {source}")
            routes[source] = {
                "pack": "DecalsNormalized",
                "asset": asset["asset_id"],
            }
    return routes


def load_landmark_routes() -> dict[str, dict[str, str]]:
    """Map tile-base source entries to normalized compound landmark assets."""
    routes: dict[str, dict[str, str]] = {}
    compound = json.loads(COMPOUND_MAPPING.read_text(encoding="utf-8"))
    for package in compound["packages"]:
        if not package["source_package"].endswith("/landmarks/tilebases.blp"):
            continue
        for asset in package["assets"]:
            routes[asset["source_entry"]] = {
                "pack": "CompoundLandmarksNormalized",
                "asset": asset["asset_id"],
            }
    return routes


def load_resource_presentation(path: Path = DEFAULT_PRESENTATION) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "c3x.resource_presentation_profiles.v0":
        raise ValueError("Resource presentation profiles have an invalid schema")
    profiles = document.get("profiles")
    default_profile = document.get("default_profile")
    bindings = document.get("resource_bindings")
    if not isinstance(profiles, dict) or default_profile not in profiles:
        raise ValueError("Resource presentation default profile is undefined")
    if not isinstance(bindings, dict):
        raise ValueError("Resource presentation bindings must be an object")
    for resource_id, binding in bindings.items():
        if not isinstance(binding, dict) or binding.get("profile") not in profiles:
            raise ValueError(f"Resource presentation binding is invalid: {resource_id}")
        entries = binding.get("primary_source_entries", [])
        if not isinstance(entries, list) or any(not isinstance(entry, str) for entry in entries):
            raise ValueError(f"Resource primary-source entries are invalid: {resource_id}")
    return document


def _named_entry(root: ET.Element, name: str) -> ET.Element:
    matches = [
        element
        for element in root.iter("Element")
        if (named := element.find("./m_Name")) is not None and named.get("text") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one ArtDef entry {name}, found {len(matches)}")
    return matches[0]


def _collections(entry: ET.Element, name: str) -> list[ET.Element]:
    return [
        collection
        for collection in entry.findall("./m_ChildCollections/Element")
        if (named := collection.find("./m_CollectionName")) is not None
        and named.get("text") == name
    ]


def _values(item: ET.Element) -> dict[str, ET.Element]:
    result: dict[str, ET.Element] = {}
    for value in item.findall("./m_Fields/m_Values/Element"):
        parameter = value.find("./m_ParamName")
        if parameter is not None and parameter.get("text"):
            result[parameter.get("text", "")] = value
    return result


def _scalar(value: ET.Element | None) -> str | None:
    if value is None:
        return None
    child = next((child for child in value if child.tag != "m_ParamName"), None)
    if child is None:
        return None
    return child.get("text", child.text)


def _xref_names(entry: ET.Element, collection_names: tuple[str, ...]) -> list[str]:
    names: list[str] = []
    for collection_name in collection_names:
        for collection in _collections(entry, collection_name):
            for item in collection.findall("./Element"):
                values = _values(item)
                value = _scalar(values.get("XrefName")) or _scalar(values.get("Xref"))
                if value and value not in names:
                    names.append(value)
    return names


def _asset_value(value: ET.Element) -> dict[str, str]:
    fields = {
        "entry": value.findtext("./m_EntryName", default=""),
        "class": value.findtext("./m_XLPClass", default=""),
        "package": value.findtext("./m_BLPPackage", default=""),
        "library": value.findtext("./m_LibraryName", default=""),
    }
    for key, field in fields.items():
        if not field:
            element = value.find({
                "entry": "./m_EntryName",
                "class": "./m_XLPClass",
                "package": "./m_BLPPackage",
                "library": "./m_LibraryName",
            }[key])
            fields[key] = "" if element is None else element.get("text", "")
    if not fields["entry"] or not fields["package"]:
        raise ValueError("ArtDef Asset value has no entry or package")
    return fields


def _placement(item: ET.Element) -> dict[str, Any] | None:
    values = _values(item)
    asset = values.get("Asset")
    if asset is None:
        return None
    result: dict[str, Any] = {"asset": _asset_value(asset)}
    converters = {
        "Scale": ("scale", float, 1.0),
        "Count": ("count", int, 1),
        "ScaleVariation": ("scale_variation", float, 0.0),
        "MinCount": ("min_count", int, 0),
        "Priority": ("priority", int, 0),
        "Width": ("width", float, 0.0),
        "LowendReduction": ("low_end_reduction", float, 0.0),
    }
    for source, (target, converter, default) in converters.items():
        raw = _scalar(values.get(source))
        result[target] = default if raw in (None, "") else converter(raw)
    for source, target in (
        ("ShowDecal", "show_decal"),
        ("IsCenterModel", "is_center_model"),
        ("AllowOverlap", "allow_overlap"),
    ):
        result[target] = (_scalar(values.get(source)) or "false").lower() == "true"
    result["rotate_mode"] = _scalar(values.get("RotateMode")) or "RotateZ"
    result["name"] = (item.find("./m_Name").get("text", "") if item.find("./m_Name") is not None else "")
    return result


def _clutter_placements(clutter_root: ET.Element, set_name: str) -> list[dict[str, Any]]:
    entry = _named_entry(clutter_root, set_name)
    plants = _collections(entry, "Plants")
    if len(plants) != 1:
        raise ValueError(f"Expected one Plants collection for {set_name}, found {len(plants)}")
    return [
        placement
        for item in plants[0].findall("./Element")
        if (placement := _placement(item)) is not None
    ]


def _landmark_assets(landmark_root: ET.Element, xref: str) -> list[dict[str, Any]]:
    entry = _named_entry(landmark_root, xref)
    assets = []
    for value in entry.findall(".//m_Fields/m_Values/Element"):
        parameter = value.find("./m_ParamName")
        if parameter is not None and parameter.get("text") == "Asset":
            candidate = _asset_value(value)
            if candidate not in assets:
                assets.append(candidate)
    if not assets:
        raise ValueError(f"Landmark ArtDef entry {xref} has no Asset")
    return assets


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_inventory(mapping_path: Path, assets_root: Path) -> dict[str, Any]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    artdefs = assets_root / "Base" / "ArtDefs"
    resource_root = ET.parse(artdefs / "Resources.artdef").getroot()
    feature_root = ET.parse(artdefs / "Features.artdef").getroot()
    clutter_root = ET.parse(artdefs / "Clutter.artdef").getroot()
    landmark_root = ET.parse(artdefs / "Landmarks.artdef").getroot()
    resources = []
    unique_assets: dict[tuple[str, str], dict[str, str]] = {}
    for item in mapping["mappings"]:
        target_root = resource_root if item["target_kind"] == "resource" else feature_root
        entry = _named_entry(target_root, item["civ6_artdef"])
        clutter_sets = _xref_names(entry, ("Clutter", "ClutterVariants"))
        landmark_xrefs = _xref_names(entry, ("Landmark",))
        placements = []
        for set_name in clutter_sets:
            for placement in _clutter_placements(clutter_root, set_name):
                placement = {**placement, "clutter_set": set_name}
                placements.append(placement)
                asset = placement["asset"]
                unique_assets[(asset["package"], asset["entry"])] = asset
        landmarks = []
        for xref in landmark_xrefs:
            for asset in _landmark_assets(landmark_root, xref):
                landmarks.append({"xref": xref, "asset": asset})
                unique_assets[(asset["package"], asset["entry"])] = asset
        resources.append(
            {
                "resource_id": "resource/" + _slug(item["civ3_id"].removeprefix("GOOD_")),
                "civ3_biq_index": item["civ3_biq_index"],
                "civ3_id": item["civ3_id"],
                "civ3_name": item["civ3_name"],
                "civ6_artdef": item["civ6_artdef"],
                "match": item["match"],
                "confidence": item["confidence"],
                "clutter_sets": clutter_sets,
                "placements": placements,
                "landmarks": landmarks,
            }
        )
    package_counts: dict[str, int] = {}
    for package, _entry in unique_assets:
        package_counts[package] = package_counts.get(package, 0) + 1
    return {
        "schema": "c3x.resource_source_inventory.v0",
        "mapping_count": len(resources),
        "resources": resources,
        "unique_assets": sorted(unique_assets.values(), key=lambda asset: (asset["package"], asset["entry"])),
        "summary": {
            "resources": len(resources),
            "resources_with_clutter": sum(bool(item["placements"]) for item in resources),
            "resources_with_landmarks": sum(bool(item["landmarks"]) for item in resources),
            "placements": sum(len(item["placements"]) for item in resources),
            "unique_assets": len(unique_assets),
            "packages": package_counts,
        },
    }


def probe_static_assets(
    inventory: dict[str, Any], assets_root: Path, pack: Path
) -> dict[str, Any]:
    sources = [
        asset for asset in inventory["unique_assets"] if asset["package"] == "environment/clutter"
    ]
    if not sources:
        raise ValueError("Resource inventory contains no environment/clutter assets")
    blp_root = assets_root / "Base" / "Platforms" / "Windows" / "BLPs"
    package_path = blp_root / "environment" / "clutter.blp"
    package = IndexedStaticPackage(package_path, "Tree_Pine_01")
    shared_data = blp_root / "SHARED_DATA"
    results = []
    special_routes = load_special_asset_routes()
    used_stems: set[str] = set()
    for source in sources:
        stem = _slug(source["entry"])
        if stem in used_stems:
            raise ValueError(f"Normalized resource asset stem collision: {stem}")
        used_stems.add(stem)
        spec = {
            "source_name": source["entry"],
            "asset_id": "resource.asset." + stem,
            "manifest_key": "resource/assets/" + stem,
            "stem": stem,
            "group": "resource",
        }
        try:
            manifest_asset, evidence = clutter_blp_extractor.build_feature(
                package,
                shared_data,
                pack,
                spec,
                allow_wrapping_uvs=True,
                allow_optional_maps=True,
            )
            results.append(
                {
                    "source": source,
                    "status": "normalized",
                    "asset_id": spec["asset_id"],
                    "manifest_key": spec["manifest_key"],
                    "manifest_asset": manifest_asset,
                    "evidence": evidence,
                }
            )
        except (OSError, ValueError, ET.ParseError) as error:
            route = special_routes.get(source["entry"])
            if route is None:
                results.append({"source": source, "status": "unsupported", "reason": str(error)})
            else:
                results.append(
                    {
                        "source": source,
                        "status": "routed",
                        "route": route,
                        "simple_profile_rejection": str(error),
                    }
                )
    return {
        "schema": "c3x.resource_static_probe.v0",
        "source_package": str(package_path),
        "allocation_count": len(package.allocations),
        "assets": results,
        "summary": {
            "candidates": len(results),
            "normalized": sum(item["status"] == "normalized" for item in results),
            "routed": sum(item["status"] == "routed" for item in results),
            "unsupported": sum(item["status"] == "unsupported" for item in results),
        },
    }


def build_static_pack(
    inventory: dict[str, Any], assets_root: Path, pack: Path,
    presentation_path: Path = DEFAULT_PRESENTATION,
) -> dict[str, Any]:
    """Write every currently proven static body and a source-independent manifest.

    Unsupported source entries stay in the evidence report instead of silently
    acquiring a guessed interpretation. Resources retain all usable authored
    placements, so partially supported sets are immediately useful in Terrain Lab.
    """
    probe = probe_static_assets(inventory, assets_root, pack)
    by_source = {
        (item["source"]["package"], item["source"]["entry"]): item
        for item in probe["assets"]
    }
    assets = {
        item["manifest_key"]: item["manifest_asset"]
        for item in probe["assets"]
        if item["status"] == "normalized"
    }
    landmark_routes = load_landmark_routes()
    presentation_document = load_resource_presentation(presentation_path)
    routed_packs = sorted(
        {item["route"]["pack"] for item in probe["assets"] if item["status"] == "routed"}
        | {
            route["pack"]
            for resource in inventory["resources"]
            for landmark in resource.get("landmarks", [])
            if (route := landmark_routes.get(landmark["asset"]["entry"])) is not None
        }
    )
    routed_manifests = {}
    for pack_name in routed_packs:
        manifest_path = RENDERER_ROOT / "packs" / pack_name / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Missing specialized resource pack: {pack_name}")
        specialized = json.loads(manifest_path.read_text(encoding="utf-8"))
        if specialized.get("schema") != "c3x.asset_pack.v0":
            raise ValueError(f"Specialized resource pack has an invalid schema: {pack_name}")
        routed_manifests[pack_name] = specialized
    for item in probe["assets"]:
        if item["status"] != "routed":
            continue
        route = item["route"]
        if route["asset"] not in routed_manifests[route["pack"]].get("assets", {}):
            raise ValueError(
                f"Specialized resource route is absent: {route['pack']}:{route['asset']}"
            )
    skin_report_path = DEFAULT_REPORT.with_name("resource_skin_extract.json")
    skin_report = (
        json.loads(skin_report_path.read_text(encoding="utf-8"))
        if skin_report_path.is_file()
        else {"assets": []}
    )
    skins = {}
    skin_validation = {}
    for item in skin_report["assets"]:
        if item["status"] != "normalized" or not all(
            (pack / item["asset"][role]).is_file()
            for role in ("mesh", "skeleton", "material")
        ):
            continue
        skeleton = normalized_skin.load_skeleton(pack / item["asset"]["skeleton"])
        mesh = normalized_skin.load_mesh(
            pack / item["asset"]["mesh"], len(skeleton["bones"])
        )
        skins[item["resource_id"]] = item
        skin_validation[item["resource_id"]] = {
            "rest_pose": normalized_skin.validate_rest_pose(mesh, skeleton)
        }
    for resource_id, item in skins.items():
        assets[resource_id + "/landmark"] = item["asset"]
    resources: dict[str, Any] = {}
    for source_resource in inventory["resources"]:
        resource_id = source_resource["resource_id"]
        binding = presentation_document["resource_bindings"].get(resource_id, {})
        profile_name = binding.get("profile", presentation_document["default_profile"])
        presentation = {
            "profile": profile_name,
            **presentation_document["profiles"][profile_name],
            "subject_candidates": [],
        }
        primary_entries = set(binding.get("primary_source_entries", []))
        authored_entries = {
            placement["asset"]["entry"] for placement in source_resource["placements"]
        } | {
            landmark["asset"]["entry"] for landmark in source_resource.get("landmarks", [])
        }
        if unknown_entries := primary_entries - authored_entries:
            raise ValueError(
                f"Resource presentation names unknown source entries for {resource_id}: "
                + ", ".join(sorted(unknown_entries))
            )
        placements = []
        omitted = []
        for source_placement in source_resource["placements"]:
            source_asset = source_placement["asset"]
            result = by_source[(source_asset["package"], source_asset["entry"])]
            if result["status"] not in ("normalized", "routed"):
                omitted.append(
                    {"entry": source_asset["entry"], "reason": result["reason"]}
                )
                continue
            placement = {
                key: value
                for key, value in source_placement.items()
                if key not in ("asset", "clutter_set", "name")
            }
            if result["status"] == "normalized":
                placement["asset"] = result["manifest_key"]
            else:
                placement["asset"] = result["route"]["asset"]
                placement["pack"] = result["route"]["pack"]
            placements.append(placement)
            if source_asset["entry"] in primary_entries:
                presentation["subject_candidates"].append(
                    {key: placement[key] for key in ("asset", "pack") if key in placement}
                )
        landmark_asset = resource_id + "/landmark" if resource_id in skins else None
        landmark_route = None
        for landmark in source_resource.get("landmarks", []):
            source_entry = landmark["asset"]["entry"]
            if landmark_asset is None and source_entry in landmark_routes:
                landmark_route = landmark_routes[source_entry]
            if source_entry in primary_entries:
                if landmark_asset is not None:
                    presentation["subject_candidates"].append({"asset": landmark_asset})
                elif source_entry in landmark_routes:
                    presentation["subject_candidates"].append(landmark_routes[source_entry])
        presentation["subject_candidates"] = [
            dict(candidate)
            for candidate in {
                tuple(sorted(candidate.items())) for candidate in presentation["subject_candidates"]
            }
        ]
        presentation["subject_candidates"].sort(
            key=lambda candidate: (candidate.get("pack", ""), candidate["asset"])
        )
        if presentation["composition"] == "single_primary_subject" and not presentation["subject_candidates"]:
            raise ValueError(f"Single-subject resource has no normalized candidate: {resource_id}")
        resources[resource_id] = {
            "placements": placements,
            "source_placement_count": len(source_resource["placements"]),
            "omitted_source_entries": omitted,
            "landmark_asset": landmark_asset,
            "landmark_route": landmark_route,
            "presentation": presentation,
        }

    animations = {}
    for resource_id, relative in (
        ("resource/fish", "animations/fish_ambient.c3anim"),
        ("resource/whales", "animations/whale_idle.c3anim"),
    ):
        clip = pack / relative
        if clip.is_file():
            if resource_id in skins:
                skin_asset = skins[resource_id]["asset"]
                skeleton = normalized_skin.load_skeleton(pack / skin_asset["skeleton"])
                mesh = normalized_skin.load_mesh(
                    pack / skin_asset["mesh"], len(skeleton["bones"])
                )
                loaded_clip = normalized_animation.load_clip(clip)
                skin_validation[resource_id]["animation_binding"] = normalized_skin.bind_clip(
                    mesh, skeleton, loaded_clip, 0
                )
            pose_relative = relative.removesuffix(".c3anim") + ".c3pose"
            pose_path = pack / pose_relative
            pose_validation = None
            if pose_path.is_file() and resource_id in skins:
                pose_cache = normalized_pose_cache.load_pose_cache(pose_path)
                pose_validation = normalized_pose_cache.validate_skeleton_binding(
                    pose_cache, skeleton
                )
                skin_validation[resource_id]["pose_cache_binding"] = pose_validation
            animations[resource_id] = {
                "clip": relative,
                "group_index": 0,
                "loop": True,
                "sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
                "binding_status": (
                    "normalized_skin" if resource_id in skins else "awaiting_normalized_skin"
                ),
                "pose_status": (
                    "validated_model_aware_pose_cache"
                    if pose_validation is not None
                    else (
                        "model_aware_sampling_required"
                        if resource_id == "resource/fish"
                        else "validated_cpu_skin"
                    )
                ),
            }
            if pose_validation is not None:
                animations[resource_id]["pose_cache"] = pose_relative
                animations[resource_id]["pose_cache_sha256"] = hashlib.sha256(
                    pose_path.read_bytes()
                ).hexdigest()

    manifest = {
        "schema": "c3x.resource_pack.v0",
        "pack_id": "local.resources.normalized",
        "assets": assets,
        "resources": resources,
        "animations": animations,
        "pack_dependencies": routed_packs,
        "runtime_source_dependency": None,
    }
    write_json(pack / "manifest.json", manifest)
    coverage = {
        resource_id: {
            "normalized_placements": len(value["placements"]),
            "source_placements": value["source_placement_count"],
            "complete": (
                value["source_placement_count"] > 0
                and len(value["placements"]) == value["source_placement_count"]
            ) or value["landmark_asset"] is not None or value["landmark_route"] is not None,
        }
        for resource_id, value in resources.items()
    }
    return {
        **probe,
        "schema": "c3x.resource_pack_build.v0",
        "pack": str(pack),
        "manifest": str(pack / "manifest.json"),
        "coverage": coverage,
        "skin_validation": skin_validation,
        "summary": {
            **probe["summary"],
            "manifest_assets": len(assets),
            "resources": len(resources),
            "complete_static_resources": sum(item["complete"] for item in coverage.values()),
            "animation_clips": len(animations),
            "pose_caches": sum("pose_cache" in item for item in animations.values()),
            "routed_assets": probe["summary"].get("routed", 0),
            "skinned_landmarks": len(skins),
            "single_subject_resources": sum(
                value["presentation"]["composition"] == "single_primary_subject"
                for value in resources.values()
            ),
        },
    }


def extract_landmark_animations(
    inventory: dict[str, Any], assets_root: Path, output: Path
) -> dict[str, Any]:
    package_profiles = {
        "environment/clutter": 12.0,
        "landmarks/tilebases": 100.0,
    }
    uses: dict[tuple[str, str], set[str]] = defaultdict(set)
    for resource in inventory["resources"]:
        for placement in resource["placements"]:
            asset = placement["asset"]
            if asset["package"] in package_profiles:
                uses[(asset["package"], asset["entry"])].add(resource["resource_id"])
        for landmark in resource["landmarks"]:
            asset = landmark["asset"]
            if asset["package"] in package_profiles:
                uses[(asset["package"], asset["entry"])].add(resource["resource_id"])
    if not uses:
        raise ValueError("Resource inventory contains no animation-capable package assets")

    packages = []
    unique_clips: dict[tuple[str, int], dict[str, Any]] = {}
    for logical_package, source_units_per_tile in package_profiles.items():
        entries = sorted(entry for package_name, entry in uses if package_name == logical_package)
        if not entries:
            continue
        package_path = (
            assets_root / "Base" / "Platforms" / "Windows" / "BLPs" / (logical_package + ".blp")
        )
        package_bytes = package_path.read_bytes()
        initial_entry = next(
            (
                entry
                for entry in entries
                if package_bytes.count(entry.encode("ascii") + b"\0") == 1
            ),
            None,
        )
        if initial_entry is None:
            raise ValueError(f"{logical_package} has no unique resource entry for package indexing")
        package = IndexedStaticPackage(package_path, initial_entry)
        animation_arrays = [
            pointer
            for pointer in range(1, len(package.allocations) + 1)
            if package.type_name(pointer) == "BLP::AnimationEntry"
        ]
        if len(animation_arrays) != 1:
            raise ValueError(
                f"Expected one BLP animation table in {logical_package}, found {len(animation_arrays)}"
            )
        animation_array = animation_arrays[0]
        if len(package.bytes_for(animation_array)) % 64:
            raise ValueError(f"{logical_package} animation table does not use 64-byte records")
        asset_records = []
        package_stem = _slug(logical_package)
        for entry in entries:
            try:
                package.select_direct_string(entry)
                _landmark, _user_data, base_model = clutter_blp_extractor.landmark_base_model(package)
            except ValueError as exc:
                asset_records.append(
                    {
                        "source_entry": entry,
                        "resource_ids": sorted(uses[(logical_package, entry)]),
                        "clips": [],
                        "binding_status": "source_name_ambiguous",
                        "reason": str(exc),
                    }
                )
                continue
            descriptors = package.pointer_fields(
                base_model, "FGXModelFramework::BehaviorDesc::AnimationDesc"
            )
            if len(descriptors) > 1:
                raise ValueError(
                    f"Expected at most one animation descriptor for {entry}, found {len(descriptors)}"
                )
            indices: list[int] = []
            if descriptors:
                raw_descriptors = package.bytes_for(descriptors[0][1])
                if len(raw_descriptors) % 8:
                    raise ValueError(f"Animation descriptors for {entry} do not use 8-byte records")
                indices = sorted(
                    {
                        struct.unpack_from("<I", raw_descriptors, offset + 4)[0]
                        for offset in range(0, len(raw_descriptors), 8)
                    }
                )
            clips = []
            for index in indices:
                key = (logical_package, index)
                if key not in unique_clips:
                    record = package.array_element(animation_array, index)
                    name = package.string_value(struct.unpack_from("<Q", record, 8)[0])
                    if not name:
                        raise ValueError(f"Animation entry {index} in {logical_package} has no name")
                    data_offset = struct.unpack_from("<Q", record, 0x20)[0]
                    data_bytes = struct.unpack_from("<Q", record, 0x28)[0]
                    name_hash = struct.unpack_from("<I", record, 0x30)[0]
                    if name_hash != fnv1a32(name):
                        raise ValueError(f"Animation entry hash mismatch: {name}")
                    payload = package.big_data(data_offset, data_bytes)
                    if payload[:16].hex() != "e59b495e6f631f141e13eba990beedc4":
                        raise ValueError(
                            f"Animation entry does not start with supported Granny magic: {name}"
                        )
                    relative = Path(package_stem) / f"{index:04d}_{_slug(name)}.fgx"
                    target = output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                    unique_clips[key] = {
                        "source_package": logical_package,
                        "name": name,
                        "table_index": index,
                        "raw_fgx": relative.as_posix(),
                        "normalized_clip": (
                            Path("animations/resources") / relative.with_suffix(".c3anim")
                        ).as_posix(),
                        "translation_scale": 1.0 / source_units_per_tile,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "fnv1a32": f"0x{name_hash:08x}",
                    }
                clips.append(unique_clips[key])
            asset_records.append(
                {
                    "source_entry": entry,
                    "resource_ids": sorted(uses[(logical_package, entry)]),
                    "clips": [
                        {
                            "name": clip["name"],
                            "table_index": clip["table_index"],
                            "raw_fgx": clip["raw_fgx"],
                            "normalized_clip": clip["normalized_clip"],
                        }
                        for clip in clips
                    ],
                    "binding_status": (
                        "raw_clips_extracted_body_profile_pending"
                        if clips and logical_package == "environment/clutter"
                        else "raw_clips_extracted"
                        if clips
                        else "static"
                    ),
                }
            )
        packages.append(
            {
                "source_package": logical_package,
                "source_units_per_tile": source_units_per_tile,
                "animation_table_entries": package.allocations[animation_array - 1]["element_count"],
                "assets": asset_records,
            }
        )
    clips = sorted(unique_clips.values(), key=lambda item: (item["source_package"], item["table_index"]))
    return {
        "schema": "c3x.resource_animation_extract.v0",
        "packages": packages,
        "unique_clips": clips,
        "summary": {
            "assets": sum(len(package["assets"]) for package in packages),
            "animated_assets": sum(
                bool(asset["clips"]) for package in packages for asset in package["assets"]
            ),
            "unique_clips": len(clips),
            "bytes": sum(clip["bytes"] for clip in clips),
            "resources_with_animation": len(
                {
                    resource_id
                    for package in packages
                    for asset in package["assets"]
                    if asset["clips"]
                    for resource_id in asset["resource_ids"]
                }
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--presentation", type=Path, default=DEFAULT_PRESENTATION)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--probe-static", action="store_true")
    parser.add_argument("--probe-pack", type=Path, default=DEFAULT_PROBE_PACK)
    parser.add_argument("--build-static-pack", action="store_true")
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--extract-landmark-animations", action="store_true")
    parser.add_argument("--animation-stage", type=Path, default=DEFAULT_ANIMATION_STAGE)
    parser.add_argument("--extract-landmark-skins", action="store_true")
    args = parser.parse_args(argv)
    try:
        inventory = build_inventory(args.mapping, args.assets_root)
        write_json(args.report, inventory)
        if args.probe_static:
            probe = probe_static_assets(inventory, args.assets_root, args.probe_pack)
            write_json(args.report.with_name("resource_static_probe.json"), probe)
        if args.build_static_pack:
            build = build_static_pack(
                inventory, args.assets_root, args.pack, args.presentation
            )
            write_json(args.report.with_name("resource_pack_build.json"), build)
        if args.extract_landmark_animations:
            animations = extract_landmark_animations(
                inventory, args.assets_root, args.animation_stage
            )
            write_json(args.report.with_name("resource_animation_extract.json"), animations)
        if args.extract_landmark_skins:
            skins = resource_skin_extractor.extract_skins(
                inventory, args.assets_root, args.pack
            )
            write_json(args.report.with_name("resource_skin_extract.json"), skins)
    except (OSError, ValueError, ET.ParseError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1
    print(json.dumps(inventory["summary"], indent=2, sort_keys=True))
    if args.probe_static:
        print(json.dumps(probe["summary"], indent=2, sort_keys=True))
    if args.build_static_pack:
        print(json.dumps(build["summary"], indent=2, sort_keys=True))
    if args.extract_landmark_animations:
        print(json.dumps(animations["summary"], indent=2, sort_keys=True))
    if args.extract_landmark_skins:
        print(json.dumps(skins["summary"], indent=2, sort_keys=True))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
