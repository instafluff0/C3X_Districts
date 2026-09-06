#!/usr/bin/env python3
"""Build a deterministic census of Civ III map-visible art and coverage gaps."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import posixpath
import struct
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "c3x.civ3_art_inventory.v0"
LAYER_ORDER = ("base", "ptw", "conquests", "scenario")
UNIT_ANIMATION_SECTION = "Animations"
INVENTORY_ROOT = Path(__file__).resolve().parent
DEFAULT_ATLAS_CONTRACTS = INVENTORY_ROOT / "vanilla_atlas_layouts.json"
DEFAULT_BIQ_SEMANTICS = INVENTORY_ROOT / "vanilla_conquests_biq_semantics.json"
FLC_DIRECTION_NAMES = ("southwest", "south", "southeast", "east", "northeast", "north", "northwest", "west")

# These are rendering responsibilities, including layers C3X intentionally leaves to Civ III.
RENDER_LAYERS: tuple[dict[str, str], ...] = (
    {"id": "terrain_base", "scope": "map", "default_ownership": "replace_candidate"},
    {"id": "terrain_transition", "scope": "map", "default_ownership": "replace_candidate"},
    {"id": "terrain_feature", "scope": "map", "default_ownership": "replace_candidate"},
    {"id": "water", "scope": "map", "default_ownership": "replace_candidate"},
    {"id": "river", "scope": "map", "default_ownership": "replace_candidate"},
    {"id": "infrastructure", "scope": "map", "default_ownership": "replace_candidate"},
    {"id": "resource", "scope": "map", "default_ownership": "replace_candidate"},
    {"id": "city", "scope": "map", "default_ownership": "replace_candidate"},
    {"id": "unit", "scope": "map", "default_ownership": "replace_candidate"},
    {"id": "effect", "scope": "map", "default_ownership": "replace_candidate"},
    {"id": "fog_of_war", "scope": "map", "default_ownership": "retained_civ3"},
    {"id": "territory_border", "scope": "map", "default_ownership": "retained_civ3"},
    {"id": "map_grid", "scope": "map", "default_ownership": "retained_civ3"},
    {"id": "selection_highlight", "scope": "map", "default_ownership": "retained_civ3"},
    {"id": "movement_path", "scope": "map", "default_ownership": "retained_civ3"},
    {"id": "city_label", "scope": "map", "default_ownership": "retained_civ3"},
    {"id": "unit_status", "scope": "map", "default_ownership": "retained_civ3"},
    {"id": "map_cursor", "scope": "map", "default_ownership": "retained_civ3"},
    {"id": "editor_marker", "scope": "editor", "default_ownership": "out_of_runtime_scope"},
    {"id": "minimap", "scope": "hud", "default_ownership": "retained_civ3"},
    {"id": "hud", "scope": "hud", "default_ownership": "retained_civ3"},
)

TERRAIN_FILE_LAYERS = {
    "craters.pcx": "infrastructure",
    "deltarivers.pcx": "river",
    "editfog.pcx": "editor_marker",
    "floodplains.pcx": "terrain_feature",
    "fogofwar.pcx": "fog_of_war",
    "goodyhuts.pcx": "infrastructure",
    "grassland forests.pcx": "terrain_feature",
    "hill forests.pcx": "terrain_feature",
    "hill jungle.pcx": "terrain_feature",
    "irrigation desett.pcx": "infrastructure",
    "irrigation plains.pcx": "infrastructure",
    "irrigation tundra.pcx": "infrastructure",
    "irrigation.pcx": "infrastructure",
    "landmark_terrain.pcx": "terrain_transition",
    "lmforests.pcx": "terrain_feature",
    "lmhills.pcx": "terrain_feature",
    "lmmountains.pcx": "terrain_feature",
    "marsh.pcx": "terrain_feature",
    "mountain forests.pcx": "terrain_feature",
    "mountain jungles.pcx": "terrain_feature",
    "mountains-snow.pcx": "terrain_feature",
    "mountains.pcx": "terrain_feature",
    "mtnrivers.pcx": "river",
    "plains forests.pcx": "terrain_feature",
    "polaricecaps-final.pcx": "water",
    "pollution.pcx": "infrastructure",
    "railroads.pcx": "infrastructure",
    "roads.pcx": "infrastructure",
    "startloc.pcx": "editor_marker",
    "terrainbuildings.pcx": "infrastructure",
    "territory.pcx": "territory_border",
    "tnt.pcx": "infrastructure",
    "tundra forests.pcx": "terrain_feature",
    "volcanos forests.pcx": "terrain_feature",
    "volcanos jungles.pcx": "terrain_feature",
    "volcanos-snow.pcx": "terrain_feature",
    "volcanos.pcx": "terrain_feature",
    "waterfalls.pcx": "river",
    "x_airfields and detect_shadow.pcx": "infrastructure",
    "x_airfields and detect.pcx": "infrastructure",
    "x_victory.pcx": "infrastructure",
    "xhills.pcx": "terrain_feature",
}


def normalized_key(path: Path) -> str:
    return path.as_posix().casefold()


def classify_terrain_file(name: str) -> str:
    folded = name.casefold()
    explicit = TERRAIN_FILE_LAYERS.get(folded)
    if explicit:
        return explicit
    stem = Path(folded).stem
    if stem.startswith(("lw", "w")) and stem.endswith(("cso", "ooo", "sss")):
        return "water"
    if stem.startswith(("lx", "x")) and len(stem) == (5 if stem.startswith("lx") else 4):
        return "terrain_transition"
    return "unclassified"


def pcx_metadata(path: Path) -> dict[str, int]:
    data = path.read_bytes()[:128]
    if len(data) < 128 or data[0] != 0x0A:
        raise ValueError(f"Not a valid PCX header: {path}")
    xmin, ymin, xmax, ymax = struct.unpack_from("<4H", data, 4)
    if xmax < xmin or ymax < ymin:
        raise ValueError(f"Invalid PCX bounds: {path}")
    return {
        "width": xmax - xmin + 1,
        "height": ymax - ymin + 1,
        "bits_per_plane": data[3],
        "planes": data[65],
    }


def flc_metadata(path: Path) -> dict[str, Any]:
    data = path.read_bytes()[:128]
    if len(data) < 128:
        raise ValueError(f"FLC header is too short: {path}")
    size, magic, frames, width, height, depth = struct.unpack_from("<I5H", data, 0)
    if magic not in {0xAF11, 0xAF12}:
        raise ValueError(f"Unsupported FLI/FLC magic 0x{magic:04X}: {path}")
    num_anims = struct.unpack_from("<H", data, 96)[0]
    anim_length = struct.unpack_from("<H", data, 98)[0]
    return {
        "declared_bytes": size,
        "frames": frames,
        "width": width,
        "height": height,
        "depth": depth,
        "direction_count": num_anims,
        "frames_per_direction": anim_length,
        "direction_order": list(FLC_DIRECTION_NAMES[:num_anims]) if num_anims <= 8 else [],
        "shadow_encoding": "palette_indices_240_254",
        "smoke_encoding": "palette_indices_224_239",
        "transparency_index": 255,
    }


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def contract_by_basename(catalog: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    if catalog is None:
        return by_name
    for contract in catalog.get("contracts", []):
        for basename in contract.get("basenames", []):
            folded = str(basename).casefold()
            if folded in by_name:
                raise ValueError(f"Duplicate atlas contract for {basename}")
            by_name[folded] = contract
    return by_name


def expand_contract_cells(contract: dict[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    next_index = 0
    for group in contract.get("groups", []):
        origin_x, origin_y = group["origin"]
        cell_w, cell_h = group["cell"]
        stride_x, stride_y = group.get("stride", group["cell"])
        runtime_indices = group.get("runtime_indices")
        group_offset = 0
        for row in range(group["rows"]):
            for column in range(group["columns"]):
                runtime_index: int | str = next_index
                if runtime_indices is not None:
                    runtime_index = runtime_indices[group_offset]
                cells.append(
                    {
                        "group": group["name"],
                        "source_index": next_index,
                        "runtime_index": runtime_index,
                        "rect": [
                            origin_x + column * stride_x,
                            origin_y + row * stride_y,
                            cell_w,
                            cell_h,
                        ],
                    }
                )
                next_index += 1
                group_offset += 1
    return cells


def apply_atlas_contract(
    record: dict[str, Any],
    by_basename: dict[str, dict[str, Any]],
    reachable_resource_icons: set[int],
) -> None:
    atlas = record.get("atlas")
    if atlas is None:
        return
    contract = by_basename.get(Path(record["asset_path"]).name.casefold())
    if contract is None:
        return
    cells = expand_contract_cells(contract)
    image = record.get("image", {})
    bounds_errors = [
        cell["source_index"]
        for cell in cells
        if cell["rect"][0] < 0
        or cell["rect"][1] < 0
        or cell["rect"][0] + cell["rect"][2] > image.get("width", 0)
        or cell["rect"][1] + cell["rect"][3] > image.get("height", 0)
    ]
    authored_capacity = int(contract["authored_capacity"])
    capacity_error = authored_capacity != len(cells)
    reachable = contract.get("reachable", {})
    if reachable.get("mode") == "biq_good_icon":
        reachable_indices = sorted(reachable_resource_icons)
    elif reachable.get("mode") == "indices":
        reachable_indices = sorted(int(value) for value in reachable.get("indices", []))
    else:
        reachable_indices = sorted(int(cell["runtime_index"]) for cell in cells)
    reachable_set = set(reachable_indices)
    for cell in cells:
        cell["classification"] = (
            "vanilla_fallback" if int(cell["runtime_index"]) in reachable_set else "unreachable"
        )
    atlas.update(
        {
            "status": "resolved" if not bounds_errors and not capacity_error else "invalid_contract",
            "contract_id": contract["id"],
            "authored_capacity": authored_capacity,
            "reachable_indices": reachable_indices,
            "unreachable_indices": sorted(
                int(cell["runtime_index"]) for cell in cells if cell["classification"] == "unreachable"
            ),
            "cells": cells,
            "bounds_errors": bounds_errors,
            "capacity_error": capacity_error,
            "evidence": [
                "Civ3Conquests Map_Renderer/load-image slice rectangles",
                "Conquests BIQ selector values" if reachable.get("mode") == "biq_good_icon" else "Civ3 runtime array capacity",
            ],
        }
    )


def root_specs(install_root: Path, scenario_art_roots: Iterable[Path] = ()) -> list[dict[str, Any]]:
    specs = [
        {"id": "base", "priority": 0, "path": install_root / "Art"},
        {"id": "ptw", "priority": 1, "path": install_root / "Civ3PTW" / "Art"},
        {"id": "conquests", "priority": 2, "path": install_root / "Conquests" / "Art"},
    ]
    for index, path in enumerate(scenario_art_roots):
        art_path = path if path.name.casefold() == "art" else path / "Art"
        specs.append({"id": f"scenario:{index}", "priority": 3 + index, "path": art_path})
    return specs


def relative_asset_path(path: Path, art_root: Path) -> str:
    return "Art/" + path.relative_to(art_root).as_posix()


def file_record(path: Path, art_root: Path, root_id: str, priority: int, render_layer: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "asset_path": relative_asset_path(path, art_root),
        "source_layer": root_id,
        "source_priority": priority,
        "render_layer": render_layer,
        "extension": path.suffix.casefold(),
        "bytes": path.stat().st_size,
    }
    try:
        if path.suffix.casefold() == ".pcx":
            record["image"] = pcx_metadata(path)
            record["atlas"] = {
                "status": "requires_layout_contract",
                "cell_count": None,
                "evidence": [],
            }
        elif path.suffix.casefold() in {".flc", ".fli"}:
            record["animation"] = flc_metadata(path)
    except (OSError, ValueError) as exc:
        record["metadata_error"] = str(exc)
    return record


def read_unit_animations(unit_dir: Path) -> tuple[str | None, dict[str, str], list[str]]:
    ini_files = sorted(unit_dir.glob("*.ini"), key=lambda item: item.name.casefold())
    preferred = next((path for path in ini_files if path.stem.casefold() == unit_dir.name.casefold()), None)
    ini_path = preferred or (ini_files[0] if ini_files else None)
    if ini_path is None:
        return None, {}, []
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    warnings: list[str] = []
    try:
        parser.read(ini_path, encoding="latin-1")
    except configparser.Error as exc:
        return ini_path.name, {}, [str(exc)]
    animations: dict[str, str] = {}
    section = next((name for name in parser.sections() if name.casefold() == UNIT_ANIMATION_SECTION.casefold()), None)
    if section:
        for key, value in parser.items(section):
            value = value.strip()
            if value:
                animations[key.strip().upper()] = value.replace("\\", "/")
    else:
        warnings.append("missing Animations section")
    return ini_path.name, dict(sorted(animations.items())), warnings


def discover_file_assets(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in specs:
        art_root = spec["path"]
        if not art_root.is_dir():
            continue
        terrain_root = art_root / "Terrain"
        for path in sorted(terrain_root.glob("*.pcx"), key=lambda item: item.name.casefold()):
            records.append(file_record(path, art_root, spec["id"], spec["priority"], classify_terrain_file(path.name)))
        cities_root = art_root / "Cities"
        for path in sorted(cities_root.glob("*.pcx"), key=lambda item: item.name.casefold()):
            records.append(file_record(path, art_root, spec["id"], spec["priority"], "city"))
        for path in sorted(art_root.glob("resources*.pcx"), key=lambda item: item.name.casefold()):
            records.append(file_record(path, art_root, spec["id"], spec["priority"], "resource"))
        animations_root = art_root / "Animations"
        for path in sorted(animations_root.rglob("*"), key=lambda item: normalized_key(item)):
            if path.is_file() and path.suffix.casefold() in {".flc", ".fli", ".ini"}:
                records.append(file_record(path, art_root, spec["id"], spec["priority"], "effect"))
    return records


def discover_units(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for spec in specs:
        art_root = spec["path"]
        units_root = art_root / "Units"
        if not units_root.is_dir():
            continue
        for unit_dir in sorted((path for path in units_root.iterdir() if path.is_dir()), key=lambda item: item.name.casefold()):
            ini_name, animations, warnings = read_unit_animations(unit_dir)
            # Units/Palettes and the shipped Units/Test support folder are not
            # selectable units and intentionally have no INI.
            if ini_name is None:
                continue
            animation_records = []
            for action, relative in animations.items():
                animation_record: dict[str, Any] = {
                    "action": action,
                    "path": relative,
                }
                if relative.strip() == ".":
                    animation_record["resolution_status"] = "disabled"
                    animation_records.append(animation_record)
                    continue
                logical_path = posixpath.normpath(f"Units/{unit_dir.name}/{relative}")
                resolved = None
                for candidate_spec in reversed(specs):
                    candidate = candidate_spec["path"].joinpath(*logical_path.split("/"))
                    if candidate.is_file():
                        resolved = (candidate_spec, candidate)
                        break
                animation_record["resolution_status"] = "resolved" if resolved else "missing"
                animation_record["resolved_asset_path"] = "Art/" + logical_path
                animation_record["resolved_source_layer"] = resolved[0]["id"] if resolved else None
                if resolved and resolved[1].suffix.casefold() in {".flc", ".fli"}:
                    try:
                        animation_record.update(flc_metadata(resolved[1]))
                    except (OSError, ValueError) as exc:
                        animation_record["metadata_error"] = str(exc)
                animation_records.append(animation_record)
            units.append(
                {
                    "name": unit_dir.name,
                    "source_layer": spec["id"],
                    "source_priority": spec["priority"],
                    "asset_path": relative_asset_path(unit_dir, art_root),
                    "ini": ini_name,
                    "animations": animation_records,
                    "warnings": warnings,
                    "semantic_binding": {
                        "status": "requires_biq_or_runtime_evidence",
                        "unit_type_id": None,
                    },
                }
            )
    return units


def mark_effective(records: list[dict[str, Any]], identity_fields: tuple[str, ...]) -> None:
    winners: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in records:
        key = tuple(str(record[field]).casefold() for field in identity_fields)
        winner = winners.get(key)
        if winner is None or record["source_priority"] > winner["source_priority"]:
            winners[key] = record
    for record in records:
        key = tuple(str(record[field]).casefold() for field in identity_fields)
        record["effective"] = record is winners[key]


def build_inventory(
    install_root: Path,
    scenario_art_roots: Iterable[Path] = (),
    *,
    atlas_contracts: dict[str, Any] | None = None,
    biq_semantics: dict[str, Any] | None = None,
    runtime_census: dict[str, Any] | None = None,
) -> dict[str, Any]:
    install_root = install_root.resolve()
    specs = root_specs(install_root, scenario_art_roots)
    files = discover_file_assets(specs)
    units = discover_units(specs)
    mark_effective(files, ("asset_path",))
    mark_effective(units, ("name",))

    effective_files = [record for record in files if record["effective"]]
    effective_units = [record for record in units if record["effective"]]
    for record in files:
        if not record["effective"]:
            record["classification"] = "unreachable"
            record["classification_evidence"] = "shadowed by a higher-precedence art root"
        elif record["render_layer"] == "editor_marker":
            record["classification"] = "not_map_rendered"
            record["classification_evidence"] = "editor-only marker"
        else:
            record["classification"] = "vanilla_fallback"
            record["classification_evidence"] = "effective Civ III asset retained until its M7 category replacement"
    for record in units:
        if not record["effective"]:
            record["classification"] = "unreachable"
            record["classification_evidence"] = "shadowed by a higher-precedence unit directory"
    reachable_resource_icons = {
        int(record["icon_index"])
        for record in (biq_semantics or {}).get("resources", [])
    }
    contracts = contract_by_basename(atlas_contracts)
    for record in effective_files:
        apply_atlas_contract(record, contracts, reachable_resource_icons)

    folder_to_unit_types: dict[str, set[str]] = {}
    for unit_type in (biq_semantics or {}).get("unit_types", []):
        for variant in unit_type.get("art_variants", []):
            folder_to_unit_types.setdefault(str(variant["art_folder"]).casefold(), set()).add(
                str(unit_type["civilopedia_entry"])
            )
    effective_unit_names = {record["name"].casefold() for record in effective_units}
    for record in effective_units:
        unit_type_ids = sorted(folder_to_unit_types.get(record["name"].casefold(), set()), key=str.casefold)
        if unit_type_ids:
            record["classification"] = "vanilla_fallback"
            record["classification_evidence"] = "reachable through a vanilla Conquests unit selector"
            record["semantic_binding"] = {
                "status": "resolved",
                "unit_type_ids": unit_type_ids,
                "classification": "vanilla_fallback",
                "evidence": ["Conquests BIQ PRTO.civilopediaEntry", "layered PediaIcons ANIMNAME binding"],
            }
        elif biq_semantics is not None:
            record["classification"] = "unreachable"
            record["classification_evidence"] = "not referenced by the vanilla Conquests selector catalog"
            record["semantic_binding"] = {
                "status": "unreachable",
                "unit_type_ids": [],
                "classification": "unreachable",
                "evidence": ["not referenced by any effective vanilla Conquests PediaIcons ANIMNAME selector"],
            }

    semantic_unit_gaps = []
    effective_unit_by_name = {record["name"].casefold(): record for record in effective_units}
    for unit_type in (biq_semantics or {}).get("unit_types", []):
        variants = unit_type.get("art_variants", [])
        missing_folders = sorted(
            {
                str(variant["art_folder"])
                for variant in variants
                if str(variant["art_folder"]).casefold() not in effective_unit_names
            },
            key=str.casefold,
        )
        if not variants or missing_folders:
            semantic_unit_gaps.append(
                {
                    "civilopedia_entry": unit_type.get("civilopedia_entry"),
                    "reason": "no_animation_binding" if not variants else "missing_art_folders",
                    "missing_art_folders": missing_folders,
                }
            )
        correlations = []
        runtime_states: set[str] = set()
        for variant in variants:
            art_record = effective_unit_by_name.get(str(variant["art_folder"]).casefold())
            actions = []
            if art_record is not None:
                for animation in art_record["animations"]:
                    runtime_states.add(animation["action"])
                    actions.append(
                        {
                            "action": animation["action"],
                            "path": animation["path"],
                            "resolution_status": animation["resolution_status"],
                            "direction_count": animation.get("direction_count"),
                            "frames_per_direction": animation.get("frames_per_direction"),
                            "direction_order": animation.get("direction_order", []),
                            "shadow_encoding": animation.get("shadow_encoding"),
                            "projectile_or_effect": (
                                "animation_or_civ3_effect_path"
                                if animation["action"].startswith(("ATTACK", "BOMB", "DEATH", "VICTORY"))
                                else "not_applicable"
                            ),
                        }
                    )
            correlations.append(
                {
                    "selector_key": variant["key"],
                    "art_folder": variant["art_folder"],
                    "source_layer": variant["source_layer"],
                    "status": "resolved" if art_record is not None else "missing",
                    "classification": "vanilla_fallback" if art_record is not None else "unresolved",
                    "ini": art_record["ini"] if art_record is not None else None,
                    "actions": actions,
                }
            )
        unit_type["classification"] = "vanilla_fallback" if correlations and not missing_folders else "unresolved"
        unit_type["art_correlation"] = correlations
        unit_type["runtime_states"] = sorted(runtime_states)
        unit_type["direction_contract"] = {
            "order": list(FLC_DIRECTION_NAMES),
            "exception_policy": "honor FlicAnimHeader num_anims; missiles, nuclear effects, and paradrop/build clips may be one- or two-direction",
        }

    if biq_semantics is not None:
        for terrain in biq_semantics.get("terrain_types", []):
            terrain["classification"] = "vanilla_fallback"
            terrain["evidence"] = "BIQ TERR index selects a classified terrain/feature family"
        for resource in biq_semantics.get("resources", []):
            resource["classification"] = "vanilla_fallback"
            resource["evidence"] = "BIQ GOOD.icon resolves to resources.pcx and resources_shadows.pcx"
        for civilization in biq_semantics.get("civilizations", []):
            civilization["classification"] = "not_map_rendered"
            civilization["city_art_classification"] = "vanilla_fallback"
            civilization["evidence"] = "RACE culture_group is selector metadata for retained city art"
        for transformation in biq_semantics.get("terrain_transformations", []):
            transformation["classification"] = "vanilla_fallback"
            transformation["evidence"] = "worker order changes a classified tile infrastructure/terrain state"
        if isinstance(biq_semantics.get("city_selectors"), dict):
            biq_semantics["city_selectors"]["classification"] = "vanilla_fallback"
            biq_semantics["city_selectors"]["evidence"] = "Civ III city loader indexes culture, era, size, wall, and status slices"

    ownership_decisions = []
    for layer in RENDER_LAYERS:
        default = layer["default_ownership"]
        ownership_decisions.append(
            {
                **layer,
                "classification": "not_map_rendered" if default == "out_of_runtime_scope" else "vanilla_fallback",
                "evidence": (
                    "editor-only source is excluded from game runtime"
                    if default == "out_of_runtime_scope"
                    else "M5 proven map insertion boundary retains Civ III compositing for this layer"
                    if default == "retained_civ3"
                    else "M6 inventory baseline; replacement advances category-by-category in M7"
                ),
            }
        )

    unclassified = [record["asset_path"] for record in effective_files if record["render_layer"] == "unclassified"]
    unresolved_atlases = [
        record["asset_path"] for record in effective_files
        if record.get("atlas", {}).get("status") != "resolved" and record.get("atlas")
    ]
    unresolved_units = [
        record["name"]
        for record in effective_units
        if record["semantic_binding"]["status"] not in {"resolved", "unreachable"}
    ]
    missing_animation_files = [
        f"{record['name']}:{animation['action']}={animation['path']}"
        for record in effective_units
        for animation in record["animations"]
        if animation["resolution_status"] == "missing"
    ]
    root_records = [
        {
            "id": spec["id"],
            "priority": spec["priority"],
            "relative_location": (
                spec["path"].relative_to(install_root).as_posix()
                if spec["path"].is_relative_to(install_root)
                else "<external-scenario-art-root>"
            ),
            "present": spec["path"].is_dir(),
        }
        for spec in specs
    ]
    return {
        "schema": SCHEMA,
        "target": "vanilla_civilization_iii_complete_conquests_map_rendering",
        "districts_in_scope": False,
        "source_roots": root_records,
        "render_layers": list(RENDER_LAYERS),
        "ownership_decisions": ownership_decisions,
        "file_assets": sorted(files, key=lambda item: (normalized_key(Path(item["asset_path"])), item["source_priority"])),
        "units": sorted(units, key=lambda item: (item["name"].casefold(), item["source_priority"])),
        "semantic_inventory": biq_semantics,
        "runtime_selector_census": runtime_census,
        "summary": {
            "effective_file_assets": len(effective_files),
            "effective_unit_directories": len(effective_units),
            "effective_files_by_render_layer": dict(sorted(Counter(record["render_layer"] for record in effective_files).items())),
            "unclassified_effective_files": len(unclassified),
            "unresolved_atlases": len(unresolved_atlases),
            "unresolved_unit_bindings": len(unresolved_units),
            "missing_unit_animation_files_in_selected_layer": len(missing_animation_files),
            "biq_primary_unit_types": len((biq_semantics or {}).get("unit_types", [])),
            "biq_resources": len((biq_semantics or {}).get("resources", [])),
            "biq_terrain_types": len((biq_semantics or {}).get("terrain_types", [])),
            "semantic_unit_gaps": len(semantic_unit_gaps),
            "runtime_unknown_selectors": len((runtime_census or {}).get("unknown_selectors", [])),
        },
        "completeness": {
            "status": (
                "incomplete"
                if unclassified
                or unresolved_atlases
                or unresolved_units
                or missing_animation_files
                or semantic_unit_gaps
                or biq_semantics is None
                or runtime_census is None
                or (runtime_census or {}).get("unknown_selectors")
                else "complete"
            ),
            "unclassified_effective_files": sorted(unclassified, key=str.casefold),
            "unresolved_atlases": sorted(unresolved_atlases, key=str.casefold),
            "unresolved_unit_bindings": sorted(unresolved_units, key=str.casefold),
            "missing_unit_animation_files_in_selected_layer": sorted(missing_animation_files, key=str.casefold),
            "semantic_unit_gaps": semantic_unit_gaps,
            "runtime_unknown_selectors": (runtime_census or {}).get("unknown_selectors", []),
            "required_future_evidence": [
                *([] if biq_semantics is not None else ["BIQ semantic records"]),
                *([] if atlas_contracts is not None else ["atlas layout contracts"]),
                *([] if runtime_census is not None else ["runtime selector census"]),
            ],
        },
    }


def canonical_json(inventory: dict[str, Any]) -> str:
    return json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


DIGIT_FONT = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}


def decode_pcx_rgb(path: Path) -> tuple[int, int, bytearray]:
    data = path.read_bytes()
    if len(data) < 897 or data[0] != 0x0A or data[3] != 8 or data[65] != 1:
        raise ValueError(f"Contact-sheet decoder requires an 8-bit single-plane PCX: {path}")
    xmin, ymin, xmax, ymax = struct.unpack_from("<4H", data, 4)
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    bytes_per_line = struct.unpack_from("<H", data, 66)[0]
    palette_pos = len(data) - 769
    if palette_pos <= 128 or data[palette_pos] != 12:
        raise ValueError(f"PCX lacks a 256-color palette: {path}")
    palette = data[palette_pos + 1 :]
    decoded = bytearray()
    pos = 128
    required = bytes_per_line * height
    while pos < palette_pos and len(decoded) < required:
        value = data[pos]
        pos += 1
        if value >= 0xC0:
            if pos >= palette_pos:
                break
            count = value & 0x3F
            decoded.extend([data[pos]] * count)
            pos += 1
        else:
            decoded.append(value)
    if len(decoded) < required:
        raise ValueError(f"PCX pixel payload is truncated: {path}")
    rgb = bytearray(width * height * 3)
    for y in range(height):
        row = decoded[y * bytes_per_line : y * bytes_per_line + width]
        for x, index in enumerate(row):
            source = index * 3
            dest = (y * width + x) * 3
            rgb[dest : dest + 3] = palette[source : source + 3]
    return width, height, rgb


def set_rgb(pixels: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < height:
        offset = (y * width + x) * 3
        pixels[offset : offset + 3] = bytes(color)


def annotate_cells(width: int, height: int, pixels: bytearray, cells: list[dict[str, Any]]) -> None:
    for cell in cells:
        x, y, cell_w, cell_h = cell["rect"]
        color = (255, 48, 48) if cell["classification"] != "unreachable" else (255, 192, 0)
        for px in range(x, min(width, x + cell_w)):
            set_rgb(pixels, width, height, px, y, color)
            set_rgb(pixels, width, height, px, y + cell_h - 1, color)
        for py in range(y, min(height, y + cell_h)):
            set_rgb(pixels, width, height, x, py, color)
            set_rgb(pixels, width, height, x + cell_w - 1, py, color)
        label = str(cell["runtime_index"])
        label_w = max(1, len(label) * 4 - 1)
        for py in range(y + 1, min(height, y + 8)):
            for px in range(x + 1, min(width, x + label_w + 3)):
                set_rgb(pixels, width, height, px, py, (0, 0, 0))
        for char_index, char in enumerate(label):
            glyph = DIGIT_FONT.get(char)
            if glyph is None:
                continue
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        set_rgb(pixels, width, height, x + 2 + char_index * 4 + gx, y + 2 + gy, (255, 255, 255))


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def write_rgb_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    rows = b"".join(b"\x00" + bytes(pixels[y * width * 3 : (y + 1) * width * 3]) for y in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(rows, 9))
    png += png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def generate_contact_sheets(
    inventory: dict[str, Any],
    install_root: Path,
    scenario_art_roots: Iterable[Path],
    output_dir: Path,
) -> list[dict[str, Any]]:
    specs = {spec["id"]: spec for spec in root_specs(install_root.resolve(), scenario_art_roots)}
    manifest = []
    for record in inventory["file_assets"]:
        atlas = record.get("atlas")
        if not record["effective"] or not atlas or atlas.get("status") != "resolved":
            continue
        source = specs[record["source_layer"]]["path"].joinpath(*Path(record["asset_path"]).parts[1:])
        digest = hashlib.sha256(record["asset_path"].casefold().encode("utf-8")).hexdigest()[:10]
        stem = "".join(char if char.isalnum() else "_" for char in Path(record["asset_path"]).stem.casefold()).strip("_")
        filename = f"{record['render_layer']}_{stem}_{digest}.png"
        destination = output_dir / filename
        width, height, pixels = decode_pcx_rgb(source)
        annotate_cells(width, height, pixels, atlas["cells"])
        write_rgb_png(destination, width, height, pixels)
        atlas["contact_sheet"] = f"contact_sheets/{filename}"
        manifest.append(
            {
                "asset_path": record["asset_path"],
                "contact_sheet": atlas["contact_sheet"],
                "contract_id": atlas["contract_id"],
                "annotated_cells": len(atlas["cells"]),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
    manifest.sort(key=lambda item: item["asset_path"].casefold())
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    inventory["contact_sheets"] = manifest
    return manifest


def markdown_summary(inventory: dict[str, Any]) -> str:
    summary = inventory["summary"]
    completeness = inventory["completeness"]
    lines = [
        "# Vanilla Civ III Map Art Inventory",
        "",
        "This report is generated from layered Base, Play the World, Conquests, and optional scenario art roots.",
        "Districts are intentionally out of scope. File counts are not sprite counts until atlas layout and runtime reachability are proven.",
        "",
        "## Current Census",
        "",
        f"- Effective file assets: {summary['effective_file_assets']}",
        f"- Effective unit directories: {summary['effective_unit_directories']}",
        f"- Unclassified effective files: {summary['unclassified_effective_files']}",
        f"- Atlases awaiting verified cell/index layouts: {summary['unresolved_atlases']}",
        f"- Units awaiting BIQ/runtime bindings: {summary['unresolved_unit_bindings']}",
        f"- Selected-layer unit animation references not found: {summary['missing_unit_animation_files_in_selected_layer']}",
        f"- BIQ primary unit types: {summary['biq_primary_unit_types']}",
        f"- BIQ resources / terrain types: {summary['biq_resources']} / {summary['biq_terrain_types']}",
        f"- BIQ unit types with missing art evidence: {summary['semantic_unit_gaps']}",
        f"- Unknown runtime selectors: {summary['runtime_unknown_selectors']}",
        f"- Completeness: `{completeness['status']}`",
        "",
        "## Render Responsibilities",
        "",
        "| Layer | Scope | Default ownership |",
        "|---|---|---|",
    ]
    for layer in inventory["render_layers"]:
        lines.append(f"| `{layer['id']}` | `{layer['scope']}` | `{layer['default_ownership']}` |")
    lines.extend(["", "## Effective File Assets", "", "| Path | Layer | Source | Dimensions | Atlas status |", "|---|---|---|---|---|"])
    for record in inventory["file_assets"]:
        if not record["effective"]:
            continue
        image = record.get("image")
        dimensions = f"{image['width']}x{image['height']}" if image else "-"
        atlas_status = record.get("atlas", {}).get("status", "not_an_atlas")
        lines.append(
            f"| `{record['asset_path']}` | `{record['render_layer']}` | `{record['source_layer']}` | {dimensions} | `{atlas_status}` |"
        )
    lines.extend(
        [
            "",
            "## Completion Gate",
            "",
            "M6.0 is complete only when every BIQ-defined or runtime-reachable map visual is classified as mapped, vanilla fallback, not map-rendered, or unreachable with evidence. Unknown atlas layouts, semantic bindings, and ownership decisions keep the gate open.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory layered vanilla Civ III map art")
    parser.add_argument("--install-root", type=Path, required=True, help="Civilization III Complete installation root")
    parser.add_argument("--scenario-art-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True, help="Output JSON ledger")
    parser.add_argument("--markdown", type=Path, help="Optional generated Markdown summary")
    parser.add_argument("--atlas-contracts", type=Path, default=DEFAULT_ATLAS_CONTRACTS)
    parser.add_argument("--biq-semantics", type=Path, default=DEFAULT_BIQ_SEMANTICS)
    parser.add_argument("--runtime-census", type=Path, default=INVENTORY_ROOT / "runtime_selector_census.json")
    parser.add_argument("--contact-sheets", type=Path, help="Optional directory for local annotated PNG contact sheets")
    parser.add_argument("--fail-on-unclassified", action="store_true")
    parser.add_argument("--fail-on-unresolved", action="store_true")
    args = parser.parse_args()

    inventory = build_inventory(
        args.install_root,
        args.scenario_art_root,
        atlas_contracts=load_json(args.atlas_contracts),
        biq_semantics=load_json(args.biq_semantics),
        runtime_census=load_json(args.runtime_census),
    )
    if args.contact_sheets:
        generate_contact_sheets(inventory, args.install_root, args.scenario_art_root, args.contact_sheets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(inventory), encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_summary(inventory), encoding="utf-8")
    print(json.dumps(inventory["summary"], indent=2, sort_keys=True))
    if args.fail_on_unclassified and inventory["summary"]["unclassified_effective_files"]:
        return 2
    if args.fail_on_unresolved and inventory["completeness"]["status"] != "complete":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
