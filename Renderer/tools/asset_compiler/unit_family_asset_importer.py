#!/usr/bin/env python3
"""Compile representative non-Warrior unit families into a generic local pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import normalized_animation
from Renderer.tools.asset_compiler.grassland_pack_builder import validate_runtime_independence
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage
from Renderer.tools.asset_compiler.unit_member_resolver import ASSETS_ROOT, resolve_unit
from Renderer.tools.asset_compiler.unit_model_extractor import _compile_component


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRATEGY = Path(__file__).with_name("unit_family_strategy.json")
DEFAULT_ACTION_CONTRACT = Path(__file__).with_name("unit_action_conversion.json")
DEFAULT_OWNER_COLOR_CONTRACT = Path(__file__).with_name("unit_owner_color_runtime.json")
DEFAULT_PACK = RENDERER_ROOT / "packs" / "UnitFamilyLab"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "units" / "family_build.json"
SAFE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
REQUIRED_ACTIONS = {"idle", "move", "attack", "death"}
BASIC_ACTIONS = {"idle", "fidget", "move", "fortify", "attack", "defend", "victory", "death"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_strategy(path: Path = DEFAULT_STRATEGY) -> dict[str, Any]:
    strategy = json.loads(path.read_text(encoding="utf-8"))
    if strategy.get("schema") != "c3x.source_unit_family_strategy.v0":
        raise ValueError("unsupported unit-family strategy schema")
    if strategy.get("runtime_integration") != "not_enabled":
        raise ValueError("unit-family intake must remain offline-only")
    units = strategy.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("unit-family strategy contains no proof units")
    slugs = []
    for unit in units:
        slug = unit.get("slug")
        if not isinstance(slug, str) or not SAFE_ID.fullmatch(slug):
            raise ValueError("unit-family strategy contains an invalid slug")
        if not isinstance(unit.get("source_artdef"), str) or not unit["source_artdef"].startswith("UNIT_"):
            raise ValueError(f"{slug} has an invalid source ArtDef")
        if unit.get("domain") not in {"land", "sea", "air"}:
            raise ValueError(f"{slug} has an invalid domain")
        actions = unit.get("actions")
        if not isinstance(actions, dict) or set(actions) != REQUIRED_ACTIONS:
            raise ValueError(f"{slug} must define idle, move, attack, and death")
        for action, record in {**actions, **unit.get("additional_actions", {})}.items():
            if not SAFE_ID.fullmatch(action) or not isinstance(record, dict):
                raise ValueError(f"{slug} contains an invalid action")
            if not isinstance(record.get("loop"), bool):
                raise ValueError(f"{slug}/{action} has no loop policy")
            source = record.get("source")
            alias = record.get("alias")
            if (source is None) == (alias is None):
                raise ValueError(f"{slug}/{action} must define exactly one source or alias")
            if source is not None and (
                not isinstance(source, str) or not source.startswith("ANIMATION_")
            ):
                raise ValueError(f"{slug}/{action} has an invalid source clip")
            if alias is not None and (
                not isinstance(alias, str) or not SAFE_ID.fullmatch(alias)
            ):
                raise ValueError(f"{slug}/{action} has an invalid action alias")
        all_actions = {**actions, **unit.get("additional_actions", {})}
        for action in all_actions:
            seen = set()
            current = action
            while "alias" in all_actions[current]:
                if current in seen:
                    raise ValueError(f"{slug}/{action} contains an action-alias cycle")
                seen.add(current)
                current = all_actions[current]["alias"]
                if current not in all_actions:
                    raise ValueError(f"{slug}/{action} aliases unknown action {current}")
        minimum = unit.get("minimum_matching_tracks")
        if not isinstance(minimum, int) or minimum < 1:
            raise ValueError(f"{slug} has an invalid track threshold")
        if not isinstance(unit.get("archetype"), str) or not SAFE_ID.fullmatch(unit["archetype"]):
            raise ValueError(f"{slug} has an invalid archetype")
        slugs.append(slug)
    if len(slugs) != len(set(slugs)):
        raise ValueError("unit-family slugs must be unique")
    runtime = strategy.get("runtime", {})
    if runtime.get("direction_count") != 8 or runtime.get("body_ownership") != "unit_body_only":
        raise ValueError("unit-family runtime contract must preserve eight directions and body-only ownership")
    return strategy


def load_action_contract(path: Path = DEFAULT_ACTION_CONTRACT) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != "c3x.unit_action_conversion.v0":
        raise ValueError("unsupported unit-action conversion schema")
    if contract.get("runtime_integration") != "not_enabled":
        raise ValueError("unit-action conversion must remain offline-only")
    actions = contract.get("actions")
    if not isinstance(actions, dict) or set(actions) != BASIC_ACTIONS:
        raise ValueError("unit-action contract must define exactly the eight basic actions")
    slots = {}
    for action, record in actions.items():
        if record.get("playback") not in {"loop", "clamp"}:
            raise ValueError(f"unit-action contract has invalid playback for {action}")
        for slot in record.get("civ3_slots", []):
            if slot in slots:
                raise ValueError(f"Civ III slot {slot} is assigned to multiple logical actions")
            slots[slot] = action
    if slots.get("ATTACK1") != "attack" or slots.get("ATTACK2") != "attack":
        raise ValueError("ATTACK1 and ATTACK2 must initially alias logical attack")
    if actions["defend"].get("civ3_slots") != []:
        raise ValueError("defend must remain event-derived rather than an invented FLC slot")
    return contract


def load_owner_color_contract(path: Path = DEFAULT_OWNER_COLOR_CONTRACT) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != "c3x.unit_owner_color_runtime.v0":
        raise ValueError("unsupported unit owner-color runtime schema")
    if contract.get("runtime_integration") != "not_enabled":
        raise ValueError("unit owner-color intake must remain offline-only")
    conversion = contract.get("asset_conversion", {})
    if conversion.get("bake_owner_variants") is not False:
        raise ValueError("unit owner colors must not be baked into per-civ variants")
    if conversion.get("material_contract") != "neutral_base_plus_civ_color_weight":
        raise ValueError("unit owner-color material contract is invalid")
    if conversion.get("primary_ramp_indices") != [0, 15]:
        raise ValueError("unit owner-color primary ramp must preserve Civ III indices 0..15")
    selection = contract.get("runtime_selection", {})
    if (
        selection.get("instance_selector") != "display_color_table_id"
        or selection.get("display_civ_authority") != "civ3_native_unit_body_color_selection"
        or selection.get("do_not_assume_owner_equals_display_civ") is not True
    ):
        raise ValueError("unit owner-color selection must follow Civ III's displayed identity")
    lut = contract.get("gpu_lut", {})
    if (
        lut.get("format") != "rgba8_unorm_srgb"
        or lut.get("width") != 64
        or lut.get("height") != 32
        or lut.get("row_semantic") != "display_color_table_id"
    ):
        raise ValueError("unit owner-color GPU LUT contract is invalid")
    invalidation = contract.get("invalidation", {})
    if invalidation.get("owner_or_display_identity_change") != "update_instance_selector_only":
        raise ValueError("owner changes must not rebuild converted unit art")
    if contract.get("scenario_policy", {}).get("source") != "effective_tables_already_loaded_by_civ3":
        raise ValueError("scenario owner colors must use Civ III's effective loaded tables")
    authoring = contract.get("authoring", {})
    if (
        authoring.get("scope") != "per_material_component_not_per_unit_code"
        or authoring.get("override_key") != "stable_logical_asset_id"
        or set(authoring.get("allowed_modes", [])) != {"none", "source_mask", "authored_mask"}
    ):
        raise ValueError("unit owner-color authoring contract is not source-agnostic")
    gate = contract.get("coverage_gate", {})
    minimum = gate.get("minimum_changed_pixels", {})
    if (
        gate.get("measurement")
        != "changed_screen_pixels_against_neutral_at_normal_and_reduced_civ3_scale"
        or gate.get("palette_sample_policy")
        != "maximum_coverage_across_distinct_runtime_color_tables"
        or gate.get("normal_scale_factor") != 1.0
        or not isinstance(gate.get("reduced_scale_factor"), (int, float))
        or not 0.0 < gate["reduced_scale_factor"] < 1.0
        or not isinstance(gate.get("rgb_delta_threshold"), int)
        or not 1 <= gate["rgb_delta_threshold"] <= 255
        or not isinstance(minimum.get("normal"), int)
        or not isinstance(minimum.get("reduced"), int)
        or minimum["normal"] < 1
        or minimum["reduced"] < 1
    ):
        raise ValueError("unit owner-color coverage gate is invalid")
    return contract


def load_owner_color_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "c3x.unit_owner_color_overrides.v0":
        raise ValueError("unsupported unit owner-color override schema")
    overrides = document.get("overrides")
    if not isinstance(overrides, dict):
        raise ValueError("unit owner-color overrides must be keyed by logical asset ID")
    for asset_id, record in overrides.items():
        if not isinstance(asset_id, str) or not asset_id.startswith("unit/") or not isinstance(record, dict):
            raise ValueError("unit owner-color override has an invalid logical asset ID")
        if record.get("mode") not in {"none", "source_mask", "authored_mask", "solid_color"}:
            raise ValueError(f"unit owner-color override {asset_id} has an invalid mode")
        strength = record.get("strength", 0.0)
        if not isinstance(strength, (int, float)) or not 0.0 <= strength <= 1.0:
            raise ValueError(f"unit owner-color override {asset_id} has an invalid strength")
        if record.get("mask_source") not in {"base_color_alpha_inverse", "constant_one"}:
            raise ValueError(f"unit owner-color override {asset_id} has an invalid mask source")
        palette_index = record.get("representative_palette_index", 6)
        if not isinstance(palette_index, int) or not 0 <= palette_index < 64:
            raise ValueError(f"unit owner-color override {asset_id} has an invalid palette index")
    return overrides


def default_owner_color_for_component(
    component: dict[str, Any], tint_strength: float
) -> dict[str, Any]:
    """Translate Civ VI's generic tint semantics into pack-local mask metadata."""
    if component.get("tint") != "USE_CIV_COLOR":
        return {
            "mode": "none",
            "mask_source": "constant_one",
            "strength": 0.0,
            "representative_palette_index": 6,
        }
    dedicated_geometry = component.get("role") == "TeamColor"
    return {
        "mode": "solid_color" if dedicated_geometry else "source_mask",
        "mask_source": "constant_one" if dedicated_geometry else "base_color_alpha_inverse",
        "strength": tint_strength,
        "representative_palette_index": 6,
    }


def _physical_package(assets_root: Path, content: str, logical: str) -> Path:
    relative = Path(logical)
    if relative.is_absolute() or ".." in relative.parts or "\\" in logical:
        raise ValueError(f"unsafe unit package path: {logical}")
    path = assets_root / content / "Platforms" / "Windows" / "BLPs" / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _initial_entry(path: Path, entries: list[str]) -> str:
    data = path.read_bytes()
    unique = [entry for entry in entries if data.count(entry.encode("ascii") + b"\0") == 1]
    if not unique:
        raise ValueError(f"unit package has no unique initialization entry: {path}")
    return unique[0]


def _clip_record(path: Path, loop: bool) -> dict[str, Any]:
    clip = normalized_animation.load_clip(path)
    return {
        "clip": path.as_posix(),
        "duration": clip.duration,
        "sample_rate": clip.sample_rate,
        "frame_count": clip.frame_count,
        "track_groups": len(clip.groups),
        "tracks": sum(len(group.tracks) for group in clip.groups),
        "clip_form": "pose" if clip.frame_count <= 2 else "motion",
        "loop": loop,
        "sha256": _sha256(path),
        "binding_status": "converted_unvalidated",
    }


def compile_unit_families(
    assets_root: Path,
    strategy_path: Path = DEFAULT_STRATEGY,
    pack: Path = DEFAULT_PACK,
    report_path: Path = DEFAULT_REPORT,
    owner_color_overrides_path: Path | None = None,
) -> dict[str, Any]:
    strategy = load_strategy(strategy_path)
    action_contract = load_action_contract()
    owner_color_contract = load_owner_color_contract()
    owner_color_overrides = load_owner_color_overrides(owner_color_overrides_path)
    default_tint_strength = owner_color_contract["shader"]["lab_calibration"]["strength"]
    shared_data = assets_root / strategy["source_content"] / "Platforms" / "Windows" / "BLPs" / "SHARED_DATA"
    if not shared_data.is_dir():
        raise FileNotFoundError(shared_data)
    resolved = [(unit, resolve_unit(assets_root, unit["source_artdef"], "Any")) for unit in strategy["units"]]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit, recipe in resolved:
        for component in recipe["selected_components"]:
            grouped[component["source_package"]].append({"unit": unit, "component": component})
    packages = {}
    package_reports = {}
    for logical, items in grouped.items():
        path = _physical_package(assets_root, strategy["source_content"], logical)
        packages[logical] = IndexedStaticPackage(
            path, _initial_entry(path, [item["component"]["source_entry"] for item in items])
        )
        package_reports[logical] = {"path": str(path), "sha256": _sha256(path)}

    assets = {}
    units = {}
    animation_records = {}
    component_evidence = []
    source_animation_evidence = []
    texture_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for unit, source_recipe in resolved:
        slug = unit["slug"]
        component_records = []
        role_counts: dict[str, int] = defaultdict(int)
        for component in source_recipe["selected_components"]:
            role = re.sub(r"[^a-z0-9]+", "_", component["role"].lower()).strip("_")
            role_counts[role] += 1
            key = role if role_counts[role] == 1 else f"{role}_{role_counts[role]}"
            asset, evidence = _compile_component(
                packages[component["source_package"]],
                shared_data,
                pack,
                component,
                texture_cache,
                slug,
                key,
            )
            asset_id = f"unit/{slug}/{key}"
            owner_color = owner_color_overrides.get(asset_id)
            if owner_color is None:
                owner_color = default_owner_color_for_component(
                    component, default_tint_strength
                )
            component_path = pack / asset["component"]
            component_document = json.loads(component_path.read_text(encoding="utf-8"))
            component_document["owner_color"] = owner_color
            _write_json(component_path, component_document)
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
            component_evidence.append(
                {
                    "unit": unit["source_artdef"],
                    "source_package": component["source_package"],
                    "owner_color": owner_color,
                    **evidence,
                }
            )
        actions = {}
        all_actions = {**unit["actions"], **unit.get("additional_actions", {})}
        for action, action_record in all_actions.items():
            source_action = action
            while "alias" in all_actions[source_action]:
                source_action = all_actions[source_action]["alias"]
            source = all_actions[source_action]
            source_path = shared_data / source["source"]
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            relative = Path("animations") / "unit" / slug / f"{source_action}.c3anim"
            output_path = pack / relative
            animation_id = f"animation/unit/{slug}/{action}"
            actions[action] = animation_id
            source_animation_evidence.append(
                {
                    "unit": unit["source_artdef"],
                    "action": action,
                    "source": source["source"],
                    "source_action": source_action,
                    "alias": action_record.get("alias"),
                    "source_match": action_record.get("source_match", "direct"),
                    "source_path": str(source_path),
                    "source_sha256": _sha256(source_path),
                    "converted": output_path.is_file(),
                }
            )
            animation_records[animation_id] = (
                {
                    **_clip_record(output_path, action_record["loop"]),
                    "clip": relative.as_posix(),
                    "source_action": source_action,
                    "source_match": action_record.get("source_match", "direct"),
                    **(
                        {"alias_of": f"animation/unit/{slug}/{source_action}"}
                        if source_action != action
                        else {}
                    ),
                }
                if output_path.is_file()
                else {
                    "status": "pending_offline_conversion",
                    "loop": action_record["loop"],
                    "source_action": source_action,
                }
            )
        driver = next(
            record["asset"]
            for record in component_records
            if json.loads((pack / assets[record["asset"]]["component"]).read_text(encoding="utf-8"))["binding_mode"]
            == "vertex_skin"
        )
        recipe_relative = Path("units") / f"{slug}_recipe.json"
        _write_json(
            pack / recipe_relative,
            {
                "schema": "c3x.unit_recipe.v0",
                "unit_id": f"unit/{slug}",
                "civ3_ids": unit["civ3_ids"],
                "domain": unit["domain"],
                "archetype": unit["archetype"],
                "member": {
                    "count": source_recipe["member"]["count"],
                    "member_scale": source_recipe["member"]["member_scale"],
                    "variation_scale": source_recipe["member"]["variation_scale"],
                },
                "components": component_records,
                "animation_driver": driver,
                "minimum_matching_tracks": unit["minimum_matching_tracks"],
                "formation": source_recipe["formation"],
                "movement": source_recipe["movement"],
                "actions": actions,
                "runtime_integration": "not_enabled",
            },
        )
        units[f"unit/{slug}"] = {"recipe": recipe_relative.as_posix()}

    unknown_owner_color_overrides = sorted(set(owner_color_overrides) - set(assets))
    if unknown_owner_color_overrides:
        raise ValueError(
            "unit owner-color overrides reference unknown logical assets: "
            + ", ".join(unknown_owner_color_overrides)
        )

    manifest = {
        "schema": "c3x.unit_pack.v0",
        "name": "UnitFamilyLab",
        "source_policy": "Local licensed-source import; derived art is not redistributable.",
        "units": units,
        "assets": assets,
        "animations": animation_records,
        "action_contract": "units/action_contract.json",
        "owner_color_contract": "units/owner_color_runtime.json",
        "runtime_contract": strategy["runtime"],
        "runtime_integration": "not_enabled",
    }
    _write_json(pack / manifest["action_contract"], action_contract)
    _write_json(pack / manifest["owner_color_contract"], owner_color_contract)
    _write_json(pack / "manifest.json", manifest)
    independence_errors = validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError("Runtime unit-family pack is source-dependent: " + "; ".join(independence_errors))
    report = {
        "schema": "c3x.source_unit_family_build.v0",
        "strategy": {"path": str(strategy_path), "sha256": _sha256(strategy_path)},
        "action_contract": {
            "path": str(DEFAULT_ACTION_CONTRACT),
            "sha256": _sha256(DEFAULT_ACTION_CONTRACT),
        },
        "owner_color_contract": {
            "path": str(DEFAULT_OWNER_COLOR_CONTRACT),
            "sha256": _sha256(DEFAULT_OWNER_COLOR_CONTRACT),
        },
        "owner_color_overrides": (
            None
            if owner_color_overrides_path is None
            else {"path": str(owner_color_overrides_path), "sha256": _sha256(owner_color_overrides_path)}
        ),
        "packages": package_reports,
        "units": [
            {
                "source_artdef": unit["source_artdef"],
                "slug": unit["slug"],
                "domain": unit["domain"],
                "archetype": unit["archetype"],
                "source_recipe": recipe,
            }
            for unit, recipe in resolved
        ],
        "components": component_evidence,
        "animations": source_animation_evidence,
        "deferred_archetype_probes": strategy["deferred_archetype_probes"],
        "outputs": {
            "pack": str(pack),
            "units": len(units),
            "components": len(assets),
            "skinned_components": sum(item["geometry"]["skinned"] for item in component_evidence),
            "rigid_components": sum(not item["geometry"]["skinned"] for item in component_evidence),
            "textures": len(texture_cache),
            "action_slots": len(animation_records),
            "converted_actions": sum(record.get("clip") is not None for record in animation_records.values()),
            "unique_converted_clips": len(
                {record["clip"] for record in animation_records.values() if record.get("clip")}
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
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--owner-color-overrides",
        type=Path,
        help="optional source-agnostic per-component owner-color authoring sidecar",
    )
    args = parser.parse_args(argv)
    try:
        report = compile_unit_families(
            args.assets_root,
            args.strategy,
            args.pack,
            args.report,
            args.owner_color_overrides,
        )
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
