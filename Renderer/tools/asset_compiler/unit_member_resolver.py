#!/usr/bin/env python3
"""Resolve Civ VI unit/member/bin composition into an explicit lab recipe."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ASSETS_ROOT = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common"
    / "Sid Meier's Civilization VI/Civ6.app/Contents/Assets"
)
DEFAULT_REPORT = Path(__file__).resolve().parents[2] / "preview/out/units/warrior_recipe.json"


def _text(node: ET.Element | None, child: str) -> str:
    if node is None:
        return ""
    value = node.find(child)
    if value is None:
        return ""
    return value.get("text", value.text or "")


def _name(node: ET.Element) -> str:
    return _text(node, "m_Name")


def _items(collection: ET.Element | None) -> list[ET.Element]:
    return [] if collection is None else list(collection.findall("Element"))


def _root_collection(document: ET.Element, name: str) -> ET.Element:
    roots = document.find("m_RootCollections")
    if roots is None:
        raise ValueError("ArtDef has no root collections")
    matches = [item for item in roots.findall("Element") if _text(item, "m_CollectionName") == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} root collection, found {len(matches)}")
    return matches[0]


def _named_item(collection: ET.Element | None, name: str) -> ET.Element:
    matches = [item for item in _items(collection) if _name(item) == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one item {name!r}, found {len(matches)}")
    return matches[0]


def _child_collection(node: ET.Element, name: str) -> ET.Element:
    parent = node.find("m_ChildCollections")
    matches = [] if parent is None else [
        item for item in parent.findall("Element") if _text(item, "m_CollectionName") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} child collection under {_name(node)!r}, found {len(matches)}")
    return matches[0]


def _field_nodes(node: ET.Element) -> list[ET.Element]:
    values = node.find("m_Fields/m_Values")
    return [] if values is None else list(values.findall("Element"))


def _field(node: ET.Element, name: str) -> ET.Element | None:
    matches = [value for value in _field_nodes(node) if _text(value, "m_ParamName") == name]
    if len(matches) > 1:
        raise ValueError(f"Duplicated field {name!r} under {_name(node)!r}")
    return matches[0] if matches else None


def _reference(node: ET.Element, name: str, required: bool = True) -> str:
    value = _field(node, name)
    result = _text(value, "m_ElementName")
    if required and not result:
        raise ValueError(f"Missing reference field {name!r} under {_name(node)!r}")
    return result


def _string(node: ET.Element, name: str, default: str = "") -> str:
    value = _field(node, name)
    result = _text(value, "m_Value")
    return result if result else default


def _float(node: ET.Element, name: str, default: float | None = None) -> float:
    value = _field(node, name)
    raw = "" if value is None else _text(value, "m_fValue")
    if not raw:
        if default is None:
            raise ValueError(f"Missing float field {name!r} under {_name(node)!r}")
        return default
    return float(raw)


def _int(node: ET.Element, name: str, default: int | None = None) -> int:
    value = _field(node, name)
    raw = "" if value is None else _text(value, "m_nValue")
    if not raw:
        if default is None:
            raise ValueError(f"Missing integer field {name!r} under {_name(node)!r}")
        return default
    return int(raw)


def _bool(node: ET.Element, name: str, default: bool = False) -> bool:
    value = _field(node, name)
    raw = "" if value is None else _text(value, "m_bValue")
    return default if not raw else raw.lower() == "true"


def _blp_asset(node: ET.Element) -> dict[str, Any]:
    fields = _field_nodes(node)
    entries = [value for value in fields if _text(value, "m_ParamName") == "Asset"]
    if len(entries) != 1:
        raise ValueError(f"Bin candidate {_name(node)!r} does not have one Asset field")
    entry = entries[0]
    name = _text(entry, "m_EntryName")
    package = _text(entry, "m_BLPPackage")
    if not name or not package:
        raise ValueError(f"Bin candidate {_name(node)!r} has an incomplete BLP binding")
    return {
        "candidate": _name(node),
        "source_entry": name,
        "source_package": package + ("" if package.lower().endswith(".blp") else ".blp"),
        "library": _text(entry, "m_LibraryName"),
        "scale": _float(node, "Scale", 1.0),
    }


def _bin(attachment_bins: ET.Element, path: str, culture: str) -> dict[str, Any]:
    if path.startswith("V:") and "/" not in path and len(path) > 2:
        return {
            "bin": path,
            "requested_culture": culture,
            "selected_culture": None,
            "bin_tint": None,
            "candidates": [],
            "selection": None,
            "selection_rule": "virtual member attachment resolved by compound-unit composition",
            "virtual_member": path[2:],
        }
    parts = path.split("/")
    if len(parts) == 2 and all(parts):
        category_name, bin_name = parts
        requested_candidate = None
    elif len(parts) == 4 and all(parts) and parts[2] == "#":
        category_name, bin_name, _, requested_candidate = parts
    else:
        raise ValueError(
            f"Unit attachment bin path is not Category/Bin or Category/Bin/#/Candidate: {path!r}"
        )
    category = _named_item(attachment_bins, category_name)
    leaf = _named_item(_child_collection(category, "Groups"), bin_name)
    cultures = _child_collection(leaf, "Cultures")
    available = {_name(item): item for item in _items(cultures)}
    selected_culture = culture if culture in available else "Any"
    if selected_culture not in available:
        raise ValueError(f"Unit attachment bin {path!r} has neither {culture!r} nor Any")
    culture_node = available[selected_culture]
    tint = _reference(culture_node, "Tint", required=False)
    candidates = [_blp_asset(item) for item in _items(_child_collection(culture_node, "Assets"))]
    if not candidates:
        raise ValueError(f"Unit attachment bin {path!r}/{selected_culture} has no candidates")
    if requested_candidate is None:
        selection = candidates[0]
        selection_rule = "first declared candidate in exact culture, otherwise Any"
    else:
        matches = [item for item in candidates if item["candidate"] == requested_candidate]
        if len(matches) != 1:
            raise ValueError(
                f"Unit attachment bin {category_name}/{bin_name}/{selected_culture} "
                f"does not contain exactly one requested candidate {requested_candidate!r}"
            )
        selection = matches[0]
        selection_rule = "explicit # candidate in exact culture, otherwise Any"
    return {
        "bin": path,
        "requested_culture": culture,
        "selected_culture": selected_culture,
        "bin_tint": tint or None,
        "candidates": candidates,
        "selection": selection,
        "selection_rule": selection_rule,
    }


def _find_root_item(document: ET.Element, collection: str, name: str) -> ET.Element:
    return _named_item(_root_collection(document, collection), name)


def resolve_unit(
    assets_root: Path,
    unit_name: str,
    culture: str = "Any",
    variation_name: str | None = None,
    member_index: int | None = None,
) -> dict[str, Any]:
    units_path = assets_root / "Base/ArtDefs/Units.artdef"
    bins_path = assets_root / "Base/ArtDefs/Unit_Bins.artdef"
    units = ET.parse(units_path).getroot()
    bins = ET.parse(bins_path).getroot()
    unit = _find_root_item(units, "Units", unit_name)
    members = _items(_child_collection(unit, "Members"))
    if member_index is None:
        if len(members) != 1:
            raise ValueError(
                f"Compound unit has {len(members)} member recipes; select one with member_index"
            )
        selected_member_index = 0
    else:
        if not isinstance(member_index, int) or not 0 <= member_index < len(members):
            raise ValueError(
                f"Member recipe index {member_index!r} is outside 0..{len(members) - 1}"
            )
        selected_member_index = member_index
    member_binding = members[selected_member_index]
    member_name = _reference(member_binding, "Type")
    member = _find_root_item(units, "UnitMemberTypes", member_name)
    culture_nodes = {_name(item): item for item in _items(_child_collection(member, "Cultures"))}
    member_culture = culture if culture in culture_nodes else "Any"
    if member_culture not in culture_nodes:
        raise ValueError(f"Member {member_name!r} has neither {culture!r} nor Any")
    variations = _items(_child_collection(culture_nodes[member_culture], "Variations"))
    if not variations:
        raise ValueError(f"Member {member_name!r}/{member_culture} has no variations")
    if variation_name is None:
        variation = variations[0]
    else:
        matches = [item for item in variations if _name(item) == variation_name]
        if len(matches) != 1:
            raise ValueError(
                f"Member {member_name!r}/{member_culture} does not contain exactly one "
                f"variation {variation_name!r}"
            )
        variation = matches[0]
    attachments = []
    attachment_bins = _root_collection(bins, "UnitAttachmentBins")
    for attachment in _items(_child_collection(variation, "Attachments")):
        bin_paths = [_name(item) for item in _items(_child_collection(attachment, "Bins"))]
        attachments.append(
            {
                "role": _name(attachment),
                "point": _string(attachment, "Point", "Root"),
                "attachment_tint": _reference(attachment, "Tint", required=False) or None,
                "bins": [_bin(attachment_bins, path, culture) for path in bin_paths],
            }
        )
    formation_name = _reference(unit, "Formation")
    formation = _find_root_item(units, "UnitFormationTypes", formation_name)
    movement_name = _reference(member, "Movement")
    movement = _find_root_item(units, "UnitMovementTypes", movement_name)
    unit_combat_name = _reference(unit, "UnitCombat", required=False)
    combat_formation_name = None
    offsets = []
    if unit_combat_name:
        unit_combat = _find_root_item(units, "UnitCombat", unit_combat_name)
        combat_formation_name = _reference(unit_combat, "CombatFormation", required=False) or None
        if combat_formation_name:
            combat_formation = _find_root_item(units, "CombatFormation", combat_formation_name)
            offsets = [
                {"forward": _float(item, "Forward"), "left": _float(item, "Left")}
                for item in _items(_child_collection(combat_formation, "Offsets"))
            ]
    selected_components = []
    virtual_attachments = []
    for attachment in attachments:
        for bin_info in attachment["bins"]:
            selection = bin_info["selection"]
            if selection is None:
                virtual_attachments.append(
                    {
                        "role": attachment["role"],
                        "point": attachment["point"],
                        "member": bin_info["virtual_member"],
                    }
                )
            elif selection["source_entry"] != "EmptyUnitAttachment":
                selected_components.append(
                    {
                        "role": attachment["role"],
                        "point": attachment["point"],
                        "tint": attachment["attachment_tint"] or bin_info["bin_tint"],
                        "bin": bin_info["bin"],
                        **selection,
                    }
                )
    return {
        "schema": "c3x.source_unit_recipe.v0",
        "unit": unit_name,
        "culture": culture,
        "member": {
            "type": member_name,
            "recipe_index": selected_member_index,
            "recipe_count": len(members),
            "count": _int(member_binding, "Count"),
            "member_scale": _float(member_binding, "Scale", 1.0),
            "selected_culture": member_culture,
            "variation": _name(variation),
            "variation_scale": _float(variation, "Scale", 1.0),
            "is_attachment": _bool(variation, "IsAttachment"),
            "attachments": attachments,
        },
        "selected_components": selected_components,
        "virtual_attachments": virtual_attachments,
        "formation": {
            "name": formation_name,
            "spacing_x": _float(formation, "SpacingX"),
            "spacing_y": _float(formation, "SpacingY"),
            "stagger_increment": _int(formation, "StaggerIncrement"),
            "first_row_width": _int(formation, "FirstRowWidth"),
            "layout": _reference(formation, "Type"),
            "combat": combat_formation_name,
            "combat_offsets": offsets,
        },
        "movement": {
            "name": movement_name,
            "total_time": _float(movement, "TotalTime"),
            "ease_in": _float(movement, "EaseIn"),
            "ease_out": _float(movement, "EaseOut"),
        },
        "actions": (
            {
                "idle": "ANIMATION_Warrior_IdleB",
                "move": "ANIMATION_UnitMedium_Run_SwordAndShieldA",
                "attack": "ANIMATION_Warrior_AttackMeleeB",
                "death": "ANIMATION_Warrior_DeathMeleeA",
            }
            if unit_name == "UNIT_WARRIOR"
            else {}
        ),
        "source": {
            "units_artdef": str(units_path),
            "unit_bins_artdef": str(bins_path),
            "runtime_integration": "not_enabled",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=ASSETS_ROOT)
    parser.add_argument("--unit", default="UNIT_WARRIOR")
    parser.add_argument("--culture", default="Any")
    parser.add_argument("--variation", help="select an exact member variation instead of the first")
    parser.add_argument("--member-index", type=int, help="select one recipe from a compound unit")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        result = resolve_unit(
            args.assets_root,
            args.unit,
            args.culture,
            args.variation,
            args.member_index,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, ET.ParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Resolved {result['unit']}: {result['member']['count']} members, {len(result['selected_components'])} non-empty components")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
