#!/usr/bin/env python3
"""Verify the installed Civ VI camp graph and Civ III camp-state boundary."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from Renderer.tools.asset_compiler.artdef_graph_resolver import (
    DEFAULT_ASSETS_ROOT,
    index_artdefs,
    resolve_target_graph,
    resolve_terminal_packages,
)


DEFAULT_REPORT = (
    REPOSITORY_ROOT
    / "Renderer"
    / "preview"
    / "out"
    / "tile_objects"
    / "barbarian_camp_source_probe.json"
)
GAMEPLAY_IMPROVEMENTS = Path("Base/Assets/Gameplay/Data/Improvements.xml")
DECOMPILED_CIV3 = REPOSITORY_ROOT / "ref" / "Civ3Conquests_master.exe.c"
EXPECTED_ROOTS = {"VIL_BAR_01", "VIL_BAR_IND"}
EXPECTED_PACKAGE = "Base/Platforms/Windows/BLPs/landmarks/tilebases.blp"


def _improvement_suppresses_resource(path: Path, name: str) -> bool:
    root = ET.parse(path).getroot()
    for item in root.findall("./m_RootCollections/Element/Element"):
        item_name = item.find("m_Name")
        if item_name is None or item_name.attrib.get("text") != name:
            continue
        for value in item.findall(".//m_Values/Element"):
            parameter = value.find("m_ParamName")
            if parameter is None or parameter.attrib.get("text") != "SuppressResource":
                continue
            enabled = value.find("m_bValue")
            return enabled is not None and (enabled.text or "").strip().lower() == "true"
    raise ValueError(f"Installed ArtDef has no {name} SuppressResource field")


def probe(assets_root: Path) -> dict:
    index = index_artdefs(assets_root)
    if index["parse_errors"]:
        raise ValueError(f"Installed ArtDef parse failures: {len(index['parse_errors'])}")
    graph = resolve_target_graph(
        index,
        "improvement",
        "IMPROVEMENT_BARBARIAN_CAMP",
        "tile_object/barbarian_camp",
    )
    resolve_terminal_packages([graph], index, assets_root)
    roots = {
        terminal["entry"]
        for terminal in graph["terminals"]
        if terminal["scope"] == "map_visual"
    }
    if roots != EXPECTED_ROOTS:
        raise ValueError(f"Unexpected barbarian-camp roots: {sorted(roots)}")
    unresolved = [
        terminal
        for terminal in graph["terminals"]
        if terminal["scope"] == "map_visual"
        and (
            terminal.get("status") != "resolved"
            or terminal.get("package_path") != EXPECTED_PACKAGE
        )
    ]
    if unresolved:
        raise ValueError("Barbarian-camp roots do not resolve uniquely to the expected package")
    suppresses_resource = _improvement_suppresses_resource(
        assets_root / "Base/ArtDefs/Improvements.artdef",
        "IMPROVEMENT_BARBARIAN_CAMP",
    )
    if not suppresses_resource:
        raise ValueError("Installed Civ VI barbarian camp no longer suppresses resources")

    gameplay_path = assets_root / GAMEPLAY_IMPROVEMENTS
    gameplay_root = ET.parse(gameplay_path).getroot()
    row = gameplay_root.find(
        ".//Improvements/Row[@ImprovementType='IMPROVEMENT_BARBARIAN_CAMP']"
    )
    if row is None:
        raise ValueError("Installed gameplay data has no barbarian-camp row")
    expected_gameplay = {
        "BarbarianCamp": "true",
        "RemoveOnEntry": "true",
        "DispersalGold": "50",
    }
    if any(row.attrib.get(key) != value for key, value in expected_gameplay.items()):
        raise ValueError("Installed barbarian-camp gameplay flags changed")

    civ3 = DECOMPILED_CIV3.read_text(encoding="utf-8", errors="replace")
    civ3_fragments = {
        "viewer_conditioned_overlay_bit": "return (bool)((byte)(uVar1 >> 7) & 1);",
        "draw_uses_tribe_id": "uVar9 = (*pTVar4->vtable->m44_Get_Barbarian_TribeID)",
        "draw_has_dedicated_camp_call": "m23_Draw_Barbarian_Camp)(uVar9);",
        "capture_clears_camp_bit": "m51_Unset_Tile_Flags)(pTVar4,0,0x80,-1,-1);",
        "capture_releases_tribe_id": "m55_Set_Barbarian_TribeID)(pTVar4,-1);",
        "spawn_creates_units_separately": "pUVar7 = spawn_unit(this_00,bic_data.General.BarbarianBasicUnitID",
    }
    missing = [name for name, fragment in civ3_fragments.items() if fragment not in civ3]
    if missing:
        raise ValueError("Civ III camp evidence changed: " + ", ".join(missing))

    return {
        "schema": "c3x.barbarian_camp_source_probe.v0",
        "status": "passed",
        "civ6": {
            "artdef_chain": [
                "Base/ArtDefs/Improvements.artdef:IMPROVEMENT_BARBARIAN_CAMP",
                "Base/ArtDefs/Landmarks.artdef:LM_BARBARIAN_CAMP",
            ],
            "root_assets": sorted(roots),
            "package": EXPECTED_PACKAGE,
            "gameplay_flags": expected_gameplay,
            "source_suppresses_resource": suppresses_resource,
        },
        "civ3": {
            "source": "ref/Civ3Conquests_master.exe.c",
            "verified_boundaries": sorted(civ3_fragments),
            "renderer_state_authority": [
                "Tile.m7_Check_Barbarian_Camp(viewer_civ_id)",
                "Tile.m44_Get_Barbarian_TribeID",
            ],
        },
        "runtime_integration": "not_enabled",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    try:
        report = probe(args.assets_root)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
