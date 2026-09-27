#!/usr/bin/env python3
"""Inventory installed Civ VI wall ArtDefs against the local city-wall intake.

This only reads the installed source and writes source names, never source art.
The resulting report is portable: paths are relative to the Civ VI Assets root.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from Renderer.tools.asset_compiler.artdef_graph_resolver import DEFAULT_ASSETS_ROOT


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = Path(__file__).with_name("city_adjunct_sets.json")
DEFAULT_REPORT = ROOT / "lab" / "out" / "cities" / "wall-source-audit.json"


def _values(item: ET.Element, tag: str) -> list[str]:
    return [element.attrib["text"] for element in item.iter(tag) if "text" in element.attrib]


def audit(assets_root: Path, mapping_path: Path = DEFAULT_MAPPING) -> dict:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    imported = {(asset["source_package"], asset["source_entry"])
                for asset in mapping["assets"]}
    sets = []
    packages = sorted(path.relative_to(assets_root).as_posix()
                      for path in assets_root.rglob("*.blp")
                      if "wall" in path.name.lower())
    for path in sorted(assets_root.rglob("Walls.artdef")):
        source = path.relative_to(assets_root).as_posix()
        tree = ET.parse(path).getroot()
        for collection in tree.findall("./m_RootCollections/Element"):
            collection_name = collection.find("m_CollectionName").attrib["text"]
            if collection_name not in ("WallSet", "TowerSet"):
                continue
            for item in collection.findall("./Element"):
                name = item.find("m_Name").attrib["text"]
                fields = item.find("m_Fields")
                gameplay = []
                if fields is not None:
                    for field in fields.findall("./m_Values/Element"):
                        param = field.find("m_ParamName")
                        value = field.find("m_Value")
                        if param is not None and value is not None and param.attrib.get("text") == "Gameplay Name":
                            gameplay.append(value.attrib.get("text", ""))
                entries = sorted(set(_values(item, "m_EntryName")))
                sets.append({
                    "artdef": source,
                    "collection": collection_name,
                    "name": name,
                    "gameplay": gameplay,
                    "entries": entries,
                })
    # ArtDef names are the primary evidence. Package paths are included so
    # a missing named set cannot be mistaken for absent source geometry.
    mapped_entries = sorted({entry for _, entry in imported})
    artdef_entries = sorted({entry for item in sets for entry in item["entries"]})
    return {
        "schema": "c3x.wall_source_audit.v0",
        "source_artdefs": sorted({item["artdef"] for item in sets}),
        "wall_packages": packages,
        "sets": sets,
        "mapped_city_wall_entries": mapped_entries,
        "unmapped_artdef_entries": sorted(set(artdef_entries) - set(mapped_entries)),
        "mapped_entries_not_in_artdefs": sorted(set(mapped_entries) - set(artdef_entries)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = audit(args.assets_root, args.mapping)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artdefs": len(report["source_artdefs"]),
                      "wall_packages": len(report["wall_packages"]),
                      "wall_sets": sum(item["collection"] == "WallSet" for item in report["sets"]),
                      "tower_sets": sum(item["collection"] == "TowerSet" for item in report["sets"]),
                      "mapped_city_wall_pieces": len(report["mapped_city_wall_entries"]),
                      "unmapped_entries": len(report["unmapped_artdef_entries"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
