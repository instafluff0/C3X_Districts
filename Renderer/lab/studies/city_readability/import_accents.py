#!/usr/bin/env python3
"""Import era accent buildings into a local, generic Lab pack.

Industrial cities in Civ III read through their smokestacks; modern ones
through high-rises. The installed source package holds suitable district
buildings. This offline step normalizes the selected entries with the same
compound-landmark adapter used for palaces. The output pack is ignored local
data and is never redistributed; runtime consumes only the compiled city pack.

    python3 Renderer/lab/studies/city_readability/import_accents.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.tools.asset_compiler.artdef_graph_resolver import DEFAULT_ASSETS_ROOT
from Renderer.tools.asset_compiler.compound_landmark_importer import _compile_asset
from Renderer.tools.asset_compiler.improvement_asset_importer import _shared_roots
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage

PACK = ROOT / "Renderer/packs/CityAccentsLab"
LANDMARKS = "Base/Platforms/Windows/BLPs/landmarks/"
# role -> source entries tried in order of the listed packages.
ACCENTS = {
    "industrial/factory": "DIS_PRD_Factory",
    "industrial/power_plant": "DIS_PRD_PowerPlant",
    "industrial/workshop": "DIS_PRD_IND_Workshop",
    "modern/workshop": "DIS_PRD_Mod_Workshop",
    "modern/electronics": "DIS_PRD_Modern_Japan_04ElectronicsFactory",
    "modern/apartment_lg": "DIS_NBH_Apartment_A_Lg",
    "modern/hotel_lg": "DIS_NBH_Hotel_A_Lg",
    "industrial/stack": "DIS_PRD_FactoryStack",
    "industrial/boiler_top": "LM_I_Boiler_Stack_Top",
    "modern/water_tower": "DIS_NBH_WaterTower",
    "industrial/silo": "IMP_Farm_IND_Bld_Silo",
}
PACKAGES = ("hero_buildings.blp", "city_buildings.blp", "tilebases.blp")


def asset_id(role: str) -> str:
    return "city/accent/" + role.replace("_", "-")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--pack", type=Path, default=PACK)
    parser.add_argument("--only", nargs="*", default=())
    args = parser.parse_args(argv)
    pack = args.pack.resolve()
    pack.relative_to(ROOT / "Renderer/packs")
    pack.mkdir(parents=True, exist_ok=True)
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema": "c3x.asset_pack.v0", "name": "CityAccentsLab",
        "display_name": "Lab city era accents",
        "source_policy": "Local licensed-source import; derived art is not redistributable.",
        "assets": {}, "runtime_integration": "not_enabled"}
    texture_cache: dict = {}
    report = {}
    for role, entry in ACCENTS.items():
        if args.only and role not in args.only:
            continue
        error = None
        for name in PACKAGES:
            relative = LANDMARKS + name
            source = args.assets_root / relative
            try:
                package = IndexedStaticPackage(source, entry)
                asset, evidence = _compile_asset(package, _shared_roots(args.assets_root, relative), pack,
                                                 entry, asset_id(role), 12.0, texture_cache,
                                                 terrain_edit_policy="preserve_unresolved",
                                                 auxiliary_uvs=True, omit_empty_material_draws=True)
            except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
                error = f"{name}: {exc}"
                continue
            manifest["assets"][asset_id(role)] = asset
            report[role] = {"entry": entry, "package": relative,
                            "source_sha256": hashlib.sha256(package.data).hexdigest(),
                            "parts": len(evidence.get("geometry", []))}
            print("IMPORTED", role, entry, name, flush=True)
            break
        else:
            report[role] = {"entry": entry, "error": error}
            print("FAILED", role, entry, error, flush=True)
    manifest["assets"] = dict(sorted(manifest["assets"].items()))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (pack / "import-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
