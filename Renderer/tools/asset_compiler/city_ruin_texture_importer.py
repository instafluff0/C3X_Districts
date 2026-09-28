#!/usr/bin/env python3
"""Import local Civ VI ruin-debris textures as source-independent city candidates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler.clutter_blp_extractor import extract_civbig_texture


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_ROOT = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common"
    / "Sid Meier's Civilization VI/Civ6.app/Contents/Assets"
)
DEFAULT_MAPPING = Path(__file__).with_name("city_ruin_texture_sets.json")
DEFAULT_PACK = RENDERER_ROOT / "packs/CityRuinsCandidates"
DEFAULT_REPORT = RENDERER_ROOT / "preview/out/city_ruins/texture_import.json"
SAFE_ID = re.compile(r"^city/ruins/[a-z0-9_]+$")


def load_mapping(path: Path) -> dict:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping.get("schema") != "c3x.source_city_ruin_texture_mapping.v0":
        raise ValueError("Unsupported city-ruin texture mapping")
    if mapping.get("source_root") != "Base/Platforms/Windows/BLPs/SHARED_DATA":
        raise ValueError("Unexpected city-ruin source root")
    textures = mapping.get("textures")
    if not isinstance(textures, list) or not textures:
        raise ValueError("City-ruin mapping has no textures")
    ids: set[str] = set()
    sources: set[str] = set()
    for item in textures:
        if not isinstance(item, dict) or set(item) != {"source_entry", "asset_id", "usage"}:
            raise ValueError("Invalid city-ruin texture record")
        source, asset_id, usage = (item[key] for key in ("source_entry", "asset_id", "usage"))
        if not isinstance(source, str) or not re.fullmatch(r"TEXTURE_[A-Za-z0-9_]+", source):
            raise ValueError("Invalid city-ruin source entry")
        if not isinstance(asset_id, str) or not SAFE_ID.fullmatch(asset_id):
            raise ValueError("Invalid city-ruin asset ID")
        if not isinstance(usage, str) or not usage or source in sources or asset_id in ids:
            raise ValueError("Invalid or duplicate city-ruin texture mapping")
        sources.add(source)
        ids.add(asset_id)
    return mapping


def compile_textures(mapping_path: Path, assets_root: Path, pack: Path, report_path: Path) -> dict:
    mapping = load_mapping(mapping_path)
    source_root = assets_root / mapping["source_root"]
    textures = {}
    evidence = []
    for item in mapping["textures"]:
        source = source_root / item["source_entry"]
        relative = f"textures/ruins/{item['asset_id'].split('/')[-1]}.dds"
        info = extract_civbig_texture(source, pack / relative)
        textures[item["asset_id"]] = {
            "texture": relative,
            "usage": item["usage"],
            "format": info["format_name"],
            "color_space": info["color_space"],
            "width": info["width"],
            "height": info["height"],
            "mip_count": info["mip_count"],
            "sampling": {"address_u": "clamp", "address_v": "clamp"},
            "binding_status": "candidate_texture_only",
        }
        evidence.append({"source_entry": item["source_entry"], "asset_id": item["asset_id"], **info})
    manifest = {
        "schema": "c3x.city_ruin_texture_pack.v0",
        "pack_id": "local.city_ruins.candidates",
        "textures": textures,
        "runtime_source_dependency": None,
        "runtime_status": "not_enabled",
    }
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "schema": "c3x.city_ruin_texture_import.v0",
        "source_package_evidence": mapping["source_package_evidence"],
        "textures": evidence,
        "summary": {"textures": len(evidence), "bytes": sum((pack / item["texture"]).stat().st_size for item in textures.values())},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        report = compile_textures(args.mapping, args.assets_root, args.pack, args.report)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
