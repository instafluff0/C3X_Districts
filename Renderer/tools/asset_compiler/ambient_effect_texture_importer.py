#!/usr/bin/env python3
"""Convert a conservative Civ VI ambient-effect texture slice into generic DDS assets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler.clutter_blp_extractor import extract_civbig_texture


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_ROOT = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common"
    / "Sid Meier's Civilization VI/Civ6.app/Contents/Assets"
)
DEFAULT_MAPPING = Path(__file__).with_name("ambient_effect_texture_sets.json")
DEFAULT_PACK = RENDERER_ROOT / "packs" / "AmbientEffectsNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "ambient_effects" / "texture_import.json"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_mapping(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "c3x.source_ambient_effect_texture_mapping.v0":
        raise ValueError("Ambient-effect texture mapping has an invalid schema")
    textures = document.get("textures")
    if not isinstance(textures, list) or not textures:
        raise ValueError("Ambient-effect texture mapping contains no textures")
    ids: set[str] = set()
    sources: set[str] = set()
    for item in textures:
        if not isinstance(item, dict) or set(item) != {"source_entry", "asset_id", "usage"}:
            raise ValueError("Ambient-effect texture mapping record is invalid")
        if not all(isinstance(item[key], str) and item[key] for key in item):
            raise ValueError("Ambient-effect texture mapping fields must be non-empty strings")
        if item["asset_id"] in ids or item["source_entry"] in sources:
            raise ValueError("Ambient-effect texture mapping contains a duplicate")
        ids.add(item["asset_id"])
        sources.add(item["source_entry"])
    return document


def compile_textures(
    mapping_path: Path, assets_root: Path, pack: Path, report_path: Path
) -> dict[str, Any]:
    mapping = load_mapping(mapping_path)
    source_root = assets_root / mapping["source_root"]
    manifest_textures = {}
    evidence = []
    for item in mapping["textures"]:
        source = source_root / item["source_entry"]
        relative = f"textures/effects/{_slug(item['asset_id'])}.dds"
        info = extract_civbig_texture(source, pack / relative)
        manifest_textures[item["asset_id"]] = {
            "texture": relative,
            "usage": item["usage"],
            "format": info["format_name"],
            "color_space": info["color_space"],
            "width": info["width"],
            "height": info["height"],
            "mip_count": info["mip_count"],
            "sampling": {"address_u": "clamp", "address_v": "clamp"},
            "animation_layout": "unresolved",
            "binding_status": "texture_only_no_particle_or_light_contract",
        }
        evidence.append(
            {
                "source_entry": item["source_entry"],
                "asset_id": item["asset_id"],
                "source": str(source),
                **info,
            }
        )
    manifest = {
        "schema": "c3x.ambient_effect_texture_pack.v0",
        "pack_id": "local.ambient_effects.normalized",
        "textures": manifest_textures,
        "runtime_source_dependency": None,
        "runtime_status": "not_enabled",
    }
    _write_json(pack / "manifest.json", manifest)
    report = {
        "schema": "c3x.ambient_effect_texture_import.v0",
        "pack": str(pack),
        "textures": evidence,
        "summary": {
            "textures": len(evidence),
            "bytes": sum((pack / manifest_textures[item["asset_id"]]["texture"]).stat().st_size for item in mapping["textures"]),
            "sprite_layouts_resolved": 0,
            "effect_bindings_resolved": 0,
        },
    }
    _write_json(report_path, report)
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
