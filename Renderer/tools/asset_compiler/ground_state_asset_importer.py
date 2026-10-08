#!/usr/bin/env python3
"""Import local Civ VI art for Civ III ground states (pollution, craters, ruins).

Writes the ignored GroundStatesNormalized pack: the ruin props through the
compound-landmark importer, each clutter decal's exact quad (non-indexed, see
generic_decal_compiler.decode_decal_mesh) with its textures, and standalone
textures. The runtime never reads this pack; build_site_runtime composes it.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler.clutter_blp_extractor import (
    TYPE_INDEX_BUFFER, TYPE_TEXTURE, TYPE_VERTEX_BUFFER, decode_buffer_entry, decode_texture_entry,
    extract_civbig_texture, landmark_base_model)
from Renderer.tools.asset_compiler.compound_landmark_importer import compile_compound_landmarks, default_assets_root
from Renderer.tools.asset_compiler.generic_decal_compiler import (
    TYPE_DECAL, TYPE_DECAL_VECTOR, decode_decal_descriptor, decode_decal_mesh)
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAPPING = Path(__file__).with_name("ground_state_sources.json")
DEFAULT_PACK = ROOT / "Renderer/packs/GroundStatesNormalized"


def shared_data_for(package: str) -> list[str]:
    root = package.split("/Platforms/")[0]
    roots = [f"{root}/Platforms/Windows/BLPs/SHARED_DATA"]
    return roots + ([] if root == "Base" else ["Base/Platforms/Windows/BLPs/SHARED_DATA"])


def decal_quads(assets_root: Path, package_path: str, entry: str) -> list[dict]:
    """Every decal of a clutter entry: its triangles, footprint and texture names."""
    package = IndexedStaticPackage(assets_root / package_path, entry)
    package.select_direct_string(entry)
    _landmark, user_data, _base = landmark_base_model(package)
    vector = package.pointer_fields(user_data, TYPE_DECAL_VECTOR)[0][1]
    pointer = package.pointer_fields(vector, TYPE_DECAL)[0][1]
    count = package.allocations[pointer - 1]["element_count"]
    textures = package.unique_allocation(TYPE_TEXTURE)
    vertices = package.unique_allocation(TYPE_VERTEX_BUFFER)
    indices = package.unique_allocation(TYPE_INDEX_BUFFER)
    result = []
    for index in range(count):
        raw = package.array_element(pointer, index)
        descriptor = decode_decal_descriptor(raw, lambda k: decode_texture_entry(package, textures, k), 100.0,
                                             required_roles=("base_color",))
        buffer_index = struct.unpack_from("<I", raw, 0x3C)[0]
        vertex_entry = decode_buffer_entry(package, vertices, buffer_index, True)
        index_entry = decode_buffer_entry(package, indices, buffer_index, False)
        mesh, _ = decode_decal_mesh(raw, descriptor["footprint_bounds"],
                                    package.big_data(vertex_entry["offset"], vertex_entry["bytes"]),
                                    package.big_data(index_entry["offset"], index_entry["bytes"]),
                                    vertex_entry["count"], index_entry["count"])
        result.append({"mesh": mesh, "footprint": descriptor["footprint_bounds"],
                       "textures": {role: value["name"] for role, value in descriptor["textures"].items()}})
    return result


def extract(assets_root: Path, roots: list[str], name: str, pack: Path, asset_id: str) -> dict:
    for root in roots:
        source = assets_root / root / name
        if source.is_file():
            target = pack / "textures" / f"{asset_id}.dds"
            info = extract_civbig_texture(source, target)
            return {"texture": target.relative_to(pack).as_posix(), "source_entry": name,
                    "format": info["format_name"], "width": info["width"], "height": info["height"],
                    "source_sha256": info["source_sha256"]}
    raise ValueError(f"Missing ground-state texture {name}")


def build(assets_root: Path, mapping_path: Path, pack: Path) -> dict:
    mapping = json.loads(mapping_path.read_text())
    if mapping.get("schema") != "c3x.source_ground_state_mapping.v0":
        raise ValueError("Unsupported ground-state mapping")
    pack.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        props_mapping = Path(folder) / "props.json"
        props_mapping.write_text(json.dumps(mapping["props"]))
        compile_compound_landmarks(assets_root, props_mapping, pack / "props", pack / "props_report.json",
                                   False, "reject")
    manifest = {"schema": "c3x.ground_state_sources.v0", "props": "props/manifest.json",
                "decals": {}, "textures": {}, "runtime_source_dependency": None}
    for item in mapping["decals"]:
        decals = decal_quads(assets_root, item["source_package"], item["source_entry"])
        roots = shared_data_for(item["source_package"])
        records = []
        for number, decal in enumerate(decals):
            channels = {role: extract(assets_root, roots, name, pack, f"{item['asset_id']}_{number}_{role}")
                        for role, name in decal["textures"].items() if role in ("base_color", "height")}
            records.append({"vertices": decal["mesh"]["vertices"], "indices": decal["mesh"]["indices"],
                            "footprint": decal["footprint"], "channels": channels})
        manifest["decals"][item["asset_id"]] = {"source_entry": item["source_entry"], "decals": records}
    for item in mapping["textures"]:
        manifest["textures"][item["asset_id"]] = extract(assets_root, [item["shared_data"]], item["source_entry"],
                                                         pack, item["asset_id"])
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=default_assets_root())
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    args = parser.parse_args(argv)
    manifest = build(args.assets_root, args.mapping, args.pack)
    print(f"{len(manifest['decals'])} decal sources, {len(manifest['textures'])} textures -> {args.pack}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
