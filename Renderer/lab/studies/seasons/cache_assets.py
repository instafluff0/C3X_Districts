#!/usr/bin/env python3
"""Preserve bounded seasonal art before removing the source installation.

The local ignored cache is an input, not disposable Lab output. Snow descriptors
are compiled to generic meshes/materials; no source package is copied. Only the
small blossom payload is kept raw for exact reproducibility of its adaptation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CACHE = HERE / "assets"
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.seasons.asset_inputs import assets_root

BLOSSOMS = "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_FX_Blossoms"
FLOWERS = {
    "blossoms": BLOSSOMS,
    "flowered-color": "Base/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_DiffuseTint_Foliage_Bld_Flowered_Color_B_null",
    "flowered-white": "Base/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_DiffuseTint_Foliage_Bld_Flowered_White_B_null",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(cache=CACHE):
    from Renderer.tools.asset_compiler.grassland_pack_builder import validate_runtime_independence
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest.get("schema") != "c3x.lab.seasonal_cache.v1":
        raise ValueError("Unsupported seasonal cache schema")
    for row in manifest["files"]:
        path = (cache / row["path"]).resolve()
        path.relative_to(cache.resolve())
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Seasonal input changed: " + row["path"])
    errors = validate_runtime_independence(cache / "snow-decals")
    if errors:
        raise ValueError("Snow pack validation failed: " + "; ".join(errors))
    pack = json.loads((cache / "snow-decals/manifest.json").read_text())
    if len(pack["assets"]) != 11:
        raise ValueError("Seasonal cache must retain all eleven snow variants")
    return manifest


def preserve(root):
    from Renderer.tools.asset_compiler.generic_decal_compiler import compile_decal_pack
    from Renderer.tools.asset_compiler.clutter_blp_extractor import extract_civbig_texture
    if CACHE.exists():
        return verify()
    if shutil.disk_usage(HERE).free < 256 * 1024**2:
        raise ValueError("Less than 256 MiB free for the bounded seasonal cache")
    names = [f"TER_Snow_Decal{i:02d}" for i in range(1, 10)] + ["TER_Snow_Decal_Dark01", "TER_Snow_Decal_Dark02"]
    mapping = {
        "schema": "c3x.source_decal_mapping.v0", "source_units_per_tile": 12.0,
        "sources": {"package": "Base/Platforms/Windows/BLPs/environment/clutter.blp",
                    "shared_data": "Base/Platforms/Windows/BLPs/SHARED_DATA",
                    "artdef": "Base/ArtDefs/Clutter.artdef"},
        "groups": [{"group_id": "terrain/snow/surface", "artdef_set": "CLUTTER_SNOW", "collection": "Plants",
                    "assets": [{"source_asset": name, "asset_id": f"terrain/snow/decal_{i:02d}"}
                               for i, name in enumerate(names, 1)]}],
    }
    # Build transactionally beside the final cache. The compiler's temporary
    # report contains absolute paths and is deliberately not retained.
    with tempfile.TemporaryDirectory(prefix="seasonal-cache-", dir=HERE) as directory:
        work = Path(directory)
        payload = work / "assets"
        payload.mkdir()
        spec = work / "mapping.json"
        spec.write_text(json.dumps(mapping))
        report = compile_decal_pack(root, spec, payload / "snow-decals", work / "report.json")
        pack_path = payload / "snow-decals/manifest.json"
        pack = json.loads(pack_path.read_text())
        pack.update(name="SeasonalSnowDecals", display_name="Seasonal Snow Decals")
        pack_path.write_text(json.dumps(pack, indent=2) + "\n")
        flower_rows = []
        for name, relative in FLOWERS.items():
            target = payload / "flowers" / (name + ".dds")
            info = extract_civbig_texture(root / relative, target)
            flower_rows.append({"role": name, "source": relative, "source_sha256": info["source_sha256"],
                                "path": target.relative_to(payload).as_posix(),
                                "width": info["width"], "height": info["height"], "format": info["dxgi_format"]})
        source = payload / "flowers/blossoms.source"
        shutil.copyfile(root / BLOSSOMS, source)
        files = [{"path": p.relative_to(payload).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
                 for p in sorted(payload.rglob("*")) if p.is_file()]
        manifest = {
            "schema": "c3x.lab.seasonal_cache.v1",
            "retention": "Preserved local licensed inputs; keep when cleaning Lab output or uninstalling the source game.",
            "scope": "Eleven normalized snow decal variants with meshes, UVs, placement and four shared channels; three flower textures; exact blossom source.",
            "snow_source_mapping": mapping,
            "snow_source_hashes": {k: {"path": mapping["sources"][k], "sha256": v["sha256"]}
                                   for k, v in report["sources"].items() if v["sha256"]},
            "flowers": flower_rows, "files": files, "bytes": sum(row["bytes"] for row in files),
            "runtime_independence": "Generic snow pack validated; blossom raw format is offline adaptation input only.",
        }
        (payload / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify(payload)
        payload.rename(CACHE)
    return verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Check preserved inputs without accessing the source game")
    parser.add_argument("--assets-root", type=Path)
    args = parser.parse_args()
    manifest = verify() if args.verify else preserve(args.assets_root or assets_root())
    print(f"PASS seasonal cache: {len(manifest['files'])} files, {manifest['bytes'] / 1024**2:.2f} MiB; no source installation needed for verification")


if __name__ == "__main__":
    main()
