#!/usr/bin/env python3
"""Compile locally installed water and shoreline resources into a generic pack.

The importer is intentionally source-specific.  Its output is ordinary DDS plus
source-agnostic C3X JSON, so neither the renderer nor a shipped definition needs
the original package layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import c3x_asset_compiler
from Renderer.tools.asset_compiler import terrain_relief_builder


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common"
    / "Sid Meier's Civilization VI/Civ6.app/Contents/Assets"
)
DEFAULT_PACK = RENDERER_ROOT / "packs" / "TerrainNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "water_import.json"
DEFAULT_RELIEF_PACKAGE = (
    DEFAULT_ASSETS / "Base" / "Platforms" / "Windows" / "BLPs" /
    "terrain" / "TerrainElementSet_Base.blp"
)

# Each source file has an explicit generic runtime identity.  Hash-suffixed
# density ramps are retained as variants rather than silently deduplicated.
SOURCE_TEXTURES = {
    "surface/large_lean0": "TEXTURE_TER_Water_Bumps_Large_Lean0_0x9be17165",
    "surface/large_lean1": "TEXTURE_TER_Water_Bumps_Large_Lean1_0x9be17165",
    "surface/small_lean0": "TEXTURE_TER_Water_Bumps_Small_Lean0_0x9be17165",
    "surface/small_lean1": "TEXTURE_TER_Water_Bumps_Small_Lean1_0x9be17165",
    "surface/small_secondary_lean0": "TEXTURE_TER_Water_Bumps_Small2_Lean0_0x3879e4e5",
    "surface/small_secondary_lean1": "TEXTURE_TER_Water_Bumps_Small2_Lean1_0x3879e4e5",
    "surface/river_lean0": "TEXTURE_TER_Water_River_Bump_Lean0_0x9be17165",
    "surface/river_lean1": "TEXTURE_TER_Water_River_Bump_Lean1_0x9be17165",
    "surface/tiling_mask": "TEXTURE_WaterTiling_M",
    "surface/tiling_normal0": "TEXTURE_WaterTiling_N0",
    "surface/tiling_normal1": "TEXTURE_WaterTiling_N1",
    "surface/non_tiling_mask": "TEXTURE_WaterNonTiling_M",
    "surface/non_tiling_normal0": "TEXTURE_WaterNonTiling_N0",
    "surface/non_tiling_normal1": "TEXTURE_WaterNonTiling_N1",
    "surface/gloss": "TEXTURE_Water_G",
    "profiles/coast/dark": "TEXTURE_TER_WaterCoast_Density_DarkMap_0x961666cd",
    "profiles/coast/scatter": "TEXTURE_TER_WaterCoast_Density_ScatterMap_0x961666cd",
    "profiles/lake/dark": "TEXTURE_TER_WaterLake_Density_DarkMap_0x6b28231d",
    "profiles/lake/scatter": "TEXTURE_TER_WaterLake_Density_ScatterMap_0x6b28231d",
    "profiles/island/dark": "TEXTURE_TER_WaterGalapagos_Density_DarkMap_0x5b120645",
    "profiles/island/scatter": "TEXTURE_TER_WaterGalapagos_Density_ScatterMap_0x5b120645",
    "profiles/river_source/dark": "TEXTURE_TER_WaterRiverSource_Density_DarkMap_0x6c139095",
    "profiles/river_source/scatter": "TEXTURE_TER_WaterRiverSource_Density_ScatterMap_0x6c139095",
    "profiles/opaque/variant_01_dark": "TEXTURE_TER_WaterOpaque_Density_DarkMap_0x6fe39a3f",
    "profiles/opaque/variant_01_scatter": "TEXTURE_TER_WaterOpaque_Density_ScatterMap_0x6fe39a3f",
    "profiles/opaque/variant_02_dark": "TEXTURE_TER_WaterOpaque_Density_DarkMap_0x96092132",
    "profiles/opaque/variant_02_scatter": "TEXTURE_TER_WaterOpaque_Density_ScatterMap_0x96092132",
    "profiles/tropical/variant_01_dark": "TEXTURE_TER_WaterTropical_Density_DarkMap_0x4c351125",
    "profiles/tropical/variant_01_scatter": "TEXTURE_TER_WaterTropical_Density_ScatterMap_0x4c351125",
    "profiles/tropical/variant_02_dark": "TEXTURE_TER_WaterTropical_Density_DarkMap_0x5ce8e135",
    "profiles/tropical/variant_02_scatter": "TEXTURE_TER_WaterTropical_Density_ScatterMap_0x5ce8e135",
    "profiles/default/variant_01_dark": "TEXTURE_TER_Water_Density_DarkMap_0x961666cd",
    "profiles/default/variant_01_scatter": "TEXTURE_TER_Water_Density_ScatterMap_0x961666cd",
    "profiles/default/variant_02_dark": "TEXTURE_TER_Water_Density_DarkMap_0xe6b8e128",
    "profiles/default/variant_02_scatter": "TEXTURE_TER_Water_Density_ScatterMap_0xe6b8e128",
    "effects/mist": "TEXTURE_FX_WaterMist01",
    "effects/waterfall": "TEXTURE_FX_Waterfall_a_02",
    "effects/crash_foam": "TEXTURE_FX_Wave_Crash_Foam",
    "effects/ripples_primary": "TEXTURE_FXt_Water_Ripples",
    "effects/ripples_secondary": "TEXTURE_FXt_Water_Ripples02",
    "effects/splash_primary": "TEXTURE_FXt_Water_Splash_01",
    "effects/splash_secondary": "TEXTURE_FXt_Water_Splash_05",
    "effects/turbulence": "TEXTURE_FXt_Water_Turbulence_01",
    "effects/splash_sheet": "TEXTURE_FX_Splash",
    "shore/cliff_rocks_gloss": "TEXTURE_Cliff_Rocks_G",
    "shore/cliff_rocks_normal0": "TEXTURE_Cliff_Rocks_N0",
    "shore/cliff_rocks_normal1": "TEXTURE_Cliff_Rocks_N1",
    "shore/cliff_tile_alpha": "TEXTURE_Cliff_Tile_A",
    "shore/cliff_tile_gloss": "TEXTURE_Cliff_Tile_G",
    "shore/cliff_tile_normal0": "TEXTURE_Cliff_Tile_N0",
    "shore/cliff_tile_normal1": "TEXTURE_Cliff_Tile_N1",
    "shore/beach_blanket_base": "TEXTURE_Decal_Beach_Blanket_B",
    "shore/beach_blanket_fog": "TEXTURE_Decal_Beach_Blanket_FOW",
    "shore/ocean_decal_base": "TEXTURE_TER_Ocean_Decal_B",
    "shore/ocean_decal_height": "TEXTURE_TER_Ocean_Decal_H",
    "river/source_decal_base": "TEXTURE_TER_RiverSource_Decal_B",
    "river/source_decal_fog": "TEXTURE_TER_RiverSource_Decal_FOW",
    "river/source_decal_height": "TEXTURE_TER_RiverSource_Decal_H",
    "river/clutter_decal_base": "TEXTURE_TER_River_Clutter_Decal_B",
    "river/clutter_decal_height": "TEXTURE_TER_River_Clutter_Decal_H",
    "river/clutter_gloss": "TEXTURE_TER_River_Clutter_G",
    "river/clutter_normal0": "TEXTURE_TER_River_Clutter_N0",
    "river/clutter_normal1": "TEXTURE_TER_River_Clutter_N1",
    "reference/coast": "TEXTURE_Terrain_Coast",
    "reference/ocean": "TEXTURE_Terrain_Ocean",
    "reference/river": "TEXTURE_Terrain_River",
    "terrain/snow_decal_base": "TEXTURE_TER_Snow_Decal_B",
    "terrain/snow_decal_fog": "TEXTURE_TER_Snow_Decal_FOW",
    "terrain/snow_decal_gloss": "TEXTURE_TER_Snow_Decal_G",
    "terrain/snow_decal_height": "TEXTURE_TER_Snow_Decal_H",
}

# Optional installed expansion resources that are directly useful to Civ III's
# flood plains, flooded coast, rivers, and volcano terrain.  District/building
# assets stay deferred to their later ownership milestones.
OPTIONAL_TEXTURES = {
    "effects/waterfall_secondary": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_FX_Waterfall_a_01",
    "effects/flood_rapids_primary": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_FXt_WaterFloodRapids_Mask01",
    "effects/flood_rapids_secondary": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_FXt_WaterFloodRapids_Mask02",
    "effects/flood_wave": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_FXt_Wave_Flood",
    "profiles/terraced_spring/variant_01_dark": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_WaterPamukkale_Density_DarkMap_0x5e99002a",
    "profiles/terraced_spring/variant_01_scatter": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_WaterPamukkale_Density_ScatterMap_0x5e99002a",
    "profiles/terraced_spring/variant_02_dark": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_WaterPamukkale_Density_DarkMap_0xbe37c5b9",
    "profiles/terraced_spring/variant_02_scatter": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_WaterPamukkale_Density_ScatterMap_0xbe37c5b9",
    "profiles/river_flood/variant_01_dark": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_WaterRiverFlood_Density_DarkMap_0x3092839d",
    "profiles/river_flood/variant_01_scatter": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_WaterRiverFlood_Density_ScatterMap_0x3092839d",
    "profiles/river_flood/variant_02_dark": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_WaterRiverFlood_Density_DarkMap_0xaf308b15",
    "profiles/river_flood/variant_02_scatter": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_WaterRiverFlood_Density_ScatterMap_0xaf308b15",
    "shore/submerged_base": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_CoastSubmerged_B",
    "shore/submerged_height": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_CoastSubmerged_H",
    "flood/decal_base": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_Flood_Decal_B",
    "flood/decal_fog": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_Flood_Decal_FOW",
    "flood/decal_gloss": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_Flood_Decal_G",
    "flood/decal_height": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_Flood_Decal_H",
    "floodplain/decal_fog": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_Floodplain_Decal_FOW",
    "floodplain/decal_gloss": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_Floodplain_Decal_G",
    "floodplain/decal_height": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_Floodplain_Decal_H",
    "floodplain/grassland_base": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_Floodplain_Grassland_Decal_B",
    "floodplain/plains_base": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_TER_Floodplain_Plains_Decal_B",
    "volcano/base": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_Feature_Volcano_B",
    "volcano/height": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_Feature_Volcano_H",
    "volcano/active_base": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_Active_Volcano_B",
    "volcano/active_specular": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_Active_Volcano_S",
}
OPTIONAL_HEIGHT_BLOBS = {
    "terrain/coast_flood_height": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/BLOB_TER_CoastFlood_HMEdit",
    "terrain/coast_submerged_height": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/BLOB_TER_CoastSubmerged_HMEdit",
    "terrain/floodplain_height": "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/BLOB_FEATURE_Floodplains_HMEdit",
}

FLOAT_VALUE_RE = re.compile(
    r'<Element class="AssetObjects\.\.FloatValue">\s*<m_fValue>([-+0-9.eE]+)</m_fValue>\s*'
    r'<m_ParamName text="([^"]+)"/>', re.DOTALL,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_float_parameters(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    return {name: float(value) for value, name in FLOAT_VALUE_RE.findall(text)}


def runtime_texture_path(role: str) -> str:
    return f"textures/water/{role}.dds"


def extract_height_edit_blob(source: Path, target: Path) -> dict[str, Any]:
    """Decode the bounded 8-bit height samples in a Terrain_EditHeightmap blob."""
    data = source.read_bytes()
    if len(data) < 514 or data[:6] != b"CIVBIG" or b"Terrain_EditHeightmap" not in data[:128]:
        raise ValueError(f"Invalid terrain height-edit blob: {source}")
    width, height = struct.unpack_from("<II", data, 24)
    declared_payload = struct.unpack_from("<I", data, 8)[0]
    pixel_bytes = width * height * 2
    data_offset = len(data) - pixel_bytes
    if width < 1 or height < 1 or width > 4096 or height > 4096 or data_offset < 128:
        raise ValueError(f"Invalid terrain height-edit dimensions: {source}")
    if declared_payload + 440 != len(data):
        raise ValueError(f"Terrain height-edit payload size mismatch: {source}")
    encoded = data[data_offset:]
    if len(encoded) != pixel_bytes or any(encoded[0::2]):
        raise ValueError(f"Terrain height-edit samples exceed normalized 8-bit range: {source}")
    pixels = encoded[1::2]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(terrain_relief_builder.make_r8_dds(width, height, pixels))
    return {
        "width": width,
        "height": height,
        "mip_count": 1,
        "dxgi_format": 61,
        "format_name": "R8_UNORM",
        "payload_bytes": len(pixels),
    }


def _runtime_catalog(extracted: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "c3x.water_asset_catalog.v0",
        "textures": {
            role: {
                "texture": runtime_texture_path(role),
                "format": record["format_name"],
                "width": record["width"],
                "height": record["height"],
                "mip_count": record["mip_count"],
            }
            for role, record in sorted(extracted.items())
        },
        "relief": "water/relief.json",
        "profiles": "water/profiles.json",
    }


def _runtime_profiles(extracted: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "c3x.water_profiles.v0",
        "surface": {
            "large_lean": [runtime_texture_path("surface/large_lean0"), runtime_texture_path("surface/large_lean1")],
            "small_lean": [runtime_texture_path("surface/small_lean0"), runtime_texture_path("surface/small_lean1")],
            "small_secondary_lean": [runtime_texture_path("surface/small_secondary_lean0"), runtime_texture_path("surface/small_secondary_lean1")],
            "river_lean": [runtime_texture_path("surface/river_lean0"), runtime_texture_path("surface/river_lean1")],
            "fresnel_f0": 0.001,
            "environment_fresnel_exponent": 4.0,
            "refraction_scale": 0.4,
            "dynamic_specular_exponent": 850.0,
            "sun_specular_exponent": 5000.0,
        },
        "surf": {
            "foam": runtime_texture_path("effects/crash_foam"),
            "ripples": runtime_texture_path("effects/ripples_primary"),
            "turbulence": runtime_texture_path("effects/turbulence"),
            "whitecap_min": 0.0,
            "whitecap_max": 0.35,
            "whitecap_power": 4.0,
            "wave_width_px": 20.0,
            "wave_length_px": 128.0,
            "crash_distance_px": 8.0,
        },
        "profile_density": {
            role.removeprefix("profiles/"): runtime_texture_path(role)
            for role in sorted(extracted)
            if role.startswith("profiles/")
        },
    }


def _runtime_relief() -> dict[str, Any]:
    families = {}
    for family, channels in terrain_relief_builder.WATER_RELIEF_FAMILIES.items():
        families[family] = {
            role: [
                f"textures/water/relief/{family}/{role}_lod0.dds",
                f"textures/water/relief/{family}/{role}_lod1.dds",
            ]
            for role in channels
        }
    return {"schema": "c3x.water_relief.v0", "families": families}


def compile_water_pack(assets_root: Path, relief_package: Path, pack: Path, report: Path) -> dict[str, Any]:
    shared = assets_root / "Base" / "Platforms" / "Windows" / "BLPs" / "SHARED_DATA"
    if not shared.is_dir():
        raise ValueError(f"Missing shared texture directory: {shared}")

    extracted: dict[str, Any] = {}
    sources = [(role, shared / source_name) for role, source_name in SOURCE_TEXTURES.items()]
    sources.extend(
        (role, assets_root / relative)
        for role, relative in OPTIONAL_TEXTURES.items()
        if (assets_root / relative).is_file()
    )
    for role, source in sources:
        source_name = str(source.relative_to(assets_root))
        if not source.is_file():
            raise ValueError(f"Missing required water texture: {source_name}")
        target = pack / runtime_texture_path(role)
        info = c3x_asset_compiler.extract_civbig_to_dds(source, target)
        extracted[role] = {
            **info,
            "source": source_name,
            "output": runtime_texture_path(role),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }

    height_blob_count = 0
    for role, relative in OPTIONAL_HEIGHT_BLOBS.items():
        source = assets_root / relative
        if not source.is_file():
            continue
        target = pack / runtime_texture_path(role)
        info = extract_height_edit_blob(source, target)
        extracted[role] = {
            **info,
            "source": relative,
            "output": runtime_texture_path(role),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
        height_blob_count += 1

    relief_report = terrain_relief_builder.extract_water_relief_resources(
        relief_package, pack / "textures" / "water" / "relief"
    )
    catalog = _runtime_catalog(extracted)
    profiles = _runtime_profiles(extracted)
    relief = _runtime_relief()
    write_json(pack / "water" / "catalog.json", catalog)
    write_json(pack / "water" / "profiles.json", profiles)
    write_json(pack / "water" / "relief.json", relief)

    artdefs = assets_root / "Base" / "ArtDefs"
    source_parameters = {
        name: parse_float_parameters(artdefs / name)
        for name in ("Water.artdef", "Wave.artdef")
    }
    result = {
        "schema": "c3x.local_water_pack_build.v0",
        "source_root": str(assets_root),
        "texture_count": len(extracted),
        "base_texture_count": len(SOURCE_TEXTURES),
        "optional_texture_count": len(extracted) - len(SOURCE_TEXTURES),
        "height_blob_count": height_blob_count,
        "relief_texture_count": relief_report["compiled_texture_count"],
        "textures": extracted,
        "source_parameters": source_parameters,
        "relief_evidence": relief_report,
        "runtime": {
            "catalog": "water/catalog.json",
            "profiles": "water/profiles.json",
            "relief": "water/relief.json",
        },
    }
    write_json(report, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--relief-package", type=Path, default=DEFAULT_RELIEF_PACKAGE)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        result = compile_water_pack(args.assets, args.relief_package, args.pack, args.report)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Imported {result['texture_count']} water/shore textures and "
          f"{result['relief_texture_count']} relief channels")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
