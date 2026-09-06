#!/usr/bin/env python3
"""Compile locally installed terrain materials into a generic multi-terrain pack."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.preview import render_textured_patch
from Renderer.tools.asset_compiler import civblp_material_resolver
from Renderer.tools.asset_compiler import civblp_probe
from Renderer.tools.asset_compiler import grassland_pack_builder
from Renderer.tools.asset_compiler import terrain_geometry_resolver
from Renderer.tools.asset_compiler import terrain_relief_builder
from Renderer.tools.asset_compiler import water_pack_builder


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK = RENDERER_ROOT / "packs" / "TerrainNormalized"
DEFAULT_REPORT = RENDERER_ROOT / "preview" / "out" / "terrain_normalized_build.json"

MATERIAL_TARGETS = {
    "desert": "ART_DEF_TERRAIN_MATERIAL_DESERT",
    "plains": "ART_DEF_TERRAIN_MATERIAL_PLAINS",
    "grassland": "ART_DEF_TERRAIN_MATERIAL_GRASSLAND",
    "tundra": "ART_DEF_TERRAIN_MATERIAL_TUNDRA_BLEND",
    "flood_plain": "ART_DEF_TERRAIN_MATERIAL_PLAINS",
    "hills": "ART_DEF_TERRAIN_MATERIAL_GRASSHILL_TOP",
    "mountains": "ART_DEF_TERRAIN_MATERIAL_MTN_BASE",
    "forest": "ART_DEF_TERRAIN_MATERIAL_GRASSLAND",
    "jungle": "ART_DEF_TERRAIN_MATERIAL_GRASSMARSH",
    "marsh": "ART_DEF_TERRAIN_MATERIAL_GRASSMARSH",
    "volcano": "ART_DEF_TERRAIN_MATERIAL_MTN_DARK_BASE",
    "coast": "ART_DEF_TERRAIN_MATERIAL_SHALLOWS",
    "sea": "ART_DEF_TERRAIN_MATERIAL_COAST_WIDE",
    "ocean": "ART_DEF_TERRAIN_MATERIAL_OCEAN",
}

# Optional authored material used only on sufficiently elevated parts of the
# surface.  TerrainStyle.artdef leaves desert and tundra high materials empty;
# do not invent replacements for those omissions.
ELEVATED_MATERIAL_TARGETS = {
    "plains": "ART_DEF_TERRAIN_MATERIAL_PLAINS_TOP",
    "grassland": "ART_DEF_TERRAIN_MATERIAL_GRASS_TOP",
    "flood_plain": "ART_DEF_TERRAIN_MATERIAL_PLAINS_TOP",
    "mountains": "ART_DEF_TERRAIN_MATERIAL_MTN_TOP",
    "forest": "ART_DEF_TERRAIN_MATERIAL_GRASS_TOP",
    "volcano": "ART_DEF_TERRAIN_MATERIAL_MTN_TOP",
}

MOUNTAIN_LAYER_TARGETS = {
    "snow": "ART_DEF_TERRAIN_MATERIAL_MTN_SNOW",
    "desert_base": "ART_DEF_TERRAIN_MATERIAL_MTN_DESERT_BASE",
    "desert_stripe_1": "ART_DEF_TERRAIN_MATERIAL_MTN_DESERT_STRIPE01",
    "desert_stripe_2": "ART_DEF_TERRAIN_MATERIAL_MTN_DESERT_STRIPE02",
    "desert_stripe_3": "ART_DEF_TERRAIN_MATERIAL_MTN_DESERT_STRIPE03",
}

COAST_LAYER_TARGETS = {
    "beach": "ART_DEF_TERRAIN_MATERIAL_BEACH",
    "cliff": "ART_DEF_TERRAIN_MATERIAL_CLIFF",
    "cliff_white": "ART_DEF_TERRAIN_MATERIAL_CLIFF_WHITE",
}

COAST_LAYER_OCCURRENCES = {"beach": 0}

# Complete the useful material library even when a material is not yet the
# primary Civ III terrain identity.  This gives future biome, river, ice, salt,
# and island rules real authored inputs without adding source-specific runtime
# names or inventing substitutions.
AUXILIARY_MATERIAL_TARGETS = {
    "desert_hills": "ART_DEF_TERRAIN_MATERIAL_DESERT_HILLS",
    "island_base": "ART_DEF_TERRAIN_MATERIAL_GALAPAGOS_BASE",
    "ice": "ART_DEF_TERRAIN_MATERIAL_ICE",
    "plains_hills_top": "ART_DEF_TERRAIN_MATERIAL_PLAINSHILL_TOP",
    "river_bed": "ART_DEF_TERRAIN_MATERIAL_RIVER",
    "salt_flat": "ART_DEF_TERRAIN_MATERIAL_SALT_BASE",
    "snow": "ART_DEF_TERRAIN_MATERIAL_SNOW",
    "tundra_base": "ART_DEF_TERRAIN_MATERIAL_TUNDRA",
}
AUXILIARY_MATERIAL_OCCURRENCES = {"tundra_base": 0}

EXPLICIT_FALLBACKS = {
    "vegetation_models": "Civ III forest, jungle, and marsh bodies remain a complete retained overlay until normalized model dependencies are available",
    "transitions": "generic runtime edge blending replaces base-material boundaries; authored shoreline bodies remain Civ III-owned",
    "polar_ice": "requires complete geometry and water-edge dependencies",
    "landmarks": "requires exact landmark selector and complete replacement dependencies",
}

RELIEF_BY_TERRAIN = {
    "desert": ("surface_detail", 2.0, "continuous"),
    "plains": ("surface_detail", 2.5, "continuous"),
    "grassland": ("surface_detail", 3.0, "continuous"),
    "tundra": ("surface_detail", 2.0, "continuous"),
    "flood_plain": ("surface_detail", 1.5, "continuous"),
    "hills": ("surface_detail", 16.0, "connected_hills"),
    "mountains": ("mountain_atlas", 64.0, "mountain_atlas"),
    "forest": ("surface_detail", 3.0, "continuous"),
    "jungle": ("surface_detail", 3.5, "continuous"),
    "marsh": ("surface_detail", 1.0, "continuous"),
    "volcano": ("mountain_atlas", 70.0, "mountain_atlas"),
}

MATERIAL_OCCURRENCES = {
    "mountains": 0,
    "ocean": 1,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def infer_assets_root(package: Path) -> Path | None:
    for parent in package.resolve().parents:
        if parent.name == "Assets" and (parent / "Base" / "ArtDefs").is_dir():
            return parent
    return None


def build_local_terrain_pack(
    package: Path, mesh: Path, pack: Path, report: Path, relief_package: Path | None = None,
    assets_root: Path | None = None,
    material_occurrences: dict[str, int] | None = None,
) -> dict[str, Any]:
    normalized_mesh = json.loads(mesh.read_text(encoding="utf-8"))
    mesh_errors = terrain_geometry_resolver.validate_normalized_mesh(normalized_mesh)
    if mesh_errors:
        raise ValueError("Normalized mesh failed validation: " + "; ".join(mesh_errors))

    mesh_target = pack / "meshes" / "flat_terrain_patch.json"
    mesh_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mesh, mesh_target)

    relief_source = relief_package or package.with_name("TerrainElementSet_Base.blp")
    resolved_assets_root = assets_root or infer_assets_root(package)
    relief_stage = report.parent / "terrain_relief"
    relief_report = terrain_relief_builder.extract_relief_resources(relief_source, relief_stage)
    authored_relief_report = terrain_relief_builder.compile_authored_relief_sets(relief_source, pack)
    for role in terrain_relief_builder.RELIEF_OUTPUTS:
        target = pack / "textures" / f"relief_{role}.dds"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relief_stage / f"{role}.dds", target)

    assets: dict[str, Any] = {}
    source_evidence = []
    material_cache: dict[str, dict[str, tuple[dict[str, Any], str]]] = {}
    def material_channels(target: str, occurrence: int | None = None) -> dict[str, tuple[dict[str, Any], str]]:
        if target not in material_cache:
            binding = civblp_material_resolver.resolve_file(package, target, occurrence)
            texture_key = target.removeprefix("ART_DEF_TERRAIN_MATERIAL_").lower()
            channels: dict[str, tuple[dict[str, Any], str]] = {}
            for role in ("base_color", "height", "specular"):
                staging = report.parent / "terrain_textures" / f"{texture_key}_{role}.dds"
                info = grassland_pack_builder.extract_embedded_texture_role(
                    package, binding, role, staging
                )
                if role == "base_color":
                    texture = render_textured_patch.DdsBc3Texture.from_file(staging)
                    if texture.dxgi_format not in (77, 78) or texture.width <= 0 or texture.height <= 0:
                        raise ValueError(f"{target} did not produce a bounded BC3 base color")
                elif info["dxgi_format"] not in (80, 81):
                    raise ValueError(f"{target} {role} did not produce a bounded BC4 texture")
                relative = f"textures/{texture_key}_{role}.dds"
                target_path = pack / relative
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(staging, target_path)
                channels[role] = (info, relative)
            material_cache[target] = channels
        return material_cache[target]

    def channel_document(channel: tuple[dict[str, Any], str]) -> dict[str, Any]:
        info, relative = channel
        return {
            "texture": relative,
            "format": info["format_name"],
            "color_space": info["color_space"],
            "uv_channel": "uv0",
            "address_u": "repeat",
            "address_v": "repeat",
        }

    occurrences = MATERIAL_OCCURRENCES if material_occurrences is None else material_occurrences
    for terrain, target in MATERIAL_TARGETS.items():
        channels = material_channels(target, occurrences.get(terrain))
        texture_info, texture_relative = channels["base_color"]
        material_relative = f"materials/{terrain}.json"
        relief = RELIEF_BY_TERRAIN.get(terrain)
        material_document = {
            "schema": "c3x.material.v0",
            "name": terrain,
            "base_color": {
                "texture": texture_relative,
                "format": texture_info["format_name"],
                "color_space": texture_info["color_space"],
                "uv_channel": "uv0",
                "address_u": "repeat",
                "address_v": "repeat",
            },
            "surface_class": "water" if terrain in {"coast", "sea", "ocean"} else "land",
            "status": "normalized_local_import",
        }
        for role in ("height", "specular"):
            channel_info, channel_relative = channels[role]
            material_document[role] = {
                "texture": channel_relative,
                "format": channel_info["format_name"],
                "color_space": channel_info["color_space"],
                "uv_channel": "uv0",
                "address_u": "repeat",
                "address_v": "repeat",
            }
        elevated_target = ELEVATED_MATERIAL_TARGETS.get(terrain)
        elevated_channels = material_channels(elevated_target) if elevated_target else None
        if elevated_channels is not None:
            material_document["elevated"] = {}
            for role in ("base_color", "height", "specular"):
                channel_info, channel_relative = elevated_channels[role]
                material_document["elevated"][role] = {
                    "texture": channel_relative,
                    "format": channel_info["format_name"],
                    "color_space": channel_info["color_space"],
                    "uv_channel": "uv0",
                    "address_u": "repeat",
                    "address_v": "repeat",
                }
        if relief is not None:
            relief_role, height_scale, profile = relief
            material_document["relief"] = {
                "texture": f"textures/relief_{relief_role}.dds",
                "format": "R8_UNORM",
                "height_scale_px": height_scale,
                "profile": profile,
            }
        if terrain == "hills":
            material_document["authored_relief_set"] = "relief/hills.json"
        elif terrain in {"mountains", "volcano"}:
            material_document["authored_relief_set"] = "relief/mountains.json"
        if terrain == "mountains":
            material_document["authored_layers"] = {}
            for layer_name, layer_target in MOUNTAIN_LAYER_TARGETS.items():
                layer_channels = material_channels(layer_target)
                material_document["authored_layers"][layer_name] = {}
                for role in ("base_color", "height", "specular"):
                    channel_info, channel_relative = layer_channels[role]
                    material_document["authored_layers"][layer_name][role] = {
                        "texture": channel_relative,
                        "format": channel_info["format_name"],
                        "color_space": channel_info["color_space"],
                        "uv_channel": "uv0",
                        "address_u": "repeat",
                        "address_v": "repeat",
                    }
        elif terrain == "coast":
            material_document["authored_layers"] = {}
            for layer_name, layer_target in COAST_LAYER_TARGETS.items():
                layer_channels = material_channels(
                    layer_target, COAST_LAYER_OCCURRENCES.get(layer_name)
                )
                material_document["authored_layers"][layer_name] = {}
                for role in ("base_color", "height", "specular"):
                    channel_info, channel_relative = layer_channels[role]
                    material_document["authored_layers"][layer_name][role] = {
                        "texture": channel_relative,
                        "format": channel_info["format_name"],
                        "color_space": channel_info["color_space"],
                        "uv_channel": "uv0",
                        "address_u": "repeat",
                        "address_v": "repeat",
                    }
            if resolved_assets_root is not None:
                material_document["water_surface"] = {
                    name: {
                        "texture": water_pack_builder.runtime_texture_path(f"surface/{role}"),
                        "address_u": "repeat",
                        "address_v": "repeat",
                    }
                    for name, role in {
                        "large_lean0": "large_lean0",
                        "large_lean1": "large_lean1",
                        "small_lean0": "small_lean0",
                        "small_lean1": "small_lean1",
                    }.items()
                }
                material_document["water_surface"]["foam"] = {
                    "texture": water_pack_builder.runtime_texture_path("effects/crash_foam"),
                    "address_u": "repeat",
                    "address_v": "repeat",
                }
        write_json(
            pack / material_relative,
            material_document,
        )
        logical_id = f"terrain/{terrain}/base"
        assets[logical_id] = {
            "type": "terrain",
            "mesh": "meshes/flat_terrain_patch.json",
            "material": material_relative,
        }
        source_evidence.append(
            {
                "terrain": terrain,
                "source_target": target,
                "source_texture": texture_info["logical_name"],
                "dds_sha256": texture_info["dds_sha256"],
                "channels": {
                    role: {
                        "source_texture": channels[role][0]["logical_name"],
                        "dds_sha256": channels[role][0]["dds_sha256"],
                        "format": channels[role][0]["format_name"],
                    }
                    for role in ("base_color", "height", "specular")
                },
                "elevated_source_target": elevated_target,
                "elevated_channels": None if elevated_channels is None else {
                    role: {
                        "source_texture": elevated_channels[role][0]["logical_name"],
                        "dds_sha256": elevated_channels[role][0]["dds_sha256"],
                        "format": elevated_channels[role][0]["format_name"],
                    }
                    for role in ("base_color", "height", "specular")
                },
            }
        )

    material_library = {}
    auxiliary_evidence = []
    for name, target in AUXILIARY_MATERIAL_TARGETS.items():
        channels = material_channels(target, AUXILIARY_MATERIAL_OCCURRENCES.get(name))
        relative = f"materials/library/{name}.json"
        write_json(pack / relative, {
            "schema": "c3x.material.v0",
            "name": name,
            "base_color": channel_document(channels["base_color"]),
            "height": channel_document(channels["height"]),
            "specular": channel_document(channels["specular"]),
            "surface_class": "water" if name == "river_bed" else "land",
            "status": "normalized_local_import",
        })
        material_library[name] = relative
        auxiliary_evidence.append({
            "name": name,
            "source_target": target,
            "channels": {
                role: {
                    "source_texture": channels[role][0]["logical_name"],
                    "dds_sha256": channels[role][0]["dds_sha256"],
                    "format": channels[role][0]["format_name"],
                }
                for role in ("base_color", "height", "specular")
            },
        })

    water_report = None
    if resolved_assets_root is not None:
        water_report = water_pack_builder.compile_water_pack(
            resolved_assets_root,
            relief_source,
            pack,
            report.parent / "water_import.json",
        )

    runtime_coverage = {
        "schema": "c3x.terrain_runtime_coverage.v0",
        "mapped": sorted(assets),
        "fallbacks": [
            {"family": family, "ownership": "civ3", "reason": reason}
            for family, reason in sorted(EXPLICIT_FALLBACKS.items())
        ],
        "transition_policy": "deterministic material boundary; unsupported authored blends remain transparent",
    }
    manifest = {
        "schema": "c3x.asset_pack.v0",
        "name": "TerrainNormalized",
        "display_name": "Normalized Terrain Materials",
        "source_policy": "Local licensed-source import; derived textures are not redistributable.",
        "projection": {
            "tile_width_px": 128,
            "tile_height_px": 64,
            "height_scale_px": 54,
            "basis": {"x": [64, 32], "y": [-64, 32], "z": [0, -54]},
        },
        "assets": assets,
        "material_library": material_library,
        "relief_sets": authored_relief_report["runtime_sets"],
        "water": None if water_report is None else water_report["runtime"],
        "coverage": "runtime_coverage.json",
    }
    write_json(pack / "runtime_coverage.json", runtime_coverage)
    write_json(pack / "manifest.json", manifest)
    independence_errors = grassland_pack_builder.validate_runtime_independence(pack)
    if independence_errors:
        raise ValueError("Runtime pack is not source-independent: " + "; ".join(independence_errors))

    build_report = {
        "schema": "c3x.local_terrain_pack_build.v0",
        "source_package": str(package),
        "normalized_pack": str(pack),
        "mapped_count": len(assets),
        "fallback_count": len(EXPLICIT_FALLBACKS),
        "source_evidence": source_evidence,
        "auxiliary_material_evidence": auxiliary_evidence,
        "relief_evidence": relief_report,
        "authored_relief_evidence": authored_relief_report,
        "water_evidence": water_report,
        "runtime_coverage": runtime_coverage,
    }
    write_json(report, build_report)
    return build_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=civblp_probe.DEFAULT_PACKAGE)
    parser.add_argument("--mesh", type=Path, default=terrain_geometry_resolver.DEFAULT_MESH)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--assets-root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_local_terrain_pack(
            args.package, args.mesh, args.pack, args.report, assets_root=args.assets_root
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.pack / 'manifest.json'} with {result['mapped_count']} mapped terrain materials")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
