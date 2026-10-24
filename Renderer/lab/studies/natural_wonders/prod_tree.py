#!/usr/bin/env python3
"""Production port of the natural-wonder kits, staged as anchored patches.

The integration (docs/natural_wonder_rendering.md, "Game Integration") touches
files other sessions edit at the same time, so it is developed here as anchored
edits, the way stamp_tree.py stages the Lab study:

    python3 Renderer/lab/studies/natural_wonders/prod_tree.py prepare   # private tree + patches
    python3 Renderer/lab/studies/natural_wonders/prod_tree.py build     # VM compile of that tree
    python3 Renderer/lab/studies/natural_wonders/prod_tree.py check     # anchors against the checkout
    python3 Renderer/lab/studies/natural_wonders/prod_tree.py apply     # write into the checkout

`apply` is for the reviewed promotion only (the integration lead's review, after
the frame ABI owner's change has landed). Every edit must match its anchor exactly
once in the current checkout, so a moved anchor fails loudly instead of landing
in the wrong place.

Effect kinds (city effect layer, material 90 + kind): 4 geyser, 5 waterfall
ribbon, 9 spray; records of kind 7 (lake) and 8 (fjord channel) are water drawn by
the production water layers. Kind 6 is Winter's flurry.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.mountains import private_tree
from Renderer.lab.studies.natural_wonders import (prod_api, prod_bodies, prod_injected, prod_lab, prod_labels,
                                                  prod_relief, prod_seasons, prod_topology, prod_water, stamp_tree)

NAME = "prod"
private_tree.OUT = ROOT / "Renderer/lab/out/natural_wonders"

EFFECTS = (stamp_tree.GEYSER_WATERFALL
           .replace("// Lab only (studies/natural_wonders): a geyser (94).",
                    "// Natural wonders (docs/natural_wonder_rendering.md): a geyser (94).")
           .replace("// Lab only (studies/natural_wonders): a waterfall sheet (95)",
                    "// Natural wonders: a waterfall sheet (95)")
           .replace("where the spray (96) takes over.", "where the spray (99) takes over.")
           .replace("// Lab only (studies/natural_wonders): spray (96)", "// Natural wonders: spray (99)"))

REGIONS = r"""    // Natural wonders (docs/natural_wonder_rendering.md): where no volcano owns the
    // relief, the owner channel carries a wonder stamp's region weights (x eases to
    // the tile's own ground on vegetated slopes, y to the wonder's special material)
    // and w its region kind: 1 snow, 2 travertine, 3 banded sandstone, 4 a
    // stratovolcano's brown flanks under snow.
    if (input.material.y > 1.5 && input.volcano_owner.z < 0.001 && input.volcano_owner.w > 0.5) {
        float wonder_kind = round(input.volcano_owner.w);
        float wonder_ground = saturate(input.volcano_owner.x);
        float wonder_special = saturate(input.volcano_owner.y);
        // Civ VI's vegetated host ground: the skin's grass, since a stamped tile's own
        // biome channels describe the mountain.
        float3 vegetated = ground_surface_sample(GrassColor, input, 0.43, float2(.31, .17), false).rgb;
        float vegetated_height = ground_surface_sample(GrassHeight, input, 0.43, float2(.31, .17), false).r;
        albedo = lerp(albedo, vegetated, wonder_ground);
        height_detail = lerp(height_detail, vegetated_height, wonder_ground);
        float grain = saturate(height_detail * 1.6);
        float3 snow_rock = triplanar(SnowColor, input.world * MTN_ROCK_SCALE, geometric);
        float3 special;
        [branch] if (wonder_kind < 1.5) {
            special = snow_rock;
        } else if (wonder_kind < 2.5) {
            special = lerp(ground_albedo, float3(0.80, 0.77, 0.69), 0.75) * lerp(0.80, 1.06, grain);
        } else if (wonder_kind < 3.5) {
            float band = frac(input.world.z * 7.0 + height_detail * 0.4);
            special = lerp(float3(0.42, 0.16, 0.08), float3(0.58, 0.26, 0.14), smoothstep(0.25, 0.75, band)) *
                lerp(0.86, 1.04, grain);
        } else {
            // The special weight ramps volcanic brown (0.5) to snow (1).
            special = lerp(float3(0.40, 0.26, 0.17) * lerp(0.72, 1.06, grain), snow_rock, saturate(wonder_special * 2 - 1));
            wonder_special = saturate(wonder_special * 2);
        }
        albedo = lerp(albedo, special, wonder_special);
    }
"""

COLLAR = stamp_tree.PATCHES["Renderer/lab/shared/shaders/hydrology/coastal_waves.hlsl"][0][1].replace(
    "    // Lab only (studies/natural_wonders): a water wonder's ribbon (base terrain 2 + seed) is a\n",
    "    // Natural wonders: a water wonder's ribbon (base terrain 2 + seed) is a\n")

# The Lab's wonder_effects() (stamp_tree.py), with production kinds: spray is 9, and
# only lake (7) and channel (8) records belong to the water layers.
_LAB_EFFECTS = stamp_tree.PATCHES["Renderer/native/city_fidelity/compiler.h"][1][1]
WONDER_EFFECTS = (_LAB_EFFECTS
    .replace("// Lab only (studies/natural_wonders): a natural wonder's attached effects. A record is\n",
             "// A natural wonder's attached effects (docs/natural_wonder_rendering.md). A record is\n")
    .replace("        if(e[0]>6.5f)continue;  // still water: the production water layer (c3x_renderer.cpp)\n",
             "        if(e[0]>float(effect_lake)-.5f && e[0]<float(effect_channel)+.5f)continue;  // water layers (c3x_renderer.cpp)\n")
    .replace("        if(e[0]>4.5f && e[0]<5.5f){", "        if(e[0]==float(effect_waterfall)){"))
assert WONDER_EFFECTS != _LAB_EFFECTS and "Lab only" not in WONDER_EFFECTS

# New files live under prod/ at their checkout paths.
PROD = Path(__file__).resolve().parent / "prod"
NEW_FILES = {path.relative_to(PROD).as_posix(): path for path in sorted(PROD.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts}

PATCHES = {
    "Renderer/native/c3x_renderer.cpp": [
        ('#include "render_core/compressed_world_store.h"\n',
         '#include "render_core/compressed_world_store.h"\n#include "render_core/season_stand.h"\n'),
        ("    // Seasons D4: 0 every season; 1 a summer forest stand hidden in Winter;\n"
         "    // 2 its precomputed Winter stand, drawn only in Winter.\n    unsigned char season_stand=0;",
         "    // Seasons: the seasons this chunk draws in (render_core/season_stand.h);\n"
         "    // 0 every season. Forest stands and natural-wonder pines.\n    unsigned char season_stand=0;"),
        ("    // Frame season 2: forest tiles draw their Winter stands (seasons D4).\n    bool winter_season=false;",
         "    // The frame's season (0..3) and the season stands its chunks carry (bit m:\n"
         "    // a chunk with stand m exists).\n    int season_index=0;std::uint16_t season_stands_used=0;"),
        ("    // Seasons D4: whether a chunk draws in the current season.\n"
         "    bool season_shown(CachedVertexChunk const& chunk) const {\n"
         "        return !chunk.season_stand || (chunk.season_stand==2)==winter_season;\n    }",
         "    // Seasons: whether a chunk draws in the current season.\n"
         "    bool season_shown(CachedVertexChunk const& chunk) const {\n"
         "        return c3x_renderer::render_core::season_stand_shown(chunk.season_stand,season_index);\n    }"),
        ("        winter_season=frame_season(frame)==2;", "        season_index=frame_season(frame)&3;"),
        ("                    chunk.season_stand=stand?2:winter_stand?1:0;",
         "                    chunk.season_stand=stand?c3x_renderer::render_core::season_stand_winter:\n"
         "                        winter_stand?c3x_renderer::render_core::season_stand_outside_winter:0;\n"
         "                    season_stands_used|=std::uint16_t(1u<<chunk.season_stand);")],
    "Renderer/sandbox/fresh_pipeline.h": [
        ("        // Seasons D4: entering or leaving Winter swaps the forest stands. The\n",
         "        // Seasons: a change that shows or hides any season stand (forest and\n"
         "        // natural-wonder pines at Winter's edges) swaps the stands. The\n"),
        ("        renderer.winter_season=season==2;\n        if(previous_season!=INT_MIN && (previous_season==2)!=(season==2)){",
         "        renderer.season_index=season&3;\n        if(previous_season!=INT_MIN &&\n"
         "           c3x_renderer::render_core::season_stands_differ(renderer.season_stands_used,previous_season,season)){")],
    "Renderer/native/test_water_motion.py": [
        ('#include "Renderer/native/environment_runtime.h"\n#include <cstdio>',
         '#include "Renderer/native/environment_runtime.h"\n#include "Renderer/native/render_core/season_stand.h"\n#include <cstdio>'),
        (" struct Renderer {bool winter_season=false;float lighting_hour=-1.f;",
         " struct Renderer {int season_index=0;std::uint16_t season_stands_used=std::uint16_t(\n"
         "   (1u<<c3x_renderer::render_core::season_stand_outside_winter)|(1u<<c3x_renderer::render_core::season_stand_winter));\n"
         "  float lighting_hour=-1.f;"),
        (" assert(pipeline.retirements==0 && !pipeline.renderer.winter_season);",
         " assert(pipeline.retirements==0 && pipeline.renderer.season_index!=2);"),
        (" assert(pipeline.renderer.winter_season && pipeline.retirements==1 && pipeline.copied.season==2);",
         " assert(pipeline.renderer.season_index==2 && pipeline.retirements==1 && pipeline.copied.season==2);"),
        (" assert(!pipeline.renderer.winter_season && pipeline.retirements==2);",
         " assert(pipeline.renderer.season_index==1 && pipeline.retirements==2);")],
    "Renderer/native/city_fidelity/runtime.h": [
        ("enum EffectKind:unsigned {effect_flame=0,effect_smoke=1,effect_night_light=2,effect_volcano_plume=3,effect_kind_count};",
         "enum EffectKind:unsigned {effect_flame=0,effect_smoke=1,effect_night_light=2,effect_volcano_plume=3,effect_kind_count};\n"
         "// Natural-wonder effect records (NaturalWonderRuntime): kinds outside the city packs'\n"
         "// range; 6 is Winter's flurry. Lake and channel records are water, not quads.\n"
         "enum WonderEffectKind:unsigned {effect_geyser=4,effect_waterfall=5,effect_lake=7,effect_channel=8,effect_spray=9};")],
    "Renderer/native/city_fidelity/compiler.h": [
        ("        bool rises=e.kind==float(effect_smoke) || e.kind==float(effect_volcano_plume);",
         "        // Natural-wonder geysers and spray rise from their foot too.\n"
         "        bool rises=e.kind==float(effect_smoke) || e.kind==float(effect_volcano_plume) ||\n"
         "            e.kind==float(effect_geyser) || e.kind==float(effect_spray);"),
        ("    return output.chunks.size()>before;\n}\ntemplate<class Height,class Project,class Stop=ContinueCompilation,class Site=EverySite>",
         WONDER_EFFECTS)],
    "Renderer/lab/shared/shaders/objects/city_scene_material.hlsl": [
        ("float4 shape,float3 look){", "float4 shape,float3 look,bool keep=false){"),
        ("  clip(alpha-.004);\n  float grey=", "  if(!keep)clip(alpha-.004);\n  float grey="),
        ("float4 q8_city_effect(FeaturePixelInput p){", EFFECTS + "float4 q8_city_effect(FeaturePixelInput p){"),
        (" if(kind>2.5){\n  // A volcano's plume (93)",
         " if(kind>3.5 && kind<4.5)return q8_effect_geyser(q,pixel,t,seed,strength,night);\n"
         " if(kind>4.5 && kind<5.5)return q8_effect_waterfall(q,pixel,t,seed,strength,night);\n"
         " if(kind>8.5 && kind<9.5)return q8_effect_spray(q,pixel,t,seed,strength,night);\n"
         " if(kind>2.5){\n  // A volcano's plume (93)"),
        # Quad units per pixel along the effect's own axes: identical for screen-aligned quads,
        # and right for the waterfall ribbon, which runs down the face at any screen angle.
        (" float2 pixel=float2(max(abs(ddx(p.uv.x)),1e-5),max(abs(ddy(p.uv.y)),1e-5));",
         " float2 pixel=float2(max(length(float2(ddx(p.uv.x),ddy(p.uv.x))),1e-5),\n"
         "                     max(length(float2(ddx(p.uv.y),ddy(p.uv.y))),1e-5));")],
    "Renderer/lab/shared/shaders/hydrology/coastal_waves.hlsl": [
        ("    float seed=input.base_terrain,seed2=input.real_terrain;\n", COLLAR)],
    "Renderer/lab/shared/shaders/relief/beauty_mountain.hlsl": [
        ("        albedo = lerp(albedo, graded, rock_albedo_coverage);\n    }\n",
         "        albedo = lerp(albedo, graded, rock_albedo_coverage);\n    }\n" + REGIONS)],
}

for _module in (prod_relief, prod_water, prod_api, prod_injected, prod_labels, prod_lab, prod_bodies, prod_topology,
                prod_seasons):
    for _relative, _edits in _module.PATCHES.items():
        PATCHES.setdefault(_relative, []).extend(_edits)


def edit(text: str, relative: str, edits: list[tuple[str, str]]) -> str:
    for old, new in edits:
        if text.count(old) != 1:
            raise SystemExit(f"{relative}: anchor not found exactly once: {old[:70]!r}")
        text = text.replace(old, new)
    return text


def prepare() -> None:
    root = private_tree.tree(NAME)
    private_tree.own(NAME, list(PATCHES) + list(NEW_FILES))
    for relative in PATCHES:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes((ROOT / relative).read_bytes())
    private_tree.sync(NAME)
    for relative, edits in PATCHES.items():
        path = root / relative
        path.write_text(edit(path.read_text(), relative, edits))
    for relative, source in NEW_FILES.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(source.read_bytes())
    print("patched", len(PATCHES), "files and added", len(NEW_FILES), "in", root.relative_to(ROOT))


def check() -> None:
    for relative, edits in PATCHES.items():
        edit((ROOT / relative).read_text(), relative, edits)
    clashes = [relative for relative in NEW_FILES if (ROOT / relative).exists()]
    if clashes:
        raise SystemExit(f"new files already exist in the checkout: {clashes}")
    print("all", sum(map(len, PATCHES.values())), "anchors match the checkout once;", len(NEW_FILES), "new files free")


def apply() -> None:
    check()
    for relative, edits in PATCHES.items():
        path = ROOT / relative
        path.write_text(edit(path.read_text(), relative, edits))
    for relative, source in NEW_FILES.items():
        (ROOT / relative).write_bytes(source.read_bytes())
    print("applied", len(PATCHES), "files and", len(NEW_FILES), "new files to the checkout")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("prepare", "build", "check", "apply"))
    args = parser.parse_args()
    {"prepare": prepare, "build": lambda: private_tree.build(NAME), "check": check, "apply": apply}[args.action]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
