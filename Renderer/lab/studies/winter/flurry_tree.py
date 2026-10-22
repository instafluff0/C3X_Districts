#!/usr/bin/env python3
"""Private Lab tree: a gentle snow flurry over mountain tiles, drawn by Renderer64.

The flurry is one more attached effect in the city effect layer, the per-frame
layer that animates chimney smoke and volcano plumes over the retained terrain:

- Renderer/native/city_fidelity/compiler.h: mountain_flurry() emits one rising
  camera-facing quad (kind 6, material 96) standing on the tile's front corner,
  so its flakes draw in front of the tile's own mountain while nearer relief
  can still hide them (the effect layer reads the retained depth);
- Renderer/native/object_preparation.h and source_fidelity/geometry.h: both
  object paths emit it for mountain tiles (real terrain 6) while the Lab switch
  C3X_LAB_MOUNTAIN_FLURRY is set (production would gate on the Winter season);
- Renderer/lab/shared/shaders/objects/city_scene_material.hlsl: the flurry
  shader (three parallax layers of hashed cells, as flurry.py), regenerated
  into city_fidelity/city.hlsl;
- Renderer/native/biq_preview.cpp: attached-effect frames at 15 fps
  (C3X_LAB_EFFECT_FRAMES), the natural-wonders study's clip loop;
- Renderer/native/c3x_renderer.cpp: C3X_LAB_WINTER_PINES selects the authored
  snowy-pine variety for every pine forest (production: only on tundra).

The checkout is never written; the tree lives in Renderer/lab/out/winter/tree-flurry/.

    python3 Renderer/lab/studies/winter/flurry_tree.py prepare   # sync + patch
    python3 Renderer/lab/studies/winter/flurry_tree.py build     # VM compile
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.mountains import private_tree

NAME = "flurry"
private_tree.OUT = ROOT / "Renderer/lab/out/winter"

FLURRY_SHADER = r'''// Lab only (studies/winter): a gentle flurry of small snowflakes over a mountain tile (96).
// Three parallax layers of cells fall through the quad; each cell holds at most one
// flake at a hashed position with a hashed sway phase. Sizes are screen pixels at tile
// width 128, scaled with the zoom; the column fades in at its top and out at its foot.
float q8_flurry_hash(float2 p,float s){return frac(sin(dot(p,float2(127.1,311.7))+s*74.7)*43758.5453);}
float4 q8_effect_flurry(float2 q,float2 pixel,float t,float seed,float strength,float night){
  float2 s=float2(q.x/pixel.x,q.y/pixel.y);
  float zoom=(2/pixel.x)/(1.05*128);
  float alpha=0;
  [unroll]for(int layer=0;layer<3;layer++){
   float cell=(layer==0?22.:layer==1?15.:10.)*zoom,speed=(layer==0?20.:layer==1?14.:9.)*zoom;
   float radius=(layer==0?1.25:layer==1?.95:.7)*zoom,sway=(layer==0?3.:layer==1?2.2:1.5)*zoom;
   float opacity=layer==0?.9:layer==1?.7:.5,share=layer==2?.5:.55;
   float2 g=float2(s.x,s.y+speed*t);
   float column=floor(g.x/cell);
   g.x+=sway*sin(6.2832*(t/2.6+q8_flurry_hash(float2(column,layer),seed+3.1)));
   float2 id=floor(g/cell),f=frac(g/cell);
   float2 c=.2+.6*float2(q8_flurry_hash(id,seed+layer),q8_flurry_hash(id.yx,seed+layer+9.3));
   float present=step(q8_flurry_hash(id+17,seed+layer*5),share);
   alpha=max(alpha,present*opacity*saturate(radius+.6-length((f-c)*cell)));
  }
  alpha*=smoothstep(1,.65,q.y)*smoothstep(0,.2,q.y)*smoothstep(1,.75,abs(q.x))*strength;
  clip(alpha-.004);
  // Bright as the other effects (scene white near 1.3 after grading); moonlit at night.
  return float4(lerp(float3(1.26,1.28,1.32),float3(.50,.58,.82),night),alpha);
}
'''

PATCHES = {
    "Renderer/native/city_fidelity/compiler.h": [
        ("        bool rises=e.kind==float(effect_smoke) || e.kind==float(effect_volcano_plume);",
         "        // Lab only (studies/winter): the mountain flurry (6) rises from its foot too.\n"
         "        bool rises=e.kind==float(effect_smoke) || e.kind==float(effect_volcano_plume) || e.kind>5.5f;"),
        ("    append_effects(library,{plume},std::make_shared<Lighting>(),project_natural,indexed,output);\n"
         "    return output.chunks.size()>before;\n}\n",
         "    append_effects(library,{plume},std::make_shared<Lighting>(),project_natural,indexed,output);\n"
         "    return output.chunks.size()>before;\n}\n"
         "// Lab only (studies/winter): a gentle snow flurry over a mountain tile. The quad stands\n"
         "// on the tile's front corner (column+1, row), about three tile heights tall and a tile\n"
         "// wide, so it is in front of the tile's own relief; `ground` is the height there.\n"
         "template<class Project>\n"
         "bool mountain_flurry(Library const& library,int nc,int nr,float ground,\n"
         "        Project project_natural,bool indexed,Surfaces& output){\n"
         "    EffectAnchor flurry{{float(nc)+1.f,float(nr),ground/112},\n"
         "        Effect{{0,0,0},6.f,1.05f,3.6f,float((unsigned(nc)*73u^unsigned(nr)*151u)%64u),1.f}};\n"
         "    auto before=output.chunks.size();\n"
         "    append_effects(library,{flurry},std::make_shared<Lighting>(),project_natural,indexed,output);\n"
         "    return output.chunks.size()>before;\n}\n")],
    "Renderer/native/object_preparation.h": [
        ("    bool const volcano=capture.real_terrain_type==10 && input.city_ready;\n"
         "    if(!capture.road_mask && !capture.railroad_mask && capture.city_id<0 && !volcano &&",
         "    bool const volcano=capture.real_terrain_type==10 && input.city_ready;\n"
         "    // Lab only (studies/winter): mountains carry a snow flurry while the switch is set.\n"
         "    bool const flurry=capture.real_terrain_type==6 && input.city_ready && GetEnvironmentVariableA(\"C3X_LAB_MOUNTAIN_FLURRY\",nullptr,0)!=0;\n"
         "    if(!capture.road_mask && !capture.railroad_mask && capture.city_id<0 && !volcano && !flurry &&"),
        ("    if(stop())return {};\n    result->rivers.assign(",
         "    if(flurry && library.effect_material<library.materials.size() && !queries.tile(nc,nr).hidden){\n"
         "        city_fidelity::Surfaces snow;\n"
         "        GroundProjection projection{nc,nr,input.projection.half_w,input.projection.half_h,\n"
         "            input.projection.relief_projection_scale,float(input.projection.content_view_height)};\n"
         "        if(city_fidelity::mountain_flurry(library,nc,nr,height_natural(float(nc)+1.f,float(nr)),projection,true,snow) &&\n"
         "           !adopt_city(snow))return {};\n"
         "    }\n"
         "    if(stop())return {};\n    result->rivers.assign(")],
    "Renderer/native/source_fidelity/geometry.h": [
        ("            for(auto& chunk:plume.chunks)city_chunks.push_back(std::move(chunk));\n    }\n",
         "            for(auto& chunk:plume.chunks)city_chunks.push_back(std::move(chunk));\n    }\n"
         "    // Lab only (studies/winter): a snow flurry over each explored mountain tile.\n"
         "    if(!prepared_objects && city_profile && cities.ready && owner.real==6 && GetEnvironmentVariableA(\"C3X_LAB_MOUNTAIN_FLURRY\",nullptr,0)!=0 &&\n"
         "       !queries.tile(nc,nr).hidden){\n"
         "        c3x_renderer::city_fidelity::Surfaces snow;\n"
         "        if(c3x_renderer::city_fidelity::mountain_flurry(cities.library,nc,nr,height_natural(float(nc)+1.f,float(nr)),\n"
         "                project_natural,false,snow))\n"
         "            for(auto& chunk:snow.chunks)city_chunks.push_back(std::move(chunk));\n"
         "    }\n")],
    "Renderer/lab/shared/shaders/objects/city_scene_material.hlsl": [
        ("float4 q8_city_effect(FeaturePixelInput p){", FLURRY_SHADER + "float4 q8_city_effect(FeaturePixelInput p){"),
        (" if(kind>2.5){\n  // A volcano's plume (93)",
         " if(kind>5.5 && kind<6.5)return q8_effect_flurry(q,pixel,t,seed,strength,night);\n"
         " if(kind>2.5){\n  // A volcano's plume (93)")],
    # Lab only: Winter selects the authored snowy-pine variety for every pine
    # forest (production picks it only on tundra); the user's pick, 2026-10-22.
    "Renderer/native/c3x_renderer.cpp": [
        ("                t.terrain_type==3?unsigned(forest_kinds::snow_pine_forest):unsigned(forest_kinds::pine_forest);};",
         "                (t.terrain_type==3 || GetEnvironmentVariableA(\"C3X_LAB_WINTER_PINES\",nullptr,0)!=0)?\n"
         "                unsigned(forest_kinds::snow_pine_forest):unsigned(forest_kinds::pine_forest);};")],
    "Renderer/native/biq_preview.cpp": [
        ("    if(ok)ok=lab_verify_objects(frame,output);\n",
         "    if(ok)ok=lab_verify_objects(frame,output);\n"
         "    // Lab only (studies/winter, after studies/natural_wonders): attached-effect frames at\n"
         "    // 15 fps on the visual clock; an unchanged view reuses its raster, so each redraws cold.\n"
         "    char effect_frames[8]={};\n"
         "    if(ok && GetEnvironmentVariableA(\"C3X_LAB_EFFECT_FRAMES\",effect_frames,sizeof(effect_frames))){\n"
         "        auto base=frame.presentation_time_ticks;\n"
         "        for(int n=0;n<std::atoi(effect_frames) && ok;++n){\n"
         "            frame.presentation_time_ticks=base+n*frame.presentation_frequency/15;\n"
         "            reset();\n"
         "            ok=set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK &&\n"
         "               render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK;\n"
         "            char suffix[32];sprintf_s(suffix,\".effect-%03d.bmp\",n);\n"
         "            ok=ok && write_bmp((std::string(argv[5])+suffix).c_str(),output);\n"
         "        }\n"
         "        std::printf(\"EFFECT frames: %s count=%s\\n\",ok?\"saved\":\"FAIL\",effect_frames);\n"
         "        return ok?0:1;\n"
         "    }\n")],
}


def prepare() -> None:
    private_tree.sync(NAME)
    root = private_tree.tree(NAME)
    private_tree.own(NAME, list(PATCHES))
    for relative, edits in PATCHES.items():
        base = (root / ".base" / relative).read_text()
        text = base
        for anchor, replacement in edits:
            if text.count(anchor) != 1:
                raise SystemExit(f"{relative}: anchor occurs {text.count(anchor)} times: {anchor[:70]!r}")
            text = text.replace(anchor, replacement)
        (root / relative).write_text(text)
    private_tree.shaders(NAME)
    print("patched", len(PATCHES), "files in", root.relative_to(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare", "build"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        private_tree.build(NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
