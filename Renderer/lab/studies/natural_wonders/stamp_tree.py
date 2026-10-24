#!/usr/bin/env python3
"""Private Lab tree that puts a natural wonder's height field into Renderer64's terrain relief.

Renderer64's mountains blend into the ground because their relief is part of
the terrain mesh: the mountain shader hands off from the ordinary ground
material and normal at zero rise. This tree adds one Lab-only stamp beside the
volcano stamp so a wonder's own (prepared) height field gets the same terrain
integration, mountain materials and snow rules:

- Renderer/lab/shared/natural/data.h: an optional wonder field and its span,
  height and turn, read from Renderer/packs/NaturalWonderStampLab/;
- Renderer/lab/shared/natural/mountain_shape.h: MountainFlag natural_wonder and
  add_wonder(), sampling the baked displacement directly;
- Renderer/native/render_core/terrain_query.h, world_topology.h: topology bit 28;
- Renderer/native/source_fidelity/terrain_compiler.h, geometry.h: pass the flag;
- Renderer/native/biq_preview.cpp: a Lab scene marker (BIQ bonus 0x40000000);
- attached effects (effects.bin, kits.py effect_records): city_fidelity/compiler.h
  wonder_effects() emits them in the city effect layer from object_preparation.h and the
  legacy geometry.h path; city_scene_material.hlsl draws a geyser (94) and a waterfall (95).

The checkout is never written; the private tree lives in
Renderer/lab/out/natural_wonders/tree-stamp/ (mountains study private_tree.py).

    python3 Renderer/lab/studies/natural_wonders/stamp_tree.py prepare   # sync + patch
    python3 Renderer/lab/studies/natural_wonders/stamp_tree.py build     # VM compile
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.mountains import private_tree

NAME = "stamp"
private_tree.OUT = ROOT / "Renderer/lab/out/natural_wonders"
MARKER = 0x40000000

# Lab effect shaders for wonders (city_scene_material.hlsl): geyser steam and a waterfall.
GEYSER_WATERFALL = r'''// Lab only (studies/natural_wonders): a geyser (94). Each cycle a white water jet climbs
// from the vent, holds and collapses into a spray crown; steam billows up with it and
// lingers after; between eruptions the vent only steams. Strength under ~.6 only steams.
// Water and steam run brighter than smoke (scene white is near 1.3 after grading).
float4 q8_effect_geyser(float2 q,float2 pixel,float t,float seed,float strength,float night){
  float H=pixel.x/pixel.y,Y=q.y*H;
  float phase=frac(t/11.0);
  float erupt=smoothstep(0,.05,phase)*(1-smoothstep(.30,.46,phase));
  float steam=smoothstep(.02,.16,phase)*(1-smoothstep(.42,.95,phase));
  float top=erupt*(.56+.05*sin(t*6.1+seed))*H*saturate((strength-.6)*3);
  float rise=saturate(Y/max(top,1e-3));
  float streak=.6*q8_effect_noise(float2(q.x*11+seed,(Y-t*3.5)*1.1))+.4*q8_effect_noise(float2(q.x*23+seed,(Y-t*4.5)*2.6));
  float wobble=.04*sin(Y*2.7+t*5.3)*rise;
  float body=abs(q.x-wobble)/(.09+.17*pow(rise,1.5));
  float jet=smoothstep(1,.35,body+(streak-.5)*.5*rise)*(1-smoothstep(top*.6,top,Y))*step(1e-3,top)*lerp(.95,.55,rise);
  float2 crown=float2(q.x-wobble,(Y-top*.9)/(.35+.2*top));
  float spray=exp(-dot(crown,crown)*2.6)*smoothstep(.3,.8,q8_effect_noise(float2(q.x*7+t*1.7,Y*7-t*2.5)))*
              erupt*step(1e-3,top)*.8;
  float4 water=float4(float3(1.24,1.28,1.32)*lerp(.9,1.03,streak)*lerp(1,.3,night),saturate((jet*.7+spray*.6)*min(strength,1)));
  // Steam-only vents (fumaroles) steam steadily; the geyser's steam follows its eruption.
  float vapour_strength=strength<.6?min(strength*1.35,.7):strength*lerp(.4,.82,steam);
  float4 vapour=q8_effect_plume(q,pixel,t*.55,seed,vapour_strength,night,
                                float4(3.6,.66,.2,.5),float3(.86,.14,.03),true);
  vapour.rgb*=1.3;
  float4 result=q8_effect_over(water,vapour);
  clip(result.a-.004);
  return result;
}
// Lab only (studies/natural_wonders): a waterfall sheet (95) draped down the face from its
// lip (v=1) to its foot (v=0): streaked water falls fast, widens and frays toward the foot,
// where the spray (96) takes over.
float4 q8_effect_waterfall(float2 q,float2 pixel,float t,float seed,float strength,float night){
  float H=pixel.x/pixel.y,Y=q.y*H,drop=1-q.y;
  float spread=lerp(.62,.92,pow(drop,1.2));
  float sway=(q8_effect_noise(float2(Y*.5+seed,t*.6))-.5)*.05*drop;
  float x=(q.x-sway)/spread;
  float fall=Y+t*3.4;
  // Streaks run along the fall (stretched noise), so no cell of the noise lattice shows.
  float streak=.55*q8_effect_noise(float2(x*4.3+seed,fall*.32))+.3*q8_effect_noise(float2(x*9.7+seed*3,(Y+t*4.6)*.55))
              +.15*q8_effect_noise(float2(x*21+seed*5,(Y+t*5.2)*1.1));
  // A steady, full sheet: its edge frays a little, never pinching the fall narrow.
  float fray=(q8_effect_noise(float2(x*1.3+seed,fall*.45))-.5)*(.1+.3*drop);
  float body=smoothstep(1,.5,abs(x)+fray);
  float sheet=body*lerp(.55,1,smoothstep(.2,.8,streak))*lerp(1.0,.65,drop*drop)*smoothstep(0,.1,q.y);
  float lip=smoothstep(.04,0,drop)*smoothstep(1.3,.6,abs(x));
  float3 tone=float3(1.30,1.34,1.38)*lerp(.72,1.06,streak)*lerp(1.04,.86,saturate(x*.5+.5));
  float4 result=float4((tone+lip*.1)*lerp(1,.28,night),saturate((sheet+lip*.5)*strength));
  clip(result.a-.004);
  return result;
}
// Lab only (studies/natural_wonders): spray (96) billowing up from a waterfall's plunge
// pool, fading in above the quad's foot and out below its top.
float4 q8_effect_spray(float2 q,float2 pixel,float t,float seed,float strength,float night){
  float H=pixel.x/pixel.y,Y=q.y*H;
  float mist=0,shade=0;
  [unroll]for(int k=0;k<8;k++){
   float cycle=t/3.2+k/8.0,phase=frac(cycle);
   float r=.22+.3*phase;
   float2 c=float2((q8_effect_noise(float2(seed*5+k,floor(cycle)))-.5)*.5,r*1.1+.08+phase*max(H-1.1,.2)*.65);
   float2 d=(float2(q.x,Y)-c)/r;
   float erode=(q8_effect_noise(float2(q.x*3.4+k*1.7,Y*3.4-t*.5))-.5)*.75;
   float a=smoothstep(1,.3,length(d)+erode)*smoothstep(0,.12,phase)*pow(1-phase,1.1)*.65;
   mist=1-(1-mist)*(1-a);shade+=a*saturate(.6-.4*d.x+.4*d.y);
  }
  float haze=exp(-pow((Y-.32)/.26,2))*smoothstep(.85,.15,abs(q.x))*.14
             *(.5+q8_effect_noise(float2(q.x*2.6+t*.25+seed,Y*2.2-t*.4)));
  mist=(1-(1-mist)*(1-haze))*smoothstep(0,.3,Y)*smoothstep(H,H*.65,Y)*smoothstep(1,.7,abs(q.x));
  float4 result=float4(float3(1.18,1.21,1.25)*lerp(.9,1.05,saturate(shade))*lerp(1,.28,night),saturate(mist*strength));
  clip(result.a-.004);
  return result;
}
'''

PATCHES = {
    "Renderer/native/render_core/terrain_query.h": [
        (
        "struct Tile { int base = 2, real = 2; bool present = false, hidden = false, snow = false, active = false, erupting = false; };",
        "// wonder: Lab-only natural-wonder relief stamp marker (bit 28).\n"
        "struct Tile { int base = 2, real = 2; bool present = false, hidden = false, snow = false, active = false, erupting = false, wonder = false; };")],
    "Renderer/native/render_core/coastal_waves.h": [
        ("struct WavePoint { Point position; float distance,along,coverage; };",
         "struct WavePoint { Point position; float distance,along,coverage; };\n"
         "// Lab only (studies/natural_wonders): a water wonder's wave ribbons round its islets,\n"
         "// five floats per point (x, y offsets from the wonder tile's centre, distance, along,\n"
         "// coverage) as a triangle list; NaturalData loads them with the wonder stamp pack.\n"
         "inline std::vector<float>& wonder_wave_points(){static std::vector<float> points;return points;}\n"
         "inline std::vector<WavePoint> wonder_waves(WorldCoast const& coast,int c,int r){\n"
         "    std::vector<WavePoint> out;auto const& points=wonder_wave_points();\n"
         "    if(points.empty() || !coast.world().tile(c,r).wonder)return out;\n"
         "    for(std::size_t i=0;i+5<=points.size();i+=5)\n"
         "        out.push_back({Point{c+.5+points[i],r+.5+points[i+1]},points[i+2],points[i+3],points[i+4]});\n"
         "    return out;\n"
         "}"),
        ("    if(!coastal_wave_site(coast,c,r,chosen) || !coastal_wave_spaced(coast,c,r,chosen))return {};",
         "    auto wonder=wonder_waves(coast,c,r);\n"
         "    if(!coastal_wave_site(coast,c,r,chosen) || !coastal_wave_spaced(coast,c,r,chosen))return wonder;"),
        ("    if(!active)return {};", "    if(!active)return wonder;"),
        ("        for(auto i:{a,b,e,a,e,d})result.push_back(grid[i]);\n    }\n    return result;",
         "        for(auto i:{a,b,e,a,e,d})result.push_back(grid[i]);\n    }\n"
         "    result.insert(result.end(),wonder.begin(),wonder.end());\n    return result;")],
    "Renderer/native/render_core/world_topology.h": [(
        "(value&(1u<<26))!=0,(value&(1u<<24))!=0,(value&(1u<<27))!=0};",
        "(value&(1u<<26))!=0,(value&(1u<<24))!=0,(value&(1u<<27))!=0,(value&(1u<<28))!=0};")],
    "Renderer/native/source_fidelity/terrain_compiler.h": [(
        "return (t.snow?1u:0u)|(t.active?2u:0u)|(t.erupting?4u:0u);};",
        "return (t.snow?1u:0u)|(t.active?2u:0u)|(t.erupting?4u:0u)|(t.wonder?8u:0u);};")],
    "Renderer/native/source_fidelity/geometry.h": [(
        "return (t.snow?1u:0u)|(t.active?2u:0u)|(t.erupting?4u:0u);};",
        "return (t.snow?1u:0u)|(t.active?2u:0u)|(t.erupting?4u:0u)|(t.wonder?8u:0u);};"),
        ("            for(auto& chunk:plume.chunks)city_chunks.push_back(std::move(chunk));\n    }\n",
         "            for(auto& chunk:plume.chunks)city_chunks.push_back(std::move(chunk));\n    }\n"
         "    // Lab only: a natural wonder's attached effects (stamp_tree.py) on its rendered relief.\n"
         "    if(!prepared_objects && city_profile && cities.ready && queries.tile(nc,nr).wonder && !natural.wonder_effects.empty()){\n"
         "        MountainShape const shape(natural,nc,nr,lookup_natural,mountain_flags);\n"
         "        auto relief_at=[&](float x,float y){\n"
         "            auto shore=shore_sample_at(x,y);\n"
         "            float river_scale=river_terrain_near?smooth01((float(river_at(x,y))-6.f)/16.f):1.f;\n"
         "            return height_natural(x,y)+shape.sample(natural,x,y).displacement*\n"
         "                coast_relief(float(shore.distance),float(shore.beach_width))*river_scale*hidden_taper_at(x,y);\n"
         "        };\n"
         "        c3x_renderer::city_fidelity::Surfaces effects;\n"
         "        if(c3x_renderer::city_fidelity::wonder_effects(cities.library,nc,nr,natural.wonder_effects,relief_at,\n"
         "                project_natural,false,effects))\n"
         "            for(auto& chunk:effects.chunks)city_chunks.push_back(std::move(chunk));\n"
         "    }\n")],
    "Renderer/native/biq_preview.cpp": [
        ("(t.real==6 && (t.bonus&0x10u) ? 1u<<26 : 0);",
         "(t.real==6 && (t.bonus&0x10u) ? 1u<<26 : 0)|\n"
         "        // Lab marker: a natural-wonder relief stamp on this mountain-hosted tile.\n"
         "        ((t.bonus&0x40000000u) ? 1u<<28 : 0);"),
        ("tile.square_parts = source.bonus;", "tile.square_parts = source.bonus & ~0x40000000u;"),
        # Effect clips: an unchanged view reuses its last raster, so each frame redraws cold.
        ("    if(ok)ok=lab_verify_objects(frame,output);\n",
         "    if(ok)ok=lab_verify_objects(frame,output);\n"
         "    // Lab only (studies/natural_wonders): attached-effect frames at 15 fps on the visual clock.\n"
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
    "Renderer/native/city_fidelity/compiler.h": [
        ("        bool rises=e.kind==float(effect_smoke) || e.kind==float(effect_volcano_plume);",
         "        // Lab only: natural-wonder steam (4) and waterfalls (5) rise from their foot too.\n"
         "        bool rises=e.kind==float(effect_smoke) || e.kind==float(effect_volcano_plume) || e.kind>3.5f;"),
        ("    return output.chunks.size()>before;\n}\ntemplate<class Height,class Project,class Stop=ContinueCompilation,class Site=EverySite>",
         "    return output.chunks.size()>before;\n}\n"
         "// Lab only (studies/natural_wonders): a natural wonder's attached effects. A record is\n"
         "// kind, foot and top offsets from the tile centre, lift (mountain units), width, height\n"
         "// (tile units; 0 reaches the lifted top on screen), strength and seed. Waterfalls (5)\n"
         "// are a ribbon draped from the top point down the face to the foot, standing off it by\n"
         "// the lift; the rest are camera-facing quads rising from the foot.\n"
         "template<class Surface,class Project>\n"
         "bool wonder_effects(Library const& library,int nc,int nr,std::vector<float> const& records,\n"
         "        Surface surface,Project project_natural,bool indexed,Surfaces& output){\n"
         "    std::vector<EffectAnchor> anchors;\n"
         "    Chunk ribbon;ribbon.material=library.effect_material;ribbon.lighting=std::make_shared<Lighting>();\n"
         "    ribbon.terrain_conforming=true;ribbon.effect=true;\n"
         "    std::vector<fidelity::MapVertex> strip;\n"
         "    for(std::size_t i=0;i+10<=records.size();i+=10){\n"
         "        float const* e=records.data()+i;\n"
         "        float fx=float(nc)+.5f+e[1],fy=float(nr)+.5f+e[2],tx=float(nc)+.5f+e[3],ty=float(nr)+.5f+e[4];\n"
         "        if(e[0]>6.5f)continue;  // still water: the production water layer (c3x_renderer.cpp)\n"
         "        if(e[0]>4.5f && e[0]<5.5f){\n"
         "            float dx=tx-fx,dy=ty-fy,length=std::max(std::sqrt(dx*dx+dy*dy),1e-4f);\n"
         "            float ax=-dy/length*e[6]*.5f,ay=dx/length*e[6]*.5f;\n"
         "            unsigned base=unsigned(strip.size());constexpr int steps=32;\n"
         "            // Free fall: from the lip the water drops on a smooth arc to its foot instead of\n"
         "            // following each ledge of the face; it stays clear of the rock, smoothed so a\n"
         "            // ledge bends it gently rather than in a step.\n"
         "            float top=surface(tx,ty)+e[5],bottom=surface(fx,fy)+e[5]*.5f;\n"
         "            std::array<float,steps+1> clear{},fall{};\n"
         "            for(int s=0;s<=steps;++s){\n"
         "                float v=float(s)/steps,x=fx+dx*v,y=fy+dy*v,p=1-v;\n"
         "                clear[s]=std::max({surface(x,y),surface(x+ax,y+ay),surface(x-ax,y-ay)})+e[5]*.6f;\n"
         "                fall[s]=std::max(top-(top-bottom)*p*p,clear[s]);\n"
         "            }\n"
         "            for(int pass=0;pass<4;++pass)for(int s=1;s<steps;++s)\n"
         "                fall[s]=std::max(clear[s],(fall[s-1]+2*fall[s]+fall[s+1])*.25f);\n"
         "            for(int s=0;s<=steps;++s){\n"
         "                float v=float(s)/steps,x=fx+dx*v,y=fy+dy*v,z=fall[s];\n"
         "                for(int side=-1;side<=1;side+=2){\n"
         "                    auto out=project_natural(x+ax*side,y+ay*side,z);\n"
         "                    out.u=float(side);out.v=v;out.normal_x=out.normal_y=0;out.normal_z=1;\n"
         "                    out.macro_u=e[9];out.macro_v=e[8];\n"
         "                    out.material_grass=1;out.material_plains=out.material_desert=out.material_marsh=0;\n"
         "                    out.authored_relief_height=out.authored_relief_blend=0;\n"
         "                    out.relief_owner_u=e[6];out.relief_owner_v=length;\n"
         "                    out.base_terrain=90.f+e[0];strip.push_back(out);\n"
         "                }\n"
         "            }\n"
         "            for(unsigned s=0;s<unsigned(steps);++s){\n"
         "                unsigned a=base+2*s,quad[]={a,a+1,a+3,a,a+3,a+2};\n"
         "                ribbon.indices.insert(ribbon.indices.end(),std::begin(quad),std::end(quad));\n"
         "            }\n"
         "            continue;\n"
         "        }\n"
         "        float foot=surface(fx,fy),height=e[7];\n"
         "        if(!(height>0)){\n"
         "            auto a=project_natural(fx,fy,foot),b=project_natural(tx,ty,surface(tx,ty)+e[5]),\n"
         "                 c=project_natural(fx,fy,foot+112.f);\n"
         "            height=(a.y-b.y)/std::max(1e-3f,a.y-c.y);\n"
         "        }\n"
         "        anchors.push_back({{fx,fy,foot/112},Effect{{0,0,0},e[0],e[6],std::max(height,.05f),e[9],e[8]}});\n"
         "    }\n"
         "    auto before=output.chunks.size();\n"
         "    append_effects(library,anchors,std::make_shared<Lighting>(),project_natural,indexed,output);\n"
         "    if(!strip.empty() && library.effect_material<library.materials.size()){\n"
         "        if(indexed)ribbon.vertices=std::move(strip);\n"
         "        else {for(unsigned index:ribbon.indices)ribbon.vertices.push_back(strip[index]);ribbon.indices.clear();}\n"
         "        output.chunks.push_back(std::move(ribbon));\n"
         "    }\n"
         "    return output.chunks.size()>before;\n"
         "}\n"
         "template<class Height,class Project,class Stop=ContinueCompilation,class Site=EverySite>")],
    "Renderer/native/object_preparation.h": [
        ("    bool const volcano=capture.real_terrain_type==10 && input.city_ready;\n"
         "    if(!capture.road_mask && !capture.railroad_mask && capture.city_id<0 && !volcano &&",
         "    bool const volcano=capture.real_terrain_type==10 && input.city_ready;\n"
         "    // Lab only: a natural wonder's attached effects (stamp_tree.py) on its mountain-hosted tile.\n"
         "    bool const wonder=capture.real_terrain_type==6 && input.city_ready && !natural.wonder_effects.empty();\n"
         "    if(!capture.road_mask && !capture.railroad_mask && capture.city_id<0 && !volcano && !wonder &&"),
        ("    if(input.projection.tile.road_mask || input.projection.tile.railroad_mask || volcano ||",
         "    if(input.projection.tile.road_mask || input.projection.tile.railroad_mask || volcano || wonder ||"),
        # The wonder stamp (natural_wonder flag) shapes the surface its effects stand on.
        ("        mountain_shape=MountainShape(natural,nc,nr,lookup_natural,[](int,int){return false;});",
         "        mountain_shape=MountainShape(natural,nc,nr,lookup_natural,\n"
         "            [&](int c,int r){return queries.tile(c,r).wonder?8u:0u;});"),
        ("    if(stop())return {};\n    result->rivers.assign(",
         "    if(wonder && library.effect_material<library.materials.size() && queries.tile(nc,nr).wonder){\n"
         "        city_fidelity::Surfaces effects;\n"
         "        GroundProjection projection{nc,nr,input.projection.half_w,input.projection.half_h,\n"
         "            input.projection.relief_projection_scale,float(input.projection.content_view_height)};\n"
         "        if(city_fidelity::wonder_effects(library,nc,nr,natural.wonder_effects,route_height,projection,true,effects) &&\n"
         "           !adopt_city(effects))return {};\n"
         "    }\n"
         "    if(stop())return {};\n    result->rivers.assign(")],
    # Resource bodies (the wonder kit's pieces) seat on the wonder stamp, not a generic mountain.
    "Renderer/native/c3x_renderer.cpp": [
        # Lab only: a wonder wave ribbon carries a flag and its own seed in its coverage (2 + seed,
        # negative where masked) and passes them on as base terrain (both wave-cell paths).
        ("\n                            out.base_terrain=seed;out.real_terrain=seed2;out.surface_kind=7;",
         "\n                            out.base_terrain=std::abs(point.coverage)>1.5f?std::abs(point.coverage):seed;"
         "out.real_terrain=std::abs(point.coverage)>1.5f?std::fmod((std::abs(point.coverage)-2.f)*7.13f+.37f,1.f):seed2;out.surface_kind=7;"),
        ("\n                out.base_terrain=seed;out.real_terrain=seed2;out.surface_kind=7;",
         "\n                out.base_terrain=std::abs(point.coverage)>1.5f?std::abs(point.coverage):seed;"
         "out.real_terrain=std::abs(point.coverage)>1.5f?std::fmod((std::abs(point.coverage)-2.f)*7.13f+.37f,1.f):seed2;out.surface_kind=7;"),
        ("            bool water_coverage=(world_ground && shared_hit) || !cull_empty_water || !environment_profile ||",
         "            // Lab only (studies/natural_wonders): a wonder's channel as river water.\n"
         "            std::vector<Vertex> wonder_river;\n"
         "            bool water_coverage=(world_ground && shared_hit) || !cull_empty_water || !environment_profile ||"),
        ("                auto prior_chunks=compiled.mesh->layers[layer].size();",
         "                if(layer==geometry_river && !wonder_river.empty()){\n"
         "                    if(!cache_geometry_layer(mesh_upload,wonder_river,compiled.mesh->layers[layer],prewarming,compiled.byte_count,foreground_pending,\n                                false,false,nullptr,nullptr,nullptr,nullptr,world_ground?3u:0u)){\n"
         "                        tile_geometry_cache_bytes-=compiled.byte_count;return false;\n"
         "                    }\n"
         "                    compiled.byte_count+=compiled.mesh->layers[layer].back().byte_count;\n"
         "                }\n"
         "                auto prior_chunks=compiled.mesh->layers[layer].size();"),
        ("[&](int mc,int mr){return queries.natural_tile(mc,mr);},[](int,int){return false;});}",
         "[&](int mc,int mr){return queries.natural_tile(mc,mr);},\n"
         "                        [&](int mc,int mr){return queries.tile(mc,mr).wonder?8u:0u;});}"),
        ('                if(pickup_profile && (layer==geometry_bed || layer==geometry_water)) {\n                    if(!water_coverage)continue;\n                    // All flat layers share exact vertex/index data. Surface\n                    // kind is the per-draw b1 value; no duplicate allocation.\n                    compiled.mesh->layers[layer]=compiled.mesh->layers[geometry_underlay];\n                    for(auto& chunk:compiled.mesh->layers[layer]){\n                        if(chunk.buffer)chunk.buffer->AddRef();if(chunk.indices)chunk.indices->AddRef();chunk.byte_count=0;\n                    }\n                    continue;\n                }', "                if(pickup_profile && (layer==geometry_bed || layer==geometry_water)) {\n                    // Lab only (studies/natural_wonders): a wonder's lake (7) or channel (8) is a\n                    // flat production bed and water surface at its own level. Its hydrology fields give a\n                    // rocky shore, negative distance inside and depth toward the middle, so the\n                    // water material shades it as every other water (Fresnel, sky, mirrored\n                    // objects, glints, shadows).\n                    std::vector<Vertex> wonder_water;\n                    int const wonder_c=(tile.tile_x+tile.tile_y)/2,wonder_r=(tile.tile_x-tile.tile_y)/2;\n                    if(!natural.wonder_effects.empty() && queries.tile(wonder_c,wonder_r).wonder){\n                        auto const& records=natural.wonder_effects;\n                        for(std::size_t at=0;at+10<=records.size();at+=10){\n                            float const* e=records.data()+at;\n                            if(e[0]<6.5f || e[0]>8.5f || (e[0]>7.5f && layer!=geometry_water))continue;\n                            bool const channel=e[0]>7.5f;\n                            float fx=float(wonder_c)+.5f+e[1],fy=float(wonder_r)+.5f+e[2];\n                            float tx=float(wonder_c)+.5f+e[3],ty=float(wonder_r)+.5f+e[4];\n                            float dx=tx-fx,dy=ty-fy,length=std::max(std::sqrt(dx*dx+dy*dy),1e-4f);\n                            float half=e[6]*.5f,ax=-dy/length,ay=dx/length;\n                            constexpr int along=36,across=18;\n                            float const water_ground=natural_height_at(float(wonder_c)+.5f,float(wonder_r)+.5f);\n                            std::vector<Vertex> grid;\n                            for(int s=0;s<=along;++s)for(int k=0;k<=across;++k){\n                                float v=channel?float(s)/along*1.65f-.55f:float(s)/along*1.2f-.1f;\n                                float u=(float(k)/across*2-1)*(channel?1.7f:1.2f);\n                                // Past the mouth (v<0) a channel's strip itself fans out toward the sea\n                                // with a wandering edge: the river material fills its full width once it\n                                // reaches the sea, so the strip's outline is the plume's outline.\n                                float flare=std::clamp((.15f-v)/.6f,0.f,1.f);\n                                flare=channel?1+1.1f*flare*flare*(3-2*flare)+.12f*flare*std::sin(v*length*19.f+(u>0?1.3f:4.1f)):1.f;\n                                float x=fx+dx*v+ax*u*half*flare,y=fy+dy*v+ay*u*half*flare;\n                                float rise=0;\n                                float distance=natural.wonder_water(x,y,float(wonder_c)+.5f,float(wonder_r)+.5f,&rise);\n                                float terrain=resource_height_at(x,y),level=water_ground+rise,above=level-terrain;\n                                // A channel is a river decal on the ground (under the walls, on the floor).\n                                Vertex out=natural_projection(x,y,channel?std::min(level,terrain+1.f):level);\n                                // Water stands where the terrain is below its surface inside the shape.\n                                distance=std::max(distance,-above/112.f);\n                                out.macro_u=x*.5f;out.macro_v=y*.5f;out.panel=1;\n                                // Marks wonder water for the shared receiver query (keeps a third of the cast shadow).\n                                out.shadow_visibility=2;out.ambient_visibility=1;\n                                out.base_terrain=out.real_terrain=11;out.shore_distance=distance;\n                                out.shore_true_distance=distance;out.shore_beach_width=0;out.shore_rockiness=1;\n                                out.shore_depth=std::clamp(above/30.f,0.f,.8f);\n                                if(channel){\n                                    // A channel is river water (the river layer's material): its\n                                    // distance band from the centreline, flow toward the mouth, and\n                                    // the real coast distance so it opens into the sea.\n                                    // Wandering banks; above the water level the walls close the channel.\n                                    float wander=std::sin(v*length*23.f+u*2.f)*.6f+std::sin(v*length*47.f+1.7f)*.4f;\n                                    out.surface_kind=9;\n                                    out.river_distance=std::max(5.f*std::abs(u)+wander,above<0?6.f-above*.5f:0.f);\n                                    out.river_branch_count=out.river_mouth_distance=1000.f;out.river_padding=0;\n                                    out.relief_owner_u=-dx/length;out.relief_owner_v=-dy/length;\n                                    out.relief_owner_coverage=out.relief_owner_state=0;\n                                    out.shore_true_distance=out.shore_distance=float(queries.shore(x,y).distance);\n                                    out.shore_depth=0;\n                                }\n                                grid.push_back(out);\n                            }\n                            for(int s=0;s<along;++s)for(int k=0;k<across;++k){\n                                unsigned a=unsigned(s*(across+1)+k),b=a+1,c=a+unsigned(across+1),d=c+1;\n                                for(unsigned corner:{a,c,b,b,c,d})(channel?wonder_river:wonder_water).push_back(grid[corner]);\n                            }\n                        }\n                    }\n                    if(!water_coverage && wonder_water.empty())continue;\n                    if(water_coverage){\n                    // All flat layers share exact vertex/index data. Surface\n                    // kind is the per-draw b1 value; no duplicate allocation.\n                    compiled.mesh->layers[layer]=compiled.mesh->layers[geometry_underlay];\n                    for(auto& chunk:compiled.mesh->layers[layer]){\n                        if(chunk.buffer)chunk.buffer->AddRef();if(chunk.indices)chunk.indices->AddRef();chunk.byte_count=0;\n                    }\n                    }\n                    if(!wonder_water.empty()){\n                        if(!cache_geometry_layer(mesh_upload,wonder_water,compiled.mesh->layers[layer],prewarming,compiled.byte_count,foreground_pending,\n                                false,false,nullptr,nullptr,nullptr,nullptr,world_ground?3u:0u)){\n                            tile_geometry_cache_bytes-=compiled.byte_count;return false;\n                        }\n                        compiled.byte_count+=compiled.mesh->layers[layer].back().byte_count;\n                    }\n                    continue;\n                }")],
    "Renderer/lab/shared/natural/data.h": [
        ("    bool volcano_ready=false;\n",
         "    bool volcano_ready=false;\n"
         "    // Lab only (studies/natural_wonders): one natural-wonder relief stamp, a baked\n"
         "    // displacement field with its span (tiles), height (relief units) and turn.\n"
         "    HeightField wonder_height,wonder_ground,wonder_special;\n"
         "    // Attached effects (kits.py effect_records): ten floats per effect.\n"
         "    std::vector<float> wonder_effects;\n"
         "    float wonder_span=0,wonder_units=0,wonder_turn=0;\n"
         "    bool wonder_ready=false;\n"),
        ("        volcano_ready=load_volcano_element(read);\n",
         "        volcano_ready=load_volcano_element(read);\n"
         "        wonder_ready=load_wonder_element(read);\n"
         "        load_wonder_waves(read);\n"),
        ("    void measure_volcano_trend(){",
         "    template<class Read> bool load_wonder_element(Read read){\n"
         "        constexpr char const* pack=\"Renderer/packs/NaturalWonderStampLab/\";\n"
         "        std::vector<std::uint8_t> bytes,params;\n"
         "        if(!read(std::string(pack)+\"stamp.bin\",params)||params.size()!=12)return false;\n"
         "        std::memcpy(&wonder_span,params.data(),4);std::memcpy(&wonder_units,params.data()+4,4);\n"
         "        std::memcpy(&wonder_turn,params.data()+8,4);\n"
         "        if(!(wonder_span>0) || !(wonder_units>0))return false;\n"
         "        if(!read(std::string(pack)+\"height_lod0.dds\",bytes)||bytes.size()<148)return false;\n"
         "        unsigned format=0;std::memcpy(&format,bytes.data()+128,4);\n"
         "        if(format!=61)return false;\n"
         "        auto& f=wonder_height;std::memcpy(&f.height,bytes.data()+12,4);std::memcpy(&f.width,bytes.data()+16,4);\n"
         "        if(!f.width||f.width>1024||f.height!=f.width||148ull+std::uint64_t(f.width)*f.height>bytes.size())return false;\n"
         "        f.pixels.assign(bytes.begin()+148,bytes.begin()+148+std::size_t(f.width)*f.height);\n"
         "        f.minimum=0;f.maximum=1;\n"
         "        // Optional region weights (vegetated ground, the wonder's special material).\n"
         "        HeightField* weights[]={&wonder_ground,&wonder_special};\n"
         "        char const* names[]={\"ground_lod0.dds\",\"special_lod0.dds\"};\n"
         "        for(unsigned i=0;i<2;i++){\n"
         "            std::vector<std::uint8_t> extra;auto& w=*weights[i];w.pixels.clear();\n"
         "            if(!read(std::string(pack)+names[i],extra)||extra.size()<148)continue;\n"
         "            unsigned fmt=0;std::memcpy(&fmt,extra.data()+128,4);if(fmt!=61)continue;\n"
         "            std::memcpy(&w.height,extra.data()+12,4);std::memcpy(&w.width,extra.data()+16,4);\n"
         "            if(w.width!=f.width||w.height!=f.height||148ull+std::uint64_t(w.width)*w.height>extra.size())continue;\n"
         "            w.pixels.assign(extra.begin()+148,extra.begin()+148+std::size_t(w.width)*w.height);\n"
         "            w.minimum=0;w.maximum=1;\n"
         "        }\n"
         "        std::vector<std::uint8_t> effects;wonder_effects.clear();\n"
         "        if(read(std::string(pack)+\"effects.bin\",effects) && !effects.empty() && effects.size()%40==0){\n"
         "            wonder_effects.resize(effects.size()/4);std::memcpy(wonder_effects.data(),effects.data(),effects.size());\n"
         "        }\n"
         "        return true;\n"
         "    }\n"
         "    // Lab only: signed distance (tiles, negative inside) to a wonder's still water\n"
         "    // (records of kind 7 lake, 8 channel) whose stamp is centred at (cx, cy).\n"
         "    // rise (optional) receives the water's level there in relief units above the natural\n"
         "    // ground (e[5] at the top end, e[7] at the foot, level for a lake).\n"
         "    float wonder_water(float x,float y,float cx,float cy,float* rise=nullptr) const {\n"
         "        float best=1e9f;\n"
         "        for(std::size_t i=0;i+10<=wonder_effects.size();i+=10){\n"
         "            float const* e=wonder_effects.data()+i;\n"
         "            if(e[0]<6.5f || e[0]>8.5f)continue;\n"
         "            float fx=cx+e[1],fy=cy+e[2],dx=e[3]-e[1],dy=e[4]-e[2];\n"
         "            float length=std::max(std::sqrt(dx*dx+dy*dy),1e-4f),half=e[6]*.5f;\n"
         "            float px=x-fx,py=y-fy,along=(px*dx+py*dy)/(length*length),across=(-px*dy+py*dx)/length/half;\n"
         "            float wobble=.05f*std::sin(x*23.f+y*7.f)+.04f*std::sin(y*31.f-x*11.f);\n"
         "            float distance;\n"
         "            if(e[0]>7.5f){\n"
         "                float beyond=std::max(-along,along-1)*length,side=(std::abs(across)-1)*half;\n"
         "                distance=std::hypot(std::max(beyond,0.f),std::max(side,0.f))+std::min(std::max(beyond,side),0.f)+wobble*half;\n"
         "            }else distance=(std::sqrt(across*across+(2*along-1)*(2*along-1))-1+wobble)*std::min(half,length*.5f);\n"
         "            if(distance<best && rise)*rise=e[7]+(e[5]-e[7])*std::clamp(along,0.f,1.f);\n"
         "            best=std::min(best,distance);\n"
         "        }\n"
         "        return best;\n"
         "    }\n"
         "    void measure_volcano_trend(){"),
        ("    void measure_volcano_trend(){",
         "    template<class Read> void load_wonder_waves(Read read){\n"
         "        auto& points=::c3x_renderer::render_core::wonder_wave_points();points.clear();\n"
         "        std::vector<std::uint8_t> bytes;\n"
         "        if(!read(std::string(\"Renderer/packs/NaturalWonderStampLab/waves.bin\"),bytes) || bytes.size()%20)return;\n"
         "        points.resize(bytes.size()/4);std::memcpy(points.data(),bytes.data(),bytes.size());\n"
         "    }\n"
         "    void measure_volcano_trend(){"),
        ('#include "../../../native/source_fidelity/kernels.h"',
         '#include "../../../native/source_fidelity/kernels.h"\n#include "../../../native/render_core/coastal_waves.h"')],
    "Renderer/lab/shared/natural/mountain_shape.h": [
        ("enum MountainFlag : unsigned { mountain_snow_cap=1u, volcano_active=2u, volcano_erupting=4u };",
         "enum MountainFlag : unsigned { mountain_snow_cap=1u, volcano_active=2u, volcano_erupting=4u, natural_wonder=8u };"),
        ("struct MountainCell { bool mountain=false,volcano=false,snow=false; unsigned activity=0; std::uint32_t seed=0; };",
         "struct MountainCell { bool mountain=false,volcano=false,snow=false; unsigned activity=0; std::uint32_t seed=0; bool wonder=false; };"),
        ("    unsigned activity=0;\n    bool mirror=false;\n};",
         "    unsigned activity=0;\n    bool mirror=false,baked=false;\n};"),
        ("    MountainShape()=default;",
         "    // Lab only: a natural wonder's baked displacement, fixed turn, its own span and height.\n"
         "    void add_wonder(NaturalData const& natural,float x,float y,bool snow){\n"
         "        auto& p=pieces[count++];\n"
         "        p.height_field=p.blend_field=&natural.wonder_height;p.baked=true;\n"
         "        p.center_x=x;p.center_y=y;p.turn_cos=std::cos(natural.wonder_turn);p.turn_sin=std::sin(natural.wonder_turn);\n"
         "        p.inv_long=p.inv_cross=1/natural.wonder_span;\n"
         "        p.height=natural.wonder_units;p.reach=std::min(1.06f,reach_scale*natural.wonder_span);p.snow=snow?1.f:0.f;\n"
         "    }\n"
         "    MountainShape()=default;"),
        ("                    volcano?((bits&volcano_erupting)?2u:(bits&volcano_active)?1u:0u):0u,mountain_seed(t)};",
         "                    volcano?((bits&volcano_erupting)?2u:(bits&volcano_active)?1u:0u):0u,mountain_seed(t),\n"
         "                    !volcano && (bits&natural_wonder)!=0};"),
        ("            std::uint32_t seed=mix(owner.seed);\n",
         "            std::uint32_t seed=mix(owner.seed);\n"
         "            if(owner.wonder && natural.wonder_ready){\n"
         "                add_wonder(natural,float(nc+dc)+.5f,float(nr+dr)+.5f,owner.snow);\n"
         "                continue;\n"
         "            }\n"),
        ("            float shape=std::max(0.f,h)*smooth01((blend-.28f)/.34f);\n",
         "            float shape=std::max(0.f,h)*smooth01((blend-.28f)/.34f);\n"
         "            if(p.baked)shape=std::max(0.f,h);\n"),
        # Snow and top-rock bands follow each wonder's own peak, not the standard mountain height.
        ("        MountainSample result;float second=0,volcano=0,other=0;",
         "        MountainSample result;float second=0,volcano=0,other=0,norm=height,wonder_g=0,wonder_s=0;"),
        ("                result.u=u;result.v=v;result.snow=p.snow;",
         "                result.u=u;result.v=v;result.snow=p.snow;norm=p.baked?p.height:height;\n"
         "                wonder_g=p.baked?natural.wonder_ground.sample(u,v):0.f;\n"
         "                wonder_s=p.baked?natural.wonder_special.sample(u,v):0.f;"),
        ("        result.height=std::min(1.f,result.displacement/height);",
         "        result.height=std::min(1.f,result.displacement/norm);\n"
         "        // Region weights belong to the dominant wonder stamp; they ease out where a\n"
         "        // neighbouring mountain or saddle takes over the surface.\n"
         "        float owned=smooth01((result.dominant-second)/blend_k+.5f);\n"
         "        result.wonder_ground=wonder_g*owned;result.wonder_special=wonder_s*owned;"),
        ("    float volcano=0,volcano_u=.5f,volcano_v=.5f,channel=0;unsigned activity=0;",
         "    float volcano=0,volcano_u=.5f,volcano_v=.5f,channel=0;unsigned activity=0;\n"
         "    // Lab only: a natural-wonder stamp's region weights (vegetated ground, special material).\n"
         "    float wonder_ground=0,wonder_special=0;")],
    "Renderer/lab/shared/natural/relief_mesh_body.h": [
        ("                    out.base_terrain=42+coverage*source_weight;",
         "                    // Lab only: the relief opens where it falls below its wonder's still water\n"
         "                    // (drawn by the production water layers), as land does at a shore. Height\n"
         "                    // above the water is linear across each triangle, so the shader's coverage\n"
         "                    // clip follows the exact waterline rather than the vertex grid.\n"
         "                    float wonder_shore=1e9f;\n"
         "                    for(unsigned piece=0;piece<mountain_shape.count;++piece)if(mountain_shape.pieces[piece].baked){\n"
         "                        auto const& wonder_piece=mountain_shape.pieces[piece];float rise=0;\n"
         "                        if(natural.wonder_water(world_x,world_y,wonder_piece.center_x,wonder_piece.center_y,&rise)<0)\n"
         "                            wonder_shore=std::min(wonder_shore,(elevation-height_natural(wonder_piece.center_x,wonder_piece.center_y)-rise)/6.f);\n"
         "                    }\n"
         "                    out.base_terrain=42+std::max(-.5f,std::min(coverage*source_weight,wonder_shore));"),
        ("out.relief_owner_u=sample.volcano_u-.5f;out.relief_owner_v=sample.volcano_v-.5f;",
         "// Lab only: without a volcano, the owner channel carries a wonder stamp's region weights.\n"
         "                    out.relief_owner_u=sample.volcano>0?sample.volcano_u-.5f:sample.wonder_ground*flat_blend;\n"
         "                    out.relief_owner_v=sample.volcano>0?sample.volcano_v-.5f:sample.wonder_special*flat_blend;")],
    "Renderer/lab/shared/shaders/objects/city_scene_material.hlsl": [
        ("float4 shape,float3 look){", "float4 shape,float3 look,bool keep=false){"),
        ("  clip(alpha-.004);\n  float grey=", "  if(!keep)clip(alpha-.004);\n  float grey="),
        ("float4 q8_city_effect(FeaturePixelInput p){", GEYSER_WATERFALL + "float4 q8_city_effect(FeaturePixelInput p){"),
        (" if(kind>2.5){",
         " if(kind>5.5)return q8_effect_spray(q,pixel,t,seed,strength,night);\n"
         " if(kind>4.5)return q8_effect_waterfall(q,pixel,t,seed,strength,night);\n"
         " if(kind>3.5)return q8_effect_geyser(q,pixel,t,seed,strength,night);\n"
         " if(kind>2.5){"),
        # Quad units per pixel along the effect's own axes: identical for screen-aligned quads,
        # and right for the waterfall ribbon, which runs down the face at any screen angle.
        (" float2 pixel=float2(max(abs(ddx(p.uv.x)),1e-5),max(abs(ddy(p.uv.y)),1e-5));",
         " float2 pixel=float2(max(length(float2(ddx(p.uv.x),ddy(p.uv.x))),1e-5),\n"
         "                     max(length(float2(ddx(p.uv.y),ddy(p.uv.y))),1e-5));")],
    "Renderer/lab/shared/shaders/hydrology/coastal_waves.hlsl": [(
        "    float seed=input.base_terrain,seed2=input.real_terrain;\n",
        "    // Lab only (studies/natural_wonders): a water wonder's ribbon (base terrain 2 + seed) is a\n"
        "    // foam collar hugging its rock, as Civ VI rings Ha Long Bay's pillars: the connected fine\n"
        "    // foam in world space, densest at the waterline, its reach slowly breathing.\n"
        "    if(input.base_terrain>1.5){\n"
        "        float collar_seed=input.base_terrain-2,d=input.uv.x,t=coastal_time.x;\n"
        "        float2 world=input.macro_uv*2;\n"
        "        float broad=coastal_foam.Sample(material_sampler,world*float2(.12,.24)+collar_seed*.37).r;\n"
        "        float reach=.18+.16*broad+.04*sin(t*.7+collar_seed*6.283);\n"
        "        float2 drift=float2(t*.004,-t*.003);\n"
        "        float lace=coastal_foam.Sample(material_sampler,world*float2(.9,1.8)+drift+collar_seed).r;\n"
        "        float fine=coastal_foam.Sample(material_sampler,world*float2(1.7,3.4)-drift*1.4+collar_seed*3).r;\n"
        "        float pattern=saturate((max(lace,fine*.85)-.1)*2.6);\n"
        "        float falloff=1-smoothstep(reach*.2,reach,d);\n"
        "        float rim=1-smoothstep(0,.05,d);\n"
        "        float coverage=saturate(pattern*falloff*.75+rim*(.15+.45*pattern))*.85;\n"
        "        clip(coverage-.01);\n"
        "        float3 lit=float3(.84,.88,.90)*q6_receiver_illumination(input,float3(0,0,1),1,1);\n"
        "        return float4(lit*coverage,coverage);\n"
        "    }\n"
        "    float seed=input.base_terrain,seed2=input.real_terrain;\n")],
    # The profile mountain shaders are generated from this source by the tree build.
    "Renderer/lab/shared/shaders/relief/beauty_mountain.hlsl": [
        ("        albedo = lerp(albedo, graded, rock_albedo_coverage);\n    }\n",
         "        albedo = lerp(albedo, graded, rock_albedo_coverage);\n    }\n"
         "#ifdef WONDER_REGION_KIND\n"
         "    // Lab only (studies/natural_wonders): where no volcano owns the relief, the owner\n"
         "    // channel carries a wonder stamp's region weights: x eases to the tile's own ground\n"
         "    // (vegetated slopes), y to the wonder's special material (1 snow, 2 travertine,\n"
         "    // 3 banded sandstone).\n"
         "    if (input.material.y > 1.5 && input.volcano_owner.z < 0.001) {\n"
         "        float wonder_ground = saturate(input.volcano_owner.x);\n"
         "        float wonder_special = saturate(input.volcano_owner.y);\n"
         "        // Civ VI's vegetated host ground (region 13): the skin's grass, since a\n"
         "        // mountain-hosted tile's own biome channels describe the mountain.\n"
         "        float3 vegetated = ground_surface_sample(GrassColor, input, 0.43, float2(.31, .17), false).rgb;\n"
         "        float vegetated_height = ground_surface_sample(GrassHeight, input, 0.43, float2(.31, .17), false).r;\n"
         "        albedo = lerp(albedo, vegetated, wonder_ground);\n"
         "        height_detail = lerp(height_detail, vegetated_height, wonder_ground);\n"
         "        float grain = saturate(height_detail * 1.6);\n"
         "#if WONDER_REGION_KIND == 1\n"
         "        float3 special = triplanar(SnowColor, input.world * MTN_ROCK_SCALE, geometric);\n"
         "#elif WONDER_REGION_KIND == 4\n"
         "        // A stratovolcano: the special weight ramps volcanic brown (0.5) to snow (1).\n"
         "        float3 special = lerp(float3(0.40, 0.26, 0.17) * lerp(0.72, 1.06, grain),\n"
         "                              triplanar(SnowColor, input.world * MTN_ROCK_SCALE, geometric),\n"
         "                              saturate(wonder_special * 2 - 1));\n"
         "        wonder_special = saturate(wonder_special * 2);\n"
         "#elif WONDER_REGION_KIND == 2\n"
         "        float3 special = lerp(ground_albedo, float3(0.80, 0.77, 0.69), 0.75) * lerp(0.80, 1.06, grain);\n"
         "#else\n"
         "        float band = frac(input.world.z * 7.0 + height_detail * 0.4);\n"
         "        float3 special = lerp(float3(0.42, 0.16, 0.08), float3(0.58, 0.26, 0.14), smoothstep(0.25, 0.75, band)) *\n"
         "                         lerp(0.86, 1.04, grain);\n"
         "#endif\n"
         "        albedo = lerp(albedo, special, wonder_special);\n"
         "    }\n"
         "#endif\n")],
}


def prepare() -> None:
    root = private_tree.tree(NAME)
    private_tree.own(NAME, list(PATCHES))
    # Owned files start from the checkout each time, so patches apply to its current text.
    for relative in PATCHES:
        (root / relative).write_bytes((ROOT / relative).read_bytes())
    private_tree.sync(NAME)
    for relative, edits in PATCHES.items():
        path = root / relative
        text = path.read_text()
        for old, new in edits:
            if text.count(old) != 1:
                raise SystemExit(f"{relative}: patch anchor not found once: {old[:70]!r}")
            text = text.replace(old, new)
        path.write_text(text)
    print("patched", len(PATCHES), "files in", root.relative_to(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("prepare", "build"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        private_tree.build(NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
