#!/usr/bin/env python3
"""Generate the native binding adapter from immutable pickup shader modules.

The reference tree is never edited. Binding, gameplay state and world-wrap
adaptations are explicit here; source material/lighting equations stay pinned.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent


def generate():
    native = ROOT.parent
    production = (native / 'terrain_rendering.hlsl').read_text()
    start = production.index('#ifdef C3X_GAME_RENDERER')
    end = production.index('#else', start)
    settings = production[start:end]
    records = []
    for source in sorted((ROOT / 'reference/shaders').rglob('*.hlsl')):
        if source.name == 'frozen_l21.hlsl':
            continue
        text = source.read_text()
        if source.name == 'scene_linear_v1.hlsl':
            begin = text.index('#ifdef C3X_GAME_RENDERER')
            finish = text.index('#else', begin)
            text = text[:begin] + settings + text[finish:]
            # The captured source OWNER supplies activity on neighboring land.
            text = text.replace('? input.active_effect : q4_volcano_active(input,fixture_active)',
                                '? input.relief_material.w : q4_volcano_active(input,fixture_active)')
        if source.name == 'frame_shadow_v1.hlsl':
            text = text.replace('register(b1)', 'register(b2)')
        if source.parent.name == 'hydrology' and source.name == 'scene_material_v1.hlsl':
            begin = text.index('float2 q3_source_world(')
            finish = text.index('float4 q3_authored_bed_detail(', begin)
            text = text[:begin] + '''// Captured world dimensions/flags; b1 remains native viewport, b2 shadows.
cbuffer C3XWorldSettings : register(b3) {
 float4 c3x_world_dimensions; // width, height, wrap-x, wrap-y
 float4 c3x_world_material; // common repeat period, reserved
};
float2 q3_source_world(PixelInput input) {
 float2 world=input.macro_uv*2;
 float2 raw=float2(world.x+world.y,world.x-world.y);
 if(c3x_world_dimensions.z>.5)
  raw.x-=floor(raw.x/c3x_world_dimensions.x)*c3x_world_dimensions.x;
 if(c3x_world_dimensions.w>.5)
  raw.y-=floor(raw.y/c3x_world_dimensions.y)*c3x_world_dimensions.y;
 return float2(raw.x+raw.y,raw.x-raw.y)*.5;
}
float q3_source_repeat(float requested) {
 float period=c3x_world_material.x;
 return period>0?round(requested*period)/period:requested;
}
''' + text[finish:]
        target = ROOT / 'generated' / source.relative_to(ROOT / 'reference')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        records.append({'path': target.relative_to(ROOT).as_posix(),
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    adapter = (native / 'integrated_terrain.hlsl').read_text()
    adapter = adapter.replace('#include "terrain_rendering.hlsl"', '''#define Q6_TEXEL_RECEIVER_OFFSET 1
#define Q4_RELIEF_MATERIAL_DATA 1
#define Q4_BROAD_RELIEF 1
#define Q4_VOLCANO_FOOTPRINT .3875
#define Q3_SOURCE_WATER_NORMALS 1
#define Q4_COASTAL_ROCKS 1
#define Q3_COAST_DETAIL 1
#define Q6_GAMEPLAY_NIGHT 1
#define Q4_BIQ_CONTINUOUS_DESERT 1
#define Q4_BIQ_DUNE_COVERAGE 1
#define Q4_COMBINED_ROCK_PROJECTION 1
#define Q3_STATIC_OPTICS_V2 1
#define Q6_WORLD_SHADOWS 1
#include "generated/shaders/hydrology/scene_linear.hlsl"''')
    adapter = adapter.replace('    float material_tundra : TEXCOORD13;', '''    float material_tundra : TEXCOORD13;
    float4 q6_world : TEXCOORD14;
    float4 hydrology_data : TEXCOORD15;
    float4 relief_material : TEXCOORD16;''')
    adapter = adapter.replace('    output.material_tundra = input.material_tundra;', '''    output.material_tundra = input.material_tundra;
    output.q6_world = input.q6_world;
    output.hydrology_data = input.hydrology_data;
    output.relief_material = input.relief_material;''')
    adapter = adapter.replace('    output.material_index = input.base_terrain;', '''    output.material_index = input.base_terrain;
    output.q6_world = input.q6_world;''')
    # Native coverage lives in accumulated alpha; there is no wire-packet MRT.
    adapter = adapter.replace('return PSMain(input);', 'return PSMain(input).color;')
    adapter = adapter.replace('return PSFeature(input);', 'return PSFeature(input).color;')
    # The native standard include handler does not implement the Lab packet
    # preprocessor's nested include search. Flatten the pinned closure offline;
    # runtime compilation then has one deterministic, hashable input file.
    def expand(text, directory):
        def include(match):
            path = (directory / match.group(1)).resolve()
            path.relative_to(ROOT)
            return expand(path.read_text(), path.parent)
        return re.sub(r'^#include "([^"]+)"\s*$', include, text, flags=re.MULTILINE)
    adapter = expand(adapter, ROOT)
    (ROOT / 'integrated_v2.hlsl').write_text(adapter)
    records.append({'path': 'integrated_v2.hlsl',
                    'sha256': hashlib.sha256(adapter.encode()).hexdigest()})
    (ROOT / 'generated.json').write_text(json.dumps({'schema': 'c3x.native.shader_adapter.v1',
        'candidate': 'lab_v2_terrain_lighting_r1', 'files': records}, indent=2)+'\n')


if __name__ == '__main__':
    generate()
