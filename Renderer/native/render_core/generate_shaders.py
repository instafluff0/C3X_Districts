#!/usr/bin/env python3
"""Generate the native binding adapter from the preserved shader sources.

The source tree is never edited. Binding, gameplay state and world-wrap
adaptations are explicit here; source material/lighting equations stay pinned.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent


def generate(source_root=None, output_root=None, output_name='terrain_scene.hlsl', *, complete_rock_channels=False):
    source_root = (Path(source_root) if source_root else ROOT / 'source').resolve()
    output_root = (Path(output_root) if output_root else ROOT).resolve()
    native = ROOT.parent
    production = (native / 'terrain_rendering.hlsl').read_text()
    start = production.index('#ifdef C3X_GAME_RENDERER')
    end = production.index('#else', start)
    settings = production[start:end]
    records = []
    for source in sorted((source_root / 'shaders').rglob('*.hlsl')):
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
            # Resource bodies retain their full source texture and mip chain.
            # Apply a restrained gameplay-scale detail bias only to true
            # resource materials; shared mine/city slot reuse remains neutral.
            for slot in range(8):
                old = (f'albedo = resource_base_texture_{slot}.Sample('
                       'material_sampler, input.uv).rgb;')
                new = (f'albedo = resource_base_texture_{slot}.SampleBias('
                       'material_sampler, input.uv, resource_weight * -0.45).rgb;')
                assert text.count(old) == 1
                text = text.replace(old, new)
            old = '''        float feature_form = raised_form_response(signed_diffuse);
        light *= feature_form;'''
            assert text.count(old) == 1
            text = text.replace(old, '''        float feature_form = raised_form_response(signed_diffuse);
        // Small resource bodies need more ambient-facing readability than
        // relief while keeping the shared sun direction and authored normals.
        // Open the opposing face without flattening lit crowns or lifting the
        // rest of the scene.
        float resource_form = lerp(0.70, 1.14,
            smoothstep(0.08, 0.84, signed_diffuse));
        light *= lerp(feature_form, resource_form, resource_weight);
        light += resource_weight * environment_ambient_color * 0.16;''')
            marker = '''    if (input.panel > 0.5 && input.surface_kind > 13.5 && input.surface_kind < 14.5)
    {'''
            assert text.count(marker) == 1
            start = text.index(marker)
            brace = text.index('{', start)
            finish = brace + 1
            depth = 1
            while depth:
                depth += (text[finish] == '{') - (text[finish] == '}')
                finish += 1
            text = text[:finish] + '''
    if (input.panel > 0.5 && input.surface_kind > 14.5 && input.surface_kind < 15.5)
    {
        // Animated resources project their current posed source triangles.
        // Preserve the source alpha-cutout cards instead of shadowing their
        // complete rectangular geometry.  This is the same coverage channel
        // and cutoff used by the resource body pass.
        float coverage = resource_base_texture_0.Sample(
            material_sampler, input.uv).a;
        clip(coverage - 0.08);
        // One restrained sample anchors the body without a generic blob or a
        // full static-scene rerender on every animation tick.
        float alpha = environment_shadow_strength * c3x_dynamic_shadow_opacity *
            smoothstep(0.08, 0.35, coverage);
        clip(alpha - 0.004);
        return float4(0.008, 0.011, 0.016, alpha);
    }''' + text[finish:]
            text=text.replace('tile_object_weight * 0.07);','tile_object_weight * 0.07 * (1.0-step(0.175, material_fraction)));')
            policy = native.parent / 'lab/shared/shaders/lighting/shadow_policy.hlsl'
            text = policy.read_text() + '\n' + text
        if source.name == 'frame_shadow_v1.hlsl':
            text = text.replace('register(b1)', 'register(b2)')
            begin = text.index('float q6_world_visibility(')
            receiver = (ROOT / 'shadow_receiver.hlsl').read_text().replace(
                '// C3X_SHARED_PAGED_SHADOW',
                (native.parent / 'lab/shared/shaders/lighting/paged_shadow_v1.hlsl').read_text())
            text = text[:begin] + receiver + '\n#endif\n'
        if source.name == 'scene_shadow_v1.hlsl':
            text = text.replace('q6_world_visibility(feature_base_texture_0,',
                                'q6_world_visibility(pickup_shadow_terrain,')
            text = text.replace('q6_world_visibility(shallow_bed_texture,',
                                'q6_world_visibility(pickup_shadow_feature,')
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
        target = output_root / 'generated' / source.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        records.append({'path': target.relative_to(output_root).as_posix(),
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
    if complete_rock_channels:
        adapter = adapter.replace('#define Q6_TEXEL_RECEIVER_OFFSET 1',
            '#define Q4_COMPLETE_ROCK_CHANNELS 1\n#define Q4_COHERENT_ROCK_CHANNELS 1\n#define Q6_TEXEL_RECEIVER_OFFSET 1')
    adapter = adapter.replace('    float material_tundra : TEXCOORD13;', '''    float material_tundra : TEXCOORD13;
    float4 q6_world : TEXCOORD14;
    float4 hydrology_data : TEXCOORD15;
    float4 relief_material : TEXCOORD16;''')
    adapter = adapter.replace('PixelInput VSIntegrated(IntegratedVertexInput input)\n{',
        'PixelInput VSIntegrated(IntegratedVertexInput input)\n{\n'
        '    if(c3x_viewport_reserved.y>0)input.surface_kind=c3x_viewport_reserved.y;')
    adapter = adapter.replace('    output.material_tundra = input.material_tundra;', '''    output.material_tundra = input.material_tundra;
    output.q6_world = input.q6_world;
    output.hydrology_data = input.hydrology_data;
    output.relief_material = input.relief_material;''')
    adapter = adapter.replace('    output.material_index = input.base_terrain;', '''    output.material_index = input.base_terrain;
    output.q6_world = input.q6_world;''')
    # Features upload only the full-precision fields consumed by this entry.
    # A separate signature avoids requiring unused terrain input elements.
    adapter = adapter.replace('FeaturePixelInput VSIntegratedFeature(IntegratedVertexInput input)\n{', '''struct PackedFeatureInput {
 float3 position:POSITION;float2 uv:TEXCOORD0;float3 normal:NORMAL;
 float material:TEXCOORD6;float3 world:TEXCOORD14;
};
FeaturePixelInput VSIntegratedFeature(PackedFeatureInput packed)
{
 IntegratedVertexInput input=(IntegratedVertexInput)0;
 input.position=packed.position;input.uv=packed.uv;input.geometry_normal=packed.normal;
 input.base_terrain=packed.material;input.q6_world=float4(packed.world,1);''')
    # Native coverage lives in accumulated alpha; there is no wire-packet MRT.
    adapter = adapter.replace('return PSMain(input);', 'return PSMain(input).color;')
    adapter = adapter.replace('return PSFeature(input);', 'return PSFeature(input).color;')
    # The native standard include handler does not implement the Lab packet
    # preprocessor's nested include search. Flatten the pinned closure offline;
    # runtime compilation then has one deterministic, hashable input file.
    def expand(text, directory):
        def include(match):
            path = (directory / match.group(1)).resolve()
            path.relative_to(output_root)
            return expand(path.read_text(), path.parent)
        return re.sub(r'^#include "([^"]+)"\s*$', include, text, flags=re.MULTILINE)
    adapter = expand(adapter, output_root)
    (output_root / output_name).write_text(adapter)
    records.append({'path': output_name,
                    'sha256': hashlib.sha256(adapter.encode()).hexdigest()})
    manifest_name = 'shader_manifest.json' if output_name == 'terrain_scene.hlsl' else 'generated.json'
    (output_root / manifest_name).write_text(json.dumps({'schema': 'c3x.native.shader_adapter.v1',
        'component': 'render_core' if output_name == 'terrain_scene.hlsl' else 'source-fidelity-retained-hydrology',
        'files': records}, indent=2)+'\n')


if __name__ == '__main__':
    generate()
