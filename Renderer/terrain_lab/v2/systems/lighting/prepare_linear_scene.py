#!/usr/bin/env python3
"""Extract a source-compatible scene-linear shader; never rewrite frozen inputs.

Only output operations are removed. Material sampling and authored source data
remain intact. Shared packet conversion must supply per-pass depth/blend state.
"""
import hashlib
import json
import re
from pathlib import Path

V2 = Path(__file__).resolve().parents[2]

def generate():
    source = V2 / 'shaders/common/frozen_l21.hlsl'
    s = source.read_text()
    edits = []
    for m in re.finditer(r'\bpow\(', s):
        depth, i, comma = 1, m.end(), None
        while depth:
            if s[i] == '(':
                depth += 1
            elif s[i] == ')':
                depth -= 1
            elif s[i] == ',' and depth == 1:
                comma = i
            i += 1
        if comma is not None and re.fullmatch(r'\s*1\.0\s*/\s*2\.2\s*', s[comma+1:i-1]):
            argument = s[m.end():comma].strip()
            # Remove only the display clamp, retaining physical material masks.
            if argument.startswith('saturate(') and argument.endswith(')'):
                argument = argument[len('saturate('):-1]
            edits.append((m.start(), i, '(' + argument + ')'))
    assert len(edits) == 12, 'frozen shader output extraction drift'
    for start, end, value in reversed(edits):
        s = s[:start] + value + s[end:]
    old = 'return exposure * (l13a_layout > 0.5 ? environment_exposure : 1.0);'
    assert s.count(old) == 1
    s = s.replace(old, 'return 1.0; // Q6: exposure belongs to final output only.')
    old = 'return l13a_layout > 0.5 ? color / (1.0 + color * 0.30) : color;'
    assert s.count(old) == 1
    s = s.replace(old, 'return color; // Q6: shared output applies the shoulder.')
    marker = '    return ambient + sunlight + moonlight;'
    assert s.count(marker)==1
    s=s.replace(marker,'''#ifdef Q6_GAMEPLAY_NIGHT
    // One shared clock-driven response for every composed receiver. Preserve
    // noon radiance; retain directional moon form instead of lifting black in
    // a postprocess or exposing terrain independently from foliage/water.
    ambient *= lerp(1.0, 1.8, environment_night_activation);
    moonlight *= 2.2;
#endif
'''+marker)
    # Three legacy diagnostic-only branches bypass frame_output_exposure.
    s = s.replace('coast_source * exposure', 'coast_source').replace('legacy_bed * exposure', 'legacy_bed')
    s = s.replace('float4 PSMain(PixelInput input) : SV_TARGET', 'float4 q6_raw_main(PixelInput input)')
    s = s.replace('float4 PSFeature(FeaturePixelInput input) : SV_TARGET', 'float4 q6_raw_feature(FeaturePixelInput input)')
    # Q3 owns continuous shoreline classification. Do not bake the inherited
    # land sand blend into albedo before its replacement material hook.
    marker = '    else if (shoreline_integrated > 0.5)\n    {\n        float transition_noise0'
    assert s.count(marker) == 1
    start=s.index(marker);brace=s.index('{',start);end=brace+1;depth=1
    while depth:
        if s[end]=='{':depth+=1
        elif s[end]=='}':depth-=1
        end+=1
    s=s[:start]+'#ifndef Q3_SHORE_MATERIAL\n'+s[start:end]+'\n#endif'+s[end:]
    # Opt-in Q0 world-geometry extension. Neither depth nor screen XY is used
    # to reconstruct world height. Legacy projected shadow vertices are invalid.
    marker = '    float material_tundra : TEXCOORD13;'
    assert s.count(marker) == 2
    s = s.replace(marker, marker + '''
#if defined(Q6_WORLD_SHADOWS) || defined(Q3_HYDROLOGY_DATA)
    float4 q6_world : TEXCOORD14;
#endif
#ifdef Q3_HYDROLOGY_DATA
    float4 hydrology_data : TEXCOORD15;
#endif''')
    marker = '    float material_index : TEXCOORD1;'
    assert s.count(marker) == 2
    s = s.replace(marker, marker + '''
#if defined(Q6_WORLD_SHADOWS) || defined(Q3_HYDROLOGY_DATA)
    float4 q6_world : TEXCOORD2;
#endif''')
    for marker in ['    output.material_tundra = input.material_tundra;',
                   '    output.material_index = input.material_index;']:
        assert s.count(marker) == 1
        s = s.replace(marker, marker + '''
#if defined(Q6_WORLD_SHADOWS) || defined(Q3_HYDROLOGY_DATA)
    output.q6_world = input.q6_world;
#endif''')
    marker = '    output.material_tundra = input.material_tundra;'
    s=s.replace(marker,marker+'''
#ifdef Q3_HYDROLOGY_DATA
    output.hydrology_data = input.hydrology_data;
#endif''')
    # Seven material evaluation call sites, not the function definition.
    calls = list(re.finditer(r'frame_illumination\(', s))
    assert len(calls) == 8
    for call in reversed(calls[1:]):
        s = s[:call.start()] + 'q6_receiver_illumination(input, ' + s[call.end():]
    marker = '        // L13A replaces the historical fixed-noon key'
    assert s.count(marker) == 1
    s = s.replace(marker, '''#ifdef Q3_SHORE_MATERIAL
        q3_shore_material(input, world_position, albedo, material_normal);
#endif
#ifdef Q2_MATERIAL_RESPONSE
        q2_material_form(input, world_position, geometry_normal, albedo, material_normal);
#endif
#ifdef Q8_DEBUG_ALBEDO
        return float4(albedo,1);
#endif
#ifdef Q8_DEBUG_NORMAL
        return float4(material_normal*.5+.5,1);
#endif
''' + marker)
    marker = '        float3 view_direction = float3(0.0, 0.0, 1.0);'
    assert s.count(marker) == 1
    s = s.replace(marker, '''#ifdef Q2_MATERIAL_RESPONSE
        q2_material_specular(input, world_position, geometry_normal, specular);
#endif
''' + marker)
    # External material include comes after texture/struct declarations.
    marker = '    float dune_weight = dune_region_weight(world_position);'
    assert s.count(marker)==1
    s=s.replace(marker,marker+'''
#ifdef Q4_BIQ_DUNE_COVERAGE
    // The old four-tile gallery rectangle is not a gameplay material selector.
    // The shared continuous desert weight preserves soft biome transitions.
    if(biq_layout>.5) dune_weight=saturate(input.material_weights.z);
#endif''')
    marker = '''    if (biq_layout > 0.5 && (input.base_terrain > 0.5 || input.real_terrain > 0.5))
        dune_weight = 0.0;'''
    assert s.count(marker)==1
    s=s.replace(marker,marker+'''
#ifdef Q4_BIQ_CONTINUOUS_DESERT
    if(biq_layout>.5) dune_weight=saturate(input.material_weights.z);
#endif''')
    # Reuse Q4's source-material projection in the combined scene. This is an
    # opt-in C3X projection adaptation, not a recovered source-engine shader.
    for name in ('mountain_base', 'mountain_top', 'mountain_snow',
                 'desert_mountain_base', 'desert_mountain_stripe1',
                 'desert_mountain_stripe2', 'desert_mountain_stripe3'):
        old = name + '_texture.Sample(material_sampler, input.uv).rgb'
        s = re.sub(r'\b' + re.escape(old),
                   'q4_relief_color(' + name + '_texture, input)', s)
    marker = 'float4 q6_raw_feature(FeaturePixelInput input)'
    s = s.replace(marker, '''#ifdef Q2_MATERIAL_RESPONSE
#include "../../terrain/scene_material_v1.hlsl"
#endif
#include "../../relief/combined_material.hlsl"
#include "../scene_shadow_v1.hlsl"
#ifdef Q3_WATER_MATERIAL
float4 q3_water_material(PixelInput input);
#endif
#ifdef Q3_SHORE_MATERIAL
void q3_shore_material(PixelInput input, float2 world_position, inout float3 albedo, inout float3 material_normal);
#endif

''' + marker)
    marker = 'float4 q6_raw_main(PixelInput input)\n{'
    assert s.count(marker) == 1
    s = s.replace(marker, marker + '''
#ifdef Q3_WATER_MATERIAL
    if(input.panel > .5 && ((input.surface_kind > 3.5 && input.surface_kind < 6.5)
        || (input.surface_kind > 8.5 && input.surface_kind < 9.5)))
        return q3_water_material(input);
#endif''')
    marker = '        // The density ramps operate over optical depth'
    assert s.count(marker)==1
    s=s.replace(marker,'''#ifdef Q6_WORLD_SHADOWS
        sun_glint *= q6_receiver_visibility(input, water_normal, 1.0);
#endif
'''+marker)
    marker = '        float3 highlight_color = l13a_layout > 0.5'
    assert s.count(marker)==1
    s=s.replace(marker,'''#ifdef Q6_WORLD_SHADOWS
        highlight *= q6_receiver_visibility(input, material_normal, 1.0);
#endif
'''+marker)
    s += '''
// Hardware composition consumes premultiplied scene-linear color exactly once.
struct Q6SceneOutput { float4 color : SV_Target0; float validity : SV_Target1; };
Q6SceneOutput q6_scene_output(float4 raw) {
    Q6SceneOutput o;
    float alpha = saturate(raw.a);
    clip(alpha - 0.000001);
    o.color = float4(max(raw.rgb, 0.0) * alpha, alpha);
    o.validity = 1.0;
    return o;
}
Q6SceneOutput PSMain(PixelInput input) { return q6_scene_output(q6_raw_main(input)); }
Q6SceneOutput PSFeature(FeaturePixelInput input) { return q6_scene_output(q6_raw_feature(input)); }
'''
    target = V2 / 'shaders/lighting/generated/scene_linear_v1.hlsl'
    target.parent.mkdir(exist_ok=True)
    target.write_text(s)
    record = {'schema': 'c3x.q6.scene_linear_adapter.v1',
              'source': str(source.relative_to(V2)),
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'output_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'display_conversions_removed': len(edits),
              'contract': 'q6_scene_linear_premultiplied_v1',
              'limitations': ['Inherited projected shadow triangles remain provisional.',
                              'Frozen source binding completeness is not Q7 composition acceptance.']}
    (V2 / 'audits/lighting/scene_linear_provenance.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))

if __name__ == '__main__':
    generate()
