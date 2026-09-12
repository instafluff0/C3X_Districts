#!/usr/bin/env python3
"""Mountain-body controls using the frozen accepted volcano Lab scene.

Uses renderer.py's scene/Windows dispatcher. Only private shared shaders change;
the accepted volcano study, production sources, art and staged DLL are inputs.
"""
from pathlib import Path
import argparse
import os
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer import renderer
from Renderer.lab import platform, preparation

OUT = ROOT / 'Renderer/lab/out/mountains/body-study'
SEED = ROOT / 'Renderer/lab/out/volcanoes/material-study'
MIRROR = OUT / 'root'
SOURCE = 'Renderer/lab/shared/shaders/relief/beauty_mountain.hlsl'
VARIANTS = ('current', 'source-color', 'calmer-normal', 'clean-rock', 'balanced-rock', 'no-fine-cavity', 'slope-rock', 'collar-no-bump', 'collar-triplanar')


def collar_source(source, variant, preserve_volcano):
    source = variant_source(source, 'slope-rock', preserve_volcano)
    helper = '''
float study_collar(P input) {
    float rise = max(0, input.world.z - input.base_relief - 2.5/112.0);
    float slope = 1-saturate(normalize(input.normal).z);
    return smoothstep(.01,.12,rise) * smoothstep(.16,.48,slope);
}
float4 study_ground_sample(Texture2D tex, P input, float scale, float2 offset, bool rotated) {
    float3 p=input.world;
    float3 n=normalize(input.normal);
    if(rotated) { p=float3(p.y,-p.x,p.z); n=float3(n.y,-n.x,n.z); }
    float3 w=pow(abs(n),5); w/=max(dot(w,1),.00001);
    float4 top=tex.Sample(Wrap,p.xy*scale+offset);
    float4 sides=tex.Sample(Wrap,p.yz*scale+offset)*w.x +
                 tex.Sample(Wrap,p.xz*scale+offset)*w.y + top*w.z;
    return lerp(top,sides,study_collar(input));
}
'''
    if preserve_volcano:
        helper = helper.replace('    return smoothstep(.01,.12,rise)', '''    float2 delta=input.world.xy-float2(16.5,.5);
    float volcano=smoothstep(.025,.20,input.world.z) *
        (1-smoothstep(.60,.78,max(abs(delta.x),abs(delta.y))));
    return (1-volcano) * smoothstep(.01,.12,rise)''')
    source = replace_once(source, 'void ground_material(', helper + '\nvoid ground_material(')
    if variant == 'collar-no-bump':
        return replace_once(source, 'float2(ddx(ground_height), ddy(ground_height)), rock_derivatives',
                            'float2(ddx(ground_height), ddy(ground_height)) * (1-study_collar(input)), rock_derivatives')
    start=source.index('void ground_material(')
    end=source.index('Output shade(P input)',start)
    body=source[start:end]
    scales={'uv0':('0.43','float2(.31,.17)','false'),
            'uv1':('0.43*.91','float2(.63,.29)','true'),
            'tundra_uv':('0.43*.84','float2(.19,.71)','false')}
    def sample(match):
        tex,uv,hill=match.groups(); scale,offset,rotated=scales[uv]
        if hill:scale=f'({scale})*1.08';offset=f'{offset}*1.08'
        return f'study_ground_sample({tex}, input, {scale}, {offset}, {rotated})'
    body,count=re.subn(r'(\w+)\.Sample\(Wrap, (uv0|uv1|tundra_uv)( \* 1\.08)?\)',sample,body)
    if count != 20: raise ValueError(f'Ground sample inventory changed: {count}')
    return source[:start]+body+source[end:]


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Frozen mountain shader changed: ' + old)
    return source.replace(old, new)


def variant_source(source, variant, preserve_volcano=True):
    if variant.startswith('collar-'):
        return collar_source(source, variant, preserve_volcano)
    if variant == 'current':
        return source
    source = replace_once(source, '        float base = 1 - top - snow;', '''        float base = 1 - top - snow;
        // Lab body-only mask: preserve both snow layers and the accepted
        // neighboring volcano, which inherits this material's normal response.
        float2 study_offset = input.world.xy - float2(16.5,.5);
        float study_volcano = smoothstep(.025,.20,input.world.z) *
            (1-smoothstep(.60,.78,max(abs(study_offset.x),abs(study_offset.y))));
        float study_body = (1-smoothstep(.48,.62,height)) * (1-study_volcano);''')
    if variant == 'slope-rock':
        source = replace_once(source,
            '(1-smoothstep(.48,.62,height)) * (1-study_volcano)',
            'smoothstep(.38,.75,mountain_rise) * (1-study_volcano)')
    if not preserve_volcano:
        source = source.replace('* (1-study_volcano)', '* 1.0')
    if variant in ('source-color', 'clean-rock', 'balanced-rock', 'slope-rock'):
        contrast = {'balanced-rock': '(study_body * .72)',
                    'slope-rock': '(study_body * .92)'}.get(variant, 'study_body')
        source = replace_once(source,
            '        mountain_albedo *= lerp(1.0, grain, 1 - snow);',
            f'        mountain_albedo *= lerp(1.0, grain, (1 - snow) * (1-{contrast}));\n'
            f'        rock_crevice = lerp(rock_crevice, 1.0, {contrast});')
        source = replace_once(source,
            '        mountain_albedo *= lerp(0.82, 1.06, rock_micro_relief);',
            f'        mountain_albedo *= lerp(lerp(0.82, 1.06, rock_micro_relief), 1.0, {contrast}' +
            (' * (1-snow)' if variant == 'slope-rock' else '') + ');')
    if variant == 'no-fine-cavity':
        source = replace_once(source,
            'rock_crevice_visibility(fine_rock, fine_neighborhood), 1.0, snow);',
            'lerp(rock_crevice_visibility(fine_rock, fine_neighborhood), 1.0, study_body), 1.0, snow);')
    if variant in ('calmer-normal', 'clean-rock'):
        source = replace_once(source,
            'rock_derivatives += triplanar_height_derivatives(RockHeight, fine_world, geometric) * 0.12 * (1-snow);',
            'rock_derivatives += triplanar_height_derivatives(RockHeight, fine_world, geometric) * lerp(0.12, 0.04, study_body) * (1-snow);')
    return source


def setup():
    if (OUT / 'baseline.hlsl').exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = renderer.read(SEED / 'render-224.json')
    if renderer.checksum(SEED / 'C3XRenderer.dll') != receipt['candidate_sha256']:
        raise ValueError('Volcano study candidate identity changed')
    # Only assets are hardlinked and remain strictly read-only.
    for p in (SEED / 'root').rglob('*'):
        if not p.is_file():
            continue
        q = MIRROR / p.relative_to(SEED / 'root')
        q.parent.mkdir(parents=True, exist_ok=True)
        if 'packs' in p.relative_to(SEED / 'root').parts:
            if not q.exists():
                os.link(p, q)
        else:
            shutil.copyfile(p, q)
    for name in ('C3XRenderer.dll', 'native_preview.exe', 'candidate-build.json'):
        shutil.copyfile(SEED / name, OUT / name)
    shader = MIRROR / SOURCE
    accepted = next(r for r in receipt['renders'] if r['variant'] == 'skin-lava-shadow-probe' and r['case'] == 'gameplay')
    for name, digest in accepted['shaders'].items():
        if renderer.checksum(MIRROR / name) != digest:
            raise ValueError('Frozen shaders differ from accepted volcano preview: ' + name)
    shutil.copyfile(shader, OUT / 'baseline.hlsl')
    renderer.write(OUT / 'baseline.json', dict(accepted_render=accepted,
        candidate_sha256=receipt['candidate_sha256'],
        preview_sha256=receipt['preview_sha256'], diagnostic_only=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--category', choices=['mountains', 'volcanoes'], default='volcanoes')
    parser.add_argument('--cases', nargs='+', default=['gameplay'])
    parser.add_argument('--zoom', type=int, default=224)
    parser.add_argument('--hour', type=int, default=12)
    parser.add_argument('--repeat', action='store_true')
    args = parser.parse_args()
    setup()
    terrain_name = 'Renderer/lab/shared/shaders/relief/beauty_terrain.hlsl'
    caster_name = 'Renderer/native/render_core/source_caster.hlsl'
    for label, name in [('terrain', terrain_name), ('caster', caster_name)]:
        if not (OUT / f'baseline-{label}.hlsl').exists():
            shutil.copyfile(MIRROR / name, OUT / f'baseline-{label}.hlsl')
    from PIL import Image
    original_run = platform.run_native_fixture
    def isolated(directory, command, run_id):
        batch = directory / 'render.bat'
        batch.write_text(replace_once(batch.read_text(), 'C3XRenderer.dll" ..\\.. ',
            'C3XRenderer.dll" ..\\lab\\out\\mountains\\body-study\\root '))
        return original_run(directory, command, run_id)
    platform.run_native_fixture = isolated
    source = (OUT / 'baseline.hlsl').read_text()
    terrain = (OUT / 'baseline-terrain.hlsl').read_text()
    caster = (OUT / 'baseline-caster.hlsl').read_text()
    if args.category == 'mountains':
        # Remove only the known fixed volcano probe from both natural shaders.
        # Ordinary mountain fixtures must not acquire a phantom volcano skin.
        def without_volcano(text):
            start = text.index('    // Fixed source-UV routing proof,')
            end = text.index('    float ndl =', start)
            return (text[:start] + text[end:]).replace(
                'Texture2D StudyVolcanoColor : register(t69);\nTexture2D StudyLavaColor : register(t71);\n', '')
        source, terrain = without_volcano(source), without_volcano(terrain)
        caster = caster.replace('float boundary:TEXCOORD4;float3 world:TEXCOORD5;', 'float boundary:TEXCOORD4;')
        caster = caster.replace('o.boundary=i.material;o.world=i.world.xyz;return o;', 'o.boundary=i.material;return o;')
        caster = replace_once(caster, '''  bool volcano_body=i.world.z>.045 && all(abs(i.world.xy-float2(16.5,.5))<.80);
  if(!volcano_body)clip(smoothstep(.08,.72,i.coverage)-.45);return i.depth;''',
            '  clip(smoothstep(.08,.72,i.coverage)-.45);return i.depth;')
    (MIRROR / terrain_name).write_text(terrain)
    (MIRROR / caster_name).write_text(caster)
    records_path = OUT / 'renders.json'
    records = renderer.read(records_path) if records_path.exists() else []
    try:
        for variant in args.variants:
            (MIRROR / SOURCE).write_text(variant_source(source, variant, args.category == 'volcanoes'))
            preparation.generate(MIRROR)
            for case in args.cases:
                key = f'{args.category}-{case}-h{args.hour:02}-z{args.zoom}'
                destination = OUT / variant / (key + ('-repeat' if args.repeat else ''))
                print(f'Rendering {variant} {key}', flush=True)
                record = renderer.native_render(args.category, case, args.hour, args.zoom,
                    destination, candidate=OUT / 'C3XRenderer.dll', preview=OUT / 'native_preview.exe')
                Image.open(ROOT / record['image']).convert('RGB').save(destination / 'preview.png')
                record.update(variant=variant, category=args.category, repeat=args.repeat,
                    shader_sha256=renderer.checksum(MIRROR / SOURCE))
                records.append(record)
                renderer.write(records_path, records)
    finally:
        platform.run_native_fixture = original_run


if __name__ == '__main__':
    main()
