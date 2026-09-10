#!/usr/bin/env python3
"""Isolated production-renderer material diagnosis; never stages a DLL or art.

Run with the workspace Python (Pillow). Uses renderer.py's category fixture and
Windows dispatcher. Shared shader sources are adapted only inside lab/out.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer import renderer
from Renderer.lab import platform, preparation

OUT = ROOT / 'Renderer/lab/out/mountains/zebra-study'
MIRROR = OUT / 'root'
SOURCE = 'Renderer/lab/shared/shaders/relief/beauty_mountain.hlsl'
TOP = 'float top = smoothstep(0.08, 0.62, mountain_rise) * (1 - snow);'
SNOW = 'float snow = smoothstep(0.79, 0.94, height) * smoothstep(0.24, 0.72, geometric.z);'


def variant_source(source, variant):
    if variant == 'current':
        return source
    if source.count(TOP) != 1 or source.count(SNOW) != 1:
        raise ValueError('Historical probes require --baseline-shader with the pre-promotion source; use current for the accepted shader')
    if variant in ('micro-relief', 'micro-relief-balanced'):
        source = variant_source(source, 'grain-focused')
        source = source.replace('snow * 0.60) * 0.25;',
                                'snow * 0.60) * ' + ('0.04;' if variant == 'micro-relief' else '0.08;'))
        if variant == 'micro-relief-balanced':
            source = source.replace('fine_world, geometric) * 0.12 * (1-snow);',
                                    'fine_world, geometric) * 0.08 * (1-snow);')
        return source
    if variant in ('stone-scale-3', 'stone-scale-5'):
        source = variant_source(source, 'snowcaps')
        scale = 3 if variant == 'stone-scale-3' else 5
        source = source.replace('p *= Quality.y;', f'p *= Quality.y * {scale}.0;')
        source = source.replace('Quality.z * lerp(1.0, 1.60, rock_detail_coverage)',
                                f'Quality.z * lerp(1.0, {1.60 / scale:.6f}, rock_detail_coverage)')
        return source
    if variant in ('gradient', 'gentle-bump', 'grain-focused'):
        source = variant_source(source, 'snowcaps')
        if variant == 'gentle-bump':
            return source.replace('Quality.z * lerp(1.0, 1.60, rock_detail_coverage)',
                                  'Quality.z * lerp(1.0, 0.48, rock_detail_coverage)')
        helper = (Path(__file__).parent / 'height_derivatives.hlsl').read_text()
        source = source.replace('float ggx(', helper + '\nfloat ggx(')
        source = source.replace('    float rock_crevice = 1;',
                                '    float rock_crevice = 1;\n    float2 rock_derivatives = 0;')
        key = '        float fine_rock = triplanar_scalar(RockHeight,\n            fine_world, geometric);'
        assert source.count(key) == 1
        source = source.replace(key, key + '''
        float2 rock_gradient = study_height_derivatives(RockHeight, input.world, geometric);
        float2 layered_gradient = rock_gradient * base +
            study_height_derivatives(TopHeight, input.world, geometric) * top +
            study_height_derivatives(SnowHeight, input.world, geometric) * snow;
        rock_derivatives = lerp(layered_gradient, rock_gradient, snow * 0.60);
        rock_derivatives += study_height_derivatives(RockHeight, fine_world, geometric) * 0.12 * (1-snow);
''')
        key = 'detail_normal(geometric, input.world,\n                                                    height_detail, rock_normal_strength)'
        assert source.count(key) == 1
        source = source.replace(key,
            'study_detail_normal(geometric, input.world, lerp(float2(ddx(ground_height), ddy(ground_height)), rock_derivatives, rock_detail_coverage), rock_normal_strength)')
        if variant == 'grain-focused':
            source = source.replace('rock_derivatives = lerp(layered_gradient, rock_gradient, snow * 0.60);',
                                    'rock_derivatives = lerp(layered_gradient, rock_gradient, snow * 0.60) * 0.25;')
        return source
    if variant.startswith('probe-'):
        source = variant_source(source, 'snowcaps')
        if variant in ('probe-contrast', 'probe-geometry'):
            source = source.replace('float crevice_visibility = lerp(1.0, rock_crevice, rock_albedo_coverage);',
                                    'float crevice_visibility = 1;').replace(
                                    'mountain_albedo *= lerp(1.0, grain, 1 - snow);', '')
        if variant in ('probe-normal', 'probe-geometry'):
            source = source.replace('float rock_normal_strength = Quality.z * lerp(1.0, 1.60, rock_detail_coverage);',
                                    'float rock_normal_strength = Quality.z * (1-rock_detail_coverage);')
        if variant == 'probe-geometry':
            source = source.replace('albedo = lerp(ground_albedo, mountain_albedo, rock_albedo_coverage);',
                                    'albedo = lerp(ground_albedo, float3(.35,.35,.35), rock_albedo_coverage);')
        if variant == 'probe-shadow':
            source = source.replace('    float ndl = saturate(dot(normal, light_direction));',
                                    '    shadow = 1;\n    float ndl = saturate(dot(normal, light_direction));')
        if variant == 'probe-rotate':
            source = source.replace('texture_map.Sample(Wrap, p.yz)', 'texture_map.Sample(Wrap, float2(p.z,-p.y))').replace(
                                    'texture_map.Sample(Wrap, p.xz)', 'texture_map.Sample(Wrap, float2(p.z,-p.x))').replace(
                                    'texture_map.SampleBias(Wrap, p.yz, 3)', 'texture_map.SampleBias(Wrap, float2(p.z,-p.y), 3)').replace(
                                    'texture_map.SampleBias(Wrap, p.xz, 3)', 'texture_map.SampleBias(Wrap, float2(p.z,-p.x), 3)')
        return source
    if variant == 'no-top':
        return source.replace(TOP, 'float top = 0; // Diagnostic: retain base rock and existing snow.')
    if variant == 'snowcaps':
        return source.replace(TOP,
            'float top = smoothstep(0.52, 0.68, height) * (1 - snow);').replace(SNOW,
            'float snow = smoothstep(0.62, 0.78, height) * smoothstep(0.02, 0.25, geometric.z);')
    if variant == 'summit':
        return source.replace(TOP,
            'float top = smoothstep(0.60, 0.78, height) * (1 - snow);').replace(SNOW,
            'float snow = smoothstep(0.76, 0.90, height) * smoothstep(0.04, 0.35, geometric.z);')
    if variant == 'no-extra-contrast':
        return source.replace('float crevice_visibility = lerp(1.0, rock_crevice, rock_albedo_coverage);',
                              'float crevice_visibility = 1;').replace(
                              'mountain_albedo *= lerp(1.0, grain, 1 - snow);', '')
    assert variant == 'current'
    return source


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    # Independent shader copies, including unchanged native-only shaders.
    for p in (ROOT / 'Renderer/native').rglob('*.hlsl'):
        q = MIRROR / p.relative_to(ROOT)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
    for name in preparation.input_paths(ROOT):
        q = MIRROR / name
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, q)
    # Read-only hardlinked art avoids duplicating several GB of source assets.
    # This study never writes pack files, including the linked inputs.
    packs = MIRROR / 'Renderer/packs'
    if not packs.exists():
        shutil.copytree(ROOT / 'Renderer/packs', packs, copy_function=os.link)
    for p in (ROOT / 'Renderer').glob('*.txt'):
        shutil.copyfile(p, MIRROR / 'Renderer' / p.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', default=['current'],
                        choices=['current', 'no-top', 'summit', 'snowcaps', 'no-extra-contrast', 'probe-contrast', 'probe-normal', 'probe-shadow', 'probe-geometry', 'probe-rotate', 'gradient', 'gentle-bump', 'grain-focused', 'stone-scale-3', 'stone-scale-5', 'micro-relief', 'micro-relief-balanced'])
    parser.add_argument('--cases', nargs='+', default=['detail', 'gameplay', 'coastal'])
    parser.add_argument('--hour', type=int, default=12)
    parser.add_argument('--zoom', type=int, default=128)
    parser.add_argument('--reuse-candidate', action='store_true',
                        help='Continue with the already copied study DLL during unrelated C++ work')
    parser.add_argument('--baseline-shader', type=Path, help='Pre-promotion source for historical diagnostic variants')
    args = parser.parse_args()
    if not args.reuse_candidate:
        renderer.require_current_candidate()
    renderer.ensure_preview_tool()
    setup()
    candidate = OUT / 'C3XRenderer.dll'
    if not args.reuse_candidate:
        shutil.copyfile(ROOT / 'Renderer/native/build/candidate/C3XRenderer.dll', candidate)
        shutil.copyfile(renderer.LAB / '.cache/native-build.json', OUT / 'candidate-build.json')
    if not candidate.is_file():
        raise ValueError('No existing isolated study candidate')
    staged = ROOT / 'Renderer/bin/C3XRenderer.dll'
    staged_hash = renderer.checksum(staged)
    native_run = platform.run_native_fixture

    def isolated(directory, command, run_id):
        batch = directory / 'render.bat'
        body = batch.read_text()
        needle = 'C3XRenderer.dll" ..\\.. '
        assert body.count(needle) == 1
        body = body.replace(needle, 'C3XRenderer.dll" ..\\lab\\out\\mountains\\zebra-study\\root ')
        batch.write_text(body)
        try:
            return native_run(directory, command, run_id)
        except ValueError:
            process = platform.fixture_process(directory, run_id)
            if process is None:
                raise
            return platform.wait_native_fixture(directory, run_id, process)

    platform.run_native_fixture = isolated
    source = (args.baseline_shader or ROOT / SOURCE).read_text()
    receipt = OUT / f'render-{args.hour}-{args.zoom}.json'
    previous = renderer.read(receipt) if receipt.exists() else {}
    records = previous.get('renders', []) if previous.get('candidate_sha256') == renderer.checksum(candidate) else []
    try:
        for variant in args.variants:
            (MIRROR / SOURCE).write_text(variant_source(source, variant))
            preparation.generate(MIRROR)
            shaders = {renderer.relative(p): renderer.checksum(p) for p in
                       (MIRROR / 'Renderer/native').rglob('*.hlsl')}
            for case in args.cases:
                destination = OUT / variant / f'{case}-h{args.hour:02}-z{args.zoom}'
                renderer.native_render('mountains', case, args.hour, args.zoom,
                                       destination, candidate=candidate)
                bmp = destination / f'{case}-h{args.hour:02}-z{args.zoom}.bmp'
                from PIL import Image
                Image.open(bmp).convert('RGB').save(destination / 'preview.png')
                records = [r for r in records if (r['variant'], r['case'], r['hour'], r['zoom']) !=
                           (variant, case, args.hour, args.zoom)]
                records.append(dict(variant=variant, case=case, hour=args.hour, zoom=args.zoom,
                                    image=renderer.relative(bmp), sha256=renderer.checksum(bmp),
                                    shaders=shaders))
            renderer.write(receipt, dict(
                fixture='synthetic category scene, production D3D11 renderer',
                candidate_sha256=renderer.checksum(candidate), renders=records,
                production_changed=False))
    finally:
        platform.run_native_fixture = native_run
        assert renderer.checksum(staged) == staged_hash, 'Staged DLL changed during study'


if __name__ == '__main__':
    main()
