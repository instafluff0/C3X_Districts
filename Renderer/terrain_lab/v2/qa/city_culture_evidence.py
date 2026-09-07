"""Check saved individual-house culture studies and composed night-light inputs."""
import json
import subprocess
from PIL import Image, ImageDraw
from city_growth_evidence import ROOT, V2, OUT, FIX, read, sha, placement, clearance
from city_growth_hierarchy_evidence import difference
from city_scene_pass import executable, Cache
from city_facade_light_probe import derive

BASE = OUT/'city-culture-r2'
CASES = {'asian-seven': 58, 'ancient-seven': 59, 'asian-medium': 60,
         'ancient-medium': 61, 'asian-large': 62, 'asian-holdout': 64}
SELECTED = ['asian-medium', 'ancient-medium', 'asian-large', 'asian-holdout']


def main():
    frame = executable(V2/'qa/city_shadow_frame_contract.cpp', Cache(V2/'app/.cache'))
    catalog = read(ROOT/'Renderer/packs/CityStudyAuxiliaryUV/city_catalog.json')
    cases, augmentations, windows = {}, {}, []
    for name, revision in CASES.items():
        path = next((FIX/f'city-scene-r{revision}').glob('*/augmentation.json'))
        a = read(path); augmentations[name] = a
        assert a['generator_profile']['era_policy'] == 'single_current_era_user_preference'
        assert all(i['asset'] in catalog['pools'][a['pool']]['components'] for i in a['instances'])
        expected = 7 if name.endswith('seven') else 24 if name == 'asian-large' else 16
        assert len(a['instances']) == expected
        assert read(path.parent/'surface.json')['region']['region']['extent'] == [10, 10]
        raw = OUT/f'city-scene-r{revision}'/path.parent.name/'combined'
        result = BASE/name/'render'; source_report = read(raw/'report.json')
        # Windows can consume prepared closures while Metal parity is pending.
        jobs = read(result/'batch.json'); report = source_report
        assert [str(ROOT/p['path']) for p in report['packets']] == [job[0] for job in jobs]
        assert sha(raw/'postprocess/source.hlsl') == sha(result/'postprocess/source.hlsl')
        lights = read(BASE/name/'lights.json')
        assert lights['augmentation_sha256'] == sha(path)
        assert len(lights['blockers']) == expected and len(lights['lights']) <= expected*4
        complete = derive(a, read(path.parent/'surface.json'), 128)
        assert len(lights['lights']) <= 32
        assert {l['owner'] for l in lights['lights']} == {l['owner'] for l in complete['lights']}
        assert {l['owner'] for l in lights['lights']} <= set(range(expected))
        for texture, digest in lights['texture_sha256'].items():
            assert sha(ROOT/texture) == digest
        reference = read(ROOT/a['shadow_frame_report']['path'])
        packet_checks = []
        for index in range(2):
            p = ROOT/report['packets'][index]['path']
            assert sha(p) == report['packets'][index]['sha256']
            packet_checks.append(json.loads(subprocess.check_output([str(frame), str(p),
                       str(ROOT/reference['outputs'][index]['packet']), str(p)], text=True)))
        parity = read(BASE/f'windows-{name}/evidence.json')
        assert len(parity['results']) == 2
        for index, row in enumerate(parity['results']):
            if name.endswith('seven'):assert row['metrics']['pass']
            else:assert row['metrics'] is None, 'Reassess this checkpoint after Metal parity completes'
            for key, f in [('d3d11_sha256', BASE/f'windows-{name}'/row['frame']),
                           ('packet_sha256', ROOT/report['packets'][index]['path']),
                           ('shader_sha256', result/'shaders/source.hlsl'),
                           ('reflection_sha256', result/'shaders/reflection/source.hlsl'),
                           ('post_sha256', result/'postprocess/source.hlsl')]:
                assert row[key] == sha(f)
        windows.append(parity)
        cases[name] = {'revision': revision, 'augmentation_sha256': sha(path),
                       'source_biq_sha256': a['source_biq_sha256'], 'pool': a['pool'],
                       'body_count': expected, 'scale': a['instances'][0]['scale'],
                       'lights': len(lights['lights']), 'unbounded_light_proxies': len(complete['lights']),
                       'all_emitting_buildings_represented': True, 'blockers': len(lights['blockers']),
                       'visual_selection': 'Capacity witness only; scattered outer houses need another layout approach' if name == 'asian-large' else 'Provisional neighborhood/readability candidate' if name in SELECTED else 'Matched seven-house baseline',
                       'clearance': clearance(a), 'packet_frame_checks': packet_checks,
                       'metal_parity': 'pass' if name.endswith('seven') else 'pending; prolonged Metal compilation stopped',
                       'window_emission_and_geometry_packets_unchanged_by_spill': True}
    for name, old in [('asian-medium', 'asian-seven'), ('ancient-medium', 'ancient-seven')]:
        a, b = augmentations[name], augmentations[old]
        for key in ('projection', 'source_biq_sha256', 'anchor_tile', 'uniform_scale_factor'):
            assert a[key] == b[key]
        assert a['instances'][0]['scale'] == b['instances'][0]['scale']
        cases[name]['matched_density_pixels'] = [difference(BASE/f'windows-{old}'/f'h{h:02}-z1-pan00.bmp',
              BASE/f'windows-{name}'/f'h{h:02}-z1-pan00.bmp', (630, 280, 1030, 565)) for h in (12, 0)]
    medium, large = augmentations['asian-medium'], augmentations['asian-large']
    assert [placement(i) for i in medium['instances']] == [placement(i) for i in large['instances'][:16]]
    result = {'classification': 'Provisional individual-house density and night-readability improvement; general city quality remains open',
              'cases': cases, 'windows': windows, 'asian_medium_large_exact_prefix': True,
              'verification_scope': 'Twelve completed Windows frames; four baseline Metal comparisons pass, eight denser-frame comparisons remain pending. No full-backend acceptance.',
              'failed_holdout_large_plan': read(next((FIX/'city-scene-r63').glob('*/growth-search.json'))),
              'holdout_limit': 'Same palette, scale, density, envelopes and lighting; plan only current medium stage after future large search exhausted budget',
              'previous_goal_turn': 'progress: completed growth hierarchy, visual evidence and storage cleanup',
              'remaining': ['Other culture/era/size and palace combinations', 'Facade and inter-building detail',
                            'Future large holdout placement', 'New shoreline reflection coverage; prior capital control retained',
                            'Replace large shader-embedded light arrays with generic runtime data to address Metal compilation cost',
                            'All native delivery and human/milestone gates']}
    target = V2/'audits/beauty/CITY_CULTURE_DENSITY_EVIDENCE.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    sheet = Image.new('RGB', (760, 1040), (25, 25, 25)); draw = ImageDraw.Draw(sheet)
    for row, (before, after, hour) in enumerate([('asian-seven', 'asian-medium', 12),
              ('asian-seven', 'asian-medium', 0), ('ancient-seven', 'ancient-medium', 12),
              ('ancient-seven', 'ancient-medium', 0)]):
        for col, name in enumerate((before, after)):
            sheet.paste(Image.open(BASE/f'windows-{name}'/f'h{hour:02}-z1-pan00.bmp').convert('RGB').crop((650, 300, 1030, 540)), (col*380, row*260+20))
            draw.text((col*380+5, row*260+4), name+f' | {hour:02}:00', fill='white')
    sheet.save(BASE/'selected-native-comparison.png')
    print(target.relative_to(ROOT))


if __name__ == '__main__':
    main()
