"""Verify saved single-era growth composition at matched gameplay cameras."""
import json
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from city_growth_evidence import ROOT, V2, OUT, FIX, read, sha, placement, clearance
from city_scene_pass import executable, Cache, city

BASE = OUT / 'city-growth-hierarchy-r1'
CASES = {'inland-small': 53, 'inland-medium': 54, 'inland-large': 51,
         'wilderness-small': 56, 'wilderness-medium': 55, 'holdout-medium': 57}
PREVIOUS = {'wilderness-small': BASE / 'previous-small-fixed-frame/render',
            'wilderness-medium': OUT / 'city-settlement-ground-r2/wilderness/render',
            'holdout-medium': OUT / 'city-settlement-ground-r2/freshshadow/render',
            'inland-large': OUT / 'city-settlement-ground-r2/inland/render'}
ROIS = {'wilderness-small': (755, 240, 1000, 450),
        'wilderness-medium': (755, 240, 1000, 450),
        'holdout-medium': (525, 345, 770, 535),
        'inland-large': (650, 300, 980, 550)}


def difference(before, after, roi=None):
    a = np.asarray(Image.open(before).convert('RGB')).astype(int)
    b = np.asarray(Image.open(after).convert('RGB')).astype(int)
    assert a.shape == b.shape == (800, 1360, 3)
    d = abs(a-b).max(2)
    ys, xs = np.where(d > 2)
    result = {'changed_pixels_gt_2': len(xs), 'max_channel_delta': int(d.max()),
              'bounds': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
              'before_sha256': sha(before), 'after_sha256': sha(after)}
    if roi:
        x0, y0, x1, y1 = roi
        d[y0:y1, x0:x1] = 0
        result['outside_city_roi_max'] = int(d.max())
        result['outside_city_roi_changed_pixels'] = int((d > 0).sum())
        assert d.max() <= 1, result
    return result


def main():
    cache = Cache(V2 / 'app/.cache')
    frame = executable(V2 / 'qa/city_shadow_frame_contract.cpp', cache)
    ground = executable(V2 / 'qa/settlement_ground_contract.cpp', cache)
    cases, augmentations, parity = {}, {}, []
    for name, revision in CASES.items():
        path = next((FIX / f'city-scene-r{revision}').glob('*/augmentation.json'))
        a = read(path); augmentations[name] = a
        assert a['pool'] == 'city/pool/american/modern' and a['graduated_growth']
        assert a['generator_profile']['era_policy'] == 'single_current_era_user_preference'
        assert len(a['instances']) == (4, 7, 11)[a['size']]
        assert all(i['scale'] == 2.9177169657033994 for i in a['instances'])
        assert read(path.parent / 'surface.json')['region']['region']['extent'] == [10, 10]
        limits = a['layout_attempts'][0]['slot_extents']
        for i in a['instances']:
            bounds = [v+i['offset'][j % 2]+(-.012 if j < 2 else .012)
                      for j, v in enumerate(i['local_bounds'])]
            assert max(abs(v) for v in bounds) <= limits[i['slot']]+1e-8
        heights = [city.component(i['asset'], ROOT / a['pack'])['hi'][2]*i['scale']*
                   a['source_z_pixels_per_unit'] for i in a['instances']]
        raw = OUT / f'city-scene-r{revision}' / path.parent.name
        reference = read(ROOT / a['shadow_frame_report']['path'])
        data = read(BASE / name / 'ground/settlement.json')
        render = BASE / name / 'ground/render'
        checks = []
        for index, packet in enumerate(data['packets']):
            for key in ('original', 'output'):
                assert sha(ROOT / packet[key]) == packet[key+'_sha256']
            checks.append(json.loads(subprocess.check_output([str(ground), str(ROOT / packet['original']),
                          str(ROOT / packet['output']), str(packet['insertion_draw'])], text=True)))
            original = raw / f'combined-{index}.packet'
            ref = ROOT / reference['outputs'][index]['packet']
            checks.append(json.loads(subprocess.check_output([str(frame), str(original), str(ref), str(original)], text=True)))
        windows = read(BASE / f'windows-{name}/evidence.json')
        assert len(windows['results']) == 2
        report = read(render / 'report.json')
        for index, row in enumerate(windows['results']):
            assert row['metrics']['pass']
            for key, file in [('d3d11_sha256', BASE/f'windows-{name}'/row['frame']),
                              ('shader_sha256', render/'shaders/source.hlsl'),
                              ('reflection_sha256', render/'shaders/reflection/source.hlsl'),
                              ('post_sha256', render/'postprocess/source.hlsl'),
                              ('packet_sha256', ROOT/report['packets'][index]['path'])]:
                assert row[key] == sha(file)
        parity.append(windows)
        cases[name] = {'augmentation_sha256': sha(path), 'source_biq_sha256': a['source_biq_sha256'],
                       'projection': a['projection'], 'body_count': len(heights),
                       'height_pixels_above_source_ground': heights, 'clearance': clearance(a),
                       'packet_checks': checks, 'ground_tile_period': data['tile_period']}
        if name in PREVIOUS:
            cases[name]['matched_pixels'] = [difference(PREVIOUS[name]/f'h{h:02}-z1-pan00.png',
                       render/f'h{h:02}-z1-pan00.png', ROIS[name]) for h in (12, 0)]
    for group in [('inland-small', 'inland-medium', 'inland-large'), ('wilderness-small', 'wilderness-medium')]:
        last = augmentations[group[-1]]
        for name in group:
            a = augmentations[name]
            assert [placement(i) for i in a['instances']] == [placement(i) for i in last['instances'][:len(a['instances'])]]
            for key in ('projection', 'source_biq_sha256', 'anchor_tile'):
                assert a[key] == last[key]
            assert cases[name]['ground_tile_period'] == cases[group[-1]]['ground_tile_period']
        maxima = [max(cases[name]['height_pixels_above_source_ground']) for name in group]
        assert all(a < b for a, b in zip(maxima, maxima[1:]))
    controls = []
    old = read(OUT/'city-settlement-ground-r2/small/render/report.json')
    ref = read(OUT/'city-growth-r1/r46-fixed-shadow-frame/report.json')
    for index in range(2):
        controls.append(json.loads(subprocess.check_output([str(frame), str(ROOT/old['packets'][index]['path']),
              str(ROOT/ref['outputs'][index]['packet']), str(BASE/f'previous-small-fixed-frame/combined-{index}.packet')], text=True)))
    reflection = [difference(BASE/f'city-reflection-off/render/h{h:02}-z1-pan00.png',
                   BASE/f'wilderness-medium/ground/render/h{h:02}-z1-pan00.png') for h in (12, 0)]
    evidence = {'classification': 'Provisional modern growth improvement, not general city or milestone approval',
                'cases': cases, 'stable_growth_prefixes': True, 'windows': parity,
                'previous_small_frame_controls': controls, 'city_reflection_control': reflection,
                'reflection_interpretation': 'Only 3 day and 2 night pixels exceed 2/255; no substantial reflection claim for this view',
                'holdout': 'Same recipe on freshshadow without local growth tuning; region had earlier city evidence',
                'failed_large_wilderness': read(next((FIX/'city-scene-r52').glob('*/growth-search.json'))),
                'remaining': ['Facade material richness and environment response', 'Broader single-era culture, size and capital coverage',
                              'Large wilderness fit; budget exhaustion is not an infeasibility proof', 'Native integration and all approval gates']}
    target = V2/'audits/beauty/CITY_GROWTH_HIERARCHY_EVIDENCE.json'
    target.write_text(json.dumps(evidence, indent=2)+'\n')
    sheet = Image.new('RGB', (760, 920), (25, 25, 25)); draw = ImageDraw.Draw(sheet)
    crops = [(755, 230, 1135, 440), (755, 230, 1135, 440), (480, 330, 860, 540), (650, 310, 1030, 520)]
    for row, (name, crop) in enumerate(zip(PREVIOUS, crops)):
        for col, (folder, label) in enumerate([(PREVIOUS[name], 'Previous, matched frame'), (BASE/name/'ground/render', 'Growth hierarchy')]):
            sheet.paste(Image.open(folder/'h12-z1-pan00.png').convert('RGB').crop(crop), (col*380, row*230+20))
            draw.text((col*380+5, row*230+4), name+' | '+label, fill='white')
    sheet.save(BASE/'previous-native-comparison.png')
    print(target.relative_to(ROOT))


if __name__ == '__main__':
    main()
