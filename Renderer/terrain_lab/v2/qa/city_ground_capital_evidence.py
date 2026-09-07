"""Recheck preserved r18 grounding and r19 palace comparisons; no acceptance gate."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
V2 = ROOT / 'Renderer/terrain_lab/v2'
OUT = V2 / 'audits/beauty/out'
FIX = V2 / 'fixtures/beauty'


def read(path):
    return json.loads(path.read_text())


def pixels(before, after, box):
    rows = []
    for hour in (12, 0):
        for zoom in (1, 2):
            filename = f'h{hour:02}-z{zoom}-pan00.png'
            old = np.asarray(Image.open(before / filename).convert('RGB')).astype(int)
            path = after / filename
            new = np.asarray(Image.open(path).convert('RGB')).astype(int)
            assert old.shape == new.shape
            delta = np.abs(old - new).max(axis=2)
            outside = delta.copy()
            x0, y0, x1, y1 = [n // zoom for n in box]
            outside[y0:y1, x0:x1] = 0
            assert outside.max() <= 1, 'comparison changed unrelated scene pixels'
            ys, xs = np.where(delta > 2)
            assert len(xs), 'comparison has no measurable visible contribution'
            lake = delta[491//zoom:501//zoom, 858//zoom:885//zoom]
            rows.append({'hour': hour, 'zoom': zoom, 'changed_pixels_gt_2': len(xs),
                         'max_channel_delta': int(delta.max()),
                         'bounds': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                         'outside_roi_max': int(outside.max()),
                         'lake_max_channel_delta': int(lake.max()),
                         'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows


def main():
    grounded = []
    for name, previous, roi in (
        ('american-modern-s1-at7-5', 16, (800, 400, 960, 475)),
        ('european-medieval-s1', 8, (370, 370, 510, 460)),
    ):
        before = read(FIX / f'city-scene-r{previous}' / name / 'augmentation.json')
        after = read(FIX / 'city-scene-r18' / name / 'augmentation.json')
        for key in ('source_biq_sha256', 'anchor_tile', 'projection', 'size', 'pool'):
            assert before[key] == after[key], key
        for a, b in zip(before['instances'], after['instances']):
            for key in ('asset', 'slot', 'scale', 'rotation', 'offset', 'local_bounds', 'ground_height_range'):
                assert a[key] == b[key], key
        assert len(before['instances']) == len(after['instances']) == 7
        assert len(after['compound_ground']['draws']) == 7
        grounded.append({'name': name, 'matched_buildings': 7,
                         'ground': after['compound_ground'],
                         'clipping': [i.get('ground_clipping', []) for i in after['instances']],
                         'pixels': pixels(OUT / f'city-scene-r{previous}' / name / 'combined',
                                          OUT / 'city-scene-r18' / name / 'combined', roi)})
    name = 'american-modern-s1-capital-at7-5'
    control = name.replace('-capital', '-capital-control')
    capital = read(FIX / 'city-scene-r19' / name / 'augmentation.json')
    ordinary = read(FIX / 'city-scene-r19' / control / 'augmentation.json')
    assert [i for i in capital['instances'] if i['slot'] != 'capital'] == ordinary['instances']
    assert capital['capital']['reserved_site'] == ordinary['capital']['reserved_site']
    assert len(ordinary['instances']) == 7 and not ordinary['capital']['drawn']
    parity = [read(OUT / directory / 'evidence.json') for directory in
              ('city-scene-r18/windows-medieval-ground', 'city-scene-r19/windows-capital')]
    assert all(len(p['results']) == 4 and all(r['metrics']['pass'] for r in p['results']) for p in parity)
    evidence = {'classification': 'Partial grounding improvement and palace composition diagnostic; no new city best or promotion',
                'era_policy': 'single current era', 'grounding': grounded,
                'capital': {'mapping': capital['capital']['mapping'], 'matched_houses': 7,
                            'placement_attempts': capital['capital']['placement_attempts'],
                            'pixels': pixels(OUT / 'city-scene-r19' / control / 'combined',
                                             OUT / 'city-scene-r19' / name / 'combined', (840, 390, 920, 515)),
                            'finding': 'Dome and windows contribute; foreground towers obscure facade. No measurable palace contribution in the fixed interior lake ROI.'},
                'standalone_windows_parity': parity, 'free_disk_gib': round(shutil.disk_usage(V2).free / 1024**3, 2),
                'remaining': ['Palace prominence and coherent city layout', 'Ground material state/height response',
                              'Full sizes/cultures and fresh terrain regions', 'Complete material and night lighting scope',
                              'All existing visual/integration gates']}
    target = V2 / 'audits/beauty/CITY_GROUND_CAPITAL_r18_r19_EVIDENCE.json'
    target.write_text(json.dumps(evidence, indent=2) + '\n')
    print(target.relative_to(ROOT))


if __name__ == '__main__':
    main()
