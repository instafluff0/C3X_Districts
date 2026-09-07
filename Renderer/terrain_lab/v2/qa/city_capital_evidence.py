"""Check the preserved r13 palace/control pair and produce gameplay-size crops.

Uses Pillow and NumPy. This is Lab evidence, not native promotion or approval.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
V2 = ROOT / 'Renderer/terrain_lab/v2'


def read(path):
    return json.loads(path.read_text())


def main():
    name = 'american-ancient-s1-capital'
    fixtures = V2 / 'fixtures/beauty/city-scene-r13'
    output = V2 / 'audits/beauty/out/city-scene-r13'
    capital = read(fixtures / name / 'augmentation.json')
    control = read(fixtures / (name + '-control') / 'augmentation.json')
    houses = [x for x in capital['instances'] if x['slot'] != 'capital']
    palaces = [x for x in capital['instances'] if x['slot'] == 'capital']
    assert houses == control['instances'], 'control moved surrounding buildings'
    assert len(palaces) == 1 and not control['capital']['drawn']
    for key in ('source_biq_sha256', 'anchor_tile', 'projection', 'size', 'pool'):
        assert capital[key] == control[key], key
    for instance in capital['instances']:
        assert instance['ground_height_range'][1] - instance['ground_height_range'][0] <= 3
    # Reimported palace bodies differ only by the newly retained UV channels.
    old = ROOT / 'Renderer/packs/FutureGateCandidates'
    new = ROOT / 'Renderer/packs/CityPalaceStudy'
    geometry = []
    for asset, entry in read(new / 'manifest.json')['assets'].items():
        landmark = read(new / entry['landmark'])
        for path in landmark['components']['geometry']:
            before, after = read(old / path), read(new / path)
            assert before['topology'] == after['topology']
            assert len(before['vertices']) == len(after['vertices'])
            for a, b in zip(before['vertices'], after['vertices']):
                assert all(a[k] == b[k] for k in ('position', 'normal', 'uv0'))
                assert 'uv1' in b and 'uv2' in b
            geometry.append({'asset': asset, 'mesh': path,
                             'vertices': len(after['vertices']),
                             'sha256': hashlib.sha256((new / path).read_bytes()).hexdigest()})
    review = output / 'review'
    review.mkdir(exist_ok=True)
    sheet = Image.new('RGB', (680, 580), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    rows = []
    for row, hour in enumerate((12, 0)):
        for zoom in (1, 2):
            filename = f'h{hour:02d}-z{zoom}-pan00.bmp'
            a = Image.open(output / (name + '-control') / 'combined' / filename).convert('RGB')
            b = Image.open(output / name / 'combined' / filename).convert('RGB')
            difference = np.abs(np.asarray(a).astype(int) - np.asarray(b).astype(int))
            box = tuple(x // zoom for x in (270, 270, 600, 530))
            delta = difference[box[1]:box[3], box[0]:box[2]]
            changed = int(np.count_nonzero(np.max(delta, axis=2) > 2))
            assert changed > 20, 'palace must visibly change gameplay pixels'
            rows.append({'hour': hour, 'zoom': zoom, 'city_changed_pixels_gt_2': changed,
                         'city_max_channel_delta': int(delta.max())})
            if zoom == 1:
                for col, (image, label) in enumerate(((a, 'Matched control'), (b, 'Capital palace'))):
                    sheet.paste(image.crop(box), (col * 340, row * 290 + 30))
                    draw.text((col * 340 + 8, row * 290 + 8),
                              f'{label} / {"Noon" if hour else "Midnight"}', fill='white')
    sheet.save(review / 'capital-native.png')
    parity = read(output / 'windows-capital/evidence.json')['results']
    assert len(parity) == 4 and all(x['metrics']['pass'] for x in parity)
    evidence = {'classification': 'Lab visual comparison; no native promotion',
                'terrain_tiles': 100, 'matching_houses': len(houses),
                'palace': palaces[0], 'source_geometry_preserved': geometry,
                'pixel_differences': rows, 'native_capital_indicator': 'retained',
                'standalone_windows_parity': parity,
                'remaining': ['Other culture mappings', 'Colonial attached components',
                              'Production authoritative capital-state binding',
                              'Full city visual and integration gates']}
    (V2 / 'audits/beauty/CITY_CAPITAL_r13_EVIDENCE.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'matching_houses': len(houses), 'pixel_differences': rows}))


if __name__ == '__main__':
    main()
