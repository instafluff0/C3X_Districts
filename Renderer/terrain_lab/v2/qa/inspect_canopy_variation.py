"""Verify fixed scene inputs and source transforms; emit native-size pixel pairs."""
import json
from pathlib import Path
import sys

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parents[2]
sys.path.insert(0, str(V2 / 'qa'))
from verify_gameplay_terrain import load, sha, difference


def main():
    from PIL import Image, ImageDraw
    base = V2 / 'audits/beauty/out'
    dest = base / 'canopy-variation-r1/review'
    dest.mkdir(parents=True, exist_ok=True)
    evidence = {'schema': 'c3x.canopy_variation_evidence.v1', 'approval': None, 'visual_accepted': False, 'regions': []}
    for region in ('coastal', 'inland', 'wilderness', 'freshcanopy'):
        old = base / ('canopy-variation-baseline' if region == 'freshcanopy' else 'shadow-receiver-r1') / region
        new = base / 'canopy-variation-r1' / region
        before, after = load(old / 'report.json'), load(new / 'report.json')
        for key in ('real_map', 'terrain', 'scenarios', 'viewport', 'tile_count', 'packs', 'settings'):
            assert before['effective']['fixture'][key] == after['effective']['fixture'][key], (region, key)
        assert before['effective']['pack_hash'] == after['effective']['pack_hash']
        assert before['effective']['shader_hashes'] == after['effective']['shader_hashes']
        bm, am = dict(before['effective']['module']), dict(after['effective']['module'])
        bm.pop('id'); am.pop('id'); assert am.pop('canopy_variation') == 1
        assert bm == am
        record = {'region': region, 'frames': []}
        for previous, frame in zip(before['outputs'], after['outputs']):
            for key in ('hour', 'zoom', 'offset'):
                assert previous[key] == frame[key]
            for f in (previous, frame):
                assert sha(ROOT / f['image']) == f['sha256']
                assert sha(ROOT / f['source_metadata']['path']) == f['source_metadata']['sha256']
            a, b = (load(ROOT / f['source_metadata']['path']) for f in (previous, frame))
            for key in ('meshes', 'textures', 'draw_texture_bindings'):
                assert a[key] == b[key], (region, key)
            assert len(a['instances']) == len(b['instances'])
            moved = 0
            for x, y in zip(a['instances'], b['instances']):
                x, y = dict(x), dict(y)
                moved += x.pop('source_xy_anchor', None) != y.pop('source_xy_anchor', None)
                x.pop('ground_authoring_height', None); y.pop('ground_authoring_height', None)
                assert x == y, (region, 'source scale/rotation/order changed')
            assert moved > 0
            rect = [360, 220, 1000, 540] if frame['zoom'] == 1 else [0, 0, 680, 400]
            w, h = rect[2] - rect[0], rect[3] - rect[1]
            sheet = Image.new('RGB', (w*2+20, h+30), (25, 28, 31)); draw = ImageDraw.Draw(sheet)
            for i, (f, label) in enumerate(((previous, 'previous best'), (frame, 'canopy variation'))):
                sheet.paste(Image.open(ROOT / f['image']).convert('RGB').crop(rect), (i*(w+20), 30))
                draw.text((i*(w+20)+10, 8), region+' | '+label, fill='white')
            output = dest / f"{region}-h{frame['hour']:02}-z{frame['zoom']}.png"
            sheet.save(output)
            record['frames'].append({'hour': frame['hour'], 'zoom': frame['zoom'],
                'before_sha256': previous['sha256'], 'after_sha256': frame['sha256'],
                'source_instances': len(b['instances']), 'moved_instances': moved,
                'source_meshes_textures_counts_scale_yaw_preserved': True,
                'crop': rect, 'resampled': False, 'comparison': output.relative_to(ROOT).as_posix(),
                'difference': difference(ROOT / previous['image'], ROOT / frame['image'])})
        evidence['regions'].append(record)
    output = V2 / 'audits/beauty/CANOPY_VARIATION_r1_EVIDENCE.json'
    output.write_text(json.dumps(evidence, indent=2)+'\n')
    print('PASS 16 matched frames; source meshes, textures, counts, sizes and rotations preserved')


if __name__ == '__main__':
    main()
