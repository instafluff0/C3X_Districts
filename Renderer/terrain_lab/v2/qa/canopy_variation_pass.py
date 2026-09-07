"""Compose stable canopy slot variation beside the retained river/relief scene."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
V2 = ROOT / 'Renderer/terrain_lab/v2'
sys.path.insert(0, str(V2 / 'qa'))
from coastal_pass import save
from shadow_receiver_pass import REGIONS
sys.path.insert(0, str(V2 / 'app'))
import real_map
REGIONS = REGIONS + ['freshcanopy']


def foundation():
    out = V2 / 'fixtures/beauty/river-vegetation-foundation/freshcanopy'
    if (out / 'BENCHMARKS.json').exists():
        return out
    reg, _ = real_map.load_registry()
    request = {'source_sha256': reg['source']['sha256'], 'regions': [{
        'requested_id': 'beauty-freshcanopy-100-v1', 'origin': [76, 58],
        'extent': [10, 10], 'halo': 6, 'role': 'user_evaluation',
        'camera': {'viewport': [1360, 800], 'zooms': [1, 2], 'hours': [12, 0]}}]}
    save(out / 'region-request.json', request)
    if not any(r['id'] == 'beauty-freshcanopy-100-v1' for r in reg['regions']):
        real_map.register(out / 'region-request.json')
    real_map.export('beauty-freshcanopy-100-v1', out, 'Q8-beauty', False)
    exported = json.loads((out / 'fixture.json').read_text())
    source = V2 / 'fixtures/beauty/shadow-receiver-r1/coastal'
    f = json.loads((source / 'fixture.json').read_text())
    for key in ('real_map', 'terrain', 'tile_count', 'viewport', 'id'):
        f[key] = exported[key]
    for key, old in f['scenarios'].items():
        header = (ROOT / old).read_text().strip().split(',')
        header[4] = f['real_map']['region']['terrain_sha256']
        path = out / (key + '.csv')
        save(path, ','.join(header) + '\n')
        f['scenarios'][key] = path.relative_to(ROOT).as_posix()
    m = json.loads((source / 'terrain.module.json').read_text())
    shader = (ROOT / m['shader']).read_text().replace('ORIGIN_X 56.5', 'ORIGIN_X 66.5').replace('ORIGIN_Y 18.5', 'ORIGIN_Y 8.5')
    save(out / 'combined.hlsl', shader)
    m['shader'] = (out / 'combined.hlsl').relative_to(ROOT).as_posix()
    f['modules'] = [(out / 'terrain.module.json').relative_to(ROOT).as_posix()]
    save(out / 'terrain.module.json', m)
    (out / 'fixture.json').write_text(json.dumps(f, indent=2) + '\n')
    real_map.validate_provenance(f)
    save(out / 'BENCHMARKS.json', {
        'region': f['real_map']['region'], 'projection': m['projection'],
        'gameplay_crop': [360, 220, 1000, 540],
        'selection': 'Before viewing: maximum squared wrapped origin distance from existing beauty regions among even-coordinate 100-tile crops with >=8 forest, >=8 jungle and >=15 river tiles. Nine forest, 24 jungle, 19 river tiles, four hills, five mountains. No regional tuning.'})
    return out


def prepare(region, baseline=False):
    source = foundation() if region == 'freshcanopy' else V2 / 'fixtures/beauty/shadow-receiver-r1' / region
    f = json.loads((source / 'fixture.json').read_text())
    m = json.loads((ROOT / f['modules'][0]).read_text())
    name = 'canopy-variation-baseline' if baseline else 'canopy-variation-r1'
    out = V2 / 'fixtures/beauty' / name / region
    f['id'] = m['id'] = name + '-' + region
    if not baseline:
        m['canopy_variation'] = 1
    f['modules'] = [(out / 'terrain.module.json').relative_to(ROOT).as_posix()]
    save(out / 'terrain.module.json', m)
    save(out / 'fixture.json', f)
    return out / 'fixture.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region', choices=REGIONS + ['all'], required=True)
    p.add_argument('--hours', nargs='+', type=int, choices=[0, 6, 12, 18], default=[12, 0])
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--baseline', action='store_true')
    a = p.parse_args()
    for region in REGIONS if a.region == 'all' else [a.region]:
        f = prepare(region, a.baseline)
        if a.prepare_only:
            continue
        name = 'canopy-variation-baseline' if a.baseline else 'canopy-variation-r1'
        out = V2 / 'audits/beauty/out' / name / region
        if (out / 'report.json').exists():
            raise ValueError('Preserved render already exists')
        subprocess.run([sys.executable, str(V2 / 'app/runner.py'), 'compose',
                        '--fixture', str(f), '--candidate', name,
                        '--output', str(out), '--hours', *map(str, a.hours)],
                       check=True, cwd=ROOT)
        for bmp in out.glob('h*-z*-pan00.bmp'):
            subprocess.run(['sips', '-s', 'format', 'png', str(bmp), '--out',
                            str(bmp.with_suffix('.png'))], check=True,
                           stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
