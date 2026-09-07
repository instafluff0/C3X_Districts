"""Compose an opt-in shared river corridor on the frozen canopy benchmarks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
V2 = ROOT / 'Renderer/terrain_lab/v2'
sys.path.insert(0, str(V2 / 'qa'))
from canopy_variation_pass import REGIONS, prepare as prepare_canopy
from coastal_pass import save


def prepare(region, revision='r1'):
    f = json.loads(prepare_canopy(region).read_text())
    m = json.loads((ROOT / f['modules'][0]).read_text())
    name = 'river-corridor-' + revision
    out = V2 / 'fixtures/beauty' / name / region
    f['id'] = m['id'] = name + '-' + region
    m['river_corridor'] = 1
    if revision in ('r3','r4'):m['river_bank_rocks']=1
    if revision == 'r4':
        profile=json.loads((V2/'fixtures/beauty/source-river-pools-r1/provenance.json').read_text())
        m['river_pool_profiles']={'path':profile['output'],'sha256':profile['output_sha256']}
    if revision in ('r2','r3','r4'):
        shader = '#define Q3_CONTINUOUS_RIVERS 1\n' + (ROOT / m['shader']).read_text()
        save(out / 'combined.hlsl', shader)
        m['shader'] = (out / 'combined.hlsl').relative_to(ROOT).as_posix()
    f['modules'] = [(out / 'terrain.module.json').relative_to(ROOT).as_posix()]
    save(out / 'terrain.module.json', m)
    save(out / 'fixture.json', f)
    return out / 'fixture.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region', choices=REGIONS + ['all'], required=True)
    p.add_argument('--hours', nargs='+', type=int, choices=[0, 6, 12, 18], default=[12, 0])
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--revision', choices=['r1', 'r2','r3','r4'], default='r2')
    a = p.parse_args()
    for region in REGIONS if a.region == 'all' else [a.region]:
        f = prepare(region, a.revision)
        if a.prepare_only:
            continue
        name = 'river-corridor-' + a.revision
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
