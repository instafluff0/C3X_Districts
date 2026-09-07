"""Recover selected city components' exact ground triangles and atlas UVs.

Descriptor selection is an explicit diagnostic, not recovered state semantics.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
V2 = ROOT / 'Renderer/terrain_lab/v2'
sys.path.insert(0, str(V2 / 'systems/terrain'))
from prepare_ground_decals import decode
sys.path.insert(0, str(ROOT / 'Renderer/tools/asset_compiler'))
from indexed_static_package import IndexedStaticPackage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--asset')
    selection.add_argument('--pool', help='Recover every selected component in one culture/era pool')
    parser.add_argument('--descriptor-index', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('preserve existing city ground evidence')
    report = json.loads((V2 / 'audits/beauty/out/city-source-expanded-r1/build.json').read_text())
    if args.pool:
        pool_id = args.pool if args.pool.startswith('city/pool/') else 'city/pool/' + args.pool
        selected = next(pool['selected'] for pool in report['pools'] if pool['pool'] == pool_id)
    else:
        asset_id = args.asset or 'city/component/81e6bc964c4f7c5a'
        selected = [next(x for pool in report['pools'] for x in pool['selected'] if x['asset_id'] == asset_id)]
    assets = Path.home() / "Library/Application Support/Steam/steamapps/common/Sid Meier's Civilization VI/Civ6.app/Contents/Assets"
    pack = ROOT / 'Renderer/packs/CityStudyAuxiliaryUV'
    manifest = json.loads((pack / 'manifest.json').read_text())
    packages = {}
    parts = {}
    for item in selected:
        if item['package'] not in packages:
            packages[item['package']] = IndexedStaticPackage(assets / item['package'], item['entry'])
        source = decode(packages[item['package']], item['entry'], args.descriptor_index)
        landmark = json.loads((pack / manifest['assets'][item['asset_id']]['landmark']).read_text())
        decal = json.loads((pack / landmark['components']['decal']).read_text())
        if decal['schema'] == 'c3x.decal_set.v0':
            decal = json.loads((pack / decal['decals'][args.descriptor_index]).read_text())
        vertices = []
        bounds = source['bounds']
        for x, y, u, v in source['vertices']:
            vertices.append({'position': [(bounds[0] + x * (bounds[2] - bounds[0])) / 100,
                                          (bounds[1] + y * (bounds[3] - bounds[1])) / 100, .0001],
                             'uv0': [u, v], 'normal': [0, 0, 1]})
        channel = dict(decal['channels']['base_color'])
        channel['texture'] = (pack / channel['texture']).relative_to(ROOT).as_posix()
        parts[item['asset_id']] = [{'mesh': {'vertices': vertices, 'topology': {'indices': list(range(len(vertices)))}},
                                  'material': {'alpha_mode': 'blend', 'channels': {'base_color': channel}},
                                  'source_descriptor_index': args.descriptor_index,
                                  'source_descriptor_sha256': source['source_descriptor_sha256'],
                                  'source_vertex_slice_sha256': source['vertex_slice_sha256']}]
    data = {'schema': 'c3x.lab.city_ground_parts.v1',
            'classification': 'explicit operational-material probe; source descriptor state selector remains unproven',
            'parts': parts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(str(args.output))


if __name__ == '__main__':
    main()
