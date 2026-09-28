"""Recover auxiliary texture coordinates for selected normalized city derivatives.

Source decoding is offline. Only coordinates matched to the derivative's exact
geometry are exported; layouts, positions, normals and materials stay intact.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Renderer.lab.shared.cities.assets import component
from Renderer.lab.shared.cities.fingerprint import geometry_digest
from Renderer.lab.studies.cities.trim_subsurface import trim_mesh
from Renderer.tools.asset_compiler.city_asset_importer import DEFAULT_ASSETS_ROOT, _shared_roots
from Renderer.tools.asset_compiler.city_adjunct_asset_importer import load_mapping
from Renderer.tools.asset_compiler.compound_landmark_importer import _compile_asset
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage
from Renderer.tools.asset_compiler.wall_mesh_filter import remove_uv_island_components, trim_skirt_and_ground


def transfer(source, derivative):
    def key(v):
        return tuple(round(x,7) for name in ('position','uv0') for x in v[name])
    # Clipping at grade may create interpolated vertices and shift the vertical
    # origin. Infer that already-applied offset from unchanged XY/UV witnesses.
    heights=defaultdict(list)
    for v in source['vertices']:
        heights[tuple(v['position'][:2]+v['uv0'])].append(v['position'][2])
    grades=Counter()
    for v in derivative['vertices']:
        for z in heights[tuple(v['position'][:2]+v['uv0'])]:
            grades[round(z-v['position'][2],8)]+=1
    for grade in dict.fromkeys([None,0.,*(g for g,_ in grades.most_common(3))]):
        candidate=source if grade is None else trim_mesh(source,grade)[0]
        if candidate is None:continue
        if (candidate['topology']==derivative['topology'] and
            len(candidate['vertices'])==len(derivative['vertices']) and
            all(key(a)==key(b) for a,b in zip(candidate['vertices'],derivative['vertices']))):
            return dict(geometry_digest=geometry_digest(derivative),grade=grade,
                        uv1=[v['uv1'] for v in candidate['vertices']],
                        uv2=[v['uv2'] for v in candidate['vertices']])
        by_vertex=defaultdict(set)
        for v in candidate['vertices']:
            by_vertex[key(v)].add(tuple(v['uv1']+v['uv2']))
        matches=[by_vertex[key(v)] for v in derivative['vertices']]
        if all(len(found)==1 for found in matches):
            values=[next(iter(found)) for found in matches]
            return dict(geometry_digest=geometry_digest(derivative),
                        uv1=[v[:2] for v in values],uv2=[v[2:] for v in values],grade=grade)
    raise ValueError('Auxiliary UVs do not match every derivative vertex: '+derivative['asset_id'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets-root',type=Path,default=DEFAULT_ASSETS_ROOT)
    ap.add_argument('--city-pack',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'Renderer/packs/CityRecipeAuxiliaryUV/uv.json')
    args=ap.parse_args()
    manifest=json.loads((args.city_pack/'manifest.json').read_text())
    needed={g['asset'] for g in manifest['gaps']}
    sources={}
    for era in ('industrial','modern'):
        report=ROOT/f'Renderer/lab/out/cities/all-era-source-auditions/{era}/source-report.json'
        for pool in json.loads(report.read_text())['pools']:
            for item in pool['selected']:sources[item['asset_id']]=(item['package'],item['entry'],100.)
    mapping=load_mapping()
    for item in mapping['assets']:
        sources[item['asset_id']]=(item['source_package'],item['source_entry'],mapping['source_units_per_tile'])
    output={'schema':'c3x.normalized_auxiliary_uv.v1','meshes':{},'packages':{}}
    grouped=defaultdict(list)
    for asset in sorted(needed):
        source_asset=asset.replace('/modern_low/','/modern_barricade/')
        package,entry,scale=sources[source_asset]
        grouped[package].append((asset,entry,scale))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='uv-source-',dir=ROOT/'Renderer/native/build/cities-borders') as directory:
        pack=Path(directory)
        textures={}
        for package,selected in grouped.items():
            options=mapping.get('package_options',{}).get(package,{})
            decoder=IndexedStaticPackage(args.assets_root/package,selected[0][1],
                allow_declared_size_mismatch=options.get('allow_declared_size_mismatch',False))
            output['packages'][package]=hashlib.sha256(decoder.data).hexdigest()
            for asset,entry,scale in selected:
                record,_=_compile_asset(decoder,_shared_roots(args.assets_root,package),pack,
                    entry,asset,scale,textures,auxiliary_uvs=True)
                landmark=json.loads((pack/record['landmark']).read_text())
                imported={}
                for path in landmark['components']['geometry']:
                    mesh=json.loads((pack/path).read_text())
                    if '/modern_low/' in asset:
                        mesh,_=remove_uv_island_components(mesh,tuple(mapping['derived_kits']['modern_clean']['uv_rect']))
                        mesh,_=trim_skirt_and_ground(mesh,mapping['derived_kits']['modern_low']['floor_z'])
                    imported[mesh['asset_id']]=mesh
                model=next(m for m in manifest['models'] if m['asset']==asset)
                for mesh,material in component(asset,Path(model['pack']))['parts']:
                    if not any(channel in material['channels'] and not all(uv in v for v in mesh['vertices'])
                        for channel,uv in [('ambient_occlusion','uv1'),('emissive','uv2')]):continue
                    matches=[]
                    candidates=([imported[mesh['asset_id']]] if mesh['asset_id'] in imported else imported.values())
                    for source in candidates:
                        try: matches.append(transfer(source,mesh))
                        except ValueError: pass
                    if not matches or any(m!=matches[0] for m in matches):
                        (ROOT/'Renderer/native/build/cities-borders/uv-mismatch.json').write_text(json.dumps(dict(source=list(imported.values()),derivative=mesh)))
                        raise ValueError('Missing or ambiguous derivative recovery: '+mesh['asset_id'])
                    output['meshes'][geometry_digest(mesh)]=matches[0]
                print('RECOVERED',asset,flush=True)
    args.output.write_text(json.dumps(output,separators=(',',':'))+'\n')
    print('Meshes:',len(output['meshes']),flush=True)

if __name__=='__main__':main()
