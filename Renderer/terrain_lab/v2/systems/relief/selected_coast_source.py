"""Audit and locally extract the selected skin's exact ordinary cliff assets."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[5]
V2=ROOT/'Renderer/terrain_lab/v2'
sys.path.insert(0,str(ROOT))
from Renderer.tools.asset_compiler.clutter_blp_extractor import StaticPackage,build_feature
from Renderer.tools.asset_compiler.shore_blp_extractor import SHORE_SPECS

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    source=Path(os.environ.get('C3X_CIV6_ENVIRONMENT_SKIN',
        str(Path.home()/'Library/Application Support/Steam/steamapps/workshop/content/289070/1702339134')))
    blps=source/'Platforms/Windows/BLPs'
    package_path=blps/'environment/clutter.blp'
    out=V2/'fixtures/beauty/selected-coast-source-v1'
    out.mkdir(parents=True,exist_ok=True)
    record_path=out/'provenance.json'
    package_hash=sha(package_path)
    if record_path.exists():
        record=json.loads(record_path.read_text())
        assert record['package_sha256']==package_hash
        assert all(sha(out/p)==h for p,h in record['output_sha256'].items())
        return out
    package=StaticPackage(package_path,'TER_Cliffs_Rock01')
    records=[]
    for spec in SHORE_SPECS[:6]:
        _,evidence=build_feature(package,blps/'SHARED_DATA',out,spec)
        name=spec['stem']
        old=ROOT/'Renderer/packs/ShoreNormalized'
        mesh=json.loads((out/f'meshes/features/{name}.json').read_text())
        original=json.loads((old/f'meshes/features/{name}.json').read_text())
        channels={}
        for role,t in evidence['textures'].items():
            p=out/f'textures/features/{name}_{role}.dds'
            old_role={'normal_0':'normal_0','normal_1':'normal_1'}.get(role,role)
            baseline=old/f'textures/features/{name}_{old_role}.dds'
            channels[role]={'entry':t['name'],'source_sha256':t['source_sha256'],
                'dds_sha256':sha(p),'base_pack_matches':baseline.exists() and sha(p)==sha(baseline)}
        records.append({'asset':spec['source_name'],'channels':channels,
            'source_vertices_match_base_pack':mesh['vertices']==original['vertices'],
            'source_indices_match_base_pack':mesh['topology']['indices']==original['topology']['indices']})
    outputs={p.relative_to(out).as_posix():sha(p) for folder in ['meshes','materials','textures'] for p in (out/folder).rglob('*') if p.is_file()}
    record={'classification':'source_reuse','source_locator':'C3X_CIV6_ENVIRONMENT_SKIN or installed selected skin',
        'package_sha256':package_hash,'assets':records,'output_sha256':outputs,
        'appearance_model':'Exact selected source payloads; C3X lighting remains an adaptation.'}
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'package_sha256':package_hash,'assets':records},indent=2))
    return out

if __name__=='__main__':prepare()
