"""Verify installed era grounding materials against existing normalized atlases.

Reads material records only. Does not generate connecting roads or copy assets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from Renderer.tools.asset_compiler import route_style_importer as source


def sha(data):return hashlib.sha256(data).hexdigest()


def records():
    report=ROOT/'Renderer/terrain_lab/v2/fixtures/beauty/city-generator-source-r2/source-report.json'
    generators=json.loads(report.read_text())
    package_path='Base/Platforms/Windows/BLPs/route_decal_materials.blp'
    package=source.IndexedStaticPackage(source.MAC_ASSETS_ROOT/package_path,'Decal_Parts_Modern_01')
    textures=Path('Renderer/packs/CityStudyAuxiliaryUV/textures/compound')
    old=textures/'base_color_b333b8babb7742d9.dds'
    for era,tag,name,suffix in [('modern','ARTERA_MODERN','Decal_Parts_Modern_01','185f6344efc21ad8'),
                              ('medieval','ARTERA_CLASSICAL','Decal_Parts_Classical_01','1674b3260ef12664')]:
        material,evidence=source.decode_route_material_entry(package,name)
        payload,metadata=source._decode_embedded_texture(package,package.unique_allocation(source.TYPE_TEXTURE),material['base_color']['index'])
        replacement=textures/f'base_color_{suffix}.dds'
        if (ROOT/replacement).read_bytes()[148:]!=payload:raise ValueError('normalized atlas does not match installed material payload')
        matches=[]
        for document in generators['sources']:
            path=source.MAC_ASSETS_ROOT/document['path']
            if sha(path.read_bytes())!=document['sha256']:raise ValueError('regenerate changed generator source report')
            for record in document['GroundingMaterials']:
                if record['fields']['Tag_Era']['m_ElementName']==tag:
                    if record['fields']['Normal Material']['m_EntryName']!=name:raise ValueError('conflicting source era grounding selector')
                    matches.append({'artdef':document['path'],'sha256':document['sha256'],'record':record})
        if not matches:raise ValueError('era grounding selector missing')
        yield era,{'schema':'c3x.lab.ground_binding_override.v1',
            'classification':'Explicit Lab era-ground binding hypothesis; no recovered engine height/state selector',
            'expected':{'texture':old.as_posix(),'sha256':sha((ROOT/old).read_bytes())},
            'replacement':{'texture':replacement.as_posix(),'sha256':sha((ROOT/replacement).read_bytes())},
            'source':{'package':package_path,'sha256':sha(package.data),'material':name,'material_evidence':evidence,
                      'texture':material['base_color'],'payload_sha256':sha(payload),'metadata':metadata,
                      'matches_existing_normalized_payload_exactly':True,'generator_bindings':matches},
            'unchanged':['source ground triangles','source atlas UVs','city body materials','city era','city placement',
                         'terrain','shadow map','local lights','reflection and glow shaders'],
            'unresolved':['HeightRange application','normal/pillaged state selection','ground height channel','broader coverage between compounds']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify',action='store_true')
    a=p.parse_args();output=ROOT/a.output
    for era,value in records():
        target=output/f'{era}.json'
        if a.verify:
            if json.loads(target.read_text())!=value:raise ValueError('saved ground binding evidence changed')
        else:
            if target.exists():raise ValueError('preserve existing grounding evidence')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(value,indent=2)+'\n')
    print('PASS era grounding source bindings and normalized payload identity')


if __name__=='__main__':main()
