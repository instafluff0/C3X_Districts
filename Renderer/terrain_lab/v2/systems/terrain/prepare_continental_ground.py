"""Recover installed continental ground channels without changing runtime packs."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[5]
V2=ROOT/'Renderer/terrain_lab/v2'
sys.path.insert(0,str(ROOT/'Renderer/tools/asset_compiler'))
import terrain_relief_builder as reader


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    steam=Path.home()/'Library/Application Support/Steam/steamapps'
    assets=Path(os.environ.get('C3X_CIV6_ASSETS',str(steam/"common/Sid Meier's Civilization VI/Civ6.app/Contents/Assets")))
    skin=Path(os.environ.get('C3X_CIV6_ENVIRONMENT_SKIN',str(steam/'workshop/content/289070/1702339134')))
    relative='Platforms/Windows/BLPs/terrain/TerrainElementSet_Base.blp'
    out=V2/'fixtures/beauty/source-continental-r1';out.mkdir(parents=True,exist_ok=True)
    (out/'.gitignore').write_text('*.dds\nground_fields.h\n')
    records=[]
    for family,root in [('base',assets/'Base'),('selected_overlay',skin)]:
        path=root/relative;_,elements,report=reader.inspect_terrain_element_package(path)
        with path.open('rb') as source:
            for name,e in elements.items():
                if 'CONTINENTAL_HILL' not in name:continue
                record={'source_tree':family,'package_relative':relative,'package_sha256':sha(path.read_bytes()),
                    'entry':name,'grid_dimensions':e['grid_dimensions'],'parameters':e['parameters'],'channels':[]}
                for role,lods in e['channels'].items():
                    for lod in lods:
                        source.seek(report['big_data_offset']+lod['relative_offset']);b=source.read(lod['bytes'])
                        if len(b)!=lod['bytes']:raise ValueError('short source channel')
                        filename=family+'-'+name.removeprefix('ART_DEF_TERRAIN_ELEMENT_').lower()+f'-{role}-lod{lod["level"]}.dds'
                        target=out/filename
                        dds=reader.make_r8_dds(lod['width'],lod['height'],b,62 if role=='region_ids' else 61)
                        target.write_bytes(dds)
                        record['channels'].append({'role':role,'lod':lod['level'],'dimensions':[lod['width'],lod['height']],
                            'source_resource':lod['name'],'source_index':lod['index'],'payload_sha256':sha(b),
                            'min':min(b),'max':max(b),'mean':sum(b)/len(b),'dds':target.relative_to(ROOT).as_posix(),'dds_sha256':sha(dds)})
                records.append(record)
    report={'schema':'c3x.continental_ground_source_audit.v1','classification':'recovered source channels; engine placement, blend and unit evaluation unproven',
        'records':records}
    header=['// Local derived source fields, area filtered for the gameplay mesh.','namespace q2_continental {']
    compiled=[]
    for family in ('grassland','plains'):
        source=out/f'selected_overlay-continental_hill_{family}-height-lod0.dds'
        pixels=source.read_bytes()[148:];filtered=[]
        for y in range(64):
            for x in range(64):
                total=sum(sum(pixels[(y*32+j)*2048+x*32:(y*32+j)*2048+x*32+32]) for j in range(32))
                filtered.append(round(total/1024))
        header.append('inline unsigned char const '+family+'[4096]={'+','.join(map(str,filtered))+'};')
        compiled.append({'family':family,'size':[64,64],'filter':'32x32 area average, rounded to R8',
            'source_dds_sha256':sha(source.read_bytes()),'filtered_payload_sha256':sha(bytes(filtered))})
    header.append('}')
    (out/'ground_fields.h').write_text('\n'.join(header)+'\n')
    report['compiled_fields']=compiled;report['compiled_header_sha256']=sha((out/'ground_fields.h').read_bytes())
    (out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{'source_tree':r['source_tree'],'entry':r['entry'],'parameters':r['parameters'],
        'channels':[{k:c[k] for k in ('role','lod','dimensions','min','max','mean')} for c in r['channels']]} for r in records]))


if __name__=='__main__':main()
