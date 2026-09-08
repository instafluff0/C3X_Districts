"""Normalize omitted city channels into a small generic, fingerprinted overlay."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from Renderer.tools.asset_compiler import compound_landmark_importer as source
from Renderer.tools.asset_compiler.clutter_blp_extractor import TYPE_TEXTURE
from Renderer.tools.asset_compiler.c3x_asset_compiler import parse_civbig_header,make_dds_dx10_header
from Renderer.tools.asset_compiler.opacity_coverage import bc4_blocks_to_bc3_alpha
from Renderer.lab.shared.cities.source_selection import selection


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    select=parser.add_mutually_exclusive_group(required=True)
    select.add_argument('--pool')
    select.add_argument('--palace',help='Generic root ID in the existing normalized palace library')
    parser.add_argument('--output',type=Path,required=True,help='New directory beneath Renderer; contains only extra textures and a mapping')
    parser.add_argument('--resume',action='store_true',help='Retry an incomplete import only; never replace a completed mapping')
    a=parser.parse_args();out=a.output.resolve();out.relative_to(ROOT/'Renderer')
    if out.exists() and (not a.resume or (out/'mapping.json').exists()):raise ValueError('preserve existing material overlay')
    selected,pack,units,report=selection(a.pool,a.palace)
    assets={r['asset_id']:r for r in report['assets']};packages={};materials={};evidence=[];textures={}
    manifest=json.loads((pack/'manifest.json').read_text())
    out.mkdir(parents=True,exist_ok=a.resume);(out/'.gitignore').write_text('*.json\ntextures/\n')
    for item in selected:
        key=item['package']
        if key not in packages:packages[key]=source.IndexedStaticPackage(source.MAC_ASSETS_ROOT/key,item['entry'])
        package=packages[key];package.select_direct_string(item['entry'])
        _,owner,model=source.landmark_base_model(package);ma=package.pointer_fields(model,source.TYPE_MATERIAL)[0][1]
        ta=package.unique_allocation(TYPE_TEXTURE)
        landmark=json.loads((pack/manifest['assets'][item['asset_id']]['landmark']).read_text())
        for record in assets[item['asset_id']]['materials']:
            index=record['material_index'];normalized=record['normalized_material_index']
            material=json.loads((pack/landmark['components']['materials'][normalized]).read_text())
            # Match the existing normalized source material before adding anything.
            for role,c in material['channels'].items():
                original=pack/c['texture']
                if hashlib.sha256(original.read_bytes()).hexdigest()!=record['texture_slots'][role]['dds_sha256']:
                    raise ValueError('existing material texture differs from source conversion evidence')
                c['texture']=original.relative_to(ROOT).as_posix()
            ud=struct.unpack_from('<Q',package.array_element(ma,index))[0]
            raw=package.bytes_for(package.pointer_fields(ud,source.TYPE_MATERIAL_DATA)[0][1]);channels={}
            for role,offset,kind in (('metalness',0x2c,'Generic_Metalness'),('opacity',0x30,'Generic_OPAC')):
                number=struct.unpack_from('<I',raw,offset)[0]
                if number==0xffffffff:continue
                entry=source.decode_texture_entry(package,ta,number)
                if entry['class']!=kind:raise ValueError('unexpected extra material texture class')
                pieces=Path(key).parts;blp_root=Path(*pieces[:pieces.index('BLPs')+1])
                src=source.MAC_ASSETS_ROOT/blp_root/'SHARED_DATA'/entry['name']
                data=src.read_bytes();h=hashlib.sha256(data).hexdigest();cache_key=(role,h)
                if cache_key not in textures:
                    info=parse_civbig_header(data);payload=data[48:48+info['payload_bytes']]
                    if role=='opacity':
                        if info['dxgi_format']!=80:raise ValueError('opacity coverage currently requires BC4 UNORM')
                        payload=bc4_blocks_to_bc3_alpha(payload)
                        info={**info,'dxgi_format':77,'payload_bytes':len(payload),'format_name':'BC3_UNORM'}
                    target=out/'textures'/f'{role}_{h[:16]}.dds';target.parent.mkdir(exist_ok=True)
                    target.write_bytes(make_dds_dx10_header(info)+payload)
                    textures[cache_key]={'texture':target.relative_to(ROOT).as_posix(),'format':info['format_name'],
                                         'color_space':'srgb' if info['format_name'].endswith('_SRGB') else 'linear',
                                         'component':'a' if role=='opacity' else 'r'}
                channels[role]={**textures[cache_key],'address_u':record['uv_address'],'address_v':record['uv_address']}
                evidence.append({'asset':item['asset_id'],'source_material':index,'normalized_material':normalized,
                                 'role':role,'source_class':kind,'source_sha256':h,'source_offset':offset,
                                 'texture':channels[role]['texture'],'dds_sha256':hashlib.sha256((ROOT/channels[role]['texture']).read_bytes()).hexdigest()})
            materials[item['asset_id']+':'+material['name']]={'material_digest':digest(material),'channels':channels}
    mapping={'schema':'c3x.lab.material_overlay.v1','materials':materials,
             'opacity_policy':'Coverage mask in alpha, UV0; single-sample threshold 0.5 is the explicit Lab adapter',
             'source_runtime_dependency':None}
    (out/'mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (out/'source-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'materials':len(materials),'extra_bindings':len(evidence),'unique_textures':len(textures),'mapping':(out/'mapping.json').relative_to(ROOT).as_posix()}))


if __name__=='__main__':main()
