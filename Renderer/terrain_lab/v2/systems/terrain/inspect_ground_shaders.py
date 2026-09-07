"""Inventory installed DXBC containers relevant to terrain material composition."""
import hashlib
import json
import os
from pathlib import Path
import re
import struct

ROOT=Path(__file__).resolve().parents[5];V2=ROOT/'Renderer/terrain_lab/v2'


def main():
    assets=Path(os.environ.get('C3X_CIV6_ASSETS',str(Path.home()/"Library/Application Support/Steam/steamapps/common/Sid Meier's Civilization VI/Civ6.app/Contents/Assets")))
    relative='Base/Binaries/Win64Steam/ShaderAutoGen_Windows_DX11_FinalRelease.bs'
    data=(assets/relative).read_bytes();out=V2/'audits/beauty/out/ground-shader-source';out.mkdir(parents=True,exist_ok=True)
    rows=[];total=0
    for match in re.finditer(b'DXBC',data):
        offset=match.start()
        if offset+32>len(data):continue
        length,count=struct.unpack_from('<II',data,offset+24)
        if count>64 or length<32+count*4 or offset+length>len(data):continue
        blob=data[offset:offset+length];chunks={}
        for i in range(count):
            start=struct.unpack_from('<I',blob,32+i*4)[0]
            if start+8>len(blob):raise ValueError('invalid DXBC chunk offset')
            name=blob[start:start+4].decode('ascii');n=struct.unpack_from('<I',blob,start+4)[0]
            if start+8+n>len(blob):raise ValueError('invalid DXBC chunk size')
            chunks[name]=blob[start+8:start+8+n]
        total+=1
        rdef=chunks.get('RDEF',b'')
        names=sorted({m.group().decode('ascii') for m in re.finditer(rb'[ -~]{5,}',rdef)})
        if not any("WorldView_Terrain" in n or "TerrainBlendTextureArray" in n or
                   "BaseColor" in n or "HeightScale" in n or "CacheBake" in n or
                   n in {"HeightToNormalDynamics", "tx_fuzz", "tx_basecolor"}
                   for n in names):continue
        program=chunks.get('SHEX',chunks.get('SHDR',b''))
        stage=struct.unpack_from('<I',program)[0]>>16 if program else None
        path=out/f'shader-{offset:08x}.dxbc';path.write_bytes(blob)
        rows.append({'offset':offset,'length':length,'sha256':hashlib.sha256(blob).hexdigest(),
            'stage':stage,'chunks':list(chunks),'reflection_strings':names,'local_dxbc':path.relative_to(ROOT).as_posix()})
    report={'classification':'compiled-source inventory; runtime variant selection and instruction meaning unproven',
        'package_relative':relative,'package_sha256':hashlib.sha256(data).hexdigest(),
        'valid_dxbc_containers':total,'selected_containers':rows}
    (V2/'audits/beauty/GROUND_SHADER_SOURCE_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'valid_containers':total,'selected':len(rows),
        'stages':{str(k):sum(r['stage']==k for r in rows) for k in sorted({r['stage'] for r in rows})},
        'reflection_names_max':max(len(r['reflection_strings']) for r in rows)}))


if __name__=='__main__':main()
