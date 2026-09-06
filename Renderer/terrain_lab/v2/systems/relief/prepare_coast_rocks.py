"""Copy exact normalized source channels/mesh data into a local opt-in bundle."""
import hashlib
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[5]
V2=ROOT/'Renderer/terrain_lab/v2'

def prepare(selected=False):
    pack=ROOT/'Renderer/packs/ShoreNormalized'
    if selected:
        from selected_coast_source import prepare as selected_source
        pack=selected_source()
    out=V2/'fixtures/beauty'/('source-coast-rocks-v2' if selected else 'source-coast-rocks-v1')
    out.mkdir(parents=True,exist_ok=True)
    textures=[];assets=[];sources={}
    def string(s):
        b=s.encode();return struct.pack('<I',len(b))+b
    def record(p):
        sources[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    for kind,count in [('large',4),('small',2)]:
        for i in range(1,count+1):
            name=f'cliff_{kind}_{i:02}'
            mp=pack/f'meshes/features/{name}.json';matp=pack/f'materials/features/{name}.json'
            record(mp);record(matp);mesh=json.loads(mp.read_text());mat=json.loads(matp.read_text())
            index=len(textures)
            for path in [mat['base_color']['texture'],mat['lean_normal']['texture_0'],
                         mat['lean_normal']['texture_1'],mat['gloss']['texture']]:
                p=pack/path;record(p);textures.append(p.relative_to(ROOT).as_posix())
            vertices=mesh['vertices'];indices=mesh['topology']['indices']
            b=bytearray(string(name)+struct.pack('<III',index,len(vertices),len(indices)))
            for v in vertices:b.extend(struct.pack('<8f',*(v['position']+v['normal']+v['uv0'])))
            b.extend(struct.pack(f'<{len(indices)}I',*indices));assets.append(b)
    payload=b'C3XVEG1\0'+struct.pack('<IIII',1,len(textures),len(assets),1)
    payload+=b''.join(string(t) for t in textures)+b''.join(assets)
    payload+=string('coastal_rock')+struct.pack('<I',len(assets))
    for i in range(len(assets)):payload+=struct.pack('<IffIIIIff',i,1.,0.,1,0,1,1,0.,0.)
    target=out/'coastal_rocks.bin'
    if target.exists() and target.read_bytes()!=payload:raise ValueError('Source bundle changed; version required')
    target.write_bytes(payload)
    record={'classification':'source_reuse','uvs_and_source_proportions_preserved':True,
            'sources':sources,'bundle_sha256':hashlib.sha256(payload).hexdigest(),
            'channels':['base_color','lean_normal_0','lean_normal_1','gloss'],
            'appearance_model':'C3X source material adaptation; not a recovered engine shader'}
    (out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    return {'path':target.relative_to(ROOT).as_posix(),'sha256':record['bundle_sha256']}

if __name__=='__main__':print(json.dumps(prepare(),indent=2))
