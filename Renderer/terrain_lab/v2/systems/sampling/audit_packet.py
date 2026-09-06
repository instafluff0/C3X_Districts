"""Audit uploaded texture dimensions and matched local normalized DDS payloads."""
from pathlib import Path
import hashlib,json,struct,sys
ROOT=Path(__file__).resolve().parents[5]

def main():
    report=Path(sys.argv[1]);r=json.loads(report.read_text());packet=ROOT/r['outputs'][0]['packet']
    textures=[]
    with packet.open('rb') as f:
        def u32():return struct.unpack('<I',f.read(4))[0]
        magic,version,w,h,down=[u32() for _ in range(5)]
        if magic!=0x32514c43 or version not in (1,2,3):raise ValueError('unknown packet')
        if version==3:f.read(24)
        for index in range(u32()):
            tw,th,fmt,n=[u32() for _ in range(4)];sizes=[];digest=hashlib.sha256()
            for mip in range(n):
                pitch,size=u32(),u32();external=bool(size&0x80000000);size&=0x7fffffff
                if external:
                    ref=f.read(64).decode();data=(Path(str(packet)+'.blobs')/ref).read_bytes()
                    if hashlib.sha256(data).hexdigest()!=ref or len(data)!=size:raise ValueError('corrupt mip')
                else:data=f.read(size)
                digest.update(data);sizes.append({'level':mip,'width':max(1,tw>>mip),'height':max(1,th>>mip),'bytes':size,'pitch':pitch})
            textures.append({'index':index+1,'width':tw,'height':th,'format':fmt,'mips':sizes,'payload_sha256':digest.hexdigest(),'normalized_sources':[]})
    byhash={}
    for p in (ROOT/'Renderer/packs').rglob('*.dds'):
        b=p.read_bytes()
        if b[:4]!=b'DDS ' or b[84:88]!=b'DX10':continue
        key=hashlib.sha256(b[148:]).hexdigest()
        byhash.setdefault(key,[]).append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(b).hexdigest()})
    for t in textures:t['normalized_sources']=byhash.get(t['payload_sha256'],[])
    out={'schema':'c3x.q1.upload_audit.v1','packet_version':version,'packet_sha256':hashlib.sha256(packet.read_bytes()).hexdigest(),
        'internal_size':[w,h],'downsample':down,'textures':textures,
        'upload_bytes':sum(m['bytes'] for t in textures for m in t['mips']),
        'max_uploaded_dimension':max(max(t['width'],t['height']) for t in textures),
        'source_policy':'Matched normalized DDS payloads prove uploaded dimensions, not original largest cooked-source mip. Source extraction copies full CIVBIG payload; unavailable upstream conversion history remains unproven.'}
    (report.parent/'uploaded-textures.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='textures'},indent=2));print('textures',len(textures),'matched',sum(bool(t['normalized_sources']) for t in textures))

if __name__=='__main__':main()
