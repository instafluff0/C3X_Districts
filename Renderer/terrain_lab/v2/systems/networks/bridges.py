"""Read generic normalized rigid bridge bundles; preserve source UV/proportions."""
import struct,math
from pathlib import Path
from network import unit,sub,dot,mul
class Bundle:
    def __init__(self,path):
        data=Path(path).read_bytes();self.data=data;self.offset=8
        if data[:8]!=b'C3XVEG1\0':raise ValueError('invalid normalized bridge bundle')
        version,nt,na,ng=self.take('4I')
        if version!=1 or not 0<nt<=8 or not 0<na<=256 or not 0<ng<=64:raise ValueError('unsupported bridge bundle')
        self.textures=[self.string() for _ in range(nt)];self.assets=[];self.groups={}
        for _ in range(na):
            name=self.string();texture,nv,ni=self.take('3I')
            if texture>=nt or not 0<nv<=100000 or not 0<ni<=300000 or ni%3:raise ValueError('bridge bounds')
            vertices=[self.take('8f') for _ in range(nv)];indices=self.take(f'{ni}I')
            if max(indices)>=nv:raise ValueError('bridge index')
            self.assets.append(dict(id=name,texture=texture,vertices=vertices,indices=indices))
        for _ in range(ng):
            name=self.string();n,=self.take('I');self.groups[name]=[self.take('IffIIIIff') for _ in range(n)]
        if self.offset!=len(data):raise ValueError('trailing bridge data')
    def take(self,fmt):
        size=struct.calcsize('<'+fmt)
        if self.offset+size>len(self.data):raise ValueError('truncated bridge')
        result=struct.unpack_from('<'+fmt,self.data,self.offset);self.offset+=size;return result
    def string(self):
        n,=self.take('I')
        if n>4096 or self.offset+n>len(self.data):raise ValueError('invalid bridge string')
        s=self.data[self.offset:self.offset+n].decode();self.offset+=n;return s
    def add(self,mesh,crossing,stage,pillaged,rail):
        group='bridge_'+('railroad' if rail else ('modern' if stage>=3 else 'industrial' if stage>=2 else 'medieval'))+('_pillaged' if pillaged else '_normal')
        placements=self.groups[group]
        if len(placements)!=1:raise ValueError('unsupported multipart bridge')
        asset=self.assets[placements[0][0]];vs=asset['vertices'];lo=[min(v[k] for v in vs) for k in range(3)];hi=[max(v[k] for v in vs) for k in range(3)]
        # Source X is the bridge longitudinal axis (normalized rigid bundle contract).
        scale=(crossing.width+12)/(hi[0]-lo[0]);t=unit(crossing.tangent);n=(-t[1],t[0])
        center=[(a+b)*.5 for a,b in zip(lo,hi)]
        # Normalized source Z=0 deck datum maps to supplied deck plane; no nonuniform warping.
        for index in asset['indices']:
            v=vs[index];x=(v[0]-center[0])*scale;y=(v[1]-center[1])*scale;z=v[2]*scale+crossing.deck_z
            xy=(crossing.xy[0]+x*t[0]+y*n[0],crossing.xy[1]+x*t[1]+y*n[1])
            normal=(v[3]*t[0]+v[4]*n[0],v[3]*t[1]+v[4]*n[1],v[5])
            mesh.vertices.append((*xy,z,*normal,1,1,1,v[6],v[7],10+asset['texture'],0))
        return {'asset_id':asset['id'],'group':group,'uniform_scale':scale,'texture':self.textures[asset['texture']],'source_bounds':[lo,hi],'crossing_id':crossing.id,'transform':'uniform scale; rotate source X to route tangent; center on supplied crossing'}
if __name__=='__main__':
    import sys,json
    b=Bundle(sys.argv[1]);print(json.dumps({'textures':b.textures,'groups':{k:v[0][0] for k,v in b.groups.items()},'bounds':{a['id']:[[min(v[k] for v in a['vertices']) for k in range(3)],[max(v[k] for v in a['vertices']) for k in range(3)]] for a in b.assets}},indent=2))
