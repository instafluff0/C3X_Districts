#!/usr/bin/env python3
"""Q4 source-backed geometry experiment. Requires numpy and Pillow for offline preparation.
The ground/shore evaluator is an explicit temporary proxy, not a Q2/Q3 replacement.
"""
from pathlib import Path
import argparse, hashlib, json, math, struct, subprocess
import numpy as np
from PIL import Image
from clearance import clear, source_footprint, q5_clear, city_envelopes
ROOT=Path(__file__).resolve().parents[5]
V2=ROOT/'Renderer/terrain_lab/v2'
OWN=V2/'fixtures/relief'
SKIN=ROOT/'Renderer/packs/Civ5EnvironmentSkin'

def rel(p): return Path(p).relative_to(ROOT).as_posix()
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def smooth(a,b,x):
    t=np.clip((x-a)/(b-a),0,1); return t*t*(3-2*t)
def sample(a,u,v):
    h,w=a.shape; x=np.clip(u,0,1)*(w-1); y=np.clip(v,0,1)*(h-1)
    i=x.astype(int); j=y.astype(int); fx=x-i; fy=y-j
    return (a[j,i]*(1-fx)+a[j,np.minimum(i+1,w-1)]*fx)*(1-fy)+(a[np.minimum(j+1,h-1),i]*(1-fx)+a[np.minimum(j+1,h-1),np.minimum(i+1,w-1)]*fx)*fy

def height_dds(p):
    b=Path(p).read_bytes(); h,w=struct.unpack_from('<II',b,12)
    # Normalized R8 DX10 data; preserve scalar source without reinterpretation.
    assert b[84:88]==b'DX10' and struct.unpack_from('<I',b,128)[0] in (61,62)
    return np.frombuffer(b,dtype=np.uint8,count=w*h,offset=148).reshape(h,w).astype(float)/255

class Fixture:
    def __init__(self,name,revision=1):
        self.name=name; self.revision=revision; self.n=6;self.viewport=(1024,768);self.projection=(512,98,66,33,93.338)
        self.sources={}; hp=OWN/"selected-source/hills_height_lod0.dds"; self.hillfield=height_dds(hp);self.sources[rel(hp)]=digest(hp);self.hill_amplitude=.65*(10/14); self.draws=[]; self.textures=[]; self.texmap={}; self.transforms=[]
        self.mass_sources=[];self.rejected=[];self.envelopes=[];self.q5_envelope=None;
        q5path=V2/"fixtures/networks/source-clearance/clearance.json"
        if name=="biomes-clearance" and q5path.is_file():
            self.q5_envelope=json.loads(q5path.read_text());self.sources[rel(q5path)]=digest(q5path)
        if name=="biomes-clearance" and self.q5_envelope is None:self.envelopes=json.loads((OWN/"clearance-proxy.json").read_text())["envelopes"]
        self.body_blends={}; self.casters=[]; self.coast=[]; self.masses=[]; self.real=None; self.hydro=None; self.tiles=[]
        if name.startswith("real-"):
            self.real=json.loads((OWN/name/"fixture.json").read_text())
            city_path=V2/'audits/objects/CITY_VEGETATION_WITNESS.json'
            if city_path.is_file():
                self.envelopes+=city_envelopes(json.loads(city_path.read_text()),self.real['real_map'])
                self.sources[rel(city_path)]=digest(city_path)
            self.tiles=[list(map(int,line.split(","))) for line in (ROOT/self.real["terrain"]).read_text().splitlines()[1:] if line and not line.startswith("#")]
            binary=OWN/name/"sample-hydrology"
            subprocess.run(["clang++","-std=c++17","-O2",str(V2/"systems/relief/sample_hydrology.cpp"),"-o",str(binary)],check=True)
            fieldfile=OWN/name/"hydrology.bin"
            subprocess.run([str(binary),str(ROOT/self.real["terrain"]),str(fieldfile)],check=True)
            self.hydro=np.fromfile(fieldfile,dtype="<f4").reshape(257,257,5)
            self.coast=np.loadtxt(str(fieldfile)+".coast",delimiter=",").reshape(-1,5) if Path(str(fieldfile)+".coast").stat().st_size else []
            ep=Path(str(fieldfile)+".exclusions")
            if ep.stat().st_size:
                for ax,ay,bx,by,radius in np.loadtxt(ep,delimiter=",").reshape(-1,5):
                    self.envelopes.append({"points":[[ax+1.5,ay+1.5],[bx+1.5,by+1.5]],"half_width":float(radius),"clearance":0,"provider":"Q3 Field.exclusions"})
            self.sources[rel(V2/"systems/hydrology/field.h")]=digest(V2/"systems/hydrology/field.h")
        if name in ('range','combined','volcano'):
            for i,(cx,cy) in enumerate([(2.1,1.6),(3.25,1.75),(4.15,2.35)] if name!='volcano' else [(3,3)]):
                volcano=name=='volcano'
                p=(ROOT/'Renderer/packs/TerrainElementsNormalized/textures/terrain_elements/terrain_feature_volcano/height_lod0.dds') if volcano else SKIN/f'textures/relief/mountains/standard/variant_0{i+1}/height_lod0.dds'
                field=height_dds(p); self.sources[rel(p)]=digest(p)
                self.masses.append((cx,cy,1.75,i*.45,field,.75471735 if not volcano else .75471735));self.mass_sources.append(p)
        if self.real:
            for i,tile in enumerate(self.tiles):
                tx,ty,rx,ry,base,kind,*_=tile
                if kind!=6 or not (-1<=tx<=4 and -1<=ty<=4):continue
                p=SKIN/f'textures/relief/mountains/standard/variant_0{i%5+1}/height_lod0.dds'
                field=height_dds(p);self.sources[rel(p)]=digest(p)
                self.masses.append((tx+1.5,ty+1.5,1.7,(rx*13+ry*7)%4*math.pi/2,field,.75471735));self.mass_sources.append(p)
        self.bounds=(-.5,-.5,7,7)
    def coords(self,x,y):
        # Canonical orientation rotates only the diagnostic source fixture.
        if self.name.startswith('coast-r'):
            for _ in range(int(self.name[-1])): x,y=6-y,x
        return x,y
    def fields(self,x,y):
        x,y=self.coords(x,y)
        if self.real:
            h=[sample(self.hydro[:,:,i],x/6,y/6) for i in range(5)]
            dist,rocky,bed,rd,rw=h
            hill=np.zeros_like(x)
            for tx,ty,rx,ry,base,kind,*_ in self.tiles:
                if kind==5: hill+=np.exp(-(((x-tx-1.5)/.78)**2+((y-ty-1.5)/.78)**2)*1.25)
            hill=np.clip(hill,0,1)*sample(self.hillfield,(x*.085+.11)%1,(y*.085+.19)%1)*self.hill_amplitude
            cliff=smooth(-.065,.11,dist)
            z=bed+hill*((1-rocky)*smooth(0,.4,dist)+rocky*cliff)
            z=np.where(rd<rw/2+.12,np.minimum(z,bed+hill*smooth(rw/2,rw/2+.12,rd)),z)
            rock=rocky*(1-smooth(.06,.26,dist))*smooth(-.09,-.02,dist)
            sand=(1-rocky)*(1-smooth(.1,.35,dist))
            return z,dist,np.zeros_like(rock),sand
        if self.name=='island':
            dist=1.78-np.sqrt((x-3)**2+(y-3)**2)+.07*np.sin(x*8+y*3)
        elif self.name=='cove':
            dist=4.2-x-.95*np.exp(-((y-3)/.8)**2)+.09*np.sin(y*6)
        else:
            dist=4.15-x+.32*np.sin(y*1.35)+.075*np.sin(y*5.8+.4)
        if self.name in ('range','volcano','dunes','biomes','biomes-clearance','source-rock'): dist=np.ones_like(x)*10
        hill=sample(self.hillfield,(x*.085+.11)%1,(y*.085+.19)%1)*self.hill_amplitude*smooth(-.6,1.2,y)*(1-smooth(4.5,5.5,y))
        hill*=smooth(.0,1.0,x)
        if self.name in ('dunes','source-rock'): hill=np.zeros_like(x)
        if self.name in ('range','volcano','biomes','biomes-clearance'): hill*=.4
        # Named hills abut water. Only lowlands receive a beach ribbon.
        rocky=smooth(.15,.55,y)*(1-smooth(4.85,5.35,y))
        if self.revision==0: rocky=np.zeros_like(x)
        shore=smooth(-.10,.30,dist)
        cliff=smooth(-.085,.11,dist)
        z=-.055+(hill+.072)*((1-rocky)*shore+rocky*cliff)
        z+=0 # No invented high-frequency hill displacement.
        z+=0 # Rock bodies come only from the normalized source meshes.
        rock=rocky*(1-smooth(.06,.28,dist))*smooth(-.09,-.015,dist)
        sand=(1-rocky)*(1-smooth(.12,.42,dist))
        if self.name=='dunes':
            mask=smooth(.3,1.1,x)*(1-smooth(4.8,5.7,x))*smooth(.3,1.0,y)*(1-smooth(4.8,5.7,y))
            # Direct source material height; macro scale remains a diagnostic
            # until the source dune-layer construction is recovered.
            hp=SKIN/'textures/desert_hills_height.dds'
            field=np.asarray(Image.open(hp)).astype(float)/255
            self.sources[rel(hp)]=digest(hp)
            z=.02+mask*sample(field,x/6,y/6)*.55
            sand=np.ones_like(x); rock=np.zeros_like(x)
        return z,dist,np.zeros_like(rock),sand
    def ground(self,x,y):
        z=self.fields(x,y)[0]
        # Ground shoulders are separate from rigid source bodies.
        for cx,cy,scale,yaw,field,aspect in self.masses:
            r=np.sqrt((x-cx)**2+(y-cy)**2)
            shoulder=(.017+.069*(1-smooth(.95,1.55,r)))*smooth(0,.4,self.fields(x,y)[1])
            z=np.where(self.fields(x,y)[1]>0,np.maximum(z,shoulder),z)
        return z
    def source_height(self,x,y,m):
        cx,cy,s,a,f,aspect=m; dx=(x-cx)/s;dy=(y-cy)/s
        u=dx*np.cos(a)+dy*np.sin(a)+.5;v=-dx*np.sin(a)+dy*np.cos(a)+.5
        h=sample(f,u,v)*aspect*s+.07
        return np.where((u>=0)&(u<=1)&(v>=0)&(v<=1),h,-1)
    def top(self,x,y):
        z=self.ground(x,y)
        for m in self.masses: z=np.maximum(z,self.source_height(x,y,m))
        return z
    def texture(self,p):
        p=Path(p); key=rel(p)
        if key in self.texmap: return self.texmap[key]
        b=p.read_bytes(); h,w=struct.unpack_from('<II',b,12); count=struct.unpack_from('<I',b,28)[0] or 1
        four=b[84:88]; offset=128
        if four==b'DX10': fmt=struct.unpack_from('<I',b,128)[0];offset=148
        else: fmt={b'DXT1':72,b'DXT5':78,b'ATI1':80,b'ATI2':83}[four]
        if fmt==71:fmt=72
        if fmt==77:fmt=78
        mips=[]; ww,hh=w,h
        for _ in range(count):
            size=8 if fmt in (71,72,80) else 16
            pitch=ww if fmt in (61,62) else ((ww+3)//4)*size; n=pitch*(hh if fmt in (61,62) else ((hh+3)//4))
            mips.append((pitch,b[offset:offset+n]));offset+=n;ww=max(1,ww//2);hh=max(1,hh//2)
        self.textures.append((w,h,fmt,mips)); idx=len(self.textures);self.texmap[key]=idx;self.sources[key]=digest(p);return idx
    def mesh(self,xyz,norm,uv,blend,texture):
        if len(xyz)==0: return
        xyz=np.asarray(xyz); norm=np.asarray(norm); uv=np.asarray(uv); blend=np.broadcast_to(blend,(len(xyz),4))
        # World Euclidean axes, exact Civ III 2:1 projected ground basis.
        ox,oy,px,py,pz=self.projection
        sx=ox+(xyz[:,0]-xyz[:,1])*px
        sy=oy+(xyz[:,0]+xyz[:,1])*py-xyz[:,2]*pz
        pos=np.stack([sx/(self.viewport[0]/2)-1,1-sy/(self.viewport[1]/2),.72-(xyz[:,0]+xyz[:,1]+xyz[:,2]*(2*py/pz))*(.168/self.n)],axis=-1)
        vertices=np.concatenate([pos,norm,uv,xyz,blend],axis=-1).astype('<f4')
        self.draws.append((vertices,self.texture(texture),0,0))
    def surface(self):
        n=241 if self.n==6 else 401; t=np.linspace(0,self.n,n);x,y=np.meshgrid(t,t);z=self.ground(x,y)
        _,dist,rock,sand=self.fields(x,y)
        dzdy,dzdx=np.gradient(z,t,t); normals=np.stack([-dzdx,-dzdy,np.ones_like(z)],axis=-1);normals/=np.linalg.norm(normals,axis=-1)[...,None]
        xyz=np.stack([x,y,z],axis=-1).reshape(-1,3);uv=np.stack([x*.48,y*.48],axis=-1).reshape(-1,2)
        weights=np.stack([sand,rock,np.zeros_like(z),np.zeros_like(z)],axis=-1).reshape(-1,4)
        a=np.arange(n*n).reshape(n,n)[:-1,:-1].ravel(); ix=np.stack([a,a+1,a+n+1,a,a+n+1,a+n],axis=-1).ravel()
        self.mesh(xyz[ix],normals.reshape(-1,3)[ix],uv[ix],weights[ix],SKIN/'textures/grassland_base_color.dds')
        # Explicit Q3 water proxy; flat, static and no waves or surf.
        sel=np.min(dist.reshape(-1)[ix].reshape(-1,3),axis=1)<0
        wi=ix.reshape(-1,3)[sel].ravel();wxyz=xyz.copy();wxyz[:,2]=-.012
        self.mesh(wxyz[wi],np.tile([0,0,1],(len(wi),1)),uv[wi],[0,0,1,0],SKIN/'textures/grassland_base_color.dds')
    def bodies(self):
        for mi,m in enumerate(self.masses):
            cx,cy,s,a,f,aspect=m; n=97;u,v=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n)); z=sample(f,u,v)*aspect
            x=u-.5;y=v-.5; xyz=np.stack([cx+s*(x*np.cos(a)-y*np.sin(a)),cy+s*(x*np.sin(a)+y*np.cos(a)),.07+s*z],axis=-1)
            dy,dx=np.gradient(z,1/(n-1),1/(n-1));norm=np.stack([-dx*np.cos(a)+dy*np.sin(a),-dx*np.sin(a)-dy*np.cos(a),np.ones_like(z)],axis=-1);norm/=np.linalg.norm(norm,axis=-1)[...,None]
            idx=np.arange(n*n).reshape(n,n)[:-1,:-1].ravel();ix=np.stack([idx,idx+1,idx+n+1,idx,idx+n+1,idx+n],axis=-1).ravel()
            # Source UVs remain [0,1]. No per-instance body deformation or diamond taper.
            blend=np.stack([smooth(.75,.8125,z/aspect),np.zeros_like(z),np.zeros_like(z),np.ones_like(z)*4],axis=-1).reshape(-1,4)
            self.mesh(xyz.reshape(-1,3)[ix],norm.reshape(-1,3)[ix],np.stack([u,v],axis=-1).reshape(-1,2)[ix],blend[ix],SKIN/'textures/mtn_base_base_color.dds')
            vertices,texture,_,_=self.draws[-1]
            self.draws[-1]=(vertices,texture,self.texture(SKIN/'textures/mtn_base_height.dds'),0)
            bp=self.mass_sources[mi].with_name('blend_lod0.dds')
            self.body_blends[len(self.draws)-1]=self.texture(bp)
            self.transforms.append({'kind':'source_height_grid','instance':mi,'uniform_scale':s,'yaw':a,'translation':[cx,cy,.07],'height_to_width':aspect,'uv_preserved':True,'normalized_conventional_mesh_available':False})
    def source_mesh(self,pack,name,x,y,s,a,z=None):
        p=pack/f'meshes/features/{name}.json';d=json.loads(p.read_text());self.sources[rel(p)]=digest(p)
        footprint=source_footprint(d['bounds'],x,y,s,a)
        if 'Vegetation' in pack.name and (not clear(footprint,self.envelopes,margin=0) or not q5_clear(footprint,self.q5_envelope)):
            self.rejected.append({'mesh':rel(p),'anchor':[x,y],'reason':'source crown bounds intersects corridor'});return
        mat=json.loads((pack/f'materials/features/{name}.json').read_text())
        pos=np.array([v['position'] for v in d['vertices']]);norm=np.array([v['normal'] for v in d['vertices']]);uv=np.array([v['uv0'] for v in d['vertices']]);ix=np.array(d['topology']['indices'])
        rotation=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        z=float(self.ground(np.array(x),np.array(y))) if z is None else z
        out=pos@rotation.T*s+[x,y,z];norm=norm@rotation.T
        has_normal='lean_normal' in mat
        mode=(1 if mat.get("alpha_mode")!="opaque" else 0)+(2 if has_normal else 0)
        self.mesh(out[ix],norm[ix],uv[ix],[0,0,0,mode],pack/mat['base_color']['texture'])
        if has_normal:
            v,t,_,_=self.draws[-1]
            self.draws[-1]=(v,t,self.texture(pack/mat['lean_normal']['texture_0']),self.texture(pack/mat['gloss']['texture']))
        if mat.get("alpha_mode")=="opaque": self.casters.append(out[ix].reshape(-1,3,3))
        self.transforms.append({'kind':'normalized_mesh','draw_index':len(self.draws)-1,'material':rel(pack/f'materials/features/{name}.json'),'material_sha256':digest(pack/f'materials/features/{name}.json'),'mesh':rel(p),'uniform_scale':s,'yaw':a,'translation':[x,y,z],'uv_preserved':True})
    def dressing(self):
        if self.name=='source-rock':
            self.source_mesh(ROOT/'Renderer/packs/ShoreNormalized','cliff_large_02',3,3,.52,.4,z=.018)
        # Only actual normalized source meshes create exposed rock geometry.
        if self.revision>=5 and self.name not in ('range','volcano','dunes','biomes','biomes-clearance','source-rock'):
            points=[]
            if self.real:
                for ax,ay,bx,by,rocky in self.coast:
                    if rocky>.15:points.append(((ax+bx)/2+1.5,(ay+by)/2+1.5))
            else:
                grid=np.linspace(0,6,73);xx,yy=np.meshgrid(grid,grid);dd=self.fields(xx,yy)[1]
                for iy in range(72):
                    for ix in range(72):
                        xy=[(grid[ix],grid[iy]),(grid[ix+1],grid[iy]),(grid[ix+1],grid[iy+1]),(grid[ix],grid[iy+1])]
                        ds=[dd[iy,ix],dd[iy,ix+1],dd[iy+1,ix+1],dd[iy+1,ix]]
                        for tri in [(0,1,2),(0,2,3)]:
                            cuts=[]
                            for a,b in zip(tri,tri[1:]+tri[:1]):
                                if (ds[a]>0)!=(ds[b]>0):
                                    t=ds[a]/(ds[a]-ds[b]);cuts.append(tuple(xy[a][k]+t*(xy[b][k]-xy[a][k]) for k in range(2)))
                            if len(cuts)==2:points.append(tuple((cuts[0][k]+cuts[1][k])/2 for k in range(2)))
            used=[]
            for j,(x,y) in enumerate(points):
                if any((x-a)**2+(y-b)**2<.085**2 for a,b in used):continue
                e=.01;d=self.fields(np.array(x),np.array(y))[1]
                dx=(self.fields(np.array(x+e),np.array(y))[1]-d)/e;dy=(self.fields(np.array(x),np.array(y+e))[1]-d)/e
                norm=max(.001,float(np.hypot(dx,dy)));dx/=norm;dy/=norm
                top=float(self.ground(np.array(x+dx*.28),np.array(y+dy*.28)))
                if top<.045:continue
                used.append((x,y)); variant=1+(j*7)%4
                meshname=f'cliff_large_0{variant}'
                pack=ROOT/'Renderer/packs/ShoreNormalized'
                bounds=json.loads((pack/f'meshes/features/{meshname}.json').read_text())['bounds']
                scale=(top+.035)/bounds['maximum'][2]*(.98+.16*math.sin(j*3.7))
                # Use more of each broad source body below the waterline. Uniform
                # scale plus deeper translation connects faces without reshaping.
                bury=.32*(top+.035);scale*=1.32
                yaw=math.atan2(float(dy),float(dx))+j*.71
                self.source_mesh(pack,meshname,x+float(dx)*.085,y+float(dy)*.085,scale,yaw,z=-.04-bury)
                if j%5==0:
                    self.source_mesh(pack,f'cliff_small_0{1+j%2}',x-float(dx)*.11,y-float(dy)*.11,scale*.46,yaw+.4,z=-.05)
        if self.real:
            rng=np.random.default_rng(410)
            for tx,ty,rx,ry,base,kind,*_ in self.tiles:
                if kind not in (7,8) or not (-1<=tx<=4 and -1<=ty<=4):continue
                rng=np.random.default_rng((rx*73856093 ^ ry*19349663 ^ 410)&0xffffffff)
                for j in range(12):
                    x=tx+1.5+rng.uniform(-.51,.51);y=ty+1.5+rng.uniform(-.51,.51)
                    name=('forest_leafy_v1_01','forest_leafy_v4_02') if kind==7 else ('jungle_palm_02','jungle_plant_01')
                    self.source_mesh(ROOT/'Renderer/packs/Civ5EnvironmentVegetation',name[j%2],x,y,rng.uniform(.75,.95) if kind==7 else rng.uniform(.40,.54),rng.uniform(0,6.28))
        if self.name in ('biomes','biomes-clearance','combined'):
            rng=np.random.default_rng(410)
            for j in range(95):
                x,y=rng.uniform(.5,5.4,2)
                if self.fields(np.array(x),np.array(y))[1]<.3:continue
                if self.name=='combined' and y<3:continue
                name=('forest_leafy_v1_01','forest_leafy_v4_02','jungle_palm_02','jungle_plant_01')[int(x>3)*2+j%2]
                s=rng.uniform(.75,.95) if x<=3 else rng.uniform(.40,.54)
                self.source_mesh(ROOT/'Renderer/packs/Civ5EnvironmentVegetation',name,x,y,s,rng.uniform(0,6.28))
    def write(self):
        self.surface();self.bodies();self.dressing()
        # Batch source instances by material instead of allocating one GPU buffer
        # per tree. The packet contract caps buffers at 256.
        groups={};lookup={}
        for i,(v,t,n,g) in enumerate(self.draws):
            key=(t,n,g,self.body_blends.get(i,0))
            if key not in groups:groups[key]=[]
            lookup[i]=(list(groups).index(key),sum(len(a) for a in groups[key]))
            groups[key].append(v)
        self.draws=[];self.body_blends={}
        for (t,n,g,foot),vertices in groups.items():
            self.body_blends[len(self.draws)]=foot
            self.draws.append((np.concatenate(vertices),t,n,g))
        for transform in self.transforms:
            if 'draw_index' in transform:
                transform['draw_index'],transform['vertex_start']=lookup[transform['draw_index']]

        x,y=np.meshgrid(np.linspace(-.5,self.n+.5,512),np.linspace(-.5,self.n+.5,512));h=np.clip((self.top(x,y)+.25)/2,0,1)
        for triangles in self.casters:
            for tri in triangles:
                uv=(tri[:,:2]+.5)/(self.n+1)*511
                lo=np.maximum(0,np.floor(uv.min(axis=0)).astype(int));hi=np.minimum(511,np.ceil(uv.max(axis=0)).astype(int))
                if np.any(hi<lo):continue
                gx,gy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
                a,b,c=uv;det=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
                if abs(det)<1e-9:continue
                u=((b[1]-c[1])*(gx-c[0])+(c[0]-b[0])*(gy-c[1]))/det
                v=((c[1]-a[1])*(gx-c[0])+(a[0]-c[0])*(gy-c[1]))/det
                w=1-u-v;mask=(u>=0)&(v>=0)&(w>=0)
                z=(u*tri[0,2]+v*tri[1,2]+w*tri[2,2]+.25)/2
                h[gy,gx]=np.where(mask,np.maximum(h[gy,gx],z),h[gy,gx])
        self.textures.append((512,512,61,[(512,(h*255).astype('u1').tobytes())]));heightid=len(self.textures)
        sand=self.texture(SKIN/'textures/desert_base_color.dds');mountaintop=self.texture(SKIN/'textures/mtn_top_base_color.dds');rock=self.texture(SKIN/'textures/mtn_top_base_color.dds')
        out=OWN/f'{self.name}-r{self.revision}';out.mkdir(parents=True,exist_ok=True)
        packet=out/'geometry.packet';f=packet.open('wb')
        def ints(*a):f.write(struct.pack('<'+'I'*len(a),*a))
        def blob(b):ints(len(b));f.write(b)
        ints(0x32514c43,2,*self.viewport,1,len(self.textures))
        for w,h,fmt,mips in self.textures:
            ints(w,h,fmt,len(mips))
            for pitch,b in mips:ints(pitch);blob(b)
        ints(1+len(self.draws));blob(struct.pack('<24f',*([0]*20+list(self.bounds))))
        for vertices,_,_,_ in self.draws:blob(vertices.tobytes())
        ints(len(self.draws))
        for i,(v,texture,normal,gloss) in enumerate(self.draws):
            ints(i+1,0,len(v),60,0,1,0,5)
            for comps,off in [(3,0),(3,12),(2,24),(3,32),(4,44)]:ints(comps,off)
            ints(texture,heightid,mountaintop if v[0,14]==4 else sand,rock,normal,gloss,self.body_blends.get(i,0),*([0]*121))
        f.close()
        import sys
        sys.path.insert(0,str(V2/'app'))
        try:
            from packet_store import compact_packet
            compact_packet(packet,OWN/'.content')
        finally:sys.path.pop(0)
        terrain=out/'proxy.csv';terrain.write_text(f'C3X_BIQ_TERRAIN_WINDOW_V2,{self.n},{self.n},{self.n*self.n},0,0,60,60,0\n'+''.join(f'{x},{y},{x+y},{y-x},2,2,0,0,0\n' for y in range(self.n) for x in range(self.n)))
        fixture=json.loads((V2/'tests/platform/micro.fixture.json').read_text());fixture.update(id=f'relief-{self.name}-r{self.revision}',track='Q4-relief',tile_count=self.n*self.n,viewport=list(self.viewport),terrain=rel(terrain),modules=[rel(V2/'systems/relief/relief.module.json')],references=['civ6.hills','civ6.mountain','civ6.rocky_hill_coast'],isolations=['relief'],scenarios={'relief_packet':rel(packet)})
        if self.real:
            fixture['terrain']=self.real['terrain'];fixture['tile_count']=self.real['tile_count'];fixture['real_map']=self.real['real_map']
        (out/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
        evidence={'schema':'c3x.q4.source_geometry.v1','class':'synthetic_stress_case','proxy_inputs':['Q2 continuous ground','Q3 signed shore and static water'],'classification':{'rocks':'source_adaptation','hills':'source_adaptation','mountains_volcano':'diagnostic: exact source-height pixels, unproven physical coordinate reconstruction','dunes':'diagnostic_proxy: source material height at macro scale','water':'diagnostic_proxy'},'selected_beauty':False,'source_files':self.sources,'transforms':self.transforms,'geometry_sha256':digest(packet),'generator_sha256':digest(__file__),'fixture':rel(out/'fixture.json'),'revision':self.revision,'q5_envelope':self.q5_envelope,'clearance_envelopes':self.envelopes,'rejected_instances':self.rejected}
        if self.real:
            evidence['class']='unaltered_real_terrain';evidence['real_map']=self.real['real_map'];evidence['proxy_inputs']=['Q2 ground material provisional','Q6 shared receiver provisional'];evidence['terrain_sha256']=digest(ROOT/self.real['terrain'])
        (out/'provenance.json').write_text(json.dumps(evidence,indent=2)+'\n')
        shadow_draws=[]
        for i,(vertices,texture,normal,gloss) in enumerate(self.draws):
            mode=int(vertices[0,14]);water=bool(vertices[0,13]>.5)
            shadow_draws.append(dict(draw_index=i,vertex_buffer=i+1,vertex_count=len(vertices),triangle_count=len(vertices)//3,casts=not water,receives=True,pose='static, already transformed in packet world positions',alpha_mode='cutout' if mode%2 else 'opaque',alpha_texture=texture if mode%2 else None,alpha_cutoff=.32 if mode%2 else None,source_footprint_texture=self.body_blends.get(i),source_footprint_cutoff=.025 if mode==4 else None))
        shadow_manifest=dict(schema='c3x.q4.caster_packet_manifest.v1',packet=rel(packet),packet_sha256=digest(packet),source_provenance=rel(out/'provenance.json'),coordinate_space='q4_orthonormal_tile_units_v1',vertex_layout=dict(stride=60,world_xyz_offset=32,source_uv_offset=24,scalar='float32_little_endian',topology='nonindexed_triangle_list'),draws=shadow_draws,shadow_owner='Q6 shared actual-triangle light-depth and receiver evaluation',limits=['Current private height-atlas shadow is diagnostic only.','Cutout filtering/cutoff and shared source-world conversion need convergence validation.','Clearance envelopes are placement constraints, never shadow geometry.','Mountain coordinate hypotheses remain diagnostic.'])
        (out/'shadow_casters.json').write_text(json.dumps(shadow_manifest,indent=2)+'\n')
        print(rel(out/'fixture.json'))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('name');a.add_argument('--revision',type=int,default=1);v=a.parse_args();Fixture(v.name,v.revision).write()
