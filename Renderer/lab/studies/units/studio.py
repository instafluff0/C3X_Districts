"""Independent CPU unit studio: no game renderer, sprite canvas, or cached bitmap.

Source geometry/palettes and decoded authored frames are read-only inputs.
Lighting is an explicit studio choice; this does not claim the Civ VI engine.
"""
from pathlib import Path
import argparse, json, struct, io
import numpy as np
from PIL import Image
from .prepare import ROOT,SOURCE,read
from .frame_probe import OUT as FRAME_ROOT
OUT=ROOT/'Renderer/lab/out/units/studio'


def normalize(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-15)

def srgb(v):return np.where(v<=.04045,v/12.92,((v+.055)/1.055)**2.4)

def display(v):
    v=np.maximum(v,0)
    return np.clip(np.where(v<=.0031308,12.92*v,1.055*v**(1/2.4)-.055),0,1)


def sample(texture,uv,clamp=False):
    h,w,_=texture.shape;p=uv*np.array([w,h])-.5
    ij=np.floor(p).astype(int);f=p-ij
    x0,x1=ij[:,0],ij[:,0]+1;y0,y1=ij[:,1],ij[:,1]+1
    if clamp:x0,x1,y0,y1=np.clip(x0,0,w-1),np.clip(x1,0,w-1),np.clip(y0,0,h-1),np.clip(y1,0,h-1)
    else:x0,x1,y0,y1=x0%w,x1%w,y0%h,y1%h
    return (texture[y0,x0]*(1-f[:,0,None])+texture[y0,x1]*f[:,0,None])*(1-f[:,1,None])+(texture[y1,x0]*(1-f[:,0,None])+texture[y1,x1]*f[:,0,None])*f[:,1,None]


def load_parts(action='idle',phase=0):
    unit=read(SOURCE/'manifest.json')['units']['unit/warrior'];frames=read(FRAME_ROOT/'frames.json')['components'];parts=[];texture_cache={}
    for p in unit['actions'][action]['parts']:
        b=(SOURCE/p['mesh']).read_bytes();_,n,ni,nb,nf=struct.unpack_from('<5I',b,8)
        rows=np.array([struct.unpack_from('<8f4I4f',b,32+i*64) for i in range(n)])
        indices=np.frombuffer(b,dtype='<u4',offset=32+n*64,count=ni).reshape(-1,3)
        m=np.frombuffer(b,dtype='<f4',offset=32+n*64+ni*4).reshape(nf,nb,4,4)
        f=phase*(nf-1);a=min(int(f),nf-1);c=min(a+1,nf-1);palette=m[a]*(1-(f-a))+m[c]*(f-a)
        joints=rows[:,8:12].astype(int);weights=rows[:,12:16]
        matrices=sum(weights[:,i,None,None]*palette[joints[:,i]] for i in range(4))
        pos=np.einsum('vi,vij->vj',np.c_[rows[:,:3],np.ones(n)],matrices)[:,:3]
        # Preserve the established normal transform. Tangents use source linear skin transforms.
        linear=matrices[:,:3,:3];cofactor=np.stack([np.cross(linear[:,1],linear[:,2]),np.cross(linear[:,2],linear[:,0]),np.cross(linear[:,0],linear[:,1])],axis=1)
        det=np.linalg.det(linear);normal_matrix=cofactor/np.where(abs(det)>1e-12,det,1)[:,None,None]
        normal=normalize(np.einsum('vi,vij->vj',rows[:,3:6],normal_matrix))
        tangent=normalize(np.einsum('vi,vij->vj',np.array(frames[p['asset']]['tangents']),linear))
        bitangent=normalize(np.einsum('vi,vij->vj',np.array(frames[p['asset']]['bitangents']),linear))
        textures={}
        for name,channel in p['material']['channels'].items():
            key=channel['texture']
            if key not in texture_cache:
                raw=bytearray((SOURCE/key).read_bytes());fmt=struct.unpack_from('<I',raw,128)[0]
                # Pillow decodes the same BC block bits through UNORM; explicit
                # sRGB conversion below retains the source color-space meaning.
                if fmt in (72,78):struct.pack_into('<I',raw,128,fmt-1)
                tex=np.array(Image.open(io.BytesIO(raw)).convert('RGBA')).astype(np.float32)/255
                if channel['color_space']=='srgb':tex[:,:,:3]=srgb(tex[:,:,:3])
                texture_cache[key]=tex
            textures[name]=(texture_cache[key],channel.get('address_u')=='clamp')
        parts.append({'asset':p['asset'],'position':pos,'normal':normal,'tangent':tangent,'bitangent':bitangent,'uv':rows[:,6:8],
                      'indices':indices,'material':p['material'],'textures':textures})
    anatomy=np.concatenate([p['position'] for p in parts if p['asset'].split('/')[-1] in ('body','head','armor')])
    low=anatomy[:,2].min();height=np.ptp(anatomy[:,2])
    for p in parts:p['position']=(p['position']-np.array([0,0,low]))/height
    return parts


def raster(points,indices,width,height):
    """Yield visible triangle pixel centers and barycentrics; no geometry simplification."""
    for tri in indices:
        p=points[tri];lo=np.maximum(np.floor(p[:,:2].min(axis=0)).astype(int),0);hi=np.minimum(np.ceil(p[:,:2].max(axis=0)).astype(int),[width-1,height-1])
        if np.any(hi<lo):continue
        x,y=np.meshgrid(np.arange(lo[0],hi[0]+1)+.5,np.arange(lo[1],hi[1]+1)+.5)
        denominator=(p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
        if abs(denominator)<1e-10:continue
        a=((p[1,1]-p[2,1])*(x-p[2,0])+(p[2,0]-p[1,0])*(y-p[2,1]))/denominator
        b=((p[2,1]-p[0,1])*(x-p[2,0])+(p[0,0]-p[2,0])*(y-p[2,1]))/denominator
        inside=(a>=-1e-7)&(b>=-1e-7)&(a+b<=1+1e-7)
        yield tri,x[inside].astype(int),y[inside].astype(int),np.stack((a[inside],b[inside],1-a[inside]-b[inside]),axis=-1)


def render(size=1200,elevation=40,yaw=45,source_specular=True,normal_detail=True):
    parts=load_parts();angle=np.radians(elevation);az=np.radians(yaw)
    view=np.array([np.cos(az)*np.cos(angle),np.sin(az)*np.cos(angle),np.sin(angle)])
    right=normalize(np.cross(np.array([0,0,1]),view));up=np.cross(view,right)
    camera=np.stack((right,up,view),axis=1);scale=size*.68;center=np.array([size*.52,size*.75])
    depth=np.full((size,size),-np.inf,dtype=np.float32);ids=np.full((size,size),-1,dtype=np.int16)
    g=np.zeros((size,size,14),dtype=np.float32)
    for k,p in enumerate(parts):
        projected=p['position']@camera;projected[:,:2]=projected[:,:2]*[scale,-scale]+center
        attributes=np.c_[p['position'],p['normal'],p['tangent'],p['bitangent'],p['uv']]
        for tri,x,y,bary in raster(projected,p['indices'],size,size):
            z=bary@projected[tri,2];keep=z>depth[y,x];x,y,bary,z=x[keep],y[keep],bary[keep],z[keep]
            values=bary@attributes[tri];keep=values[:,2]>=-1e-6;x,y,values,z=x[keep],y[keep],values[keep],z[keep]
            depth[y,x]=z;ids[y,x]=k;g[y,x]=values
    # One studio key plus hemispherical fill. These are deliberate reference
    # lighting parameters, not recovered Civ VI time-of-day constants.
    light=normalize(np.array([-.35,-.65,1.0]));key=np.array([3.2,3.05,2.9]);sky=np.array([.42,.50,.66]);ground=np.array([.12,.10,.08])
    rgb=np.full((size,size,3),srgb(np.array([.55,.59,.59])),dtype=np.float32)
    owner=srgb(np.array([.125,.357,.867]))
    for k,p in enumerate(parts):
        y,x=np.where(ids==k);v=g[y,x];uv=v[:,12:14];n=normalize(v[:,3:6]);t=v[:,6:9];bt=v[:,9:12]
        def tex(name,default):
            return sample(*((p['textures'][name][0],uv,p['textures'][name][1]))) if name in p['textures'] else np.broadcast_to(default,(len(x),len(default)))
        base=tex('base_color',[1,1,1,1]);mat=p['material'];tint=np.array(mat['tint_rgb'])
        if mat['owner_color']['mode']!='none':tint=owner
        albedo=base[:,:3]*(1-base[:,3,None]+base[:,3,None]*tint)
        xy=tex('normal_0',[.5,.5,1,1])[:,:2]*2-1
        if normal_detail:n=normalize(t*xy[:,0,None]+bt*xy[:,1,None]+n*np.sqrt(np.maximum(0,1-np.sum(xy*xy,axis=1)))[:,None])
        ao=tex('ambient_occlusion',[1,1,1,1])[:,0]
        ndl=np.maximum(n@light,0);hemi=np.clip(n[:,2]*.5+.5,0,1)
        fill=ground+(sky-ground)*hemi[:,None]
        radiance=albedo*(fill*ao[:,None]+key*ndl[:,None]/np.pi)
        rough=tex('gloss',[.1,.3,.1,1])[:,:3]
        if source_specular:
            geometric=normalize(v[:,3:6]);h=normalize(light+view);hz=geometric@h
            offset=np.stack((t@h,bt@h),axis=-1)/np.maximum(hz[:,None],1e-6)-xy
            inv=.5/np.maximum(rough[:,:2],1e-6);lobes=inv*np.exp(-np.minimum(inv*np.sum(offset*offset,axis=1)[:,None],256))
            distribution=.25*(rough[:,2]+lobes@np.array([1/3,2/3]))
            f0=.04*(1-np.clip(np.sqrt(np.pi*rough[:,2])-.35,0,1))**2
            fresnel=f0+(1-f0)*(1-np.clip(h@light,0,1))**5
            radiance+=key*(distribution*fresnel*ndl*(hz>0))[:,None]
        rgb[y,x]=radiance
    # Direct display transfer: no C3X max-channel tone compressor, no sharpening.
    result=Image.fromarray(np.uint8(np.round(display(rgb)*255)))
    OUT.mkdir(parents=True,exist_ok=True);name=f'warrior-e{elevation}-a{yaw}-s{size}'
    result.save(OUT/(name+'.png'))
    (OUT/(name+'.json')).write_text(json.dumps({'size':size,'camera_elevation':elevation,'camera_yaw':yaw,'body_world_height':1,
        'input_components':len(parts),'input_vertices':sum(len(p['position']) for p in parts),'input_triangles':sum(len(p['indices']) for p in parts),
        'source_preserved':['geometry','UVs','authored frames','source first idle pose','base/tint','normal_0','AO','three cooked roughness channels'],
        'inferred':['studio key and hemispherical fill','orthographic camera','direct sRGB display exposure'],
        'not_implemented':['cast shadows','second LEAN variance constant','source environment cube/SH','metalness extra-slot intake'],
        'clipped_display_fraction':float(np.any(rgb[ids>=0]>1,axis=-1).mean())},indent=2)+'\n')
    print('PASS independent studio',name,flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--size',type=int,default=1200);parser.add_argument('--elevation',type=int,default=40);parser.add_argument('--yaw',type=int,default=45);args=parser.parse_args()
    render(args.size,args.elevation,args.yaw)
