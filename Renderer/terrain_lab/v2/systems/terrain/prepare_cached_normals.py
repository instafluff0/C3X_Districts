"""Source-kernel normal baking diagnostic; cache scale is explicitly calibrated."""
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[5]
V2=ROOT/'Renderer/terrain_lab/v2'


def encoded_normals(padded,scale=1):
    a=padded
    gx=a[:-2,2:]+2*a[1:-1,2:]+a[2:,2:]-a[:-2,:-2]-2*a[1:-1,:-2]-a[2:,:-2]
    gy=a[2:,:-2]+2*a[2:,1:-1]+a[2:,2:]-a[:-2,:-2]-2*a[:-2,1:-1]-a[:-2,2:]
    v=np.stack([gx*scale,-gy*scale,np.ones_like(gx)],axis=-1)
    v/=np.linalg.norm(v,axis=-1,keepdims=True)
    return v[:,:,:2]*.5+.5


def verify_source():
    folder=V2/'audits/beauty/out/ground-shader-source'
    a=np.fromfile(folder/'normal-probe-input.f32',dtype='<f4').reshape(34,34)
    output=np.fromfile(folder/'normal-probe-output.rgba8',dtype=np.uint8).reshape(32,32,4)
    expected=np.rint(encoded_normals(a)*255).astype(np.uint8)
    delta=np.abs(expected.astype(int)-output[:,:,:2].astype(int))
    assert delta.max()==0 and np.max(output[:,:,2:])==0
    return {'source_shader_sha256':hashlib.sha256((folder/'shader-00426326.dxbc').read_bytes()).hexdigest(),
        'source_disassembly_sha256':hashlib.sha256((folder/'shader-00426326.asm').read_bytes()).hexdigest(),
        'input_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
        'gpu_output_sha256':hashlib.sha256(output.tobytes()).hexdigest(),
        'pixels':1024,'max_rgba8_channel_error':int(delta.max()),'dispatch':'D3D11 WARP, original source bytecode'}


def write_dds(path,values):
    height,width=values.shape[:2]
    levels=[]
    while True:
        levels.append(np.rint(np.clip(values,0,1)*65535).astype('<u2').tobytes())
        if values.shape[0]==1:break
        values=(values[::2,::2]+values[1::2,::2]+values[::2,1::2]+values[1::2,1::2])*.25
    header=[0]*31
    header[:7]=[124,0x2100f,height,width,width*4,0,len(levels)]
    header[18:22]=[32,4,struct.unpack('<I',b'DX10')[0],0]
    header[26]=0x401008
    path.write_bytes(b'DDS '+struct.pack('<31I',*header)+struct.pack('<5I',35,3,0,1,0)+b''.join(levels))


def main():
    proof=verify_source()
    destination=V2/'fixtures/beauty/source-normal-cache-r1'
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'.gitignore').write_text('*.dds\n')
    source=ROOT/'Renderer/packs/Civ5EnvironmentSkin'
    records=[]
    for name in ['grassland','plains','desert','marsh','tundra']:
        material=json.loads((source/'materials'/(name+'.json')).read_text())
        path=source/material['height']['texture']
        im=Image.open(path).convert('L')
        original_size=im.size
        im=im.resize((1024,1024),Image.Resampling.BOX)
        field=np.array(im,dtype=np.float32)/255
        normals=encoded_normals(np.pad(field,1,mode='wrap'))
        target=destination/(name+'.dds')
        write_dds(target,normals)
        records.append({'material':name,'height':path.relative_to(ROOT).as_posix(),
            'height_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_size':list(original_size),'cache_size':[1024,1024],
            'normal':target.relative_to(ROOT).as_posix(),'normal_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    report={'classification':'source-kernel reconstruction verified; per-material cache size/scale and processing boundary remain Lab hypotheses',
        'normal_scale':1,'input_height':'normalized source red, BOX downsample before kernel',
        'mips':'average encoded RG after normal construction, RG16_UNORM',
        'limitation':'per-material baking precedes cross-material blending; source engine bakes combined height',
        'source_compute_parity':proof,'materials':records}
    (destination/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'source_compute_max_error':0,'prepared_materials':len(records)}))


if __name__=='__main__':main()
