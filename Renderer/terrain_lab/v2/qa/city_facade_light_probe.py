"""Derive explicitly approximate facade light proxies and render matched packets.

Requires Pillow/NumPy only offline. Source light/VFX attachment binding is not
claimed; the generic light record is derived from sampled emissive geometry.
"""
import argparse
from collections import defaultdict
import json
import io
import math
from pathlib import Path
import subprocess
import struct
import shutil
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[4];V2=ROOT/'Renderer/terrain_lab/v2'
sys.path.insert(0,str(V2/'systems/objects'));import presentation as city
sys.path.insert(0,str(V2/'app'));import runner
from cache import file_hash


def linear(rgb):return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)


def emission_texture(path):
    payload=bytearray(path.read_bytes())
    if payload[84:88]==b'DX10':
        fmt=struct.unpack_from('<I',payload,128)[0]
        if fmt in (72,75,78):struct.pack_into('<I',payload,128,fmt-1)
    # Pillow's BC1/2/3 DX10 decoder omits their SRGB aliases. Decode the same
    # compressed blocks via the UNORM alias, then linearize explicitly.
    return linear(np.array(Image.open(io.BytesIO(payload)).convert('RGB'),dtype=float)/255)


def bilinear(image,uv):
    h,w,_=image.shape;x=np.clip(uv[0]*w-.5,0,w-1);y=np.clip(uv[1]*h-.5,0,h-1)
    x0=int(x);y0=int(y);x1=min(x0+1,w-1);y1=min(y0+1,h-1)
    return (image[y0,x0]*(1-(x-x0))+image[y0,x1]*(x-x0))*(1-(y-y0))+ (image[y1,x0]*(1-(x-x0))+image[y1,x1]*(x-x0))*(y-y0)


def bounded_lights(lights,budget):
    """Keep one strongest proxy per emitting body, then the strongest remainder."""
    if not 1<=budget<=128:raise ValueError('invalid facade proxy budget')
    if len(lights)<=budget:return lights
    groups=defaultdict(list)
    def energy(index):
        light=lights[index]
        return light['intensity']*float(np.dot(light['color_linear'],[.2126,.7152,.0722]))*light['range']**2
    for index,light in enumerate(lights):groups[light['owner']].append(index)
    if len(groups)>budget:raise ValueError('proxy budget cannot represent every emitting building')
    selected={max(indices,key=lambda i:(energy(i),-i)) for indices in groups.values()}
    remaining=sorted((i for i in range(len(lights)) if i not in selected),key=lambda i:(-energy(i),i))
    selected.update(remaining[:budget-len(selected)])
    return [light for i,light in enumerate(lights) if i in selected]


def facade_plane_proxy(points,normals,weights,offset=.012):
    """Place a light beyond its sampled wall, retaining the wall's orientation."""
    direction=np.average(normals,axis=0,weights=weights);direction[2]=0
    length=np.linalg.norm(direction)
    if length<.5:raise ValueError('facade samples have incompatible directions')
    direction/=length
    center=np.average(points,axis=0,weights=weights)
    support=max(points@direction)
    return center+direction*(support-float(center@direction)+offset),direction


def derive(augmentation,surface,light_budget=48,source_facade_slots=()):
    if augmentation['emissive_uv']!=2 or augmentation['grounding']!='source_z_zero':raise ValueError('probe requires verified UV2 emission and source ground zero')
    metric=1/augmentation['scene_world_z_per_source_unit'];pack=Path(augmentation['pack'])
    normal_binding=augmentation.get('source_normals') or {}
    frames=city.read(normal_binding['mapping'])['meshes'] if normal_binding.get('mapping') else {}
    textures={};lights=[];boxes=[];bindings=[]
    for owner,instance in enumerate(augmentation['instances']):
        body_pack=Path(augmentation['capital']['mapping'].get('pack',pack)) if instance['slot']=='capital' else pack
        body=city.component(instance['asset'],body_pack);scale=instance['scale'];rotation=instance['rotation']
        site=surface['samples'][instance['sample_start']]
        origin=np.array([site['column']+site['u'],-(site['row']+1-site['v']),site['height']/112*metric])
        center=np.array([(body['lo'][0]+body['hi'][0])/2,(body['lo'][1]+body['hi'][1])/2,0])
        local=instance['local_bounds']
        boxes.append({'low':(origin+np.array([local[0],local[1],max(0,body['lo'][2])*scale])).tolist(),
                      'high':(origin+np.array([local[2],local[3],body['hi'][2]*scale])).tolist(),'owner':owner})
        groups=defaultdict(list);group_normals=defaultdict(list)
        for mesh,material in body['parts']:
            channel=material['channels'].get('emissive')
            if not channel or material['alpha_mode']!='opaque':continue
            if channel.get('color_space')!='srgb':raise ValueError('probe currently requires an explicitly SRGB emissive texture')
            # Match the displayed UV2 emission pass, which uses decal_sampler
            # (clamp); imported material addressing also serves base-color UV0.
            path=channel['texture']
            if path not in textures:textures[path]=emission_texture(ROOT/path)
            vertices=mesh['vertices'];normals=frames.get(mesh.get('asset_id'),{}).get('normals')
            indices=mesh['topology']['indices']
            for start in range(0,len(indices),3):
                ids=indices[start:start+3];v=[vertices[i] for i in ids]
                p=np.array([x['position'] for x in v]);uv=np.array([x['uv2'] for x in v])
                n=np.array([normals[i] if normals else vertices[i]['normal'] for i in ids]).mean(0)
                n/=max(np.linalg.norm(n),1e-9)
                if abs(n[2])>.55:continue
                area=np.linalg.norm(np.cross(p[1]-p[0],p[2]-p[0]))*.5*scale*scale
                if area<1e-8:continue
                # Fixed barycentric quadrature samples the lower window rows.
                for u in range(8):
                    for w in range(8-u):
                        weights=np.array([(u+1/3)/8,(w+1/3)/8,1-(u+w+2/3)/8])
                        pos=weights@p
                        if pos[2]<0 or pos[2]>body['hi'][2]*.4:continue
                        color=bilinear(textures[path],weights@uv)
                        luminance=float(color@np.array([.2126,.7152,.0722]))
                        if luminance<.015:continue
                        world_normal=np.array(city.rotate(n.tolist(),rotation));axis=int(np.argmax(abs(world_normal[:2])))
                        sign=1 if world_normal[axis]>0 else -1
                        point=np.array(city.rotate((pos-center).tolist(),rotation))*scale
                        groups[axis,sign].append((point,color,area/36,luminance))
                        if instance['slot'] in source_facade_slots:group_normals[axis,sign].append(world_normal)
        for (axis,sign),samples in sorted(groups.items()):
            weights=np.array([a*l for _,_,a,l in samples]);total=float(weights.sum())
            if total<1e-6:continue
            pos=sum(p*w for (p,_,_,_),w in zip(samples,weights))/total
            sample_center=pos.copy()
            # Put the proxy on the outer facade plane to avoid an averaged
            # inset centroid living inside its owning building.
            pos[axis]=local[axis+2 if sign>0 else axis]+sign*.012
            direction=[0.,0.,0.];direction[axis]=float(sign)
            if instance['slot'] in source_facade_slots:
                pos,direction=facade_plane_proxy(np.array([p for p,_,_,_ in samples]),np.array(group_normals[axis,sign]),weights)
                direction=direction.tolist()
            color=sum(c*a for _,c,a,_ in samples)/sum(a for _,_,a,_ in samples)
            span=max(local[2]-local[0],local[3]-local[1]);radius=max(.32,min(.55,span*1.6))
            intensity=augmentation['emissive_gain']*min(2.5,max(.35,total/(radius*radius)*30))
            lights.append({'owner':owner,'position':(origin+pos).tolist(),'range':radius,'color_linear':color.tolist(),
                           'intensity':intensity,'direction':direction,'sample_count':len(samples),
                           'source_sample_centroid':(origin+sample_center).tolist()})
        bindings.append({'asset':body['id'],'source_sockets':body['sockets'],
                         'classification':'Existing unresolved sockets are not enabled by this derived facade approximation'})
    # Individual-house palettes need more bodies than assembled modern blocks.
    # Each body contributes at most four facade groups; the receiver shader
    # already rejects pixels outside the complete light envelope.
    if not lights or len(lights)>128 or len(boxes)>32:raise ValueError('local light bounds exceeded or no emitting facade samples')
    source_light_count=len(lights)
    lights=bounded_lights(lights,light_budget)
    return {'schema':'c3x.lab.local_facade_lights.v1','classification':'Authored emissive-spill approximation; not decoded source light bindings or exact area-light transport',
            'z_metric':metric,'emission_sampling':'UV2 clamp, matching the displayed separate emission shader',
            'lights':lights,'blockers':boxes,'texture_sha256':{p:file_hash(ROOT/p) for p in textures},'source_attachment_audit':bindings,
            'proxy_budget':light_budget,'unbounded_proxy_count':source_light_count,
            'proxy_selection':'Strongest energy proxy per emitting body, then strongest remaining; original order retained. Window emission unchanged.'}


def shader_data(data,gain):
    def array(name,values):
        rows=[','.join(f'{float(x):.10f}' for x in row) for row in values]
        return f'static const float4 {name}[{len(rows)}]={{'+','.join('float4('+r+')' for r in rows)+'};\n'
    text=f'#define Q8_LOCAL_LIGHT_GAIN {gain:.8f}\n#define Q8_LOCAL_Z_METRIC {data["z_metric"]:.10f}\n#define Q8_LOCAL_LIGHT_COUNT {len(data["lights"])}\n#define Q8_LOCAL_BLOCKER_COUNT {len(data["blockers"])}\n'
    text+=array('Q8LocalPositionRange',[l['position']+[l['range']] for l in data['lights']])
    text+=array('Q8LocalColorIntensity',[l['color_linear']+[l['intensity']] for l in data['lights']])
    text+=array('Q8LocalDirectionOwner',[l['direction']+[l['owner']] for l in data['lights']])
    text+=array('Q8LocalBoxLow',[b['low']+[0] for b in data['blockers']])
    text+=array('Q8LocalBoxHigh',[b['high']+[0] for b in data['blockers']])
    low=[min(l['position'][i]-l['range'] for l in data['lights']) for i in range(3)]
    high=[max(l['position'][i]+l['range'] for l in data['lights']) for i in range(3)]
    for name,values in [('Q8LocalEnvelopeLow',low),('Q8LocalEnvelopeHigh',high)]:
        text+=f'static const float3 {name}=float3('+','.join(f'{v:.10f}' for v in values)+');\n'
    return text+(V2/'shaders/lighting/local_facade_lights.hlsl').read_text()+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--augmentation',type=Path,required=True)
    p.add_argument('--source-render',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gain',type=float,default=1)
    p.add_argument('--light-budget',type=int,default=48,help='Bounded spill proxies; every emitting body retains at least one')
    p.add_argument('--prepare-only',action='store_true',help='Prepare independent backend inputs without invoking Metal')
    p.add_argument('--resume',action='store_true',help='Repair preparation before any rendered output exists')
    p.add_argument('--no-blockers',action='store_true',help='Unselected diagnostic: disable local building occlusion')
    p.add_argument('--no-object-reflections',action='store_true',help='Diagnostic: retain local illumination but disable the object reflection pass')
    a=p.parse_args()
    if not 0<=a.gain<=4:raise ValueError('bounded light gain required')
    source=ROOT/a.source_render;output=ROOT/a.output
    output.resolve().relative_to(V2/'audits/beauty/out')
    if shutil.disk_usage(V2).free<8*1024**3:raise ValueError('preserve at least 8 GiB free disk space')
    if output.exists() and any(output.iterdir()) and not a.resume:raise ValueError('preserve prior local light probe')
    if a.resume and ((output/'render/report.json').exists() or list((output/'render').glob('*.bmp'))):raise ValueError('preserve rendered result')
    output.mkdir(parents=True,exist_ok=True)
    augmentation=city.read(a.augmentation);surface=city.read(a.augmentation.parent/'surface.json')
    data=derive(augmentation,surface,a.light_budget);data['gain']=a.gain
    data['local_building_occlusion']=not a.no_blockers;data['object_reflections']=not a.no_object_reflections
    data['augmentation_sha256']=file_hash(ROOT/a.augmentation)
    (output/'lights.json').write_text(json.dumps(data,indent=2)+'\n')
    report=city.read(a.source_render/'report.json');input_report=ROOT/report['source_report']
    for name,path in [('combined',source/'shaders/source.hlsl'),('reflection',source/'shaders/reflection/source.hlsl')]:
        shader=path.read_text();marker='float q6_receiver_visibility(PixelInput input,float3 normal,float legacy_shadow) {'
        if shader.count(marker)!=1:raise ValueError('shared receiver entry changed')
        shader=shader.replace(marker,f'#define Q8_LOCAL_OCCLUSION {int(not a.no_blockers)}\n'+shader_data(data,a.gain)+marker)
        marker='return frame_illumination(normal,'
        if shader.count(marker)!=3:raise ValueError('shared receiver lighting wrappers changed')
        shader=shader.replace(marker,'return q8_local_irradiance(input.q6_world,normal,ambient_visibility)+frame_illumination(normal,')
        if a.no_object_reflections:
            count=shader.count('#define Q3_OBJECT_REFLECTION 1')
            if (name=='combined' and count!=1) or count>1:raise ValueError('object reflection control marker changed')
            shader=shader.replace('#define Q3_OBJECT_REFLECTION 1','// Object reflection disabled for explicit diagnostic')
        (output/f'{name}.hlsl').write_text(shader)
    subprocess.run([sys.executable,str(V2/'qa/replay_shader.py'),'--report',str(input_report),'--shader',str(output/'combined.hlsl'),
                    '--post-shader',str(source/'postprocess/source.hlsl'),'--output',str(output/'render')]+
                    ([] if a.no_object_reflections else ['--reflection-shader',str(output/'reflection.hlsl')])+
                    (['--prepare-only'] if a.prepare_only else []),check=True)


if __name__=='__main__':main()
