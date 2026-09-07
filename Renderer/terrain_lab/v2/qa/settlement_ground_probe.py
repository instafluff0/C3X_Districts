"""Compose an authored settlement underlay on fixed city packets and lighting."""
import argparse
import io
import json
import math
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import sys

from PIL import Image
from city_scene_pass import ROOT,V2,Cache,executable,compact_packet,rel,save,city
from cache import file_hash
from city_ground_geometry import clip_ground_to_land_cells
from settlement_ground import coverage,grid,convex_hull


def prepare(augmentation,parts,binding,output,margin,feather,capital_footprint='bounds'):
    data=json.loads((ROOT/augmentation).read_text());surface=json.loads((ROOT/augmentation.parent/'surface.json').read_text())
    ground=json.loads((ROOT/parts).read_text());material=json.loads((ROOT/binding).read_text())['replacement']
    path=ROOT/material['texture']
    if file_hash(path)!=material['sha256']:raise ValueError('atlas fingerprint changed')
    raw=bytearray(path.read_bytes())
    if struct.unpack_from('<I',raw,128)[0]!=78:raise ValueError('requires normalized BC3 SRGB atlas')
    struct.pack_into('<I',raw,128,77)
    image=Image.open(io.BytesIO(raw)).convert('RGBA')
    atlas_uv=[.62,.56,.94,.86]
    crop=image.crop(tuple(round(v*s) for v,s in zip(atlas_uv,image.size*2)))
    alpha_range=crop.getchannel('A').getextrema()
    if alpha_range[0]<128:raise ValueError('underlay crop crosses a transparent atlas-piece boundary')
    boxes=[];coverage_boxes=[];polygons=[];densities=[];footprint_sources=[]
    scales=[instance['scale'] for instance in data['instances'] if instance['slot']!='capital']
    scale=scales[0]
    if any(abs(s-scale)>1e-8 for s in scales):raise ValueError('one-era underlay requires a uniform ordinary city scale')
    for instance in data['instances']:
        site=surface['samples'][instance['sample_start']];x=site['column']+site['u'];y=site['row']+1-site['v']
        x0,y0,x1,y1=instance['local_bounds'];box=[x+x0,y-y1,x+x1,y-y0];boxes.append(box)
        if instance['slot']=='capital' and capital_footprint=='source-hull':
            mapping=data['capital']['mapping'];pack=Path(mapping['pack'])
            body=city.component(instance['asset'],pack);points=[]
            for mesh,body_material in body['parts']:
                if body_material['alpha_mode']=='blend':continue
                for v in mesh['vertices']:
                    local=[v['position'][j]-(body['lo'][j]+body['hi'][j])/2 for j in (0,1)]+[0]
                    dx,dy,_=city.rotate(local,instance['rotation'])
                    points.append([x+dx*instance['scale'],y-dy*instance['scale']])
            hull=convex_hull(points)
            if any(not box[0]-1e-8<=p[0]<=box[2]+1e-8 or not box[1]-1e-8<=p[1]<=box[3]+1e-8 for p in hull):
                raise ValueError('source footprint no longer matches frozen instance bounds')
            polygons.append(hull)
            manifest=json.loads((ROOT/pack/'manifest.json').read_text())
            landmark=pack/manifest['assets'][instance['asset']]['landmark']
            source=json.loads((ROOT/landmark).read_text())
            paths=[landmark]+[pack/p for p in source['components']['geometry']]
            footprint_sources.append({'asset':instance['asset'],'files':{p.as_posix():file_hash(ROOT/p) for p in paths}})
        else:coverage_boxes.append(box)
    # Use the complete configured source pool, not the visible growth-stage
    # subset, so existing paving coordinates do not shift when a city grows.
    for parts_for_asset in ground['parts'].values():
        for part in parts_for_asset:
            mesh=part['mesh'];vertices=mesh['vertices'];indices=mesh['topology']['indices']
            for i in range(0,len(indices),3):
                a,b,c=[vertices[k] for k in indices[i:i+3]]
                det=lambda key:(b[key][0]-a[key][0])*(c[key][1]-a[key][1])-(c[key][0]-a[key][0])*(b[key][1]-a[key][1])
                area=abs(det('position'))*scale**2
                if area>1e-12:densities.append(math.sqrt(abs(det('uv0'))*image.width*image.height/area))
    density=statistics.median(densities)
    if not 50<=density<=2000:raise ValueError('source-derived texel density outside city probe range')
    period=[(atlas_uv[i+2]-atlas_uv[i])*image.size[i]/density for i in (0,1)]
    xy,triangles=grid(boxes,margin);alpha=[coverage(x,y,coverage_boxes,margin,feather,polygons) for x,y in xy]
    points=output/'points.csv'
    points.write_text(''.join(f'{math.floor(x)},{math.floor(y)},{x-math.floor(x)},{1-(y-math.floor(y))}\n' for x,y in xy))
    subprocess.run([sys.executable,str(V2/'app/surface_query.py'),'--fixture',str(ROOT/surface['fixture']),
                    '--points',str(points),'--output',str(output/'surface.json')],check=True)
    queried=json.loads((output/'surface.json').read_text());samples=queried['samples'];projection=queried['projection']
    if projection!=data['projection']:raise ValueError('fixed city projection changed')
    cells={(int(s['column']),int(s['row'])):s['base'] for s in samples}
    excluded={(int(s['column']),int(s['row'])) for s in samples if s['real'] in (7,8)}
    vertices=[]
    for i,s in enumerate(samples):
        x,y=xy[i]
        vertices.append([s['screen_x']/projection['width']*2-1,1-(s['screen_y']-.005)/projection['height']*2,
                         s['depth']-.0000005,x/period[0],y/period[1],s['normal_x'],s['normal_y'],s['normal_z'],
                         62+alpha[i],x,y,s['height']/112,1])
    emitted=[]
    for indices in triangles:
        if max(alpha[i] for i in indices)==0:continue
        emitted.extend(clip_ground_to_land_cells([vertices[i] for i in indices],
                       [samples[i]['shore_distance'] for i in indices],cells,excluded_cells=excluded))
    if not emitted:raise ValueError('no legal settlement ground coverage')
    wire=struct.pack('<II',0x31524753,len(emitted))+b''.join(struct.pack('<13f',*v) for v in emitted)
    (output/'ground.bin').write_bytes(wire)
    return {'schema':'c3x.lab.settlement_ground.v1','classification':'Authored union of building footprints, not recovered source generator geometry',
            'augmentation':augmentation.as_posix(),'augmentation_sha256':file_hash(ROOT/augmentation),
            'ground_parts':parts.as_posix(),'ground_parts_sha256':file_hash(ROOT/parts),
            'atlas':material,'atlas_uv':atlas_uv,'atlas_alpha_range':alpha_range,
            'atlas_crop':'Manually inspected unmarked interior; source alpha retained, mirrored at source-derived texel density',
            'texels_per_tile':density,'density_basis':'complete configured source pool, independent of visible growth stage',
            'uniform_ordinary_city_scale':scale,'tile_period':period,'margin':margin,'feather':feather,'boxes':boxes,
            'capital_footprint':capital_footprint,'coverage_boxes':coverage_boxes,'coverage_polygons':polygons,
            'footprint_sources':footprint_sources,
            'excluded_vegetation_cells':sorted(excluded),'grid_step':.025,'grid_vertices':len(xy),'emitted_vertices':len(emitted),
            'shore_boundary':-.02,'surface_sha256':file_hash(output/'surface.json'),'ground_sha256':file_hash(output/'ground.bin'),
            'remaining':['Source engine ground-height/material-state semantics','Complete route/river mesh clearance','Height-map/specular ground response']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-render',type=Path,required=True);p.add_argument('--augmentation',type=Path,required=True)
    p.add_argument('--ground-parts',type=Path,required=True);p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--margin',type=float,default=.1);p.add_argument('--feather',type=float,default=.025)
    p.add_argument('--gain',type=float,default=1)
    p.add_argument('--capital-footprint',choices=['bounds','source-hull'],default='bounds',
                   help='Preserve authored palace orientation in the added paving footprint')
    a=p.parse_args();source=ROOT/a.source_render;output=ROOT/a.output
    output.resolve().relative_to(V2/'audits/beauty/out')
    if output.exists():raise ValueError('preserve earlier settlement candidate')
    if not .02<=a.margin<=.2 or not .01<=a.feather<=a.margin or not 0<=a.gain<=1:raise ValueError('bounded settlement parameters required')
    if shutil.disk_usage(V2).free<8*1024**3:raise ValueError('preserve 8 GiB free space')
    output.mkdir(parents=True)
    data=prepare(a.augmentation,a.ground_parts,a.binding,output,a.margin,a.feather,a.capital_footprint);data['gain']=a.gain
    base=json.loads((source/'report.json').read_text());input_path=ROOT/base['source_report']
    report=json.loads(input_path.read_text());jobs=json.loads((input_path.parent/'batch.json').read_text())
    exe=executable(V2/'qa/append_settlement_ground.cpp',Cache(V2/'app/.cache'))
    data['packets']=[]
    for i,(job,row) in enumerate(zip(jobs,report['outputs'])):
        packet=output/f'combined-{i}.packet';original=ROOT/row['packet']
        result=subprocess.run([str(exe),str(original),str(output/'ground.bin'),str(ROOT/data['atlas']['texture']),str(packet)],check=True,capture_output=True,text=True)
        compact_packet(packet,V2/'app/.cache/content')
        data['packets'].append({'original':rel(original),'original_sha256':file_hash(original),'output':rel(packet),
                               'output_sha256':file_hash(packet),**json.loads(result.stdout)})
        row['packet']=rel(packet);job[0]=str(packet)
    save(output/'report.json',report);save(output/'batch.json',jobs);save(output/'settlement.json',data)
    for name,path in [('combined',source/'shaders/source.hlsl'),('reflection',source/'shaders/reflection/source.hlsl')]:
        text=path.read_text();marker='Q6SceneOutput Q8_CITY_FEATURE_ENTRY(FeaturePixelInput p) {'
        if text.count(marker)!=1:raise ValueError('city feature entry changed')
        code=f'#define Q8_SETTLEMENT_GAIN {a.gain:.8f}\n#define Q8_SETTLEMENT_ATLAS float4('+','.join(map(str,data['atlas_uv']))+')\n'
        text=text.replace(marker,code+(V2/'shaders/objects/settlement_ground.hlsl').read_text()+'\n'+marker)
        marker='float4 ground=city_base_texture_0.Sample(decal_sampler,p.uv);'
        if text.count(marker)!=1:raise ValueError('city ground sampler changed')
        text=text.replace(marker,'float4 ground=p.material_index>61.5?q8_settlement_ground_sample(p):city_base_texture_0.Sample(decal_sampler,p.uv);')
        (output/f'{name}.hlsl').write_text(text)
    subprocess.run([sys.executable,str(V2/'qa/replay_shader.py'),'--report',str(output/'report.json'),
                    '--shader',str(output/'combined.hlsl'),'--reflection-shader',str(output/'reflection.hlsl'),
                    '--post-shader',str(source/'postprocess/source.hlsl'),'--output',str(output/'render')],check=True)


if __name__=='__main__':main()
