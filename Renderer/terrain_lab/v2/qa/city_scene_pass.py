"""Preserved city composition probe on the fixed 100-tile coastal terrain.

City instances are explicit Lab augmentation, not captured BIQ city state. Source
parts, UVs and uniform preprojection transforms are retained. The terrain packet
is copied, cities appended, and shared shadows rebuilt before GPU reflections.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4];V2=ROOT/'Renderer/terrain_lab/v2';OUT=V2/'audits/beauty/out'
sys.path.insert(0,str(V2/'app'))
import runner
from cache import Cache,file_hash
sys.path.insert(0,str(V2/'systems/objects'))
import presentation as city

def rel(path):return path.relative_to(ROOT).as_posix()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def run(args):subprocess.run([str(x) for x in args],cwd=ROOT,check=True)
def executable(source,cache):
    obj=runner.compile_cpp(cache,source)
    exe=cache.artifact('module-executable',{'object':file_hash(obj)},lambda dst:runner.run(['clang++',obj,'-o',dst]))
    exe.chmod(0o755);return exe

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision',type=int,required=True)
    parser.add_argument('--pool',default='european/medieval')
    parser.add_argument('--size',type=int,choices=[0,1,2],default=1)
    parser.add_argument('--factor',type=float,default=1)
    parser.add_argument('--channels',action='store_true')
    parser.add_argument('--expanded',action='store_true')
    parser.add_argument('--authored-ground',action='store_true',help='place source z=0 at ground; retain negative foundation skirts underground')
    parser.add_argument('--emissive-gain',type=float,default=1.45)
    parser.add_argument('--emissive-uv',type=int,choices=[0,1,2],default=0)
    parser.add_argument('--glow',action='store_true')
    parser.add_argument('--anchor',type=int,nargs=2,default=[3,2])
    parser.add_argument('--all-zooms',action='store_true')
    parser.add_argument('--resume',action='store_true',help='retry an input/build failure before any combined render exists')
    a=parser.parse_args()
    if not .5<=a.factor<=2:raise ValueError('bounded uniform scale factor required')
    if not 0<=a.emissive_gain<=12 or any(v<0 or v>9 for v in a.anchor):raise ValueError('city parameter bounds')
    pack=Path('Renderer/packs/CityStudyExpanded') if a.expanded else city.PACK
    if a.emissive_uv:
        if not a.expanded:raise ValueError('auxiliary coordinates require the separate expanded study pack')
        pack=Path('Renderer/packs/CityStudyAuxiliaryUV')
    pool='city/pool/'+a.pool;catalog=city.read(pack/'city_catalog.json')
    if pool not in catalog['pools']:raise ValueError('unknown pool')
    name=a.pool.replace('/','-')+f'-s{a.size}'
    if a.anchor!=[3,2]:name+='-at'+'-'.join(map(str,a.anchor))
    fixture=V2/f'fixtures/beauty/city-scene-r{a.revision}'/name
    output=OUT/f'city-scene-r{a.revision}'/name
    if (fixture.exists() or output.exists()) and not a.resume:raise ValueError('preserved city probe exists')
    if a.resume:
        if (output/'combined').exists():raise ValueError('preserve existing rendered result; use a new revision')
        if (fixture/'augmentation.json').exists():
            prior=json.loads((fixture/'augmentation.json').read_text())
            assert (prior['pool'],prior['size'],prior['uniform_scale_factor'])==(pool,a.size,a.factor)
    fixture.mkdir(parents=True,exist_ok=a.resume);output.mkdir(parents=True,exist_ok=a.resume)
    assets=[city.component(x,pack) for x in catalog['pools'][pool]['components']]
    anchor=a.anchor
    # Freeze a dense buildability witness before arranging any source bodies.
    # The signed shore query is negative on land (opposite optical water data).
    foundation=V2/'fixtures/beauty/city-scene-foundation'/('coastal' if anchor==[3,2] else 'coastal-'+'-'.join(map(str,anchor)))
    foundation.mkdir(parents=True,exist_ok=True)
    gridfile=foundation/'surface.json';gridstep=.04
    if not gridfile.exists():
        grid=[]
        for iy in range(51):
            for ix in range(51):
                wx=anchor[0]+.5-1+ix*gridstep;wy=anchor[1]+.5+1-iy*gridstep
                col=math.floor(wx);row=math.floor(wy);grid.append([col,row,wx-col,1-(wy-row)])
        pointfile=foundation/'points.csv';pointfile.write_text(''.join(','.join(map(str,p))+'\n' for p in grid))
        run([sys.executable,V2/'app/surface_query.py','--fixture',V2/'fixtures/beauty/river-corridor-r3/coastal/fixture.json','--points',pointfile,'--output',gridfile])
    grid=json.loads(gridfile.read_text())['samples']
    def buildable(box):
        values=[]
        for x,y in [(box[0],box[1]),(box[0],box[3]),(box[2],box[1]),(box[2],box[3]),((box[0]+box[2])/2,(box[1]+box[3])/2)]:
            ix=round((x+1)/gridstep);iy=round((y+1)/gridstep)
            if not (0<=ix<=50 and 0<=iy<=50):return False
            s=grid[iy*51+ix];values.append(s['height'])
            if s['base']>=11 or s['shore_distance']>-.05:return False
        return max(values)-min(values)<=2.5
    source_scale=None;ordering=None
    if a.expanded:
        reference=[city.component(x) for x in city.read(city.PACK/'city_catalog.json')['pools'][pool]['components']]
        source_scale=city.layout(reference,0,factor=a.factor)[0]['scale']
        # Keep one assembled neighborhood as a focal compound; alternate block
        # variants are alternatives, not four mandatory adjacent neighborhoods.
        source_report=json.loads((OUT/'city-source-expanded-r1/build.json').read_text())
        selected=next(p for p in source_report['pools'] if p['pool']==pool)['selected']
        blocks={p['asset_id'] for p in selected if '_Block_' in p['entry']}
        standalone=sorted([x for x in assets if x['id'] not in blocks],key=lambda x:(-(x['hi'][2]-x['lo'][2]),x['id']))
        compounds=sorted([x for x in assets if x['id'] in blocks],key=lambda x:(-(x['hi'][2]-x['lo'][2]),x['id']))
        ordering=compounds[:1]+standalone
        while len(ordering)<11:ordering+=standalone
    footprint_limit=[.65,.8,.95][a.size] if a.expanded else None
    layout=city.layout(assets,a.size,factor=a.factor,buildable=buildable,source_scale=source_scale,ordering=ordering,footprint_limit=footprint_limit)
    points=[];instances=[]
    for inst in layout:
        body=inst['asset'];positions=[]
        for mesh,_ in body['parts']:
            for v in mesh['vertices']:
                source=[v['position'][0]-(body['lo'][0]+body['hi'][0])*.5,
                        v['position'][1]-(body['lo'][1]+body['hi'][1])*.5,v['position'][2]-(0 if a.authored_ground else body['lo'][2])]
                positions.append([x*inst['scale'] for x in city.rotate(source,inst['rotation'])])
        bounds=[min(v[0] for v in positions),min(v[1] for v in positions),max(v[0] for v in positions),max(v[1] for v in positions)]
        sample_start=len(points)
        for dx,dy in [(0,0),(bounds[0],bounds[1]),(bounds[0],bounds[3]),(bounds[2],bounds[1]),(bounds[2],bounds[3])]:
            wx=anchor[0]+.5+inst['x']+dx;wy=anchor[1]+.5-inst['y']-dy
            col=math.floor(wx);row=math.floor(wy);points.append([col,row,wx-col,1-(wy-row)])
        instances.append({'asset':body['id'],'slot':inst['slot'],'scale':inst['scale'],'rotation':inst['rotation'],
                          'offset':[inst['x'],inst['y']],'local_bounds':bounds,'sample_start':sample_start})
    pointfile=fixture/'points.csv';pointfile.write_text(''.join(','.join(map(str,p))+'\n' for p in points))
    run([sys.executable,V2/'app/surface_query.py','--fixture',V2/'fixtures/beauty/river-corridor-r3/coastal/fixture.json',
         '--points',pointfile,'--output',fixture/'surface.json'])
    surface=json.loads((fixture/'surface.json').read_text());projection=surface['projection']
    width,height=map(int,[projection['width'],projection['height']]);half=projection['half_width'];half_y=projection['half_height'];vertical=projection['vertical_scale']
    groups=defaultdict(list);inputs={};materials=[]
    for inst,record in zip(layout,instances):
        samples=surface['samples'][record['sample_start']:record['sample_start']+5];site=samples[0]
        record['ground_height_range']=[min(x['height'] for x in samples),max(x['height'] for x in samples)]
        record['minimum_shore_distance']=min(x['shore_distance'] for x in samples)
        if any(x['base']>=11 or x['shore_distance']>-.02 for x in samples):raise ValueError('building footprint reaches water')
        if record['ground_height_range'][1]-record['ground_height_range'][0]>3:raise ValueError('building site needs terrain foundation handling')
        body=inst['asset']
        for mesh,mat in body['parts']:
            if mat['alpha_mode']!='opaque':raise ValueError('unsupported city alpha contract')
            ch=mat['channels'];keys=['base_color','emissive','ambient_occlusion','normal_0']
            textures=tuple(ch.get(k,{}).get('texture','') for k in keys)
            for tex in textures:
                if tex:inputs[tex]=file_hash(ROOT/tex)
            if ch not in materials:materials.append(ch)
            channel_bits=(1 if textures[2] else 0)+(2 if textures[3] else 0)
            vertices=[]
            for v in mesh['vertices']:
                source=[v['position'][0]-(body['lo'][0]+body['hi'][0])*.5,v['position'][1]-(body['lo'][1]+body['hi'][1])*.5,v['position'][2]-(0 if a.authored_ground else body['lo'][2])]
                x,y,z=[q*inst['scale'] for q in city.rotate(source,inst['rotation'])]
                # Use the existing Q7 source projection; publish the conversion
                # into the terrain's authoring-height world coordinate explicitly.
                height_pixels=z*80.9543
                sx=site['screen_x']+(x-y)*half;sy=site['screen_y']+(x+y)*half_y-height_pixels
                depth=site['depth']-(x+y)*half_y/height*.75-height_pixels/vertical*.0012
                world=[site['column']+site['u']+x,site['row']+1-site['v']-y,(site['height']+height_pixels/vertical)/112]
                vertices.append([sx/width*2-1,1-sy/height*2,depth,*v['uv0'],*city.rotate(v['normal'],inst['rotation']),40+channel_bits,*world,1])
            groups[(textures,False)].extend(vertices[i] for i in mesh['topology']['indices'])
            if a.emissive_uv and textures[1]:
                emission_vertices=[]
                for source,v in zip(mesh['vertices'],vertices):
                    emission_vertices.append([*v[:3],*source[f'uv{a.emissive_uv}'],*v[5:8],80,*v[9:]])
                groups[(textures,True)].extend(emission_vertices[i] for i in mesh['topology']['indices'])
    wire=bytearray(struct.pack('<II',0x38514353,len(groups)))
    for (textures,emission_only),verts in sorted(groups.items(),key=lambda x:(x[0][1],x[0][0])):
        for path in textures:
            b=path.encode();wire+=struct.pack('<I',len(b))+b
        wire+=struct.pack('<I',len(verts))
        for v in verts:wire+=struct.pack('<13f',*v)
    (fixture/'city.bin').write_bytes(wire)
    save(fixture/'augmentation.json',{'classification':'source_adaptation; explicit Lab city augmentation',
         'source_biq_sha256':surface['region']['source_sha256'],'anchor_tile':anchor,'pool':pool,'size':a.size,
         'uniform_scale_factor':a.factor,'instances':instances,'textures':inputs,'material_declarations':materials,
         'pack':pack.as_posix(),'expanded_pool':a.expanded,'emissive_gain':a.emissive_gain,
         'emissive_uv':a.emissive_uv,'hdr_glow':a.glow,
         'grounding':'source_z_zero' if a.authored_ground else 'lowest_source_vertex',
         'footprint_half_extent_tiles':footprint_limit,'cross_tile_extent_authorization':'user permits slight city overlap, especially larger cities',
         'projection':projection,'source_z_pixels_per_unit':80.9543,'scene_world_z_per_source_unit':80.9543/(vertical*112),
         'material_channels_enabled':['base_color','emissive']+(['ambient_occlusion'] if a.channels else []),
         'remaining':['source normal/gloss interpretation','full coast/route/vegetation envelopes','capital and wall states','all culture/era/size matrix']})
    base=OUT/'river-corridor-r3/coastal';report=json.loads((base/'report.json').read_text());jobs=json.loads((base/'batch.json').read_text())
    pairs=[(j,r) for j,r in zip(jobs,report['outputs']) if a.all_zooms or r['zoom']==1]
    cache=Cache(V2/'app/.cache');append=executable(V2/'qa/append_city_scene.cpp',cache)
    shadows=executable(V2/'systems/lighting/scene_shadow.cpp',cache)
    for i,(job,row) in enumerate(pairs):
        combined=output/f'city-{i}.packet';shadowed=output/f'combined-{i}.packet'
        run([append,job[0],fixture/'city.bin',combined]);run([shadows,combined,shadowed,row['hour'],base/'report.json'])
        row['packet']=rel(shadowed);job[0]=str(shadowed)
    report['outputs']=[r for _,r in pairs];save(output/'report.json',report);save(output/'batch.json',[j for j,_ in pairs])
    common=fixture/'city.hlsl'
    common.write_text('#define Q3_NATURAL_WATER 1\n#define PSFeature Q8LegacyPSFeature\n'
        '#include "../../river-corridor-r3/coastal/combined.hlsl"\n#undef PSFeature\n'+
        f'#define Q8_CITY_CHANNELS {int(a.channels)}\n#define Q8_CITY_SEPARATE_EMISSION {int(a.emissive_uv>0)}\n#define Q8_CITY_EMISSIVE_GAIN {a.emissive_gain}\n#include "../../../../shaders/objects/city_scene_material.hlsl"\n')
    shader=fixture/'combined.hlsl';shader.write_text(f'#define Q3_OBJECT_REFLECTION 1\n#define Q3_REFLECTION_SIZE float2({width}.0,{height}.0)\n#include "city.hlsl"\n')
    reflected=fixture/'reflection.hlsl';reflected.write_text('#define VSMain Q3OriginalVSMain\n#define VSFeature Q3OriginalVSFeature\n#define PSMain Q3OriginalPSMain\n#define Q8_CITY_FEATURE_ENTRY Q3OriginalPSFeature\n#include "city.hlsl"\n#undef VSMain\n#undef VSFeature\n#undef PSMain\n'+f'#define Q3_REFLECTION_HEIGHT_NDC {4*.82*half/height:.12f}\n#include "../../../../shaders/hydrology/planar_reflection_pass.hlsl"\n')
    run([sys.executable,V2/'qa/replay_shader.py','--report',output/'report.json','--shader',shader,
         '--reflection-shader',reflected,'--output',output/'combined']+
         (['--post-shader',V2/'shaders/common/hdr_glow_tiled.hlsl'] if a.glow else []))

if __name__=='__main__':main()
