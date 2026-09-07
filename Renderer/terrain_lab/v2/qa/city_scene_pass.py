"""Preserved city composition probes on the fixed 100-tile terrain benchmarks.

City instances are explicit Lab augmentation, not captured BIQ city state. Source
parts, UVs and uniform preprojection transforms are retained. The terrain packet
is copied, cities appended, and shared shadows rebuilt before GPU reflections.
"""
import argparse
from collections import defaultdict
import json
import math
import shutil
import statistics
from pathlib import Path
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4];V2=ROOT/'Renderer/terrain_lab/v2';OUT=V2/'audits/beauty/out'
sys.path.insert(0,str(V2/'app'))
import runner
from cache import Cache,file_hash
from packet_store import compact_packet
sys.path.insert(0,str(V2/'systems/objects'))
import presentation as city
from city_generator_layout import select_components
from city_ground_geometry import clip_ground_triangle

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
    parser.add_argument('--region',choices=['coastal','inland','wilderness','freshcanopy'],default='coastal')
    parser.add_argument('--size',type=int,choices=[0,1,2],default=1)
    parser.add_argument('--factor',type=float,default=1)
    parser.add_argument('--footprint-limit',type=float,help='Explicit Lab city half-extent in tiles; bounded to the sampled terrain envelope')
    parser.add_argument('--channels',action='store_true')
    parser.add_argument('--source-addressing',action='store_true',help='Respect normalized repeat/clamp addressing on city surface textures')
    parser.add_argument('--surface-detail',action='store_true',help='Adapt source slope-map detail to the geometric tangent frame')
    parser.add_argument('--compound-ground',type=Path,help='Explicit normalized source ground-part mapping for this Lab comparison')
    parser.add_argument('--expanded',action='store_true')
    parser.add_argument('--authored-ground',action='store_true',help='place source z=0 at ground; retain negative foundation skirts underground')
    parser.add_argument('--emissive-gain',type=float,default=1.45)
    parser.add_argument('--emissive-uv',type=int,choices=[0,1,2],default=0)
    parser.add_argument('--glow',action='store_true')
    parser.add_argument('--weighted-growth',action='store_true',help='Count a whole source neighborhood by footprint rather than as one house')
    parser.add_argument('--graduated-growth',action='store_true',help='Grow from lower standalone buildings toward taller buildings and a late neighborhood block')
    parser.add_argument('--generator-profile',type=Path,help='Use recovered generator parameters while preserving the user-selected single-era appearance')
    parser.add_argument('--historical-era-mix',action='store_true',help='Reproduce the rejected multi-era diagnostic; not the selected city appearance')
    parser.add_argument('--capital',action='store_true',help='Add the explicitly mapped palace to this Lab capital city')
    parser.add_argument('--capital-composition',action='store_true',help='Keep the city compact while discouraging foreground coverage of its palace')
    parser.add_argument('--omit-capital',action='store_true',help='Matched control: retain the reserved palace site but omit its draws')
    parser.add_argument('--anchor',type=int,nargs=2,default=[3,2])
    parser.add_argument('--all-zooms',action='store_true')
    parser.add_argument('--resume',action='store_true',help='retry an input/build failure before any combined render exists')
    a=parser.parse_args()
    terrain_fixture=V2/f'fixtures/beauty/river-corridor-r3/{a.region}/fixture.json'
    if shutil.disk_usage(V2).free<8*1024**3:raise ValueError('capture stopped: preserve at least 8 GiB free disk space')
    if not .5<=a.factor<=2:raise ValueError('bounded uniform scale factor required')
    if a.footprint_limit is not None and not .5<=a.footprint_limit<=1:raise ValueError('city footprint must remain in the sampled terrain envelope')
    if not 0<=a.emissive_gain<=12 or any(v<0 or v>9 for v in a.anchor):raise ValueError('city parameter bounds')
    if a.omit_capital and not a.capital:raise ValueError('capital control requires --capital')
    if a.capital_composition and not a.capital:raise ValueError('capital composition requires --capital')
    if a.graduated_growth and (not a.expanded or a.weighted_growth):raise ValueError('graduated growth requires expanded assets and excludes the alternative weighted recipe')
    if a.generator_profile and (not a.expanded or a.graduated_growth or a.weighted_growth):raise ValueError('generator profile requires expanded assets and excludes alternative growth recipes')
    if a.historical_era_mix and not a.generator_profile:raise ValueError('historical diagnostic requires generator profile')
    capital_mapping=city.read(V2.relative_to(ROOT)/'systems/objects/capital_styles.json') if a.capital else None
    if a.capital and a.pool not in capital_mapping['styles']:raise ValueError('no explicit Lab palace style mapping for this pool')
    pack=Path('Renderer/packs/CityStudyExpanded') if a.expanded else city.PACK
    if a.emissive_uv:
        if not a.expanded:raise ValueError('auxiliary coordinates require the separate expanded study pack')
        pack=Path('Renderer/packs/CityStudyAuxiliaryUV')
    pool='city/pool/'+a.pool;catalog=city.read(pack/'city_catalog.json')
    if pool not in catalog['pools']:raise ValueError('unknown pool')
    name=a.pool.replace('/','-')+f'-s{a.size}'
    if a.capital:name+='-capital'+('-control' if a.omit_capital else '')
    if a.region!='coastal':name+='-'+a.region
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
    foundation=V2/'fixtures/beauty/city-scene-foundation'/(a.region if anchor==[3,2] else a.region+'-'+'-'.join(map(str,anchor)))
    foundation.mkdir(parents=True,exist_ok=True)
    gridfile=foundation/'surface.json';gridstep=.04
    if not gridfile.exists():
        grid=[]
        for iy in range(51):
            for ix in range(51):
                wx=anchor[0]+.5-1+ix*gridstep;wy=anchor[1]+.5+1-iy*gridstep
                col=math.floor(wx);row=math.floor(wy);grid.append([col,row,wx-col,1-(wy-row)])
        pointfile=foundation/'points.csv';pointfile.write_text(''.join(','.join(map(str,p))+'\n' for p in grid))
        run([sys.executable,V2/'app/surface_query.py','--fixture',terrain_fixture,'--points',pointfile,'--output',gridfile])
    grid=json.loads(gridfile.read_text())['samples']
    def buildable(box):
        values=[]
        for x,y in [(box[0],box[1]),(box[0],box[3]),(box[2],box[1]),(box[2],box[3]),((box[0]+box[2])/2,(box[1]+box[3])/2)]:
            ix=round((x+1)/gridstep);iy=round((y+1)/gridstep)
            if not (0<=ix<=50 and 0<=iy<=50):return False
            s=grid[iy*51+ix];values.append(s['height'])
            if s['base']>=11 or s['shore_distance']>-.05:return False
        return max(values)-min(values)<=2.5
    source_scale=None;ordering=None;stage_counts=None;compound_weight=1
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
        if a.graduated_growth:
            ordering=list(reversed(standalone))+compounds[:1]
        while len(ordering)<11:ordering+=standalone
        if a.weighted_growth and compounds:
            area=lambda x:(x['hi'][0]-x['lo'][0])*(x['hi'][1]-x['lo'][1])
            compound_weight=min(4,max(1,math.ceil(area(compounds[0])/statistics.median(area(x) for x in standalone))))
            stage_counts=[1+max(0,budget-compound_weight) for budget in (4,7,11)]
    generator=None
    if a.generator_profile:
        generator=city.read(a.generator_profile)
        style,era=a.pool.split('/')
        source_layers=generator['era_layers'][era]
        layers=source_layers if a.historical_era_mix else [{'era':era,'order_from_center':0,'weight':1.0}]
        layer_assets={x['era']:[city.component(asset,pack) for asset in catalog['pools'][f"city/pool/{style}/{x['era']}"]['components']] for x in layers}
        compound_ids={x['asset_id'] for row in source_report['pools'] for x in row['selected'] if '_Block_' in x['entry']}
        ordering=select_components(layers,layer_assets,compound_ids)
        assets=[asset for values in layer_assets.values() for asset in values]
        source_scale*=generator['model_scale']
    footprint_limit=[.65,.8,.95][a.size] if a.expanded else None
    if a.footprint_limit is not None:footprint_limit=a.footprint_limit
    palace=None;palace_site=None;palace_attempts=[]
    if a.capital:
        mapping=capital_mapping['styles'][a.pool]
        body=city.component(mapping['asset'],Path(mapping.get('pack',capital_mapping['pack'])))
        span=max(body['hi'][j]-body['lo'][j] for j in (0,1))
        palace_scale=mapping['footprint_span_tiles']/span*a.factor/1.5
        # A legal palace site can still strand the last ordinary building.
        # Try bounded alternative civic sites before rejecting the whole city.
        # Ordinary city geometry, scale, dry-land gates and order stay fixed.
        def palace_buildable(box):
            center=[(box[0]+box[2])/2,(box[1]+box[3])/2]
            return buildable(box) and all(math.dist(center,p)>.001 for p in palace_attempts)
        for attempt in range(25):
            palace=city.layout([body],0,buildable=palace_buildable,source_scale=palace_scale,
                               footprint_limit=footprint_limit,stage_counts=[1,1,1])[0]
            palace['slot']='capital'
            palace_attempts.append([palace['x'],palace['y']])
            hx=(body['hi'][0]-body['lo'][0])*palace_scale/2+.024
            hy=(body['hi'][1]-body['lo'][1])*palace_scale/2+.024
            palace_site=[palace['x']-hx,palace['y']-hy,palace['x']+hx,palace['y']+hy]
            def house_buildable(box):
                overlaps=box[0]<palace_site[2] and box[2]>palace_site[0] and box[1]<palace_site[3] and box[3]>palace_site[1]
                return not overlaps and buildable(box)
            try:
                layout=city.layout(assets,a.size,recipe='compact' if a.capital_composition else 'stable',factor=a.factor,buildable=house_buildable,source_scale=source_scale,ordering=ordering,footprint_limit=footprint_limit,stage_counts=stage_counts,focal_instance=palace if a.capital_composition else None,source_ground_zero=a.authored_ground and a.capital_composition)
                break
            except ValueError as error:
                if not str(error).startswith('city footprint cannot fit'):raise
        else:raise ValueError('city footprint cannot fit after 25 bounded palace sites')
    else:
        layout=city.layout(assets,a.size,factor=a.factor,buildable=buildable,source_scale=source_scale,ordering=ordering,footprint_limit=footprint_limit,stage_counts=stage_counts)
    if palace and not a.omit_capital:layout.append(palace)
    ground_parts=city.read(a.compound_ground)['parts'] if a.compound_ground else {}
    ground_draws=[]
    for inst in layout:
        body=inst['asset']
        if body['id'] in ground_parts:
            extra=ground_parts[body['id']]
            projected=[]
            for part in extra:
                mesh=part['mesh'];vertices=[]
                for start in range(0,len(mesh['topology']['indices']),3):
                    corners=[mesh['vertices'][i] for i in mesh['topology']['indices'][start:start+3]]
                    span=max(math.dist(x['position'],y['position'])*inst['scale'] for x in corners for y in corners)
                    steps=max(1,min(24,math.ceil(span/.06)))
                    def vertex(i,j):
                        weights=[1-(i+j)/steps,i/steps,j/steps]
                        return {key:[sum(w*c[key][axis] for w,c in zip(weights,corners)) for axis in range(n)]
                                for key,n in [('position',3),('uv0',2),('normal',3)]}
                    for i in range(steps):
                        for j in range(steps-i):
                            vertices.extend([vertex(i,j),vertex(i+1,j),vertex(i,j+1)])
                            if i+j<steps-1:vertices.extend([vertex(i+1,j),vertex(i+1,j+1),vertex(i,j+1)])
                projected.append(({'vertices':vertices,'topology':{'indices':list(range(len(vertices)))}},part['material']))
            inst['asset']={**body,'parts':body['parts']+projected}
            ground_draws.append({'asset':body['id'],'slot':inst['slot'],'parts':len(extra)})
    points=[];instances=[]
    for inst in layout:
        body=inst['asset'];positions=[]
        for mesh,material in body['parts']:
            if material['alpha_mode']=='blend':continue
            for v in mesh['vertices']:
                source=[v['position'][0]-(body['lo'][0]+body['hi'][0])*.5,
                        v['position'][1]-(body['lo'][1]+body['hi'][1])*.5,v['position'][2]-(0 if a.authored_ground else body['lo'][2])]
                positions.append([x*inst['scale'] for x in city.rotate(source,inst['rotation'])])
        bounds=[min(v[0] for v in positions),min(v[1] for v in positions),max(v[0] for v in positions),max(v[1] for v in positions)]
        sample_start=len(points)
        for dx,dy in [(0,0),(bounds[0],bounds[1]),(bounds[0],bounds[3]),(bounds[2],bounds[1]),(bounds[2],bounds[3])]:
            wx=anchor[0]+.5+inst['x']+dx;wy=anchor[1]+.5-inst['y']-dy
            col=math.floor(wx);row=math.floor(wy);points.append([col,row,wx-col,1-(wy-row)])
        ground_samples={}
        for part_index,(mesh,material) in enumerate(body['parts']):
            if material['alpha_mode']!='blend':continue
            ground_samples[part_index]=len(points)
            for v in mesh['vertices']:
                local=[v['position'][j]-(body['lo'][j]+body['hi'][j])/2 for j in (0,1)]+[0]
                dx,dy,_=[x*inst['scale'] for x in city.rotate(local,inst['rotation'])]
                wx=anchor[0]+.5+inst['x']+dx;wy=anchor[1]+.5-inst['y']-dy
                col=math.floor(wx);row=math.floor(wy);points.append([col,row,wx-col,1-(wy-row)])
        instances.append({'asset':body['id'],'slot':inst['slot'],'scale':inst['scale'],'rotation':inst['rotation'],
                          'offset':[inst['x'],inst['y']],'local_bounds':bounds,'sample_start':sample_start,
                          **({'era_layer':body['era_layer'],'order_from_center':body['order_from_center']} if 'era_layer' in body else {}),
                          **({'ground_samples':ground_samples} if ground_samples else {})})
    pointfile=fixture/'points.csv';pointfile.write_text(''.join(','.join(map(str,p))+'\n' for p in points))
    run([sys.executable,V2/'app/surface_query.py','--fixture',terrain_fixture,
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
        for part_index,(mesh,mat) in enumerate(body['parts']):
            ground=mat['alpha_mode']=='blend'
            if mat['alpha_mode'] not in ('opaque','blend'):raise ValueError('unsupported city alpha contract')
            ch=mat['channels'];keys=['base_color','emissive','ambient_occlusion','normal_0']
            textures=tuple(ch.get(k,{}).get('texture','') for k in keys)
            for tex in textures:
                if tex:inputs[tex]=file_hash(ROOT/tex)
            if ch not in materials:materials.append(ch)
            channel_bits=(1 if textures[2] else 0)+(2 if textures[3] else 0)
            if a.source_addressing and ch['base_color']['address_u']=='repeat':channel_bits+=4
            vertices=[];shore_distances=[]
            for vertex_index,v in enumerate(mesh['vertices']):
                source=[v['position'][0]-(body['lo'][0]+body['hi'][0])*.5,v['position'][1]-(body['lo'][1]+body['hi'][1])*.5,v['position'][2]-(0 if a.authored_ground else body['lo'][2])]
                x,y,z=[q*inst['scale'] for q in city.rotate(source,inst['rotation'])]
                # Use the existing Q7 source projection; publish the conversion
                # into the terrain's authoring-height world coordinate explicitly.
                height_pixels=z*80.9543
                sx=site['screen_x']+(x-y)*half;sy=site['screen_y']+(x+y)*half_y-height_pixels
                depth=site['depth']-(x+y)*half_y/height*.75-height_pixels/vertical*.0012
                world=[site['column']+site['u']+x,site['row']+1-site['v']-y,(site['height']+height_pixels/vertical)/112]
                if ground:
                    sample=surface['samples'][record['ground_samples'][part_index]+vertex_index]
                    if sample['base']>=11 and sample['shore_distance']<=-.02:raise ValueError('city ground land/water query disagrees')
                    shore_distances.append(sample['shore_distance'])
                    sx=sample['screen_x'];sy=sample['screen_y']-.015;depth=sample['depth']-.000001
                    world=[sample['column']+sample['u'],sample['row']+1-sample['v'],sample['height']/112]
                vertices.append([sx/width*2-1,1-sy/height*2,depth,*v['uv0'],*city.rotate(v['normal'],inst['rotation']),60 if ground else 40+channel_bits,*world,1])
            indices=mesh['topology']['indices']
            if ground:
                emitted=[]
                for start in range(0,len(indices),3):
                    triangle=indices[start:start+3]
                    emitted.extend(clip_ground_triangle([vertices[i] for i in triangle],[shore_distances[i] for i in triangle]))
                record.setdefault('ground_clipping',[]).append({'part':part_index,'input_triangles':len(indices)//3,
                    'output_triangles':len(emitted)//3,'wet_input_vertices':sum(d>-.02 for d in shore_distances),
                    'shore_boundary':-.02,'classification':'local linear shore approximation on tessellated source triangles'})
                groups[(textures,False)].extend(emitted)
            else:groups[(textures,False)].extend(vertices[i] for i in indices)
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
         'benchmark_region':a.region,'source_biq_sha256':surface['region']['source_sha256'],'anchor_tile':anchor,'pool':pool,'size':a.size,
         'uniform_scale_factor':a.factor,'instances':instances,'textures':inputs,'material_declarations':materials,
         'pack':pack.as_posix(),'expanded_pool':a.expanded,'emissive_gain':a.emissive_gain,
         'emissive_uv':a.emissive_uv,'hdr_glow':a.glow,
         'source_addressing':a.source_addressing,'surface_detail':a.surface_detail,
         'compound_ground':{'mapping':str(a.compound_ground) if a.compound_ground else None,
                            'mapping_sha256':file_hash(ROOT/a.compound_ground) if a.compound_ground else None,
                            'draws':ground_draws,'classification':'explicit source-material/triangle probe; descriptor state and height response unproven'},
         'weighted_growth':a.weighted_growth,'compound_house_equivalents':compound_weight,'stage_component_counts':stage_counts,
         'graduated_growth':a.graduated_growth,
         'generator_profile':{'path':str(a.generator_profile) if a.generator_profile else None,
                              'sha256':file_hash(ROOT/a.generator_profile) if a.generator_profile else None,
                              'era_policy':'rejected_historical_mix_diagnostic' if a.historical_era_mix else 'single_current_era_user_preference',
                              'used':(['era weights','center ordering','uniform model scale'] if a.historical_era_mix else ['uniform model scale']) if generator else [],
                              'adapter':'stable weighted choices; bounded ring preference; source engine algorithm not recovered'},
         'capital':{'requested':a.capital,'drawn':bool(palace and not a.omit_capital),'reserved_site':palace_site,'placement_attempts':palace_attempts,
                    'composition':'compact_with_focal_visibility_preference' if a.capital_composition else 'first_legal_layout',
                    'mapping':capital_mapping['styles'][a.pool] if a.capital else None,
                    'authority':'explicit Lab fixture only; production must use captured Civ III capital status',
                    'native_capital_indicator':'retained'},
         'grounding':'source_z_zero' if a.authored_ground else 'lowest_source_vertex',
         'footprint_half_extent_tiles':footprint_limit,'cross_tile_extent_authorization':'user permits slight city overlap, especially larger cities',
         'projection':projection,'source_z_pixels_per_unit':80.9543,'scene_world_z_per_source_unit':80.9543/(vertical*112),
         'material_channels_enabled':['base_color','emissive']+(['ambient_occlusion'] if a.channels else [])+(['normal_0_slope_adaptation'] if a.surface_detail else []),
         'remaining':['source normal/gloss interpretation','full coast/route/vegetation envelopes','capital and wall states','all culture/era/size matrix']})
    base=OUT/f'river-corridor-r3/{a.region}';report=json.loads((base/'report.json').read_text());jobs=json.loads((base/'batch.json').read_text())
    pairs=[(j,r) for j,r in zip(jobs,report['outputs']) if a.all_zooms or r['zoom']==1]
    cache=Cache(V2/'app/.cache');append=executable(V2/'qa/append_city_scene.cpp',cache)
    shadows=executable(V2/'systems/lighting/scene_shadow.cpp',cache)
    for i,(job,row) in enumerate(pairs):
        combined=output/f'city-{i}.packet';shadowed=output/f'combined-{i}.packet'
        run([append,job[0],fixture/'city.bin',combined]);run([shadows,combined,shadowed,row['hour'],base/'report.json'])
        # The pre-shadow copy is disposable; the final packet is replay input.
        combined.unlink()
        # Share existing immutable terrain mips/buffers instead of retaining a
        # full terrain copy for every city, hour and zoom.
        compact_packet(shadowed,V2/'app/.cache/content')
        row['packet']=rel(shadowed);job[0]=str(shadowed)
    report['outputs']=[r for _,r in pairs];save(output/'report.json',report);save(output/'batch.json',[j for j,_ in pairs])
    common=fixture/'city.hlsl'
    common.write_text('#define Q3_NATURAL_WATER 1\n#define PSFeature Q8LegacyPSFeature\n'
        f'#include "../../river-corridor-r3/{a.region}/combined.hlsl"\n#undef PSFeature\n'+
        f'#define Q8_CITY_CHANNELS {int(a.channels)}\n#define Q8_CITY_SURFACE_DETAIL {int(a.surface_detail)}\n#define Q8_CITY_WORLD_Z_TO_SOURCE {vertical*112/80.9543:.12f}\n#define Q8_CITY_SEPARATE_EMISSION {int(a.emissive_uv>0)}\n#define Q8_CITY_EMISSIVE_GAIN {a.emissive_gain}\n#include "../../../../shaders/objects/city_scene_material.hlsl"\n')
    shader=fixture/'combined.hlsl';shader.write_text(f'#define Q3_OBJECT_REFLECTION 1\n#define Q3_REFLECTION_SIZE float2({width}.0,{height}.0)\n#include "city.hlsl"\n')
    reflected=fixture/'reflection.hlsl';reflected.write_text('#define VSMain Q3OriginalVSMain\n#define VSFeature Q3OriginalVSFeature\n#define PSMain Q3OriginalPSMain\n#define Q8_CITY_FEATURE_ENTRY Q3OriginalPSFeature\n#include "city.hlsl"\n#undef VSMain\n#undef VSFeature\n#undef PSMain\n'+f'#define Q3_REFLECTION_HEIGHT_NDC {4*.82*half/height:.12f}\n#include "../../../../shaders/hydrology/planar_reflection_pass.hlsl"\n')
    run([sys.executable,V2/'qa/replay_shader.py','--report',output/'report.json','--shader',shader,
         '--reflection-shader',reflected,'--output',output/'combined']+
         (['--post-shader',V2/'shaders/common/hdr_glow_tiled.hlsl'] if a.glow else []))

if __name__=='__main__':main()
