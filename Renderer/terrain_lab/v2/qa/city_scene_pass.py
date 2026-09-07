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
from city_ground_geometry import clip_ground_to_land_cells
from mesh_fingerprint import geometry_digest,material_digest
from city_growth_layout import solve as solve_growth, bounds as growth_bounds
from city_exclusion import Exclusion

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
    parser.add_argument('--region',choices=['coastal','inland','wilderness','freshcanopy','freshshadow','freshwater'],default='coastal')
    parser.add_argument('--size',type=int,choices=[0,1,2],default=1)
    parser.add_argument('--factor',type=float,default=1)
    parser.add_argument('--footprint-limit',type=float,help='Explicit Lab city half-extent in tiles; bounded to the sampled terrain envelope')
    parser.add_argument('--channels',action='store_true',help='Legacy unselected UV0 AO diagnostic; use --ao-uv 1 for the tested medieval source atlas')
    parser.add_argument('--ao-uv',type=int,choices=[0,1,2],help='Explicit source AO coordinate study with a separate interpolated float2')
    parser.add_argument('--ao-strength',type=float,default=1,help='Bounded source AO contribution; zero is the matched disabled control')
    parser.add_argument('--source-addressing',action='store_true',help='Respect normalized repeat/clamp addressing on city surface textures')
    parser.add_argument('--surface-detail',action='store_true',help='Adapt source slope-map detail to the geometric tangent frame')
    parser.add_argument('--source-normals',type=Path,help='Generic normalized per-mesh normal override, with geometry fingerprint checks')
    parser.add_argument('--source-surface',choices=['off','normal','lit'],default='off',help='Opt-in source frame and normal texture; lit also evaluates the cooked dual-lobe roughness texture')
    parser.add_argument('--extra-materials',type=Path,help='Generic fingerprinted metalness and coverage-mask overlay')
    parser.add_argument('--opacity-cutouts',action='store_true',help='Apply coverage masks consistently to city bodies, emission, reflections and shadows')
    parser.add_argument('--metalness',action='store_true',help='Direct-light metalness diagnostic; full environment reflection remains required')
    parser.add_argument('--vegetation-clearance',type=float,help='Bounded tile-space margin around unchanged forest/jungle cells; rejects colliding city layouts')
    parser.add_argument('--layout-retries',type=int,default=0,help='Bounded alternative sites for the current growth stage; earlier-stage placements stay fixed')
    parser.add_argument('--growth-search-nodes',type=int,default=0,help='Bounded constrained-first search with quarter-turn choices for new city bodies')
    parser.add_argument('--growth-grid-step',type=float,choices=[.035,.07],default=.07,help='Placement grid spacing; terrain clearance samples stay fixed')
    parser.add_argument('--growth-plan-size',type=int,choices=[0,1,2],help='Plan this growth stage first, then render the requested prefix without moving earlier buildings')
    parser.add_argument('--growth-stage-extents',type=float,nargs=3,help='Optional small/medium/large footprint envelopes while solving the complete growth plan')
    parser.add_argument('--growth-neighbor-gap',type=float,help='Prefer a compact envelope and require the final house neighborhood to connect within this tile-space gap')
    parser.add_argument('--river-clearance-pixels',type=float,help='Exclude the rendered river distance-field corridor from city footprints; 12.4 conservatively includes the current banks')
    parser.add_argument('--stage-component-counts',type=int,nargs=3,help='Explicit small/medium/large body budgets for constrained growth; supports individual-house palettes')
    parser.add_argument('--preserve-layout',type=Path,help='Existing Lab augmentation whose growth-prefix placements must remain exact')
    parser.add_argument('--layout-only',action='store_true',help='Write the planned placements without compiling or rendering a scene')
    parser.add_argument('--shadow-frame-report',type=Path,help='Explicit prior city report supplying the fixed light grid for matched layout comparisons')
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
    if a.source_surface!='off' and (not a.source_normals or a.ao_uv is None):raise ValueError('source surface requires an explicit source-frame mapping and auxiliary AO layout')
    if a.extra_materials and (a.source_surface=='off' or a.emissive_uv!=2):raise ValueError('extra materials require source frame layout and UV2 emission')
    if (a.opacity_cutouts or a.metalness) and not a.extra_materials:raise ValueError('extra material flags require a mapping')
    if a.vegetation_clearance is not None and not 0<=a.vegetation_clearance<=.3:raise ValueError('vegetation clearance must be in [0,.3] tiles')
    if not 0<=a.layout_retries<=64 or (a.layout_retries and a.capital):raise ValueError('ordinary city layout retries must be in [0,64]; capitals use their existing dedicated search')
    if not 0<=a.growth_search_nodes<=200000:raise ValueError('growth search node bound')
    if a.growth_search_nodes and (a.layout_retries or not a.expanded):raise ValueError('growth search requires expanded city assets and excludes legacy retry mode')
    if a.preserve_layout and not a.growth_search_nodes:raise ValueError('preserved layout requires growth search')
    if a.growth_plan_size is not None and (not a.growth_search_nodes or a.growth_plan_size<a.size):raise ValueError('growth plan must cover the rendered size')
    if a.growth_stage_extents and (not a.growth_search_nodes or any(not .3<=v<=1 for v in a.growth_stage_extents) or sorted(a.growth_stage_extents)!=a.growth_stage_extents):raise ValueError('growth stage extents require ordered bounded sizes and the constrained search')
    if a.growth_neighbor_gap is not None and (not a.growth_search_nodes or not .02<=a.growth_neighbor_gap<=.3):raise ValueError('neighbor gap requires constrained search and a gap in [.02,.3]')
    if a.river_clearance_pixels is not None and (not math.isfinite(a.river_clearance_pixels) or not 6.2<=a.river_clearance_pixels<=16):raise ValueError('river exclusion must include the full water width and remain in [6.2,16] pixels')
    if a.stage_component_counts and (not a.growth_search_nodes or a.weighted_growth or any(not 1<=v<=32 for v in a.stage_component_counts) or any(x>=y for x,y in zip(a.stage_component_counts,a.stage_component_counts[1:]))):raise ValueError('explicit body budgets require constrained growth and three increasing counts up to 32')
    terrain_series={'freshshadow':'shadow-receiver-r1','freshwater':'water-natural-foundation'}.get(a.region,'river-corridor-r3')
    terrain_fixture=V2/f'fixtures/beauty/{terrain_series}/{a.region}/fixture.json'
    if shutil.disk_usage(V2).free<8*1024**3:raise ValueError('capture stopped: preserve at least 8 GiB free disk space')
    if not .5<=a.factor<=2:raise ValueError('bounded uniform scale factor required')
    if a.footprint_limit is not None and not .5<=a.footprint_limit<=1:raise ValueError('city footprint must remain in the sampled terrain envelope')
    if not 0<=a.emissive_gain<=12 or any(v<0 or v>9 for v in a.anchor):raise ValueError('city parameter bounds')
    if a.omit_capital and not a.capital:raise ValueError('capital control requires --capital')
    if a.capital_composition and not a.capital:raise ValueError('capital composition requires --capital')
    if not 0<=a.ao_strength<=1 or (a.ao_strength!=1 and a.ao_uv is None):raise ValueError('AO strength requires explicit auxiliary AO and must be in [0,1]')
    if a.graduated_growth and (not a.expanded or a.weighted_growth):raise ValueError('graduated growth requires expanded assets and excludes the alternative weighted recipe')
    if a.generator_profile and (not a.expanded or a.graduated_growth or a.weighted_growth):raise ValueError('generator profile requires expanded assets and excludes alternative growth recipes')
    if a.historical_era_mix and not a.generator_profile:raise ValueError('historical diagnostic requires generator profile')
    capital_mapping=city.read(V2.relative_to(ROOT)/'systems/objects/capital_styles.json') if a.capital else None
    if a.capital and a.pool not in capital_mapping['styles']:raise ValueError('no explicit Lab palace style mapping for this pool')
    pack=Path('Renderer/packs/CityStudyExpanded') if a.expanded else city.PACK
    if a.emissive_uv or a.ao_uv is not None:
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
    exclusion=None;river_evidence=None
    if a.river_clearance_pixels is not None:
        terrain_report=OUT/f'{terrain_series}/{a.region}/report.json'
        terrain_row=next(r for r in json.loads(terrain_report.read_text())['outputs'] if r['zoom']==1)
        packet=ROOT/terrain_row['packet']
        cache=Cache(V2/'app/.cache')
        exporter=executable(V2/'qa/river_city_exclusion.cpp',cache)
        exclusion_path=fixture/'river-exclusion.json'
        run([exporter,packet,*anchor,a.river_clearance_pixels,exclusion_path])
        river_evidence=json.loads(exclusion_path.read_text())
        river_evidence.update(anchor_tile=anchor,terrain_packet=rel(packet),terrain_packet_sha256=file_hash(packet),
            classification='Lab rendered distance-field adapter; generic convex city-local exclusion polygons')
        save(exclusion_path,river_evidence)
        exclusion=Exclusion(river_evidence['polygons'])
        river_evidence={k:v for k,v in river_evidence.items() if k!='polygons'}
        river_evidence.update(path=rel(exclusion_path),sha256=file_hash(exclusion_path))
    def buildable_uncached(box):
        if exclusion is not None and exclusion.blocks(box):return False
        if a.vegetation_clearance is not None:
            margin=a.vegetation_clearance
            x0=math.floor((box[0]-margin+1)/gridstep);x1=math.ceil((box[2]+margin+1)/gridstep)
            y0=math.floor((box[1]-margin+1)/gridstep);y1=math.ceil((box[3]+margin+1)/gridstep)
            if min(x0,y0)<0 or max(x1,y1)>50:return False
            if any(grid[y*51+x]['real'] in (7,8) for y in range(y0,y1+1) for x in range(x0,x1+1)):return False
        values=[]
        for x,y in [(box[0],box[1]),(box[0],box[3]),(box[2],box[1]),(box[2],box[3]),((box[0]+box[2])/2,(box[1]+box[3])/2)]:
            ix=round((x+1)/gridstep);iy=round((y+1)/gridstep)
            if not (0<=ix<=50 and 0<=iy<=50):return False
            s=grid[iy*51+ix];values.append(s['height'])
            if s['base']>=11 or s['shore_distance']>-.05:return False
        return max(values)-min(values)<=2.5
    buildability_cache={}
    def buildable(box):
        key=tuple(box)
        if key not in buildability_cache:buildability_cache[key]=buildable_uncached(box)
        return buildability_cache[key]
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
    if a.stage_component_counts:stage_counts=list(a.stage_component_counts)
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
    palace=None;palace_site=None;palace_attempts=[];core_center=[0,0]
    layout_attempts=[]
    if a.capital:
        mapping=capital_mapping['styles'][a.pool]
        core=mapping.get('center_half_extent_tiles')
        if core is not None and (not math.isfinite(core) or not 0<core<=footprint_limit):raise ValueError('invalid palace center envelope')
        reference=mapping.get('center_reference','tile_anchor')
        if reference not in ('tile_anchor','dry_land_centroid'):raise ValueError('unknown palace center reference')
        if reference=='dry_land_centroid':
            dry=[[-1+(i%51)*gridstep,-1+(i//51)*gridstep] for i,s in enumerate(grid)
                 if s['base']<11 and s['shore_distance']<=-.05 and
                 abs(-1+(i%51)*gridstep)<=footprint_limit and abs(-1+(i//51)*gridstep)<=footprint_limit]
            if not dry:raise ValueError('capital footprint has no dry land')
            core_center=[sum(p[j] for p in dry)/len(dry) for j in range(2)]
        body=city.component(mapping['asset'],Path(mapping.get('pack',capital_mapping['pack'])))
        span=max(body['hi'][j]-body['lo'][j] for j in (0,1))
        palace_scale=mapping['footprint_span_tiles']/span*a.factor/1.5
        preserved_capital=None;preserved_houses=[]
        if a.preserve_layout:
            previous=city.read(a.preserve_layout)
            for key,value in [('pool',pool),('anchor_tile',anchor),('benchmark_region',a.region),('uniform_scale_factor',a.factor)]:
                if previous[key]!=value:raise ValueError('preserved capital layout mismatch: '+key)
            preserved_capital=next(i for i in previous['instances'] if i['slot']=='capital')
            preserved_houses=[i for i in previous['instances'] if i['slot']!='capital']
            if preserved_capital['asset']!=body['id'] or abs(preserved_capital['scale']-palace_scale)>1e-10:raise ValueError('preserved palace changed')
        remaining_nodes=a.growth_search_nodes
        # A legal palace site can still strand the last ordinary building.
        # Try bounded alternative civic sites before rejecting the whole city.
        # Ordinary city geometry, scale, dry-land gates and order stay fixed.
        def palace_buildable(box):
            center=[(box[0]+box[2])/2,(box[1]+box[3])/2]
            if a.capital_composition and core is not None and any(abs(v-core_center[j])>core+1e-10 for j,v in enumerate(center)):return False
            return buildable(box) and all(math.dist(center,p)>(.21 if a.growth_search_nodes else .001) for p in palace_attempts)
        for attempt in range(25):
            if preserved_capital:
                palace=dict(asset=body,slot='capital',x=preserved_capital['offset'][0],y=preserved_capital['offset'][1],rotation=preserved_capital['rotation'],scale=palace_scale)
                if core is not None and any(abs(v-core_center[j])>core+1e-10 for j,v in enumerate(preserved_capital['offset'])):raise ValueError('preserved palace violates the center envelope')
            else:
                try:
                    palace=city.layout([body],0,buildable=palace_buildable,source_scale=palace_scale,
                                       footprint_limit=footprint_limit,stage_counts=[1,1,1])[0]
                except ValueError as error:
                    if not a.growth_search_nodes or not str(error).startswith('city footprint cannot fit'):raise
                    save(fixture/'capital-growth-search.json',{'status':'no_remaining_separated_palace_sites','total_node_limit':a.growth_search_nodes,
                         'nodes_used':a.growth_search_nodes-remaining_nodes,'site_minimum_separation':.21,'attempts':layout_attempts})
                    raise ValueError('capital growth has no remaining separated palace sites') from error
            palace['slot']='capital'
            palace_attempts.append([palace['x'],palace['y']])
            local=growth_bounds(body,palace['rotation'],palace_scale)
            if preserved_capital and any(abs(v-w)>1e-8 for v,w in zip(local,preserved_capital['local_bounds'])):raise ValueError('preserved palace geometry changed')
            palace_box=[v+(palace['x'],palace['y'])[i%2] for i,v in enumerate(local)]
            palace_site=[v+(-.024 if i<2 else .024) for i,v in enumerate(palace_box)]
            if any(abs(v)>footprint_limit for v in palace_site) or not buildable(palace_site):
                if preserved_capital:raise ValueError('preserved palace violates terrain clearance')
                continue
            def house_buildable(box):
                overlaps=box[0]<palace_site[2] and box[2]>palace_site[0] and box[1]<palace_site[3] and box[3]>palace_site[1]
                return not overlaps and buildable(box)
            if a.growth_search_nodes:
                plan_size=a.size if a.growth_plan_size is None else a.growth_plan_size
                counts=stage_counts or [4,7,11]
                limits=[a.growth_stage_extents[next(stage for stage,n in enumerate(counts) if i<n)]
                        for i in range(counts[plan_size])] if a.growth_stage_extents else None
                # Source-space projection is a ranking approximation only; the
                # actual grounded facade must still pass combined visual review.
                projected={}
                def screen_box(asset,rotation,scale):
                    key=(asset['id'],rotation,scale)
                    if key not in projected:
                        points=[]
                        for mesh,_ in asset['parts']:
                            for v in mesh['vertices']:
                                point=[v['position'][0]-(asset['lo'][0]+asset['hi'][0])*.5,
                                       v['position'][1]-(asset['lo'][1]+asset['hi'][1])*.5,
                                       v['position'][2]-(0 if a.authored_ground else asset['lo'][2])]
                                points.append(city.project([q*scale for q in city.rotate(point,rotation)]))
                        projected[key]=[min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)]
                    return projected[key]
                pbox=screen_box(body,palace['rotation'],palace_scale);dx,dy=city.project([palace['x'],palace['y'],0])
                pbox=[v+(dx,dy)[i%2] for i,v in enumerate(pbox)]
                def focal_cost(asset,rotation,x,y,scale):
                    if x+y<=palace['x']+palace['y']:return 0
                    box=screen_box(asset,rotation,scale);dx,dy=city.project([x,y,0]);box=[v+(dx,dy)[i%2] for i,v in enumerate(box)]
                    return 2*max(0,min(box[2],pbox[2])-max(box[0],pbox[0]))*max(0,min(box[3],pbox[3])-max(box[1],pbox[1]))/((pbox[2]-pbox[0])*(pbox[3]-pbox[1]))
                # Reserve budget for distinct palace sites instead of spending
                # it all around the first otherwise legal central landmark.
                site_budget=remaining_nodes if preserved_capital else min(2000,remaining_nodes)
                layout,search=solve_growth(ordering,counts[plan_size],source_scale,footprint_limit,house_buildable,
                     preserved_houses,site_budget,a.growth_grid_step,limits,a.growth_neighbor_gap,
                     counts[:plan_size+1] if a.growth_neighbor_gap is not None else (),[palace_box],focal_cost if a.capital_composition else None,
                     staged_connection=a.growth_neighbor_gap is not None)
                remaining_nodes-=search['nodes'];search.update(palace_attempt=attempt,palace_site=palace_site)
                layout_attempts.append(search);save(fixture/'growth-search.json',search)
                save(fixture/'capital-growth-search.json',{'total_node_limit':a.growth_search_nodes,'nodes_used':a.growth_search_nodes-remaining_nodes,'site_minimum_separation':.21,'attempts':layout_attempts})
                if layout is not None:
                    search['planned_size']=plan_size
                    search['planned_instances']=[{**{k:v for k,v in i.items() if k!='asset'},'asset':i['asset']['id']} for i in layout]
                    layout=layout[:counts[a.size]];break
                if not remaining_nodes or preserved_capital:raise ValueError('capital growth search '+search['status'])
                continue
            try:
                layout=city.layout(assets,a.size,recipe='compact' if a.capital_composition else 'stable',factor=a.factor,buildable=house_buildable,source_scale=source_scale,ordering=ordering,footprint_limit=footprint_limit,stage_counts=stage_counts,focal_instance=palace if a.capital_composition else None,source_ground_zero=a.authored_ground and a.capital_composition)
                break
            except ValueError as error:
                if not str(error).startswith('city footprint cannot fit'):raise
        else:raise ValueError('city footprint cannot fit after 25 bounded palace sites')
    elif a.growth_search_nodes:
        preserved=[]
        if a.preserve_layout:
            previous=city.read(a.preserve_layout)
            for key,value in [('pool',pool),('anchor_tile',anchor),('benchmark_region',a.region),('uniform_scale_factor',a.factor)]:
                if previous[key]!=value:raise ValueError('preserved city layout mismatch: '+key)
            preserved=previous['instances']
        plan_size=a.size if a.growth_plan_size is None else a.growth_plan_size
        counts=stage_counts or [4,7,11]
        limits=[a.growth_stage_extents[next(stage for stage,n in enumerate(counts) if i<n)]
                for i in range(counts[plan_size])] if a.growth_stage_extents else None
        layout,search=solve_growth(ordering,counts[plan_size],source_scale,
                                   footprint_limit,buildable,preserved,a.growth_search_nodes,a.growth_grid_step,limits,a.growth_neighbor_gap,
                                   counts[:plan_size+1] if a.growth_neighbor_gap is not None else ())
        layout_attempts.append(search);save(fixture/'growth-search.json',search)
        if layout is None:raise ValueError('city growth search '+search['status'])
        search['planned_size']=plan_size
        search['planned_instances']=[{**{k:v for k,v in i.items() if k!='asset'},'asset':i['asset']['id']} for i in layout]
        layout=layout[:(stage_counts or [4,7,11])[a.size]]
    else:
        counts=stage_counts or [4,7,11];prefix=counts[a.size-1] if a.size else 0
        plans=[{}]+[{slot:rank} for rank in range(1,65) for slot in range(counts[a.size]-2,prefix-1,-1)][:a.layout_retries]
        for ranks in plans:
            try:
                layout=city.layout(assets,a.size,factor=a.factor,buildable=buildable,source_scale=source_scale,ordering=ordering,footprint_limit=footprint_limit,stage_counts=stage_counts,candidate_ranks=ranks)
                layout_attempts.append({'candidate_ranks':ranks,'status':'fit'});break
            except ValueError as error:
                if not str(error).startswith('city footprint cannot fit'):raise
                layout_attempts.append({'candidate_ranks':ranks,'status':'no_fit','reason':str(error)})
                save(fixture/'layout-attempts.json',{'preserved_prefix_count':prefix,'attempts':layout_attempts})
        else:raise ValueError(f'city footprint cannot fit after {len(plans)} bounded attempts; prior growth stage preserved')
    if palace and not a.omit_capital:layout.append(palace)
    if a.layout_only:
        save(fixture/'layout-plan.json',{'classification':'Unrendered Lab layout; visual acceptance pending',
             'pool':pool,'size':a.size,'anchor_tile':anchor,'benchmark_region':a.region,
             'river_exclusion':river_evidence,
             'instances':[{**{k:v for k,v in i.items() if k!='asset'},'asset':i['asset']['id']} for i in layout],
             'search':layout_attempts})
        print(rel(fixture/'layout-plan.json'));return
    normal_overrides=city.read(a.source_normals)['meshes'] if a.source_normals else {}
    normal_meshes={};normal_unmapped=set()
    for inst in layout:
        body=inst['asset'];parts=[]
        for mesh,material in body['parts']:
            key=mesh.get('asset_id')
            if key in normal_overrides:
                if key not in normal_meshes:
                    override=normal_overrides[key]
                    if geometry_digest(mesh)!=override['geometry_digest']:raise ValueError('normal override geometry changed')
                    normals=override['normals']
                    if len(normals)!=len(mesh['vertices']) or any(len(n)!=3 or not all(math.isfinite(x) for x in n) or abs(sum(x*x for x in n)-1)>1e-5 for n in normals):
                        raise ValueError('invalid normalized source normal data')
                    normal_meshes[key]={**mesh,'vertices':[{**v,'normal':n} for v,n in zip(mesh['vertices'],normals)]}
                    if a.source_surface!='off':
                        for field,attribute in (('tangents','tangent'),('bitangents','bitangent')):
                            values=override[field]
                            if len(values)!=len(normals) or any(len(n)!=3 or not all(math.isfinite(x) for x in n) or abs(sum(x*x for x in n)-1)>1e-5 for n in values):raise ValueError('invalid source frame')
                            for vertex,n in zip(normal_meshes[key]['vertices'],values):vertex[attribute]=n
                mesh=normal_meshes[key]
            elif a.source_normals:normal_unmapped.add(key)
            parts.append((mesh,material))
        if a.source_normals:inst['asset']={**body,'parts':parts}
    if a.source_normals and not normal_meshes:raise ValueError('normal override did not match any city mesh')
    if a.source_surface!='off' and normal_unmapped:raise ValueError('source surface requires all body frames')
    material_applied=[]
    if a.extra_materials:
        overlay=city.read(a.extra_materials)
        if overlay['schema']!='c3x.lab.material_overlay.v1':raise ValueError('unknown material overlay')
        for inst in layout:
            body=inst['asset'];parts=[]
            for mesh,material in body['parts']:
                key=body['id']+':'+material['name'];record=overlay['materials'][key]
                if material_digest(material)!=record['material_digest']:raise ValueError('material overlay no longer matches source material')
                if any(k not in ('metalness','opacity') for k in record['channels']):raise ValueError('unsupported extra material channel')
                parts.append((mesh,{**material,'channels':{**material['channels'],**record['channels']}}))
                material_applied.append(key)
            inst['asset']={**body,'parts':parts}
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
    cell_bases={(int(s['column']),int(s['row'])):s['base'] for s in grid+surface['samples']}
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
            if a.source_surface!='off':keys.append('gloss')
            if a.extra_materials:keys.extend(['metalness','opacity'])
            textures=tuple(ch.get(k,{}).get('texture','') for k in keys)
            for tex in textures:
                if tex:inputs[tex]=file_hash(ROOT/tex)
            if ch not in materials:materials.append(ch)
            channel_bits=(1 if textures[2] else 0)+(2 if textures[3] else 0)
            if a.source_addressing and ch['base_color']['address_u']=='repeat':channel_bits+=4
            if a.source_surface!='off' and textures[4]:channel_bits+=8
            if a.extra_materials:
                if a.metalness and textures[5]:channel_bits+=16
                if a.opacity_cutouts and textures[6]:channel_bits+=32
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
                    shore_distances.append(sample['shore_distance'])
                    sx=sample['screen_x'];sy=sample['screen_y']-.015;depth=sample['depth']-.000001
                    world=[sample['column']+sample['u'],sample['row']+1-sample['v'],sample['height']/112]
                vertices.append([sx/width*2-1,1-sy/height*2,depth,*v['uv0'],*city.rotate(v['normal'],inst['rotation']),60 if ground else (100 if a.extra_materials else 40)+channel_bits,*world,1])
                if a.ao_uv is not None:vertices[-1].extend(v.get(f'uv{a.ao_uv}',[0,0]))
                if a.source_surface!='off':
                    vertices[-1].extend(city.rotate(v.get('tangent',[1,0,0]),inst['rotation']))
                    vertices[-1].extend(city.rotate(v.get('bitangent',[0,1,0]),inst['rotation']))
                if a.extra_materials:vertices[-1].extend(v.get('uv2',[0,0]))
            indices=mesh['topology']['indices']
            if ground:
                emitted=[]
                for start in range(0,len(indices),3):
                    triangle=indices[start:start+3]
                    emitted.extend(clip_ground_to_land_cells([vertices[i] for i in triangle],[shore_distances[i] for i in triangle],cell_bases))
                record.setdefault('ground_clipping',[]).append({'part':part_index,'input_triangles':len(indices)//3,
                    'output_triangles':len(emitted)//3,'wet_input_vertices':sum(d>-.02 for d in shore_distances),
                    'shore_boundary':-.02,'classification':'exact dry tile-cell clipping plus local linear shore approximation on tessellated source triangles'})
                groups[(textures,False)].extend(emitted)
            else:groups[(textures,False)].extend(vertices[i] for i in indices)
            if a.emissive_uv and textures[1]:
                emission_vertices=[]
                for source,v in zip(mesh['vertices'],vertices):
                    emission_vertices.append([*v[:3],*(source['uv0'] if a.extra_materials else source[f'uv{a.emissive_uv}']),*v[5:8],200+channel_bits if a.extra_materials else 80,*v[9:]])
                groups[(textures,True)].extend(emission_vertices[i] for i in mesh['topology']['indices'])
    wire=bytearray(struct.pack('<II',0x3B514353 if a.extra_materials else 0x3A514353 if a.source_surface!='off' else 0x39514353 if a.ao_uv is not None else 0x38514353,len(groups)))
    for (textures,emission_only),verts in sorted(groups.items(),key=lambda x:(x[0][1],x[0][0])):
        for path in textures:
            b=path.encode();wire+=struct.pack('<I',len(b))+b
        wire+=struct.pack('<I',len(verts))
        for v in verts:wire+=struct.pack('<23f' if a.extra_materials else '<21f' if a.source_surface!='off' else '<15f' if a.ao_uv is not None else '<13f',*v)
    (fixture/'city.bin').write_bytes(wire)
    save(fixture/'augmentation.json',{'classification':'source_adaptation; explicit Lab city augmentation',
         'benchmark_region':a.region,'source_biq_sha256':surface['region']['source_sha256'],'anchor_tile':anchor,'pool':pool,'size':a.size,
         'uniform_scale_factor':a.factor,'instances':instances,'textures':inputs,'material_declarations':materials,
         'pack':pack.as_posix(),'expanded_pool':a.expanded,'emissive_gain':a.emissive_gain,'source_surface':a.source_surface,
         'emissive_uv':a.emissive_uv,'hdr_glow':a.glow,
         'source_addressing':a.source_addressing,'surface_detail':a.surface_detail,'ao_uv':a.ao_uv,'ao_strength':a.ao_strength,
         'extra_materials':{'mapping':str(a.extra_materials) if a.extra_materials else None,'sha256':file_hash(ROOT/a.extra_materials) if a.extra_materials else None,
                            'applied':sorted(set(material_applied)),'opacity_cutouts':a.opacity_cutouts,'metalness':a.metalness},
         'vegetation_clearance':a.vegetation_clearance,
         'river_exclusion':river_evidence,
         'preserved_layout':{'path':a.preserve_layout.as_posix(),'sha256':file_hash(ROOT/a.preserve_layout)} if a.preserve_layout else None,
         'shadow_frame_report':{'path':a.shadow_frame_report.as_posix(),'sha256':file_hash(ROOT/a.shadow_frame_report)} if a.shadow_frame_report else None,
         'layout_attempts':layout_attempts,
         'source_normals':{'mapping':str(a.source_normals) if a.source_normals else None,
                           'sha256':file_hash(ROOT/a.source_normals) if a.source_normals else None,
                           'applied':sorted(normal_meshes),'unmapped':sorted(normal_unmapped)},
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
                    'center_offset':core_center,
                    'composition':'compact_with_focal_visibility_preference' if a.capital_composition else 'first_legal_layout',
                    'mapping':capital_mapping['styles'][a.pool] if a.capital else None,
                    'authority':'explicit Lab fixture only; production must use captured Civ III capital status',
                    'native_capital_indicator':'retained'},
         'grounding':'source_z_zero' if a.authored_ground else 'lowest_source_vertex',
         'footprint_half_extent_tiles':footprint_limit,'cross_tile_extent_authorization':'user permits slight city overlap, especially larger cities',
         'projection':projection,'source_z_pixels_per_unit':80.9543,'scene_world_z_per_source_unit':80.9543/(vertical*112),
         'material_channels_enabled':(['base_color','emissive']
             +(['ambient_occlusion'] if a.channels or a.ao_uv is not None else [])
             +(['normal_0_slope_adaptation'] if a.surface_detail else [])
             +(['normal_0_authored_frame'] if a.source_surface!='off' else [])
             +(['cooked_gloss_direct_specular'] if a.source_surface=='lit' else [])
             +(['opacity_coverage'] if a.opacity_cutouts else [])
             +(['metalness_direct_diagnostic'] if a.metalness else [])),
         'remaining':['source normal/gloss interpretation','full coast/route/vegetation envelopes','capital and wall states','all culture/era/size matrix']})
    base=OUT/f'{terrain_series}/{a.region}';report=json.loads((base/'report.json').read_text());jobs=json.loads((base/'batch.json').read_text())
    pairs=[(j,r) for j,r in zip(jobs,report['outputs']) if a.all_zooms or r['zoom']==1]
    cache=Cache(V2/'app/.cache');append=executable(V2/'qa/append_city_scene.cpp',cache)
    shadows=executable(V2/'systems/lighting/scene_shadow.cpp',cache)
    reference_frames={(r['hour'],r['zoom']):ROOT/r['packet'] for r in city.read(a.shadow_frame_report)['outputs']} if a.shadow_frame_report else {}
    for i,(job,row) in enumerate(pairs):
        combined=output/f'city-{i}.packet';shadowed=output/f'combined-{i}.packet'
        run([append,job[0],fixture/'city.bin',combined])
        run([shadows,combined,shadowed,row['hour'],base/'report.json']+
            ([reference_frames[row['hour'],row['zoom']]] if a.shadow_frame_report else []))
        # The pre-shadow copy is disposable; the final packet is replay input.
        combined.unlink()
        # Share existing immutable terrain mips/buffers instead of retaining a
        # full terrain copy for every city, hour and zoom.
        compact_packet(shadowed,V2/'app/.cache/content')
        row['packet']=rel(shadowed);job[0]=str(shadowed)
    report['outputs']=[r for _,r in pairs];save(output/'report.json',report);save(output/'batch.json',[j for j,_ in pairs])
    common=fixture/'city.hlsl'
    terrain_include=f'../../{terrain_series}/{a.region}/combined.hlsl'
    if a.ao_uv is not None:
        # Local shader closure only: no mutation of the shared terrain input.
        source=runner.shader_source(terrain_fixture.parent/'combined.hlsl')
        marker='    float4 q6_world : TEXCOORD2;'
        assert source.count(marker)==2
        source=source.replace(marker,marker+'\n    float2 city_ao_uv : TEXCOORD3;')
        marker='    output.material_index = input.material_index;'
        assert source.count(marker)==1
        source=source.replace(marker,marker+'\n    output.city_ao_uv = input.city_ao_uv;')
        if a.source_surface!='off':
            marker='    float2 city_ao_uv : TEXCOORD3;'
            assert source.count(marker)==2
            source=source.replace(marker,marker+'\n    float3 city_tangent : TEXCOORD4;\n    float3 city_bitangent : TEXCOORD5;')
            marker='    output.city_ao_uv = input.city_ao_uv;'
            source=source.replace(marker,marker+'\n    output.city_tangent = input.city_tangent;\n    output.city_bitangent = input.city_bitangent;')
            if a.extra_materials:
                marker='    float3 city_bitangent : TEXCOORD5;'
                source=source.replace(marker,marker+'\n    float2 city_emissive_uv : TEXCOORD6;')
                marker='    output.city_bitangent = input.city_bitangent;'
                source=source.replace(marker,marker+'\n    output.city_emissive_uv = input.city_emissive_uv;')
        (fixture/'terrain-auxiliary.hlsl').write_text(source)
        terrain_include='terrain-auxiliary.hlsl'
    common.write_text('#define Q3_NATURAL_WATER 1\n#define PSFeature Q8LegacyPSFeature\n'
        f'#include "{terrain_include}"\n#undef PSFeature\n'+f'#define Q8_CITY_AUXILIARY_AO {int(a.ao_uv is not None)}\n#define Q8_CITY_AO_STRENGTH {a.ao_strength:.9f}\n'+
        f'#define Q8_CITY_EXTRA_MATERIALS {int(bool(a.extra_materials))}\n#define Q8_CITY_SOURCE_SURFACE {int(a.source_surface!="off")}\n#define Q8_CITY_SOURCE_SPECULAR {int(a.source_surface=="lit")}\n#define Q8_CITY_VIEW_DIRECTION normalize(float3(1,1,{2*half_y/80.9543:.12f}))\n'+
        f'#define Q8_CITY_CHANNELS {int(a.channels)}\n#define Q8_CITY_SURFACE_DETAIL {int(a.surface_detail)}\n#define Q8_CITY_WORLD_Z_TO_SOURCE {vertical*112/80.9543:.12f}\n#define Q8_CITY_SEPARATE_EMISSION {int(a.emissive_uv>0)}\n#define Q8_CITY_EMISSIVE_GAIN {a.emissive_gain}\n#include "../../../../shaders/objects/city_scene_material.hlsl"\n')
    shader=fixture/'combined.hlsl';shader.write_text(f'#define Q3_OBJECT_REFLECTION 1\n#define Q3_REFLECTION_SIZE float2({width}.0,{height}.0)\n#include "city.hlsl"\n')
    reflected=fixture/'reflection.hlsl';reflected.write_text('#define VSMain Q3OriginalVSMain\n#define VSFeature Q3OriginalVSFeature\n#define PSMain Q3OriginalPSMain\n#define Q8_CITY_FEATURE_ENTRY Q3OriginalPSFeature\n#include "city.hlsl"\n#undef VSMain\n#undef VSFeature\n#undef PSMain\n'+f'#define Q3_REFLECTION_HEIGHT_NDC {4*.82*half/height:.12f}\n#include "../../../../shaders/hydrology/planar_reflection_pass.hlsl"\n')
    run([sys.executable,V2/'qa/replay_shader.py','--report',output/'report.json','--shader',shader,
         '--reflection-shader',reflected,'--output',output/'combined']+
         (['--post-shader',V2/'shaders/common/hdr_glow_tiled.hlsl'] if a.glow else []))

if __name__=='__main__':main()
