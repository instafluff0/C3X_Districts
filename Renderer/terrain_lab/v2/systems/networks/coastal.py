"""Prepare exact Q8 coastal routes, pinned Q0 surface samples and Q4 corridors.

Only Q5 paths are written. The four gameplay links and terrain remain identical;
the augmentation owner is rebound explicitly for Q0's owned replay validation.
"""
import hashlib,json,math,struct,subprocess,sys,copy
from pathlib import Path
from network import Node,Edge,Graph,Mesh
from source_routes import build_source_routes
from exchange import validate
ROOT=Path(__file__).resolve().parents[5];V2=ROOT/'Renderer/terrain_lab/v2'
OUT=V2/'fixtures/networks/coastal-r01'
def rel(p):return p.relative_to(ROOT).as_posix()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def graph_for(recipe):
    routes=recipe['placement']['routes'];tiles=sorted({tuple(e[k]) for e in routes for k in ('from','to')})
    ids={p:i for i,p in enumerate(tiles)}
    nodes=[Node(ids[(c,r)],(c+r,c-r),(64*(c+r+1),64*(c-r))) for c,r in tiles]
    edges=[Edge(i,ids[tuple(e['from'])],ids[tuple(e['to'])],rail=e['kind']=='rail',stage=1) for i,e in enumerate(routes)]
    return Graph(nodes,edges)
def query_point(x,y):
    c,r=(x+y)/128,(x-y)/128
    return math.floor(c),math.floor(r),c-math.floor(c),1-(r-math.floor(r))
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    recipe_path=V2/'fixtures/beauty/coastal-r01/RECIPE.json'
    original=V2/'fixtures/beauty/coastal-r01/after.fixture.json'
    recipe=json.loads(recipe_path.read_text());f=json.loads(original.read_text())
    inputs={rel(p):sha(p) for p in (recipe_path,original)}
    for key,value in list(f['scenarios'].items())+[('terrain',f['terrain'])]:
        source=ROOT/value;target=OUT/source.name;target.write_bytes(source.read_bytes());inputs[value]=sha(source)
        if key=='terrain':f[key]=rel(target)
        else:f['scenarios'][key]=rel(target)
    overlay=ROOT/f['real_map']['overlay'];target=OUT/'augmentation.json';augmentation=json.loads(overlay.read_text())
    augmentation['owner']='Q5-networks';write(target,augmentation)
    f['real_map']['overlay']=rel(target);f['real_map']['overlay_sha256']=sha(target);inputs[rel(overlay)]=sha(overlay)
    f.update(id='q5-coastal-query',track='Q5-networks')
    write(OUT/'query.fixture.json',f)
    graph=graph_for(recipe);mesh=build_source_routes(graph,Mesh(lambda x,y:0),rail_width=12)
    unique=list(dict.fromkeys((v[0],v[1]) for v in mesh.vertices))
    (OUT/'points.csv').write_text(''.join(','.join(map(str,query_point(*p)))+'\n' for p in unique))
    subprocess.run([sys.executable,str(V2/'app/surface_query.py'),'--fixture',rel(OUT/'query.fixture.json'),'--points',rel(OUT/'points.csv'),'--output',rel(OUT/'surface.json')],cwd=ROOT,check=True)
    surface=json.loads((OUT/'surface.json').read_text());samples=dict(zip(unique,surface['samples']))
    if len(samples)!=len(unique):raise ValueError('surface query incomplete')
    w,h=f['viewport'];lift=.35;vertices=[];envelopes=[]
    for v in mesh.vertices:
        s=samples[(v[0],v[1])];lift_height=lift*112/64
        x,y=s['screen_x'],s['screen_y']-lift_height*surface['projection']['vertical_scale']
        world=((v[0]+v[1])/128,(v[0]-v[1])/128,s['height']/112+lift/64,1)
        vertices.append((x/w*2-1,1-y/h*2,s['depth']-lift_height*.0012,
                         s['normal_x'],s['normal_y'],s['normal_z'],*v[6:13],*world))
    payload=struct.pack('<4I',0x514e4331,w,h,len(vertices))+b''.join(struct.pack('<17f',*v) for v in vertices)
    (OUT/'scene.bin').write_bytes(payload)
    # Exact rendered triangle footprint union (not a proxy straight line).
    for i in range(0,len(mesh.vertices),3):
        vs=mesh.vertices[i:i+3];z=[vertices[j][15]*64 for j in range(i,i+3)]
        envelopes.append(dict(id=f'road-triangle:{i//3}',kind='road',polygon=[list(v[:2]) for v in vs],height_range=[min(z),max(z)],clearance=4.,source_geometry_sha256=sha(OUT/'scene.bin')))
    data=dict(schema='c3x.lab_v2.corridors.v1',coordinate_space='civ3_raw_delta_pixels_v1',terrain_sha256=sha(OUT/'terrain.csv'),region_id=f['real_map']['region_id'],provider='Q5-networks',revision=2,wrap_period=[6400,0],envelopes=envelopes,source_geometry=dict(path=rel(OUT/'scene.bin'),sha256=sha(OUT/'scene.bin')),classification='source_adaptation',halo_complete=True,halo_scope='Complete four-edge declared Q8 augmentation; no captured routes or inferred off-crop links.',source_recipe=dict(path=rel(recipe_path),sha256=sha(recipe_path)))
    validate(data);write(OUT/'corridors.json',data)
    sidecar=dict(path=rel(OUT/'corridors.json'),sha256=sha(OUT/'corridors.json'),schema=data['schema'],owner='Q5-networks')
    # Serialize the exact Q4-owned callback input format; all placement decisions
    # remain in Q4's published implementation, evaluated on actual source vertices.
    cache=OUT/'clearance.csv'
    cache.write_text('C3X_Q4_CLEARANCE_V1\n'+''.join(','.join(map(str,[e['clearance'],*data['wrap_period'],len(e['polygon']),*(v for p in e['polygon'] for v in p)]))+'\n' for e in envelopes))
    index={k:copy.deepcopy(data[k]) for k in ('schema','coordinate_space','terrain_sha256','region_id','provider','revision','wrap_period')}
    index.update(envelopes=[],relief_clearance=dict(path=rel(cache),sha256=sha(cache)),source_corridors=sidecar)
    write(OUT/'q4-clearance.json',index)
    base=json.loads((ROOT/f['modules'][0]).read_text());base.update(id='q5-coastal-base',owner='Q5-networks',shader='Renderer/terrain_lab/v2/shaders/networks/coastal_base.hlsl')
    for k in ('terrain_hooks','hydrology_hooks'):base[k]['owner']='Q8-beauty'
    post=base.pop('packet_postprocessor');write(OUT/'base.module.json',base)
    q5=dict(schema='c3x.lab_v2.module.v1',id='q5-coastal-routes',owner='Q5-networks',provider='cpp_packet',contract=1,source='Renderer/terrain_lab/v2/systems/networks/coastal.cpp',shader='Renderer/terrain_lab/v2/shaders/networks/coastal.hlsl',color_branch='q6_scene_linear_premultiplied_v1',after=['q5-coastal-base'])
    write(OUT/'routes.module.json',q5)
    f.update(id='q5-coastal-routes',modules=[rel(OUT/'base.module.json'),rel(OUT/'routes.module.json')],packet_postprocessor=post,sidecars=[sidecar])
    write(OUT/'routes.fixture.json',f)
    base['placement_hooks']=dict(header='Renderer/terrain_lab/v2/systems/relief/placement_adapter.h',owner='Q4-relief',initialize='q4_placement::initialize',accept_vegetation='q4_placement::accept_vegetation')
    write(OUT/'clear.module.json',base)
    f.update(id='q5-coastal-clear',modules=[rel(OUT/'clear.module.json'),rel(OUT/'routes.module.json')])
    f['sidecars'].append(dict(path=rel(OUT/'q4-clearance.json'),sha256=sha(OUT/'q4-clearance.json'),schema=data['schema'],owner='Q5-networks'))
    write(OUT/'clear.fixture.json',f)
    route_pack=ROOT/f['packs']['routes'];source_files=list((route_pack/'materials/routes').glob('*.json'))
    for path in source_files:
        mat=json.loads(path.read_text())
        inputs[rel(path)]=sha(path)
    write(OUT/'provenance.json',dict(schema='c3x.q5.coastal.v1',classification='source_adaptation',inputs=inputs,terrain_sha256=sha(OUT/'terrain.csv'),augmentation_sha256=sha(target),surface_query_sha256=sha(OUT/'surface.json'),mesh_sha256=sha(OUT/'scene.bin'),edge_count=len(graph.edges),vertex_count=len(vertices),surface_lift_raw_pixels=lift,geometry='Q5 curves and full-width exact Q0 height queries; Q8 topology unchanged; no bridges in this river-free recipe.',stage='Medieval source atlas: matches inherited Q8 renderer style override1.',shadow='Ground ribbons are actual world-space receivers of the final Q6 mesh/alpha shadow field; caster=false.',clearance='Q4 full transformed source vertex callback; four-pixel margin from rendered triangle footprint; no whole-tile clearing.',pending=['Q4 review of source clearance composition','Final composed terrain sampler when relief geometry changes','Broader developed-map and heldout acceptance']))
    print(rel(OUT/'clear.fixture.json'))
if __name__=='__main__':main()
