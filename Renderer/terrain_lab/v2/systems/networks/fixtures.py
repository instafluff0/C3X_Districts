"""Reproducible, explicitly synthetic Q5 fixtures. No BIQ state is fabricated."""
from pathlib import Path
import argparse, hashlib, json, math, struct
from dataclasses import asdict
from network import fit_crossing_grade,Node,Edge,Crossing,Graph,Mesh,build_routes,unit,sub,length,lerp
ROOT=Path(__file__).resolve().parents[5]
BASE=ROOT/'Renderer/terrain_lab/v2'
OUT=BASE/'fixtures/networks'
def rel(p):return p.relative_to(ROOT).as_posix()
def save_json(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def height(x,y):
    return 44*math.exp(-((x-105)/145)**2-((y-510)/165)**2)+22*math.exp(-((x+185)/130)**2-((y-245)/110)**2)
def world(u,v):return ((u-v)*64,(u+v)*64)
def create(name='dense',curved=True,rail_width=5.4,control=False,isolation=False,context=False,source=False):
    nodes=[Node(v*7+u,(u-v,u+v),world(u,v)) for v in range(7) for u in range(7)]
    edges=[];seen=set()
    def connect(a,b,rail=False,stage=1,pillaged=False):
        ia=a[1]*7+a[0];ib=b[1]*7+b[0];key=tuple(sorted((ia,ib)))
        if key in seen:return
        seen.add(key);pa,pb=nodes[ia].xy,nodes[ib].xy;c=None
        # An explicit synthetic horizontal hydrology edge at world y=352.
        if (pa[1]-352)*(pb[1]-352)<0:
            t=(352-pa[1])/(pb[1]-pa[1]);xy=lerp(pa,pb,t);direction=unit(sub(pb,pa))
            c=fit_crossing_grade(Crossing(len(edges),1000,xy,direction,14/abs(direction[1]),height(*xy)+2.2),height)
        edges.append(Edge(len(edges),ia,ib,rail,stage,pillaged,crossing=c))
    def run(points,rail=False,stage=1):
        for a,b in zip(points,points[1:]):connect(a,b,rail,stage)
    run([(u,3) for u in range(7)],True)
    run([(3,v) for v in range(7)],True)
    run([(u,1) for u in range(1,6)])
    run([(5,v) for v in range(1,6)])
    run([(u,5) for u in range(5,0,-1)])
    run([(1,v) for v in range(5,0,-1)])
    run([(1,4),(2,5),(3,6)],False,0)
    run([(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)],False,2) # Visually north-south.
    run([(0,6),(1,5),(2,4),(3,3),(4,2),(5,1),(6,0)],False,1)
    connect((4,4),(5,4),False,3,True)
    if context:
        # Industrial settlement/countryside: one rail corridor, road loop and spurs.
        keep=[]
        for e in edges:
            a,b=nodes[e.a].raw,nodes[e.b].raw
            na,nb=nodes[e.a],nodes[e.b]
            ua,va=(na.raw[0]+na.raw[1])//2,(na.raw[1]-na.raw[0])//2
            ub,vb=(nb.raw[0]+nb.raw[1])//2,(nb.raw[1]-nb.raw[0])//2
            if e.rail or (1<=ua<=5 and 1<=ub<=5 and 1<=va<=5 and 1<=vb<=5 and (ua==ub or va==vb)):
                keep.append(Edge(e.id,e.a,e.b,e.rail and va==vb,1,False,e.wrap,e.crossing))
        edges=keep
    graph=Graph(nodes,edges); mesh=Mesh(height)
    if not isolation:
        # Small, explicit sampled height proxy; Q2 replaces this surface at convergence.
        for v in range(-64 if context else 0,176 if context else 112):
            for u in range(-64 if context else 0,176 if context else 112):
                uv=((u/16-.5,v/16-.5),((u+1)/16-.5,v/16-.5),((u+1)/16-.5,(v+1)/16-.5),(u/16-.5,(v+1)/16-.5))
                points=[(*world(*p),height(*world(*p))) for p in uv]
                x,y=world((u+.5)/16-.5,(v+.5)/16-.5)
                noise=1+.035*math.sin(x*.43+y*.71)+.035*math.sin(x*.12-y*.27)
                dry=.5+.5*math.sin(x*.007+y*.004)
                color=((.17+dry*.045)*noise,(.235+dry*.015)*noise,(.07+dry*.006)*noise)
                mesh.quad(*points,color)
        water=[(-384+i*6,352) for i in range(129)]
        # Clip proxy channel to the diamond at this y; no topology inferred.
        water=[p for p in water if abs(p[0])<min(p[1]+64,832-p[1])]
        mesh.strip(water,18,(.23,.245,.135),lift=.14)
        mesh.strip(water,14,(.065,.17,.19),lift=.18)
        # Owned diagnostic city/worked-site proxies, distinct from source assets.
        for u,v in ((1,1),(5,3),(3,5)):
            x,y=world(u,v)
            for dx,dy,w,h in ((-17,-16,12,11),(-3,-17,10,17),(11,-13,9,12)):
                box(mesh,x+dx,y+dy,w,9,h,(.46,.39,.26))
        for u,v in ((2,1),(4,1),(1,4),(5,5)):
            x,y=world(u,v)
            for k in range(4):mesh.strip([(x-14+k*6,y-22),(x-14+k*6,y-8)],2.7,(.25,.28,.09),lift=.25)
    terrain_count=len(mesh.vertices)
    bridge_evidence=[]
    if not control:
        if source:
            from source_routes import build_source_routes
            build_source_routes(graph,mesh,curved,rail_width)
            from bridges import Bundle
            bundle=Bundle(ROOT/'Renderer/packs/RouteDoodadsNormalized/bridge_runtime.bin')
            for e in graph.edges:
                if e.crossing:bridge_evidence.append(bundle.add(mesh,e.crossing,e.stage,e.pillaged,e.rail))
        else:build_routes(graph,mesh,curved,rail_width)
    out=OUT/name;out.mkdir(parents=True,exist_ok=True)
    mesh_path=out/'scene.bin'
    with mesh_path.open('wb') as f:
        f.write(struct.pack('<II4f',0x354e4555 if source else 0x354e4554,len(mesh.vertices),0,187,1,0))
        for v in mesh.vertices:
            if source and len(v)==9:v=(*v,0,0,-1,0)
            f.write(struct.pack('<13f' if source else '<9f',*v))
    terrain=out/'terrain.csv'
    terrain.write_text('C3X_BIQ_TERRAIN_WINDOW_V2,7,7,49,0,0,14,14,0\n'+''.join(f'{u},{v},{u-v},{u+v},2,2,0,0,0\n' for v in range(7) for u in range(7)))
    provenance={'class':'constructed_stress_case','source_dataset':None,'source_hash':None,'surface':'explicit analytic proxy v1, not accepted terrain','hydrology':'explicit synthetic horizontal edge 1000, y=352; width 14','objects':'three authored diagnostic city markers and four worked plots; not captured state','generator':rel(Path(__file__)),'graph':{'nodes':[asdict(n) for n in nodes],'edges':[asdict(e) for e in edges]},'controls':{'curved':curved,'rail_width':rail_width,'no_routes':control,'isolation':isolation,'context':context,'source_routes':source},'mesh_sha256':hashlib.sha256(mesh_path.read_bytes()).hexdigest(),'terrain_vertices':terrain_count,'route_vertices':len(mesh.vertices)-terrain_count,'bridges':bridge_evidence,'component_classification':{'routes':'source_adaptation' if source else 'diagnostic_proxy','bridges':'source_adaptation' if source else 'diagnostic_proxy','ground':'diagnostic_proxy','cities':'diagnostic_proxy','worked_plots':'diagnostic_proxy'}}
    save_json(out/'provenance.json',provenance)
    from clearance import publish
    save_json(out/'clearance.json',publish(graph,height,bridge_evidence))
    fixture={'schema':'c3x.lab_v2.fixture.v1','id':name,'track':'Q5-networks','campaign':'Q1','tile_count':49,'viewport':[640,384] if context else [960,576],'terrain':rel(terrain),'modules':[rel(BASE/'systems/networks/module.json')],'packs':{'terrain':'Renderer/packs/Civ5EnvironmentSkin','vegetation':'Renderer/packs/Civ5EnvironmentVegetation','decals':'Renderer/packs/DecalsNormalized','relief':'Renderer/packs/TerrainElementsNormalized','shore':'Renderer/packs/ShoreNormalized','routes':'Renderer/packs/RouteStylesNormalized','bridges':'Renderer/packs/RouteDoodadsNormalized'},'references':['civ6.roads','civ3.real_gameplay_layout'],'isolations':['no_routes','networks_only'],'settings':{'anisotropy':8,'mip_bias':0,'samples':4,'render_scale':1,'postprocess':'box','camera_offsets':[[0,0]]},'scenarios':{'network_mesh':rel(mesh_path),'network_provenance':rel(out/'provenance.json'),'route_clearance':rel(out/'clearance.json')}}
    if source:
        from bridges import Bundle
        bundle=Bundle(ROOT/'Renderer/packs/RouteDoodadsNormalized/bridge_runtime.bin')
        texture_list=out/'bridge_textures.txt'
        texture_list.write_text(''.join('Renderer/packs/RouteDoodadsNormalized/'+p+'\n' for p in bundle.textures))
        fixture['scenarios']['bridge_textures']=rel(texture_list)
        files=[ROOT/'Renderer/packs/RouteDoodadsNormalized/bridge_runtime.bin']
        files += [ROOT/p for p in texture_list.read_text().splitlines()]
        pack=ROOT/'Renderer/packs/RouteStylesNormalized'
        files += list((pack/'route_styles').glob('*.json'))+list((pack/'materials/routes').glob('*.json'))
        for mat in (pack/'materials/routes').glob('*.json'):
            for channel in json.loads(mat.read_text())['channels'].values():files.append(pack/channel['texture'])
        save_json(out/'source_art.json',{'classification':'source_adaptation','geometry':'generated route ribbons; rigid source bridges, uniform transforms and unchanged UVs','source_files':{rel(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'channels_used':['base_color, authored alpha','source bridge normals'],'channels_pending':['route height BC5: decoding/normal semantics not asserted'],'background':'diagnostic_proxy'})
        fixture['scenarios']['source_art']=rel(out/'source_art.json')
    save_json(out/'fixture.json',fixture)
    return graph,mesh

def box(mesh,x,y,w,d,h,c):
    p=[(x-w/2,y-d/2),(x+w/2,y-d/2),(x+w/2,y+d/2),(x-w/2,y+d/2)]
    z=max(height(*a) for a in p);bottom=[(*a,z) for a in p];top=[(*a,z+h) for a in p]
    mesh.quad(*top,c)
    for i in range(4):mesh.quad(bottom[i],bottom[(i+1)%4],top[(i+1)%4],top[i],tuple(v*.8 for v in c))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='dense');p.add_argument('--straight',action='store_true');p.add_argument('--rail-width',type=float,default=5.4);p.add_argument('--no-routes',action='store_true');p.add_argument('--isolation',action='store_true');p.add_argument('--context',action='store_true');p.add_argument('--source',action='store_true');a=p.parse_args()
    g,m=create(a.name,not a.straight,a.rail_width,a.no_routes,a.isolation,a.context,a.source)
    print(f'{a.name}: {len(g.edges)} exact edges; {len(m.vertices)} vertices')
