"""Q7 city augmentation over immutable registered terrain; exact Q0 anchors."""
import argparse,copy,json,subprocess,sys
from pathlib import Path
import presentation as p

def prepare(region,compact=False):
    source=p.FIX/('real-mixed' if region=='mixed' else 'real-holdout')
    out=p.FIX/'generated'/('registered-'+region+('-v3' if compact else '-v2'))
    (p.ROOT/out).mkdir(parents=True,exist_ok=True)
    f=p.read(source/'fixture.json');f['viewport']=[592,376]
    f['id']='q7-registered-'+region
    cityrow=(p.ROOT/source/'cities.csv').read_text().splitlines()[1].split(',')
    column,row=map(int,cityrow[:2])
    lines=(p.ROOT/source/'cities.csv').read_text().splitlines();header=lines[0].split(',');header[3]='0'
    (p.ROOT/out/'no-cities.csv').write_text(','.join(header)+'\n')
    f['scenarios']['cities']=str(out/'no-cities.csv')
    overlay=p.read(source/'augmentation.json');overlay.update(profile='q7_registered_presentation_v1',owner='Q7-presentation')
    overlay['objects']=[];overlay['presentation_layer']=dict(classification='source_adaptation',kind='city',tile=[column,row],size=0,pool='city/pool/american/ancient',note='deterministic Lab placement, not captured city state')
    p.write(out/'augmentation.json',overlay)
    f['real_map'].update(overlay=str(out/'augmentation.json'),overlay_sha256=p.sha(out/'augmentation.json'),scenario_hashes={key:p.sha(Path(path)) for key,path in f['scenarios'].items()})
    m=p.read(source/'module.json');m.update(id='q7-registered-terrain',shader=str(p.V2/'shaders/objects/presentation.hlsl'))
    p.write(out/'terrain.module.json',m);f['modules']=[str(out/'terrain.module.json')]
    p.write(out/'terrain.fixture.json',f)
    before=copy.deepcopy(f);before['scenarios']['cities']=str(source/'cities.csv');before['real_map']['scenario_hashes']['cities']=p.sha(source/'cities.csv');before['id']+='-before'
    p.write(out/'before.fixture.json',before)
    (p.ROOT/out/'points.csv').write_text(''.join(f'{column},{row},{u},{v}\n' for u,v in [(0.5,0.5),(.1,.1),(.9,.1),(.9,.9),(.1,.9)]))
    subprocess.run([sys.executable,'Renderer/terrain_lab/v2/app/surface_query.py','--fixture',str(out/'terrain.fixture.json'),'--points',str(out/'points.csv'),'--output',str(out/'surface.json')],cwd=p.ROOT,check=True)
    surface=p.read(out/'surface.json');anchor=surface['samples'][0]
    if anchor['base']>=11:raise ValueError('illegal city domain')
    if abs(surface['projection']['half_width']-64)>1e-6:raise ValueError('Q7 normal tile scale drift')
    p.WORLD_DRAWS.clear();draws=p.defaultdict(list);records=[]
    # Conservative local road-entry proxy; final published Q5 curve envelopes
    # remain a convergence obligation, never represented as matched here.
    from clearance_adapter import projected_plane
    p.ACTIVE_CORRIDORS=dict(schema='c3x.q5.route_clearance.v1',world_wrap=[0,0],halo_complete=True,classification='diagnostic_proxy',entries=[dict(id='declared-city-road-entry',kind='road',shape='capsule_chain',points=[projected_plane([-.03,0]),projected_plane([.8,0])],occupied_radius=14,clearance_radius=18)])
    p.emit_city(draws,records,'city/pool/american/ancient',0,[anchor['screen_x'],anchor['screen_y']],f['viewport'],'compact' if compact else 'stable',1)
    # Match the sampled terrain depth at the city origin while leaving source
    # local Z and rigid proportions unchanged. Flat city-site corner witness is
    # recorded; uneven sites must use per-building samples before acceptance.
    naive=.94-anchor['screen_y']/f['viewport'][1]*.75
    bias=anchor['depth']-naive
    payload=bytearray(p.struct.pack('<II',0x37515043,len(draws)))
    for (base,em),verts in sorted(draws.items()):
        for s in (base,em):b=s.encode();payload+=p.struct.pack('<I',len(b))+b
        payload+=p.struct.pack('<I',len(verts))
        for v in verts:v=list(v);v[2]+=bias;payload+=p.struct.pack('<9f',*v)
    (p.ROOT/out/'geometry.bin').write_bytes(payload)
    f['packs']['presentation_geometry']=str(out)
    f['modules'].append(str(p.OWN/'presentation.module.json'))
    p.write(out/'fixture.json',f)
    p.write(out/'layout.json',dict(schema='c3x.q7.real_layout.v1',components=records,source_sha256=f['real_map']['source_sha256'],region=region,source_coordinates=f['real_map']['region']['origin'],anchor=anchor,surface_query_sha256=p.sha(out/'surface.json'),ground_depth_bias=bias,footprint_surface_height_range=[min(x['height'] for x in surface['samples']),max(x['height'] for x in surface['samples'])],clearance=p.ACTIVE_CORRIDORS,classification='source_adaptation',pending=['Q5 final route envelope','Q3 bank exclusion where present','shared terrain receiver/city caster shadow','Q4 vegetation/city composition']))
    print(out/'fixture.json')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('region',choices=['mixed','mixed-holdout']);ap.add_argument('--compact',action='store_true');a=ap.parse_args();prepare(a.region,a.compact)
