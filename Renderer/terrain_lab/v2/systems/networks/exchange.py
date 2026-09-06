"""Adapt Q5's tested sweep envelopes to the shared scene-exchange contract."""
from pathlib import Path
import hashlib,json,math,sys
from network import unit,sub,add,mul
ROOT=Path(__file__).resolve().parents[5];BASE=ROOT/'Renderer/terrain_lab/v2'
sys.path.insert(0,str(BASE/'shared'))
from scene_exchange import validate

def export(fixture_path):
    fixture_path=fixture_path.resolve()
    f=json.loads(fixture_path.read_text());local=json.loads((ROOT/f['scenarios']['route_clearance']).read_text())
    source=hashlib.sha256((ROOT/f['scenarios']['network_mesh']).read_bytes()).hexdigest();envelopes=[]
    def emit(e,poly,suffix):
        envelopes.append({'id':e['id']+suffix,'kind':'bridge' if e['kind']=='bridge' else 'rail' if e['kind']=='road_and_rail' else 'road','polygon':[list(p) for p in poly],'height_range':e['height_range'],'clearance':e['clearance_radius']-e['occupied_radius'],'source_geometry_sha256':source})
    for e in local['entries']:
        pts=e['points'];radius=e['occupied_radius']
        if e['shape']=='polygon':emit(e,pts,'');continue
        for i,(a,b) in enumerate(zip(pts,pts[1:])):
            if a==b:continue
            t=unit(sub(b,a));n=(-t[1],t[0]);r=radius
            emit(e,[add(a,mul(n,-r)),add(b,mul(n,-r)),add(b,mul(n,r)),add(a,mul(n,r))],f':segment:{i}')
        # Circumscribed disks conservatively cover exact swept end/junction caps.
        for i,p in enumerate(pts):
            r=radius/math.cos(math.pi/12)
            emit(e,[add(p,(r*math.cos(math.tau*k/12),r*math.sin(math.tau*k/12))) for k in range(12)],f':cap:{i}')
    data={'schema':'c3x.lab_v2.corridors.v1','coordinate_space':'civ3_raw_delta_pixels_v1','terrain_sha256':hashlib.sha256((ROOT/f['terrain']).read_bytes()).hexdigest(),'region_id':f['id'],'provider':'Q5-networks','revision':1,'wrap_period':local['world_wrap'],'envelopes':envelopes,'classification':'source_adaptation','fixture_class':'constructed_stress_case','halo_complete':False,'source_geometry':{'path':f['scenarios']['network_mesh'],'sha256':source},'polygonization':'Rendered strip quads plus circumscribed 12-sided caps; <=0.212px conservative excess at radius6. Bridge rigid XY bounds preserved.'}
    validate(data);out=fixture_path.parent/'corridors.json';out.write_text(json.dumps(data,separators=(',',':'))+'\n')
    f['sidecars']=[{'path':out.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'schema':data['schema'],'owner':'Q5-networks'}]
    fixture_path.write_text(json.dumps(f,indent=2)+'\n');return out
if __name__=='__main__':print(export(Path(sys.argv[1])).relative_to(ROOT))
