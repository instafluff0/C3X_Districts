"""Adapt authoritative registered Q7 source sidecars to Q0 wire6 geometry."""
import argparse,copy,struct
from pathlib import Path
import presentation as p

def prepare(region,compact=False):
    source=p.FIX/'generated'/('registered-'+region+('-v3' if compact else '-v2'))
    out=p.FIX/'generated'/('registered-'+region+('-linear-v2' if compact else '-linear-v1'))
    (p.ROOT/out).mkdir(parents=True,exist_ok=True)
    world=p.read(source/'source-world-v1.json');f=p.read(source/'fixture.json')
    width,height=f['viewport'];payload=bytearray(struct.pack('<II',0x37515044,len(world['draws'])))
    for draw in world['draws']:
        if draw['alpha_mode']!='opaque':raise ValueError('source alpha family requires explicit runtime adoption')
        for key in ['base_color','emissive']:
            value=draw['channels'].get(key,{}).get('texture','').encode()
            payload+=struct.pack('<I',len(value))+value
        payload+=struct.pack('<I',len(draw['vertices']))
        for v in draw['vertices']:
            # Exact projection is published by Q7 using the Q0 query. No inverse
            # camera or depth reconstruction occurs in the provider or shader.
            payload+=struct.pack('<13f',v[8]/width*2-1,1-v[9]/height*2,v[10],v[3],v[4],*v[5:8],29,*v[:3],1)
    (p.ROOT/out/'geometry.bin').write_bytes(payload)
    terrain=p.read(source/'terrain.module.json');terrain.update(id='q7-linear-terrain-v1',shader=str(p.V2/'shaders/objects/terrain_linear_v1.hlsl'),
        linear_adapter=1,color_branch='q6_scene_linear_premultiplied_v1',world_positions=1)
    terrain.pop('packet_postprocessor',None);p.write(out/'terrain.module.json',terrain)
    f.update(id='q7-'+region+'-linear-v1',modules=[str(out/'terrain.module.json'),str(p.OWN/'linear.module.json')],
        packet_postprocessor=dict(source=str(p.V2/'systems/lighting/scene_shadow.cpp'),owner='Q6-lighting',contract=1))
    f['packs']['presentation_geometry']=str(out)
    f['settings'].update(samples=4,anisotropy=8,mip_bias=0,render_scale=1,
        postprocess=dict(shader=str(p.V2/'shaders/sampling/linear_reconstruct.hlsl'),owner='Q1-sampling',contract=2))
    p.write(out/'fixture.json',f)
    before=copy.deepcopy(f);before['id']+='-before';before['modules']=before['modules'][:1]
    before['packs'].pop('presentation_geometry')
    inherited=p.read(source/'before.fixture.json')
    before['scenarios']['cities']=inherited['scenarios']['cities']
    before['real_map']['scenario_hashes']['cities']=p.sha(Path(before['scenarios']['cities']))
    p.write(out/'before.fixture.json',before)
    p.write(out/'provenance.json',dict(schema='c3x.q7.linear_adoption.v1',source_world=str(source/'source-world-v1.json'),source_world_sha256=p.sha(source/'source-world-v1.json'),
        source_fixture=str(source/'fixture.json'),geometry_sha256=p.sha(out/'geometry.bin'),placement='unchanged',material_channels='base and source emission only; full material fidelity pending',
        interfaces=['Q0 wire6 generic geometry flags','Q6 final composed mesh/alpha shadow field','Q1 post contract2 linear reconstruction']))
    print(out/'fixture.json')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('region',choices=['mixed','mixed-holdout']);ap.add_argument('--compact',action='store_true');a=ap.parse_args();prepare(a.region,a.compact)
