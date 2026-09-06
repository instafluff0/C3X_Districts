#!/usr/bin/env python3
"""Independently fit source-to-Q7 world transforms and verify source UVs.

Reads published Q7 world/layout artifacts. No emitter functions are reused and
no owner artifact is modified. Planar source parts cannot prove a full 3D affine
transform and remain explicitly unidentifiable rather than silently passing.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[5]
PACK=ROOT/'Renderer/packs/CityComponentsNormalized'
AUDIT=ROOT/'Renderer/terrain_lab/v2/audits/sampling'

def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('fixture',type=Path);a=ap.parse_args()
    directory=a.fixture.resolve();layout=read(directory/'layout.json');world=read(directory/'world.json')
    manifest=read(PACK/'manifest.json');draws={(d['base_color'],d['emissive']):np.asarray(d['vertices'],float) for d in world['draws']}
    cursors={k:0 for k in draws};rows=[]
    for instance in layout['components']:
        if 'pool' not in instance or 'footprint' not in instance:raise ValueError('this witness requires city components only')
        landmark_path=PACK/manifest['assets'][instance['asset']]['landmark'];landmark=read(landmark_path)
        for binding in landmark['draw_bindings']:
            if 'worked' not in binding['states']:continue
            mesh_path=PACK/landmark['components']['geometry'][binding['geometry']]
            material_path=PACK/landmark['components']['materials'][binding['material']]
            mesh=read(mesh_path);material=read(material_path);channels=material['channels']
            texture=lambda k:(PACK/channels[k]['texture']).relative_to(ROOT).as_posix() if k in channels else ''
            key=(texture('base_color'),texture('emissive'));indices=np.asarray(mesh['topology']['indices'])
            source=np.asarray([v['position'] for v in mesh['vertices']],float)[indices]
            uv=np.asarray([v['uv0'] for v in mesh['vertices']],float)[indices]
            normals=np.asarray([v['normal'] for v in mesh['vertices']],float)[indices]
            start=cursors[key];actual=draws[key][start:start+len(source)];cursors[key]+=len(source)
            if actual.shape!=(len(source),8):raise ValueError('source/draw span mismatch')
            basis=np.column_stack([source,np.ones(len(source))]);fit,_,rank,_=np.linalg.lstsq(basis,actual[:,:3],rcond=None)
            residual=float(np.max(np.abs(basis@fit-actual[:,:3])))
            uv_error=float(np.max(np.abs(uv-actual[:,3:5])))
            row=dict(asset=instance['asset'],slot=instance['slot'],pool=instance['pool'],size=instance['size'],
                source_mesh=mesh_path.relative_to(ROOT).as_posix(),source_sha256=digest(mesh_path),
                source_material=material_path.relative_to(ROOT).as_posix(),material_sha256=digest(material_path),
                vertices=len(source),affine_rank=int(rank),position_fit_error=residual,uv_max_error=uv_error,
                declared_material_channels=sorted(channels),bound_material_channels=[k for k in ['base_color','emissive'] if k in channels],
                pending_material_channels=sorted(set(channels)-{'base_color','emissive'}),alpha_mode=material.get('alpha_mode'))
            if rank==4:
                matrix=fit[:3].T;scales=np.linalg.svd(matrix,compute_uv=False);scale=float(scales.mean())
                normal_error=float(np.max(np.abs(normals@(matrix/scale).T-actual[:,5:8])))
                row.update(singular_values=scales.tolist(),axis_ratio=float(scales.max()/scales.min()),
                    inferred_uniform_scale=scale,declared_scale=instance['scale'],normal_max_error=normal_error,
                    determinant=float(np.linalg.det(matrix)))
                row['passed']=bool(row['axis_ratio']<1.00001 and residual<1e-7 and uv_error==0 and normal_error<1e-6 and row['determinant']>0)
            else:row.update(passed=False,reason='planar/degenerate source part cannot identify full 3D transform')
            rows.append(row)
    if any(cursors[k]!=len(draws[k]) for k in draws):raise ValueError('unmatched world vertices')
    # Verify the actual serialized Q7 geometry against its published world
    # contract, including UV/normal precision. This does not prove unbound
    # material channels or shared shadow/receiver behavior.
    fixture=read(directory/'fixture.json');width,height=fixture['viewport']
    data=(directory/'geometry.bin').read_bytes();offset=8
    magic,count=struct.unpack_from('<2I',data)
    if magic!=0x37515043:raise ValueError('Q7 geometry version')
    def string():
        nonlocal offset
        n,=struct.unpack_from('<I',data,offset);offset+=4;s=data[offset:offset+n].decode();offset+=n;return s
    packet_errors=[]
    for _ in range(count):
        base=string();emissive=string();n,=struct.unpack_from('<I',data,offset);offset+=4
        packed=np.frombuffer(data,dtype='<f4',count=n*9,offset=offset).reshape(n,9);offset+=n*36
        world_draw=draws[(base,emissive)];x,y,z=world_draw[:,:3].T
        expected=np.column_stack([(x-y)*128/width,-((x+y)*32-z*80.9543)*2/height,
            .94-(height*.5+(x+y)*32)/height*.75-z*.20732,world_draw[:,3:]])
        packet_errors.append(float(np.max(np.abs(packed[:,:8]-expected.astype('<f4')))))
    if offset!=len(data) or max(packet_errors)>2e-6:raise ValueError('world/serialized geometry mismatch')
    result=dict(schema='c3x.q1.q7_world_audit.v1',fixture=directory.relative_to(ROOT).as_posix(),
        world_sha256=digest(directory/'world.json'),layout_sha256=digest(directory/'layout.json'),
        geometry_sha256=digest(directory/'geometry.bin'),world_to_serialized_max_error=max(packet_errors),
        parts=rows,full_rank_parts=sum(x['affine_rank']==4 for x in rows),passed_full_rank_parts=sum(x['passed'] for x in rows),
        unidentifiable_parts=sum(x['affine_rank']<4 for x in rows),uv_max_error=max(x['uv_max_error'] for x in rows),
        interpretation='Source-to-published-Q7-world evidence only. Runtime packet, alpha/channel completeness, real terrain placement and shared shadows remain separate gates.')
    path=AUDIT/(directory.name+'-q7-world-metrics.json');path.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='parts'})
    if any(x['uv_max_error']!=0 or x['position_fit_error']>1e-7 or (x['affine_rank']==4 and not x['passed']) for x in rows):raise ValueError('source transform/UV/normal mismatch')

if __name__=='__main__':main()
