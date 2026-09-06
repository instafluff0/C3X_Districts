#!/usr/bin/env python3
"""Inspect actual normalized source-bundle UVs/normals before screen projection.

These are source distributions, not proof of runtime instance transform fidelity.
No authored UVs are normalized, stretched, or silently repaired.
"""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np

ROOT=Path(__file__).resolve().parents[5]
OUTPUT=ROOT/'Renderer/terrain_lab/v2/audits/sampling/source-mesh-metrics.json'
BUNDLES=['CompoundUnitLab/unit_tank_runtime.bin','CityComponentsNormalized/city_runtime.bin',
         'ResourceNormalized/resource_runtime.bin','Civ5EnvironmentVegetation/vegetation_runtime.bin']

def inspect(path):
    data=path.read_bytes();offset=8
    if data[:8]!=b'C3XVEG1\0':raise ValueError('unknown source bundle')
    def uints(n):
        nonlocal offset
        values=struct.unpack_from('<'+'I'*n,data,offset);offset+=4*n;return values
    def string():
        nonlocal offset
        n,=uints(1);s=data[offset:offset+n].decode();offset+=n;return s
    version,nt,na,ng=uints(4)
    if version!=1:raise ValueError('unknown version')
    textures=[string() for _ in range(nt)];assets=[]
    for _ in range(na):
        name=string();texture,nv,ni=uints(3)
        vertices=np.frombuffer(data,dtype='<f4',count=nv*8,offset=offset).reshape(nv,8).astype(float);offset+=nv*32
        indices=np.frombuffer(data,dtype='<u4',count=ni,offset=offset).reshape(-1,3);offset+=ni*4
        if not np.isfinite(vertices).all() or np.max(indices)>=nv:raise ValueError('invalid source mesh')
        triangles=vertices[indices];e1=triangles[:,1,:3]-triangles[:,0,:3];e2=triangles[:,2,:3]-triangles[:,0,:3]
        a=np.linalg.norm(e1,axis=1);b=np.sum(e1*e2,axis=1)/np.maximum(a,1e-30)
        c=np.sqrt(np.maximum(np.sum(e2*e2,axis=1)-b*b,0))
        uv1=triangles[:,1,6:]-triangles[:,0,6:];uv2=triangles[:,2,6:]-triangles[:,0,6:]
        det=uv1[:,0]*uv2[:,1]-uv1[:,1]*uv2[:,0]
        good=(a>1e-10)&(c>1e-10)&(np.abs(det)>1e-12)
        j=np.stack([uv1[good]/a[good,None],(uv2[good]-uv1[good]*b[good,None]/a[good,None])/c[good,None]],axis=2)
        s=np.linalg.svd(j,compute_uv=False);anis=s[:,0]/s[:,1];density=np.sqrt(s[:,0]*s[:,1])
        norms=np.linalg.norm(vertices[:,3:6],axis=1)
        assets.append(dict(id=name,vertices=nv,triangles=len(indices),texture=textures[texture],
            geometry_or_uv_degenerate=int((~good).sum()),
            anisotropy_percentiles=dict(zip(['p50','p95','p99','max'],np.percentile(anis,[50,95,99,100]).tolist())) if len(anis) else None,
            uv_density_percentiles=dict(zip(['p05','p50','p95'],np.percentile(density,[5,50,95]).tolist())) if len(density) else None,
            source_triangles_above_16=int((anis>16).sum()),normal_length_range=[float(norms.min()),float(norms.max())],
            source_uv_sha256=hashlib.sha256(vertices[:,6:].astype('<f4').tobytes()).hexdigest()))
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(data).hexdigest(),assets=assets)

def main():
    result=dict(schema='c3x.q1.source_mesh_audit.v1',
        interpretation='Authored baseline; >16 flags investigation, not permission to alter source UVs. Runtime preprojection transforms and source/import equivalence remain separate gates.',
        bundles=[inspect(ROOT/'Renderer/packs'/name) for name in BUNDLES])
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
    print('Audited',sum(len(x['assets']) for x in result['bundles']),'actual normalized mesh assets')

if __name__=='__main__':main()
