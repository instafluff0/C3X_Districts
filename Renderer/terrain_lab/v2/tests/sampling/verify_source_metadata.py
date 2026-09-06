#!/usr/bin/env python3
"""Validate Q0 source sidecar against normalized binary bundles and DDS files."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
def sha(data):return hashlib.sha256(data).hexdigest()

def bundle(path):
    data=path.read_bytes();offset=8
    if data[:8]!=b'C3XVEG1\0':raise ValueError('source bundle format')
    def u(n=1):
        nonlocal offset
        values=struct.unpack_from('<'+'I'*n,data,offset);offset+=4*n;return values
    def string():
        nonlocal offset
        n,=u();s=data[offset:offset+n].decode();offset+=n;return s
    version,nt,na,ng=u(4);textures=[string() for _ in range(nt)];assets={}
    for _ in range(na):
        name=string();ti,nv,ni=u(3);vertices=data[offset:offset+nv*32];offset+=nv*32
        indices=data[offset:offset+ni*4];offset+=ni*4
        assets[sha(vertices)]=dict(id=name,vertices_sha256=sha(vertices),indices_sha256=sha(indices),
            uv_sha256=sha(b''.join(vertices[i+24:i+32] for i in range(0,len(vertices),32))),
            normal_sha256=sha(b''.join(vertices[i+12:i+24] for i in range(0,len(vertices),32))),
            vertex_count=nv,index_count=ni,base_color_source=textures[ti])
    return sha(data),assets

def main():
    ap=argparse.ArgumentParser();ap.add_argument('report',type=Path);a=ap.parse_args()
    r=json.loads(a.report.read_text());results=[];cache={}
    for output in r['outputs']:
        identity=output['source_metadata'];raw=(ROOT/identity['path']).read_bytes()
        if sha(raw)!=identity['sha256']:raise ValueError('metadata changed')
        m=json.loads(raw);ids=set();warnings=[]
        for source in m['meshes']:
            path=ROOT/source['bundle']
            if path not in cache:cache[path]=bundle(path)
            digest,assets=cache[path]
            if digest!=source['bundle_sha256']:raise ValueError('bundle changed')
            original=assets[source['vertices_sha256']]
            for key in ['vertices_sha256','indices_sha256','uv_sha256','normal_sha256','vertex_count','index_count']:
                if source[key]!=original[key]:raise ValueError('source mesh mismatch: '+key)
            if Path(source['base_color_source']).name!=Path(original['base_color_source']).name:raise ValueError('source texture identity mismatch')
            ids.add(source['vertices_sha256'])
        for texture in m['textures']:
            data=(ROOT/texture['path']).read_bytes()
            if sha(data)!=texture['sha256']:raise ValueError('source texture changed')
            if struct.unpack_from('<I',data,128)[0]!=texture['source_format']:raise ValueError('DDS format mismatch')
        for instance in m['instances']:
            if instance['mesh_sha256'] not in ids or instance['source_uniform_scale']<=0:raise ValueError('invalid instance identity/scale')
            if not instance['legacy_vertical_calibration_is_uniform_world_transform']:
                warnings.append(dict(id=instance['id'],projected_z=instance['projected_z_pixels_per_source_unit'],
                    ground_unit_z=instance['ground_projection_vertical_scale']*instance['world_z_authoring_divisor'],
                    classification='declared nonuniform legacy projection; not an accepted uniform source-body transform'))
        results.append(dict(zoom=output['zoom'],source_metadata=identity,verified_meshes=len(m['meshes']),
            verified_textures=len(m['textures']),verified_instances=len(m['instances']),
            tangent_stream=m['tangent_stream'],projection_exceptions=warnings))
    target=a.report.parent/'verified-source-metadata.json';target.write_text(json.dumps(results,indent=2)+'\n')
    print('PASS source identities at',len(results),'zooms;',results[0]['verified_meshes'],'meshes,',results[0]['verified_textures'],'textures,',results[0]['verified_instances'],'instances; legacy projection exceptions explicitly unresolved')

if __name__=='__main__':main()
