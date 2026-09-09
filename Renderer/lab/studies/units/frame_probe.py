"""Recover local Warrior tangent frames without changing normalized source packs.

The installed skinned-object VS independently oct-decodes TANGENT.xy and .zw.
The byte offset is validated against orthogonality and UV orientation below;
its association with the source input layout is empirical, not reflected DXBC.
"""
from pathlib import Path
import hashlib
import json
import struct
import re
from collections import Counter
import numpy as np
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage
from Renderer.tools.asset_compiler.packed_static_frame import decode_octahedral_snorm8
from Renderer.tools.asset_compiler.unit_model_extractor import compile_unit_warrior, _compile_component
from Renderer.tools.asset_compiler.unit_member_resolver import ASSETS_ROOT, resolve_unit
from Renderer.tools.asset_compiler.unit_family_asset_importer import _physical_package, _initial_entry
from .prepare import ROOT, read

OUT=ROOT/'Renderer/lab/out/units/source-frame-pack'
REPORT=ROOT/'Renderer/lab/out/units/source-frame-build.json'


def recover(subjects=None, output=OUT):
    if subjects is None:
        report=compile_unit_warrior(ASSETS_ROOT,output,REPORT)
        for c in report['components']:
            c['asset_id']='unit/warrior/'+c['role'].lower()
            c['package']=report['source_package']['path']
        packages={report['source_package']['path']:IndexedStaticPackage(Path(report['source_package']['path']),report['components'][0]['source_entry'])}
    else:
        report={'components':[]};assets={};packages={};cache={}
        for slug,source_id,member in subjects:
            recipe=resolve_unit(ASSETS_ROOT,source_id,'Any',member_index=member)
            counts=Counter()
            for component in recipe['selected_components']:
                physical=_physical_package(ASSETS_ROOT,'Base',component['source_package']);key=str(physical)
                if key not in packages:packages[key]=IndexedStaticPackage(physical,_initial_entry(physical,[component['source_entry'],'Warrior_Armor_01']))
                role=re.sub(r'[^a-z0-9]+','_',component['role'].lower()).strip('_');counts[role]+=1
                role=role if counts[role]==1 else role+'_'+str(counts[role])
                asset,evidence=_compile_component(packages[key],ASSETS_ROOT/'Base/Platforms/Windows/BLPs/SHARED_DATA',output,component,cache,slug,role)
                aid='unit/'+slug+'/'+role;assets[aid]=asset;evidence.update(asset_id=aid,package=key);report['components'].append(evidence)
        output.mkdir(parents=True,exist_ok=True)
        (output/'manifest.json').write_text(json.dumps({'assets':assets},indent=2)+'\n')
        (output/'source-build.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest=read(output/'manifest.json');result={'schema':'c3x.unit_frame_probe.v1','components':{}}
    for component in report['components']:
        aid=component['asset_id'];package=packages[component['package']]
        doc=read(output/manifest['assets'][aid]['component'])
        for relative,part in zip(doc['meshes'],component['geometry']['parts']):
            vb=part['source_vertex_buffer'];raw=package.big_data(vb['offset'],vb['bytes'])
            if hashlib.sha256(raw).hexdigest()!=part['vertex_sha256']:raise ValueError('Source changed')
            mesh=read(output/relative);vertices=mesh['vertices'];indices=np.array(mesh['topology']['indices']).reshape(-1,3)
            # Current Warrior primitives reference every source vertex in order.
            if len(vertices)!=vb['count']:raise ValueError('Primitive needs explicit vertex remapping')
            for i,v in enumerate(vertices):
                p=np.array(struct.unpack_from('<3e',raw,i*vb['stride']))/100
                uv=struct.unpack_from('<2e',raw,i*vb['stride']+8)
                if not np.allclose(p,v['position'],atol=6e-9,rtol=0) or not np.allclose(uv,v['uv0'],atol=6e-9,rtol=0):
                    raise ValueError('Source/normalized vertex identity mismatch')
            normal=np.array([decode_octahedral_snorm8(raw,i*vb['stride']+6) for i in range(vb['count'])])
            candidates={}
            for offset in range(12,vb['stride']-3,4):
                tangent=np.array([decode_octahedral_snorm8(raw,i*vb['stride']+offset) for i in range(vb['count'])])
                bitangent=np.array([decode_octahedral_snorm8(raw,i*vb['stride']+offset+2) for i in range(vb['count'])])
                dots=np.concatenate((np.sum(normal*tangent,axis=1),np.sum(normal*bitangent,axis=1),np.sum(tangent*bitangent,axis=1)))
                candidates[offset]=float(np.mean(np.abs(dots)))
            offset=min(candidates,key=candidates.get)
            expected=20 if part['skinned'] else 12
            if offset!=expected or candidates[offset]*2>=min(v for k,v in candidates.items() if k!=offset):raise ValueError('Ambiguous packed tangent offset '+aid+' '+str(candidates))
            tangent=np.array([decode_octahedral_snorm8(raw,i*vb['stride']+offset) for i in range(vb['count'])])
            bitangent=np.array([decode_octahedral_snorm8(raw,i*vb['stride']+offset+2) for i in range(vb['count'])])
            pair_dot=float(np.mean(np.abs(np.sum(tangent*bitangent,axis=1))))
            if pair_dot>.015:raise ValueError('Packed tangent pair is not orthogonal '+aid)
            pos=np.array([v['position'] for v in vertices]);uv=np.array([v['uv0'] for v in vertices])
            e1,e2=pos[indices[:,1]]-pos[indices[:,0]],pos[indices[:,2]]-pos[indices[:,0]]
            u1,u2=uv[indices[:,1]]-uv[indices[:,0]],uv[indices[:,2]]-uv[indices[:,0]]
            det=u1[:,0]*u2[:,1]-u1[:,1]*u2[:,0];valid=np.abs(det)>1e-10
            du=(e1*u2[:,1,None]-e2*u1[:,1,None])[valid]/det[valid,None]
            dv=(e2*u1[:,0,None]-e1*u2[:,0,None])[valid]/det[valid,None]
            du/=np.linalg.norm(du,axis=1)[:,None];dv/=np.linalg.norm(dv,axis=1)[:,None]
            td=np.sum(tangent[indices[valid]].mean(axis=1)*du,axis=1)
            bd=np.sum(bitangent[indices[valid]].mean(axis=1)*dv,axis=1)
            result['components'][aid]={'source_vertex_sha256':part['vertex_sha256'],'stride':vb['stride'],
                'offset':offset,'tangent_pair_mean_abs_dot':pair_dot,'candidate_mean_abs_frame_dots':candidates,'vertices':len(vertices),
                'uv_tangent_mean_dot':float(td.mean()),'uv_bitangent_mean_dot':float(bd.mean()),
                'uv_tangent_positive_fraction':float((td>0).mean()),'uv_bitangent_positive_fraction':float((bd>0).mean()),
                'tangents':tangent.tolist(),'bitangents':bitangent.tolist(),
                'normalized_mesh':relative}
    (output/'frames.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    for aid,v in recover()['components'].items():
        print(aid,v['vertices'],'offset',v['offset'],'orthogonality',round(v['candidate_mean_abs_frame_dots'][v['offset']],5),
              'UV agreement',round(v['uv_tangent_mean_dot'],4),round(v['uv_bitangent_mean_dot'],4))
