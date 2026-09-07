"""Compare packed source normals with existing normalized geometry; emit a Lab override."""
import argparse
import json
import math
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
sys.path.insert(0,str(ROOT))
from Renderer.tools.asset_compiler import compound_landmark_importer as source
from Renderer.tools.asset_compiler.packed_static_frame import decode_octahedral_snorm8
sys.path.insert(0,str(V2/'systems/objects'))
from mesh_fingerprint import geometry_digest
from city_source_selection import selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    select=parser.add_mutually_exclusive_group(required=True)
    select.add_argument('--pool')
    select.add_argument('--palace',help='Generic root ID in the existing normalized palace library')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-frame',action='store_true',help='Also decode the two tangent directions established by the installed rigid-model vertex shader')
    a=parser.parse_args()
    if a.output.exists():raise ValueError('preserve existing normal-source evidence')
    selected,pack,units,report=selection(a.pool,a.palace)
    manifest=json.loads((pack/'manifest.json').read_text());packages={};overrides={};records=[]
    for item in selected:
        if item['package'] not in packages:
            packages[item['package']]=source.IndexedStaticPackage(source.MAC_ASSETS_ROOT/item['package'],item['entry'])
        package=packages[item['package']];package.select_direct_string(item['entry'])
        _,owner,model=source.landmark_base_model(package);states,_=source._decode_states(package,owner)
        primitive_array=package.pointer_fields(model,source.TYPE_PRIM_GROUP)[0][1]
        material_array=package.pointer_fields(model,source.TYPE_MATERIAL)[0][1]
        landmark=json.loads((pack/manifest['assets'][item['asset_id']]['landmark']).read_text())
        targets={}
        for name in landmark['components']['geometry']:
            mesh=json.loads((pack/name).read_text());targets.setdefault(geometry_digest(mesh),[]).append(mesh)
        for pi in range(package.allocations[primitive_array-1]['element_count']):
            primitive=source._decode_primitive(package,package.array_element(primitive_array,pi),states,package.allocations[material_array-1]['element_count'])
            ve=source.decode_buffer_entry(package,package.unique_allocation(source.TYPE_VERTEX_BUFFER),primitive['vertex_buffer'],True)
            if (ve['format'],ve['stride'])!=(0x315CFCD9,24):
                raise ValueError('packed normal interpretation is only verified for static profile 0x315CFCD9 / stride 24')
            ie=source.decode_buffer_entry(package,package.unique_allocation(source.TYPE_INDEX_BUFFER),primitive['index_buffer'],False)
            vb=package.big_data(ve['offset'],ve['bytes']);ib=package.big_data(ie['offset'],ie['bytes'])
            mesh,evidence=source._normalize_geometry(vb,ib,ve,ie,primitive,units,None,auxiliary_uvs=True)
            digest=geometry_digest(mesh)
            if digest not in targets:raise ValueError('source geometry does not match normalized study mesh')
            indices=[i+primitive['base_vertex'] for i in struct.unpack('<'+str(ie['count'])+'H',ib)[primitive['first_index']:primitive['first_index']+primitive['index_count']]]
            used=set()
            for start in range(0,len(indices),3):
                tri=indices[start:start+3]
                positions=[[struct.unpack_from('<e',vb,i*ve['stride']+axis*2)[0]/units for axis in range(3)] for i in tri]
                u=[positions[1][j]-positions[0][j] for j in range(3)];v=[positions[2][j]-positions[0][j] for j in range(3)]
                cross=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
                if math.sqrt(sum(x*x for x in cross))>1e-12:used.update(tri)
            referenced=sorted(used);assert len(referenced)==len(mesh['vertices'])
            normals=[decode_octahedral_snorm8(vb,index*ve['stride']+6) for index in referenced]
            dots=[sum(x*y for x,y in zip(n,v['normal'])) for n,v in zip(normals,mesh['vertices'])]
            entry={'geometry_digest':digest,'normals':normals}
            if a.include_frame:
                entry['tangents']=[decode_octahedral_snorm8(vb,index*ve['stride']+12) for index in referenced]
                entry['bitangents']=[decode_octahedral_snorm8(vb,index*ve['stride']+14) for index in referenced]
            for target in targets[digest]:
                key=target['asset_id']
                if key in overrides:assert overrides[key]==entry
                overrides[key]=entry
            records.append({'asset':item['asset_id'],'primitive':pi,'profile':ve['format'],'stride':ve['stride'],
                            'vertices':len(normals),'source_vertex_sha256':evidence['vertex_sha256'],
                            'minimum_geometric_dot':min(dots),'mean_geometric_dot':sum(dots)/len(dots),
                            'negative_geometric_dots':sum(x<0 for x in dots)})
        if any(t['asset_id'] not in overrides for ts in targets.values() for t in ts):raise ValueError('missing source mesh normal mapping')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'schema':'c3x.lab.vertex_normals.v1','classification':'source octahedral directions; shader-backed frame decoding when include-frame is set; source BRDF integration remains partial',
                                    'frame_offsets':[12,14] if a.include_frame else None,
                                    'offset':6,'encoding':'two signed normalized octahedral bytes','meshes':overrides,'evidence':records},indent=2)+'\n')
    print(json.dumps({'meshes':len(overrides),'primitives':len(records),'minimum_dot':min(x['minimum_geometric_dot'] for x in records),
                      'negative_dots':sum(x['negative_geometric_dots'] for x in records)}))


if __name__=='__main__':main()
