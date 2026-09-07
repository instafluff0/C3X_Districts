"""Inspect unconsumed static vertex bytes against geometric normals, without pack edits."""
import json
import struct
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from Renderer.tools.asset_compiler import compound_landmark_importer as source


def main():
    entry='DIS_CTY_RE_Bld_MD_A_01'
    package=source.IndexedStaticPackage(source.MAC_ASSETS_ROOT/'Base/Platforms/Windows/BLPs/landmarks/city_buildings.blp',entry)
    _,owner,model=source.landmark_base_model(package)
    states,_=source._decode_states(package,owner)
    primitive_array=package.pointer_fields(model,source.TYPE_PRIM_GROUP)[0][1]
    material_array=package.pointer_fields(model,source.TYPE_MATERIAL)[0][1]
    primitive=source._decode_primitive(package,package.array_element(primitive_array,0),states,package.allocations[material_array-1]['element_count'])
    ve=source.decode_buffer_entry(package,package.unique_allocation(source.TYPE_VERTEX_BUFFER),primitive['vertex_buffer'],True)
    ie=source.decode_buffer_entry(package,package.unique_allocation(source.TYPE_INDEX_BUFFER),primitive['index_buffer'],False)
    vb=package.big_data(ve['offset'],ve['bytes']);ib=package.big_data(ie['offset'],ie['bytes'])
    mesh,evidence=source._normalize_geometry(vb,ib,ve,ie,primitive,100,None,auxiliary_uvs=True)
    indices=struct.unpack('<'+str(ie['count'])+'H',ib)[primitive['first_index']:primitive['first_index']+primitive['index_count']]
    referenced=sorted({i+primitive['base_vertex'] for i in indices})
    assert len(referenced)==len(mesh['vertices'])
    rows=[]
    for index,vertex in zip(referenced,mesh['vertices']):
        raw=vb[index*ve['stride']:(index+1)*ve['stride']]
        assert all(abs(struct.unpack_from('<e',raw,axis*2)[0]/100-vertex['position'][axis])<1e-7 for axis in range(3))
        rows.append({'normal':vertex['normal'],'raw':raw.hex(),'uv':vertex['uv0']})
    output=ROOT/'Renderer/terrain_lab/v2/fixtures/beauty/city-tangent-source-r1'
    output.mkdir(exist_ok=True)
    if (output/'data.json').exists():raise ValueError('preserve existing raw vertex evidence')
    (output/'data.json').write_text(json.dumps({'entry':entry,'stride':ve['stride'],'profile':ve['format'],'source_vertex_sha256':evidence['vertex_sha256'],'rows':rows},indent=2)+'\n')
    print('vertices',len(rows),'stride',ve['stride'])
    for row in rows[:12]:print(row)


if __name__=='__main__':main()
