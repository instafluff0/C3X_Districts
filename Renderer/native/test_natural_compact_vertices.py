"""Stage 5 exact compaction: 56-byte natural vertices give identical shader inputs.

A world-projected natural mesh whose uv equals world xy and whose relief owner
is zero, bit for bit, drops position, uv and owner (92 -> 56 bytes). Its input
layouts alias position and uv to world (the vertex shaders replace position
from world whenever natural_projection.z > 0) and read the owner from a
stride-0 zero stream. Every other natural mesh keeps all 92 bytes.
"""
import re
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp

NATIVE = ROOT / 'Renderer/native'
COMPONENTS = {'DXGI_FORMAT_R32_FLOAT': 1, 'DXGI_FORMAT_R32G32_FLOAT': 2,
              'DXGI_FORMAT_R32G32B32_FLOAT': 3, 'DXGI_FORMAT_R32G32B32A32_FLOAT': 4}


def natural_layouts():
    """(components, full offset, compact offset, compact slot) per natural shader input."""
    source = (NATIVE / 'source_fidelity/runtime.h').read_text()
    block = re.search(r'D3D11_INPUT_ELEMENT_DESC e\[\]=\{(.*?)\};', source, re.S).group(1)
    full = [(COMPONENTS[f], int(o)) for f, o in
            re.findall(r'\{"\w+",\d+,(DXGI_FORMAT_\w+),0,(\d+),D3D11_INPUT_PER_VERTEX_DATA,0\}', block)]
    compact = [int(v) for v in re.search(r'unsigned compact_offsets\[\]=\{([^}]*)\}', source).group(1).split(',')]
    assert len(full) == len(compact) == 8, (full, compact)
    assert 'e[7].InputSlot=1;' in source, 'the owner must come from slot 1'
    return [(n, f, c, 1 if i == 7 else 0) for i, ((n, f), c) in enumerate(zip(full, compact))]


# Shadow caster natural inputs: uv, base terrain, world, desert weight, owner.
CASTER_FULL = [(2, 40), (1, 72), (4, 12), (1, 56), (4, 76)]
CASTER_COMPACT = [(2, 0, 0), (1, 52, 0), (4, 0, 0), (1, 36, 0), (4, 0, 1)]
# Territory borders: position (unused by the shader) and world.
BORDER_FULL = [(3, 0), (3, 12)]
BORDER_COMPACT = [(3, 0, 0), (3, 0, 0)]


def table(rows):
    return ','.join('{%s}' % ','.join(str(v) for v in row) for row in rows)


class NaturalCompactVertexTests(unittest.TestCase):
    def test_compact_inputs_match_the_full_format(self):
        layouts = natural_layouts()
        run_cpp(r'''
#include <array>
#include <cmath>
#include <cassert>
#include <cstring>
#include "Renderer/native/render_core/prepared_mesh.h"
using namespace c3x_renderer::render_core;
struct Element {unsigned components,full,compact,slot;};
Element natural[]={''' + table(layouts) + r'''};
Element caster[]={''' + table([(n, f, c, s) for (n, f), (_, c, s) in zip(CASTER_FULL, CASTER_COMPACT)]) + r'''};
Element border[]={''' + table([(n, f, c, s) for (n, f), (_, c, s) in zip(BORDER_FULL, BORDER_COMPACT)]) + r'''};
float fetch(PreparedMesh const& m,unsigned vertex,unsigned offset,unsigned component,unsigned slot){
 if(slot)return 0.f; // the stride-0 zero stream
 float value;std::memcpy(&value,m.vertices.data()+vertex*m.vertex_stride+offset+component*4,4);return value;}
std::vector<Vertex> grid(bool owner,bool uv){
 std::vector<Vertex> out;
 for(int y=0;y<3;++y)for(int x=0;x<3;++x)for(int corner=0;corner<2;++corner){
  Vertex v{};v.world_x=3.f+x*.25f+corner*.125f;v.world_y=7.f-y*.25f;v.world_z=.01f*float(x+y);v.world_valid=1+.5f*corner;
  v.x=100.f+x;v.y=50.f+y;v.z=9.f;v.u=v.world_x;v.v=v.world_y;
  v.normal_x=.1f*x;v.normal_y=.2f;v.normal_z=.9f;v.material_grass=.25f;v.material_plains=.5f+x;v.material_desert=.125f*y;
  v.material_marsh=.0625f;v.authored_relief_height=1.5f+y;v.authored_relief_blend=.75f;v.base_terrain=float(x+2*y);
  if(owner && x==1 && y==2)v.relief_owner_coverage=.5f;
  if(uv && x==2 && y==1)v.u=v.world_x+1.f;
  out.push_back(v);out.push_back(v);}
 return out;}
PreparedMesh prepare(std::vector<Vertex> const& source,bool projected){
 MeshFormat f;f.natural=true;f.world_projected=projected;PreparedMesh m;
 assert(prepare_mesh(source,nullptr,f,m,[]{return false;}));return m;}
void same_inputs(Element const* elements,unsigned count,PreparedMesh const& full,PreparedMesh const& compact,bool position_first){
 assert(full.vertices.size()/full.vertex_stride==compact.vertices.size()/compact.vertex_stride);
 for(unsigned i=0;i<full.vertices.size()/full.vertex_stride;++i)
  for(unsigned e=0;e<count;++e)for(unsigned c=0;c<elements[e].components;++c){
   // Position is replaced from world in every natural vertex shader: the
   // compact layout aliases it to world xyz.
   float expected=position_first && e==0?fetch(full,i,12,c,0):fetch(full,i,elements[e].full,c,0);
   float actual=fetch(compact,i,elements[e].compact,c,elements[e].slot);
   assert(std::memcmp(&expected,&actual,4)==0);
  }}
int main(){
 auto source=grid(false,false);
 auto full=prepare(source,false),compact=prepare(source,true);
 assert(full.vertex_stride==natural_vertex_stride && compact.vertex_stride==natural_compact_stride);
 assert(compact.vertices.size()*92==full.vertices.size()*56);
 assert(compact.indices==full.indices && compact.index_count==full.index_count);
 assert(compact.bounds==full.bounds && compact.world_low==full.world_low && compact.world_high==full.world_high);
 assert(std::memcmp(&compact.projected_bounds,&full.projected_bounds,sizeof(full.projected_bounds))==0);
 same_inputs(natural,8,full,compact,true);
 same_inputs(caster,5,full,compact,false);
 same_inputs(border,2,full,compact,true);
 // Any vertex outside the exact rule keeps the whole mesh at 92 bytes.
 assert(prepare(grid(true,false),true).vertex_stride==natural_vertex_stride);
 assert(prepare(grid(false,true),true).vertex_stride==natural_vertex_stride);
 Vertex negative=source[0];negative.relief_owner_u=-0.f;
 assert(!natural_vertex_compact(negative));
 // Only the pickup natural format compacts.
 MeshFormat legacy;legacy.pickup=false;legacy.natural=true;legacy.world_projected=true;PreparedMesh m;
 assert(prepare_mesh(source,nullptr,legacy,m,[]{return false;}) && m.vertex_stride==120u);
}
''')

    def test_every_natural_consumer_binds_the_compact_layout(self):
        shadow = (NATIVE / 'render_core/source_shadow.h').read_text()
        fresh = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        borders = (NATIVE / 'gpu_territory_borders.h').read_text()
        renderer = (NATIVE / 'c3x_renderer.cpp').read_text()
        self.assertTrue('elements[0].AlignedByteOffset=0;elements[1].AlignedByteOffset=52;elements[2].AlignedByteOffset=0;\n'
                        '            elements[3].AlignedByteOffset=36;elements[4].AlignedByteOffset=0;elements[4].InputSlot=1;' in shadow,
                        'source_shadow.h compact caster layout')
        self.assertTrue('e[0].AlignedByteOffset=0;e[1].AlignedByteOffset=52;e[2].AlignedByteOffset=0;\n'
                        '        e[3].AlignedByteOffset=36;e[4].AlignedByteOffset=0;e[4].InputSlot=1;' in fresh,
                        'fresh_pipeline.h compact caster layout')
        for name, text in (('source_shadow.h', shadow), ('fresh_pipeline.h', fresh)):
            self.assertTrue('stride==56?natural_compact_layout' in text, name + ' selects the compact caster layout')
            self.assertTrue('IASetVertexBuffers(1,1,&zero_stream' in text, name + ' binds the zero owner stream')
        self.assertTrue('elements[1].AlignedByteOffset=0;\n        if(FAILED(device->CreateInputLayout(elements,2,'
                        'vs->GetBufferPointer(),vs->GetBufferSize(),&compact_layout)))' in borders)
        self.assertTrue('vertex_stride==56?compact_layout.Get()' in borders)
        # Both chunk draw loops pick the layout per chunk stride.
        self.assertTrue('if(layer>=geometry_natural_terrain)natural.bind_format(context,stride);' in renderer)
        self.assertTrue('if(layer>=geometry_natural_terrain)renderer.natural.bind_format(context,stride);' in fresh)

    def test_only_world_projected_meshes_compact(self):
        renderer = (NATIVE / 'c3x_renderer.cpp').read_text()
        compiler = (NATIVE / 'source_fidelity/terrain_compiler.h').read_text()
        self.assertTrue('format.world_projected=natural_vertex && projection_kind && !record && compact_natural_vertices;' in renderer)
        self.assertTrue('if(mesh.vertex_stride==c3x_renderer::render_core::natural_compact_stride && '
                        '(!projection_kind || record))return false;' in renderer)
        self.assertTrue('input.world_projected=world_content && compact_natural_vertices;' in renderer)
        self.assertTrue('std::uint64_t(input.world_projected)*32+' in renderer, 'terrain keys separate the two formats')
        self.assertTrue('format.world_projected=input.world_projected;' in compiler)
        # Natural chunks with a projection kind always receive natural_projection.
        self.assertTrue('if(natural_world || source_chunk.projection_kind || source_chunk.instances)' in renderer)
        self.assertTrue('chunk.natural_projection[2]=float(shadow_tile_width);' in renderer)


if __name__ == '__main__':
    unittest.main()
