"""Stage 5 exact compaction: integrated ground meshes elide constant elements.

An integrated (168-byte) mesh drops every input element whose bytes are the
same on all its vertices; the chunk keeps the full 168-byte constant row
beside its vertices. Input layouts read an elided element from that row at a
stride-0 slot 1, and the pulled shader reads it from the row's page copy. Each
consumer must therefore fetch exactly the bytes of the full format.
"""
import re
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp

NATIVE = ROOT / 'Renderer/native'
WIDTH = {'DXGI_FORMAT_R32_FLOAT': 4, 'DXGI_FORMAT_R32G32_FLOAT': 8,
         'DXGI_FORMAT_R32G32B32_FLOAT': 12, 'DXGI_FORMAT_R32G32B32A32_FLOAT': 16}


def layout(text):
    return [(int(o), WIDTH[f]) for f, o in
            re.findall(r'\{\s*"\w+",\s*\d+,\s*(DXGI_FORMAT_\w+),\s*0,\s*(\d+),\s*D3D11_INPUT_PER_VERTEX_DATA,\s*0\}', text)]


def table(rows):
    return ','.join('{%d,%d}' % row for row in rows)


class IntegratedElisionTests(unittest.TestCase):
    def test_every_consumer_fetches_the_full_format(self):
        renderer = (NATIVE / 'c3x_renderer.cpp').read_text()
        main = layout(renderer.split('integrated_input_elements() {', 1)[1].split('}};', 1)[0])
        self.assertEqual(len(main), 19)
        borders = layout((NATIVE / 'gpu_territory_borders.h').read_text().split(
            'ID3D11InputLayout* elided_layout(', 1)[1].split('};', 1)[0])
        self.assertEqual(borders, [(0, 12), (120, 12)])
        # The ordinary shadow caster's integrated layout (uv, material, world,
        # coverage, owner); elided ground never casts, but the mapping holds.
        caster = [(12, 8), (60, 4), (120, 16), (76, 4), (152, 16)]
        fresh = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        helper = fresh.split('std::string const field_address=R"(', 1)[1].split(')";', 1)[0]
        helper = helper.split('uint c3x_pulled_field(', 1)[1]
        helper = 'uint c3x_pulled_field(' + helper.replace('[unroll]', '')
        run_cpp(r'''
#include <array>
#include <cmath>
#include <cassert>
#include <cstring>
#include "Renderer/native/render_core/prepared_mesh.h"
using namespace c3x_renderer::render_core;
using uint=unsigned;
// The pulled shader's address helper, compiled as C++.
static uint c3x_pulled_elided,c3x_pulled_template;
static uint c3x_pulled_element_bytes[19];
''' + helper + r'''
struct Element {unsigned offset,bytes;};
Element main_layout[]={''' + table(main) + r'''},border[]={''' + table(borders) + r'''},caster[]={''' + table(caster) + r'''};
struct Desc {unsigned AlignedByteOffset,InputSlot;unsigned bytes;};
std::vector<Vertex> mesh(bool river){
 std::vector<Vertex> out;
 for(int y=0;y<5;++y)for(int x=0;x<5;++x){
  Vertex v{};v.world_x=3+x*.25f;v.world_y=7-y*.25f;v.world_z=2.5f/112;v.world_valid=1;
  v.x=x*32.f;v.y=y*16.f;v.z=v.y;v.u=.3f*x;v.v=.7f*y;v.panel=1;v.normal_z=1;
  v.shadow_visibility=v.ambient_visibility=1;v.macro_u=v.world_x*.5f;v.macro_v=v.world_y*.5f;
  v.surface_kind=.5f;v.surface_coordinate=.1f*x;v.base_terrain=2;v.real_terrain=3;
  v.material_grass=.1f*x;v.material_plains=.2f;v.material_desert=.05f*y;v.material_marsh=0;
  v.shore_distance=float(x+y);v.river_distance=v.river_branch_count=v.river_mouth_distance=v.river_padding=1000;
  v.active_effect=1;v.material_tundra=.01f*y;v.shore_true_distance=float(x);v.shore_beach_width=.2f;v.shore_rockiness=.1f*y;v.shore_depth=0;
  if(river){v.river_distance=float(x);v.river_padding=float(y);v.normal_x=.1f*x;v.relief_owner_u=.2f*y;}
  out.push_back(v);}
 return out;}
PreparedMesh prepare(std::vector<Vertex> const& source,bool elide,unsigned kind=3){
 MeshFormat f;f.projection_kind=kind;f.elide_constants=elide;PreparedMesh m;
 std::vector<unsigned> topology;for(unsigned i=0;i+2<source.size();++i){topology.push_back(i);topology.push_back(i+1);topology.push_back(i+2);}
 assert(prepare_mesh(source,&topology,f,m,[]{return false;}));return m;}
void same(PreparedMesh const& full,PreparedMesh const& compact,Element const* layout,unsigned count){
 unsigned vertices=unsigned(full.vertices.size()/full.vertex_stride);
 assert(compact.vertices.size()==std::size_t(vertices)*compact.vertex_stride);
 for(unsigned e=0;e<count;++e){
  Desc d{layout[e].offset,0,layout[e].bytes};
  assert(remap_integrated_layout(&d,1,compact.elided,[](Desc const& x){return x.bytes;}));
  for(unsigned i=0;i<vertices;++i){
   auto const* expected=full.vertices.data()+i*full.vertex_stride+layout[e].offset;
   auto const* actual=d.InputSlot?compact.constants.data()+d.AlignedByteOffset:
    compact.vertices.data()+i*compact.vertex_stride+d.AlignedByteOffset;
   assert(std::memcmp(expected,actual,layout[e].bytes)==0);}
 }}
void same_pulled(PreparedMesh const& full,PreparedMesh const& compact){
 // A page holds the compact vertices followed by the constant row.
 std::vector<unsigned char> page(compact.vertices.begin(),compact.vertices.end());
 page.insert(page.end(),compact.constants.begin(),compact.constants.end());
 c3x_pulled_elided=compact.elided;c3x_pulled_template=unsigned(compact.vertices.size());
 for(unsigned e=0;e<integrated_element_count;++e)c3x_pulled_element_bytes[e]=integrated_elements[e].bytes;
 unsigned vertices=unsigned(full.vertices.size()/full.vertex_stride);
 for(unsigned i=0;i<vertices;++i)for(unsigned e=0;e<integrated_element_count;++e){
  auto const& d=integrated_elements[e];uint at=c3x_pulled_field(i*compact.vertex_stride,e,d.offset);
  assert(at+d.bytes<=page.size() && std::memcmp(full.vertices.data()+i*full.vertex_stride+d.offset,page.data()+at,d.bytes)==0);}}
int main(){
 for(bool river:{false,true})for(unsigned kind:{0u,3u}){
  auto source=mesh(river);auto full=prepare(source,false,kind),compact=prepare(source,true,kind);
  assert(full.vertex_stride==integrated_vertex_bytes && !full.elided);
  assert(compact.elided && !(compact.elided&1u) && compact.vertex_stride==integrated_stride(compact.elided));
  assert(compact.vertex_stride<full.vertex_stride && compact.indices==full.indices && compact.bounds==full.bounds);
  same(full,compact,main_layout,19);same(full,compact,border,2);same(full,compact,caster,5);same_pulled(full,compact);
 }
 // The underlay keeps only its varying elements: 168 -> 88 bytes here.
 auto underlay=prepare(mesh(false),true);assert(underlay.vertex_stride==88);
 // A mesh whose every element varies stays full.
 auto varied=mesh(true);for(std::size_t i=0;i<varied.size();++i){auto& v=varied[i];float f=float(i+1);
  v.panel=v.normal_y=v.shadow_visibility=v.surface_kind=v.base_terrain=v.real_terrain=v.authored_relief_height=
  v.active_effect=v.river_mouth_distance=v.relief_owner_state=f;}
 auto kept=prepare(varied,true);assert(!kept.elided && kept.vertex_stride==integrated_vertex_bytes);
 // Natural, feature and city formats never elide.
 MeshFormat natural;natural.natural=true;natural.elide_constants=true;PreparedMesh n;
 assert(prepare_mesh(mesh(false),nullptr,natural,n,[]{return false;}) && !n.elided);
}
''')

    def test_codec_keeps_the_format_and_rejects_older_entries(self):
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/world_backing_codec.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 PreparedWorld ground;ground.kind=WorldPreparationKind::ground;
 ground.ground=std::make_unique<fidelity::PreparedGround>();ground.terrain=std::make_unique<fidelity::TerrainSurfaces>();
 auto& m=ground.ground->meshes[0];m.elided=(1u<<2)|(1u<<3);m.vertex_stride=render_core::integrated_stride(m.elided);
 m.index_stride=2;m.vertices.assign(m.vertex_stride*2,7);m.constants.fill(9);m.constants[20]=1;m.template_offset=44;
 auto bytes=WorldBackingCodec::encode(ground);assert(!bytes.empty());
 auto restored=WorldBackingCodec::decode(bytes,WorldPreparationKind::ground);assert(restored);
 auto const& r=restored->ground->meshes[0];
 assert(r.elided==m.elided && r.constants==m.constants && r.vertices==m.vertices && r.vertex_stride==m.vertex_stride);
 assert(r.template_offset==0); // GPU placement belongs to the uploader
 auto old=bytes;std::uint32_t five=5;std::memcpy(old.data(),&five,4);assert(!WorldBackingCodec::decode(old,WorldPreparationKind::ground));
 // A stride that disagrees with the mask is rejected.
 ground.ground->meshes[0].vertex_stride=168;ground.ground->meshes[0].vertices.assign(168*2,7);
 assert(!WorldBackingCodec::decode(WorldBackingCodec::encode(ground),WorldPreparationKind::ground));
}
''')

    def test_consumers_test_the_mask_before_the_stride(self):
        renderer = (NATIVE / 'c3x_renderer.cpp').read_text()
        fresh = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        borders = (NATIVE / 'gpu_territory_borders.h').read_text()
        pages = (NATIVE / 'render_core/pulled_mesh_pages.h').read_text()
        self.assertTrue('else if(pickup_profile && layer<geometry_feature && !bind_integrated(chunk.content(),integrated_bound))' in renderer)
        self.assertTrue('else if(renderer.pickup_profile && layer<geometry_feature && !renderer.bind_integrated(mesh,integrated_bound))' in fresh)
        self.assertTrue('auto* layout=chunk.elided?elided_layout(device,chunk.elided):' in borders)
        self.assertTrue('chunk.content().elided)continue;' in renderer, 'elided ground never casts')
        self.assertTrue('(m.vertex_stride==pulled_stride || (!pulled_feature_layer && m.elided))' in fresh)
        self.assertTrue('source.elided=m.elided;source.template_offset=m.template_offset;' in fresh)
        self.assertTrue('r.elided=s.elided;r.template_byte=' in pages and 's.elided?row_bytes:0' in pages)
        for site in ('if(mesh.elided)chunk.template_offset=upload.append(mesh.constants.data(),mesh.constants.size());',
                     'if(mesh.elided)mesh.template_offset=upload.append(mesh.constants.data(),mesh.constants.size());',
                     'if(part.mesh.elided)part.mesh.template_offset=upload.append(part.mesh.constants.data(),part.mesh.constants.size());'):
            self.assertTrue(site in renderer, site)


if __name__ == '__main__':
    unittest.main()
