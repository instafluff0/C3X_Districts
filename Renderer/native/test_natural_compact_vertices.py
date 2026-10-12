"""Stage 5 natural vertex formats give the shader inputs of the 92-byte format.

Step 1 (exact): a world-projected natural mesh whose uv equals world xy and
whose relief owner is zero, bit for bit, drops position, uv and owner
(92 -> 56 bytes). Its input layouts alias position and uv to world (the vertex
shaders replace position from world whenever natural_projection.z > 0) and read
the owner from a stride-0 zero stream.

Step 2 (quantized, approved October 11): world-projected meshes store the
normal as SNORM16x4 and the material, relief and owner channels as half floats
(92 -> 68, 56 -> 40 bytes), expanded by the input assembler. World, uv,
position and base terrain stay 32-bit and must match bit for bit; quantized
inputs must equal the input assembler's conversion of the exact value.
"""
import re
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp

NATIVE = ROOT / 'Renderer/native'
COMPONENTS = {'DXGI_FORMAT_R32_FLOAT': 1, 'DXGI_FORMAT_R32G32_FLOAT': 2,
              'DXGI_FORMAT_R32G32B32_FLOAT': 3, 'DXGI_FORMAT_R32G32B32A32_FLOAT': 4}


def natural_components():
    """Shader components of each natural input, in natural_offsets order."""
    source = (NATIVE / 'source_fidelity/runtime.h').read_text()
    block = re.search(r'D3D11_INPUT_ELEMENT_DESC e\[\]=\{(.*?)\};', source, re.S).group(1)
    full = [(COMPONENTS[f], int(o)) for f, o in
            re.findall(r'\{"\w+",\d+,(DXGI_FORMAT_\w+),0,(\d+),D3D11_INPUT_PER_VERTEX_DATA,0\}', block)]
    assert [o for _, o in full] == [0, 12, 28, 40, 48, 64, 72, 76], full
    assert 'unsigned offsets[]={o.position,o.world,o.normal,o.uv,o.material,o.relief,o.base,o.owner};' in source
    return [n for n, _ in full]


def table(rows):
    return ','.join('{%s}' % ','.join(str(v) for v in row) for row in rows)


class NaturalCompactVertexTests(unittest.TestCase):
    def test_every_format_matches_the_full_inputs(self):
        components = natural_components()
        run_cpp(r'''
#include <array>
#include <cmath>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <vector>
#include "Renderer/native/render_core/prepared_mesh.h"
using namespace c3x_renderer::render_core;
enum Kind {f32,snorm,f16,zero};
unsigned const components[]={''' + ','.join(map(str, components)) + r'''};
float half_value(std::uint16_t h){
 int exponent=(h>>10)&31;float mantissa=float(h&1023);
 float magnitude=exponent==0?std::ldexp(mantissa,-24):exponent==31?INFINITY:std::ldexp(1024.f+mantissa,exponent-25);
 return h&0x8000?-magnitude:magnitude;}
// The input assembler's conversion, from first principles: the nearest half
// (ties to the even code) and the nearest SNORM16 step.
float nearest_half(float v){
 static std::vector<std::pair<float,std::uint16_t>> codes=[]{std::vector<std::pair<float,std::uint16_t>> r;
  for(unsigned h=0;h<0x7c00;++h)r.push_back({half_value(std::uint16_t(h)),std::uint16_t(h)});return r;}();
 float a=std::fabs(v);auto it=std::lower_bound(codes.begin(),codes.end(),std::make_pair(a,std::uint16_t(0)));
 auto best=it==codes.end()?codes.back():*it;
 if(it!=codes.begin()){auto below=*(it-1);float db=a-below.first,da=best.first-a;
  if(db<da || (db==da && (below.second&1)==0))best=below;}
 return v<0?-best.first:best.first;}
float fetch(PreparedMesh const& m,unsigned vertex,unsigned offset,unsigned component,Kind kind){
 auto const* p=m.vertices.data()+vertex*m.vertex_stride+offset;
 if(kind==zero)return 0.f; // the stride-0 zero stream
 if(kind==snorm){std::int16_t value;std::memcpy(&value,p+component*2,2);return std::max(float(value)/32767.f,-1.f);}
 if(kind==f16){std::uint16_t value;std::memcpy(&value,p+component*2,2);return half_value(value);}
 float value;std::memcpy(&value,p+component*4,4);return value;}
void same(float expected,float actual,Kind kind){
 if(kind==f32 || kind==zero){assert(std::memcmp(&expected,&actual,4)==0);return;}
 if(kind==f16){assert(actual==nearest_half(expected));return;}
 assert(std::fabs(actual-expected)<=.5f/32767.f+1e-7f);}
std::vector<Vertex> grid(bool owner,bool uv){
 std::vector<Vertex> out;
 for(int y=0;y<3;++y)for(int x=0;x<3;++x)for(int corner=0;corner<2;++corner){
  Vertex v{};v.world_x=3.f+x*.25f+corner*.125f;v.world_y=7.f-y*.25f;v.world_z=.0137f*float(x+y);v.world_valid=1+.5f*corner;
  v.x=100.f+x;v.y=50.f+y;v.z=9.f;v.u=v.world_x;v.v=v.world_y;
  float n[]={.1371f*x-.13f,.2113f*corner-.07f,.9f};float l=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
  v.normal_x=n[0]/l;v.normal_y=n[1]/l;v.normal_z=n[2]/l;
  v.material_grass=.2371f*x+.0137f*y-.05f;v.material_plains=x==0?1.f+.0731f*y:x==1?2.5f:5.f+y;
  v.material_desert=.1251f*y+.003f*corner;v.material_marsh=.0625f+.311f*x;
  v.authored_relief_height=13.37f+y;v.authored_relief_blend=.7513f-.1f*x;v.base_terrain=x==2?42.f+.3713f*y:-10.f+.2f*x;
  if(owner && x==1){v.relief_owner_u=-.4137f+.1f*y;v.relief_owner_v=.6251f;v.relief_owner_coverage=.5f+.01f*corner;v.relief_owner_state=float(2*y+1)+.25f;}
  if(uv && x==2 && y==1)v.u=v.world_x+1.f;
  out.push_back(v);out.push_back(v);}
 return out;}
PreparedMesh prepare(std::vector<Vertex> const& source,bool projected,bool quantized){
 MeshFormat f;f.natural=true;f.world_projected=projected;f.quantized=quantized;PreparedMesh m;
 assert(prepare_mesh(source,nullptr,f,m,[]{return false;}));return m;}
// Main natural layout, in natural_offsets order: position, world, normal, uv,
// material, relief, base terrain, owner.
void main_inputs(PreparedMesh const& full,PreparedMesh const& actual){
 auto o=natural_offsets(actual.vertex_stride),f=natural_offsets(full.vertex_stride);
 unsigned offsets[]={o.position,o.world,o.normal,o.uv,o.material,o.relief,o.base,o.owner};
 unsigned full_offsets[]={f.position,f.world,f.normal,f.uv,f.material,f.relief,f.base,f.owner};
 Kind kinds[]={f32,f32,o.quantized?snorm:f32,f32,o.quantized?f16:f32,o.quantized?f16:f32,f32,o.zero_owner?zero:o.quantized?f16:f32};
 // Position is replaced from world in every natural vertex shader: compact
 // formats alias it to world xyz.
 if(o.zero_owner)full_offsets[0]=f.world;
 for(unsigned i=0;i<full.vertices.size()/full.vertex_stride;++i)
  for(unsigned e=0;e<8;++e)for(unsigned c=0;c<components[e];++c)
   same(fetch(full,i,full_offsets[e],c,f32),fetch(actual,i,offsets[e],c,kinds[e]),kinds[e]);}
// Shadow casters: uv, base terrain, world, coverage (the desert channel), owner.
void caster_inputs(PreparedMesh const& full,PreparedMesh const& actual){
 auto o=natural_offsets(actual.vertex_stride);
 unsigned counts[]={2,1,4,1,4},full_offsets[]={40,72,12,56,76};
 unsigned offsets[]={o.uv,o.base,o.world,o.material+(o.quantized?4u:8u),o.owner};
 Kind kinds[]={f32,f32,f32,o.quantized?f16:f32,o.zero_owner?zero:o.quantized?f16:f32};
 for(unsigned i=0;i<full.vertices.size()/full.vertex_stride;++i)
  for(unsigned e=0;e<5;++e)for(unsigned c=0;c<counts[e];++c)
   same(fetch(full,i,full_offsets[e],c,f32),fetch(actual,i,offsets[e],c,kinds[e]),kinds[e]);}
// Territory borders read world: at 12 through the natural layout, at 0
// through the compact layout (chosen by natural_offsets(stride).world).
void border_inputs(PreparedMesh const& full,PreparedMesh const& actual){
 unsigned offset=natural_offsets(actual.vertex_stride).world==0?0u:12u;
 for(unsigned i=0;i<full.vertices.size()/full.vertex_stride;++i)for(unsigned c=0;c<3;++c)
  same(fetch(full,i,12,c,f32),fetch(actual,i,offset,c,f32),f32);}
void check(PreparedMesh const& full,PreparedMesh const& actual,unsigned stride){
 assert(full.vertex_stride==natural_vertex_stride && actual.vertex_stride==stride);
 assert(actual.vertices.size()*92==full.vertices.size()*stride);
 assert(actual.indices==full.indices && actual.index_count==full.index_count);
 assert(actual.bounds==full.bounds && actual.world_low==full.world_low && actual.world_high==full.world_high);
 assert(std::memcmp(&actual.projected_bounds,&full.projected_bounds,sizeof(full.projected_bounds))==0);
 main_inputs(full,actual);caster_inputs(full,actual);border_inputs(full,actual);}
int main(){
 // The reference encoder agrees with production on every half code's
 // neighbourhood (ties, subnormals, the largest finite value).
 for(unsigned h=0;h<0x7c00;h+=7){float v=half_value(std::uint16_t(h));
  for(float probe:{v,std::nextafter(v,0.f),std::nextafter(v,INFINITY),-v}){
   if(std::fabs(probe)>65504.f)continue;
   assert(half_value(half_bits(probe))==nearest_half(probe));}}
 for(unsigned h=1;h<0x7bff;h+=13){float mid=(half_value(std::uint16_t(h))+half_value(std::uint16_t(h+1)))*.5f;
  assert(half_value(half_bits(mid))==nearest_half(mid));}
 // Class codes stay exact: thresholds at x.5 and owner activity.
 for(float code:{1.f,1.25f,2.f,2.5f,5.f,6.f,7.f,8.f,3.25f,42.f})assert(half_value(half_bits(code))==code);
 auto source=grid(false,false),owned=grid(true,true);
 auto full=prepare(source,false,false),owned_full=prepare(owned,false,false);
 check(full,prepare(source,true,false),natural_compact_stride);
 check(full,prepare(source,true,true),natural_quantized_compact_stride);
 check(owned_full,prepare(owned,true,true),natural_quantized_stride);
 // Without world projection nothing compacts or quantizes.
 assert(prepare(source,false,true).vertex_stride==natural_vertex_stride);
 // Any vertex outside the exact rule keeps the full layout.
 assert(prepare(grid(true,false),true,false).vertex_stride==natural_vertex_stride);
 assert(prepare(grid(false,true),true,false).vertex_stride==natural_vertex_stride);
 Vertex negative=source[0];negative.relief_owner_u=-0.f;
 assert(!natural_vertex_compact(negative));
 // A value outside the 16-bit range keeps the exact format.
 auto wide=source;wide[3].normal_x=1.0001f;
 assert(prepare(wide,true,true).vertex_stride==natural_compact_stride);
 wide=source;wide[5].authored_relief_height=70000.f;
 assert(prepare(wide,true,true).vertex_stride==natural_compact_stride);
 // Only the pickup natural format compacts.
 MeshFormat legacy;legacy.pickup=false;legacy.natural=true;legacy.world_projected=true;legacy.quantized=true;PreparedMesh m;
 assert(prepare_mesh(source,nullptr,legacy,m,[]{return false;}) && m.vertex_stride==120u);
 for(unsigned stride:{natural_vertex_stride,natural_compact_stride,natural_quantized_stride,natural_quantized_compact_stride})
  assert(natural_stride(stride) && natural_format_index(stride)==(stride==92?0u:stride==56?1u:stride==68?2u:3u));
 assert(!natural_stride(48) && !natural_stride(88) && !natural_stride(168));
}
''')

    def test_every_natural_consumer_binds_the_format_layouts(self):
        runtime = (NATIVE / 'source_fidelity/runtime.h').read_text()
        shadow = (NATIVE / 'render_core/source_shadow.h').read_text()
        fresh = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        borders = (NATIVE / 'gpu_territory_borders.h').read_text()
        renderer = (NATIVE / 'c3x_renderer.cpp').read_text()
        # The main layouts: offsets from natural_offsets, 16-bit formats where
        # the format is quantized (the kinds main_inputs assumes).
        for line in ('e[7].InputSlot=o.zero_owner?1:0;',
                     'e[2].Format=o.quantized?DXGI_FORMAT_R16G16B16A16_SNORM:DXGI_FORMAT_R32G32B32_FLOAT;',
                     'e[4].Format=o.quantized?DXGI_FORMAT_R16G16B16A16_FLOAT:DXGI_FORMAT_R32G32B32A32_FLOAT;',
                     'e[5].Format=o.quantized?DXGI_FORMAT_R16G16_FLOAT:DXGI_FORMAT_R32G32_FLOAT;',
                     'e[7].Format=o.quantized && !o.zero_owner?DXGI_FORMAT_R16G16B16A16_FLOAT:DXGI_FORMAT_R32G32B32A32_FLOAT;',
                     'auto* wanted=format?format_layout[bound_provider][format-1]:layout[bound_provider];',
                     'if(render_core::natural_offsets(stride).zero_owner){UINT none=0;c->IASetVertexBuffers(1,1,&zero_stream'):
            self.assertIn(line, runtime)
        # The caster layouts (the kinds caster_inputs assumes), legacy and fresh.
        for name, text, e in (('source_shadow.h', shadow, 'elements'), ('fresh_pipeline.h', fresh, 'e')):
            for line in ('%s[0].AlignedByteOffset=o.uv;%s[1].AlignedByteOffset=o.base;' % (e, e),
                         '%s[3].AlignedByteOffset=o.material+(o.quantized?4:8);' % e,
                         '%s[3].Format=o.quantized?DXGI_FORMAT_R16G16_FLOAT:DXGI_FORMAT_R32_FLOAT;' % e,
                         '%s[4].AlignedByteOffset=o.owner;%s[4].InputSlot=o.zero_owner?1:0;' % (e, e),
                         '%s[4].Format=o.quantized && !o.zero_owner?DXGI_FORMAT_R16G16B16A16_FLOAT:DXGI_FORMAT_R32G32B32A32_FLOAT;' % e,
                         '&natural_layouts[f]);'):
                self.assertIn(line, text, name)
            self.assertIn('natural_offsets(', text, name)
            self.assertIn('IASetVertexBuffers(1,1,&zero_stream', text, name + ' binds the zero owner stream')
        self.assertIn('?natural_layouts[natural_format_index(c.stride)]', shadow)
        self.assertIn('natural?natural_layouts[c3x_renderer::render_core::natural_format_index(caster.stride)]', fresh)
        self.assertIn('elements[1].AlignedByteOffset=0;\n        if(FAILED(device->CreateInputLayout(elements,2,'
                      'vs->GetBufferPointer(),vs->GetBufferSize(),&compact_layout)))', borders)
        self.assertIn('render_core::natural_offsets(chunk.vertex_stride).world==0?compact_layout.Get():natural_layout.Get();', borders)
        # Both chunk draw loops pick the layout per chunk stride.
        self.assertIn('if(layer>=geometry_natural_terrain)natural.bind_format(context,stride);', renderer)
        self.assertIn('if(layer>=geometry_natural_terrain)renderer.natural.bind_format(context,stride);', fresh)

    def test_only_world_projected_meshes_compact(self):
        renderer = (NATIVE / 'c3x_renderer.cpp').read_text()
        compiler = (NATIVE / 'source_fidelity/terrain_compiler.h').read_text()
        self.assertIn('format.world_projected=natural_vertex && projection_kind && !record && compact_vertices;\n'
                      '            format.quantized=quantized_vertices;', renderer)
        self.assertIn('if(natural_vertex && mesh.vertex_stride!=c3x_renderer::render_core::natural_vertex_stride && '
                      '(!projection_kind || record))return false;', renderer)
        self.assertIn('input.world_projected=world_content && compact_vertices;input.quantized=quantized_vertices;', renderer)
        self.assertIn('std::uint64_t(input.world_projected && input.quantized)*64+std::uint64_t(input.world_projected)*32+', renderer,
                      'terrain keys separate the formats')
        self.assertIn('format.world_projected=input.world_projected;format.quantized=input.quantized;', compiler)
        # Natural chunks with a projection kind always receive natural_projection.
        self.assertIn('if(natural_world || source_chunk.projection_kind || source_chunk.instances)', renderer)
        self.assertIn('chunk.natural_projection[2]=float(shadow_tile_width);', renderer)


if __name__ == '__main__':
    unittest.main()
