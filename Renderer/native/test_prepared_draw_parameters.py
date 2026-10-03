"""Execute production prepared water submission against streaming draw receipts."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method


GPU_STUB = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
using UINT=unsigned;using LONG=int;using HRESULT=int;
constexpr int FALSE=0,D3D11_FEATURE_D3D11_OPTIONS=1,D3D11_USAGE_DYNAMIC=1,
 D3D11_USAGE_IMMUTABLE=2,D3D11_BIND_CONSTANT_BUFFER=4,D3D11_CPU_ACCESS_WRITE=8,
 D3D11_MAP_WRITE_DISCARD=1,D3D11_MAP_WRITE_NO_OVERWRITE=2;
bool FAILED(HRESULT h){return h<0;}bool SUCCEEDED(HRESULT h){return h>=0;}
#define __uuidof(...) 1
#define C3X_RENDERER64_FRESH 1
struct D3D11_RECT {LONG left,top,right,bottom;};
struct D3D11_BUFFER_DESC {unsigned ByteWidth=0,Usage=0,BindFlags=0,CPUAccessFlags=0;};
struct D3D11_SUBRESOURCE_DATA {void const* pSysMem=nullptr;};
struct D3D11_MAPPED_SUBRESOURCE {void* pData=nullptr;};
struct D3D11_FEATURE_DATA_D3D11_OPTIONS {bool ConstantBufferOffsetting=true,MapNoOverwriteOnDynamicConstantBuffer=true;};
unsigned buffers_live=0,buffers_released=0;
struct ID3D11Buffer {
 std::vector<unsigned char> bytes;unsigned id=0;bool counted=false;
 void Release(){assert(counted&&buffers_live);--buffers_live;++buffers_released;delete this;}
};
struct ID3D11Device {
 bool fail=false,offsetting=true;unsigned creates=0;
 HRESULT CheckFeatureSupport(int,D3D11_FEATURE_DATA_D3D11_OPTIONS* o,unsigned){o->ConstantBufferOffsetting=offsetting;return 0;}
 HRESULT CreateBuffer(D3D11_BUFFER_DESC const* d,D3D11_SUBRESOURCE_DATA const* data,ID3D11Buffer** output){
  if(fail)return -1;
  *output=new ID3D11Buffer;(*output)->bytes.resize(d->ByteWidth);(*output)->counted=true;
  if(data)std::memcpy((*output)->bytes.data(),data->pSysMem,d->ByteWidth);++creates;++buffers_live;return 0;
 }
 long GetDeviceRemovedReason(){return 0;}
};
struct ViewportShaderSettings {
 float translation[2]={},depth_translation=0,padding=0,inverse_size[2]={},reserved[2]={},natural_projection[4]={};
};
struct DrawReceipt {
 unsigned mesh=0,count=0,instances=0;ViewportShaderSettings parameters{};std::array<float,8> material{};
 bool operator==(DrawReceipt const& other)const{return mesh==other.mesh&&count==other.count&&instances==other.instances&&
  !std::memcmp(&parameters,&other.parameters,sizeof(parameters))&&!std::memcmp(material.data(),other.material.data(),sizeof(material));}
};
struct ID3D11DeviceContext1 {
 ID3D11Buffer *viewport=nullptr,*vertex=nullptr;unsigned first=0,maps=0,bindings=0,material_updates=0;
 std::array<float,8> material{};std::vector<DrawReceipt> draws;
 HRESULT QueryInterface(int,void** result){*result=this;return 0;}void Release(){}
 HRESULT Map(ID3D11Buffer* b,int,int,int,D3D11_MAPPED_SUBRESOURCE* m){m->pData=b->bytes.data();++maps;return 0;}
 void Unmap(ID3D11Buffer*,int){}
 void VSSetConstantBuffers1(unsigned slot,unsigned count,ID3D11Buffer* const* b,UINT const* offset,UINT const* size){
  assert(slot==1&&count==1&&*size==16);viewport=*b;first=*offset*16;++bindings;
 }
 void VSSetConstantBuffers(unsigned slot,unsigned,ID3D11Buffer* const* b){if(slot==1){viewport=*b;first=0;}}
 void UpdateSubresource(ID3D11Buffer* b,int,void*,void const* data,int,int){
  std::memcpy(b->bytes.data(),data,b->bytes.size());if(b->id==3){std::memcpy(material.data(),data,32);++material_updates;}
 }
 void IASetVertexBuffers(unsigned,unsigned,ID3D11Buffer* const* b,UINT*,UINT*){vertex=*b;}
 void IASetIndexBuffer(ID3D11Buffer*,unsigned,unsigned){}
 template<class... A>void VSSetShaderResources(A... ){}
 template<class... A>void PSSetShaderResources(A... ){}
 template<class... A>void PSSetConstantBuffers(A... ){}
 template<class... A>void PSSetSamplers(A... ){}
 template<class... A>void IASetInputLayout(A... ){}
 template<class... A>void VSSetShader(A... ){}
 template<class... A>void PSSetShader(A... ){}
 template<class... A>void OMSetBlendState(A... ){}
 template<class... A>void OMSetDepthStencilState(A... ){}
 void draw(unsigned count,unsigned instances){
  assert(viewport&&first+sizeof(ViewportShaderSettings)<=viewport->bytes.size());DrawReceipt d;
  d.mesh=vertex?vertex->id:0;d.count=count;d.instances=instances;d.material=material;
  std::memcpy(&d.parameters,viewport->bytes.data()+first,sizeof(d.parameters));draws.push_back(d);
 }
 void DrawIndexed(unsigned count,int,int){draw(count,1);}
 void DrawIndexedInstanced(unsigned count,unsigned instances,int,int,int){draw(count,instances);}
};
using ID3D11DeviceContext=ID3D11DeviceContext1;
using ID3D11SamplerState=int;
unsigned GetEnvironmentVariableA(char const*,char*,unsigned){return 0;}
'''


def production_harness():
    fresh = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()
    stream = (ROOT / "Renderer/native/render_core/draw_parameter_stream.h").read_text()
    prepared = (ROOT / "Renderer/native/render_core/prepared_draw_parameters.h").read_text()
    shared = (ROOT / "Renderer/native/render_core/shared_instance_submission.h").read_text()
    issue = method(fresh, "    bool issue_records(")
    grouping = shared[shared.index("inline bool opaque_rigid_material("):shared.rindex("}}")]
    return GPU_STUB + stream.replace("#include <d3d11_1.h>", "") + prepared.replace("#include <d3d11_1.h>", "") + r'''
#include "Renderer/native/render_core/geometry_draws.h"
#include "Renderer/native/render_core/water_material_frame.h"
namespace c3x_renderer {namespace render_core {
''' + grouping + r'''
}}
enum GeometryLayer:unsigned {geometry_underlay,geometry_bed,geometry_water,geometry_river,geometry_wave,geometry_city,geometry_shadow,geometry_layer_count};
struct Mesh {
 struct Bounds {float left=0,top=0,right=10,bottom=10;} bounds;
 struct World {float low[3]={},high[3]={};} world_bounds;
 ID3D11Buffer *buffer=nullptr,*indices=nullptr,*resource_instance=nullptr;
 int* animation_texture=nullptr;
 unsigned vertex_offset=0,index_offset=0,index_count=6,index_format=0,projection_kind=3,vertex_stride=32;
 unsigned city_material=0xffffffffu,city_environment=0,city_atlas=0;
 float instance_material=8,visual_time=-1;bool rigid_source=false;
 int translation_x=0,translation_y=0;float natural_projection[4]={};
};
using GeometryDrawView=c3x_renderer::render_core::GeometryDrawView<Mesh,geometry_layer_count>;
using GeometryDrawReference=GeometryDrawView::Reference;
struct Renderer {
 ID3D11Device device_value;ID3D11DeviceContext1 context_value;
 ID3D11Device* device=&device_value;ID3D11DeviceContext1* context=&context_value;
 std::uint64_t device_generation=1,content_revision=1;
 unsigned content_view_width=1600,content_view_height=1100;
 bool city_profile=true,pickup_profile=true,environment_profile=true,water_scene_active=true;
 unsigned frame_draw_calls=0,culls=0;
 c3x_renderer::render_core::WaterMaterialFrame water_material;
 float wave_time_seconds=1;
 ID3D11Buffer viewport_storage,water_storage,wave_storage;
 ID3D11Buffer *viewport_settings_buffer=&viewport_storage,*water_frame=&water_storage,*wave_frame=&wave_storage,
  *shadow_settings_buffer=nullptr;
 int *feature_input_layout=nullptr,*input_layout=nullptr,*feature_vertex_shader=nullptr,*vertex_shader=nullptr,
  *feature_pixel_shader=nullptr,*resource_input_layout=nullptr,*resource_shadow_vertex_shader=nullptr,*resource_body_vertex_shader=nullptr,
  *blend_state=nullptr,*depth_state=nullptr,*terrain_sampler=nullptr,*decal_sampler=nullptr,*natural_wrap=nullptr,*natural_clamp=nullptr;
 std::array<int*,8> city_emissive_views{},city_base_views{},resource_texture_views{};
 struct Reflection {float height_pixels=1;int *ps[2]={},*vs[2]={};}reflection;
 struct Packets {
  struct Range {explicit operator bool()const{return false;}bool contiguous(Range const&)const{return false;}};
  unsigned uploaded_bytes=0,placement_copies=0;template<class... A>void issue(A... ){}
 }ordered_rigid_packets;int* ordered_rigid_layout=nullptr;
 template<class... A>void prepare_ordered_rigid_packets(A... ){}
 struct Rigid {int* resident_layout=nullptr;int* resident_vertex[2]={};}rigid_sources;
 struct Range {unsigned first=0,count=1;explicit operator bool()const{return count;}};
 struct Front {int* view=nullptr;Range find(unsigned key){return {key,1};}};
 struct Shared {
  ID3D11Buffer* selection_buffer=nullptr;unsigned selection_offset=0,queries=0;
  bool select_indices(ID3D11Device*,ID3D11DeviceContext1*,Front*,unsigned const*,unsigned){++queries;return true;}
 }shared_instances;
 struct Cities {
  struct Material {bool ground=false;};struct Library {std::array<Material,4> materials;}library;
  template<class... A>void bind(A... ){}template<class... A>void bind_emission(A... ){}
  bool emits(unsigned material){return material==1;}
 }cities;
 Renderer(){viewport_storage.bytes.resize(sizeof(ViewportShaderSettings));water_storage.bytes.resize(32);water_storage.id=3;
  wave_storage.bytes.resize(16);water_material.time=7.5f;water_material.drift[1]=.2f;}
 unsigned shared_instance_draw_key(unsigned,GeometryDrawReference const& draw){return draw.content().buffer->id;}
 template<class... A>bool reject_shared_instance_range(A... ){return false;}
 bool chunk_intersects_region(GeometryDrawReference const& draw,ViewportShaderSettings const& viewport,D3D11_RECT rect,bool){
  ++culls;auto left=draw.bounds().left+draw.translation_x()+viewport.translation[0];return left<rect.right&&left+10>rect.left;
 }
};
Renderer* current_renderer=nullptr;
auto& sandbox_active_reflection(){return current_renderer->reflection;}
struct Work {
 bool enabled=true;unsigned uploads=0;std::uint64_t bytes=0;struct Calls {unsigned copies=0;}calls;
 struct Row {unsigned tested_records=0,accepted_records=0,submitted_instances=0;}rows[geometry_layer_count];
 auto& row(unsigned layer){return rows[layer];}
 void upload(unsigned size,unsigned){if(size){++uploads;bytes+=size;}}
 void upload_buffer(ID3D11Buffer* b){++uploads;bytes+=b->bytes.size();}
 void draw(unsigned,unsigned,unsigned){}
};
struct Harness {
 Renderer renderer;Work work;
 c3x_renderer::render_core::DrawParameterStream parameters;
 using PreparedParameters=c3x_renderer::render_core::PreparedDrawParameters<ViewportShaderSettings,geometry_layer_count>;
 PreparedParameters water_parameters;
 GeometryDrawView::Records water_visible;
 std::uint64_t revision=1,visibility_revision=1;float projection_zoom=1;
 Renderer::Front front;Renderer::Front* shared_front=&front;
''' + method(fresh, "    struct PhaseConstantCounts {") + r''';
 PhaseConstantCounts phase_constant_counts;
 std::uint64_t view_revision(){return revision;}
 D3D11_RECT source_bounds(ViewportShaderSettings const&,D3D11_RECT rect,bool){return rect;}
''' + issue + r'''
};
struct Fixture {
 Harness h;
 std::vector<Mesh> meshes;
 std::vector<ID3D11Buffer> vertices;
 ViewportShaderSettings viewport;D3D11_RECT rect={-1000,-1000,10000,10000};
 Fixture(unsigned count=777):meshes(count),vertices(count){
  viewport.translation[0]=7;viewport.translation[1]=-23;viewport.depth_translation=89;
  viewport.inverse_size[0]=1.f/1600;viewport.inverse_size[1]=1.f/1100;
  for(unsigned i=0;i<count;++i){
   vertices[i].id=i+1;auto& mesh=meshes[i];mesh.buffer=&vertices[i];mesh.projection_kind=1+i%4;
   GeometryDrawView::Record r(mesh);r.translation_x=int(i%97)*13;r.translation_y=-int(i)*3;
   for(unsigned j=0;j<4;++j)r.natural_projection[j]=float(i*4+j)*.375f;
   r.water_visible=i%17!=0;h.water_visible[geometry_water].push_back(r);
  }
 }
 std::vector<DrawReceipt> submit(bool resident=true,bool mirrored=false,GeometryLayer layer=geometry_water){
  current_renderer=&h.renderer;h.renderer.context->draws.clear();
  // draw_layer's bind_common establishes this binding before issue_records.
  h.renderer.context->VSSetConstantBuffers(1,1,&h.renderer.viewport_settings_buffer);
  // A separate records identity exercises the original streaming path.
  GeometryDrawView::Records copy;if(!resident)copy=h.water_visible;
  assert(h.issue_records(resident?h.water_visible:copy,layer,viewport,rect,mirrored));
  assert(h.renderer.context->viewport==h.renderer.viewport_settings_buffer);return h.renderer.context->draws;
 }
};
'''


class PreparedDrawParameterTests(unittest.TestCase):
    def test_exact_draws_reuse_owned_pages_after_unrelated_ring_overwrite(self):
        run_cpp(production_harness() + r'''
int main(){
 {Fixture f;auto oracle=f.submit(false);auto& h=f.h;
  assert(f.submit()==oracle&&!h.water_parameters.builds&&h.water_parameters.misses==1);
  auto first=f.submit();assert(first==oracle&&h.water_parameters.builds==1&&h.water_parameters.batch_count==4);
  auto maps=h.renderer.context->maps,culls=h.renderer.culls,creates=h.renderer.device->creates;
  assert(h.water_parameters.uploaded_bytes==777*256&&h.water_parameters.uploads==4);
  assert(h.water_parameters.gpu_bytes()==777*256);
  assert(h.water_parameters.metadata_bytes()>=sizeof(h.water_parameters));
  assert(h.water_parameters.cold_upload_ms>=0);
  std::array<ViewportShaderSettings,256> unrelated{};for(auto& p:unrelated)p.translation[0]=-8888;
  assert(h.parameters.upload(unrelated.data(),256));++maps;
  // Change time, not geometry. Every draw must receive new material state.
  h.renderer.water_material.time+=.25f;h.renderer.water_material.drift[1]-=.05f;
  oracle=f.submit(false);maps=h.renderer.context->maps;culls=h.renderer.culls;
  auto second=f.submit();assert(second==oracle&&h.water_parameters.reuses==1);
  assert(h.renderer.context->maps==maps&&h.renderer.culls==culls&&h.renderer.device->creates==creates);
  assert(h.water_parameters.reused_records==777);
  // Explicit still visibility remains per occurrence and changes on a new epoch.
  h.water_visible[geometry_water][1].water_visible=false;++h.visibility_revision;
  auto changed=f.submit();assert(changed==f.submit(false));assert(f.submit()==changed);assert(h.water_parameters.builds==2);
  assert(buffers_live==5); // Four owned pages plus the one dynamic stream.
 }
 assert(!buffers_live);
}
''')

    def test_every_view_membership_device_and_projection_dependency_rebuilds(self):
        run_cpp(production_harness() + r'''
int main(){
 {Fixture f(310);auto& h=f.h;f.submit();f.submit();unsigned expected=1;
  auto check=[&](){auto actual=f.submit();assert(actual==f.submit(false));assert(f.submit()==actual);assert(h.water_parameters.builds==++expected);};
  ++h.visibility_revision;check();++h.revision;check();++h.renderer.content_revision;check();++h.renderer.device_generation;check();
  f.viewport.translation[0]+=1;check();f.viewport.translation[1]+=1;check();f.viewport.depth_translation+=1;check();
  f.viewport.inverse_size[0]+=.001f;check();f.viewport.reserved[0]=9;check();f.viewport.natural_projection[1]=8;check();
  f.rect.left+=1000;check();f.rect.right-=9900;check();h.projection_zoom=2;check();
  ++h.renderer.content_view_width;check();++h.renderer.content_view_height;check();
  h.renderer.pickup_profile=false;check();h.renderer.city_profile=false;check();h.renderer.reflection.height_pixels=5;check();
  // Reorder at constant count plus replace storage: a new exact generation must
  // never index retired membership. No cached pointers or mesh leases exist.
  auto replacement=h.water_visible;std::reverse(replacement[geometry_water].begin(),replacement[geometry_water].end());
  h.water_visible.swap(replacement);replacement={};++h.visibility_revision;check();
  h.water_parameters.clear();check();
  auto count=h.water_parameters.builds;auto before=h.renderer.context->maps;
  f.submit(true,true);assert(h.water_parameters.builds==count&&h.renderer.context->maps>before);
 }
 assert(!buffers_live);
}
''')

    def test_allocation_capacity_and_unsupported_stream_keep_original_path(self):
        run_cpp(production_harness() + r'''
int main(){
 {Fixture f(780);auto& h=f.h;auto oracle=f.submit(false);f.submit();h.renderer.device->fail=true;
  assert(f.submit()==oracle&&h.water_parameters.fallbacks==1&&!h.water_parameters.batch_count);
  auto builds=h.water_parameters.builds;h.renderer.device->fail=false;
  assert(f.submit()==oracle&&h.water_parameters.builds==builds&&!h.water_parameters.batch_count);
  ++h.visibility_revision;assert(f.submit()==oracle&&!h.water_parameters.batch_count);
  assert(f.submit()==oracle&&h.water_parameters.batch_count==4);
 }
 assert(!buffers_live);
 {Fixture f(256*65);f.rect.right=20000;auto& h=f.h;auto oracle=f.submit(false);f.submit();
  assert(f.submit()==oracle&&h.water_parameters.fallbacks==1&&!h.water_parameters.batch_count);
  assert(buffers_live==1);auto builds=h.water_parameters.builds;
  assert(f.submit()==oracle&&h.water_parameters.builds==builds);
 }
 assert(!buffers_live);
 {Fixture f(300);auto& h=f.h;h.renderer.device->offsetting=false;
  auto oracle=f.submit(false);assert(f.submit()==oracle&&!h.water_parameters.builds&&!buffers_live);
 }
}
''')

    def test_moving_view_streams_without_allocating_new_pages(self):
        run_cpp(production_harness() + r'''
int main(){
 {Fixture f(600);auto& h=f.h;
  // Simulate continuous scrolling and zoom. Each first observation streams;
  // only the first stable pair admits immutable pages.
  f.submit(false);auto allocations=h.renderer.device->creates;
  for(unsigned frame=0;frame<50;++frame){f.viewport.translation[0]+=3;h.projection_zoom+=.01f;
   auto draw=f.submit();assert(draw==f.submit(false));
   assert(h.renderer.device->creates==allocations&&!h.water_parameters.builds&&!h.water_parameters.batch_count);
  }
  assert(h.water_parameters.misses==50);
  auto oracle=f.submit(false);assert(f.submit()==oracle&&h.water_parameters.builds==1);
  allocations=h.renderer.device->creates;assert(f.submit()==oracle&&h.water_parameters.reuses==1);
  assert(h.renderer.device->creates==allocations);
 }
 assert(!buffers_live);
}
''')

    def test_census_is_opt_in_stable_and_once_per_owner(self):
        run_cpp(r'''
#include <array>
#include <cassert>
#include "Renderer/native/render_core/submission_census.h"
int main(){
 using Census=c3x_renderer::render_core::SubmissionCensus<std::array<unsigned,3>>;
 Census census;std::array<unsigned,3> key={1,2,3};
 for(unsigned n=0;n<100;++n)assert(!census.observe(false,key));
 for(unsigned n=0;n<100;++n){++key[1];assert(!census.observe(true,key));}
 assert(!census.observe(true,key));assert(census.observe(true,key));
 for(unsigned n=0;n<100;++n){++key[2];assert(!census.observe(true,key));}
 assert(!census.observe(false,key));assert(census.reported);
 Census reset;assert(!reset.observe(true,key));assert(!reset.observe(true,key));
 assert(reset.observe(true,key));
 Census disabled;assert(!disabled.observe(true,key));assert(!disabled.observe(true,key));
 assert(!disabled.observe(false,key));assert(!disabled.observe(true,key));
 assert(!disabled.observe(true,key));assert(disabled.observe(true,key));
}
''')

    def test_mixed_rigid_native_city_animation_resource_and_phase_branches_match(self):
        run_cpp(production_harness() + r'''
int main(){
 {Fixture f(780);auto& h=f.h;int texture=1;
  for(unsigned i=0;i<f.meshes.size();++i){auto& m=f.meshes[i];
   if(i%9==0)m.rigid_source=true;
   if(i%9==1)m.city_material=1;
   if(i%9==2)m.city_material=2;
   if(i%9==3)m.resource_instance=&h.renderer.viewport_storage;
   if(i%9==4)m.animation_texture=&texture;
   if(i%9==5)m.visual_time=2;
  }
  auto oracle=f.submit(false);assert(f.submit()==oracle);assert(f.submit()==oracle);auto prior=h.renderer.shared_instances.queries;
  h.renderer.water_material.time=90;h.renderer.water_scene_active=false;oracle=f.submit(false);
  assert(f.submit()==oracle&&h.water_parameters.reuses==1&&h.renderer.shared_instances.queries>prior);
  // Separate layers own separate pages, even when their counts match.
  h.water_visible[geometry_city]=h.water_visible[geometry_water];++h.visibility_revision;
  oracle=f.submit(false,false,geometry_city);assert(f.submit(true,false,geometry_city)==oracle);
  assert(f.submit(true,false,geometry_city)==oracle);
 }
 assert(!buffers_live);
}
''')


if __name__ == "__main__":
    unittest.main()
