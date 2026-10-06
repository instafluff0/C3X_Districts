"""Execute camera-free input validation and the production required-region drain."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import ROOT, run_cpp


class LoadingWorldPreparationTests(unittest.TestCase):
    def test_world_only_contract_and_required_retry(self):
        source = Path(__file__).with_name('c3x_renderer.cpp').read_text()
        start = source.index('    bool prepare_world_sources(')
        sources = source[start:source.index('    bool render(', start)]
        start = source.index('    bool prepare_required_world(')
        method = source[start:source.index('    bool prepare_loading_sources(', start)]
        command = source.split('} else if(command==Command::prepare_world_loading){', 1)[1].split(
            '} else if (command == Command::require_world_changes)', 1)[0]
        run_cpp(r'''
#define C3X_RENDERER64_FRESH
#include "Renderer/native/render_core/world_input_capture.h"
#include "Renderer/native/render_core/world_preparation_region.h"
#include "Renderer/native/render_core/world_coast.h"
#include "Renderer/native/render_core/viewer_topology.h"
#include <algorithm>
#include <cassert>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>
using namespace c3x_renderer::render_core;
namespace c3x_renderer {struct Signature {std::uint64_t complete=1;};
Signature terrain_frame_signature(c3x_renderer_frame_v1 const&,unsigned,unsigned){return {};}
namespace fidelity {struct PatchDetail {PatchDetail()=default;PatchDetail(int,unsigned){}
 unsigned identity()const{return 64;}};}}
template<std::size_t N,class... A>void sprintf_s(char(&b)[N],char const* f,A... a){std::snprintf(b,N,f,a...);}
struct LARGE_INTEGER {long long QuadPart=0;};
void QueryPerformanceCounter(LARGE_INTEGER* value){++value->QuadPart;}
unsigned GetEnvironmentVariableA(char const*,char*,unsigned){return 0;}
struct RendererState {
 CapturedScene topology_cache;unsigned content_revision=1,device_generation=1;
 bool canonical_world_preparation=true,gpu_output_mode=false,pickup_profile=true,fidelity_profile=true;
 bool loading_preparation=false,loading_world_only=false;unsigned patch_pixels=0;
 bool loading_gpu_residency=false,world_gpu_residency_ready=false;
 bool world_gpu_capacity_refused=false,world_gpu_allocation_failed=false,world_gpu_allocation_pressure=false;
 struct Record {unsigned state=0;};std::map<std::uint64_t,Record> world_gpu_records;
 std::vector<Record*> world_gpu_current;
 std::map<std::uint64_t,c3x_renderer_tile_v1> gpu_owners;
 unsigned gpu_calls=0,gpu_uploads=0,gpu_reports=0,gpu_capacity=0,gpu_fail=0;
 bool gpu_complete=false;
 void world_gpu_report(bool complete){++gpu_reports;gpu_complete=complete;gpu_capacity=0;
  for(auto const& row:world_gpu_records)gpu_capacity+=row.second.state==2;}
 WorldCoast world_coast;ViewerTopology viewer_topology;
 struct Natural {unsigned calls=0;std::int64_t revision=-1;
  void update_rivers(WorldTopology const& world,std::int64_t r){
   assert(!world.empty());++calls;revision=r;}}natural;
 struct {unsigned calls=0;std::int64_t revision=-1;
  void bind(Natural const&,WorldTopology const& world,std::int64_t r){
   assert(!world.empty());++calls;revision=r;}}cliff_query_scratch;
 c3x_renderer::fidelity::PatchDetail patch_detail;
 struct {struct Stats {std::size_t bytes=0;};Stats statistics(){return {};}}world_backing;
 struct {void write(char const*,char const*,bool){}double milliseconds(long long v){return double(v);}}trace;
 std::size_t tile_geometry_cache_bytes=0;unsigned calls=0;int fail_call=0;
''' + sources + r'''
 bool render(c3x_renderer_frame_v1 const& f,c3x_renderer_output_v1&,int,
  std::atomic<bool> const*,std::uint64_t,unsigned const* selected,unsigned count,c3x_renderer_frame_v1 const* basis){
  assert(gpu_output_mode && f.target_width==8 && f.target_height==8 && f.tile_width==128 && f.tile_height==64);
  assert(count && selected && basis && basis->tile_count==0);
  assert(world_coast.revision()==basis->world_topology_revision);
  assert(natural.revision==basis->world_topology_revision && cliff_query_scratch.revision==natural.revision);
  auto sample=world_coast.sample({4,4},[](auto,auto){},[](auto,auto){});
  assert(std::isfinite(sample.distance));
  if(loading_gpu_residency){
   assert(count==1 && world_gpu_residency_ready);++gpu_calls;
   auto const& tile=f.tiles[selected[0]];auto key=topology_cache.key(tile.tile_x,tile.tile_y);
   auto& record=world_gpu_records[key];world_gpu_current={&record};
   world_gpu_capacity_refused=world_gpu_allocation_failed=false;
   if(gpu_fail==1 && tile.tile_x%4<2){world_gpu_capacity_refused=true;return false;}
   if(gpu_fail==2){world_gpu_allocation_failed=true;return false;}
   if(gpu_fail==3)return false;
   auto facts=CapturedScene::content(tile);auto retained=gpu_owners.find(key);
   if(retained==gpu_owners.end() || std::memcmp(&retained->second,&facts,sizeof(facts))){gpu_owners[key]=facts;++gpu_uploads;}
   record.state=1;return true;
  }
  ++calls;return int(calls)!=fail_call;
 }
};
struct Harness {
 RendererState renderer_state;ScenePublication scene_changes;WorldPreparationSchedule world_schedule;
 std::atomic<unsigned> loading_prepared{0},loading_regions{0};std::uint64_t world_prepare_sequence=0,world_initialization_scope=0;
 bool scene_changes_ok=true,fail_sources=false;unsigned source_calls=0;
 struct {unsigned passes=1,cursor=0;}world_input;
 c3x_renderer_camera_identity_v1 job_required_world_identity{1,2,4,3};
 std::shared_ptr<int> prepared_map;std::weak_ptr<int> retired;
 void retain(){prepared_map=std::make_shared<int>(1);retired=prepared_map;}
 unsigned retired_scenes=0;void retire_completed_scene(){++retired_scenes;}
 bool prepare_loading_sources(){
  // The old view must lose its only preparation owner before source mutation.
  assert(retired.expired());++source_calls;
  if(fail_sources)throw std::runtime_error("source failure");return true;
 }
 struct GpuOutputMode {RendererState& state;bool prior;
  GpuOutputMode(RendererState& s,bool gpu,bool):state(s),prior(s.gpu_output_mode){s.gpu_output_mode=gpu;}
  ~GpuOutputMode(){state.gpu_output_mode=prior;}};
''' + method + r'''
 int prepare_loading_command(){int result=C3X_RENDERER_RESULT_ERROR;
''' + command + r'''
 return result;}
};
int main(){
 std::vector<unsigned> topology(128,2|(2<<8));c3x_renderer_frame_v1 f{};
 f.api_version=C3X_RENDERER_API_VERSION;f.struct_size=sizeof(f);f.hour=12;
 f.world_width_tiles=f.world_height_tiles=16;f.world_topology=topology.data();f.world_topology_count=128;f.world_topology_revision=3;
 c3x_renderer_camera_request_v1 request{C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&f,{1,2,4,3}};
 assert(valid_loading_world(&request));auto valid=f;
 for(auto member:{&c3x_renderer_frame_v1::target_width,&c3x_renderer_frame_v1::tile_width}){
  f.*member=1;assert(!valid_loading_world(&request));f=valid;
 }
 f.tile_count=1;assert(!valid_loading_world(&request));f=valid;
 f.world_topology_count=127;assert(!valid_loading_world(&request));f=valid;
 request.identity.scene_epoch=4;assert(!valid_loading_world(&request));request.identity.scene_epoch=3;
 Harness h;assert(h.scene_changes.capture(f,request.identity));bool changed=false;
 assert(h.scene_changes.apply(h.renderer_state.topology_cache,changed));
 // A fresh owner has no foreground draw to initialize canonical recipe inputs.
 bool cold_failure=false;
 try{h.renderer_state.world_coast.sample({4,4},[](auto,auto){},[](auto,auto){});}
 catch(std::runtime_error const&){cold_failure=true;}assert(cold_failure);
 // No permitted objects is valid empty ownership, not an unavailable region.
 assert(h.prepare_required_world(h.scene_changes.state(),nullptr,true));
 assert(h.world_schedule.completed==4 && !h.renderer_state.calls);
 assert(h.renderer_state.world_coast.revision()==3 && h.renderer_state.natural.calls==1);
 assert(h.renderer_state.world_coast.world().dimensions().width==16);
 assert(h.renderer_state.world_coast.world().at(127)==topology[127]);
 h.world_schedule.clear();
 for(int y=0;y<16;++y)for(int x=y&1;x<16;x+=2){c3x_renderer_tile_v1 t{};
  t.tile_x=x;t.tile_y=y;t.terrain_type=t.real_terrain_type=2;
  t.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_PREFETCH;
  assert(h.renderer_state.topology_cache.publish(t,changed));
 }
 h.renderer_state.fail_call=2;
 assert(!h.prepare_required_world(h.scene_changes.state(),nullptr,true));
 assert(h.world_schedule.completed==1 && h.world_schedule.unavailable==1 && h.renderer_state.calls==2);
 assert(!h.renderer_state.gpu_output_mode);
 h.renderer_state.fail_call=0;assert(h.prepare_required_world(h.scene_changes.state(),nullptr,true));
 assert(h.world_schedule.completed==4 && !h.world_schedule.unavailable && h.renderer_state.calls==5);
 assert(h.prepare_required_world(h.scene_changes.state(),nullptr,true) && h.renderer_state.calls==5);
 assert(h.renderer_state.gpu_uploads==128 && h.renderer_state.gpu_complete);
 auto uploads=h.renderer_state.gpu_uploads;
 assert(h.prepare_required_world(h.scene_changes.state(),nullptr,true));
 assert(h.renderer_state.gpu_uploads==uploads && !h.renderer_state.loading_gpu_residency);
 // Optional capacity refusal keeps RAM completion and previously resident
 // owners. The actual second pass visits later smaller cores after refusal.
 h.renderer_state.gpu_owners.clear();h.renderer_state.gpu_fail=1;
 assert(h.prepare_required_world(h.scene_changes.state(),nullptr,true));
 assert(h.renderer_state.gpu_complete && h.renderer_state.gpu_capacity==64);
 assert(h.renderer_state.gpu_owners.size()==64);
 uploads=h.renderer_state.gpu_uploads;
 assert(h.prepare_required_world(h.scene_changes.state(),nullptr,true));
 assert(h.renderer_state.gpu_uploads==uploads); // repeated pressure never churns survivors
 h.renderer_state.gpu_fail=0;assert(h.prepare_required_world(h.scene_changes.state(),nullptr,true));
 assert(h.renderer_state.gpu_uploads==uploads+64 && h.renderer_state.gpu_owners.size()==128);
 // Allocation/source failures remain explicit, with no GPU-complete receipt.
 for(unsigned failure:{2u,3u}){h.renderer_state.gpu_fail=failure;
  assert(!h.prepare_required_world(h.scene_changes.state(),nullptr,true));
  assert(!h.renderer_state.gpu_complete && !h.renderer_state.loading_gpu_residency && h.renderer_state.world_gpu_records.empty());
 }
 h.renderer_state.gpu_fail=0;
 h.world_schedule.invalidate(valid,0,0);std::atomic<bool> cancel{true};
 assert(!h.prepare_required_world(h.scene_changes.state(),&cancel,true));
 assert(!h.world_schedule.empty() && !h.world_schedule.unavailable && h.renderer_state.calls==5);
 cancel=false;assert(h.prepare_required_world(h.scene_changes.state(),&cancel,true));
 assert(h.world_schedule.completed==4 && !h.renderer_state.gpu_output_mode);
 // Missing source authority is an explicit failure, with no recipe render.
 auto missing=f;missing.world_topology_count=0;auto calls=h.renderer_state.calls;
 assert(!h.renderer_state.prepare_world_sources(missing) && h.renderer_state.calls==calls);
 // A later topology revision refreshes the shared source-only dependency too.
 topology[0]=11|(11<<8);f.world_topology_revision=4;
 assert(h.renderer_state.prepare_world_sources(f));
 assert(h.renderer_state.world_coast.revision()==4 && h.renderer_state.natural.revision==4);
 assert(h.renderer_state.world_coast.world().at(0)==topology[0]);
 // Execute the actual command gate: stale/incomplete scopes keep their owner.
 Harness absent;absent.retain();
 assert(absent.prepare_loading_command()==C3X_RENDERER_RESULT_SUPERSEDED && !absent.retired.expired());
 h.retain();auto unchanged=h.prepared_map.get();calls=h.renderer_state.calls;
 auto rejected=[&]{assert(h.prepare_loading_command()==C3X_RENDERER_RESULT_SUPERSEDED);
  assert(h.prepared_map.get()==unchanged && !h.retired.expired() && !h.source_calls && h.renderer_state.calls==calls);};
 h.scene_changes_ok=false;rejected();h.scene_changes_ok=true;
 h.world_input.passes=0;rejected();h.world_input.passes=1;
 h.world_input.cursor=1;rejected();h.world_input.cursor=0;
 for(auto member:{&c3x_renderer_camera_identity_v1::map_epoch,&c3x_renderer_camera_identity_v1::viewer_epoch,
     &c3x_renderer_camera_identity_v1::visibility_epoch,&c3x_renderer_camera_identity_v1::scene_epoch}){
  ++(h.job_required_world_identity.*member);rejected();--(h.job_required_world_identity.*member);
 }
 // Valid preparation retires before even a failing source setup, and leaves
 // the existing explicit source/recipe errors and loading-scope RAII intact.
 h.fail_sources=true;bool failed=false;
 try{h.prepare_loading_command();}catch(std::runtime_error const&){failed=true;}
 assert(failed && h.retired.expired() && !h.prepared_map && h.renderer_state.calls==calls);
 h.fail_sources=false;h.retain();h.world_schedule.clear();h.renderer_state.fail_call=int(calls)+1;
 assert(h.prepare_loading_command()==C3X_RENDERER_RESULT_ERROR && h.retired.expired());
 assert(!h.renderer_state.loading_preparation && !h.renderer_state.loading_world_only && !h.renderer_state.gpu_output_mode);
 h.renderer_state.fail_call=0;h.retain();
 assert(h.prepare_loading_command()==C3X_RENDERER_RESULT_OK && h.retired.expired());
 assert(h.world_schedule.completed==4 && !h.world_schedule.unavailable && !h.loading_regions);
 assert(!h.renderer_state.loading_preparation && !h.renderer_state.loading_world_only && !h.renderer_state.gpu_output_mode);
 // The actual region/key traversal keeps unchanged copied generations, while
 // a publication edit rearms its affected recipe region and replaces that core.
 auto changed_core=h.renderer_state.topology_cache.world_view().current(h.renderer_state.topology_cache.key(0,0))->occurrence;
 changed_core.has_effect=1;assert(h.renderer_state.topology_cache.publish(changed_core,changed) && changed);
 h.world_schedule.invalidate(valid,0,0);uploads=h.renderer_state.gpu_uploads;
 assert(h.prepare_required_world(h.scene_changes.state(),nullptr,true));
 assert(h.renderer_state.gpu_uploads==uploads+1 && h.renderer_state.gpu_owners.size()==128);
}
''')


    def test_recognized_save_extension_requires_actual_restoration(self):
        source = (ROOT / 'injected_code.c').read_text()
        restore = source[source.index('patch_move_game_data ('):source.index('patch_MappedFile_deinit_after_saving_or_loading (')]
        start = restore.index('\tMappedFile * save =')
        prefix = restore[start:restore.index('\t\tbyte * cursor = seg;', start)]
        # The game's native pointer arithmetic is x86; use the host pointer width.
        prefix = prefix.replace('(int)save->base_addr', '(std::uintptr_t)save->base_addr')
        start = restore.index('\t\tif (error_chunk_name == NULL && cursor == seg + seg_size)')
        proof = restore[start:restore.index('\tif ((! save_else_load) &&', start)]
        final = restore[restore.rfind('\tif (is->current_config.enable_custom_rendering &&'):restore.rfind('\treturn tr;') + len('\treturn tr;')]
        run_cpp(r"""
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>
#include <vector>
using byte=unsigned char;
struct MappedFile {void* base_addr=nullptr;int size=0;}file;
struct {struct {bool enable_custom_rendering=true;}current_config;MappedFile* accessing_save_file=&file;}state,*is=&state;
struct {bool is_now_loading_game=true;}form,*p_main_screen_form=&form;
constexpr byte bookend[4]={0x22,'C','3','X'};
int prepares=0,allocations=0,parse_mode=0;bool allocation_ok=true;
bool match_save_segment_bookend(byte* b){return !std::memcmp(b,bookend,4);}
int int_from_bytes(byte* b){int result=0;std::memcpy(&result,b,4);return result;}
byte* allocate(int count){++allocations;return allocation_ok?new byte[count]:nullptr;}
void release(byte* p){delete[] p;}
int prepare_custom_renderer_loading_world(){++prepares;return 1;}
#define malloc allocate
#define free release
int restore(int tr,bool save_else_load){bool renderer_restore_ok=tr>0;
""" + prefix + r"""
 // Chunk restoration is tested elsewhere; execute the real surrounding
 // admission and exact-consumption certificate with each parser outcome.
 byte* cursor=seg+(parse_mode==0?seg_size:parse_mode==1?0:seg_size+1);
 char* error_chunk_name=parse_mode==1?(char*)"failed chunk":nullptr;
""" + proof + final + r"""
}
void reset(std::vector<byte>& data){state={};form={};file={data.data(),int(data.size())};
 prepares=allocations=parse_mode=0;allocation_ok=true;}
std::vector<byte> extension(){std::vector<byte> data(16);int size=4;
 std::memcpy(data.data(),bookend,4);std::memcpy(data.data()+8,&size,4);std::memcpy(data.data()+12,bookend,4);return data;}
int main(){
 // A native-only old save remains eligible; no optional extension is invented.
 std::vector<byte> old(8);reset(old);assert(restore(5,false)==5 && prepares==1 && !allocations);
 auto valid=extension();reset(valid);assert(restore(5,false)==5 && prepares==1 && allocations==1);
 for(int size:{0,-1,INT_MAX}){auto data=extension();std::memcpy(data.data()+8,&size,4);reset(data);
  assert(restore(5,false)==5 && !prepares && !allocations);}
 std::vector<byte> short_extension(bookend,bookend+4);reset(short_extension);
 assert(restore(5,false)==5 && !prepares && !allocations);
 auto missing_header=extension();missing_header[0]=0;reset(missing_header);
 assert(restore(5,false)==5 && !prepares && !allocations);
 reset(valid);allocation_ok=false;assert(restore(5,false)==5 && !prepares && allocations==1);
 for(int outcome:{1,2}){reset(valid);parse_mode=outcome;assert(restore(5,false)==5 && !prepares && allocations==1);}
 reset(valid);assert(restore(0,false)==0 && !prepares);
 reset(valid);assert(restore(5,true)==5 && !prepares && !allocations);
 reset(valid);state.current_config.enable_custom_rendering=false;
 assert(restore(5,false)==5 && !prepares && allocations==1);
}
""")


    def test_bloom_shaders_prepare_without_view_and_survive_target_resize(self):
        source = (ROOT / 'Renderer/sandbox/bloom.h').read_text()
        methods = source[source.index('struct SandboxBloom {'):source.index('    bool draw(')] + '};'
        run_cpp(r"""
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdint>
struct SandboxPassWorkload{};
using HRESULT=int;
constexpr HRESULT S_OK=0;
bool FAILED(HRESULT h){return h<0;}bool SUCCEEDED(HRESULT h){return h>=0;}
constexpr int D3DCOMPILE_OPTIMIZATION_LEVEL3=1,D3D11_BIND_CONSTANT_BUFFER=2,
 D3D11_FILTER_MIN_MAG_MIP_LINEAR=3,D3D11_TEXTURE_ADDRESS_CLAMP=4,
 D3D11_FILL_SOLID=5,D3D11_CULL_NONE=6,DXGI_FORMAT_R16G16B16A16_FLOAT=7,
 D3D11_BIND_RENDER_TARGET=8,D3D11_BIND_SHADER_RESOURCE=16;
constexpr float D3D11_FLOAT32_MAX=1.e30f;
int alive=0,compiles=0,compile_fail=0,texture_calls=0,texture_fail=0;
struct Resource {Resource(){++alive;}void Release(){--alive;delete this;}
 void* GetBufferPointer(){return this;}std::size_t GetBufferSize(){return 1;}};
using ID3DBlob=Resource;using ID3D11Texture2D=Resource;using ID3D11RenderTargetView=Resource;
using ID3D11ShaderResourceView=Resource;using ID3D11VertexShader=Resource;using ID3D11PixelShader=Resource;
using ID3D11Buffer=Resource;using ID3D11SamplerState=Resource;using ID3D11RasterizerState=Resource;
struct D3D11_BUFFER_DESC {unsigned ByteWidth=0,BindFlags=0;};
struct D3D11_SAMPLER_DESC {int Filter=0,AddressU=0,AddressV=0,AddressW=0;float MaxLOD=0;};
struct D3D11_RASTERIZER_DESC {int FillMode=0,CullMode=0;bool DepthClipEnable=false;};
struct D3D11_TEXTURE2D_DESC {unsigned Width=0,Height=0,ArraySize=0,MipLevels=0;
 struct {unsigned Count=0;}SampleDesc;int Format=0,BindFlags=0;};
HRESULT D3DCompile(char const*,std::size_t,char const*,void*,void*,char const*,char const*,int,int,Resource** out,Resource**){
 ++compiles;if(compiles==compile_fail)return -1;*out=new Resource;return 0;
}
struct Device {
 HRESULT CreateVertexShader(void*,std::size_t,void*,Resource** out){*out=new Resource;return 0;}
 HRESULT CreatePixelShader(void*,std::size_t,void*,Resource** out){*out=new Resource;return 0;}
 HRESULT CreateBuffer(D3D11_BUFFER_DESC*,void*,Resource** out){*out=new Resource;return 0;}
 HRESULT CreateSamplerState(D3D11_SAMPLER_DESC*,Resource** out){*out=new Resource;return 0;}
 HRESULT CreateRasterizerState(D3D11_RASTERIZER_DESC*,Resource** out){*out=new Resource;return 0;}
 HRESULT CreateTexture2D(D3D11_TEXTURE2D_DESC*,void*,Resource** out){
  ++texture_calls;if(texture_calls==texture_fail)return -1;*out=new Resource;return 0;}
 HRESULT CreateRenderTargetView(Resource*,void*,Resource** out){*out=new Resource;return 0;}
 HRESULT CreateShaderResourceView(Resource*,void*,Resource** out){*out=new Resource;return 0;}
}device;
struct {Device* device;}renderer{&device};
""" + methods + r"""
int main(){
 {SandboxBloom bloom;assert(bloom.ensure_shaders());
  assert(compiles==3 && texture_calls==0 && !bloom.width && !bloom.height && !bloom.view[0]);
  auto vertex=bloom.vertex,blur=bloom.blur;assert(bloom.ensure(11,7));
  assert(bloom.width==6 && bloom.height==4 && compiles==3 && texture_calls==2);
  assert(bloom.ensure(33,15));assert(bloom.width==17 && bloom.height==8 && compiles==3 && texture_calls==4);
  assert(bloom.vertex==vertex && bloom.blur==blur);
  texture_fail=texture_calls+1;assert(!bloom.ensure(61,27));
  assert(!bloom.view[0] && !bloom.width && !bloom.height && bloom.vertex==vertex && bloom.blur==blur);
  texture_fail=0;assert(bloom.ensure(61,27) && compiles==3);
 }assert(!alive);
 compiles=0;compile_fail=2;
 {SandboxBloom bloom;assert(!bloom.ensure_shaders() && !alive && !bloom.blur && !bloom.vertex);
  compile_fail=0;assert(bloom.ensure_shaders() && compiles==5);
 }assert(!alive);
}
""")


if __name__ == '__main__':
    unittest.main()
