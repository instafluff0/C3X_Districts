"""Owned compilation survives view replacement; retirement remains a real barrier."""
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.lab.platform import ROOT

class DurablePreparationTests(unittest.TestCase):
    def test_actual_world_reserves_charge_packets_and_publication_once(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        start=source.index('                auto attachments=Budget::scene_limit+')
        budget=source[start:source.index('\n            }',start)]
        run_cpp(r'''#include "Renderer/native/render_core/frame_working_set.h"
#include "Renderer/native/render_core/shadow_sampling_grid.h"
#include <atomic>
#include <cassert>
#include <memory>
using Budget=c3x_renderer::render_core::FrameWorkingSet;
struct OrderedRigidPackets {static constexpr std::size_t budget=64u*Budget::mib;};
struct State {
 std::size_t base=0,publication_capacity_bytes=160u*Budget::mib,publication_working_bytes=0;
 std::size_t tile_geometry_cache_bytes=512u*Budget::mib,terrain_patch_index_bytes=0,tile_geometry_runtime_budget=0;
 bool measured_gpu=true,world_gpu_allocation_pressure=false,loading_gpu_residency=true;
 struct Packet {std::size_t bytes=0;std::size_t gpu_bytes()const{return bytes;}}ordered_rigid_packets,shared_instances;
 struct Retired {std::atomic<std::size_t> bytes{0};};std::shared_ptr<Retired> retired_content=std::make_shared<Retired>();
 struct Composition {std::size_t bytes=0;std::size_t allocation_bytes()const{return bytes;}};
 std::shared_ptr<Composition> gpu_composition=std::make_shared<Composition>();
 struct {std::size_t ullAvailPhys=12ull*1024*Budget::mib,ullTotalPhys=16ull*1024*Budget::mib;}content_memory;
 std::size_t frame_working_bytes()const{return base+ordered_rigid_packets.bytes;}
 std::size_t calculate(){std::size_t gpu_headroom=8ull*1024*Budget::mib,ceiling=gpu_headroom;unsigned requested_workers=4;
''' + budget + r'''
 return tile_geometry_runtime_budget;}
};
int main(){State s;auto empty=s.calculate();
 // Existing packet bytes satisfy only their own missing reserve.
 s.ordered_rigid_packets.bytes=32u*Budget::mib;auto packets=s.calculate();
 assert(packets==empty+16u*Budget::mib);
 s.publication_working_bytes=32u*Budget::mib;auto published=s.calculate();
 assert(published==packets+16u*Budget::mib);
 s.publication_working_bytes=320u*Budget::mib;auto full=s.calculate();
 s.publication_working_bytes=640u*Budget::mib;assert(s.calculate()==full);
 s.world_gpu_allocation_pressure=true;assert(s.calculate()==s.tile_geometry_cache_bytes);
 s.world_gpu_allocation_pressure=false;s.content_memory.ullAvailPhys=0;s.loading_gpu_residency=false;
 assert(s.calculate()<s.tile_geometry_cache_bytes);
}
''')

    def test_actual_mesh_failures_classify_only_proven_out_of_memory(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        start=source.index('            ID3D11Buffer* allocations[2]={nullptr,nullptr};')
        create=source[start:source.index('            for(unsigned owner=0;owner<2;',start)]
        start=source.index('            auto hr=device->CreateBuffer(&desc,&initial,&chunk.indices);')
        grid=source[start:source.index('            try{terrain_patch_indices.emplace',start)]
        run_cpp(r'''#include <atomic>
#include <array>
#include <cassert>
#include <cstdio>
#include <thread>
using HRESULT=long;
constexpr HRESULT E_OUTOFMEMORY=-2147024882L,removed=-2147467259L;
bool FAILED(HRESULT r){return r<0;}bool SUCCEEDED(HRESULT r){return r>=0;}
struct D3D11_BUFFER_DESC {unsigned ByteWidth=0,Usage=0,BindFlags=0;};
struct D3D11_SUBRESOURCE_DATA {void const* pSysMem=nullptr;};
constexpr unsigned D3D11_USAGE_IMMUTABLE=1,D3D11_BIND_VERTEX_BUFFER=2,D3D11_BIND_INDEX_BUFFER=4;
std::atomic<unsigned> live{0};
struct ID3D11Buffer {ID3D11Buffer(){++live;}void Release(){--live;delete this;}};
struct Device {HRESULT failure[2]={};
 HRESULT CreateBuffer(D3D11_BUFFER_DESC const* d,D3D11_SUBRESOURCE_DATA const*,ID3D11Buffer** result){
  auto hr=failure[d->ByteWidth/4-1];if(hr<0)return hr;*result=new ID3D11Buffer;return hr;}
 HRESULT GetDeviceRemovedReason(){return removed;}};
#include "Renderer/native/render_core/immutable_mesh_upload.h"
struct State {Device instance,*device=&instance;
 std::array<c3x_renderer::render_core::ImmutableMeshUpload,2> mesh_uploads;
 bool loading_gpu_residency=true,world_gpu_capacity_refused=false,world_gpu_allocation_pressure=false,world_gpu_allocation_failed=false;
 std::size_t tile_geometry_cache_bytes=100;struct {std::size_t byte_count=100;}compiled;
 bool execute(){
''' + create + r'''
 for(auto buffer:allocations)if(buffer)buffer->Release();return true;}
 bool grid(){D3D11_BUFFER_DESC desc{4};D3D11_SUBRESOURCE_DATA initial{};
  struct {ID3D11Buffer* indices=nullptr;}chunk;
''' + grid + r'''
 chunk.indices->Release();return true;}
};
int main(){unsigned data[2]={1,2};
 for(bool loading:{false,true})for(HRESULT first:{0L,E_OUTOFMEMORY,removed})
 for(HRESULT second:{0L,E_OUTOFMEMORY,removed}){
  State s;s.loading_gpu_residency=loading;s.instance.failure[0]=first;s.instance.failure[1]=second;
  s.mesh_uploads[0].append(data,4);s.mesh_uploads[1].append(data,8);
  auto okay=s.execute();assert(okay==(!first && !second) && !live);
  auto capacity=loading && !okay && first!=removed && second!=removed;
  assert(s.world_gpu_capacity_refused==capacity && s.world_gpu_allocation_pressure==capacity);
  assert(s.world_gpu_allocation_failed==(loading && !okay && !capacity));
  assert(s.tile_geometry_cache_bytes==(okay?100:0));
 }
 for(bool loading:{false,true})for(HRESULT failure:{0L,E_OUTOFMEMORY,removed}){
  State s;s.loading_gpu_residency=loading;s.instance.failure[0]=failure;
  assert(s.grid()==!failure && !live);
  assert(s.world_gpu_capacity_refused==(loading && failure==E_OUTOFMEMORY));
  assert(s.world_gpu_allocation_failed==(loading && failure==removed));
 }
 Device device;c3x_renderer::render_core::ImmutableMeshUpload empty;ID3D11Buffer* buffer=nullptr;long hr=99;
 assert(empty.create(&device,&buffer,&hr) && !hr && !buffer);
 c3x_renderer::render_core::ImmutableMeshUpload packed;packed.append(data,4);
 assert(packed.create(&device,&buffer));buffer->Release();assert(!live);
}
''')

    def test_actual_loading_admission_refuses_without_evicting_or_churning(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        start=source.index('    bool make_tile_cache_room(')
        method=source[start:source.index('    bool cache_geometry_layer(',start)]
        run_cpp(r'''#include <atomic>
#include <cassert>
#include <cstdio>
#include <memory>
#include <unordered_map>
constexpr unsigned viewport_cache_capacity=8;
template<std::size_t N,class... A>void sprintf_s(char(&b)[N],char const* f,A... a){std::snprintf(b,N,f,a...);}
struct Cache {unsigned signature=0;std::size_t byte_count=0;bool prefetched=false;
 std::uint64_t animation_epoch=0,last_used=0;};
using CachedTileGeometry=Cache;
struct State {
 std::unordered_multimap<unsigned,Cache> tile_geometry_cache;
 std::size_t tile_geometry_cache_bytes=0,prefetched_geometry_bytes=0,tile_geometry_runtime_budget=512;
 std::size_t terrain_patch_index_bytes=4,tile_geometry_cache_capacity=16;
 std::uint64_t tile_geometry_epoch=100;unsigned frame_tiles_evicted=0,cache_evictions=0,releases=0;
 bool loading_gpu_residency=true,world_gpu_capacity_refused=false;
 struct Retired {std::atomic<std::size_t> bytes{32};};std::shared_ptr<Retired> retired_content=std::make_shared<Retired>();
 struct {void write(char const*,char const*,bool){}}trace;
 struct {Cache* resolve(Cache* value){return value;}}resident_content;
 struct Candidates {template<class M,class R,class F>Cache* next(M& map,R&,std::uint64_t,F){return map.empty()?nullptr:&map.begin()->second;}}residency_candidates;
 void release_resident_content(Cache&){++releases;}
''' + method + r'''};
int main(){State s;for(unsigned i=0;i<3;++i)s.tile_geometry_cache.emplace(i,Cache{i,100,true});
 s.tile_geometry_cache_bytes=s.prefetched_geometry_bytes=300;
 assert(!s.make_tile_cache_room(180) && s.world_gpu_capacity_refused);
 assert(s.tile_geometry_cache.size()==3 && s.tile_geometry_cache_bytes==300 && !s.releases && !s.frame_tiles_evicted);
 s.world_gpu_capacity_refused=false;assert(s.make_tile_cache_room(64) && !s.world_gpu_capacity_refused);
 s.tile_geometry_runtime_budget=128;
 for(unsigned i=0;i<3;++i)assert(!s.make_tile_cache_room(0));
 assert(s.tile_geometry_cache_bytes==300 && s.tile_geometry_cache.size()==3 && !s.releases);
 // Foreground selected-content admission retains its existing eviction path.
 s.loading_gpu_residency=false;assert(s.make_tile_cache_room(0));
 assert(s.tile_geometry_cache_bytes==0 && s.tile_geometry_cache.empty() && s.releases==3 && s.frame_tiles_evicted==3);
}
''')

    def test_completed_recipes_admit_measured_gpu_growth_without_duplicate_ram_reserve(self):
        run_cpp(r'''#include "Renderer/native/render_core/frame_working_set.h"
#include <cassert>
using F=c3x_renderer::render_core::FrameWorkingSet;
int main(){
 auto owned=2ull*1024*F::mib,ceiling=8ull*1024*F::mib;
 for(unsigned workers:{0u,1u,4u,6u,12u})for(unsigned free:{0u,128u,4096u,12288u})
 for(unsigned adapter:{0u,512u,2048u,8192u})for(unsigned future:{0u,512u,1536u}){
  auto available=std::size_t(free)*F::mib,headroom=std::size_t(adapter)*F::mib;
  auto reserve=std::max(2048ull*F::mib,16ull*1024*F::mib/6)+512ull*F::mib+
   std::max(2u,std::min(workers,6u)+1u)*16ull*F::mib+std::min(workers,6u)*48ull*F::mib;
  auto result=F::world_geometry(available,16ull*1024*F::mib,owned,headroom,future*F::mib,ceiling,workers,true);
  assert(result>=owned && result<=ceiling);
  auto growth=result-owned,usable=available>reserve?available-reserve:0;
  usable=usable>future*F::mib?usable-future*F::mib:0;
  assert(2*growth<=usable);
  auto gpu=headroom>future*F::mib?headroom-future*F::mib:0;
  assert(growth<=gpu-gpu/5);
  if(!adapter || free==0)assert(result==owned);
 }
 auto loaded=F::world_geometry(12ull*1024*F::mib,16ull*1024*F::mib,owned,8ull*1024*F::mib,0,ceiling,4,true);
 auto preparing=F::residency(12ull*1024*F::mib,16ull*1024*F::mib,owned,200ull*F::mib,8ull*1024*F::mib,ceiling,4);
 assert(loaded>preparing.geometry); // compact recipes already exist
 auto shrunk=F::world_geometry(0,16ull*1024*F::mib,loaded,0,1536ull*F::mib,ceiling,4,true);
 assert(shrunk==loaded); // no loading sweep eviction
 auto foreground=F::world_geometry(0,16ull*1024*F::mib,loaded,0,1536ull*F::mib,ceiling,4,false);
 auto old_pressure=F::residency(0,16ull*1024*F::mib,loaded,0,0,ceiling,4);
 assert(foreground==old_pressure.geometry && foreground<loaded);
}
''')

    def test_measured_residency_balances_ram_gpu_and_pressure(self):
        run_cpp(r'''#include "Renderer/native/render_core/frame_working_set.h"
#include <cassert>
using F=c3x_renderer::render_core::FrameWorkingSet;
int main(){
 auto ample=F::residency(12ull*1024*F::mib,16ull*1024*F::mib,2ull*1024*F::mib,200ull*F::mib,4ull*1024*F::mib,8ull*1024*F::mib,4);
 assert(ample.geometry>2ull*1024*F::mib && ample.prepared>200ull*F::mib);
 auto adapter_full=F::residency(12ull*1024*F::mib,16ull*1024*F::mib,2ull*1024*F::mib,200ull*F::mib,0,8ull*1024*F::mib,4);
 assert(adapter_full.geometry==2ull*1024*F::mib);
 auto pressure=F::residency(128ull*F::mib,16ull*1024*F::mib,2ull*1024*F::mib,200ull*F::mib,4ull*1024*F::mib,8ull*1024*F::mib,4);
 assert(pressure.geometry<2ull*1024*F::mib && pressure.prepared==200ull*F::mib);
 auto explicit_cap=F::residency(12ull*1024*F::mib,16ull*1024*F::mib,128ull*F::mib,0,4ull*1024*F::mib,384ull*F::mib,4);
 assert(explicit_cap.geometry<=384ull*F::mib);
 // A future compact byte costs one physical byte; future GPU geometry uses
 // the conservative two-byte charge for allocation and driver/CPU mirrors.
 for(unsigned workers:{0u,1u,4u,6u,12u})for(unsigned free_mib:{0u,128u,2048u,4096u,12288u})
 for(unsigned gpu_mib:{0u,128u,1024u,4096u}){
  auto lanes=std::min(workers,6u);auto physical=16ull*1024*F::mib;
  auto reserve=std::max(2048ull*F::mib,physical/6)+512ull*F::mib+
      std::max(2u,lanes+1u)*16ull*F::mib+lanes*48ull*F::mib;
  auto available=free_mib*F::mib,headroom=gpu_mib*F::mib;
  auto before_geometry=2ull*1024*F::mib,before_prepared=200ull*F::mib;
  auto result=F::residency(available,physical,before_geometry,before_prepared,headroom,8ull*1024*F::mib,workers);
  auto usable=available>reserve?available-reserve:0;
  auto compact_growth=result.prepared-before_prepared;
  auto geometry_growth=result.geometry>before_geometry?result.geometry-before_geometry:0;
  assert(result.prepared>=before_prepared && result.geometry<=8ull*1024*F::mib);
  assert(compact_growth+2*geometry_growth<=usable && geometry_growth<=headroom-headroom/5);
  if(available<=reserve)assert(!compact_growth && result.geometry<=before_geometry);
  if(workers>6){auto clamped=F::residency(available,physical,before_geometry,before_prepared,headroom,8ull*1024*F::mib,6);
   assert(result.geometry==clamped.geometry && result.prepared==clamped.prepared);}
 }
}
''')

    def test_retarget_does_not_join_or_duplicate_active_work(self):
        run_cpp(r'''
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Result {int value;size_t bytes()const{return 8u*1024u*1024u;}};
using Input=std::shared_ptr<int const>;using Queue=ContentPreparation<int,Input,Result>;
int main(){
 std::atomic<bool> entered{false},release{false},obsolete{false};std::atomic<int> compiled{0};
 Queue q;auto source=std::make_shared<int const>(17);std::weak_ptr<int const> lifetime=source;
 auto compile=[&](Input const& input,auto const& stop,unsigned){
  ++compiled;if(*input==17){entered=true;while(!release && !stop)std::this_thread::yield();}
  return std::make_unique<Result>(Result{*input});
 };
 q.schedule({{1,source}},compile,1,{1},32u*1024u*1024u);source.reset();
 while(!entered)std::this_thread::yield();
 // Must return while job 1 is blocked. The duplicate never starts, and the
 // replacement input cannot mutate the running job's value.
 q.schedule({{1,std::make_shared<int const>(99)},{2,std::make_shared<int const>(23)}},compile,1,{2,1},32u*1024u*1024u);
 assert(!lifetime.expired() && compiled==1);
 obsolete=true;assert(!q.take(1,false,[&]{return obsolete.load();}));
 assert(!lifetime.expired());release=true;
 auto a=q.take(1),b=q.take(2);assert(a&&b&&a->value==17&&b->value==23);
 q.clear();assert(lifetime.expired()&&compiled==2);
 // Asset/device retirement cancels and joins active input owners.
 entered=false;release=false;source=std::make_shared<int const>(17);lifetime=source;
 q.schedule({{3,source}},compile,1,{3},32u*1024u*1024u);source.reset();
 while(!entered)std::this_thread::yield();q.clear();assert(lifetime.expired());
 assert(!q.statistics().active && !q.statistics().pending && !q.statistics().bytes);
}
''')

    def test_pressure_budget_preserves_requested_parallelism_with_ready_content(self):
        run_cpp(r'''#include "Renderer/native/render_core/content_preparation.h"
#include "Renderer/native/render_core/frame_working_set.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Result{size_t bytes()const{return 8u*1024u*1024u;}};
int main(){
 std::atomic<unsigned> entered{0};std::atomic<bool> release{false};
 ContentPreparation<int,int,Result> q;
 auto compile=[&](int value,auto const& stop,unsigned){if(value){++entered;while(!release && !stop)std::this_thread::yield();}return std::make_unique<Result>();};
 auto budget=FrameWorkingSet::content(500u*1024u*1024u,512u*1024u*1024u,768u*1024u*1024u,4).preparation;
 q.schedule({{0,0}},compile,4,{0},budget);
 auto until=std::chrono::steady_clock::now()+std::chrono::seconds(5);
 while(q.statistics().built!=1 && std::chrono::steady_clock::now()<until)std::this_thread::yield();assert(q.statistics().built==1);
 q.schedule({{1,1},{2,2},{3,3},{4,4}},compile,4,{0,1,2,3,4},budget);
 while(entered!=4 && std::chrono::steady_clock::now()<until)std::this_thread::yield();
 assert(entered==4);release=true;q.clear();
}''')

    def test_world_lease_survives_retarget_but_changed_inputs_refuse_adoption(self):
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include "Renderer/native/render_core/content_preparation.h"
#include "Renderer/native/render_core/prepared_world_validity.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Part {
 std::unordered_map<std::size_t,std::uint32_t> world;
 std::unordered_map<std::uint64_t,std::uint64_t> topology,coast;
 std::vector<int> rivers;
};
struct Result {std::unique_ptr<Part> ground,terrain,objects;c3x_renderer::WorldPreparationKind kind=c3x_renderer::WorldPreparationKind::combined;std::size_t bytes()const{return 1024;}};
struct Coast {Coast const& world()const{return *this;}unsigned at(std::size_t)const{return 0;}
 std::uint64_t node_revision(std::uint64_t)const{return 0;}};
struct Rivers {using CellProof=std::vector<int>;bool valid(CellProof const&)const{return true;}};
int main(){
 CapturedScene source;c3x_renderer_frame_v1 frame={};frame.world_width_tiles=16;frame.world_height_tiles=16;frame.world_wrap_x=1;
 c3x_renderer_camera_identity_v1 identity{};source.publication_scope(frame,identity,1);
 c3x_renderer_tile_v1 tile={};tile.tile_x=2;tile.tile_y=2;tile.tile_flags=C3X_RENDERER_TILE_RENDER;
 tile.anchor_x=20;tile.road_mask=1;bool changed=false;assert(source.publish(tile,changed));
 auto owned=source.world_snapshot();auto id=owned->key(18,2);assert(id==source.key(2,2));
 using Input=std::shared_ptr<CapturedScene::WorldSnapshot const>;using Queue=ContentPreparation<int,Input,Result>;
 Queue q;std::atomic<bool> entered{false},release{false};auto semantic=owned->current(id)->semantic;
 std::weak_ptr<CapturedScene::WorldSnapshot const> lifetime=owned;
 q.schedule({{1,owned}},[&](Input const& input,auto const& stop,unsigned){
  entered=true;while(!release && !stop)std::this_thread::yield();
  assert(input->current(id)->semantic==semantic && input->current(id)->occurrence.anchor_x==0);
  auto result=std::make_unique<Result>();result->ground=std::make_unique<Part>();
  result->terrain=std::make_unique<Part>();result->objects=std::make_unique<Part>();
  result->ground->topology[id]=input->current(id)->semantic;return result;
 },1,{1},Queue::byte_limit);owned.reset();while(!entered)std::this_thread::yield();
 // Published local change replaces world input while the old job continues.
 tile.road_mask=2;assert(source.publish(tile,changed));auto latest=source.world_snapshot();
 tile.anchor_x=55;frame.tiles=&tile;frame.tile_count=1;assert(source.begin(frame));
 assert(source.update(tile,1,2,3,CapturedScene::topology(tile)));source.finish();
 frame.tile_count=0;assert(source.begin(frame));source.finish();
 assert(!source.current(id) && source.world_snapshot()==latest && !lifetime.expired());
 release=true;auto result=q.take(1);q.pause();assert(result);
 Coast coast;Rivers rivers;assert(!prepared_world_valid(*result,coast,*latest,rivers));
 assert(lifetime.expired());q.clear();source={};assert(latest->current(id)->occurrence.road_mask==2);
 assert(!latest->current(latest->key(4,4)));
}
''')

    def test_rigid_batching_preserves_adjacent_draw_order(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        start=source.index('    bool draw_cached_geometry(')
        loop_start=source.index('            for(unsigned i=0;i<selected.size();){',start)
        fallback=source.index('                auto const& first_chunk=selected[i].content();',loop_start)
        end=source.index('            selected.clear();return true;',fallback)
        # Execute the existing fallback merge after packet selection. The
        # packet consumer itself has its separate actual-shader native oracle.
        loop=source[loop_start:source.index('\n',loop_start)]+ '\n'+source[fallback:end]
        run_cpp(r'''#include <vector>
#include <cassert>
struct Mesh {bool rigid_source;int buffer,indices,vertex_offset,index_offset,index_count,index_format,projection_kind;};
struct Draw {Mesh mesh;Mesh const& content()const{return mesh;}};
int main(){
 Mesh a={true,1,2,0,0,36,1,2},b=a;b.rigid_source=false;
 Mesh c=a;c.index_offset=128;
 std::vector<Draw> selected={{a},{a},{b},{a},{c},{c}};
 std::vector<bool> packets(selected.size(),false);
 std::vector<int> parameters={0,1,2,3,4,5};std::vector<unsigned> sizes,order;
 auto issue=[&](Draw const&,int first,unsigned index,unsigned count){assert(first==int(index));sizes.push_back(count);for(unsigned n=0;n<count;++n)order.push_back(index+n);};
'''+loop+r'''
 assert((sizes==std::vector<unsigned>{2,1,1,2}));
 assert((order==std::vector<unsigned>{0,1,2,3,4,5}));
}''')

    def test_draw_prelude_shares_resource_handoff_and_failure_is_terminal(self):
        run_cpp(r'''
#include "Renderer/native/gpu_image_worker_client.h"
#include <cassert>
using namespace c3x_gpu_images;
unsigned calls=0,draws=0,fail=0;long long next_id=10;
int execute(c3x_renderer_gpu_images_v1 const* r,c3x_renderer_gpu_result_v1* out,unsigned* pixels,unsigned count){
 ++calls;draws+=r->command_count;if(fail==2)return C3X_RENDERER_RESULT_ERROR;
 if(fail==1)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
 out->image=next_id++;out->pixel_count=count;if(pixels)for(unsigned i=0;i<count;++i)pixels[i]=0x1234;
 return C3X_RENDERER_RESULT_OK;
}
int main(){
 c3x_renderer_gpu_frame_v1 frame={};frame.struct_size=sizeof(frame);frame.ticket=frame.session=1;
 WorkerClient client(execute,frame);Command draw={};draw.destination=9;
 assert(client.submit(&draw,1));assert(client.create(2,2,Format::bgra32));assert(calls==1&&draws==1&&client.flushed());
 unsigned pixels[4]={};assert(client.submit(&draw,1));assert(client.upload(10,1,pixels,4));assert(calls==2&&draws==2);
 assert(client.submit(&draw,1));assert(client.readback(10,pixels,4));assert(calls==3&&draws==3&&pixels[0]==0x1234);
 assert(client.submit(&draw,1));fail=1;assert(!client.create(2,2,Format::bgra32));assert(client.flushed());
 fail=0;assert(client.create(2,2,Format::bgra32));assert(draws==4); // prelude is not repeated on admission refusal
 assert(client.submit(&draw,1));auto before=calls;assert(!client.create(0,2,Format::bgra32));
 assert(calls==before+1&&draws==5&&client.flushed()); // rejected dimensions still flush the preceding valid draw
 assert(client.submit(&draw,1));fail=2;bool threw=false;try{client.destroy(10);}catch(...){threw=true;}assert(threw);
 auto previous=calls;try{client.flush();client.create(2,2,Format::bgra32);}catch(...){}assert(calls==previous);
}
''')
