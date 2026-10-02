"""Owned compilation survives view replacement; retirement remains a real barrier."""
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.lab.platform import ROOT

class DurablePreparationTests(unittest.TestCase):
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
        loop='            for(unsigned i=0;i<selected.size();){'+source.split('            for(unsigned i=0;i<selected.size();){',1)[1].split('            selected.clear();return true;',1)[0]
        run_cpp(r'''#include <vector>
#include <cassert>
struct Mesh {bool rigid_source;int buffer,indices,vertex_offset,index_offset,index_count,index_format,projection_kind;};
struct Draw {Mesh mesh;Mesh const& content()const{return mesh;}};
int main(){
 Mesh a={true,1,2,0,0,36,1,2},b=a;b.rigid_source=false;
 Mesh c=a;c.index_offset=128;
 std::vector<Draw> selected={{a},{a},{b},{a},{c},{c}};
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
