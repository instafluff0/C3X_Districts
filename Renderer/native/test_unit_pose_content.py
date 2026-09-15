"""Execute immutable unit CPU compilation and owned-input queue behavior."""
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.lab.platform import ROOT


class UnitPoseContentTests(unittest.TestCase):
    def test_parallel_exact_content_and_native_pose_dependencies(self):
        run_cpp(r'''
#include "Renderer/native/unit_pose_content.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 auto mesh=std::make_shared<AnimationMesh>();mesh->bones=1;mesh->frames=2;mesh->duration=1;
 mesh->vertices.resize(3);mesh->indices={0,1,2};
 for(unsigned i=0;i<3;++i){auto& v=mesh->vertices[i];
  v.source.position[0]=i==1?.2f:-.2f;v.source.position[1]=i==2?.2f:-.2f;v.source.position[2]=.3f;v.source.normal[2]=1;
  v.tangent={1,0,0};v.bitangent={0,1,0};v.weights={1,0,0,0};}
 for(unsigned f=0;f<2;++f)for(float v:{1.f,0.f,0.f,0.f,0.f,1.f,0.f,0.f,0.f,0.f,1.f,0.f,float(f)*.1f,0.f,0.f,1.f})mesh->palettes.push_back(v);
 UnitPoseInput input;input.source.meshes={mesh};input.width=input.height=240;input.anchor_x=input.anchor_y=120;
 input.direction=3;input.phase=.5;input.light_x=1;
 std::atomic<bool> cancel{false};auto control=UnitPoseCompiler{}(input,cancel,0);assert(control);
 auto selected_input=input;selected_input.gpu_shadow=true;
 auto selected=UnitPoseCompiler{}(selected_input,cancel,0);assert(selected);
 assert(selected->uploads==control->uploads && selected->shadow.heights.empty() && selected->ground_shadow.empty());
 assert(selected->ground_projection[8]==control->shadow.extent && !selected->shadow_triangles.empty());
 // Independent consumer of the selected pass input reproduces the CPU raster.
 std::vector<float> heights(control->shadow.heights.size(),-1.f);int extent=control->shadow.extent;
 for(auto const& t:selected->shadow_triangles)for(int y=0;y<extent;++y)for(int x=0;x<extent;++x){
  float px=x+.5f,py=y+.5f,u=((t[4]-px)*(t[9]-py)-(t[5]-py)*(t[8]-px))/t[3];
  float v=((t[8]-px)*(t[1]-py)-(t[9]-py)*(t[0]-px))/t[3],w=1-u-v,z=u*t[2]+v*t[6]+w*t[10];
  if(u>=0&&v>=0&&w>=0&&z>=.002f)heights[y*extent+x]=std::max(heights[y*extent+x],z);
 }
 assert(heights==control->shadow.heights);
 using Pool=render_core::ContentPreparation<int,UnitPoseInput,UnitPoseContent>;
 Pool pool;pool.configure({},UnitPoseCompiler{},2);
 assert(pool.offer({1,input},32));assert(pool.offer({1,input},32));
 auto other=input;other.direction=7;assert(pool.offer({2,other},32));
 // Releasing the original owner leaves immutable job leases valid.
 mesh.reset();input.source.meshes.clear();other.source.meshes.clear();
 auto result=pool.take(1);assert(result && result->uploads==control->uploads && result->shadow.heights==control->shadow.heights && result->ground_shadow==control->ground_shadow);
 // Independent legacy finishing formula validates worker-prepared shadow bytes.
 for(int y=0;y<input.height;++y)for(int x=0;x<input.width;++x){
  float sx=(float(x)+.5f-float(input.anchor_x))/(64*input.zoom),sy=(float(y)+.5f-float(input.anchor_y))/(32*input.zoom);
  float fade=std::clamp(float(std::min({x,y,input.width-1-x,input.height-1-y}))/3,0.f,1.f);
  unsigned expected=unsigned(255*lighting::c3x_dynamic_shadow_opacity*input.shadow_strength*fade*control->shadow.coverage((sx+sy)*.5f,(sy-sx)*.5f));
  assert(result->ground_shadow[std::size_t(y)*input.width+x]==expected);
 }
 auto turned=pool.take(2);assert(turned && turned->uploads!=control->uploads);
 pool.pause();assert(pool.statistics().built==2 && pool.statistics().active_peak<=2);
 pool.clear();assert(pool.statistics().bytes==0 && pool.statistics().pending==0);
 // Reject oversized pass inputs before skinning or allocating the record list.
 auto oversized=std::make_shared<AnimationMesh>();oversized->indices.resize((16u*1024u*1024u/48+1)*3);
 UnitPoseInput excessive;excessive.gpu_shadow=true;excessive.source.meshes={oversized};
 assert(!UnitPoseCompiler{}(excessive,cancel,0));
 // Cancellation cannot publish a partial pose.
 cancel=true;UnitPoseInput cancelled_input;cancelled_input.source.meshes={std::make_shared<AnimationMesh>()};
 assert(!UnitPoseCompiler{}(cancelled_input,cancel,0));
}
''')

    def test_cpu_validity_reuses_owner_color_but_rejects_pose_changes(self):
        source=(ROOT/"Renderer/native/unit_body_renderer.h").read_text()
        key="struct Key {"+source.split("struct Key {",1)[1].split("    struct PoseSelection",1)[0]
        function="PoseKey content_key("+source.split("PoseKey content_key(",1)[1].split("    unsigned preparation_workers",1)[0]
        run_cpp(r'''
#include <array>
#include <cassert>
struct Owner {
'''+key+"using PoseKey=std::array<int,11>;"+function+r'''
};
int main(){
 Owner owner;Owner::Key key={1,2,3,4,16,240,240,1000,12,0,0x205bdd};
 auto original=owner.content_key(key);assert(owner.content_key(key,true)!=original);auto other=key;other.color=0xdd4422;
 assert(!(other==key) && owner.content_key(other)==original); // Reuse CPU content, recolor on GPU.
 for(auto field:{&Owner::Key::action,&Owner::Key::direction,&Owner::Key::cursor,&Owner::Key::frames,
                 &Owner::Key::width,&Owner::Key::height,&Owner::Key::scale_milli,&Owner::Key::hour,&Owner::Key::season}) {
  other=key;++(other.*field);assert(owner.content_key(other)!=original);
 }
 other=key;++other.unit;assert(owner.content_key(other)!=original);
}
''')

    def test_advancing_input_replaces_unused_predictions_under_pressure(self):
        run_cpp(r'''
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Result {int value;std::size_t bytes()const{return 8u*1024u*1024u;}};
int main(){
 ContentPreparation<int,int,Result> pool;
 pool.configure({},[](auto const& input,auto const&,unsigned){return std::make_unique<Result>(Result{input});});
 assert(pool.offer({1,1},32));
 auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
 while(pool.statistics().built<1){assert(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();}
 // The unused result closes the speculative refill gate.
 assert(pool.offer({2,2},32));assert(pool.offer({2,2},32,true));
 while(pool.statistics().built<2){assert(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();}
 assert(pool.take(2)->value==2);assert(pool.statistics().evicted==0);
 assert(pool.statistics().peak_bytes<=16u*1024u*1024u);
}
''')

    def test_advancing_predictions_keep_due_order(self):
        run_cpp(r'''
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Result {int value;std::size_t bytes()const{return 1;}};
int main(){
 ContentPreparation<int,int,Result> pool;std::atomic<bool> entered{false},release{false};std::vector<int> order;
 pool.configure({},[&](int const& input,auto const&,unsigned){
  order.push_back(input);if(input==1){entered=true;while(!release)std::this_thread::yield();}
  return std::make_unique<Result>(Result{input});
 });
 pool.offer({1,1},32,true);while(!entered)std::this_thread::yield();
 pool.offer({9,9},32);pool.offer({2,2},32,true);pool.offer({3,3},32,true);release=true;
 auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
 while(pool.statistics().built<4){assert(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();}
 assert(pool.take(1));assert(pool.take(2));assert(pool.take(3));assert(pool.take(9));pool.pause();
 assert((order==std::vector<int>{1,2,3,9})); // No reverse-order churn from newer predictions.
}
''')

    def test_owned_queue_joins_matching_work_and_releases_leases(self):
        run_cpp(r'''
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Result {int value;std::size_t bytes()const{return 1;}};
int main(){
 using Pool=ContentPreparation<int,std::shared_ptr<int const>,Result>;
 Pool pool;std::atomic<bool> entered{false},release{false};std::atomic<int> calls{0};
 pool.configure({},[&](auto const& input,auto const& cancelled,unsigned){
  ++calls;entered=true;while(!release && !cancelled)std::this_thread::yield();
  return std::make_unique<Result>(Result{*input});
 },2);
 auto input=std::make_shared<int const>(42);std::weak_ptr<int const> lease=input;
 assert(pool.offer({1,input},1));while(!entered)std::this_thread::yield();
 assert(pool.offer({1,input},1)); // Matching active work is never duplicated.
 input.reset();assert(!lease.expired());release=true;
 assert(pool.take(1,true)->value==42);pool.clear();assert(lease.expired() && calls==1);
}
''')


if __name__ == "__main__":
    unittest.main()
