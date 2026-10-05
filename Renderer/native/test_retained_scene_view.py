"""Real completed-view borrow survives partial construction and exceptions."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

class RetainedSceneViewTests(unittest.TestCase):
    def test_interleaved_front_and_preparation_share_immutable_content(self):
        run_cpp(r'''
#include "Renderer/native/render_core/retained_scene_view.h"
#include "Renderer/native/render_core/scene_membership.h"
#include <cassert>
#include <stdexcept>
using namespace c3x_renderer::render_core;
struct Chunk {struct Bounds {int left,top,right,bottom;} bounds{};
 int translation_x=0,translation_y=0;float natural_projection[4]={};};
int main(){
 SceneMembership<Chunk,2> membership;Chunk chunk;
 auto mesh=std::make_shared<int>(7);std::weak_ptr<int> weak=mesh;
 membership.retain({1,1},mesh);membership.edit(0).push_back(chunk);
 int width=128,epoch=1;std::vector<int> anchors{4,8};
 auto fields=std::tie(width,epoch,membership,anchors);
 auto completed=retain_scene_view(fields);
 auto old=membership.revision();mesh.reset();
 membership.clear();width=256;epoch=2;anchors={9};
 auto pending=membership.revision();assert(pending!=old&&!weak.expired());
 for(int i=0;i<8;++i){
  try {
   auto front=completed.borrow(fields);
   assert(width==128&&epoch==1&&membership.revision()==old&&anchors.size()==2);
   // Animation may update occurrence state, but not the pending view.
   membership.edit(0)[0].translation_x=i;old=membership.revision();
   assert(old!=pending);
   if(i%2)throw std::runtime_error("pending unit asset");
  }catch(std::runtime_error const&){}
  assert(width==256&&epoch==2&&membership.revision()==pending&&anchors==std::vector<int>{9});
 }
 // Cancelling scratch cannot retire the displayed generation.
 membership.clear();assert(!weak.expired());
 {auto front=completed.borrow(fields);assert(membership[0][0].translation_x==7);}
}
''')


    def test_pressure_retires_only_optional_mesh_pins_and_retries_once(self):
        source=Path('Renderer/native/c3x_renderer.cpp').read_text()
        start=source.index('                            if(!complete && completed_scene_usable()')
        end=source.index('\n#endif',start)
        retry=source[start:end]
        run_cpp(r'''
#include <cassert>
#include <atomic>
#include <vector>
bool retained=true;int retired=0,services=0;
bool completed_scene_usable(){return retained;}
void retire_completed_scene(){retained=false;++retired;}
struct {int pauses=0;void pause_motion(long long){++pauses;}}unit_instances;
struct State {
 int discarded=0,renders=0;bool succeeds=true;
 struct {void write(char const*,char const*,bool){}}trace;
 void discard_scene_view(){++discarded;}
 template<class F>bool render(int,int&,int,std::atomic<bool>*,int,void*,int,void*,F service){++renders;service();return succeeds;}
}renderer_state;
struct Worker {
 std::atomic<bool> camera_cancelled{false};long long visual_ticks=123;int job_frame=0,output=0;
 void service_camera_preparation(){++services;}
 bool run(bool complete){
'''+retry+r'''
 return complete;
 }
};
int main(){
 Worker worker;
 assert(worker.run(true)&&retired==0&&renderer_state.renders==0);
 worker.camera_cancelled=true;assert(!worker.run(false)&&retired==0);
 worker.camera_cancelled=false;
 assert(worker.run(false)&&retired==1&&unit_instances.pauses==1&&renderer_state.discarded==1&&services==1);
 assert(!worker.run(false)&&renderer_state.renders==1); // no retry loop after optional pins are gone
}
''')

if __name__=='__main__':unittest.main()
