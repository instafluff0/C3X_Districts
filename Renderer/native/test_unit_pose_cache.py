"""Execute production unit-pose admission and bounded memory transitions."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class UnitPoseCacheTests(unittest.TestCase):
    def test_ambient_units_have_stable_offsets_without_changing_directed_cursors(self):
        source=(ROOT/"Renderer/native/unit_body_renderer.h").read_text()
        body="NativeUnitDraw draw;"+source.split("NativeUnitDraw draw;",1)[1].split("        int scale_milli=",1)[0]
        run_cpp(r'''
#include "Renderer/native/animation_runtime.h"
#include "Renderer/native/unit_animation_runtime.h"
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
using namespace c3x_renderer;
struct Action {bool loop=true,ambient=true;double duration=8.0/3;unsigned frames=81;};
bool sample(c3x_renderer_unit_v1 request,Action const* action,int& frame,double& phase){
 char const* failure_reason=nullptr;
'''+body+r'''
 (void)failure_reason;frame=pose_cursor;phase=pose.phase;return true;
}
int main(){
 c3x_renderer_unit_v1 request={};request.unit_id=10000;request.action=1;request.direction=3;
 request.action_cursor=7;request.frame_count=16;request.sprite_width=request.sprite_height=191;
 request.presentation_time_ticks=1000000;request.presentation_frequency=1000000;
 Action ambient;int a,b,c;double phase=0;
 assert(sample(request,&ambient,a,phase));request.unit_id++;assert(sample(request,&ambient,b,phase));
 request.unit_id++;assert(sample(request,&ambient,c,phase));assert(a!=b && a!=c && b!=c);
 request.unit_id=10000;request.body_x=400;request.body_y=-100;request.projection_scale_milli=1500;
 assert(sample(request,&ambient,b,phase) && a==b);
 request.presentation_time_ticks+=8000000;assert(sample(request,&ambient,b,phase) && a==b);
 Action directed;directed.ambient=false;directed.loop=false;
 for(int action:{2,3,8})for(int id:{10000,10001,10002}) {
  request.action=action;request.unit_id=id;assert(sample(request,&directed,b,phase));
  assert(b==7 && phase==7.0/15); // Includes the native-directed fidget action.
 }
 request.action_cursor=1000;assert(sample(request,&directed,b,phase) && b==15 && phase==1);
}
''')

    def test_larger_cache_shrinks_and_oversize_admission_preserves_current_pixels(self):
        source=(ROOT/"Renderer/native/unit_body_renderer.h").read_text()
        configure="void configure_pose_cache("+source.split("void configure_pose_cache(",1)[1].split("    template<class T>",1)[0]
        admission=source.split("        // This optional owner retains exact posed pixels",1)[1].split('        failure_reason="none";return true;',1)[0]
        admission="// This optional owner retains exact posed pixels"+admission
        run_cpp(r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
struct State {
 struct Cached {unsigned key;std::uint64_t used;std::vector<std::uint32_t> pixels;unsigned cast_pixels;};
 std::vector<Cached> cache;std::vector<std::uint32_t> pixels;
 std::size_t cache_bytes=0,pose_cache_budget=8*1024*1024,pose_cache_entries=128;
 std::uint64_t serial=0;unsigned cast_pixels=7;
'''+configure+r'''
 void admit(unsigned key) {
'''+admission+r'''
 }
};
int main() {
 State state;state.pixels.assign(1024*1024,0xff123456); // Four MiB current body.
 state.admit(1);state.admit(2);state.admit(3);
 assert(state.cache.size()==2 && state.cache[0].key==2 && state.cache_bytes==8*1024*1024);
 state.configure_pose_cache(true);assert(state.cache.size()==2 && state.pose_cache_budget==256*1024*1024);
 for(unsigned key=4;key<=8;++key)state.admit(key);
 assert(state.cache.size()==7 && state.cache_bytes==28*1024*1024);
 state.configure_pose_cache(false);
 assert(state.cache.size()==2 && state.cache[0].key==7 && state.cache[1].key==8);
 assert(state.pose_cache_entries==128 && state.cache_bytes==8*1024*1024);
 state.pose_cache_budget=1;auto current=state.pixels;
 state.admit(9);assert(state.pixels==current && state.cache.size()==2);
 state.pose_cache_budget=8*1024*1024;state.pose_cache_entries=1;
 state.admit(10);assert(state.cache.size()==1 && state.cache[0].key==10);
 assert(state.cache[0].pixels==current && state.cache[0].cast_pixels==7);
 state.pixels[0]=0;assert(state.cache[0].pixels[0]==0xff123456); // Independent owner.
}
''')


if __name__ == "__main__":
    unittest.main()
