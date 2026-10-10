"""Explored resources keep moving in fog, as water does.

Resource animations (whales, fish, animals) on explored tiles that were not
currently visible were posed at time zero and did not count as moving, so
they stood still while the water around them moved (the user, October 10).
Never-explored objects still contribute nothing.
"""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class FoggedResourceMotionTests(unittest.TestCase):
    def test_explored_fogged_resources_advance_and_count_as_moving(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        start = source.index('            unsigned visibility=visibility_pass?visibility_coverage.state(anchor.tile_x,anchor.tile_y):2;')
        end = source.index('animation.mesh.duration,anchor.seed);', start) + len('animation.mesh.duration,anchor.seed);')
        pose = source[start:end]
        self.assertEqual(source.count('++visible_resource_animations;++moving_resources;'), 2)
        run_cpp(r'''
#include <cassert>
#include <cstdint>
namespace c3x_renderer {double ambient_animation_time(std::int64_t ticks,std::int64_t,double,unsigned){return double(ticks);}}
struct Coverage {unsigned value=0;unsigned state(int,int)const{return value;}} visibility_coverage;
struct Anchor {int tile_x=0,tile_y=0;unsigned seed=0;} anchor;
struct Mesh {double duration=1;};struct Animation {Mesh mesh;} animation;
struct Frame {std::int64_t presentation_frequency=24000000;} frame;
bool visibility_pass=true;std::int64_t ticks=4321;
double pose_time(unsigned state,bool& skipped){visibility_coverage.value=state;skipped=true;
 for(int once=0;once<1;++once){
''' + pose + r'''
  skipped=false;return time;}
 return -1;}
int main(){
 bool skipped=false;
 assert(pose_time(2,skipped)==4321.0&&!skipped);   // visible
 assert(pose_time(1,skipped)==4321.0&&!skipped);   // explored, fogged: moves too
 pose_time(0,skipped);assert(skipped);              // never explored: absent
}
''')


if __name__ == '__main__':
    unittest.main()
