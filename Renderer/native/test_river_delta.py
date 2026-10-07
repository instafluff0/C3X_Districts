"""River mouths fan into a small delta.

A river that reaches the sea keeps its main outlet and gains two narrower
distributaries. They leave the last reach a little upstream, bend toward the
sea on either side of the outlet, carry the downstream flow tangent and read
as smaller channels (the distance includes a narrow offset). The branches are
deterministic per river edge, so overlapping river pages agree, and the Lab
and Renderer64 game builds (C3X_RENDERER64_FRESH) build the same delta.
"""
import unittest

from Renderer.native.native_cpp_test import run_cpp

PROGRAM = r'''
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include "Renderer/native/render_core/world_topology.h"
#include "Renderer/native/source_fidelity/river_corridor.h"
using namespace c3x_renderer::render_core;
river::Corridor reach(){
 // An east-flowing river along row 8 reaching the sea at column 31.
 WorldTopology world;World dims{32,32,true,false};
 std::vector<std::uint32_t> tiles(512,2u|(2u<<8));world.update(dims,tiles.data(),tiles.size());
 for(int c=20;c<=30;++c)tiles[world.index(c,8)]|=32u<<16;
 for(int r=4;r<=12;++r)tiles[world.index(31,r)]=12u|(12u<<8);
 world.update(dims,tiles.data(),tiles.size());
 hydro::Field field;field.map_width=32;field.map_height=32;field.wraps=true;
 for(int r=3;r<14;r++)for(int c=16;c<34;c++){auto i=world.index(c,r);auto v=world.at(i);
  field.tiles[{c,r}]={c,r,c+r,c-r,int(v&255),int((v>>8)&255),unsigned((v>>16)&255),world.river_flow(i)};
 }
 river::Corridor corridor;corridor.build(field,[](double,double){return 0.;});
 return corridor;
}
int main(){
 auto corridor=reach();
 unsigned mouths=0;river::Terminal mouth{};
 for(auto const& t:corridor.terminals)if(t.mouth){++mouths;mouth=t;}
 assert(mouths==1);
 // Unique branch segments (buckets repeat each segment per covered cell).
 std::vector<river::Segment> branches;
 for(auto const& bucket:corridor.buckets)for(auto const& s:bucket.second)
  if(s.narrow>0){bool seen=false;for(auto const& b:branches)seen=seen || (b.a.x==s.a.x && b.a.y==s.a.y && b.b.x==s.b.x && b.b.y==s.b.y);
   if(!seen)branches.push_back(s);}
 assert(branches.size()==20);
 // Downstream tangents follow each branch.
 for(auto const& s:branches){auto step=river::from_screen(s.b)-river::from_screen(s.a);
  assert(std::hypot(s.flow.x,s.flow.y)>.99 && step.x*s.flow.x+step.y*s.flow.y>0);}
 // The two branch ends lie on opposite sides of the outlet, apart, and
 // read as water narrower than the main channel.
 std::vector<river::Segment> ends;
 for(auto const& s:branches){bool continues=false;
  for(auto const& t:branches)continues=continues || (t.a.x==s.b.x && t.a.y==s.b.y);
  if(!continues)ends.push_back(s);}
 assert(ends.size()==2);
 auto end0=river::from_screen(ends[0].b),end1=river::from_screen(ends[1].b);
 double spread=std::hypot(end0.x-end1.x,end0.y-end1.y);
 std::printf("mouth=(%.2f,%.2f) ends=(%.2f,%.2f) (%.2f,%.2f) spread=%.2f\n",
  mouth.p.x,mouth.p.y,end0.x,end0.y,end1.x,end1.y,spread);
 assert(spread>.5);
 for(auto end:{end0,end1}){
  auto sample=corridor.sample(end);
  assert(std::abs(sample.distance-1.8)<1e-6);
  assert(std::hypot(end.x-mouth.p.x,end.y-mouth.p.y)>.4);
 }
 // Deterministic: an independent build (another page) inserts the same delta.
 auto again=reach();
 for(auto const& bucket:corridor.buckets){auto other=again.buckets.find(bucket.first);
  assert(other!=again.buckets.end() && other->second.size()==bucket.second.size());}
 std::puts("PASS river delta: two narrower distributaries fan out beside the outlet");
}
'''


class RiverDeltaTests(unittest.TestCase):
    def test_lab_mouth_gains_two_distributaries(self):
        run_cpp(PROGRAM)

    def test_game_builds_share_the_delta(self):
        run_cpp('#define C3X_RENDERER64_FRESH 1\n' + PROGRAM)


if __name__ == '__main__':
    unittest.main()
