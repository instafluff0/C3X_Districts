"""River mouths fan into a small delta whose every branch reaches the sea.

A river that reaches the sea keeps its main outlet and gains two to four
narrower distributaries. They leave the last reach a little upstream, carry
the downstream flow tangent, read as smaller channels (the distance includes
a narrow offset) and fan out to separate water tiles beside the outlet. Each
ends at its water tile's center, as the outlet does, where the optical shore
keeps a water core. Branches once ended at a fixed angle and length, and on
real coasts many stopped on the beach short of the sea (user, October 19).
The branches are deterministic per river edge, so overlapping river pages
agree, and the Lab and Renderer64 game builds (C3X_RENDERER64_FRESH) build
the same delta.
"""
import unittest

from Renderer.native.native_cpp_test import run_cpp

PROGRAM = r'''
#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include "Renderer/native/render_core/world_topology.h"
#include "Renderer/native/render_core/terrain_query.h"
#include "Renderer/native/source_fidelity/river_corridor.h"
using namespace c3x_renderer::render_core;
struct Reach {WorldTopology world;river::Corridor corridor;};
// An east-flowing river along row r0 reaching the sea where water(c,r).
template<class Water> void reach(Reach& out,int r0,Water water){
 World dims{128,128,true,false};
 std::vector<std::uint32_t> tiles(128*128/2,2u|(2u<<8));out.world.update(dims,tiles.data(),tiles.size());
 for(int r=0;r<64;++r)for(int c=0;c<64;++c){auto i=out.world.index(c,r);if(i!=std::size_t(-1) && water(c,r))tiles[i]=12u|(12u<<8);}
 int end=20;while(!water(end,r0-1) && !water(end,r0))++end;
 for(int c=20;c<end;++c)tiles[out.world.index(c,r0)]|=32u<<16;
 out.world.update(dims,tiles.data(),tiles.size());
 hydro::Field field;field.map_width=128;field.map_height=128;field.wraps=true;
 for(int r=r0-8;r<r0+8;r++)for(int c=end-12;c<end+6;c++){auto i=out.world.index(c,r);auto v=out.world.at(i);
  field.tiles[{c,r}]={c,r,c+r,c-r,int(v&255),int((v>>8)&255),unsigned((v>>16)&255),out.world.river_flow(i)};
 }
 out.corridor.build(field,[](double,double){return 0.;});
}
// Unique branch segments (buckets repeat each segment per covered cell).
std::vector<river::Segment> branches(river::Corridor const& corridor){
 std::vector<river::Segment> found;
 for(auto const& bucket:corridor.buckets)for(auto const& s:bucket.second)
  if(s.narrow>0){bool seen=false;for(auto const& b:found)seen=seen || (b.a.x==s.a.x && b.a.y==s.a.y && b.b.x==s.b.x && b.b.y==s.b.y);
   if(!seen)found.push_back(s);}
 return found;
}
std::vector<hydro::P> ends(std::vector<river::Segment> const& found){
 std::vector<hydro::P> result;
 for(auto const& s:found){bool continues=false;
  for(auto const& t:found)continues=continues || (t.a.x==s.b.x && t.a.y==s.b.y);
  if(!continues)result.push_back(river::from_screen(s.b));}
 return result;
}
// The production shore field's signed distance (tiles; negative in the sea).
double shore(WorldTopology const& world,hydro::P end){
 ShoreField<std::function<Tile(int,int)>> field(world.dimensions(),[&](int c,int r){return world.tile(c,r);});
 Point p{end.x-.5,end.y-.5};std::vector<CoastSegment> coast;
 for(int y=int(std::floor(p.y))-3;y<=int(std::floor(p.y))+3;++y)for(int x=int(std::floor(p.x))-3;x<=int(std::floor(p.x))+3;++x){
  auto const& cell=field.cell(x,y);coast.insert(coast.end(),cell.begin(),cell.end());}
 return field.sample(p,coast).distance;
}
int main(){
 // A straight coast at column 31.
 Reach straight;reach(straight,8,[](int c,int r){return c>=31;});
 unsigned mouths=0;river::Terminal mouth{};
 for(auto const& t:straight.corridor.terminals)if(t.mouth){++mouths;mouth=t;}
 assert(mouths==1);
 auto found=branches(straight.corridor);auto tips=ends(found);
 assert(tips.size()>=2 && tips.size()<=4 && found.size()==12*tips.size());
 // Downstream tangents follow each branch.
 for(auto const& s:found){auto step=river::from_screen(s.b)-river::from_screen(s.a);
  assert(std::hypot(s.flow.x,s.flow.y)>.99 && step.x*s.flow.x+step.y*s.flow.y>0);}
 // Each branch ends at its own water tile's center, narrower than the main
 // channel, apart from the others.
 for(unsigned i=0;i<tips.size();++i){auto end=tips[i];
  assert(std::abs(end.x-std::floor(end.x)-.5)<1e-9 && std::abs(end.y-std::floor(end.y)-.5)<1e-9);
  assert(std::floor(end.x)>=31);
  assert(std::abs(straight.corridor.sample(end).distance-1.8)<1e-6);
  assert(std::hypot(end.x-mouth.p.x,end.y-mouth.p.y)>.4);
  for(unsigned j=0;j<i;++j)assert(std::hypot(end.x-tips[j].x,end.y-tips[j].y)>.9);
 }
 // Deterministic: an independent build (another page) inserts the same delta.
 Reach again;reach(again,8,[](int c,int r){return c>=31;});
 for(auto const& bucket:straight.corridor.buckets){auto other=again.corridor.buckets.find(bucket.first);
  assert(other!=again.corridor.buckets.end() && other->second.size()==bucket.second.size());}
 // Every branch reaches the sea on straight, diagonal, inlet, corner and
 // notched coasts: past the shore by more than the river surface's fade
 // (outlet alpha is zero below -.28), never stopped on the beach.
 std::vector<std::function<bool(int,int,int)>> coasts={
  [](int c,int,int){return c>=36;},
  [](int c,int r,int r0){return c-(r-r0)>=36;},
  [](int c,int r,int r0){return c+(r-r0)>=36;},
  [](int c,int r,int r0){return c>=37 || (c==36 && std::abs(r-r0)<=1);},
  [](int c,int r,int r0){return c>=36 && r>=r0;},
  [](int c,int r,int r0){return c>=37 || (c==36 && r==r0);},
 };
 unsigned total=0,widest=0;double worst=-9;
 for(auto const& coast:coasts)for(int r0=10;r0<54;++r0){
  Reach river;reach(river,r0,[&](int c,int r){return coast(c,r,r0);});
  bool has_mouth=false;for(auto const& t:river.corridor.terminals)has_mouth=has_mouth || t.mouth;
  if(!has_mouth)continue;
  auto tips=ends(branches(river.corridor));
  widest=std::max<unsigned>(widest,unsigned(tips.size()));
  for(auto end:tips){double distance=shore(river.world,end);worst=std::max(worst,distance);++total;}
 }
 std::printf("branch ends=%u worst shore distance=%.3f widest delta=%u\n",total,worst,widest);
 assert(total>250 && worst<-.25);
 // More than two branches where the coast offers more water tiles.
 assert(widest>=3);
 std::puts("PASS river delta: two to four distributaries, each ending in the sea");
}
'''


class RiverDeltaTests(unittest.TestCase):
    def test_lab_mouth_branches_reach_the_sea(self):
        run_cpp(PROGRAM, timeout=600)

    def test_game_builds_share_the_delta(self):
        run_cpp('#define C3X_RENDERER64_FRESH 1\n' + PROGRAM, timeout=600)


if __name__ == '__main__':
    unittest.main()
