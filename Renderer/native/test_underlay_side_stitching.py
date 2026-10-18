"""Neighbouring flat underlays meet without T-junctions.

The underlay's density differs between neighbours (river tiles 32, coastal 16,
others 8 at 1x). Unequal grids left T-junctions on shared sides, which showed
in game as thin dark seams on beaches and shallows (an early-game save:
(27,73)|(28,74) and (30,70)|(31,69)). The test runs the production
append_ground_layer and underlay call, extracted from ground_compiler.h, for
every pair of neighbouring densities and checks that both tiles place the same
vertices on their shared side, and that each tile's triangles still cover it
exactly once.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPILER = ROOT / "Renderer/native/source_fidelity/ground_compiler.h"
PIPELINE = ROOT / "Renderer/native/source_fidelity/prepared_ground.h"


def underlay_block(source):
    # The underlay's grid setup precedes append_ground_layer (absent before
    # the fix, when the underlay call used each tile's own grid alone).
    anchor = "    // Neighbouring underlays differ in density"
    start = source.index(anchor) if anchor in source else source.index("    auto append_ground_layer = ")
    end = source.index("    if (input.ground < 11 && (!input.fidelity_profile || input.draw_marsh))")
    return source[start:end]


HARNESS = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include "Renderer/lab/shared/natural/world.h"
#include "Renderer/lab/shared/natural/vertex.h"
using MapVertex=c3x_renderer::fidelity::MapVertex;
using UINT=unsigned;
struct CachedGroundGrid {int divisions=0;float layer=0;std::vector<MapVertex> vertices;std::vector<std::array<float,3>> samples;
 int sample_stride(int)const{return 0;} MapVertex project(std::size_t,int,int)const{return {};}};
RULE
// One test world: each tile's river flag and centre shore distance.
struct Site {bool river=false;double shore=5;};
std::map<std::pair<int,int>,Site> world;
struct Mesh {std::vector<MapVertex> vertices;std::vector<UINT> indices;};
Mesh underlay(int c,int r,int width){
 struct Input {bool world_ground=false,retain_ground_grids=false,reuse_nested_ground_grids=false,prewarming=false;
  bool cull_river_reach=false,fidelity_profile=true,pickup_profile=true;int flat_grid=8;} input;
 struct Destination {std::vector<MapVertex> underlay_vertices;std::vector<UINT> underlay_indices;
  std::vector<CachedGroundGrid> pending_grids;} destination;
 struct Frame {int tile_width,tile_height;} frame{width,width/2};
 struct Tile {int tile_x,tile_y;} tile{c+r,c-r};
 std::vector<CachedGroundGrid> const* cached_grid_source=nullptr;
 unsigned ground_grid_hit_counter=0;
 Site self=world[{c,r}];
 bool river_near=self.river;
 // The callers' flat_grid rule (c3x_renderer.cpp make_ground_job and the
 // foreground pickup path): coastal tiles 16 at >=96 px, others 8.
 input.flat_grid=std::abs(self.shore)<1.5 && width>=96?16:8;
 auto river_channel_cut=[](float){return 0.f;};
 auto river_node_distance=[](float,float,unsigned){return 1000.f;};
 struct Point {float relief[3]={},normal_delta[2]={};} point;
 auto ground_point_at=[&](float,float)->Point&{return point;};
 auto cancelled=[](){return false;};
 auto make_ground_vertex=[&](float u,float v,float layer){
  MapVertex out={};out.u=u;out.v=v;out.x=u;out.y=v;out.world_x=float(c)+u;out.world_y=float(r)+1-v;out.material_grass=layer;
  return out;};
 struct Shore {double distance;};
 auto shore_sample_at=[&](float u,float v){Site s=world[{int(std::floor(u)),int(std::floor(v))}];return Shore{s.shore};};
 // prepared_ground.h's neighbour density, by the same rule as the tile's own.
 auto underlay_grid_at=[&](int nc,int nr){auto found=world.find({nc,nr});
  return found==world.end()?0:NEIGHBOUR;};
 (void)underlay_grid_at;(void)shore_sample_at;
BLOCK
 return {destination.underlay_vertices,destination.underlay_indices};
}
// World-space points one tile places on a side: v=0 (r+1), u=1 (c+1), v=1 (r), u=0 (c).
std::vector<std::pair<float,float>> side_points(Mesh const& m,unsigned side){
 std::set<std::pair<float,float>> used;
 for(auto i:m.indices){auto const& p=m.vertices[i];
  bool on=side==0?p.v==0.f:side==1?p.u==1.f:side==2?p.v==1.f:p.u==0.f;
  if(on)used.insert({p.world_x,p.world_y});}
 return {used.begin(),used.end()};
}
// Triangles cover the unit cell exactly once and meet each side only along
// consecutive side points: no gap, overlap, sliver or T-junction.
void watertight(Mesh const& m){
 double area=0;
 std::array<std::map<float,int>,4> segments;
 for(std::size_t t=0;t<m.indices.size();t+=3){
  MapVertex const* p[3]={&m.vertices[m.indices[t]],&m.vertices[m.indices[t+1]],&m.vertices[m.indices[t+2]]};
  double a=.5*((p[1]->u-p[0]->u)*(p[2]->v-p[0]->v)-(p[2]->u-p[0]->u)*(p[1]->v-p[0]->v));
  assert(a>1e-9);area+=a;
  for(int e=0;e<3;++e){auto const& q=*p[e];auto const& s=*p[(e+1)%3];
   for(unsigned side=0;side<4;++side){
    bool on=side==0?q.v==0.f&&s.v==0.f:side==1?q.u==1.f&&s.u==1.f:side==2?q.v==1.f&&s.v==1.f:q.u==0.f&&s.u==0.f;
    if(on)++segments[side][side&1?std::min(q.v,s.v):std::min(q.u,s.u)];}}
 }
 assert(std::abs(area-1)<1e-5);
 for(unsigned side=0;side<4;++side){
  float covered=0;for(auto const& s:segments[side]){assert(s.second==1);}
  std::vector<float> points;for(auto i:m.indices){auto const& p=m.vertices[i];
   bool on=side==0?p.v==0.f:side==1?p.u==1.f:side==2?p.v==1.f:p.u==0.f;if(on)points.push_back(side&1?p.v:p.u);}
  std::sort(points.begin(),points.end());points.erase(std::unique(points.begin(),points.end()),points.end());
  assert(points.front()==0.f && points.back()==1.f && segments[side].size()==points.size()-1);
  for(std::size_t i=0;i+1<points.size();++i)assert(segments[side].count(points[i]));
  (void)covered;
 }
}
unsigned grid(Site s,int width){int flat=std::abs(s.shore)<1.5&&width>=96?16:8;return s.river?std::max(flat,width>=96?32:16):flat;}
int main(){
 // The user's seams: river tile (c50,r-23) beside coast (c51,r-23), across
 // u=51; river tile (c50,r-20) beside coast (c50,r-19), across v=-19.
 world={{{50,-23},{true,.5}},{{51,-23},{false,-.5}},{{50,-20},{true,.5}},{{50,-19},{false,-.4}}};
 {auto a=underlay(50,-23,128),b=underlay(51,-23,128);
  auto shared_a=side_points(a,1),shared_b=side_points(b,3);
  std::printf("u=51 side: river tile %zu points, coast tile %zu points\n",shared_a.size(),shared_b.size());
  assert(shared_a==shared_b && shared_a.size()==33);}
 {auto a=underlay(50,-20,128),b=underlay(50,-19,128);
  auto shared_a=side_points(a,0),shared_b=side_points(b,2);
  std::printf("v=-19 side: river tile %zu points, coast tile %zu points\n",shared_a.size(),shared_b.size());
  assert(shared_a==shared_b && shared_a.size()==33);}
 // Every density pair, on every side, at 1x and zoomed out.
 Site kinds[3]={{false,5},{false,.5},{true,.5}};
 int across[4][2]={{0,1},{1,0},{0,-1},{-1,0}};
 unsigned cases=0;
 for(int width:{128,64})for(auto self:kinds)for(auto other:kinds)for(unsigned side=0;side<4;++side){
  world.clear();world[{10,10}]=self;world[{10+across[side][0],10+across[side][1]}]=other;
  auto a=underlay(10,10,width),b=underlay(10+across[side][0],10+across[side][1],width);
  watertight(a);watertight(b);
  auto pa=side_points(a,side),pb=side_points(b,(side+2)%4);
  assert(pa==pb);
  assert(pa.size()==std::max(grid(self,width),grid(other,width))+1);
  // A side is split only toward a denser neighbour; matching tiles keep
  // the plain grid (no extra vertices or triangles).
  if(grid(self,width)==grid(other,width)){
   unsigned n=grid(self,width);
   assert(a.vertices.size()==(n+1)*(n+1) && a.indices.size()==n*n*6);
  }
  ++cases;
 }
 std::printf("%u neighbour cases meet exactly\n",cases);
}
'''


class UnderlaySideStitchingTests(unittest.TestCase):
    def test_neighbouring_underlays_share_side_vertices(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = COMPILER.read_text()
        rule = ""
        neighbour = "0"
        if "inline int underlay_grid(" in source:
            rule = "inline int underlay_grid(" + source.split("inline int underlay_grid(", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
            neighbour = "underlay_grid(found->second.river,found->second.shore,width)"
        program = HARNESS.replace("RULE", rule).replace("NEIGHBOUR", neighbour).replace("BLOCK", underlay_block(source))
        with tempfile.TemporaryDirectory(prefix="c3x-underlay-stitch-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            subprocess.run([compiler, "-std=c++17", "-O1", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_pipeline_supplies_neighbour_density_by_the_same_rule(self):
        # The neighbour callable must use underlay_grid, with observed reads,
        # so a neighbour's river or shore change re-places the shared side.
        pipeline = PIPELINE.read_text()
        callable_source = pipeline.split("auto ground_underlay_grid_at=", 1)[1].split("};", 1)[0]
        self.assertIn("underlay_grid(", callable_source)
        self.assertIn("ground_observe_world(", callable_source)
        self.assertIn("ground_query_scratch.rivers.river_affects(", callable_source)
        self.assertIn("ground_shore_sample_at(", callable_source)
        call = pipeline.split("c3x_renderer::fidelity::compile_ground_surfaces(", 1)[1].split(");", 1)[0]
        self.assertIn("ground_underlay_grid_at", call)


if __name__ == "__main__":
    unittest.main()
