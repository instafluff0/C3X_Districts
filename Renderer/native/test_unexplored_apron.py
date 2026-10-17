"""Ground in the rim of unexplored tiles, for the native fog blend.

The user's native Civ III reference (October 17): fog of war and unexplored
black blend across the tile edge, half at the edge and complete a fifth of a
tile into either tile, so native shows a sliver of an unexplored tile's own
ground at its rim. Revealed ground keeps its real relief (ViewerTopology
real relief), so explored tiles now extend:

- flat ground at the datum, with the rim's own materials and no relief or
  features, a quarter tile into every unexplored edge and corner neighbour
  (natural terrain here; underlay, bed, water and river in the ground
  compiler);
- on screen-down edges (x=column+1, y=row) raised above the datum, the
  edge's own surface continued down to that rim. Without it the flat
  underlay showed between the raised edge and the fog as a dull band (the
  user's screenshots, October 17; the datum taper fixed the same band in
  review 2026-10-04 but reshaped ground on reveal).

No shader changes: the runtime pack draws all of it as ordinary ground.
"""
import os
import unittest

from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_hill_decal_compaction import PREAMBLE


@unittest.skipIf(os.name == "nt", "Host-only contracts never dispatch Windows or VM tools")
class UnexploredRimTests(unittest.TestCase):
    def test_natural_ground_reaches_into_unexplored_neighbours(self):
        run_cpp(PREAMBLE + r'''
#include <cmath>
#include <cstdio>
// Vertices beyond the tile's own footprint, by side; all lie at the datum
// within a quarter tile. An apron vertex keeps its raised edge's material.
struct Outside {unsigned count=0,apron=0;bool sides[3][3]={};};
Outside outside(TerrainSurfaces const& s,int nc,int nr){
 Outside a;
 for(unsigned layer:{0u,2u}){auto const& mesh=s.meshes[layer];if(!mesh.vertex_stride)continue;
  auto o=render_core::natural_offsets(mesh.vertex_stride);assert(!o.quantized);
  for(std::size_t at=0;at<mesh.vertices.size();at+=mesh.vertex_stride){float w[3],grass=0;
   std::memcpy(w,mesh.vertices.data()+at+o.world,12);std::memcpy(&grass,mesh.vertices.data()+at+o.material,4);
   int dc=w[0]<float(nc)-1e-4f?-1:w[0]>float(nc+1)+1e-4f?1:0,dr=w[1]<float(nr)-1e-4f?-1:w[1]>float(nr+1)+1e-4f?1:0;
   if(!dc && !dr)continue;
   assert(w[0]>float(nc)-.2501f && w[0]<float(nc+1)+.2501f && w[1]>float(nr)-.2501f && w[1]<float(nr+1)+.2501f);
   assert(std::fabs(w[2]*112-2.5f)<1e-4f);
   ++a.count;a.sides[dr+1][dc+1]=true;a.apron+=grass>0;
  }
 }
 return a;
}
int main(){
 auto natural=natural_fixture();std::array<ReliefFields,14> assets;render_core::WorldCoast world;
 render_core::World dimensions{16,16,false,false};
 std::vector<std::uint32_t> data(128,2+(5<<8));
 auto input=hill_input();int nc=(input.tile_x+input.tile_y)/2,nr=(input.tile_x-input.tile_y)/2;
 std::int64_t revision=0;
 auto compile=[&](std::vector<std::uint32_t> const& values){
  world.update(dimensions,values.data(),values.size(),++revision);input.world_revision=revision;
  TerrainCompileScratch scratch;auto s=compile_terrain_surfaces(natural,assets,world,input,scratch,[]{return false;},false);
  assert(s);return outside(*s,nc,nr);};
 auto unexplored=[&](std::vector<int> tiles){auto copy=data;
  for(std::size_t i=0;i+1<tiles.size();i+=2)copy[world.world().index(nc+tiles[i],nr+tiles[i+1])]|=1u<<28;return copy;};
 world.update(dimensions,data.data(),data.size(),++revision);
 for(bool indexed:{true,false}){input.indexed=indexed;
  // Every neighbour explored: nothing beyond the tile.
  assert(compile(data).count==0);
  // Back edges and a back corner: flat rims only there, no apron.
  auto back=compile(unexplored({-1,0,0,1,-1,1}));
  assert(back.count>0 && back.apron==0 && back.sides[1][0] && back.sides[2][1] && back.sides[2][0]);
  assert(!back.sides[1][2] && !back.sides[0][1] && !back.sides[0][0] && !back.sides[0][2] && !back.sides[2][2]);
  // Each raised front edge: its rim and the continued edge surface.
  for(int side=0;side<2;++side){
   auto front=compile(unexplored({side?0:1,side?-1:0}));
   std::printf("side=%d indexed=%d outside=%u apron=%u\n",side,int(indexed),front.count,front.apron);
   assert(front.apron>0 && front.sides[side?0:1][side?1:2]);
   for(int r=0;r<3;++r)for(int c=0;c<3;++c)assert(front.sides[r][c]==(r==(side?0:1) && c==(side?1:2)));
  }
 }
}
''', timeout=120)

    def test_underlay_water_and_rivers_reach_into_unexplored_neighbours(self):
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/source_fidelity/ground_preparation.h"
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <cstdio>
using namespace c3x_renderer;
struct RiverNode {int lattice_x,lattice_y;unsigned degree;bool touches_water;};
int main(){
 std::vector<std::uint32_t> bits(512,2|(2<<8));
 for(int y=0;y<32;++y)for(int x=y&1;x<32;x+=2){unsigned kind=x<12?11u:2u;bits[y*16+x/2]=kind|(kind<<8);}
 int tx=14,ty=12,nc=(tx+ty)/2,nr=(tx-ty)/2;
 render_core::CapturedScene topology;c3x_renderer_frame_v1 capture{};
 capture.world_width_tiles=capture.world_height_tiles=32;capture.world_wrap_x=capture.world_wrap_y=1;
 assert(topology.begin(capture));
 for(int y=0;y<32;++y)for(int x=y&1;x<32;x+=2){c3x_renderer_tile_v1 tile{};tile.tile_x=x;tile.tile_y=y;
  tile.terrain_type=int(bits[y*16+x/2]&255);tile.real_terrain_type=tile.terrain_type;tile.tile_flags=C3X_RENDERER_TILE_RENDER;
  assert(topology.update(tile,tile.terrain_type,tile.real_terrain_type,0,123+unsigned(x+y)));}
 topology.finish();
 std::vector<RiverNode const*> nodes;
 auto source=[](int,unsigned,int,float,float){return 0.f;};auto dune=[](float,float){return 0.f;};
 auto river=[](auto const&,float,float){return 1000.f;};auto relief=[](float,float){return std::array<float,3>{};};
 auto weights=[](float,float){return std::array<float,5>{1,0,0,0,0};};auto shore=[](float,float,float,float){return 0.f;};
 auto ndc=[](float x){return x;};auto key=[&](int x,int y){return topology.key(x,y);};
 c3x_renderer_frame_v1 frame{};frame.tile_width=128;frame.tile_height=64;frame.world_topology_revision=1;
 frame.world_width_tiles=frame.world_height_tiles=32;frame.world_wrap_x=frame.world_wrap_y=1;
 fidelity::GroundCompileInput input;input.tile.tile_x=tx;input.tile.tile_y=ty;
 input.tile.terrain_type=2;input.tile.real_terrain_type=2;
 input.pickup_profile=input.fidelity_profile=true;input.world_ground=true;input.ground=2;
 input.half_w=64;input.half_h=32;input.relief_projection_scale=128/224.f*.82f;
 input.flat_grid=8;input.tile_ground_grid=16;input.shadow_grid=8;
 std::int64_t revision=0;
 auto prepare=[&](std::vector<std::pair<int,int>> hidden){
  auto copy=bits;render_core::WorldCoast coast;coast.update({32,32,true,true},copy.data(),copy.size(),++revision);
  for(auto n:hidden)copy[coast.world().index(nc+n.first,nr+n.second)]|=1u<<28;
  coast.update({32,32,true,true},copy.data(),copy.size(),++revision);
  fidelity::NaturalWorld natural;natural.update_rivers(coast.world(),revision);
  fidelity::SurfaceQueryScratch scratch;std::shared_ptr<std::vector<fidelity::CachedGroundGrid>> grids;
  auto center=coast.sample({nc+.5,nr+.5},[](auto,auto){},[](auto,auto){});
  auto uv=[&](float u,float v,float scale){return fidelity::ground_surface_uv(input.tile,frame,u,v,scale);};
  auto ground=fidelity::prepare_ground(input,frame,natural,scratch,coast,topology,nodes,grids,center,2.f,true,true,
   key,source,dune,river,relief,weights,shore,uv,ndc,ndc,[]{return false;});
  assert(ground);return ground;};
 auto plain=prepare({});
 auto edge=prepare({{1,0}}),corner=prepare({{-1,1}}),both=prepare({{1,0},{-1,1}});
 auto const& u0=plain->meshes[0];
 std::printf("underlay indices plain=%u edge=%u corner=%u both=%u\n",unsigned(u0.index_count),
  unsigned(edge->meshes[0].index_count),unsigned(corner->meshes[0].index_count),unsigned(both->meshes[0].index_count));
 // An edge strip is 8x1 cells, a corner one cell: 48 and 6 more indices.
 assert(edge->meshes[0].index_count==u0.index_count+48 && corner->meshes[0].index_count==u0.index_count+6);
 assert(both->meshes[0].index_count==u0.index_count+54);
 // The bounds reach a quarter tile beyond the tile, toward that neighbour only.
 assert(edge->meshes[0].world_high[0]>u0.world_high[0]+.2f && edge->meshes[0].world_low[1]==u0.world_low[1]);
 // Reading the neighbours is a recorded dependency, so a reveal recompiles.
 assert(edge->world.size()>=plain->world.size() && edge->world!=plain->world);
}
''', timeout=90)


if __name__ == '__main__':
    unittest.main()
