"""Durable world membership and camera-independent preparation contracts."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT=Path(__file__).resolve().parents[2]


class WorldContentTests(unittest.TestCase):
    def test_spatial_membership_survives_views_and_releases_with_owner(self):
        run_cpp(r'''
#include "Renderer/native/render_core/world_pass_index.h"
#include <cassert>
#include <limits>
using c3x_renderer::render_core::WorldPassIndex;
int main(){
 WorldPassIndex index;std::vector<WorldPassIndex::Key> out;
 // One object crosses cells and a wrapped occurrence uses another raw anchor.
 assert(index.add(7,100,-.25,-1.25,2.5,.5));
 assert(index.add(8,200,99.75,-1.25,102.5,.5));
 auto bytes=index.bytes();
 for(int zoom:{32,64,128,160,192,224})for(int pan:{-1400,0,3000}){
  double left=pan+zoom*.5,right=pan+zoom;
  assert(index.query((left-pan)/zoom,-1,(right-pan)/zoom,0,out));
  assert(out.size()==1 && out[0]==100);assert(index.bytes()==bytes);
 }
 // Selection is conservative on cell edges, and never misses an intersection.
 for(int i=0;i<200;++i)assert(index.add(10+i,1000+i,i*.375,-i*.125,i*.375+.7,-i*.125+.8));
 for(int i=0;i<200;++i){
  assert(index.query(i*.375+.2,-i*.125+.2,i*.375+.4,-i*.125+.4,out));
  assert(std::find(out.begin(),out.end(),1000+i)!=out.end());
 }
 index.erase(7);assert(!index.contains(100));assert(index.contains(200));
 assert(index.add(300,100,10,20,11,21)); // Address recycled by a new immutable owner.
 assert(index.query(-1,-2,3,1,out));assert(std::find(out.begin(),out.end(),100)==out.end());
 index.erase(7);assert(index.contains(100));index.erase(300);assert(!index.contains(100));
 assert(!index.add(400,300,0,0,1e12,1e12));
 assert(!index.add(400,300,0,0,std::numeric_limits<double>::infinity(),1));
 for(unsigned i=0;i<100000;++i){if(!index.add(10000+i,10000+i,i*3.,0,i*3.+.5,1))break;}
 assert(index.bytes()<=index.budget);
 index.clear();assert(index.bytes()==0);assert(index.query(-100,-100,100,100,out) && out.empty());
}
''')

    def test_preparation_key_preserves_content_and_detail_across_camera_changes(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        method='c3x_renderer::fidelity::TerrainCompileInput terrain_compile_input('+source.split(
            'c3x_renderer::fidelity::TerrainCompileInput terrain_compile_input(',1)[1].split('\n    bool terrain_result_valid',1)[0]
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <array>
#include <cstdint>
#include <cassert>
namespace c3x_renderer {namespace fidelity {
struct Detail {unsigned value=5;unsigned identity()const{return value;}};
struct TerrainCompileInput {
 using Key=std::array<std::uint64_t,12>;Key key{};
 int tile_x,tile_y,real_terrain_type,ground,tile_width,tile_height,target_height;
 long long world_revision;Detail detail;
 bool river_ready,skip_flat_shore,separate_relief,indexed,retain_height;
};
}}
struct State {
 bool city_profile=true,retained_world=true,share_world_meshes=true,river_assets_ready=true;
 int content_view_height=900;unsigned content_revision=13;
 c3x_renderer::fidelity::Detail patch_detail;
'''+method+r'''
};
int main(){
 State state;c3x_renderer_tile_v1 tile={};tile.tile_x=2;tile.tile_y=4;tile.real_terrain_type=6;
 c3x_renderer_frame_v1 frame={};frame.tile_width=128;frame.tile_height=64;
 frame.world_width_tiles=frame.world_height_tiles=100;
 auto compile=[&]{return state.terrain_compile_input(tile,frame,2,true,true,true,true,state.share_world_meshes);};
 auto first=compile();assert(first.tile_width==128 && first.target_height==128);
 for(int zoom:{64,128,160,192,224}){
  frame.tile_width=zoom;frame.tile_height=zoom/2;state.content_view_height=1200;
  tile.anchor_x+=37;frame.world_topology_revision++;
  auto next=compile();assert(first.key==next.key);
  // Local proof validation, not this global revision, decides reuse.
  assert(next.world_revision==frame.world_topology_revision);
 }
 state.patch_detail.value++;assert(first.key!=compile().key);state.patch_detail.value--;
 state.content_revision++;assert(first.key!=compile().key);state.content_revision--;
 tile.real_terrain_type=7;assert(first.key!=compile().key);tile.real_terrain_type=6;
 state.share_world_meshes=false;assert(compile().tile_width==224 && first.key!=compile().key);
}
''')

    def test_pass_queries_require_current_occurrence_and_preserve_order(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        method='bool query_region_inputs('+source.split('bool query_region_inputs(',1)[1].split(
            '\n    void prepare_region_contributors',1)[0]
        run_cpp(r'''
#include "Renderer/native/render_core/world_pass_index.h"
#include "Renderer/native/render_core/region_contributor_index.h"
#include <cassert>
struct State {
 c3x_renderer::render_core::WorldPassIndex world_pass_index;
 c3x_renderer::render_core::RegionContributorIndex region_contributors;
 std::unordered_map<std::uintptr_t,std::vector<std::pair<unsigned,unsigned>>> world_pass_occurrences;
 bool world_pass_affine=true;int shadow_tile_width=128;
 double world_pass_x=0,world_pass_y=0;float world_pass_reflection=0;
'''+method+r'''
};
int main(){
 State s;using Item=std::pair<unsigned,unsigned>;std::vector<Item> out;
 assert(s.world_pass_index.add(1,100,1,1,2,2));
 assert(s.world_pass_index.add(2,200,1,1,2,2)); // retained but not currently observed
 s.world_pass_occurrences[100]={{5,3},{2,1}};
 s.region_contributors.add(0,{0,0},128,128,256,256);s.region_contributors.ready=true;
 assert(s.query_region_inputs(0,128,128,128,out));
 assert((out==std::vector<Item>{{0,0},{2,1},{5,3}}));
 auto bytes=s.world_pass_index.bytes();
 s.region_contributors.clear();s.region_contributors.ready=true;
 for(int width:{64,128,160,192,224}){
  s.shadow_tile_width=width;s.world_pass_x=300;s.world_pass_y=-100;
  assert(s.query_region_inputs(0,300+width,-100+width,width,out));
  assert((out==std::vector<Item>{{2,1},{5,3}}));assert(s.world_pass_index.bytes()==bytes);
 }
 s.shadow_tile_width=128;s.world_pass_x=s.world_pass_y=0;s.world_pass_reflection=256;
 assert(s.query_region_inputs(1,128,384,128,out));assert(out.size()==2);
 s.world_pass_occurrences.clear();assert(s.query_region_inputs(1,128,384,128,out) && out.empty());
 s.world_pass_affine=false;s.region_contributors.ready=false;assert(!s.query_region_inputs(0,0,0,128,out));
}
''')
