"""Resident source/instance pose bounds and packed shader inputs."""
import unittest
from pathlib import Path
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.source_fidelity.prepare import function


class ResourceInstanceTests(unittest.TestCase):
    def test_pose_only_skips_legacy_grid_and_keeps_visibility_clock(self):
        source = Path(__file__).with_name('c3x_renderer.cpp').read_text()
        compose = source.split('    bool compose_resource_animations(', 1)[1]
        setup = compose.split('        using c3x_renderer::render_core::RasterRegionAxis;', 1)[1].split(
            '        std::vector<c3x_renderer::FeatureSourceVertex> posed;', 1)[0]
        samples = compose.split('        for (auto const & anchor:resource_anchors) {', 1)[1].split(
            '            if(city_profile) {', 1)[0]
        clock = function(source, 'resource_clock')
        ticks = compose.split('        auto ticks=clock*', 1)[1].split(';', 1)[0]
        run_cpp(r'''
#include "Renderer/native/render_core/resource_instances.h"
#include "Renderer/native/render_core/visibility_coverage.h"
#include "Renderer/native/render_core/raster_grid.h"
#include <cassert>
#include <string>
using namespace c3x_renderer;using namespace c3x_renderer::render_core;
using LONG=int;struct D3D11_RECT {LONG left,top,right,bottom;};
struct Sampler {
 VisibilityCoverage visibility_coverage;bool visibility_pass=true,world_backdrops=true;
 int width=2240,height=1260;
 struct {unsigned pose_samples=0,backdrop_blocks=0;}resource_preparation;
 struct Anchor {unsigned asset=0,seed=0;int tile_x=4,tile_y=4;};
 std::vector<Anchor> resource_anchors=std::vector<Anchor>(1);
 struct Animation {std::string name="whales";AnimationMesh mesh;};
 std::vector<Animation> resource_animations=std::vector<Animation>(1);
 std::vector<std::pair<double,float>> posed;
''' + clock + r'''
 void grid(c3x_renderer_frame_v1 const& frame,bool pose_only){
  using c3x_renderer::render_core::RasterRegionAxis;
''' + setup + r'''
  dirty({0,0,width,height});
  assert(pose_only?dirty_blocks.empty():std::any_of(dirty_blocks.begin(),dirty_blocks.end(),[](auto v){return v!=0;}));
 }
 bool sample(c3x_renderer_frame_v1 const& frame,bool pose_only=true){
  posed.clear();resource_preparation.pose_samples=0;
  if(!visibility_coverage.capture(frame,1,1,1,1))return false;
  auto clock=resource_clock(frame,pose_only?60:15);auto ticks=clock*''' + ticks + r''';
  for(auto const& anchor:resource_anchors){''' + samples + r'''
   AnimationPose pose;assert(sample_animation_pose(animation.mesh,time,true,pose));
   posed.emplace_back(time,pose.positions[0][12]);assert(submersion==.17f);
  }
  return true;
 }
};
int main(){
 Sampler sampler;auto& mesh=sampler.resource_animations[0].mesh;
 mesh.bones=1;mesh.frames=2;mesh.duration=2;mesh.palettes.resize(32);
 for(unsigned f=0;f<2;++f){auto p=mesh.palettes.data()+16*f;p[0]=p[5]=p[10]=p[15]=1;p[12]=float(f*10);}
 c3x_renderer_tile_v1 tile{};tile.tile_x=tile.tile_y=4;tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_BITS;
 c3x_renderer_frame_v1 frame{};frame.tiles=&tile;frame.tile_count=1;
 frame.target_width=2240;frame.target_height=1260;frame.tile_width=128;frame.tile_height=64;
 frame.presentation_frequency=1000;frame.presentation_time_ticks=250;
 sampler.grid(frame,true);assert(!sampler.resource_preparation.backdrop_blocks);
 sampler.grid(frame,false);assert(sampler.resource_preparation.backdrop_blocks>0);
 assert(sampler.sample(frame) && sampler.posed.size()==1);auto first=sampler.posed[0];
 frame.presentation_time_ticks=750;assert(sampler.sample(frame) && sampler.visibility_coverage.captures==1);
 assert(sampler.resource_preparation.pose_samples==1 && sampler.posed[0].first>first.first && sampler.posed[0].second>first.second);
 // Explored fog retains the seed's fixed pose; hide removes it. Revealing the
 // same occurrence samples the absolute live clock rather than restarting it.
 tile.tile_flags&=~C3X_RENDERER_TILE_VISIBLE;assert(sampler.sample(frame));first=sampler.posed[0];
 frame.presentation_time_ticks=1500;assert(sampler.sample(frame) && sampler.posed[0]==first);
 tile.tile_flags&=~C3X_RENDERER_TILE_EXPLORED;assert(sampler.sample(frame) && sampler.posed.empty() && !sampler.resource_preparation.pose_samples);
 tile.tile_flags|=C3X_RENDERER_TILE_VISIBILITY_BITS;assert(sampler.sample(frame) && sampler.posed[0].first>1 && sampler.posed[0].second>first.second);
 sampler.visibility_pass=false;tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN;
 assert(sampler.sample(frame) && sampler.resource_preparation.pose_samples==1);
}
''')

    def test_bounds_cover_interpolation_skinning_and_collapsed_bones(self):
        run_cpp(r'''
#include "Renderer/native/render_core/resource_instances.h"
#include <cassert>
#include <fstream>
#include <iterator>
#include <string>
using namespace c3x_renderer;
using namespace c3x_renderer::render_core;
void verify(AnimationMesh const& mesh){
 ResourceSourceBounds bounds;assert(bounds.prepare(mesh));
 for(unsigned f=0;f<17;++f){
  double time=mesh.duration*(double(f)/16.0);
  AnimationPose pose;assert(sample_animation_pose(mesh,time,true,pose));
  ResourceSourceBounds::Box box;assert(bounds.posed(pose,mesh.bones,box));
  std::vector<FeatureSourceVertex> expected;assert(sample_animation_mesh(mesh,time,true,expected));
  auto bytes=resource_pose_bytes(mesh.bones);assert(bytes%16==0 && bytes<=65536);
  std::vector<float> packed(bytes/4+4,1234);pack_resource_pose(pose,mesh.bones,packed.data());
  for(unsigned i=0;i<16;++i)assert(packed[i]==1234); // Placement belongs to the occurrence.
  for(unsigned i=0;i<4;++i)assert(packed[bytes/4+i]==1234);
  for(unsigned i=0;i<mesh.vertices.size();++i){
   auto const& v=mesh.vertices[i];float p[3]={},n[3]={};
   for(unsigned j=0;j<4;++j){auto m=packed.data()+16+28*v.joints[j];auto normal=m+16;
    for(unsigned a=0;a<3;++a){
     p[a]+=v.weights[j]*(v.source.position[0]*m[a]+v.source.position[1]*m[4+a]+v.source.position[2]*m[8+a]+m[12+a]);
     n[a]+=v.weights[j]*(v.source.normal[0]*normal[a]+v.source.normal[1]*normal[4+a]+v.source.normal[2]*normal[8+a]);
    }
   }
   float length=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
   for(unsigned a=0;a<3;++a){
    assert(p[a]==expected[i].position[a]);
    assert((length>1e-12f?n[a]/length:v.source.normal[a])==expected[i].normal[a]);
    assert(p[a]>=box.low[a] && p[a]<=box.high[a]);
   }
  }
 }
}
int main(){
 assert(!resource_pose_bytes(0) && !resource_pose_bytes(257));
 assert(resource_pose_bytes(256)==28736);
 AnimationMesh m;m.bones=2;m.frames=2;m.duration=2;m.palettes.resize(64);
 for(unsigned frame=0;frame<2;++frame)for(unsigned b=0;b<2;++b){auto p=m.palettes.data()+frame*32+b*16;
  p[0]=b?0:1+float(frame);p[5]=b?0:2;p[10]=b?0:.25f;p[15]=1;p[4]=b?0:.5f;p[12]=float(frame*4+b);
 }
 for(unsigned i=0;i<8;++i){AnimationVertex v={};
  for(unsigned a=0;a<3;++a){v.source.position[a]=(i&(1u<<a))?2.f:-3.f;v.source.normal[a]=float(a+1);}
  v.joints={0,1,0,0};v.weights={.4f,.6f,0,0};m.vertices.push_back(v);
 }
 verify(m);ResourceSourceBounds b;assert(b.prepare(m));
 AnimationPose p;assert(sample_animation_pose(m,0,true,p));p.positions[0][0]=INFINITY;
 ResourceSourceBounds::Box invalid;assert(!b.posed(p,2,invalid));
 // Exercise current locally available normalized clips without requiring them
 // for source-only tests. Bounds must contain actual interpolated animal poses.
 for(auto name:{"horses_0_0.bin","cattle_2_0.bin","fish_0.bin","wheat_0_0.bin"}){
  std::ifstream f(std::string("Renderer/packs/ResourceAnimationRuntime/clips/")+name,std::ios::binary);
  if(!f)continue;std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)),{});
  AnimationMesh asset;assert(decode_animation_mesh(bytes,asset));verify(asset);
 }
}
''')
