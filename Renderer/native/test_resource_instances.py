"""Resident source/instance pose bounds and packed shader inputs."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ResourceInstanceTests(unittest.TestCase):
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
