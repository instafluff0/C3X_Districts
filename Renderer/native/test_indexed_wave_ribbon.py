"""Exact indexed coastal triangles and full production vertex channels."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class IndexedWaveRibbonTests(unittest.TestCase):
    def test_production_channels_triangle_order_wrap_and_capacity(self):
        run_cpp(r'''
#include "Renderer/native/render_core/indexed_wave_ribbon.h"
#include "Renderer/lab/shared/natural/vertex.h"
#include <cassert>
using namespace c3x_renderer::render_core;
using Vertex=c3x_renderer::fidelity::MapVertex;
Vertex project(WavePoint const& point,int c,int r,unsigned tile_width,bool retained){
 float hw=tile_width*.5f,hh=tile_width*.25f,cu=retained?float(c):17.5f,rv=retained?float(r):-.5f;
 float u=float(point.position.x)-cu,v=1-(float(point.position.y)-rv);
 Vertex out={};out.x=(retained?0.f:113.f)+hw+(u-v)*hw;
 out.y=(retained?0.f:-72.f)+(u+v)*hh;out.z=out.y+.08f;
 out.u=point.distance;out.v=point.along;out.panel=1;out.normal_z=1;
 out.shadow_visibility=1;out.ambient_visibility=point.coverage;
 out.base_terrain=.375f;out.real_terrain=.7125f;out.surface_kind=7;
 out.world_x=float(point.position.x);out.world_y=float(point.position.y);out.world_z=2.5f/112;out.world_valid=1;
 out.macro_u=out.world_x*.5f;out.macro_v=out.world_y*.5f;return out;
}
int main(){
 static_assert(sizeof(Vertex)==168,"full production wave channels");
 World world{32,32,true,true};WorldCoast coast;std::vector<std::uint32_t> topology(512);
 unsigned active=0,masked=0;
 for(unsigned shape=0;shape<3;++shape){
  for(int y=0;y<32;++y)for(int x=y&1;x<32;x+=2){
   int boundary=16+(shape?int(3*std::sin(y*.45)):0);
   topology[(y*32+x)/2]=x<boundary?(2|(2<<8)):(11|(11<<8));
  }
  coast.update(world,topology.data(),topology.size(),shape+1);
  for(int y=8;y<24;++y)for(int x=9+(y&1);x<23;x+=2){
   int c=(x+y)/2,r=(x-y)/2;
   for(float scale:{.3f,.7f,1.f})for(int wrap:{0,16}){
    auto expanded=coastal_wave_ribbon(coast,c+wrap,r+wrap,scale);
    auto indexed=indexed_wave_ribbon(expanded);
    if(expanded.empty()){assert(indexed.empty() && indexed.vertices.empty());continue;}
    ++active;assert(expanded.size()==1152 && indexed.indices.size()==1152 && indexed.vertices.size()==231);
    for(unsigned i=0;i<expanded.size();++i){
     auto const& old=expanded[i];auto const& now=indexed.vertices[indexed.indices[i]];
     assert(!std::memcmp(&old.position.x,&now.position.x,sizeof(double)) &&
            !std::memcmp(&old.position.y,&now.position.y,sizeof(double)));
     if(!old.coverage)++masked;
     for(unsigned width:{64u,128u,160u})for(bool retained:{false,true}){
      auto a=project(old,c+wrap,r+wrap,width,retained),b=project(now,c+wrap,r+wrap,width,retained);
      assert(!std::memcmp(&a,&b,sizeof(Vertex))); // Triangle expansion stays bit exact.
     }
    }
    unsigned cursor=0;
    for(unsigned row=0;row<32;++row)for(unsigned column=0;column<6;++column){
     unsigned a=row*7+column,b=a+1,d=a+7,e=d+1;
     for(auto expected:{a,b,e,a,e,d})assert(indexed.indices[cursor++]==expected);
    }
    // Preserve literal geometry if a producer changes one duplicated channel.
    auto inconsistent=expanded;inconsistent[3].coverage=.123456f;
    auto fallback=indexed_wave_ribbon(inconsistent);assert(fallback.vertices.size()==1152);
    for(unsigned i=0;i<1152;++i)assert(fallback.indices[i]==i);
   }
  }
 }
 assert(active && masked); // Includes masked shore joins, not only full beach.
 assert(indexed_wave_ribbon({{{1.,2.},.1f,.2f,.3f}}).indices==std::vector<unsigned>{0});
 constexpr std::size_t old_bytes=1152*(sizeof(Vertex)+sizeof(unsigned));
 constexpr std::size_t new_bytes=231*sizeof(Vertex)+1152*sizeof(unsigned);
 static_assert(old_bytes==198144 && new_bytes==43416,"no quality reduction");
 // The recorded native64 failure admitted 84 cells then rejected cell 85.
 assert(84*old_bytes==16644096 && 85*old_bytes>16u*1024u*1024u);
 assert(85*new_bytes<16u*1024u*1024u);
}
''')


if __name__ == '__main__':
    unittest.main()
