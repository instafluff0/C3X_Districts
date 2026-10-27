"""Mountains and volcanoes keep their height beside water: their rock meets the
sea as a rocky coast instead of sinking behind the flat beach ramp."""
import unittest
from pathlib import Path
from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


class MountainCoastHeightTests(unittest.TestCase):
    def test_coastal_mountains_and_volcanoes_stand_at_full_height(self):
        # The beach ramp (coast_relief) reaches full height only .58 tiles past
        # the beach; a one-tile island's shore is ~.34 tiles from its centre,
        # so island peaks kept 0% of their height, a peninsula tip 6% and a
        # mountain on a straight coast 59%.
        run_cpp(r'''
#include "Renderer/lab/shared/natural/queries.h"
#include "Renderer/lab/shared/natural/mountain_shape.h"
#include <cassert>
#include <cstdio>
using namespace c3x_renderer;
using namespace c3x_renderer::fidelity;
int main(){
 NaturalWorld natural;
 // A ridged synthetic stamp and crater cone stand in for the pack fields.
 natural.fields.resize(2);
 for(auto& f:natural.fields){f.width=f.height=256;f.pixels.resize(256*256);}
 for(auto* f:{&natural.volcano_height,&natural.volcano_blend,&natural.volcano_channel}){f->width=f->height=256;f->pixels.resize(256*256);}
 for(unsigned y=0;y<256;y++)for(unsigned x=0;x<256;x++){
  float u=(x+.5f)/256-.5f,v=(y+.5f)/256-.5f,r=std::sqrt(u*u+v*v),a=std::atan2(v,u);
  float ridge=1-.35f*(1-std::cos(5*a+r*9))*.5f*std::min(1.f,r/.1f);
  natural.fields[0].pixels[y*256+x]=std::uint8_t(255*std::clamp((1-r/.34f)*ridge,0.f,1.f));
  natural.fields[1].pixels[y*256+x]=std::uint8_t(255*std::clamp((.38f-r)/.08f,0.f,1.f));
  float cone=std::clamp(1-r/.30f,0.f,1.f)-.25f*std::clamp(1-r/.04f,0.f,1.f);
  natural.volcano_height.pixels[y*256+x]=std::uint8_t(255*cone);
  natural.volcano_blend.pixels[y*256+x]=std::uint8_t(255*std::clamp((.34f-r)/.06f,0.f,1.f));
 }
 for(unsigned i=0;i<5;i++){natural.macro[i][0]=0;natural.macro[i][1]=1;natural.macro_axis[i]=.3f*float(i);}
 natural.volcano_channel.maximum=1;natural.volcano_ready=true;natural.measure_volcano_trend();
 render_core::World dims{128,128,true,true};
 std::vector<unsigned> tiles(128*128/2,2|(2<<8));
 auto at=[&](int c,int r)->unsigned&{int x=render_core::mod(c+r,128),y=render_core::mod(c-r,128);return tiles[(y*128+x)/2];};
 auto set=[&](int c,int r,unsigned real){at(c,r)=(real>=11?real:2u)|(real<<8);};
 // Sea from column 30; a peninsula runs out to a mountain at its tip.
 for(int r=-64;r<64;++r)for(int c=30;c<64;++c)set(c,r,r==30 && c<=32?2u:12u);
 struct Case{char const* name;int c,r;unsigned real;bool coastal;};
 Case cases[]={
  {"inland mountain",10,0,6,false},{"inland volcano",10,6,10,false},
  {"coast mountain",29,-10,6,true},{"coast volcano",29,-4,10,true},
  {"island mountain",45,-10,6,true},{"island volcano",45,0,10,true},
  {"peninsula mountain",33,30,6,true}};
 for(auto& k:cases)set(k.c,k.r,k.real);
 render_core::WorldCoast coast;coast.update(dims,tiles.data(),tiles.size(),1);
 auto none=[](auto,auto){};
 unsigned raised=0,inland=0;float translucent=0;
 for(auto& k:cases){
  render_core::ExactPointCache<render_core::ShoreSample> scratch;
  SurfaceQueries query(coast,scratch,k.c+k.r,k.c-k.r,none,none,true,nullptr);
  MountainShape shape(natural,k.c,k.r,[&](int c,int r){return query.natural_tile(c,r);},[](int,int){return 0u;});
  float raw=0,drawn=0;
  for(float y=k.r-.5f;y<=k.r+1.5f;y+=1/64.f)for(float x=k.c-.5f;x<=k.c+1.5f;x+=1/64.f){
   auto sample=shape.sample(natural,x,y);
   auto s=query.shore(x,y);
   float d=float(s.distance),bw=float(s.beach_width);
   float rise=sample.displacement*mountain_coast_relief(d,bw,sample.stone);
   // The relief mesh's coverage (relief_mesh_body.h) over the flat beach.
   float coverage=mountain_coast_coverage(coast_coverage(d,bw),d,sample.stone);
   // Monotone: the rocky coast only ever adds height and coverage.
   assert(rise>=sample.displacement*coast_relief(d,bw)-1e-4f && coverage>=coast_coverage(d,bw)-1e-6f);
   // Raised rock is opaque where the stamp stands; its low foot keeps the
   // beach, raised at most a few units under the fade.
   if(sample.stone>=32.f && rise>0)assert(coverage==1.f);
   translucent=std::max(translucent,rise*(1-coverage));
   if(rise>1)++raised;
   // Away from the water nothing changes.
   if(coast_relief(d,bw)==1.f){assert(rise==sample.displacement);++inland;}
   if(x>=k.c && x<=k.c+1 && y>=k.r && y<=k.r+1){raw=std::max(raw,sample.displacement);drawn=std::max(drawn,rise);}
  }
  std::printf("%-20s raw %6.1f drawn %6.1f (%3.0f%%)\n",k.name,raw,drawn,100*drawn/raw);
  assert(raw>.7f*(k.real==10?MountainShape::volcano_height:MountainShape::height));
  assert(drawn>=(k.coastal?.85f:1.f)*raw);
 }
 assert(raised>0 && inland>0 && translucent<6.f);
 // A stone-free (wonder or flat) sample keeps the beach rule exactly.
 for(float d=-.2f;d<1.f;d+=.01f){
  assert(mountain_coast_relief(d,.15f,0)==coast_relief(d,.15f));
  assert(mountain_coast_coverage(.3f,d,0)==.3f);
 }
}
''')

    def test_mountain_consumers_share_the_coast_rule(self):
        # Routes, resources, the volcano plume and wonder effects sit on the
        # rock the mesh draws: every mountain-displacement consumer scales by
        # the same mountain coast rule, never the plain beach ramp.
        consumers = {
            "Renderer/lab/shared/natural/relief_mesh_body.h": 2,
            "Renderer/lab/shared/natural/test_mesh.cpp": 2,
            "Renderer/native/object_preparation.h": 1,
            "Renderer/native/c3x_renderer.cpp": 1,
            "Renderer/native/source_fidelity/geometry.h": 2,
        }
        for path, count in consumers.items():
            text = (ROOT / path).read_text()
            self.assertGreaterEqual(text.count("mountain_coast_relief("), count, path)
            for line in text.splitlines():
                if "displacement" in line and "coast_relief(" in line:
                    self.assertIn("mountain_coast_relief(", line, f"{path}: {line.strip()}")


if __name__ == "__main__":
    unittest.main()
