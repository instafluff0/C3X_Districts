"""Shore rockiness and beach width are continuous, so hills keep no sheer walls.

A ShoreSample's distance is continuous, but rockiness and beach width were
read at the nearest coast point. That point jumps where two coast segments are
equally near, and so did both values. Hill and coastal-rim relief scale by
them, so the terrain stepped there: a sheer wall across a hill behind a rocky
coast (drawn as vertical streaks) and dashes on its low skirt. Seen in game on
a hill at a coast corner. SurfaceQueries now reads both at the point itself
within the reach where any consumer uses them.
"""
import unittest

from Renderer.native.native_cpp_test import run_cpp

WORLD = r'''
#include "Renderer/lab/shared/natural/queries.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
using namespace c3x_renderer;
struct Scene {
 render_core::World dims{48,64,true,true};
 std::vector<unsigned> tiles=std::vector<unsigned>(48*64/2,2|(2<<8));
 render_core::WorldCoast coast;
 unsigned& at(int c,int r){int x=render_core::mod(c+r,48),y=render_core::mod(c-r,64);return tiles[(y*48+x)/2];}
 Scene(){
  // Sea for c>=20 and for r<=9: an L-shaped coast around a land corner at
  // (20,10), with hills along only the u=20 arm, so the two arms' feet differ
  // in rockiness and the nearest foot switches between them inland.
  for(int c=0;c<48;++c)for(int r=-30;r<=30;++r)if(c>=20||r<=9)at(c,r)=12|(12<<8);
  for(int r=10;r<=16;++r)at(19,r)=2|(5<<8);
  coast.update(dims,tiles.data(),tiles.size(),1);
 }
};
'''


class ShoreAttributeContinuityTests(unittest.TestCase):
    def test_rockiness_and_beach_width_have_no_jumps(self):
        run_cpp(WORLD + r'''
int main(){
 Scene scene;
 auto none=[](auto,auto){};
 // Production evaluates each point through its owner tile's queries.
 std::map<std::pair<int,int>,std::unique_ptr<render_core::ExactPointCache<render_core::ShoreSample>>> caches;
 std::map<std::pair<int,int>,std::unique_ptr<fidelity::SurfaceQueries<decltype(none),decltype(none)>>> owners;
 auto shore=[&](double u,double v){
  int c=int(std::floor(u)),r=int(std::floor(v));auto key=std::make_pair(c,r);
  if(!owners.count(key)){
   caches[key]=std::make_unique<render_core::ExactPointCache<render_core::ShoreSample>>();
   owners[key]=std::make_unique<fidelity::SurfaceQueries<decltype(none),decltype(none)>>(scene.coast,*caches[key],c+r,c-r,none,none);
  }
  return owners[key]->shore(float(u),float(v));
 };
 double rocky=0,beach=0;unsigned pairs=0;
 for(double u=17.0;u<20.0;u+=.02)for(double v=10.0;v<13.0;v+=.02){
  auto a=shore(u,v);
  if(a.distance<0 || a.distance>.85)continue;
  for(auto b:{shore(u+.02,v),shore(u,v+.02)}){
   rocky=std::max(rocky,std::abs(a.rocky-b.rocky));
   beach=std::max(beach,std::abs(a.beach_width-b.beach_width));++pairs;
  }
 }
 std::printf("largest step over .02 tile: rocky %.3f, beach width %.4f (%u pairs)\n",rocky,beach,pairs);
 assert(pairs>1000);
 assert(rocky<.06);
 assert(beach<.01);
 // The farthest consumer, the ground's world_valid ramp, reads beach width to
 // ~1.25 tiles: switching back to the coast's values must not show there.
 auto valid=[](render_core::ShoreSample s){return fidelity::coast_ramp((float(s.distance)-float(s.beach_width)-.10f)/.90f);};
 double ramp=0;unsigned far=0;
 for(double u=14.0;u<20.0;u+=.02)for(double v=10.0;v<16.0;v+=.02){
  auto a=shore(u,v);if(a.distance<0 || a.distance>2)continue;far+=a.distance>1.2;
  ramp=std::max({ramp,double(std::abs(valid(a)-valid(shore(u+.02,v)))),double(std::abs(valid(a)-valid(shore(u,v+.02))))});
 }
 std::printf("largest world_valid step over .02 tile %.4f\n",ramp);
 assert(far>1000);
 assert(ramp<.05); // the ramp itself climbs up to .033 per .02 tile
}
''')

    def test_hill_behind_a_rocky_corner_has_no_wall(self):
        run_cpp(WORLD + r'''
int main(){
 Scene scene;
 fidelity::NaturalWorld natural;
 natural.fields.resize(1);auto& f=natural.fields[0];f.width=f.height=4;
 for(unsigned i=0;i<16;++i)f.pixels.push_back(std::uint8_t((i*37+11)%256));
 f.minimum=11/255.f;f.maximum=1;natural.terrain[14]=0;
 auto none=[](auto,auto){};auto flat=[](float,float){return 0.f;};
 std::map<std::pair<int,int>,std::unique_ptr<render_core::ExactPointCache<render_core::ShoreSample>>> caches;
 std::map<std::pair<int,int>,std::unique_ptr<fidelity::SurfaceQueries<decltype(none),decltype(none)>>> owners;
 auto height=[&](double u,double v){
  int c=int(std::floor(u)),r=int(std::floor(v));auto key=std::make_pair(c,r);
  if(!owners.count(key)){
   caches[key]=std::make_unique<render_core::ExactPointCache<render_core::ShoreSample>>();
   owners[key]=std::make_unique<fidelity::SurfaceQueries<decltype(none),decltype(none)>>(scene.coast,*caches[key],c+r,c-r,none,none);
  }
  return owners[key]->height(natural,flat,float(u),float(v));
 };
 // A wall keeps its full height however close two samples are; the intended
 // rocky cliff (about 40 units over .14 tile) rises under 1 unit per .0025.
 double worst=0,tallest=0;
 for(double u=18.0;u<20.0;u+=.0125)for(double v=10.0;v<13.0;v+=.0125){
  double h=height(u,v);tallest=std::max(tallest,h-2.5);
  worst=std::max({worst,std::abs(h-height(u+.0025,v)),std::abs(h-height(u,v+.0025))});
 }
 std::printf("hill rise %.1f units, largest step over .0025 tile %.2f units\n",tallest,worst);
 assert(tallest>10);
 assert(worst<1.5);
}
''')


if __name__ == "__main__":
    unittest.main()
