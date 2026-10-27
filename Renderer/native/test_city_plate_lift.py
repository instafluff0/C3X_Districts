"""City plates stay over hill terrain: the ground mesh no longer streaks through them.

A city's paving (and any source ground part) is draped on the natural surface
and drawn after the terrain with read-only depth. Both meshes are flat between
their own points: the terrain on its tile grid (24 per tile around hills in
the resident world), the plate on a .02-tile grid. On a curved hill they cross;
at .005 units of lift the terrain cut dark streaks through the plates of cities
founded on hills. Measured on the real natural and city packs, the terrain rose
over ~40% of a hill city's plate, by up to ~1 unit for a lone hill and ~2-4
where hills meet. The plate now lies ground_lift units up, still under draped
farm fields (~2.9 units) and routes so they keep drawing over its fringe.
"""
import unittest

from Renderer.native.native_cpp_test import run_cpp

PROGRAM = r'''
#include "Renderer/native/city_fidelity/compiler.h"
#include "Renderer/lab/shared/natural/queries.h"
#include "Renderer/lab/shared/natural/ground.h"
#include <cassert>
#include <cstdio>
using namespace c3x_renderer;
int main(){
 // The farm field lift (object_compiler.h: .016 tile at 150 units per tile over
 // .82 relief) bounds the plate from above.
 assert(city_fidelity::ground_lift>1.f && city_fidelity::ground_lift<.016f*150/.82f);
 fidelity::NaturalWorld natural;
 natural.fields.resize(1);auto& f=natural.fields[0];f.width=f.height=4;
 for(unsigned i=0;i<16;++i)f.pixels.push_back(std::uint8_t((i*37+11)%256));
 f.minimum=11/255.f;f.maximum=1;natural.terrain[14]=0;
 // A plate on a .02-tile grid reaching .46 tile from the centre, as authored.
 city_fidelity::Library library;library.materials.resize(1);
 library.materials[0].channels=4;library.materials[0].ground=1;
 city_fidelity::Composition city;city.anchor_layout=1;
 city.paving.material=0;city.paving.period[0]=city.paving.period[1]=.45f;
 constexpr int cells=46;
 for(int j=0;j<=cells;++j)for(int i=0;i<=cells;++i)
  city.paving.vertices.push_back({-.46f+.02f*i,-.46f+.02f*j,1});
 for(int j=0;j<cells;++j)for(int i=0;i<cells;++i){
  unsigned a=unsigned(j*(cells+1)+i),b=a+1,c=b+cells+1,d=a+cells+1;
  for(unsigned k:{a,b,c,a,c,d})city.paving.indices.push_back(k);
 }
 library.compositions.push_back(city);
 auto none=[](auto,auto){};auto flat=[](float,float){return 0.f;};
 int const nc=16,nr=12;
 // A lone hill city, and one whose plate reaches into the valley between its
 // hill and the next.
 for(auto const& hills:{std::vector<std::array<int,2>>{{16,12}},{{16,12},{17,12}}}){
  render_core::World dims{48,64,true,true};
  std::vector<unsigned> tiles(48*64/2,2|(2<<8));
  auto at=[&](int c,int r)->unsigned&{int x=render_core::mod(c+r,48),y=render_core::mod(c-r,64);return tiles[(y*48+x)/2];};
  for(auto h:hills)at(h[0],h[1])=(at(h[0],h[1])&~(255u<<8))|(5u<<8);
  render_core::WorldCoast coast;coast.update(dims,tiles.data(),tiles.size(),1);
  render_core::ExactPointCache<render_core::ShoreSample> scratch;
  fidelity::SurfaceQueries query(coast,scratch,nc+nr,nc-nr,none,none);
  auto height=[&](float x,float y){return query.height(natural,flat,x,y);};
  // The ground compiler's grid: (a,b,c) and (a,c,d) per cell, v up the tile.
  constexpr int grid=24;
  auto terrain=[&](float x,float y){
   int c=int(std::floor(x)),r=int(std::floor(y));
   float gu=(x-float(c))*grid,gv=(1.f-(y-float(r)))*grid;
   int ix=std::min(grid-1,int(gu)),iy=std::min(grid-1,int(gv));float fx=gu-float(ix),fy=gv-float(iy);
   auto h=[&](int i,int j){return height(float(c)+float(i)/grid,float(r)+1.f-float(j)/grid);};
   float a=h(ix,iy),b=h(ix+1,iy),cc=h(ix+1,iy+1),d=h(ix,iy+1);
   return fx>=fy?a+(b-a)*fx+(cc-b)*fy:a+(cc-d)*fx+(d-a)*fy;
  };
  fidelity::GroundProjection projection{nc,nr,64,32,128.f/224.f*.82f,768};
  city_fidelity::Surfaces out;
  assert(city_fidelity::compile(library,library.compositions[0],nc,nr,height,projection,out,
    city_fidelity::ContinueCompilation{},true));
  auto const& plate=out.chunks.front();
  assert(plate.terrain_conforming && plate.indices.size()==std::size_t(cells)*cells*6);
  for(auto const& v:plate.vertices)
   assert(std::abs(v.world_z*112-height(v.world_x,v.world_y)-city_fidelity::ground_lift)<1e-3f);
  float through=-1e9f,before=-1e9f;unsigned raised=0;
  for(std::size_t t=0;t<plate.indices.size();t+=3){
   auto const& A=plate.vertices[plate.indices[t]];auto const& B=plate.vertices[plate.indices[t+1]];
   auto const& C=plate.vertices[plate.indices[t+2]];
   for(int i=0;i<=4;++i)for(int j=0;j<=4-i;++j){
    float u=i/4.f,v=j/4.f,w=1-u-v;
    float x=w*A.world_x+u*B.world_x+v*C.world_x,y=w*A.world_y+u*B.world_y+v*C.world_y;
    float drawn=(w*A.world_z+u*B.world_z+v*C.world_z)*112,ground=terrain(x,y);
    raised+=ground>12.f;
    through=std::max(through,ground-drawn);
    before=std::max(before,ground-(drawn-city_fidelity::ground_lift+.005f));
   }
  }
  std::printf("%zu hill(s): terrain rose %.3f units through the plate at the old .005 lift, %.3f now\n",
   hills.size(),before,through);
  assert(raised>1000 && before>.25f);
  assert(through<0);
 }
}
'''


class CityPlateLiftTests(unittest.TestCase):
    def test_hill_terrain_stays_under_city_plates(self):
        run_cpp(PROGRAM, timeout=120)


if __name__ == "__main__":
    unittest.main()
