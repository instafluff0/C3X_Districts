"""Hills step down to rivers, and river distances stay exact past the reach they
shape: no hill ground over the drawn water, no relief cliffs on cell lines."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class HillRiverBankTests(unittest.TestCase):
    def test_hills_leave_the_river_surface_and_invalidate(self):
        # Civ III rivers run on tile edges but hill bodies reach across them.
        # Without the bank a hill stood ~34 units over this water.
        run_cpp(r'''
#include "Renderer/lab/shared/natural/queries.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 fidelity::NaturalWorld natural;
 natural.fields.resize(1);auto& f=natural.fields[0];f.width=f.height=4;
 for(unsigned i=0;i<16;++i)f.pixels.push_back(std::uint8_t((i*37+11)%256));
 f.minimum=11/255.f;f.maximum=1;natural.terrain[14]=0;
 render_core::World dims{48,64,true,true};
 // A straight river along node row 12 between two hills facing each other
 // across it (a valley), and a lone hill beside it.
 std::vector<unsigned> tiles(48*64/2,2|(2<<8));
 auto at=[&](int c,int r)->unsigned&{int x=render_core::mod(c+r,48),y=render_core::mod(c-r,64);return tiles[(y*48+x)/2];};
 for(int c=10;c<24;++c){at(c,12)|=32u<<16;at(c,11)|=2u<<16;}
 for(auto h:{std::array<int,2>{16,12},{16,11},{20,12}})at(h[0],h[1])=(at(h[0],h[1])&~(255u<<8))|(5u<<8);
 render_core::WorldCoast coast;coast.update(dims,tiles.data(),tiles.size(),1);
 natural.update_rivers(coast.world(),1);
 auto none=[](auto,auto){};auto flat=[](float,float){return 0.f;};
 auto authored=[&](float x,float y){
  return natural.height(x,y,[&](int c,int r){auto t=coast.world().tile(c,r);
   return fidelity::Tile{render_core::mod(c+r,48),render_core::mod(c-r,64),c,r,t.real};});};
 unsigned water=0,buried=0,kept=0;float picked_x=0,picked_y=0;
 fidelity::NaturalWorld::CellInputs inputs;
 {
  fidelity::NaturalWorld::DependencyScope scope(natural,&inputs);
  for(auto owner:{std::array<int,2>{16,12},{16,11},{20,12}}){
   render_core::ExactPointCache<render_core::ShoreSample> scratch;
   fidelity::SurfaceQueries query(coast,scratch,owner[0]+owner[1],owner[0]-owner[1],none,none,true,&natural);
   for(float y=owner[1]-.75f;y<=owner[1]+1.75f;y+=1/32.f)for(float x=owner[0]-.75f;x<=owner[0]+1.75f;x+=1/32.f){
    float d=float(natural.river_sample({x,y}).distance);
    float rise=query.height(natural,flat,x,y)-2.5f,source=authored(x,y)-2.5f;
    // Banks only lower the authored hill; they never raise or invert it.
    assert(rise>=-1e-4f && rise<=source+1e-4f);
    // The drawn water ends 7.4 source pixels from the centerline.
    if(d<7.4f){assert(rise<1e-3f);++water;if(source>2){++buried;picked_x=x;picked_y=y;}}
    // A tile from the river the hill keeps its authored shape exactly.
    if(d>64 && source>0){assert(rise==source);++kept;}
   }
  }
 }
 assert(water>0 && buried>0 && kept>0);
 // The river is a dependency: removing it restores the hill there.
 fidelity::NaturalWorld::CellProof proof(inputs.begin(),inputs.end());assert(natural.valid(proof));
 for(auto&t:tiles)t&=~(255u<<16);
 coast.update(dims,tiles.data(),tiles.size(),2);natural.update_rivers(coast.world(),2);
 assert(!natural.valid(proof));
 render_core::ExactPointCache<render_core::ShoreSample> scratch;
 int c=int(std::floor(picked_x)),r=int(std::floor(picked_y));
 fidelity::SurfaceQueries changed(coast,scratch,c+r,c-r,none,none,true,&natural);
 assert(changed.height(natural,flat,picked_x,picked_y)==authored(picked_x,picked_y));
}
''')

    def test_river_distance_is_exact_and_continuous_to_its_reach(self):
        # A sample reads only its own cell's bucket. With segments filed only
        # .65 tiles out, distances past ~37 screen pixels jumped at cell lines
        # (to 1000 when the segment was missing) and low relief, which ramps
        # to 52 pixels, cliffed there; hill banks ramp to 53.
        run_cpp(r'''
#include "Renderer/lab/shared/natural/queries.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 fidelity::NaturalWorld natural;
 for(auto& f:natural.low_relief.fields){f.width=f.height=16;f.amplitude=28;f.span=96;
  for(unsigned y=0;y<16;++y)for(unsigned x=0;x<16;++x)f.pixels.push_back((x*11+y*7)%256);}
 render_core::World dims{48,64,true,true};
 std::vector<unsigned> tiles(48*64/2,2|(2<<8));
 auto at=[&](int c,int r)->unsigned&{int x=render_core::mod(c+r,48),y=render_core::mod(c-r,64);return tiles[(y*48+x)/2];};
 // Coast from column 17; the meander's mouth there adds delta distributaries.
 for(int r=-20;r<40;++r)for(int c=17;c<30;++c)at(c,r)=11|(11<<8);
 int c=8,r=4;
 for(char move:std::string("ccrrcrrrccrcrrccrrc")){
  if(move=='c'){at(c,r)|=32u<<16;at(c,r-1)|=2u<<16;++c;}
  else{at(c,r)|=128u<<16;at(c-1,r)|=8u<<16;++r;}
 }
 render_core::WorldCoast coast;coast.update(dims,tiles.data(),tiles.size(),1);
 natural.update_rivers(coast.world(),1);
 constexpr double reach=54;
 auto distance=[&](double x,double y){return std::min(reach,natural.river_sample({x,y}).distance);};
 unsigned pairs=0,exact=0,affected=0,narrowed=0;
 auto none=[](auto,auto){};
 for(int cell_r=0;cell_r<=24;++cell_r)for(int cell_c=4;cell_c<=24;++cell_c){
  // Every segment of the page this cell reads, delta reaches included.
  auto const& field=natural.river_page(cell_c+.5,cell_r+.5);
  std::vector<river::Segment> segments;
  for(auto const& bucket:field.buckets)segments.insert(segments.end(),bucket.second.begin(),bucket.second.end());
  double pool=1e9;
  for(auto const& t:field.terminals)if(!t.mouth)pool=std::min(pool,hydro::length(t.p-hydro::P{cell_c+.5,cell_r+.5}));
  // River terrain detail keeps exactly the cells within .65 tiles.
  bool near_rule=field.terminal_buckets.count({cell_c,cell_r})>0;
  for(auto const& s:segments){
   auto a=river::from_screen(s.a),b=river::from_screen(s.b);
   near_rule=near_rule || (std::min(a.x,b.x)-.65<cell_c+1 && std::max(a.x,b.x)+.65>=cell_c &&
                           std::min(a.y,b.y)-.65<cell_r+1 && std::max(a.y,b.y)+.65>=cell_r);
  }
  assert(natural.river_affects(cell_c,cell_r)==near_rule);affected+=near_rule;
  render_core::ExactPointCache<render_core::ShoreSample> scratch;
  fidelity::SurfaceQueries query(coast,scratch,cell_c+cell_r,cell_c-cell_r,none,none,true,&natural);
  for(int k=0;k<16;++k){
   double t=k/16.+1/32.,x=cell_c+t,y=cell_r+t;
   if(pool>1.6){
    double brute=1e9,plain=1e9;
    for(auto const& s:segments){
     double d=river::distance(river::screen({x,cell_r+.5}),s.a,s.b);
     plain=std::min(plain,d);brute=std::min(brute,d+s.narrow);
    }
    if(brute<reach){assert(std::abs(natural.river_sample({x,cell_r+.5}).distance-brute)<1e-6);++exact;narrowed+=brute!=plain;}
   }
   // Either side of the cell's lower-left edges: the distance and the low
   // relief it shapes are continuous.
   assert(std::abs(distance(cell_c-1e-5,y)-distance(cell_c+1e-5,y))<.01);
   assert(std::abs(distance(x,cell_r-1e-5)-distance(x,cell_r+1e-5))<.01);
   assert(std::abs(query.low_height(natural,cell_c-1e-5f,float(y))-query.low_height(natural,cell_c+1e-5f,float(y)))<.01f);
   pairs+=2;
  }
 }
 assert(exact>0 && affected>0 && narrowed>0);
}
''')


if __name__ == "__main__":
    unittest.main()
