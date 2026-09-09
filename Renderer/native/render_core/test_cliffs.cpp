#include "cliff_placement.h"
#include "world_topology.h"
#include <cassert>
int main(){using namespace c3x_renderer::render_core;
 World w{32,24,true,true};WorldTopology t;std::vector<uint32_t> v(384,5|(5<<8));
 t.update(w,v.data(),v.size());auto lookup=[&](int c,int r){return t.tile(c,r);};
 ShoreField field(w,lookup);auto contour=[&](int c,int r){return field.cell(c,r);};
 auto p=cliff_placements(w,10,0,lookup,[&](int c,int r){return t.index(c,r);},
 [](double,double){return 50.;},[](double x,double){return x;},[](unsigned){return 1.;},contour);assert(p.empty());
 for(int y=0;y<24;y++)for(int x=y&1;x<32;x+=2)v[(y*32+x)/2]=x<16?(5|(5<<8)):(11|(11<<8));
 t.update(w,v.data(),v.size());field.clear_scratch();
 auto height=[](double,double){return 50.;};auto shore=[](double u,double r){return 16.-u-r;};
 auto index=[&](int c,int r){return t.index(c,r);};auto maximum=[](unsigned){return 1.;};
 std::vector<CliffPlacement> all;
 for(int y=4;y<20;y++)for(int x=14+(y&1);x<=17;x+=2){
  int c=(x+y)/2,r=(x-y)/2;
  auto first=cliff_placements(w,c,r,lookup,index,height,shore,maximum,contour);
  auto again=cliff_placements(w,c,r,lookup,index,height,shore,maximum,contour);
  assert(first.size()==again.size());
  for(size_t i=0;i<first.size();i++){
   assert(first[i].position.x==again[i].position.x && first[i].yaw==again[i].yaw);
   assert(first[i].asset<8);
   if(first[i].asset<4){
    assert(first[i].scale>=.34 && first[i].scale<=.42);
    assert(first[i].z>=2.5/112.);
    assert(first[i].z+first[i].scale>=(50.+2.5)/112.);
   } else assert(first[i].scale>=.229 && first[i].scale<=.311);
   all.push_back(first[i]);
  }
 }
 assert(!all.empty());
 for(size_t i=0;i<all.size();i++)for(size_t j=0;j<i;j++){
  if(all[i].asset>=4 || all[j].asset>=4)continue;
  auto d=all[i].position-all[j].position;assert(dot(d,d)>=.30*.30-1e-7);
 }
 // An authored recipe overrides hard-coded asset choices and scale ranges.
 auto recipe=[](bool small,unsigned){return CliffRecipe{small?7u:2u,1.5,0};};
 bool witnessed=false;
 for(int y=4;y<20;y++)for(int x=14+(y&1);x<=17;x+=2){
  auto p=cliff_placements(w,(x+y)/2,(x-y)/2,lookup,index,height,shore,maximum,contour,recipe);
  for(auto const& a:p){
   witnessed=true;
   assert(a.asset==2 || a.asset==7);
   if(a.asset==2)assert(std::abs(a.scale-.38*1.5)<1e-8);
   else assert(std::abs(a.scale-.27*1.5)<1e-8);
  }
 }
 assert(witnessed);
}
