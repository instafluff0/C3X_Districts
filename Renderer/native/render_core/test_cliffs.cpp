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
    assert(first[i].scale>=.40 && first[i].scale<=.50);
    assert(first[i].z>=2.5/112.);
    assert(first[i].z+first[i].scale>=(50.+2.5)/112.);
   } else assert((first[i].scale>=.229 && first[i].scale<=.311) ||
                 (first[i].scale>=.339 && first[i].scale<=.461));
   all.push_back(first[i]);
  }
 }
 assert(!all.empty());
 for(size_t i=0;i<all.size();i++)for(size_t j=0;j<i;j++){
  if(all[i].asset>=4 || all[j].asset>=4)continue;
  auto d=all[i].position-all[j].position;assert(dot(d,d)>=.20*.20-1e-7);
 }
 // An authored recipe overrides hard-coded asset choices and scale ranges.
 auto recipe=[](bool small,unsigned){return CliffRecipe{small?7u:2u,1.5,0};};
 bool witnessed=false;
 for(int y=4;y<20;y++)for(int x=14+(y&1);x<=17;x+=2){
  auto p=cliff_placements(w,(x+y)/2,(x-y)/2,lookup,index,height,shore,maximum,contour,recipe);
  for(auto const& a:p){
   witnessed=true;
   assert(a.asset==2 || a.asset==7);
   if(a.asset==2)assert(std::abs(a.scale-(50./112.)*1.5)<1e-8);
   else assert(std::abs(a.scale-.27*1.5)<1e-8 || std::abs(a.scale-.40*1.5)<1e-8);
  }
 }
 assert(witnessed);
 // Upper details attach to their own position on a sloped terrain surface.
 auto slope=[](double x,double y){return 50.+x*2+y*3;};
 unsigned attached=0;
 for(int y=4;y<20;y++)for(int x=14+(y&1);x<=17;x+=2){
  auto values=cliff_placements(w,(x+y)/2,(x-y)/2,lookup,index,slope,shore,maximum,contour);
  for(auto const& a:values)if(a.asset>=4 && a.z>.1){
   assert(std::abs(a.z-(slope(a.position.x,a.position.y)+2.5)/112.)<1e-8);attached++;
  }
 }
 assert(attached>0);
 // A sloped source face stays perpendicular to its transformed normal at
 // every orientation. A reflected normal Y with unreflected positions fails.
 for(double yaw:{0.,.7,-1.3,3.14})for(float vertical:{1.f,1.6332753f})for(double scale:{.27,.52}) {
  CliffPlacement instance;instance.position={2,-3};instance.z=.2;instance.yaw=yaw;instance.scale=scale;
  CliffTransform transform(instance,vertical);
  std::array<float,3> a{0,0,0},b{1,0,.7f},c{0,1,.4f},n{-.7f,-.4f,1};
  auto pa=transform.position(a),pb=transform.position(b),pc=transform.position(c),pn=transform.normal(n);
  std::array<float,3> u,v,cross;
  for(int i=0;i<3;i++){u[i]=pb[i]-pa[i];v[i]=pc[i]-pa[i];}
  cross={u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};
  double dot=0,nn=0,cc=0;
  for(int i=0;i<3;i++){dot+=cross[i]*pn[i];nn+=pn[i]*pn[i];cc+=cross[i]*cross[i];}
  assert(dot/std::sqrt(nn*cc)>.99999);
 }

}
