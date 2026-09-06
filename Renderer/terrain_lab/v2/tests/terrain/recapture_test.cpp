#include "../../systems/terrain/scene_adapter.h"
#include <iostream>
#include <iomanip>
int main() {
 double weight_error=0,uv_error=0;int samples=0;
 for(auto name:{"dry","cold","wet"}) {
  std::string root="Renderer/terrain_lab/v2/fixtures/terrain/real-q2-";
  q2_scene::initialize((root+name+"/terrain.csv").c_str());auto first=q2_scene::surface;
  q2_scene::initialize((root+name+"-holdout/terrain.csv").c_str());auto next=q2_scene::surface;
  auto a=first.at(0,0),b=next.at(0,0);
  double dc=(b->x-a->x+b->y-a->y)*.5,dr=(b->x-a->x-b->y+a->y)*.5;
  for(int i=0;i<=256;i++) {
   double y=i/64.;auto x=first.sample(dc,y),z=next.sample(0,y-dr);
   for(int j=0;j<5;j++)weight_error=std::max(weight_error,std::abs(x.weights[j]-z.weights[j]));
   float u[2],v[2];q2_scene::surface=first;q2_scene::material_uv(float(dc),float(y),.26f,u);
   q2_scene::surface=next;q2_scene::material_uv(0,float(y-dr),.26f,v);
   for(int j=0;j<2;j++)uv_error=std::max(uv_error,double(std::abs(u[j]-v[j])));
   ++samples;
  }
 }
 std::cout<<std::setprecision(14)<<"{\"actual_neighbor_samples\":"<<samples<<",\"max_weight_delta\":"<<weight_error<<",\"max_uv_delta\":"<<uv_error<<"}\n";
 return weight_error>1e-12||uv_error>1e-6;
}
