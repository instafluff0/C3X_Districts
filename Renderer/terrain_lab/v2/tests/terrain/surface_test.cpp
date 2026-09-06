#include "../../systems/terrain/surface.h"
#include <cstdio>
int main(){double seam=0,wrap=0,baseline=0,total=0;unsigned samples=0;
 for(int a=0;a<14;a++)for(int b=a;b<14;b++)for(int axis=0;axis<2;axis++)for(int reverse=0;reverse<2;reverse++){
 q2::Surface s;for(int y=-2;y<6;y++)for(int x=-2;x<6;x++){
 int real=(((axis?y:x)>=2)^reverse)?b:a;int base=real<4||real>=11?real:real==4?0:2;s.tiles.push_back({x,y,98+x+y,50+x-y,base,real});}
 for(int i=0;i<=32;i++){
  double t=i/32.;auto v=s.sample(axis?1+t:2,axis?2:1+t);auto adjacent=s.sample(axis?1+t:1+1.,axis?1+1.:1+t);
  q2::Surface alias=s;for(auto& tile:alias.tiles)tile.x+=100;
  auto w=alias.sample(axis?1+t:2,axis?2:1+t);
  auto old=s.sample(axis?1+t:2,axis?2:1+t,true),old_alias=alias.sample(axis?1+t:2,axis?2:1+t,true);
  double sum=0;for(int k=0;k<5;k++){seam=std::max(seam,std::abs(v.weights[k]-adjacent.weights[k]));wrap=std::max(wrap,std::abs(v.weights[k]-w.weights[k]));baseline=std::max(baseline,std::abs(old.weights[k]-old_alias.weights[k]));if(v.weights[k]<-1e-12||v.weights[k]>1+1e-12)return 2;sum+=v.weights[k];}total=std::max(total,std::abs(sum-1));samples++;
 }}
 std::printf("{\"samples\":%u,\"max_shared_weight_delta\":%.12g,\"max_wrap_weight_delta\":%.12g,\"baseline_raw_noise_wrap_delta\":%.12g,\"max_weight_sum_error\":%.12g,\"base_height\":2.5,\"geometry_normal\":[0,0,1]}\n",samples,seam,wrap,baseline,total);
 return seam>1e-6||wrap>1e-6||total>1e-6||baseline<.01;
}
