#include "../../fixtures/beauty/source-continental-r1/ground_fields.h"
#include "../../fixtures/beauty/source-continental-r1/map_context.h"
#include "../../systems/terrain/continental_ground.h"
#include <iostream>
#include <iomanip>
#include <cassert>
int main(int argc,char**argv) {
 assert(argc==3);
 q2_continental::initialize(argv[1]);auto first=q2_continental::ground_surface;
 q2_continental::initialize(argv[2]);auto next=q2_continental::ground_surface;
 auto a=*first.at(0,0),b=*next.at(0,0);
 int dx=b.x-a.x;
 if(dx>first.width/2)dx-=first.width;if(dx< -first.width/2)dx+=first.width;
 float dc=(dx+b.y-a.y)*.5f,dr=(dx-b.y+a.y)*.5f;
 double error=0;unsigned count=0;
 for(int j=-12;j<=31;j++)for(int i=-12;i<=31;i++) {
  float x=i*.5f,y=j*.5f;
  // Compare the shared domain of actual Q0 height queries: each capture owns
  // a six-cell halo. Beyond it Q0 returns the water datum without calling us.
  if(x-dc< -6 || y-dr< -6 || x-dc>=16 || y-dr>=16)continue;
  q2_continental::ground_surface=first;float h=q2_continental::ground_height(x,y);
  q2_continental::ground_surface=next;float k=q2_continental::ground_height(x-dc,y-dr);
  assert(h>=0&&h<=14&&k>=0&&k<=14);
  error=std::max(error,double(std::abs(h-k)));count++;
 }
 std::cout<<std::setprecision(12)<<"{\"samples\":"<<count<<",\"maximum_crop_or_wrap_delta_px\":"<<error<<"}\n";
 return error>1e-5;
}
