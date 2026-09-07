#pragma once
// Generic source-height fields supplied by the selected fixture profile.
// Ground height is in projected pixels, with water/river constraints in Q0.
#include "scene_adapter.h"
namespace q2_continental {
inline q2::Surface ground_surface;
inline void initialize(char const* path) {
 q2_scene::initialize(path);ground_surface=q2_scene::surface;
 if(ground_surface.width!=map_width || ground_surface.height!=map_height)
  throw std::runtime_error("continental material context dimensions differ");
 auto origin=*ground_surface.at(0,0);
 // Height/normal/shadow queries can reach the source halo's outer edge.
 // Supply the extra interpolation ring from the same authoritative BIQ.
 for(int r=-8;r<ground_surface.rows+8;r++)for(int c=-8;c<ground_surface.columns+8;c++) {
  if(ground_surface.at(c,r))continue;
  int x=(origin.x+c+r)%map_width;if(x<0)x+=map_width;
  int y=origin.y+c-r;
  unsigned cell=y>=0&&y<map_height?map_terrain[y*map_width+x]:0xbb;
  ground_surface.tiles.push_back({c,r,x,y,int(cell&15),int(cell>>4)});
 }
}
inline double sample(unsigned char const* field,double u,double v) {
 double x=u*64-.5,y=v*64-.5;
 int ix=int(std::floor(x)),iy=int(std::floor(y));double tx=x-ix,ty=y-iy;
 auto basis=[](double t,int i) {
  if(i==0)return (1-t)*(1-t)*(1-t)/6;
  if(i==1)return (3*t*t*t-6*t*t+4)/6;
  if(i==2)return (-3*t*t*t+3*t*t+3*t+1)/6;
  return t*t*t/6;
 };
 double value=0;
 for(int j=0;j<4;j++)for(int i=0;i<4;i++)
  value+=field[((iy+j-1)&63)*64+((ix+i-1)&63)]*basis(tx,i)*basis(ty,j);
 return value/255;
}
inline float ground_height(float x,float y) {
 auto s=ground_surface.sample(x,y);
 // Six complete source periods across each map dimension preserve wrap and
 // viewport independence. Period and pixel conversion remain Lab calibration.
 double u=s.raw_x*6/ground_surface.width,v=s.raw_y*6/ground_surface.height;
 return float(sample(grassland,u,v)*14*s.weights[0]+sample(plains,u,v)*10*s.weights[1]);
}
}
