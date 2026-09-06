#pragma once
// Q2 adapter to Q0's opt-in material-only scene hook. No second base plane.
#include "surface.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
namespace q2_scene {
inline q2::Surface surface;
inline void initialize(char const* path) {
 std::ifstream input(path);if(!input)throw std::runtime_error("Q2 adapter terrain missing");
 std::string line;std::getline(input,line);surface=q2::Surface{};
 unsigned count=0,halo=0;int origin_column=0,origin_row=0;
 if(std::sscanf(line.c_str(),"C3X_BIQ_TERRAIN_WINDOW_V2,%d,%d,%u,%d,%d,%d,%d,%u",&surface.columns,&surface.rows,&count,&origin_column,&origin_row,&surface.width,&surface.height,&halo)!=8)
  throw std::runtime_error("Q2 adapter requires BIQ window v2");
 while(std::getline(input,line)) {
  q2::Tile tile{};
  if(std::sscanf(line.c_str(),"%d,%d,%d,%d,%d,%d",&tile.column,&tile.row,&tile.x,&tile.y,&tile.base,&tile.real)!=6)
   throw std::runtime_error("Q2 adapter invalid tile");
  surface.tiles.push_back(tile);
 }
 if(surface.tiles.size()!=count+halo)throw std::runtime_error("Q2 adapter count mismatch");
}
inline void material_uv(float x,float y,float uv_scale,float uv[2]) {
 auto origin=surface.at(0,0);
 double raw_x=origin->x+double(x)+double(y)-1;
 double raw_y=origin->y+double(x)-double(y);
 uv[0]=float(raw_x*std::max(1.,std::round(surface.width*uv_scale*.5))/surface.width);
 uv[1]=float(raw_y*std::max(1.,std::round(surface.height*uv_scale*.5))/surface.height);
}
inline void material_weights(float x,float y,float weights[5]) {
 auto sample=surface.sample(x,y);
 for(int i=0;i<5;i++)weights[i]=float(sample.weights[i]);
}
}
