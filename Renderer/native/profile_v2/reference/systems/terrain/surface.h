#pragma once
// Q2 continuous base field v1. Relief and water composition remain external.
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace q2 {
struct Tile { int column,row,x,y,base,real; };
struct Sample { double height=2.5; std::array<double,3> normal{0,0,1}; std::array<double,5> weights{}; double raw_x,raw_y; };
inline double smooth(double x) { x=std::clamp(x,0.,1.); return x*x*x*(x*(x*6-15)+10); }
inline double periodic(double x,double period) { return x-std::floor(x/period)*period; }
inline double wave(double x,double y,double width,double height,int nx,int ny,double phase) {
 return std::sin(6.283185307179586*(nx*periodic(x,width)/width+ny*periodic(y,height)/height)+phase);
}
struct Surface {
 int columns=4,rows=4,width=100,height=100; std::vector<Tile> tiles;
 Tile const* at(int c,int r) const { for(auto& t:tiles)if(t.column==c&&t.row==r)return &t;return nullptr; }
 int material(Tile const& t) const { if(t.real==9)return 3;if(t.real==4)return 1;return t.base==0?2:t.base==1?1:t.base==3?4:0; }
 std::array<double,5> center(int c,int r) const {
  std::array<double,5> w{};auto t=at(c,r);
  if(!t)throw std::runtime_error("Q2 missing adjacency halo");
  if(t->base<11){w[material(*t)]=1;return w;}
  double total=0;
  for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {auto n=at(c+x,r+y);if(n&&n->base<11){double k=x&&y?.7:1.;w[material(*n)]+=k;total+=k;}}
  if(total==0)w[0]=1;else for(auto& k:w)k/=total;
  return w;
 }
 Sample sample(double x,double y,bool baseline=false) const {
  auto origin=at(0,0);if(!origin)throw std::runtime_error("missing origin");Sample s;
  s.raw_x=origin->x+x+y-1; s.raw_y=origin->y+x-y;
  int nx=std::max(1,int(std::round(width/7.))),ny=std::max(1,int(std::round(height/6.)));
  double wx=.12*wave(s.raw_x,s.raw_y,width,height,nx,ny,.4)+.035*wave(s.raw_x,s.raw_y,width,height,nx*2,-ny,.7);
  double wy=.12*wave(s.raw_x,s.raw_y,width,height,nx,-ny,.9)+.035*wave(s.raw_x,s.raw_y,width,height,nx,ny*2,1.7);
  if(baseline){wx=std::sin(s.raw_x*.83+s.raw_y*1.19)*.1+std::sin(s.raw_x*2.31-s.raw_y*.67)*.035;wy=std::sin(s.raw_x*1.07-s.raw_y*.91)*.1+std::sin(s.raw_x*.59+s.raw_y*2.03)*.035;}
  double gx=x-.5+wx,gy=y-.5+wy;int ix=int(std::floor(gx)),iy=int(std::floor(gy));
  double tx=smooth(gx-ix),ty=smooth(gy-iy);
  if(baseline){auto old=[](double a){a=std::clamp((a-.2)/.6,0.,1.);return a*a*(3-2*a);};tx=old(gx-ix);ty=old(gy-iy);}
  for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++){auto w=center(ix+dx,iy+dy);double k=(dx?tx:1-tx)*(dy?ty:1-ty);for(int i=0;i<5;i++)s.weights[i]+=w[i]*k;}
  return s;
 }
};
}
