#include "render_core/coastal_waves.h"
#include <cassert>
#include <cstdio>
int main(){using namespace c3x_renderer::render_core;
 World w{32,32,true,true};WorldCoast coast;std::vector<uint32_t> data(512);
 auto fill=[&](int real){for(int y=0;y<32;++y)for(int x=y&1;x<32;x+=2)data[(y*32+x)/2]=x<16?(2|(real<<8)):(11|(11<<8));};
 fill(2);coast.update(w,data.data(),data.size(),1);
 unsigned active=0;int found_c=0,found_r=0;
 for(int y=8;y<24;++y)for(int x=13+(y&1);x<19;x+=2){
  int c=(x+y)/2,r=(x-y)/2;auto ribbon=coastal_wave_ribbon(coast,c,r,.7f);
  if(ribbon.empty())continue;found_c=c;found_r=r;++active;
  auto repeat=coastal_wave_ribbon(coast,c,r,.7f);assert(repeat.size()==ribbon.size());
  for(unsigned i=0;i<ribbon.size();++i){auto const&p=ribbon[i];assert(p.position.x==repeat[i].position.x && p.position.y==repeat[i].position.y);assert(std::isfinite(p.position.x));assert(p.distance>0 && p.distance<1);assert(p.along>=0 && p.along<=1);}
 }
 assert(active>0);
 auto first=coastal_wave_ribbon(coast,found_c,found_r,.7f);
 auto wrapped=coastal_wave_ribbon(coast,found_c+16,found_r+16,.7f);
 assert(first.size()==wrapped.size());
 for(unsigned i=0;i<first.size();++i){assert(std::abs(first[i].position.x+16-wrapped[i].position.x)<1e-6);assert(std::abs(first[i].position.y+16-wrapped[i].position.y)<1e-6);assert(first[i].coverage==wrapped[i].coverage);}
 for(int relief:{5,6}){fill(relief);coast.update(w,data.data(),data.size(),relief);
  for(int y=8;y<24;++y)for(int x=13+(y&1);x<19;x+=2)assert(coastal_wave_ribbon(coast,(x+y)/2,(x-y)/2,.7f).empty());}
 fill(2);coast.update(w,data.data(),data.size(),9);assert(!coastal_wave_ribbon(coast,found_c,found_r,.7f).empty());
 // Longer fronts must not bridge across dry land inside curved coves.
 for(int y=0;y<32;++y)for(int x=y&1;x<32;x+=2)
  data[(y*32+x)/2]=x<16+int(3*std::sin(y*.45))?(2|(2<<8)):(11|(11<<8));
 coast.update(w,data.data(),data.size(),11);
 for(int y=8;y<24;++y)for(int x=9+(y&1);x<23;x+=2){
  auto ribbon=coastal_wave_ribbon(coast,(x+y)/2,(x-y)/2,1.f);
  for(unsigned i=0;i<ribbon.size();i+=3){
   if(ribbon[i].coverage<1 || ribbon[i+1].coverage<1 || ribbon[i+2].coverage<1)continue;
   Point p=(ribbon[i].position+ribbon[i+1].position+ribbon[i+2].position)*(1./3.);
   auto observe=[](auto...){ };assert(coast.sample(p,observe,observe).distance<=.002);
  }
 }
 std::vector<Point> origins;unsigned candidates=0;
 for(int y=0;y<32;++y)for(int x=y&1;x<32;x+=2){
  int c=(x+y)/2,r=(x-y)/2;CoastSegment site;
  if(!coastal_wave_site(coast,c,r,site))continue;
  ++candidates;bool selected=coastal_wave_spaced(coast,c,r,site);
  CoastSegment copy;assert(coastal_wave_site(coast,c+16,r+16,copy));
  assert(selected==coastal_wave_spaced(coast,c+16,r+16,copy));
  if(selected)origins.push_back((site.a+site.b)*.5);
 }
 assert(!origins.empty() && origins.size()<candidates);
 for(unsigned i=0;i<origins.size();++i)for(unsigned j=0;j<i;++j)
  for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x){
   Point offset{double((x+y)*16),double((x-y)*16)};
   assert(length(origins[i]-origins[j]+offset)>=.85-1e-8);
  }
 std::printf("PASS coastal beaches, relief exclusion, deterministic spacing/wrap, cove safety, topology edit; origins=%zu/%u\n",origins.size(),candidates);
}
