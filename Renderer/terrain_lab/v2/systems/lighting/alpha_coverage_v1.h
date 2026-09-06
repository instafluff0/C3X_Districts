#pragma once
#include "../../contracts/packet_v1.h"
#include <cmath>
namespace q6 {
// Source-independent BC1/BC3 alpha coverage for CPU light-depth preparation.
// Finest-mip nearest coverage is diagnostic; Q0's eventual GPU shadow pass
// must supply footprint-aware alpha coverage at distant views.
inline float alpha_nearest(labv2::Texture const& texture,float u,float v){
 auto const& m=texture.mips[0];unsigned x=unsigned((u-floor(u))*texture.width)%texture.width,y=unsigned((v-floor(v))*texture.height)%texture.height;
 unsigned texel=(y%4)*4+x%4;
 if(texture.format==71||texture.format==72){
   auto p=m.bytes.data()+(y/4)*m.pitch+(x/4)*8;
   unsigned c0=p[0]+p[1]*256,c1=p[2]+p[3]*256;
   unsigned bits=unsigned(p[4])|(unsigned(p[5])<<8)|(unsigned(p[6])<<16)|(unsigned(p[7])<<24);
   return c0<=c1&&((bits>>(texel*2))&3)==3?0.f:1.f;
 }
 if(texture.format==77||texture.format==78){
   auto p=m.bytes.data()+(y/4)*m.pitch+(x/4)*16;
   float a[8]={p[0]/255.f,p[1]/255.f};
   if(p[0]>p[1])for(int i=2;i<8;i++)a[i]=((8-i)*a[0]+(i-1)*a[1])/7;
   else{for(int i=2;i<6;i++)a[i]=((6-i)*a[0]+(i-1)*a[1])/5;a[6]=0;a[7]=1;}
   uint64_t bits=0;for(int i=0;i<6;i++)bits|=uint64_t(p[i+2])<<(8*i);
   return a[(bits>>(texel*3))&7];
 }
 throw std::runtime_error("Q6 alpha coverage requires declared BC1/BC3 texture");
}
}
