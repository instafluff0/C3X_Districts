#pragma once
#include "../contracts/packet_v1.h"
#include <algorithm>
#include <cstring>
namespace labv2 {
inline float half_float(uint16_t h) {
  unsigned sign = h >> 15, exp = (h >> 10) & 31, mant = h & 1023;
  float v = exp == 0 ? std::ldexp(float(mant), -24)
            : exp == 31 ? (mant ? NAN : INFINITY)
                        : std::ldexp(float(mant + 1024), int(exp) - 25);
  return sign ? -v : v;
}
// CPU output reference shared by both backends. Matches Q6 response_v1.hlsl.
// Linear premultiplied RGBA and independent R8 validity are reconstructed first.
inline std::vector<uint8_t> display_pixels(const std::vector<uint16_t>& rgba,
    const std::vector<uint8_t>& validity, unsigned w, unsigned h, unsigned scale,
    const Packet& p) {
  unsigned ow=w/scale, oh=h/scale;
  std::vector<uint8_t> out(size_t(ow)*oh*4);
  for(unsigned y=0;y<oh;y++) for(unsigned x=0;x<ow;x++) {
    if(x<p.valid_rect[0]||y<p.valid_rect[1]||x>=p.valid_rect[2]||y>=p.valid_rect[3]) continue;
    double c[4]={}, valid=0;
    for(unsigned dy=0;dy<scale;dy++) for(unsigned dx=0;dx<scale;dx++) {
      size_t i=size_t(y*scale+dy)*w+x*scale+dx;
      if (!validity[i]) continue;
      valid+=validity[i]/255.0;
      for(unsigned k=0;k<4;k++) {
        float v=half_float(rgba[i*4+k]);
        if(!std::isfinite(v)) throw std::runtime_error("nonfinite linear attachment");
        c[k]+=v;
      }
    }
    double n=double(scale)*scale;
    for(auto &v:c) v/=n;
    if(valid==0||c[3]<=1e-6) continue;
    for(unsigned k=0;k<3;k++) c[k]=(std::max)(0.0,c[k]/c[3]*p.exposure);
    double shoulder=1+(std::max)(c[0],(std::max)(c[1],c[2]));
    for(unsigned k=0;k<3;k++) {
      double v=c[k]/shoulder;
      v=v<=.0031308 ? 12.92*v : 1.055*std::pow(v,1/2.4)-.055;
      out[(size_t(y)*ow+x)*4+2-k]=uint8_t(std::lround((std::min)(1.0,(std::max)(0.0,v))*255));
    }
    out[(size_t(y)*ow+x)*4+3]=uint8_t(std::lround((std::min)(1.0,(std::max)(0.0,c[3]))*255));
  }
  return out;
}
inline void write_attachment(const std::string& path,const void* bytes,size_t n) {
  FILE* f=open_path(path,"wb");
  if(!f) throw std::runtime_error("attachment write failed");
  bool ok=fwrite(bytes,1,n,f)==n; fclose(f);
  if(!ok) throw std::runtime_error("attachment write truncated");
}
}
