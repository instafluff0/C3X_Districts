// Inspect emitted source fields independently of shader input-location mapping.
#include <algorithm>
#include "../../contracts/packet_v1.h"
#include <cstring>
#include <iostream>
int main(int argc,char**argv){try{
 if(argc!=2)return 2;auto p=labv2::read_packet(argv[1]);
 double land=0,water=0;float lo=1e9,hi=-1e9,depth=0,rocky=0;unsigned draws=0;
 for(auto const& d:p.draws){if(d.feature)continue;if(d.attributes.size()<2)throw std::runtime_error("source attributes missing");
  auto a=d.attributes.back();if(a.components!=4)throw std::runtime_error("hydrology field layout");draws++;
  for(unsigned i=0;i<d.count;i++){float v[4];std::memcpy(v,p.buffers[d.vertex_buffer].data()+i*d.stride+a.offset,sizeof(v));
   for(float x:v)if(!std::isfinite(x))throw std::runtime_error("nonfinite emitted sample");
   if(v[2]<0||v[2]>1||v[3]<0)throw std::runtime_error("invalid hydrology sample");
   lo=std::min(lo,v[0]);hi=std::max(hi,v[0]);depth=std::max(depth,v[3]);rocky=std::max(rocky,v[2]);(v[0]>=0?land:water)++;
  }
 }
 std::cout<<"{\"draws\":"<<draws<<",\"land_vertex_samples\":"<<land<<",\"water_vertex_samples\":"<<water<<",\"min_signed_distance\":"<<lo<<",\"max_signed_distance\":"<<hi<<",\"max_depth\":"<<depth<<",\"max_rockiness\":"<<rocky<<"}\n";
 return land>0&&water>0?0:1;
}catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}}
