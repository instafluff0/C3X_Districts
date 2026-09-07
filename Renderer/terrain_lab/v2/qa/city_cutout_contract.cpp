// Inspect the generic city cutout contract in the actual shared replay packet.
#include "../contracts/packet_v1.h"
#include "../systems/lighting/alpha_coverage_v1.h"
#include <cstdio>
#include <cstring>
#include <map>

int main(int argc,char**argv){
 try{
  if(argc!=3)throw std::runtime_error("usage: city_cutout_contract packet on|off");
  auto p=labv2::read_packet(argv[1]);unsigned bodies=0,emissions=0;
  std::map<unsigned,bool> masks;
  for(auto const& d:p.draws){
   if(!d.feature || d.stride!=92)continue;
   auto const& v=p.buffers[d.vertex_buffer];float material=0;std::memcpy(&material,v.data()+32,4);
   if(material<99.5f)continue;
   bool emission=material>=199.5f;int bits=int(std::round(material-(emission?200:100)));
   if(d.attributes.size()!=9 || d.attributes[8].components!=2 || d.attributes[8].offset!=84)
    throw std::runtime_error("missing separate emission coordinates");
   if(!(bits&32))continue;
   unsigned binding=d.textures[121];if(!binding)throw std::runtime_error("missing opacity texture");
   if(emission){
    if(d.geometry_flags!=0 || d.blend_mode!=2)throw std::runtime_error("masked emission must not cast another shadow");
    emissions++;
   }else{
    if(d.geometry_flags!=7 || d.alpha_texture_slot!=121 || d.alpha_cutoff!=.5f || d.uv_attribute!=1)
     throw std::runtime_error("body and shadow masks disagree");
    bodies++;
   }
   if(!masks.count(binding)){
    auto const& texture=p.textures[binding-1];
    if(texture.format!=77)throw std::runtime_error("coverage requires linear BC3 alpha");
    bool low=false,high=false;
    for(unsigned y=0;y<texture.height;++y)for(unsigned x=0;x<texture.width;++x){
     float a=q6::alpha_nearest(texture,(x+.5f)/texture.width,(y+.5f)/texture.height);
     low|=a<.5f;high|=a>=.5f;
    }
    if(!low || !high)throw std::runtime_error("mask does not contain both openings and opaque coverage");
    masks[binding]=true;
   }
  }
  if((std::string(argv[2])=="on")!=(bodies>0))throw std::runtime_error("unexpected cutout activation");
  std::printf("{\"masked_body_draws\":%u,\"masked_emission_draws\":%u,\"coverage_textures\":%zu,\"pass\":true}\n",bodies,emissions,masks.size());
  return 0;
 }catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
