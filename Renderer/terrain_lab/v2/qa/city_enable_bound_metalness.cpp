// Enable the generic metalness channel already bound in a preserved city packet.
// Geometry, textures, lighting, shadows and all other material bits stay fixed.
#include "../contracts/packet_v1.h"
#include <cstdio>
#include <cstring>
#include <map>

int main(int argc,char**argv) {
 try {
  if(argc!=3)throw std::runtime_error("usage: city_enable_bound_metalness input output");
  auto p=labv2::read_packet(argv[1]);std::map<unsigned,bool> seen;unsigned draws=0,vertices=0;
  for(auto const& d:p.draws) {
   if(!d.feature || d.stride!=92 || !d.textures[120])continue;
   auto& data=p.buffers.at(d.vertex_buffer);
   if(data.size()%92 || data.size()<36)throw std::runtime_error("invalid city vertex buffer");
   float first=0;std::memcpy(&first,data.data()+32,4);
   if(first<99.5f || first>263.5f)continue;
   if(d.attributes.size()!=9 || d.attributes[3].offset!=32 || d.attributes[3].components!=1)
    throw std::runtime_error("unsupported city material layout");
   draws++;
   if(seen[d.vertex_buffer])continue;
   seen[d.vertex_buffer]=true;
   for(size_t pos=32;pos<data.size();pos+=92) {
    float material=0;std::memcpy(&material,data.data()+pos,4);
    // Texture-batched city parts can retain different per-triangle address
    // flags. Preserve every vertex's flags rather than copying the first one.
    int origin=material>=199.5f?200:100,bits=int(std::round(material-origin));
    if(bits<0 || bits>63 || material!=float(origin+bits) || (origin==200)!=(first>=199.5f))
     throw std::runtime_error("unsupported material channels");
    float updated=float(origin+(bits|16));
    std::memcpy(data.data()+pos,&updated,4);vertices++;
   }
  }
  if(!draws)throw std::runtime_error("no bound city metalness textures");
  if(!labv2::write_packet(argv[2],p))throw std::runtime_error("could not write material trial packet");
  std::printf("{\"city_draws_with_bound_metalness\":%u,\"vertex_buffers\":%zu,\"material_vertices\":%u}\n",draws,seen.size(),vertices);
  return 0;
 }catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
