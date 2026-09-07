// Compare city triangles across material-layout upgrades and draw rebatching.
#include "../contracts/packet_v1.h"
#include <cstring>
#include <map>

std::map<std::string,unsigned> city_triangles(labv2::Packet const& p) {
 std::map<std::string,unsigned> result;
 for(auto const& d:p.draws) {
  if(!d.feature || d.stride<52)continue;
  auto const& data=p.buffers.at(d.vertex_buffer);
  float first=0;std::memcpy(&first,data.data()+32,4);
  if(first<39.5f)continue;
  if(d.count%3)throw std::runtime_error("nontriangular city draw");
  for(size_t tri=0;tri<d.count;tri+=3) {
   std::string key;
   for(size_t corner=0;corner<3;++corner) {
    auto v=data.data()+(tri+corner)*d.stride;float material=0;std::memcpy(&material,v+32,4);
    bool emission=(material>=79.5f && material<89.5f) || material>=199.5f;
    char kind=emission?'e':material>=59.5f && material<69.5f?'g':'b';key+=kind;
    key.append((char const*)v,12); // Authoritative projected position/depth.
    key.append((char const*)v+(emission && d.stride==92?84:12),8); // Base or emissive UV.
    key.append((char const*)v+36,16); // World coordinates and receiver validity.
   }
   result[key]++;
  }
 }
 return result;
}

int main(int argc,char**argv) {
 try {
  if(argc!=3)throw std::runtime_error("usage: city_geometry_material_contract original restored");
  auto a=labv2::read_packet(argv[1]),b=labv2::read_packet(argv[2]);
  if(a.width!=b.width || a.height!=b.height || a.valid_rect!=b.valid_rect || a.exposure!=b.exposure)
   throw std::runtime_error("city frame changed");
  auto before=city_triangles(a),after=city_triangles(b);
  if(before.empty() || before!=after)throw std::runtime_error("city position, topology, world coordinates or base/emissive UV changed");
  unsigned count=0;for(auto const& row:before)count+=row.second;
  std::printf("{\"pass\":true,\"city_triangles\":%u,\"positions_topology_world_and_primary_uvs_exact\":true}\n",count);
  return 0;
 }catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
