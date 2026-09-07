// Inspect the actual preserved light grid and unchanged city/terrain geometry.
#include "../contracts/packet_v1.h"
#include <iostream>
#include <cstring>

int main(int argc,char**argv) {
 try {
  if(argc!=4)throw std::runtime_error("usage: city_shadow_frame_contract source reference result");
  auto source=labv2::read_packet(argv[1]),reference=labv2::read_packet(argv[2]),result=labv2::read_packet(argv[3]);
  auto const& expected=reference.buffers.at(reference.draws.front().frame_buffer);
  if(expected.size()!=80 || source.draws.size()!=result.draws.size())throw std::runtime_error("frame/draw count mismatch");
  for(size_t i=0;i<source.draws.size();++i) {
   auto const& a=source.draws[i];auto const& b=result.draws[i];
   if(a.count!=b.count || a.stride!=b.stride || a.geometry_flags!=b.geometry_flags ||
      a.alpha_cutoff!=b.alpha_cutoff || a.blend_mode!=b.blend_mode ||
      source.buffers.at(a.vertex_buffer)!=result.buffers.at(b.vertex_buffer))
    throw std::runtime_error("frame control changed geometry or draw coverage");
   auto const& frame=result.buffers.at(b.frame_buffer);
   if(frame.size()!=80 || std::memcmp(frame.data(),expected.data(),64))throw std::runtime_error("light grid changed");
  }
  auto const& before=source.buffers.at(source.draws.front().frame_buffer);
  bool same_frame=before.size()==80 && !std::memcmp(before.data(),expected.data(),64);
  if(same_frame) {
   auto const& a=source.draws.front();auto const& b=result.draws.front();
   auto const& old=source.textures.at(a.textures[a.feature?17:25]-1);
   auto const& rebuilt=result.textures.at(b.textures[b.feature?17:25]-1);
   if(old.width!=rebuilt.width || old.height!=rebuilt.height || old.format!=rebuilt.format || old.mips.size()!=rebuilt.mips.size())
    throw std::runtime_error("unchanged frame changed shadow texture shape");
   for(size_t i=0;i<old.mips.size();++i)if(old.mips[i].pitch!=rebuilt.mips[i].pitch || old.mips[i].bytes!=rebuilt.mips[i].bytes)
    throw std::runtime_error("unchanged frame changed shadow texture pixels");
  }
  float old_z,reference_z;
  std::memcpy(&old_z,before.data()+56,4);std::memcpy(&reference_z,expected.data()+56,4);
  std::cout<<"{\"pass\":true,\"draws\":"<<source.draws.size()
           <<",\"source_origin_z\":"<<old_z<<",\"reference_origin_z\":"<<reference_z
           <<",\"identical_frame_texture_check\":"<<(same_frame?"true":"false")<<"}\n";
  return 0;
 } catch(std::exception const& e) {std::cerr<<e.what()<<"\n";return 1;}
}
