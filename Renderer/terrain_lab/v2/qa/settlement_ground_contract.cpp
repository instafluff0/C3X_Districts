// Independently inspect the emitted packets: one receiver draw, no other changes.
#include "../contracts/packet_v1.h"
#include <cstring>

bool same_draw(labv2::Draw const& a,labv2::Draw const& b) {
 if(a.vertex_buffer!=b.vertex_buffer || a.constant_buffer!=b.constant_buffer || a.count!=b.count || a.stride!=b.stride ||
    a.feature!=b.feature || a.depth!=b.depth || a.clear_depth!=b.clear_depth || a.depth_mode!=b.depth_mode ||
    a.blend_mode!=b.blend_mode || a.shader_index!=b.shader_index || a.frame_buffer!=b.frame_buffer ||
    a.world_attribute!=b.world_attribute || a.normal_attribute!=b.normal_attribute || a.uv_attribute!=b.uv_attribute ||
    a.alpha_texture_slot!=b.alpha_texture_slot || a.geometry_flags!=b.geometry_flags || a.alpha_cutoff!=b.alpha_cutoff ||
    a.textures!=b.textures || a.attributes.size()!=b.attributes.size())return false;
 for(size_t i=0;i<a.attributes.size();++i)if(a.attributes[i].components!=b.attributes[i].components || a.attributes[i].offset!=b.attributes[i].offset)return false;
 return true;
}

int main(int argc,char**argv) {
 try {
  if(argc!=4)throw std::runtime_error("usage: settlement_ground_contract original result insertion_index");
  auto a=labv2::read_packet(argv[1]),b=labv2::read_packet(argv[2]);size_t insertion=std::stoul(argv[3]);
  if(a.width!=b.width || a.height!=b.height || a.downsample!=b.downsample || a.color_branch!=b.color_branch ||
     a.valid_rect!=b.valid_rect || a.geometry_contract!=b.geometry_contract || a.binding_contract!=b.binding_contract ||
     a.shader_count!=b.shader_count || a.exposure!=b.exposure || a.textures.size()!=b.textures.size() ||
     a.buffers.size()+1!=b.buffers.size() || a.draws.size()+1!=b.draws.size() || insertion>=b.draws.size())
   throw std::runtime_error("unexpected settlement packet shape or frame change");
  for(size_t i=0;i<a.buffers.size();++i)if(a.buffers[i]!=b.buffers[i])throw std::runtime_error("original geometry/constants changed");
  for(size_t i=0;i<a.textures.size();++i){auto const& x=a.textures[i];auto const& y=b.textures[i];
   if(x.width!=y.width || x.height!=y.height || x.format!=y.format || x.mips.size()!=y.mips.size())throw std::runtime_error("existing texture layout changed");
   for(size_t j=0;j<x.mips.size();++j)if(x.mips[j].pitch!=y.mips[j].pitch || x.mips[j].bytes!=y.mips[j].bytes)throw std::runtime_error("existing texture/shadow pixels changed");
  }
  for(size_t i=0;i<a.draws.size();++i)if(!same_draw(a.draws[i],b.draws[i+(i>=insertion)]))throw std::runtime_error("existing draw or material binding changed");
  auto const& d=b.draws[insertion];
  if(d.geometry_flags!=2 || d.depth_mode!=1 || d.blend_mode!=1 || d.vertex_buffer!=a.buffers.size() || !d.feature)
   throw std::runtime_error("underlay is not a separate noncasting alpha receiver");
  auto const& vertices=b.buffers[d.vertex_buffer];
  for(unsigned i=0;i<d.count;++i){float material;memcpy(&material,vertices.data()+size_t(i)*d.stride+32,4);
   if(material<62 || material>63)throw std::runtime_error("underlay alpha range changed");}
  std::printf("{\"pass\":true,\"original_draws\":%zu,\"added_ground_vertices\":%u,\"original_geometry_constants_materials_shadow_textures_unchanged\":true}\n",a.draws.size(),d.count);
  return 0;
 }catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
