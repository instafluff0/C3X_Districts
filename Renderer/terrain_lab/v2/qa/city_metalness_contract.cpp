// Independently inspect a channel-only edit; reject changes to any other byte.
#define main unused_settlement_contract_entry
#include "settlement_ground_contract.cpp"
#undef main

int main(int argc,char**argv) {
 try {
  if(argc!=3)throw std::runtime_error("usage: city_metalness_contract original result");
  auto a=labv2::read_packet(argv[1]),b=labv2::read_packet(argv[2]);
  if(a.width!=b.width || a.height!=b.height || a.downsample!=b.downsample || a.color_branch!=b.color_branch ||
     a.valid_rect!=b.valid_rect || a.geometry_contract!=b.geometry_contract || a.binding_contract!=b.binding_contract ||
     a.shader_count!=b.shader_count || a.exposure!=b.exposure || a.textures.size()!=b.textures.size() ||
     a.buffers.size()!=b.buffers.size() || a.draws.size()!=b.draws.size())throw std::runtime_error("packet shape changed");
  for(size_t i=0;i<a.textures.size();++i) {
   auto const& x=a.textures[i];auto const& y=b.textures[i];
   if(x.width!=y.width || x.height!=y.height || x.format!=y.format || x.mips.size()!=y.mips.size())throw std::runtime_error("texture shape changed");
   for(size_t j=0;j<x.mips.size();++j)if(x.mips[j].pitch!=y.mips[j].pitch || x.mips[j].bytes!=y.mips[j].bytes)throw std::runtime_error("material/shadow texture changed");
  }
  std::vector<bool> allowed(a.buffers.size(),false);unsigned changed=0;
  for(size_t i=0;i<a.draws.size();++i) {
   auto const& d=a.draws[i];if(!same_draw(d,b.draws[i]))throw std::runtime_error("draw/binding changed");
   auto const& vb=a.buffers.at(d.vertex_buffer);
   if(d.feature && d.stride==92 && vb.size()>=36) {
    float m=0;std::memcpy(&m,vb.data()+32,4);
    if(m>=99.5f && m<=263.5f && d.textures[120])allowed[d.vertex_buffer]=true;
   }
  }
  for(size_t index=0;index<a.buffers.size();++index) {
   auto expected=a.buffers[index];auto const& actual=b.buffers[index];
   if(expected.size()!=actual.size())throw std::runtime_error("buffer size changed");
   if(allowed[index]) {
    if(expected.size()%92)throw std::runtime_error("invalid city buffer");
    for(size_t v=0;v<expected.size()/92;++v) {
     float old=0,now=0;std::memcpy(&old,expected.data()+v*92+32,4);std::memcpy(&now,actual.data()+v*92+32,4);
     int origin=old>=199.5f?200:100,bits=int(old)-origin;
     if(bits<0 || bits>63 || old!=float(origin+bits) || now!=float(origin+(bits|16)))throw std::runtime_error("wrong material channel edit");
     if(old!=now)changed++;
     std::memcpy(expected.data()+v*92+32,&now,4);
    }
   }
   if(expected!=actual)throw std::runtime_error("geometry, UV, normal, constant or shadow frame byte changed");
  }
  if(!changed)throw std::runtime_error("no metalness channels changed");
  std::printf("{\"pass\":true,\"changed_material_vertices\":%u,\"all_other_buffer_bytes_draws_and_textures_exact\":true}\n",changed);
  return 0;
 }catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
