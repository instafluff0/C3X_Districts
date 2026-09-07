// Replace a fingerprinted ground-only atlas binding in an existing Lab packet.
// No geometry, shadows, body materials, draw state or lighting are regenerated.
#define main city_unused_frozen_entry
#include "../shared/frozen_scene.cpp"
#undef main
#include "../shared/environment_runtime.cpp"

bool same_texture(labv2::Texture const& a,labv2::Texture const& b) {
 if(a.width!=b.width || a.height!=b.height || a.format!=b.format || a.mips.size()!=b.mips.size())return false;
 for(size_t i=0;i<a.mips.size();++i)if(a.mips[i].pitch!=b.mips[i].pitch || a.mips[i].bytes!=b.mips[i].bytes)return false;
 return true;
}

int main(int argc,char**argv) {
 try {
  if(argc!=5)throw std::runtime_error("usage: city_ground_binding_probe input.packet expected.dds replacement.dds output.packet");
  recorded=labv2::read_packet(argv[1]);
  auto original=recorded;
  ID3D11Device device;
  auto texture=[&](char const* path){
   std::vector<uint8_t> bytes;if(!read_file(path,bytes)||bytes.size()<148)throw std::runtime_error("ground DDS missing");
   unsigned fmt=read_u32(bytes,128);if(fmt!=78)throw std::runtime_error("ground probe requires normalized BC3 SRGB atlas");
   ID3D11ShaderResourceView* view=nullptr;unsigned w,h;
   if(!load_dds(&device,path,fmt,&view,w,h))throw std::runtime_error("ground DDS load failed");
   auto value=recorded.textures.at(view->id-1);release(view);
   recorded.textures.resize(original.textures.size());return value;
  };
  auto expected=texture(argv[2]),replacement=texture(argv[3]);
  unsigned replacement_id=0;
  for(unsigned i=0;i<recorded.textures.size();++i)if(same_texture(recorded.textures[i],replacement)){replacement_id=i+1;break;}
  if(!replacement_id){recorded.textures.push_back(replacement);replacement_id=unsigned(recorded.textures.size());}
  unsigned changed=0,vertices=0;
  for(auto& d:recorded.draws){
   if(!d.feature || d.attributes.size()<5 || d.attributes[3].offset!=32 || d.stride<52 || !d.count)continue;
   auto const& v=recorded.buffers.at(d.vertex_buffer);float material=0;memcpy(&material,v.data()+32,4);
   if(material!=60)continue;
   if(d.depth_mode!=1 || d.blend_mode!=1 || d.geometry_flags!=2 || !d.textures[124])throw std::runtime_error("ground draw state changed");
   if(!same_texture(recorded.textures.at(d.textures[124]-1),expected))continue;
   for(unsigned i=0;i<d.count;++i){float m=0;memcpy(&m,v.data()+size_t(i)*d.stride+32,4);if(m!=60)throw std::runtime_error("mixed ground/body draw");}
   // An identical-material control must also retain its original resource ID;
   // rebinding an equal duplicate is unnecessary and can alter GPU rounding.
   if(!same_texture(expected,replacement))d.textures[124]=replacement_id;
   changed++;vertices+=d.count;
  }
  if(!changed)throw std::runtime_error("expected ground atlas did not match any city draw");
  if(recorded.buffers!=original.buffers)throw std::runtime_error("binding probe changed geometry or constants");
  if(!labv2::write_packet(argv[4],recorded))throw std::runtime_error("packet write failed");
  std::printf("{\"matched_ground_draws\":%u,\"ground_vertices\":%u,\"all_buffers_unchanged\":true,\"pass\":true}\n",changed,vertices);
  return 0;
 }catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
