// Independently verify that only shared-frame bindings and appended data changed.
#define main unused_settlement_contract_entry
#include "settlement_ground_contract.cpp"
#undef main

int main(int argc,char**argv) {
 try {
  if(argc!=4)throw std::runtime_error("usage: frame_data_contract original result payload");
  auto a=labv2::read_packet(argv[1]),b=labv2::read_packet(argv[2]);
  FILE* f=labv2::open_path(argv[3],"rb");if(!f)throw std::runtime_error("missing payload");
  fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);if(n<=0 || n>65536){fclose(f);throw std::runtime_error("invalid payload size");}
  std::vector<uint8_t> data(n);bool ok=fread(data.data(),1,data.size(),f)==data.size();fclose(f);
  if(!ok)throw std::runtime_error("truncated payload");
  if(a.width!=b.width || a.height!=b.height || a.downsample!=b.downsample || a.color_branch!=b.color_branch ||
     a.valid_rect!=b.valid_rect || a.geometry_contract!=b.geometry_contract || a.binding_contract!=b.binding_contract ||
     a.shader_count!=b.shader_count || a.exposure!=b.exposure || a.textures.size()!=b.textures.size() ||
     a.buffers.size()>=b.buffers.size() || a.draws.size()!=b.draws.size())throw std::runtime_error("unexpected packet shape change");
  for(size_t i=0;i<a.buffers.size();++i)if(a.buffers[i]!=b.buffers[i])throw std::runtime_error("original buffer changed");
  for(size_t i=0;i<a.textures.size();++i){auto const& x=a.textures[i];auto const& y=b.textures[i];
   if(x.width!=y.width || x.height!=y.height || x.format!=y.format || x.mips.size()!=y.mips.size())throw std::runtime_error("texture shape changed");
   for(size_t j=0;j<x.mips.size();++j)if(x.mips[j].pitch!=y.mips[j].pitch || x.mips[j].bytes!=y.mips[j].bytes)throw std::runtime_error("material or shadow texture changed");
  }
  for(size_t i=0;i<a.draws.size();++i){
   auto expected=a.draws[i];auto const& actual=b.draws[i];
   if(actual.frame_buffer<a.buffers.size())throw std::runtime_error("missing appended frame");
   auto const& old=a.buffers.at(expected.frame_buffer);auto const& current=b.buffers.at(actual.frame_buffer);
   if(old.size()!=80 || current.size()!=old.size()+data.size() ||
      !std::equal(old.begin(),old.end(),current.begin()) ||
      !std::equal(data.begin(),data.end(),current.begin()+old.size()))throw std::runtime_error("frame prefix or payload mismatch");
   expected.frame_buffer=actual.frame_buffer;
   if(!same_draw(expected,actual))throw std::runtime_error("draw geometry, material or state changed");
  }
  std::printf("{\"pass\":true,\"draws\":%zu,\"payload_bytes\":%zu,\"original_buffers_textures_draws_and_shadow_prefix_preserved\":true}\n",a.draws.size(),data.size());
  return 0;
 }catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
