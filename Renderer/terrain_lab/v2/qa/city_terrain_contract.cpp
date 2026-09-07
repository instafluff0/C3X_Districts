// Check preserved terrain/object data after city append and shared-shadow rebuild.
#define main unused_settlement_contract
#include "settlement_ground_contract.cpp"
#undef main

int main(int argc,char**argv) {
 try {
  if(argc!=3)throw std::runtime_error("usage: city_terrain_contract terrain combined");
  auto a=labv2::read_packet(argv[1]),b=labv2::read_packet(argv[2]);
  if(a.width!=b.width || a.height!=b.height || a.downsample!=b.downsample || a.color_branch!=b.color_branch ||
     a.valid_rect!=b.valid_rect || a.shader_count!=b.shader_count || a.exposure!=b.exposure ||
     a.draws.size()>=b.draws.size() || a.textures.size()>b.textures.size() || a.buffers.size()>b.buffers.size())
   throw std::runtime_error("city changed terrain frame or removed original data");
  for(size_t i=0;i<a.buffers.size();++i)if(a.buffers[i]!=b.buffers[i])throw std::runtime_error("original buffer changed");
  for(size_t i=0;i<a.textures.size();++i){auto const& x=a.textures[i];auto const& y=b.textures[i];
   if(x.width!=y.width || x.height!=y.height || x.format!=y.format || x.mips.size()!=y.mips.size())throw std::runtime_error("original texture shape changed");
   for(size_t j=0;j<x.mips.size();++j)if(x.mips[j].pitch!=y.mips[j].pitch || x.mips[j].bytes!=y.mips[j].bytes)throw std::runtime_error("original texture bytes changed");
  }
  size_t padded=0;
  for(size_t i=0;i<a.draws.size();++i){auto const& x=a.draws[i];auto y=b.draws[i];
   if(x.feature){
    if(y.stride!=x.stride+40 || y.attributes.size()!=x.attributes.size()+4)throw std::runtime_error("unexpected auxiliary feature layout");
    unsigned components[]={2,3,3,2},offsets[]={0,8,20,32};
    for(size_t k=0;k<4;++k){auto attr=y.attributes[x.attributes.size()+k];
     if(attr.components!=components[k] || attr.offset!=x.stride+offsets[k])throw std::runtime_error("auxiliary feature attributes changed");}
    auto const& before=a.buffers.at(x.vertex_buffer);auto const& after=b.buffers.at(y.vertex_buffer);
    for(size_t v=0;v<x.count;++v){
     if(std::memcmp(before.data()+v*x.stride,after.data()+v*y.stride,x.stride))throw std::runtime_error("original feature vertex changed");
     for(size_t k=x.stride;k<y.stride;++k)if(after[v*y.stride+k])throw std::runtime_error("nonzero feature padding");
    }
    y.stride=x.stride;y.vertex_buffer=x.vertex_buffer;y.attributes.resize(x.attributes.size());padded++;
   }
   y.frame_buffer=x.frame_buffer;y.textures[x.feature?17:25]=x.textures[x.feature?17:25];
   if(!same_draw(x,y))throw std::runtime_error("terrain material or draw state changed");
  }
  std::printf("{\"pass\":true,\"original_draws\":%zu,\"zero_padded_features\":%zu,\"original_geometry_materials_constants_unchanged\":true}\n",a.draws.size(),padded);
  return 0;
 }catch(std::exception const&e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
