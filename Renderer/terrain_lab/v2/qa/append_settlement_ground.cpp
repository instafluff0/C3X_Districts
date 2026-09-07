// Append a generic projected settlement underlay before existing city decals.
// All previous draw/resource identities, caster geometry and shadow maps remain.
#include "../contracts/packet_v1.h"
#include <fstream>
#include <cstring>

int main(int argc,char**argv) {
 try {
  if(argc!=5)throw std::runtime_error("usage: append_settlement_ground input.packet ground.bin atlas.dds output.packet");
  auto p=labv2::read_packet(argv[1]);
  std::ifstream atlas(argv[3],std::ios::binary);uint32_t header[37];atlas.read((char*)header,sizeof(header));
  if(!atlas || header[0]!=0x20534444 || header[21]!=0x30315844 || header[32]!=78)
   throw std::runtime_error("settlement atlas requires normalized BC3 SRGB DDS");
  labv2::Texture texture;texture.width=header[4];texture.height=header[3];texture.format=78;
  unsigned w=texture.width,h=texture.height;
  if(!w || !h || w>4096 || h>4096 || !header[7] || header[7]>13)throw std::runtime_error("atlas dimensions");
  for(unsigned i=0;i<header[7];++i){labv2::Mip mip;mip.pitch=((w+3)/4)*16;
   mip.bytes.resize(size_t(mip.pitch)*((h+3)/4));atlas.read((char*)mip.bytes.data(),mip.bytes.size());
   texture.mips.push_back(std::move(mip));w=std::max(1u,w/2);h=std::max(1u,h/2);}
  if(!atlas || atlas.peek()!=EOF)throw std::runtime_error("atlas byte layout");
  unsigned template_index=unsigned(p.draws.size());
  for(unsigned i=0;i<p.draws.size();++i){
   auto const& draw=p.draws[i];if(!draw.feature || draw.stride<52 || !draw.textures[124])continue;
   float m;memcpy(&m,p.buffers.at(draw.vertex_buffer).data()+32,4);if(m!=60)continue;
   auto const& t=p.textures.at(draw.textures[124]-1);
   bool same=t.width==texture.width && t.height==texture.height && t.format==texture.format && t.mips.size()==texture.mips.size();
   if(same)for(size_t j=0;j<t.mips.size();++j)same&=t.mips[j].pitch==texture.mips[j].pitch && t.mips[j].bytes==texture.mips[j].bytes;
   if(same){template_index=i;break;}
  }
  if(template_index>=p.draws.size())throw std::runtime_error("ground template draw missing");
  auto d=p.draws[template_index];float material;
  memcpy(&material,p.buffers.at(d.vertex_buffer).data()+32,4);
  if(material!=60 || !d.feature || d.geometry_flags!=2 || d.blend_mode!=1 || d.depth_mode!=1)
   throw std::runtime_error("settlement requires an existing noncasting ground draw");
  std::ifstream file(argv[2],std::ios::binary);unsigned magic=0,count=0;
  file.read((char*)&magic,4);file.read((char*)&count,4);
  if(magic!=0x31524753 || count>180000 || count<3 || count%3 || d.stride<52)
   throw std::runtime_error("invalid settlement ground wire");
  std::vector<uint8_t> vertices(size_t(count)*d.stride,0);
  for(unsigned i=0;i<count;++i){
   float values[13];file.read((char*)values,sizeof(values));
   for(float value:values)if(!std::isfinite(value))throw std::runtime_error("nonfinite settlement vertex");
   if(values[8]<62 || values[8]>63)throw std::runtime_error("settlement alpha material range");
   memcpy(vertices.data()+size_t(i)*d.stride,values,sizeof(values));
  }
  if(!file || file.peek()!=EOF)throw std::runtime_error("truncated/trailing settlement wire");
  d.count=count;d.vertex_buffer=unsigned(p.buffers.size());p.buffers.push_back(std::move(vertices));
  size_t insertion=0;
  for(;insertion<p.draws.size();++insertion){
   auto const& draw=p.draws[insertion];if(!draw.feature || draw.stride<52)continue;
   memcpy(&material,p.buffers.at(draw.vertex_buffer).data()+32,4);if(material==60)break;
  }
  p.draws.insert(p.draws.begin()+insertion,d);
  if(!labv2::write_packet(argv[4],p))throw std::runtime_error("settlement packet write failed");
  std::printf("{\"vertices\":%u,\"insertion_draw\":%zu,\"template_draw\":%u,\"caster_added\":false,\"pass\":true}\n",count,insertion,template_index);
  return 0;
 }catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
