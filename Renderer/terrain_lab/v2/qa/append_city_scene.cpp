// Append complete normalized city bodies to a copied single-namespace Lab packet.
// Terrain and existing object geometry stay intact; Q6 then rebuilds common shadows.
#define main city_unused_frozen_entry
#include "../shared/frozen_scene.cpp"
#undef main
#include "../shared/environment_runtime.cpp"
#include <fstream>
#include <map>

int main(int argc,char**argv) {
 try {
  if(argc!=4)throw std::runtime_error("usage: append_city_scene terrain.packet city.bin output.packet");
  recorded=labv2::read_packet(argv[1]);
  if(recorded.color_branch!=1 || recorded.shader_count!=1 || recorded.draws.empty())
   throw std::runtime_error("city probe requires scene-linear single-namespace terrain");
  auto constants=recorded.draws[0].constant_buffer;
  recorded.geometry_contract=1;
  std::ifstream in(argv[2],std::ios::binary);
  auto u32=[&](){uint32_t v=0;in.read((char*)&v,4);return v;};
  auto string=[&](){auto n=u32();if(n>4096)throw std::runtime_error("city path limit");
   std::string s(n,'\0');in.read(s.data(),n);return s;};
  if(u32()!=0x38514353)throw std::runtime_error("city scene wire version");
  ID3D11Device device;std::map<std::string,unsigned> known;
  auto texture=[&](std::string const& path,bool srgb){
   if(path.empty())return 0u;
   std::string key=path+(srgb?"/srgb":"/linear");
   if(known.count(key))return known[key];
   std::vector<uint8_t> bytes;if(!read_file(path,bytes)||bytes.size()<148)throw std::runtime_error("city DDS missing");
   unsigned fmt=read_u32(bytes,128);if(srgb && fmt==71)fmt=72;if(srgb && fmt==77)fmt=78;
   ID3D11ShaderResourceView* view=nullptr;unsigned w,h;
   if(!load_dds(&device,path,fmt,&view,w,h))throw std::runtime_error("city DDS load failed");
   auto id=view->id;release(view);known[key]=id;return id;
  };
  unsigned count=u32();if(count>512)throw std::runtime_error("city draw limit");
  for(unsigned i=0;i<count;++i){
   auto base=string(),emission=string(),ao=string(),normal=string();unsigned n=u32();
   if(n<3 || n>3000000 || n%3)throw std::runtime_error("city vertex limit");
   std::vector<uint8_t> vertices(size_t(n)*52);in.read((char*)vertices.data(),vertices.size());
   labv2::Draw d;d.feature=1;d.depth=1;d.clear_depth=0;d.depth_mode=2;d.blend_mode=0;
   d.constant_buffer=constants;d.vertex_buffer=unsigned(recorded.buffers.size());d.count=n;d.stride=52;
   d.attributes={{3,0},{2,12},{3,20},{1,32},{4,36}};
   d.world_attribute=4;d.normal_attribute=2;d.uv_attribute=1;d.geometry_flags=3;
   float material=0;memcpy(&material,vertices.data()+32,4);
   if(material>=79.5f){d.depth_mode=1;d.blend_mode=2;d.geometry_flags=0;}
   d.textures[124]=texture(base,true);d.textures[116]=texture(emission,true);
   d.textures[118]=texture(ao,false);d.textures[119]=texture(normal,false);
   recorded.buffers.push_back(std::move(vertices));recorded.draws.push_back(d);
  }
  if(!in || in.peek()!=EOF)throw std::runtime_error("truncated/trailing city geometry");
  return labv2::write_packet(argv[3],recorded)?0:1;
 }catch(std::exception const&e){fprintf(stderr,"city scene: %s\n",e.what());return 1;}
}
