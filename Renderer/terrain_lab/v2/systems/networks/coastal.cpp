// Q5 source ribbons, projected by the pinned Q0 surface query; wire6 receivers.
#define Q5_TEXTURE_ONLY 1
#include "module.cpp"
int main(int argc,char**argv) {
 try {
  if(argc!=7)return 2;
  std::ifstream input(argv[6]);std::stringstream text;text<<input.rdbuf();
  std::string descriptor=text.str();std::smatch match;
  if(!std::regex_search(descriptor,match,std::regex("\"path\"\\s*:\\s*\"([^\"]*/corridors\\.json)\"")))return 3;
  std::ifstream sidecar(match[1].str());std::stringstream st;st<<sidecar.rdbuf();std::string closure=st.str();
  if(!std::regex_search(closure,match,std::regex("\"source_geometry\"\\s*:\\s*\\{\\s*\"path\"\\s*:\\s*\"([^\"]+)\"")))return 3;
  std::ifstream mesh(match[1].str(),std::ios::binary);uint32_t header[4];mesh.read((char*)header,16);
  if(!mesh||header[0]!=0x514e4331||header[3]>3000000||header[3]%3)return 4;
  labv2::Packet p;p.width=atoi(argv[2]);p.height=atoi(argv[3]);p.downsample=atoi(argv[5]);
  if(header[1]!=p.width||header[2]!=p.height)throw std::runtime_error("re-query route projection for changed viewport");
  p.geometry_contract=1;p.color_branch=1;p.valid_rect={0,0,p.width/p.downsample,p.height/p.downsample};
  auto e=c3x_renderer::evaluate_environment(float(atof(argv[4])),0);p.exposure=e.exposure;
  p.buffers.resize(2);p.buffers[0].resize(size_t(header[3])*68);mesh.read((char*)p.buffers[0].data(),p.buffers[0].size());
  if(!mesh||mesh.peek()!=EOF)return 5;
  float constants[]={float(p.width),float(p.height),0,0,0,0,0,0,
   e.sun_direction[0],e.sun_direction[1],e.sun_direction[2],e.sun_intensity,
   e.sun_color[0],e.sun_color[1],e.sun_color[2],e.exposure,
   e.moon_direction[0],e.moon_direction[1],e.moon_direction[2],e.moon_intensity,
   e.moon_color[0],e.moon_color[1],e.moon_color[2],0,
   e.ambient_color[0],e.ambient_color[1],e.ambient_color[2],0};
  p.buffers[1].resize(sizeof(constants));memcpy(p.buffers[1].data(),constants,sizeof(constants));
  labv2::Draw d;d.count=header[3];d.stride=68;d.constant_buffer=1;d.depth_mode=1;d.blend_mode=1;
  d.attributes={{3,0},{3,12},{3,24},{4,36},{4,52}};
  d.world_attribute=4;d.normal_attribute=1;d.geometry_flags=2; // Ground ribbons receive; no invented ribbon shadows.
  if(!std::regex_search(descriptor,match,std::regex("\"routes\"\\s*:\\s*\"([^\"]+)\"")))return 6;
  std::string pack=match[1];
  for(int i=0;i<10;i++) {
   std::ifstream mat(pack+"/materials/routes/material_0"+std::to_string(i)+".json");
   std::stringstream mt;mt<<mat.rdbuf();std::string content=mt.str();
   if(!std::regex_search(content,match,std::regex("\"texture\"\\s*:\\s*\"([^\"]+)\"")))return 7;
   p.textures.push_back(load_texture(pack+"/"+match[1].str()));d.textures[i]=i+1;
  }
  p.draws.push_back(d);return labv2::write_packet(argv[1],p)?0:8;
 }catch(std::exception const&e){fprintf(stderr,"Q5 coastal: %s\n",e.what());return 1;}
}
