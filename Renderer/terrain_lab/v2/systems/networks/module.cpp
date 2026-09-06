// Thin packet adapter. Prepared geometry remains owned by Q5, backend by Q0.
#include "../../contracts/packet_v1.h"
#include "../../shared/environment_runtime.cpp"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
labv2::Texture load_texture(std::string path) {
 std::ifstream f(path,std::ios::binary);uint32_t h[37]{};f.read((char*)h,148);
 if(!f||h[0]!=0x20534444||h[21]!=0x30315844)throw std::runtime_error("requires normalized DX10 DDS");
 labv2::Texture t;t.height=h[3];t.width=h[4];t.format=h[32];if(t.format==77)t.format=78;if(t.format==71)t.format=72;
 unsigned block=t.format==78||t.format==77?16:t.format==72||t.format==71?8:0;
 if(!block||!t.width||!t.height||h[7]>15)throw std::runtime_error("unsupported route material");
 unsigned w=t.width,v=t.height;for(unsigned i=0;i<std::max(1u,h[7]);i++){
  labv2::Mip m;m.pitch=((w+3)/4)*block;m.bytes.resize(m.pitch*((v+3)/4));
  f.read((char*)m.bytes.data(),m.bytes.size());if(!f)throw std::runtime_error("truncated route texture");
  t.mips.push_back(std::move(m));w=std::max(1u,w/2);v=std::max(1u,v/2);
 }return t;
}
#ifndef Q5_TEXTURE_ONLY
int main(int argc,char**argv) {
 try {
  if(argc!=7)return 2;
  std::ifstream input(argv[6]);std::stringstream text;text<<input.rdbuf();std::smatch match;
  std::string descriptor=text.str();
  if(!std::regex_search(descriptor,match,std::regex("\"network_mesh\"\\s*:\\s*\"([^\"]+)\"")))return 3;
  std::ifstream mesh(match[1].str(),std::ios::binary);uint32_t magic=0,count=0;
  float camera[4];mesh.read((char*)&magic,4);mesh.read((char*)&count,4);mesh.read((char*)camera,16);
  if((magic!=0x354e4554&&magic!=0x354e4555)||count>3000000||count%3)return 4;
  labv2::Packet p;p.width=atoi(argv[2]);p.height=atoi(argv[3]);p.downsample=atoi(argv[5]);
  p.buffers.resize(2);p.buffers[0].resize(size_t(count)*52);
  for(uint32_t i=0;i<count;i++){
   float v[13]={};v[11]=-1;mesh.read((char*)v,magic==0x354e4555?52:36);
   memcpy(p.buffers[0].data()+i*52,v,52);
  }
  if(!mesh||mesh.peek()!=EOF)return 5;
  // Q0 currently binds constants only to fragment stages; project on the CPU.
  for(uint32_t i=0;i<count;i++) {
   float *v=reinterpret_cast<float*>(p.buffers[0].data()+i*52);
   float x=v[0],y=v[1],z=v[2];
   v[0]=((x-camera[0])*camera[2]+p.width*.5f)/p.width*2-1;
   v[1]=1-((y*.5f-z-camera[1])*camera[2]+p.height*.5f)/p.height*2;
   v[2]=.5f-(y+z*2)*.00015f;
  }
  auto e=c3x_renderer::evaluate_environment(float(atof(argv[4])),0);
#ifdef Q5_SCENE_LINEAR
  p.color_branch=1;p.exposure=e.exposure;p.valid_rect={0,0,p.width/p.downsample,p.height/p.downsample};
#endif
  float constants[]={float(p.width),float(p.height),camera[0],camera[1],camera[2],camera[3],0,0,
   e.sun_direction[0],e.sun_direction[1],e.sun_direction[2],e.sun_intensity,
   e.sun_color[0],e.sun_color[1],e.sun_color[2],e.exposure,
   e.moon_direction[0],e.moon_direction[1],e.moon_direction[2],e.moon_intensity,
   e.moon_color[0],e.moon_color[1],e.moon_color[2],0,
   e.ambient_color[0],e.ambient_color[1],e.ambient_color[2],0};
  p.buffers[1].resize(sizeof(constants));memcpy(p.buffers[1].data(),constants,sizeof(constants));
  labv2::Draw d;d.count=count;d.stride=52;d.constant_buffer=1;d.depth=1;d.clear_depth=1;
  #ifdef Q5_SCENE_LINEAR
  d.depth_mode=2;d.blend_mode=1;d.clear_depth=0;
#endif
  d.attributes={{3,0},{3,12},{3,24},{4,36}};
  if(magic==0x354e4555){
   if(!std::regex_search(descriptor,match,std::regex("\"routes\"\\s*:\\s*\"([^\"]+)\"")))return 7;
   std::string pack=match[1];
   for(int i=0;i<10;i++){
    std::string name=pack+"/materials/routes/material_0"+std::to_string(i)+".json";
    std::ifstream mat(name);std::stringstream mt;mt<<mat.rdbuf();std::string content=mt.str();
    if(!std::regex_search(content,match,std::regex("\"texture\"\\s*:\\s*\"([^\"]+)\"")))return 8;
    p.textures.push_back(load_texture(pack+"/"+match[1].str()));d.textures[i]=i+1;
   }
   if(std::regex_search(descriptor,match,std::regex("\"bridge_textures\"\\s*:\\s*\"([^\"]+)\""))){
    std::ifstream list(match[1].str());std::string path;unsigned i=10;
    while(std::getline(list,path)){if(i>=18)return 9;p.textures.push_back(load_texture(path));d.textures[i]=i+1;i++;}
    if(i!=18)return 10;
   }

  }p.draws.push_back(d);
  return labv2::write_packet(argv[1],p)?0:6;
 }catch(std::exception const&e){fprintf(stderr,"network: %s\n",e.what());return 1;}
}
#endif
