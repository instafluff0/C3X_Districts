// Q2-owned packet provider: base-material isolation over authoritative CSV facts.
#include "../../contracts/packet_v1.h"
#include "../../shared/environment_runtime.cpp"
#include "surface.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <cstring>
std::string read(std::string p){std::ifstream f(p);if(!f)throw std::runtime_error("missing input "+p);return std::string(std::istreambuf_iterator<char>(f),{});}
// Minimal member lookup with nesting/string awareness; fixture schema is validated by Q0.
std::string member(std::string const& s,std::string key){
 int depth=0;for(size_t i=0;i<s.size();i++){
  if(s[i]=='"'){size_t start=++i;while(i<s.size()&&(s[i]!='"'||(i>0&&s[i-1]=='\\')))i++;
   if(depth==1&&s.substr(start,i-start)==key){size_t pos=s.find(':',i+1);if(pos==std::string::npos)break;return s.substr(pos+1);}}
  else if(s[i]=='{'||s[i]=='[')depth++;else if(s[i]=='}'||s[i]==']')depth--;
 }throw std::runtime_error("missing JSON member "+key);
}
std::string field(std::string const& s,std::string key){auto value=member(s,key);auto a=value.find('"'),b=value.find('"',a+1);if(a==std::string::npos||b==std::string::npos)throw std::runtime_error("invalid string");return value.substr(a+1,b-a-1);}

labv2::Texture texture(std::string path){
 std::ifstream f(path,std::ios::binary);uint32_t h[37]{};f.read((char*)h,148);
 if(!f||h[0]!=0x20534444||h[21]!=0x30315844)throw std::runtime_error("requires normalized DX10 DDS");
 labv2::Texture t;t.height=h[3];t.width=h[4];t.format=h[32];unsigned block=t.format==78||t.format==77?16:t.format==80?8:0;
 if(!block||!t.width||!t.height||h[7]>15)throw std::runtime_error("unsupported material storage");
 unsigned w=t.width,v=t.height;for(unsigned i=0;i<std::max(1u,h[7]);i++){labv2::Mip m;m.pitch=((w+3)/4)*block;m.bytes.resize(m.pitch*((v+3)/4));f.read((char*)m.bytes.data(),m.bytes.size());if(!f)throw std::runtime_error("truncated texture");t.mips.push_back(std::move(m));w=std::max(1u,w/2);v=std::max(1u,v/2);}return t;
}
struct Vertex {float position[3],uv[2],weights[4],tundra;};
struct Constants {float sun[4],moon[4],sun_color[4],moon_color[4],ambient[4];};
int main(int argc,char**argv){try{
 if(argc!=7)return 2;labv2::Packet p;p.width=std::atoi(argv[2]);p.height=std::atoi(argv[3]);p.downsample=std::atoi(argv[5]);
 auto descriptor=read(argv[6]);std::string id=field(descriptor,"id");bool baseline=id.find("baseline")!=std::string::npos;
 q2::Surface surface;std::istringstream csv(read(field(descriptor,"terrain")));std::string line;std::getline(csv,line);unsigned count,halo;int oc,orr;
 if(std::sscanf(line.c_str(),"C3X_BIQ_TERRAIN_WINDOW_V2,%d,%d,%u,%d,%d,%d,%d,%u",&surface.columns,&surface.rows,&count,&oc,&orr,&surface.width,&surface.height,&halo)!=8)throw std::runtime_error("invalid terrain header");
 while(std::getline(csv,line)){q2::Tile t;if(std::sscanf(line.c_str(),"%d,%d,%d,%d,%d,%d",&t.column,&t.row,&t.x,&t.y,&t.base,&t.real)!=6)throw std::runtime_error("invalid tile");surface.tiles.push_back(t);}
 if(surface.tiles.size()!=count+halo)throw std::runtime_error("tile count drift");
 // The normalized pack manifest, not a source adapter, defines every texture role.
 auto pack=field(member(descriptor,"packs"),"terrain");
 for(auto name:{"grassland","plains","desert","marsh","tundra"}){
  auto material=read(pack+"/materials/"+name+".json");
  for(auto role:{"base_color","height","specular"}){
   auto channel=member(material,role);
   p.textures.push_back(texture(pack+"/"+field(channel,"texture")));
  }
 }
 float scale=1.0f; // Civ III normal zoom: exact 128x64 tile basis; viewport crops the scene.
 std::vector<Vertex> vertices;auto vertex=[&](double x,double y){auto s=surface.sample(x,y,baseline);Vertex v{};
  double px=(x-y-(surface.columns-surface.rows)*.5)*64*scale;double py=(x+y-(surface.columns+surface.rows)*.5)*32*scale;
  v.position[0]=float(px*2/p.width);v.position[1]=float(-py*2/p.height);v.position[2]=.5;
  // Integer repeats around the map torus: no phase jump at aliases or wrap.
  v.uv[0]=float(s.raw_x*std::max(1.,std::round(surface.width*.13))/surface.width);
  v.uv[1]=float(s.raw_y*std::max(1.,std::round(surface.height*.13))/surface.height);
  for(int i=0;i<4;i++)v.weights[i]=float(s.weights[i]);v.tundra=float(s.weights[4]);return v;};
 for(int y=0;y<surface.rows*24;y++)for(int x=0;x<surface.columns*24;x++){double a=x/24.,b=y/24.;auto tl=vertex(a,b),tr=vertex(a+1/24.,b),bl=vertex(a,b+1/24.),br=vertex(a+1/24.,b+1/24.);for(auto v:{tl,tr,br,tl,br,bl})vertices.push_back(v);}
 auto e=c3x_renderer::evaluate_environment(float(std::atoi(argv[4])),0);Constants c{};for(int i=0;i<3;i++){c.sun[i]=e.sun_direction[i];c.moon[i]=e.moon_direction[i];c.sun_color[i]=e.sun_color[i]*e.sun_intensity;c.moon_color[i]=e.moon_color[i]*e.moon_intensity;c.ambient[i]=e.ambient_color[i];}c.ambient[3]=e.exposure;
 p.buffers.resize(2);p.buffers[0].resize(vertices.size()*sizeof(Vertex));std::memcpy(p.buffers[0].data(),vertices.data(),p.buffers[0].size());p.buffers[1].resize(sizeof(c));std::memcpy(p.buffers[1].data(),&c,sizeof(c));
 labv2::Draw d;d.count=vertices.size();d.stride=sizeof(Vertex);d.constant_buffer=1;d.depth=1;d.clear_depth=1;d.attributes={{3,0},{2,12},{4,20},{1,36}};for(unsigned i=0;i<p.textures.size();i++)d.textures[i]=i+1;p.draws.push_back(d);
 return labv2::write_packet(argv[1],p)?0:1;
 }catch(std::exception const& e){std::fprintf(stderr,"Q2: %s\n",e.what());return 1;}}
