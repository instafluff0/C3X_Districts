#include "field.h"
#include "../../contracts/packet_v1.h"
#include "../../shared/environment_runtime.cpp"
#include <cstring>
#include <regex>
#include <iostream>
std::string read(std::string path){std::ifstream f(path);if(!f)throw std::runtime_error("missing module input");return {std::istreambuf_iterator<char>(f),{}};}
std::string value(std::string text,std::string key,std::string suffix=""){
 std::regex re("\""+key+"\"\\s*:\\s*\"([^\"]+)\"");for(auto it=std::sregex_iterator(text.begin(),text.end(),re);it!=std::sregex_iterator();++it){std::string s=(*it)[1];if(suffix.empty()||(s.size()>=suffix.size()&&s.substr(s.size()-suffix.size())==suffix))return s;}throw std::runtime_error("missing field "+key);
}
double number(std::string text,std::string key,double fallback){std::regex re("\""+key+"\"\\s*:\\s*(-?[0-9.]+)");std::smatch m;return std::regex_search(text,m,re)?std::stod(m[1]):fallback;}
labv2::Texture dds(std::string path){
 std::string b=read(path);auto u=[&](size_t off){uint32_t v;if(off+4>b.size())throw std::runtime_error("truncated DDS");memcpy(&v,b.data()+off,4);return v;};
 if(b.substr(0,4)!="DDS "||b.substr(84,4)!="DX10")throw std::runtime_error("expected normalized DX10 DDS");
 labv2::Texture t;t.width=u(16);t.height=u(12);t.format=u(128);unsigned w=t.width,h=t.height,n=u(28);size_t off=148;
 if(t.format==71)t.format=72;else if(t.format==77)t.format=78; // declared color roles: hardware sRGB decode
 unsigned block=(t.format==72||t.format==80)?8:16;if(t.format!=72&&t.format!=78&&t.format!=80)throw std::runtime_error("unsupported normalized color DDS");
 for(unsigned i=0;i<std::max(1u,n);i++){labv2::Mip m;m.pitch=std::max(1u,(w+3)/4)*block;size_t size=m.pitch*std::max(1u,(h+3)/4);if(off+size>b.size())throw std::runtime_error("truncated DDS mip");m.bytes.assign(b.begin()+off,b.begin()+off+size);t.mips.push_back(m);off+=size;w=std::max(1u,w/2);h=std::max(1u,h/2);}return t;
}
struct Vertex {float p[3],uv[2],normal[3],shore[4],river[4];};
int main(int argc,char**argv){try{
 if(argc!=7)return 2;std::string input=read(argv[6]);hydro::Field f(value(input,"terrain",".csv"));
 std::string control=read(value(input,"controls"));f.wraps=number(control,"wrap_x",1)!=0;f.build();int mode=int(number(control,"mode",0));double legacy=number(control,"legacy",0);
 labv2::Packet p;p.width=unsigned(atoi(argv[2]));p.height=unsigned(atoi(argv[3]));p.downsample=unsigned(atoi(argv[5]));
 auto env=c3x_renderer::evaluate_environment(float(atof(argv[4])),0);
 if(number(control,"scene_linear",0)!=0){p.color_branch=1;p.valid_rect={0,0,p.width/p.downsample,p.height/p.downsample};p.exposure=env.exposure;}
 float constants[]={env.sun_direction[0],env.sun_direction[1],env.sun_direction[2],env.sun_intensity,
 env.sun_color[0],env.sun_color[1],env.sun_color[2],env.exposure,
 env.moon_direction[0],env.moon_direction[1],env.moon_direction[2],env.moon_intensity,
 env.moon_color[0],env.moon_color[1],env.moon_color[2],float(mode),
 env.ambient_color[0],env.ambient_color[1],env.ambient_color[2],float(legacy),
 float(f.wraps?f.map_width*.5:0),0,0,0};
 int nx=f.cols*28,ny=f.rows*28;std::vector<Vertex> grid((nx+1)*(ny+1));
 double sx=number(control,"tile_halfwidth",double(p.width)*.88/(f.cols+f.rows)),sy=sx*.5;
 double padding=number(control,"padding",0);
 for(int y=0;y<=ny;y++)for(int x=0;x<=nx;x++){
  hydro::P q{-.5-padding+double(x)*(f.cols+2*padding)/nx,-.5-padding+double(y)*(f.rows+2*padding)/ny};auto s=f.sample(q);
  double rd=s.river_distance-s.river_width*.5,land=hydro::smooth(0,.14,s.shore_distance);
  double hill=f.occupancy(q,true)*.37*land;
  double h=std::max(s.height,0.)+hill*hydro::smooth(.025,.80,rd);
  if(mode==2)h=s.height; // bed-only exposes continuous submerged slope.
  double cx=q.x-number(control,"center_x",(f.cols-1)*.5),cy=q.y-number(control,"center_y",(f.rows-1)*.5);
  Vertex v{};v.p[0]=float((cx-cy)*sx*2/p.width);v.p[1]=float((-(cx+cy)*sy+h*sx)*2/p.height);
  v.p[2]=float(.5+(cx+cy)*.014-h*.035);v.uv[0]=float(q.x+(f.origin_x+f.origin_y)*.5);v.uv[1]=float(q.y+(f.origin_x-f.origin_y)*.5);
  v.normal[2]=1;v.shore[0]=float(s.shore_distance);v.shore[1]=float(s.beach_width);v.shore[2]=float(s.rocky);v.shore[3]=float(s.depth);
  v.river[0]=float(rd);v.river[1]=float(s.wetness);v.river[2]=float(hill);v.river[3]=float(s.height);grid[y*(nx+1)+x]=v;
 }
 std::vector<Vertex> vertices;vertices.reserve(nx*ny*6);
 for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){int a=y*(nx+1)+x,b=a+1,c=a+nx+2,d=a+nx+1;for(int i:{a,b,c,a,c,d})vertices.push_back(grid[i]);}
 p.buffers.resize(2);p.buffers[0].resize(vertices.size()*sizeof(Vertex));memcpy(p.buffers[0].data(),vertices.data(),p.buffers[0].size());p.buffers[1].resize(sizeof(constants));memcpy(p.buffers[1].data(),constants,sizeof(constants));
 std::string material_root=value(control,"material_root");for(auto name:{"grassland","beach","shallows","cliff"})p.textures.push_back(dds(material_root+"/"+name+"_base_color.dds"));
 p.textures.push_back(dds(material_root+"/shallows_height.dds"));
 labv2::Draw draw;draw.count=uint32_t(vertices.size());draw.stride=sizeof(Vertex);draw.constant_buffer=1;draw.depth=1;
 draw.attributes={{3,0},{2,12},{3,20},{4,32},{4,48}};for(unsigned i=0;i<p.textures.size();i++)draw.textures[i]=i+1;p.draws.push_back(draw);
 return labv2::write_packet(argv[1],p)?0:1;
 }catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}}
