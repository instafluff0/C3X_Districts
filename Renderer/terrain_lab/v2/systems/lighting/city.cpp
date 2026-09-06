// Q6 CPU shadow-field preparation, consumed by the shared GPU backend.
// Generic normalized geometry; no source-engine formats or private presenter.
#include "../../contracts/packet_v1.h"
#include "../../shared/environment_runtime.cpp"
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <regex>
#include "alpha_coverage_v1.h"
#include "shadow_field_v1.h"
using namespace q6;
int main(int argc,char**argv){
 if(argc!=7)return 2;
 std::ifstream input(argv[6]);std::string descriptor((std::istreambuf_iterator<char>(input)),{});
 std::smatch match;std::regex geometry("\"lighting_geometry\"\\s*:\\s*\"([^\"]+)\"");
 if(!std::regex_search(descriptor,match,geometry))throw std::runtime_error("missing lighting geometry descriptor");
 auto p=labv2::read_packet(match[1].str().c_str());
 // This provider uses the published FEATURE semantic order: position, UV,
 // normal, world TEXCOORD1, material TEXCOORD2. Native main order differs.
 p.draws[0].feature=1;
 p.width=unsigned(atoi(argv[2]));p.height=unsigned(atoi(argv[3]));p.downsample=unsigned(atoi(argv[5]));
 auto e=c3x_renderer::evaluate_environment(float(atof(argv[4])),0);
#ifdef Q6_LINEAR
 p.color_branch=1;p.exposure=e.exposure;p.valid_rect={0,0,p.width/p.downsample,p.height/p.downsample};
 p.draws[0].depth_mode=2;p.draws[0].blend_mode=0;p.draws[0].clear_depth=0;
#endif
 F3 light=norm({e.sun_direction[0]*e.sun_intensity+e.moon_direction[0]*e.moon_intensity,e.sun_direction[1]*e.sun_intensity+e.moon_direction[1]*e.moon_intensity,e.sun_direction[2]*e.sun_intensity+e.moon_direction[2]*e.moon_intensity});
 float horizontal=sqrt(light[0]*light[0]+light[1]*light[1]);
 F3 L=norm({light[0]/horizontal,light[1]/horizontal,1.35f});
 F3 U=norm(cross({0,0,1},L)),V=cross(L,U);
 constexpr int S=1024;constexpr float span=6;
 auto& buffer=p.buffers[0];float* verts=reinterpret_cast<float*>(buffer.data());size_t count=buffer.size()/48;
 std::vector<WorldTriangle> triangles;
 std::vector<size_t> triangle_vertices;
 for(size_t i=0;i<count;i+=3){
   if(verts[i*12+11]>7.5f)continue;
   WorldTriangle triangle;
   for(int j=0;j<3;j++){float* v=verts+(i+j)*12;triangle[j]={v[8],v[9],v[10]};}
   triangles.push_back(triangle);triangle_vertices.push_back(i);
 }
 auto coverage=[&](size_t tri,float a,float b,float c){
   float* v=verts+triangle_vertices[tri]*12;
   float u=v[3]*a+v[15]*b+v[27]*c, w=v[4]*a+v[16]*b+v[28]*c;
   unsigned binding=p.draws[0].textures[unsigned(v[11])];
   return binding && q6::alpha_nearest(p.textures[binding-1],u,w)>=.5f;
 };
 p.textures.push_back(raster_shadow_field(triangles,U,V,L,S,span,coverage));
 F3 right=norm({1,-1,0}),camera=norm({1,1,.81649658f}),up=cross(right,camera);
 for(size_t i=0;i<count;i++){float* v=verts+i*12;F3 w={v[8],v[9],v[10]};v[0]=dot(w,right)/1.40f;v[1]=(dot(w,up)-.12f)/(.9333333f);v[2]=.5f-dot(w,camera)*.10f;
#ifdef Q6_NATIVE_PROJECTION
 v[0]=(w[0]-w[1])*128/p.width;
 v[1]=(-(w[0]+w[1])*32+w[2]*80.9543f)*2/p.height;
#endif
 }
 float cb[]={e.sun_direction[0],e.sun_direction[1],e.sun_direction[2],e.sun_intensity,
 e.sun_color[0],e.sun_color[1],e.sun_color[2],e.exposure,
 e.moon_direction[0],e.moon_direction[1],e.moon_direction[2],e.moon_intensity,
 e.moon_color[0],e.moon_color[1],e.moon_color[2],e.night_activation,
 e.ambient_color[0],e.ambient_color[1],e.ambient_color[2],e.emissive_scale,
 U[0],U[1],U[2],span,V[0],V[1],V[2],float(S),L[0],L[1],L[2],0,
 float(p.width),float(p.height),0,0};
 p.buffers[1].resize(sizeof(cb));memcpy(p.buffers[1].data(),cb,sizeof(cb));
 p.draws[0].textures[16]=uint32_t(p.textures.size());
#ifdef Q6_REVERSE
 for(size_t i=0;i<count/2;i+=3)for(int k=0;k<36;k++)std::swap(verts[i*12+k],verts[(count-3-i)*12+k]);
#endif
 return labv2::write_packet(argv[1],p)?0:1;
}
