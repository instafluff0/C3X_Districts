// Q6-owned source packet postprocessor. All projections/world data come from Q0.
#include "../../shared/environment_runtime.cpp"
#include "shadow_field_v1.h"
#include "alpha_coverage_v1.h"
#include <cstdlib>
#include <iostream>
using namespace q6;
struct CasterCoverage { std::array<std::array<float,2>,3> uv; unsigned binding; float cutoff=.5f; };
float value_at(std::vector<uint8_t> const& b,size_t offset) {
 float value;memcpy(&value,b.data()+offset,4);return value;
}
int main(int argc,char**argv) {
 if(argc!=5)return 2;
 auto p=labv2::read_packet(argv[1]);
 if(p.color_branch!=1)throw std::runtime_error("Q6 shadows require scene-linear packet");
 auto e=c3x_renderer::evaluate_environment(float(atof(argv[3])),0);
 auto basis=build_shadow_frame(std::vector<WorldTriangle>{},e,64,6);
 std::vector<WorldTriangle> triangles;
 std::vector<CasterCoverage> coverage;
 std::vector<labv2::Draw> draws;
 F3 low={1e20f,1e20f,1e20f},high={-1e20f,-1e20f,-1e20f};
 unsigned removed=0,receivers=0;
 for(auto draw:p.draws) {
  bool generic=draw.world_attribute!=UINT32_MAX;
  unsigned world_index=generic?draw.world_attribute:(draw.feature?4:16);
  if(draw.attributes.size()<=world_index || draw.attributes[world_index].components!=4)
   throw std::runtime_error("Q0 authoritative world attribute is required");
  auto const& source=p.buffers.at(draw.vertex_buffer);
  auto world_offset=draw.attributes[world_index].offset;
  auto uv_offset=generic && draw.uv_attribute!=UINT32_MAX?draw.attributes[draw.uv_attribute].offset:draw.attributes[1].offset;
  auto class_offset=generic?0:draw.attributes[draw.feature?3:6].offset;
  std::vector<uint8_t> filtered;
  for(unsigned i=0;i<draw.count;i+=3) {
   float material=generic?0:value_at(source,size_t(i)*draw.stride+class_offset);
   int kind=int(std::floor(material+.5f));
   bool legacy_shadow=!generic && !draw.feature && (kind==7||kind==10||kind==12||kind==14);
   if(legacy_shadow){removed++;continue;}
   filtered.insert(filtered.end(),source.begin()+size_t(i)*draw.stride,
     source.begin()+size_t(i+3)*draw.stride);
   WorldTriangle tri;CasterCoverage c{};bool valid=true;
   for(int j=0;j<3;j++) {
    size_t start=size_t(i+j)*draw.stride;
    valid=valid && value_at(source,start+world_offset+12)>.5f;
    for(int axis=0;axis<3;axis++) {
     float f=value_at(source,start+world_offset+axis*4);
     if(!std::isfinite(f))throw std::runtime_error("nonfinite world receiver");
     tri[j][axis]=f;low[axis]=std::min(low[axis],f);high[axis]=std::max(high[axis],f);
    }
    c.uv[j]={value_at(source,start+uv_offset),value_at(source,start+uv_offset+4)};
   }
   if(!valid)continue;
   receivers++;
   // Only actual opaque/cutout source geometry casts. Bed, water and decals do not.
   bool caster=generic ? bool(draw.geometry_flags&1) :
    (draw.feature ? draw.blend_mode==0 : material>.75f && material<3.5f);
   if(!caster)continue;
   if(generic) {
    if(draw.blend_mode!=0)throw std::runtime_error("translucent shadow casting requires explicit optical model");
    if(draw.geometry_flags&4) {
     if(draw.alpha_texture_slot==UINT32_MAX || draw.uv_attribute==UINT32_MAX)
      throw std::runtime_error("cutout caster requires source alpha and UV");
     c.binding=draw.textures[draw.alpha_texture_slot];c.cutoff=draw.alpha_cutoff;
     if(!c.binding)throw std::runtime_error("unbound generic caster alpha");
    }
   } else if(draw.feature) {
    int slot=kind<4?25+kind:kind<8?94+kind-4:kind<13?89+kind-8:
             kind<21?108+kind-13:kind<29?116+kind-21:124+kind-29;
    if(slot<0||slot>=128)throw std::runtime_error("unknown source caster material");
    c.binding=draw.textures[slot];
    if(!c.binding)throw std::runtime_error("missing source caster alpha binding");
   }
   triangles.push_back(tri);coverage.push_back(c);
  }
  if(filtered.empty())continue;
  if(filtered.size()!=size_t(draw.count)*draw.stride) {
   draw.vertex_buffer=unsigned(p.buffers.size());draw.count=unsigned(filtered.size()/draw.stride);
   p.buffers.push_back(std::move(filtered));
  }
  draws.push_back(draw);
 }
 if(!receivers||triangles.empty())throw std::runtime_error("world scene has no receiver/caster geometry");
 F3 origin;for(int i=0;i<3;i++)origin[i]=(low[i]+high[i])*.5f;
 float extent=0;
 for(int corner=0;corner<8;corner++) {
  F3 v;for(int axis=0;axis<3;axis++)v[axis]=((corner&(1<<axis))?high[axis]:low[axis])-origin[axis];
  extent=std::max(extent,std::max(std::abs(dot(v,basis.U)),
   std::max(std::abs(dot(v,basis.V)),std::abs(dot(v,basis.L)))));
 }
 float span=std::ceil((extent*2+.5f)*4)/4;
 for(auto& triangle:triangles)for(auto& v:triangle)for(int i=0;i<3;i++)v[i]-=origin[i];
 auto alpha=[&](size_t i,float a,float b,float c) {
  auto const& mask=coverage[i];if(!mask.binding)return true;
  float u=mask.uv[0][0]*a+mask.uv[1][0]*b+mask.uv[2][0]*c;
  float v=mask.uv[0][1]*a+mask.uv[1][1]*b+mask.uv[2][1]*c;
  return alpha_nearest(p.textures.at(mask.binding-1),u,v)>=mask.cutoff;
 };
 // Keep shadow texel size bounded as the viewport grows; never broaden contact
 // or penumbra just because another tile enters the scene.
 int resolution=1024;
 while(span/resolution>6.f/1024 && resolution<4096)resolution*=2;
 p.textures.push_back(raster_shadow_field(triangles,basis.U,basis.V,basis.L,resolution,span,alpha));
 float constants[]={basis.U[0],basis.U[1],basis.U[2],span,
  basis.V[0],basis.V[1],basis.V[2],float(resolution),basis.L[0],basis.L[1],basis.L[2],0,
  origin[0],origin[1],origin[2],0,1,1,0,0};
 unsigned frame=unsigned(p.buffers.size());p.buffers.emplace_back(sizeof(constants));
 memcpy(p.buffers.back().data(),constants,sizeof(constants));
 constants[16]=0;unsigned no_receive_frame=unsigned(p.buffers.size());
 p.buffers.emplace_back(sizeof(constants));memcpy(p.buffers.back().data(),constants,sizeof(constants));
 for(auto& draw:draws) {
  bool receive=draw.world_attribute==UINT32_MAX || bool(draw.geometry_flags&2);
  draw.frame_buffer=receive?frame:no_receive_frame;
  draw.textures[draw.feature?17:25]=unsigned(p.textures.size());
 }
 p.draws=std::move(draws);p.binding_contract=2;
 std::cout<<"Q6 actual source casters="<<triangles.size()<<" receivers="<<receivers
  <<" removed legacy shadow triangles="<<removed<<" field_span="<<span<<"\n";
 return labv2::write_packet(argv[2],p)?0:1;
}
