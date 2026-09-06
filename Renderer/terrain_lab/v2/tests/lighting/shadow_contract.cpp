#include "../../systems/lighting/shadow_field_v1.h"
#include "../../shared/environment_runtime.cpp"
#include <cassert>
#include <algorithm>
#include <iostream>
#include "../../systems/lighting/alpha_coverage_v1.h"
int main(){
 using namespace q6;
 std::vector<WorldTriangle> tris={{{F3{-.3f,-.3f,.6f},F3{.3f,-.3f,.6f},F3{-.3f,.3f,.6f}}},{{F3{.3f,-.3f,.6f},F3{.3f,.3f,.6f},F3{-.3f,.3f,.6f}}}};
 for(int hour:{0,6,12,18}){
   auto e=c3x_renderer::evaluate_environment(float(hour),0);
   auto f=build_shadow_frame(tris,e,128,4);
   auto reversed=tris;std::reverse(reversed.begin(),reversed.end());
   assert(f.texture.mips[0].bytes==build_shadow_frame(reversed,e,128,4).texture.mips[0].bytes);
   assert(std::abs(dot(f.U,f.V))<1e-6 && std::abs(dot(f.L,f.L)-1)<1e-6);
   assert(std::abs(f.L[2]/std::hypot(f.L[0],f.L[1])-1.35)<1e-5);
   // World ray from caster center reaches a real ground receiver at z=0.
   F3 ground={-.6f*f.L[0]/f.L[2],-.6f*f.L[1]/f.L[2],0};
   int x=int((dot(ground,f.U)/4+.5f)*128),y=int((dot(ground,f.V)/4+.5f)*128);
   uint16_t raw=0;memcpy(&raw,f.texture.mips[0].bytes.data()+(y*128+x)*8,2);
   assert(raw/65535.f>dot(ground,f.L)/4+.5f+.05f);
   c3x_renderer::AmbientAttachment attachment={};attachment.activation_policy=c3x_renderer::ActivationPolicy::night;attachment.animated=false;
   c3x_renderer::AttachmentInput input={0,0,true,true,true};
   assert(c3x_renderer::evaluate_attachment(attachment,input,e,float(hour)).visible_animation_count==0);
 }
 auto environment=c3x_renderer::evaluate_environment(12,0);auto basis=build_shadow_frame(tris,environment,128,4);
 auto empty=raster_shadow_field(tris,basis.U,basis.V,basis.L,128,4,[](size_t,float,float,float){return false;});
 assert(std::all_of(empty.mips[0].bytes.begin(),empty.mips[0].bytes.end(),[](uint8_t x){return x==0;}));
 labv2::Texture t;t.width=4;t.height=4;t.format=72;t.mips.push_back({8,std::vector<uint8_t>(8,0)});
 assert(alpha_nearest(t,.1,.1)==1);for(int i=4;i<8;i++)t.mips[0].bytes[i]=255;
 assert(alpha_nearest(t,.1,.1)==0);
 t.format=78;t.mips[0]={16,std::vector<uint8_t>(16,0)};t.mips[0].bytes[0]=255;
 assert(alpha_nearest(t,.1,.1)==1);t.mips[0].bytes[2]=1;assert(alpha_nearest(t,.1,.1)==0);
 bool rejected=false;try{build_shadow_frame(tris,c3x_renderer::evaluate_environment(12,0),128,.1f);}catch(std::runtime_error const&){rejected=true;}assert(rejected);
 std::cout<<"PASS four-phase shadow direction, receiver intersection, permutation, extent and static-idle contract\n";
}
