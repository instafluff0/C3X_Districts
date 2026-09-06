#pragma once
// Q6 Lab v2 shared world-triangle visibility field. CPU preparation only;
// the existing Q0 GPU backend owns draw/depth/attachment execution.
#include "../../contracts/packet_v1.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
namespace q6 {
using F3=std::array<float,3>;
inline F3 norm(F3 v){float n=sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);for(auto& x:v)x/=n;return v;}
inline float dot(F3 a,F3 b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline F3 cross(F3 a,F3 b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}

using WorldTriangle=std::array<F3,3>;
// U,V,L are a right-handed orthonormal light basis, oriented toward the light.
// All supplied receiver/caster world coordinates must fit the declared span.
// Cutout geometry requires the coverage callback; opaque_shadow_field is opaque only.
template<class Coverage>
labv2::Texture raster_shadow_field(std::vector<WorldTriangle> const& triangles,
                                  F3 U,F3 V,F3 L,int resolution,float span,Coverage coverage) {
 if(resolution<64 || resolution>4096 || !std::isfinite(span) || span<=0)
   throw std::runtime_error("invalid Q6 shadow field extent");
 for(auto const& tri:triangles)for(auto const& w:tri) {
   for(float value:w)if(!std::isfinite(value))throw std::runtime_error("nonfinite shadow geometry");
   if(std::abs(dot(w,U))>=span*.5f || std::abs(dot(w,V))>=span*.5f || std::abs(dot(w,L))>=span*.5f)
     throw std::runtime_error("Q6 shadow extent excludes caster; expand extent including receiver halo");
 }
 const int S=resolution;
 std::vector<float> depth(S*S,0);

 // Rasterize actual world triangles to light depth. Cutout callbacks interpolate
 // source UVs with these barycentric weights before accepting a depth sample.
 for(size_t triangle_index=0;triangle_index<triangles.size();triangle_index++){
   auto const& triangle=triangles[triangle_index];
   float q[3][3];
   for(int j=0;j<3;j++){F3 w=triangle[j];q[j][0]=(dot(w,U)/span+.5f)*S;q[j][1]=(dot(w,V)/span+.5f)*S;q[j][2]=dot(w,L)/span+.5f;}
   float denom=(q[1][1]-q[2][1])*(q[0][0]-q[2][0])+(q[2][0]-q[1][0])*(q[0][1]-q[2][1]);if(fabs(denom)<1e-8)continue;
   int x0=std::max(0,int(floor(std::min(q[0][0],std::min(q[1][0],q[2][0]))))),x1=std::min(S-1,int(ceil(std::max(q[0][0],std::max(q[1][0],q[2][0])))));
   int y0=std::max(0,int(floor(std::min(q[0][1],std::min(q[1][1],q[2][1]))))),y1=std::min(S-1,int(ceil(std::max(q[0][1],std::max(q[1][1],q[2][1])))));
   for(int y=y0;y<=y1;y++)for(int x=x0;x<=x1;x++){
     float a=((q[1][1]-q[2][1])*(x+.5f-q[2][0])+(q[2][0]-q[1][0])*(y+.5f-q[2][1]))/denom;
     float b=((q[2][1]-q[0][1])*(x+.5f-q[2][0])+(q[0][0]-q[2][0])*(y+.5f-q[2][1]))/denom;
     float c=1-a-b;if(a>=0&&b>=0&&c>=0&&coverage(triangle_index,a,b,c))depth[y*S+x]=std::max(depth[y*S+x],a*q[0][2]+b*q[1][2]+c*q[2][2]);
   }
 }
 labv2::Texture shadow;shadow.width=S;shadow.height=S;shadow.format=11;
 labv2::Mip mip;mip.pitch=S*8;mip.bytes.resize(S*S*8);
 for(int i=0;i<S*S;i++){uint16_t z=uint16_t(std::clamp(depth[i],0.f,1.f)*65535+.5f);for(int c=0;c<4;c++)memcpy(mip.bytes.data()+i*8+c*2,&z,2);}
 shadow.mips.push_back(std::move(mip));return shadow;

}
inline labv2::Texture opaque_shadow_field(std::vector<WorldTriangle> const& triangles,
                                  F3 U,F3 V,F3 L,int resolution=1024,float span=6) {
 return raster_shadow_field(triangles,U,V,L,resolution,span,[](size_t,float,float,float){return true;});
}
struct ShadowFrame {
 F3 U,V,L; float span; int resolution; labv2::Texture texture;
};
// EnvironmentState-compatible input: captured shared environment, not local time.
template<class Environment>
ShadowFrame build_shadow_frame(std::vector<WorldTriangle> const& triangles,
                               Environment const& e,int resolution=1024,float span=6) {
 F3 combined={e.sun_direction[0]*e.sun_intensity+e.moon_direction[0]*e.moon_intensity,
              e.sun_direction[1]*e.sun_intensity+e.moon_direction[1]*e.moon_intensity,
              e.sun_direction[2]*e.sun_intensity+e.moon_direction[2]*e.moon_intensity};
 float h=std::hypot(combined[0],combined[1]);
 F3 L=h>1e-6f?norm({combined[0]/h,combined[1]/h,1.35f}):norm({-1,0,1.35f});
 F3 U=norm(cross({0,0,1},L)),V=cross(L,U);
 return {U,V,L,span,resolution,opaque_shadow_field(triangles,U,V,L,resolution,span)};
}
} // namespace q6
