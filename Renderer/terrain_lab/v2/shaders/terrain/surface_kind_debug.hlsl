// Diagnostic proxy: identify ownership of pale straight-edged surfaces.
#define PSMain q2_frozen_pixel
#include "../common/frozen_l21.hlsl"
#undef PSMain
float4 PSMain(PixelInput input):SV_TARGET {
 float4 source=q2_frozen_pixel(input);
 float a=source.a;clip(a-.001);
 float k=input.surface_kind;
 if(k<.25)return float4(.1,.6,.1,a); // base land
 if(k<.75)return float4(.4,.2,.6,a); // flat underlay
 if(k<1.5)return float4(.6,.5,.2,a); // land body
 if(k<2.5)return float4(1,.6,0,a); // beach
 if(k<3.5)return float4(1,0,0,a); // cliff
 if(k<5.5)return float4(0,1,1,a); // beds
 return float4(0,.2,1,a); // other water/river/specialized
}
