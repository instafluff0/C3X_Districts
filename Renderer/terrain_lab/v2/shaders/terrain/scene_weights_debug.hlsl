// Diagnostic proxy: identify ownership of pale straight-edged surfaces.
#define PSMain q2_frozen_pixel
#include "../common/frozen_l21.hlsl"
#undef PSMain
float4 PSMain(PixelInput input):SV_TARGET {
 float4 source=q2_frozen_pixel(input);
 float a=source.a;clip(a-.001);
 float4 w=input.material_weights;
 float3 color=w.x*float3(.1,.6,.1)+w.y*float3(.6,.35,.1)+w.z*float3(1,1,0)+w.w*float3(1,0,0)+input.material_tundra;
 return float4(color,a);
}
