// Opt-in Lab HDR lens response, before the shared tone curve. Source assets and
// material masks stay intact. Reconstruction/alpha/validity remain Q1-owned.
// Threshold is above the measured diffuse scene range in the fixed city probes.
// This is optical glow, not a substitute for local light transport.
#include "../sampling/reconstruction_v1.hlsl"
Texture2D<float4> Scene:register(t0);
RWTexture2D<float4> Output:register(u1);
Texture2D<float> Validity:register(t3);
RWTexture2D<float> OutputValidity:register(u4);
cbuffer PostSettings:register(b2) {uint2 input_size;uint2 output_size;int4 valid_rect;};
#ifndef Q8_GLOW_GAIN
#define Q8_GLOW_GAIN 1.8
#endif
float3 q8_highlight(int2 p) {
 if(any(p<0)||any(p>=int2(input_size)))return 0;
 float4 c=Scene.Load(int3(p,0));
 if(Validity.Load(int3(p,0))<=0||c.a<=0)return 0;
 float peak=max(c.r,max(c.g,c.b));
 return c.rgb*(max(peak-1.0,0)/max(peak,0.0001));
}
[numthreads(8,8,1)]
void CSPost(uint3 id:SV_DispatchThreadID) {
 if(any(id.xy>=output_size))return;
 Q1Reconstructed r=q1_reconstruct_box(Scene,Validity,input_size,output_size,valid_rect,int2(id.xy));
 float3 glow=0;float total=0;
 int2 at=int2((float2(id.xy)+0.5)*float2(input_size)/float2(output_size));
 // Two compact optical scales; fixed internal-pixel radii scale with the scene
 // through the established gameplay reconstruction, including zoom 2.
 for(int y=-8;y<=8;y++)for(int x=-8;x<=8;x++) {
  float d=float(x*x+y*y);
  float w=exp(-d/5.0)+0.12*exp(-d/28.0);
  glow+=q8_highlight(at+int2(x,y))*w;total+=w;
 }
 r.rgba.rgb+=glow/max(total,0.0001)*Q8_GLOW_GAIN*r.rgba.a;
 Output[id.xy]=r.rgba;OutputValidity[id.xy]=r.coverage;
}
