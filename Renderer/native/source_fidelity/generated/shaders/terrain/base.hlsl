// Q2 source-material structure; Q6 output response on provisional display attachment.
#include "../lighting/response_v1.hlsl"
#include "material_response_v1.hlsl"
#ifndef Q2_DETAIL
#define Q2_DETAIL 1
#endif
#ifndef Q2_ISOLATION
#define Q2_ISOLATION 0
#endif
Texture2D<float4> material_0 : register(t0);
Texture2D<float4> material_1 : register(t1);
Texture2D<float4> material_2 : register(t2);
Texture2D<float4> material_3 : register(t3);
Texture2D<float4> material_4 : register(t4);
Texture2D<float4> material_5 : register(t5);
Texture2D<float4> material_6 : register(t6);
Texture2D<float4> material_7 : register(t7);
Texture2D<float4> material_8 : register(t8);
Texture2D<float4> material_9 : register(t9);
Texture2D<float4> material_10 : register(t10);
Texture2D<float4> material_11 : register(t11);
Texture2D<float4> material_12 : register(t12);
Texture2D<float4> material_13 : register(t13);
Texture2D<float4> material_14 : register(t14);
SamplerState material_sampler : register(s0);
cbuffer Frame : register(b0) {float4 sun,moon,sun_color,moon_color,ambient;};
struct V {float3 position:POSITION;float2 uv:TEXCOORD0;float4 weights:TEXCOORD1;float tundra:TEXCOORD2;};
struct P {float4 position:SV_POSITION;float2 uv:TEXCOORD0;float4 weights:TEXCOORD1;float tundra:TEXCOORD2;};
P VSMain(V v){P p;p.position=float4(v.position,1);p.uv=v.uv;p.weights=v.weights;p.tundra=v.tundra;return p;}
P VSFeature(V v){return VSMain(v);}
float4 PSMain(P p):SV_TARGET {
 float w[5]={p.weights.x,p.weights.y,p.weights.z,p.weights.w,p.tundra};
 float3 color=0;float h=0,spec=0,hx=0,hy=0;
 float step=.002;float3 normal=float3(0,0,1);
 {
  color+=material_0.Sample(material_sampler,p.uv).rgb*w[0];
  float baseheight=material_1.Sample(material_sampler,p.uv).r;
  float fine=material_1.Sample(material_sampler,p.uv*8.0).r;
  float middle=material_1.Sample(material_sampler,p.uv*3.0).r;
  h+=(baseheight+.22*middle+.07*fine)*w[0];
  spec+=material_2.Sample(material_sampler,p.uv).r*w[0];
  hx+=(material_1.Sample(material_sampler,p.uv+float2(step,0)).r-baseheight
      +.22*(material_1.Sample(material_sampler,(p.uv+float2(step,0))*3).r-middle)
      +.07*(material_1.Sample(material_sampler,(p.uv+float2(step,0))*8).r-fine))*w[0];
  hy+=(material_1.Sample(material_sampler,p.uv+float2(0,step)).r-baseheight
      +.22*(material_1.Sample(material_sampler,(p.uv+float2(0,step))*3).r-middle)
      +.07*(material_1.Sample(material_sampler,(p.uv+float2(0,step))*8).r-fine))*w[0];
  }
 {
  color+=material_3.Sample(material_sampler,p.uv).rgb*w[1];
  float baseheight=material_4.Sample(material_sampler,p.uv).r;
  float fine=material_4.Sample(material_sampler,p.uv*8.0).r;
  float middle=material_4.Sample(material_sampler,p.uv*3.0).r;
  h+=(baseheight+.22*middle+.07*fine)*w[1];
  spec+=material_5.Sample(material_sampler,p.uv).r*w[1];
  hx+=(material_4.Sample(material_sampler,p.uv+float2(step,0)).r-baseheight
      +.22*(material_4.Sample(material_sampler,(p.uv+float2(step,0))*3).r-middle)
      +.07*(material_4.Sample(material_sampler,(p.uv+float2(step,0))*8).r-fine))*w[1];
  hy+=(material_4.Sample(material_sampler,p.uv+float2(0,step)).r-baseheight
      +.22*(material_4.Sample(material_sampler,(p.uv+float2(0,step))*3).r-middle)
      +.07*(material_4.Sample(material_sampler,(p.uv+float2(0,step))*8).r-fine))*w[1];
  }
 {
  color+=material_6.Sample(material_sampler,p.uv).rgb*w[2];
  float baseheight=material_7.Sample(material_sampler,p.uv).r;
  float fine=material_7.Sample(material_sampler,p.uv*8.0).r;
  float middle=material_7.Sample(material_sampler,p.uv*3.0).r;
  h+=(baseheight+.22*middle+.07*fine)*w[2];
  spec+=material_8.Sample(material_sampler,p.uv).r*w[2];
  hx+=(material_7.Sample(material_sampler,p.uv+float2(step,0)).r-baseheight
      +.22*(material_7.Sample(material_sampler,(p.uv+float2(step,0))*3).r-middle)
      +.07*(material_7.Sample(material_sampler,(p.uv+float2(step,0))*8).r-fine))*w[2];
  hy+=(material_7.Sample(material_sampler,p.uv+float2(0,step)).r-baseheight
      +.22*(material_7.Sample(material_sampler,(p.uv+float2(0,step))*3).r-middle)
      +.07*(material_7.Sample(material_sampler,(p.uv+float2(0,step))*8).r-fine))*w[2];
  }
 {
  color+=material_9.Sample(material_sampler,p.uv).rgb*w[3];
  float baseheight=material_10.Sample(material_sampler,p.uv).r;
  float fine=material_10.Sample(material_sampler,p.uv*8.0).r;
  float middle=material_10.Sample(material_sampler,p.uv*3.0).r;
  h+=(baseheight+.22*middle+.07*fine)*w[3];
  spec+=material_11.Sample(material_sampler,p.uv).r*w[3];
  hx+=(material_10.Sample(material_sampler,p.uv+float2(step,0)).r-baseheight
      +.22*(material_10.Sample(material_sampler,(p.uv+float2(step,0))*3).r-middle)
      +.07*(material_10.Sample(material_sampler,(p.uv+float2(step,0))*8).r-fine))*w[3];
  hy+=(material_10.Sample(material_sampler,p.uv+float2(0,step)).r-baseheight
      +.22*(material_10.Sample(material_sampler,(p.uv+float2(0,step))*3).r-middle)
      +.07*(material_10.Sample(material_sampler,(p.uv+float2(0,step))*8).r-fine))*w[3];
  }
 {
  color+=material_12.Sample(material_sampler,p.uv).rgb*w[4];
  float baseheight=material_13.Sample(material_sampler,p.uv).r;
  float fine=material_13.Sample(material_sampler,p.uv*8.0).r;
  float middle=material_13.Sample(material_sampler,p.uv*3.0).r;
  h+=(baseheight+.22*middle+.07*fine)*w[4];
  spec+=material_14.Sample(material_sampler,p.uv).r*w[4];
  hx+=(material_13.Sample(material_sampler,p.uv+float2(step,0)).r-baseheight
      +.22*(material_13.Sample(material_sampler,(p.uv+float2(step,0))*3).r-middle)
      +.07*(material_13.Sample(material_sampler,(p.uv+float2(step,0))*8).r-fine))*w[4];
  hy+=(material_13.Sample(material_sampler,p.uv+float2(0,step)).r-baseheight
      +.22*(material_13.Sample(material_sampler,(p.uv+float2(0,step))*3).r-middle)
      +.07*(material_13.Sample(material_sampler,(p.uv+float2(0,step))*8).r-fine))*w[4];
  }
 Q2MaterialDetailV1 detail=q2_material_detail_v1(color,h,hx,hy,spec,Q2_DETAIL!=0);
 normal=detail.tangent_normal;color=detail.albedo;float roughness=detail.roughness;
 if(Q2_ISOLATION==1)return float4(p.weights.xyz+p.tundra*.8,1);
 if(Q2_ISOLATION==2)return float4(normal*.5+.5,1);
 if(Q2_ISOLATION==3)return float4(h.xxx,1);
 if(Q2_ISOLATION==4)return float4(roughness.xxx,1);
 if(Q2_ISOLATION==5)return float4(pow(saturate(color),1/2.2),1);
 float3 light=ambient.rgb+sun_color.rgb*saturate(dot(normal,sun.xyz))+moon_color.rgb*saturate(dot(normal,moon.xyz));
 float3 lit_color=color*light;
 // Restrained broad response only; no terrain glitter or invented analytic light.
 if(Q2_DETAIL)lit_color+=(1-roughness)*.015*pow(saturate(dot(reflect(-sun.xyz,normal),float3(0,0,1))),8)*sun_color.rgb;
 return float4(q6_srgb_encode(q6_display_linear(lit_color,ambient.a)),1);
}
float4 PSFeature(P p):SV_TARGET{return PSMain(p);}
