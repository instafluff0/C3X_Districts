#include "response_v1.hlsl"
#include "shadow_visibility_v1.hlsl"
#ifndef Q6_SHADOWS
#define Q6_SHADOWS 1
#endif
#ifndef Q6_CONTACT
#define Q6_CONTACT 1
#endif
#ifndef Q6_EMISSIVE_ONLY
#define Q6_EMISSIVE_ONLY 0
#endif
#ifndef Q6_LEGACY
#define Q6_LEGACY 0
#endif
Texture2D Base0:register(t0);Texture2D Emission0:register(t8);
Texture2D Base1:register(t1);Texture2D Emission1:register(t9);
Texture2D Base2:register(t2);Texture2D Emission2:register(t10);
Texture2D Base3:register(t3);Texture2D Emission3:register(t11);
Texture2D Base4:register(t4);Texture2D Emission4:register(t12);
Texture2D Base5:register(t5);Texture2D Emission5:register(t13);
Texture2D Base6:register(t6);Texture2D Emission6:register(t14);
Texture2D Base7:register(t7);Texture2D Emission7:register(t15);
Texture2D Shadow:register(t16);SamplerState material_sampler:register(s0);
cbuffer Frame:register(b0){float4 Sun,SunColorExposure,Moon,MoonColorNight,AmbientEmission,ShadowU,ShadowV,ShadowL,Viewport;};
struct V {float3 position:POSITION;float2 uv:TEXCOORD0;float3 normal:NORMAL0;float3 world:TEXCOORD1;float material:TEXCOORD2;};
struct P {float4 position:SV_POSITION;float2 uv:TEXCOORD0;float3 normal:NORMAL0;float3 world:TEXCOORD1;float material:TEXCOORD2;};
P VSMain(V v){P p;p.position=float4(v.position,1);p.uv=v.uv;p.normal=v.normal;p.world=v.world;p.material=v.material;return p;}
P VSFeature(V v){return VSMain(v);}
float3 city_radiance(P p){
 float4 base_sample=float4(.24,.28,.19,1);float3 emissive=0;
 if(p.material<0.5){base_sample=Base0.Sample(material_sampler,p.uv);emissive=Emission0.Sample(material_sampler,p.uv).rgb;}
 else if(p.material<1.5){base_sample=Base1.Sample(material_sampler,p.uv);emissive=Emission1.Sample(material_sampler,p.uv).rgb;}
 else if(p.material<2.5){base_sample=Base2.Sample(material_sampler,p.uv);emissive=Emission2.Sample(material_sampler,p.uv).rgb;}
 else if(p.material<3.5){base_sample=Base3.Sample(material_sampler,p.uv);emissive=Emission3.Sample(material_sampler,p.uv).rgb;}
 else if(p.material<4.5){base_sample=Base4.Sample(material_sampler,p.uv);emissive=Emission4.Sample(material_sampler,p.uv).rgb;}
 else if(p.material<5.5){base_sample=Base5.Sample(material_sampler,p.uv);emissive=Emission5.Sample(material_sampler,p.uv).rgb;}
 else if(p.material<6.5){base_sample=Base6.Sample(material_sampler,p.uv);emissive=Emission6.Sample(material_sampler,p.uv).rgb;}
 else if(p.material<7.5){base_sample=Base7.Sample(material_sampler,p.uv);emissive=Emission7.Sample(material_sampler,p.uv).rgb;}
 clip(base_sample.a-.5);
 float3 base=base_sample.rgb;
 float3 n=normalize(p.normal);float shadow=q6_shadow_visibility(Shadow,p.world,n,ShadowU,ShadowV,ShadowL,Q6_SHADOWS,Q6_CONTACT);
 float3 ambient=AmbientEmission.rgb*.46;
 float3 direct=SunColorExposure.rgb*Sun.w*(.16+.84*saturate(dot(n,Sun.xyz)))+
 MoonColorNight.rgb*Moon.w*(.20+.80*saturate(dot(n,Moon.xyz)))*.82;
 float3 light=base*(ambient+direct*shadow);
 float3 radiance=(Q6_EMISSIVE_ONLY?0:light)+emissive*MoonColorNight.w*AmbientEmission.w*1.45;
 return radiance;
}
#if Q6_LINEAR
struct CityTargets {float4 color:SV_Target0;float validity:SV_Target1;};
CityTargets PSMain(P p){CityTargets o;o.color=float4(city_radiance(p),1);o.validity=1;return o;}
CityTargets PSFeature(P p){return PSMain(p);}
#else
float4 PSMain(P p):SV_TARGET {
 float3 radiance=city_radiance(p);
 float3 result=Q6_LEGACY?pow(saturate(radiance*SunColorExposure.w/(1+radiance*SunColorExposure.w*.30)),1.0/2.2):q6_srgb_encode(q6_display_linear(radiance,SunColorExposure.w));
 return float4(result,1);
}
float4 PSFeature(P p):SV_TARGET{return PSMain(p);}
#endif
