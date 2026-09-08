Texture2D<float4> bridge0:register(t10);
Texture2D<float4> bridge1:register(t11);
Texture2D<float4> bridge2:register(t12);
Texture2D<float4> bridge3:register(t13);
Texture2D<float4> bridge4:register(t14);
Texture2D<float4> bridge5:register(t15);
Texture2D<float4> bridge6:register(t16);
Texture2D<float4> bridge7:register(t17);
Texture2D<float4> route0:register(t0);
Texture2D<float4> route1:register(t1);
Texture2D<float4> route2:register(t2);
Texture2D<float4> route3:register(t3);
Texture2D<float4> route4:register(t4);
Texture2D<float4> route5:register(t5);
Texture2D<float4> route6:register(t6);
Texture2D<float4> route7:register(t7);
Texture2D<float4> route8:register(t8);
Texture2D<float4> route9:register(t9);
SamplerState route_sampler:register(s0);
#ifdef Q5_WORLD_RECEIVER
Texture2D q5_shadow_field:register(t25);
#include "../lighting/frame_shadow_v1.hlsl"
#endif
// Opaque diagnostic material branch, using the shared environment evaluator.
// Transfer is local only while Q0's scene-linear attachment is unavailable.
cbuffer Frame : register(b0) {
 float4 viewport; float4 camera;
 float4 sun; float4 sunlight; float4 moon; float4 moonlight; float4 ambient;
};
struct V {float3 position:POSITION;float3 normal:TEXCOORD0;float3 color:TEXCOORD1;float4 route:TEXCOORD2;
#ifdef Q5_WORLD_RECEIVER
 float4 world:TEXCOORD3;
#endif
};
struct P {float4 position:SV_POSITION;float3 normal:TEXCOORD0;float3 color:TEXCOORD1;float4 route:TEXCOORD2;
#ifdef Q5_WORLD_RECEIVER
 float4 world:TEXCOORD3;
#endif
};
P VSMain(V v) {
 P p;p.position=float4(v.position,1);
 p.normal=v.normal;p.color=v.color;p.route=v.route;
#ifdef Q5_WORLD_RECEIVER
 p.world=v.world;
#endif
 return p;
}
P VSFeature(V v){return VSMain(v);}
float3 srgb(float3 c){return lerp(12.92*c,1.055*pow(max(c,0),1/2.4)-.055,step(.0031308,c));}
float4 road_sample(float2 uv,float stage,float pillaged){
 float4 base=lerp(route0.Sample(route_sampler,uv),route1.Sample(route_sampler,uv),pillaged);
 float4 detail=base;
 if(stage>.5&&stage<1.5)detail=lerp(route2.Sample(route_sampler,uv),route3.Sample(route_sampler,uv),pillaged);
 if(stage>1.5&&stage<2.5)detail=lerp(route4.Sample(route_sampler,uv),route5.Sample(route_sampler,uv),pillaged);
 if(stage>2.5)detail=lerp(route6.Sample(route_sampler,uv),route7.Sample(route_sampler,uv),pillaged);
 return float4(lerp(base.rgb,detail.rgb,detail.a*step(.5,stage)),max(base.a,detail.a));
}
float4 authored_route(float4 data){
 if(data.z>9.5&&data.z<10.5)return bridge0.Sample(route_sampler,data.xy);
 if(data.z>10.5&&data.z<11.5)return bridge1.Sample(route_sampler,data.xy);
 if(data.z>11.5&&data.z<12.5)return bridge2.Sample(route_sampler,data.xy);
 if(data.z>12.5&&data.z<13.5)return bridge3.Sample(route_sampler,data.xy);
 if(data.z>13.5&&data.z<14.5)return bridge4.Sample(route_sampler,data.xy);
 if(data.z>14.5&&data.z<15.5)return bridge5.Sample(route_sampler,data.xy);
 if(data.z>15.5&&data.z<16.5)return bridge6.Sample(route_sampler,data.xy);
 if(data.z>16.5&&data.z<17.5)return bridge7.Sample(route_sampler,data.xy);
 float along=frac(data.x),across=data.y,stage=data.z,pillaged=data.w;
 float2 uv=float2(along,lerp(.90606654,.99021526,(across+1)*.5));
 float4 col=road_sample(saturate(uv),min(stage,3),pillaged);
 float4 center=road_sample(float2(along,(.90606654+.99021526)*.5),min(stage,3),pillaged);
 if(stage>3.5){
  // Atlas bounds describe rectangles, not a diagonal centerline. Both long
  // steel strips occupy the upper rectangle; sleepers/ballast the lower one.
  float2 a=float2(along,lerp(.75294118,1.,(across+1)*.5));
  float2 b=float2(along,lerp(.25098039,.49803922,(across+1)*.5));
  float4 sleepers=lerp(route8.Sample(route_sampler,saturate(a)),route9.Sample(route_sampler,saturate(a)),pillaged);
  float4 steel=lerp(route8.Sample(route_sampler,saturate(b)),route9.Sample(route_sampler,saturate(b)),pillaged);
  return float4(lerp(sleepers.rgb,steel.rgb,steel.a),max(sleepers.a,steel.a));
 }
 float guard=(1-smoothstep(.46,.70,abs(across)))*.88;
 col.rgb=lerp(col.rgb,center.rgb,guard*(1-smoothstep(.02,.2,col.a)));
 col.a=max(col.a,guard);return col;
}
float4 shade_route(P p) {
 float4 material=p.route.z<-.5?float4(p.color,1):authored_route(p.route);
 if(p.route.z>=0&&p.route.z<5)material.a*=p.color.r;
 clip(material.a-.025);
 float3 n=normalize(p.normal);
 float visibility=1;
#ifdef Q5_WORLD_RECEIVER
 visibility=q6_world_visibility(q5_shadow_field,p.world,n,false);
#endif
 float3 illumination=ambient.rgb+visibility*(sunlight.rgb*sun.w*max(0,dot(n,sun.xyz))+moonlight.rgb*moon.w*max(0,dot(n,moon.xyz)));
#ifdef Q5_SCENE_LINEAR
 return float4(material.rgb*illumination*material.a,material.a);
#else
 float3 color=material.rgb*illumination*sunlight.w;
 color=color/(1+max(color.r,max(color.g,color.b)));
 return float4(srgb(color),material.a);
#endif
}
#ifdef Q5_SCENE_LINEAR
struct RouteTargets {float4 color:SV_Target0;float validity:SV_Target1;};
RouteTargets PSMain(P p){RouteTargets o;o.color=shade_route(p);o.validity=1;return o;}
RouteTargets PSFeature(P p){return PSMain(p);}
#else
float4 PSMain(P p):SV_TARGET{return shade_route(p);}
float4 PSFeature(P p):SV_TARGET{return shade_route(p);}
#endif
