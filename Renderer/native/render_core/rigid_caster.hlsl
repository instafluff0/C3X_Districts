#include "rigid_instance_geometry.hlsl"
cbuffer Caster:register(b0){float4 U;float4 V;float4 L;float4 page;float4 offset;};
struct RigidPixel {float4 position:SV_POSITION;float2 uv:TEXCOORD0;nointerpolation float material:TEXCOORD1;
 float depth:TEXCOORD2;float coverage:TEXCOORD3;float boundary:TEXCOORD4;float4 volcano:TEXCOORD5;};
RigidPixel VSSharedCaster(RigidInput i){
 float3 world=rigid_point(i).world+i.placement_view.xyz;
 RigidPixel o;o.position=float4((dot(world,U.xyz)/6-page.x)*2-1,1-(dot(world,V.xyz)/6-page.y)*2,.5,1);
 o.uv=i.source_uv;o.material=i.placement_view.w;o.depth=dot(world,L.xyz);o.coverage=1;o.boundary=o.material;o.volcano=0;return o;
}
