cbuffer Shadow:register(b0){float4 U;float4 V;float4 L;float4 Domain;float4 Depth;};
cbuffer Cutout:register(b1){float4 Flags;};
Texture2D Opacity:register(t0);SamplerState Wrap:register(s0);SamplerState Clamp:register(s1);
struct I{float3 world:POSITION;float2 uv:TEXCOORD0;float coverage:TEXCOORD1;};
struct P{float4 position:SV_POSITION;float2 uv:TEXCOORD0;float light_depth:TEXCOORD1;float coverage:TEXCOORD2;};
P VSMain(I i){P o;float2 uv=(float2(dot(i.world,U.xyz),dot(i.world,V.xyz))-Domain.xy)*Domain.zw;
    o.position=float4(uv.x*2-1,1-uv.y*2,(Depth.y-dot(i.world,L.xyz))*Depth.z,1);
    o.uv=i.uv;o.light_depth=dot(i.world,L.xyz);o.coverage=i.coverage;return o;}
P VSFeature(I i){return VSMain(i);}
float PSMain(P i):SV_Target{clip(i.coverage-.02);if(Flags.x>.5)clip(Opacity.Sample(Wrap,i.uv).r-.5);return i.light_depth;}
float PSFeature(P i):SV_Target{return PSMain(i);}
