SamplerState Wrap:register(s0);SamplerState Clamp:register(s1);
#include "seasonal_policy.hlsl"
cbuffer ProbeState:register(b6){float4 Probe;};
struct V{float3 position:POSITION;float4 world:TEXCOORD0;float3 normal:NORMAL;float2 uv:TEXCOORD1;};
struct P{float4 position:SV_POSITION;float2 world:TEXCOORD0;};
struct O{float4 color:SV_Target0;float validity:SV_Target1;};
P VSMain(V i){P o;o.position=float4(i.position,1);o.world=float2(4.6,2.3)+(i.position.xy+1)*.7;return o;}
O PSMain(P i){
    O o;o.validity=1;float2 p=i.world;
    float2 px=p+Season.z*.5,py=p+float2(Season.w*.5,-Season.w*.5);
    if(Probe.x<.5){
        float a=season_noise(p,.71,983u),x=season_noise(px,.71,983u),y=season_noise(py,.71,983u);
        o.color=float4(a,x,y,abs(ddx(a)-ddx(x))+abs(ddy(a)-ddy(y)));return o;
    }
    if(Probe.x<1.5){float3 c,cx,cy;
        float a=season_flowers(p,1,c),x=season_flowers(px,1,cx),y=season_flowers(py,1,cy);
        o.color=float4(a,x,y,length(c-cx)+length(c-cy));return o;
    }
    float3 albedo=Probe.x==4?float3(.21,.11,.045):float3(.17,.26,.055);
    float3 normal=normalize(float3(.11,-.13,1)),geometric=normalize(float3(.11,-.13,1));float gloss=.11;
    if(Probe.x==3 || Probe.x==4)season_foliage(albedo,normal,gloss,geometric,float3(p,.2),p*.17);
    else{float cover=season_ground(albedo,normal,gloss,geometric,float3(p,.2),
        Probe.x==5?float4(0,0,1,0):float4(1,0,0,0),0,Probe.x==6?1:0,.5);
        if(Probe.x==7){o.color=float4(normal,cover);return o;}}
    o.color=float4(albedo,gloss);return o;
}
