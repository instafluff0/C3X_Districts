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
    if(Probe.x==8){
        float a=season_drift(p),x=season_drift(px),y=season_drift(py);
        o.color=float4(a,x,y,abs(ddx(a)-ddx(x))+abs(ddy(a)-ddy(y)));return o;
    }
    if(Probe.x==9 || Probe.x==10){
        float3 a=float3(.17,.26,.055),x=a,y=a;
        float3 n=float3(0,0,1),nx=n,ny=n;float g=.11,gx=g,gy=g;
        if(Probe.x==9){
            season_foliage(a,n,g,n,float3(p,.2),p*.17,.42);
            season_foliage(x,nx,gx,nx,float3(px,.2),p*.17,.42);
            season_foliage(y,ny,gy,ny,float3(py,.2),p*.17,.42);
        }else{season_leaf_litter(a,float3(p,.2));season_leaf_litter(x,float3(px,.2));season_leaf_litter(y,float3(py,.2));}
        o.color=float4(season_luma(a),season_luma(x),season_luma(y),length(a-x)+length(a-y));return o;
    }
    if(Probe.x==14){
        float3 a=float3(.17,.26,.055),x=a,y=a,n=float3(0,0,1),nx=n,ny=n;float g=.11,gx=g,gy=g;
        season_ground(a,n,g,float3(0,0,1),float3(p,.2),float4(1,0,0,0),0,0,.5);
        season_ground(x,nx,gx,float3(0,0,1),float3(px,.2),float4(1,0,0,0),0,0,.5);
        season_ground(y,ny,gy,float3(0,0,1),float3(py,.2),float4(1,0,0,0),0,0,.5);
        o.color=float4(season_luma(a),season_luma(x),season_luma(y),length(a-x)+length(a-y)+length(n-nx)+length(n-ny));return o;
    }
    if(Probe.x==11 || Probe.x==12){
        float3 a=float3(.17,.26,.055),n=normalize(float3(.11,-.13,1));float g=.11;
        season_foliage(a,n,g,n,float3(p,.2),p*.17,.42,Probe.x==11?0:1);
        o.color=Probe.x==11?float4(a,g):float4(n,g);return o;
    }
    float3 albedo=Probe.x==4?float3(.21,.11,.045):float3(.17,.26,.055);
    float3 normal=normalize(float3(.11,-.13,1)),geometric=normalize(float3(.11,-.13,1));float gloss=.11;
    if(Probe.x==3 || Probe.x==4)season_foliage(albedo,normal,gloss,geometric,float3(p,.2),p*.17);
    else{float cover=season_ground(albedo,normal,gloss,geometric,float3(p,.2),
        Probe.x==5?float4(0,0,1,0):float4(1,0,0,0),0,Probe.x==6?1:0,.5);
        if(Probe.x==7){o.color=float4(normal,cover);return o;}}
    o.color=float4(albedo,gloss);return o;
}
