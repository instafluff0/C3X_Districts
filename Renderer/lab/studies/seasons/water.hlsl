cbuffer Frame:register(b0){float4 Sun;float4 SunColorExposure;float4 Ambient;float4 View;float4 Detail;};
Texture2D Sand:register(t0);
SamplerState Wrap:register(s0);SamplerState Clamp:register(s1);
#include "seasonal_policy.hlsl"
#include "lab_scene.hlsl"
struct V {float3 position:POSITION;float4 world:TEXCOORD0;float3 normal:NORMAL;float2 uv:TEXCOORD1;};
struct P {float4 position:SV_POSITION;float3 world:TEXCOORD0;};
struct Output{float4 color:SV_Target0;float validity:SV_Target1;};
P VSMain(V i){P o;o.position=float4(i.position,1);o.world=i.world.xyz;return o;}
P VSFeature(V i){return VSMain(i);}
Output shade(P i){
    Output o;float4 surface=lab_surface(i.world.xy);float shore=surface.x,river=surface.y;
    float3 sand=Sand.Sample(Wrap,i.world.xy*.43+float2(.31,.17)).rgb;
    float3 normal=float3(0,0,1);float gloss=.04;
    lab_season_ground(sand,normal,gloss,normal,i.world,0);
    float shadow=lab_shadow(i.world,normal);
    sand*=Ambient.rgb*Ambient.a+season_key(SunColorExposure.rgb)*Sun.w*.79*shadow;
    float edge=saturate(-shore*2.5);
    float3 sea=lerp(float3(.035,.20,.29),float3(.015,.09,.19),edge);
    float wave=sin((i.world.x+i.world.y)*21+sin(i.world.y*7))*.5+
        sin((i.world.x-i.world.y)*31+cos(i.world.x*9))*.5;
    float sparkle=pow(saturate(.5+wave*.5),12);
    sea+=float3(.08,.13,.18)*sparkle*.34;
    float foam=(1-smoothstep(.018,.080,abs(shore)))*.32;
    sea=lerp(sea,float3(.47,.63,.69),foam);
    float3 channel=float3(.025,.15,.25)+float3(.03,.06,.07)*sparkle*.15;
    channel*=.60+.40*shadow;
    float river_mask=(1-smoothstep(4.9,7.4,river))*smoothstep(-.08,.11,shore);
    float land=smoothstep(-.005,.028,shore);
    float3 color=lerp(sea,sand,land);
    color=lerp(color,channel,river_mask);
    o.color=float4(color,1);o.validity=1;return o;
}
Output PSMain(P i){return shade(i);} Output PSFeature(P i){return shade(i);}
