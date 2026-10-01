SamplerState Wrap:register(s0);SamplerState Clamp:register(s1);
#include "seasonal_policy.hlsl"
#include "lab_scene.hlsl"
cbuffer Frame:register(b0){float4 Sun;float4 SunColorExposure;float4 Ambient;float4 View;float4 Detail;float4 Quality;};
Texture2D SnowPatchColor:register(t0);
Texture2D SnowPatchSlope:register(t1);
Texture2D SnowPatchGloss:register(t2);
struct I{float3 position:POSITION;float4 world:TEXCOORD0;float3 normal:NORMAL;float2 uv:TEXCOORD1;float4 material:TEXCOORD2;};
struct P{float4 position:SV_POSITION;float3 world:TEXCOORD0;float3 normal:TEXCOORD1;float2 uv:TEXCOORD2;};
struct O{float4 color:SV_Target0;float validity:SV_Target1;};
P VSMain(I i){P o;o.position=float4(i.position,1);o.world=i.world.xyz;o.normal=i.normal;o.uv=i.uv;return o;}
O PSMain(P i){
    clip(Season.y>.5 && Season.x==2 ? 1 : -1);
    float4 field=lab_surface(i.world.xy);
    clip(field.x-.012);
    float4 patch=SnowPatchColor.Sample(Clamp,i.uv);
    float4 b=lab_biomes(i.world.xy);
    float alpha=patch.a*lab_land_coverage(i.world.xy)*dot(b,float4(.76,.57,.26,.84));
    clip(alpha-.018);
    float3 geometric=normalize(i.normal);
    float2 packed=SnowPatchSlope.Sample(Clamp,i.uv).rg;
    // Retained Decal_Heightmap data: derive relief from centered R rather
    // than treating unresolved G as a signed tangent slope.
    float3 normal=normalize(geometric-season_height_gradient(i.world,geometric,(packed.r-.5)*patch.a)*.020);
    float grain=clamp(season_luma(patch.rgb)*2.5,.40,1.15);
    if(SeasonSnow.x>.5){
        float mean=season_luma(SnowPatchColor.SampleBias(Clamp,i.uv,3).rgb);
        grain=clamp(season_luma(patch.rgb)/max(.03,mean),.72,1.22);
    }
    float3 color=season_linear(float3(.955,.98,1))*grain;
    color*=season_snow_palette(b,field.z);
    float sky=saturate(normal.z*.5+.5);
    float shadow=lab_shadow(i.world,normal);
    float3 key=season_key(SunColorExposure.rgb)*Sun.w;
    float3 fill=season_ambient(Ambient.rgb)*Ambient.a*lerp(.52,1,sky);
    float diffuse=.10+.90*saturate(dot(normal,LabL.xyz));
    float3 radiance=color*(fill+key*diffuse*shadow);
    float gloss=SnowPatchGloss.Sample(Clamp,i.uv).r;
    float sparkle=pow(saturate(dot(normal,normalize(LabL.xyz+View.xyz))),48)*(.015+.04*gloss);
    radiance+=key*sparkle*shadow;
    O o;o.color=float4(radiance*alpha,alpha);o.validity=1;return o;
}
