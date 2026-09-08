#pragma once
namespace c3x_renderer {
inline char const* unit_material_shader(){return R"C3XUNIT(
Texture2D<float4> base : register(t0);
Texture2D<float> shadow_map : register(t1);
Texture2D<float4> ambient_occlusion : register(t2);
Texture2D<float4> gloss_texture : register(t3);
Texture2D<float4> emissive_texture : register(t4);
SamplerState sample_base : register(s0);SamplerState sample_emission : register(s1);
cbuffer Material : register(b0) {float4 tint,owner,sun,sun_color,moon,moon_color,ambient,channels;};
cbuffer BeautyFrame : register(b1) {float4 Sun,SunColorExposure,Ambient,View,Quality;};
struct Input {float3 p:POSITION;float3 n:NORMAL;float2 uv:TEXCOORD0;float3 shadow:TEXCOORD1;};
struct Output {float4 p:SV_Position;float3 n:NORMAL;float2 uv:TEXCOORD0;float3 shadow:TEXCOORD1;};
Output VS(Input i){Output o;o.p=float4(i.p,1);o.n=i.n;o.uv=i.uv;o.shadow=i.shadow;return o;}
float ggx(float3 n, float3 light, float3 view, float roughness, float f0) {
    float3 halfway = normalize(light + view);
    float ndl = saturate(dot(n, light));
    float ndv = saturate(dot(n, view));
    float ndh = saturate(dot(n, halfway));
    float vdh = saturate(dot(view, halfway));
    float alpha = max(0.055, roughness * roughness);
    float a2 = alpha * alpha;
    float denominator = ndh * ndh * (a2 - 1) + 1;
    float distribution = a2 / max(3.14159265 * denominator * denominator, 0.0001);
    float k = (roughness + 1) * (roughness + 1) * 0.125;
    float geometry_v = ndv / max(ndv * (1 - k) + k, 0.0001);
    float geometry_l = ndl / max(ndl * (1 - k) + k, 0.0001);
    float fresnel = f0 + (1 - f0) * pow(1 - vdh, 5);
    return distribution * geometry_v * geometry_l * fresnel /
        max(4 * ndv * ndl, 0.0001) * ndl;
}


float3 beauty_unit_response(float3 albedo,float3 normal,float ao,float gloss,float3 emission,float shadow) {
 float kind=2;float3 light_direction=Sun.xyz;float diffuse=saturate(dot(normal,light_direction));
 // Preserve the source's optional-emission gate without role-based dispatch.
 struct Metadata {float2 secondary;};Metadata input;input.secondary=float2(0,channels.z);
    float sky = saturate(normal.z * 0.5 + 0.5);
    // Foliage needs readable key/fill separation at map scale; preserve a soft
    // skylight, but do not let it erase the canopy's shaded faces.
    float foliage_fill = kind > 0.5 && kind < 1.5 ? 0.72 : 1.0;
    float3 ambient = Ambient.rgb * Ambient.a * lerp(0.48, 1.0, sky) * ao * foliage_fill;
    float3 radiance = albedo * (ambient + SunColorExposure.rgb * Sun.w *
                                (0.035 + 0.965 * diffuse) * shadow);
    float roughness = lerp(0.91, 0.31, saturate(gloss));
    if (kind > 0.5 && kind < 1.5) roughness = max(roughness, 0.72);
    float specular_scale = kind > 0.5 && kind < 1.5 ? 0.10 : 0.52;
    radiance += SunColorExposure.rgb * Sun.w * specular_scale *
                ggx(normal, light_direction, normalize(View.xyz), roughness,
                    kind > 0.5 && kind < 1.5 ? 0.020 : 0.045) * shadow;
    if (!(kind > 0.5 && kind < 1.5) && input.secondary.y > 0.5)
        radiance += emission * 0.035;
    float rim = pow(1 - saturate(dot(normal, normalize(View.xyz))), 3);
    radiance += Ambient.rgb * rim * 0.08;

 return max(radiance,0);
}
float4 PS(Output i):SV_Target {
 clip(i.shadow.x);
 float4 b=base.Sample(sample_base,i.uv);if(ambient.w>.5)clip(b.a-.5);
 float3 albedo=b.rgb*tint.rgb;
 float mask=tint.w<.5?0:(tint.w<1.5?smoothstep(.06,.94,1-b.a):1);
 // Selected source modulation with Civ III's authoritative display color.
 // Multiplication retains atlas detail instead of replacing it by a luma ramp.
 albedo=lerp(albedo,albedo*(.45+owner.rgb*1.10),mask*owner.w);
 float3 n=normalize(i.n);int2 cell=int2(floor(i.shadow.yz*128));float occluded=0;
 [unroll]for(int oy=-1;oy<=1;++oy)[unroll]for(int ox=-1;ox<=1;++ox) {
  int2 q=cell+int2(ox,oy);
  if(all(q>=0) && all(q<128))occluded+=(shadow_map.Load(int3(q,0))>i.shadow.x+.006)?1.0/9:0;
 }
 float ao=channels.x>.5?lerp(.48,1,ambient_occlusion.Sample(sample_base,i.uv).r):1;
 float gloss=channels.y>.5?gloss_texture.Sample(sample_base,i.uv).r:.08;
 float3 emission=channels.z>.5?emissive_texture.Sample(sample_emission,i.uv).rgb:0;
 return float4(beauty_unit_response(albedo,n,ao,gloss,emission,1-.78*occluded),1);
}
)C3XUNIT";}
}
