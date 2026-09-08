// Q7 material bindings; shared Q6 visibility and response own illumination.
#define PSFeature Q7FrozenPSFeature
#include "../common/frozen_l21.hlsl"
#undef PSFeature
#include "../lighting/shadow_visibility_v1.hlsl"
#include "../lighting/response_v1.hlsl"
float4 PSFeature(FeaturePixelInput p):SV_TARGET {
    if(p.material_index>=0)return Q7FrozenPSFeature(p);
    // Inverse of Q7's pinned orthographic projection, including its linear
    // depth convention. This preserves the frozen FeatureVertex ABI while Q0
    // prepares a semantic world-position interface. Never applies to v1 draws.
    float2 viewport=float2(roads_enabled,roads_only);
    float sy=p.position.y-viewport.y*.5;
    float dz=.94-p.position.z-viewport.y*.5*.75/viewport.y;
    float z=(dz-sy*.75/viewport.y)/(80.9543*.75/viewport.y+.20732);
    float sum=(sy+80.9543*z)/32;
    float difference=(p.position.x-viewport.x*.5)/64;
    float3 world=float3((sum+difference)*.5,(sum-difference)*.5,z);
    float3 n=normalize(p.geometry_normal);
    float shadow=q6_shadow_visibility(road_bridge_base_texture_7,world,n,
        float4(height_texel,normal_strength,exposure),
        float4(lab_mode,beauty_relief_enabled,beauty_water_enabled,shoreline_integrated),
        float4(promotion_tile_layout,scene_width,scene_height,0),dune_only>.5,l10_layout>.5);
    float4 source=city_base_texture_0.Sample(material_sampler,p.uv);
    if(p.material_index< -99)clip(source.a-.08);
    float3 base=source.rgb;
    float3 emissive=resource_base_texture_0.Sample(material_sampler,p.uv).rgb;
    float3 ambient=environment_ambient_color*.46;
    float3 direct=environment_sun_color*environment_sun_intensity*(.16+.84*saturate(dot(n,environment_sun_direction)))+
        environment_moon_color*environment_moon_intensity*(.20+.80*saturate(dot(n,environment_moon_direction)))*.82;
    float3 radiance=(biq_layout>.5?0:base*(ambient+direct*shadow))+
        emissive*environment_night_activation*environment_emissive_scale*1.45;
    if(marsh_enabled>1.5)return float4((1-p.position.z).xxx,1);
    if(marsh_enabled>.5)return float4(frac(abs(p.material_index)*float3(.317,.571,.713)),1);
    return float4(q6_srgb_encode(q6_display_linear(radiance,environment_exposure)),1);
}
