// Q7 source city materials in Q6's shared scene-linear and world-shadow graph.
#define Q6_WORLD_SHADOWS 1
#define PSFeature Q7UnusedFrozenFeature
#include "../lighting/generated/scene_linear_v1.hlsl"
#undef PSFeature
Q6SceneOutput PSFeature(FeaturePixelInput p) {
    float3 n=normalize(p.geometry_normal);
    float visibility=q6_world_visibility(shallow_bed_texture,p.q6_world,n,false);
    float3 base=city_base_texture_0.Sample(material_sampler,p.uv).rgb;
    float3 emissive=resource_base_texture_0.Sample(material_sampler,p.uv).rgb;
    float3 ambient=environment_ambient_color*.46;
    float3 direct=environment_sun_color*environment_sun_intensity*(.16+.84*saturate(dot(n,environment_sun_direction)))+
        environment_moon_color*environment_moon_intensity*(.20+.80*saturate(dot(n,environment_moon_direction)))*.82;
#ifdef Q7_EMISSIVE_ONLY
    base=0;
#endif
    // Current registered city source materials declare opaque; cutout families
    // require a separate source alpha contract before they enter this provider.
    return q6_scene_output(float4(base*(ambient+direct*visibility)+
        emissive*environment_night_activation*environment_emissive_scale*1.45,1));
}
