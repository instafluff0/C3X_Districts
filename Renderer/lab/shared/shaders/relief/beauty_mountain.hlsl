// One material and lighting implementation is compiled to both Metal and D3D11.
cbuffer Frame : register(b0) {
    float4 Sun;
    float4 SunColorExposure;
    float4 Ambient;
    float4 View;
    float4 Macro;
    float4 Quality;
};
Texture2D GrassColor : register(t0);
Texture2D GrassHeight : register(t1);
Texture2D GrassSpecular : register(t2);
Texture2D RockColor : register(t3);
Texture2D RockHeight : register(t4);
Texture2D RockSpecular : register(t5);
Texture2D TopColor : register(t6);
Texture2D TopHeight : register(t7);
Texture2D TopSpecular : register(t8);
Texture2D SnowColor : register(t9);
Texture2D SnowHeight : register(t10);
Texture2D SnowSpecular : register(t11);
Texture2D MacroHeight : register(t12);
#ifdef BEAUTY_TERRAIN_TRANSITIONS
Texture2D PlainsColor : register(t13);
Texture2D PlainsHeight : register(t14);
Texture2D PlainsSpecular : register(t15);
Texture2D TundraColor : register(t16);
Texture2D TundraHeight : register(t18);
Texture2D TundraSpecular : register(t19);
Texture2D DesertColor : register(t20);
Texture2D DesertHeight : register(t21);
Texture2D DesertSpecular : register(t22);
Texture2D AuthoredHillHeight : register(t23);
Texture2D GrassHillColor : register(t24);
Texture2D GrassHillHeight : register(t25);
Texture2D GrassHillSpecular : register(t26);
Texture2D PlainsHillColor : register(t27);
Texture2D PlainsHillHeight : register(t28);
Texture2D PlainsHillSpecular : register(t29);
#endif
#ifdef BEAUTY_COMPOSED_SHADOWS
Texture2D ShadowField : register(t17);
cbuffer ShadowFrame : register(b1) {
    float4 ShadowU;
    float4 ShadowV;
    float4 ShadowL;
    float4 ShadowOrigin;
    float4 ShadowFlags;
};
#include "../lighting/shadow_visibility_v1.hlsl"
#endif
SamplerState Wrap : register(s0);
SamplerState Clamp : register(s1);

struct V {
    float3 position : POSITION;
#ifdef BEAUTY_COMPOSED_SHADOWS
    float4 world : TEXCOORD0;
#else
    float3 world : TEXCOORD0;
#endif
    float3 normal : NORMAL;
    float2 uv : TEXCOORD1;
#ifdef BEAUTY_TERRAIN_TRANSITIONS
    float4 material : TEXCOORD2;
    float2 biome : TEXCOORD3;
    float coast_coverage : TEXCOORD4;
#else
    float3 material : TEXCOORD2;
#endif
};
struct P {
    float4 position : SV_POSITION;
    float3 world : TEXCOORD0;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD1;
#ifdef BEAUTY_TERRAIN_TRANSITIONS
    float4 material : TEXCOORD2;
    float2 biome : TEXCOORD3;
    float coast_coverage : TEXCOORD4;
    float base_relief : TEXCOORD5;
#else
    float3 material : TEXCOORD2;
#endif
};
struct Output { float4 color : SV_Target0; float validity : SV_Target1; };

P VSMain(V input) {
    P output;
    output.position = float4(input.position, 1);
    output.world = input.world.xyz;
    output.normal = input.normal;
    output.uv = input.uv;
    output.material = input.material;
#ifdef BEAUTY_TERRAIN_TRANSITIONS
    output.biome = input.biome;
    output.coast_coverage = input.coast_coverage;
    output.base_relief = max(0, input.world.w - 1);
#endif
    return output;
}
P VSFeature(V input) { return VSMain(input); }

float3 triplanar(Texture2D texture_map, float3 p, float3 n) {
    float3 weight = pow(abs(n), 5);
    weight /= max(dot(weight, 1), 0.00001);
    p *= Quality.y;
    return texture_map.Sample(Wrap, p.yz).rgb * weight.x +
           texture_map.Sample(Wrap, p.xz).rgb * weight.y +
           texture_map.Sample(Wrap, p.xy).rgb * weight.z;
}

float triplanar_scalar(Texture2D texture_map, float3 p, float3 n) {
    return triplanar(texture_map, p, n).r;
}

float macro_height_world(float2 world_xy) {
    float2 uv = world_xy / float2(3.20, 2.72) + 0.5;
    if (any(uv < 0) || any(uv > 1)) return 0;
    float raw = MacroHeight.SampleLevel(Clamp, uv, 0).r;
    return saturate((raw - Macro.x) / max(Macro.y - Macro.x, 0.0001)) * Macro.z;
}

float shadow_ray(float3 world, float lateral) {
    float2 toward_light = normalize(-Sun.xy);
    float2 perpendicular = float2(-toward_light.y, toward_light.x);
    world.xy += perpendicular * lateral;
    float visibility = 1;
    [unroll] for (int index = 1; index <= 20; ++index) {
        float travel = index * 0.075;
        float2 xy = world.xy + toward_light * travel;
        float ray = world.z + travel * Sun.z / max(length(Sun.xy), 0.05);
        float blocker = macro_height_world(xy);
        visibility = min(visibility, 1 - saturate((blocker - ray - 0.018) * 24));
    }
    return visibility;
}

float horizon_visibility(float3 world) {
    if (Quality.x < 0.5) return 1;
    float visibility = (shadow_ray(world, -0.045) + shadow_ray(world, 0) +
                        shadow_ray(world, 0.045)) / 3;
    return lerp(1, visibility, Quality.w);
}

float3 detail_normal(float3 geometric, float3 world, float detail) {
    float3 dx = ddx(world), dy = ddy(world);
    float3 r1 = cross(dy, geometric), r2 = cross(geometric, dx);
    float determinant = dot(dx, r1);
    float3 gradient = (ddx(detail) * r1 + ddy(detail) * r2) *
        sign(determinant) / max(abs(determinant), 0.000001);
    return normalize(geometric - gradient * Quality.z);
}

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

float3 atmosphere(float y) {
    float horizon = saturate(1 - abs(y - 0.56) * 2.25);
    float3 low = float3(0.040, 0.058, 0.074);
    float3 high = float3(0.155, 0.225, 0.300);
    return lerp(low, high, saturate(y * 0.78 + 0.18)) +
           float3(0.12, 0.085, 0.045) * horizon * 0.17;
}

void ground_material(P input, out float3 albedo, out float height_detail,
                     out float specular_map) {
#ifdef BEAUTY_TERRAIN_TRANSITIONS
    // Use the terrain provider's world-aligned sampling and normalized BIQ
    // family weights so the mountain collar is the terrain beneath it, not a
    // second grass-only surface. The broad authored modulation also matches.
    float2 uv0 = input.world.xy * 0.43 + float2(0.31, 0.17);
    float2 uv1 = float2(input.world.y, -input.world.x) * (0.43 * 0.91) +
                 float2(0.63, 0.29);
    float2 tundra_uv = input.world.xy * (0.43 * 0.84) + float2(0.19, 0.71);
    float desert_weight = saturate(input.biome.y);
    float tundra_weight = saturate(input.material.w /
        max(1 - desert_weight, 0.00001));
    float plains_weight = saturate(input.biome.x /
        max(1 - desert_weight - input.material.w, 0.00001));
    float3 grass = GrassColor.Sample(Wrap, uv0).rgb;
    float3 plains = PlainsColor.Sample(Wrap, uv1).rgb;
    float3 tundra = TundraColor.Sample(Wrap, tundra_uv).rgb;
    albedo = lerp(lerp(grass, plains, plains_weight), tundra, tundra_weight);
    height_detail = lerp(lerp(GrassHeight.Sample(Wrap, uv0).r,
                                      PlainsHeight.Sample(Wrap, uv1).r,
                                      plains_weight),
                               TundraHeight.Sample(Wrap, tundra_uv).r,
                               tundra_weight);
    specular_map = lerp(lerp(GrassSpecular.Sample(Wrap, uv0).r,
                                    PlainsSpecular.Sample(Wrap, uv1).r,
                                    plains_weight),
                             TundraSpecular.Sample(Wrap, tundra_uv).r,
                             tundra_weight);
    albedo = lerp(albedo, DesertColor.Sample(Wrap, uv0).rgb, desert_weight);
    height_detail = lerp(height_detail, DesertHeight.Sample(Wrap, uv0).r,
                         desert_weight);
    specular_map = lerp(specular_map, DesertSpecular.Sample(Wrap, uv0).r,
                        desert_weight);
    // The fourth world component carries only the underlying authored relief.
    // Reconstruct the same hill-top family response at the mountain collar so
    // an adjacent hill cannot look like a separate mesh pushed through it.
    float slope = 1 - saturate(normalize(input.normal).z);
    float hill_weight = smoothstep(0.008, 0.18, input.base_relief) *
                        saturate(0.42 + slope * 2.2) * (1 - tundra_weight);
    float3 hill = lerp(GrassHillColor.Sample(Wrap, uv0 * 1.08).rgb,
                       PlainsHillColor.Sample(Wrap, uv1 * 1.08).rgb,
                       plains_weight);
    hill = lerp(hill, tundra, tundra_weight);
    float hill_h = lerp(GrassHillHeight.Sample(Wrap, uv0 * 1.08).r,
                        PlainsHillHeight.Sample(Wrap, uv1 * 1.08).r,
                        plains_weight);
    hill_h = lerp(hill_h, TundraHeight.Sample(Wrap, tundra_uv).r,
                  tundra_weight);
    float hill_s = lerp(GrassHillSpecular.Sample(Wrap, uv0 * 1.08).r,
                        PlainsHillSpecular.Sample(Wrap, uv1 * 1.08).r,
                        plains_weight);
    hill_s = lerp(hill_s, TundraSpecular.Sample(Wrap, tundra_uv).r,
                  tundra_weight);
    albedo = lerp(albedo, hill, hill_weight * 0.90);
    height_detail = lerp(height_detail, hill_h, hill_weight);
    specular_map = lerp(specular_map, hill_s, hill_weight);
    float broad = AuthoredHillHeight.Sample(Wrap,
        input.world.xy * 0.035 + float2(0.73, 0.21)).r;
    albedo *= lerp(0.90, 1.08, broad);
#else
    float2 uv = input.world.xy * 0.27 + 0.5;
    albedo = GrassColor.Sample(Wrap, uv).rgb;
    height_detail = GrassHeight.Sample(Wrap, uv).r;
    specular_map = GrassSpecular.Sample(Wrap, uv).r;
#endif
}

Output shade(P input) {
    Output output;
    if (input.material.y < 0.5) {
        output.color = float4(atmosphere(input.uv.y), 1);
        output.validity = 1;
        return output;
    }

    float3 geometric = normalize(input.normal);
    float3 albedo;
    float height_detail;
    float specular_map;
    float mountain_coverage = 0;
#ifdef BEAUTY_TERRAIN_TRANSITIONS
    // 42 remains the source-shadow mountain discriminator. Its fractional
    // range carries the authoritative coast/source-family coverage.
    float coast_alpha = saturate(input.coast_coverage - 42);
    clip(coast_alpha - 0.001);
#else
    float coast_alpha = 1;
#endif
    float3 ground_albedo;
    float ground_height;
    float ground_specular;
    ground_material(input, ground_albedo, ground_height, ground_specular);
    if (input.material.y < 1.5) {
        albedo = ground_albedo;
        float ground_luma = dot(albedo, float3(0.2126, 0.7152, 0.0722));
        albedo = lerp(albedo, ground_luma.xxx, 0.15) * float3(1.03, 1.0, 0.92);
        height_detail = ground_height;
        specular_map = ground_specular;
    } else if (Quality.x < 0.5) {
        // Control: the old failure mode stretches one planar albedo lookup over
        // steep faces and omits the source material response.
        float footprint = smoothstep(0.08, 0.72, input.material.z);
        mountain_coverage = footprint;
        clip(footprint - 0.015);
        albedo = lerp(ground_albedo, RockColor.Sample(Wrap, input.uv).rgb, footprint);
        height_detail = ground_height;
        specular_map = ground_specular * (1 - footprint);
    } else {
        float height = input.material.x;
        float footprint = smoothstep(0.08, 0.72, input.material.z);
        mountain_coverage = footprint;
        clip(footprint - 0.015);
        float snow = smoothstep(0.79, 0.94, height) * smoothstep(0.24, 0.72, geometric.z);
        float top = smoothstep(0.34, 0.73, height) * (1 - snow);
        float base = 1 - top - snow;
        float3 mountain_albedo = triplanar(RockColor, input.world, geometric) * base +
                                 triplanar(TopColor, input.world, geometric) * top +
                                 triplanar(SnowColor, input.world, geometric) * snow;
        albedo = lerp(ground_albedo, mountain_albedo, footprint);
        float mountain_detail = triplanar_scalar(RockHeight, input.world, geometric) * base +
                                triplanar_scalar(TopHeight, input.world, geometric) * top +
                                triplanar_scalar(SnowHeight, input.world, geometric) * snow;
        height_detail = lerp(ground_height, mountain_detail, footprint);
        float mountain_specular = triplanar_scalar(RockSpecular, input.world, geometric) * base +
                                  triplanar_scalar(TopSpecular, input.world, geometric) * top +
                                  triplanar_scalar(SnowSpecular, input.world, geometric) * snow;
        specular_map = lerp(ground_specular, mountain_specular, footprint);
    }
    if (input.material.y > 1.5 && Quality.x > 0.5) {
        // Civ VI's broad cool skylight is a major part of its gray-rock read;
        // preserve authored luminance/detail while avoiding raw brown albedo.
        // Apply it only to the rock footprint; grading the inherited ground
        // was the visible gray halo at plains and shoreline transitions.
        float rock_luma = dot(albedo, float3(0.2126, 0.7152, 0.0722));
        float3 graded = lerp(albedo, rock_luma.xxx, 0.28) * float3(0.96, 1.0, 1.07);
        albedo = lerp(albedo, graded, mountain_coverage);
    }
    float3 normal = Quality.x > 0.5 ? detail_normal(geometric, input.world, height_detail) : geometric;
#ifdef BEAUTY_COMPOSED_SHADOWS
    // The shared shadow-frame light is authoritative for both the BRDF and
    // projection, so every mountain face and cast shadow agrees in direction.
    float3 light_direction = ShadowL.xyz;
    float received_shadow = q6_shadow_visibility(ShadowField, input.world, normal,
        ShadowU, ShadowV, ShadowL, ShadowFlags.x > 0.5, true);
    // The low-coverage collar is only a material/geometry transition into the
    // authoritative ground. Let full cast shadow return with the rock body;
    // otherwise a tall coastal peak draws a detached dark wedge on its collar.
    float shadow = lerp(1.0, received_shadow,
                        smoothstep(0.12, 0.62, mountain_coverage));
#else
    float3 light_direction = Sun.xyz;
    float shadow = horizon_visibility(input.world);
#endif
    float ndl = saturate(dot(normal, light_direction));
    float wrap = saturate((dot(normal, light_direction) + 0.18) / 1.18);
    float altitude = input.material.y > 1.5 ? input.material.x : 0;
    float mountain_cavity = lerp(0.76, 1.0, smoothstep(0.03, 0.48, altitude));
    // The terminal collar represents inherited ground, so it must not retain
    // the mountain's cavity darkening as its opacity falls away.
    float cavity = input.material.y > 1.5 ?
        lerp(1.0, mountain_cavity, mountain_coverage) : 1.0;
    float sky = saturate(normal.z * 0.5 + 0.5);
    float3 ambient = Ambient.rgb * Ambient.a * lerp(0.52, 1.0, sky) * cavity;
    float3 diffuse = albedo * (ambient + SunColorExposure.rgb * Sun.w *
                               (0.055 + 0.945 * wrap) * shadow);
    float roughness = lerp(0.88, 0.38, saturate(specular_map));
    float specular = Quality.x > 0.5 ? ggx(normal, light_direction, normalize(View.xyz), roughness, 0.045) : 0;
    float rim = Quality.x > 0.5 ? pow(1 - saturate(dot(normal, normalize(View.xyz))), 3) *
        saturate(dot(normal, -light_direction) * 0.5 + 0.5) : 0;
    float3 radiance = diffuse + SunColorExposure.rgb * Sun.w * specular * shadow +
                      Ambient.rgb * rim * 0.13;
    // The mountain mesh deliberately extends beyond the rock footprint so its
    // joined normals and height can settle cleanly into the authoritative
    // terrain. Cross-fade that final collar instead of drawing its ground-like
    // material as an opaque, slightly raised lip (most visible on beaches).
    float surface_alpha = coast_alpha;
    if (input.material.y > 1.5)
        surface_alpha *= smoothstep(0.10, 0.48, mountain_coverage);
    clip(surface_alpha - 0.001);
    output.color = float4(max(radiance, 0) * surface_alpha, surface_alpha);
    output.validity = surface_alpha;
    return output;
}

Output PSMain(P input) { return shade(input); }
Output PSFeature(P input) { return shade(input); }
