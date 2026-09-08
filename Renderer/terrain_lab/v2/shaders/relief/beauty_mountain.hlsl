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
    float3 material : TEXCOORD2;
};
struct P {
    float4 position : SV_POSITION;
    float3 world : TEXCOORD0;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD1;
    float3 material : TEXCOORD2;
};
struct Output { float4 color : SV_Target0; float validity : SV_Target1; };

P VSMain(V input) {
    P output;
    output.position = float4(input.position, 1);
    output.world = input.world.xyz;
    output.normal = input.normal;
    output.uv = input.uv;
    output.material = input.material;
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
    if (input.material.y < 1.5) {
        float2 ground_uv = input.world.xy * 0.27 + 0.5;
        albedo = GrassColor.Sample(Wrap, ground_uv).rgb;
        float ground_luma = dot(albedo, float3(0.2126, 0.7152, 0.0722));
        albedo = lerp(albedo, ground_luma.xxx, 0.15) * float3(1.03, 1.0, 0.92);
        height_detail = GrassHeight.Sample(Wrap, ground_uv).r;
        specular_map = GrassSpecular.Sample(Wrap, ground_uv).r;
    } else if (Quality.x < 0.5) {
        // Control: the old failure mode stretches one planar albedo lookup over
        // steep faces and omits the source material response.
        float footprint = smoothstep(0.08, 0.72, input.material.z);
        clip(footprint - 0.015);
        float3 grass = GrassColor.Sample(Wrap, input.world.xy * 0.27 + 0.5).rgb;
        albedo = lerp(grass, RockColor.Sample(Wrap, input.uv).rgb, footprint);
        height_detail = GrassHeight.Sample(Wrap, input.world.xy * 0.27 + 0.5).r;
        specular_map = 0;
    } else {
        float height = input.material.x;
        float footprint = smoothstep(0.08, 0.72, input.material.z);
        clip(footprint - 0.015);
        float snow = smoothstep(0.79, 0.94, height) * smoothstep(0.24, 0.72, geometric.z);
        float top = smoothstep(0.34, 0.73, height) * (1 - snow);
        float base = 1 - top - snow;
        float3 mountain_albedo = triplanar(RockColor, input.world, geometric) * base +
                                 triplanar(TopColor, input.world, geometric) * top +
                                 triplanar(SnowColor, input.world, geometric) * snow;
        float3 grass = GrassColor.Sample(Wrap, input.world.xy * 0.27 + 0.5).rgb;
        albedo = lerp(grass, mountain_albedo, footprint);
        float mountain_detail = triplanar_scalar(RockHeight, input.world, geometric) * base +
                                triplanar_scalar(TopHeight, input.world, geometric) * top +
                                triplanar_scalar(SnowHeight, input.world, geometric) * snow;
        height_detail = lerp(GrassHeight.Sample(Wrap, input.world.xy * 0.27 + 0.5).r,
                             mountain_detail, footprint);
        float mountain_specular = triplanar_scalar(RockSpecular, input.world, geometric) * base +
                                  triplanar_scalar(TopSpecular, input.world, geometric) * top +
                                  triplanar_scalar(SnowSpecular, input.world, geometric) * snow;
        specular_map = mountain_specular * footprint;
    }
    if (input.material.y > 1.5 && Quality.x > 0.5) {
        // Civ VI's broad cool skylight is a major part of its gray-rock read;
        // preserve authored luminance/detail while avoiding raw brown albedo.
        float rock_luma = dot(albedo, float3(0.2126, 0.7152, 0.0722));
        albedo = lerp(albedo, rock_luma.xxx, 0.28) * float3(0.96, 1.0, 1.07);
    }
    float3 normal = Quality.x > 0.5 ? detail_normal(geometric, input.world, height_detail) : geometric;
#ifdef BEAUTY_COMPOSED_SHADOWS
    // The shared shadow-frame light is authoritative for both the BRDF and
    // projection, so every mountain face and cast shadow agrees in direction.
    float3 light_direction = ShadowL.xyz;
    float shadow = q6_shadow_visibility(ShadowField, input.world, normal,
        ShadowU, ShadowV, ShadowL, ShadowFlags.x > 0.5, true);
#else
    float3 light_direction = Sun.xyz;
    float shadow = horizon_visibility(input.world);
#endif
    float ndl = saturate(dot(normal, light_direction));
    float wrap = saturate((dot(normal, light_direction) + 0.18) / 1.18);
    float altitude = input.material.y > 1.5 ? input.material.x : 0;
    float cavity = lerp(0.76, 1.0, smoothstep(0.03, 0.48, altitude));
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
    output.color = float4(max(radiance, 0), 1);
    output.validity = 1;
    return output;
}

Output PSMain(P input) { return shade(input); }
Output PSFeature(P input) { return shade(input); }
