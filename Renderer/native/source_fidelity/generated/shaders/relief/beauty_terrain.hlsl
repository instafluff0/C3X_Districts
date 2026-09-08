cbuffer Frame : register(b0) {
    float4 Sun;
    float4 SunColorExposure;
    float4 Ambient;
    float4 View;
    float4 Detail;
};
Texture2D GrassColor : register(t0);
Texture2D GrassHeight : register(t1);
Texture2D GrassSpecular : register(t2);
Texture2D GrassHillColor : register(t3);
Texture2D GrassHillHeight : register(t4);
Texture2D GrassHillSpecular : register(t5);
Texture2D PlainsColor : register(t6);
Texture2D PlainsHeight : register(t7);
Texture2D PlainsSpecular : register(t8);
Texture2D PlainsHillColor : register(t9);
Texture2D PlainsHillHeight : register(t10);
Texture2D PlainsHillSpecular : register(t11);
Texture2D HillDecalColor : register(t12);
Texture2D HillDecalNormal : register(t13);
Texture2D AuthoredHillHeight : register(t14);
Texture2D TundraColor : register(t15);
Texture2D TundraHeight : register(t16);
#ifdef BEAUTY_COMPOSED_SHADOWS
Texture2D ShadowField : register(t17);
Texture2D TundraSpecular : register(t18);
cbuffer ShadowFrame : register(b1) {
    float4 ShadowU;
    float4 ShadowV;
    float4 ShadowL;
    float4 ShadowOrigin;
    float4 ShadowFlags;
};
#include "../lighting/shadow_visibility_v1.hlsl"
#else
Texture2D TundraSpecular : register(t17);
#endif
Texture2D ForestFloorColor : register(t22);
Texture2D ForestFloorHeight : register(t23);
Texture2D JungleFloorColor : register(t24);
Texture2D JungleFloorHeight : register(t25);
Texture2D PlainsSurfaceColor : register(t26);
Texture2D PlainsSurfaceHeight : register(t27);
Texture2D DesertDuneColor : register(t28);
Texture2D DesertDuneHeight : register(t29);
Texture2D SurfaceDetail : register(t30);
SamplerState Wrap : register(s0);
SamplerState Clamp : register(s1);

struct V {
    float3 position : POSITION;
    float3 world : TEXCOORD0;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD1;
    float4 material : TEXCOORD2;
};
struct P {
    float4 position : SV_POSITION;
    float3 world : TEXCOORD0;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD1;
    float4 material : TEXCOORD2;
};
struct Output { float4 color : SV_Target0; float validity : SV_Target1; };

P VSMain(V input) {
    P output;
    output.position = float4(input.position, 1);
    output.world = input.world;
    output.normal = input.normal;
    output.uv = input.uv;
    output.material = input.material;
    return output;
}
P VSFeature(V input) { return VSMain(input); }

float3 detail_normal_strength(float3 geometric, float3 world, float detail, float strength) {
    float3 dx = ddx(world), dy = ddy(world);
    float3 r1 = cross(dy, geometric), r2 = cross(geometric, dx);
    float determinant = dot(dx, r1);
    float3 gradient = (ddx(detail) * r1 + ddy(detail) * r2) *
        sign(determinant) / max(abs(determinant), 0.000001);
    return normalize(geometric - gradient * strength);
}

float3 detail_normal(float3 geometric, float3 world, float detail) {
    return detail_normal_strength(geometric, world, detail, Detail.y);
}

float3 decal_normal(P input, float2 packed) {
    float3 geometric = normalize(input.normal);
    float3 dx = ddx(input.world), dy = ddy(input.world);
    float2 ux = ddx(input.uv), uy = ddy(input.uv);
    float determinant = ux.x * uy.y - ux.y * uy.x;
    if (abs(determinant) < 0.0000001) return geometric;
    float3 tangent = normalize((dx * uy.y - dy * ux.y) / determinant);
    float3 bitangent = normalize((dy * ux.x - dx * uy.x) / determinant);
    return normalize(geometric + tangent * (packed.x * 2 - 1) * 0.38 +
                     bitangent * (packed.y * 2 - 1) * 0.38);
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
    float gv = ndv / max(ndv * (1 - k) + k, 0.0001);
    float gl = ndl / max(ndl * (1 - k) + k, 0.0001);
    float fresnel = f0 + (1 - f0) * pow(1 - vdh, 5);
    return distribution * gv * gl * fresnel / max(4 * ndv * ndl, 0.0001) * ndl;
}

float3 atmosphere(float y) {
    float horizon = saturate(1 - abs(y - 0.56) * 2.25);
    return lerp(float3(0.040, 0.058, 0.074), float3(0.155, 0.225, 0.300),
                saturate(y * 0.78 + 0.18)) + float3(0.12, 0.085, 0.045) * horizon * 0.17;
}

float surface_shape(float2 world) {
    float broad = SurfaceDetail.Sample(Wrap, world * 0.071 + float2(0.13, 0.37)).r;
    float crossed = SurfaceDetail.Sample(Wrap,
        float2(world.y, -world.x) * 0.183 + float2(0.61, 0.29)).r;
    return saturate(broad * 0.72 + crossed * 0.28);
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
    float surface_occlusion = 1;
    float alpha = 1;

    if (input.material.y > 4.5) {
        bool desert_dune = input.material.y > 6.5;
        bool plains_surface = input.material.y > 5.5 && !desert_dune;
        float4 patch = desert_dune ? DesertDuneColor.Sample(Clamp, input.uv) :
                       (plains_surface ? PlainsSurfaceColor.Sample(Clamp, input.uv) :
                                         HillDecalColor.Sample(Clamp, input.uv));
        float2 packed = desert_dune ? DesertDuneHeight.Sample(Clamp, input.uv).rg :
                        (plains_surface ? PlainsSurfaceHeight.Sample(Clamp, input.uv).rg :
                                          HillDecalNormal.Sample(Clamp, input.uv).rg);
        clip(patch.a - 0.015);
        if (desert_dune) {
            float3 substrate = DesertColor.Sample(Wrap,
                input.world.xy * Detail.x + float2(0.31, 0.17)).rgb;
            albedo = lerp(substrate, patch.rgb, 0.42);
            packed = lerp(0.5.xx, packed, 0.58);
        } else {
            albedo = patch.rgb;
        }
        geometric = decal_normal(input, packed);
        height_detail = packed.r;
        specular_map = desert_dune ? 0.08 : 0.04;
        surface_occlusion = lerp(1.0, 0.80, saturate(length(packed * 2 - 1)));
        // The source patch carries its own coverage. The interpolated
        // authoritative biome field fades it at ecotones and terrain edits.
        alpha *= patch.a * smoothstep(0.015, 0.42, input.material.z) *
                 (desert_dune ? 0.62 : 1.0);
    } else if (input.material.y > 2.5) {
        bool jungle_floor = input.material.y > 3.5;
        float4 floor_sample = jungle_floor ? JungleFloorColor.Sample(Clamp, input.uv) :
                                             ForestFloorColor.Sample(Clamp, input.uv);
        float2 floor_normal = jungle_floor ? JungleFloorHeight.Sample(Clamp, input.uv).rg :
                                             ForestFloorHeight.Sample(Clamp, input.uv).rg;
        clip(floor_sample.a - 0.06);
        // The decoded decal is the confirmed source albedo. Civ VI's final
        // vegetation-floor response is much darker beneath the canopy than a
        // normally lit terrain decal; the exact engine AO equation is not in
        // the package, so retain the source hue while reconstructing that
        // canopy attenuation explicitly.
        float3 canopy_tint = float3(0.58, 0.58, 0.58);
        if (jungle_floor) canopy_tint = float3(0.30, 0.28, 0.52);
        albedo = floor_sample.rgb * canopy_tint;
        geometric = decal_normal(input, floor_normal);
        height_detail = floor_normal.r;
        specular_map = 0.04;
        alpha *= floor_sample.a;
    } else if (input.material.y > 1.5) {
        float4 decal = HillDecalColor.Sample(Clamp, input.uv);
        clip(decal.a - 0.015);
        // The decal defines the irregular authored patch footprint. Its paired
        // hill-top material supplies the denser exposed-stone field visible in
        // the source game. Both inputs are source-authored; this combination is
        // an inferred Lab response, not a claim about Firaxis' shader equation.
        float2 stone_uv = float2(input.world.y, -input.world.x) * 0.71 + float2(0.37, 0.59);
        float3 stone_source = GrassHillColor.Sample(Wrap, stone_uv).rgb;
        float stone_ratio = stone_source.b / max(stone_source.g, 0.025);
        float rock = smoothstep(0.18, 0.42, stone_ratio);
        float stone_luma = dot(stone_source, float3(0.2126, 0.7152, 0.0722));
        float3 stone = lerp(stone_source, stone_luma.xxx, 0.34) *
                       float3(0.91, 0.97, 1.10) * 0.84;
        albedo = lerp(decal.rgb, stone, rock * 0.94);
        float stone_height = GrassHillHeight.Sample(Wrap, stone_uv).r;
        geometric = detail_normal_strength(geometric, input.world,
                                           stone_height * 0.58 + rock * 0.42, 0.11);
        height_detail = stone_height;
        specular_map = 0.06;
        alpha = decal.a;
    } else {
        float2 uv0 = input.world.xy * Detail.x + float2(0.31, 0.17);
        float2 uv1 = float2(input.world.y, -input.world.x) * (Detail.x * 0.91) + float2(0.63, 0.29);
        float3 grass = GrassColor.Sample(Wrap, uv0).rgb;
        float3 plains = PlainsColor.Sample(Wrap, uv1).rgb;
        float2 tundra_uv = input.world.xy * (Detail.x * 0.84) + float2(0.19, 0.71);
        float3 tundra = TundraColor.Sample(Wrap, tundra_uv).rgb;
        float grass_h = GrassHeight.Sample(Wrap, uv0).r;
        float plains_h = PlainsHeight.Sample(Wrap, uv1).r;
        float tundra_h = TundraHeight.Sample(Wrap, tundra_uv).r;
        float grass_s = GrassSpecular.Sample(Wrap, uv0).r;
        float plains_s = PlainsSpecular.Sample(Wrap, uv1).r;
        float tundra_s = TundraSpecular.Sample(Wrap, tundra_uv).r;
        float plains_weight = smoothstep(0.18, 0.88, input.material.w);
        float tundra_weight = smoothstep(1.12, 1.82, input.material.w);
        float3 base = lerp(lerp(grass, plains, plains_weight), tundra, tundra_weight);
        float base_h = lerp(lerp(grass_h, plains_h, plains_weight), tundra_h, tundra_weight);
        float base_s = lerp(lerp(grass_s, plains_s, plains_weight), tundra_s, tundra_weight);

        float slope = 1 - saturate(geometric.z);
        float rocky_band = input.material.z * smoothstep(0.09, 0.43, input.material.x) *
                           (1 - tundra_weight) *
                           saturate(0.42 + slope * 2.2);
        float3 hill = lerp(GrassHillColor.Sample(Wrap, uv0 * 1.08).rgb,
                           PlainsHillColor.Sample(Wrap, uv1 * 1.08).rgb, plains_weight);
        hill = lerp(hill, tundra, tundra_weight);
        float hill_h = lerp(GrassHillHeight.Sample(Wrap, uv0 * 1.08).r,
                            PlainsHillHeight.Sample(Wrap, uv1 * 1.08).r, plains_weight);
        hill_h = lerp(hill_h, tundra_h, tundra_weight);
        float hill_s = lerp(GrassHillSpecular.Sample(Wrap, uv0 * 1.08).r,
                            PlainsHillSpecular.Sample(Wrap, uv1 * 1.08).r, plains_weight);
        hill_s = lerp(hill_s, tundra_s, tundra_weight);
        float hill_ratio = hill.r / max(hill.g, 0.025);
        float hill_blue_ratio = hill.b / max(hill.g, 0.025);
        float hill_rock = max(smoothstep(0.83, 1.02, hill_ratio),
                              smoothstep(0.31, 0.62, hill_blue_ratio));
        float hill_luma = dot(hill, float3(0.2126, 0.7152, 0.0722));
        float3 hill_stone = lerp(hill_luma.xxx, hill, 0.16) * float3(0.91, 0.96, 1.10) * 1.20;
        hill = lerp(hill, hill_stone, hill_rock * 0.82);
        albedo = lerp(base, hill, rocky_band * 0.90);
        height_detail = lerp(base_h, hill_h, rocky_band);
        specular_map = lerp(base_s, hill_s, rocky_band);
        geometric = detail_normal(geometric, input.world, height_detail);

        // Source-backed detail supplies a continuous material-scale response.
        // Cooler lows and warm dry highs add readable regional variation
        // without perturbing the flat-ground geometry or drawing tile borders.
        float broad = surface_shape(input.world.xy);
        albedo *= lerp(float3(0.88, 0.94, 0.97),
                       float3(1.09, 1.045, 0.91), broad);
    }

#ifdef BEAUTY_COMPOSED_SHADOWS
    // Q6ShadowL is also the projection direction used to build the shared
    // shadow field. One vector must drive both face light and cast shadows.
    float3 light_direction = ShadowL.xyz;
#else
    float3 light_direction = Sun.xyz;
#endif
    float ndl = saturate(dot(geometric, light_direction));
    float wrap = saturate((dot(geometric, light_direction) + 0.20) / 1.20);
    float sky = saturate(geometric.z * 0.5 + 0.5);
    float cavity = lerp(0.79, 1.0, smoothstep(0.02, 0.30, input.material.x));
    float3 ambient = Ambient.rgb * Ambient.a * lerp(0.56, 1.0, sky) * cavity * surface_occlusion;
#ifdef BEAUTY_COMPOSED_SHADOWS
    float shadow = q6_shadow_visibility(ShadowField, input.world, geometric,
        ShadowU, ShadowV, ShadowL, ShadowFlags.x > 0.5, true);
#else
    float shadow = 1.0;
#endif
    float3 diffuse = albedo * (ambient + SunColorExposure.rgb * Sun.w *
                               (0.07 + 0.93 * wrap) * shadow);
    float roughness = lerp(0.92, 0.48, saturate(specular_map));
    float specular = ggx(geometric, light_direction, normalize(View.xyz), roughness, 0.035);
    float3 radiance = diffuse + SunColorExposure.rgb * Sun.w * specular * shadow;
    output.color = float4(max(radiance, 0) * alpha, alpha);
    output.validity = alpha;
    return output;
}

Output PSMain(P input) { return shade(input); }
Output PSFeature(P input) { return shade(input); }
