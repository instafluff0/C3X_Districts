"""Autumn shader patches for the study's private shader roots.

A look is a set of #define calibrations, a per-tree leaf palette and anchored
edits to the flattened production shaders. An edit whose anchor is missing or
ambiguous fails loudly, so production drift can never silently drop part of a
look. All colour targets were measured on the selected concept (OKLab L/C/h).
"""
from __future__ import annotations

from pathlib import Path

COMMON = Path(__file__).with_name("autumn_common.hlsl").read_text()
CITY = "Renderer/native/city_fidelity/"
TREES = CITY + "objects.hlsl"
RELIEF = CITY + "terrain.hlsl"
MOUNTAIN = CITY + "mountain.hlsl"
CASTER = "Renderer/native/source_fidelity/instance_caster.hlsl"
RIGID = CITY + "rigid_feature.hlsl"
FEATURE_ONLY = (RIGID, CITY + "resource_body.hlsl")
FIELD_SHADERS = FEATURE_ONLY + (CITY + "feature.hlsl", CITY + "city.hlsl")
INTEGRATED = tuple(CITY + name for name in ("hydrology.hlsl", "feature.hlsl", "city.hlsl"))

# (cumulative share, OKLab lightness, chroma, hue) per crown colour.
PALETTE_V1 = [(0.30, 0.76, 0.150, 84), (0.44, 0.80, 0.140, 93), (0.68, 0.69, 0.160, 58),
              (0.78, 0.67, 0.120, 43), (0.86, 0.56, 0.140, 35), (1.00, 0.64, 0.095, 104)]
# Gold-led: less coral/russet (read pink at game zoom), an amber step.
PALETTE_V2 = [(0.34, 0.75, 0.150, 84), (0.48, 0.79, 0.140, 92), (0.60, 0.70, 0.150, 70),
              (0.82, 0.67, 0.155, 59), (0.88, 0.64, 0.120, 50), (0.93, 0.55, 0.130, 42),
              (1.00, 0.62, 0.090, 104)]

# Coral and russet darker and more orange: lighter versions read pink.
PALETTE_V4 = [(0.34, 0.75, 0.150, 84), (0.48, 0.79, 0.140, 92), (0.60, 0.70, 0.150, 70),
              (0.82, 0.67, 0.155, 59), (0.88, 0.60, 0.130, 54), (0.93, 0.51, 0.125, 47),
              (1.00, 0.62, 0.090, 104)]

V1 = {
    "AUTUMN_PATCH_SCALE": 0.45, "AUTUMN_GROUND_SHIFT": -20, "AUTUMN_PATCH_HUE": 16,
    "AUTUMN_STRAW_CHROMA": 1.12, "AUTUMN_GRASS_CHROMA": 0.84, "AUTUMN_GRASS_LIGHT": 0.92,
    "AUTUMN_PATCH_LIGHT": 0.10, "AUTUMN_LEAF_LIGHT": 1.0, "AUTUMN_LEAF_DETAIL": 1.0,
    "AUTUMN_BROADLEAF_GB": 3.2, "AUTUMN_CROWN": 0.6, "AUTUMN_WRAP": 0.35, "AUTUMN_FILL": 0.95,
    "AUTUMN_CROWN_BASE": 1.0, "AUTUMN_CROWN_TOP": 1.0, "AUTUMN_LEAF_CHROMA": 1.0,
    "AUTUMN_SAND_FLOOR": -360, "AUTUMN_LEAF_BIAS": 0, "AUTUMN_HILL_TURF": 1.0,
    "AUTUMN_WATER_HUE": 240, "AUTUMN_WATER_PULL": 0.7, "AUTUMN_WATER_CHROMA": 1.9, "AUTUMN_WATER_LIGHT": 0.94,
    "AUTUMN_GREEN_PULL": 0, "AUTUMN_RIVER_WATER": 1.0, "AUTUMN_TRANSLUCENCY": 0, "AUTUMN_PALE_KEEP": 0,
    "AUTUMN_HILL_FLOOR": 0.0201, "AUTUMN_FIELD_CELLS": 2.5, "AUTUMN_LARCH_SHARE": 0, "AUTUMN_SHRUB_SHARE": 0,
    "AUTUMN_ROAD_DARK": "float3(1, 1, 1)", "AUTUMN_WATER_NIGHT": 1.0, "AUTUMN_FIELD": "autumn_field",
    "AUTUMN_FARM_HUE": 82, "AUTUMN_FARM_PULL": 0.85, "AUTUMN_FARM_CHROMA": 0.85, "AUTUMN_FARM_LIGHT": 1.04,
    "AUTUMN_WATER_PULL_START": 100,
}
V2 = {**V1, "AUTUMN_PATCH_LIGHT": 0.18, "AUTUMN_PATCH_HUE": 18, "AUTUMN_LEAF_DETAIL": 1.35,
      "AUTUMN_CROWN": 0.7, "AUTUMN_WRAP": 0.12, "AUTUMN_FILL": 0.80, "AUTUMN_TURF": 1.1,
      "AUTUMN_HILL_CHROMA": 1.2}
# Crowns read as round puffs: dark undersides, brighter tops, less leaf speckle.
V3 = {**V2, "AUTUMN_CROWN": 0.85, "AUTUMN_LEAF_DETAIL": 0.8, "AUTUMN_CROWN_BASE": 0.5,
      "AUTUMN_CROWN_TOP": 1.08, "AUTUMN_LEAF_CHROMA": 0.92, "AUTUMN_WRAP": 0.2, "AUTUMN_FILL": 0.85}
# Hill turf as warm as the flats, sand kept honey, softer leaf speckle.
V4 = {**V3, "AUTUMN_SAND_FLOOR": 70, "AUTUMN_LEAF_BIAS": 1.0, "AUTUMN_HILL_TURF": 1.32}
# Greenest turf pulled further, rivers keep half their colour, backlit leaves glow.
V6 = {**V4, "AUTUMN_GREEN_PULL": 0.6, "AUTUMN_RIVER_WATER": 0.5, "AUTUMN_TRANSLUCENCY": 0.45,
      "AUTUMN_PALE_KEEP": 0.85}
# Khaki floor for near-grey hill turf, stronger glow through backlit leaves.
V7 = {**V6, "AUTUMN_HILL_FLOOR": 0.055, "AUTUMN_TRANSLUCENCY": 0.8}
# Context pass: harvest fields; a golden larch share in conifer stands and
# red-orange shrubs, so Civ III pine forests also read as autumn.
# Damp, darker dirt roads stay readable on gold turf; the deeper water fades
# back toward the shared look at night instead of glowing.
V8 = {**V7, "AUTUMN_LARCH_SHARE": 0.28, "AUTUMN_SHRUB_SHARE": 0.7,
      "AUTUMN_ROAD_DARK": "float3(0.80, 0.76, 0.74)", "AUTUMN_WATER_NIGHT": 0.3}
# User, 2026-10-21: golden pines look odd; just tint farms; cities unchanged.
V9 = {**V8, "AUTUMN_LARCH_SHARE": 0, "AUTUMN_SHRUB_SHARE": 0, "AUTUMN_FIELD": "autumn_farm"}
# Paler harvest-gold plots keep the field pattern readable against olive turf.
V10 = {**V9, "AUTUMN_FARM_HUE": 88, "AUTUMN_FARM_CHROMA": 0.75, "AUTUMN_FARM_LIGHT": 1.16}
# Coast keeps its teal hue (v10 pulled it halfway to the sea's blue).
V11 = {**V10, "AUTUMN_WATER_PULL_START": 205}


def palette(entries) -> str:
    branches = []
    for index, (share, light, chroma, hue) in enumerate(entries):
        test = "" if index == len(entries) - 1 else f"if (seed < {share}) "
        prefix = "    " if index == 0 else "    else "
        branches.append(f"{prefix}{test}pal = float3({light}, {chroma}, {hue});")
    return ("// Per-tree palette; the atlas keeps its own light/dark leaf structure.\n"
            "float3 autumn_leaf_color(float3 albedo, float3 average, float seed) {\n"
            "    float3 pal;\n" + "\n".join(branches) + "\n"
            "    pal.z += (frac(seed * 7.31) - 0.5) * 10;\n"
            "    pal.x += (frac(seed * 13.7) - 0.5) * 0.06;\n"
            "    float3 src = autumn_lch(albedo), avg = autumn_lch(average);\n"
            "    float L = pal.x * AUTUMN_LEAF_LIGHT + (src.x - avg.x) * AUTUMN_LEAF_DETAIL;\n"
            "    float C = pal.y * AUTUMN_LEAF_CHROMA * saturate(0.45 + 0.75 * src.x / max(avg.x, 0.05));\n"
            "    float h = pal.z + clamp(src.z - avg.z, -25, 25) * 0.3;\n"
            "    return autumn_from_lch(float3(max(L, 0.05), C, h));\n}\n")


def edits(values, leaves, *, turf=False, mountains=False, warm_hills=False, water=False, fields=False,
          conifers=False):
    common = "".join(f"#define {key} {value}\n" for key, value in values.items()) + COMMON + palette(leaves)
    result = [
        # Per-tree seed and an ellipsoidal crown normal travel with each instance.
        ((TREES,), "    float2 secondary : TEXCOORD3;\n};\nstruct Output",
         "    float2 secondary : TEXCOORD3;\n    nointerpolation float autumn_seed : TEXCOORD4;\n"
         "    float4 crown : TEXCOORD5;\n};\nstruct Output"),
        ((TREES,), "    output.secondary = input.secondary;\n    return output;",
         "    output.secondary = input.secondary;\n    output.autumn_seed = 0;\n"
         "    output.crown = float4(input.normal, 1);\n    return output;"),
        ((TREES,), " P o=VSMain(v);\n o.position.xy=(floor(v.position.xy*256+.5)/256+i.placement_view.xy)",
         " P o=VSMain(v);\n"
         " o.autumn_seed=frac(sin(dot(i.place0.xy+i.place0.zw*7.3+i.place1.xy*3.1,float2(12.9898,78.233)))*43758.5453);\n"
         " {float3 q=i.source_position-float3(0,0,.15);float3 g=float3(q.x/.04,q.y/.04,q.z/.012);\n"
         "  o.crown=float4(normalize(float3(g.x*co-g.y*si,-(g.x*si+g.y*co),g.z/(150.f/(.82f*64.f)))),saturate(i.source_position.z/.24));}\n"
         " o.position.xy=(floor(v.position.xy*256+.5)/256+i.placement_view.xy)"),
        ((TREES, RELIEF), "Output shade(P input) {", common + "Output shade(P input) {"),
        ((TREES,), "    float3 normal = geometric;\n    if (kind < 0.5) {",
         "    float3 normal = geometric;\n    bool autumn_leaf = false;\n    if (kind < 0.5) {"),
        ((TREES,), "        if (has_normal)\n            normal = mapped_normal(geometric, input.world, input.uv,\n"
                   "                                   foliage || repeat_address);",
         "        if (foliage) {\n"
         "            // Lab stand-in for pack metadata: the broadleaf atlas is yellow-green.\n"
         "            float3 average = BaseColor.SampleLevel(Clamp, float2(0.5, 0.5), 15).rgb;\n"
         "            autumn_leaf = average.g > AUTUMN_BROADLEAF_GB * average.b;\n"
         "            float leafness = saturate((base.g - base.r) * 25 + 0.75);\n"
         "            if (autumn_leaf)\n"
         "                albedo = lerp(albedo, autumn_leaf_color(lerp(albedo, BaseColor.SampleBias(Wrap, input.uv, AUTUMN_LEAF_BIAS).rgb,\n"
         "                    AUTUMN_LEAF_BIAS > 0 ? 0.7 : 0), average, input.autumn_seed), leafness) *\n"
         "                    lerp(AUTUMN_CROWN_BASE, AUTUMN_CROWN_TOP, smoothstep(0.15, 0.85, input.crown.w));\n"
         "        }\n"
         "        if (has_normal)\n            normal = mapped_normal(geometric, input.world, input.uv,\n"
         "                                   foliage || repeat_address);"),
        ((TREES,), "    float ndl = saturate(dot(normal, light_direction));\n    float diffuse = ndl;",
         "    if (autumn_leaf) normal = normalize(lerp(normal, normalize(input.crown.xyz), AUTUMN_CROWN));\n"
         "    float ndl = saturate(dot(normal, light_direction));\n"
         "    float diffuse = autumn_leaf ? saturate((dot(normal, light_direction) + AUTUMN_WRAP) / (1 + AUTUMN_WRAP)) : ndl;"),
        # Sunlight through backlit leaves (low evening sun behind the crowns).
        ((TREES,), "    if (!(kind > 0.5 && kind < 1.5) && input.secondary.y > 0.5)\n        radiance += Emissive.Sample(Clamp, input.uv).rgb * 0.035;",
         "    if (autumn_leaf)\n"
         "        radiance += albedo * SunColorExposure.rgb * Sun.w * shadow * AUTUMN_TRANSLUCENCY *\n"
         "            pow(saturate(dot(normalize(View.xyz), -light_direction) * 0.5 + 0.5), 2) * (1 - ndl);\n"
         "    if (!(kind > 0.5 && kind < 1.5) && input.secondary.y > 0.5)\n        radiance += Emissive.Sample(Clamp, input.uv).rgb * 0.035;"),
        ((TREES,), "    float foliage_fill = kind > 0.5 && kind < 1.5 ? 0.72 : 1.0;",
         "    float foliage_fill = autumn_leaf ? AUTUMN_FILL : (kind > 0.5 && kind < 1.5 ? 0.72 : 1.0);"),
        # Land ground, hills, hill decals and forest floor (natural relief shader).
        ((RELIEF,), "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction",
         "    if (input.material.y > 2.5 && input.material.y < 3.5) albedo = autumn_litter(albedo, input.world.xy);\n"
         "    else if (!(input.material.y > 3.5 && input.material.y < 4.5)) {\n"
         "        albedo = autumn_ground(albedo, input.world.xy);\n"
         "        float autumn_hill = input.material.y < 1.5 ? input.material.z * smoothstep(0.02, 0.2, input.material.x) : 0;\n"
         "        float3 autumn_hill_lch = autumn_lch(albedo) * float3(1, lerp(1, AUTUMN_HILL_TURF, autumn_hill), 1);\n"
         "        // Near-grey hill turf (tundra edges) gets a dry khaki floor instead of staying grey-green.\n"
         "        float autumn_grey = autumn_hill * (1 - smoothstep(0.02, AUTUMN_HILL_FLOOR, autumn_hill_lch.y));\n"
         "        autumn_hill_lch.z = lerp(autumn_hill_lch.z, 84, autumn_grey);\n"
         "        autumn_hill_lch.y = lerp(autumn_hill_lch.y, max(autumn_hill_lch.y, AUTUMN_HILL_FLOOR), autumn_hill);\n"
         "        albedo = autumn_from_lch(autumn_hill_lch);\n"
         "    }\n"
         "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction"),
        # Coasts and beaches (integrated terrain shaders).
        # q6_raw_feature precedes q6_raw_main, so helpers go before both.
        (INTEGRATED, "float4 q6_raw_feature(FeaturePixelInput input)\n{", common + "float4 q6_raw_feature(FeaturePixelInput input)\n{"),
        (INTEGRATED, "#ifdef Q8_DEBUG_ALBEDO\n        return float4(albedo,1);",
         "        albedo = autumn_ground(albedo, world_position);\n#ifdef Q8_DEBUG_ALBEDO\n        return float4(albedo,1);"),
    ]
    if turf:
        result += [
            # Dry turf: amplify the source grass/plains pattern against its own
            # coarse mean, so existing blades and tufts carry the extra contrast.
            ((RELIEF,), "        albedo *= 1 + clamp((grain - 0.303) * 0.9, -0.16, 0.22) * grass_plains_detail;",
             "        albedo *= 1 + clamp((grain - 0.303) * 0.9, -0.16, 0.22) * grass_plains_detail;\n"
             "        {\n"
             "            float3 lum = float3(0.2126, 0.7152, 0.0722);\n"
             "            float grass_ratio = dot(grass_fine, lum) / max(dot(grass_low_blend, lum), 0.01);\n"
             "            float plains_ratio = dot(plains, lum) / max(dot(PlainsColor.SampleBias(Wrap, uv1, 5.0).rgb, lum), 0.01);\n"
             "            float turf = lerp(grass_ratio, plains_ratio, plains_weight) - 1;\n"
             "            albedo *= 1 + clamp(turf * AUTUMN_TURF, -0.35, 0.45) * (1 - tundra_weight) * (1 - desert_weight) * (1 - rocky_band);\n"
             "        }"),
            # Hill stone keeps its grey relief but loses the cool cast beside gold turf.
            ((RELIEF,), "        float3 hill_stone = lerp(hill_luma.xxx, hill, 0.16) * float3(0.91, 0.96, 1.10) * 1.20;",
             "        float3 hill_stone = lerp(hill_luma.xxx, hill, 0.16) * float3(1.0, 0.98, 0.94) * 1.22;"),
            ((RELIEF,), "        albedo = lerp(base, hill, rocky_band * 0.90);",
             "        albedo = lerp(base, hill, rocky_band * 0.90);\n"
             "        albedo = lerp(albedo, autumn_from_lch(autumn_lch(albedo) * float3(1, AUTUMN_HILL_CHROMA, 1)), rocky_band);"),
        ]
    if warm_hills:
        result = [edit for edit in result if not edit[1].startswith("        float3 hill_stone")]
        result += [
            # Concept hills read as light khaki turf over grey stone, not grey mounds.
            ((RELIEF,), "        float3 hill_stone = lerp(hill_luma.xxx, hill, 0.16) * float3(0.91, 0.96, 1.10) * 1.20;",
             "        float3 hill_stone = lerp(hill_luma.xxx, hill, 0.30) * float3(1.04, 0.99, 0.90) * 1.30;"),
            ((RELIEF,), "        float3 stone = lerp(stone_source, stone_luma.xxx, 0.34) *\n                       float3(0.91, 0.97, 1.10) * 0.84;",
             "        float3 stone = lerp(stone_source, stone_luma.xxx, 0.34) *\n                       float3(1.0, 0.97, 0.92) * 0.98;"),
        ]
    if water:
        result += [
            (INTEGRATED, "        return q3_water_material(input);\n#endif",
             "    {\n        float4 autumn_surface = q3_water_material(input);\n"
             "        float autumn_strength = (input.surface_kind > 8.5 && input.surface_kind < 9.5 ? AUTUMN_RIVER_WATER : 1) *\n"
             "            lerp(AUTUMN_WATER_NIGHT, 1, saturate(environment_sun_intensity * 2.5));\n"
             "        return float4(autumn_water(autumn_surface.rgb, autumn_strength), autumn_surface.a);\n    }\n#endif"),
        ]
    if fields:
        result += [
            ((CITY + "hydrology.hlsl",), "                lerp(0.90, 1.06, pattern_ground);\n            return float4(frame_tone_map(track_albedo * track_light *",
             "                lerp(0.90, 1.06, pattern_ground);\n            track_albedo *= AUTUMN_ROAD_DARK;\n"
             "            return float4(frame_tone_map(track_albedo * track_light *"),
            # Farm kit fields draw through every q6_raw_feature copy.
            (FEATURE_ONLY, "float4 q6_raw_feature(FeaturePixelInput input)\n{", common + "float4 q6_raw_feature(FeaturePixelInput input)\n{"),
            (FIELD_SHADERS, "            albedo = lerp(albedo, farm_kit_ground_tint(albedo, farm_kit_terrain),\n"
                       "                0.7 - 0.35 * saturate((farm_kit_terrain - 0.65) / 0.35));\n    }\n",
             "            albedo = lerp(albedo, farm_kit_ground_tint(albedo, farm_kit_terrain),\n"
             "                0.7 - 0.35 * saturate((farm_kit_terrain - 0.65) / 0.35));\n    }\n"
             "    if (farm_kit_field_weight > 0.5)\n"
             "        albedo = mine_slot < 0.5 ? AUTUMN_FIELD(albedo, input.q6_world.xy) : autumn_ground(albedo, input.q6_world.xy);\n"),
        ]
    if conifers:
        result = [edit for edit in result if not edit[1].startswith("        if (has_normal)")]
        result += [
            ((TREES,), "        if (has_normal)\n            normal = mapped_normal(geometric, input.world, input.uv,\n"
                       "                                   foliage || repeat_address);",
             "        if (foliage) {\n"
             "            // Lab stand-in for pack metadata: classify each atlas by its average colour.\n"
             "            float3 average = BaseColor.SampleLevel(Clamp, float2(0.5, 0.5), 15).rgb;\n"
             "            float gb = average.g / max(average.b, 0.001);\n"
             "            autumn_leaf = gb > AUTUMN_BROADLEAF_GB;\n"
             "            bool conifer = gb > 1.95 && gb < 2.6 && average.r < 0.85 * average.g;\n"
             "            bool shrub = gb > 1.6 && gb <= 1.95;\n"
             "            float leafness = saturate((base.g - base.r) * 25 + 0.75);\n"
             "            float3 soft = lerp(albedo, BaseColor.SampleBias(Wrap, input.uv, AUTUMN_LEAF_BIAS).rgb, AUTUMN_LEAF_BIAS > 0 ? 0.7 : 0);\n"
             "            if (autumn_leaf)\n"
             "                albedo = lerp(albedo, autumn_leaf_color(soft, average, input.autumn_seed), leafness) *\n"
             "                    lerp(AUTUMN_CROWN_BASE, AUTUMN_CROWN_TOP, smoothstep(0.15, 0.85, input.crown.w));\n"
             "            else if (conifer && input.autumn_seed < AUTUMN_LARCH_SHARE) {\n"
             "                // Larch: a conifer whose needles turn gold.\n"
             "                float3 lch = autumn_lch(albedo), avg = autumn_lch(average);\n"
             "                albedo = lerp(albedo, autumn_from_lch(float3(0.68 + (lch.x - avg.x), 0.125, 86 - 6 * frac(input.autumn_seed * 9.1))), leafness);\n"
             "            } else if (shrub && input.autumn_seed < AUTUMN_SHRUB_SHARE) {\n"
             "                float3 lch = autumn_lch(albedo), avg = autumn_lch(average);\n"
             "                albedo = lerp(albedo, autumn_from_lch(float3(0.58 + (lch.x - avg.x), 0.14, 38 + 18 * frac(input.autumn_seed * 5.3))), leafness);\n"
             "            }\n"
             "        }\n"
             "        if (has_normal)\n            normal = mapped_normal(geometric, input.world, input.uv,\n"
             "                                   foliage || repeat_address);"),
        ]
    if mountains:
        result += [
            ((MOUNTAIN,), "Output shade(P input) {", common + "Output shade(P input) {"),
            ((MOUNTAIN,), "#endif\n#ifdef SANDBOX_TERRAIN_MATERIAL\n#ifdef BEAUTY_VOLCANO_MATERIAL",
             "    albedo = autumn_ground(albedo, input.world.xy);\n"
             "#endif\n#ifdef SANDBOX_TERRAIN_MATERIAL\n#ifdef BEAUTY_VOLCANO_MATERIAL"),
        ]
    return result


def taller(height: float):
    """Proportion experiment for every forest body (all seasons), colour and shadow.

    The pack stores trees at 50% of source height; this scales that again in
    the instance transform with the matching inverse normal transform.
    """
    files = (TREES, CASTER)
    return [
        (files, " precise float height=i.place1.w+i.source_position.z*scale*z_basis*112;",
         f" precise float height=i.place1.w+i.source_position.z*scale*z_basis*112*{height};"),
        ((TREES,), "-(i.source_normal.x*si+i.source_normal.y*co),i.source_normal.z/(150.f/(.82f*64.f)));",
         f"-(i.source_normal.x*si+i.source_normal.y*co),i.source_normal.z/(150.f/(.82f*64.f)*{height}));"),
        ((TREES,), "g.z/(150.f/(.82f*64.f))))", f"g.z/(150.f/(.82f*64.f)*{height})))"),
    ]


def tiers(ocean_dark: float = 0.62, ocean_cover: float = 0.5):
    """Season-independent: Civ III ocean reads deeper than sea.

    Production water depth is distance-to-shore only and levels off ~2 tiles
    out, so sea and ocean match. The vertex's smooth per-type value
    (surface_coordinate: coast .34, sea .68, ocean 1.0) darkens ocean's final
    colour and hides more of the bed, without hard tile edges.
    """
    hydro = CITY + "hydrology.hlsl"
    return [
        ((hydro,), "float4 q3_natural_water(PixelInput input) {\n float depth=max(0,input.hydrology_data.w);",
         "float4 q3_natural_water(PixelInput input) {\n float depth=max(0,input.hydrology_data.w);\n"
         " // Lab tier cue: ocean (type value 1.0) deeper than sea (.68).\n"
         " float tier_ocean=smoothstep(.72,.97,input.surface_coordinate);"),
        ((hydro,), " return float4(premult/max(alpha,.0001),alpha);",
         f" alpha=lerp(alpha,1,{ocean_cover}*tier_ocean);\n"
         f" return float4(premult/max(alpha,.0001)*lerp(1,{ocean_dark},tier_ocean),alpha);"),
    ]


PROBE = [
    # Diagnostic: natural relief = magenta, natural mountain tiles = cyan.
    ((RELIEF,), "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction",
     "    albedo = float3(1, 0, 1);\n#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction"),
    ((MOUNTAIN,), "#endif\n#ifdef SANDBOX_TERRAIN_MATERIAL\n#ifdef BEAUTY_VOLCANO_MATERIAL",
     "    albedo = float3(0, 1, 1);\n#endif\n#ifdef SANDBOX_TERRAIN_MATERIAL\n#ifdef BEAUTY_VOLCANO_MATERIAL"),
]

LOOKS = {
    "identity": [],
    "probe": PROBE,
    "v1": edits(V1, PALETTE_V1),
    "v2": edits(V2, PALETTE_V2, turf=True, mountains=True),
    "v3": edits(V3, PALETTE_V2, turf=True, mountains=True, warm_hills=True),
    "v4": edits(V4, PALETTE_V4, turf=True, mountains=True, warm_hills=True),
}
LOOKS["v4-tall"] = LOOKS["v4"] + taller(1.4)
LOOKS["v5"] = edits(V4, PALETTE_V4, turf=True, mountains=True, warm_hills=True, water=True)
LOOKS["v6"] = edits(V6, PALETTE_V4, turf=True, mountains=True, warm_hills=True, water=True)
LOOKS["v7"] = edits(V7, PALETTE_V4, turf=True, mountains=True, warm_hills=True, water=True)
LOOKS["v8"] = edits(V8, PALETTE_V4, turf=True, mountains=True, warm_hills=True, water=True, fields=True, conifers=True)
LOOKS["v9"] = edits(V9, PALETTE_V4, turf=True, mountains=True, warm_hills=True, water=True, fields=True)
LOOKS["v10"] = edits(V10, PALETTE_V4, turf=True, mountains=True, warm_hills=True, water=True, fields=True)
LOOKS["v11"] = edits(V11, PALETTE_V4, turf=True, mountains=True, warm_hills=True, water=True, fields=True)
LOOKS["summer-tall"] = taller(1.4)
LOOKS["summer-tiers"] = tiers()
# Diagnostic: natural-water pixels painted by the vertex type value
# (yellow coast <.5, green sea .5-.84, magenta ocean >.84).
LOOKS["tier-probe"] = [
    ((CITY + "hydrology.hlsl",), "float4 q3_natural_water(PixelInput input) {\n float depth=max(0,input.hydrology_data.w);",
     "float4 q3_natural_water(PixelInput input) {\n"
     " if(true){float t=input.surface_coordinate;return float4(t<.5?float3(1,1,0):t<.84?float3(0,1,0):float3(1,0,1),1);}\n"
     " float depth=max(0,input.hydrology_data.w);"),
]
LOOKS["v11-tiers"] = LOOKS["v11"] + tiers()


def apply(look: str, relative: str, text: str) -> str:
    for paths, anchor, replacement in LOOKS[look]:
        if relative not in paths:
            continue
        if text.count(anchor) != 1:
            raise ValueError(f"{look}: anchor occurs {text.count(anchor)} times in {relative}: {anchor[:60]!r}")
        text = text.replace(anchor, replacement)
    return text
