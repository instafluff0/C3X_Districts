"""Winter shader patches for the study's private shader roots.

A look is a set of #define calibrations and anchored edits to the flattened
production shaders. An edit whose anchor is missing or ambiguous fails loudly,
so production drift can never silently drop part of a look. Colour targets
were measured on the selected concept (winter-icy-blue.png, OKLab).
"""
from __future__ import annotations

from pathlib import Path

COMMON = Path(__file__).with_name("winter_common.hlsl").read_text()
CITY = "Renderer/native/city_fidelity/"
TREES = CITY + "objects.hlsl"
RELIEF = CITY + "terrain.hlsl"
MOUNTAIN = CITY + "mountain.hlsl"

V1 = {
    "WINTER_SNOW_LOW": "float3(0.70, 0.77, 0.88)", "WINTER_SNOW_HIGH": "float3(0.84, 0.87, 0.92)",
    "WINTER_SNOW_SPECK": 0.06, "WINTER_SNOW_GLOSS": 0.35,
    "WINTER_DORMANT_CHROMA": 0.45, "WINTER_DORMANT_HUE": 75, "WINTER_DORMANT_LIGHT": 0.85,
    "WINTER_DEPTH": 0.75, "WINTER_SCOUR": 0.5, "WINTER_SCOUR_SCALE": 0.9,
    "WINTER_SHED_LOW": 0.35, "WINTER_SHED_HIGH": 0.7,
    "WINTER_DRIFT_SCALE": 2.0, "WINTER_DRIFT_RELIEF": 0.04, "WINTER_SMOOTH": 0.7,
    "WINTER_FLOOR_DEPTH": 0.55, "WINTER_STONE_DEPTH": 0.45,
    "WINTER_SKY": "float3(0.82, 0.95, 1.30)",
    "WINTER_MTN_SNOW_LOW": 0.30, "WINTER_MTN_SNOW_HIGH": 0.48, "WINTER_MTN_UP_LOW": 0.45, "WINTER_MTN_UP_HIGH": 0.75,
    "WINTER_TREE_UP_LOW": 0.15, "WINTER_TREE_UP_HIGH": 0.55, "WINTER_TREE_SNOW": "float3(0.80, 0.84, 0.90)",
    "WINTER_NEEDLE_LIGHT": 0.70, "WINTER_NEEDLE_CHROMA": 0.55, "WINTER_NEEDLE_HUE": 175,
    "WINTER_WOOD_UP_LOW": 0.35, "WINTER_WOOD_UP_HIGH": 0.70,
    "WINTER_DESERT_DEPTH": 1.0, "WINTER_CRUST_SCALE": 1.3, "WINTER_CRUST_LIGHT": 0, "WINTER_CRUST": 0,
    "WINTER_TREE_LIFT": 2.5, "WINTER_NEEDLE_PULL": 0.35,
    "WINTER_ROCK_CHROMA": 1.0, "WINTER_ROCK_LIGHT": 1.0, "WINTER_TREE_FILL": 1.0,
}


# v5 tree hook: foliage classes by atlas average (linear g/b; pack metadata in
# production). Authored snowy pines keep their own snow (cool fill only); pines,
# shrubs and jungle get bough snow over dark needles; broadleaf crowns either
# the same (WINTER_BARE 0) or a bare winter canopy: grey-brown twig mass,
# thinned so the snow below shows, a little snow caught on top.
TREES_CLASSED = (
    "    float winter_snow = 0;\n"
    "    if (kind > 0.5 && kind < 1.5) {\n"
    "        float3 lch = winter_lch(albedo);\n"
    "        float3 average = BaseColor.SampleLevel(Clamp, float2(0.5, 0.5), 15).rgb;\n"
    "        float average_light = winter_lch(average).x;\n"
    "        float gb = average.g / max(average.b, 0.001);\n"
    "        float rel = lch.x / max(average_light, 0.05);\n"
    "        float lift = (lch.x - average_light) * WINTER_TREE_LIFT;\n"
    "        if (gb < WINTER_SNOWY_GB) {\n"
    "            winter_snow = saturate((rel - 0.9) * 2);\n"
    "        } else if (gb > WINTER_BROADLEAF_GB && WINTER_BARE > 0) {\n"
    "            clip(winter_noise(input.uv * WINTER_BARE_SCALE) - WINTER_BARE);\n"
    "            float3 twig = winter_from_lch(float3(WINTER_TWIG_LIGHT * clamp(rel, 0.55, 1.45), WINTER_TWIG_CHROMA, WINTER_TWIG_HUE));\n"
    "            winter_snow = smoothstep(WINTER_TREE_UP_LOW, WINTER_TREE_UP_HIGH, normal.z + lift) * WINTER_BARE_SNOW;\n"
    "            albedo = lerp(twig, WINTER_TREE_SNOW, winter_snow);\n"
    "        } else {\n"
    "            winter_snow = smoothstep(WINTER_TREE_UP_LOW, WINTER_TREE_UP_HIGH, normal.z + lift);\n"
    "            float3 needle = winter_from_lch(float3(lch.x * WINTER_NEEDLE_LIGHT, lch.y * WINTER_NEEDLE_CHROMA,\n"
    "                lerp(lch.z, WINTER_NEEDLE_HUE, WINTER_NEEDLE_PULL)));\n"
    "            albedo = lerp(needle, WINTER_TREE_SNOW * (0.9 + 0.2 * saturate(rel - 0.5)), winter_snow);\n"
    "        }\n"
    "    } else if (kind < 0.5) {\n"
    "        winter_snow = winter_cover(1, 0.5, 0, input.world.xy);\n"
    "        albedo = lerp(winter_dormant(albedo), winter_snow_albedo(0.5, input.world.xy), winter_snow);\n"
    "    }\n"
)


# v13 relief hook: each Civ III land type keeps a winter signature (user,
# 2026-10-22: the types were indistinguishable under uniform snow, 0.1-0.6).
# Type weights come from the relief's own continuous biome inputs, so
# transitions stay soft. Per type [grassland, plains, desert, tundra]: snow
# depth, the exposed ground's lightness factor, chroma and hue. Grassland shows
# sage-green tussocks in clumps, plains golden stubble, desert tan sand on the
# crests (snow in the hollows), tundra deep, smooth, bluer snow. Hill rock
# keeps the grey dormant stone.
def relief_types(patterns=False):
    return (
    "    float winter_snow;\n"
    "    {\n"
    "        float kind = input.material.y;\n"
    "        float3 mesh_normal = normalize(input.normal);\n"
    "        float slope = 1 - saturate(mesh_normal.z);\n"
    "        float depth = kind > 2.5 && kind < 4.5 ? WINTER_FLOOR_DEPTH : (kind > 1.5 && kind < 2.5 ? WINTER_STONE_DEPTH : 1.0);\n"
    "        float w_desert = kind < 1.5 ? saturate(input.biome.y) : (kind > 6.5 && kind < 7.5 ? 1 : 0);\n"
    "        float w_plains = kind < 1.5 ? saturate(input.biome.x) : (kind > 5.5 && kind < 6.5 ? 1 : 0);\n"
    "        float w_tundra = kind < 1.5 ? saturate(input.material.w) : 0;\n"
    "        float4 types = float4(saturate(1 - w_desert - w_plains - w_tundra), w_plains, w_desert, w_tundra);\n"
    "        types /= max(dot(types, float4(1, 1, 1, 1)), 1e-4);\n"
    "        depth *= dot(types, WINTER_TYPE_DEPTH);\n"
    + ("        // Grassland tussocks gather in clumps rather than an even speckle.\n"
       "        float clump = winter_fbm(input.world.xy * WINTER_TUSSOCK_SCALE) - 0.5;\n"
       "        winter_snow = winter_cover(depth, height_detail + clump * WINTER_TUSSOCK * types.x, slope, input.world.xy);\n"
       if not patterns else
       # v14: absolute material heights mostly sat above the snow line on desert
       # (no snow at all) and below it on grass/plains (no ground). Each type now
       # exposes ground through its own pattern with a known share showing.
       "        float2 wind = float2(dot(input.world.xy, float2(0.8, 0.6)), dot(input.world.xy, float2(-0.6, 0.8)));\n"
       "        float clump = winter_fbm(input.world.xy * WINTER_TUSSOCK_SCALE);\n"
       "        float streak = winter_noise(float2(wind.x * 3.0, wind.y * 22.0)) * 0.6 +\n"
       "                       winter_noise(float2(wind.x * 6.1, wind.y * 41.0) + 5.0) * 0.4;\n"
       "        float ripple = 0.5 + 0.5 * sin(wind.y * WINTER_RIPPLE_FREQ + winter_fbm(input.world.xy * 1.7) * 6.0);\n"
       "        float shown = types.x * smoothstep(WINTER_SHOW.x - 0.05, WINTER_SHOW.x + 0.05, clump) +\n"
       "                      types.y * smoothstep(WINTER_SHOW.y - 0.05, WINTER_SHOW.y + 0.05, streak) +\n"
       "                      types.z * smoothstep(WINTER_SHOW.z - 0.12, WINTER_SHOW.z + 0.12, ripple);\n"
       "        winter_snow = winter_cover(depth, height_detail, slope, input.world.xy) * (1 - saturate(shown));\n") +
    "        float crust = (SurfaceDetail.Sample(Wrap, input.world.xy * WINTER_CRUST_SCALE + float2(0.41, 0.73)).r - 0.303) *\n"
    "            (1 - WINTER_TUNDRA_SMOOTH * types.w);\n"
    "        float3 snow = winter_snow_albedo(height_detail, input.world.xy) * (1 + crust * WINTER_CRUST_LIGHT) *\n"
    "            lerp(1, WINTER_TUNDRA_SNOW, types.w);\n"
    "        // Exposed ground keeps the source's own light/dark grain with its type's colour.\n"
    "        float3 source = winter_lch(albedo);\n"
    "        float3 exposed = winter_from_lch(float3(source.x * dot(types, WINTER_TYPE_LIGHT),\n"
    "            dot(types, WINTER_TYPE_CHROMA) * saturate(0.5 + source.y / 0.08), dot(types, WINTER_TYPE_HUE)));\n"
    "        float rock = kind < 1.5 ? input.material.z * smoothstep(0.09, 0.43, input.material.x) : (kind > 1.5 && kind < 2.5 ? 1 : 0);\n"
    "        exposed = lerp(exposed, winter_dormant(albedo), saturate(rock));\n"
    "        albedo = lerp(exposed, snow, winter_snow);\n"
    "        float3 drift = detail_normal_strength(mesh_normal, input.world,\n"
    "            winter_fbm(input.world.xy * WINTER_DRIFT_SCALE) + crust * WINTER_CRUST, WINTER_DRIFT_RELIEF);\n"
    "        geometric = normalize(lerp(geometric, drift, winter_snow * WINTER_SMOOTH));\n"
    "        specular_map = lerp(specular_map, WINTER_SNOW_GLOSS, winter_snow);\n"
    "    }\n"
    )


# v15 land signatures, shared by the relief and the mountains' ground collars.
# types = [grassland, plains, desert, tundra] from the continuous biome inputs.
# Grassland: ragged clumps of sage tussocks; plains: an even scatter of small
# dark-gold stubble flecks; desert: pale grey sand on broken, wandering ripple
# crests with snow in the troughs; tundra: deep smooth bluer snow.
SIGNATURES = """
float4 winter_types(float plains, float desert, float tundra) {
    float4 types = float4(saturate(1 - plains - desert - tundra), plains, desert, tundra);
    return types / max(dot(types, float4(1, 1, 1, 1)), 1e-4);
}
// Wandering dune ripples (1 on crests).
float winter_ripple(float2 world) {
    float turn = (winter_fbm(world * 0.6) - 0.5) * 2.4;
    float2 wind = float2(cos(turn), sin(turn));
    return 0.5 + 0.5 * sin(dot(world, wind) * WINTER_RIPPLE_FREQ + winter_fbm(world * 2.3) * 8.0);
}
float winter_type_shown(float4 types, float2 world) {
    float clump = winter_fbm(world * WINTER_TUSSOCK_SCALE) + (winter_noise(world * 35.0) - 0.5) * 0.25;
    float fleck = winter_noise(world * WINTER_STUBBLE_SCALE) * 0.7 + winter_noise(world * WINTER_STUBBLE_SCALE * 1.9 + 3.0) * 0.3;
    float ripple = winter_ripple(world);
    float broken = lerp(0.45, 1.0, smoothstep(0.35, 0.65, winter_fbm(world * 2.7 + 11.0)));
    return saturate(types.x * smoothstep(WINTER_SHOW.x - 0.04, WINTER_SHOW.x + 0.04, clump) +
                    types.y * smoothstep(WINTER_SHOW.y - 0.05, WINTER_SHOW.y + 0.05, fleck) +
                    types.z * smoothstep(WINTER_SHOW.z - 0.15, WINTER_SHOW.z + 0.15, ripple) * broken);
}
// Exposed ground keeps the source's own light/dark grain with its type's colour.
float3 winter_type_exposed(float3 albedo, float4 types) {
    float3 source = winter_lch(albedo);
    return winter_from_lch(float3(source.x * dot(types, WINTER_TYPE_LIGHT),
        dot(types, WINTER_TYPE_CHROMA) * saturate(0.5 + source.y / 0.08), dot(types, WINTER_TYPE_HUE)));
}
"""

RELIEF_SIGNATURES = (
    "    float winter_snow;\n"
    "    {\n"
    "        float kind = input.material.y;\n"
    "        float3 mesh_normal = normalize(input.normal);\n"
    "        float slope = 1 - saturate(mesh_normal.z);\n"
    "        float depth = kind > 2.5 && kind < 4.5 ? WINTER_FLOOR_DEPTH : (kind > 1.5 && kind < 2.5 ? WINTER_STONE_DEPTH : 1.0);\n"
    "        float4 types = winter_types(kind < 1.5 ? saturate(input.biome.x) : (kind > 5.5 && kind < 6.5 ? 1 : 0),\n"
    "                                    kind < 1.5 ? saturate(input.biome.y) : (kind > 6.5 && kind < 7.5 ? 1 : 0),\n"
    "                                    kind < 1.5 ? saturate(input.material.w) : 0);\n"
    "        depth *= dot(types, WINTER_TYPE_DEPTH);\n"
    "        float rock = kind < 1.5 ? input.material.z * smoothstep(0.09, 0.43, input.material.x) : (kind > 1.5 && kind < 2.5 ? 1 : 0);\n"
    "        winter_snow = winter_cover(depth, height_detail, slope, input.world.xy) *\n"
    "            (1 - winter_type_shown(types, input.world.xy) * (1 - saturate(rock)));\n"
    "        float crust = (SurfaceDetail.Sample(Wrap, input.world.xy * WINTER_CRUST_SCALE + float2(0.41, 0.73)).r - 0.303) *\n"
    "            (1 - WINTER_TUNDRA_SMOOTH * types.w);\n"
    "        float3 snow = winter_snow_albedo(height_detail, input.world.xy) * (1 + crust * WINTER_CRUST_LIGHT) *\n"
    "            lerp(1, WINTER_TUNDRA_SNOW, types.w);\n"
    "        float3 exposed = lerp(winter_type_exposed(albedo, types), winter_dormant(albedo), saturate(rock));\n"
    "        albedo = lerp(exposed, snow, winter_snow);\n"
    "        float3 drift = detail_normal_strength(mesh_normal, input.world,\n"
    "            winter_fbm(input.world.xy * WINTER_DRIFT_SCALE) + crust * WINTER_CRUST, WINTER_DRIFT_RELIEF);\n"
    "        geometric = normalize(lerp(geometric, drift, winter_snow * WINTER_SMOOTH));\n"
    "        specular_map = lerp(specular_map, WINTER_SNOW_GLOSS, winter_snow);\n"
    "    }\n"
)
# v17 (user, 2026-10-22: "so splotchy"; grassland and mountains fully covered,
# a few splotches elsewhere at most). The types separate mainly through the
# snow itself: tundra icier blue-grey with a wind crust, plains faintly warm
# cream with sparse gold stubble, desert warmer beige with dune ripples in the
# snow's relief and sand on a few crests. Mountains keep the uniform cover.
RELIEF_SUBTLE = RELIEF_SIGNATURES.replace(
    "        float crust = (SurfaceDetail.Sample(Wrap, input.world.xy * WINTER_CRUST_SCALE + float2(0.41, 0.73)).r - 0.303) *\n"
    "            (1 - WINTER_TUNDRA_SMOOTH * types.w);\n"
    "        float3 snow = winter_snow_albedo(height_detail, input.world.xy) * (1 + crust * WINTER_CRUST_LIGHT) *\n"
    "            lerp(1, WINTER_TUNDRA_SNOW, types.w);\n",
    "        float crust = (SurfaceDetail.Sample(Wrap, input.world.xy * WINTER_CRUST_SCALE + float2(0.41, 0.73)).r - 0.303) *\n"
    "            (1 + WINTER_TUNDRA_CRUST * types.w);\n"
    "        float3 snow = winter_snow_albedo(height_detail, input.world.xy) * (1 + crust * WINTER_CRUST_LIGHT) *\n"
    "            (types.x + types.y * WINTER_PLAINS_SNOW + types.z * WINTER_DESERT_SNOW + types.w * WINTER_TUNDRA_SNOW);\n"
    "        float dune = (winter_ripple(input.world.xy) - 0.5) * WINTER_DESERT_RELIEF * types.z;\n").replace(
    "            winter_fbm(input.world.xy * WINTER_DRIFT_SCALE) + crust * WINTER_CRUST, WINTER_DRIFT_RELIEF);\n",
    "            winter_fbm(input.world.xy * WINTER_DRIFT_SCALE) + crust * WINTER_CRUST + dune, WINTER_DRIFT_RELIEF);\n").replace(
    "        specular_map = lerp(specular_map, WINTER_SNOW_GLOSS, winter_snow);\n",
    "        specular_map = lerp(specular_map, WINTER_SNOW_GLOSS + WINTER_TUNDRA_GLOSS * types.w, winter_snow);\n")
assert RELIEF_SUBTLE.count("WINTER_PLAINS_SNOW") == 1 and RELIEF_SUBTLE.count("+ dune") == 1 and RELIEF_SUBTLE.count("WINTER_TUNDRA_GLOSS") == 1

MOUNTAIN_SIGNATURES = (
    "    float winter_snow;\n"
    "    {\n"
    "        float4 types = winter_types(saturate(input.biome.x), saturate(input.biome.y), saturate(input.material.w));\n"
    "        float ground_cover = winter_cover(dot(types, WINTER_TYPE_DEPTH), height_detail, 1 - saturate(normalize(input.normal).z),\n"
    "            input.world.xy) * (1 - winter_type_shown(types, input.world.xy));\n"
    "        float rock_cover = smoothstep(WINTER_MTN_UP_LOW, WINTER_MTN_UP_HIGH, normal.z) * 0.85;\n"
    "        winter_snow = lerp(ground_cover, rock_cover, rock_albedo_coverage);\n"
    "        float3 rock_lch = winter_lch(albedo) * float3(WINTER_ROCK_LIGHT, WINTER_ROCK_CHROMA, 1);\n"
    "        float3 bare = lerp(winter_type_exposed(albedo, types), winter_from_lch(rock_lch), rock_albedo_coverage);\n"
    "        float3 snow = winter_snow_albedo(height_detail, input.world.xy) * lerp(1, WINTER_TUNDRA_SNOW, types.w * (1 - rock_albedo_coverage));\n"
    "        albedo = lerp(bare, snow, winter_snow);\n"
    "        specular_map = lerp(specular_map, WINTER_SNOW_GLOSS, winter_snow);\n"
    "    }\n"
)


def edits(values, trees="laden", ground="uniform"):
    common = "".join(f"#define {key} {value}\n" for key, value in values.items()) + COMMON + \
        (SIGNATURES if ground in ("signatures", "subtle") else "")
    result = [
        # ---- Relief: land, hills, hill stone, forest floor and ground decals.
        ((RELIEF, TREES, MOUNTAIN), "Output shade(P input) {", common + "Output shade(P input) {"),
        ((RELIEF,), "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction",
         "    float winter_snow;\n"
         "    {\n"
         "        float kind = input.material.y;\n"
         "        float3 mesh_normal = normalize(input.normal);\n"
         "        float slope = 1 - saturate(mesh_normal.z);\n"
         "        float depth = kind > 2.5 && kind < 4.5 ? WINTER_FLOOR_DEPTH : (kind > 1.5 && kind < 2.5 ? WINTER_STONE_DEPTH : 1.0);\n"
         "        // Wind keeps open desert thinly covered; its ripples and sand show.\n"
         "        float desert = kind < 1.5 ? saturate(input.biome.y) : (kind > 6.5 && kind < 7.5 ? 1 : 0);\n"
         "        depth *= lerp(1, WINTER_DESERT_DEPTH, desert);\n"
         "        winter_snow = winter_cover(depth, height_detail, slope, input.world.xy);\n"
         "        // Authored surface detail (mean .303) gives the crust its grain.\n"
         "        float crust = SurfaceDetail.Sample(Wrap, input.world.xy * WINTER_CRUST_SCALE + float2(0.41, 0.73)).r - 0.303;\n"
         "        float3 snow = winter_snow_albedo(height_detail, input.world.xy) * (1 + crust * WINTER_CRUST_LIGHT);\n"
         "        albedo = lerp(winter_dormant(albedo), snow, winter_snow);\n"
         "        // Snow buries the grain: soft drifts and crust replace most micro relief.\n"
         "        float3 drift = detail_normal_strength(mesh_normal, input.world,\n"
         "            winter_fbm(input.world.xy * WINTER_DRIFT_SCALE) + crust * WINTER_CRUST, WINTER_DRIFT_RELIEF);\n"
         "        geometric = normalize(lerp(geometric, drift, winter_snow * WINTER_SMOOTH));\n"
         "        specular_map = lerp(specular_map, WINTER_SNOW_GLOSS, winter_snow);\n"
         "    }\n"
         "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction"),
        ((RELIEF,), "    float3 ambient = Ambient.rgb * Ambient.a * lerp(0.56, 1.0, sky) * cavity * surface_occlusion;",
         "    float3 ambient = Ambient.rgb * Ambient.a * lerp(0.56, 1.0, sky) * cavity * surface_occlusion *\n"
         "        lerp(1, WINTER_SKY, winter_snow);"),
        # ---- Mountains: every mountain snow-capped, lower and fuller cover.
        ((MOUNTAIN,), "        float snow_cap = saturate((input.material.y - 2) * 2);\n"
                      "        float snow = smoothstep(0.62, 0.78, height) * smoothstep(MTN_SNOW_SLOPE_LOW, MTN_SNOW_SLOPE_HIGH, geometric.z) * snow_cap;",
         "        float snow_cap = 1;\n"
         "        float snow = smoothstep(WINTER_MTN_SNOW_LOW, WINTER_MTN_SNOW_HIGH, height) *\n"
         "            smoothstep(WINTER_MTN_UP_LOW, WINTER_MTN_UP_HIGH, geometric.z) * snow_cap;"),
        ((MOUNTAIN,), "#endif\n#ifdef SANDBOX_TERRAIN_MATERIAL\n#ifdef BEAUTY_VOLCANO_MATERIAL",
         "    float winter_snow;\n"
         "    {\n"
         "        float ground_cover = winter_cover(1, height_detail, 1 - saturate(normalize(input.normal).z), input.world.xy);\n"
         "        float rock_cover = smoothstep(WINTER_MTN_UP_LOW, WINTER_MTN_UP_HIGH, normal.z) * 0.85;\n"
         "        winter_snow = lerp(ground_cover, rock_cover, rock_albedo_coverage);\n"
         "        float3 rock_lch = winter_lch(albedo) * float3(WINTER_ROCK_LIGHT, WINTER_ROCK_CHROMA, 1);\n"
         "        float3 bare = lerp(winter_dormant(albedo), winter_from_lch(rock_lch), rock_albedo_coverage);\n"
         "        albedo = lerp(bare, winter_snow_albedo(height_detail, input.world.xy), winter_snow);\n"
         "        specular_map = lerp(specular_map, WINTER_SNOW_GLOSS, winter_snow);\n"
         "    }\n"
         "#endif\n#ifdef SANDBOX_TERRAIN_MATERIAL\n#ifdef BEAUTY_VOLCANO_MATERIAL"),
        ((MOUNTAIN,), "    float3 ambient = Ambient.rgb * Ambient.a * lerp(ambient_floor, 1.0, sky) * cavity * crevice_visibility;",
         "    float3 ambient = Ambient.rgb * Ambient.a * lerp(ambient_floor, 1.0, sky) * cavity * crevice_visibility *\n"
         "        lerp(1, WINTER_SKY, winter_snow);"),
        # ---- Trees: snow on up-facing boughs and wood; dark cool needles beneath.
        ((TREES,), "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Use the exact vector that projects the Q6 field.",
         "    float winter_snow = 0;\n"
         "    if (kind > 0.5 && kind < 1.5) {\n"
         "        float3 lch = winter_lch(albedo);\n"
         "        float3 average = BaseColor.SampleLevel(Clamp, float2(0.5, 0.5), 15).rgb;\n"
         "        float lift = (lch.x - winter_lch(average).x) * WINTER_TREE_LIFT;\n"
         "        winter_snow = smoothstep(WINTER_TREE_UP_LOW, WINTER_TREE_UP_HIGH, normal.z + lift);\n"
         "        float3 needle = winter_from_lch(float3(lch.x * WINTER_NEEDLE_LIGHT, lch.y * WINTER_NEEDLE_CHROMA,\n"
         "            lerp(lch.z, WINTER_NEEDLE_HUE, WINTER_NEEDLE_PULL)));\n"
         "        albedo = lerp(needle, WINTER_TREE_SNOW * (0.9 + 0.2 * saturate(lch.x / max(winter_lch(average).x, 0.05) - 0.5)), winter_snow);\n"
         "    } else if (kind > 1.5) {\n"
         "        winter_snow = smoothstep(WINTER_WOOD_UP_LOW, WINTER_WOOD_UP_HIGH, normal.z);\n"
         "        albedo = lerp(albedo, WINTER_TREE_SNOW, winter_snow);\n"
         "    } else {\n"
         "        winter_snow = winter_cover(1, 0.5, 0, input.world.xy);\n"
         "        albedo = lerp(winter_dormant(albedo), winter_snow_albedo(0.5, input.world.xy), winter_snow);\n"
         "    }\n"
         "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Use the exact vector that projects the Q6 field."),
        ((TREES,), "    float3 ambient = Ambient.rgb * Ambient.a * lerp(0.48, 1.0, sky) * ao * foliage_fill;",
         "    float3 ambient = Ambient.rgb * Ambient.a * lerp(0.48, 1.0, sky) * ao * foliage_fill *\n"
         "        lerp(1, WINTER_SKY, winter_snow) * WINTER_TREE_FILL;"),
    ]
    if ground == "signatures":
        relief_anchor = "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction"
        mountain_anchor = "#endif\n#ifdef SANDBOX_TERRAIN_MATERIAL\n#ifdef BEAUTY_VOLCANO_MATERIAL"
        result = [(paths, a, RELIEF_SIGNATURES + a if paths == (RELIEF,) and a == relief_anchor else
                   MOUNTAIN_SIGNATURES + a if paths == (MOUNTAIN,) and a == mountain_anchor else r)
                  for paths, a, r in result]
    if ground == "subtle":
        anchor = "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction"
        result = [(paths, a, RELIEF_SUBTLE + a if paths == (RELIEF,) and a == anchor else r) for paths, a, r in result]
    if ground in ("types", "patterns"):
        anchor = "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction"
        result = [(paths, a, relief_types(ground == "patterns") + anchor if paths == (RELIEF,) and a == anchor else r)
                  for paths, a, r in result]
    if trees == "classed":
        # Hue lock: keep the tree pass's own shading (luminance) but give snow
        # the concept's cool white, bluer where it is unlit.
        result.append(((TREES,), "    output.color = float4(max(radiance, 0), 1);",
            "    {\n"
            "        float3 winter_w = float3(0.2126, 0.7152, 0.0722);\n"
            "        float3 tint = lerp(WINTER_TREE_SHADE_TINT, WINTER_TREE_LIT_TINT, saturate(diffuse * shadow));\n"
            "        radiance = lerp(radiance, dot(radiance, winter_w) * tint / dot(tint, winter_w), winter_snow * WINTER_TREE_LOCK);\n"
            "    }\n"
            "    output.color = float4(max(radiance, 0), 1);"))
        anchor = "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Use the exact vector that projects the Q6 field."
        result = [(paths, a, TREES_CLASSED + anchor if a == anchor else r) for paths, a, r in result]
    return result


INTEGRATED = tuple(CITY + name for name in ("hydrology.hlsl", "feature.hlsl", "city.hlsl",
                                             "rigid_feature.hlsl", "resource_body.hlsl"))
# Farm kit fields, resources, rocks, plants and city bodies draw through these.
FIELD_SHADERS = tuple(CITY + name for name in ("feature.hlsl", "city.hlsl", "rigid_feature.hlsl",
                                                "resource_body.hlsl"))
# Integrated shaders have Civ VI's authored snow albedo bound at t8.
INTEGRATED_COMMON = """
// Authored snow albedo (mountain_snow_texture, t8), normalised to the study's
// snow colour so relief (procedural) and integrated surfaces meet seamlessly.
float3 winter_snow_tex(float2 world) {
    float3 tex = mountain_snow_texture.Sample(material_sampler, world * WINTER_SNOW_TEX).rgb;
    float3 mean = mountain_snow_texture.SampleLevel(material_sampler, float2(0.5, 0.5), 11).rgb;
    float detail = winter_luma(tex) / max(winter_luma(mean), 0.05) - 1;
    return WINTER_SNOW_HIGH * (1 + detail * WINTER_SNOW_TEX_DETAIL);
}
// Winter fields: snow lies in the furrows and the ridges show through as dark
// soil or pale stubble, chosen per plot-sized cell; the crop rows survive.
float3 winter_field(float3 albedo, float2 world, float3 snow) {
    float3 lch = winter_lch(albedo);
    float pick = winter_hash(floor(world * WINTER_FIELD_CELLS) + 0.37);
    float ridge = saturate((lch.x - WINTER_FIELD_MID) * WINTER_FIELD_CONTRAST + 0.5);
    float depth = lerp(WINTER_FIELD_COVER_LOW, WINTER_FIELD_COVER_HIGH, pick);
    float cover = smoothstep(depth - 0.25, depth + 0.25, 1 - ridge);
    float3 soil = pick < WINTER_FIELD_STUBBLE ? WINTER_FIELD_STRAW : WINTER_FIELD_SOIL;
    soil *= 0.75 + 0.5 * ridge;
    return lerp(soil, snow, cover);
}
// Second field pass keyed on the farm kit's own structure (no world cells, so
// no seams inside a plot): tan field paths become dark hedgerow lines, ripe
// (yellow) plots pale stubble, green plots ploughed soil; snow settles in the
// furrows. The farm ground around the plots is mostly snow.
float3 winter_field_kit(float3 albedo, float2 world, float3 snow, bool ground) {
    float3 lch = winter_lch(albedo);
    float speck = winter_noise(world * 61.0);
    if (ground)
        return lerp(winter_dormant(albedo), snow, saturate(WINTER_FIELD_GROUND + (speck - 0.5) * 0.3));
    float path = smoothstep(-0.01, 0.04, albedo.r - albedo.g);
    float ripe = smoothstep(108, 96, lch.z);
    float ridge = saturate((lch.x - WINTER_FIELD_MID) * WINTER_FIELD_CONTRAST + 0.5);
    float depth = lerp(WINTER_FIELD_COVER_LOW, WINTER_FIELD_COVER_HIGH, ripe);
    float cover = smoothstep(depth - 0.25, depth + 0.25, 1 - ridge + (speck - 0.5) * 0.2);
    float3 soil = lerp(WINTER_FIELD_SOIL, WINTER_FIELD_STRAW, ripe) * (0.8 + 0.4 * ridge);
    float3 hedge = lerp(WINTER_FIELD_HEDGE, snow, WINTER_FIELD_HEDGE_SNOW * speck);
    return lerp(lerp(soil, snow, cover), hedge, path);
}
// Third field pass: rows against each plot's own blurred mean (the crop
// texture resampled at a higher mip), and a per-plot state picked from that
// mean colour: fallow under snow with faint rows, ploughed (dark furrows), or
// straw stubble. Tan field paths stay as dark hedgerow lines.
float3 winter_field_plot(float3 albedo, float3 mean, float2 world, float3 snow) {
    float3 lch = winter_lch(albedo), plot = winter_lch(mean);
    float speck = winter_noise(world * 61.0);
    float pick = frac(plot.x * 53.1 + plot.z * 0.37 + plot.y * 91.7);
    float path = smoothstep(-0.01, 0.04, albedo.r - albedo.g);
    float ridge = saturate((lch.x - plot.x) * WINTER_FIELD_CONTRAST + 0.5);
    float depth = pick < WINTER_FIELD_FALLOW ? WINTER_FIELD_COVER_HIGH : WINTER_FIELD_COVER_LOW;
    float3 soil = pick < WINTER_FIELD_STUBBLE ? WINTER_FIELD_SOIL : WINTER_FIELD_STRAW;
    float cover = smoothstep(depth - 0.2, depth + 0.2, 1 - ridge + (speck - 0.5) * 0.15);
    soil *= 0.8 + 0.4 * ridge;
    float3 hedge = lerp(WINTER_FIELD_HEDGE, snow, WINTER_FIELD_HEDGE_SNOW * speck);
    return lerp(lerp(soil, snow * WINTER_FIELD_SNOW_GAIN, cover), hedge, path);
}
"""


# Icy rivers (user, 2026-10-21: "more icy looking to match the color of
# oceans"): the open channel takes the sea's winter blue; ice shelves grow in
# from both banks with a broken edge, a few floes drift mid-channel, and the
# banks carry snow. distance_pixels runs from the centre line (waterline ~5.8).
RIVER_ICE = (
    "  {\n"
    "    float3 river_lch = winter_lch(surface);\n"
    "    float river_hue = river_lch.z < 0 ? river_lch.z + 360 : river_lch.z;\n"
    "    // Like the sea's, the recolour fades at dusk and night (v12; v9-v11 kept it on).\n"
    "    float river_day = lerp(1, lerp(WINTER_WATER_NIGHT, 1, saturate(environment_sun_intensity * 2.5)), WINTER_RIVER_DAYLIGHT);\n"
    "    river_lch.z = lerp(river_hue, lerp(river_hue, WINTER_WATER_HUE, WINTER_WATER_PULL), river_day);\n"
    "    river_lch.y = lerp(river_lch.y, max(river_lch.y * WINTER_RIVER_CHROMA, WINTER_RIVER_MIN_CHROMA), river_day);\n"
    "    river_lch.x *= lerp(1, WINTER_RIVER_LIGHT, river_day);\n"
    "    float3 river_open = winter_from_lch(river_lch);\n"
    "    float shelf_edge = winter_noise(world * WINTER_RIVER_ICE_SCALE) * 0.65 + winter_noise(world * WINTER_RIVER_ICE_SCALE * 3.1 + 7.7) * 0.35;\n"
    "    float shelf = smoothstep(WINTER_RIVER_ICE_IN, WINTER_RIVER_ICE_IN + 0.6, distance_pixels + (shelf_edge - 0.5) * WINTER_RIVER_ICE_BREAK);\n"
    "    float floe = smoothstep(WINTER_RIVER_FLOE, WINTER_RIVER_FLOE + 0.04, winter_noise(world * WINTER_RIVER_FLOE_SCALE + 3.3)) *\n"
    "        (1 - smoothstep(1.5, 3.0, distance_pixels));\n"
    "    float river_ice = saturate(max(shelf, floe * 0.85));\n"
    "    // Ice is pale cyan-white, a little greyer where thin, lit as flat ground.\n"
    "    float3 ice_tone = lerp(WINTER_RIVER_ICE_THIN, winter_snow_tex(world), smoothstep(0, 1.2, distance_pixels - WINTER_RIVER_ICE_IN));\n"
    "    float3 ice_lit = ice_tone * q6_receiver_illumination(input, float3(0, 0, 1), 1, 1);\n"
    "    surface = lerp(river_open, ice_lit + specular * 0.3, river_ice * WINTER_RIVER_ICE);\n"
    "    shore = lerp(winter_dormant(shore), winter_snow_tex(world), WINTER_RIVER_BANK_SNOW);\n"
    "  }\n"
    "  return float4(lerp(shore*bank_light,surface,water),bank*outlet);"
)
# v24 river banks (user, 2026-10-22: the grey bank "reads like an un-textured
# gray line"). The band was v9's shelf ice: a fixed 2.2 px strip on both sides
# of every reach, and grey because integrated shaders light snow about 1/1.4 as
# bright as the relief does. Now shelf ice grows from the banks in some
# reaches and not others, with a scalloped edge; snow on it continues the
# ground's (gain-matched), ending in a thin band of bare glassy ice with a few
# bare windows. Open reaches meet the snow with a broken, faintly wet rim.
RIVER_BANKS = RIVER_ICE.split("    float shelf_edge")[0] + (
    "    float3 flat_light = q6_receiver_illumination(input, float3(0, 0, 1), 1, 1);\n"
    "    float3 snow_lit = winter_snow_tex(world) * WINTER_RIVER_SNOW_GAIN * flat_light;\n"
    "    float reach_ice = smoothstep(WINTER_SHELF_OPEN, WINTER_SHELF_OPEN + 0.25, winter_noise(world * WINTER_SHELF_REACH + 5.7));\n"
    "    float scallop = winter_noise(world * WINTER_SHELF_SCALLOP) * 0.7 + winter_noise(world * WINTER_SHELF_SCALLOP * 2.9 + 1.3) * 0.3 - 0.5;\n"
    "    float shelf_from = water_width + 0.5 - reach_ice * (WINTER_SHELF_WIDTH * (0.45 + 0.55 * winter_noise(world * WINTER_SHELF_REACH * 2.3 + 2.1))\n"
    "        - scallop * WINTER_SHELF_BREAK);\n"
    "    float shelf = smoothstep(shelf_from - 0.2, shelf_from + 0.2, shoreline);\n"
    "    float floe = smoothstep(WINTER_RIVER_FLOE, WINTER_RIVER_FLOE + 0.04, winter_noise(world * WINTER_RIVER_FLOE_SCALE + 3.3)) *\n"
    "        (1 - smoothstep(1.5, 3.0, distance_pixels));\n"
    "    float river_ice = saturate(max(shelf, floe * 0.85));\n"
    "    float glass = 1 - smoothstep(shelf_from + 0.15, shelf_from + WINTER_SHELF_GLASS, shoreline);\n"
    "    float window = smoothstep(0.31, 0.24, winter_fbm(world * WINTER_SHELF_WINDOW + 9.1));\n"
    "    float3 bare_lit = lerp(river_open, WINTER_RIVER_ICE_THIN * flat_light, 0.6);\n"
    "    float3 ice_lit = lerp(snow_lit, bare_lit, saturate(max(glass * shelf, window * 0.75)) * (1 - floe));\n"
    "    surface = lerp(river_open, ice_lit + specular * 0.3, river_ice * WINTER_RIVER_ICE);\n"
    "    float open_bank = 1 - reach_ice;\n"
    "    float rim = open_bank * (1 - smoothstep(water_width + 0.1, water_width + 0.8 + sediment_noise * 0.9, shoreline)) *\n"
    "        smoothstep(0.35, 0.55, sediment_noise * 0.6 + (shore_noise + 0.5) * 0.4) * WINTER_RIVER_RIM;\n"
    "    shore = winter_snow_tex(world) * WINTER_RIVER_SNOW_GAIN * lerp(float3(1, 1, 1), WINTER_RIVER_RIM_TONE, rim);\n"
    "  }\n"
    "  return float4(lerp(shore*bank_light,surface,water),bank*outlet);"
)


# v22 roofs (user, 2026-10-22: "too dull a gray ... lacks texture"). The city
# look's soft shoulder (~.42, for whitewash) squeezed roof snow to about half
# the ground snow's brightness; the old coat also flattened the roof relief
# and sampled the snow texture at map scale. Now: the roof's tile/thatch relief
# mostly survives, crust at building scale, bluer and darker in the roof's own
# creases (AO), and the snow is lit like the ground's, outside the city look.
ROOFS_LIT = (
    " float winter_roof=0;float3 winter_roof_snow=0;float3 winter_roof_n=n;\n"
    " {\n"
    "  float3 winter_up=normalize(p.geometry_normal);\n"
    "  float2 rw=p.q6_world.xy;\n"
    "  winter_roof=saturate(smoothstep(WINTER_ROOF_LOW,WINTER_ROOF_HIGH,winter_up.z)*(WINTER_ROOF_COVER+(winter_noise(rw*WINTER_ROOF_NOISE)-.5)*.5));\n"
    "  float3 fine=mountain_snow_texture.Sample(material_sampler,rw*WINTER_ROOF_TEX).rgb;\n"
    "  float3 mean=mountain_snow_texture.SampleLevel(material_sampler,float2(.5,.5),11).rgb;\n"
    "  float crust=winter_luma(fine)/max(winter_luma(mean),.05)-1+(winter_noise(rw*WINTER_ROOF_TEX*7.3)-.5)*.25;\n"
    "  winter_roof_snow=WINTER_SNOW_HIGH*(1+crust*WINTER_ROOF_DETAIL)*lerp(WINTER_ROOF_CREASE,float3(1,1,1),saturate(ao*1.2));\n"
    "  winter_roof_n=normalize(lerp(n,winter_up,winter_roof*WINTER_ROOF_FLATTEN));\n"
    "  base=lerp(base,winter_roof_snow,winter_roof);\n"
    "  n=winter_roof_n;\n"
    "  metalness*=1-winter_roof;\n"
    " }\n"
    " float3 lit=base*(1-metalness)*q6_receiver_illumination(p,n,1,ao);"
)
# v23 roofs: v22 still measured ~60% of the ground snow's brightness (the
# baked city AO darkens whole roofs) and its grain (2-3 px) vanished in the
# downsample. Now: snow lit with little of the baked AO and calibrated to the
# ground, the roof faces keep their own light and shade, mottling at building
# scale (~10 px), thinner bluer snow round a few bare dabs where the roof
# shows through, more of them on steep faces.
ROOFS_TEXTURED = (
    " float winter_roof=0;float3 winter_roof_snow=0;float3 winter_roof_n=n;float winter_roof_ao=lerp(1,ao,WINTER_ROOF_AO);\n"
    " {\n"
    "  float3 winter_up=normalize(p.geometry_normal);\n"
    "  float2 rw=p.q6_world.xy;\n"
    "  float slope=smoothstep(WINTER_ROOF_LOW,WINTER_ROOF_HIGH,winter_up.z);\n"
    "  float dab=winter_noise(rw*WINTER_ROOF_DAB+11.3),edge=WINTER_ROOF_BARE+(1-slope)*.3;\n"
    "  winter_roof=slope*smoothstep(edge-.03,edge+.03,dab);\n"
    "  float thin=1-smoothstep(edge,edge+.14,dab);\n"
    "  float mottle=winter_fbm(rw*WINTER_ROOF_MOTTLE)-.5;\n"
    "  winter_roof_snow=WINTER_SNOW_HIGH*(1+mottle*WINTER_ROOF_DETAIL)*lerp(float3(1,1,1),WINTER_ROOF_CREASE,thin*.5)\n"
    "   *lerp(WINTER_ROOF_CREASE,float3(1,1,1),saturate(ao*1.2));\n"
    "  winter_roof_n=normalize(lerp(n,winter_up,winter_roof*WINTER_ROOF_FLATTEN));\n"
    "  base=lerp(base,winter_roof_snow,winter_roof);\n"
    "  n=winter_roof_n;\n"
    "  metalness*=1-winter_roof;\n"
    " }\n"
    " float3 lit=base*(1-metalness)*q6_receiver_illumination(p,n,1,ao);"
)


def integrated(values, fields="cells", ice=False, rocks=False, rivers="tint", water="pull", roofs="coat"):
    common = "".join(f"#define {key} {value}\n" for key, value in values.items()) + COMMON + INTEGRATED_COMMON
    extra = [((CITY + "hydrology.hlsl",), "  return float4(lerp(shore*bank_light,surface,water),bank*outlet);",
               RIVER_BANKS if rivers == "banks" else RIVER_ICE)] if rivers in ("ice", "banks") else []
    if roofs in ("lit", "textured"):
        roof_ao = "ao" if roofs == "lit" else "winter_roof_ao"
        extra += [(FIELD_SHADERS, " lit=q8_city_look(lit,Q8_CITY_LOOK*(1-saturate(environment_night_activation)),base,Q8_CITY_LOOK_PALE);",
                   " lit=q8_city_look(lit,Q8_CITY_LOOK*(1-saturate(environment_night_activation)),base,Q8_CITY_LOOK_PALE);\n"
                   f" lit=lerp(lit,winter_roof_snow*q6_receiver_illumination(p,winter_roof_n,1,{roof_ao})*WINTER_ROOF_GAIN,winter_roof);")]
    extra += [
        # Coastal rocks: cooler grey stone with snow on their tops.
        (INTEGRATED, "    if(abs(input.material_index-.48)<.001)return q4_coastal_rock(input);",
         "    if(abs(input.material_index-.48)<.001) {\n"
         "        float4 rock = q4_coastal_rock(input);\n"
         "        float3 rock_up = normalize(input.geometry_normal);\n"
         "        float rock_snow = saturate(smoothstep(WINTER_SHORE_ROCK_LOW, WINTER_SHORE_ROCK_HIGH, rock_up.z) *\n"
         "            (0.7 + 0.6 * winter_noise(input.q6_world.xy * 23.0)) * WINTER_SHORE_ROCK_COVER);\n"
         "        rock.rgb = lerp(rock.rgb, winter_luma(rock.rgb).xxx, WINTER_ROCK_GREY);\n"
         "        rock.rgb = lerp(rock.rgb, winter_snow_tex(input.q6_world.xy) * q6_receiver_illumination(input, rock_up, 1, 1) *\n"
         "            frame_output_exposure(), rock_snow * rock.a);\n"
         "        return rock;\n"
         "    }"),
    ] if rocks else []
    result = extra + [
        (INTEGRATED, "float4 q6_raw_feature(FeaturePixelInput input)\n{", common + "float4 q6_raw_feature(FeaturePixelInput input)\n{"),
        # Terrain underlay drawn by the integrated shaders: beaches, banks, collars.
        (INTEGRATED, "#ifdef Q8_DEBUG_ALBEDO\n        return float4(albedo,1);",
         "        {\n"
         "            float winter_h = winter_fbm(world_position * 3.1);\n"
         "            float winter_c = winter_cover(1, winter_h * 0.9, 1 - saturate(geometry_normal.z), world_position);\n"
         "            albedo = lerp(winter_dormant(albedo), winter_snow_tex(world_position), winter_c);\n"
         "            material_normal = normalize(lerp(material_normal, geometry_normal, winter_c * WINTER_SMOOTH));\n"
         "        }\n"
         "#ifdef Q8_DEBUG_ALBEDO\n        return float4(albedo,1);"),
        # Farm fields, flat ground decals, rocks, resources, small plants (units excluded).
        (FIELD_SHADERS, "    albedo = lerp(albedo, float3(0.13, 0.115, 0.055), pollution_weight);\n"
                        "    float3 normal = normalize(input.geometry_normal);",
         "    albedo = lerp(albedo, float3(0.13, 0.115, 0.055), pollution_weight);\n"
         "    {\n"
         "        float2 winter_world = input.q6_world.xy;\n"
         "        float3 winter_up = normalize(input.geometry_normal);\n"
         "        float3 winter_white = winter_snow_tex(winter_world);\n"
         "        if (farm_kit_field_weight > 0.5)\n"
         + ("            albedo = winter_field(albedo, winter_world, winter_white);\n" if fields == "cells" else
            "            albedo = winter_field_kit(albedo, winter_world, winter_white, mine_slot > 2.5 && mine_slot < 3.5);\n"
            if fields == "kit" else
            "            albedo = mine_slot > 2.5 && mine_slot < 3.5 ?\n"
            "                winter_field_kit(albedo, winter_world, winter_white, true) :\n"
            "                winter_field_plot(albedo, resource_base_texture_0.SampleBias(material_sampler, input.uv, WINTER_FIELD_BIAS).rgb,\n"
            "                                  winter_world, winter_white);\n") +
         "        else if (unit_weight < 0.5) {\n"
         "            float flat_decal = saturate(resource_decal_weight + ground_state_weight);\n"
         "            float cover = lerp(smoothstep(WINTER_OBJ_UP_LOW, WINTER_OBJ_UP_HIGH, winter_up.z) * WINTER_OBJ_COVER,\n"
         "                               WINTER_DECAL_COVER, flat_decal);\n"
         "            cover *= 0.7 + 0.6 * winter_noise(winter_world * 23.0);\n"
         "            albedo = lerp(lerp(albedo, winter_dormant(albedo), WINTER_OBJ_DORMANT), winter_white, saturate(cover));\n"
         "        }\n"
         "    }\n"
         "    float3 normal = normalize(input.geometry_normal);"),
        # City building bodies: a snow coat on roofs and ledges.
        (FIELD_SHADERS, " float3 lit=base*(1-metalness)*q6_receiver_illumination(p,n,1,ao);",
         " {\n"
         "  float3 winter_up=normalize(p.geometry_normal);\n"
         "  float roof=smoothstep(WINTER_ROOF_LOW,WINTER_ROOF_HIGH,winter_up.z)*(WINTER_ROOF_COVER+(winter_noise(p.q6_world.xy*WINTER_ROOF_NOISE)-.5)*.5);\n"
         "  roof=saturate(roof);\n"
         "  base=lerp(base,winter_snow_tex(p.q6_world.xy),roof);\n"
         "  n=normalize(lerp(n,winter_up,roof*.8));\n"
         "  metalness*=1-roof;\n"
         " }\n"
         " float3 lit=base*(1-metalness)*q6_receiver_illumination(p,n,1,ao);"),
        # Settlement ground plates: trodden snow over the streets and yards.
        (FIELD_SHADERS, "  float3 ground_lit=ground.rgb*q6_receiver_illumination(p,ground_normal,1,1);",
         "  ground.rgb=lerp(winter_dormant(ground.rgb),winter_snow_tex(p.q6_world.xy)*WINTER_PLATE_SNOW,\n"
         "   saturate(WINTER_PLATE_COVER*(0.65+0.7*winter_noise(p.q6_world.xy*17.0))));\n"
         "  float3 ground_lit=ground.rgb*q6_receiver_illumination(p,ground_normal,1,1);"),
        # Dirt roads: darker trodden slush with snow along the verges.
        ((CITY + "hydrology.hlsl",), "                lerp(0.90, 1.06, pattern_ground);\n"
                                     "            return float4(frame_tone_map(track_albedo * track_light *",
         "                lerp(0.90, 1.06, pattern_ground);\n"
         "            track_albedo = lerp(winter_dormant(track_albedo) * WINTER_ROAD_DARK,\n"
         "                winter_snow_tex(input.macro_uv * 2.0) * 0.85, WINTER_ROAD_SNOW * (1 - coverage));\n"
         "            return float4(frame_tone_map(track_albedo * track_light *"),
        # Railway beds: the earthy bed becomes snow; ballast and steel stay.
        ((CITY + "hydrology.hlsl",), "            float3 dirt = float3(0.32, 0.20, 0.11) * lerp(0.90, 1.06, pattern_ground);",
         "            float3 dirt = lerp(float3(0.32, 0.20, 0.11), winter_snow_tex(input.macro_uv * 2.0) * 0.9, WINTER_RAIL_SNOW) *\n"
         "                lerp(0.90, 1.06, pattern_ground);"),
        # Open water: a deeper, more saturated winter blue; rivers darker.
        ((CITY + "hydrology.hlsl", CITY + "feature.hlsl", CITY + "city.hlsl"), "        return q3_water_material(input);\n#endif",
         "    {\n        float4 winter_surface = q3_water_material(input);\n"
         + ("        if (input.surface_kind > 3.5 && input.surface_kind < 4.5) {\n"
            "            // Shore and lake beds: snow at the waterline, pale ice further out;\n"
            "            // the shallow water drawn over them keeps its own tint.\n"
            "            float3 bed_light = q6_receiver_illumination(input, float3(0, 0, 1), 1, 1);\n"
            "            float bed_depth = saturate(max(0, input.hydrology_data.w) / WINTER_ICE_DEPTH);\n"
            "            float3 bed_ice = lerp(winter_snow_tex(input.macro_uv * 2.0), WINTER_ICE_SHALLOW,\n"
            "                smoothstep(0.15, 1, bed_depth + (winter_noise(input.macro_uv * 37.0) - 0.5) * 0.3));\n"
            "            winter_surface.rgb = lerp(winter_surface.rgb, bed_ice * bed_light, WINTER_ICE_BED);\n"
            "        }\n" if ice else "") +
         "        float winter_river = input.surface_kind > 8.5 && input.surface_kind < 9.5 ? 1 : 0;\n"
         "        float3 lch = winter_lch(winter_surface.rgb);\n"
         "        float hue = lch.z < 0 ? lch.z + 360 : lch.z;\n"
         "        float gate = smoothstep(150, 185, hue) * (1 - smoothstep(280, 300, hue)) *\n"
         "            lerp(WINTER_WATER_NIGHT, 1, saturate(environment_sun_intensity * 2.5)) *\n"
         + ("            (1 - winter_river);\n" if rivers in ("ice", "banks") else "            1;\n")
         + ("        lch.z = lerp(hue, WINTER_WATER_HUE, gate * WINTER_WATER_PULL);\n"
            "        lch.y *= lerp(1, lerp(WINTER_WATER_CHROMA, WINTER_RIVER_CHROMA, winter_river), gate);\n"
            "        lch.x *= lerp(1, lerp(WINTER_WATER_LIGHT, WINTER_RIVER_LIGHT, winter_river), gate);\n"
            if water == "pull" else
            # Coast, Sea and Ocean stay apart: every water hue turns by the same
            # angle (differences kept), chroma scales, and lightness steps are
            # kept or widened around a pivot rather than compressed.
            "        lch.z = hue + (WINTER_WATER_HUE - WINTER_WATER_REF_HUE) * gate;\n"
            "        lch.y *= lerp(1, WINTER_WATER_CHROMA, gate);\n"
            "        lch.x = lerp(lch.x, WINTER_WATER_PIVOT * WINTER_WATER_LIGHT + (lch.x - WINTER_WATER_PIVOT) * WINTER_WATER_CONTRAST, gate);\n"
            # The shore-distance depth saturates ~2 tiles out, so Sea and Ocean
            # share one body colour (summer separation 1.2). The tile's Civ III
            # class (real terrain 11/12/13) sets a lightness and hue step.
            "        float winter_coast = 1 - step(0.25, abs(input.real_terrain - 11));\n"
            "        float winter_ocean = 1 - step(0.25, abs(input.real_terrain - 13));\n"
            "        lch.x += (winter_coast * WINTER_COAST_LIFT - winter_ocean * WINTER_OCEAN_DROP) * gate;\n"
            "        lch.z += (winter_coast * WINTER_COAST_HUE + winter_ocean * WINTER_OCEAN_HUE) * gate;\n"
            # Smooth part: the interpolated depth inverts exactly to tiles from
            # shore (field.h: depth=.46*(1-exp(-d/.85))); open water darkens
            # gradually outward, so the class step can stay mild.
            "        float winter_offshore = -0.85 * log(max(1 - saturate(max(0, input.hydrology_data.w) / 0.46), 1e-4));\n"
            "        float winter_deep = smoothstep(WINTER_DEEP_NEAR, WINTER_DEEP_FAR, winter_offshore) * (1 - winter_river);\n"
            "        lch.x -= winter_deep * WINTER_DEEP_DROP * gate;\n"
            "        lch.z += winter_deep * WINTER_DEEP_HUE * gate;\n") +
         "        return float4(winter_from_lch(lch), winter_surface.a);\n    }\n#endif"),
    ]
    if roofs in ("lit", "textured"):
        roof = ROOFS_LIT if roofs == "lit" else ROOFS_TEXTURED
        result = [(paths, a, roof if a == " float3 lit=base*(1-metalness)*q6_receiver_illumination(p,n,1,ao);" else r)
                  for paths, a, r in result]
    return result


PROBE = [
    # Diagnostic: natural relief = magenta, natural mountain tiles = cyan.
    ((RELIEF,), "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction",
     "    albedo = float3(1, 0, 1);\n#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction"),
]

V2 = {**V1,
      "WINTER_SNOW_TEX": 0.9, "WINTER_SNOW_TEX_DETAIL": 0.8,
      "WINTER_FIELD_CELLS": 2.5, "WINTER_FIELD_MID": 0.55, "WINTER_FIELD_CONTRAST": 6.0,
      "WINTER_FIELD_COVER_LOW": 0.35, "WINTER_FIELD_COVER_HIGH": 0.75, "WINTER_FIELD_STUBBLE": 0.45,
      "WINTER_FIELD_STRAW": "float3(0.42, 0.34, 0.22)", "WINTER_FIELD_SOIL": "float3(0.17, 0.13, 0.10)",
      "WINTER_FIELD_GROUND": 0.8, "WINTER_FIELD_HEDGE": "float3(0.09, 0.08, 0.06)", "WINTER_FIELD_HEDGE_SNOW": 0.35,
      "WINTER_FIELD_FALLOW": 0.4, "WINTER_FIELD_SNOW_GAIN": 1.0, "WINTER_FIELD_BIAS": 2.5, "WINTER_ROCK_GREY": 0,
      "WINTER_SHORE_ROCK_LOW": 0.45, "WINTER_SHORE_ROCK_HIGH": 0.80, "WINTER_SHORE_ROCK_COVER": 0.85,
      "WINTER_OBJ_UP_LOW": 0.45, "WINTER_OBJ_UP_HIGH": 0.80, "WINTER_OBJ_COVER": 0.85, "WINTER_OBJ_DORMANT": 0.6,
      "WINTER_DECAL_COVER": 0.6,
      "WINTER_ROOF_LOW": 0.45, "WINTER_ROOF_HIGH": 0.75, "WINTER_ROOF_COVER": 0.9, "WINTER_ROOF_NOISE": 9.0,
      "WINTER_PLATE_SNOW": 0.92, "WINTER_PLATE_COVER": 0.65,
      "WINTER_ROAD_DARK": 0.80, "WINTER_ROAD_SNOW": 0.45, "WINTER_RAIL_SNOW": 0.7,
      "WINTER_WATER_HUE": 248, "WINTER_WATER_PULL": 0.8, "WINTER_WATER_CHROMA": 2.4, "WINTER_WATER_LIGHT": 0.92,
      "WINTER_RIVER_CHROMA": 1.6, "WINTER_RIVER_LIGHT": 0.80, "WINTER_WATER_NIGHT": 0.3,
      }

# Calibrated against the display shoulder: lit snow must sit below it (the v1
# albedo .9 compressed to a flat ~226 and shifted hue), cyan-leaning sky fill
# (a violet fill under the warm sun read pink), snow only on bough tops.
V3 = {**V2,
      "WINTER_SNOW_LOW": "float3(0.47, 0.53, 0.63)", "WINTER_SNOW_HIGH": "float3(0.60, 0.64, 0.70)",
      "WINTER_SKY": "float3(0.84, 1.0, 1.22)", "WINTER_DEPTH": 0.66, "WINTER_SCOUR": 0.6,
      "WINTER_DORMANT_CHROMA": 0.35, "WINTER_DORMANT_HUE": 95, "WINTER_DORMANT_LIGHT": 0.72,
      "WINTER_DESERT_DEPTH": 0.72, "WINTER_CRUST_SCALE": 1.3, "WINTER_CRUST_LIGHT": 0.35, "WINTER_CRUST": 0.6,
      "WINTER_DRIFT_RELIEF": 0.07, "WINTER_SMOOTH": 0.8,
      "WINTER_TREE_UP_LOW": 0.35, "WINTER_TREE_UP_HIGH": 0.75, "WINTER_TREE_LIFT": 4.0,
      "WINTER_TREE_SNOW": "float3(0.56, 0.60, 0.66)",
      "WINTER_NEEDLE_LIGHT": 0.62, "WINTER_NEEDLE_CHROMA": 0.45, "WINTER_NEEDLE_HUE": 195, "WINTER_NEEDLE_PULL": 0.5,
      }

# Diagnostic: foliage classes by atlas average (linear g/b) as flat colours:
# broadleaf red, pine green, shrub/jungle yellow, snowy pine blue, else magenta.
PROBE_TREES = [
    ((TREES,), "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Use the exact vector that projects the Q6 field.",
     "    if (kind > 0.5 && kind < 1.5) {\n"
     "        float3 average = BaseColor.SampleLevel(Clamp, float2(0.5, 0.5), 15).rgb;\n"
     "        float gb = average.g / max(average.b, 0.001);\n"
     "        albedo = gb > 3.2 ? float3(0.5, 0, 0) : gb > 1.95 ? float3(0, 0.5, 0) : gb > 1.45 ? float3(0.5, 0.5, 0) :\n"
     "                 gb > 0.5 ? float3(0, 0, 0.5) : float3(0.5, 0, 0.5);\n"
     "    } else albedo = float3(0.5, 0.5, 0.5);\n"
     "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Use the exact vector that projects the Q6 field."),
]

# Calmer ground (less crust speckle, fewer tufts), fuller desert snow, cool
# grey exposed rock, snowier mountains and hills, snow/ice shore and lake beds.
V4 = {**V3,
      "WINTER_CRUST_LIGHT": 0.18, "WINTER_SNOW_SPECK": 0.03, "WINTER_DEPTH": 0.74, "WINTER_SCOUR": 0.5,
      "WINTER_DESERT_DEPTH": 0.9, "WINTER_DORMANT_CHROMA": 0.25, "WINTER_DORMANT_HUE": 85, "WINTER_DORMANT_LIGHT": 0.70,
      "WINTER_STONE_DEPTH": 0.62, "WINTER_SHED_LOW": 0.45, "WINTER_SHED_HIGH": 0.85,
      "WINTER_MTN_SNOW_LOW": 0.12, "WINTER_MTN_SNOW_HIGH": 0.32, "WINTER_MTN_UP_LOW": 0.22, "WINTER_MTN_UP_HIGH": 0.55,
      "WINTER_ROCK_CHROMA": 0.30, "WINTER_ROCK_LIGHT": 0.92,
      "WINTER_ICE_DEPTH": 0.12, "WINTER_ICE_SHALLOW": "float3(0.36, 0.52, 0.60)", "WINTER_ICE_BED": 0.9,
      # Grey probe: lit canopy light is orange (1:.85:.75; ground 1:.95:.86)
      # because canopy ambient is cut; blue-white snow under it read pink.
      # Snow-laden crowns get strong cool sky and snow-bounce fill instead.
      "WINTER_TREE_FILL": "float3(1.7, 1.9, 2.2)", "WINTER_TREE_SNOW": "float3(0.53, 0.60, 0.66)",
      "WINTER_ROOF_LOW": 0.35, "WINTER_ROOF_HIGH": 0.6, "WINTER_ROOF_COVER": 1.0,
      "WINTER_FIELD_COVER_LOW": 0.30, "WINTER_FIELD_COVER_HIGH": 0.62, "WINTER_FIELD_CONTRAST": 7.0,
      }

# Diagnostic: the same neutral grey albedo on trees and relief, to compare
# the light each pass receives.
PROBE_GREY = [
    ((TREES,), "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Use the exact vector that projects the Q6 field.",
     "    albedo = float3(0.5, 0.5, 0.5);\n#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Use the exact vector that projects the Q6 field."),
    ((RELIEF,), "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction",
     "    albedo = float3(0.5, 0.5, 0.5);\n#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction"),
]

V5 = {**V4,
      "WINTER_SNOW_LOW": "float3(0.45, 0.52, 0.66)", "WINTER_SNOW_HIGH": "float3(0.58, 0.63, 0.73)",
      "WINTER_SKY": "float3(0.78, 0.96, 1.32)",
      "WINTER_SNOWY_GB": 1.45, "WINTER_BROADLEAF_GB": 3.2,
      "WINTER_TREE_UP_LOW": 0.45, "WINTER_TREE_UP_HIGH": 0.85, "WINTER_TREE_FILL": "float3(1.5, 1.6, 1.8)",
      "WINTER_TREE_LOCK": 1.0, "WINTER_TREE_LIT_TINT": "float3(0.97, 1.0, 1.07)", "WINTER_TREE_SHADE_TINT": "float3(0.78, 0.94, 1.30)",
      "WINTER_BARE": 0, "WINTER_BARE_SCALE": 40, "WINTER_TWIG_LIGHT": 0.40, "WINTER_TWIG_CHROMA": 0.035,
      "WINTER_TWIG_HUE": 45, "WINTER_BARE_SNOW": 0.5,
      }
# Fields keyed per plot (v6 onward).
V6 = {**V5,
      "WINTER_FIELD_CONTRAST": 9.0, "WINTER_FIELD_COVER_LOW": 0.42, "WINTER_FIELD_COVER_HIGH": 0.85,
      "WINTER_FIELD_FALLOW": 0.38, "WINTER_FIELD_STUBBLE": 0.70, "WINTER_FIELD_SNOW_GAIN": 1.14,
      "WINTER_FIELD_SOIL": "float3(0.13, 0.10, 0.08)", "WINTER_FIELD_STRAW": "float3(0.40, 0.32, 0.20)",
      "WINTER_FIELD_HEDGE": "float3(0.11, 0.10, 0.08)", "WINTER_FIELD_BIAS": 2.5,
      }
V6.update({
    "WINTER_TREE_UP_LOW": 0.62, "WINTER_TREE_UP_HIGH": 0.98, "WINTER_TREE_LIFT": 3.0, "WINTER_TREE_LOCK": 0.85,
    "WINTER_ROAD_DARK": 0.95, "WINTER_ROAD_SNOW": 0.6, "WINTER_ROCK_GREY": 0.6,
})
# Combined candidate: bare broadleaf canopy, thicker snow on shoreline rocks
# (user, 2026-10-21), a little more snow on pine boughs, darker field hedges.
V7 = {**V6, "WINTER_BARE": 0.45,
      "WINTER_SHORE_ROCK_LOW": 0.25, "WINTER_SHORE_ROCK_HIGH": 0.60, "WINTER_SHORE_ROCK_COVER": 1.15,
      "WINTER_TREE_UP_LOW": 0.55, "WINTER_TREE_UP_HIGH": 0.95,
      "WINTER_FIELD_HEDGE": "float3(0.08, 0.07, 0.06)", "WINTER_FIELD_HEDGE_SNOW": 0.25}
V7_LADEN = {**V7, "WINTER_BARE": 0}
# Measured v7 against the concept: same hues, but lit snow ~30 levels darker
# (p90 ~205 vs ~243) and open water lighter/greyer (105,129,158 vs 44,95,146).
# Brighter snow kept below the display shoulder that flattened v1; deeper
# water; browner twigs (h 45 read mauve under the orange canopy light).
V8 = {**V7,
      "WINTER_SNOW_LOW": "float3(0.52, 0.60, 0.75)", "WINTER_SNOW_HIGH": "float3(0.66, 0.71, 0.81)",
      "WINTER_TREE_SNOW": "float3(0.60, 0.67, 0.74)",
      "WINTER_WATER_LIGHT": 0.74, "WINTER_WATER_CHROMA": 3.0,
      "WINTER_TWIG_HUE": 68, "WINTER_TWIG_CHROMA": 0.028}
V8_LADEN = {**V8, "WINTER_BARE": 0}
V9 = {**V8,
      "WINTER_RIVER_CHROMA": 2.6, "WINTER_RIVER_MIN_CHROMA": 0.07, "WINTER_RIVER_LIGHT": 1.25,
      "WINTER_RIVER_ICE": 1.0, "WINTER_RIVER_ICE_IN": 3.6, "WINTER_RIVER_ICE_BREAK": 2.4, "WINTER_RIVER_ICE_SCALE": 9.0,
      "WINTER_RIVER_FLOE": 0.80, "WINTER_RIVER_FLOE_SCALE": 31.0,
      "WINTER_RIVER_ICE_THIN": "float3(0.40, 0.56, 0.64)", "WINTER_RIVER_BANK_SNOW": 0.8, "WINTER_RIVER_DAYLIGHT": 0}
V9_LADEN = {**V9, "WINTER_BARE": 0}
# Coast, Sea and Ocean readable (user, 2026-10-21). Summer seas view: coast
# L .53 h 195, sea L .45 h 241, ocean L .46 h 244.
V10 = {**V9,
       "WINTER_WATER_REF_HUE": 241, "WINTER_WATER_CHROMA": 3.0, "WINTER_WATER_PIVOT": 0.45,
       "WINTER_WATER_LIGHT": 0.90, "WINTER_WATER_CONTRAST": 1.25,
       "WINTER_COAST_LIFT": 0.03, "WINTER_COAST_HUE": -8, "WINTER_OCEAN_DROP": 0.07, "WINTER_OCEAN_HUE": 8,
       "WINTER_DEEP_NEAR": 1.0, "WINTER_DEEP_FAR": 4.5, "WINTER_DEEP_DROP": 0, "WINTER_DEEP_HUE": 0}
V10_LADEN = {**V10, "WINTER_BARE": 0}
# Smooth shore-distance darkening plus a milder exact class step (v10's step
# alone left crisp tile diamonds).
V11 = {**V10, "WINTER_COAST_LIFT": 0.015, "WINTER_COAST_HUE": -4, "WINTER_OCEAN_DROP": 0.03, "WINTER_OCEAN_HUE": 4,
       "WINTER_DEEP_NEAR": 1.0, "WINTER_DEEP_FAR": 4.5, "WINTER_DEEP_DROP": 0.07, "WINTER_DEEP_HUE": 6}
V11_LADEN = {**V11, "WINTER_BARE": 0}
# Icy rivers dim with the light (v11 rivers read neon at dusk and night).
V12 = {**V11, "WINTER_RIVER_DAYLIGHT": 1}
# Land types readable under snow. Vectors are [grassland, plains, desert, tundra].
V13 = {**V12,
       "WINTER_TYPE_DEPTH": "float4(0.80, 0.70, 0.45, 1.45)",
       "WINTER_TYPE_LIGHT": "float4(0.78, 0.95, 1.05, 0.95)",
       "WINTER_TYPE_CHROMA": "float4(0.060, 0.085, 0.095, 0.012)",
       "WINTER_TYPE_HUE": "float4(128, 84, 70, 110)",
       "WINTER_TUSSOCK": 0.35, "WINTER_TUSSOCK_SCALE": 7.0,
       "WINTER_TUNDRA_SNOW": "float3(0.95, 1.0, 1.07)", "WINTER_TUNDRA_SMOOTH": 0.6}
V13_LADEN = {**V13, "WINTER_BARE": 0}
# v13's desert was a flat beige plate (no snow, saturated sand) and grass/plains
# stayed fully white. Pattern exposure with shares of ground showing about
# grassland 25%, plains 35%, desert 50% (dune-ripple troughs hold the snow);
# paler, less saturated sand.
V14 = {**V13,
       "WINTER_TYPE_DEPTH": "float4(1.0, 1.0, 1.0, 1.45)",
       "WINTER_SHOW": "float3(0.58, 0.56, 0.50)", "WINTER_RIPPLE_FREQ": 52.0,
       "WINTER_TYPE_LIGHT": "float4(0.74, 0.92, 0.94, 0.95)",
       "WINTER_TYPE_CHROMA": "float4(0.055, 0.080, 0.060, 0.012)",
       "WINTER_TYPE_HUE": "float4(128, 84, 76, 110)"}
V14_LADEN = {**V14, "WINTER_BARE": 0}
# v14: desert and plains both read as tan/gold stripes (2.1), plains near tundra
# (3.4), regular "barcode" ripples, and the mountains' ground collars stayed
# uniformly white. Plains flecks, paler sand on broken wandering ripples,
# ragged tussocks, and the same signatures on the mountain collars.
V15 = {**V14,
       "WINTER_SHOW": "float3(0.57, 0.58, 0.46)", "WINTER_RIPPLE_FREQ": 40.0,
       "WINTER_TUSSOCK_SCALE": 8.0, "WINTER_STUBBLE_SCALE": 38.0,
       "WINTER_TYPE_LIGHT": "float4(0.70, 0.78, 1.00, 0.95)",
       "WINTER_TYPE_CHROMA": "float4(0.060, 0.100, 0.040, 0.012)",
       "WINTER_TYPE_HUE": "float4(126, 80, 72, 110)"}
V15_LADEN = {**V15, "WINTER_BARE": 0}
# v15: desert washed nearly white (fine ripples, low share showing), tundra
# close to plains and desert. Coarser ripples with more light warm sand on the
# crests; tundra a little bluer; a touch more stubble on plains.
V16_LADEN = {**V15_LADEN,
             "WINTER_SHOW": "float3(0.57, 0.55, 0.30)", "WINTER_RIPPLE_FREQ": 26.0,
             "WINTER_TYPE_LIGHT": "float4(0.70, 0.78, 0.94, 0.95)",
             "WINTER_TYPE_CHROMA": "float4(0.055, 0.100, 0.055, 0.012)",
             "WINTER_TUNDRA_SNOW": "float3(0.92, 0.99, 1.10)"}
# Grassland never shows ground (threshold 2); plains and desert only sparsely.
V17_LADEN = {**V16_LADEN,
             "WINTER_SHOW": "float3(2.0, 0.71, 0.86)",
             "WINTER_TYPE_LIGHT": "float4(0.70, 0.80, 0.96, 0.95)",
             "WINTER_TYPE_CHROMA": "float4(0.055, 0.090, 0.050, 0.012)",
             "WINTER_PLAINS_SNOW": "float3(1.02, 1.0, 0.95)", "WINTER_DESERT_SNOW": "float3(1.04, 0.99, 0.90)",
             "WINTER_TUNDRA_SNOW": "float3(0.88, 0.96, 1.08)", "WINTER_TUNDRA_CRUST": 1.2, "WINTER_TUNDRA_GLOSS": 0.15,
             "WINTER_DESERT_RELIEF": 0.15}
# v17: plains read as grassland (0.9), grass-tundra 2.8, and at tile 96 the
# water's class step showed tile diamonds. Warmer plains snow with a few more
# flecks, deeper beige desert snow, bluer tundra; half the water class step,
# more of the smooth depth darkening.
V18_LADEN = {**V17_LADEN,
             "WINTER_SHOW": "float3(2.0, 0.66, 0.86)", "WINTER_TYPE_LIGHT": "float4(0.70, 0.76, 0.96, 0.95)",
             "WINTER_PLAINS_SNOW": "float3(1.035, 1.0, 0.93)", "WINTER_DESERT_SNOW": "float3(1.06, 0.99, 0.86)",
             "WINTER_TUNDRA_SNOW": "float3(0.85, 0.94, 1.10)",
             "WINTER_OCEAN_DROP": 0.018, "WINTER_OCEAN_HUE": 3, "WINTER_COAST_LIFT": 0.01, "WINTER_DEEP_DROP": 0.09}
# v18: desert and plains merged (both cream snow with warm flecks; 3.0) and
# plains still read as grassland (1.4). Desert: no flecks, smooth warm beige
# shaped by stronger ripples. Plains: near-white snow with deeper-gold flecks.
V19_LADEN = {**V18_LADEN,
             "WINTER_SHOW": "float3(2.0, 0.60, 2.0)",
             "WINTER_TYPE_LIGHT": "float4(0.70, 0.70, 0.96, 0.95)",
             "WINTER_TYPE_CHROMA": "float4(0.055, 0.110, 0.050, 0.012)",
             "WINTER_PLAINS_SNOW": "float3(1.015, 1.0, 0.965)", "WINTER_DESERT_RELIEF": 0.25}
# Final candidate: v19's desert tint only cancelled the cool snow (neutral
# grey); a gentle cream instead, still without flecks.
V20_LADEN = {**V19_LADEN, "WINTER_DESERT_SNOW": "float3(1.09, 1.0, 0.83)"}
# Roof snow lit like the ground's, with roof relief and building-scale crust.
V22_EXTRA = {"WINTER_ROOF_TEX": 7.0, "WINTER_ROOF_DETAIL": 1.2, "WINTER_ROOF_FLATTEN": 0.35,
             "WINTER_ROOF_CREASE": "float3(0.72, 0.80, 0.95)", "WINTER_ROOF_GAIN": 1.0}
# Roof snow calibrated to the ground snow, building-scale mottling, a few bare dabs.
V23_EXTRA = {"WINTER_ROOF_AO": 0.35, "WINTER_ROOF_DAB": 9.0, "WINTER_ROOF_BARE": 0.24,
             "WINTER_ROOF_MOTTLE": 11.0, "WINTER_ROOF_DETAIL": 0.55, "WINTER_ROOF_FLATTEN": 0.1,
             "WINTER_ROOF_CREASE": "float3(0.72, 0.80, 0.95)", "WINTER_ROOF_GAIN": 1.5}
# River banks: shelf ice in some reaches only, snow matched to the ground, glassy edge.
V24_EXTRA = {"WINTER_RIVER_SNOW_GAIN": 1.4, "WINTER_SHELF_REACH": 1.6, "WINTER_SHELF_OPEN": 0.42,
             "WINTER_SHELF_WIDTH": 3.2, "WINTER_SHELF_SCALLOP": 11.0, "WINTER_SHELF_BREAK": 1.4,
             "WINTER_SHELF_GLASS": 1.0, "WINTER_SHELF_WINDOW": 7.0,
             "WINTER_RIVER_RIM": 0.6, "WINTER_RIVER_RIM_TONE": "float3(0.66, 0.74, 0.84)"}
# Plains flecks were dense and rust-orange across the band: fewer, straw-gold.
V21_LADEN = {**V20_LADEN, "WINTER_SHOW": "float3(2.0, 0.665, 2.0)",
             "WINTER_TYPE_LIGHT": "float4(0.70, 0.80, 0.96, 0.95)",
             "WINTER_TYPE_CHROMA": "float4(0.055, 0.080, 0.050, 0.012)",
             "WINTER_TYPE_HUE": "float4(126, 88, 72, 110)"}
# Broadleaf alternative: bare winter canopy.
V5_BARE = {**V5, "WINTER_BARE": 0.45}

LOOKS = {
    "identity": [],
    "probe-grey": PROBE_GREY,
    "probe-trees": PROBE_TREES,
    "v1": edits(V1),
    "v2": edits(V2) + integrated(V2),
    "v3": edits(V3) + integrated(V3),
    "v4": edits(V4) + integrated(V4, fields="kit", ice=True),
    "v5": edits(V5, trees="classed") + integrated(V5, fields="kit", ice=True),
    "v5-bare": edits(V5_BARE, trees="classed") + integrated(V5_BARE, fields="kit", ice=True),
    "v6": edits(V6, trees="classed") + integrated(V6, fields="plot", ice=True, rocks=True),
    "v7": edits(V7, trees="classed") + integrated(V7, fields="plot", ice=True, rocks=True),
    "v7-laden": edits(V7_LADEN, trees="classed") + integrated(V7_LADEN, fields="plot", ice=True, rocks=True),
    "v8": edits(V8, trees="classed") + integrated(V8, fields="plot", ice=True, rocks=True),
    "v8-laden": edits(V8_LADEN, trees="classed") + integrated(V8_LADEN, fields="plot", ice=True, rocks=True),
    "v9": edits(V9, trees="classed") + integrated(V9, fields="plot", ice=True, rocks=True, rivers="ice"),
    "v9-laden": edits(V9_LADEN, trees="classed") + integrated(V9_LADEN, fields="plot", ice=True, rocks=True, rivers="ice"),
    "v10": edits(V10, trees="classed") + integrated(V10, fields="plot", ice=True, rocks=True, rivers="ice", water="classes"),
    "v10-laden": edits(V10_LADEN, trees="classed") + integrated(V10_LADEN, fields="plot", ice=True, rocks=True, rivers="ice",
                                                            water="classes"),
    "v11": edits(V11, trees="classed") + integrated(V11, fields="plot", ice=True, rocks=True, rivers="ice", water="classes"),
    "v12": edits(V12, trees="classed") + integrated(V12, fields="plot", ice=True, rocks=True, rivers="ice", water="classes"),
    "v13": edits(V13, trees="classed", ground="types") + integrated(V13, fields="plot", ice=True, rocks=True, rivers="ice",
                                                                    water="classes"),
    "v14": edits(V14, trees="classed", ground="patterns") + integrated(V14, fields="plot", ice=True, rocks=True,
                                                                       rivers="ice", water="classes"),
    "v14-laden": edits(V14_LADEN, trees="classed", ground="patterns") + integrated(V14_LADEN, fields="plot", ice=True,
                                                                                   rocks=True, rivers="ice", water="classes"),
    "v22-laden": edits({**V21_LADEN, **V22_EXTRA}, trees="classed", ground="subtle") +
                 integrated({**V21_LADEN, **V22_EXTRA}, fields="plot", ice=True, rocks=True, rivers="ice", water="classes",
                            roofs="lit"),
    "v23-laden": edits({**V21_LADEN, **V23_EXTRA}, trees="classed", ground="subtle") +
                 integrated({**V21_LADEN, **V23_EXTRA}, fields="plot", ice=True, rocks=True, rivers="ice", water="classes", roofs="textured"),
    "v21-laden": edits(V21_LADEN, trees="classed", ground="subtle") + integrated(V21_LADEN, fields="plot", ice=True,
                                                                                 rocks=True, rivers="ice", water="classes"),
    "v20-laden": edits(V20_LADEN, trees="classed", ground="subtle") + integrated(V20_LADEN, fields="plot", ice=True,
                                                                                 rocks=True, rivers="ice", water="classes"),
    "v19-laden": edits(V19_LADEN, trees="classed", ground="subtle") + integrated(V19_LADEN, fields="plot", ice=True,
                                                                                 rocks=True, rivers="ice", water="classes"),
    "v18-laden": edits(V18_LADEN, trees="classed", ground="subtle") + integrated(V18_LADEN, fields="plot", ice=True,
                                                                                 rocks=True, rivers="ice", water="classes"),
    "v17-laden": edits(V17_LADEN, trees="classed", ground="subtle") + integrated(V17_LADEN, fields="plot", ice=True,
                                                                                 rocks=True, rivers="ice", water="classes"),
    "v16-laden": edits(V16_LADEN, trees="classed", ground="signatures") + integrated(V16_LADEN, fields="plot", ice=True,
                                                                                     rocks=True, rivers="ice", water="classes"),
    "v15-laden": edits(V15_LADEN, trees="classed", ground="signatures") + integrated(V15_LADEN, fields="plot", ice=True,
                                                                                     rocks=True, rivers="ice", water="classes"),
    "v13-laden": edits(V13_LADEN, trees="classed", ground="types") + integrated(V13_LADEN, fields="plot", ice=True,
                                                                                rocks=True, rivers="ice", water="classes"),
    "v11-laden": edits(V11_LADEN, trees="classed") + integrated(V11_LADEN, fields="plot", ice=True, rocks=True, rivers="ice",
                                                            water="classes"),
}
LOOKS["v24-laden"] = edits({**V21_LADEN, **V23_EXTRA, **V24_EXTRA}, trees="classed", ground="subtle") + \
    integrated({**V21_LADEN, **V23_EXTRA, **V24_EXTRA}, fields="plot", ice=True, rocks=True, rivers="banks",
               water="classes", roofs="textured")


# Flurry v2 (user, 2026-10-22: "needs far more randomness ... clearly the same
# rectangle shape of interweaving parallel snowflakes"). v1 moved every flake
# of a layer as one rigid grid and swayed whole columns together, and filled
# the quad to its edges. Now every column falls at its own speed and offset,
# every flake has its own spot, size, brightness and sway (phase, amplitude,
# period), the wind slants each mountain's flurry differently, and the flakes
# fill a soft rounded plume (width, height, density per mountain) whose edge
# and density drift with slow gusts. Integer hash and a bounded clock keep it
# exact late in a game. Production: seasons.py WINTER_FLURRY.
FLURRY2 = r"""// Seasons D4: a gentle flurry of small snowflakes over a mountain tile (96).
// Three parallax layers of tall cells, one flake per cell at most. Each column
// falls at its own speed with its own offset; each flake has its own spot,
// size, brightness and sway; each mountain its own plume (width, height,
// density, wind slant) whose rounded edge and density drift with slow gusts.
// Units are pixels at tile width 128; the layers thin with the zoom.
float q8_flurry_hash(float2 p,float s){
  uint3 v=uint3(int3(int2(floor(p)),int(s*97.)));
  uint h=v.x*1597334677u^v.y*3812015801u^v.z*2654435761u;
  h^=h>>16;h*=2246822519u;h^=h>>13;h*=3266489917u;h^=h>>16;
  return float(h&0xffffffu)*(1./16777216.);
}
float q8_flurry_noise(float2 p,float s){
  float2 i=floor(p),f=frac(p);f=f*f*(3-2*f);
  return lerp(lerp(q8_flurry_hash(i,s),q8_flurry_hash(i+float2(1,0),s),f.x),
              lerp(q8_flurry_hash(i+float2(0,1),s),q8_flurry_hash(i+float2(1,1),s),f.x),f.y);
}
float4 q8_effect_flurry(float2 q,float2 pixel,float t,float seed,float strength,float night){
  float zoom=(2/pixel.x)/(1.05*128);
  int layers=zoom>=1?3:zoom>=.75?2:1;
  t=fmod(t,1800.);
  float2 u=float2(q.x/pixel.x,q.y/pixel.y)/zoom;
  // This mountain's plume.
  float wind=(q8_flurry_hash(float2(seed,1),7.)-.5)*.7;
  float density=.8+.5*q8_flurry_hash(float2(seed,2),7.);
  float half_width=.62+.3*q8_flurry_hash(float2(seed,3),7.);
  float top=.62+.3*q8_flurry_hash(float2(seed,4),7.);
  // Straight noisy sides down to the foot, a rounded top: a plume over the peak.
  float2 e=float2(q.x/half_width,max(0.,q.y-.35*top)/(.65*top));
  float edge=length(e)+(q8_flurry_noise(float2(q.x*4.5,q.y*6.-t*.09),seed+21.)-.5)*.6
                     +(q8_flurry_noise(float2(q.x*1.6,q.y*2.2+t*.05),seed+23.)-.5)*.5;
  // A ragged foot: the flakes thin out at uneven heights, not on one line.
  float foot=.06+.3*q8_flurry_noise(float2(q.x*3.5,t*.05),seed+25.);
  float envelope=smoothstep(1.,.45,edge)*smoothstep(foot-.06,foot+.08,q.y);
  float gust=q8_flurry_noise(float2(u.x/70.,u.y/55.+t*.11),seed+31.);
  float alpha=0;
  [unroll]for(int layer=0;layer<3;layer++)if(layer<layers){
   float w=layer==0?24.:layer==1?17.:12.,h=w*1.3,fall=layer==0?20.:layer==1?14.:9.;
   // Each layer slants at its own angle around the mountain's wind.
   float drift=(wind+(q8_flurry_hash(float2(layer,5),seed+9.)-.5)*.5)*fall*t;
   float column=floor((u.x-drift)/w);
   float speed=fall*(.7+.6*q8_flurry_hash(float2(column,layer),seed+11.));
   float offset=q8_flurry_hash(float2(column,layer+7),seed+13.)*h;
   float2 g=float2(u.x-drift,u.y+speed*t+offset);
   float2 id=floor(g/float2(w,h)),f=frac(g/float2(w,h))*float2(w,h);
   float a=q8_flurry_hash(id,seed+layer),b=q8_flurry_hash(id.yx+3,seed+layer),
         c=q8_flurry_hash(id+17,seed+layer*5),d=q8_flurry_hash(id+29,seed+layer*3);
   float share=(layer==2?.6:.75)*density*(.45+gust);
   if(q8_flurry_hash(id+41,seed+layer*7)<share){
    float scale=layer==0?1.:layer==1?.75:.5;
    // Wide, slow flutter: up to a fifth of the cell each way, inside the cell.
    float sway=(.3+.7*d)*.2*w*sin(6.2832*(t/(1.6+2.6*c)+a*3.));
    float2 spot=float2((.3+.4*a)*w+sway,(.1+.8*b)*h);
    float radius=(layer==0?1.25:layer==1?.95:.7)*(.75+.5*d);
    float opacity=(layer==0?.9:layer==1?.7:.5)*(.7+.3*b);
    alpha=max(alpha,opacity*saturate((radius-length(f-spot))*zoom+.6));
   }
  }
  alpha*=envelope*strength;
  clip(alpha-.004);
  // Bright as the other effects (scene white near 1.3 after grading); moonlit at night.
  return float4(lerp(float3(1.26,1.28,1.32),float3(.50,.58,.82),night),alpha);
}
"""
from Renderer.lab.studies.winter.flurry_tree import FLURRY_SHADER as LAB_FLURRY
LOOKS["v24-flurry2"] = LOOKS["v24-laden"] + [(FIELD_SHADERS, LAB_FLURRY, FLURRY2)]
# v3 is FLURRY2 as it now stands (wider sway, per-layer slant, ragged plume and foot).
LOOKS["v24-flurry3"] = LOOKS["v24-flurry2"]
LOOKS["v24-flurry4"] = LOOKS["v24-flurry2"]  # v3 with a higher density floor


# Desert relief (user, 2026-10-23: desert has "a really odd, swirly look").
# winter_ripple bent its direction with fbm (+-1.2 rad) and warped its phase by
# up to 8 radians, drawing contour-line whorls that alias at map zoom.
OLD_RIPPLE = """float winter_ripple(float2 world) {
    float turn = (winter_fbm(world * 0.6) - 0.5) * 2.4;
    float2 wind = float2(cos(turn), sin(turn));
    return 0.5 + 0.5 * sin(dot(world, wind) * WINTER_RIPPLE_FREQ + winter_fbm(world * 2.3) * 8.0);
}"""
# Snow-covered transverse dunes instead: crests across one regional wind that
# turns only slowly over many tiles, gently meandering, each with a long
# windward rise and a short lee face, crest height varying along the dune.
# The relief fades out where a dune spans too few pixels to resolve.
DUNES = """float winter_ripple(float2 world) {
    float turn = (winter_noise(world * 0.12) - 0.5) * 0.8;
    float2 wind = float2(cos(0.6 + turn), sin(0.6 + turn));
    float along = dot(world, wind) * WINTER_DUNE_FREQ + (winter_noise(world * 0.9) - 0.5) * 1.6;
    float phase = frac(along);
    float profile = smoothstep(0.0, 0.78, phase) * (1 - smoothstep(0.78, 1.0, phase));
    float resolved = 1 - smoothstep(0.08, 0.25, fwidth(along));
    return 0.5 + (profile * (0.6 + 0.4 * winter_noise(world * 0.7 + 4.1)) - 0.35) * resolved;
}"""
def _desert(look, ripple, relief):
    return [(paths, a, r.replace(OLD_RIPPLE, ripple).replace("#define WINTER_DESERT_RELIEF 0.25",
            f"#define WINTER_DESERT_RELIEF {relief}\n#define WINTER_DUNE_FREQ 3.0")) for paths, a, r in LOOKS[look]]
LOOKS["v24-dunes"] = _desert("v24-flurry4", DUNES, 0.25)
LOOKS["v24-smooth"] = _desert("v24-flurry4", OLD_RIPPLE, 0)
# A few broad, low dunes: gentle undulation rather than crest lines.
LOOKS["v24-dunes-soft"] = [(paths, a, r.replace("#define WINTER_DUNE_FREQ 3.0", "#define WINTER_DUNE_FREQ 1.2"))
                           for paths, a, r in _desert("v24-flurry4", DUNES, 0.12)]
# The current look (user, 2026-10-23: broad low dunes; flurry v2). Production:
# Renderer/lab/shared/seasons (winter_natural.hlsl winter_ripple, WINTER_FLURRY).
LOOKS["v25-laden"] = LOOKS["v24-dunes-soft"]


# Farms v2 (user, 2026-10-23: Winter farms are "ugly", though clear in what
# they are). The plot pass made rows binary (contrast x9), bared near-black
# soil on most plots and drew black hedgerow dashes: a barcode. Styles:
# 1 every plot under snow, rows as soft relief, edges a cool line of snow
# shadow with a few twigs; 2 adds a per-plot patchwork (smooth snow, soft
# rows, ploughed loam thinly in the furrows, straw stubble flecks); 3 shows
# winter wheat shoots instead of stubble.
OLD_FIELDS = INTEGRATED_COMMON[INTEGRATED_COMMON.index("// Second field pass keyed on the farm kit"):
                               INTEGRATED_COMMON.index("\n}\n", INTEGRATED_COMMON.index("float3 winter_field_plot(")) + 3]
SOFT_FIELDS = """#define WINTER_FIELD_ROWS 3.0
#define WINTER_FIELD_FURROW float3(0.84, 0.89, 0.98)
#define WINTER_FIELD_EDGE float3(0.70, 0.76, 0.88)
#define WINTER_FIELD_TWIGS float3(0.22, 0.20, 0.19)
#define WINTER_FIELD_LOAM float3(0.36, 0.28, 0.21)
#define WINTER_FIELD_WHEAT float3(0.30, 0.42, 0.20)
#define WINTER_FIELD_SMOOTH 0.35
#define WINTER_FIELD_LOAM_SHOW 0.45
#define WINTER_FIELD_STRAW_SHOW 0.6
#define WINTER_FIELD_TWIG_SHOW 0.6
#define WINTER_FIELD_TWIG_FROM 0.62
#define WINTER_FIELD_PLOT_TONE 0.0
#define WINTER_FIELD_STUBBLE_ALL 0.0
#define WINTER_FIELD_STUBBLE_TONE float3(0.70, 0.58, 0.38)
// Winter farm plots (farms v2): snow over the crop rows as soft relief, a
// per-plot state in the patchwork styles, and field edges as a cool line of
// snow shadow with a few twigs, never dark dashes.
float3 winter_field_soft(float3 albedo, float plot_light, float pick, float2 world, float3 snow) {
    float3 lch = winter_lch(albedo);
    float speck = winter_noise(world * 61.0);
    float path = smoothstep(-0.01, 0.04, albedo.r - albedo.g);
    float row = saturate((lch.x - plot_light) * WINTER_FIELD_ROWS + 0.5);
    float3 white = snow * WINTER_FIELD_SNOW_GAIN;
    float3 field = lerp(white * WINTER_FIELD_FURROW, white, row);
    if (WINTER_FIELD_STYLE > 1.5) {
        if (pick < WINTER_FIELD_SMOOTH) {
            field = lerp(white, field, 0.3);
        } else if (pick >= 0.6 && pick < 0.8) {
            field = lerp(field, WINTER_FIELD_LOAM * (0.85 + 0.3 * speck), WINTER_FIELD_PLOT_TONE + (1 - row) * WINTER_FIELD_LOAM_SHOW);
        } else if (pick >= 0.8) {
            float fleck = smoothstep(0.55, 0.75, winter_noise(world * 140.0));
            float straw = WINTER_FIELD_STYLE > 3.5 ? 0.5 + 0.5 * fleck : fleck;
            field = abs(WINTER_FIELD_STYLE - 3) < 0.5 ? lerp(field, WINTER_FIELD_WHEAT, (1 - row) * 0.4)
                                                      : lerp(field, WINTER_FIELD_STRAW, WINTER_FIELD_PLOT_TONE + straw * row * WINTER_FIELD_STRAW_SHOW);
        }
    }
    field = lerp(field, WINTER_FIELD_STUBBLE_TONE, WINTER_FIELD_STUBBLE_ALL * row);
    float3 edge = lerp(white * WINTER_FIELD_EDGE, WINTER_FIELD_TWIGS,
        smoothstep(WINTER_FIELD_TWIG_FROM, WINTER_FIELD_TWIG_FROM + 0.18, speck) * WINTER_FIELD_TWIG_SHOW);
    return lerp(field, edge, path);
}
float3 winter_field_kit(float3 albedo, float2 world, float3 snow, bool ground) {
    if (ground)
        return lerp(winter_dormant(albedo), snow, saturate(WINTER_FIELD_GROUND + (winter_noise(world * 61.0) - 0.5) * 0.3));
    float ripe = smoothstep(108, 96, winter_lch(albedo).z);
    return winter_field_soft(albedo, WINTER_FIELD_MID, ripe > 0.5 ? 0.85 : 0.45, world, snow);
}
float3 winter_field_plot(float3 albedo, float3 mean, float2 world, float3 snow) {
    float3 plot = winter_lch(mean);
    return winter_field_soft(albedo, plot.x, frac(plot.x * 53.1 + plot.z * 0.37 + plot.y * 91.7), world, snow);
}
"""
def _farms(style):
    return [(paths, a, r.replace(OLD_FIELDS, f"#define WINTER_FIELD_STYLE {style}\n" + SOFT_FIELDS))
            for paths, a, r in LOOKS["v25-laden"]]
for _style in (1, 2, 3):
    LOOKS[f"v25-farms{_style}"] = _farms(_style)
# 4: a hedged patchwork. The patchwork of style 2, strong enough to read:
# firmer soft rows, ploughed plots' furrows clearly loam, stubble plots straw
# along their ridges, hedgerows broken lines of dark twigs over a cool shadow.
_HEDGED = {"#define WINTER_FIELD_ROWS 3.0": "#define WINTER_FIELD_ROWS 5.0",
           "#define WINTER_FIELD_FURROW float3(0.84, 0.89, 0.98)": "#define WINTER_FIELD_FURROW float3(0.80, 0.86, 0.97)",
           "#define WINTER_FIELD_LOAM float3(0.36, 0.28, 0.21)": "#define WINTER_FIELD_LOAM float3(0.38, 0.29, 0.21)",
           "#define WINTER_FIELD_SMOOTH 0.35": "#define WINTER_FIELD_SMOOTH 0.3",
           "#define WINTER_FIELD_LOAM_SHOW 0.45": "#define WINTER_FIELD_LOAM_SHOW 0.7",
           "#define WINTER_FIELD_STRAW_SHOW 0.6": "#define WINTER_FIELD_STRAW_SHOW 0.55",
           "#define WINTER_FIELD_TWIG_SHOW 0.6": "#define WINTER_FIELD_TWIG_SHOW 0.85",
           "#define WINTER_FIELD_TWIG_FROM 0.62": "#define WINTER_FIELD_TWIG_FROM 0.42"}
def _hedged(text):
    for a, b in _HEDGED.items():
        text = text.replace(a, b)
    return text
LOOKS["v25-farms4"] = [(paths, a, _hedged(r)) for paths, a, r in _farms(4)]
# 5: the hedged patchwork with whole-plot tone, so plots read apart at map
# zoom: ploughed fields a grey-brown tone, stubble a light golden one.
def _toned(text):
    return (_hedged(text).replace("#define WINTER_FIELD_PLOT_TONE 0.0", "#define WINTER_FIELD_PLOT_TONE 0.2")
            .replace("#define WINTER_FIELD_LOAM_SHOW 0.7", "#define WINTER_FIELD_LOAM_SHOW 0.45"))
# Hedgerow fields: white fields framed by broken dark hedgerows of twigs with
# snow on them. Stubble rows: every field shows soft golden stubble along its
# rows through thin snow.
LOOKS["v25-hedgerows"] = [(paths, a, r.replace("#define WINTER_FIELD_TWIG_FROM 0.62", "#define WINTER_FIELD_TWIG_FROM 0.22")
                           .replace("#define WINTER_FIELD_TWIG_SHOW 0.6", "#define WINTER_FIELD_TWIG_SHOW 0.9")
                           .replace("#define WINTER_FIELD_TWIGS float3(0.22, 0.20, 0.19)", "#define WINTER_FIELD_TWIGS float3(0.24, 0.21, 0.19)"))
                          for paths, a, r in _farms(1)]
LOOKS["v25-stubble"] = [(paths, a, r.replace("#define WINTER_FIELD_STUBBLE_ALL 0.0", "#define WINTER_FIELD_STUBBLE_ALL 0.4")
                         .replace("#define WINTER_FIELD_ROWS 3.0", "#define WINTER_FIELD_ROWS 5.0"))
                        for paths, a, r in _farms(1)]
LOOKS["v25-farms5"] = [(paths, a, _toned(r).replace("WINTER_FIELD_STRAW, WINTER_FIELD_PLOT_TONE",
                       "float3(0.62, 0.52, 0.34), WINTER_FIELD_PLOT_TONE")) for paths, a, r in _farms(4)]
# The current look (user, 2026-10-23: hedgerow fields for Winter farms).
# Production: winter_integrated.hlsl winter_field_soft/kit/plot.
LOOKS["v26-laden"] = LOOKS["v25-hedgerows"]
# Hedgerows v2 (2026-10-23 game check: in the 1498 AD save at map zoom the v25
# hedgerows vanish, as their twigs sit only on the kit's in-plot paths and the
# plots are white on white snow). Each plot is framed by a broken hedge along
# its own alpha rim, so every field reads apart at map zoom.
_RIM_PLOT = """#define WINTER_HEDGE_INNER {inner}
#define WINTER_HEDGE_SHOW {show}
#define WINTER_HEDGE_TWIGS 0.8
#define WINTER_HEDGE_BREAKS 12.0
float3 winter_field_plot(float3 albedo, float3 mean, float2 world, float3 snow, float rim) {{
    float3 plot = winter_lch(mean);
    float3 field = winter_field_soft(albedo, plot.x, frac(plot.x * 53.1 + plot.z * 0.37 + plot.y * 91.7), world, snow);
    float hedge = (1 - smoothstep(WINTER_HEDGE_INNER, 1.0, rim)) * smoothstep(0.2, 0.45, winter_noise(world * WINTER_HEDGE_BREAKS));
    float3 twigs = lerp(snow * WINTER_FIELD_EDGE, WINTER_FIELD_TWIGS, smoothstep(0.3, 0.55, winter_noise(world * 61.0)) * WINTER_HEDGE_TWIGS);
    return lerp(field, twigs, hedge * WINTER_HEDGE_SHOW);
}}"""
def _rim(inner, show):
    old_plot = SOFT_FIELDS[SOFT_FIELDS.index("float3 winter_field_plot("):SOFT_FIELDS.rindex("}") + 1]
    call = "                                  winter_world, winter_white);\n"
    return [(paths, a, r.replace(old_plot, _RIM_PLOT.format(inner=inner, show=show))
             .replace(call, call.replace("winter_white);", "winter_white, mine_sample.a);")))
            for paths, a, r in LOOKS["v25-hedgerows"]]
LOOKS["v27-hedges-a"] = _rim(0.8, 0.85)
LOOKS["v27-hedges-b"] = _rim(0.95, 1.0)
# The kit's alpha also dips between crop rows, so a and b hatch every row at
# close zoom. c/d read the rim from the blurred (biased) plot sample instead,
# which keeps only the plot outlines.
_BLUR_PLOT = """#define WINTER_HEDGE_LOW {low}
#define WINTER_HEDGE_HIGH {high}
#define WINTER_HEDGE_TWIGS 0.8
#define WINTER_HEDGE_BREAKS 12.0
float3 winter_field_plot(float3 albedo, float4 mean, float2 world, float3 snow) {{
    float3 plot = winter_lch(mean.rgb);
    float3 field = winter_field_soft(albedo, plot.x, frac(plot.x * 53.1 + plot.z * 0.37 + plot.y * 91.7), world, snow);
    float hedge = (1 - smoothstep(WINTER_HEDGE_LOW, WINTER_HEDGE_HIGH, mean.a)) * smoothstep(0.2, 0.45, winter_noise(world * WINTER_HEDGE_BREAKS));
    float3 twigs = lerp(snow * WINTER_FIELD_EDGE, WINTER_FIELD_TWIGS, smoothstep(0.3, 0.55, winter_noise(world * 61.0)) * WINTER_HEDGE_TWIGS);
    return lerp(field, twigs, hedge);
}}"""
def _blur(low, high):
    old_plot = SOFT_FIELDS[SOFT_FIELDS.index("float3 winter_field_plot("):SOFT_FIELDS.rindex("}") + 1]
    return [(paths, a, r.replace(old_plot, _BLUR_PLOT.format(low=low, high=high))
             .replace("input.uv, WINTER_FIELD_BIAS).rgb,", "input.uv, WINTER_FIELD_BIAS),"))
            for paths, a, r in LOOKS["v25-hedgerows"]]
LOOKS["v27-hedges-c"] = _blur(0.5, 0.8)
LOOKS["v27-hedges-d"] = _blur(0.65, 0.92)
# The current look (user, 2026-10-23: fix D). Production: winter_integrated.hlsl
# winter_field_plot (WINTER_HEDGE_*), seasons.py winter/objects.
LOOKS["v28-laden"] = LOOKS["v27-hedges-d"]


# Volcanoes (user, 2026-10-23: "Did we not add winter treatment for
# volcanoes?"). The volcano material runs after the mountain snow: it repaints
# the cone with its rock and fades its foot apron to unseasoned ground. The
# apron now takes the Winter ground, and the cone a snow cap by activity:
# from the crater rim when dormant, the lower flanks when smoldering, none
# under an eruption's fresh ash. Cooled lava streaks stay mostly dark.
VOLCANO_GROUND = ("    ground_albedo = lerp(winter_dormant(ground_albedo), winter_snow_albedo(height_detail, input.world.xy),\n"
                  "        winter_cover(1, height_detail, 1 - saturate(normalize(input.normal).z), input.world.xy));\n")
VOLCANO_CAP = """    {
        float winter_cone = volcano_coverage(input.volcano_owner, mountain_rise);
        float winter_activity = volcano_activity(input.volcano_owner);
        float winter_warm = winter_activity < 0.5 ? DORMANT : winter_activity < 1.5 ? SMOKE : ERUPT;
        float winter_r = length(input.volcano_owner.xy) + (winter_noise(input.world.xy * 9.0) - 0.5) * 0.05;
        float winter_cap = smoothstep(winter_warm, winter_warm + 0.06, winter_r) *
            smoothstep(UP_LOW, UP_HIGH, normal.z) * (1 - STREAK * volcano_channel(input.volcano_owner)) *
            (0.75 + 0.5 * winter_noise(input.world.xy * 23.0));
        albedo = lerp(albedo, winter_snow_albedo(height_detail, input.world.xy), saturate(winter_cap * COVER) * winter_cone);
    }
"""
VOLCANO_CALL = "    albedo = volcano_surface(albedo, ground_albedo, input.volcano_owner, mountain_rise);\n"
def _volcano(base, **values):
    cap = VOLCANO_CAP
    for key, value in values.items():
        cap = cap.replace(key.upper(), str(value))
    return LOOKS[base] + [((MOUNTAIN,), VOLCANO_CALL + "#endif\n    float " + tail,
                           VOLCANO_GROUND + VOLCANO_CALL + cap + "#endif\n    float " + tail)
                          for tail in ("altitude", "ndl")]
LOOKS["v29-volcano"] = _volcano("v28-laden", dormant=0.05, smoke=0.14, erupt=0.40, up_low=0.3, up_high=0.6,
                                streak=0.7, cover=1.0)
# v29's cap barely showed: the cones are steeper than mountain snow's up
# threshold. v30 reads as a snow cap: from the warm radius down to a ragged
# snowline with tongues down the flanks, a light dusting below it, bare
# near-vertical faces and mostly dark cooled lava streaks.
VOLCANO_CAP2 = """    {
        float winter_cone = volcano_coverage(input.volcano_owner, mountain_rise);
        float winter_activity = volcano_activity(input.volcano_owner);
        float winter_warm = winter_activity < 0.5 ? DORMANT : winter_activity < 1.5 ? SMOKE : ERUPT;
        float winter_r = length(input.volcano_owner.xy);
        float winter_angle = atan2(input.volcano_owner.y, input.volcano_owner.x);
        float winter_line = SNOWLINE + (winter_noise(float2(winter_angle * 2.2, 0.37)) - 0.5) * 0.12 +
            (winter_noise(float2(winter_angle * 9.0, 1.7)) - 0.5) * 0.06;
        float winter_open = smoothstep(winter_warm, winter_warm + 0.04, winter_r + (winter_noise(input.world.xy * 9.0) - 0.5) * 0.04);
        float winter_cap = winter_open * lerp(DUST, 1, 1 - smoothstep(winter_line - 0.03, winter_line + 0.03, winter_r));
        winter_cap *= smoothstep(UP_LOW, UP_HIGH, normal.z) * (1 - STREAK * volcano_channel(input.volcano_owner)) *
            (0.8 + 0.4 * winter_noise(input.world.xy * 23.0));
        albedo = lerp(albedo, winter_snow_albedo(height_detail, input.world.xy), saturate(winter_cap) * winter_cone);
    }
"""
def _volcano2(base, **values):
    cap = VOLCANO_CAP2
    for key, value in values.items():
        cap = cap.replace(key.upper(), str(value))
    return LOOKS[base] + [((MOUNTAIN,), VOLCANO_CALL + "#endif\n    float " + tail,
                           VOLCANO_GROUND + VOLCANO_CALL + cap + "#endif\n    float " + tail)
                          for tail in ("altitude", "ndl")]
LOOKS["v30-volcano"] = _volcano2("v28-laden", dormant=0.05, smoke=0.13, erupt=0.40, snowline=0.21, dust=0.3,
                                 up_low=0.02, up_high=0.3, streak=0.7)


def apply(look: str, relative: str, text: str) -> str:
    for paths, anchor, replacement in LOOKS[look]:
        if relative not in paths:
            continue
        if text.count(anchor) != 1:
            raise ValueError(f"{look}: anchor occurs {text.count(anchor)} times in {relative}: {anchor[:60]!r}")
        text = text.replace(anchor, replacement)
    return text
