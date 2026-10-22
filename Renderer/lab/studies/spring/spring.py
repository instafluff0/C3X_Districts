"""Spring shader patches for the study's private shader roots.

A look is a set of #define calibrations, a per-tree leaf palette and anchored
edits to the flattened production shaders. An edit whose anchor is missing or
ambiguous fails loudly, so production drift can never silently drop part of a
look. Colour targets were measured on the selected concept
(Renderer/lab/out/seasons/scene-concepts/spring-vibrant.png, OKLab L/C/h).

The crown lighting edits follow the autumn study (../autumn/autumn.py): a
per-tree seed and an ellipsoidal crown normal carried with each instance.
"""
from __future__ import annotations

from pathlib import Path

COMMON = Path(__file__).with_name("spring_common.hlsl").read_text()
CITY = "Renderer/native/city_fidelity/"
TREES = CITY + "objects.hlsl"
RELIEF = CITY + "terrain.hlsl"
MOUNTAIN = CITY + "mountain.hlsl"
INTEGRATED = tuple(CITY + name for name in ("hydrology.hlsl", "feature.hlsl", "city.hlsl"))

# (cumulative share, OKLab lightness, chroma, hue) per broadleaf crown.
# Fresh greens: young lime, fresh, mid and a few deeper crowns.
LEAVES_V1 = [(0.20, 0.68, 0.145, 118), (0.55, 0.62, 0.135, 127), (0.80, 0.56, 0.120, 134),
             (0.95, 0.50, 0.100, 140), (1.00, 0.72, 0.150, 112)]
# Concept crowns are one rich mid-green family (h ~125-132); yellow-limes read autumnal.
LEAVES_V2 = [(0.30, 0.63, 0.135, 124), (0.70, 0.60, 0.130, 129), (0.92, 0.56, 0.120, 133),
             (1.00, 0.66, 0.140, 121)]
# Measured v2 crowns already match the concept's L/C/h spread; the concept keeps
# each crown one green and gets its yellow from sunlit tops, so narrow per-tree variety.
LEAVES_V3 = [(0.45, 0.61, 0.132, 126), (0.85, 0.59, 0.128, 129), (1.00, 0.64, 0.136, 124)]
# Close up v3 read lime: deeper, slightly bluer mid-greens.
LEAVES_V4 = [(0.45, 0.57, 0.128, 131), (0.85, 0.55, 0.122, 134), (1.00, 0.60, 0.132, 128)]
# v4/v5 crowns measured darker and bluer than the concept (L .49 vs .535, h 127 vs 121).
LEAVES_V6 = [(0.45, 0.60, 0.130, 127), (0.85, 0.58, 0.125, 130), (1.00, 0.63, 0.135, 124)]

V1 = {
    # Turf
    "SPRING_GRASS_SHIFT": 6, "SPRING_STRAW_SHIFT": 3, "SPRING_GRASS_CHROMA": 1.10, "SPRING_STRAW_CHROMA": 1.02,
    "SPRING_GRASS_LIGHT": 1.04, "SPRING_LUSH_SCALE": 1.1, "SPRING_LUSH_HUE": 10, "SPRING_LUSH_CHROMA": 0.25,
    "SPRING_LUSH_LIGHT": 0.16, "SPRING_HILL_LIGHT": 1.08, "SPRING_HILL_CHROMA": 1.15,
    # Meadow drifts (world tiles)
    "SPRING_DRIFT_STRETCH": 1.6, "SPRING_DRIFT_SCALE": 2.4, "SPRING_DRIFT_THRESHOLD": 0.60, "SPRING_DRIFT_SOFT": 0.10,
    "SPRING_PALETTE_SCALE": 0.9, "SPRING_DENSITY": 0.75, "SPRING_SECOND_LAYER": 0.6, "SPRING_CELL": 0.045,
    "SPRING_BLOOM_MIN": 0.022, "SPRING_BLOOM_MAX": 0.034, "SPRING_BLOOM_LIGHT": 1.0, "SPRING_BLOOM_CHROMA": 1.0,
    "SPRING_AA": 0.75, "SPRING_PETAL_MODE": 0, "SPRING_JITTER": 0.42, "SPRING_UPRIGHT_CELL": 0.8,
    "SPRING_NOTCH": 0.40, "SPRING_RIM": 1.0, "SPRING_SIBLING": 0, "SPRING_SIBLING_GAP": 2.4, "SPRING_GRID_TURN": 0,
    "SPRING_SLOPE": 0, "SPRING_BLOOM_UP": 0.6, "SPRING_BLOOM_NORMAL": 0,
    "SPRING_STEM": 0, "SPRING_TOP_LIGHT": 0, "SPRING_CORAL": 0, "SPRING_SNAP": 0, "SPRING_BUNCH": 0, "SPRING_BUNCH_GAP": 1.9, "SPRING_PIXEL_SIZES": 0, "SPRING_STRAW_COLOR": 0,
    "SPRING_PLUS_FROM": 0.85,
    "SPRING_SHARE_YELLOW": 0.28, "SPRING_SHARE_WHITE": 0.48, "SPRING_SHARE_PINK": 0.68, "SPRING_SHARE_BLUE": 0.84,
    "SPRING_PLAINS_DENSITY": 0.45, "SPRING_TUNDRA_DENSITY": 0.25, "SPRING_HILL_DENSITY": 0.5,
    "SPRING_LEAF_SHADE": 0.93, "SPRING_LEAF_DRIFT": 1.0, "SPRING_TUFT": 0, "SPRING_SPRINKLE": 0,
    "SPRING_WORLEY_PALETTE": 0, "SPRING_STRAY": 0, "SPRING_CLUMP": 0, "SPRING_CLUMP_SCALE": 9, "SPRING_WOODLAND": 0,
    "SPRING_WHITE": "float3(0.95, 0.03, 100)", "SPRING_YELLOW": "float3(0.86, 0.165, 92)", "SPRING_TURF": 0, "SPRING_TURF_BIAS": 7, "SPRING_TURF_HUE": 18, "SPRING_HILL_HUE": 0, "SPRING_HILL_PULL": 0,
    # Crowns
    "SPRING_BROADLEAF_GB": 3.2, "SPRING_CROWN": 0.85, "SPRING_WRAP": 0.2, "SPRING_FILL": 0.85,
    "SPRING_CROWN_BASE": 0.6, "SPRING_CROWN_TOP": 1.10, "SPRING_LEAF_LIGHT": 1.0, "SPRING_LEAF_CHROMA": 1.0,
    "SPRING_LEAF_DETAIL": 0.9, "SPRING_TRANSLUCENCY": 0.5, "SPRING_LEAF_BIAS": 0, "SPRING_SUN_HUE": 0, "SPRING_TREE_HUE_SPREAD": 8, "SPRING_BLOSSOM_SHARE": 0, "SPRING_BLOSSOM_COVER": 0.8,
    # Water
    "SPRING_WATER_HUE": 246, "SPRING_WATER_PULL": 0.7, "SPRING_WATER_CHROMA": 1.9, "SPRING_WATER_LIGHT": 0.84,
    "SPRING_RIVER_WATER": 0.6, "SPRING_WATER_NIGHT": 0.3,
}
# Looser drifts: thinner cores, ragged fringe, lone blooms, even palette shares,
# richer petals with leafy tufts; painterly turf; yellower lighter hill turf;
# softer crowns whose sunlit tops run yellower and shaded sides bluer.
V2 = {**V1,
      "SPRING_DENSITY": 0.55, "SPRING_DRIFT_SOFT": 0.16, "SPRING_DRIFT_THRESHOLD": 0.58, "SPRING_SPRINKLE": 0.035,
      "SPRING_WORLEY_PALETTE": 1, "SPRING_PALETTE_SCALE": 1.2, "SPRING_STRAY": 0.12,
      "SPRING_SHARE_YELLOW": 0.30, "SPRING_SHARE_WHITE": 0.52, "SPRING_SHARE_PINK": 0.68, "SPRING_SHARE_BLUE": 0.82,
      "SPRING_BLOOM_MIN": 0.018, "SPRING_BLOOM_MAX": 0.036, "SPRING_BLOOM_CHROMA": 1.2,
      "SPRING_TUFT": 0.9, "SPRING_LEAF_SHADE": 0.80, "SPRING_LEAF_DRIFT": 0.3,
      "SPRING_TURF": 0.9, "SPRING_HILL_LIGHT": 1.12, "SPRING_HILL_HUE": 107, "SPRING_HILL_PULL": 0.7,
      "SPRING_LEAF_DETAIL": 0.7, "SPRING_LEAF_BIAS": 1.0, "SPRING_SUN_HUE": 9, "SPRING_CROWN_TOP": 1.06,
      "SPRING_CROWN_BASE": 0.62, "SPRING_WATER_LIGHT": 0.80}


# Firmer drifts with only rare lone blooms; blade-scale turf strokes.
V3 = {**V2,
      "SPRING_SPRINKLE": 0.006, "SPRING_DENSITY": 0.65, "SPRING_DRIFT_THRESHOLD": 0.60, "SPRING_DRIFT_SOFT": 0.12,
      "SPRING_TURF": 0.8, "SPRING_TURF_BIAS": 2.5, "SPRING_TURF_HUE": 8, "SPRING_HILL_PULL": 0.5,
      "SPRING_SUN_HUE": 14, "SPRING_TREE_HUE_SPREAD": 5}


# Cream whites, lemon yellows, clumped blooms on darker leafy tufts, richer turf.
V4 = {**V3,
      "SPRING_WHITE": "float3(0.95, 0.035, 92)", "SPRING_YELLOW": "float3(0.88, 0.15, 97)", "SPRING_BLOOM_CHROMA": 1.1,
      "SPRING_CLUMP": 0.6, "SPRING_DENSITY": 0.72, "SPRING_DRIFT_THRESHOLD": 0.58, "SPRING_SPRINKLE": 0.004,
      "SPRING_LEAF_SHADE": 0.72, "SPRING_GRASS_CHROMA": 1.18, "SPRING_STRAW_SHIFT": 0,
      "SPRING_SUN_HUE": 8, "SPRING_CROWN_TOP": 1.02}


# Game-zoom pop: bigger, richer blooms and denser cores; woodland edge flowers;
# stronger blade strokes; more crown light and shade.
V5 = {**V4,
      "SPRING_BLOOM_MIN": 0.022, "SPRING_BLOOM_MAX": 0.042, "SPRING_BLOOM_CHROMA": 1.2,
      "SPRING_YELLOW": "float3(0.88, 0.14, 97)", "SPRING_DENSITY": 0.8, "SPRING_WOODLAND": 0.9,
      "SPRING_TURF": 1.2, "SPRING_CROWN_BASE": 0.55, "SPRING_CROWN_TOP": 1.06}


# Concept bloom shares (~yellow 38, white 30, pink 12, lilac 12, blue 8%): fewer
# blue drifts, more daisy white; slightly sparser drifts; lighter crowns.
V6 = {**V5,
      "SPRING_SHARE_YELLOW": 0.32, "SPRING_SHARE_WHITE": 0.58, "SPRING_SHARE_PINK": 0.74, "SPRING_SHARE_BLUE": 0.82,
      "SPRING_DENSITY": 0.7, "SPRING_SUN_HUE": 10, "SPRING_CROWN_BASE": 0.58}


# Balanced palettes: a hashed pick left v6's view 46% pink.
V7 = {**V6, "SPRING_WORLEY_PALETTE": 2}


# The concept's drifts are small loose clusters of separate, mixed blooms with
# dark gaps, on strongly stroked turf: smaller, more numerous drifts, thinner
# cores, finer palette cells and more strays, darker tufts, stronger strokes.
V8 = {**V7, "SPRING_DRIFT_SCALE": 3.6, "SPRING_DENSITY": 0.5, "SPRING_PALETTE_SCALE": 2.4, "SPRING_STRAY": 0.3,
      "SPRING_LEAF_SHADE": 0.62, "SPRING_TURF": 1.8, "SPRING_TURF_BIAS": 3.0}


# User, 2026-10-21: flowers only on grassland, plains and hills. No tundra
# blooms and no woodland-edge flowers (forest tiles); desert never had any.
V9 = {**V8, "SPRING_TUNDRA_DENSITY": 0, "SPRING_WOODLAND": 0}


# User, 2026-10-21: prefers the earlier small, sharp, intricate blooms to v5's
# larger, fuzzier ones: smaller than v4, crisper edges, lighter tufts; a little
# more density so small blooms still read as clusters.
V10 = {**V9, "SPRING_BLOOM_MIN": 0.016, "SPRING_BLOOM_MAX": 0.032, "SPRING_AA": 0.5,
       "SPRING_LEAF_SHADE": 0.75, "SPRING_DENSITY": 0.6}


# v8-v10's strokes (1.8) bleached the turf toward pale yellow-green; v7 strength.
V11 = {**V10, "SPRING_TURF": 1.2}


# v11 read 59% yellow, 1% blue (yellow is also a companion in three palettes)
# and 3.1% bloom pixels against the concept's 3.8%: rebalance shares, fewer
# strays, more blooms at the same small size.
V12 = {**V11, "SPRING_SHARE_YELLOW": 0.24, "SPRING_SHARE_WHITE": 0.50, "SPRING_SHARE_PINK": 0.64,
       "SPRING_SHARE_BLUE": 0.80, "SPRING_STRAY": 0.2, "SPRING_DENSITY": 0.75}


# User, 2026-10-22: loves the blossoms (now part of the look); wants crisper,
# more defined blooms, worried about fluff at Renderer64's 3x zoom. Notched
# petals with per-petal shade, a hard eye and a hard contact shadow replace
# the soft lobes and soft halo.
V13 = {**V12, "SPRING_BLOSSOM_SHARE": 0.12, "SPRING_PETAL_MODE": 1, "SPRING_LEAF_SHADE": 0.72}


# v13's notched petals still read as smudges: flat on the ground the isometric
# view squashes each bloom 2:1 (2-3 px tall at 3x), and the AA footprint was
# about twice a pixel. Upright, camera-facing blooms on a screen-proportional
# grid with exact pixel footprints; a little smaller so the round blooms keep
# about the same weight; tighter jitter so they fit their cells.
V14 = {**V13, "SPRING_PETAL_MODE": 2, "SPRING_JITTER": 0.30,
       "SPRING_BLOOM_MIN": 0.016 * 0.8, "SPRING_BLOOM_MAX": 0.032 * 0.8}


# v14's deep notches made some 6 px blooms lumpy: a shallower scallop, a shaded
# petal rim (only once a bloom is a few pixels wide) and a third-pixel edge.
V15 = {**V14, "SPRING_NOTCH": 0.25, "SPRING_RIM": 0.78, "SPRING_AA": 0.35}


# User, 2026-10-22: blooms looked evenly spaced, grid-like. v14 had cut jitter
# to .30 on a grid aligned with the screen axes (nearest-neighbour directions
# peaked at 2.3x the mean). Centres anywhere in a cell (2x2 neighbour search),
# tilted grids, sibling blooms in little groups, finer and stronger clumps;
# base density lowered so the bloom count stays about the same.
V16 = {**V15, "SPRING_JITTER": 1.0, "SPRING_GRID_TURN": 0.30, "SPRING_SIBLING": 0.45,
       "SPRING_CLUMP": 0.85, "SPRING_CLUMP_SCALE": 14, "SPRING_DENSITY": 0.55}


# User, 2026-10-22: flowers should take ground depth and angle into account.
# They already sat in the terrain surface (height, occlusion, cast shadows),
# but their screen shape and spacing used flat-ground mapping, their density
# was per pixel, and petals were lit with the grass's bumpy slope normal.
V17 = {**V16, "SPRING_SLOPE": 1, "SPRING_BLOOM_NORMAL": 1}


# User, 2026-10-22: likes the concept's crisper flowers (the target panel of
# the comparisons). At pixel level the concept's heads are slightly bigger,
# top-lit, with a darker pixel beneath (stem and shade), in tight bunches on
# darker turf, and richer (coral, pink, clean white, golden yellow).
V18 = {**V17, "SPRING_STEM": 0.9, "SPRING_TOP_LIGHT": 0.16, "SPRING_CORAL": 1, "SPRING_BLOOM_CHROMA": 1.25,
       "SPRING_BLOOM_MIN": 0.016, "SPRING_BLOOM_MAX": 0.032, "SPRING_LEAF_DRIFT": 0.7,
       "SPRING_SIBLING": 0.7, "SPRING_SIBLING_GAP": 2.0, "SPRING_CLUMP": 0.9, "SPRING_CLUMP_SCALE": 22}


# At normal zoom v18's 1-2 px blooms still read as pastel mush beside the
# concept's solid pixel flowers: each disc smeared partial colour over 2-4
# pixels. Pixel-snapped centres; blooms keep their size, colour and placement.
V19 = {**V18, "SPRING_SNAP": 1}


# The concept's flowers read crisp because each is an isolated bright 1-2 px
# head with grass around it; v19's touch and merge into soft patches. Smaller
# heads, siblings apart, gentler clumping and lower density over slightly
# larger drifts (about the same number of flowers).
V20 = {**V19, "SPRING_BLOOM_MIN": 0.012, "SPRING_BLOOM_MAX": 0.026, "SPRING_SIBLING_GAP": 3.4, "SPRING_SIBLING": 0.5,
       "SPRING_CLUMP": 0.5, "SPRING_DENSITY": 0.42, "SPRING_DRIFT_THRESHOLD": 0.54, "SPRING_DRIFT_SOFT": 0.14}


# User, 2026-10-22: the concept's patches are more clumped; ours are spread
# evenly. Measured in 24 px grass blocks: concept 49% bare, top fifth of
# blocks hold 76% of petals; v20 33% and 64%. Fewer, denser, sharper drifts
# at about the same petal total (v21 moderate, v21b stronger).
V21 = {**V20, "SPRING_DRIFT_THRESHOLD": 0.61, "SPRING_DRIFT_SOFT": 0.08, "SPRING_DENSITY": 0.62,
       "SPRING_SPRINKLE": 0.0015}
V21B = {**V20, "SPRING_DRIFT_THRESHOLD": 0.64, "SPRING_DRIFT_SOFT": 0.06, "SPRING_DENSITY": 0.80,
        "SPRING_SPRINKLE": 0.0015}


# v21b reached the concept's bare share but its patches are large, ragged
# and thin inside; the concept's are smaller, more numerous, compact and dense.
# Smaller drifts, harder edges, fuller and more even interiors.
V22 = {**V21B, "SPRING_DRIFT_SCALE": 5.0, "SPRING_DRIFT_THRESHOLD": 0.66, "SPRING_DRIFT_SOFT": 0.05,
       "SPRING_DENSITY": 1.0, "SPRING_CLUMP": 0.3}
V22B = {**V21B, "SPRING_DRIFT_SCALE": 4.2, "SPRING_DRIFT_THRESHOLD": 0.64, "SPRING_DRIFT_SOFT": 0.05,
        "SPRING_DENSITY": 0.95, "SPRING_CLUMP": 0.3}


# The concept's clumps are posies: petal blobs median 3 px, largest tenth
# 16 px+ (v22b 2 px, 9 px), and 71% of 12 px grass blocks bare (v22b 65%).
# Bunches of up to five touching heads, fewer bunch centres, centres kept near
# their cell's middle so a bunch fits the 3x3 search.
V23 = {**V22B, "SPRING_BUNCH": 4, "SPRING_BUNCH_GAP": 1.9, "SPRING_JITTER": 0.5, "SPRING_DENSITY": 0.42,
       "SPRING_DRIFT_THRESHOLD": 0.65}


# v23 posies are still loose and small beside the concept's (blobs median
# 2 px / p90 10 vs 3 / 16): tighter posies, more of them in each patch, fewer
# patches.
V24 = {**V23, "SPRING_BUNCH_GAP": 1.6, "SPRING_DENSITY": 0.6, "SPRING_DRIFT_THRESHOLD": 0.68,
       "SPRING_DRIFT_SOFT": 0.04}


# v24 matches the concept's clumping numbers but reads sparse: the concept's
# patches are long thin strips along the screen horizontal, packed close, and
# its plains carry as many flowers as grassland. Longer, thinner, denser
# patches; plains density .45 -> .8.
V25 = {**V24, "SPRING_DRIFT_STRETCH": 2.6, "SPRING_DRIFT_SCALE": 6.0, "SPRING_DENSITY": 0.75,
       "SPRING_PLAINS_DENSITY": 0.8, "SPRING_DRIFT_THRESHOLD": 0.69}


# User, 2026-10-22: "clumps" meant density and positioning, not posies; v25's
# touching heads merged into fuzzy blobs. The concept's patches are dense mats
# of tiny separate crisp dots in mixed colours, with plain grass between.
# No posies or siblings, single-pixel dots (normal zoom) on a finer grid packed
# densely inside v25's strip patches, more colours mixed per patch, light
# shadows and no stems so dots stay clean.
V26 = {**V25, "SPRING_BUNCH": 0, "SPRING_SIBLING": 0, "SPRING_JITTER": 1.0, "SPRING_UPRIGHT_CELL": 0.55,
       "SPRING_BLOOM_MIN": 0.010, "SPRING_BLOOM_MAX": 0.018, "SPRING_DENSITY": 1.6, "SPRING_CLUMP": 0.2,
       "SPRING_STRAY": 0.45, "SPRING_STEM": 0, "SPRING_TUFT": 0.35, "SPRING_LEAF_DRIFT": 0.3,
       "SPRING_DRIFT_THRESHOLD": 0.69}
V26B = {**V26, "SPRING_DRIFT_THRESHOLD": 0.665}


# v26 went too far: tiny sparse dots, patches nearly invisible (2.0% petal
# pixels, concept 2.7%). Fuzz at normal zoom came from radii of .6-1.2 px
# spilling into neighbour pixels. Exact pixel shapes (solid pixel or five-pixel
# plus with a warm centre), tighter packing inside patches, more patch area
# and stronger colour.
V27 = {**V26, "SPRING_PIXEL_SIZES": 1, "SPRING_PLUS_FROM": 0.85, "SPRING_UPRIGHT_CELL": 0.7,
       "SPRING_BLOOM_MIN": 0.012, "SPRING_BLOOM_MAX": 0.026, "SPRING_DENSITY": 2.2, "SPRING_SECOND_LAYER": 0.8,
       "SPRING_DRIFT_THRESHOLD": 0.675, "SPRING_BLOOM_CHROMA": 1.4}


# On plains, white and yellow dots fade into straw: there most drifts take the
# pink, blue or lilac palettes instead.
V28 = {**V27, "SPRING_STRAW_COLOR": 0.8}


def palette(entries) -> str:
    branches = []
    for index, (share, light, chroma, hue) in enumerate(entries):
        test = "" if index == len(entries) - 1 else f"if (seed < {share}) "
        prefix = "    " if index == 0 else "    else "
        branches.append(f"{prefix}{test}pal = float3({light}, {chroma}, {hue});")
    return ("// Per-tree palette; the atlas keeps its own light/dark leaf structure.\n"
            "float3 spring_leaf_color(float3 albedo, float3 average, float seed) {\n"
            "    float3 pal;\n" + "\n".join(branches) + "\n"
            "    pal.z += (frac(seed * 7.31) - 0.5) * SPRING_TREE_HUE_SPREAD;\n"
            "    pal.x += (frac(seed * 13.7) - 0.5) * 0.05;\n"
            "    float3 src = spring_lch(albedo), avg = spring_lch(average);\n"
            "    float L = pal.x * SPRING_LEAF_LIGHT + (src.x - avg.x) * SPRING_LEAF_DETAIL;\n"
            "    float C = pal.y * SPRING_LEAF_CHROMA * saturate(0.45 + 0.75 * src.x / max(avg.x, 0.05));\n"
            "    float h = pal.z + clamp(src.z - avg.z, -25, 25) * 0.3;\n"
            "    return spring_from_lch(float3(max(L, 0.05), C, h));\n}\n")


def edits(values, leaves, *, meadow=True, crowns=True, water=True, mountains=True, turf=False, woodland=False):
    common = "".join(f"#define {key} {value}\n" for key, value in values.items()) + COMMON + palette(leaves)
    result = [
        ((TREES, RELIEF), "Output shade(P input) {", common + "Output shade(P input) {"),
        # Relief: fresh turf everywhere but forest/jungle floors, lighter hill turf.
        ((RELIEF,), "    float surface_occlusion = 1;",
         "    float surface_occlusion = 1;\n    float4 spring_bloom = 0;\n    float spring_leaf = 0;"),
        ((RELIEF,), "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction",
         "    if (!(input.material.y > 2.5 && input.material.y < 4.5)) {\n"
         "        albedo = spring_ground(albedo, input.world.xy);\n"
         "        float spring_hill = input.material.y < 1.5 ? input.material.z * smoothstep(0.02, 0.2, input.material.x) : 0;\n"
         "        float3 spring_hill_lch = spring_lch(albedo);\n"
         "        float spring_turf = spring_hill * smoothstep(0.02, 0.05, spring_hill_lch.y);\n"
         "        spring_hill_lch *= float3(lerp(1, SPRING_HILL_LIGHT, spring_turf), lerp(1, SPRING_HILL_CHROMA, spring_turf), 1);\n"
         "        spring_hill_lch.z = lerp(spring_hill_lch.z, SPRING_HILL_HUE, spring_turf * SPRING_HILL_PULL);\n"
         "        albedo = spring_from_lch(spring_hill_lch);\n"
         "        // Leafy flower stems darken the turf a little under each drift.\n"
         "        albedo *= lerp(1, SPRING_LEAF_SHADE, spring_leaf);\n"
         "        albedo = albedo * (1 - spring_bloom.a) + spring_bloom.rgb;\n"
         "        // Flower heads stand upright: light them with the slope's normal bent\n"
         "        // toward up, without the grass's micro relief or its sheen.\n"
         "        geometric = normalize(lerp(geometric, normalize(lerp(normalize(input.normal), float3(0, 0, 1), SPRING_BLOOM_UP)),\n"
         "                                   spring_bloom.a * SPRING_BLOOM_NORMAL));\n"
         "        specular_map = lerp(specular_map, 0.02, spring_bloom.a * SPRING_BLOOM_NORMAL);\n"
         "    }\n"
         "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction"),
        # Coasts and beaches (integrated terrain shaders).
        (INTEGRATED, "float4 q6_raw_feature(FeaturePixelInput input)\n{", common + "float4 q6_raw_feature(FeaturePixelInput input)\n{"),
        (INTEGRATED, "#ifdef Q8_DEBUG_ALBEDO\n        return float4(albedo,1);",
         "        albedo = spring_ground(albedo, world_position);\n#ifdef Q8_DEBUG_ALBEDO\n        return float4(albedo,1);"),
    ]
    if meadow:
        result += [
            # Flowers on open ground: grassland full, plains lighter, hill turf
            # partly; never desert or exposed hill rock (tundra from v9 on).
            ((RELIEF,), "        albedo *= 1 + clamp((grain - 0.303) * 0.9, -0.16, 0.22) * grass_plains_detail;\n    }",
             "        albedo *= 1 + clamp((grain - 0.303) * 0.9, -0.16, 0.22) * grass_plains_detail;\n"
             "        float spring_density = lerp(lerp(1, SPRING_PLAINS_DENSITY, plains_weight), SPRING_TUNDRA_DENSITY, tundra_weight) *\n"
             "            (1 - desert_weight) * (1 - rocky_band * lerp(1 - SPRING_HILL_DENSITY, 1, hill_rock));\n"
             "        spring_straw = plains_weight * (1 - desert_weight);\n"
             "        spring_bloom = spring_meadow(input.world.xy, input.world.z, spring_density, ShadowL.xy, spring_leaf);\n"
             "    }"),
        ]
    if woodland:
        result += [
            # Anemones and bluebells where the forest floor thins at the edge; a
            # bloom draws opaque so the edge reads as flowers, not bare floor.
            ((RELIEF,), "        alpha *= floor_sample.a;\n    } else if (input.material.y > 1.5) {",
             "        float spring_edge_alpha = alpha;\n"
             "        alpha *= floor_sample.a;\n"
             "        if (!jungle_floor) {\n"
             "            float spring_fringe = smoothstep(0.06, 0.22, floor_sample.a) * (1 - smoothstep(0.40, 0.85, floor_sample.a));\n"
             "            float4 spring_wood = spring_woodland(input.world.xy, spring_fringe, ShadowL.xy);\n"
             "            albedo = albedo * (1 - spring_wood.a) + spring_wood.rgb;\n"
             "            alpha = lerp(alpha, spring_edge_alpha, spring_wood.a);\n"
             "        }\n"
             "    } else if (input.material.y > 1.5) {"),
        ]
    if turf:
        # After the meadow edit, which keeps the grain line as its own first line.
        result += [
            ((RELIEF,), "        albedo *= 1 + clamp((grain - 0.303) * 0.9, -0.16, 0.22) * grass_plains_detail;\n",
             "        albedo *= 1 + clamp((grain - 0.303) * 0.9, -0.16, 0.22) * grass_plains_detail;\n"
             "        {\n"
             "            float3 lum = float3(0.2126, 0.7152, 0.0722);\n"
             "            // Against a mid mip only blade-scale strokes carry the contrast.\n"
             "            float3 grass_mean = SPRING_TURF_BIAS > 6 ? grass_low_blend : GrassColor.SampleBias(Wrap, uv0, SPRING_TURF_BIAS).rgb;\n"
             "            float grass_ratio = dot(grass_fine, lum) / max(dot(grass_mean, lum), 0.01);\n"
             "            float plains_ratio = dot(plains, lum) / max(dot(PlainsColor.SampleBias(Wrap, uv1, min(SPRING_TURF_BIAS, 5.0)).rgb, lum), 0.01);\n"
             "            float turf = clamp((lerp(grass_ratio, plains_ratio, plains_weight) - 1) * SPRING_TURF, -0.3, 0.35) *\n"
             "                (1 - tundra_weight) * (1 - desert_weight) * (1 - rocky_band);\n"
             "            float3 turf_lch = spring_lch(albedo);\n"
             "            turf_lch.x *= 1 + turf;\n"
             "            turf_lch.z -= turf * SPRING_TURF_HUE;\n"
             "            albedo = spring_from_lch(turf_lch);\n"
             "        }\n"),
        ]
    if crowns:
        result += [
            ((TREES,), "    float2 secondary : TEXCOORD3;\n};\nstruct Output",
             "    float2 secondary : TEXCOORD3;\n    nointerpolation float spring_seed : TEXCOORD4;\n"
             "    float4 crown : TEXCOORD5;\n};\nstruct Output"),
            ((TREES,), "    output.secondary = input.secondary;\n    return output;",
             "    output.secondary = input.secondary;\n    output.spring_seed = 0;\n"
             "    output.crown = float4(input.normal, 1);\n    return output;"),
            ((TREES,), " P o=VSMain(v);\n o.position.xy=(floor(v.position.xy*256+.5)/256+i.placement_view.xy)",
             " P o=VSMain(v);\n"
             " o.spring_seed=frac(sin(dot(i.place0.xy+i.place0.zw*7.3+i.place1.xy*3.1,float2(12.9898,78.233)))*43758.5453);\n"
             " {float3 q=i.source_position-float3(0,0,.15);float3 g=float3(q.x/.04,q.y/.04,q.z/.012);\n"
             "  o.crown=float4(normalize(float3(g.x*co-g.y*si,-(g.x*si+g.y*co),g.z/(150.f/(.82f*64.f)))),saturate(i.source_position.z/.24));}\n"
             " o.position.xy=(floor(v.position.xy*256+.5)/256+i.placement_view.xy)"),
            ((TREES,), "    float3 normal = geometric;\n    if (kind < 0.5) {",
             "    float3 normal = geometric;\n    bool spring_leaf = false;\n    if (kind < 0.5) {"),
            ((TREES,), "        if (has_normal)\n            normal = mapped_normal(geometric, input.world, input.uv,\n"
                       "                                   foliage || repeat_address);",
             "        if (foliage) {\n"
             "            // Lab stand-in for pack metadata: the broadleaf atlas is yellow-green.\n"
             "            float3 average = BaseColor.SampleLevel(Clamp, float2(0.5, 0.5), 15).rgb;\n"
             "            spring_leaf = average.g > SPRING_BROADLEAF_GB * average.b;\n"
             "            float leafness = saturate((base.g - base.r) * 25 + 0.75);\n"
             "            if (spring_leaf) {\n"
             "                float3 soft = lerp(albedo, BaseColor.SampleBias(Wrap, input.uv, SPRING_LEAF_BIAS).rgb, SPRING_LEAF_BIAS > 0 ? 0.7 : 0);\n"
             "                float3 fresh = spring_leaf_color(soft, average, input.spring_seed);\n"
             "                // Sunlit crown tops run yellower, shaded undersides bluer.\n"
             "                float3 fresh_lch = spring_lch(fresh);\n"
             "                fresh_lch.z -= SPRING_SUN_HUE * (smoothstep(0.15, 0.85, input.crown.w) - 0.4);\n"
             "                fresh = spring_from_lch(fresh_lch);\n"
             "                // Fruit trees in bloom: the atlas's lighter leaf clusters carry blossom.\n"
             "                if (input.spring_seed > 1 - SPRING_BLOSSOM_SHARE) {\n"
             "                    float3 src = spring_lch(albedo), avg = spring_lch(average);\n"
             "                    float pink = frac(input.spring_seed * 31.7);\n"
             "                    float3 petal = pink < 0.45 ? float3(0.93, 0.025, 20) : (pink < 0.85 ? float3(0.86, 0.075, 355) : float3(0.78, 0.115, 350));\n"
             "                    float flower = smoothstep(-0.06, 0.05, src.x - avg.x + (frac(input.spring_seed * 57.1) - 0.5) * 0.04);\n"
             "                    fresh = lerp(fresh, spring_from_lch(float3(petal.x + (src.x - avg.x) * 0.5, petal.yz)), flower * SPRING_BLOSSOM_COVER);\n"
             "                }\n"
             "                albedo = lerp(albedo, fresh, leafness) *\n"
             "                    lerp(SPRING_CROWN_BASE, SPRING_CROWN_TOP, smoothstep(0.15, 0.85, input.crown.w));\n"
             "            }\n"
             "        }\n"
             "        if (has_normal)\n            normal = mapped_normal(geometric, input.world, input.uv,\n"
             "                                   foliage || repeat_address);"),
            ((TREES,), "    float ndl = saturate(dot(normal, light_direction));\n    float diffuse = ndl;",
             "    if (spring_leaf) normal = normalize(lerp(normal, normalize(input.crown.xyz), SPRING_CROWN));\n"
             "    float ndl = saturate(dot(normal, light_direction));\n"
             "    float diffuse = spring_leaf ? saturate((dot(normal, light_direction) + SPRING_WRAP) / (1 + SPRING_WRAP)) : ndl;"),
            # Young leaves are thin: sunlight glows through backlit crowns.
            ((TREES,), "    if (!(kind > 0.5 && kind < 1.5) && input.secondary.y > 0.5)\n        radiance += Emissive.Sample(Clamp, input.uv).rgb * 0.035;",
             "    if (spring_leaf)\n"
             "        radiance += albedo * SunColorExposure.rgb * Sun.w * shadow * SPRING_TRANSLUCENCY *\n"
             "            pow(saturate(dot(normalize(View.xyz), -light_direction) * 0.5 + 0.5), 2) * (1 - ndl);\n"
             "    if (!(kind > 0.5 && kind < 1.5) && input.secondary.y > 0.5)\n        radiance += Emissive.Sample(Clamp, input.uv).rgb * 0.035;"),
            ((TREES,), "    float foliage_fill = kind > 0.5 && kind < 1.5 ? 0.72 : 1.0;",
             "    float foliage_fill = spring_leaf ? SPRING_FILL : (kind > 0.5 && kind < 1.5 ? 0.72 : 1.0);"),
        ]
    if water:
        result += [
            (INTEGRATED, "        return q3_water_material(input);\n#endif",
             "    {\n        float4 spring_surface = q3_water_material(input);\n"
             "        float spring_strength = (input.surface_kind > 8.5 && input.surface_kind < 9.5 ? SPRING_RIVER_WATER : 1) *\n"
             "            lerp(SPRING_WATER_NIGHT, 1, saturate(environment_sun_intensity * 2.5));\n"
             "        return float4(spring_water(spring_surface.rgb, spring_strength), spring_surface.a);\n    }\n#endif"),
        ]
    if mountains:
        result += [
            ((MOUNTAIN,), "Output shade(P input) {", common + "Output shade(P input) {"),
            ((MOUNTAIN,), "#endif\n#ifdef SANDBOX_TERRAIN_MATERIAL\n#ifdef BEAUTY_VOLCANO_MATERIAL",
             "    albedo = spring_ground(albedo, input.world.xy);\n"
             "#endif\n#ifdef SANDBOX_TERRAIN_MATERIAL\n#ifdef BEAUTY_VOLCANO_MATERIAL"),
        ]
    return result


PROBE = [
    # Diagnostic: natural relief = magenta, integrated terrain/coast = cyan.
    ((RELIEF,), "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction",
     "    albedo = float3(1, 0, 1);\n#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction"),
    (INTEGRATED, "#ifdef Q8_DEBUG_ALBEDO\n        return float4(albedo,1);",
     "        albedo = float3(0, 1, 1);\n#ifdef Q8_DEBUG_ALBEDO\n        return float4(albedo,1);"),
]

# Diagnostic: foliage passing the broadleaf test = magenta, other foliage = cyan,
# with the atlas average's G/B ratio in red for the cyan bodies.
FOLIAGE_PROBE = [
    ((TREES,), "        if (has_normal)\n            normal = mapped_normal(geometric, input.world, input.uv,\n"
               "                                   foliage || repeat_address);",
     "        if (foliage) {\n"
     "            float3 average = BaseColor.SampleLevel(Clamp, float2(0.5, 0.5), 15).rgb;\n"
     "            albedo = average.g > 3.2 * average.b ? float3(1, 0, 1) : float3(saturate((average.g / max(average.b, 0.001) - 1) / 4), 1, 1);\n"
     "        }\n"
     "        if (has_normal)\n            normal = mapped_normal(geometric, input.world, input.uv,\n"
     "                                   foliage || repeat_address);"),
]

LOOKS = {
    "identity": [],
    "probe": PROBE,
    "foliage-probe": FOLIAGE_PROBE,
    "v1": edits(V1, LEAVES_V1),
    "v1-ground": edits(V1, LEAVES_V1, meadow=False, crowns=False, water=False),
    "v1-blossom": edits({**V1, "SPRING_BLOSSOM_SHARE": 0.12}, LEAVES_V1),
    "v2": edits(V2, LEAVES_V2, turf=True),
    "v3": edits(V3, LEAVES_V3, turf=True),
    "v4": edits(V4, LEAVES_V4, turf=True),
    "v5": edits(V5, LEAVES_V4, turf=True, woodland=True),
    "v5-blossom": edits({**V5, "SPRING_BLOSSOM_SHARE": 0.12}, LEAVES_V4, turf=True, woodland=True),
    "v6": edits(V6, LEAVES_V6, turf=True, woodland=True),
    "v6-blossom": edits({**V6, "SPRING_BLOSSOM_SHARE": 0.12}, LEAVES_V6, turf=True, woodland=True),
    "v7": edits(V7, LEAVES_V6, turf=True, woodland=True),
    "v7-blossom": edits({**V7, "SPRING_BLOSSOM_SHARE": 0.12}, LEAVES_V6, turf=True, woodland=True),
    "v8": edits(V8, LEAVES_V6, turf=True, woodland=True),
    "v9": edits(V9, LEAVES_V6, turf=True),
    "v9-blossom": edits({**V9, "SPRING_BLOSSOM_SHARE": 0.12}, LEAVES_V6, turf=True),
    "v10": edits(V10, LEAVES_V6, turf=True),
    "v10-blossom": edits({**V10, "SPRING_BLOSSOM_SHARE": 0.12}, LEAVES_V6, turf=True),
    "v11": edits(V11, LEAVES_V6, turf=True),
    "v11-blossom": edits({**V11, "SPRING_BLOSSOM_SHARE": 0.12}, LEAVES_V6, turf=True),
    "v12": edits(V12, LEAVES_V6, turf=True),
    "v12-blossom": edits({**V12, "SPRING_BLOSSOM_SHARE": 0.12}, LEAVES_V6, turf=True),
    "v13": edits(V13, LEAVES_V6, turf=True),
}


def at_3x(values):
    """Stand-in for Renderer64's 3x zoom in the 2x `close` view.

    The Lab preview accepts tiles up to 256 px (2x); blooms and their spacing
    scaled 1.5x there cover exactly the pixels they would at 384 px (3x).
    """
    return {**values, **{key: values[key] * 1.5 for key in ("SPRING_CELL", "SPRING_BLOOM_MIN", "SPRING_BLOOM_MAX")}}


LOOKS["v12-blossom-3x"] = edits(at_3x({**V12, "SPRING_BLOSSOM_SHARE": 0.12}), LEAVES_V6, turf=True)
LOOKS["v13-3x"] = edits(at_3x(V13), LEAVES_V6, turf=True)
LOOKS["v14"] = edits(V14, LEAVES_V6, turf=True)
LOOKS["v14-3x"] = edits(at_3x(V14), LEAVES_V6, turf=True)
LOOKS["v15"] = edits(V15, LEAVES_V6, turf=True)
LOOKS["v15-3x"] = edits(at_3x(V15), LEAVES_V6, turf=True)
LOOKS["v16"] = edits(V16, LEAVES_V6, turf=True)
LOOKS["v16-3x"] = edits(at_3x(V16), LEAVES_V6, turf=True)
LOOKS["v17"] = edits(V17, LEAVES_V6, turf=True)
LOOKS["v17-3x"] = edits(at_3x(V17), LEAVES_V6, turf=True)
LOOKS["v18"] = edits(V18, LEAVES_V6, turf=True)
LOOKS["v18-3x"] = edits(at_3x(V18), LEAVES_V6, turf=True)
LOOKS["v19"] = edits(V19, LEAVES_V6, turf=True)
LOOKS["v19-3x"] = edits(at_3x(V19), LEAVES_V6, turf=True)
LOOKS["v20"] = edits(V20, LEAVES_V6, turf=True)
LOOKS["v20-3x"] = edits(at_3x(V20), LEAVES_V6, turf=True)
LOOKS["v21"] = edits(V21, LEAVES_V6, turf=True)
LOOKS["v21b"] = edits(V21B, LEAVES_V6, turf=True)
LOOKS["v22"] = edits(V22, LEAVES_V6, turf=True)
LOOKS["v22b"] = edits(V22B, LEAVES_V6, turf=True)
LOOKS["v23"] = edits(V23, LEAVES_V6, turf=True)
LOOKS["v23-3x"] = edits(at_3x(V23), LEAVES_V6, turf=True)
LOOKS["v24"] = edits(V24, LEAVES_V6, turf=True)
LOOKS["v24-3x"] = edits(at_3x(V24), LEAVES_V6, turf=True)
LOOKS["v25"] = edits(V25, LEAVES_V6, turf=True)
LOOKS["v25-3x"] = edits(at_3x(V25), LEAVES_V6, turf=True)
LOOKS["v26"] = edits(V26, LEAVES_V6, turf=True)
LOOKS["v26b"] = edits(V26B, LEAVES_V6, turf=True)
LOOKS["v26-3x"] = edits(at_3x(V26), LEAVES_V6, turf=True)
LOOKS["v27"] = edits(V27, LEAVES_V6, turf=True)
LOOKS["v27-3x"] = edits(at_3x(V27), LEAVES_V6, turf=True)
LOOKS["v28"] = edits(V28, LEAVES_V6, turf=True)
LOOKS["v28-3x"] = edits(at_3x(V28), LEAVES_V6, turf=True)
# Diagnostic: an unjittered field of large white blooms everywhere flowers can
# grow, so the shape on hills is obvious: flat-ground mapping vs height-aware.
SLOPE_PROBE = {"SPRING_DRIFT_THRESHOLD": -1, "SPRING_DENSITY": 1, "SPRING_SPRINKLE": 0, "SPRING_CLUMP": 0,
               "SPRING_SIBLING": 0, "SPRING_JITTER": 0, "SPRING_SECOND_LAYER": 0, "SPRING_HILL_DENSITY": 1,
               "SPRING_PLAINS_DENSITY": 1, "SPRING_BLOOM_MIN": 0.05, "SPRING_BLOOM_MAX": 0.05, "SPRING_CELL": 0.09,
               "SPRING_SHARE_YELLOW": 0, "SPRING_SHARE_WHITE": 1.0, "SPRING_STRAY": 0, "SPRING_GRID_TURN": 0}
LOOKS["slope-probe-flat"] = edits({**V16, **SLOPE_PROBE}, LEAVES_V6, turf=True)
LOOKS["slope-probe-height"] = edits({**V17, **SLOPE_PROBE}, LEAVES_V6, turf=True)


def apply(look: str, relative: str, text: str) -> str:
    for paths, anchor, replacement in LOOKS[look]:
        if relative not in paths:
            continue
        if text.count(anchor) != 1:
            raise ValueError(f"{look}: anchor occurs {text.count(anchor)} times in {relative}: {anchor[:60]!r}")
        text = text.replace(anchor, replacement)
    return text
