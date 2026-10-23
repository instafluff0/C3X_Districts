"""Seasonal shading for Renderer64: one shader set, every season, chosen at run time.

The approved Lab looks (Autumn v18, Winter v24-laden, Spring v20; see
`Renderer/lab/studies/*`) were each written as anchored edits to the flattened
production shaders, one season at a time. Here every one of those edits is a
*marked insertion* next to its anchor, never a replacement:

- each season's code runs inside `[branch] if (C3X_SEASON_INDEX == n)`, where the
  index comes from the frame constants (`EnvironmentState::season`), so Summer
  executes exactly the code it ran before and the other seasons cost nothing
  when they are not active;
- where a study rewrote a line, the original line stays and the season
  overrides its result inside the guard;
- the season helper libraries (`*.hlsl` beside this file) are file-scope code
  extracted from the approved looks.

Every insertion is wrapped in `/*c3x-season:TAG*/ ... /*c3x-season-end*/`, so
`strip()` restores the original bytes and `apply()` can be re-run on a tree that
already carries seasons. Insertion points are located in the stripped text, so
seasons sharing an anchor cannot disturb each other.

Game-path notes (Renderer64's fresh pipeline rewrites some shader text at run
time, see `sandbox/fresh_pipeline.h` and `sandbox/terrain_material_fast.h`):
season code never touches those runtime anchors, and the relief turf edits take
their own grass samples instead of the Lab-only `grass_fine`/`grass_low_blend`.
Open water in the game is drawn by `sandbox/water_surface.hlsl`, which already
tints Coast, Sea and Ocean; seasons recolour it relatively, keeping that tint.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CITY = "Renderer/native/city_fidelity/"
TERRAIN, MOUNTAIN, OBJECTS = CITY + "terrain.hlsl", CITY + "mountain.hlsl", CITY + "objects.hlsl"
HYDROLOGY = CITY + "hydrology.hlsl"
FEATURE_SET = tuple(CITY + n for n in ("feature.hlsl", "city.hlsl"))
FIELD_SET = FEATURE_SET + tuple(CITY + n for n in ("rigid_feature.hlsl", "resource_body.hlsl"))
INTEGRATED = (HYDROLOGY,) + FIELD_SET
WATER_SURFACE = "Renderer/sandbox/water_surface.hlsl"
NATURAL = (TERRAIN, MOUNTAIN, OBJECTS)
TARGETS = NATURAL + INTEGRATED + (WATER_SURFACE,)
AUTUMN, WINTER, SPRING = 1, 2, 3

BEGIN, END = "/*c3x-season:", "/*c3x-season-end*/"
MARKED = re.compile(r"/\*c3x-season:[^*]*\*/.*?/\*c3x-season-end\*/", re.S)


@dataclass(frozen=True)
class Edit:
    tag: str
    files: tuple
    head: str   # text before the insertion point
    tail: str   # text after it; head + tail must occur exactly once
    code: str


def library(name: str) -> str:
    return (HERE / name).read_text()


def guard(season: int, body: str) -> str:
    return f"    [branch] if (C3X_SEASON_INDEX == {season}) {{\n{body.rstrip()}\n    }}\n"


def before(tag, files, anchor, code):
    return Edit(tag, tuple(files), "", anchor, code if code.endswith("\n") else code + "\n")


def after(tag, files, anchor, code):
    return Edit(tag, tuple(files), anchor, "", "\n" + code.rstrip("\n"))


def between(tag, files, head, tail, code):
    return Edit(tag, tuple(files), head, tail, code)


# ---------------------------------------------------------------- anchors
SHADE = "Output shade(P input) {"
RAW_FEATURE = "float4 q6_raw_feature(FeaturePixelInput input)\n{"
RELIEF_LIGHT = "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Q6ShadowL is also the projection direction"
OBJECT_LIGHT = "#ifdef BEAUTY_COMPOSED_SHADOWS\n    // Use the exact vector that projects the Q6 "
MOUNTAIN_ALBEDO_END = "#endif\n#ifdef SANDBOX_TERRAIN_MATERIAL\n#ifdef BEAUTY_VOLCANO_MATERIAL"
GRAIN = "        albedo *= 1 + clamp((grain - 0.303) * 0.9, -0.16, 0.22) * grass_plains_detail;"
DEBUG_ALBEDO = "#ifdef Q8_DEBUG_ALBEDO\n        return float4(albedo,1);"
WATER_DISPATCH = "        return q3_water_material(input);\n#endif"
ROAD_HEAD = "                lerp(0.90, 1.06, pattern_ground);\n"
ROAD_TAIL = "            return float4(frame_tone_map(track_albedo * track_light *"
FLOOR_HEAD = ("        float4 floor_sample = jungle_floor ? JungleFloorColor.Sample(Clamp, input.uv) :\n"
              "                                             ForestFloorColor.Sample(Clamp, input.uv);\n"
              "        float2 floor_normal = jungle_floor ? JungleFloorHeight.Sample(Clamp, input.uv).rg :\n"
              "                                             ForestFloorHeight.Sample(Clamp, input.uv).rg;\n")
FLOOR_TAIL = "        clip(floor_sample.a - 0.06);\n"
FARM_TINT = ("            albedo = lerp(albedo, farm_kit_ground_tint(albedo, farm_kit_terrain),\n"
             "                0.7 - 0.35 * saturate((farm_kit_terrain - 0.65) / 0.35));\n    }\n")
POLLUTION = "    albedo = lerp(albedo, float3(0.13, 0.115, 0.055), pollution_weight);\n"
POLLUTION_TAIL = "    float3 normal = normalize(input.geometry_normal);"

# The same per-tree seed and rounded crown normal serve Autumn and Spring. A
# jungle body (instance_secondary.z, set by the natural runtime from the jungle
# recipes) gets seed -1: Autumn and Spring leave jungle unchanged, like conifers.
CORE = [
    after("core/terrain-frame", [TERRAIN], "    float4 Detail;",
          "    float4 C3XFramePad5;\n    float4 C3XSeason; // EnvironmentState season, progress"),
    after("core/mountain-frame", [MOUNTAIN], "    float4 Macro;\n    float4 Quality;",
          "    float4 C3XSeason; // EnvironmentState season, progress"),
    after("core/objects-frame", [OBJECTS], "    float4 View;\n    float4 Quality;",
          "    float4 C3XFramePad5;\n    float4 C3XSeason; // EnvironmentState season, progress"),
    before("core/natural-index", NATURAL, SHADE, "#define C3X_SEASON_INDEX int(C3XSeason.x + 0.5)"),
    before("core/integrated-index", INTEGRATED, RAW_FEATURE,
           "#define C3X_SEASON_INDEX int(game_resource_look.z + 0.5)"),
    # The water dispatch is a brace-less `if`: the season recolours and the
    # original return become one block, or every pixel would take the water path.
    # The (large) water material is evaluated once and each season recolours it,
    # so the integrated shaders inline it once, as before.
    # User, 2026-10-22: the coast keeps Summer's translucent look in every season.
    # Seasonal water grades fade in from Coast to Sea (the Coast/Sea/Ocean family
    # in surface_coordinate); shallow beds (kind 4) never change; rivers keep theirs.
    before("core/water-open", INTEGRATED[:3], WATER_DISPATCH,
           "    {\n        float4 c3x_water = q3_water_material(input);\n"
           "        float c3x_open_sea = input.surface_kind > 8.5 && input.surface_kind < 9.5 ? 1 :\n"
           "            input.surface_kind < 4.5 ? 0 : smoothstep(.34, .63, saturate(input.surface_coordinate));"),
    after("core/water-close", INTEGRATED[:3], "        return q3_water_material(input);", "#endif\n    }"),
    between("core/tree-varyings", [OBJECTS], "    float2 secondary : TEXCOORD3;", "\n};\nstruct Output",
            "\n    nointerpolation float season_seed : TEXCOORD4;\n    float4 crown : TEXCOORD5;"),
    between("core/tree-varyings-default", [OBJECTS], "    output.secondary = input.secondary;", "\n    return output;",
            "\n    output.season_seed = 0;\n    output.crown = float4(input.normal, 1);"),
    between("core/tree-crown", [OBJECTS], " P o=VSMain(v);",
            "\n o.position.xy=(floor(v.position.xy*256+.5)/256+i.placement_view.xy)",
            "\n o.season_seed=instance_secondary.z>.5?-1:frac(sin(dot(i.place0.xy+i.place0.zw*7.3+i.place1.xy*3.1,float2(12.9898,78.233)))*43758.5453);\n"
            " {float3 q=i.source_position-float3(0,0,.15);float3 g=float3(q.x/.04,q.y/.04,q.z/.012);\n"
            "  o.crown=float4(normalize(float3(g.x*co-g.y*si,-(g.x*si+g.y*co),g.z/(150.f/(.82f*64.f)))),saturate(i.source_position.z/.24));}"),
    between("core/tree-flags", [OBJECTS], "    float3 normal = geometric;", "\n    if (kind < 0.5) {",
            "\n    bool autumn_leaf = false;\n    bool spring_leaf = false;"),
]

# ---------------------------------------------------------------- Autumn
AUTUMN_EDITS = [
    before("autumn/library", NATURAL, SHADE, library("autumn.hlsl")),
    before("autumn/library", INTEGRATED, RAW_FEATURE, library("autumn.hlsl")),
    # Trees: gold-led crowns, rounded crown light, backlit glow.
    before("autumn/crowns", [OBJECTS], "        if (has_normal)\n            normal = mapped_normal(geometric, input.world, input.uv,",
           guard(AUTUMN, """        if (foliage) {
            float3 average = BaseColor.SampleLevel(Clamp, float2(0.5, 0.5), 15).rgb;
            autumn_leaf = average.g > AUTUMN_BROADLEAF_GB * average.b && input.season_seed >= 0;
            float leafness = saturate((base.g - base.r) * 25 + 0.75);
            if (autumn_leaf)
                albedo = lerp(albedo, autumn_leaf_color(lerp(albedo, BaseColor.SampleBias(Wrap, input.uv, AUTUMN_LEAF_BIAS).rgb,
                    AUTUMN_LEAF_BIAS > 0 ? 0.7 : 0), average, input.season_seed), leafness) *
                    lerp(AUTUMN_CROWN_BASE, AUTUMN_CROWN_TOP, smoothstep(0.15, 0.85, input.crown.w));
        }""")),
    before("autumn/crown-normal", [OBJECTS], "    float ndl = saturate(dot(normal, light_direction));",
           guard(AUTUMN, "        if (autumn_leaf) normal = normalize(lerp(normal, normalize(input.crown.xyz), AUTUMN_CROWN));")),
    after("autumn/crown-wrap", [OBJECTS], "    float diffuse = ndl;",
          guard(AUTUMN, "        if (autumn_leaf) diffuse = saturate((dot(normal, light_direction) + AUTUMN_WRAP) / (1 + AUTUMN_WRAP));")),
    after("autumn/crown-fill", [OBJECTS], "    float foliage_fill = kind > 0.5 && kind < 1.5 ? 0.72 : 1.0;",
          guard(AUTUMN, "        if (autumn_leaf) foliage_fill = AUTUMN_FILL;")),
    before("autumn/leaf-glow", [OBJECTS], "    if (!(kind > 0.5 && kind < 1.5) && input.secondary.y > 0.5)\n        radiance",
           guard(AUTUMN, """        if (autumn_leaf)
            radiance += albedo * SunColorExposure.rgb * Sun.w * shadow * AUTUMN_TRANSLUCENCY *
                pow(saturate(dot(normalize(View.xyz), -light_direction) * 0.5 + 0.5), 2) * (1 - ndl);""")),
    # Relief: turf grade and dry patches; khaki hills.
    before("autumn/ground", [TERRAIN], RELIEF_LIGHT, guard(AUTUMN, """        if (!(input.material.y > 2.5 && input.material.y < 4.5)) {
            albedo = autumn_ground(albedo, input.world.xy);
            float autumn_hill = input.material.y < 1.5 ? input.material.z * smoothstep(0.02, 0.2, input.material.x) : 0;
            float3 autumn_hill_lch = autumn_lch(albedo) * float3(1, lerp(1, AUTUMN_HILL_TURF, autumn_hill), 1);
            float autumn_grey = autumn_hill * (1 - smoothstep(0.02, AUTUMN_HILL_FLOOR, autumn_hill_lch.y));
            autumn_hill_lch.z = lerp(autumn_hill_lch.z, 84, autumn_grey);
            autumn_hill_lch.y = lerp(autumn_hill_lch.y, max(autumn_hill_lch.y, AUTUMN_HILL_FLOOR), autumn_hill);
            albedo = autumn_from_lch(autumn_hill_lch);
        }""")),
    after("autumn/turf", [TERRAIN], GRAIN, guard(AUTUMN, """        float3 lum = float3(0.2126, 0.7152, 0.0722);
        float3 autumn_fine = GrassColor.Sample(Wrap, uv0).rgb;
        float3 autumn_low = (GrassColor.SampleBias(Wrap, uv0, 7.0).rgb +
            GrassColor.SampleBias(Wrap, uv0 + float2(.37, .11), 7.0).rgb +
            GrassColor.SampleBias(Wrap, uv0 + float2(.13, .53), 7.0).rgb +
            GrassColor.SampleBias(Wrap, uv0 + float2(.61, .71), 7.0).rgb) * .25;
        float grass_ratio = dot(autumn_fine, lum) / max(dot(autumn_low, lum), 0.01);
        float plains_ratio = dot(plains, lum) / max(dot(PlainsColor.SampleBias(Wrap, uv1, 5.0).rgb, lum), 0.01);
        float turf = lerp(grass_ratio, plains_ratio, plains_weight) - 1;
        albedo *= 1 + clamp(turf * AUTUMN_TURF, -0.35, 0.45) * (1 - tundra_weight) * (1 - desert_weight) * (1 - rocky_band);""")),
    after("autumn/hill-chroma", [TERRAIN], "        albedo = lerp(base, hill, rocky_band * 0.90);",
          guard(AUTUMN, "        albedo = lerp(albedo, autumn_from_lch(autumn_lch(albedo) * float3(1, AUTUMN_HILL_CHROMA, 1)), rocky_band);")),
    after("autumn/hill-stone", [TERRAIN],
          "        float3 hill_stone = lerp(hill_luma.xxx, hill, 0.16) * float3(0.91, 0.96, 1.10) * 1.20;",
          guard(AUTUMN, "        hill_stone = lerp(hill_luma.xxx, hill, 0.30) * float3(1.04, 0.99, 0.90) * 1.30;")),
    after("autumn/decal-stone", [TERRAIN],
          "        float3 stone = lerp(stone_source, stone_luma.xxx, 0.34) *\n                       float3(0.91, 0.97, 1.10) * 0.84;",
          guard(AUTUMN, "        stone = lerp(stone_source, stone_luma.xxx, 0.34) * float3(1.0, 0.97, 0.92) * 0.98;")),
    # Forest floor under broadleaf trees: fallen leaves; none on conifer floors (3.25).
    between("autumn/fallen-leaves", [TERRAIN], FLOOR_HEAD, FLOOR_TAIL,
            "        float autumn_leaf_weight = 0;\n        float3 autumn_leaf_color = 0;\n" + guard(AUTUMN, """        if (!jungle_floor && input.material.y < 3.125) {
            float2 cell = floor(saturate(input.uv) * 2.9999);
            float2 local = frac(saturate(input.uv) * 3);
            if (fmod(cell.x + cell.y, 2) > 0.5) local.x = 1 - local.x;
            float pick = fmod(cell.x + cell.y * 3, 4);
            float2 leaf_uv = (float2(fmod(pick, 2), floor(pick / 2)) + local * 0.98 + 0.01) * 0.5;
            float4 leaves = JungleFloorColor.Sample(Clamp, leaf_uv);
            float average_light = autumn_lch(JungleFloorColor.SampleLevel(Clamp, leaf_uv, 9).rgb).x;
            float r = length((AUTUMN_LEAF_DECAL_RADIUS > 0 ? saturate(input.uv) : local) - 0.5) * 2;
            float gaps = AUTUMN_LEAF_GAPS > 0 ? smoothstep(AUTUMN_LEAF_GAPS, AUTUMN_LEAF_GAPS + 0.35,
                autumn_noise(input.world.xy * 7.0 + 3.1)) : 1;
            float pile = saturate(leaves.a * AUTUMN_LEAF_COVER * 1.4) * gaps *
                lerp(1, 1 - smoothstep(0.30, 1.0, r), AUTUMN_LEAF_FALLOFF);
            float sprinkle = AUTUMN_LEAF_SCATTER > 0 ? autumn_sprinkle(input.world.xy, r) : 0;
            autumn_leaf_weight = max(pile, sprinkle);
            float3 average_leaf = JungleFloorColor.SampleLevel(Clamp, leaf_uv, 9).rgb;
            autumn_leaf_color = autumn_fallen(sprinkle > pile ? average_leaf : leaves.rgb, average_light, input.world.xy);
            floor_sample.rgb = autumn_litter(floor_sample.rgb, input.world.xy);
            floor_normal = lerp(floor_normal, JungleFloorHeight.Sample(Clamp, leaf_uv).rg, autumn_leaf_weight);
            floor_sample.a = max(floor_sample.a, autumn_leaf_weight);
        }""")),
    after("autumn/fallen-leaves-color", [TERRAIN], "        albedo = floor_sample.rgb * canopy_tint;",
          guard(AUTUMN, """        albedo = lerp(floor_sample.rgb * canopy_tint, autumn_leaf_color * AUTUMN_LEAF_SHADE,
                      autumn_leaf_weight / max(floor_sample.a, 0.001));""")),
    before("autumn/mountain-ground", [MOUNTAIN], MOUNTAIN_ALBEDO_END,
           guard(AUTUMN, "        albedo = autumn_ground(albedo, input.world.xy);")),
    # Coasts and underlay, farms, roads, rivers (Lab open water too).
    before("autumn/underlay", INTEGRATED[:3], DEBUG_ALBEDO,
           guard(AUTUMN, "        albedo = autumn_ground(albedo, world_position);")),
    after("autumn/farms", FIELD_SET, FARM_TINT, guard(AUTUMN, """        if (farm_kit_field_weight > 0.5)
            albedo = mine_slot < 0.5 ? AUTUMN_FIELD(albedo, input.q6_world.xy) : autumn_ground(albedo, input.q6_world.xy);""")),
    between("autumn/roads", [HYDROLOGY], ROAD_HEAD, ROAD_TAIL, guard(AUTUMN, "            track_albedo *= AUTUMN_ROAD_DARK;")),
    before("autumn/water", INTEGRATED[:3], WATER_DISPATCH, guard(AUTUMN, """        float4 autumn_surface = c3x_water;
        float autumn_strength = (input.surface_kind > 8.5 && input.surface_kind < 9.5 ? AUTUMN_RIVER_WATER : 1) *
            lerp(AUTUMN_WATER_NIGHT, 1, saturate(environment_sun_intensity * 2.5)) * c3x_open_sea;
        c3x_water = float4(autumn_water(autumn_surface.rgb, autumn_strength), autumn_surface.a);""")),
]

# ---------------------------------------------------------------- Winter
# Winter's mountain flurry (seasons D4; by the Winter session): effect kind 6,
# whose quads the renderer builds per view only in Winter (prepare_flurry).
# v2 (user, 2026-10-22: "needs far more randomness ... the same rectangle
# shape of interweaving parallel snowflakes"): every column falls at its own
# speed and offset; every flake has its own spot, size, brightness and wide,
# slow sway; each parallax layer slants at its own angle around the mountain's
# wind; the flakes fill a ragged plume (width, height, density per mountain,
# noisy edges, an uneven foot) that drifts with slow gusts. Integer hash and a
# bounded clock keep it exact late in a game. Layers thin by zoom: three from
# a 128-pixel tile, two from 96, one below (the renderer draws none below 64).
# Lab: studies/winter/winter.py FLURRY2 (look v24-flurry4).
WINTER_FLURRY = """// Seasons D4: a gentle flurry of small snowflakes over a mountain tile (96).
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

WINTER_EDITS = [
    before("winter/library", NATURAL, SHADE, library("winter_natural.hlsl")),
    before("winter/library", INTEGRATED, RAW_FEATURE, library("winter_integrated.hlsl")),
    before("winter/ground", [TERRAIN], RELIEF_LIGHT, "    float winter_snow = 0;\n" + guard(WINTER, """        float kind = input.material.y;
        float3 mesh_normal = normalize(input.normal);
        float slope = 1 - saturate(mesh_normal.z);
        float depth = kind > 2.5 && kind < 4.5 ? WINTER_FLOOR_DEPTH : (kind > 1.5 && kind < 2.5 ? WINTER_STONE_DEPTH : 1.0);
        float4 types = winter_types(kind < 1.5 ? saturate(input.biome.x) : (kind > 5.5 && kind < 6.5 ? 1 : 0),
                                    kind < 1.5 ? saturate(input.biome.y) : (kind > 6.5 && kind < 7.5 ? 1 : 0),
                                    kind < 1.5 ? saturate(input.material.w) : 0);
        depth *= dot(types, WINTER_TYPE_DEPTH);
        float rock = kind < 1.5 ? input.material.z * smoothstep(0.09, 0.43, input.material.x) : (kind > 1.5 && kind < 2.5 ? 1 : 0);
        winter_snow = winter_cover(depth, height_detail, slope, input.world.xy) *
            (1 - winter_type_shown(types, input.world.xy) * (1 - saturate(rock)));
        float crust = (SurfaceDetail.Sample(Wrap, input.world.xy * WINTER_CRUST_SCALE + float2(0.41, 0.73)).r - 0.303) *
            (1 + WINTER_TUNDRA_CRUST * types.w);
        float3 snow = winter_snow_albedo(height_detail, input.world.xy) * (1 + crust * WINTER_CRUST_LIGHT) *
            (types.x + types.y * WINTER_PLAINS_SNOW + types.z * WINTER_DESERT_SNOW + types.w * WINTER_TUNDRA_SNOW);
        float dune = (winter_ripple(input.world.xy) - 0.5) * WINTER_DESERT_RELIEF * types.z;
        float3 exposed = lerp(winter_type_exposed(albedo, types), winter_dormant(albedo), saturate(rock));
        albedo = lerp(exposed, snow, winter_snow);
        float3 drift = detail_normal_strength(mesh_normal, input.world,
            winter_fbm(input.world.xy * WINTER_DRIFT_SCALE) + crust * WINTER_CRUST + dune, WINTER_DRIFT_RELIEF);
        geometric = normalize(lerp(geometric, drift, winter_snow * WINTER_SMOOTH));
        specular_map = lerp(specular_map, WINTER_SNOW_GLOSS + WINTER_TUNDRA_GLOSS * types.w, winter_snow);""")),
    after("winter/ground-sky", [TERRAIN],
          "    float3 ambient = Ambient.rgb * Ambient.a * lerp(0.56, 1.0, sky) * cavity * surface_occlusion;",
          guard(WINTER, "        ambient *= lerp(1, WINTER_SKY, winter_snow);")),
    after("winter/mountain-cap", [MOUNTAIN],
          "        float snow_cap = saturate((input.material.y - 2) * 2);\n"
          "        float snow = smoothstep(0.62, 0.78, height) * smoothstep(MTN_SNOW_SLOPE_LOW, MTN_SNOW_SLOPE_HIGH, geometric.z) * snow_cap;",
          guard(WINTER, """        snow_cap = 1;
        snow = smoothstep(WINTER_MTN_SNOW_LOW, WINTER_MTN_SNOW_HIGH, height) *
            smoothstep(WINTER_MTN_UP_LOW, WINTER_MTN_UP_HIGH, geometric.z) * snow_cap;""")),
    before("winter/mountain", [MOUNTAIN], MOUNTAIN_ALBEDO_END, "    float winter_snow = 0;\n" + guard(WINTER, """        float ground_cover = winter_cover(1, height_detail, 1 - saturate(normalize(input.normal).z), input.world.xy);
        float rock_cover = smoothstep(WINTER_MTN_UP_LOW, WINTER_MTN_UP_HIGH, normal.z) * 0.85;
        winter_snow = lerp(ground_cover, rock_cover, rock_albedo_coverage);
        float3 rock_lch = winter_lch(albedo) * float3(WINTER_ROCK_LIGHT, WINTER_ROCK_CHROMA, 1);
        float3 bare = lerp(winter_dormant(albedo), winter_from_lch(rock_lch), rock_albedo_coverage);
        albedo = lerp(bare, winter_snow_albedo(height_detail, input.world.xy), winter_snow);
        specular_map = lerp(specular_map, WINTER_SNOW_GLOSS, winter_snow);""")),
    after("winter/mountain-sky", [MOUNTAIN],
          "    float3 ambient = Ambient.rgb * Ambient.a * lerp(ambient_floor, 1.0, sky) * cavity * crevice_visibility;",
          guard(WINTER, "        ambient *= lerp(1, WINTER_SKY, winter_snow);")),
    before("winter/trees", [OBJECTS], OBJECT_LIGHT, "    float winter_snow = 0;\n" + guard(WINTER, """        if (kind > 0.5 && kind < 1.5) {
            float3 lch = winter_lch(albedo);
            float3 average = BaseColor.SampleLevel(Clamp, float2(0.5, 0.5), 15).rgb;
            float average_light = winter_lch(average).x;
            float gb = average.g / max(average.b, 0.001);
            float rel = lch.x / max(average_light, 0.05);
            float lift = (lch.x - average_light) * WINTER_TREE_LIFT;
            if (gb < WINTER_SNOWY_GB) {
                winter_snow = saturate((rel - 0.9) * 2);
            } else if (gb > WINTER_BROADLEAF_GB && WINTER_BARE > 0) {
                clip(winter_noise(input.uv * WINTER_BARE_SCALE) - WINTER_BARE);
                float3 twig = winter_from_lch(float3(WINTER_TWIG_LIGHT * clamp(rel, 0.55, 1.45), WINTER_TWIG_CHROMA, WINTER_TWIG_HUE));
                winter_snow = smoothstep(WINTER_TREE_UP_LOW, WINTER_TREE_UP_HIGH, normal.z + lift) * WINTER_BARE_SNOW;
                albedo = lerp(twig, WINTER_TREE_SNOW, winter_snow);
            } else {
                winter_snow = smoothstep(WINTER_TREE_UP_LOW, WINTER_TREE_UP_HIGH, normal.z + lift);
                float3 needle = winter_from_lch(float3(lch.x * WINTER_NEEDLE_LIGHT, lch.y * WINTER_NEEDLE_CHROMA,
                    lerp(lch.z, WINTER_NEEDLE_HUE, WINTER_NEEDLE_PULL)));
                albedo = lerp(needle, WINTER_TREE_SNOW * (0.9 + 0.2 * saturate(rel - 0.5)), winter_snow);
            }
        } else if (kind < 0.5) {
            winter_snow = winter_cover(1, 0.5, 0, input.world.xy);
            albedo = lerp(winter_dormant(albedo), winter_snow_albedo(0.5, input.world.xy), winter_snow);
        }""")),
    after("winter/tree-sky", [OBJECTS],
          "    float3 ambient = Ambient.rgb * Ambient.a * lerp(0.48, 1.0, sky) * ao * foliage_fill;",
          guard(WINTER, "        ambient *= lerp(1, WINTER_SKY, winter_snow) * WINTER_TREE_FILL;")),
    before("winter/tree-hue-lock", [OBJECTS], "    output.color = float4(max(radiance, 0), 1);", guard(WINTER, """        float3 winter_w = float3(0.2126, 0.7152, 0.0722);
        float3 tint = lerp(WINTER_TREE_SHADE_TINT, WINTER_TREE_LIT_TINT, saturate(diffuse * shadow));
        radiance = lerp(radiance, dot(radiance, winter_w) * tint / dot(tint, winter_w), winter_snow * WINTER_TREE_LOCK);""")),
    before("winter/river", [HYDROLOGY], "  return float4(lerp(shore*bank_light,surface,water),bank*outlet);", guard(WINTER, """    float3 river_lch = winter_lch(surface);
    float river_hue = river_lch.z < 0 ? river_lch.z + 360 : river_lch.z;
    float river_day = lerp(1, lerp(WINTER_WATER_NIGHT, 1, saturate(environment_sun_intensity * 2.5)), WINTER_RIVER_DAYLIGHT);
    river_lch.z = lerp(river_hue, lerp(river_hue, WINTER_WATER_HUE, WINTER_WATER_PULL), river_day);
    river_lch.y = lerp(river_lch.y, max(river_lch.y * WINTER_RIVER_CHROMA, WINTER_RIVER_MIN_CHROMA), river_day);
    river_lch.x *= lerp(1, WINTER_RIVER_LIGHT, river_day);
    float3 river_open = winter_from_lch(river_lch);
    float3 flat_light = q6_receiver_illumination(input, float3(0, 0, 1), 1, 1);
    float3 snow_lit = winter_snow_tex(world) * WINTER_RIVER_SNOW_GAIN * flat_light;
    float reach_ice = smoothstep(WINTER_SHELF_OPEN, WINTER_SHELF_OPEN + 0.25, winter_noise(world * WINTER_SHELF_REACH + 5.7));
    float scallop = winter_noise(world * WINTER_SHELF_SCALLOP) * 0.7 + winter_noise(world * WINTER_SHELF_SCALLOP * 2.9 + 1.3) * 0.3 - 0.5;
    float shelf_from = water_width + 0.5 - reach_ice * (WINTER_SHELF_WIDTH * (0.45 + 0.55 * winter_noise(world * WINTER_SHELF_REACH * 2.3 + 2.1))
        - scallop * WINTER_SHELF_BREAK);
    float shelf = smoothstep(shelf_from - 0.2, shelf_from + 0.2, shoreline);
    float floe = smoothstep(WINTER_RIVER_FLOE, WINTER_RIVER_FLOE + 0.04, winter_noise(world * WINTER_RIVER_FLOE_SCALE + 3.3)) *
        (1 - smoothstep(1.5, 3.0, distance_pixels));
    float river_ice = saturate(max(shelf, floe * 0.85));
    float glass = 1 - smoothstep(shelf_from + 0.15, shelf_from + WINTER_SHELF_GLASS, shoreline);
    float window = smoothstep(0.31, 0.24, winter_fbm(world * WINTER_SHELF_WINDOW + 9.1));
    float3 bare_lit = lerp(river_open, WINTER_RIVER_ICE_THIN * flat_light, 0.6);
    float3 ice_lit = lerp(snow_lit, bare_lit, saturate(max(glass * shelf, window * 0.75)) * (1 - floe));
    surface = lerp(river_open, ice_lit + specular * 0.3, river_ice * WINTER_RIVER_ICE);
    float open_bank = 1 - reach_ice;
    float rim = open_bank * (1 - smoothstep(water_width + 0.1, water_width + 0.8 + sediment_noise * 0.9, shoreline)) *
        smoothstep(0.35, 0.55, sediment_noise * 0.6 + (shore_noise + 0.5) * 0.4) * WINTER_RIVER_RIM;
    shore = winter_snow_tex(world) * WINTER_RIVER_SNOW_GAIN * lerp(float3(1, 1, 1), WINTER_RIVER_RIM_TONE, rim);""")),
    before("winter/roof-coat", FIELD_SET, " float3 lit=base*(1-metalness)*q6_receiver_illumination(p,n,1,ao);",
           " float winter_roof=0;float3 winter_roof_snow=0;float3 winter_roof_n=n;float winter_roof_ao=lerp(1,ao,WINTER_ROOF_AO);\n"
           + guard(WINTER, """  float3 winter_up=normalize(p.geometry_normal);
  float2 rw=p.q6_world.xy;
  float slope=smoothstep(WINTER_ROOF_LOW,WINTER_ROOF_HIGH,winter_up.z);
  float dab=winter_noise(rw*WINTER_ROOF_DAB+11.3),edge=WINTER_ROOF_BARE+(1-slope)*.3;
  winter_roof=slope*smoothstep(edge-.03,edge+.03,dab);
  float thin=1-smoothstep(edge,edge+.14,dab);
  float mottle=winter_fbm(rw*WINTER_ROOF_MOTTLE)-.5;
  winter_roof_snow=WINTER_SNOW_HIGH*(1+mottle*WINTER_ROOF_DETAIL)*lerp(float3(1,1,1),WINTER_ROOF_CREASE,thin*.5)
   *lerp(WINTER_ROOF_CREASE,float3(1,1,1),saturate(ao*1.2));
  winter_roof_n=normalize(lerp(n,winter_up,winter_roof*WINTER_ROOF_FLATTEN));
  base=lerp(base,winter_roof_snow,winter_roof);
  n=winter_roof_n;
  metalness*=1-winter_roof;""")),
    after("winter/roof-light", FIELD_SET,
          " lit=q8_city_look(lit,Q8_CITY_LOOK*(1-saturate(environment_night_activation)),base,Q8_CITY_LOOK_PALE);",
          guard(WINTER, " lit=lerp(lit,winter_roof_snow*q6_receiver_illumination(p,winter_roof_n,1,winter_roof_ao)*WINTER_ROOF_GAIN,winter_roof);")),
    before("winter/plates", FIELD_SET, "  float3 ground_lit=ground.rgb*q6_receiver_illumination(p,ground_normal,1,1);",
           guard(WINTER, """  ground.rgb=lerp(winter_dormant(ground.rgb),winter_snow_tex(p.q6_world.xy)*WINTER_PLATE_SNOW,
   saturate(WINTER_PLATE_COVER*(0.65+0.7*winter_noise(p.q6_world.xy*17.0))));""")),
    before("winter/coastal-rocks", INTEGRATED, "    if(abs(input.material_index-.48)<.001)return q4_coastal_rock(input);",
           guard(WINTER, """    if(abs(input.material_index-.48)<.001) {
        float4 rock = q4_coastal_rock(input);
        float3 rock_up = normalize(input.geometry_normal);
        float rock_snow = saturate(smoothstep(WINTER_SHORE_ROCK_LOW, WINTER_SHORE_ROCK_HIGH, rock_up.z) *
            (0.7 + 0.6 * winter_noise(input.q6_world.xy * 23.0)) * WINTER_SHORE_ROCK_COVER);
        rock.rgb = lerp(rock.rgb, winter_luma(rock.rgb).xxx, WINTER_ROCK_GREY);
        rock.rgb = lerp(rock.rgb, winter_snow_tex(input.q6_world.xy) * q6_receiver_illumination(input, rock_up, 1, 1) *
            frame_output_exposure(), rock_snow * rock.a);
        return rock;
    }""")),
    before("winter/underlay", INTEGRATED, DEBUG_ALBEDO, guard(WINTER, """            float winter_h = winter_fbm(world_position * 3.1);
            float winter_c = winter_cover(1, winter_h * 0.9, 1 - saturate(geometry_normal.z), world_position);
            albedo = lerp(winter_dormant(albedo), winter_snow_tex(world_position), winter_c);
            material_normal = normalize(lerp(material_normal, geometry_normal, winter_c * WINTER_SMOOTH));""")),
    between("winter/objects", FIELD_SET, POLLUTION, POLLUTION_TAIL, guard(WINTER, """        float2 winter_world = input.q6_world.xy;
        float3 winter_up = normalize(input.geometry_normal);
        float3 winter_white = winter_snow_tex(winter_world);
        if (farm_kit_field_weight > 0.5)
            albedo = mine_slot > 2.5 && mine_slot < 3.5 ?
                winter_field_kit(albedo, winter_world, winter_white, true) :
                winter_field_plot(albedo, resource_base_texture_0.SampleBias(material_sampler, input.uv, WINTER_FIELD_BIAS).rgb,
                                  winter_world, winter_white);
        else if (unit_weight < 0.5) {
            float flat_decal = saturate(resource_decal_weight + ground_state_weight);
            float cover = lerp(smoothstep(WINTER_OBJ_UP_LOW, WINTER_OBJ_UP_HIGH, winter_up.z) * WINTER_OBJ_COVER,
                               WINTER_DECAL_COVER, flat_decal);
            cover *= 0.7 + 0.6 * winter_noise(winter_world * 23.0);
            albedo = lerp(lerp(albedo, winter_dormant(albedo), WINTER_OBJ_DORMANT), winter_white, saturate(cover));
        }""")),
    between("winter/roads", [HYDROLOGY], ROAD_HEAD, ROAD_TAIL, guard(WINTER, """            track_albedo = lerp(winter_dormant(track_albedo) * WINTER_ROAD_DARK,
                winter_snow_tex(input.macro_uv * 2.0) * 0.85, WINTER_ROAD_SNOW * (1 - coverage));""")),
    after("winter/rails", [HYDROLOGY], "            float3 dirt = float3(0.32, 0.20, 0.11) * lerp(0.90, 1.06, pattern_ground);",
          guard(WINTER, """            dirt = lerp(float3(0.32, 0.20, 0.11), winter_snow_tex(input.macro_uv * 2.0) * 0.9, WINTER_RAIL_SNOW) *
                lerp(0.90, 1.06, pattern_ground);""")),
    before("winter/library", FIELD_SET, "float4 q8_city_effect(FeaturePixelInput p){", WINTER_FLURRY),
    # Kind 6 draws only in Winter; any other season clips it (a safety net).
    before("winter/flurry-dispatch", FIELD_SET, " if(kind>2.5){\n  // A volcano's plume (93)",
           guard(WINTER, "        if(kind>5.5 && kind<6.5)return q8_effect_flurry(q,pixel,t,seed,strength,night);")
           + "    if(kind>5.5 && kind<6.5){clip(-1);return 0;}"),
    before("winter/water", INTEGRATED[:3], WATER_DISPATCH, guard(WINTER, """        float4 winter_surface = c3x_water;
        float winter_river = input.surface_kind > 8.5 && input.surface_kind < 9.5 ? 1 : 0;
        float3 lch = winter_lch(winter_surface.rgb);
        float hue = lch.z < 0 ? lch.z + 360 : lch.z;
        float gate = smoothstep(150, 185, hue) * (1 - smoothstep(280, 300, hue)) *
            lerp(WINTER_WATER_NIGHT, 1, saturate(environment_sun_intensity * 2.5)) * (1 - winter_river) * c3x_open_sea;
        lch.z = hue + (WINTER_WATER_HUE - WINTER_WATER_REF_HUE) * gate;
        lch.y *= lerp(1, WINTER_WATER_CHROMA, gate);
        lch.x = lerp(lch.x, WINTER_WATER_PIVOT * WINTER_WATER_LIGHT + (lch.x - WINTER_WATER_PIVOT) * WINTER_WATER_CONTRAST, gate);
        float winter_coast = 1 - step(0.25, abs(input.real_terrain - 11));
        float winter_ocean = 1 - step(0.25, abs(input.real_terrain - 13));
        lch.x += (winter_coast * WINTER_COAST_LIFT - winter_ocean * WINTER_OCEAN_DROP) * gate;
        lch.z += (winter_coast * WINTER_COAST_HUE + winter_ocean * WINTER_OCEAN_HUE) * gate;
        float winter_offshore = -0.85 * log(max(1 - saturate(max(0, input.hydrology_data.w) / 0.46), 1e-4));
        float winter_deep = smoothstep(WINTER_DEEP_NEAR, WINTER_DEEP_FAR, winter_offshore) * (1 - winter_river);
        lch.x -= winter_deep * WINTER_DEEP_DROP * gate;
        lch.z += winter_deep * WINTER_DEEP_HUE * gate;
        c3x_water = float4(winter_from_lch(lch), winter_surface.a);""")),
]

# ---------------------------------------------------------------- Spring
SPRING_EDITS = [
    before("spring/library", NATURAL, SHADE, library("spring.hlsl")),
    before("spring/library", INTEGRATED[:3], RAW_FEATURE, library("spring.hlsl")),
    after("spring/locals", [TERRAIN], "    float surface_occlusion = 1;",
          "    float4 spring_bloom = 0;\n    float spring_leaf = 0;"),
    before("spring/ground", [TERRAIN], RELIEF_LIGHT, guard(SPRING, """        if (!(input.material.y > 2.5 && input.material.y < 4.5)) {
            albedo = spring_ground(albedo, input.world.xy);
            float spring_hill = input.material.y < 1.5 ? input.material.z * smoothstep(0.02, 0.2, input.material.x) : 0;
            float3 spring_hill_lch = spring_lch(albedo);
            float spring_turf = spring_hill * smoothstep(0.02, 0.05, spring_hill_lch.y);
            spring_hill_lch *= float3(lerp(1, SPRING_HILL_LIGHT, spring_turf), lerp(1, SPRING_HILL_CHROMA, spring_turf), 1);
            spring_hill_lch.z = lerp(spring_hill_lch.z, SPRING_HILL_HUE, spring_turf * SPRING_HILL_PULL);
            albedo = spring_from_lch(spring_hill_lch);
            // No leaf shade under the drifts (user, 2026-10-23: flower ground
            // looked gritty); the turf there matches the turf without flowers.
            albedo = albedo * (1 - spring_bloom.a) + spring_bloom.rgb;
            geometric = normalize(lerp(geometric, normalize(lerp(normalize(input.normal), float3(0, 0, 1), SPRING_BLOOM_UP)),
                                       spring_bloom.a * SPRING_BLOOM_NORMAL));
            specular_map = lerp(specular_map, 0.02, spring_bloom.a * SPRING_BLOOM_NORMAL);
        }""")),
    # No painterly turf: Spring keeps Summer's grass texture and only regrades
    # its colour (user, 2026-10-23: the boosted fine detail looked gritty).
    between("spring/meadow", [TERRAIN], GRAIN, "\n    }", guard(SPRING, """        float spring_density = lerp(lerp(1, SPRING_PLAINS_DENSITY, plains_weight), SPRING_TUNDRA_DENSITY, tundra_weight) *
            (1 - desert_weight) * (1 - rocky_band * lerp(1 - SPRING_HILL_DENSITY, 1, hill_rock));
        spring_straw = plains_weight * (1 - desert_weight);
        spring_bloom = spring_meadow(input.world.xy, input.world.z, spring_density, ShadowL.xy, spring_leaf);""")),
    before("spring/mountain-ground", [MOUNTAIN], MOUNTAIN_ALBEDO_END,
           guard(SPRING, "        albedo = spring_ground(albedo, input.world.xy);")),
    before("spring/crowns", [OBJECTS], "        if (has_normal)\n            normal = mapped_normal(geometric, input.world, input.uv,",
           guard(SPRING, """        if (foliage) {
            float3 average = BaseColor.SampleLevel(Clamp, float2(0.5, 0.5), 15).rgb;
            spring_leaf = average.g > SPRING_BROADLEAF_GB * average.b && input.season_seed >= 0;
            float leafness = saturate((base.g - base.r) * 25 + 0.75);
            if (spring_leaf) {
                float3 soft = lerp(albedo, BaseColor.SampleBias(Wrap, input.uv, SPRING_LEAF_BIAS).rgb, SPRING_LEAF_BIAS > 0 ? 0.7 : 0);
                float3 fresh = spring_leaf_color(soft, average, input.season_seed);
                float3 fresh_lch = spring_lch(fresh);
                fresh_lch.z -= SPRING_SUN_HUE * (smoothstep(0.15, 0.85, input.crown.w) - 0.4);
                fresh = spring_from_lch(fresh_lch);
                if (input.season_seed > 1 - SPRING_BLOSSOM_SHARE) {
                    float3 src = spring_lch(albedo), avg = spring_lch(average);
                    float pink = frac(input.season_seed * 31.7);
                    float3 petal = pink < 0.45 ? float3(0.93, 0.025, 20) : (pink < 0.85 ? float3(0.86, 0.075, 355) : float3(0.78, 0.115, 350));
                    float flower = smoothstep(-0.06, 0.05, src.x - avg.x + (frac(input.season_seed * 57.1) - 0.5) * 0.04);
                    fresh = lerp(fresh, spring_from_lch(float3(petal.x + (src.x - avg.x) * 0.5, petal.yz)), flower * SPRING_BLOSSOM_COVER);
                }
                albedo = lerp(albedo, fresh, leafness) *
                    lerp(SPRING_CROWN_BASE, SPRING_CROWN_TOP, smoothstep(0.15, 0.85, input.crown.w));
            }
        }""")),
    before("spring/crown-normal", [OBJECTS], "    float ndl = saturate(dot(normal, light_direction));",
           guard(SPRING, "        if (spring_leaf) normal = normalize(lerp(normal, normalize(input.crown.xyz), SPRING_CROWN));")),
    after("spring/crown-wrap", [OBJECTS], "    float diffuse = ndl;",
          guard(SPRING, "        if (spring_leaf) diffuse = saturate((dot(normal, light_direction) + SPRING_WRAP) / (1 + SPRING_WRAP));")),
    after("spring/crown-fill", [OBJECTS], "    float foliage_fill = kind > 0.5 && kind < 1.5 ? 0.72 : 1.0;",
          guard(SPRING, "        if (spring_leaf) foliage_fill = SPRING_FILL;")),
    before("spring/leaf-glow", [OBJECTS], "    if (!(kind > 0.5 && kind < 1.5) && input.secondary.y > 0.5)\n        radiance",
           guard(SPRING, """        if (spring_leaf)
            radiance += albedo * SunColorExposure.rgb * Sun.w * shadow * SPRING_TRANSLUCENCY *
                pow(saturate(dot(normalize(View.xyz), -light_direction) * 0.5 + 0.5), 2) * (1 - ndl);""")),
    before("spring/underlay", INTEGRATED[:3], DEBUG_ALBEDO,
           guard(SPRING, "        albedo = spring_ground(albedo, world_position);")),
    before("spring/water", INTEGRATED[:3], WATER_DISPATCH, guard(SPRING, """        float4 spring_surface = c3x_water;
        float spring_strength = (input.surface_kind > 8.5 && input.surface_kind < 9.5 ? SPRING_RIVER_WATER : 1) *
            lerp(SPRING_WATER_NIGHT, 1, saturate(environment_sun_intensity * 2.5)) * c3x_open_sea;
        c3x_water = float4(spring_water(spring_surface.rgb, spring_strength), spring_surface.a);""")),
]

# ---------------------------------------------------------------- game open water
# Relative recolour on top of the Sea/Ocean family tint: one hue offset and
# chroma/lightness scale, faded at night, never a pull to a single hue. Coast
# water keeps Summer's translucent look (user, 2026-10-22); the grade fades in
# from Coast to Sea. Low-chroma foam and glints sit outside the gate.
def _open_water(season, prefix, shift, chroma, light, pivot=None, contrast=None):
    lightness = (f"lch.x = lerp(lch.x, {pivot} * {light} + (lch.x - {pivot}) * {contrast}, gate);"
                 if pivot is not None else f"lch.x *= lerp(1, {light}, gate);")
    return guard(season, f"""        float3 lch = {prefix}_lch(color);
        float hue = lch.z < 0 ? lch.z + 360 : lch.z;
        float gate = smoothstep(150, 185, hue) * (1 - smoothstep(280, 300, hue)) * smoothstep(0.008, 0.02, lch.y) *
            lerp({prefix.upper()}_WATER_NIGHT, 1, saturate(environment_sun_intensity * 2.5)) *
            smoothstep(.34, .63, saturate(input.surface_coordinate));
        lch.z = hue + {shift} * gate;
        lch.y *= lerp(1, {chroma}, gate);
        {lightness}
        color = {prefix}_from_lch(lch);""")


WATER_EDITS = [
    before("water/open", [WATER_SURFACE], "    return q6_scene_output(float4(color, alpha)).color;",
           _open_water(AUTUMN, "autumn", 4, "AUTUMN_WATER_CHROMA", "AUTUMN_WATER_LIGHT")
           + _open_water(WINTER, "winter", "(WINTER_WATER_HUE - WINTER_WATER_REF_HUE)", "WINTER_WATER_CHROMA",
                         "WINTER_WATER_LIGHT", "WINTER_WATER_PIVOT", "WINTER_WATER_CONTRAST")
           + _open_water(SPRING, "spring", 6, "SPRING_WATER_CHROMA", "0.9")),
]

# Last at the dispatch: every season has recoloured the one water evaluation
# (Summer leaves it unchanged). The original return is preprocessed out: fxc
# would otherwise inline a second, dead copy of the water material (+50% compile).
DISPATCH_END = [before("core/water-return", INTEGRATED[:3], WATER_DISPATCH, "        return c3x_water;\n#if 0\n")]

EDITS = CORE + AUTUMN_EDITS + WINTER_EDITS + SPRING_EDITS + WATER_EDITS + DISPATCH_END

# Text that Renderer64 rewrites at run time; season insertions must leave each intact.
RUNTIME_ANCHORS = {
    TERRAIN: ["        float3 grass_fine = GrassColor.Sample(Wrap, uv0).rgb;", "        float hill_ratio =",
              "albedo = lerp(decal.rgb, stone, rock * 0.94);", "stone_height * 0.58 + rock * 0.42, 0.11);",
              "        alpha *= decal.a;", "albedo = lerp(base, hill, rocky_band * 0.90);",
              "#ifdef SANDBOX_TERRAIN_MATERIAL\n    // The sandbox retains", "    float cavity = lerp(0.79, 1.0,",
              "        float3 weights=pow(abs(face),4);", "        geometric=normalize(lerp(geometric,face,exposure));"],
    OBJECTS: ["    radiance += Ambient.rgb * rim * 0.08;"],
    WATER_SURFACE: ["    float shadow = 1;", "    // 0 A.D. separates", "float4 ShadeWaterSurface("],
}


def strip(text: str) -> str:
    return MARKED.sub("", text)


def apply(relative: str, text: str) -> str:
    """Return `text` (a target shader) with every season insertion, from a clean base."""
    base = strip(text)
    points = {}
    for edit in EDITS:
        if relative not in edit.files:
            continue
        key = (edit.head, edit.tail)
        if key not in points:
            count = base.count(edit.head + edit.tail)
            if count != 1:
                raise ValueError(f"{edit.tag}: anchor occurs {count} times in {relative}: {(edit.head + edit.tail)[:70]!r}")
            points[key] = [base.index(edit.head + edit.tail) + len(edit.head), []]
        if BEGIN in edit.code or END in edit.code:
            raise ValueError(f"{edit.tag}: code contains a season marker")
        points[key][1].append(f"{BEGIN}{edit.tag}*/{edit.code}{END}")
    result = base
    for offset, chunks in sorted(points.values(), key=lambda point: point[0], reverse=True):
        result = result[:offset] + "".join(chunks) + result[offset:]
    for anchor in RUNTIME_ANCHORS.get(relative, []):
        if base.count(anchor) != result.count(anchor):
            raise ValueError(f"season insertion breaks a runtime anchor in {relative}: {anchor[:60]!r}")
    return result


def apply_tree(root: Path) -> dict:
    """Apply seasons to every target shader under `root`; returns {relative: new text} for changed files."""
    changed = {}
    for relative in TARGETS:
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text()
        new = apply(relative, text)
        if new != text:
            changed[relative] = new
    return changed
