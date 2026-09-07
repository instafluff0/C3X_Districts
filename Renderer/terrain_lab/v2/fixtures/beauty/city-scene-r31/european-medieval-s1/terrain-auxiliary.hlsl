#define Q3_CONTINUOUS_RIVERS 1
#define Q6_TEXEL_RECEIVER_OFFSET 1
#define Q4_RELIEF_MATERIAL_DATA 1
#define Q4_BROAD_RELIEF 1
#define Q4_VOLCANO_FOOTPRINT 0.387500000
#define Q3_SOURCE_WATER_NORMALS 1
#define Q4_COASTAL_ROCKS 1
#define Q3_COAST_DETAIL 1
#define Q6_GAMEPLAY_NIGHT 1
#define Q4_BIQ_CONTINUOUS_DESERT 1
#define Q4_BIQ_DUNE_COVERAGE 1
#define Q4_COMBINED_ROCK_PROJECTION 1
#define Q3_STATIC_OPTICS_V2 1
#define Q6_WORLD_SHADOWS 1
#define Q3_MATERIAL_ORIGIN_X 56.5
#define Q3_MATERIAL_ORIGIN_Y 18.5
#define Q3_MATERIAL_WRAP_WIDTH 100
#define Q2_MATERIAL_RESPONSE 1
#define Q3_WATER_MATERIAL 1
#define Q3_SHORE_MATERIAL 1
#define Q3_HYDROLOGY_DATA 1
Texture2D base_color_texture : register(t0);
Texture2D height_texture : register(t1);
Texture2D specular_texture : register(t2);
Texture2D authored_height_texture : register(t3);
Texture2D authored_blend_texture : register(t4);
Texture2D authored_region_texture : register(t5);
Texture2D mountain_base_texture : register(t6);
Texture2D mountain_top_texture : register(t7);
Texture2D mountain_snow_texture : register(t8);
Texture2D mountain_height_texture : register(t9);
Texture2D mountain_specular_texture : register(t10);
Texture2D beach_base_texture : register(t11);
Texture2D beach_height_texture : register(t12);
Texture2D beach_specular_texture : register(t13);
Texture2D cliff_base_texture : register(t14);
Texture2D cliff_height_texture : register(t15);
Texture2D cliff_specular_texture : register(t16);
Texture2D shallow_bed_texture : register(t17);
Texture2D ocean_bed_texture : register(t18);
Texture2D water_height_texture : register(t19);
Texture2D water_large_lean0_texture : register(t20);
Texture2D water_large_lean1_texture : register(t21);
Texture2D water_small_lean0_texture : register(t22);
Texture2D water_small_lean1_texture : register(t23);
Texture2D water_foam_texture : register(t24);
Texture2D feature_base_texture_0 : register(t25);
Texture2D feature_base_texture_1 : register(t26);
Texture2D feature_base_texture_2 : register(t27);
Texture2D feature_base_texture_3 : register(t28);
Texture2D shallows_specular_texture : register(t29);
Texture2D ocean_height_texture : register(t30);
Texture2D ocean_specular_texture : register(t31);
Texture2D water_gloss_texture : register(t32);
Texture2D water_tiling_mask_texture : register(t33);
Texture2D water_non_tiling_mask_texture : register(t34);
Texture2D water_small_secondary_lean0_texture : register(t35);
Texture2D water_small_secondary_lean1_texture : register(t36);
Texture2D water_ripples_texture : register(t37);
Texture2D water_turbulence_texture : register(t38);
Texture2D coast_dark_profile_texture : register(t39);
Texture2D coast_scatter_profile_texture : register(t40);
Texture2D water_tiling_normal0_texture : register(t41);
Texture2D water_tiling_normal1_texture : register(t42);
Texture2D water_non_tiling_normal0_texture : register(t43);
Texture2D water_non_tiling_normal1_texture : register(t44);
Texture2D plains_base_texture : register(t45);
Texture2D plains_height_texture : register(t46);
Texture2D plains_specular_texture : register(t47);
Texture2D desert_base_texture : register(t48);
Texture2D desert_height_texture : register(t49);
Texture2D desert_specular_texture : register(t50);
Texture2D authored_hill_texture : register(t51);
Texture2D desert_hills_base_texture : register(t52);
Texture2D desert_hills_height_texture : register(t53);
Texture2D desert_hills_specular_texture : register(t54);
Texture2D dune_decal_base_texture : register(t55);
Texture2D dune_decal_height_texture : register(t56);
Texture2D desert_mountain_base_texture : register(t57);
Texture2D desert_mountain_stripe1_texture : register(t58);
Texture2D desert_mountain_stripe2_texture : register(t59);
Texture2D desert_mountain_stripe3_texture : register(t60);
Texture2D desert_mountain_height_texture : register(t61);
Texture2D desert_mountain_specular_texture : register(t62);
Texture2D marsh_base_texture : register(t63);
Texture2D marsh_height_texture : register(t64);
Texture2D marsh_specular_texture : register(t65);
Texture2D marsh_decal_base_texture : register(t66);
Texture2D marsh_decal_height_texture : register(t67);
Texture2D marsh_decal_specular_texture : register(t68);
Texture2D volcano_base_texture : register(t69);
Texture2D volcano_height_texture : register(t70);
Texture2D volcano_active_base_texture : register(t71);
Texture2D volcano_active_specular_texture : register(t72);
Texture2D water_decal_base_texture : register(t73);
Texture2D water_decal_height_texture : register(t74);
Texture2D grassland_decal_base_texture : register(t75);
Texture2D grassland_decal_height_texture : register(t76);
Texture2D plains_decal_base_texture : register(t77);
Texture2D plains_decal_height_texture : register(t78);
Texture2D river_base_texture : register(t79);
Texture2D river_height_texture : register(t80);
Texture2D river_specular_texture : register(t81);
Texture2D river_lean0_texture : register(t82);
Texture2D river_lean1_texture : register(t83);
Texture2D river_source_base_texture : register(t84);
Texture2D river_source_height_texture : register(t85);
Texture2D river_clutter_base_texture : register(t86);
Texture2D river_clutter_height_texture : register(t87);
Texture2D river_bank_noise_texture : register(t88);
Texture2D river_rock_base_texture_0 : register(t89);
Texture2D river_rock_base_texture_1 : register(t90);
Texture2D river_rock_base_texture_2 : register(t91);
Texture2D river_rock_base_texture_3 : register(t92);
Texture2D river_rock_base_texture_4 : register(t93);
Texture2D feature_base_texture_4 : register(t94);
Texture2D feature_base_texture_5 : register(t95);
Texture2D feature_base_texture_6 : register(t96);
Texture2D feature_base_texture_7 : register(t97);
Texture2D road_base_texture_0 : register(t98);
Texture2D road_base_texture_1 : register(t99);
Texture2D road_base_texture_2 : register(t100);
Texture2D road_base_texture_3 : register(t101);
Texture2D road_base_texture_4 : register(t102);
Texture2D road_base_texture_5 : register(t103);
Texture2D road_base_texture_6 : register(t104);
Texture2D road_base_texture_7 : register(t105);
Texture2D railroad_base_texture_0 : register(t106);
Texture2D railroad_base_texture_1 : register(t107);
Texture2D road_bridge_base_texture_0 : register(t108);
Texture2D road_bridge_base_texture_1 : register(t109);
Texture2D road_bridge_base_texture_2 : register(t110);
Texture2D road_bridge_base_texture_3 : register(t111);
Texture2D road_bridge_base_texture_4 : register(t112);
Texture2D road_bridge_base_texture_5 : register(t113);
Texture2D road_bridge_base_texture_6 : register(t114);
Texture2D road_bridge_base_texture_7 : register(t115);
Texture2D resource_base_texture_0 : register(t116);
Texture2D resource_base_texture_1 : register(t117);
Texture2D resource_base_texture_2 : register(t118);
Texture2D resource_base_texture_3 : register(t119);
Texture2D resource_base_texture_4 : register(t120);
Texture2D resource_base_texture_5 : register(t121);
Texture2D resource_base_texture_6 : register(t122);
Texture2D resource_base_texture_7 : register(t123);
Texture2D city_base_texture_0 : register(t124);
Texture2D city_base_texture_1 : register(t125);
Texture2D city_base_texture_2 : register(t126);
Texture2D city_base_texture_3 : register(t127);
SamplerState material_sampler : register(s0);
SamplerState decal_sampler : register(s1);

#ifdef C3X_GAME_RENDERER
// The game compiles the same terrain functions with a small semantic settings
// contract. Fixture selection and future systems remain Lab concerns and are
// constant-folded out of the production shader.
cbuffer TerrainSettings : register(b0)
{
    float2 height_texel;
    float normal_strength;
    float exposure;
    float3 game_light_direction;
    float game_settings_padding;
};
#define lab_mode 8.0
#define beauty_relief_enabled 1.0
#define beauty_water_enabled 1.0
#define shoreline_integrated 1.0
#define promotion_tile_layout 1.0
#define scene_width 1.0
#define scene_height 1.0
#define dune_enabled 1.0
#define dune_only 0.0
#define l10_layout 1.0
#define biq_layout 1.0
#define marsh_enabled 1.0
#define marsh_only 0.0
#define volcano_enabled 1.0
#define volcano_only 0.0
#define l12_layout 1.0
#define rivers_enabled 0.0
#define rivers_only 0.0
#define l13_layout 0.0
#define l13a_layout 0.0
#define roads_enabled 0.0
#define roads_only 0.0
#define l14_layout 0.0
#define road_style_override -1.0
#define railroads_enabled 0.0
#define railroads_only 0.0
#define l15_layout 0.0
#define railroad_padding 0.0
#define resources_enabled 0.0
#define resources_only 0.0
#define l16_layout 0.0
#define resource_padding 0.0
#define cities_enabled 0.0
#define cities_only 0.0
#define l17_layout 0.0
#define city_padding 0.0
#define environment_sun_direction float3(-0.55, -0.35, 0.22)
#define environment_sun_intensity 0.0
#define environment_sun_color float3(1.0, 1.0, 1.0)
#define environment_shadow_strength 0.0
#define environment_moon_direction float3(0.0, 0.0, 1.0)
#define environment_moon_intensity 0.0
#define environment_moon_color float3(0.0, 0.0, 0.0)
#define environment_night_activation 0.0
#define environment_ambient_color float3(0.0, 0.0, 0.0)
#define environment_exposure 1.0
#define environment_water_fresnel 1.0
#define environment_water_specular 1.0
#define environment_emissive_scale 1.0
#define environment_hour 12.0
#else
cbuffer LabSettings : register(b0)
{
    float2 height_texel;
    float normal_strength;
    float exposure;
    float lab_mode;
    float beauty_relief_enabled;
    float beauty_water_enabled;
    float shoreline_integrated;
    float promotion_tile_layout;
    float scene_width;
    float scene_height;
    float dune_enabled;
    float dune_only;
    float l10_layout;
    float biq_layout;
    float marsh_enabled;
    float marsh_only;
    float volcano_enabled;
    float volcano_only;
    float l12_layout;
    float rivers_enabled;
    float rivers_only;
    float l13_layout;
    float l13a_layout;
    float3 environment_sun_direction;
    float environment_sun_intensity;
    float3 environment_sun_color;
    float environment_shadow_strength;
    float3 environment_moon_direction;
    float environment_moon_intensity;
    float3 environment_moon_color;
    float environment_night_activation;
    float3 environment_ambient_color;
    float environment_exposure;
    float environment_water_fresnel;
    float environment_water_specular;
    float environment_emissive_scale;
    float environment_hour;
    float roads_enabled;
    float roads_only;
    float l14_layout;
    float road_style_override;
    float railroads_enabled;
    float railroads_only;
    float l15_layout;
    float railroad_padding;
    float resources_enabled;
    float resources_only;
    float l16_layout;
    float resource_padding;
    float cities_enabled;
    float cities_only;
    float l17_layout;
    float city_padding;
};
#endif

float3 frame_light_direction()
{
#ifdef C3X_GAME_RENDERER
    return normalize(game_light_direction);
#else
    if (l13a_layout < 0.5)
        return normalize(float3(-0.55, -0.35, 0.22));
    float3 combined = environment_sun_direction * environment_sun_intensity +
                      environment_moon_direction * environment_moon_intensity;
    return length(combined) > 0.001
        ? normalize(combined) : normalize(float3(-0.55, -0.35, 0.22));
#endif
}

float3 frame_illumination(float3 normal, float shadow_visibility,
                          float ambient_visibility)
{
    float sun_diffuse = saturate(dot(normal, environment_sun_direction));
    float moon_diffuse = saturate(dot(normal, environment_moon_direction));
    float direct_visibility = 1.0 - environment_shadow_strength *
        (1.0 - shadow_visibility);
    float3 ambient = environment_ambient_color *
        (0.46 * ambient_visibility);
    float3 sunlight = environment_sun_color * environment_sun_intensity *
        (0.16 + sun_diffuse * 0.84) * direct_visibility;
    float3 moonlight = environment_moon_color * environment_moon_intensity *
        (0.20 + moon_diffuse * 0.80) * direct_visibility * 0.82;
#ifdef Q6_GAMEPLAY_NIGHT
    // One shared clock-driven response for every composed receiver. Preserve
    // noon radiance; retain directional moon form instead of lifting black in
    // a postprocess or exposing terrain independently from foliage/water.
    ambient *= lerp(1.0, 1.8, environment_night_activation);
    moonlight *= 2.2;
#endif
    return ambient + sunlight + moonlight;
}

float frame_output_exposure()
{
    return 1.0; // Q6: exposure belongs to final output only.
}

float frame_cast_shadow_strength()
{
    if (l13a_layout < 0.5)
        return environment_shadow_strength;
    float light_height = saturate(frame_light_direction().z);
    float daylight_floor = 0.48 + (1.0 - light_height) * 0.30;
    float minimum_strength = lerp(0.26, daylight_floor,
                                  1.0 - environment_night_activation);
    return max(environment_shadow_strength, minimum_strength);
}

float raised_form_response(float diffuse)
{
    return lerp(0.40, 1.20, smoothstep(0.10, 0.86, diffuse));
}

float3 frame_tone_map(float3 color)
{
    return color; // Q6: shared output applies the shoulder.
}

struct VertexInput
{
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
    float panel : TEXCOORD1;
    float3 geometry_normal : NORMAL0;
    float2 shape_visibility : TEXCOORD2;
    float2 macro_uv : TEXCOORD3;
    float surface_kind : TEXCOORD4;
    float surface_coordinate : TEXCOORD5;
    float base_terrain : TEXCOORD6;
    float real_terrain : TEXCOORD7;
    float4 material_weights : TEXCOORD8;
    float terrain_depth : TEXCOORD9;
    float2 authored_relief : TEXCOORD10;
    float shore_distance : TEXCOORD11;
    float4 river_data : TEXCOORD12;
    float material_tundra : TEXCOORD13;
#if defined(Q6_WORLD_SHADOWS) || defined(Q3_HYDROLOGY_DATA)
    float4 q6_world : TEXCOORD14;
#endif
#ifdef Q3_HYDROLOGY_DATA
    float4 hydrology_data : TEXCOORD15;
#endif
#ifdef Q4_RELIEF_MATERIAL_DATA
    float4 relief_material : TEXCOORD16;
#endif
};

struct PixelInput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
    float panel : TEXCOORD1;
    float3 geometry_normal : NORMAL0;
    float2 shape_visibility : TEXCOORD2;
    float2 macro_uv : TEXCOORD3;
    float surface_kind : TEXCOORD4;
    float surface_coordinate : TEXCOORD5;
    float base_terrain : TEXCOORD6;
    float real_terrain : TEXCOORD7;
    float4 material_weights : TEXCOORD8;
    float2 authored_relief : TEXCOORD9;
    float shore_distance : TEXCOORD10;
    float4 river_data : TEXCOORD11;
    float active_effect : TEXCOORD12;
    float material_tundra : TEXCOORD13;
#if defined(Q6_WORLD_SHADOWS) || defined(Q3_HYDROLOGY_DATA)
    float4 q6_world : TEXCOORD14;
#endif
#ifdef Q3_HYDROLOGY_DATA
    float4 hydrology_data : TEXCOORD15;
#endif
#ifdef Q4_RELIEF_MATERIAL_DATA
    float4 relief_material : TEXCOORD16;
#endif
};

PixelInput VSMain(VertexInput input)
{
    PixelInput output;
    output.position = float4(input.position, input.terrain_depth, 1.0);
    output.uv = input.uv;
    output.panel = input.panel;
    output.geometry_normal = input.geometry_normal;
    output.shape_visibility = input.shape_visibility;
    output.macro_uv = input.macro_uv;
    output.surface_kind = input.surface_kind;
    output.surface_coordinate = input.surface_coordinate;
    output.base_terrain = input.base_terrain;
    output.real_terrain = input.real_terrain;
    output.material_weights = input.material_weights;
    output.authored_relief = input.authored_relief;
    output.shore_distance = input.shore_distance;
    output.river_data = input.river_data;
    output.active_effect = -1.0;
    output.material_tundra = input.material_tundra;
#ifdef Q3_HYDROLOGY_DATA
    output.hydrology_data = input.hydrology_data;
#endif
#ifdef Q4_RELIEF_MATERIAL_DATA
    output.relief_material = input.relief_material;
#endif
#if defined(Q6_WORLD_SHADOWS) || defined(Q3_HYDROLOGY_DATA)
    output.q6_world = input.q6_world;
#endif
    return output;
}

struct FeatureVertexInput
{
    float3 position : POSITION;
    float2 uv : TEXCOORD0;
    float3 geometry_normal : NORMAL0;
    float material_index : TEXCOORD1;
#if defined(Q6_WORLD_SHADOWS) || defined(Q3_HYDROLOGY_DATA)
    float4 q6_world : TEXCOORD2;
    float2 city_ao_uv : TEXCOORD3;
    float3 city_tangent : TEXCOORD4;
    float3 city_bitangent : TEXCOORD5;
#endif
};

struct FeaturePixelInput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
    float3 geometry_normal : NORMAL0;
    float material_index : TEXCOORD1;
#if defined(Q6_WORLD_SHADOWS) || defined(Q3_HYDROLOGY_DATA)
    float4 q6_world : TEXCOORD2;
    float2 city_ao_uv : TEXCOORD3;
    float3 city_tangent : TEXCOORD4;
    float3 city_bitangent : TEXCOORD5;
#endif
};

FeaturePixelInput VSFeature(FeatureVertexInput input)
{
    FeaturePixelInput output;
    output.position = float4(input.position, 1.0);
    output.uv = input.uv;
    output.geometry_normal = input.geometry_normal;
    output.material_index = input.material_index;
    output.city_ao_uv = input.city_ao_uv;
    output.city_tangent = input.city_tangent;
    output.city_bitangent = input.city_bitangent;
#if defined(Q6_WORLD_SHADOWS) || defined(Q3_HYDROLOGY_DATA)
    output.q6_world = input.q6_world;
#endif
    return output;
}

float3 promotion_material_weights(float2 world_position)
{
    // Tile rows are deliberately assigned, not randomly splatted:
    // The enlarged promotion map uses grass across its two western columns,
    // plains in the upper center/east, and desert in the lower center/east.
    // Soft borders reproduce a terrain transition while keeping each Civ III
    // diamond legible as a cell.
    float enter_right = smoothstep(1.88, 2.12, world_position.x);
    float lower_row = smoothstep(2.88, 3.12, world_position.y);
    float grass_weight = 1.0 - enter_right;
    float plains_weight = enter_right * (1.0 - lower_row);
    float desert_weight = enter_right * lower_row;
    return float3(grass_weight, plains_weight, desert_weight);
}

float dune_region_weight(float2 world_position)
{
    if (dune_enabled < 0.5)
        return 0.0;
    float left = smoothstep(2.0, 2.12, world_position.x);
    float right = 1.0 - smoothstep(5.88, 6.0, world_position.x);
    float top = smoothstep(4.0, 4.12, world_position.y);
    float bottom = 1.0 - smoothstep(7.88, 8.0, world_position.y);
    return left * right * top * bottom;
}

float2 rotate_decal_uv(float2 value, float angle)
{
    float cosine = cos(angle);
    float sine = sin(angle);
    return float2(value.x * cosine - value.y * sine,
                  value.x * sine + value.y * cosine);
}

float4 sample_dune_decal(float2 world_position, float2 center, float scale, float angle)
{
    float2 uv = rotate_decal_uv((world_position - center) / scale, angle) + 0.5;
    float inside = step(0.0, uv.x) * step(uv.x, 1.0) *
                   step(0.0, uv.y) * step(uv.y, 1.0);
    float4 sample_value = dune_decal_base_texture.Sample(decal_sampler, uv);
    sample_value.a *= inside;
    return sample_value;
}

float sample_dune_decal_height(float2 world_position, float2 center,
                               float scale, float angle)
{
    float2 uv = rotate_decal_uv((world_position - center) / scale, angle) + 0.5;
    float inside = step(0.0, uv.x) * step(uv.x, 1.0) *
                   step(0.0, uv.y) * step(uv.y, 1.0);
    return dune_decal_height_texture.Sample(decal_sampler, uv).r * inside;
}

float4 combined_dune_decal(float2 world_position)
{
    float4 large0 = sample_dune_decal(world_position, float2(3.20, 5.25), 3.00, 0.300001);
    float4 large1 = sample_dune_decal(world_position, float2(4.72, 6.65), 3.00, -0.21);
    float4 small0 = sample_dune_decal(world_position, float2(2.78, 7.18), 1.25, 0.67);
    float4 small1 = sample_dune_decal(world_position, float2(5.22, 4.72), 1.25, -0.48);
    float4 combined = large0;
    combined.rgb = lerp(combined.rgb, large1.rgb, large1.a);
    combined.a = saturate(combined.a + large1.a * (1.0 - combined.a));
    combined.rgb = lerp(combined.rgb, small0.rgb, small0.a);
    combined.a = saturate(combined.a + small0.a * (1.0 - combined.a));
    combined.rgb = lerp(combined.rgb, small1.rgb, small1.a);
    combined.a = saturate(combined.a + small1.a * (1.0 - combined.a));
    return combined;
}

float combined_dune_decal_height(float2 world_position)
{
    float large0 = sample_dune_decal_height(world_position, float2(3.20, 5.25), 3.00, 0.300001);
    float large1 = sample_dune_decal_height(world_position, float2(4.72, 6.65), 3.00, -0.21);
    float small0 = sample_dune_decal_height(world_position, float2(2.78, 7.18), 1.25, 0.67);
    float small1 = sample_dune_decal_height(world_position, float2(5.22, 4.72), 1.25, -0.48);
    return max(max(large0, large1), max(small0, small1));
}

float marsh_tile_rotation(float2 tile)
{
    return frac(sin(dot(tile, float2(12.9898, 78.233))) * 43758.5453) * 6.28318530718;
}

float4 sample_marsh_decal(float2 world_position, float2 center, float scale, float angle)
{
    float2 uv = rotate_decal_uv((world_position - center) / scale, angle) + 0.5;
    float inside = step(0.0, uv.x) * step(uv.x, 1.0) *
                   step(0.0, uv.y) * step(uv.y, 1.0);
    float4 value = marsh_decal_base_texture.Sample(decal_sampler, uv);
    value.a *= inside;
    return value;
}

float sample_marsh_decal_height(float2 world_position, float2 center,
                                 float scale, float angle)
{
    float2 uv = rotate_decal_uv((world_position - center) / scale, angle) + 0.5;
    float inside = step(0.0, uv.x) * step(uv.x, 1.0) *
                   step(0.0, uv.y) * step(uv.y, 1.0);
    return marsh_decal_height_texture.Sample(decal_sampler, uv).r * inside;
}

float4 combined_marsh_decal(float2 world_position)
{
    float2 tile = floor(world_position);
    float angle = marsh_tile_rotation(tile);
    float4 primary = sample_marsh_decal(
        world_position, tile + float2(0.48, 0.52), 1.42, angle);
    float4 secondary = sample_marsh_decal(
        world_position, tile + float2(0.24, 0.70), 0.86, angle + 1.91);
    float4 tertiary = sample_marsh_decal(
        world_position, tile + float2(0.76, 0.28), 0.72, angle - 1.37);
    float4 combined = primary;
    combined.rgb = lerp(combined.rgb, secondary.rgb, secondary.a);
    combined.a = saturate(combined.a + secondary.a * (1.0 - combined.a));
    combined.rgb = lerp(combined.rgb, tertiary.rgb, tertiary.a);
    combined.a = saturate(combined.a + tertiary.a * (1.0 - combined.a));
    return combined;
}

float combined_marsh_decal_height(float2 world_position)
{
    float2 tile = floor(world_position);
    float angle = marsh_tile_rotation(tile);
    float primary = sample_marsh_decal_height(
        world_position, tile + float2(0.48, 0.52), 1.42, angle);
    float secondary = sample_marsh_decal_height(
        world_position, tile + float2(0.24, 0.70), 0.86, angle + 1.91);
    float tertiary = sample_marsh_decal_height(
        world_position, tile + float2(0.76, 0.28), 0.72, angle - 1.37);
    return max(primary, max(secondary, tertiary));
}

float2 macro_decal_uv(float2 world_position, float scale, float2 offset)
{
    return frac(world_position / scale + offset);
}

float macro_decal_hash(float2 cell)
{
    return frac(sin(dot(cell, float2(12.9898, 78.233))) * 43758.5453);
}

float2 ocean_clutter_atlas_uv(float2 local_uv, float variant)
{
    float2 origin = variant < 0.5 ? float2(0.003, 0.003) :
                    (variant < 1.5 ? float2(0.350, 0.003) :
                    (variant < 2.5 ? float2(0.697, 0.003) :
                    (variant < 3.5 ? float2(0.003, 0.258) :
                                     float2(0.350, 0.258))));
    float2 extent = variant < 2.5 ? float2(0.300, 0.249)
                                  : float2(0.341, 0.249);
    if (variant < 1.5)
        extent.x = 0.341;
    return origin + local_uv * extent;
}

float2 coast_clutter_atlas_uv(float2 local_uv, float variant)
{
    float2 origin = variant < 0.5 ? float2(0.697, 0.513) :
                    (variant < 1.5 ? float2(0.850, 0.513) :
                    (variant < 2.5 ? float2(0.697, 0.638) :
                                     float2(0.850, 0.638)));
    return origin + local_uv * float2(0.147, 0.120);
}

float2 water_clutter_uv(float2 world_position, float scale, float2 offset)
{
    float2 projected = world_position / scale + offset;
    float variant = floor(macro_decal_hash(floor(projected)) * 5.0);
    return ocean_clutter_atlas_uv(frac(projected), variant);
}

float projected_decal_edge_fade(float2 local_uv)
{
    float edge = min(min(local_uv.x, 1.0 - local_uv.x),
                     min(local_uv.y, 1.0 - local_uv.y));
    return smoothstep(0.0, 0.075, edge);
}

float4 sample_water_clutter(float2 world_position)
{
    // Select only the five authored nonzero ocean cells from their packed
    // atlas. Sampling the entire atlas mostly hit transparent packing space.
    float2 primary_projected = world_position / 1.20 + float2(0.07, 0.19);
    float2 secondary_projected = world_position / 1.65 + float2(0.61, 0.43);
    float4 primary = water_decal_base_texture.Sample(
        decal_sampler, water_clutter_uv(world_position, 1.20, float2(0.07, 0.19)));
    float4 secondary = water_decal_base_texture.Sample(
        decal_sampler, water_clutter_uv(world_position, 1.65, float2(0.61, 0.43)));
    float primary_occupancy = 1.0 - step(0.52,
        macro_decal_hash(floor(primary_projected) + float2(19.0, 7.0)));
    float secondary_occupancy = 1.0 - step(0.32,
        macro_decal_hash(floor(secondary_projected) + float2(31.0, 13.0)));
    primary.a *= projected_decal_edge_fade(frac(primary_projected)) *
                 primary_occupancy;
    secondary.a *= projected_decal_edge_fade(frac(secondary_projected)) *
                   secondary_occupancy;
    secondary.a *= 0.52;
    float4 combined = primary;
    combined.rgb = lerp(combined.rgb, secondary.rgb, secondary.a);
    combined.a = saturate(combined.a + secondary.a * (1.0 - combined.a));
    return combined;
}

float sample_water_clutter_height(float2 world_position)
{
    float primary = water_decal_height_texture.Sample(
        decal_sampler, water_clutter_uv(world_position, 1.20, float2(0.07, 0.19))).r;
    float secondary = water_decal_height_texture.Sample(
        decal_sampler, water_clutter_uv(world_position, 1.65, float2(0.61, 0.43))).r;
    float primary_fade = projected_decal_edge_fade(
        frac(world_position / 1.20 + float2(0.07, 0.19)));
    float secondary_fade = projected_decal_edge_fade(
        frac(world_position / 1.65 + float2(0.61, 0.43)));
    return max(primary * primary_fade, secondary * secondary_fade * 0.52);
}

float4 sample_coast_clutter(float2 world_position)
{
    float2 projected = world_position / 1.20 + float2(0.29, 0.53);
    float variant = floor(macro_decal_hash(floor(projected)) * 4.0);
    float4 value = water_decal_base_texture.Sample(
        decal_sampler, coast_clutter_atlas_uv(frac(projected), variant));
    float occupancy = 1.0 - step(0.52,
        macro_decal_hash(floor(projected) + float2(43.0, 17.0)));
    value.a *= projected_decal_edge_fade(frac(projected)) * occupancy;
    return value;
}

float4 combined_water_clutter(float2 world_position, float shore_distance)
{
    float4 ocean = sample_water_clutter(world_position);
    float4 coast = sample_coast_clutter(world_position);
    coast.a *= 1.0 - smoothstep(0.08, 0.72,
        saturate(shore_distance * 1.45));
    ocean.rgb = lerp(ocean.rgb, coast.rgb, coast.a);
    ocean.a = saturate(ocean.a + coast.a * (1.0 - ocean.a));
    return ocean;
}

#ifdef Q2_SOURCE_GROUND_DECALS
// Generic local triangle/UV data is supplied by the candidate fixture.
// Placement density and source-to-C3X scale are explicit Lab adaptations.
#ifndef Q2_GROUND_CELL_COUNT
#define Q2_GROUND_CELL_COUNT 50
#endif
#ifndef Q2_GROUND_PATCH_SCALE
#define Q2_GROUND_PATCH_SCALE .32
#endif
float q2_ground_hash(float2 cell,float salt) {
 float2 raw=float2(cell.x+cell.y,cell.x-cell.y);
 raw.x-=floor(raw.x/(2*Q2_GROUND_CELL_COUNT))*(2*Q2_GROUND_CELL_COUNT);
 uint x=asuint(int(raw.x)),y=asuint(int(raw.y));
 uint h=x*1664525u+y*1013904223u+uint(salt)*374761393u;
 h=(h^(h>>16))*2246822519u;h=(h^(h>>13))*3266489917u;
 return float((h^(h>>16))&0xffffffu)/16777216.0;
}
float q2_cross2(float2 a,float2 b) {return a.x*b.y-a.y*b.x;}
bool q2_ground_uv(int id,float2 q,out float2 uv,out float2 du,out float2 dv) {
 uv=0;du=0;dv=0;
 int first=ground_decal_ranges[id].x,count=ground_decal_ranges[id].y;
 [loop] for(int j=0;j<count;j+=3) {
  float4 a=ground_decal_vertices[first+j],b=ground_decal_vertices[first+j+1],c=ground_decal_vertices[first+j+2];
  float2 ab=b.xy-a.xy,ac=c.xy-a.xy,aq=q-a.xy;
  float det=q2_cross2(ab,ac);
  float s=q2_cross2(aq,ac)/det,t=q2_cross2(ab,aq)/det;
  if(s>=0&&t>=0&&s+t<=1) {
   uv=a.zw+s*(b.zw-a.zw)+t*(c.zw-a.zw);
   du=((b.zw-a.zw)*ac.y-(c.zw-a.zw)*ab.y)/det;
   dv=(-(b.zw-a.zw)*ac.x+(c.zw-a.zw)*ab.x)/det;
   return true;
  }
 }
 return false;
}
float4 q2_ground_field(Texture2D atlas,Texture2D coverage,float2 world_position,bool plains,bool height) {
 float cell_size=(Q3_MATERIAL_WRAP_WIDTH*.5)/Q2_GROUND_CELL_COUNT;
 float2 p=(world_position+float2(Q3_MATERIAL_ORIGIN_X,Q3_MATERIAL_ORIGIN_Y))/cell_size;
 float2 dx=ddx(p),dy=ddy(p),cell=floor(p);
 float4 accum=0;
 float total=0;
 [loop] for(int id=0;id<GROUND_DECAL_COUNT;id++)
  if(ground_decal_ranges[id].w==int(plains))total+=ground_decal_ranges[id].z;
 [loop] for(int oy=-1;oy<=1;oy++) [loop] for(int ox=-1;ox<=1;ox++) {
  float2 k=cell+float2(ox,oy);
  float choice=q2_ground_hash(k,plains?41:7)*total;
  int selected=-1;
  [loop] for(int id=0;id<GROUND_DECAL_COUNT;id++) {
   if(ground_decal_ranges[id].w!=int(plains))continue;
   choice-=ground_decal_ranges[id].z;
   if(choice<0){selected=id;break;}
  }
  if(selected<0)continue;
  float2 center=k+.5+(float2(q2_ground_hash(k,11),q2_ground_hash(k,13))-.5)*.42;
  float angle=q2_ground_hash(k,17)*6.2831853;
  float cs=cos(angle),sn=sin(angle);
  float4 info=ground_decal_placement[selected];
  float2 size=info.xy*info.z*(Q2_GROUND_PATCH_SCALE/cell_size)*(1+(q2_ground_hash(k,19)*2-1)*info.w);
  float2 rel=p-center;
  float2 q=float2(cs*rel.x+sn*rel.y,-sn*rel.x+cs*rel.y)/size+.5;
  if(any(q<0)||any(q>1))continue;
  float2 uv,du,dv;
  if(!q2_ground_uv(selected,q,uv,du,dv))continue;
  float2 qdx=float2(cs*dx.x+sn*dx.y,-sn*dx.x+cs*dx.y)/size;
  float2 qdy=float2(cs*dy.x+sn*dy.y,-sn*dy.x+cs*dy.y)/size;
  float2 ux=du*qdx.x+dv*qdx.y,uy=du*qdy.x+dv*qdy.y;
  float4 sampled=atlas.SampleGrad(decal_sampler,uv,ux,uy);
  float alpha=coverage.SampleGrad(decal_sampler,uv,ux,uy).a;
  if(height)sampled.rgb=(sampled.r-.5).xxx;
  accum.rgb=lerp(accum.rgb,sampled.rgb,alpha);
  accum.a=alpha+accum.a*(1-alpha);
 }
 // Return straight color to the existing material compositor.
 if(!height)accum.rgb/=max(accum.a,.00001);
 return accum;
}

#endif
float4 sample_land_clutter(Texture2D atlas, float2 world_position,
                           float scale, float2 offset)
{
#ifdef Q2_SOURCE_GROUND_DECALS
    return q2_ground_field(atlas,atlas,world_position,abs(scale-4.25)<.01,false);
#endif
    return atlas.Sample(decal_sampler, macro_decal_uv(world_position, scale, offset));
}

float sample_land_clutter_height(Texture2D atlas, float2 world_position,
                                 float scale, float2 offset)
{
    return atlas.Sample(decal_sampler, macro_decal_uv(world_position, scale, offset)).r;
}

float sample_masked_land_clutter_height(Texture2D height_atlas, Texture2D base_atlas,
                                        float2 world_position, float scale,
                                        float2 offset)
{
#ifdef Q2_SOURCE_GROUND_DECALS
    return q2_ground_field(height_atlas,base_atlas,world_position,abs(scale-4.25)<.01,true).r;
#endif
    float2 uv = macro_decal_uv(world_position, scale, offset);
    float height = height_atlas.Sample(decal_sampler, uv).r - 0.5;
    float coverage = base_atlas.Sample(decal_sampler, uv).a;
    return height * coverage;
}

float4 sample_reused_resource_slot(float slot, float2 uv)
{
    float4 sampled = resource_base_texture_7.Sample(material_sampler, uv);
    if (slot < 0.5) sampled = resource_base_texture_0.Sample(material_sampler, uv);
    else if (slot < 1.5) sampled = resource_base_texture_1.Sample(material_sampler, uv);
    else if (slot < 2.5) sampled = resource_base_texture_2.Sample(material_sampler, uv);
    else if (slot < 3.5) sampled = resource_base_texture_3.Sample(material_sampler, uv);
    else if (slot < 4.5) sampled = resource_base_texture_4.Sample(material_sampler, uv);
    else if (slot < 5.5) sampled = resource_base_texture_5.Sample(material_sampler, uv);
    else if (slot < 6.5) sampled = resource_base_texture_6.Sample(material_sampler, uv);
    return sampled;
}

#ifdef Q2_MATERIAL_RESPONSE
#ifndef Q2_SCENE_MATERIAL_V1
#define Q2_SCENE_MATERIAL_V1
// Include after the complete scene's texture declarations and PixelInput.
// Opt-in Q2_MATERIAL_RESPONSE: supplemental SOURCE detail on existing materials.
// This never replaces source relief, water, shore, macro albedo or illumination.
#ifndef Q2_SCENE_DETAIL
#define Q2_SCENE_DETAIL 1
#endif
float q2_source_height(PixelInput input,float2 uv) {
 float4 w=max(input.material_weights,0);float t=max(input.material_tundra,0);
 float total=max(.001,dot(w,1)+t);w/=total;t/=total;
 return dot(float4(height_texture.Sample(material_sampler,uv).r,
  plains_height_texture.Sample(material_sampler,uv).r,
  desert_height_texture.Sample(material_sampler,uv).r,
  marsh_height_texture.Sample(material_sampler,uv).r),w)
  +feature_base_texture_5.Sample(material_sampler,uv).r*t;
}
float q2_secondary_height(PixelInput input,float2 uv) {
 return .22*q2_source_height(input,uv*3)+.07*q2_source_height(input,uv*8);
}
float q2_base_detail_envelope(PixelInput input,float3 geometry_normal) {
 // Continuous masks preserve source-owned raised bodies and avoid a tile flag seam.
 return (input.surface_kind>.75&&input.surface_kind<1.25?1.0:0.0)
  *(1-smoothstep(.02,.45,saturate(input.authored_relief.y)))
  *smoothstep(.65,.98,geometry_normal.z);
}
#ifdef Q2_SURFACE_GRADIENT
// A physical surface gradient from the filtered source height. The old
// unnormalized one-source-texel difference nearly vanishes when a 4096-wide
// material is minified to gameplay scale. Evaluate across the visible footprint
// and divide by that interval, then transform through the actual surface basis.
// Height amplitude is a Lab interpretation in world units, not a recovered
// source-engine parameter. This branch is opt-in while visual QA is pending.
void q2_surface_gradient(PixelInput input,float3 n,inout float3 material_normal) {
 float2 ux=ddx(input.uv),uy=ddy(input.uv);
 float2 step_uv=max(height_texel,.5*(abs(ux)+abs(uy)));
 float2 g=float2(
  q2_source_height(input,input.uv+float2(step_uv.x,0))-q2_source_height(input,input.uv-float2(step_uv.x,0)),
  q2_source_height(input,input.uv+float2(0,step_uv.y))-q2_source_height(input,input.uv-float2(0,step_uv.y))) /(2*step_uv);
 float3 px=ddx(input.q6_world.xyz),py=ddy(input.q6_world.xyz);
 float3 rx=cross(py,n),ry=cross(n,px);
 float det=dot(px,rx);
 float3 gradient=(dot(g,ux)*rx+dot(g,uy)*ry)/(abs(det)>1e-9?det:1);
 float envelope=(input.surface_kind>.75&&input.surface_kind<1.25?1.0:0.0)
  *(1-smoothstep(.02,.45,saturate(input.authored_relief.y)));
 // Source base material remains active on hills, with tangent-correct response.
 float3 delta=gradient*.016;
 delta*=min(1.0,.55/max(length(delta),.00001));
 material_normal=normalize(material_normal-delta*envelope);
}
#endif
void q2_material_form(PixelInput input,float2 world_position,float3 geometry_normal,
 inout float3 albedo,inout float3 material_normal) {
#ifdef Q2_SURFACE_GRADIENT
 q2_surface_gradient(input,geometry_normal,material_normal);
#endif
 if(!Q2_SCENE_DETAIL)return;
 float envelope=q2_base_detail_envelope(input,geometry_normal);if(envelope<=0)return;
 float h=q2_secondary_height(input,input.uv);
 float hx=q2_secondary_height(input,input.uv+float2(.002,0))-h;
 float hy=q2_secondary_height(input,input.uv+float2(0,.002))-h;
 float2 delta=clamp(float2(-hx-hy,-hx+hy)*8.485281,-.08,.08)*envelope;
 material_normal=normalize(float3(material_normal.xy+delta*geometry_normal.z,material_normal.z));
 // The zero-centered secondary field is subordinate to the selected source color.
 albedo*=1+(h-.145)*.065*envelope;
}
void q2_material_specular(PixelInput input,float2 world_position,float3 geometry_normal,
 inout float specular) {
 if(!Q2_SCENE_DETAIL)return;
 float envelope=q2_base_detail_envelope(input,geometry_normal);if(envelope<=0)return;
 float h=q2_secondary_height(input,input.uv);
 // Existing source specular remains authoritative; slight roughness variation only.
 specular*=clamp(1-(h-.145)*.04*envelope,.98,1.02);
}
#endif

#endif
#ifdef Q2_CACHED_NORMAL
#ifndef Q2_CACHED_NORMAL_IMPL
#define Q2_CACHED_NORMAL_IMPL
// Optional per-material normal cache diagnostic. Final pack semantics and the
// source engine's combined-height cache boundary remain pending.
void q2_cached_normal(PixelInput input,float3 n,inout float3 material_normal) {
    float envelope=q2_base_detail_envelope(input,n);
    if(envelope<=0)return;
    float4 w=max(input.material_weights,0);float t=max(input.material_tundra,0);
    if(marsh_enabled<.5) {w.x+=w.w;w.w=0;}
    float total=max(.001,dot(w,1)+t);w/=total;t/=total;
    float2 xy=resource_base_texture_0.Sample(material_sampler,input.uv).rg*w.x
        +resource_base_texture_1.Sample(material_sampler,input.uv).rg*w.y
        +resource_base_texture_2.Sample(material_sampler,input.uv).rg*w.z
        +resource_base_texture_3.Sample(material_sampler,input.uv).rg*w.w
        +resource_base_texture_4.Sample(material_sampler,input.uv).rg*t;
    xy=xy*2-1;
    float2 ux=ddx(input.uv),uy=ddy(input.uv);
    float3 px=ddx(input.q6_world.xyz),py=ddy(input.q6_world.xyz);
    float det=ux.x*uy.y-ux.y*uy.x;
    if(abs(det)<1e-9)return;
    float3 tangent=normalize((px*uy.y-py*ux.y)/det);
    float3 bitangent=normalize((py*ux.x-px*uy.x)/det);
    // Source encoding is (+height dx,-height dy). Convert through this scene's
    // actual UV basis so the perturbation points away from rising height.
    float3 rebuilt=normalize(n-tangent*xy.x+bitangent*xy.y);
    material_normal=normalize(lerp(material_normal,rebuilt,envelope));
}
#ifdef Q2_CACHED_OCCLUSION
float q2_cached_occlusion(PixelInput input,float3 n) {
    float envelope=q2_base_detail_envelope(input,n);
    if(envelope<=0)return 1;
    float4 w=max(input.material_weights,0);float t=max(input.material_tundra,0);
    if(marsh_enabled<.5) {w.x+=w.w;w.w=0;}
    float total=max(.001,dot(w,1)+t);w/=total;t/=total;
    float ao=resource_base_texture_0.Sample(material_sampler,input.uv).b*w.x
        +resource_base_texture_1.Sample(material_sampler,input.uv).b*w.y
        +resource_base_texture_2.Sample(material_sampler,input.uv).b*w.z
        +resource_base_texture_3.Sample(material_sampler,input.uv).b*w.w
        +resource_base_texture_4.Sample(material_sampler,input.uv).b*t;
    return lerp(1,saturate(ao),envelope);
}
#endif
#endif

#endif
// Source tiling rock projected onto three planes, adopted from Q4's witness.
// Source height/footprint, placement and source skin remain unchanged.
// Define only in candidate fixtures until visual and backend gates pass.
float3 q4_relief_color(Texture2D source, PixelInput input) {
#ifdef Q4_COMBINED_ROCK_PROJECTION
    float3 weights=pow(abs(normalize(input.geometry_normal)),4);
    weights/=max(dot(weights,1),.00001);
    // q6_world supplies authoritative geometry; macro UV alone has no height
    // and stretches a narrow row of texels over every steep mountain face.
    float3 p=input.q6_world.xyz*1.5;
    return source.Sample(material_sampler,p.yz).rgb*weights.x
        +source.Sample(material_sampler,p.xz).rgb*weights.y
        +source.Sample(material_sampler,p.xy).rgb*weights.z;
#else
    return source.Sample(material_sampler,input.uv).rgb;
#endif
}

#ifdef Q4_COHERENT_ROCK_CHANNELS
#ifndef Q4_ROCK_HEIGHT_AMPLITUDE
#define Q4_ROCK_HEIGHT_AMPLITUDE .025
#endif
// Filtered height gradients share the color projection. They are transformed
// into the actual surface tangent plane; steep faces retain their response.
// The amplitude is an explicit C3X material calibration, not an engine value.
float2 q4_rock_plane_gradient(Texture2D source,float2 uv) {
    float2 dx=ddx(uv),dy=ddy(uv);
    uint width,height;source.GetDimensions(width,height);
    float2 step_uv=max(1.0/float2(width,height),.5*(abs(dx)+abs(dy)));
    return float2(
        source.SampleGrad(material_sampler,uv+float2(step_uv.x,0),dx,dy).r-
        source.SampleGrad(material_sampler,uv-float2(step_uv.x,0),dx,dy).r,
        source.SampleGrad(material_sampler,uv+float2(0,step_uv.y),dx,dy).r-
        source.SampleGrad(material_sampler,uv-float2(0,step_uv.y),dx,dy).r)/(2*step_uv);
}
float3 q4_rock_gradient(Texture2D source,PixelInput input) {
    float3 n=normalize(input.geometry_normal),w=pow(abs(n),4);
    w/=max(dot(w,1),.00001);
    float3 p=input.q6_world.xyz*1.5;
    float2 x=q4_rock_plane_gradient(source,p.yz);
    float2 y=q4_rock_plane_gradient(source,p.xz);
    float2 z=q4_rock_plane_gradient(source,p.xy);
    float3 gradient=(float3(0,x.x,x.y)*w.x+float3(y.x,0,y.y)*w.y+
        float3(z.x,z.y,0)*w.z)*1.5;
    gradient-=n*dot(n,gradient);
    return gradient;
}
float3 q4_rock_material_normal(Texture2D source,PixelInput input) {
    return normalize(normalize(input.geometry_normal)-q4_rock_gradient(source,input)*Q4_ROCK_HEIGHT_AMPLITUDE);
}
#ifdef Q4_COMPLETE_ROCK_CHANNELS
// Slots 108..115 are unused in these terrain draws. Feature draws keep their
// original bindings. The copied-packet adapter refuses occupied terrain slots.
#define q4_snow_height road_bridge_base_texture_0
#define q4_snow_specular road_bridge_base_texture_1
#define q4_stripe1_height road_bridge_base_texture_2
#define q4_stripe1_specular road_bridge_base_texture_3
#define q4_stripe2_height road_bridge_base_texture_4
#define q4_stripe2_specular road_bridge_base_texture_5
#define q4_stripe3_height road_bridge_base_texture_6
#define q4_stripe3_specular road_bridge_base_texture_7
float4 q4_rock_layer_weights(PixelInput input,bool desert) {
    float h=input.authored_relief.x,u=smoothstep(.22,.62,h);
    float4 w=float4(1-u,u,0,0);
    if(desert) {
        float second=smoothstep(.55,.76,h),third=smoothstep(.76,.90,h);
        w=lerp(w,float4(0,0,1,0),second);
        w=lerp(w,float4(0,0,0,1),third);
    } else {
        float snow=smoothstep(.70,.84,h)*smoothstep(.18,.72,input.geometry_normal.z);
        w=lerp(w,float4(0,0,1,0),snow);
    }
    return w;
}
float3 q4_complete_rock_normal(PixelInput input,bool desert) {
    float4 w=q4_rock_layer_weights(input,desert);
    float3 g;
    if(desert) {
        g=q4_rock_gradient(desert_mountain_height_texture,input)*w.x;
        if(w.y>0)g+=q4_rock_gradient(q4_stripe1_height,input)*w.y;
        if(w.z>0)g+=q4_rock_gradient(q4_stripe2_height,input)*w.z;
        if(w.w>0)g+=q4_rock_gradient(q4_stripe3_height,input)*w.w;
    } else {
        // Normalized base and top height/specular payloads are identical.
        g=q4_rock_gradient(mountain_height_texture,input)*(w.x+w.y);
        if(w.z>0)g+=q4_rock_gradient(q4_snow_height,input)*w.z;
    }
    return normalize(normalize(input.geometry_normal)-g*Q4_ROCK_HEIGHT_AMPLITUDE);
}
float q4_complete_rock_specular(PixelInput input,bool desert) {
    float4 w=q4_rock_layer_weights(input,desert);
    if(desert)return dot(w,float4(q4_relief_color(desert_mountain_specular_texture,input).r,
        q4_relief_color(q4_stripe1_specular,input).r,q4_relief_color(q4_stripe2_specular,input).r,
        q4_relief_color(q4_stripe3_specular,input).r));
    return q4_relief_color(mountain_specular_texture,input).r*(w.x+w.y)+
        q4_relief_color(q4_snow_specular,input).r*w.z;
}
#endif
#endif

#ifdef Q2_SOURCE_ALPHA_BLEND
#ifndef Q2_SOURCE_BLEND
#define Q2_SOURCE_BLEND
// Source cache-bake bytecode weights color/spec/height by color.a^2 times
// an interpolated contribution, then resolves accumulated channels by weight.
// This diagnostic uses the Lab's existing terrain contributions; their mapping
// to the source engine's vertex contribution remains a Lab interpretation.
void q2_source_blend(inout PixelInput input) {
    if(input.surface_kind<.75 || input.surface_kind>1.25)return;
    float4 w=max(input.material_weights,0);
    float t=max(input.material_tundra,0);
    float4 a=float4(base_color_texture.Sample(material_sampler,input.uv).a,
        plains_base_texture.Sample(material_sampler,input.uv).a,
        desert_base_texture.Sample(material_sampler,input.uv).a,
        marsh_base_texture.Sample(material_sampler,input.uv).a);
    float ta=feature_base_texture_4.Sample(material_sampler,input.uv).a;
    float4 weighted=w*a*a;
    float wt=t*ta*ta;
    float total=dot(weighted,1)+wt;
    if(total>1e-8) {
        input.material_weights=weighted/total;
        input.material_tundra=wt/total;
    }
}
#endif

#endif
#ifdef Q2_CONTINENTAL_MATERIAL
// Candidate-only flat/high material graph. Terrain draw bindings 116..121;
// feature draws retain their own resource materials in the same slot range.
#define q2_grass_high_color resource_base_texture_0
#define q2_grass_high_height resource_base_texture_1
#define q2_grass_high_specular resource_base_texture_2
#define q2_plains_high_color resource_base_texture_3
#define q2_plains_high_height resource_base_texture_4
#define q2_plains_high_specular resource_base_texture_5
float2 q2_continental_mix(PixelInput input) {
 float4 w=input.material_weights/max(.001,dot(input.material_weights,1)+input.material_tundra);
 float envelope=(input.surface_kind>.75&&input.surface_kind<1.25?1.0:0.0)
    *(1-smoothstep(.02,.2,input.authored_relief.y))*smoothstep(.985,1,input.geometry_normal.z);
 // The same actual displaced geometry drives color/height/specular. The
 // threshold is a Lab hypothesis; source engine high-layer masks are unknown.
 float h=max(0,input.q6_world.z*112-2.5)/max(.01,14*w.x+10*w.y);
 return w.xy*smoothstep(.32,.48,h)*envelope;
}
float q2_continental_height_delta(PixelInput input,float2 uv) {
 float2 w=q2_continental_mix(input);
 return w.x*(q2_grass_high_height.Sample(material_sampler,uv).r-height_texture.Sample(material_sampler,uv).r)
  +w.y*(q2_plains_high_height.Sample(material_sampler,uv).r-plains_height_texture.Sample(material_sampler,uv).r);
}
float q2_continental_specular_delta(PixelInput input) {
 float2 w=q2_continental_mix(input);
 return w.x*(q2_grass_high_specular.Sample(material_sampler,input.uv).r-specular_texture.Sample(material_sampler,input.uv).r)
  +w.y*(q2_plains_high_specular.Sample(material_sampler,input.uv).r-plains_specular_texture.Sample(material_sampler,input.uv).r);
}

#endif
// Included after frozen texture, environment and pixel input declarations.
// Q0 world extension and wire5 common b1 are opt-in; ordinary linear is no-op.
#ifndef Q6_CAST_SHADOWS
#define Q6_CAST_SHADOWS 1
#endif
#ifndef Q6_SCENE_CONTACT
#define Q6_SCENE_CONTACT 1
#endif
#ifdef Q6_WORLD_SHADOWS
#ifndef Q6_FRAME_SHADOW_V1
#define Q6_FRAME_SHADOW_V1
// Shared Q6 receiver query for shadow_field_v1.h; no private clock or material bindings.
float q6_shadow_visibility(Texture2D field, float3 world, float3 normal,
 float4 ShadowU, float4 ShadowV, float4 ShadowL, bool shadows, bool contact){
 if(!shadows)return 1;
 // One shadow texel bounds the receiver footprint crossing adjacent facets.
 // Opt-in until the combined scene passes visual review; no global depth lift.
#ifndef Q6_TEXEL_RECEIVER_OFFSET
#define Q6_TEXEL_RECEIVER_OFFSET 0
#endif
 float offset=Q6_TEXEL_RECEIVER_OFFSET?min(ShadowU.w/ShadowV.w,6.0/1024.0):.001;
 float3 w=world+normal*offset;
 float2 uv=float2(dot(w,ShadowU.xyz),dot(w,ShadowV.xyz))/ShadowU.w+.5;
 float z=dot(w,ShadowL.xyz)/ShadowU.w+.5;
 // Match comparison depth to each sampled receiver-plane location. A fixed
 // bias on sloping roofs/rocks creates a stippled false self shadow.
 float2 geometry_uv=float2(dot(world,ShadowU.xyz),dot(world,ShadowV.xyz))/ShadowU.w+.5;
 float geometry_z=dot(world,ShadowL.xyz)/ShadowU.w+.5;
 float2 plane_uv=Q6_TEXEL_RECEIVER_OFFSET?geometry_uv:uv;
 float plane_z=Q6_TEXEL_RECEIVER_OFFSET?geometry_z:z;
 float2 ux=ddx(plane_uv),uy=ddy(plane_uv);float zx=ddx(plane_z),zy=ddy(plane_z);
 float determinant=ux.x*uy.y-ux.y*uy.x;
 float2 gradient=abs(determinant)>1e-12?float2(zx*uy.y-zy*ux.y,zy*ux.x-zx*uy.x)/determinant:0;
 int2 center=int2(uv*ShadowV.w);float sum=0,closest_delta=0;
 for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
   float blocker=field.Load(int3(clamp(center+int2(x,y),int2(0,0),int2(ShadowV.w-1,ShadowV.w-1)),0)).r;
   float2 sampled_uv=(float2(center+int2(x,y))+.5)/ShadowV.w;
   float receiver=z+dot(gradient,sampled_uv-uv);
   // Physical bias remains fixed when a larger scene needs a wider field.
   float bias=.00060/ShadowU.w;
   sum+=step(blocker,receiver+bias);
   if(x==0&&y==0)closest_delta=blocker-receiver;
 }
 float soft=sum/9;
 // Tighten only genuinely adjacent caster/receiver contact, no blanket AO.
 float world_gap=closest_delta*ShadowU.w;
 if(contact && world_gap>.0039 && world_gap<.024)soft=min(soft,.15);
 return soft;
}

#ifndef Q6_CAST_SHADOWS
#define Q6_CAST_SHADOWS 1
#endif
#ifndef Q6_SCENE_CONTACT
#define Q6_SCENE_CONTACT 1
#endif
// Shared wire5/6 b1 for every shader namespace; texture binding is per draw.
cbuffer Q6SharedShadow : register(b1) {
 float4 Q6ShadowU; // xyz light right; w span in normalized tile units
 float4 Q6ShadowV; // xyz light up; w resolution
 float4 Q6ShadowL;
 float4 Q6ShadowOrigin;
 float4 Q6ShadowFlags; // enabled, tighter contact, reserved, reserved
};
float q6_world_visibility(Texture2D field,float4 world,float3 normal,bool water) {
 if(world.w<=.5 || Q6ShadowFlags.x<=.5)return 1;
 return q6_shadow_visibility(field,world.xyz-Q6ShadowOrigin.xyz,normal,
  Q6ShadowU,Q6ShadowV,Q6ShadowL,Q6_CAST_SHADOWS,
  !water && Q6_SCENE_CONTACT && Q6ShadowFlags.y>.5);
}
#endif

#endif

float q6_receiver_visibility(PixelInput input,float3 normal,float legacy_shadow) {
#ifdef Q6_WORLD_SHADOWS
 if(input.q6_world.w>.5 && Q6ShadowFlags.x>.5) {
  bool water=input.surface_kind>3.5 && input.surface_kind<6.5;
  water=water || (input.surface_kind>8.5 && input.surface_kind<9.5);
  // Main t25 aliases a feature-only source binding; Q0 binds one common field.
  float visibility=q6_world_visibility(feature_base_texture_0,input.q6_world,normal,water);
  return visibility;
 }
#endif
 return legacy_shadow;
}
float3 q6_receiver_illumination(PixelInput input,float3 normal,
 float legacy_shadow,float ambient_visibility) {
 return frame_illumination(normal,q6_receiver_visibility(input,normal,legacy_shadow),ambient_visibility);
}
float3 q6_receiver_illumination(FeaturePixelInput input,float3 normal,
 float legacy_shadow,float ambient_visibility) {
#ifdef Q6_WORLD_SHADOWS
 if(input.q6_world.w>.5 && Q6ShadowFlags.x>.5) {
  // Feature t17 aliases the bed-only source binding.
  float visibility=q6_world_visibility(shallow_bed_texture,input.q6_world,normal,false);
  return frame_illumination(normal,visibility,ambient_visibility);
 }
#endif
 return frame_illumination(normal,legacy_shadow,ambient_visibility);
}

#ifdef Q4_COASTAL_ROCKS
// Per-source-material batch: t25 base, t26 LEAN0, t27 LEAN1, t28 gloss.
// UV derivatives adapt the source slope channels; this is not a recovered
// source-engine LEAN equation. The original source channels remain bound.
float4 q4_coastal_rock(FeaturePixelInput input){
 float3 albedo=feature_base_texture_0.Sample(material_sampler,input.uv).rgb;
 float2 lean=feature_base_texture_1.Sample(material_sampler,input.uv).rg*2-1;
 float moment=feature_base_texture_2.Sample(material_sampler,input.uv).r;
 float gloss=feature_base_texture_3.Sample(material_sampler,input.uv).r;
 float3 n=normalize(input.geometry_normal);
 float3 world=input.q6_world.xyz*float3(1,-1,1);
 float3 dx=ddx(world),dy=ddy(world);
 float2 ux=ddx(input.uv),uy=ddy(input.uv);
 float det=ux.x*uy.y-ux.y*uy.x;
 if(abs(det)>1e-9){
  float3 tangent=normalize((dx*uy.y-dy*ux.y)/det);
  float3 bitangent=normalize((dy*ux.x-dx*uy.x)/det);
  n=normalize(n+(tangent*lean.x+bitangent*lean.y)*.35);
 }
 float3 color=albedo*q6_receiver_illumination(input,n,1,1);
 float roughness=clamp(1-gloss+max(0,moment-dot(lean,lean)*.25)*.25,.25,1);
 float3 view=normalize(float3(0,-.52,.86));
 float3 sunhalf=normalize(environment_sun_direction+view);
 float3 moonhalf=normalize(environment_moon_direction+view);
 float spec=lerp(100,8,roughness);
 color+=gloss*.045*(environment_sun_color*environment_sun_intensity*pow(saturate(dot(n,sunhalf)),spec)
       +environment_moon_color*environment_moon_intensity*pow(saturate(dot(n,moonhalf)),spec));
 return float4(color,1);
}

#endif
#ifdef Q3_WATER_MATERIAL
float4 q3_water_material(PixelInput input);
#endif
#ifdef Q3_SHORE_MATERIAL
void q3_shore_material(PixelInput input, float2 world_position, inout float3 albedo, inout float3 material_normal);
#endif

float4 q6_raw_feature(FeaturePixelInput input)
{
#ifdef Q4_COASTAL_ROCKS
    if(abs(input.material_index-.48)<.001)return q4_coastal_rock(input);
#endif

    float3 albedo;
    float3 emissive = 0.0;
    float material_fraction = frac(input.material_index);
    float source_decal_weight = step(20.5, input.material_index) *
        (1.0 - step(28.5, input.material_index)) * step(0.005, material_fraction);
    float infrastructure_weight = step(0.195, material_fraction) *
        (1.0 - step(0.295, material_fraction)) * source_decal_weight;
    float tile_object_weight = step(0.095, material_fraction) *
        (1.0 - step(0.195, material_fraction)) * source_decal_weight;
    float unit_weight = step(0.395, material_fraction) *
        (1.0 - step(0.445, material_fraction)) * source_decal_weight;
    float resource_weight = step(20.5, input.material_index) *
        (1.0 - step(28.5, input.material_index)) *
        (1.0 - step(0.005, material_fraction));
    float mine_weight = source_decal_weight *
        (1.0 - tile_object_weight) * (1.0 - infrastructure_weight) *
        (1.0 - unit_weight);
    float raised_infrastructure_weight = infrastructure_weight *
        (1.0 - step(0.25, material_fraction));
    float pollution_weight = step(0.295, material_fraction) *
        (1.0 - step(0.305, material_fraction));
    float ground_state_weight = step(0.295, material_fraction) *
        (1.0 - step(0.320, material_fraction)) * source_decal_weight;
    float pollution_base_weight = pollution_weight *
        (1.0 - step(21.5, input.material_index));
    float pollution_detail_weight = pollution_weight *
        step(21.5, input.material_index);
    float mine_slot = floor(input.material_index + 0.001) - 21.0;
    if (input.material_index < 0.5)
        albedo = feature_base_texture_0.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 1.5)
        albedo = feature_base_texture_1.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 2.5)
        albedo = feature_base_texture_2.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 3.5)
        albedo = feature_base_texture_3.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 4.5)
        albedo = feature_base_texture_4.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 5.5)
        albedo = feature_base_texture_5.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 6.5)
        albedo = feature_base_texture_6.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 7.5)
        albedo = feature_base_texture_7.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 8.5)
        albedo = river_rock_base_texture_0.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 9.5)
        albedo = river_rock_base_texture_1.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 10.5)
        albedo = river_rock_base_texture_2.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 11.5)
        albedo = river_rock_base_texture_3.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 12.5)
        albedo = river_rock_base_texture_4.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 13.5)
        albedo = road_bridge_base_texture_0.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 14.5)
        albedo = road_bridge_base_texture_1.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 15.5)
        albedo = road_bridge_base_texture_2.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 16.5)
        albedo = road_bridge_base_texture_3.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 17.5)
        albedo = road_bridge_base_texture_4.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 18.5)
        albedo = road_bridge_base_texture_5.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 19.5)
        albedo = road_bridge_base_texture_6.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 20.5)
        albedo = road_bridge_base_texture_7.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 21.5)
        albedo = resource_base_texture_0.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 22.5)
        albedo = resource_base_texture_1.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 23.5)
        albedo = resource_base_texture_2.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 24.5)
        albedo = resource_base_texture_3.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 25.5)
        albedo = resource_base_texture_4.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 26.5)
        albedo = resource_base_texture_5.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 27.5)
        albedo = resource_base_texture_6.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 28.5)
        albedo = resource_base_texture_7.Sample(material_sampler, input.uv).rgb;
    else if (input.material_index < 29.5)
    {
        albedo = city_base_texture_0.Sample(material_sampler, input.uv).rgb;
        emissive = resource_base_texture_0.Sample(material_sampler, input.uv).rgb;
    }
    else if (input.material_index < 30.5)
    {
        albedo = city_base_texture_1.Sample(material_sampler, input.uv).rgb;
        emissive = resource_base_texture_1.Sample(material_sampler, input.uv).rgb;
    }
    else if (input.material_index < 31.5)
    {
        albedo = city_base_texture_2.Sample(material_sampler, input.uv).rgb;
        emissive = resource_base_texture_2.Sample(material_sampler, input.uv).rgb;
    }
    else
    {
        albedo = city_base_texture_3.Sample(material_sampler, input.uv).rgb;
        emissive = resource_base_texture_3.Sample(material_sampler, input.uv).rgb;
    }
    float4 mine_sample = sample_reused_resource_slot(mine_slot, input.uv);
    clip(lerp(1.0, mine_sample.a - 0.08,
              source_decal_weight * (1.0 - unit_weight)));
    float mine_emissive_code = floor(material_fraction * 100.0 + 0.5);
    if (mine_weight > 0.5 && mine_emissive_code > 1.5)
        emissive = mine_emissive_code < 2.5
            ? city_base_texture_0.Sample(material_sampler, input.uv).rgb
            : city_base_texture_1.Sample(material_sampler, input.uv).rgb;
    float infrastructure_emissive_code = floor(material_fraction * 1000.0 + 0.5) -
        floor(material_fraction * 100.0 + 0.5) * 10.0;
    if (raised_infrastructure_weight > 0.5 && infrastructure_emissive_code > 0.5)
        emissive = city_base_texture_0.Sample(material_sampler, input.uv).rgb;
    float city_weight = step(28.5, input.material_index);
    float owner_code = floor(frac(input.material_index) * 12.5 + 0.25) - 1.0;
    float3 owner_tint = owner_code < 0.5 ? float3(0.78, 0.94, 1.12) :
        (owner_code < 1.5 ? float3(1.12, 0.80, 0.72) :
        (owner_code < 2.5 ? float3(0.80, 1.08, 0.80) : float3(1.10, 0.92, 0.68)));
    albedo *= lerp(float3(1.0, 1.0, 1.0), owner_tint, city_weight * 0.10);
    float tile_object_owner = floor(material_fraction * 100.0 + 0.5) - 10.0;
    float3 tile_object_tint = tile_object_owner < 0.5 ? float3(0.78, 0.94, 1.12) :
        (tile_object_owner < 1.5 ? float3(1.12, 0.80, 0.72) :
        (tile_object_owner < 2.5 ? float3(0.80, 1.08, 0.80) :
                                  float3(1.10, 0.92, 0.68)));
    albedo *= lerp(float3(1.0, 1.0, 1.0), tile_object_tint,
                   tile_object_weight * 0.07);
    float infrastructure_owner = floor(material_fraction * 100.0 + 0.5) - 20.0;
    float3 infrastructure_tint = infrastructure_owner < 0.5
        ? float3(0.78, 0.94, 1.12)
        : (infrastructure_owner < 1.5 ? float3(1.12, 0.80, 0.72)
        : (infrastructure_owner < 2.5 ? float3(0.80, 1.08, 0.80)
                                      : float3(1.10, 0.92, 0.68)));
    albedo *= lerp(float3(1.0, 1.0, 1.0), infrastructure_tint,
                   raised_infrastructure_weight * 0.07);
    float unit_owner = floor((material_fraction - 0.40) * 100.0 + 0.05);
    float unit_style_fraction = material_fraction - 0.40 - unit_owner * 0.01;
    float unit_team_code = floor(unit_style_fraction * 1000.0 + 0.05);
    float unit_source_tint_code = floor(
        (unit_style_fraction - unit_team_code * 0.001) * 10000.0 + 0.5);
    float3 unit_color = unit_owner < 0.5 ? float3(0.08, 0.36, 0.95) :
        (unit_owner < 1.5 ? float3(0.90, 0.08, 0.05) :
        (unit_owner < 2.5 ? float3(0.08, 0.64, 0.14) :
                            float3(0.98, 0.70, 0.04)));
    // Civ VI component textures are deliberately neutral calibration maps;
    // their ArtDef Tint names supply the actual material color. These values
    // are the installed Base Units.artdef colors selected by the imported
    // components, applied before the separate civilization-color mask.
    float3 unit_neutral_tint = unit_source_tint_code < 0.5 ? float3(1.0, 1.0, 1.0) :
        (unit_source_tint_code < 1.5 ? float3(0.878, 0.765, 0.647) :
        (unit_source_tint_code < 2.5 ? float3(0.631, 0.067, 0.059) :
        (unit_source_tint_code < 3.5 ? float3(0.529, 0.286, 0.059) :
        (unit_source_tint_code < 4.5 ? float3(0.404, 0.345, 0.239) :
        (unit_source_tint_code < 5.5 ? float3(0.651, 0.514, 0.239) :
        (unit_source_tint_code < 6.5 ? float3(0.431, 0.596, 0.290) :
                                       float3(0.784, 0.580, 0.302)))))));
    float unit_source_tint_weight = step(0.5, unit_source_tint_code) * unit_weight;
    albedo = lerp(albedo, albedo * unit_neutral_tint, unit_source_tint_weight);
    // The normalized component metadata distinguishes source-alpha masks from
    // deliberately solid insignia pieces and carries strong/medium/restrained
    // authoring strengths. This keeps horses, skin, wood, steel, and cloth
    // neutral while shields, armbands, barding, sails, and vehicle panels read
    // clearly at Civ III scale.
    float unit_source_mask = smoothstep(0.06, 0.94, 1.0 - mine_sample.a);
    float unit_team_mask = unit_team_code < 0.5 ? 0.0 :
        (unit_team_code < 1.5 ? unit_source_mask :
        (unit_team_code < 2.5 ? 1.0 :
        (unit_team_code < 3.5 ? unit_source_mask :
        (unit_team_code < 4.5 ? 1.0 :
        (unit_team_code < 5.5 ? unit_source_mask : 1.0)))));
    float unit_team_strength = unit_team_code < 0.5 ? 0.0 :
        (unit_team_code < 2.5 ? 0.82 :
        (unit_team_code < 4.5 ? 0.58 : 0.35));
    // Map the neutral source value through a compact owner-color ramp instead
    // of replacing it with a flat hue. Light folds stay bright, recesses stay
    // dark, and a small source contribution preserves material texture.
    float unit_source_value = dot(albedo, float3(0.2126, 0.7152, 0.0722));
    float3 unit_color_dark = unit_color * 0.32;
    float3 unit_color_light = saturate(unit_color * 0.90 +
                                        float3(0.24, 0.24, 0.20));
    float3 unit_color_ramp = lerp(unit_color_dark, unit_color_light,
                                  smoothstep(0.08, 0.86, unit_source_value));
    float3 unit_tinted = lerp(unit_color_ramp, albedo, 0.14);
    albedo = lerp(albedo, unit_tinted,
                  unit_weight * unit_team_mask * unit_team_strength);
    // Persistent damage is source art blended into the terrain, not a square
    // replacement tile.  The crater-ground layer supplies a subdued dead-soil
    // footprint while the radiation atlas contributes only its authored alpha
    // detail; both feather before the source quad edge.
    albedo = lerp(albedo, float3(0.13, 0.115, 0.055), pollution_weight);
    float3 normal = normalize(input.geometry_normal);
    float3 light_direction = frame_light_direction();
    // Small vegetation, resources, units, cities, and raised improvements
    // otherwise collapse toward flat ambient color at Civ III scale. Preserve
    // authored normals, but expand their horizontal response so each receives
    // the same readable lit and opposing faces as relief. Flat pollution and
    // crater decals retain their unmodified normals.
    float vegetation_weight = 1.0 - step(7.5, input.material_index);
    float3 vegetation_normal = normalize(float3(
        normal.xy * 2.40, max(0.18, normal.z * 0.70)));
    float3 form_normal = normalize(lerp(normal, vegetation_normal,
                                         vegetation_weight));
    float raised_map_object_weight = saturate(
        city_weight + tile_object_weight + raised_infrastructure_weight +
        mine_weight * (1.0 - ground_state_weight));
    float raised_detail_weight = saturate(
        resource_weight + unit_weight + raised_map_object_weight);
    float horizontal_gain = 1.0 + resource_weight * 0.95 + unit_weight * 1.10 +
                            raised_map_object_weight * 0.82;
    float vertical_gain = 1.0 - resource_weight * 0.18 - unit_weight * 0.22 -
                          raised_map_object_weight * 0.16;
    float3 raised_detail_normal = normalize(float3(
        normal.xy * horizontal_gain, max(0.18, normal.z * vertical_gain)));
    form_normal = normalize(lerp(form_normal, raised_detail_normal,
                                  raised_detail_weight));
    float signed_diffuse = saturate(dot(form_normal, light_direction));
    float diffuse = l13a_layout > 0.5
        ? signed_diffuse : abs(dot(normal, light_direction));
    float3 light = l13a_layout > 0.5
        ? q6_receiver_illumination(input, form_normal, 1.0, 1.0)
        : (0.62 + diffuse * 0.62).xxx;
    if (l13a_layout > 0.5)
    {
        // Civ VI's relief readability comes chiefly from decisive opposing
        // face values rather than a uniformly dark footprint. Preserve the
        // shared environment color while opening lit crowns and deepening the
        // unlit sides of renderer-owned bodies.
        float feature_form = raised_form_response(signed_diffuse);
        light *= feature_form;
    }
    float3 lit_color = albedo * light + emissive *
        saturate(city_weight + mine_weight + raised_infrastructure_weight) *
        environment_night_activation * environment_emissive_scale * 1.45;
    float3 display_color = (frame_tone_map(
        lit_color * frame_output_exposure()));
    float2 ground_local_uv = frac(input.uv * 2.0);
    float ground_radius = length((ground_local_uv - 0.5) * 2.0);
    float ground_feather = 1.0 - smoothstep(0.58, 1.0, ground_radius);
    float pollution_alpha = pollution_base_weight * 0.30 * ground_feather +
        pollution_detail_weight * mine_sample.a * 0.62 * ground_feather;
    float crater_alpha = mine_sample.a * 0.85 * ground_feather;
    float ground_alpha = lerp(1.0,
        lerp(crater_alpha, pollution_alpha, pollution_weight),
        ground_state_weight);
    return float4(display_color, ground_alpha);
}

float4 sample_road_source(float2 uv, float style, float pillaged)
{
    float4 base_route = lerp(
        road_base_texture_0.Sample(decal_sampler, uv),
        road_base_texture_1.Sample(decal_sampler, uv), pillaged);
    float4 detail_route = base_route;
    if (style > 0.5 && style < 1.5)
        detail_route = lerp(
            road_base_texture_2.Sample(decal_sampler, uv),
            road_base_texture_3.Sample(decal_sampler, uv), pillaged);
    else if (style > 1.5 && style < 2.5)
        detail_route = lerp(
            road_base_texture_4.Sample(decal_sampler, uv),
            road_base_texture_5.Sample(decal_sampler, uv), pillaged);
    else if (style > 2.5)
        detail_route = lerp(
            road_base_texture_6.Sample(decal_sampler, uv),
            road_base_texture_7.Sample(decal_sampler, uv), pillaged);
    float layered = step(0.5, style);
    return float4(
        lerp(base_route.rgb, detail_route.rgb, detail_route.a * layered),
        max(base_route.a, detail_route.a * layered));
}

float4 sample_railroad_source(float along, float across, float pillaged)
{
    float2 rail_direction = normalize(float2(1.0, 0.24705882));
    float2 rail_perpendicular = float2(-rail_direction.y, rail_direction.x);
    float2 rail_uv_a = float2(along, lerp(0.75294118, 1.0, along)) +
                       rail_perpendicular * across * 0.070;
    float2 rail_uv_b = float2(along, lerp(0.25098039, 0.49803922, along)) +
                       rail_perpendicular * across * 0.070;
    float4 sleepers = lerp(
        railroad_base_texture_0.Sample(decal_sampler, rail_uv_a),
        railroad_base_texture_1.Sample(decal_sampler, rail_uv_a), pillaged);
    float4 ballast = lerp(
        railroad_base_texture_0.Sample(decal_sampler, rail_uv_b),
        railroad_base_texture_1.Sample(decal_sampler, rail_uv_b), pillaged);
    // The source railroad atlas stores its paired steel lines in the upper
    // strip separately from the sleeper/ballast pieces. Sample that authored
    // steel color and place both narrow rails over the authored sleepers.
    float4 steel_source = lerp(
        railroad_base_texture_0.Sample(decal_sampler, float2(frac(along), 0.095)),
        railroad_base_texture_1.Sample(decal_sampler, float2(frac(along), 0.095)),
        pillaged);
    float rail_distance = abs(abs(across) - 0.34);
    float rail_coverage = 1.0 - smoothstep(0.040, 0.075, rail_distance);
    float4 bed = ballast.a > sleepers.a ? ballast : sleepers;
    return float4(lerp(bed.rgb, steel_source.rgb, rail_coverage),
                  max(bed.a, rail_coverage));
}

bool q4_volcano_surface(PixelInput input) {
#ifdef Q4_RELIEF_MATERIAL_DATA
    return input.relief_material.z>.00001;
#else
    return abs(input.real_terrain-10)<.25;
#endif
}
float q4_volcano_coverage(PixelInput input) {
#ifdef Q4_RELIEF_MATERIAL_DATA
    return smoothstep(.02,.62,input.relief_material.z);
#else
    return smoothstep(.02,.62,input.authored_relief.y);
#endif
}
float q4_volcano_active(PixelInput input,float fallback) {
#ifdef Q4_RELIEF_MATERIAL_DATA
    return input.relief_material.w;
#else
    return fallback;
#endif
}
float4 q6_raw_main(PixelInput input)
{
#ifdef Q3_WATER_MATERIAL
    if(input.panel > .5 && ((input.surface_kind > 3.5 && input.surface_kind < 6.5)
        || (input.surface_kind > 8.5 && input.surface_kind < 9.5)))
        return q3_water_material(input);
#endif
    if (input.panel > 0.5 && input.surface_kind > 12.5 && input.surface_kind < 13.5)
    {
        float owner = input.base_terrain;
        float3 civ_color = owner < 0.5 ? float3(0.08, 0.36, 0.95) :
            (owner < 1.5 ? float3(0.90, 0.08, 0.05) :
            (owner < 2.5 ? float3(0.08, 0.64, 0.14) :
                           float3(0.98, 0.70, 0.04)));
        float edge = 1.0 - smoothstep(0.66, 1.0, abs(input.shape_visibility.x));
        float daylight = saturate(environment_sun_intensity +
                                  environment_moon_intensity * 0.45);
        float3 display_color = (frame_tone_map(
            civ_color * lerp(0.50, 1.18, daylight) * frame_output_exposure()));
        clip(edge - 0.02);
        // One owner color only: coverage softens the ribbon edge, but no
        // secondary dark/light civilization stripe is synthesized.
        return float4(display_color, edge * 0.94);
    }
    if (input.panel > 0.5 && input.surface_kind > 10.5 && input.surface_kind < 11.5)
    {
        float railroad = step(3.5, input.base_terrain);
        if ((railroad < 0.5 && roads_enabled < 0.5) ||
            (railroad > 0.5 && railroads_enabled < 0.5))
            clip(-1.0);
        clip(input.material_weights.x);
        clip(input.material_weights.y);
        clip(scene_width - input.material_weights.x);
        clip(scene_height - input.material_weights.y);
        float pillaged = step(0.5, input.real_terrain);
        float4 authored_route = railroad > 0.5
            ? sample_railroad_source(input.shape_visibility.y,
                                     input.shape_visibility.x, pillaged)
            : sample_road_source(input.uv, input.base_terrain, pillaged);
        float4 center_route = sample_road_source(
            input.macro_uv, input.base_terrain, pillaged);
        if (railroad > 0.5)
        {
            center_route = lerp(
                road_base_texture_0.Sample(decal_sampler, input.macro_uv),
                road_base_texture_1.Sample(decal_sampler, input.macro_uv), pillaged);
            float4 rail_detail = sample_railroad_source(
                input.shape_visibility.y, input.shape_visibility.x, pillaged);
            center_route = lerp(center_route, rail_detail,
                                saturate(rail_detail.a));
            authored_route = float4(
                lerp(center_route.rgb, rail_detail.rgb, rail_detail.a),
                max(center_route.a * 0.72, rail_detail.a));
        }
        // The normalized atlas contributes the visible color, variation, and
        // authored edge coverage. A narrow center ribbon is a topology guard:
        // it closes transparent pinholes without repainting or widening the
        // material, so every logical graph edge remains visually connected.
        float continuous_ribbon = 1.0 - smoothstep(
            0.46, 0.70, abs(input.shape_visibility.x));
        float alpha = max(authored_route.a, continuous_ribbon * 0.88);
        clip(alpha - 0.025);
        float center_weight = continuous_ribbon *
            (1.0 - smoothstep(0.02, 0.20, authored_route.a));
        float3 albedo = lerp(authored_route.rgb, center_route.rgb, center_weight);
        float3 normal = normalize(input.geometry_normal);
        float3 light = l13a_layout > 0.5
            ? q6_receiver_illumination(input, normal, 1.0, 1.0)
            : (0.68 + 0.72 * saturate(dot(normal, frame_light_direction()))).xxx;
        float3 display_color = (frame_tone_map(
            albedo * light * frame_output_exposure()));
        return float4(display_color, alpha);
    }
    if (roads_only > 0.5 || railroads_only > 0.5 || resources_only > 0.5 ||
        cities_only > 0.5)
        return float4(0.025, 0.028, 0.026, 1.0);
    if (input.panel > 0.5 && input.surface_kind > 9.5 && input.surface_kind < 10.5)
    {
        float alpha = (1.0 - input.shape_visibility.x) *
                      frame_cast_shadow_strength() * 0.74;
        clip(alpha - 0.004);
        return float4(0.018, 0.022, 0.030, alpha);
    }
    if (input.panel > 0.5 && input.surface_kind > 11.5 && input.surface_kind < 12.5)
    {
        // Animated unit bodies cast their actual projected source triangles.
        // Five low-opacity light-facing projections form one restrained soft
        // penumbra while limbs, weapons, turrets, mounts, and attachments
        // retain their source silhouette.
        float alpha = frame_cast_shadow_strength() * 0.16;
        clip(alpha - 0.004);
        return float4(0.010, 0.014, 0.019, alpha);
    }
    if (input.panel > 0.5 && input.surface_kind > 13.5 && input.surface_kind < 14.5)
    {
        // Static raised map objects use the same projected source-mesh method
        // and shared light vector as units. A single silhouette sample avoids
        // multiplying the large city mesh budget; blend coverage supplies the
        // restrained soft edge.
        float alpha = frame_cast_shadow_strength() * 0.46;
        clip(alpha - 0.004);
        return float4(0.010, 0.014, 0.019, alpha);
    }
    if (input.panel > 0.5 && input.surface_kind > 7.5 && input.surface_kind < 8.5)
        return float4(0.09, 0.13, 0.30, biq_layout > 0.5 ? 0.20 : 0.58);
    if (rivers_only > 0.5 &&
        !(input.panel > 0.5 && input.surface_kind > 8.5 && input.surface_kind < 9.5))
        return float4(0.025, 0.028, 0.026, 1.0);
    if (marsh_only > 0.5 && abs(input.real_terrain - 9.0) > 0.25)
        return float4(0.025, 0.028, 0.026, 1.0);
    if (volcano_only > 0.5 && !q4_volcano_surface(input))
        return float4(0.025, 0.028, 0.026, 1.0);
    if (input.panel > 0.5 && input.surface_kind > 6.5 && input.surface_kind < 7.5)
    {
        float along_fade = 1.0 - smoothstep(0.30, 1.0, input.uv.y);
        float tail_taper = lerp(1.0, 0.45,
            smoothstep(0.20, 1.0, input.uv.y));
        float lateral = abs(input.uv.x - 0.5) * 2.0 / tail_taper;
        float across_fade = 1.0 - smoothstep(0.12, 1.0, lateral);
        float alpha = biq_layout > 0.5
            ? (l13a_layout > 0.5
                ? frame_cast_shadow_strength() * 0.70 * along_fade * across_fade
                : 0.0)
            : 0.14;
        return float4(0.006, 0.010, 0.010, alpha);
    }
    if (dune_only > 0.5 && input.surface_kind > 3.5 && input.surface_kind < 6.5)
        return float4(0.025, 0.028, 0.026, 1.0);
    if (input.panel < 0.5 && lab_mode > 5.5)
    {
        float3 coast_source = input.panel < -0.5
            ? cliff_base_texture.Sample(material_sampler, input.uv).rgb
            : beach_base_texture.Sample(material_sampler, input.uv).rgb;
        return float4((coast_source), 1.0);
    }
    if (input.panel < 0.5 && lab_mode > 3.5)
    {
        if (lab_mode > 4.5 && input.panel < -1.5)
        {
            float region = authored_region_texture.Sample(material_sampler, input.uv).r;
            float3 false_color = region < 0.1 ? float3(0.02, 0.02, 0.02)
                : (region < 0.48 ? float3(0.92, 0.28, 0.18)
                : (region < 0.55 ? float3(0.20, 0.78, 0.32) : float3(0.24, 0.46, 0.96)));
            return float4(false_color, 1.0);
        }
        float source_value = lab_mode > 4.5 && input.panel < -0.5
            ? authored_blend_texture.Sample(material_sampler, input.uv).r
            : authored_height_texture.Sample(material_sampler, input.uv).r;
        return float4(source_value, source_value, source_value, 1.0);
    }
    if (input.panel > 0.5 && input.surface_kind > 8.5 && input.surface_kind < 9.5)
    {
        if (rivers_enabled < 0.5)
            clip(-1.0);
        float distance_pixels = input.river_data.x;
        float junction_weight = 1.0 - smoothstep(3.0, 18.0, input.river_data.y);
        float mouth_weight = 1.0 - smoothstep(2.0, 24.0, input.river_data.z);
        float source_weight = 1.0 - smoothstep(3.0, 22.0, input.river_data.w);
        float2 world_position = input.macro_uv * 2.0;
        float bank_noise = river_bank_noise_texture.Sample(
            material_sampler, world_position * float2(0.31, 0.47) +
            float2(0.19, 0.37)).r - 0.5;
        float bank_width = 10.2 + bank_noise * 2.6 + mouth_weight * 6.0 +
                           junction_weight * 2.0 + source_weight * 1.2;
        clip(bank_width - distance_pixels);

        float2 bed_uv = world_position * float2(0.38, 0.52) + float2(0.17, 0.31);
        float3 bed = river_base_texture.Sample(material_sampler, bed_uv).rgb;
        float bed_height = river_height_texture.Sample(material_sampler, bed_uv).r;
        float bed_specular = river_specular_texture.Sample(material_sampler, bed_uv).r;
        bed *= 0.88 + (bed_height - 0.5) * 0.34;

        float4 source = river_source_base_texture.Sample(
            decal_sampler, frac(world_position));
        float source_height = river_source_height_texture.Sample(
            decal_sampler, frac(world_position)).r;
        bed = lerp(bed, source.rgb, source.a * source_weight * 0.52);
        bed *= 0.94 + (source_height - 0.5) * source_weight * 0.18;

        float4 clutter = river_clutter_base_texture.Sample(
            decal_sampler, frac(world_position * 0.50 + float2(0.23, 0.41)));
        float clutter_edge = smoothstep(3.2, 7.4, distance_pixels);
        bed = lerp(bed, clutter.rgb, clutter.a * clutter_edge * 0.34);

        float water_width = 6.4 + bank_noise * 1.1 + mouth_weight * 5.0 +
                            junction_weight * 1.4 + source_weight * 0.8;
        float water_weight = 1.0 - smoothstep(water_width - 1.8, water_width, distance_pixels);
        float2 lean = river_lean0_texture.Sample(
            material_sampler, world_position * float2(0.92, 1.27)).rg * 2.0 - 1.0;
        float2 variance = river_lean1_texture.Sample(
            material_sampler, world_position * float2(1.31, 0.83) + float2(0.37, 0.19)).rg;
        // The alternate-skin reference treats the river as subdued gray-blue
        // water over a visible source-backed bed, not a saturated cyan stripe.
        float shallow_edge = smoothstep(0.0, water_width, distance_pixels);
        float3 water = float3(0.025, 0.080, 0.115) + bed *
            (0.045 + shallow_edge * 0.075 + dot(lean, float2(0.008, -0.006)));
        float3 river_normal = normalize(float3(-lean.x * 0.34, -lean.y * 0.34, 1.0));
        float3 light_direction = frame_light_direction();
        float3 view_direction = normalize(float3(0.0, -0.52, 0.86));
        float3 half_direction = normalize(light_direction + view_direction);
        float glint = pow(saturate(dot(river_normal, half_direction)), 96.0) *
            bed_specular * rcp(1.0 + dot(variance, float2(2.0, 2.0))) * 0.12 *
            (l13a_layout > 0.5 ? environment_water_specular : 1.0);
        float3 glint_color = l13a_layout > 0.5
            ? saturate(environment_sun_color * environment_sun_intensity +
                       environment_moon_color * environment_moon_intensity)
            : 1.0.xxx;
        water += glint * glint_color;

        float bank_fade = 1.0 - smoothstep(bank_width - 3.8, bank_width, distance_pixels);
        float3 river = lerp(bed, water, water_weight);
        float3 river_geometry_normal = normalize(input.geometry_normal);
        float diffuse = saturate(dot(river_geometry_normal, light_direction));
        float3 river_light = l13a_layout > 0.5
            ? q6_receiver_illumination(input, river_geometry_normal, 1.0, 1.0)
            : (0.78 + diffuse * 0.36).xxx;
        river *= river_light;
        float river_alpha = lerp(0.30, 0.97, water_weight) * bank_fade;
        return float4((frame_tone_map(
                          river * frame_output_exposure())),
                      river_alpha);
    }
    if (input.panel > 0.5 && input.surface_kind > 5.5 && input.surface_kind < 6.5)
    {
        float across = saturate(input.shore_distance * 1.45);
        if (shoreline_integrated < 0.5)
        {
            float2 legacy_uv = input.macro_uv * float2(1.4, 3.2) + float2(0.0, across * 0.18);
            float legacy_noise = water_foam_texture.Sample(material_sampler, legacy_uv).r;
            float legacy_broken = 0.24 + 0.76 * smoothstep(0.28, 0.76, legacy_noise);
            float legacy_crest = smoothstep(0.04, 0.20, across) *
                                 (1.0 - smoothstep(0.34, 0.92, across));
            float legacy_wash = (1.0 - smoothstep(0.0, 0.72, across)) * 0.14;
            float legacy_alpha = legacy_crest * legacy_broken * 0.38 + legacy_wash;
            return float4((float3(0.82, 0.87, 0.90)), legacy_alpha);
        }
        float2 foam_uv = input.macro_uv * float2(1.7, 3.6) + float2(across * 0.11, 0.0);
        float crash = water_foam_texture.Sample(material_sampler, foam_uv).a;
        float ripple = water_ripples_texture.Sample(material_sampler,
            input.macro_uv * float2(3.4, 5.1) + float2(0.17, 0.31)).a;
        float turbulence = water_turbulence_texture.Sample(material_sampler,
            input.macro_uv * float2(1.2, 2.4) + float2(0.43, 0.09)).a;
        float breakup = saturate(crash * 0.72 + ripple * 0.54 + turbulence * 0.48);
        float leading_crest = smoothstep(0.03, 0.16, across) *
                              (1.0 - smoothstep(0.38, 0.72, across));
        float wash = (1.0 - smoothstep(0.0, 0.92, across)) *
                     (0.055 + ripple * 0.11);
        float foam_alpha = leading_crest *
                           lerp(0.16, 0.62, smoothstep(0.05, 0.46, breakup)) + wash;
        float3 foam = float3(0.82, 0.88, 0.90);
        return float4((foam), foam_alpha);
    }
    if (input.panel > 0.5 && input.surface_kind > 4.5 && input.surface_kind < 5.5)
    {
        if (biq_layout > 0.5)
            clip(input.shore_distance - 0.001);
        float depth = saturate(input.surface_coordinate);
        float2 large_uv = input.macro_uv * float2(0.72, 0.94);
        float2 small_uv = input.macro_uv * float2(4.8, 6.1) + float2(0.31, 0.17);
        float2 large_lean = water_large_lean0_texture.Sample(material_sampler, large_uv).rg * 2.0 - 1.0;
        float2 small_lean = water_small_lean0_texture.Sample(material_sampler, small_uv).rg * 2.0 - 1.0;
        if (shoreline_integrated < 0.5)
        {
            float2 legacy_variance = water_large_lean1_texture.Sample(material_sampler, large_uv).rg +
                                     water_small_lean1_texture.Sample(material_sampler, small_uv).rg;
            float2 legacy_lean = large_lean * 0.72 + small_lean * 0.28;
            float3 legacy_normal = normalize(float3(-legacy_lean.x, -legacy_lean.y, 1.0));
            float3 legacy_view = normalize(float3(0.0, -0.52, 0.86));
            float legacy_fresnel = 0.05 + 0.36 *
                pow(1.0 - saturate(dot(legacy_normal, legacy_view)), 5.0);
            float3 legacy_light = normalize(float3(-0.55, -0.35, 0.22));
            float3 legacy_half = normalize(legacy_light + legacy_view);
            float legacy_roughness = rcp(1.0 + dot(legacy_variance, float2(2.0, 2.0)));
            float legacy_glint = pow(saturate(dot(legacy_normal, legacy_half)), 72.0) *
                                 0.34 * legacy_roughness;
            float3 legacy_water = lerp(float3(0.055, 0.24, 0.29),
                float3(0.008, 0.055, 0.15), smoothstep(0.12, 0.92, depth));
            legacy_water *= 0.96 + dot(legacy_lean, float2(0.035, -0.025));
            legacy_water += float3(0.18, 0.30, 0.34) * legacy_fresnel + legacy_glint;
            float legacy_alpha = lerp(0.34, 0.76, smoothstep(0.05, 0.95, depth));
            return float4((legacy_water), legacy_alpha);
        }
        float tiling_mask = water_tiling_mask_texture.Sample(material_sampler, input.macro_uv * 1.9).r;
        float non_tiling_mask = water_non_tiling_mask_texture.Sample(
            material_sampler, input.macro_uv * 0.55 + float2(0.21, 0.37)).r;
        float2 tiling_normal = water_tiling_normal0_texture.Sample(
            material_sampler, input.macro_uv * float2(5.2, 6.0)).rg * 2.0 - 1.0;
        float tiling_normal_variance = water_tiling_normal1_texture.Sample(
            material_sampler, input.macro_uv * float2(5.2, 6.0)).r;
        float2 non_tiling_normal = water_non_tiling_normal0_texture.Sample(
            material_sampler, input.macro_uv * float2(1.15, 1.35) + float2(0.19, 0.41)).rg *
            2.0 - 1.0;
        float non_tiling_normal_variance = water_non_tiling_normal1_texture.Sample(
            material_sampler, input.macro_uv * float2(1.15, 1.35) + float2(0.19, 0.41)).r;
        float secondary_weight = saturate(tiling_mask * 0.58 + non_tiling_mask * 0.42);
        float2 secondary_lean = water_small_secondary_lean0_texture.Sample(
            material_sampler, input.macro_uv * float2(7.1, 8.3) + float2(0.57, 0.23)).rg * 2.0 - 1.0;
        float2 lean_variance = water_large_lean1_texture.Sample(material_sampler, large_uv).rg +
                               water_small_lean1_texture.Sample(material_sampler, small_uv).rg +
                               water_small_secondary_lean1_texture.Sample(
                                   material_sampler, input.macro_uv * float2(7.1, 8.3) +
                                   float2(0.57, 0.23)).rg * 0.35;
        lean_variance += float2(tiling_normal_variance, non_tiling_normal_variance) * 0.18;
        float height_left = water_height_texture.Sample(
            material_sampler, input.uv - float2(1.0 / 256.0, 0.0)).r;
        float height_right = water_height_texture.Sample(
            material_sampler, input.uv + float2(1.0 / 256.0, 0.0)).r;
        float height_down = water_height_texture.Sample(
            material_sampler, input.uv - float2(0.0, 1.0 / 256.0)).r;
        float height_up = water_height_texture.Sample(
            material_sampler, input.uv + float2(0.0, 1.0 / 256.0)).r;
        float2 shallow_height_normal = float2(height_left - height_right, height_down - height_up);
        float2 combined_lean = large_lean * 0.64 + small_lean * 0.24 +
                               secondary_lean * secondary_weight * 0.12 +
                               tiling_normal * tiling_mask * 0.30 +
                               non_tiling_normal * non_tiling_mask * 0.22 +
                               shallow_height_normal * (1.0 - depth) * 0.20;
        float2 refracted_uv = input.uv + float2(-0.08, -0.08) +
                              combined_lean * (0.026 * (1.0 - depth));
        float refracted_ocean_mix = smoothstep(0.58, 1.0, depth);
        float3 refracted_bed = lerp(
            shallow_bed_texture.Sample(material_sampler, refracted_uv).rgb,
            ocean_bed_texture.Sample(material_sampler, refracted_uv).rgb,
            refracted_ocean_mix);
        if (biq_layout > 0.5)
        {
            float2 water_world = input.macro_uv * 2.0;
            float4 water_clutter = combined_water_clutter(
                water_world, input.shore_distance);
            refracted_bed = lerp(refracted_bed, water_clutter.rgb,
                                 water_clutter.a * 0.58);
            float clutter_height = sample_water_clutter_height(water_world);
            refracted_bed *= 1.0 + (clutter_height - 0.5) *
                                     water_clutter.a * 0.24;
        }
        float refracted_height = lerp(
            water_height_texture.Sample(material_sampler, refracted_uv).r,
            ocean_height_texture.Sample(material_sampler, refracted_uv).r,
            refracted_ocean_mix);
        refracted_bed *= 0.84 + (refracted_height - 0.426) * 1.55;
        float3 water_normal = normalize(float3(-combined_lean.x, -combined_lean.y, 1.0));
        float3 view_direction = normalize(float3(0.0, -0.52, 0.86));
        float fresnel = 0.001 + 0.999 *
            pow(1.0 - saturate(dot(water_normal, view_direction)), 4.0);
        float3 light_direction = frame_light_direction();
        float3 half_direction = normalize(light_direction + view_direction);
        float gloss = water_gloss_texture.Sample(material_sampler, input.macro_uv * 2.0).r;
        float roughness_attenuation = rcp(1.0 + dot(lean_variance, float2(2.0, 2.0)));
        float sun_glint = pow(saturate(dot(water_normal, half_direction)), 850.0) *
                          (0.18 + gloss * 0.22) * roughness_attenuation *
                          (l13a_layout > 0.5 ? environment_water_specular : 1.0);
#ifdef Q6_WORLD_SHADOWS
        sun_glint *= q6_receiver_visibility(input, water_normal, 1.0);
#endif
        // The density ramps operate over optical depth, not the entire coast
        // mesh. Compress the geometric distance so the near-shore portion
        // retains the long translucent shelf visible in the authored coast.
        float profile_coordinate = depth * 0.42;
        float4 dark_density = coast_dark_profile_texture.Sample(
            material_sampler, float2(profile_coordinate, 0.5));
        float4 scatter_density = coast_scatter_profile_texture.Sample(
            material_sampler, float2(profile_coordinate, 0.5));
        float transmission = pow(saturate(dark_density.g), 0.70);
        float3 density_water = lerp(float3(0.010, 0.040, 0.050),
                                    float3(0.12, 0.20, 0.20), transmission);
        density_water += scatter_density.rgb * float3(0.25, 0.30, 0.25);
        float3 depth_tint = lerp(float3(0.006, 0.135, 0.195),
                                 float3(0.001, 0.050, 0.145),
                                 smoothstep(0.10, 0.86, depth));
        float3 water = lerp(depth_tint, density_water, 0.05);
        float bed_visibility = smoothstep(0.09, 0.20, depth) *
                               (1.0 - smoothstep(0.62, 0.90, depth)) * 0.42;
        float3 submerged_color = refracted_bed * lerp(0.68, 0.30, depth);
        water = lerp(water, submerged_color, bed_visibility);
        float surface_light = saturate(dot(
            water_normal, normalize(float3(-0.24, -0.32, 0.92))));
        float overlay_ripple = dot(tiling_normal * tiling_mask,
                                   float2(0.18, -0.14)) +
                               dot(non_tiling_normal * non_tiling_mask,
                                   float2(-0.14, 0.17));
        water *= 0.70 + surface_light * 0.34 +
                 dot(combined_lean, float2(0.095, -0.070)) + overlay_ripple;
        float3 reflection_color = l13a_layout > 0.5
            ? lerp(environment_sun_color, environment_moon_color,
                   environment_night_activation)
            : float3(0.16, 0.27, 0.33);
        float reflection_scale = l13a_layout > 0.5
            ? environment_water_fresnel * 2.4 : 1.0;
        float3 glint_color = l13a_layout > 0.5
            ? saturate(environment_sun_color * environment_sun_intensity +
                       environment_moon_color * environment_moon_intensity)
            : 1.0.xxx;
        water += reflection_color * fresnel * reflection_scale +
                 sun_glint * glint_color;
        if (l13a_layout > 0.5)
            water *= q6_receiver_illumination(input, water_normal, 1.0, 1.0) * 0.78;
        if (biq_layout > 0.5)
        {
            // Preserve the water body and let the submerged source art read
            // through it as contour/rock contrast. Directly replacing this
            // color with the sandy decal albedo was the rejected beige-water
            // experiment; real submersion keeps the water's optical tint.
            float2 water_world = input.macro_uv * 2.0;
            float4 water_clutter = combined_water_clutter(
                water_world, input.shore_distance);
            float clutter_luminance = dot(water_clutter.rgb,
                float3(0.2126, 0.7152, 0.0722));
            float contour = saturate(0.72 + clutter_luminance * 0.90);
            float clutter_visibility = saturate(water_clutter.a * 2.0) *
                (1.0 - smoothstep(0.78, 0.99, depth)) * 0.26;
            water = lerp(water, water * contour, clutter_visibility);
        }
        float alpha = lerp(0.22, 0.97, smoothstep(0.01, 0.38, depth));
        if (biq_layout > 0.5)
        {
            // BIQ water is a whole-viewport surface field. Composite the real
            // crash/ripple/turbulence foam into that surface instead of adding
            // per-tile foam geometry, which reintroduced hairline cell seams.
            float across = saturate(input.shore_distance * 1.45);
            float2 foam_uv = input.macro_uv * float2(1.7, 3.6) +
                             float2(across * 0.11, 0.0);
            float crash = water_foam_texture.Sample(material_sampler, foam_uv).a;
            float ripple = water_ripples_texture.Sample(material_sampler,
                input.macro_uv * float2(3.4, 5.1) + float2(0.17, 0.31)).a;
            float turbulence = water_turbulence_texture.Sample(material_sampler,
                input.macro_uv * float2(1.2, 2.4) + float2(0.43, 0.09)).a;
            float breakup = saturate(crash * 0.72 + ripple * 0.54 + turbulence * 0.48);
            float crest = smoothstep(0.03, 0.16, across) *
                          (1.0 - smoothstep(0.38, 0.72, across));
            float wash = (1.0 - smoothstep(0.0, 0.92, across)) *
                         (0.022 + ripple * 0.052);
            float foam_alpha = crest *
                lerp(0.08, 0.36, smoothstep(0.05, 0.46, breakup)) + wash;
            float3 foam_color = float3(0.82, 0.88, 0.90);
            if (l13a_layout > 0.5)
                foam_color *= q6_receiver_illumination(input, float3(0.0, 0.0, 1.0), 1.0, 1.0) * 0.78;
            water = lerp(water, foam_color, foam_alpha);
            alpha = max(alpha, foam_alpha * 0.86);
        }
        return float4((frame_tone_map(
                          water * frame_output_exposure())),
                      alpha);
    }
    if (input.panel > 0.5 && input.surface_kind > 3.5 && input.surface_kind < 4.5)
    {
        if (biq_layout > 0.5)
            clip(input.shore_distance - 0.001);
        float depth = saturate(input.surface_coordinate);
        float2 bed_uv = input.uv +
                        (shoreline_integrated > 0.5 ? float2(-0.08, -0.08) : 0.0);
        float3 shallow_bed = shallow_bed_texture.Sample(material_sampler, bed_uv).rgb;
        float3 ocean_bed = ocean_bed_texture.Sample(material_sampler, bed_uv).rgb;
        if (shoreline_integrated < 0.5)
        {
            float3 legacy_bed = lerp(shallow_bed, ocean_bed, smoothstep(0.20, 0.88, depth));
            legacy_bed *= lerp(1.0, 0.38, depth);
            return float4((legacy_bed), 1.0);
        }
        float ocean_mix = smoothstep(0.58, 1.0, depth);
        float3 bed = lerp(shallow_bed, ocean_bed, ocean_mix);
        if (biq_layout > 0.5)
        {
            float2 water_world = input.macro_uv * 2.0;
            float4 water_clutter = combined_water_clutter(
                water_world, input.shore_distance);
            bed = lerp(bed, water_clutter.rgb, water_clutter.a * 0.64);
            float clutter_height = sample_water_clutter_height(water_world);
            bed *= 1.0 + (clutter_height - 0.5) *
                           water_clutter.a * 0.28;
        }
        float shallow_specular = shallows_specular_texture.Sample(material_sampler, bed_uv).r;
        float ocean_specular = ocean_specular_texture.Sample(material_sampler, bed_uv).r;
        float material_specular = lerp(shallow_specular, ocean_specular, ocean_mix);
        float shallow_height = water_height_texture.Sample(material_sampler, bed_uv).r;
        float ocean_height = ocean_height_texture.Sample(material_sampler, bed_uv).r;
        float beach_height = beach_height_texture.Sample(
            material_sampler, input.macro_uv * float2(2.7, 3.1)).r;
        float bed_detail = lerp(shallow_height, ocean_height, ocean_mix) - 0.426;
        float submerged_beach = 1.0 - smoothstep(
            0.02, 0.16, depth + (beach_height - 0.5) * 0.14);
        float3 beach_bed = beach_base_texture.Sample(
            material_sampler, input.macro_uv * float2(2.7, 3.1)).rgb;
        bed = lerp(bed, beach_bed,
                   submerged_beach * (biq_layout > 0.5 ? 0.92 : 0.68));
        bed *= lerp(1.0, 0.52, smoothstep(0.24, 0.68, depth));
        bed *= 0.98 + bed_detail * 1.35;
        bed += material_specular * 0.018;
        if (l13a_layout > 0.5)
            bed *= q6_receiver_illumination(input, float3(0.0, 0.0, 1.0), 1.0, 1.0) * 0.78;
        return float4((frame_tone_map(
                          bed * frame_output_exposure())),
                      1.0);
    }
    float2 world_position = input.macro_uv * 2.0;
    float dune_weight = dune_region_weight(world_position);
#ifdef Q4_BIQ_DUNE_COVERAGE
    // The old four-tile gallery rectangle is not a gameplay material selector.
    // The shared continuous desert weight preserves soft biome transitions.
    if(biq_layout>.5) dune_weight=saturate(input.material_weights.z);
#endif
    if (biq_layout > 0.5 && (input.base_terrain > 0.5 || input.real_terrain > 0.5))
        dune_weight = 0.0;
#ifdef Q4_BIQ_CONTINUOUS_DESERT
    if(biq_layout>.5) dune_weight=saturate(input.material_weights.z);
#endif
#ifdef Q2_SOURCE_ALPHA_BLEND
    if(biq_layout>.5) q2_source_blend(input);
#endif
    float3 albedo = base_color_texture.Sample(material_sampler, input.uv).rgb;
    if (biq_layout > 0.5)
    {
        float4 weights = input.material_weights;
        float tundra_weight = input.material_tundra;
        if (marsh_enabled < 0.5)
        {
            weights.x += weights.w;
            weights.w = 0.0;
        }
        float material_total = max(0.001, dot(weights, 1.0) + tundra_weight);
        weights /= material_total;
        tundra_weight /= material_total;
        float3 grass = base_color_texture.Sample(material_sampler, input.uv).rgb;
        float3 plains = plains_base_texture.Sample(material_sampler, input.uv).rgb;
        float3 desert = desert_base_texture.Sample(material_sampler, input.uv).rgb;
        float3 marsh = marsh_base_texture.Sample(material_sampler, input.uv).rgb;
        float3 tundra = feature_base_texture_4.Sample(material_sampler, input.uv).rgb;
        albedo = grass * weights.x + plains * weights.y + desert * weights.z +
                 marsh * weights.w + tundra * tundra_weight;
#ifdef Q2_CONTINENTAL_MATERIAL
        float2 high_mix=q2_continental_mix(input);
        albedo+=high_mix.x*(q2_grass_high_color.Sample(material_sampler,input.uv).rgb-grass)
            +high_mix.y*(q2_plains_high_color.Sample(material_sampler,input.uv).rgb-plains);
#endif

        float grass_scale = abs(input.real_terrain - 5.0) < 0.25 ? 3.85 : 4.65;
        float4 grass_clutter = sample_land_clutter(
            grassland_decal_base_texture, world_position, grass_scale,
            float2(0.13, 0.37));
        float4 plains_clutter = sample_land_clutter(
            plains_decal_base_texture, world_position, 4.25,
            float2(0.47, 0.11));
        float grass_clutter_weight = weights.x * grass_clutter.a *
            (abs(input.real_terrain - 6.0) < 0.25 ||
             q4_volcano_surface(input) ? 0.18 : 0.46);
        float plains_clutter_weight = weights.y * plains_clutter.a * 0.42;
#ifdef Q2_GROUND_DECAL_SOURCE_ALPHA
        // The source patch alpha already encodes coverage. The legacy full-
        // atlas attenuation suppressed its rock/earth detail a second time.
        grass_clutter_weight = weights.x * grass_clutter.a *
            (abs(input.real_terrain - 6.0) < .25 || q4_volcano_surface(input) ? .18 : 1.0);
        plains_clutter_weight = weights.y * plains_clutter.a;
#endif
        albedo = lerp(albedo, grass_clutter.rgb, grass_clutter_weight);
        albedo = lerp(albedo, plains_clutter.rgb, plains_clutter_weight);
        if (weights.w > 0.001 && marsh_enabled > 0.5)
        {
            float4 marsh_decal = combined_marsh_decal(world_position);
            marsh = lerp(marsh, marsh_decal.rgb, marsh_decal.a * 0.88);
            albedo = lerp(albedo, marsh, weights.w);
        }
        if (dune_weight > 0.0)
        {
            float3 desert_hills = desert_hills_base_texture.Sample(material_sampler, input.uv).rgb;
            float4 dune_decal = combined_dune_decal(world_position);
            float3 dune_color = lerp(albedo, desert_hills, 0.86);
            dune_color = lerp(dune_color, dune_decal.rgb, dune_decal.a * 0.42);
            albedo = lerp(albedo, dune_color, dune_weight);
        }
        if (volcano_enabled > 0.5 && q4_volcano_surface(input))
        {
            float fixture_active = fmod(floor(world_position.x) * 17.0 +
                                        floor(world_position.y) * 31.0, 2.0) < 1.0 ? 1.0 : 0.0;
            float active = input.active_effect >= 0.0
                ? input.active_effect : q4_volcano_active(input,fixture_active);
            float2 volcano_uv = frac(world_position);
#ifdef Q4_VOLCANO_SOURCE_MAPPING
            // Same local-v orientation and uniform footprint as the height
            // sampler. All dormant/active/slope/specular channels agree.
#ifndef Q4_VOLCANO_FOOTPRINT
#define Q4_VOLCANO_FOOTPRINT .62
#endif
            volcano_uv=.5+(float2(volcano_uv.x,1-volcano_uv.y)-.5)*Q4_VOLCANO_FOOTPRINT;
#endif
#ifdef Q4_RELIEF_MATERIAL_DATA
            volcano_uv=input.relief_material.xy;
#endif
            float3 dormant = volcano_base_texture.Sample(material_sampler, volcano_uv).rgb;
            // ActiveBase contains a nearly black field around its bright lava
            // detail. Treating it as the complete rock albedo made active
            // volcanoes nearly black, so retain the dormant authored body and
            // composite the active channel below.
            float4 active_surface = volcano_active_base_texture.Sample(
                material_sampler, volcano_uv);
            float active_luminance = max(active_surface.r,
                                     max(active_surface.g, active_surface.b));
            float lava_mask = smoothstep(0.16, 0.52, active_luminance);
            // Keep the authored dormant rock as the visible body. The active
            // map contributes restrained surface variation and its own bright
            // lava texels; its nearly black background is not a replacement
            // albedo for the whole volcano.
            float active_mix = active * active_surface.a *
                               lerp(0.12, 1.0, lava_mask);
            float3 volcano_color = lerp(dormant, active_surface.rgb, active_mix);
            albedo = lerp(albedo, volcano_color,
                          q4_volcano_coverage(input));
        }
    }
    else if (promotion_tile_layout > 0.5)
    {
        float3 material_weights = promotion_material_weights(input.macro_uv * 2.0);
        float3 plains = plains_base_texture.Sample(material_sampler, input.uv).rgb;
        float3 desert = desert_base_texture.Sample(material_sampler, input.uv).rgb;
        albedo = albedo * material_weights.x + plains * material_weights.y +
                 desert * material_weights.z;
        if (dune_weight > 0.0)
        {
            float3 desert_hills = desert_hills_base_texture.Sample(material_sampler, input.uv).rgb;
            float4 dune_decal = combined_dune_decal(world_position);
            float3 dune_color = lerp(desert, desert_hills, 0.86);
            dune_color = lerp(dune_color, dune_decal.rgb,
                              dune_decal.a * 0.42);
            albedo = lerp(albedo, dune_color, dune_weight);
        }
    }
    if (input.surface_kind > 2.5)
        albedo = cliff_base_texture.Sample(material_sampler, input.uv).rgb;
    else if (input.surface_kind > 1.5)
    {
        float3 beach = beach_base_texture.Sample(material_sampler, input.uv).rgb;
        if (shoreline_integrated > 0.5)
        {
            float3 grass = albedo;
            float transition_noise0 = beach_height_texture.Sample(
                material_sampler, input.macro_uv * float2(3.2, 4.1)).r - 0.426;
            float transition_noise1 = beach_height_texture.Sample(
                material_sampler, input.macro_uv * float2(7.3, 8.7) + float2(0.31, 0.17)).r - 0.426;
            float grass_detail = height_texture.Sample(
                material_sampler, input.macro_uv * float2(5.9, 6.7) + float2(0.11, 0.43)).r - 0.421;
            float transition_noise = transition_noise0 * 2.4 + transition_noise1 * 1.2 +
                                     grass_detail * 1.6;
            float signed_shore = input.surface_coordinate;
            float sand_mix = smoothstep(-0.42, 0.68, signed_shore + transition_noise);
            float wet_edge = smoothstep(0.66, 1.04, signed_shore + transition_noise * 0.55);
            float3 shallows = shallow_bed_texture.Sample(material_sampler, input.uv).rgb;
            float3 dry_beach = lerp(beach, shallows, 0.18);
            albedo = lerp(grass, dry_beach, sand_mix);
            albedo = lerp(albedo, shallows,
                          wet_edge * 0.30);
        }
        else
            albedo = beach;
    }
#ifndef Q3_SHORE_MATERIAL
    else if (shoreline_integrated > 0.5)
    {
        float transition_noise0 = beach_height_texture.Sample(
            material_sampler, input.macro_uv * float2(3.2, 4.1)).r - 0.426;
        float transition_noise1 = beach_height_texture.Sample(
            material_sampler, input.macro_uv * float2(7.3, 8.7) + float2(0.31, 0.17)).r - 0.426;
        float grass_detail = height_texture.Sample(
            material_sampler, input.macro_uv * float2(5.9, 6.7) + float2(0.11, 0.43)).r - 0.421;
        float transition_noise = transition_noise0 * 2.4 + transition_noise1 * 1.2 +
                                 grass_detail * 1.6;
        float signed_shore = biq_layout > 0.5
            ? input.shore_distance
            : -input.surface_coordinate * 0.78;
        float sand_mix = biq_layout > 0.5
            ? smoothstep(-0.58, 0.10, signed_shore + transition_noise * 0.12)
            : smoothstep(-0.42, 0.68, signed_shore + transition_noise);
        float3 beach = beach_base_texture.Sample(material_sampler, input.uv).rgb;
        float3 shallows = shallow_bed_texture.Sample(material_sampler, input.uv).rgb;
        float3 dry_beach = biq_layout > 0.5
            ? lerp(albedo, lerp(beach, shallows, 0.08), 0.76)
            : lerp(beach, shallows, 0.18);
        albedo = lerp(albedo, dry_beach, sand_mix);
        if (biq_layout > 0.5)
        {
            float wet_mix = smoothstep(-0.04, 0.18,
                signed_shore + transition_noise * 0.04);
            albedo = lerp(albedo, shallows, wet_mix * 0.22);
        }
    }
#endif
    float mountain_mask = 0.0;
    bool desert_mountain = false;
    bool mountain_surface = input.panel > 0.5 && input.surface_kind < 1.5 &&
        ((lab_mode > 4.5 && lab_mode < 5.5) ||
         (lab_mode > 7.5 && beauty_relief_enabled > 0.5));
    if (mountain_surface)
    {
        float2 authored_uv = input.macro_uv;
        float promotion_mountain_envelope = 1.0;
        if (promotion_tile_layout > 0.5)
        {
            bool lower_mountain = false;
            bool upper_mountain = false;
            float2 local_position = frac(world_position);
            if (biq_layout > 0.5)
#ifdef Q4_DIAGNOSTIC_DESERT_MOUNTAINS
                desert_mountain = true; // Explicit synthetic material-branch witness only.
#else
                desert_mountain = input.base_terrain < 0.5;
#endif
            else
            {
                lower_mountain = world_position.x >= 3.0 && world_position.x <= 4.0 &&
                                 world_position.y >= 1.0 && world_position.y <= 2.0;
                upper_mountain = world_position.x >= 4.0 && world_position.x <= 5.0 &&
                                 world_position.y >= 0.0 && world_position.y <= 1.0;
                desert_mountain = l10_layout > 0.5 &&
                                  world_position.x >= 8.0 && world_position.x <= 9.0 &&
                                  world_position.y >= 6.0 && world_position.y <= 7.0;
                local_position = lower_mountain
                    ? float2(world_position.x - 3.0, world_position.y - 1.0)
                    : (upper_mountain
                    ? float2(world_position.x - 4.0, world_position.y)
                    : float2(world_position.x - 8.0, world_position.y - 6.0));
                if (upper_mountain)
                    local_position.x = 1.0 - local_position.x;
                if (desert_mountain)
                    local_position.y = 1.0 - local_position.y;
            }
            if (biq_layout > 0.5)
            {
                authored_uv = local_position;
                if (input.real_terrain < 4.5 || input.real_terrain > 6.5)
                    promotion_mountain_envelope = 0.0;
            }
            else
            {
                authored_uv = float2(0.25, 0.25) + local_position * 0.50;
                float edge_distance = min(min(local_position.x, 1.0 - local_position.x),
                                          min(local_position.y, 1.0 - local_position.y));
                promotion_mountain_envelope = smoothstep(0.0, 0.10, edge_distance);
                if (!lower_mountain && !upper_mountain && !desert_mountain)
                    promotion_mountain_envelope = 0.0;
            }
        }
#ifdef Q4_BROAD_RELIEF
        if(biq_layout>.5 && abs(input.real_terrain-10)>.25)
            promotion_mountain_envelope=1; // CPU source support crosses tile ownership.
#endif
        float authored_height = biq_layout > 0.5
            ? input.authored_relief.x
            : authored_height_texture.Sample(material_sampler, authored_uv).r;
        float authored_blend = biq_layout > 0.5
            ? input.authored_relief.y
            : authored_blend_texture.Sample(material_sampler, authored_uv).r;
        // The authored blend owns the broad terrain-integrated footprint. A
        // small height floor removes only empty source pixels; multiplying the
        // two channels before thresholding erased the low shoulders and was
        // the reason the previous result read as a pasted cone.
        mountain_mask = smoothstep(0.01, 0.10, authored_height) *
                        smoothstep(0.02, 0.62, authored_blend);
        mountain_mask *= promotion_mountain_envelope;
        float3 mountain_base = desert_mountain
            ? q4_relief_color(desert_mountain_base_texture, input)
            : q4_relief_color(mountain_base_texture, input);
        float3 mountain_top = desert_mountain
            ? q4_relief_color(desert_mountain_stripe1_texture, input)
            : q4_relief_color(mountain_top_texture, input);
        float3 mountain_snow = desert_mountain
            ? q4_relief_color(desert_mountain_stripe3_texture, input)
            : q4_relief_color(mountain_snow_texture, input);
        float upper_rock = smoothstep(0.22, 0.62, authored_height);
        float snow = smoothstep(0.70, 0.84, authored_height) *
                     smoothstep(0.18, 0.72, input.geometry_normal.z);
        if (lab_mode > 7.5 && biq_layout < 0.5)
            mountain_mask *= input.surface_coordinate;
        float3 mountain_color = lerp(mountain_base, mountain_top, upper_rock);
        if (desert_mountain)
        {
            float3 stripe2 = q4_relief_color(desert_mountain_stripe2_texture, input);
            mountain_color = lerp(mountain_color, stripe2,
                                  smoothstep(0.55, 0.76, authored_height));
            mountain_color = lerp(mountain_color, mountain_snow,
                                  smoothstep(0.76, 0.90, authored_height));
        }
        else
            mountain_color = lerp(mountain_color, mountain_snow, snow);
        albedo = lerp(albedo, mountain_color, mountain_mask);
    }
    if (dune_only > 0.5)
        albedo = lerp(float3(0.025, 0.028, 0.026), albedo, dune_weight);
    float3 linear_color = albedo;

    // The left panel is always the unlit source swatch.  The right patch only
    // enables material response in the material pass, so the albedo pass is a
    // direct color/scale/filtering comparison with no hidden terrain logic.
    if (input.panel > 0.5 && lab_mode > 0.5)
    {
        float height_left = height_texture.Sample(material_sampler, input.uv - float2(height_texel.x, 0.0)).r;
        float height_right = height_texture.Sample(material_sampler, input.uv + float2(height_texel.x, 0.0)).r;
        float height_down = height_texture.Sample(material_sampler, input.uv - float2(0.0, height_texel.y)).r;
        float height_up = height_texture.Sample(material_sampler, input.uv + float2(0.0, height_texel.y)).r;
        if (promotion_tile_layout > 0.5 && input.surface_kind < 1.5)
        {
            float3 material_weights = promotion_material_weights(input.macro_uv * 2.0);
            float plains_height_left = plains_height_texture.Sample(
                material_sampler, input.uv - float2(height_texel.x, 0.0)).r;
            float plains_height_right = plains_height_texture.Sample(
                material_sampler, input.uv + float2(height_texel.x, 0.0)).r;
            float plains_height_down = plains_height_texture.Sample(
                material_sampler, input.uv - float2(0.0, height_texel.y)).r;
            float plains_height_up = plains_height_texture.Sample(
                material_sampler, input.uv + float2(0.0, height_texel.y)).r;
            float desert_height_left = desert_height_texture.Sample(
                material_sampler, input.uv - float2(height_texel.x, 0.0)).r;
            float desert_height_right = desert_height_texture.Sample(
                material_sampler, input.uv + float2(height_texel.x, 0.0)).r;
            float desert_height_down = desert_height_texture.Sample(
                material_sampler, input.uv - float2(0.0, height_texel.y)).r;
            float desert_height_up = desert_height_texture.Sample(
                material_sampler, input.uv + float2(0.0, height_texel.y)).r;
            height_left = height_left * material_weights.x +
                          plains_height_left * material_weights.y +
                          desert_height_left * material_weights.z;
            height_right = height_right * material_weights.x +
                           plains_height_right * material_weights.y +
                           desert_height_right * material_weights.z;
            height_down = height_down * material_weights.x +
                          plains_height_down * material_weights.y +
                          desert_height_down * material_weights.z;
            height_up = height_up * material_weights.x +
                        plains_height_up * material_weights.y +
                        desert_height_up * material_weights.z;
            if (dune_weight > 0.0)
            {
                float dune_height_left = desert_hills_height_texture.Sample(
                    material_sampler, input.uv - float2(height_texel.x, 0.0)).r;
                float dune_height_right = desert_hills_height_texture.Sample(
                    material_sampler, input.uv + float2(height_texel.x, 0.0)).r;
                float dune_height_down = desert_hills_height_texture.Sample(
                    material_sampler, input.uv - float2(0.0, height_texel.y)).r;
                float dune_height_up = desert_hills_height_texture.Sample(
                    material_sampler, input.uv + float2(0.0, height_texel.y)).r;
                height_left = lerp(height_left, dune_height_left, dune_weight);
                height_right = lerp(height_right, dune_height_right, dune_weight);
                height_down = lerp(height_down, dune_height_down, dune_weight);
                height_up = lerp(height_up, dune_height_up, dune_weight);
            }
        }
        if (biq_layout > 0.5 && input.surface_kind < 1.5)
        {
            float4 weights = input.material_weights;
            float tundra_weight = input.material_tundra;
            if (marsh_enabled < 0.5) { weights.x += weights.w; weights.w = 0.0; }
            float material_total = max(0.001, dot(weights, 1.0) + tundra_weight);
            weights /= material_total;
            tundra_weight /= material_total;
            float4 sample_left = float4(
                height_left,
                plains_height_texture.Sample(material_sampler, input.uv - float2(height_texel.x, 0.0)).r,
                desert_height_texture.Sample(material_sampler, input.uv - float2(height_texel.x, 0.0)).r,
                marsh_height_texture.Sample(material_sampler, input.uv - float2(height_texel.x, 0.0)).r);
            float4 sample_right = float4(
                height_right,
                plains_height_texture.Sample(material_sampler, input.uv + float2(height_texel.x, 0.0)).r,
                desert_height_texture.Sample(material_sampler, input.uv + float2(height_texel.x, 0.0)).r,
                marsh_height_texture.Sample(material_sampler, input.uv + float2(height_texel.x, 0.0)).r);
            float4 sample_down = float4(
                height_down,
                plains_height_texture.Sample(material_sampler, input.uv - float2(0.0, height_texel.y)).r,
                desert_height_texture.Sample(material_sampler, input.uv - float2(0.0, height_texel.y)).r,
                marsh_height_texture.Sample(material_sampler, input.uv - float2(0.0, height_texel.y)).r);
            float4 sample_up = float4(
                height_up,
                plains_height_texture.Sample(material_sampler, input.uv + float2(0.0, height_texel.y)).r,
                desert_height_texture.Sample(material_sampler, input.uv + float2(0.0, height_texel.y)).r,
                marsh_height_texture.Sample(material_sampler, input.uv + float2(0.0, height_texel.y)).r);
            float tundra_left = feature_base_texture_5.Sample(material_sampler,
                input.uv - float2(height_texel.x, 0.0)).r;
            float tundra_right = feature_base_texture_5.Sample(material_sampler,
                input.uv + float2(height_texel.x, 0.0)).r;
            float tundra_down = feature_base_texture_5.Sample(material_sampler,
                input.uv - float2(0.0, height_texel.y)).r;
            float tundra_up = feature_base_texture_5.Sample(material_sampler,
                input.uv + float2(0.0, height_texel.y)).r;
            height_left = dot(sample_left, weights) + tundra_left * tundra_weight;
            height_right = dot(sample_right, weights) + tundra_right * tundra_weight;
            height_down = dot(sample_down, weights) + tundra_down * tundra_weight;
            height_up = dot(sample_up, weights) + tundra_up * tundra_weight;
        }
        if (input.surface_kind > 2.5)
        {
            height_left = cliff_height_texture.Sample(material_sampler, input.uv - float2(height_texel.x, 0.0)).r;
            height_right = cliff_height_texture.Sample(material_sampler, input.uv + float2(height_texel.x, 0.0)).r;
            height_down = cliff_height_texture.Sample(material_sampler, input.uv - float2(0.0, height_texel.y)).r;
            height_up = cliff_height_texture.Sample(material_sampler, input.uv + float2(0.0, height_texel.y)).r;
        }
        else if (input.surface_kind > 1.5)
        {
            height_left = beach_height_texture.Sample(material_sampler, input.uv - float2(height_texel.x, 0.0)).r;
            height_right = beach_height_texture.Sample(material_sampler, input.uv + float2(height_texel.x, 0.0)).r;
            height_down = beach_height_texture.Sample(material_sampler, input.uv - float2(0.0, height_texel.y)).r;
            height_up = beach_height_texture.Sample(material_sampler, input.uv + float2(0.0, height_texel.y)).r;
        }
        else if (mountain_surface)
        {
            float2 left_uv = input.uv - float2(height_texel.x, 0.0);
            float2 right_uv = input.uv + float2(height_texel.x, 0.0);
            float2 down_uv = input.uv - float2(0.0, height_texel.y);
            float2 up_uv = input.uv + float2(0.0, height_texel.y);
            float mountain_height_left = desert_mountain
                ? desert_mountain_height_texture.Sample(material_sampler, left_uv).r
                : mountain_height_texture.Sample(material_sampler, left_uv).r;
            float mountain_height_right = desert_mountain
                ? desert_mountain_height_texture.Sample(material_sampler, right_uv).r
                : mountain_height_texture.Sample(material_sampler, right_uv).r;
            float mountain_height_down = desert_mountain
                ? desert_mountain_height_texture.Sample(material_sampler, down_uv).r
                : mountain_height_texture.Sample(material_sampler, down_uv).r;
            float mountain_height_up = desert_mountain
                ? desert_mountain_height_texture.Sample(material_sampler, up_uv).r
                : mountain_height_texture.Sample(material_sampler, up_uv).r;
            height_left = lerp(height_left, mountain_height_left, mountain_mask);
            height_right = lerp(height_right, mountain_height_right, mountain_mask);
            height_down = lerp(height_down, mountain_height_down, mountain_mask);
            height_up = lerp(height_up, mountain_height_up, mountain_mask);
        }
#ifdef Q2_CONTINENTAL_MATERIAL
        height_left+=q2_continental_height_delta(input,input.uv-float2(height_texel.x,0));
        height_right+=q2_continental_height_delta(input,input.uv+float2(height_texel.x,0));
        height_down+=q2_continental_height_delta(input,input.uv-float2(0,height_texel.y));
        height_up+=q2_continental_height_delta(input,input.uv+float2(0,height_texel.y));
#endif
        float3 micro_normal = normalize(float3(
            -(height_right - height_left) * normal_strength,
            -(height_up - height_down) * normal_strength,
            1.0));
        float3 geometry_normal = normalize(input.geometry_normal);
        float3 material_normal = lab_mode > 1.5
            ? normalize(float3(
                geometry_normal.x + micro_normal.x * geometry_normal.z,
                geometry_normal.y + micro_normal.y * geometry_normal.z,
                geometry_normal.z))
            : micro_normal;
#ifdef Q2_CACHED_NORMAL
        q2_cached_normal(input,geometry_normal,material_normal);
#endif

        if (biq_layout > 0.5 && input.surface_kind < 1.5 &&
            abs(input.real_terrain - 6.0) > 0.25 &&
            !q4_volcano_surface(input))
        {
            float4 clutter_weights = input.material_weights;
            clutter_weights /= max(0.001, dot(clutter_weights, 1.0) + input.material_tundra);
            float clutter_step = 0.012;
            float grass_scale = abs(input.real_terrain - 5.0) < 0.25 ? 3.85 : 4.65;
            float2 grass_offset = float2(0.13, 0.37);
            float2 plains_offset = float2(0.47, 0.11);
            float clutter_left =
                sample_masked_land_clutter_height(
                    grassland_decal_height_texture, grassland_decal_base_texture,
                    world_position - float2(clutter_step, 0.0), grass_scale,
                    grass_offset) * clutter_weights.x +
                sample_masked_land_clutter_height(
                    plains_decal_height_texture, plains_decal_base_texture,
                    world_position - float2(clutter_step, 0.0), 4.25,
                    plains_offset) * clutter_weights.y;
            float clutter_right =
                sample_masked_land_clutter_height(
                    grassland_decal_height_texture, grassland_decal_base_texture,
                    world_position + float2(clutter_step, 0.0), grass_scale,
                    grass_offset) * clutter_weights.x +
                sample_masked_land_clutter_height(
                    plains_decal_height_texture, plains_decal_base_texture,
                    world_position + float2(clutter_step, 0.0), 4.25,
                    plains_offset) * clutter_weights.y;
            float clutter_down =
                sample_masked_land_clutter_height(
                    grassland_decal_height_texture, grassland_decal_base_texture,
                    world_position - float2(0.0, clutter_step), grass_scale,
                    grass_offset) * clutter_weights.x +
                sample_masked_land_clutter_height(
                    plains_decal_height_texture, plains_decal_base_texture,
                    world_position - float2(0.0, clutter_step), 4.25,
                    plains_offset) * clutter_weights.y;
            float clutter_up =
                sample_masked_land_clutter_height(
                    grassland_decal_height_texture, grassland_decal_base_texture,
                    world_position + float2(0.0, clutter_step), grass_scale,
                    grass_offset) * clutter_weights.x +
                sample_masked_land_clutter_height(
                    plains_decal_height_texture, plains_decal_base_texture,
                    world_position + float2(0.0, clutter_step), 4.25,
                    plains_offset) * clutter_weights.y;
            material_normal = normalize(float3(
                material_normal.x - (clutter_right - clutter_left) * 0.36,
                material_normal.y - (clutter_up - clutter_down) * 0.36,
                material_normal.z));
        }
        if (dune_weight > 0.0)
        {
            float dune_step = 0.012;
            float decal_left = combined_dune_decal_height(world_position - float2(dune_step, 0.0));
            float decal_right = combined_dune_decal_height(world_position + float2(dune_step, 0.0));
            float decal_down = combined_dune_decal_height(world_position - float2(0.0, dune_step));
            float decal_up = combined_dune_decal_height(world_position + float2(0.0, dune_step));
            material_normal = normalize(float3(
                material_normal.x - (decal_right - decal_left) * 0.55 * dune_weight,
                material_normal.y - (decal_up - decal_down) * 0.55 * dune_weight,
                material_normal.z));
        }
        if (biq_layout > 0.5 && marsh_enabled > 0.5 &&
            input.material_weights.w > 0.001)
        {
            float marsh_step = 0.010;
            float marsh_left = combined_marsh_decal_height(world_position - float2(marsh_step, 0.0));
            float marsh_right = combined_marsh_decal_height(world_position + float2(marsh_step, 0.0));
            float marsh_down = combined_marsh_decal_height(world_position - float2(0.0, marsh_step));
            float marsh_up = combined_marsh_decal_height(world_position + float2(0.0, marsh_step));
            material_normal = normalize(float3(
                material_normal.x - (marsh_right - marsh_left) * 0.72 * input.material_weights.w,
                material_normal.y - (marsh_up - marsh_down) * 0.72 * input.material_weights.w,
                material_normal.z));
        }
        if (biq_layout > 0.5 && volcano_enabled > 0.5 &&
            q4_volcano_surface(input))
        {
            float2 volcano_uv = frac(world_position);
#ifdef Q4_VOLCANO_SOURCE_MAPPING
            // Same local-v orientation and uniform footprint as the height
            // sampler. All dormant/active/slope/specular channels agree.
#ifndef Q4_VOLCANO_FOOTPRINT
#define Q4_VOLCANO_FOOTPRINT .62
#endif
            volcano_uv=.5+(float2(volcano_uv.x,1-volcano_uv.y)-.5)*Q4_VOLCANO_FOOTPRINT;
#endif
#ifdef Q4_RELIEF_MATERIAL_DATA
            volcano_uv=input.relief_material.xy;
#endif
            float2 volcano_height = volcano_height_texture.Sample(
                material_sampler, volcano_uv).rg;
            material_normal = normalize(float3(
                material_normal.x + (volcano_height.x * 2.0 - 1.0) * 0.34,
                material_normal.y + (volcano_height.y * 2.0 - 1.0) * 0.34,
                material_normal.z));
        }
#ifdef Q3_SHORE_MATERIAL
        q3_shore_material(input, world_position, albedo, material_normal);
#endif
#ifdef Q2_MATERIAL_RESPONSE
        q2_material_form(input, world_position, geometry_normal, albedo, material_normal);
#endif
#ifdef Q4_COHERENT_ROCK_CHANNELS
        if(mountain_surface && input.surface_kind<1.5 && mountain_mask>0) {
#ifdef Q4_COMPLETE_ROCK_CHANNELS
            float3 rock_normal=q4_complete_rock_normal(input,desert_mountain);
#else
            float3 rock_normal=desert_mountain
                ? q4_rock_material_normal(desert_mountain_height_texture,input)
                : q4_rock_material_normal(mountain_height_texture,input);
#endif
            material_normal=normalize(lerp(material_normal,rock_normal,mountain_mask));
        }
#endif
#ifdef Q8_DEBUG_ALBEDO
        return float4(albedo,1);
#endif
#ifdef Q8_DEBUG_NORMAL
        return float4(material_normal*.5+.5,1);
#endif
        // L13A replaces the historical fixed-noon key with the same immutable
        // source-independent frame environment used by the renderer runtime.
        float3 light_direction = frame_light_direction();
        float diffuse = saturate(dot(material_normal, light_direction));
        float shadow_visibility = lab_mode > 2.5 ? input.shape_visibility.x : 1.0;
        float ambient_visibility = lab_mode > 2.5 ? input.shape_visibility.y : 1.0;
#ifdef Q2_CACHED_OCCLUSION
        ambient_visibility*=q2_cached_occlusion(input,geometry_normal);
#endif

        float3 light = l13a_layout > 0.5
            ? q6_receiver_illumination(input, material_normal, shadow_visibility, ambient_visibility)
            : (0.68 * ambient_visibility + 0.72 * diffuse * shadow_visibility).xxx;
        if (l13a_layout > 0.5 && biq_layout > 0.5 && input.surface_kind < 1.5)
        {
            float hill_surface = 1.0 - step(0.25, abs(input.real_terrain - 5.0));
            float mountain_relief = max(
                1.0 - step(0.25, abs(input.real_terrain - 6.0)),
                1.0 - step(0.25, abs(input.real_terrain - 10.0)));
            // Hill ownership is cell-local, but its visible slope is not.
            // Weight the contrast by the continuous geometric slope so the
            // lighting cannot reveal the hidden ownership diamond.
            float hill_slope = 1.0 - smoothstep(0.78, 0.995,
                                                geometry_normal.z);
            float relief_weight = saturate(hill_surface * hill_slope * 0.72 +
                                           mountain_relief * mountain_mask);
            float relief_form = raised_form_response(diffuse);
            light *= lerp(1.0, relief_form, relief_weight);
        }

        float specular = specular_texture.Sample(material_sampler, input.uv).r;
        if (promotion_tile_layout > 0.5 && input.surface_kind < 1.5)
        {
            float3 material_weights = promotion_material_weights(input.macro_uv * 2.0);
            float plains_specular = plains_specular_texture.Sample(material_sampler, input.uv).r;
            float desert_specular = desert_specular_texture.Sample(material_sampler, input.uv).r;
            specular = specular * material_weights.x + plains_specular * material_weights.y +
                       desert_specular * material_weights.z;
            if (dune_weight > 0.0)
            {
                float dune_specular = desert_hills_specular_texture.Sample(
                    material_sampler, input.uv).r;
                specular = lerp(specular, dune_specular, dune_weight);
            }
        }
        if (biq_layout > 0.5 && input.surface_kind < 1.5)
        {
            float4 weights = input.material_weights;
            float tundra_weight = input.material_tundra;
            if (marsh_enabled < 0.5) { weights.x += weights.w; weights.w = 0.0; }
            float material_total = max(0.001, dot(weights, 1.0) + tundra_weight);
            weights /= material_total;
            tundra_weight /= material_total;
            float plains_specular = plains_specular_texture.Sample(material_sampler, input.uv).r;
            float desert_specular = desert_specular_texture.Sample(material_sampler, input.uv).r;
            float marsh_specular = marsh_specular_texture.Sample(material_sampler, input.uv).r;
            float tundra_specular = feature_base_texture_6.Sample(material_sampler, input.uv).r;
            specular = dot(float4(specular, plains_specular, desert_specular, marsh_specular), weights) +
                       tundra_specular * tundra_weight;
            if (weights.w > 0.001 && marsh_enabled > 0.5)
            {
                float decal_specular = marsh_decal_specular_texture.Sample(
                    decal_sampler, frac(world_position)).r;
                specular = lerp(specular, decal_specular, 0.54 * weights.w);
            }
            if (volcano_enabled > 0.5 && q4_volcano_surface(input))
            {
                float fixture_active = fmod(floor(world_position.x) * 17.0 +
                                            floor(world_position.y) * 31.0, 2.0) < 1.0 ? 1.0 : 0.0;
                float active = input.active_effect >= 0.0
                    ? input.active_effect : q4_volcano_active(input,fixture_active);
                float2 volcano_uv = frac(world_position);
#ifdef Q4_VOLCANO_SOURCE_MAPPING
            // Same local-v orientation and uniform footprint as the height
            // sampler. All dormant/active/slope/specular channels agree.
#ifndef Q4_VOLCANO_FOOTPRINT
#define Q4_VOLCANO_FOOTPRINT .62
#endif
            volcano_uv=.5+(float2(volcano_uv.x,1-volcano_uv.y)-.5)*Q4_VOLCANO_FOOTPRINT;
#endif
#ifdef Q4_RELIEF_MATERIAL_DATA
            volcano_uv=input.relief_material.xy;
#endif
                float authored_specular = volcano_active_specular_texture.Sample(
                    material_sampler, volcano_uv).r;
                float3 active_color = volcano_active_base_texture.Sample(
                    material_sampler, volcano_uv).rgb;
                float lava_mask = smoothstep(0.16, 0.52,
                    max(active_color.r, max(active_color.g, active_color.b)));
                specular = lerp(specular, authored_specular,
                                active * lava_mask *
                                q4_volcano_coverage(input));
            }
        }
        if (input.surface_kind > 2.5)
            specular = cliff_specular_texture.Sample(material_sampler, input.uv).r;
        else if (input.surface_kind > 1.5)
            specular = beach_specular_texture.Sample(material_sampler, input.uv).r;
        else if (mountain_surface)
            specular = lerp(specular,
                            desert_mountain
                                ? (
#ifdef Q4_COMPLETE_ROCK_CHANNELS
                q4_complete_rock_specular(input,true)
#elif defined(Q4_COHERENT_ROCK_CHANNELS)
                q4_relief_color(desert_mountain_specular_texture,input).r
#else
                desert_mountain_specular_texture.Sample(material_sampler, input.uv).r
#endif
                )
                                : (
#ifdef Q4_COMPLETE_ROCK_CHANNELS
                q4_complete_rock_specular(input,false)
#elif defined(Q4_COHERENT_ROCK_CHANNELS)
                q4_relief_color(mountain_specular_texture,input).r
#else
                mountain_specular_texture.Sample(material_sampler, input.uv).r
#endif
                ),
                            mountain_mask);
#ifdef Q2_MATERIAL_RESPONSE
#ifdef Q2_CONTINENTAL_MATERIAL
        specular+=q2_continental_specular_delta(input);
#endif
        q2_material_specular(input, world_position, geometry_normal, specular);
#endif
        float3 view_direction = float3(0.0, 0.0, 1.0);
        float3 half_direction = normalize(light_direction + view_direction);
        float highlight = pow(saturate(dot(material_normal, half_direction)), 32.0) *
                          specular * 0.08 *
                          (l13a_layout > 0.5 ? environment_water_specular : 1.0);
#ifdef Q6_WORLD_SHADOWS
        highlight *= q6_receiver_visibility(input, material_normal, 1.0);
#endif
        float3 highlight_color = l13a_layout > 0.5
            ? saturate(environment_sun_color * environment_sun_intensity +
                       environment_moon_color * environment_moon_intensity)
            : 1.0.xxx;
        linear_color = albedo * light + highlight * highlight_color;
    }

    float3 display_color = (frame_tone_map(
        linear_color * frame_output_exposure()));
    return float4(display_color, 1.0);
}

// Hardware composition consumes premultiplied scene-linear color exactly once.
struct Q6SceneOutput { float4 color : SV_Target0; float validity : SV_Target1; };
Q6SceneOutput q6_scene_output(float4 raw) {
    Q6SceneOutput o;
    float alpha = saturate(raw.a);
    clip(alpha - 0.000001);
    o.color = float4(max(raw.rgb, 0.0) * alpha, alpha);
    o.validity = 1.0;
    return o;
}
Q6SceneOutput PSMain(PixelInput input) { return q6_scene_output(q6_raw_main(input)); }
Q6SceneOutput PSFeature(FeaturePixelInput input) { return q6_scene_output(q6_raw_feature(input)); }

#ifndef Q3_SCENE_MATERIAL_V1
#define Q3_SCENE_MATERIAL_V1
// Source-composed static material. Raw scene-linear RGB and straight coverage;
// Q6 wrapper premultiplies once. The Q0 CPU hook supplies exact continuous fields.
// hydrology_data = positive-land distance, beach width, rocky fraction, depth.
#ifndef Q3_MATERIAL_ORIGIN_X
#define Q3_MATERIAL_ORIGIN_X 0
#define Q3_MATERIAL_ORIGIN_Y 0
#define Q3_MATERIAL_WRAP_WIDTH 0
#endif
float2 q3_source_world(PixelInput input) {
 float2 world=input.macro_uv*2+float2(Q3_MATERIAL_ORIGIN_X,Q3_MATERIAL_ORIGIN_Y);
 if(Q3_MATERIAL_WRAP_WIDTH>0){
  float rawx=world.x+world.y,rawy=world.x-world.y;
  rawx-=floor(rawx/max(1,Q3_MATERIAL_WRAP_WIDTH))*Q3_MATERIAL_WRAP_WIDTH;
  world=float2(rawx+rawy,rawx-rawy)*.5;
 }
 return world;
}
float q3_source_repeat(float requested) {
 float period=Q3_MATERIAL_WRAP_WIDTH*.5;
 return period>0?round(requested*period)/period:requested;
}
#ifdef Q3_WATER_EFFECTS
#ifndef Q3_WATER_EFFECTS_IMPL
#define Q3_WATER_EFFECTS_IMPL
// Lab prototype inspired by the technique breakdown in Alex Tardif's Water
// Walkthrough. Independent implementation using existing local pack textures.
// Fixed phase makes replay deterministic; runtime animation clock is pending.
#ifndef Q3_WATER_TIME
#define Q3_WATER_TIME 0.0
#endif
float q3_water_hash(float2 cell) {
    float period=Q3_MATERIAL_WRAP_WIDTH*.5*q3_source_repeat(1.2);
    if(period>0)cell-=floor(cell/period)*period;
    return macro_decal_hash(cell);
}
float q3_water_noise(float2 p) {
    float2 cell=floor(p),f=frac(p);f=f*f*(3-2*f);
    return lerp(lerp(q3_water_hash(cell),q3_water_hash(cell+float2(1,0)),f.x),
        lerp(q3_water_hash(cell+float2(0,1)),q3_water_hash(cell+1),f.x),f.y);
}
float3 q3_effect_normal(PixelInput input,float3 source_normal) {
    float2 world=q3_source_world(input);
    float time=Q3_WATER_TIME;
    float2 k0=float2(q3_source_repeat(1.38),q3_source_repeat(.52));
    float2 k1=float2(q3_source_repeat(-.72),q3_source_repeat(1.96));
    float2 k2=float2(q3_source_repeat(3.44),q3_source_repeat(2.22));
    float2 slopes=normalize(k0)*cos(dot(world,k0)*6.283185-time*1.12)*.12
        +normalize(k1)*cos(dot(world,k1)*6.283185-time*.79)*.055
        +normalize(k2)*cos(dot(world,k2)*6.283185-time*1.47)*.028;
    float2 drift=float2(time*.018,-time*.012);
    float2 small=water_small_lean0_texture.Sample(material_sampler,
        world*float2(q3_source_repeat(2.4),q3_source_repeat(3.06))+drift).rg*2-1;
    float depth_fade=smoothstep(0,.14,input.hydrology_data.w);
    return normalize(float3(source_normal.xy*.60-(slopes+small*.09)*depth_fade,1));
}
void q3_effect_color(PixelInput input,float3 normal,float3 illumination,
                    inout float3 tint,inout float alpha) {
    float2 world=q3_source_world(input);
    float time=Q3_WATER_TIME,d=max(0,-input.hydrology_data.x);
    float depth=max(0,input.hydrology_data.w);
    float2 noise_uv=world*q3_source_repeat(1.2);
    float patch=q3_water_noise(noise_uv);
    float2 foam_uv=world*float2(q3_source_repeat(3.4),q3_source_repeat(4.1))+float2(time*.025,-time*.012);
    float grain=water_foam_texture.Sample(material_sampler,foam_uv).a;
    // Broken shallow-water fronts approach the coast. Do not clamp at tile edges.
    float phase=d*27-time*1.3+patch*2.4;
    float front=pow(saturate(.5+.5*cos(phase)),12);
    float shore=smoothstep(.012,.055,d)*(1-smoothstep(.16,.30,d));
    float broken=smoothstep(.22,.70,grain*.70+patch*.48);
    float foam=front*shore*broken*.65;
    float contact=exp(-d*40)*smoothstep(.002,.025,d)*broken*.26;
    foam=saturate(foam+contact);
    float3 foam_light=float3(.65,.76,.77)*q6_receiver_illumination(input,float3(0,0,1),1,1);
    // More visible wave-facing sky response, still driven by the shared rig.
    float facing=saturate(normal.y*.8+normal.x*.3+.12);
    float3 sky=environment_ambient_color*float3(.055,.085,.105)*facing;
    tint+=sky*smoothstep(.05,.22,depth);
    // Compose foam over the existing water coverage without double premultiplication.
    float3 premult=tint*alpha*(1-foam)+foam_light*foam;
    alpha=alpha+(1-alpha)*foam;
    tint=premult/max(alpha,.0001);
}
#endif

#endif
#ifdef Q3_NATURAL_WATER
#ifndef Q3_NATURAL_WATER_IMPL
#define Q3_NATURAL_WATER_IMPL
#ifndef Q3_NATURAL_COORD_SHIFT
#define Q3_NATURAL_COORD_SHIFT 0.0
#endif
#ifdef Q3_OBJECT_REFLECTION
Texture2D q3_object_reflection_texture : register(t121);
#endif
// Static Lab experiment: periodic source-detail patches and view-dependent
// reflection. This is a generic adaptation, not recovered Civ VI equations.
float q3_natural_hash(float2 cell) {
 float period=Q3_MATERIAL_WRAP_WIDTH*.5*q3_source_repeat(.6);
 if(period>0)cell-=floor(cell/period)*period;
 return macro_decal_hash(cell);
}
float q3_natural_noise(float2 p) {
 float2 c=floor(p),f=frac(p);f=f*f*(3-2*f);
 return lerp(lerp(q3_natural_hash(c),q3_natural_hash(c+float2(1,0)),f.x),
  lerp(q3_natural_hash(c+float2(0,1)),q3_natural_hash(c+1),f.x),f.y);
}
float3 q3_natural_normal(PixelInput input) {
 float2 world=q3_source_world(input)+Q3_NATURAL_COORD_SHIFT;
 float2 patch_uv=world*q3_source_repeat(.6);
 float2 warp=float2(q3_natural_noise(patch_uv),q3_natural_noise(patch_uv+float2(7,13)))-.5;
 float2 uv0=world*float2(q3_source_repeat(.36),q3_source_repeat(.48))+warp*.16;
 float2 uv1=world*float2(q3_source_repeat(.72),q3_source_repeat(.94))+warp*.12+float2(.27,.61);
 float2 a=water_large_lean0_texture.Sample(material_sampler,uv0).rg*2-1;
 float2 b=water_small_lean0_texture.Sample(material_sampler,uv1).rg*2-1;
 float2 secondary_uv=float2(world.y,-world.x)*float2(q3_source_repeat(1.12),q3_source_repeat(1.46))+warp*.1;
 float2 c=water_small_secondary_lean0_texture.Sample(material_sampler,secondary_uv).rg*2-1;
 // Rotate the crossing detail slope vector back to the world basis too.
 c=float2(-c.y,c.x);
 // Broad calm lanes interrupt the source pattern without per-tile phases.
 float envelope=lerp(.16,1,smoothstep(.20,.78,warp.x+.5));
 float2 slope=(a*.40+b*.38+c*.22)*envelope;
 slope*=lerp(.48,1,smoothstep(.01,.24,input.hydrology_data.w));
 return normalize(float3(-slope,1));
}
float4 q3_natural_water(PixelInput input) {
 float depth=max(0,input.hydrology_data.w);
 float3 normal=q3_natural_normal(input);
 float3 view=normalize(float3(0,-.52,.86));
 // The volume is lit on the mean water plane. Fine slopes change reflected
 // light, not the diffuse shading of an opaque corrugated surface.
 float3 bulk_light=q6_receiver_illumination(input,float3(0,0,1),1,1);
 float3 body=lerp(float3(.023,.074,.096),float3(.003,.015,.040),smoothstep(.18,.43,depth))*bulk_light;
 // The shared rig supplies .04 at noon and .12 at night. Treat this small
 // reflectance control as the base response, not another multiplier on .02.
 float f0=saturate(environment_water_fresnel);
 float fresnel=f0+(1-f0)*pow(1-saturate(dot(normal,view)),5);
 float3 ray=reflect(-view,normal);
 float sky_band=smoothstep(.30,.90,ray.y);
 float3 sky_light=environment_ambient_color*.6+environment_sun_color*environment_sun_intensity*.6
  +environment_moon_color*environment_moon_intensity*.6;
 float3 sky=sky_light*lerp(float3(.16,.25,.36),float3(.42,.55,.68),sky_band);
#ifdef Q3_OBJECT_REFLECTION
 // Same authoritative camera and water plane as the mirrored render target.
 // Sampling uses the linear offscreen image, never the display-tonemapped PNG.
 float2 reflected_uv=input.position.xy/Q3_REFLECTION_SIZE;
 float2 distortion=normal.xy*float2(3.0,1.5)/Q3_REFLECTION_SIZE;
 float4 object=q3_object_reflection_texture.Sample(decal_sampler,reflected_uv+distortion);
 float inside=step(0,reflected_uv.x)*step(reflected_uv.x,1)*step(0,reflected_uv.y)*step(reflected_uv.y,1);
 float object_coverage=saturate(object.a)*inside;
 sky=sky*(1-object_coverage)+object.rgb*inside;
#endif
 float2 world=q3_source_world(input)+Q3_NATURAL_COORD_SHIFT;
 float2 micro=water_small_lean0_texture.Sample(material_sampler,
  world*float2(q3_source_repeat(3.4),q3_source_repeat(4.12))+float2(.71,.29)).rg*2-1;
 float sparkle=lerp(.22,1,smoothstep(.025,.16,length(micro)));
 float3 sunhalf=normalize(view+environment_sun_direction);
 float3 moonhalf=normalize(view+environment_moon_direction);
 float3 glint=(environment_sun_color*environment_sun_intensity*pow(saturate(dot(normal,sunhalf)),48)
  +environment_moon_color*environment_moon_intensity*pow(saturate(dot(normal,moonhalf)),48))
  *sparkle*.045*environment_water_specular*q6_receiver_visibility(input,normal,1);
 float reflection=saturate(fresnel);
 float coverage=1-exp(-depth*3.2);
 float alpha=coverage+(1-coverage)*reflection;
 float3 premult=body*coverage*(1-reflection)+sky*reflection+glint;
 return float4(premult/max(alpha,.0001),alpha);
}
#endif

#endif
float4 q3_authored_bed_detail(PixelInput input) {
 float2 projected=q3_source_world(input)/1.0+float2(.29,.53);
 float variant=floor(macro_decal_hash(floor(projected))*4);
 float2 uv=coast_clutter_atlas_uv(frac(projected),variant);
 float4 detail=water_decal_base_texture.Sample(decal_sampler,uv);
 detail.a*=projected_decal_edge_fade(frac(projected))
  *(1-smoothstep(.24,.43,input.hydrology_data.w));
 return detail;
}
float3 q3_authored_bed_normal(PixelInput input) {
 float2 projected=q3_source_world(input)/1.0+float2(.29,.53);
 float variant=floor(macro_decal_hash(floor(projected))*4);
 float2 uv=coast_clutter_atlas_uv(frac(projected),variant);
 float dx=water_decal_height_texture.Sample(decal_sampler,uv+float2(.001,0)).r
  -water_decal_height_texture.Sample(decal_sampler,uv-float2(.001,0)).r;
 float dy=water_decal_height_texture.Sample(decal_sampler,uv+float2(0,.001)).r
  -water_decal_height_texture.Sample(decal_sampler,uv-float2(0,.001)).r;
 float support=q3_authored_bed_detail(input).a;
 return normalize(float3(-dx*14*support,-dy*14*support,1));
}
float3 q3_scene_bed(PixelInput input) {
 float sd=input.hydrology_data.x,rocky=saturate(input.hydrology_data.z);
 float2 uv=q3_source_world(input)*q3_source_repeat(.75);
 float3 sand=beach_base_texture.Sample(material_sampler,uv).rgb;
 float3 bed=shallow_bed_texture.Sample(material_sampler,uv).rgb;
 float2 world=q3_source_world(input);
 float4 authored=q3_authored_bed_detail(input);
 bed=lerp(bed,authored.rgb,authored.a);
 bed*=1+(sample_water_clutter_height(world)-.5)*authored.a*.30;
 float3 rock=cliff_base_texture.Sample(material_sampler,uv).rgb;
 float3 color=lerp(sand,bed,smoothstep(0,.40,-sd));
#ifdef Q3_COAST_DETAIL
 color=lerp(sand,bed,smoothstep(0,.12,-sd));
#endif
 color=lerp(color,lerp(rock,bed,smoothstep(0,.70,-sd)),rocky);
 color*=lerp(.72,1.0,smoothstep(0,.32,-sd));
 float height=water_height_texture.Sample(material_sampler,uv).r;
 // Confirmed source height detail; no animated or inferred wave channels.
 // Spectral absorption tints the actual bed before coverage compositing.
 // This preserves authored contrast in shallows without a beige offshore plate.
 float3 absorption=exp(-input.hydrology_data.w*float3(14,7,3));
 return color*clamp(.86+(height-.426)*.55,.65,1.1)*absorption;
}
void q3_shore_material(PixelInput input,float2 world_position,inout float3 albedo,inout float3 material_normal) {
 float sd=input.hydrology_data.x,width=input.hydrology_data.y;
 float rocky=saturate(input.hydrology_data.z);
 float3 sand=beach_base_texture.Sample(material_sampler,q3_source_world(input)*q3_source_repeat(.75)).rgb;
 float grain=dot(sand,float3(.2126,.7152,.0722));
 float blend=1-smoothstep(width*.25,width+.06,sd+(grain-.28)*.24);
 albedo=lerp(albedo,sand,blend*(1-rocky));
 float2 uv=q3_source_world(input)*q3_source_repeat(.75);
 float hx=beach_height_texture.Sample(material_sampler,uv+float2(.002,0)).r
  -beach_height_texture.Sample(material_sampler,uv-float2(.002,0)).r;
 float hy=beach_height_texture.Sample(material_sampler,uv+float2(0,.002)).r
  -beach_height_texture.Sample(material_sampler,uv-float2(0,.002)).r;
 float2 detail=clamp(float2(-hx-hy,-hx+hy)*12,-.18,.18)*blend*(1-rocky);
 material_normal=normalize(float3(material_normal.xy+detail*material_normal.z,material_normal.z));
 float3 rock=cliff_base_texture.Sample(material_sampler,q3_source_world(input)*q3_source_repeat(.75)).rgb;
 albedo=lerp(albedo,rock,rocky*(1-smoothstep(.03,.22,sd)));
 albedo*=1-.28*(1-smoothstep(-.02,.12,sd));
}
float4 q3_water_material(PixelInput input) {
 float kind=input.surface_kind;
#ifdef Q3_BED_ONLY
 if(kind>4.5&&kind<5.5){clip(-1);return 0;}
#endif
 // Captured surf/foam is deferred, never retained as permanent pale geometry.
 if(kind>5.5&&kind<6.5){clip(-1);return 0;}
 float sd=input.hydrology_data.x,depth=max(0,input.hydrology_data.w);
 float3 normal=float3(0,0,1);
 float source_roughness=1;
#ifdef Q3_SOURCE_WATER_NORMALS
 if(kind>4.5&&kind<5.5){
  // Static source surface phase, sampled in the same wrapped world basis as
  // the bed. Source slopes/moments drive lighting; this is a C3X adaptation,
  // not a recovered source-engine LEAN or wave animation equation.
  float2 world=q3_source_world(input);
  float2 large_uv=world*float2(q3_source_repeat(.36),q3_source_repeat(.47));
  float2 small_uv=world*float2(q3_source_repeat(2.4),q3_source_repeat(3.05))+float2(.31,.17);
  float2 large=water_large_lean0_texture.Sample(material_sampler,large_uv).rg*2-1;
  float2 small=water_small_lean0_texture.Sample(material_sampler,small_uv).rg*2-1;
  float2 variance=water_large_lean1_texture.Sample(material_sampler,large_uv).rg
   +water_small_lean1_texture.Sample(material_sampler,small_uv).rg;
  float2 lean=large*.64+small*.24;
  normal=normalize(float3(-lean,1));
  source_roughness=rcp(1+dot(variance,float2(2,2)));
 }
#endif
#ifdef Q3_WATER_EFFECTS
 if(kind>4.5&&kind<5.5)normal=q3_effect_normal(input,normal);
#endif
 float3 illumination=q6_receiver_illumination(input,normal,1,1);
 if(kind>8.5&&kind<9.5){
  // The default keeps the frozen analytic distance. Q3_CONTINUOUS_RIVERS
  // consumes the opt-in shared corridor, including its terminal presentation.
  float distance_pixels=input.river_data.x;
#ifdef Q3_CONTINUOUS_RIVERS
  float2 world=q3_source_world(input);
  float2 uv=world*q3_source_repeat(.75);
  float noise=river_bank_noise_texture.Sample(material_sampler,
   world*float2(q3_source_repeat(.31),q3_source_repeat(.47))+float2(.19,.37)).r-.5;
  float grain=river_height_texture.Sample(material_sampler,uv).r;
  float water_width=5.8+noise*.8;
  float water=1-smoothstep(water_width-1.2,water_width,distance_pixels);
  float bank_width=9.8+noise*4+(grain-.5)*1.2;
  float bank=1-smoothstep(bank_width-2.5,bank_width,distance_pixels);
  // Banks end at the optical shore; the water itself overlaps and dissolves
  // into the existing sea surface instead of ending in an offshore capsule.
  float land_bank=smoothstep(-.025,.065,sd);
  float outlet=smoothstep(-.20,.025,sd);
  bank*=outlet;clip(bank-.001);
  water=lerp(1,water,land_bank);
  float3 bed=river_base_texture.Sample(material_sampler,uv).rgb;
  float3 sand=beach_base_texture.Sample(material_sampler,uv).rgb;
  float4 clutter=river_clutter_base_texture.Sample(decal_sampler,frac(world*.5+float2(.23,.41)));
  float3 dry=lerp(bed,sand,.32)*(0.64+(grain-.5)*.30);
  dry=lerp(dry,clutter.rgb*.66,clutter.a*.30);
  float wet=1-smoothstep(water_width,bank_width-1.0,distance_pixels);
  float3 shore=lerp(dry,dry*.50,wet);
  float optical_depth=.10+.32*(1-smoothstep(0,5.5,max(0,distance_pixels)));
  float3 transmitted=bed*exp(-optical_depth*float3(8,4,2));
  float3 river=lerp(transmitted,float3(.018,.044,.052),1-exp(-optical_depth*5));
  float2 lean=river_lean0_texture.Sample(material_sampler,
    world*float2(q3_source_repeat(.92),q3_source_repeat(1.27))).rg*2-1;
  float3 river_normal=normalize(float3(-lean*.24,1));
  float3 water_light=q6_receiver_illumination(input,river_normal,1,1);
  float3 bank_light=q6_receiver_illumination(input,normalize(input.geometry_normal),1,1);
  return float4(lerp(shore*bank_light,river*water_light,water),bank);
#elif defined(Q3_STATIC_OPTICS_V2)
  // Keep the captured curve and navigable width. The source river bed remains
  // visible through shallow edges; narrow damp banks replace the sandy outline.
  float water=1-smoothstep(4.6,6.0,distance_pixels);
  float bank=1-smoothstep(6.0,7.4,distance_pixels);clip(bank-.001);
  float2 uv=q3_source_world(input)*q3_source_repeat(.75);
  float3 bed=river_base_texture.Sample(material_sampler,uv).rgb;
  float3 damp=lerp(bed,beach_base_texture.Sample(material_sampler,uv).rgb,.25)*.48;
  float optical_depth=.10+.32*(1-smoothstep(0,5.5,distance_pixels));
  float3 transmitted=bed*exp(-optical_depth*float3(8,4,2));
  float3 river=lerp(transmitted,float3(.009,.060,.075),1-exp(-optical_depth*5));
  return float4(lerp(damp,river,water)*illumination,bank);
#else
  float water=1-smoothstep(4.6,6.0,distance_pixels);
  float bank=1-smoothstep(6.0,9.0,distance_pixels);clip(bank-.001);
  float3 bed=river_base_texture.Sample(material_sampler,input.uv).rgb*.72;
  return float4(lerp(bed,float3(.065,.125,.155),water*.78)*illumination,bank*.96);
#endif
 }
 clip(-sd-.0001);
 if(kind<4.5)return float4(q3_scene_bed(input)*q6_receiver_illumination(input,q3_authored_bed_normal(input),1,1),1);
#ifdef Q3_NATURAL_WATER
 return q3_natural_water(input);
#endif
 // Optical absorption over separately shaded authored bed; no opaque water plate.
 float alpha=1-exp(-depth*3.2);
 float3 tint=lerp(float3(.023,.074,.096),float3(.003,.015,.040),smoothstep(.18,.43,depth))*illumination;
 float3 view=normalize(float3(0,-.52,.86));
 float fresnel=.02+.98*pow(1-saturate(dot(normal,view)),5);
 float3 reflection=environment_ambient_color*.2;
 float3 sunhalf=normalize(view+environment_sun_direction);
 float3 moonhalf=normalize(view+environment_moon_direction);
 float3 glint=environment_sun_color*environment_sun_intensity*pow(saturate(dot(normal,sunhalf)),180)
  +environment_moon_color*environment_moon_intensity*pow(saturate(dot(normal,moonhalf)),180);
 tint+=reflection*fresnel*environment_water_fresnel+glint*.12*source_roughness*environment_water_specular
  *q6_receiver_visibility(input,normal,1);
#ifdef Q3_WATER_EFFECTS
 q3_effect_color(input,normal,illumination,tint,alpha);
#endif
 return float4(tint,alpha);
}
#endif


