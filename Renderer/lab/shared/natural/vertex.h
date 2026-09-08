#pragma once
// Shared production map geometry layout, independent of a graphics API.
namespace c3x_renderer { namespace fidelity {
struct MapVertex {
    float x, y, z;
    float u, v;
    float panel;
    float normal_x, normal_y, normal_z;
    float shadow_visibility, ambient_visibility;
    float macro_u, macro_v;
    float surface_kind;
    float surface_coordinate;
    float base_terrain;
    float real_terrain;
    float material_grass, material_plains, material_desert, material_marsh;
    float authored_relief_height, authored_relief_blend;
    float shore_distance;
    float river_distance, river_branch_count, river_mouth_distance, river_padding;
    float active_effect;
    float material_tundra;
    float world_x, world_y, world_z, world_valid;
    float shore_true_distance, shore_beach_width, shore_rockiness, shore_depth;
    float relief_owner_u, relief_owner_v, relief_owner_coverage, relief_owner_state;
};

static_assert(sizeof(MapVertex)==168, "Production map vertex layout changed");
}}
