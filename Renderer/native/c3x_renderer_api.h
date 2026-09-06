#ifndef C3X_RENDERER_API_H
#define C3X_RENDERER_API_H

#ifdef __cplusplus
#include <cstdint>
extern "C" {
typedef std::uint32_t c3x_renderer_u32;
typedef std::int32_t c3x_renderer_i32;
typedef std::int64_t c3x_renderer_i64;
#else
#include <stdint.h>
typedef uint32_t c3x_renderer_u32;
typedef int32_t c3x_renderer_i32;
typedef int64_t c3x_renderer_i64;
#endif

#define C3X_RENDERER_API_VERSION 10u

enum c3x_renderer_result {
    C3X_RENDERER_RESULT_ERROR = 0,
    C3X_RENDERER_RESULT_OK = 1,
    C3X_RENDERER_RESULT_BAD_ARGUMENT = 2,
    C3X_RENDERER_RESULT_DEVICE_ERROR = 3
};

enum c3x_renderer_tile_flags {
    C3X_RENDERER_TILE_RENDER = 1u,
    C3X_RENDERER_TILE_VANILLA_BASE_CALL = 2u,
    C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED = 4u,
    C3X_RENDERER_TILE_CUSTOM_FEATURE_REPLACED = 8u,
    C3X_RENDERER_TILE_CUSTOM_DUNES_REPLACED = 16u,
    C3X_RENDERER_TILE_CUSTOM_RIVER_REPLACED = 32u,
    C3X_RENDERER_TILE_TOPOLOGY_HALO = 64u,
    C3X_RENDERER_TILE_CUSTOM_ROAD_REPLACED = 128u,
    C3X_RENDERER_TILE_CUSTOM_RAILROAD_REPLACED = 256u,
    C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED = 512u,
    C3X_RENDERER_TILE_CUSTOM_CITY_REPLACED = 1024u,
    C3X_RENDERER_TILE_CUSTOM_MINE_REPLACED = 2048u,
    C3X_RENDERER_TILE_CUSTOM_FARM_REPLACED = 4096u
};

enum c3x_renderer_invalidation_flags {
    C3X_RENDERER_INVALIDATE_CAMERA = 1u,
    C3X_RENDERER_INVALIDATE_SCENE = 2u,
    C3X_RENDERER_INVALIDATE_ENVIRONMENT = 4u,
    C3X_RENDERER_INVALIDATE_WRAP = 8u,
    C3X_RENDERER_INVALIDATE_PACK_DEFINITION = 16u,
    C3X_RENDERER_INVALIDATE_OWNERSHIP = 32u,
    C3X_RENDERER_INVALIDATE_DEVICE = 64u,
    C3X_RENDERER_INVALIDATE_ALL = 127u
};

enum c3x_renderer_feature_flags {
    C3X_RENDERER_FEATURE_FOREST = 1u,
    C3X_RENDERER_FEATURE_JUNGLE = 2u,
    C3X_RENDERER_FEATURE_MARSH = 4u,
    C3X_RENDERER_FEATURE_VOLCANO = 8u
};

enum c3x_renderer_improvement_flags {
    C3X_RENDERER_IMPROVEMENT_IRRIGATION = 1u,
    C3X_RENDERER_IMPROVEMENT_MINE = 2u,
    C3X_RENDERER_IMPROVEMENT_TILE_BUILDING = 4u,
    C3X_RENDERER_IMPROVEMENT_POLLUTION = 8u,
    C3X_RENDERER_IMPROVEMENT_CRATER = 16u
};

enum c3x_renderer_city_flags {
    C3X_RENDERER_CITY_CAPITAL = 1u,
    C3X_RENDERER_CITY_WALLED = 2u
};

enum c3x_renderer_dirty_flags {
    C3X_RENDERER_DIRTY_SCENE = 1u,
    C3X_RENDERER_DIRTY_STATIC_MAP = 2u,
    C3X_RENDERER_DIRTY_DYNAMIC = 4u,
    C3X_RENDERER_DIRTY_COMPOSITE = 8u,
    C3X_RENDERER_DIRTY_ALL = 15u
};

enum c3x_renderer_scheduler_state_flags {
    C3X_RENDERER_SCHEDULER_MAP_VISIBLE = 1u,
    C3X_RENDERER_SCHEDULER_FOCUSED = 2u,
    C3X_RENDERER_SCHEDULER_MODAL = 4u,
    C3X_RENDERER_SCHEDULER_DRAWING = 8u,
    C3X_RENDERER_SCHEDULER_REDRAW_PENDING = 16u
};

#pragma pack(push, 4)
struct c3x_renderer_tile_v1 {
    c3x_renderer_i32 tile_x;
    c3x_renderer_i32 tile_y;
    c3x_renderer_i32 anchor_x;
    c3x_renderer_i32 anchor_y;
    c3x_renderer_i32 terrain_type; /* underlying ground from Civ III m49 */
    c3x_renderer_u32 square_parts;
    c3x_renderer_u32 terrain_overlays;
    c3x_renderer_u32 visibility_mask;
    c3x_renderer_u32 variant_seed;
    c3x_renderer_u32 tile_flags;
    c3x_renderer_i32 real_terrain_type; /* visible square category from Civ III m50 */
    c3x_renderer_i32 resource_id;
    c3x_renderer_i32 resource_class;
    c3x_renderer_i32 tile_building_id;
    c3x_renderer_i32 city_id;
    c3x_renderer_i32 city_owner_id;
    c3x_renderer_i32 city_population;
    c3x_renderer_i32 city_size;
    c3x_renderer_i32 city_culture_group;
    c3x_renderer_i32 city_era;
    c3x_renderer_i32 unit_type_id;
    c3x_renderer_i32 unit_owner_id;
    c3x_renderer_i32 unit_class;
    c3x_renderer_i32 unit_state;
    c3x_renderer_i32 unit_damage;
    c3x_renderer_i32 unit_direction;
    c3x_renderer_u32 river_code;
    c3x_renderer_u32 road_mask;
    c3x_renderer_u32 railroad_mask;
    c3x_renderer_i32 route_style;
    c3x_renderer_u32 feature_flags;
    c3x_renderer_u32 improvement_flags;
    c3x_renderer_u32 irrigation_mask;
    c3x_renderer_u32 city_flags;
    c3x_renderer_u32 has_effect;
    c3x_renderer_i32 territory_owner_id;
    c3x_renderer_i32 fog_status;
    c3x_renderer_u32 tile_visibility;
    char resource_name[24];
    char city_owner[40];
    char city_civilization[40];
    char city_era_name[64];
    char unit_owner[40];
    char unit_civilization[40];
    char unit_era_name[64];
    char unit_type_name[32];
};

struct c3x_renderer_frame_v1 {
    c3x_renderer_u32 api_version;
    c3x_renderer_u32 struct_size;
    c3x_renderer_i32 target_width;
    c3x_renderer_i32 target_height;
    c3x_renderer_i32 clip_left;
    c3x_renderer_i32 clip_top;
    c3x_renderer_i32 clip_right;
    c3x_renderer_i32 clip_bottom;
    c3x_renderer_i32 tile_width;
    c3x_renderer_i32 tile_height;
    c3x_renderer_i32 hour;
    c3x_renderer_i32 season;
    c3x_renderer_u32 tile_count;
    struct c3x_renderer_tile_v1 const * tiles;
    c3x_renderer_i64 presentation_time_ticks;
    c3x_renderer_i64 presentation_frequency;
    c3x_renderer_u32 dirty_flags;
    c3x_renderer_u32 visible_animation_count;
    c3x_renderer_i32 world_width_tiles;
    c3x_renderer_i32 world_height_tiles;
    c3x_renderer_u32 world_wrap_x;
    c3x_renderer_u32 world_wrap_y;
};

struct c3x_renderer_output_v1 {
    c3x_renderer_u32 api_version;
    c3x_renderer_u32 struct_size;
    c3x_renderer_i32 width;
    c3x_renderer_i32 height;
    c3x_renderer_i32 stride_bytes;
    c3x_renderer_i32 clip_left;
    c3x_renderer_i32 clip_top;
    c3x_renderer_i32 clip_right;
    c3x_renderer_i32 clip_bottom;
    c3x_renderer_u32 rendered_tile_count;
    c3x_renderer_u32 fallback_tile_count;
    void const * bgra_pixels;
    c3x_renderer_u32 visible_animation_count;
    c3x_renderer_u32 request_continuous_redraw;
    c3x_renderer_i64 renderer_cpu_ticks;
    c3x_renderer_u32 textured_tile_count;
    c3x_renderer_u32 const * fallback_tile_indices;
    c3x_renderer_u32 const * replacement_tile_flags;
    c3x_renderer_u32 replacement_tile_count;
    c3x_renderer_u32 frame_invalidation_flags;
    c3x_renderer_u32 cache_hits;
    c3x_renderer_u32 cache_misses;
    c3x_renderer_u32 cache_evictions;
    c3x_renderer_u32 cache_stale_rejections;
    c3x_renderer_u32 cache_entries;
    c3x_renderer_u32 cache_capacity;
    c3x_renderer_u32 device_generation;
    c3x_renderer_u32 device_recoveries;
    c3x_renderer_i64 content_revision;
};

struct c3x_renderer_schedule_v1 {
    c3x_renderer_u32 api_version;
    c3x_renderer_u32 struct_size;
    c3x_renderer_i64 now_ticks;
    c3x_renderer_i64 last_presented_ticks;
    c3x_renderer_i64 frequency;
    c3x_renderer_i64 event_start_ticks;
    c3x_renderer_i64 event_duration_ticks;
    c3x_renderer_u32 visible_animation_count;
    c3x_renderer_u32 state_flags;
    c3x_renderer_u32 cadence_ms;
    c3x_renderer_u32 event_loops;
};

struct c3x_renderer_schedule_result_v1 {
    c3x_renderer_u32 api_version;
    c3x_renderer_u32 struct_size;
    c3x_renderer_i64 frame_timestamp_ticks;
    c3x_renderer_u32 phase_millionths;
    c3x_renderer_u32 request_redraw;
    c3x_renderer_u32 dirty_flags;
    c3x_renderer_u32 skipped_frame_count;
    c3x_renderer_u32 rebase_clock;
};

struct c3x_renderer_scene_export_v1 {
    c3x_renderer_u32 api_version;
    c3x_renderer_u32 struct_size;
    char const * output_path;
    char const * fixture_id;
    char const * profile_id;
    c3x_renderer_i32 world_seed;
    c3x_renderer_i32 world_width_tiles;
    c3x_renderer_i32 world_height_tiles;
    c3x_renderer_u32 world_wrap_x;
    c3x_renderer_u32 world_wrap_y;
};
#pragma pack(pop)

typedef c3x_renderer_u32 (*c3x_renderer_get_api_version_fn)(void);
typedef int (*c3x_renderer_set_pack_path_fn)(char const * pack_path);
typedef int (*c3x_renderer_set_definition_paths_fn)(char const * mod_root, char const * default_path,
                                                    char const * scenario_path, char const * custom_path);
typedef int (*c3x_renderer_render_fn)(struct c3x_renderer_frame_v1 const *, struct c3x_renderer_output_v1 *);
typedef int (*c3x_renderer_blit_fn)(struct c3x_renderer_output_v1 const *, void * destination_hdc);
typedef int (*c3x_renderer_export_scene_fn)(struct c3x_renderer_frame_v1 const *, struct c3x_renderer_scene_export_v1 const *);
typedef int (*c3x_renderer_schedule_fn)(struct c3x_renderer_schedule_v1 const *, struct c3x_renderer_schedule_result_v1 *);
typedef void (*c3x_renderer_reset_fn)(void);

#ifdef __cplusplus
}
#endif

#endif
