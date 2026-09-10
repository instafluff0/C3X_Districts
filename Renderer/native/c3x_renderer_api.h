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

#define C3X_RENDERER_API_VERSION 17u

enum c3x_renderer_result {
    C3X_RENDERER_RESULT_ERROR = 0,
    C3X_RENDERER_RESULT_OK = 1,
    C3X_RENDERER_RESULT_BAD_ARGUMENT = 2,
    C3X_RENDERER_RESULT_DEVICE_ERROR = 3,
    C3X_RENDERER_RESULT_PENDING = 4,
    C3X_RENDERER_RESULT_SUPERSEDED = 5,
    C3X_RENDERER_RESULT_PREVIEW = 6
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
    C3X_RENDERER_TILE_CUSTOM_FARM_REPLACED = 4096u,
    // Full authoritative appearance, captured outside the visible draw set.
    // Paired with TOPOLOGY_HALO; permits bounded idle mesh preparation.
    // A source-shadow profile may consume the necessary caster ring in foreground.
    C3X_RENDERER_TILE_PREFETCH = 8192u,
    C3X_RENDERER_TILE_CUSTOM_HUT_REPLACED = 16384u,
    C3X_RENDERER_TILE_CUSTOM_CAMP_REPLACED = 32768u
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
    C3X_RENDERER_IMPROVEMENT_CRATER = 16u,
    C3X_RENDERER_IMPROVEMENT_GOODY_HUT = 32u,
    C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP = 64u
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
    C3X_RENDERER_SCHEDULER_REDRAW_PENDING = 16u,
    C3X_RENDERER_SCHEDULER_PATHFINDER_HOLD = 32u
};

#pragma pack(push, 4)
// Optional body-only ABI, separate from the retained terrain frame contract.
struct c3x_renderer_unit_v1 {
    c3x_renderer_u32 struct_size;
    c3x_renderer_i32 unit_id, action, queued_action, direction;
    c3x_renderer_i32 action_cursor, frame_count;
    c3x_renderer_i32 body_x, body_y, sprite_width, sprite_height, reduced;
    // Thousandths of full-size Civ III unit projection. Zero retains the
    // legacy normal/reduced interpretation for standalone callers.
    c3x_renderer_i32 projection_scale_milli;
    c3x_renderer_i32 hour, season;
    c3x_renderer_u32 display_color_rgb;
    c3x_renderer_i64 presentation_time_ticks, presentation_frequency;
    char unit_key[64];
};

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
    c3x_renderer_i32 barbarian_tribe_id;
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
    // API 14: packed authoritative topology, row-major on Civ III's parity
    // lattice: index=(y*world_width_tiles+x)/2. Low bytes are base, real,
    // river mask, active effect. No anchors or object strings. Optional on the
    // frozen profile; the pickup profile requires complete world coverage.
    c3x_renderer_u32 world_topology_count;
    c3x_renderer_u32 const * world_topology;
    c3x_renderer_i64 world_topology_revision;
};

// Pointer fields borrow renderer-owned storage until the next render,
// pack/definition configuration or reset call. Copy data that must outlive
// those calls. Idle preparation and unit drawing do not invalidate map output.
// The optional camera extension below specifies its additional publication lifetime.
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
    c3x_renderer_u32 geometry_tiles_built;
    c3x_renderer_u32 geometry_tiles_reused;
    c3x_renderer_u32 geometry_tiles_evicted;
    c3x_renderer_u32 geometry_cache_bytes;
    c3x_renderer_u32 geometry_upload_bytes;
    c3x_renderer_i64 geometry_ticks;
    c3x_renderer_i64 draw_ticks;
    c3x_renderer_i64 readback_ticks;
    c3x_renderer_u32 raster_reused_pixels;
    c3x_renderer_u32 raster_draw_pixels;
    // Idle preparation telemetry. Pending belongs to the current snapshot;
    // built/cancelled/ticks are cumulative for this worker lifetime.
    c3x_renderer_u32 prefetch_tiles_pending;
    c3x_renderer_u32 prefetch_tiles_built;
    c3x_renderer_u32 prefetch_tiles_unavailable;
    c3x_renderer_u32 prefetch_tiles_cancelled;
    c3x_renderer_u32 prefetch_cache_bytes;
    c3x_renderer_i64 prefetch_ticks;
    c3x_renderer_u32 raster_cached_pixels;
    c3x_renderer_u32 prefetch_blocks_pending;
    c3x_renderer_u32 prefetch_blocks_built;
    c3x_renderer_u32 pixel_block_cache_bytes;
};

// Optional camera-publication extension. Independently versioned so synchronous
// API 17 callers and older camera callers keep their existing layouts.
#define C3X_RENDERER_CAMERA_VIEW_VERSION 1u
struct c3x_renderer_camera_identity_v1 {
    c3x_renderer_i64 map_epoch;
    c3x_renderer_i64 viewer_epoch;
    c3x_renderer_i64 visibility_epoch;
    c3x_renderer_i64 scene_epoch;
};
struct c3x_renderer_camera_request_v1 {
    c3x_renderer_u32 version, struct_size;
    struct c3x_renderer_frame_v1 const * frame;
    struct c3x_renderer_camera_identity_v1 identity;
};
struct c3x_renderer_camera_view_v1 {
    c3x_renderer_u32 version, struct_size;
    c3x_renderer_i64 ticket;
    struct c3x_renderer_camera_identity_v1 identity;
    // Exact ordered captured occurrences, anchors, clock, zoom and world basis.
    // Topology payload is not a display input: its pointer/count are zero here,
    // while world_topology_revision remains the captured revision.
    struct c3x_renderer_frame_v1 frame;
    struct c3x_renderer_output_v1 output;
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
// Optional experimental camera extension; existing render ABI stays synchronous.
// Call on the same presentation thread as render/blit/reset. Begin copies the
// complete frame and returns PENDING plus a positive ticket. A successful newer
// begin supersedes older tickets; failed begin leaves the previous request alone.
// Poll returns OK only for that exact ticket's complete pixels and ownership.
// With the experimental CAMERA_PREVIEW switch, PREVIEW may return terrain-only
// pixels for that ticket. It is not final success and owns no objects/overlays.
// Continue polling after PREVIEW; its pointers have the same publication lifetime.
// PENDING/error/superseded leave its output untouched. Poll does not advance the
// captured presentation clock. This interface supplies no redraw scheduling.
// Repeating byte-identical scalar fields and ordered payloads (at any caller
// address), with identical view epochs, may return the existing ticket. Begin
// still returns PENDING; poll that ticket to consume its preserved completion.
// Begin expires earlier synchronous borrowed outputs. Poll's borrowed publication
// survives background work, but expires at the next successful publication poll,
// synchronous render, pack/definition change or reset. Copy it for longer use.
// Synchronous render/configuration calls cancel and drain camera work first.
// Unit drawing interrupts active camera work at cancellation boundaries, keeps
// the latest immutable request, and resumes it after the UI-thread unit copy.
// Unit drawing remains synchronous and may wait for an outstanding GPU pass.
// Reset destroys tickets; callers must discard them before using another worker.
typedef int (*c3x_renderer_camera_begin_fn)(struct c3x_renderer_frame_v1 const *, c3x_renderer_i64 * ticket);
typedef int (*c3x_renderer_camera_poll_fn)(c3x_renderer_i64 ticket, struct c3x_renderer_output_v1 *);
typedef int (*c3x_renderer_camera_cancel_fn)(c3x_renderer_i64 ticket);
// Begin-view copies the frame and the caller's authoritative lifecycle epochs.
// Poll-view publishes those identities, captured occurrences and ownership with
// the pixels in one transaction; all pointers share the lifetime above. A
// consumer must reject epochs incompatible with its current visibility/viewer
// before copying any pixels or reading replacement flags. Flags index the
// returned frame.tiles, never a later capture array. This does not schedule a
// native redraw or authorize displaying an older camera with current overlays.
typedef int (*c3x_renderer_camera_begin_view_fn)(struct c3x_renderer_camera_request_v1 const *, c3x_renderer_i64 * ticket);
typedef int (*c3x_renderer_camera_poll_view_fn)(c3x_renderer_i64 ticket, struct c3x_renderer_camera_view_v1 *);
typedef int (*c3x_renderer_blit_fn)(struct c3x_renderer_output_v1 const *, void * destination_hdc);
typedef int (*c3x_renderer_unit_draw_fn)(struct c3x_renderer_unit_v1 const *, void * destination_hdc);
/* Optional extension: native underlay resolves color-key canvas antialiasing. */
typedef int (*c3x_renderer_unit_draw_background_fn)(struct c3x_renderer_unit_v1 const *, void * destination_hdc, void * background_hdc);
/* Optional extension; writes drawn left/top/right/bottom only on success. */
typedef int (*c3x_renderer_unit_draw_expanded_fn)(struct c3x_renderer_unit_v1 const *, void * destination_hdc, void * background_hdc, int * bounds_ltrb);
// Optional unit configuration. Set before definition/asset loading; defaults off.
typedef int (*c3x_renderer_set_unit_rendering_fn)(int enabled);
typedef int (*c3x_renderer_export_scene_fn)(struct c3x_renderer_frame_v1 const *, struct c3x_renderer_scene_export_v1 const *);
typedef int (*c3x_renderer_schedule_fn)(struct c3x_renderer_schedule_v1 const *, struct c3x_renderer_schedule_result_v1 *);
typedef void (*c3x_renderer_reset_fn)(void);

#ifdef __cplusplus
}
#endif

#endif
