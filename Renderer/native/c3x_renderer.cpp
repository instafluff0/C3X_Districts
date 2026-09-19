#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <array>
#include <cctype>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "c3x_renderer_api.h"
#include "terrain_scene_runtime.h"
#include "object_compiler.h"
#include "animation_runtime.h"
#include "benchmark_oracle.h"
#include "environment_runtime.h"
#include "terrain_definition_runtime.h"
#include "renderer_trace.h"
#include "gpu_frame_api.h"
#include "gpu_composition_session.h"
#include "gpu_visibility.h"
#include "render_core/dynamic_scene_input.h"
#include "gpu_native_presenter.h"
#include "native_screen_bridge.h"
#include "native_observation.h"
#include "native_lifetime_registry.h"
#include "native_composition_owner.h"
#include "asset_content_hash.h"
#include "scroll_damage.h"
#include "river_node_locality.h"
#include "pixel_block_cache.h"
#include "render_core/render_region_cache.h"
#include "render_core/region_contributor_index.h"
#include "render_core/world_pass_index.h"
#include "render_core/center_shore_cache.h"
#include "render_core/projected_mesh_bounds.h"
#include "color_quantization.h"
#include "render_core/terrain_query.h"
#include "render_core/world_coast.h"
#include "render_core/captured_scene.h"
#include "prepared_view_area.h"
#include "render_core/geometry_draws.h"
#include "render_core/draw_parameter_stream.h"
#include "render_core/immutable_mesh_upload.h"
#include "render_core/prepared_mesh.h"
#include "render_core/resource_instances.h"
#include "render_core/scene_depth.h"
#include "render_core/scene_surface.h"
#include "render_core/scene_guard.h"
#include "render_core/coastal_waves.h"
#include "render_core/relief_query.h"
#include "render_core/exact_point_cache.h"
#include "render_core/frame_telemetry.h"
#include "render_core/raster_grid.h"
#include "render_core/water_coverage.h"
#include "render_core/wave_retention.h"
#include "render_core/cliff_placement.h"
#include "render_core/source_shadow.h"
#include "render_core/linear_target.h"
#include "source_fidelity/runtime.h"
#include "source_fidelity/light_frame.h"
#include "source_fidelity/terrain_compiler.h"
#include "source_fidelity/cliff_compiler.h"
#include "source_fidelity/ground_preparation.h"
#include "source_fidelity/surface_query_scratch.h"
#include "environment_refresh/reflection.h"
#include "unit_body_renderer.h"
#include "render_core/unit_frame_preparation.h"
#include "render_core/unit_instances.h"
#include "city_fidelity/gpu.h"
#include "city_fidelity/compiler.h"
#include "world_preparation.h"
#include "city_fidelity/glow.h"
#include "../lab/shared/natural/ground.h"
#include "../lab/shared/natural/queries.h"
#include "../lab/shared/natural/relief.h"
#include "../lab/shared/natural/mesh.h"

namespace {

constexpr c3x_renderer_u32 viewport_cache_capacity = 32u;
// Isolated benchmark experiment only; normal/game builds retain their budgets.
#ifdef C3X_RENDERER_BENCHMARK_LARGE_CACHE
constexpr std::size_t default_viewport_cache_budget = 128u * 1024u * 1024u;
constexpr std::size_t natural_mesh_cache_budget = 192u * 1024u * 1024u;
constexpr std::size_t natural_mesh_cache_capacity = 2048u;
constexpr std::size_t default_resource_backdrop_cache_budget = 288u * 1024u * 1024u;
constexpr std::size_t tile_geometry_cache_budget = C3X_RENDERER_BENCHMARK_GPU_CACHE_MIB * 1024u * 1024u;
constexpr std::size_t tile_geometry_cache_capacity = C3X_RENDERER_BENCHMARK_GPU_CACHE_MIB / 192u * 4096u;
#else
// Reassign 96 MiB of the former bitmap budget to reusable world mesh data.
constexpr std::size_t default_viewport_cache_budget = 32u * 1024u * 1024u;
constexpr std::size_t natural_mesh_cache_budget = 96u * 1024u * 1024u;
constexpr std::size_t natural_mesh_cache_capacity = 1024u;
constexpr std::size_t default_resource_backdrop_cache_budget = 128u * 1024u * 1024u;
static_assert(default_viewport_cache_budget+natural_mesh_cache_budget==128u*1024u*1024u,
    "World mesh reuse must not increase the combined CPU cache budget");
// Includes active buffers: a frame pins its entries, never a second copy.
// The 2240x1192 witness exceeds the old 192 MiB cap in its first view.
// User-authorized modern-machine tier; keep CPU caches at their separate caps.
// A compile-time override retains explicit pressure testing of smaller tiers.
#ifndef C3X_RENDERER_GPU_GEOMETRY_MIB
#define C3X_RENDERER_GPU_GEOMETRY_MIB 768
#endif
static_assert(C3X_RENDERER_GPU_GEOMETRY_MIB>=192 && C3X_RENDERER_GPU_GEOMETRY_MIB<=1024,
    "Geometry tier must leave room for other resources in the x86 host");
constexpr std::size_t tile_geometry_cache_budget = C3X_RENDERER_GPU_GEOMETRY_MIB * 1024u * 1024u;
constexpr std::size_t tile_geometry_cache_capacity = C3X_RENDERER_GPU_GEOMETRY_MIB / 192u * 4096u;
#endif

using Vertex = c3x_renderer::fidelity::MapVertex;

// The production shader exposes only settings that are meaningful to the game.
// Standalone fixture and promotion switches are compile-time concerns outside
// this runtime contract.
struct TerrainShaderSettings {
    float height_texel[2];
    float normal_strength;
    float exposure;
    float light_direction[3];
    float sun_intensity;
    float sun_color[3];
    float shadow_strength;
    float moon_direction[3];
    float moon_intensity;
    float moon_color[3];
    float night_activation;
    float ambient_color[3];
    float environment_exposure;
    float water_fresnel;
    float water_specular;
    float emissive_scale;
    float hour;
};

struct ViewportShaderSettings {
    float translation[2];
    float depth_translation;
    float padding;
    float inverse_size[2];
    float reserved[2];
    float natural_projection[4];
};

struct TerrainTexture : c3x_renderer::fidelity::ReliefFields {
    ID3D11ShaderResourceView * view = nullptr;
    ID3D11ShaderResourceView * material_height_view = nullptr;
    ID3D11ShaderResourceView * specular_view = nullptr;
    ID3D11ShaderResourceView * elevated_view = nullptr;
    ID3D11ShaderResourceView * elevated_height_view = nullptr;
    ID3D11ShaderResourceView * elevated_specular_view = nullptr;
    std::array<ID3D11ShaderResourceView *, 5> relief_layer_views = {};
    std::array<ID3D11ShaderResourceView *, 5> water_surface_views = {};
    std::vector<std::uint8_t> dds;
    std::vector<std::uint8_t> material_height_dds;
    std::vector<std::uint8_t> specular_dds;
    std::vector<std::uint8_t> elevated_dds;
    std::vector<std::uint8_t> elevated_height_dds;
    std::vector<std::uint8_t> elevated_specular_dds;
    std::array<std::vector<std::uint8_t>, 5> relief_layer_dds;
    std::array<std::vector<std::uint8_t>, 5> water_surface_dds;
    float height_scale_px = 0.0f;
    int relief_profile = 0;
    bool configured = false;
};

struct CachedViewport {
    c3x_renderer::TerrainFrameSignature signature;
    std::vector<std::uint32_t> pixels;
    std::vector<c3x_renderer_tile_v1> tiles;
    std::vector<c3x_renderer_u32> replacement_flags;
    // Weak identities only: retaining a bitmap must not pin GPU allocations.
    std::vector<c3x_renderer::render_core::ContentHandle> tile_keys;
    std::vector<c3x_renderer_u32> fallback_indices;
    c3x_renderer_u32 rendered_tile_count = 0;
    c3x_renderer_u32 fallback_tile_count = 0;
    c3x_renderer_u32 textured_tile_count = 0;
    std::size_t byte_count = 0;
};

struct CachedGeometry {
    c3x_renderer::TerrainFrameSignature signature;
    std::vector<c3x_renderer_tile_v1> tiles;
    std::vector<c3x_renderer_u32> replacement_flags;
    std::vector<c3x_renderer_u32> fallback_indices;
    std::vector<c3x_renderer::render_core::ContentHandle> tile_keys;
    c3x_renderer_u32 rendered_tile_count = 0;
    c3x_renderer_u32 fallback_tile_count = 0;
    c3x_renderer_u32 textured_tile_count = 0;
    bool valid = false;

    void clear() {
        tiles.clear();
        replacement_flags.clear();
        fallback_indices.clear();
        tile_keys.clear();
        rendered_tile_count = fallback_tile_count = textured_tile_count = 0;
        valid = false;
    }
};

struct CachedVertexChunk {
    ID3D11Buffer * buffer = nullptr;
    ID3D11Buffer * indices = nullptr;
    UINT index_count = 0;
    DXGI_FORMAT index_format = DXGI_FORMAT_R32_UINT;
    UINT vertex_stride = 120;
    UINT vertex_offset = 0, index_offset = 0;
    std::size_t byte_count = 0;
    int translation_x = 0, translation_y = 0;
    D3D11_RECT bounds = {};
    c3x_renderer::render_core::SourceShadow::Bounds world_bounds;
    c3x_renderer::render_core::ProjectedMeshBounds projected_bounds;
    std::uint64_t version = 0;
    ID3D11ShaderResourceView * animation_texture = nullptr; // borrowed, dynamic pass only
    ID3D11Buffer * resource_instance = nullptr; // borrowed instance constants, dynamic pass only
    std::shared_ptr<std::vector<c3x_renderer::fidelity::MeshInstance> const> instances;
    float instance_material=40;
    float visual_time=-1; // Optional wave sample; negative follows the visible clock.
    unsigned city_material=0xffffffffu;
    bool city_environment=false;
    float city_atlas[4]={};
    std::shared_ptr<c3x_renderer::city_fidelity::Lighting> city_lighting;
    float natural_projection[4] = {};
    // 1: natural world; 2: normalized feature/height; 3: normalized ground;
    // 4: authored world content with the viewport-defined depth basis.
    unsigned projection_kind=0;
    int source_tile_width=0;
};

using PendingCityChunk=c3x_renderer::city_fidelity::Chunk;

// Own one immutable coast-cell occurrence. Raw lattice coordinates preserve
// wrapped world/shadow coordinates; the viewport borrows references and anchors.
struct RetainedWaveCell {
    CachedVertexChunk chunk;
    std::uint64_t used=0;
    std::size_t bytes=0;
    RetainedWaveCell()=default;
    RetainedWaveCell(RetainedWaveCell const&)=delete;
    RetainedWaveCell& operator=(RetainedWaveCell const&)=delete;
    RetainedWaveCell(RetainedWaveCell&& other) noexcept
        :chunk(std::move(other.chunk)),used(other.used),bytes(other.bytes) {
        other.chunk.buffer=nullptr;other.chunk.indices=nullptr;
    }
    ~RetainedWaveCell(){if(chunk.buffer)chunk.buffer->Release();if(chunk.indices)chunk.indices->Release();}
};

struct ResourceAnimation {
    std::string name;
    c3x_renderer::AnimationMesh mesh;
    c3x_renderer::render_core::ResourceSourceBounds source_bounds;
    ID3D11Buffer * vertices = nullptr;
    std::vector<std::uint8_t> dds;
    ID3D11ShaderResourceView * view = nullptr;
    ID3D11Buffer * indices = nullptr;
    float scale = 1, yaw = 0, offset[3] = {};
    unsigned count = 1;
};
struct ResourceAnchor {
    unsigned asset = 0, seed = 0;
    float u = .5f, v = .5f, world_u = 0, world_v = 0, ground = 0;
    int anchor_x = 0, anchor_y = 0, tile_x = 0, tile_y = 0;
};
struct ResourceBackdrop {
    int x=0,y=0;
    std::int64_t depth_origin=0;
    ID3D11Texture2D * color=nullptr,* depth=nullptr;
    std::uint64_t signature=0,used=0;
    std::size_t bytes=0;
    c3x_renderer::render_core::RenderRegionKey dependencies;
};
struct ResourceBuffer {
    ID3D11Buffer * vertices = nullptr;
    ID3D11Buffer * shadow_vertices = nullptr;
    unsigned capacity = 0, shadow_capacity = 0;
};

enum GeometryLayer : std::size_t {
    geometry_underlay,
    geometry_land,
    geometry_bed,
    geometry_water,
    geometry_river,
    geometry_route,
    geometry_shadow,
    geometry_wave,
    geometry_feature,
    geometry_city,
    geometry_wall,
    geometry_mine,
    geometry_farm,
    geometry_site,
    geometry_cliff0, geometry_cliff1, geometry_cliff2, geometry_cliff3,
    geometry_cliff4, geometry_cliff5, geometry_cliff6, geometry_cliff7,
    geometry_natural_terrain, geometry_natural_decal, geometry_natural_mountain,
    geometry_natural_forest0,
    geometry_layer_count = geometry_natural_forest0 + 22
};

using GeometryDrawRecord=c3x_renderer::render_core::GeometryDrawRecord<CachedVertexChunk>;
using GeometryDrawView=c3x_renderer::render_core::GeometryDrawView<CachedVertexChunk,geometry_layer_count>;
using GeometryDrawReference=GeometryDrawView::Reference;
static_assert(sizeof(GeometryDrawRecord)<=sizeof(CachedVertexChunk)/2,"occurrence metadata must be smaller than owned content");

struct CachedTileGeometry {
    std::uint64_t signature = 0, version = 0;
    std::array<std::uint64_t,20> compile_context={};
    std::uint64_t validity_epoch=0;
    int validity_anchor_x=0,validity_anchor_y=0;
    bool validity=false;
    // Bindings borrow the existing owner; eviction invalidates their generation.
    c3x_renderer::render_core::ContentHandle binding, natural_content;
    bool shared_natural = false;
    bool world_objects=false,world_ground=false;
    int source_tile_width=0;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> appearance_dependencies;
    int tile_x = 0, tile_y = 0;
    std::vector<ResourceAnchor> resource_anchors;
    bool replaces_resource = false;
    bool prefetched = false;
    CachedTileGeometry() = default;
    CachedTileGeometry(CachedTileGeometry const &) = delete;
    CachedTileGeometry(CachedTileGeometry &&) noexcept = default;
    ~CachedTileGeometry() {
        for (auto & layer : buffers) for (auto & chunk : layer) {
            if (chunk.buffer) chunk.buffer->Release();
            if (chunk.indices) chunk.indices->Release();
        }
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> dependencies;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> coast_dependencies;
    std::vector<std::pair<std::size_t, std::uint32_t>> world_dependencies;
    c3x_renderer::fidelity::NaturalWorld::CellProof river_dependencies;
    std::array<std::vector<CachedVertexChunk>, geometry_layer_count> buffers;
    std::vector<std::pair<std::uint64_t, std::array<int, 2>>> anchor_dependencies;
    std::size_t byte_count = 0;
    std::uint64_t last_used = 0;
    std::uint64_t animation_epoch = 0;
    int anchor_x = 0, anchor_y = 0;
};

// Authored natural meshes are independent of the screen projection. Retain
// indexed material/world data across zoom levels; only their three projected
// coordinates need rebuilding. This CPU tier has its own strict memory cap.
struct NaturalMesh {
    std::vector<std::array<float,23>> vertices;
    std::vector<UINT> indices;
};
struct NaturalTile {
    std::array<NaturalMesh, geometry_layer_count-geometry_natural_terrain> layers;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> dependencies, coast_dependencies, appearance_dependencies;
    std::vector<std::pair<std::size_t,std::uint32_t>> world_dependencies;
    c3x_renderer::fidelity::NaturalWorld::CellProof river_dependencies;
    std::size_t bytes=0;
    std::uint64_t used=0;
    int tile_width=0,tile_height=0,target_height=0,tile_x=0,tile_y=0;
};

// Hash bytes, then compare all bytes: repeated triangle vertices share storage
// without merging seams, normals, materials, or merely similar positions.
using VertexHash=c3x_renderer::render_core::VertexHash;
using VertexEqual=c3x_renderer::render_core::VertexEqual;

// GroundPoint/CachedGroundGrid now live in source_fidelity/ground_compiler.h,
// the single definition shared with compile_ground_surfaces; alias both
// names here since the rest of this file (CachedGroundTile, cache eviction)
// still refers to them unqualified.
using GroundPoint = c3x_renderer::fidelity::GroundPoint;
using CachedGroundGrid = c3x_renderer::fidelity::CachedGroundGrid;
struct CachedGroundTile {
    std::uint64_t signature=0,used=0;
    int x=0,y=0;
    std::size_t bytes=0;
    // A shared_ptr, not an owned vector: a future worker job may capture a
    // copy of this handle before it starts (see the ground call site) so the
    // grids stay alive even if this map entry is evicted by a *different*
    // tile's admission logic while that job is still reading them -- the
    // same lease pattern std::shared_ptr provides everywhere else, applied
    // here since CachedGroundTile is ground-exclusive state.
    std::shared_ptr<std::vector<CachedGroundGrid>> grids;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> dependencies,coast_dependencies;
    std::vector<std::pair<std::size_t,std::uint32_t>> world_dependencies;
    c3x_renderer::fidelity::NaturalWorld::CellProof river_dependencies;
};

using RiverNode=c3x_renderer::fidelity::GroundRiverNode;

struct SceneTopology : c3x_renderer::render_core::CapturedScene {
    std::uint64_t signature = 0;
    std::vector<RiverNode> rivers;
};

// Locate a direct member within one JSON object. Never cross into a nested
// authored layer or a sibling when a channel is absent. Material files are
// key-order independent (the selected skin sorts authored_layers before base).
std::size_t json_member_position(std::vector<std::uint8_t> const& data,
                                char const* key, std::size_t start=0) {
    auto whitespace=[&](std::size_t& p) {while(p<data.size() && std::isspace(data[p])) ++p;};
    auto string_end=[&](std::size_t p) {
        for(++p;p<data.size();++p) {
            if(data[p]=='\\') {if(++p>=data.size())break;}
            else if(data[p]=='"')return p;
        }
        return std::string::npos;
    };
    if(start>=data.size())return std::string::npos;
    whitespace(start);
    if(start<data.size() && data[start]=='"') {
        start=string_end(start);if(start==std::string::npos)return start;
        ++start;whitespace(start);
        if(start>=data.size() || data[start++]!=':')return std::string::npos;
        whitespace(start);
    }
    if(start>=data.size() || data[start]!='{')return std::string::npos;
    int depth=1;
    for(std::size_t p=start+1;p<data.size();++p) {
        if(data[p]=='"') {
            auto end=string_end(p);if(end==std::string::npos)return end;
            auto after=end+1;whitespace(after);
            if(depth==1 && after<data.size() && data[after]==':' &&
                end-p-1==std::strlen(key) && std::memcmp(data.data()+p+1,key,end-p-1)==0)return p;
            p=end;
        } else if(data[p]=='{' || data[p]=='[')++depth;
        else if(data[p]=='}' || data[p]==']') {if(--depth==0)break;}
    }
    return std::string::npos;
}

class RendererState {
public:
    // Geometry/depth units belong to the native viewport, independently of
    // the larger working surface used for optional nearby preparation.
    int content_view_width=0,content_view_height=0;
    // Output eligibility is separate from the retained static working extent.
    D3D11_RECT selected_output={};
    bool selected_output_active=false,selected_output_changed=false;
    bool gpu_output_mode=false,cpu_output_stale=false;
    bool visibility_pass=false;
    c3x_renderer::GpuVisibility visibility_gpu;
    c3x_renderer::render_core::VisibilityCoverage visibility_coverage;
    std::vector<std::uint32_t> visibility_pixels;
    ID3D11Texture2D* gpu_map_texture=nullptr;
    bool gpu_map_valid=false;
    std::uint64_t scene_generation=0,unit_scene_captures=0,unit_scene_rejections=0,unit_scene_evictions=0;
    std::vector<std::weak_ptr<c3x_renderer::UnitSceneRegion>> unit_scene_regions;
    std::shared_ptr<std::size_t> unit_scene_bytes=std::make_shared<std::size_t>(0);
    c3x_renderer::render_core::LinearTarget unit_scene_work;
    unsigned frame_output_readbacks=0;
    std::unique_ptr<c3x_gpu_images::Session> gpu_composition;
    std::int64_t gpu_serial=0;

    std::vector<D3D11_RECT> scene_pending_finish;
    std::size_t viewport_cache_budget=default_viewport_cache_budget;
    std::size_t resource_backdrop_cache_budget=default_resource_backdrop_cache_budget;
    c3x_renderer::UnitBodyRenderer unit_bodies;
    bool unit_rendering_enabled=false;
    bool fidelity_profile = false, fidelity_shadow_control = false;
    bool environment_profile = false;
    bool city_profile=false;
    bool clip_dirty_blocks=false;
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
    unsigned diagnostic_routes=0; // 1: omit surface draws; 2: also omit surface construction.
    bool diagnostic_half_pixels=false;
    unsigned diagnostic_animation=0; // Cumulative benchmark-only GPU phase ablations.
#endif
    bool bounded_post=false;
    bool shared_scene_surface=false,scene_surface_requested=false;
    c3x_renderer::render_core::LinearTarget scene_scratch;
    std::vector<D3D11_RECT> scene_dynamic_damage;
    std::uint64_t scene_static_signature=0;
    c3x_renderer::render_core::LinearRestore scene_restore;
    std::int64_t scene_static_depth_origin=0;
    bool scene_overlap=false;int scene_dx=0,scene_dy=0;
    std::vector<D3D11_RECT> scene_damage;
    c3x_renderer::render_core::SceneGuard<D3D11_RECT> scene_guard;
    c3x_renderer::render_core::RenderRegionKey scene_guard_context;
    int scene_guard_pad=0;bool scene_guard_failed=false;
    std::uint64_t scene_guard_depth_origin=0;
    int scene_region_size=128;
    int scene_region_height=128;
    bool cull_empty_water=false;
    bool world_backdrops=false,backdrop_reuse_control=false;
    bool world_waves=false,wave_reuse_control=false;
    bool animation_readback_atlas=false;
    bool world_raster_grid=false;
    bool world_regions=false,world_regions_control=false,local_region_revisions=false;
    bool region_diagnostics=false;
    bool region_receiver_shadows=false;
    bool composition_receiver_index=false;
    bool tight_natural_bounds=false;
    int region_input_ring=2;
    std::int64_t region_origin_x=0,region_origin_y=0;
    c3x_renderer::render_core::RenderRegionCache<ID3D11Texture2D> render_regions;
    c3x_renderer::render_core::RenderRegionKey region_context;
    c3x_renderer::render_core::SourceShadow::PreparedCasters retained_region_casters;
    c3x_renderer::render_core::RegionContributorIndex region_contributors;
    c3x_renderer::render_core::WorldPassIndex world_pass_index;
    std::unordered_map<std::uintptr_t,std::vector<std::pair<unsigned,unsigned>>> world_pass_occurrences;
    bool world_pass_affine=false;
    double world_pass_x=0,world_pass_y=0;
    float world_pass_reflection=0;

    c3x_renderer::render_core::CenterShoreCache center_shore_cache;
    double frame_center_shore_ms=0;
    std::size_t frame_region_hits=0,frame_region_misses=0,frame_region_hit_pixels=0,frame_region_rejected=0;
    std::array<double,3> frame_region_phase_ms{}; // contributors, lights, shadow dependencies
    double frame_tile_validation_ms=0,frame_tile_append_ms=0,frame_topology_ms=0;

    c3x_renderer::city_fidelity::Gpu cities;
    c3x_renderer::city_fidelity::Glow city_glow;
    c3x_renderer::city_fidelity::Glow region_glow;
    c3x_renderer::environment_refresh::Reflection reflection;
    c3x_renderer::environment_refresh::Reflection region_reflection;
    std::string fidelity_root;
    c3x_renderer::fidelity::Natural natural;
    c3x_renderer::fidelity::TerrainPreparation terrain_preparation;
    c3x_renderer::fidelity::GroundTask::Queue ground_preparation;
    c3x_renderer::fidelity::GroundPreparation selected_ground_preparation;
    std::array<c3x_renderer::fidelity::TerrainCompileScratch,6> terrain_scratch;
    std::array<c3x_renderer::fidelity::SurfaceQueryScratch,6> world_ground_scratch;
    c3x_renderer::fidelity::TerrainCompileScratch foreground_terrain_scratch;
    // Cliff generation's own private copy of the ground/terrain query
    // machinery (SurfaceQueries/ReliefSurface/NaturalWorld caches), so it can
    // eventually query height/shore/river data without sharing the per-tile
    // `queries`/`pickup_surface` ground uses. A persistent member (not a
    // per-frame local) to match foreground_terrain_scratch and avoid
    // rebuilding its river-page cache every frame.
    c3x_renderer::fidelity::SurfaceQueryScratch cliff_query_scratch;
    // Ground owns query scratch per task; the scoped join precedes source mutation.
    unsigned cpu_terrain_workers=0;
    bool world_preparation=false;
    std::size_t cpu_preparation_budget=16u*1024u*1024u;
    RendererTrace trace;
    c3x_renderer::render_core::GpuFrameTelemetry gpu_telemetry;
    bool profiling=false;
    using AnimationGpu=c3x_renderer::render_core::GpuAnimationTelemetry;
    AnimationGpu animation_gpu;
    void poll_animation_gpu(){
        animation_gpu.poll(context,[&](AnimationGpu::Sample const& sample){
            UINT64 covered=0;unsigned count=0;
            for(unsigned i=0;i<AnimationGpu::phase_count;++i){covered+=sample.ticks[i];count+=sample.counts[i];}
            double scale=sample.valid?1000.0/sample.frequency:0;
            char detail[768];sprintf_s(detail,
                "sample_sequence=%llu valid=%u frequency=%llu gpu_span_ms=%.6f unassigned_span_ms=%.6f "
                "background_ms=%.6f import_ms=%.6f receivers_ms=%.6f shadow_ms=%.6f body_ms=%.6f finish_ms=%.6f transfer_ms=%.6f "
                "background_spans=%u import_spans=%u receivers_spans=%u shadow_spans=%u body_spans=%u finish_spans=%u transfer_spans=%u spans=%u ring_skipped=%u",
                sample.sequence,unsigned(sample.valid),sample.frequency,sample.total*scale,
                (sample.total>=covered?sample.total-covered:0)*scale,
                sample.ticks[0]*scale,sample.ticks[1]*scale,sample.ticks[2]*scale,sample.ticks[3]*scale,
                sample.ticks[4]*scale,sample.ticks[5]*scale,sample.ticks[6]*scale,
                sample.counts[0],sample.counts[1],sample.counts[2],sample.counts[3],sample.counts[4],sample.counts[5],sample.counts[6],count,sample.skipped);
            trace.write("animation-gpu-phases",detail,true);
        });
    }

    std::size_t sampled_geometry_bucket=~std::size_t(0);
    std::uint64_t frame_draw_calls=0,frame_parameter_updates=0,frame_bounds_tests=0;
    std::uint64_t frame_pass_setups=0,frame_active_layers=0;
    double frame_geometry_issue_ms=0,frame_scene_select_ms=0,frame_scene_execute_ms=0;
    unsigned frame_content_uploads=0,frame_prepared_meshes=0,frame_foreground_meshes=0;
    std::size_t frame_prepared_vertex_bytes=0;
    unsigned frame_caster_preparations=0;
    std::size_t frame_post_lanes=0;
    void memory_sample(char const* phase) {
        if(!profiling)return;
        auto sample=c3x_renderer::render_core::AddressSpaceSample::capture();
        std::size_t view_draw_bytes=sizeof(geometry_vertex_buffers);
        for(auto const& layer:geometry_vertex_buffers)view_draw_bytes+=layer.capacity()*sizeof(GeometryDrawRecord);
        char detail[640];sprintf_s(detail,
            "phase=%s available_virtual=%llu largest_free_region=%llu committed_va=%llu reserved_va=%llu "
            "gpu_geometry=%zu gpu_geometry_cap=%zu natural_cpu=%zu ground_cpu=%zu natural_cpu_cap=%zu "
            "viewport_cpu=%zu viewport_cpu_cap=%zu backdrop_gpu=%zu backdrop_gpu_cap=%zu pixels_capacity=%zu scene_cpu=%zu content_bindings_cpu=%zu view_draw_cpu=%zu",
            phase,sample.available,sample.largest,sample.committed,sample.reserved,
            tile_geometry_cache_bytes,tile_geometry_cache_budget,natural_mesh_cache_bytes,ground_grid_cache_bytes,
            natural_mesh_cache_budget,viewport_cache_bytes,viewport_cache_budget,resource_backdrop_bytes,
            resource_backdrop_cache_budget,pixels.capacity()*sizeof(pixels[0]),topology_cache.bytes(),resident_content.bytes(),view_draw_bytes);
        trace.write("memory-sample",detail,true);
        sprintf_s(detail,"linear_frame=%zu linear_block=%zu reflection=%zu glow=%zu region_reflection=%zu region_glow=%zu wave_geometry=%zu",
            linear_frame.bytes(),linear_block.bytes(),reflection.linear.bytes(),city_glow.linear.bytes(),
            region_reflection.linear.bytes(),region_glow.linear.bytes(),wave_geometry_bytes);
        trace.write("memory-linear-scratch",detail,true);
    }
    SceneTopology topology_cache;
    bool retained_world=true;
    c3x_renderer::fidelity::PatchLayouts patch_layouts;
    c3x_renderer::fidelity::PatchDetail patch_detail;
    unsigned patch_pixels=0;
    bool tree_instances_enabled=false;
    std::map<unsigned,ID3D11Buffer*> terrain_patch_indices;
    std::size_t terrain_patch_index_bytes=0;
    unsigned frame_patch_index_reuses=0;
    unsigned frame_instances_ready=0,frame_world_object_hits=0,frame_world_object_builds=0;
    std::uint64_t requested_signature = 0;
    char const * frame_cache_path = "cold";
    c3x_renderer_i64 frame_geometry_ticks = 0, frame_draw_ticks = 0, frame_readback_ticks = 0;
    ID3D11Device * device = nullptr;
    ID3D11DeviceContext * context = nullptr;
    ID3D11VertexShader * vertex_shader = nullptr;
    ID3D11PixelShader * pixel_shader = nullptr;
    ID3D11VertexShader * feature_vertex_shader = nullptr;
    ID3D11PixelShader * feature_pixel_shader = nullptr;
    ID3D11PixelShader * resource_shadow_shader = nullptr;
    ID3D11VertexShader * resource_body_vertex_shader=nullptr,*resource_shadow_vertex_shader=nullptr;
    ID3D11InputLayout * resource_input_layout=nullptr;
    ID3D11InputLayout * input_layout = nullptr;
    ID3D11InputLayout * feature_input_layout = nullptr;
    ID3D11Buffer * terrain_settings_buffer = nullptr;
    ID3D11Buffer * viewport_settings_buffer = nullptr;
    c3x_renderer::render_core::DrawParameterStream draw_parameters;
    ID3D11Buffer * world_settings_buffer = nullptr;
    ID3D11Buffer * shadow_settings_buffer = nullptr;
    bool pickup_profile = false;
    c3x_renderer::render_core::SourceShadow source_shadow;
    std::array<float,12> shadow_basis{};
    int shadow_tile_width=128,shadow_tile_height=64;
    c3x_renderer::render_core::WorldCoast world_coast;
    c3x_renderer_i64 geometry_world_revision = -1;
    float display_exposure = 1.0f;
    c3x_renderer::render_core::LinearTarget linear_frame, linear_block;
    c3x_renderer::render_core::LinearOutput linear_output;
    ID3D11BlendState * blend_state = nullptr;
    ID3D11DepthStencilState * depth_state = nullptr;
    ID3D11RasterizerState * rasterizer_state = nullptr;
    ID3D11SamplerState * natural_wrap = nullptr, *natural_clamp = nullptr;
    ID3D11SamplerState * terrain_sampler = nullptr;
    ID3D11SamplerState * decal_sampler = nullptr;
    ID3D11Texture2D * render_texture = nullptr;
    ID3D11RenderTargetView * render_target = nullptr;
    ID3D11Texture2D * depth_texture = nullptr;
    ID3D11DepthStencilView * depth_target = nullptr;
    ID3D11Texture2D * readback_texture = nullptr;
    ID3D11Texture2D * animation_readback_texture = nullptr;
    unsigned animation_readback_width=0,animation_readback_height=0;
    int width = 0;
    int height = 0;
    std::vector<std::uint32_t> pixels;
    std::vector<c3x_renderer_u32> fallback_tile_indices;
    std::vector<c3x_renderer_u32> replacement_tile_flags;
    std::array<TerrainTexture, c3x_renderer::terrain_type_count> terrain_textures;
    TerrainTexture dune_surface;
    c3x_renderer::FeatureBundle feature_bundle;
    std::array<std::vector<std::uint8_t>, 8> feature_texture_dds;
    std::array<ID3D11ShaderResourceView *, 8> feature_texture_views = {};
    std::array<ID3D11ShaderResourceView*,128> material_views{};
    bool material_views_valid=false;
    // Extra terrain channels not already represented by TerrainTexture are
    // held here and bound to the shared shader's stable register contract.
    std::array<std::vector<std::uint8_t>, 19> terrain_extra_dds;
    std::array<ID3D11ShaderResourceView *, 19> terrain_extra_views = {};
    bool terrain_extra_assets_ready = false;
    bool authored_relief_assets_ready = false;
    std::string integrated_shader_path = "integrated_terrain.hlsl";
    std::vector<std::uint8_t> dune_decal_base_dds;
    std::vector<std::uint8_t> dune_decal_height_dds;
    ID3D11ShaderResourceView * dune_decal_base_view = nullptr;
    ID3D11ShaderResourceView * dune_decal_height_view = nullptr;
    std::vector<std::uint8_t> marsh_decal_base_dds;
    std::vector<std::uint8_t> marsh_decal_height_dds;
    std::vector<std::uint8_t> marsh_decal_specular_dds;
    ID3D11ShaderResourceView * marsh_decal_base_view = nullptr;
    ID3D11ShaderResourceView * marsh_decal_height_view = nullptr;
    ID3D11ShaderResourceView * marsh_decal_specular_view = nullptr;
    std::vector<std::uint8_t> volcano_base_dds;
    std::vector<std::uint8_t> volcano_height_dds;
    std::vector<std::uint8_t> volcano_active_base_dds;
    std::vector<std::uint8_t> volcano_active_specular_dds;
    ID3D11ShaderResourceView * volcano_base_view = nullptr;
    ID3D11ShaderResourceView * volcano_height_view = nullptr;
    ID3D11ShaderResourceView * volcano_active_base_view = nullptr;
    ID3D11ShaderResourceView * volcano_active_specular_view = nullptr;
    std::vector<std::uint8_t> water_clutter_base_dds;
    std::vector<std::uint8_t> water_clutter_height_dds;
    std::vector<std::uint8_t> grass_clutter_base_dds;
    std::vector<std::uint8_t> grass_clutter_height_dds;
    std::vector<std::uint8_t> plains_clutter_base_dds;
    std::vector<std::uint8_t> plains_clutter_height_dds;
    ID3D11ShaderResourceView * water_clutter_base_view = nullptr;
    ID3D11ShaderResourceView * water_clutter_height_view = nullptr;
    ID3D11ShaderResourceView * grass_clutter_base_view = nullptr;
    ID3D11ShaderResourceView * grass_clutter_height_view = nullptr;
    ID3D11ShaderResourceView * plains_clutter_base_view = nullptr;
    ID3D11ShaderResourceView * plains_clutter_height_view = nullptr;
    std::array<std::vector<std::uint8_t>, 10> river_surface_dds;
    std::array<ID3D11ShaderResourceView *, 10> river_surface_views = {};
    c3x_renderer::FeatureBundle river_rock_bundle;
    c3x_renderer::FeatureBundle cliff_bundle;
    std::array<std::vector<std::uint8_t>,32> cliff_dds;
    std::array<ID3D11ShaderResourceView *,32> cliff_views = {};
    bool cliff_assets_ready = false;
    std::array<std::vector<std::uint8_t>, 5> river_rock_texture_dds;
    std::array<ID3D11ShaderResourceView *, 5> river_rock_texture_views = {};
    std::array<std::vector<std::uint8_t>, 10> route_texture_dds;
    std::array<ID3D11ShaderResourceView *, 10> route_texture_views = {};
    c3x_renderer::FeatureBundle bridge_bundle;
    std::array<std::vector<std::uint8_t>, 8> bridge_texture_dds;
    std::array<ID3D11ShaderResourceView *, 8> bridge_texture_views = {};
    c3x_renderer::FeatureBundle resource_bundle;
    std::array<std::vector<std::uint8_t>, 8> resource_texture_dds;
    std::array<ID3D11ShaderResourceView *, 8> resource_texture_views = {};
    c3x_renderer::FeatureBundle city_bundle;
    std::array<std::vector<std::uint8_t>, 4> city_base_dds;
    std::array<ID3D11ShaderResourceView *, 4> city_base_views = {};
    std::array<std::vector<std::uint8_t>, 4> city_emissive_dds;
    std::array<ID3D11ShaderResourceView *, 4> city_emissive_views = {};
    c3x_renderer::FeatureBundle wall_bundle;
    std::vector<std::uint8_t> wall_texture_dds;
    ID3D11ShaderResourceView * wall_texture_view = nullptr;
    c3x_renderer::FeatureBundle mine_bundle;
    std::array<std::vector<std::uint8_t>, 6> mine_base_dds;
    std::array<ID3D11ShaderResourceView *, 6> mine_base_views = {};
    std::array<std::vector<std::uint8_t>, 2> mine_emissive_dds;
    std::array<ID3D11ShaderResourceView *, 2> mine_emissive_views = {};
    c3x_renderer::FeatureBundle site_bundle;
    std::array<std::vector<std::uint8_t>, 8> site_dds;
    std::array<ID3D11ShaderResourceView *, 8> site_views = {};
    bool site_assets_ready = false;
    c3x_renderer::FeatureBundle farm_bundle;
    std::array<std::vector<std::uint8_t>, 6> farm_base_dds;
    std::array<ID3D11ShaderResourceView *, 6> farm_base_views = {};
    std::array<std::vector<std::uint8_t>, 2> farm_emissive_dds;
    std::array<ID3D11ShaderResourceView *, 2> farm_emissive_views = {};
    c3x_renderer::TerrainFrameSignature cached_signature;
    c3x_renderer::TerrainFrameSignature previous_signature;
    std::uint64_t content_revision = 0;
    std::uint64_t previous_content_revision = 0;
    c3x_renderer_u32 cached_rendered_tile_count = 0;
    c3x_renderer_u32 cached_fallback_tile_count = 0;
    c3x_renderer_u32 cached_textured_tile_count = 0;
    std::vector<ResourceAnimation> resource_animations;
    std::vector<ResourceAnchor> resource_anchors;
    std::vector<ResourceBuffer> resource_buffers;
    std::vector<ResourceBackdrop> resource_backdrops;
    std::int64_t scene_depth_origin=0;
    std::uint64_t resource_backdrop_epoch=0;
    std::size_t resource_backdrop_bytes=0;
    c3x_renderer_i64 resource_composite_ticks=0;
    std::vector<std::uint32_t> resource_pixels;
    std::uint64_t resource_pixel_signature = 0;
    c3x_renderer_i64 resource_pixel_clock = -1;
    unsigned visible_resource_animations = 0, visible_wave_animations = 0, moving_resources = 0;
    float wave_time_seconds=0;
    bool wave_attempted=false,wave_ready=false;
    ID3D11PixelShader* wave_shader=nullptr;
    ID3D11Buffer* wave_frame=nullptr;
    std::array<ID3D11ShaderResourceView*,3> wave_views={};
    std::vector<CachedVertexChunk> wave_chunks;
    std::uint64_t wave_signature=0;
    std::size_t wave_geometry_bytes=0,wave_upload_bytes=0;
    std::map<std::pair<int,int>,RetainedWaveCell> retained_wave_cells;
    std::uint64_t retained_wave_scope=0,retained_wave_epoch=0;
    unsigned wave_cells_built=0,wave_cells_reused=0;
    unsigned ambient_count() const {return moving_resources+visible_wave_animations;}
    unsigned posed_count() const {return visible_resource_animations+unsigned(wave_chunks.size());}
    void reset_waves() {
        for(auto& c:wave_chunks){release(c.buffer);release(c.indices);}wave_chunks.clear();
        retained_wave_cells.clear();retained_wave_scope=retained_wave_epoch=0;
        wave_signature=0;visible_wave_animations=0;wave_geometry_bytes=wave_upload_bytes=0;
    }
    c3x_renderer_u32 cached_visible_animation_count = 0;
    c3x_renderer_u32 cached_request_continuous_redraw = 0;
    c3x_renderer_u32 cache_hits = 0;
    c3x_renderer_u32 cache_misses = 0;
    c3x_renderer_u32 cache_evictions = 0;
    c3x_renderer_u32 cache_stale_rejections = 0;
    c3x_renderer_u32 device_generation = 1;
    c3x_renderer_u32 device_recoveries = 0;
    std::vector<c3x_renderer_tile_v1> cached_tiles;
    std::vector<c3x_renderer_u32> cached_replacement_tile_flags;
    std::vector<CachedViewport> viewport_cache;
    std::size_t viewport_cache_bytes = 0;
    CachedGeometry geometry_cache;
    GeometryDrawView::Records geometry_vertex_buffers;
    std::unordered_multimap<std::uint64_t, CachedTileGeometry> tile_geometry_cache;
    c3x_renderer::render_core::ResidentContent<CachedTileGeometry> resident_content{tile_geometry_cache_capacity};
    std::size_t tile_geometry_cache_bytes = 0, prefetched_geometry_bytes = 0;
    std::size_t tile_geometry_runtime_budget = tile_geometry_cache_budget;
    std::unordered_map<std::uint64_t,NaturalTile> natural_mesh_cache;
    std::size_t natural_mesh_cache_bytes=0;
    std::unordered_map<std::uint64_t,CachedGroundTile> ground_grid_cache;
    std::size_t ground_grid_cache_bytes=0;
    unsigned frame_ground_grid_hits=0;
    unsigned frame_natural_hits=0;
    std::uint64_t tile_geometry_epoch = 0, tile_geometry_version = 0;
    std::vector<c3x_renderer::TileFootprint> geometry_footprints, bitmap_footprints;
    c3x_renderer::TerrainFrameSignature bitmap_footprint_signature;
    unsigned frame_tiles_built = 0, frame_tiles_reused = 0;
    unsigned frame_tiles_evicted = 0;
    std::size_t frame_upload_bytes = 0;
    ViewportShaderSettings geometry_viewport_settings = {};
    std::vector<D3D11_RECT> raster_rects;
    c3x_renderer_u32 raster_reused_pixels = 0, raster_draw_pixels = 0;

    c3x_renderer::PixelBlockCache pixel_blocks;
    c3x_renderer::TileFootprint prepared_footprint;
    std::vector<c3x_renderer::TileFootprint> pixel_neighborhood;
    std::vector<c3x_renderer::PixelRect> pixel_prepare_rects;
    std::size_t pixel_prepare_cursor = 0;
    int pixel_phase_x = 0, pixel_phase_y = 0;
    bool pixel_prepare_started = false, pixel_readback_pending = false;
    c3x_renderer::PixelBlock pending_pixel_block;
    ID3D11Texture2D * block_texture = nullptr, * block_depth_texture = nullptr, * block_readback = nullptr;
    ID3D11RenderTargetView * block_target = nullptr;
    ID3D11DepthStencilView * block_depth = nullptr;
    unsigned raster_cached_pixels = 0, prepared_blocks = 0;

    bool cache_valid = false;
    bool feature_assets_ready = false;
    bool dune_assets_ready = false;
    bool marsh_assets_ready = false;
    bool volcano_assets_ready = false;
    bool clutter_assets_ready = false;
    bool river_assets_ready = false;
    bool route_assets_ready = false;
    bool resource_assets_ready = false;
    bool city_assets_ready = false;
    bool mine_assets_ready = false;
    bool farm_assets_ready = false;

    ~RendererState() {
        reset();
    }

#ifdef C3X_RENDERER_BENCHMARK_ORACLE
    void trim_to_prepared(c3x_renderer_benchmark_oracle_trim_v1 & result, bool oracle_limits=true) {
        terrain_preparation.clear();
        for(auto& scratch:terrain_scratch)scratch.reset();foreground_terrain_scratch.reset();
        for(auto& scratch:world_ground_scratch){scratch.rivers.reset_world();scratch.reset_tile();}
        result = {};
        result.version = C3X_RENDERER_BENCHMARK_ORACLE_VERSION;
        result.struct_size = sizeof(result);
        result.cleared_viewport_bytes = viewport_cache_bytes +
            pixels.capacity() * sizeof(pixels[0]) +
            resource_pixels.capacity() * sizeof(resource_pixels[0]);
        result.cleared_region_bytes = render_regions.gpu_bytes + render_regions.metadata_bytes;
        result.cleared_pixel_block_bytes = pixel_blocks.bytes;
        result.cleared_backdrop_bytes = resource_backdrop_bytes;
        result.retained_geometry_bytes = tile_geometry_cache_bytes;
        result.retained_natural_bytes = natural_mesh_cache_bytes;
        result.retained_ground_bytes = ground_grid_cache_bytes;
        result.retained_wave_bytes = wave_geometry_bytes;
        result.retained_unit_pose_bytes = unit_bodies.cache_bytes;
        result.retained_unit_payload_bytes = unit_bodies.resident_bytes;
        result.retained_shadow_bytes = source_shadow.view ? 128u * 1024u * 1024u : 0;
        result.retained_other_bytes = center_shore_cache.bytes;
        result.retained_geometry_entries = static_cast<std::uint32_t>(
            std::min<std::size_t>(tile_geometry_cache.size(), UINT32_MAX));
        result.retained_unit_pose_entries = static_cast<std::uint32_t>(
            std::min<std::size_t>(unit_bodies.cached_pose_entries(), UINT32_MAX));
        result.retained_wave_entries = static_cast<std::uint32_t>(
            std::min<std::size_t>(retained_wave_cells.size(), UINT32_MAX));

        render_regions.clear();
        region_context.clear();
        retained_region_casters = {};
        viewport_cache.clear();
        viewport_cache_bytes = 0;
        cancel_pixel_preparation();
        pixel_blocks.clear();
        clear_resource_backdrops();
        std::fill(resource_pixels.begin(),resource_pixels.end(),0);
        resource_pixel_signature = 0;
        resource_pixel_clock = -1;
        moving_resources = visible_resource_animations = visible_wave_animations = 0;
        for(auto& chunk:wave_chunks){release(chunk.buffer);release(chunk.indices);}
        wave_chunks.clear();wave_signature=0;wave_upload_bytes=0;
        geometry_cache.clear();
        clear_geometry_vertex_buffers();
        geometry_footprints.clear();
        bitmap_footprints.clear();
        bitmap_footprint_signature = {};
        cached_tiles.clear();
        cached_replacement_tile_flags.clear();
        cached_signature = {};
        previous_signature = {};
        cached_rendered_tile_count = cached_fallback_tile_count = cached_textured_tile_count = 0;
        cached_visible_animation_count = cached_request_continuous_redraw = 0;
        cache_valid = false;
        std::fill(pixels.begin(),pixels.end(),0);
        if(oracle_limits)tile_geometry_runtime_budget=512u*1024u*1024u;
        while(!tile_geometry_cache.empty() && tile_geometry_cache_bytes>tile_geometry_runtime_budget) {
            auto oldest=std::min_element(tile_geometry_cache.begin(),tile_geometry_cache.end(),[](auto const& a,auto const& b){
                return a.second.last_used<b.second.last_used;});
            tile_geometry_cache_bytes-=oldest->second.byte_count;
            if(oldest->second.prefetched)prefetched_geometry_bytes-=oldest->second.byte_count;
            release_geometry_vertex_buffers(oldest->second.buffers);release_resident_content(oldest->second);tile_geometry_cache.erase(oldest);
            ++result.capacity_geometry_evictions;
        }
        if(oracle_limits)result.capacity_pose_evictions=unit_bodies.benchmark_limit_pose_cache();
        result.retained_geometry_bytes=tile_geometry_cache_bytes;
        result.retained_unit_pose_bytes=unit_bodies.cache_bytes;
        result.retained_geometry_entries=static_cast<std::uint32_t>(
            (std::min<std::size_t>)(tile_geometry_cache.size(),UINT32_MAX));
        result.retained_unit_pose_entries=static_cast<std::uint32_t>(
            (std::min<std::size_t>)(unit_bodies.cached_pose_entries(),UINT32_MAX));
        char detail[384];
        std::snprintf(detail,sizeof(detail),
            "cleared_viewport=%llu cleared_regions=%llu cleared_blocks=%llu cleared_backdrops=%llu retained_geometry=%llu retained_poses=%llu retained_payloads=%llu",
            static_cast<unsigned long long>(result.cleared_viewport_bytes),
            static_cast<unsigned long long>(result.cleared_region_bytes),
            static_cast<unsigned long long>(result.cleared_pixel_block_bytes),
            static_cast<unsigned long long>(result.cleared_backdrop_bytes),
            static_cast<unsigned long long>(result.retained_geometry_bytes),
            static_cast<unsigned long long>(result.retained_unit_pose_bytes),
            static_cast<unsigned long long>(result.retained_unit_payload_bytes));
        trace.write("oracle-trim",detail,true);
    }
#endif

    template <typename T>
    void release(T *& value) {
        if (value != nullptr) {
            value->Release();
            value = nullptr;
        }
    }

    void reset_targets() {
        scene_restore.reset();
        ++scene_generation;unit_scene_work.reset();scene_scratch.reset();scene_guard.reset();scene_guard_context.clear();scene_guard_pad=0;scene_dynamic_damage.clear();scene_static_signature=0;scene_overlap=false;scene_damage.clear();
        cancel_pixel_preparation();
        linear_frame.reset(); linear_block.reset(); reflection.linear.reset();region_reflection.linear.reset();region_glow.linear.reset();
        pixel_blocks.clear();
        release(block_readback); release(block_depth); release(block_depth_texture);
        release(block_target); release(block_texture);
        release(gpu_map_texture);gpu_map_valid=false;
        release(readback_texture);
        release(animation_readback_texture);
        animation_readback_width=animation_readback_height=0;
        release(depth_target);
        release(depth_texture);
        release(render_target);
        release(render_texture);
        width = 0;
        height = 0;
        pixels.clear();
        cache_valid = false;
    }

    void clear_resource_backdrops() {
        ++scene_generation;unit_scene_work.reset();scene_scratch.reset();scene_guard.reset();scene_guard_context.clear();scene_guard_pad=0;scene_dynamic_damage.clear();scene_static_signature=0;scene_overlap=false;scene_damage.clear();
        for(auto & block:resource_backdrops){release(block.color);release(block.depth);}
        resource_backdrops.clear();resource_backdrop_epoch=0;resource_backdrop_bytes=0;
    }
    bool make_resource_backdrop_room(std::size_t bytes,std::uint64_t signature) {
        if(bytes>resource_backdrop_cache_budget)return false;
        while(resource_backdrop_bytes>resource_backdrop_cache_budget-bytes){
            auto oldest=resource_backdrops.end();
            for(auto it=resource_backdrops.begin();it!=resource_backdrops.end();++it)
                if(it->signature!=signature && (oldest==resource_backdrops.end() || it->used<oldest->used))oldest=it;
            // Do not cycle through the current view when it exceeds the cap.
            // Uncached blocks still render normally, with fresh animated poses.
            if(oldest==resource_backdrops.end())return false;
            resource_backdrop_bytes-=oldest->bytes;
            release(oldest->color);release(oldest->depth);resource_backdrops.erase(oldest);
        }
        return true;
    }
    void reset_resource_buffers() {
        clear_resource_backdrops();
        for (auto & buffer : resource_buffers) {
            release(buffer.vertices);
            release(buffer.shadow_vertices);
        }
        resource_buffers.clear(); resource_pixels.clear();
        resource_pixel_signature = 0; resource_pixel_clock = -1; moving_resources = visible_resource_animations = 0;
        for (auto & animation : resource_animations) {
            release(animation.view); release(animation.indices);release(animation.vertices);
        }
    }

    void reset() {
        material_views={};material_views_valid=false;
        gpu_composition.reset();visibility_gpu.reset();visibility_pixels.clear();
        terrain_preparation.clear();
        for(auto& scratch:terrain_scratch)scratch.reset();foreground_terrain_scratch.reset();
        for(auto& scratch:world_ground_scratch){scratch.rivers.reset_world();scratch.reset_tile();}
        memory_sample("before-reset");
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
        tile_geometry_runtime_budget=tile_geometry_cache_budget;
        unit_bodies.benchmark_reset_pose_limit();
#endif
        render_regions.clear();region_context.clear();retained_region_casters={};
        gpu_telemetry.reset();animation_gpu.reset();
        sampled_geometry_bucket=~std::size_t(0);
        unit_bodies.reset_gpu();
        reset_resource_buffers();
        if (context != nullptr)
            context->ClearState();
        reset_targets();
        world_coast.clear();center_shore_cache.clear(); geometry_world_revision = -1; source_shadow.clear(); natural.reset(); reflection.reset();cities.reset();city_glow.reset();
        region_reflection.reset();region_glow.reset();
        for (TerrainTexture & texture : terrain_textures)
        {
            release(texture.view);
            release(texture.material_height_view);
            release(texture.specular_view);
            release(texture.elevated_view);
            release(texture.elevated_height_view);
            release(texture.elevated_specular_view);
            for (ID3D11ShaderResourceView *& view : texture.relief_layer_views)
                release(view);
            for (ID3D11ShaderResourceView *& view : texture.water_surface_views)
                release(view);
        }
        release(dune_surface.view);
        release(dune_surface.material_height_view);
        release(dune_surface.specular_view);
        release(dune_decal_base_view);
        release(dune_decal_height_view);
        release(marsh_decal_base_view);
        release(marsh_decal_height_view);
        release(marsh_decal_specular_view);
        release(volcano_base_view);
        release(volcano_height_view);
        release(volcano_active_base_view);
        release(volcano_active_specular_view);
        release(water_clutter_base_view);
        release(water_clutter_height_view);
        release(grass_clutter_base_view);
        release(grass_clutter_height_view);
        release(plains_clutter_base_view);
        release(plains_clutter_height_view);
        for (auto & view : cliff_views) release(view);
        for (ID3D11ShaderResourceView *& view : feature_texture_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : terrain_extra_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : river_surface_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : river_rock_texture_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : route_texture_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : bridge_texture_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : resource_texture_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : city_base_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : city_emissive_views)
            release(view);
        release(wall_texture_view);
        for (ID3D11ShaderResourceView *& view : site_views) release(view);
        for (ID3D11ShaderResourceView *& view : mine_base_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : mine_emissive_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : farm_base_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : farm_emissive_views)
            release(view);
        release(natural_wrap);release(natural_clamp);
        release(decal_sampler);
        release(terrain_sampler);
        release(rasterizer_state);
        release(depth_state);
        release(blend_state);
        release(input_layout); release(feature_input_layout);
        release(resource_body_vertex_shader);release(resource_shadow_vertex_shader);release(resource_input_layout);
        release(terrain_settings_buffer);
        draw_parameters.clear();
        release(viewport_settings_buffer);
        release(world_settings_buffer); release(shadow_settings_buffer);
        linear_output.reset();
        clear_geometry_vertex_buffers();
        clear_tile_geometry_cache();
        release(feature_pixel_shader);
        release(resource_shadow_shader);
        release(feature_vertex_shader);
        release(pixel_shader);
        reset_waves();release(wave_shader);release(wave_frame);
        for(auto&view:wave_views)release(view);
        wave_attempted=wave_ready=false;
        release(vertex_shader);
        release(context);
        release(device);
        cached_tiles.clear();
        cached_replacement_tile_flags.clear();
        viewport_cache.clear();
        viewport_cache_bytes = 0;
        geometry_cache.clear();
        cache_valid = false;
        if (device_generation != 0xffffffffu)
            ++device_generation;
        memory_sample("after-reset");
    }

    // UI presentation needs the device, but no terrain assets or shaders.
    bool initialize_device() {
        if (device != nullptr)
            return true;

        char budget_detail[256];sprintf_s(budget_detail,"gpu_geometry=%zu natural_cpu=%zu viewport_cpu=%zu backdrop_gpu=%zu region_size=%d",
            tile_geometry_cache_budget,natural_mesh_cache_budget,viewport_cache_budget,resource_backdrop_cache_budget,scene_region_size);
        trace.write("cache-budgets",budget_detail,true);

        LARGE_INTEGER device_begin={},device_end={};
        if(trace.buffered)QueryPerformanceCounter(&device_begin);
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        D3D_FEATURE_LEVEL levels[] = {
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0
        };
        D3D_FEATURE_LEVEL selected = D3D_FEATURE_LEVEL_10_0;
        HRESULT hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
            levels, static_cast<UINT>(std::size(levels)), D3D11_SDK_VERSION,
            &device, &selected, &context);
        if (FAILED(hr)) {
            hr = D3D11CreateDevice(
                nullptr, D3D_DRIVER_TYPE_WARP, nullptr, flags,
                levels, static_cast<UINT>(std::size(levels)), D3D11_SDK_VERSION,
                &device, &selected, &context);
        }
        if (FAILED(hr)) {
            reset();
            return false;
        }

        if(trace.buffered){QueryPerformanceCounter(&device_end);char detail[128];
            sprintf_s(detail,"begin=%lld end=%lld elapsed_ms=%.3f",device_begin.QuadPart,device_end.QuadPart,
                trace.milliseconds(device_end.QuadPart-device_begin.QuadPart));trace.write("setup-device",detail,true);}
        if (pickup_profile && selected < D3D_FEATURE_LEVEL_11_0) {
            trace.write("profile-unavailable", "pickup-r1 requires D3D feature level 11", true);
            reset(); return false;
        }

        return true;
    }

    bool initialize() {
        // All failed material initialization paths reset these resources.
        if(vertex_shader)return true;
        if(!initialize_device())return false;
        HRESULT hr=S_OK;
        // Production executes the material and feature entry points copied
        // from the approved Lab handoff, isolated from in-progress Lab edits.
        auto compile_terrain_shader = [this](char const * entry, char const * target,
                                             ID3DBlob ** blob) {
            std::string selected_shader=integrated_shader_path;
            if(fidelity_profile && std::strstr(entry,"Feature"))selected_shader=fidelity_root+(city_profile?"/Renderer/native/city_fidelity/feature.hlsl":environment_profile?"/Renderer/native/environment_refresh/feature.hlsl":"/Renderer/native/render_core/terrain_scene.hlsl");
            if(city_profile && (!std::strcmp(entry,"VSResourceBody") || !std::strcmp(entry,"VSResourceShadow")))
                selected_shader=fidelity_root+"/Renderer/native/city_fidelity/"+
                    (std::strcmp(entry,"VSResourceBody")==0?"resource_body.hlsl":"resource_shadow.hlsl");
            int count = MultiByteToWideChar(CP_UTF8, 0, selected_shader.c_str(),
                                            -1, nullptr, 0);
            if (count <= 0)
                return false;
            std::wstring wide_path(static_cast<std::size_t>(count), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, selected_shader.c_str(), -1,
                                wide_path.data(), count);
            ID3DBlob * errors = nullptr;
            LARGE_INTEGER shader_begin={},shader_end={};
            if(trace.buffered)QueryPerformanceCounter(&shader_begin);
            HRESULT result = pickup_profile ? c3x_renderer::render_core::compile_cached(
                wide_path.c_str(),entry,target,blob,&errors) : D3DCompileFromFile(
                wide_path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
                entry, target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, blob, &errors);
            if(trace.buffered){QueryPerformanceCounter(&shader_end);char detail[192];
                sprintf_s(detail,"entry=%s begin=%lld end=%lld elapsed_ms=%.3f",entry,shader_begin.QuadPart,shader_end.QuadPart,
                    trace.milliseconds(shader_end.QuadPart-shader_begin.QuadPart));trace.write("setup-shader",detail,true);}
            if (errors != nullptr) {
                trace.write("shader-compile",static_cast<char const *>(errors->GetBufferPointer()),true);
                errors->Release();
            }
            if(FAILED(result))trace.write("shader-entry-failed",entry,true);
            return SUCCEEDED(result);
        };
        ID3DBlob * vertex_blob = nullptr;
        ID3DBlob * pixel_blob = nullptr;
        ID3DBlob * feature_vertex_blob = nullptr;
        ID3DBlob * feature_pixel_blob = nullptr;
        char const * vs_target = pickup_profile ? "vs_5_0" : "vs_4_0";
        char const * ps_target = pickup_profile ? "ps_5_0" : "ps_4_0";
        if (!compile_terrain_shader("VSIntegrated", vs_target, &vertex_blob) ||
            !compile_terrain_shader("PSIntegrated", ps_target, &pixel_blob) ||
            !compile_terrain_shader("VSIntegratedFeature", vs_target, &feature_vertex_blob) ||
            !compile_terrain_shader("PSIntegratedFeature", ps_target, &feature_pixel_blob)) {
            release(feature_pixel_blob);
            release(feature_vertex_blob);
            release(pixel_blob);
            release(vertex_blob);
            reset();
            return false;
        }

        hr = device->CreateVertexShader(vertex_blob->GetBufferPointer(), vertex_blob->GetBufferSize(),
                                        nullptr, &vertex_shader);
        if (SUCCEEDED(hr))
            hr = device->CreatePixelShader(pixel_blob->GetBufferPointer(), pixel_blob->GetBufferSize(),
                                           nullptr, &pixel_shader);
        if (SUCCEEDED(hr))
            hr = device->CreateVertexShader(feature_vertex_blob->GetBufferPointer(),
                                            feature_vertex_blob->GetBufferSize(), nullptr,
                                            &feature_vertex_shader);
        if (SUCCEEDED(hr))
            hr = device->CreatePixelShader(feature_pixel_blob->GetBufferPointer(),
                                           feature_pixel_blob->GetBufferSize(), nullptr,
                                           &feature_pixel_shader);
        if(city_profile && SUCCEEDED(hr)) {
            ID3DBlob* blob=nullptr;
            if(!compile_terrain_shader("PSResourceShadow",ps_target,&blob))hr=E_FAIL;
            else hr=device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&resource_shadow_shader);
            release(blob);
            if(SUCCEEDED(hr)) {
                if(!compile_terrain_shader("VSResourceBody",vs_target,&blob))hr=E_FAIL;
                else {
                    hr=device->CreateVertexShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&resource_body_vertex_shader);
                    D3D11_INPUT_ELEMENT_DESC elements[]={
                        {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
                        {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
                        {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,24,D3D11_INPUT_PER_VERTEX_DATA,0},
                        {"BLENDINDICES",0,DXGI_FORMAT_R32G32B32A32_UINT,0,56,D3D11_INPUT_PER_VERTEX_DATA,0},
                        {"BLENDWEIGHT",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,72,D3D11_INPUT_PER_VERTEX_DATA,0}};
                    static_assert(sizeof(c3x_renderer::AnimationVertex)==88,"Resource source GPU layout");
                    if(SUCCEEDED(hr))hr=device->CreateInputLayout(elements,UINT(std::size(elements)),blob->GetBufferPointer(),blob->GetBufferSize(),&resource_input_layout);
                }
                release(blob);
            }
            if(SUCCEEDED(hr)) {
                if(!compile_terrain_shader("VSResourceShadow",vs_target,&blob))hr=E_FAIL;
                else hr=device->CreateVertexShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&resource_shadow_vertex_shader);
                release(blob);
            }
        }
        if(environment_profile && SUCCEEDED(hr)) {
            ID3DBlob* blob=nullptr;
            if(!compile_terrain_shader("PSCoastalWave",ps_target,&blob))hr=E_FAIL;
            else {hr=device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&wave_shader);release(blob);}
            D3D11_BUFFER_DESC desc={};desc.ByteWidth=16;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(SUCCEEDED(hr))hr=device->CreateBuffer(&desc,nullptr,&wave_frame);
        }
        D3D11_INPUT_ELEMENT_DESC elements[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 1, DXGI_FORMAT_R32_FLOAT, 0, 20, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 2, DXGI_FORMAT_R32G32_FLOAT, 0, 36, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 3, DXGI_FORMAT_R32G32_FLOAT, 0, 44, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 4, DXGI_FORMAT_R32_FLOAT, 0, 52, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 5, DXGI_FORMAT_R32_FLOAT, 0, 56, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 6, DXGI_FORMAT_R32_FLOAT, 0, 60, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 7, DXGI_FORMAT_R32_FLOAT, 0, 64, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 8, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 68, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 9, DXGI_FORMAT_R32G32_FLOAT, 0, 84, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 10, DXGI_FORMAT_R32_FLOAT, 0, 92, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 11, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 96, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 12, DXGI_FORMAT_R32_FLOAT, 0, 112, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 13, DXGI_FORMAT_R32_FLOAT, 0, 116, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 14, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 120, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 15, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 136, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 16, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 152, D3D11_INPUT_PER_VERTEX_DATA, 0}
        };
        if (SUCCEEDED(hr)) {
            hr = device->CreateInputLayout(elements, static_cast<UINT>(std::size(elements) - (pickup_profile ? 0 : 3)),
                                            vertex_blob->GetBufferPointer(), vertex_blob->GetBufferSize(),
                                            &input_layout);
        }
        if(pickup_profile && SUCCEEDED(hr)) {
            D3D11_INPUT_ELEMENT_DESC feature_elements[]={
                {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,20,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",6,DXGI_FORMAT_R32_FLOAT,0,32,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",14,DXGI_FORMAT_R32G32B32_FLOAT,0,36,D3D11_INPUT_PER_VERTEX_DATA,0}};
            hr=device->CreateInputLayout(feature_elements,5,feature_vertex_blob->GetBufferPointer(),
                feature_vertex_blob->GetBufferSize(),&feature_input_layout);
        }
        release(feature_pixel_blob);
        release(feature_vertex_blob);
        release(pixel_blob);
        release(vertex_blob);
        if (FAILED(hr)) {
            char reason[96];sprintf_s(reason,"shader/layout HRESULT=0x%08lx",static_cast<unsigned long>(hr));
            trace.write("initialize-failed",reason,true);
            reset();
            return false;
        }

        TerrainShaderSettings settings = {};
        settings.height_texel[0] = 1.0f / 2048.0f;
        settings.height_texel[1] = 1.0f / 2048.0f;
        settings.normal_strength = 4.0f;
        settings.exposure = 1.0f;
        settings.light_direction[0] = -0.55f;
        settings.light_direction[1] = -0.35f;
        settings.light_direction[2] = 0.22f;
        settings.sun_intensity = 1.0f;
        settings.sun_color[0] = settings.sun_color[1] = settings.sun_color[2] = 1.0f;
        settings.shadow_strength = 0.84f;
        settings.moon_direction[2] = 1.0f;
        settings.ambient_color[0] = settings.ambient_color[1] = settings.ambient_color[2] = 0.78f;
        settings.environment_exposure = 1.0f;
        settings.water_fresnel = 0.04f;
        settings.water_specular = 0.62f;
        settings.emissive_scale = 0.25f;
        settings.hour = 12.0f;
        D3D11_BUFFER_DESC settings_desc = {};
        settings_desc.ByteWidth = sizeof(settings);
        settings_desc.Usage = D3D11_USAGE_DEFAULT;
        settings_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA settings_data = {&settings, 0, 0};
        hr = device->CreateBuffer(&settings_desc, &settings_data,
                                  &terrain_settings_buffer);
        ViewportShaderSettings viewport_settings = {};
        D3D11_BUFFER_DESC viewport_settings_desc = {};
        viewport_settings_desc.ByteWidth = sizeof(viewport_settings);
        viewport_settings_desc.Usage = D3D11_USAGE_DEFAULT;
        viewport_settings_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA viewport_settings_data = {&viewport_settings, 0, 0};
        if (SUCCEEDED(hr))
            hr = device->CreateBuffer(&viewport_settings_desc,
                                      &viewport_settings_data,
                                      &viewport_settings_buffer);

        if (pickup_profile && SUCCEEDED(hr)) {
            D3D11_BUFFER_DESC desc = {};
            desc.ByteWidth = 32; desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            hr = device->CreateBuffer(&desc, nullptr, &world_settings_buffer);
            desc.ByteWidth = 80;
            if (SUCCEEDED(hr)) hr = device->CreateBuffer(&desc, nullptr, &shadow_settings_buffer);
            if (SUCCEEDED(hr) && !linear_output.ensure(device)) hr = E_FAIL;
            std::string source_path=fidelity_profile ? fidelity_root+(city_profile?"/Renderer/native/city_fidelity/source_caster.hlsl":environment_profile?"/Renderer/native/environment_refresh/source_caster.hlsl":"/Renderer/native/render_core/source_caster.hlsl") : integrated_shader_path.substr(0,integrated_shader_path.find_last_of("\\/"))+"/source_caster.hlsl";
            int count=MultiByteToWideChar(CP_UTF8,0,source_path.c_str(),-1,nullptr,0);
            std::wstring wide(static_cast<std::size_t>(count),L'\0');
            MultiByteToWideChar(CP_UTF8,0,source_path.c_str(),-1,wide.data(),count);
            if (SUCCEEDED(hr) && !source_shadow.ensure(device,wide.c_str())) hr=E_FAIL;
        }

        D3D11_BLEND_DESC blend = {};
        blend.RenderTarget[0].BlendEnable = TRUE;
        blend.RenderTarget[0].SrcBlend = pickup_profile ? D3D11_BLEND_ONE : D3D11_BLEND_SRC_ALPHA;
        blend.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
        blend.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
        blend.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
        blend.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
        blend.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
        blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        if (SUCCEEDED(hr))
            hr = device->CreateBlendState(&blend, &blend_state);

        D3D11_DEPTH_STENCIL_DESC depth = {};
        depth.DepthEnable = TRUE;
        depth.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        depth.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
        if (SUCCEEDED(hr))
            hr = device->CreateDepthStencilState(&depth, &depth_state);

        D3D11_RASTERIZER_DESC raster = {};
        raster.FillMode = D3D11_FILL_SOLID;
        raster.CullMode = D3D11_CULL_NONE;
        raster.DepthClipEnable = TRUE;
        raster.ScissorEnable = TRUE;
        if (SUCCEEDED(hr))
            hr = device->CreateRasterizerState(&raster, &rasterizer_state);
        D3D11_SAMPLER_DESC sampler = {};
        sampler.Filter = D3D11_FILTER_ANISOTROPIC;
        sampler.MaxAnisotropy = 8;
        sampler.MipLODBias = 0.0f;
        sampler.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
        sampler.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
        sampler.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
        sampler.MaxLOD = D3D11_FLOAT32_MAX;
        if (SUCCEEDED(hr))
            hr = device->CreateSamplerState(&sampler, &terrain_sampler);
        sampler.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        if (SUCCEEDED(hr))
            hr = device->CreateSamplerState(&sampler, &decal_sampler);
        if(SUCCEEDED(hr) && fidelity_profile){
            sampler.MaxAnisotropy=16;sampler.MipLODBias=-1;
            hr=device->CreateSamplerState(&sampler,&natural_clamp);
            sampler.AddressU=sampler.AddressV=sampler.AddressW=D3D11_TEXTURE_ADDRESS_WRAP;
            if(SUCCEEDED(hr))hr=device->CreateSamplerState(&sampler,&natural_wrap);
        }
        if (FAILED(hr)) {
            reset();
            return false;
        }
        return true;
    }

    static bool read_file(char const * path, std::vector<std::uint8_t> & output) {
        HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                  FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE)
            return false;
        LARGE_INTEGER size = {};
        bool ok = GetFileSizeEx(file, &size) != 0 && size.QuadPart > 0 && size.QuadPart <= 256ll * 1024ll * 1024ll;
        if (ok) {
            output.resize(static_cast<std::size_t>(size.QuadPart));
            DWORD bytes_read = 0;
            ok = ReadFile(file, output.data(), static_cast<DWORD>(output.size()), &bytes_read, nullptr) != 0 &&
                 bytes_read == static_cast<DWORD>(output.size());
        }
        CloseHandle(file);
        if (!ok)
            output.clear();
        return ok;
    }

    static bool pack_path(char const * root, char const * relative, char * output, std::size_t capacity) {
        std::size_t root_length = std::strlen(root);
        std::size_t relative_length = std::strlen(relative);
        if (root_length == 0 || root_length + relative_length + 2 > capacity)
            return false;
        std::memcpy(output, root, root_length);
        output[root_length] = '\\';
        std::memcpy(output + root_length + 1, relative, relative_length + 1);
        return true;
    }

    static bool safe_relative(std::string const & path) {
        if (path.empty() || path.front() == '/' || path.front() == '\\' || path.find(':') != std::string::npos)
            return false;
        std::string normalized = path;
        std::replace(normalized.begin(), normalized.end(), '\\', '/');
        std::size_t cursor = 0;
        while (cursor <= normalized.size()) {
            std::size_t end = normalized.find('/', cursor);
            std::string part = normalized.substr(cursor, end == std::string::npos ? std::string::npos : end - cursor);
            if (part == "..")
                return false;
            if (end == std::string::npos)
                break;
            cursor = end + 1;
        }
        return true;
    }

    static bool contains(std::vector<std::uint8_t> const & data, char const * text) {
        auto const * begin = reinterpret_cast<char const *>(data.data());
        auto const * end = begin + data.size();
        std::size_t length = std::strlen(text);
        return std::search(begin, end, text, text + length) != end;
    }

    static std::size_t find_text(std::vector<std::uint8_t> const & data, std::string const & text,
                                 std::size_t start = 0) {
        if (start > data.size())
            return std::string::npos;
        auto begin = data.begin() + static_cast<std::ptrdiff_t>(start);
        auto found = std::search(begin, data.end(), text.begin(), text.end());
        return found == data.end() ? std::string::npos : static_cast<std::size_t>(found - data.begin());
    }

    static bool json_string_after(std::vector<std::uint8_t> const & data, char const * key,
                                  std::size_t start, std::string & value) {
        std::string marker = std::string("\"") + key + "\"";
        std::size_t position = json_member_position(data, key, start);
        if (position == std::string::npos)
            return false;
        position = find_text(data, ":", position + marker.size());
        if (position == std::string::npos)
            return false;
        auto const * bytes = reinterpret_cast<char const *>(data.data());
        while (++position < data.size() && std::isspace(static_cast<unsigned char>(bytes[position])) != 0) {}
        if (position >= data.size() || bytes[position] != '\"')
            return false;
        std::size_t end = position + 1;
        while (end < data.size() && bytes[end] != '\"')
            ++end;
        if (end >= data.size())
            return false;
        value.assign(bytes + position + 1, bytes + end);
        return safe_relative(value);
    }

    static bool json_number_after(std::vector<std::uint8_t> const & data, char const * key,
                                  std::size_t start, float & value) {
        std::string marker = std::string("\"") + key + "\"";
        std::size_t position = json_member_position(data, key, start);
        if (position == std::string::npos)
            return false;
        position = find_text(data, ":", position + marker.size());
        if (position == std::string::npos)
            return false;
        auto const * bytes = reinterpret_cast<char const *>(data.data());
        while (++position < data.size() && std::isspace(static_cast<unsigned char>(bytes[position])) != 0) {}
        char number[48] = {};
        std::size_t count = 0;
        while (position < data.size() && count + 1 < std::size(number) &&
               (std::isdigit(static_cast<unsigned char>(bytes[position])) != 0 ||
                bytes[position] == '-' || bytes[position] == '+' || bytes[position] == '.' ||
                bytes[position] == 'e' || bytes[position] == 'E'))
            number[count++] = bytes[position++];
        if (count == 0 || (position < data.size() &&
            std::isspace(static_cast<unsigned char>(bytes[position])) == 0 &&
            bytes[position] != ',' && bytes[position] != '}' && bytes[position] != ']'))
            return false;
        char * end = nullptr;
        value = std::strtof(number, &end);
        return end != number && *end == '\0' && std::isfinite(value);
    }

    static std::uint32_t read_u32(std::vector<std::uint8_t> const & data, std::size_t offset) {
        return static_cast<std::uint32_t>(data[offset]) |
               (static_cast<std::uint32_t>(data[offset + 1]) << 8) |
               (static_cast<std::uint32_t>(data[offset + 2]) << 16) |
               (static_cast<std::uint32_t>(data[offset + 3]) << 24);
    }

    void clear_terrain_assets() {
        material_views={};material_views_valid=false;
        terrain_preparation.clear();
        for(auto& scratch:terrain_scratch)scratch.reset();foreground_terrain_scratch.reset();
        for(auto& scratch:world_ground_scratch){scratch.rivers.reset_world();scratch.reset_tile();}
        for (TerrainTexture & texture : terrain_textures) {
            release(texture.view);
            release(texture.material_height_view);
            release(texture.specular_view);
            release(texture.elevated_view);
            release(texture.elevated_height_view);
            release(texture.elevated_specular_view);
            for (ID3D11ShaderResourceView *& view : texture.relief_layer_views)
                release(view);
            for (ID3D11ShaderResourceView *& view : texture.water_surface_views)
                release(view);
            texture.dds.clear();
            texture.material_height_dds.clear();
            texture.specular_dds.clear();
            texture.elevated_dds.clear();
            texture.elevated_height_dds.clear();
            texture.elevated_specular_dds.clear();
            for (std::vector<std::uint8_t> & layer : texture.relief_layer_dds)
                layer.clear();
            for (std::vector<std::uint8_t> & channel : texture.water_surface_dds)
                channel.clear();
            texture.height_pixels.clear();
            texture.blend_pixels.clear();
            for (std::vector<std::uint8_t> & field : texture.relief_height_variants)
                field.clear();
            for (std::vector<std::uint8_t> & field : texture.relief_blend_variants)
                field.clear();
            texture.relief_variant_widths.fill(0);
            texture.relief_variant_heights.fill(0);
            texture.relief_height_minimum.fill(0.0f);
            texture.relief_height_maximum.fill(0.0f);
            texture.relief_blend_minimum.fill(0.0f);
            texture.relief_blend_maximum.fill(0.0f);
            texture.height_width = 0;
            texture.height_height = 0;
            texture.height_minimum = 0.0f;
            texture.height_maximum = 1.0f;
            texture.blend_minimum = 0.0f;
            texture.blend_maximum = 1.0f;
            texture.height_scale_px = 0.0f;
            texture.relief_profile = 0;
            texture.configured = false;
        }
        release(dune_surface.view);
        release(dune_surface.material_height_view);
        release(dune_surface.specular_view);
        dune_surface.dds.clear();
        dune_surface.material_height_dds.clear();
        dune_surface.specular_dds.clear();
        dune_surface.configured = false;
        release(dune_decal_base_view);
        release(dune_decal_height_view);
        dune_decal_base_dds.clear();
        dune_decal_height_dds.clear();
        release(marsh_decal_base_view);
        release(marsh_decal_height_view);
        release(marsh_decal_specular_view);
        marsh_decal_base_dds.clear();
        marsh_decal_height_dds.clear();
        marsh_decal_specular_dds.clear();
        release(volcano_base_view);
        release(volcano_height_view);
        release(volcano_active_base_view);
        release(volcano_active_specular_view);
        volcano_base_dds.clear();
        volcano_height_dds.clear();
        volcano_active_base_dds.clear();
        volcano_active_specular_dds.clear();
        release(water_clutter_base_view);
        release(water_clutter_height_view);
        release(grass_clutter_base_view);
        release(grass_clutter_height_view);
        release(plains_clutter_base_view);
        release(plains_clutter_height_view);
        water_clutter_base_dds.clear();
        water_clutter_height_dds.clear();
        grass_clutter_base_dds.clear();
        grass_clutter_height_dds.clear();
        plains_clutter_base_dds.clear();
        plains_clutter_height_dds.clear();
        for (std::size_t index = 0; index < river_surface_views.size(); ++index) {
            release(river_surface_views[index]);
            river_surface_dds[index].clear();
        }
        for (std::size_t index = 0; index < river_rock_texture_views.size(); ++index) {
            release(river_rock_texture_views[index]);
            river_rock_texture_dds[index].clear();
        }
        river_rock_bundle = {};
        cliff_bundle = {}; cliff_assets_ready = false;
        for (auto & view : cliff_views) release(view);
        for (auto & data : cliff_dds) data.clear();
        for (std::size_t index = 0; index < route_texture_views.size(); ++index) {
            release(route_texture_views[index]);
            route_texture_dds[index].clear();
        }
        for (std::size_t index = 0; index < bridge_texture_views.size(); ++index) {
            release(bridge_texture_views[index]);
            bridge_texture_dds[index].clear();
        }
        bridge_bundle = {};
        for (std::size_t index = 0; index < resource_texture_views.size(); ++index) {
            release(resource_texture_views[index]);
            resource_texture_dds[index].clear();
        }
        resource_bundle = {};
        for (std::size_t index = 0; index < city_base_views.size(); ++index) {
            release(city_base_views[index]);
            release(city_emissive_views[index]);
            city_base_dds[index].clear();
            city_emissive_dds[index].clear();
        }
        city_bundle = {};
        release(wall_texture_view);
        wall_texture_dds.clear();
        wall_bundle = {};
        for (std::size_t index = 0; index < mine_base_views.size(); ++index) {
            release(mine_base_views[index]);
            mine_base_dds[index].clear();
        }
        for (std::size_t index = 0; index < mine_emissive_views.size(); ++index) {
            release(mine_emissive_views[index]);
            mine_emissive_dds[index].clear();
        }
        mine_bundle = {};
        for (unsigned i=0;i<site_views.size();++i) {release(site_views[i]);site_dds[i].clear();}
        site_bundle={};site_assets_ready=false;
        for (std::size_t index = 0; index < farm_base_views.size(); ++index) {
            release(farm_base_views[index]);
            farm_base_dds[index].clear();
        }
        for (std::size_t index = 0; index < farm_emissive_views.size(); ++index) {
            release(farm_emissive_views[index]);
            farm_emissive_dds[index].clear();
        }
        farm_bundle = {};
        for (std::size_t index = 0; index < feature_texture_views.size(); ++index) {
            release(feature_texture_views[index]);
            feature_texture_dds[index].clear();
        }
        for (std::size_t index = 0; index < terrain_extra_views.size(); ++index) {
            release(terrain_extra_views[index]);
            terrain_extra_dds[index].clear();
        }
        feature_bundle = {};
        feature_assets_ready = false;
        dune_assets_ready = false;
        marsh_assets_ready = false;
        volcano_assets_ready = false;
        clutter_assets_ready = false;
        river_assets_ready = false;
        route_assets_ready = false;
        resource_assets_ready = false;
        reset_resource_buffers(); resource_animations.clear();
        unit_bodies.clear();
        city_assets_ready = false;
        mine_assets_ready = false;
        farm_assets_ready = false;
        terrain_extra_assets_ready = false;
        authored_relief_assets_ready = false;
        cache_valid = false;
        viewport_cache.clear();
        viewport_cache_bytes = 0;
        clear_geometry_vertex_buffers();
        clear_tile_geometry_cache();
        geometry_cache.clear();

    }

    void mix_content_revision(std::vector<std::uint8_t> const & data) {
        if (content_revision == 0)
            content_revision = 1469598103934665603ull;
        // Process full payloads in x86-sized blocks; retain the existing
        // 64-bit aggregate key and its per-payload invalidation boundary.
        auto digest=c3x_renderer::asset_content_hash(data.data(),data.size());
        auto bytes=reinterpret_cast<unsigned char const*>(digest.data());
        for (std::size_t i=0;i<sizeof(digest);++i) {
            auto value=bytes[i];
            content_revision ^= value;
            content_revision *= 1099511628211ull;
        }
    }

    bool load_material_only(char const * root, char const * material_relative,
                            TerrainTexture & output) {
        char path[4 * MAX_PATH];
        std::vector<std::uint8_t> material;
        if (!pack_path(root, material_relative, path, std::size(path)) ||
            !read_file(path, material) || !contains(material, "c3x.material.v0"))
            return false;
        std::array<std::pair<char const *, std::vector<std::uint8_t> *>, 3> channels = {
            std::make_pair("base_color", &output.dds),
            std::make_pair("height", &output.material_height_dds),
            std::make_pair("specular", &output.specular_dds)
        };
        for (std::size_t index = 0; index < channels.size(); ++index) {
            std::size_t position = json_member_position(material, channels[index].first);
            std::string relative;
            std::vector<std::uint8_t> dds;
            if (position == std::string::npos ||
                !json_string_after(material, "texture", position, relative) ||
                !pack_path(root, relative.c_str(), path, std::size(path)) ||
                !read_file(path, dds) || dds.size() < 156 ||
                std::memcmp(dds.data(), "DDS ", 4) != 0 ||
                std::memcmp(dds.data() + 84, "DX10", 4) != 0)
                return false;
            std::uint32_t format = read_u32(dds, 128);
            bool valid = index == 0 ?
                (format == DXGI_FORMAT_BC3_UNORM || format == DXGI_FORMAT_BC3_UNORM_SRGB) :
                (format == DXGI_FORMAT_BC4_UNORM || format == DXGI_FORMAT_BC4_SNORM);
            if (!valid)
                return false;
            mix_content_revision(dds);
            channels[index].second->swap(dds);
        }
        mix_content_revision(material);
        output.configured = true;
        return true;
    }

    bool load_decal_channel(char const * root, std::vector<std::uint8_t> const & record,
                            char const * channel, std::vector<std::uint8_t> & output,
                            std::uint32_t expected_format) {
        std::size_t position = find_text(record, std::string("\"") + channel + "\"");
        std::string relative;
        char path[4 * MAX_PATH];
        if (position == std::string::npos ||
            !json_string_after(record, "texture", position, relative) ||
            !pack_path(root, relative.c_str(), path, std::size(path)) ||
            !read_file(path, output) || output.size() < 156 ||
            std::memcmp(output.data(), "DDS ", 4) != 0 ||
            std::memcmp(output.data() + 84, "DX10", 4) != 0 ||
            read_u32(output, 128) != expected_format)
            return false;
        mix_content_revision(output);
        return true;
    }

    bool load_dds_bytes(char const * root, char const * relative,
                        std::vector<std::uint8_t> & output,
                        std::uint32_t expected_format,
                        std::uint32_t alternate_format = 0) {
        char path[4 * MAX_PATH];
        if (!pack_path(root, relative, path, std::size(path)) ||
            !read_file(path, output) || output.size() < 156 ||
            std::memcmp(output.data(), "DDS ", 4) != 0 ||
            std::memcmp(output.data() + 84, "DX10", 4) != 0 ||
            (read_u32(output, 128) != expected_format &&
             read_u32(output, 128) != alternate_format))
            return false;
        mix_content_revision(output);
        return true;
    }

    bool load_r8_field(char const * root, char const * relative,
                       std::vector<std::uint8_t> & field_pixels,
                       std::uint32_t & field_width, std::uint32_t & field_height) {
        std::vector<std::uint8_t> dds;
        char path[4 * MAX_PATH];
        if (!pack_path(root, relative, path, std::size(path)) ||
            !read_file(path, dds) || dds.size() < 149 ||
            std::memcmp(dds.data(), "DDS ", 4) != 0 ||
            std::memcmp(dds.data() + 84, "DX10", 4) != 0 ||
            read_u32(dds, 128) != DXGI_FORMAT_R8_UNORM)
            return false;
        field_width = read_u32(dds, 16);
        field_height = read_u32(dds, 12);
        if (field_width == 0 || field_height == 0 ||
            field_width > 4096 || field_height > 4096 ||
            148ull + static_cast<std::uint64_t>(field_width) * field_height > dds.size())
            return false;
        field_pixels.assign(dds.begin() + 148,
            dds.begin() + 148 + static_cast<std::ptrdiff_t>(field_width * field_height));
        mix_content_revision(dds);
        return true;
    }

    void configure_integrated_assets(char const * terrain_root,
                                     c3x_renderer::RendererPackRoots const & companion_packs) {
        std::string packs_root = terrain_root == nullptr ? "" : terrain_root;
        std::size_t slash = packs_root.find_last_of("\\/");
        if (slash == std::string::npos)
            return;
        packs_root.resize(slash);
        std::string vegetation_root = companion_packs.vegetation.empty() ?
            packs_root + "\\VegetationNormalized" : companion_packs.vegetation;
        std::string decal_root = companion_packs.decals.empty() ?
            packs_root + "\\DecalsNormalized" : companion_packs.decals;
        std::string terrain_elements_root = companion_packs.terrain_elements.empty() ?
            packs_root + "\\TerrainElementsNormalized" : companion_packs.terrain_elements;
        std::string shore_root = companion_packs.shore.empty() ?
            packs_root + "\\ShoreNormalized" : companion_packs.shore;
        bool dune_material = load_material_only(
            terrain_root, "materials\\library\\desert_hills.json", dune_surface);
        std::vector<std::uint8_t> decal_record;
        char decal_path[4 * MAX_PATH];
        bool decal = pack_path(decal_root.c_str(),
            "decals\\terrain_desert_dune_decal_01.json", decal_path, std::size(decal_path)) &&
            read_file(decal_path, decal_record) && contains(decal_record, "c3x.decal.v0") &&
            load_decal_channel(decal_root.c_str(), decal_record, "base_color",
                               dune_decal_base_dds, DXGI_FORMAT_BC3_UNORM_SRGB) &&
            load_decal_channel(decal_root.c_str(), decal_record, "height",
                               dune_decal_height_dds, DXGI_FORMAT_BC5_UNORM);
        if (decal)
            mix_content_revision(decal_record);
        dune_assets_ready = dune_material && decal;

        TerrainTexture & hills = terrain_textures[5];
        std::uint32_t hill_width = 0, hill_height = 0;
        bool hill_geometry = load_r8_field(
            terrain_root, "textures\\relief\\hills\\standard\\height_lod0.dds",
            hills.height_pixels, hill_width, hill_height);
        if (pickup_profile) {
            std::string root = packs_root + "\\TerrainProfileR1";
            hill_geometry = load_r8_field(root.c_str(), "height.dds",
                hills.height_pixels, hill_width, hill_height);
            std::string path = shore_root + "\\cliff_runtime.bin";
            std::vector<std::uint8_t> bytes;
            cliff_assets_ready = read_file(path.c_str(), bytes) &&
                load_feature_bundle(path, cliff_bundle) && cliff_bundle.assets.size() == 8 &&
                cliff_bundle.texture_paths.size() == 32 &&
                find_feature_group(cliff_bundle, "cliff_large") != nullptr &&
                find_feature_group(cliff_bundle, "cliff_small") != nullptr;
            unsigned formats[] = {72,83,80,72};
            if (cliff_assets_ready) {
                mix_content_revision(bytes);
                for (std::size_t i=0; i<cliff_dds.size(); ++i)
                    cliff_assets_ready = cliff_assets_ready && load_dds_bytes(shore_root.c_str(),
                        cliff_bundle.texture_paths[i].c_str(), cliff_dds[i], formats[i%4], 0);
            }
            trace.write("pickup-assets", cliff_assets_ready && hill_geometry ?
                "selected hill and 8 source cliffs / 32 channels ready" : "selected source asset missing", true);
        }
        if (hill_geometry) {
            hills.height_width = hill_width;
            hills.height_height = hill_height;
            measure_field_limits(hills.height_pixels,
                                 hills.height_minimum, hills.height_maximum);
            hills.height_scale_px = 52.0f;
            hills.relief_profile = 2;
        }

        TerrainTexture & mountains = terrain_textures[6];
        bool mountain_geometry = true;
        for (int variant = 0; variant < 5; ++variant) {
            char height_path[256], blend_path[256];
            sprintf_s(height_path,
                "textures\\relief\\mountains\\standard\\variant_%02d\\height_lod0.dds",
                variant + 1);
            sprintf_s(blend_path,
                "textures\\relief\\mountains\\standard\\variant_%02d\\blend_lod0.dds",
                variant + 1);
            std::uint32_t height_width = 0, height_height = 0;
            std::uint32_t blend_width = 0, blend_height = 0;
            mountain_geometry = mountain_geometry &&
                load_r8_field(terrain_root, height_path,
                    mountains.relief_height_variants[variant],
                    height_width, height_height) &&
                load_r8_field(terrain_root, blend_path,
                    mountains.relief_blend_variants[variant],
                    blend_width, blend_height) &&
                height_width == blend_width && height_height == blend_height;
            mountains.relief_variant_widths[variant] = height_width;
            mountains.relief_variant_heights[variant] = height_height;
            measure_field_limits(mountains.relief_height_variants[variant],
                mountains.relief_height_minimum[variant],
                mountains.relief_height_maximum[variant]);
            measure_field_limits(mountains.relief_blend_variants[variant],
                mountains.relief_blend_minimum[variant],
                mountains.relief_blend_maximum[variant]);
        }
        if (mountain_geometry) {
            mountains.height_scale_px = 104.0f;
            mountains.relief_profile = 4;
        }
        authored_relief_assets_ready = hills.configured && hill_geometry &&
            mountains.configured && mountain_geometry;

        std::vector<std::uint8_t> marsh_record;
        char marsh_path[4 * MAX_PATH];
        bool marsh_decal = pack_path(decal_root.c_str(),
            "decals\\terrain_marsh_decal_01.json", marsh_path, std::size(marsh_path)) &&
            read_file(marsh_path, marsh_record) && contains(marsh_record, "c3x.decal.v0") &&
            load_decal_channel(decal_root.c_str(), marsh_record, "base_color",
                               marsh_decal_base_dds, DXGI_FORMAT_BC3_UNORM_SRGB) &&
            load_decal_channel(decal_root.c_str(), marsh_record, "height",
                               marsh_decal_height_dds, DXGI_FORMAT_BC5_UNORM) &&
            load_decal_channel(decal_root.c_str(), marsh_record, "specular",
                               marsh_decal_specular_dds, DXGI_FORMAT_BC4_UNORM);
        if (marsh_decal)
            mix_content_revision(marsh_record);
        marsh_assets_ready = terrain_textures[9].configured && marsh_decal;

        TerrainTexture & volcano = terrain_textures[10];
        std::uint32_t volcano_width = 0, volcano_height = 0;
        std::uint32_t blend_width = 0, blend_height = 0;
        bool volcano_geometry = load_r8_field(
            terrain_elements_root.c_str(),
            "textures\\terrain_elements\\terrain_feature_volcano\\height_lod0.dds",
            volcano.height_pixels, volcano_width, volcano_height) &&
            load_r8_field(
                terrain_elements_root.c_str(),
                "textures\\terrain_elements\\terrain_feature_volcano\\blend_lod0.dds",
                volcano.blend_pixels, blend_width, blend_height) &&
            volcano_width == blend_width && volcano_height == blend_height;
        if (volcano_geometry) {
            volcano.height_width = volcano_width;
            volcano.height_height = volcano_height;
            measure_field_limits(volcano.height_pixels,
                                 volcano.height_minimum, volcano.height_maximum);
            measure_field_limits(volcano.blend_pixels,
                                 volcano.blend_minimum, volcano.blend_maximum);
            volcano.height_scale_px = 104.0f;
            volcano.relief_profile = 5;
        }
        volcano_assets_ready = volcano.configured && volcano_geometry &&
            load_dds_bytes(terrain_root, "textures\\water\\volcano\\base.dds",
                           volcano_base_dds, DXGI_FORMAT_BC3_UNORM_SRGB,
                           DXGI_FORMAT_BC3_UNORM) &&
            load_dds_bytes(terrain_root, "textures\\water\\volcano\\height.dds",
                           volcano_height_dds, DXGI_FORMAT_BC5_UNORM) &&
            load_dds_bytes(terrain_root, "textures\\water\\volcano\\active_base.dds",
                           volcano_active_base_dds, DXGI_FORMAT_BC3_UNORM_SRGB,
                           DXGI_FORMAT_BC3_UNORM) &&
            load_dds_bytes(terrain_root, "textures\\water\\volcano\\active_specular.dds",
                           volcano_active_specular_dds, DXGI_FORMAT_BC4_UNORM);

        std::vector<std::uint8_t> water_clutter_record, grass_clutter_record,
            plains_clutter_record;
        char clutter_path[4 * MAX_PATH];
        bool water_clutter = pack_path(decal_root.c_str(),
            "decals\\terrain_water_ocean_decal_01.json", clutter_path,
            std::size(clutter_path)) && read_file(clutter_path, water_clutter_record) &&
            load_decal_channel(decal_root.c_str(), water_clutter_record, "base_color",
                               water_clutter_base_dds, DXGI_FORMAT_BC3_UNORM_SRGB) &&
            load_decal_channel(decal_root.c_str(), water_clutter_record, "height",
                               water_clutter_height_dds, DXGI_FORMAT_BC5_UNORM);
        bool grass_clutter = pack_path(decal_root.c_str(),
            "decals\\terrain_grassland_decal_02.json", clutter_path,
            std::size(clutter_path)) && read_file(clutter_path, grass_clutter_record) &&
            load_decal_channel(decal_root.c_str(), grass_clutter_record, "base_color",
                               grass_clutter_base_dds, DXGI_FORMAT_BC3_UNORM_SRGB) &&
            load_decal_channel(decal_root.c_str(), grass_clutter_record, "height",
                               grass_clutter_height_dds, DXGI_FORMAT_BC5_UNORM);
        bool plains_clutter = pack_path(decal_root.c_str(),
            "decals\\terrain_plains_decal_01.json", clutter_path,
            std::size(clutter_path)) && read_file(clutter_path, plains_clutter_record) &&
            load_decal_channel(decal_root.c_str(), plains_clutter_record, "base_color",
                               plains_clutter_base_dds, DXGI_FORMAT_BC3_UNORM_SRGB) &&
            load_decal_channel(decal_root.c_str(), plains_clutter_record, "height",
                               plains_clutter_height_dds, DXGI_FORMAT_BC5_UNORM);
        clutter_assets_ready = water_clutter && grass_clutter && plains_clutter;
        if (clutter_assets_ready) {
            mix_content_revision(water_clutter_record);
            mix_content_revision(grass_clutter_record);
            mix_content_revision(plains_clutter_record);
        }

        std::string bundle_path = vegetation_root + "\\vegetation_runtime.bin";
        feature_assets_ready = load_feature_bundle(bundle_path, feature_bundle) &&
            feature_bundle.texture_paths.size() <= feature_texture_dds.size();
        if (feature_assets_ready) {
            for (std::size_t index = 0; index < feature_bundle.texture_paths.size(); ++index) {
                char texture_path[4 * MAX_PATH];
                if (!pack_path(vegetation_root.c_str(),
                               feature_bundle.texture_paths[index].c_str(), texture_path,
                               std::size(texture_path)) ||
                    !read_file(texture_path, feature_texture_dds[index]) ||
                    feature_texture_dds[index].size() < 156 ||
                    std::memcmp(feature_texture_dds[index].data(), "DDS ", 4) != 0 ||
                    std::memcmp(feature_texture_dds[index].data() + 84, "DX10", 4) != 0 ||
                    (read_u32(feature_texture_dds[index], 128) != DXGI_FORMAT_BC1_UNORM_SRGB &&
                     read_u32(feature_texture_dds[index], 128) != DXGI_FORMAT_BC1_UNORM &&
                     read_u32(feature_texture_dds[index], 128) != DXGI_FORMAT_BC3_UNORM_SRGB &&
                     read_u32(feature_texture_dds[index], 128) != DXGI_FORMAT_BC3_UNORM)) {
                    feature_assets_ready = false;
                    break;
                }
                mix_content_revision(feature_texture_dds[index]);
            }
        }

        struct ExtraTexture {
            char const * path;
            std::uint32_t format;
            std::uint32_t alternate;
        };
        std::array<ExtraTexture, 19> const extras = {{
            {"textures\\beach_height.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\beach_specular.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\cliff_height.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\cliff_specular.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\water\\surface\\gloss.dds", DXGI_FORMAT_BC1_UNORM_SRGB,
             DXGI_FORMAT_BC1_UNORM},
            {"textures\\water\\surface\\tiling_mask.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\water\\surface\\non_tiling_mask.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\water\\surface\\small_secondary_lean0.dds",
             DXGI_FORMAT_R16G16B16A16_UNORM, 0},
            {"textures\\water\\surface\\small_secondary_lean1.dds",
             DXGI_FORMAT_R16G16_UNORM, 0},
            {"textures\\water\\effects\\ripples_primary.dds",
             DXGI_FORMAT_BC3_UNORM_SRGB, DXGI_FORMAT_BC3_UNORM},
            {"textures\\water\\effects\\turbulence.dds",
             DXGI_FORMAT_BC3_UNORM_SRGB, DXGI_FORMAT_BC3_UNORM},
            {"textures\\water\\profiles\\coast\\dark.dds",
             DXGI_FORMAT_R16G16B16A16_FLOAT, 0},
            {"textures\\water\\profiles\\coast\\scatter.dds",
             DXGI_FORMAT_R16G16B16A16_FLOAT, 0},
            {"textures\\water\\surface\\tiling_normal0.dds", DXGI_FORMAT_BC5_UNORM, 0},
            {"textures\\water\\surface\\tiling_normal1.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\water\\surface\\non_tiling_normal0.dds", DXGI_FORMAT_BC5_UNORM, 0},
            {"textures\\water\\surface\\non_tiling_normal1.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\mtn_desert_base_height.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\mtn_desert_base_specular.dds", DXGI_FORMAT_BC4_UNORM, 0},
        }};
        terrain_extra_assets_ready = true;
        for (std::size_t index = 0; index < extras.size(); ++index)
            terrain_extra_assets_ready = terrain_extra_assets_ready && load_dds_bytes(
                terrain_root, extras[index].path, terrain_extra_dds[index],
                extras[index].format, extras[index].alternate);

        std::array<ExtraTexture, 10> const river_channels = {{
            {"textures\\river_base_color.dds", DXGI_FORMAT_BC3_UNORM_SRGB,
             DXGI_FORMAT_BC3_UNORM},
            {"textures\\river_height.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\river_specular.dds", DXGI_FORMAT_BC4_UNORM, 0},
            {"textures\\water\\surface\\river_lean0.dds",
             DXGI_FORMAT_R16G16B16A16_UNORM, 0},
            {"textures\\water\\surface\\river_lean1.dds",
             DXGI_FORMAT_R16G16_UNORM, 0},
            {"textures\\water\\river\\source_decal_base.dds",
             DXGI_FORMAT_BC3_UNORM_SRGB, DXGI_FORMAT_BC3_UNORM},
            {"textures\\water\\river\\source_decal_height.dds",
             DXGI_FORMAT_BC5_UNORM, 0},
            {"textures\\water\\river\\clutter_decal_base.dds",
             DXGI_FORMAT_BC3_UNORM_SRGB, DXGI_FORMAT_BC3_UNORM},
            {"textures\\water\\river\\clutter_decal_height.dds",
             DXGI_FORMAT_BC5_UNORM, 0},
            {"textures\\water\\relief\\river_bank_noise\\height_lod0.dds",
             DXGI_FORMAT_R8_UNORM, 0},
        }};
        bool river_surface_ready = true;
        for (std::size_t index = 0; index < river_channels.size(); ++index)
            river_surface_ready = river_surface_ready && load_dds_bytes(
                terrain_root, river_channels[index].path, river_surface_dds[index],
                river_channels[index].format, river_channels[index].alternate);
        std::string river_bundle_path = shore_root + "\\shore_runtime.bin";
        std::vector<std::uint8_t> river_bundle_bytes;
        bool river_rocks_ready = read_file(river_bundle_path.c_str(), river_bundle_bytes) &&
            load_feature_bundle(river_bundle_path, river_rock_bundle) &&
            river_rock_bundle.texture_paths.size() == river_rock_texture_dds.size() &&
            find_feature_group(river_rock_bundle, "river_rock") != nullptr;
        if (river_rocks_ready) {
            mix_content_revision(river_bundle_bytes);
            for (std::size_t index = 0; index < river_rock_texture_dds.size(); ++index) {
                char texture_path[4 * MAX_PATH];
                river_rocks_ready = pack_path(
                    shore_root.c_str(), river_rock_bundle.texture_paths[index].c_str(),
                    texture_path, std::size(texture_path)) &&
                    read_file(texture_path, river_rock_texture_dds[index]) &&
                    river_rock_texture_dds[index].size() >= 156 &&
                    std::memcmp(river_rock_texture_dds[index].data(), "DDS ", 4) == 0 &&
                    std::memcmp(river_rock_texture_dds[index].data() + 84, "DX10", 4) == 0 &&
                    (read_u32(river_rock_texture_dds[index], 128) == DXGI_FORMAT_BC1_UNORM_SRGB ||
                     read_u32(river_rock_texture_dds[index], 128) == DXGI_FORMAT_BC1_UNORM);
                if (!river_rocks_ready)
                    break;
                mix_content_revision(river_rock_texture_dds[index]);
            }
        }
        river_assets_ready = river_surface_ready && river_rocks_ready;

        std::string route_root = packs_root + "\\RouteStylesNormalized";
        char const * route_textures[] = {
            "textures\\routes\\base_color_f5c58170975c3c8b.dds",
            "textures\\routes\\base_color_6c2f0cc0e38fff7b.dds",
            "textures\\routes\\base_color_5ae696c133345a83.dds",
            "textures\\routes\\base_color_ba7297551e6ece35.dds",
            "textures\\routes\\base_color_3631908100843d0d.dds",
            "textures\\routes\\base_color_678d822a9338aab5.dds",
            "textures\\routes\\base_color_fc46370d90163f16.dds",
            "textures\\routes\\base_color_a832a6518fdab3b1.dds",
            "textures\\routes\\base_color_c3ae5ee0b5879164.dds",
            "textures\\routes\\base_color_39a5855bea35f6d1.dds",
        };
        route_assets_ready = true;
        for (std::size_t index = 0; index < route_texture_dds.size(); ++index)
            route_assets_ready = route_assets_ready && load_dds_bytes(
                route_root.c_str(), route_textures[index], route_texture_dds[index],
                DXGI_FORMAT_BC3_UNORM_SRGB, DXGI_FORMAT_BC3_UNORM);

        auto load_runtime_bundle = [&](std::string const & root, char const * filename,
                                       c3x_renderer::FeatureBundle & bundle,
                                       auto & texture_bytes) {
            std::string runtime_path = root + "\\" + filename;
            std::vector<std::uint8_t> runtime_bytes;
            if (!read_file(runtime_path.c_str(), runtime_bytes) ||
                !load_feature_bundle(runtime_path, bundle) ||
                bundle.texture_paths.size() != texture_bytes.size())
                return false;
            mix_content_revision(runtime_bytes);
            for (std::size_t index = 0; index < texture_bytes.size(); ++index)
                if (!load_dds_bytes(root.c_str(), bundle.texture_paths[index].c_str(),
                                    texture_bytes[index], DXGI_FORMAT_BC1_UNORM_SRGB,
                                    DXGI_FORMAT_BC1_UNORM))
                    return false;
            return true;
        };
        std::string bridge_root = packs_root + "\\RouteDoodadsNormalized";
        route_assets_ready = route_assets_ready && load_runtime_bundle(
            bridge_root, "bridge_runtime.bin", bridge_bundle, bridge_texture_dds);

        std::string resource_root = packs_root + "\\ResourceNormalized";
        resource_assets_ready = load_runtime_bundle(
            resource_root, "resource_runtime.bin", resource_bundle, resource_texture_dds);

        // Bind the catalog once. Payloads are resident only when needed.
        char unit_pack[128]="UnitAnimationFidelity",unit_root[4*MAX_PATH];
        if(GetEnvironmentVariableA("C3X_RENDERER_UNIT_PACK",unit_pack,sizeof(unit_pack))>=sizeof(unit_pack))
            strcpy_s(unit_pack,"UnitAnimationFidelity");
        if(unit_rendering_enabled && (!pack_path(packs_root.c_str(),unit_pack,unit_root,std::size(unit_root)) ||
            !load_unit_animations(unit_root))) {
            unit_bodies.clear();trace.write("unit-bind","complete unit pack rejected; native bodies retained",true);
        }
        if (pickup_profile) {
            std::string root = packs_root + "\\ResourceAnimationRuntime";
            std::vector<std::uint8_t> definitions;
            if (read_file((root + "\\bindings.json").c_str(), definitions)) {
                auto bindings = json_member_position(definitions, "bindings");
                for (char const * name : {"horses","cattle","wheat","fish","whales","game","furs","ivory","bananas","rubber"}) {
                    auto record = json_member_position(definitions, name, bindings);
                    ResourceAnimation animation; animation.name = name;
                    std::string mesh_path, texture_path; float count = 0;
                    char path[4 * MAX_PATH]; std::vector<std::uint8_t> bytes;
                    bool ok = record != std::string::npos &&
                        json_string_after(definitions,"mesh",record,mesh_path) &&
                        json_string_after(definitions,"texture",record,texture_path) &&
                        json_number_after(definitions,"scale",record,animation.scale) &&
                        json_number_after(definitions,"yaw",record,animation.yaw) &&
                        json_number_after(definitions,"count",record,count) && count>=1 && count<=5 &&
                        json_number_after(definitions,"offset_x",record,animation.offset[0]) &&
                        json_number_after(definitions,"offset_y",record,animation.offset[1]) &&
                        json_number_after(definitions,"offset_z",record,animation.offset[2]) &&
                        animation.scale>0 && animation.scale<100 &&
                        pack_path(root.c_str(),mesh_path.c_str(),path,std::size(path)) &&
                        read_file(path,bytes) && c3x_renderer::decode_animation_mesh(bytes,animation.mesh) &&
                        animation.source_bounds.prepare(animation.mesh) &&
                        load_dds_bytes(root.c_str(),texture_path.c_str(),animation.dds,
                            DXGI_FORMAT_BC1_UNORM_SRGB,DXGI_FORMAT_BC1_UNORM);
                    if (ok) {
                        animation.count = static_cast<unsigned>(count);
                        mix_content_revision(bytes); mix_content_revision(animation.dds);
                        resource_animations.push_back(std::move(animation));
                    }
                    char detail[192]; sprintf_s(detail,"resource=%s result=%s facing=SE payload_bytes=%zu",
                        name,ok?"ready":"rejected",bytes.size()); trace.write("animation-bind",detail,true);
                }
                mix_content_revision(definitions);
            }
        }

        std::string city_root = packs_root + "\\CityComponentsNormalized";
        std::string wall_root = packs_root + "\\CityAdjunctsNormalized";
        std::string city_runtime_path = city_root + "\\city_runtime.bin";
        std::string wall_runtime_path = wall_root + "\\wall_runtime.bin";
        std::vector<std::uint8_t> city_runtime_bytes, wall_runtime_bytes;
        city_assets_ready = read_file(city_runtime_path.c_str(), city_runtime_bytes) &&
            load_feature_bundle(city_runtime_path, city_bundle) &&
            city_bundle.texture_paths.size() == 8u &&
            read_file(wall_runtime_path.c_str(), wall_runtime_bytes) &&
            load_feature_bundle(wall_runtime_path, wall_bundle) &&
            wall_bundle.texture_paths.size() == 1u;
        if (city_assets_ready) {
            mix_content_revision(city_runtime_bytes);
            mix_content_revision(wall_runtime_bytes);
            for (std::size_t index = 0; index < city_base_dds.size(); ++index) {
                city_assets_ready = city_assets_ready && load_dds_bytes(
                    city_root.c_str(), city_bundle.texture_paths[index].c_str(),
                    city_base_dds[index], DXGI_FORMAT_BC1_UNORM_SRGB, DXGI_FORMAT_BC1_UNORM) &&
                    load_dds_bytes(city_root.c_str(), city_bundle.texture_paths[index + 4u].c_str(),
                                   city_emissive_dds[index], DXGI_FORMAT_BC1_UNORM_SRGB,
                                   DXGI_FORMAT_BC1_UNORM);
            }
            city_assets_ready = city_assets_ready && load_dds_bytes(
                wall_root.c_str(), wall_bundle.texture_paths[0].c_str(),
                wall_texture_dds, DXGI_FORMAT_BC1_UNORM_SRGB, DXGI_FORMAT_BC1_UNORM);
        }

        std::string site_root = packs_root + "\\TileSitesRuntime";
        std::vector<std::uint8_t> site_bytes;
        site_assets_ready=read_file((site_root+"\\sites.bin").c_str(),site_bytes) &&
            load_feature_bundle(site_root+"\\sites.bin",site_bundle) && site_bundle.texture_paths.size()==8;
        if(site_assets_ready) {
            mix_content_revision(site_bytes);
            for(unsigned i=0;i<site_dds.size();++i)
                site_assets_ready=site_assets_ready &&
                    (load_dds_bytes(site_root.c_str(),site_bundle.texture_paths[i].c_str(),site_dds[i],DXGI_FORMAT_BC1_UNORM_SRGB,DXGI_FORMAT_BC3_UNORM_SRGB) ||
                     load_dds_bytes(site_root.c_str(),site_bundle.texture_paths[i].c_str(),site_dds[i],DXGI_FORMAT_BC1_UNORM,DXGI_FORMAT_BC3_UNORM));
        }
        std::string improvement_root = packs_root + "\\ImprovementsNormalized";
        auto load_improvement = [&](char const * runtime_name,
                                    c3x_renderer::FeatureBundle & bundle,
                                    auto & base_dds, auto & emissive_dds,
                                    DWORD error_base) {
            std::string runtime_path = improvement_root + "\\" + runtime_name;
            std::vector<std::uint8_t> runtime_bytes;
            if (!read_file(runtime_path.c_str(), runtime_bytes) ||
                !load_feature_bundle(runtime_path, bundle) ||
                bundle.texture_paths.size() != base_dds.size() + emissive_dds.size()) {
                SetLastError(error_base);
                return false;
            }
            mix_content_revision(runtime_bytes);
            for (std::size_t source_index = 0; source_index < bundle.texture_paths.size(); ++source_index) {
                char path[4 * MAX_PATH];
                std::vector<std::uint8_t> & output = source_index < base_dds.size()
                    ? base_dds[source_index] : emissive_dds[source_index - base_dds.size()];
                if (!pack_path(improvement_root.c_str(), bundle.texture_paths[source_index].c_str(),
                               path, std::size(path)) ||
                    !read_file(path, output) || output.size() < 156 ||
                    std::memcmp(output.data(), "DDS ", 4) != 0 ||
                    std::memcmp(output.data() + 84, "DX10", 4) != 0) {
                    SetLastError(error_base + 1u + static_cast<DWORD>(source_index));
                    return false;
                }
                std::uint32_t format = read_u32(output, 128);
                if (format != DXGI_FORMAT_BC1_UNORM &&
                    format != DXGI_FORMAT_BC1_UNORM_SRGB &&
                    format != DXGI_FORMAT_BC3_UNORM &&
                    format != DXGI_FORMAT_BC3_UNORM_SRGB) {
                    SetLastError(error_base + 1u + static_cast<DWORD>(source_index));
                    return false;
                }
                mix_content_revision(output);
            }
            return true;
        };
        mine_assets_ready = load_improvement(
            "mine_runtime.bin", mine_bundle, mine_base_dds, mine_emissive_dds, 1801u);
        farm_assets_ready = load_improvement(
            "farm_runtime.bin", farm_bundle, farm_base_dds, farm_emissive_dds, 1831u);
    }

    bool configure_asset(int terrain_type, char const * root, char const * logical_asset_id) {
        if (terrain_type < 0 || terrain_type >= c3x_renderer::terrain_type_count || root == nullptr ||
            root[0] == '\0' || logical_asset_id == nullptr || logical_asset_id[0] == '\0')
            return false;
        char path[4 * MAX_PATH];
        std::vector<std::uint8_t> manifest, mesh, material, dds, height_dds;
        if (!pack_path(root, "manifest.json", path, std::size(path)) || !read_file(path, manifest) ||
            !contains(manifest, "c3x.asset_pack.v0"))
            return false;
        std::string logical_marker = std::string("\"") + logical_asset_id + "\"";
        std::size_t asset_position = find_text(manifest, logical_marker);
        std::string mesh_relative, material_relative, texture_relative;
        if (asset_position == std::string::npos ||
            !json_string_after(manifest, "mesh", asset_position, mesh_relative) ||
            !json_string_after(manifest, "material", asset_position, material_relative))
            return false;
        if (!pack_path(root, mesh_relative.c_str(), path, std::size(path)) || !read_file(path, mesh) ||
            !contains(mesh, "c3x.normalized_mesh.v0") || !contains(mesh, "\"primitive\": \"triangles\""))
            return false;
        if (!pack_path(root, material_relative.c_str(), path, std::size(path)) || !read_file(path, material) ||
            !contains(material, "c3x.material.v0") || !json_string_after(material, "texture", json_member_position(material, "base_color"), texture_relative))
            return false;
        if (!pack_path(root, texture_relative.c_str(), path, std::size(path)) || !read_file(path, dds) ||
            dds.size() < 164 || std::memcmp(dds.data(), "DDS ", 4) != 0 || read_u32(dds, 4) != 124 ||
            std::memcmp(dds.data() + 84, "DX10", 4) != 0 ||
            (read_u32(dds, 128) != DXGI_FORMAT_BC3_UNORM && read_u32(dds, 128) != DXGI_FORMAT_BC3_UNORM_SRGB))
            return false;
        TerrainTexture & texture = terrain_textures[terrain_type];
        release(texture.view);
        release(texture.material_height_view);
        release(texture.specular_view);
        release(texture.elevated_view);
        release(texture.elevated_height_view);
        release(texture.elevated_specular_view);
        for (ID3D11ShaderResourceView *& view : texture.relief_layer_views)
            release(view);
        for (ID3D11ShaderResourceView *& view : texture.water_surface_views)
            release(view);
        texture.dds.swap(dds);
        texture.material_height_dds.clear();
        texture.specular_dds.clear();
        texture.elevated_dds.clear();
        texture.elevated_height_dds.clear();
        texture.elevated_specular_dds.clear();
        for (std::vector<std::uint8_t> & layer : texture.relief_layer_dds)
            layer.clear();
        for (std::vector<std::uint8_t> & channel : texture.water_surface_dds)
            channel.clear();
        texture.height_pixels.clear();
        texture.height_width = 0;
        texture.height_height = 0;
        texture.height_scale_px = 0.0f;
        texture.relief_profile = 0;
        for (auto const & channel : std::array<std::pair<char const *, std::vector<std::uint8_t> *>, 2>{
                 std::make_pair("height", &texture.material_height_dds),
                 std::make_pair("specular", &texture.specular_dds)}) {
            std::size_t channel_position = json_member_position(material, channel.first);
            std::string channel_relative;
            std::vector<std::uint8_t> channel_dds;
            if (channel_position == std::string::npos ||
                !json_string_after(material, "texture", channel_position, channel_relative) ||
                !pack_path(root, channel_relative.c_str(), path, std::size(path)) ||
                !read_file(path, channel_dds) || channel_dds.size() < 156 ||
                std::memcmp(channel_dds.data(), "DDS ", 4) != 0 || read_u32(channel_dds, 4) != 124 ||
                std::memcmp(channel_dds.data() + 84, "DX10", 4) != 0 ||
                (read_u32(channel_dds, 128) != DXGI_FORMAT_BC4_UNORM &&
                 read_u32(channel_dds, 128) != DXGI_FORMAT_BC4_SNORM))
                continue;
            channel.second->swap(channel_dds);
        }
        std::size_t elevated_position = json_member_position(material, "elevated");
        if (elevated_position != std::string::npos) {
            for (auto const & channel : std::array<std::pair<char const *, std::vector<std::uint8_t> *>, 3>{
                     std::make_pair("base_color", &texture.elevated_dds),
                     std::make_pair("height", &texture.elevated_height_dds),
                     std::make_pair("specular", &texture.elevated_specular_dds)}) {
                std::size_t channel_position = json_member_position(material, channel.first, elevated_position);
                std::string channel_relative;
                std::vector<std::uint8_t> channel_dds;
                if (channel_position == std::string::npos ||
                    !json_string_after(material, "texture", channel_position, channel_relative) ||
                    !pack_path(root, channel_relative.c_str(), path, std::size(path)) ||
                    !read_file(path, channel_dds) || channel_dds.size() < 156 ||
                    std::memcmp(channel_dds.data(), "DDS ", 4) != 0 || read_u32(channel_dds, 4) != 124 ||
                    std::memcmp(channel_dds.data() + 84, "DX10", 4) != 0)
                    return false;
                std::uint32_t format = read_u32(channel_dds, 128);
                bool format_valid = channel.first == std::string("base_color") ?
                    (format == DXGI_FORMAT_BC3_UNORM || format == DXGI_FORMAT_BC3_UNORM_SRGB) :
                    (format == DXGI_FORMAT_BC4_UNORM || format == DXGI_FORMAT_BC4_SNORM);
                if (!format_valid)
                    return false;
                channel.second->swap(channel_dds);
            }
        }
        std::size_t authored_layers_position = json_member_position(material, "authored_layers");
        if (authored_layers_position != std::string::npos) {
            std::array<char const *, 5> layer_names = {};
            std::size_t layer_count = 0;
            if (terrain_type == 6) {
                layer_names = {"snow", "desert_base", "desert_stripe_1", "desert_stripe_2", "desert_stripe_3"};
                layer_count = 5;
            } else if (terrain_type == 11) {
                layer_names = {"beach", "cliff", "cliff_white", nullptr, nullptr};
                layer_count = 3;
            } else {
                return false;
            }
            for (std::size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
                std::size_t layer_position = json_member_position(material, layer_names[layer_index], authored_layers_position);
                std::size_t channel_position = json_member_position(material, "base_color", layer_position);
                std::string channel_relative;
                std::vector<std::uint8_t> channel_dds;
                if (layer_position == std::string::npos || channel_position == std::string::npos ||
                    !json_string_after(material, "texture", channel_position, channel_relative) ||
                    !pack_path(root, channel_relative.c_str(), path, std::size(path)) ||
                    !read_file(path, channel_dds) || channel_dds.size() < 164 ||
                    std::memcmp(channel_dds.data(), "DDS ", 4) != 0 || read_u32(channel_dds, 4) != 124 ||
                    std::memcmp(channel_dds.data() + 84, "DX10", 4) != 0 ||
                    (read_u32(channel_dds, 128) != DXGI_FORMAT_BC3_UNORM &&
                     read_u32(channel_dds, 128) != DXGI_FORMAT_BC3_UNORM_SRGB))
                    return false;
                texture.relief_layer_dds[layer_index].swap(channel_dds);
            }
        }
        std::size_t water_surface_position = json_member_position(material, "water_surface");
        if (water_surface_position != std::string::npos) {
            if (terrain_type != 11)
                return false;
            std::array<char const *, 5> channel_names = {
                "large_lean0", "large_lean1", "small_lean0", "small_lean1", "foam"
            };
            std::array<std::uint32_t, 5> expected_formats = {
                DXGI_FORMAT_R16G16B16A16_UNORM, DXGI_FORMAT_R16G16_UNORM,
                DXGI_FORMAT_R16G16B16A16_UNORM, DXGI_FORMAT_R16G16_UNORM,
                DXGI_FORMAT_BC3_UNORM_SRGB,
            };
            for (std::size_t channel_index = 0; channel_index < channel_names.size(); ++channel_index) {
                std::size_t channel_position = json_member_position(material, channel_names[channel_index], water_surface_position);
                std::string channel_relative;
                std::vector<std::uint8_t> channel_dds;
                if (channel_position == std::string::npos ||
                    !json_string_after(material, "texture", channel_position, channel_relative) ||
                    !pack_path(root, channel_relative.c_str(), path, std::size(path)) ||
                    !read_file(path, channel_dds) || channel_dds.size() < 156 ||
                    std::memcmp(channel_dds.data(), "DDS ", 4) != 0 || read_u32(channel_dds, 4) != 124 ||
                    std::memcmp(channel_dds.data() + 84, "DX10", 4) != 0)
                    return false;
                std::uint32_t format = read_u32(channel_dds, 128);
                bool format_matches = format == expected_formats[channel_index] ||
                    (channel_index == 4 && format == DXGI_FORMAT_BC3_UNORM);
                if (!format_matches)
                    return false;
                texture.water_surface_dds[channel_index].swap(channel_dds);
            }
        }
        std::size_t relief_position = json_member_position(material, "relief");
        if (relief_position != std::string::npos) {
            std::string height_relative, profile;
            float scale = 0.0f;
            if (!json_string_after(material, "texture", relief_position, height_relative) ||
                !json_string_after(material, "profile", relief_position, profile) ||
                !json_number_after(material, "height_scale_px", relief_position, scale) ||
                scale < 0.0f || scale > 256.0f ||
                !pack_path(root, height_relative.c_str(), path, std::size(path)) ||
                !read_file(path, height_dds) || height_dds.size() < 149 ||
                std::memcmp(height_dds.data(), "DDS ", 4) != 0 || read_u32(height_dds, 4) != 124 ||
                std::memcmp(height_dds.data() + 84, "DX10", 4) != 0 ||
                read_u32(height_dds, 128) != DXGI_FORMAT_R8_UNORM)
                return false;
            std::uint32_t height_width = read_u32(height_dds, 16);
            std::uint32_t height_height = read_u32(height_dds, 12);
            if (height_width == 0 || height_height == 0 || height_width > 4096 || height_height > 4096 ||
                148ull + static_cast<std::uint64_t>(height_width) * height_height > height_dds.size())
                return false;
            if (profile == "continuous")
                texture.relief_profile = 1;
            else if (profile == "connected_hills")
                texture.relief_profile = 2;
            else if (profile == "mountain_massif")
                texture.relief_profile = 3;
            else if (profile == "mountain_atlas")
                texture.relief_profile = 4;
            else
                return false;
            texture.height_width = height_width;
            texture.height_height = height_height;
            texture.height_scale_px = scale;
            texture.height_pixels.assign(height_dds.begin() + 148,
                height_dds.begin() + 148 + static_cast<std::ptrdiff_t>(height_width * height_height));
        }
        texture.configured = true;
        mix_content_revision(manifest);
        mix_content_revision(mesh);
        mix_content_revision(material);
        mix_content_revision(texture.dds);
        return true;
    }

    bool configure_pack(char const * root) {
        clear_terrain_assets();
        previous_content_revision = content_revision;
        content_revision = 0;
        return configure_asset(2, root, "terrain/grassland/base");
    }

    bool configure_definitions(char const * mod_root, char const * default_path,
                               char const * scenario_path, char const * custom_path) {
        terrain_preparation.clear();
        for(auto& scratch:terrain_scratch)scratch.reset();foreground_terrain_scratch.reset();
        for(auto& scratch:world_ground_scratch){scratch.rivers.reset_world();scratch.reset_tile();}
        char requested_profile[32] = {};
        GetEnvironmentVariableA("C3X_RENDERER_VISUAL_PROFILE", requested_profile, sizeof(requested_profile));
        bool use_city=requested_profile[0]==0 || std::strcmp(requested_profile,"city-fidelity")==0;
        bool use_environment=use_city || std::strcmp(requested_profile,"environment-refresh")==0;
        bool use_fidelity = use_environment || requested_profile[0] == 0 || std::strcmp(requested_profile, "source-fidelity-r13") == 0;
        if(fidelity_profile != use_fidelity || environment_profile!=use_environment || city_profile!=use_city) reset();
        city_profile=use_city;
        environment_profile=use_environment;
        fidelity_profile = use_fidelity;
        char control[8]={};
        bool explicit_surface=GetEnvironmentVariableA("C3X_RENDERER_SHARED_SCENE_SURFACE",control,sizeof(control))!=0;
        bool requested_surface=explicit_surface && std::strcmp(control,"1")==0;
        if(!explicit_surface){
            char waves[8]={},reflections[8]={};
            GetEnvironmentVariableA("C3X_RENDERER_WAVES",waves,sizeof(waves));
            GetEnvironmentVariableA("C3X_RENDERER_REFLECTION_CONTROL",reflections,sizeof(reflections));
            requested_surface=city_profile && std::strcmp(waves,"0")==0 && std::strcmp(reflections,"1")==0;
        }
        scene_surface_requested=requested_surface;
        // API 18 capture owns fog. The off control exists only in benchmark builds.
        visibility_pass=true;
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
        char visibility_option[8]={};
        if(GetEnvironmentVariableA("C3X_RENDERER_VISIBILITY_PASS",visibility_option,sizeof(visibility_option)))
            visibility_pass=std::strcmp(visibility_option,"0")!=0;
#endif
        if(requested_surface!=shared_scene_surface){reset_targets();clear_resource_backdrops();}
        shared_scene_surface=requested_surface;
        bool three_zoom_memory=c3x_renderer::NavigationOptions::retained(GetEnvironmentVariableA,"C3X_RENDERER_THREE_ZOOM_MEMORY");
        auto viewport_limit=three_zoom_memory?std::max(default_viewport_cache_budget,std::size_t(64u*1024u*1024u)):default_viewport_cache_budget;
        auto backdrop_limit=three_zoom_memory?std::size_t(832u*1024u*1024u):default_resource_backdrop_cache_budget;
        // Budget changes retire optional owners before admitting under the new
        // bound. They never change target resolution, MSAA or retained depth.
        if(viewport_cache_budget!=viewport_limit){viewport_cache.clear();viewport_cache_bytes=0;}
        if(resource_backdrop_cache_budget!=backdrop_limit)clear_resource_backdrops();
        viewport_cache_budget=viewport_limit;resource_backdrop_cache_budget=backdrop_limit;
        clip_dirty_blocks=GetEnvironmentVariableA("C3X_RENDERER_BLOCK_CLIP",control,sizeof(control)) && std::strcmp(control,"1")==0;
        bounded_post=GetEnvironmentVariableA("C3X_RENDERER_BOUNDED_POST",control,sizeof(control)) && std::strcmp(control,"1")==0;
        GetEnvironmentVariableA("C3X_RENDERER_REGION_SIZE",control,sizeof(control));
        scene_region_size=std::strcmp(control,"2240")==0?2240:std::strcmp(control,"256")==0?256:std::strcmp(control,"512")==0?512:128;
        scene_region_height=scene_region_size==2240?256:scene_region_size;
        world_raster_grid=GetEnvironmentVariableA("C3X_RENDERER_WORLD_RASTER_GRID",control,sizeof(control)) && std::strcmp(control,"1")==0;
        world_regions=world_raster_grid && city_profile && scene_region_size==128 &&
            GetEnvironmentVariableA("C3X_RENDERER_WORLD_REGIONS",control,sizeof(control)) && std::strcmp(control,"1")==0;
        world_regions_control=GetEnvironmentVariableA("C3X_RENDERER_WORLD_REGIONS_CONTROL",control,sizeof(control)) && std::strcmp(control,"1")==0;
        local_region_revisions=GetEnvironmentVariableA("C3X_RENDERER_LOCAL_REGION_REVISIONS",control,sizeof(control)) && std::strcmp(control,"1")==0;
        region_diagnostics=GetEnvironmentVariableA("C3X_RENDERER_REGION_DIAGNOSTICS",control,sizeof(control)) && std::strcmp(control,"1")==0;
        region_receiver_shadows=GetEnvironmentVariableA("C3X_RENDERER_REGION_RECEIVER_SHADOWS",control,sizeof(control)) && std::strcmp(control,"1")==0;
        composition_receiver_index=c3x_renderer::NavigationOptions::retained(GetEnvironmentVariableA,"C3X_RENDERER_COMPOSITION_RECEIVER_INDEX");
        tight_natural_bounds=GetEnvironmentVariableA("C3X_RENDERER_TIGHT_NATURAL_BOUNDS",control,sizeof(control)) && std::strcmp(control,"1")==0;
        GetEnvironmentVariableA("C3X_RENDERER_REGION_INPUT_RING",control,sizeof(control));
        region_input_ring=std::strcmp(control,"4")==0?4:2;
        GetEnvironmentVariableA("C3X_RENDERER_REGION_METADATA_MIB",control,sizeof(control));
        std::size_t metadata_limit=std::strcmp(control,"32")==0?32u*1024u*1024u:96u*1024u*1024u;
        if(render_regions.metadata_limit!=metadata_limit)render_regions.clear();
        render_regions.metadata_limit=metadata_limit;
        cull_empty_water=GetEnvironmentVariableA("C3X_RENDERER_WATER_COVERAGE",control,sizeof(control)) && std::strcmp(control,"1")==0;
        world_backdrops=world_raster_grid && c3x_renderer::NavigationOptions::retained(GetEnvironmentVariableA,"C3X_RENDERER_WORLD_BACKDROPS");
        backdrop_reuse_control=GetEnvironmentVariableA("C3X_RENDERER_BACKDROP_REUSE_CONTROL",control,sizeof(control)) && std::strcmp(control,"1")==0;
        world_waves=c3x_renderer::NavigationOptions::retained(GetEnvironmentVariableA,"C3X_RENDERER_WORLD_WAVES");
        wave_reuse_control=GetEnvironmentVariableA("C3X_RENDERER_WAVE_REUSE_CONTROL",control,sizeof(control)) && std::strcmp(control,"1")==0;
        unit_bodies.direct_scene=!(GetEnvironmentVariableA("C3X_RENDERER_UNIT_SCENE_CONTROL",control,sizeof(control)) && std::strcmp(control,"1")==0);
        animation_readback_atlas=GetEnvironmentVariableA("C3X_RENDERER_ANIMATION_READBACK_ATLAS",control,sizeof(control)) && std::strcmp(control,"1")==0;
        reflection.enabled=!(GetEnvironmentVariableA("C3X_RENDERER_REFLECTION_CONTROL",control,sizeof(control)) && std::strcmp(control,"1")==0);
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
        GetEnvironmentVariableA("C3X_RENDERER_DIAGNOSTIC_ROUTES",control,sizeof(control));
        diagnostic_routes=std::strcmp(control,"draw")==0?1u:std::strcmp(control,"all")==0?2u:0u;
        diagnostic_half_pixels=GetEnvironmentVariableA("C3X_RENDERER_DIAGNOSTIC_HALF_PIXELS",control,sizeof(control)) && std::strcmp(control,"1")==0;
        char animation_ablation[40]={};
        GetEnvironmentVariableA("C3X_RENDERER_DIAGNOSTIC_ANIMATION",animation_ablation,sizeof(animation_ablation));
        diagnostic_animation=std::strcmp(animation_ablation,"body-shadow")==0?1u:
            std::strcmp(animation_ablation,"body-shadow-finish")==0?2u:
            std::strcmp(animation_ablation,"body-shadow-finish-import")==0?3u:0u;
#endif
        fidelity_shadow_control=GetEnvironmentVariableA("C3X_RENDERER_FIDELITY_SHADOW_CONTROL",control,sizeof(control)) && std::strcmp(control,"1")==0;
        fidelity_root = mod_root ? mod_root : "";
        bool use_pickup = std::strcmp(requested_profile, "frozen") != 0;
        if (pickup_profile != use_pickup) reset();
        pickup_profile = use_pickup;
        char shader_path[4 * MAX_PATH];
        if (mod_root != nullptr &&
            pack_path(mod_root, city_profile ? "Renderer\\native\\city_fidelity\\hydrology.hlsl" : environment_profile ? "Renderer\\native\\environment_refresh\\hydrology.hlsl" : fidelity_profile ? "Renderer\\native\\source_fidelity\\hydrology.hlsl" : pickup_profile ? "Renderer\\native\\render_core\\terrain_scene.hlsl" :
                      "Renderer\\native\\integrated_terrain.hlsl",
                      shader_path, std::size(shader_path)) &&
            GetFileAttributesA(shader_path) != INVALID_FILE_ATTRIBUTES)
            integrated_shader_path = shader_path;
        std::array<c3x_renderer::TerrainAssetBinding, c3x_renderer::terrain_type_count> bindings;
        c3x_renderer::RendererPackRoots companion_packs;
        std::string diagnostic;
        if (!c3x_renderer::load_terrain_definition_layers(
                mod_root, default_path, scenario_path, custom_path, bindings,
                companion_packs, diagnostic))
            return false;
        std::string default_name = default_path == nullptr ? "" : default_path;
        std::replace(default_name.begin(), default_name.end(), '/', '\\');
        std::size_t default_separator = default_name.find_last_of('\\');
        std::string default_filename = default_separator == std::string::npos
            ? default_name : default_name.substr(default_separator + 1);
        bool production = default_filename == "default.custom_rendering.txt";
        clear_terrain_assets();
        previous_content_revision = content_revision;
        content_revision = 0;
        std::vector<std::uint8_t> shader_source;
        if (!read_file(integrated_shader_path.c_str(), shader_source) && production)
            return false;
        if (!shader_source.empty())
            mix_content_revision(shader_source);
        char production_shader_path[4 * MAX_PATH];
        shader_source.clear();
        bool has_production_shader = mod_root != nullptr &&
            pack_path(mod_root, "Renderer\\native\\terrain_rendering.hlsl",
                      production_shader_path, std::size(production_shader_path)) &&
            read_file(production_shader_path, shader_source);
        if (!has_production_shader && production)
            return false;
        if (has_production_shader)
            mix_content_revision(shader_source);
        for (char const * path : {default_path, scenario_path, custom_path}) {
            std::vector<std::uint8_t> definition;
            if (path != nullptr && path[0] != '\0' && read_file(path, definition))
                mix_content_revision(definition);
        }
        char const * terrain_root = nullptr;
        for (int index = 0; index < c3x_renderer::terrain_type_count; ++index) {
            c3x_renderer::TerrainAssetBinding const & binding = bindings[index];
            if (binding.configured) {
                if (!configure_asset(index, binding.pack_root.c_str(),
                                     binding.logical_asset_id.c_str())) {
                    if (production)
                        return false;
                    continue;
                }
                if (terrain_root == nullptr)
                    terrain_root = binding.pack_root.c_str();
            }
        }
        if (production) {
            if (terrain_root == nullptr)
                return false;
            for (TerrainTexture const & texture : terrain_textures)
                if (!texture.configured)
                    return false;
            configure_integrated_assets(terrain_root, companion_packs);
            if (!terrain_extra_assets_ready || !authored_relief_assets_ready ||
                !dune_assets_ready ||
                !marsh_assets_ready || !volcano_assets_ready ||
                !clutter_assets_ready || !feature_assets_ready || !river_assets_ready ||
                !route_assets_ready || !resource_assets_ready || !city_assets_ready ||
                !mine_assets_ready || !farm_assets_ready)
                return false;
        }
        return true;
    }

    bool ensure_dds_texture(std::vector<std::uint8_t> const & dds,
                            ID3D11ShaderResourceView *& view, bool required, bool full_resolution = false) {
        if (view != nullptr)
            return true;
        if (dds.empty())
            return !required;
        if (device == nullptr || dds.size() < 156)
            return false;
        material_views_valid=false;
        std::uint32_t width_px = read_u32(dds, 16);
        std::uint32_t height_px = read_u32(dds, 12);
        std::uint32_t mip_count = std::max(1u, read_u32(dds, 28));
        DXGI_FORMAT format = static_cast<DXGI_FORMAT>(read_u32(dds, 128));
        if (width_px == 0 || height_px == 0 || width_px > 16384 || height_px > 16384 || mip_count > 15)
            return false;
        std::uint32_t block_bytes = 0;
        std::uint32_t bytes_per_pixel = 0;
        if (format == DXGI_FORMAT_BC1_UNORM || format == DXGI_FORMAT_BC1_UNORM_SRGB ||
            format == DXGI_FORMAT_BC4_UNORM || format == DXGI_FORMAT_BC4_SNORM)
            block_bytes = 8u;
        else if (format == DXGI_FORMAT_BC3_UNORM || format == DXGI_FORMAT_BC3_UNORM_SRGB ||
                 format == DXGI_FORMAT_BC5_UNORM)
            block_bytes = 16u;
        else if (format == DXGI_FORMAT_R16G16B16A16_UNORM ||
                 format == DXGI_FORMAT_R16G16B16A16_FLOAT)
            bytes_per_pixel = 8u;
        else if (format == DXGI_FORMAT_R16G16_UNORM || format == DXGI_FORMAT_R8G8B8A8_UNORM || format == DXGI_FORMAT_R32_FLOAT)
            bytes_per_pixel = 4u;
        else if (format == DXGI_FORMAT_R8_UNORM)
            bytes_per_pixel = 1u;
        if (format != DXGI_FORMAT_BC1_UNORM && format != DXGI_FORMAT_BC1_UNORM_SRGB &&
            format != DXGI_FORMAT_BC3_UNORM && format != DXGI_FORMAT_BC3_UNORM_SRGB &&
            format != DXGI_FORMAT_BC4_UNORM && format != DXGI_FORMAT_BC4_SNORM &&
            format != DXGI_FORMAT_BC5_UNORM &&
            format != DXGI_FORMAT_R16G16B16A16_UNORM &&
            format != DXGI_FORMAT_R16G16B16A16_FLOAT &&
            format != DXGI_FORMAT_R16G16_UNORM && format != DXGI_FORMAT_R8G8B8A8_UNORM && format != DXGI_FORMAT_R32_FLOAT &&
            format != DXGI_FORMAT_R8_UNORM)
            return false;
        std::vector<D3D11_SUBRESOURCE_DATA> subresources(mip_count);
        std::size_t offset = 148;
        std::uint32_t mip_width = width_px, mip_height = height_px;
        for (std::uint32_t mip = 0; mip < mip_count; ++mip) {
            std::uint32_t row_pitch = bytes_per_pixel != 0
                ? mip_width * bytes_per_pixel
                : std::max(1u, (mip_width + 3) / 4) * block_bytes;
            std::uint32_t rows = bytes_per_pixel != 0
                ? mip_height : std::max(1u, (mip_height + 3) / 4);
            std::size_t byte_count = static_cast<std::size_t>(row_pitch) * rows;
            if (offset + byte_count > dds.size())
                return false;
            subresources[mip].pSysMem = dds.data() + offset;
            subresources[mip].SysMemPitch = row_pitch;
            subresources[mip].SysMemSlicePitch = static_cast<UINT>(byte_count);
            offset += byte_count;
            mip_width = std::max(1u, mip_width / 2);
            mip_height = std::max(1u, mip_height / 2);
        }
        // Civ III is a 32-bit process and only displays 128x64 map cells. Keep
        // the authored mip chain, but do not allocate source-detail levels that
        // cannot contribute at the in-game projection scale.
        std::uint32_t first_mip = 0;
        std::uint32_t runtime_width = width_px, runtime_height = height_px;
        while (!full_resolution && first_mip + 1 < mip_count &&
               (runtime_width > 2048 || runtime_height > 2048)) {
            ++first_mip;
            runtime_width = std::max(1u, runtime_width / 2);
            runtime_height = std::max(1u, runtime_height / 2);
        }
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = runtime_width;
        desc.Height = runtime_height;
        desc.MipLevels = mip_count - first_mip;
        desc.ArraySize = 1;
        desc.Format = format;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        ID3D11Texture2D * texture = nullptr;
        HRESULT hr = device->CreateTexture2D(&desc, subresources.data() + first_mip, &texture);
        if (SUCCEEDED(hr))
            hr = device->CreateShaderResourceView(texture, nullptr, &view);
        release(texture);
        return SUCCEEDED(hr);
    }

    bool ensure_pack_texture(int terrain_type) {
        TerrainTexture & asset = terrain_textures[terrain_type];
        if (!asset.configured)
            return false;
        bool layers_ready = true;
        for (std::size_t index = 0; index < asset.relief_layer_dds.size(); ++index)
            layers_ready = layers_ready && ensure_dds_texture(
                asset.relief_layer_dds[index], asset.relief_layer_views[index], false);
        for (std::size_t index = 0; index < asset.water_surface_dds.size(); ++index)
            layers_ready = layers_ready && ensure_dds_texture(
                asset.water_surface_dds[index], asset.water_surface_views[index], false);
        return layers_ready && ensure_dds_texture(asset.dds, asset.view, true) &&
            ensure_dds_texture(asset.material_height_dds, asset.material_height_view, false) &&
            ensure_dds_texture(asset.specular_dds, asset.specular_view, false) &&
            ensure_dds_texture(asset.elevated_dds, asset.elevated_view, false) &&
            ensure_dds_texture(asset.elevated_height_dds, asset.elevated_height_view, false) &&
            ensure_dds_texture(asset.elevated_specular_dds, asset.elevated_specular_view, false);
    }

    bool ensure_terrain_textures() {
        if (pickup_profile) {
            if (!cliff_assets_ready) return false;
            for (std::size_t i=0;i<cliff_dds.size();++i)
                if (!ensure_dds_texture(cliff_dds[i],cliff_views[i],true)) return false;
        }
        if (dune_assets_ready) {
            dune_assets_ready = ensure_dds_texture(dune_surface.dds, dune_surface.view, true) &&
                ensure_dds_texture(dune_surface.material_height_dds,
                                   dune_surface.material_height_view, true) &&
                ensure_dds_texture(dune_surface.specular_dds, dune_surface.specular_view, true) &&
                ensure_dds_texture(dune_decal_base_dds, dune_decal_base_view, true) &&
                ensure_dds_texture(dune_decal_height_dds, dune_decal_height_view, true);
        }
        if (feature_assets_ready) {
            for (std::size_t index = 0; index < feature_bundle.texture_paths.size(); ++index)
                feature_assets_ready = feature_assets_ready && ensure_dds_texture(
                    feature_texture_dds[index], feature_texture_views[index], true);
        }
        if (marsh_assets_ready) {
            marsh_assets_ready = ensure_dds_texture(marsh_decal_base_dds, marsh_decal_base_view, true) &&
                ensure_dds_texture(marsh_decal_height_dds, marsh_decal_height_view, true) &&
                ensure_dds_texture(marsh_decal_specular_dds, marsh_decal_specular_view, true);
        }
        if (volcano_assets_ready) {
            volcano_assets_ready = ensure_dds_texture(volcano_base_dds, volcano_base_view, true) &&
                ensure_dds_texture(volcano_height_dds, volcano_height_view, true) &&
                ensure_dds_texture(volcano_active_base_dds, volcano_active_base_view, true) &&
                ensure_dds_texture(volcano_active_specular_dds,
                                   volcano_active_specular_view, true);
        }
        if (clutter_assets_ready) {
            clutter_assets_ready = ensure_dds_texture(water_clutter_base_dds,
                                                      water_clutter_base_view, true) &&
                ensure_dds_texture(water_clutter_height_dds,
                                   water_clutter_height_view, true) &&
                ensure_dds_texture(grass_clutter_base_dds,
                                   grass_clutter_base_view, true) &&
                ensure_dds_texture(grass_clutter_height_dds,
                                   grass_clutter_height_view, true) &&
                ensure_dds_texture(plains_clutter_base_dds,
                                   plains_clutter_base_view, true) &&
                ensure_dds_texture(plains_clutter_height_dds,
                                   plains_clutter_height_view, true);
        }
        if (terrain_extra_assets_ready) {
            for (std::size_t index = 0; index < terrain_extra_dds.size(); ++index)
                if (!ensure_dds_texture(terrain_extra_dds[index], terrain_extra_views[index], true))
                    return false;
        }
        if (river_assets_ready) {
            for (std::size_t index = 0; index < river_surface_dds.size(); ++index)
                river_assets_ready = river_assets_ready && ensure_dds_texture(
                    river_surface_dds[index], river_surface_views[index], true);
            for (std::size_t index = 0; index < river_rock_texture_dds.size(); ++index)
                river_assets_ready = river_assets_ready && ensure_dds_texture(
                    river_rock_texture_dds[index], river_rock_texture_views[index], true);
        }
        if (route_assets_ready) {
            for (std::size_t index = 0; index < route_texture_views.size(); ++index)
                route_assets_ready = route_assets_ready && ensure_dds_texture(
                    route_texture_dds[index], route_texture_views[index], true);
            for (std::size_t index = 0; index < bridge_texture_views.size(); ++index)
                route_assets_ready = route_assets_ready && ensure_dds_texture(
                    bridge_texture_dds[index], bridge_texture_views[index], true);
        }
        for (auto & animation : resource_animations)
            if (!ensure_dds_texture(animation.dds, animation.view, true)) return false;
        if (resource_assets_ready) {
            for (std::size_t index = 0; index < resource_texture_views.size(); ++index)
                resource_assets_ready = resource_assets_ready && ensure_dds_texture(
                    resource_texture_dds[index], resource_texture_views[index], true);
        }
        if (city_assets_ready) {
            for (std::size_t index = 0; index < city_base_views.size(); ++index) {
                city_assets_ready = city_assets_ready && ensure_dds_texture(
                    city_base_dds[index], city_base_views[index], true) &&
                    ensure_dds_texture(city_emissive_dds[index], city_emissive_views[index], true);
            }
            city_assets_ready = city_assets_ready && ensure_dds_texture(
                wall_texture_dds, wall_texture_view, true);
        }
        if (mine_assets_ready) {
            for (std::size_t index = 0; index < mine_base_views.size(); ++index)
                mine_assets_ready = mine_assets_ready && ensure_dds_texture(
                    mine_base_dds[index], mine_base_views[index], true);
            for (std::size_t index = 0; index < mine_emissive_views.size(); ++index)
                mine_assets_ready = mine_assets_ready && ensure_dds_texture(
                    mine_emissive_dds[index], mine_emissive_views[index], true);
        }
        if(site_assets_ready) for(unsigned i=0;i<site_views.size();++i)
            site_assets_ready=site_assets_ready && ensure_dds_texture(site_dds[i],site_views[i],true);
        if (farm_assets_ready) {
            for (std::size_t index = 0; index < farm_base_views.size(); ++index)
                farm_assets_ready = farm_assets_ready && ensure_dds_texture(
                    farm_base_dds[index], farm_base_views[index], true);
            for (std::size_t index = 0; index < farm_emissive_views.size(); ++index)
                farm_assets_ready = farm_assets_ready && ensure_dds_texture(
                    farm_emissive_dds[index], farm_emissive_views[index], true);
        }
        return true;
    }

    bool ensure_targets(int requested_width, int requested_height) {
        if (requested_width == width && requested_height == height && render_texture != nullptr)
            return true;
        reset_targets();

        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = static_cast<UINT>(requested_width);
        desc.Height = static_cast<UINT>(requested_height);
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET;
        HRESULT hr = device->CreateTexture2D(&desc, nullptr, &render_texture);
        if (SUCCEEDED(hr))
            hr = device->CreateRenderTargetView(render_texture, nullptr, &render_target);

        D3D11_TEXTURE2D_DESC depth_desc = {};
        depth_desc.Width = static_cast<UINT>(requested_width);
        depth_desc.Height = static_cast<UINT>(requested_height);
        depth_desc.MipLevels = 1;
        depth_desc.ArraySize = 1;
        depth_desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        depth_desc.SampleDesc.Count = 1;
        depth_desc.Usage = D3D11_USAGE_DEFAULT;
        depth_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
        if (SUCCEEDED(hr))
            hr = device->CreateTexture2D(&depth_desc, nullptr, &depth_texture);
        if (SUCCEEDED(hr))
            hr = device->CreateDepthStencilView(depth_texture, nullptr, &depth_target);

        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (SUCCEEDED(hr))
            hr = device->CreateTexture2D(&desc, nullptr, &readback_texture);
        if (FAILED(hr)) {
            reset_targets();
            return false;
        }
        width = requested_width;
        height = requested_height;
        pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
        return true;
    }

    static int ground_type(c3x_renderer_tile_v1 const & tile) {
        // Real terrain types 5-10 are relief or vegetation over the base
        // biome.  They are composed separately and must not become a
        // tile-sized ground decal.  Flood plain (4) and water remain direct
        // surface families.
        if (tile.real_terrain_type == 4 || tile.real_terrain_type >= 11)
            return tile.real_terrain_type;
        return tile.terrain_type;
    }

    static int relief_type(c3x_renderer_tile_v1 const & tile) {
        // Volcanoes, hills, and mountains are geometry categories over their
        // underlying ground terrain.
        return tile.real_terrain_type == 5 || tile.real_terrain_type == 6 ||
               tile.real_terrain_type == 10
            ? tile.real_terrain_type : -1;
    }

    static void measure_field_limits(std::vector<std::uint8_t> const & pixels,
                                     float & minimum, float & maximum) {
        if (pixels.empty()) {
            minimum = 0.0f;
            maximum = 1.0f;
            return;
        }
        auto limits = std::minmax_element(pixels.begin(), pixels.end());
        minimum = static_cast<float>(*limits.first) / 255.0f;
        maximum = static_cast<float>(*limits.second) / 255.0f;
    }

    static float sample_byte_field(std::vector<std::uint8_t> const & pixels,
                                   std::uint32_t width, std::uint32_t height,
                                   float u, float v, bool wrap) {
        if (pixels.empty() || width == 0 || height == 0)
            return 0.0f;
        if (wrap) {
            u -= std::floor(u);
            v -= std::floor(v);
        } else {
            u = std::clamp(u, 0.0f, 1.0f);
            v = std::clamp(v, 0.0f, 1.0f);
        }
        float x = u * static_cast<float>(wrap ? width : width - 1);
        float y = v * static_cast<float>(wrap ? height : height - 1);
        std::uint32_t x0 = static_cast<std::uint32_t>(x);
        std::uint32_t y0 = static_cast<std::uint32_t>(y);
        if (wrap) { x0 %= width; y0 %= height; }
        std::uint32_t x1 = wrap ? (x0 + 1) % width : std::min(x0 + 1, width - 1);
        std::uint32_t y1 = wrap ? (y0 + 1) % height : std::min(y0 + 1, height - 1);
        float fx = x - static_cast<float>(x0);
        float fy = y - static_cast<float>(y0);
        auto value = [&pixels, width](std::uint32_t px, std::uint32_t py) {
            return static_cast<float>(pixels[static_cast<std::size_t>(py) * width + px]) / 255.0f;
        };
        float top = value(x0, y0) * (1.0f - fx) + value(x1, y0) * fx;
        float bottom = value(x0, y1) * (1.0f - fx) + value(x1, y1) * fx;
        return top * (1.0f - fy) + bottom * fy;
    }

    static float sample_height_field(TerrainTexture const & asset, float u, float v, bool wrap) {
        return sample_byte_field(asset.height_pixels, asset.height_width,
                                 asset.height_height, u, v, wrap);
    }

    static float sample_normalized_field(std::vector<std::uint8_t> const & pixels,
                                         std::uint32_t width, std::uint32_t height,
                                         float minimum, float maximum,
                                         float u, float v) {
        return c3x_renderer::fidelity::sample_normalized_field(pixels,width,height,minimum,maximum,u,v);
    }

    static float smooth_edge(float distance) {
        float amount = std::clamp(distance / 0.42f, 0.0f, 1.0f);
        return amount * amount * (3.0f - 2.0f * amount);
    }

    static float smoothstep01(float value) {
        value = std::clamp(value, 0.0f, 1.0f);
        return value * value * (3.0f - 2.0f * value);
    }

    static std::array<float, 3> relief_height(
            TerrainTexture const & asset, float u, float v,
            float world_u, float world_v, int base_ground,
            c3x_renderer_u32 seed, bool const connected_edges[4]) {
        if (asset.relief_profile == 1)
            return {(sample_height_field(asset, world_u, world_v, true) - 0.5f) *
                asset.height_scale_px, 0.0f, 0.0f};
        if (asset.relief_profile == 2) {
            // Match the accepted Lab hill body: a source-space nine-tap
            // low-pass extracts the authored macro landform, and a calibrated
            // remap removes the source field's positive floor. Neighbor
            // topology weights outside this function own its continuous skirt.
            (void)u;
            (void)v;
            (void)base_ground;
            (void)seed;
            (void)connected_edges;
            constexpr float radius = 0.018f;
            float source_u = 0.11f + world_u * 0.035f;
            float source_v = 0.17f + world_v * 0.035f;
            auto sample = [&asset](float x, float y) {
                return sample_normalized_field(asset.height_pixels,
                    asset.height_width, asset.height_height,
                    asset.height_minimum, asset.height_maximum, x, y);
            };
            float center = sample(source_u, source_v) * 4.0f;
            float cardinal = sample(source_u - radius, source_v) +
                sample(source_u + radius, source_v) +
                sample(source_u, source_v - radius) +
                sample(source_u, source_v + radius);
            float diagonal = sample(source_u - radius, source_v - radius) +
                sample(source_u + radius, source_v - radius) +
                sample(source_u - radius, source_v + radius) +
                sample(source_u + radius, source_v + radius);
            float authored_macro = (center + cardinal * 2.0f + diagonal) / 16.0f;
            float shape = smoothstep01((authored_macro - 0.22f) / 0.38f);
            return {shape * asset.height_scale_px, 0.0f, 0.0f};
        }
        if (asset.relief_profile == 4) {
            // Mountain bodies are composed across neighboring cells below,
            // using the Lab's chain sampler verbatim. A tile-local fallback
            // here would reintroduce the divergent diamond-shaped mountain.
            return {0.0f, 0.0f, 0.0f};
        }
        if (asset.relief_profile == 3) {
            float sample_u = u, sample_v = v;
            switch (seed & 3u) {
            case 1: sample_u = v; sample_v = 1.0f - u; break;
            case 2: sample_u = 1.0f - u; sample_v = 1.0f - v; break;
            case 3: sample_u = 1.0f - v; sample_v = u; break;
            default: break;
            }
            return {sample_height_field(asset, sample_u, sample_v, false) *
                asset.height_scale_px, 0.0f, 0.0f};
        }
        if (asset.relief_profile == 5) {
            // Use the normalized ordinary-volcano height and blend fields
            // exactly as authored. Only deterministic rigid orientation,
            // bounded aspect, Civ III footprint fit, and vertical calibration
            // are applied. Adjacent mountain/volcano cells enlarge the same
            // source footprint so shoulders overlap without connector geometry.
            bool connected = connected_edges[0] || connected_edges[1] ||
                connected_edges[2] || connected_edges[3];
            float footprint = connected ? 0.60f : 1.0f;
            float aspect = (seed & 4u) != 0 ? 0.88f : 1.12f;
            float source_u = 0.5f + (u - 0.5f) * footprint * aspect;
            float source_v = 0.5f + (v - 0.5f) * footprint / aspect;
            switch (seed & 3u) {
            case 1: std::swap(source_u, source_v); break;
            case 2: source_u = 1.0f - source_u; break;
            case 3: std::swap(source_u, source_v); source_u = 1.0f - source_u; break;
            default: break;
            }
            if (source_u < 0.0f || source_u > 1.0f ||
                source_v < 0.0f || source_v > 1.0f)
                return {0.0f, 0.0f, 0.0f};
            float height = sample_height_field(asset, source_u, source_v, false);
            float blend = sample_byte_field(asset.blend_pixels,
                asset.height_width, asset.height_height, source_u, source_v, false);
            float blend_weight = std::clamp(blend / 0.34f, 0.0f, 1.0f);
            blend_weight = blend_weight * blend_weight * (3.0f - 2.0f * blend_weight);
            float vertical = ((seed >> 3) & 1u) != 0 ? 104.0f : 88.0f;
            return {height * blend_weight * vertical, height, blend};
        }
        return {0.0f, 0.0f, 0.0f};
    }

    bool load_unit_animations(std::string const& root) {
        std::vector<std::uint8_t> data;
        if(!read_file((root+"\\bindings.json").c_str(),data))return false;
        float count=0;
        if(!json_number_after(data,"unit_count",0,count) || count<1 || count>128 || count!=int(count))return false;
        std::unordered_map<std::string,unsigned> mesh_ids,texture_ids;
        for(int i=0;i<int(count);++i) {
            auto location=json_member_position(data,("unit"+std::to_string(i)).c_str());
            float complete=0,keys=0;
            if(!json_number_after(data,"complete",location,complete))return false;
            if(complete!=1){trace.write("unit-bind",("unit="+std::to_string(i)+" unresolved material contract; native body").c_str(),true);continue;}
            c3x_renderer::UnitBodyRenderer::Unit unit;
            float sample_scale=1;
            if(json_number_after(data,"sample_scale",location,sample_scale) &&
               sample_scale!=1 && sample_scale!=2 && sample_scale!=4)return false;
            unit.sample_scale=int(sample_scale);
            float minimum_canvas=0;
            if(json_number_after(data,"minimum_canvas",location,minimum_canvas) &&
               (minimum_canvas<0 || minimum_canvas>512 || minimum_canvas!=std::floor(minimum_canvas)))return false;
            unit.minimum_canvas=int(minimum_canvas);
            if(!json_number_after(data,"key_count",location,keys) || keys<1 || keys>16 || keys!=int(keys) ||
               !json_number_after(data,"scale",location,unit.scale) || unit.scale<=0 || unit.scale>10 ||
               !json_number_after(data,"yaw_offset",location,unit.yaw_offset) ||
               !json_number_after(data,"offset_z",location,unit.offset_z))return false;
            for(int key=0;key<int(keys);++key) {
                std::string value;
                if(!json_string_after(data,("key"+std::to_string(key)).c_str(),location,value) || value.size()>63)return false;
                unit.keys.push_back(value);
            }
            for(char const* name:{"idle","move","attack","death","fortify","fidget","victory","capture","defend","fortress","build","road","mine","irrigate","jungle","forest","plant"}) {
                auto action_location=json_member_position(data,name,location);
                if(action_location==std::string::npos)continue;
                float parts=0,loop=0;
                if(!json_number_after(data,"part_count",action_location,parts) || parts<1 || parts>32 || parts!=int(parts) ||
                   !json_number_after(data,"loop",action_location,loop) || (loop!=0 && loop!=1))return false;
                c3x_renderer::UnitBodyRenderer::Action action;action.name=name;action.loop=loop!=0;
                float ambient=0;
                if(json_number_after(data,"ambient",action_location,ambient)) {
                    if((ambient!=0 && ambient!=1) || (ambient==1 && !action.loop))return false;
                    action.ambient=ambient==1;
                }
                float ambient_frames=0;
                if(action.ambient &&
                   (!json_number_after(data,"duration",action_location,action.duration) || action.duration<=0 || action.duration>3600 ||
                    !json_number_after(data,"frames",action_location,ambient_frames) || ambient_frames<2 || ambient_frames>4096 || ambient_frames!=int(ambient_frames)))return false;
                if(action.ambient)action.frames=unsigned(ambient_frames);
                float exit_clip=0;
                if(json_number_after(data,"allow_exit_clip",action_location,exit_clip) &&
                   (exit_clip!=0 && (exit_clip!=1 || action.name!="death")))return false;
                action.allow_exit_clip=exit_clip==1;
                for(int part_index=0;part_index<int(parts);++part_index) {
                    auto record=json_member_position(data,("part"+std::to_string(part_index)).c_str(),action_location);
                    c3x_renderer::UnitBodyRenderer::Part part;
                    float address=0;
                    if(json_number_after(data,"address_mode",record,address) &&
                       (address<0 || address>3 || address!=int(address)))return false;
                    part.address=unsigned(address);
                    std::string mesh,texture;char path[4*MAX_PATH];
                    if(!json_string_after(data,"mesh",record,mesh) || !json_string_after(data,"texture",record,texture) ||
                       !json_number_after(data,"tint_r",record,part.tint[0]) || !json_number_after(data,"tint_g",record,part.tint[1]) ||
                       !json_number_after(data,"tint_b",record,part.tint[2]) || !json_number_after(data,"owner_mask",record,part.mask) ||
                       !json_number_after(data,"owner_strength",record,part.strength) || !json_number_after(data,"cutout",record,part.cutout) ||
                       part.mask<0 || part.mask>2 || part.strength<0 || part.strength>1 || part.cutout<0 || part.cutout>1)return false;
                    if(mesh_ids.find(mesh)==mesh_ids.end()) {
                        c3x_renderer::UnitBodyRenderer::Mesh bound;
                        if(!pack_path(root.c_str(),mesh.c_str(),path,std::size(path)))return false;
                        bound.path=path;mesh_ids[mesh]=unsigned(unit_bodies.meshes.size());unit_bodies.meshes.push_back(std::move(bound));
                    }
                    if(texture_ids.find(texture)==texture_ids.end()) {
                        c3x_renderer::UnitBodyRenderer::Texture bound;
                        if(!pack_path(root.c_str(),texture.c_str(),path,std::size(path)))return false;
                        bound.path=path;texture_ids[texture]=unsigned(unit_bodies.textures.size());unit_bodies.textures.push_back(std::move(bound));
                    }
                    part.mesh=mesh_ids.at(mesh);part.texture=texture_ids.at(texture);
                    char const* material_fields[]={"ao_texture","gloss_texture","emissive_texture","normal_texture"};
                    for(unsigned channel=0;channel<4;++channel) {
                        std::string relative;
                        if(!json_string_after(data,material_fields[channel],record,relative))continue;
                        if(texture_ids.find(relative)==texture_ids.end()) {
                            c3x_renderer::UnitBodyRenderer::Texture bound;
                            if(!pack_path(root.c_str(),relative.c_str(),path,std::size(path)))return false;
                            bound.path=path;texture_ids[relative]=unsigned(unit_bodies.textures.size());
                            unit_bodies.textures.push_back(std::move(bound));
                        }
                        part.material_textures[channel]=texture_ids.at(relative);
                    }
                    json_number_after(data,"material_model",record,part.material_model);
                    action.parts.push_back(part);
                }
                unit.actions.push_back(std::move(action));
            }
            trace.write("unit-bind",(unit.keys.front()+" animation catalog ready; payloads load on demand").c_str(),true);
            unit_bodies.units.push_back(std::move(unit));
        }
        return !unit_bodies.units.empty();
    }

    bool prepare_unit_action(c3x_renderer::UnitBodyRenderer::Action const& action) {
        // All parts of the current action are pinned together. CPU palettes,
        // index buffers and compressed texture copies share a fixed residency
        // budget; sprite-cache hits never enter this path or touch disk.
        auto & bodies=unit_bodies;
        auto used=++bodies.payload_serial;
        auto reserve=[&](std::size_t bytes) {
            constexpr std::size_t budget=96u*1024u*1024u;
            if(bytes>budget)return false;
            while(bodies.resident_bytes>budget-bytes) {
                std::uint64_t oldest=UINT64_MAX;int mesh_id=-1,texture_id=-1;
                for(unsigned i=0;i<bodies.meshes.size();++i) {
                    auto const& m=bodies.meshes[i];
                    bool pinned=std::any_of(action.parts.begin(),action.parts.end(),[&](auto const& p){return p.mesh==i;});
                    if(m.bytes && !pinned && m.animation.use_count()<=1 && m.used<oldest){oldest=m.used;mesh_id=int(i);texture_id=-1;}
                }
                for(unsigned i=0;i<bodies.textures.size();++i) {
                    auto const& t=bodies.textures[i];
                    bool pinned=std::any_of(action.parts.begin(),action.parts.end(),[&](auto const& p){return p.texture==i || std::find(std::begin(p.material_textures),std::end(p.material_textures),i)!=std::end(p.material_textures);});
                    if(t.bytes && !pinned && t.used<oldest){oldest=t.used;texture_id=int(i);mesh_id=-1;}
                }
                if(mesh_id>=0) {
                    auto & m=bodies.meshes[mesh_id];bodies.resident_bytes-=m.bytes;m.bytes=0;
                    bodies.release(m.indices);m.animation={};
                } else if(texture_id>=0) {
                    auto & t=bodies.textures[texture_id];bodies.resident_bytes-=t.bytes;t.bytes=0;
                    bodies.release(t.view);std::vector<std::uint8_t>().swap(t.dds);
                } else {
                    // Immutable CPU jobs pin animation assets in the existing
                    // 96 MiB owner. Revoke optional leases before denying a
                    // demanded payload; no unaccounted shadow asset cache.
                    bool leased=std::any_of(bodies.meshes.begin(),bodies.meshes.end(),[](auto const& m){return m.animation.use_count()>1;});
                    if(!leased)return false;
                    bodies.release_pose_leases();
                }
            }
            return true;
        };
        unsigned loads=0;
        for(auto const& part:action.parts) {
            if(part.mesh>=bodies.meshes.size() || part.texture>=bodies.textures.size())return false;
            auto & mesh=bodies.meshes[part.mesh];
            if(mesh.failed)return false;
            if(!mesh.bytes) {
                std::vector<std::uint8_t> payload;c3x_renderer::AnimationMesh decoded;
                if(!read_file(mesh.path.c_str(),payload) || !c3x_renderer::decode_animation_mesh(payload,decoded)) {
                    mesh.failed=true;trace.write("unit-payload","invalid mesh; only this kit falls back",true);return false;
                }
                std::size_t bytes=decoded.vertices.capacity()*sizeof(c3x_renderer::AnimationVertex)+
                    decoded.indices.capacity()*sizeof(std::uint32_t)*2+decoded.palettes.capacity()*sizeof(float);
                if(!reserve(bytes))return false;
                mesh.animation=std::make_shared<c3x_renderer::AnimationMesh const>(std::move(decoded));mesh.bytes=bytes;bodies.resident_bytes+=bytes;++loads;
            }
            mesh.used=used;
            unsigned material_ids[]={part.texture,part.material_textures[0],part.material_textures[1],part.material_textures[2],part.material_textures[3]};
            for(unsigned channel=0;channel<5;++channel) {
            unsigned id=material_ids[channel];if(id==UINT32_MAX)continue;
            if(id>=bodies.textures.size())return false;
            auto & texture=bodies.textures[id];if(texture.failed)return false;
            if(!texture.bytes) {
                std::vector<std::uint8_t> dds;
                if(!read_file(texture.path.c_str(),dds) || dds.size()<156 ||
                   std::memcmp(dds.data(),"DDS ",4)!=0 || std::memcmp(dds.data()+84,"DX10",4)!=0 ||
                   (channel==0 && read_u32(dds,128)!=DXGI_FORMAT_BC3_UNORM_SRGB && read_u32(dds,128)!=DXGI_FORMAT_BC1_UNORM_SRGB)) {
                    texture.failed=true;trace.write("unit-payload","missing texture; only this kit falls back",true);return false;
                }
                // Budget the retained DDS plus its GPU compressed copy.
                if(!reserve(dds.size()*2))return false;
                if(!ensure_dds_texture(dds,texture.view,true,true)) {
                    texture.failed=true;trace.write("unit-payload","invalid texture; only this kit falls back",true);return false;
                }
                char detail[160];std::snprintf(detail,sizeof(detail),
                    "channel=%u width=%u height=%u mips=%u first_mip=0 anisotropy=16 mip_bias=0 address=%u",
                    channel,read_u32(dds,16),read_u32(dds,12),std::max(1u,read_u32(dds,28)),part.address);
                trace.write("unit-texture",detail,true);
                texture.bytes=dds.size()*2;texture.dds=std::move(dds);bodies.resident_bytes+=texture.bytes;++loads;
            } else if(!texture.view && !ensure_dds_texture(texture.dds,texture.view,true,true))return false;
            texture.used=used;
            }
        }
        if(loads) {
            char message[160];std::snprintf(message,sizeof(message),"action=%s loaded=%u resident_bytes=%zu budget_bytes=%u",
                action.name.c_str(),loads,bodies.resident_bytes,96u*1024u*1024u);
            trace.write("unit-payload",message,true);
        }
        return true;
    }

    int resource_animation_for(c3x_renderer_tile_v1 const & tile) const {
        if (tile.resource_id < 0 || resource_animations.empty()) return -1;
        std::string name(tile.resource_name, std::find(std::begin(tile.resource_name),std::end(tile.resource_name),'\0'));
        std::transform(name.begin(),name.end(),name.begin(),[](unsigned char c){return char(std::tolower(c));});
        for (unsigned i=0;i<resource_animations.size();++i)
            if (name.find(resource_animations[i].name)!=std::string::npos) return int(i);
        return -1;
    }

    bool frame_has_resource_animation(c3x_renderer_frame_v1 const & frame) const {
        if(wave_ready)for(unsigned i=0;i<frame.tile_count;++i)
            if((frame.tiles[i].tile_flags&C3X_RENDERER_TILE_RENDER) && frame.tiles[i].terrain_type>=11)return true;
        if (resource_animations.empty()) return false;
        for (unsigned i=0;i<frame.tile_count;++i)
            if ((frame.tiles[i].tile_flags&C3X_RENDERER_TILE_RENDER) && resource_animation_for(frame.tiles[i])>=0)
                return true;
        return false;
    }

    bool can_prepare_ambient() const {
        return city_profile && shared_scene_surface && moving_resources && !visible_wave_animations;
    }

    static c3x_renderer_i64 resource_clock(c3x_renderer_frame_v1 const & frame) {
        return frame.presentation_time_ticks/std::max<c3x_renderer_i64>(1,frame.presentation_frequency/15);
    }

    bool make_wave_cell_room(std::size_t bytes) {
        constexpr std::size_t budget=32u*1024u*1024u,entries=16384;
        if(bytes>budget)return false;
        while(retained_wave_cells.size()>=entries || wave_geometry_bytes>budget-bytes) {
            auto oldest=retained_wave_cells.end();
            for(auto it=retained_wave_cells.begin();it!=retained_wave_cells.end();++it)
                if(it->second.used!=retained_wave_epoch &&
                   (oldest==retained_wave_cells.end() || it->second.used<oldest->second.used))oldest=it;
            if(oldest==retained_wave_cells.end())return false;
            wave_geometry_bytes-=oldest->second.bytes;retained_wave_cells.erase(oldest);
        }
        return true;
    }

    bool prepare_retained_wave_chunks(c3x_renderer_frame_v1 const& frame) {
        using namespace c3x_renderer::render_core;
        // Coast geometry is independent of light and animation time. Keep the
        // full topology revision until local coast dependency proofs exist.
        auto scope=wave_geometry_scope(frame,content_revision,device_generation);
        if(scope!=retained_wave_scope || wave_reuse_control)reset_waves();
        retained_wave_scope=scope;
        for(auto& chunk:wave_chunks){release(chunk.buffer);release(chunk.indices);}wave_chunks.clear();
        wave_signature=0;
        if(!wave_ready || !frame.tile_count)return true;
        if(++retained_wave_epoch==0){reset_waves();retained_wave_scope=scope;retained_wave_epoch=1;}
        auto cells=captured_wave_cells(frame,C3X_RENDERER_TILE_RENDER);
        auto anchor=frame.tiles;
        for(unsigned i=0;i<frame.tile_count;++i)if(frame.tiles[i].tile_flags&C3X_RENDERER_TILE_RENDER){anchor=frame.tiles+i;break;}
        // Pin the complete requested set before any admission can evict cells.
        for(auto const& cell:cells){auto found=retained_wave_cells.find(cell.first);
            if(found!=retained_wave_cells.end())found->second.used=retained_wave_epoch;}
        float hw=frame.tile_width*.5f,hh=frame.tile_height*.5f;
        int dx=int(geometry_viewport_settings.translation[0]),dy=int(geometry_viewport_settings.translation[1]);
        for(auto const& entry:cells) {
            int c=entry.first.first,r=entry.first.second;
            auto found=retained_wave_cells.find(entry.first);
            if(found==retained_wave_cells.end()) {
                RetainedWaveCell cell;cell.used=retained_wave_epoch;
                auto identity=world_coast.world().index(c,r);
                if(identity!=std::size_t(-1)) {
                    auto hash=c3x_renderer::stable_hash(unsigned(identity)*193u+71u);
                    float seed=c3x_renderer::stable_random(hash),seed2=c3x_renderer::stable_random(hash+23u);
                    auto ribbon=coastal_wave_ribbon(world_coast,c,r,.30f+.70f*seed);
                    cell.bytes=ribbon.size()*(sizeof(Vertex)+sizeof(unsigned));
                    if(!make_wave_cell_room(cell.bytes)){trace.write("coastal-wave-budget","active cells exceed 32 MiB/16384 entries; no truncation",true);return false;}
                    if(!ribbon.empty()) {
                        std::vector<Vertex> vertices;vertices.reserve(ribbon.size());
                        auto& chunk=cell.chunk;chunk.bounds={LONG_MAX,LONG_MAX,LONG_MIN,LONG_MIN};
                        for(unsigned axis=0;axis<3;++axis){chunk.world_bounds.low[axis]=1e9f;chunk.world_bounds.high[axis]=-1e9f;}
                        for(auto const& point:ribbon) {
                            float u=float(point.position.x)-float(c),v=1-(float(point.position.y)-float(r));
                            Vertex out={};out.x=hw+(u-v)*hw;out.y=(u+v)*hh;out.z=out.y+.08f;
                            out.u=point.distance;out.v=point.along;out.panel=1;out.normal_z=1;
                            out.shadow_visibility=1;out.ambient_visibility=point.coverage;
                            out.base_terrain=seed;out.real_terrain=seed2;out.surface_kind=7;
                            out.world_x=float(point.position.x);out.world_y=float(point.position.y);out.world_z=2.5f/112;out.world_valid=1;
                            out.macro_u=out.world_x*.5f;out.macro_v=out.world_y*.5f;vertices.push_back(out);
                            chunk.bounds.left=std::min(chunk.bounds.left,LONG(std::floor(out.x))-2);chunk.bounds.right=std::max(chunk.bounds.right,LONG(std::ceil(out.x))+2);
                            chunk.bounds.top=std::min(chunk.bounds.top,LONG(std::floor(out.y))-2);chunk.bounds.bottom=std::max(chunk.bounds.bottom,LONG(std::ceil(out.y))+2);
                            float world[]={out.world_x,out.world_y,out.world_z};for(unsigned axis=0;axis<3;++axis){
                                chunk.world_bounds.low[axis]=std::min(chunk.world_bounds.low[axis],world[axis]);chunk.world_bounds.high[axis]=std::max(chunk.world_bounds.high[axis],world[axis]);}
                        }
                        std::vector<unsigned> indices(vertices.size());for(unsigned i=0;i<indices.size();++i)indices[i]=i;
                        D3D11_BUFFER_DESC desc={};desc.ByteWidth=unsigned(vertices.size()*sizeof(Vertex));desc.Usage=D3D11_USAGE_IMMUTABLE;desc.BindFlags=D3D11_BIND_VERTEX_BUFFER;
                        D3D11_SUBRESOURCE_DATA data={};data.pSysMem=vertices.data();
                        if(FAILED(device->CreateBuffer(&desc,&data,&chunk.buffer)))return false;
                        desc.ByteWidth=unsigned(indices.size()*sizeof(unsigned));desc.BindFlags=D3D11_BIND_INDEX_BUFFER;data.pSysMem=indices.data();
                        if(FAILED(device->CreateBuffer(&desc,&data,&chunk.indices)))return false;
                        chunk.vertex_stride=sizeof(Vertex);chunk.index_count=unsigned(indices.size());chunk.version=scope;
                    }
                } else if(!make_wave_cell_room(0))return false;
                auto bytes=cell.bytes;
                found=retained_wave_cells.emplace(entry.first,std::move(cell)).first;
                wave_geometry_bytes+=bytes;wave_upload_bytes+=bytes;++wave_cells_built;
            } else ++wave_cells_reused;
            auto const& owner=found->second.chunk;if(!owner.buffer)continue;
            unsigned visibility=visibility_pass?visibility_coverage.state(c+r,c-r):2;if(!visibility)continue;
            auto chunk=owner;chunk.visual_time=visibility==2?-1.f:0.f;
            // Cell-local vertices stay immutable. Visible occurrences provide
            // their authoritative screen transform through the existing buffer.
            chunk.translation_x=anchor->anchor_x+(c+r-anchor->tile_x)*int(hw)-dx;
            chunk.translation_y=anchor->anchor_y+(c-r-anchor->tile_y)*int(hh)-dy;
            if(chunk.bounds.right+chunk.translation_x+dx<=0 || chunk.bounds.left+chunk.translation_x+dx>=width ||
               chunk.bounds.bottom+chunk.translation_y+dy<=0 || chunk.bounds.top+chunk.translation_y+dy>=height)continue;
            wave_chunks.push_back(chunk);chunk.buffer->AddRef();chunk.indices->AddRef();
        }
        wave_signature=cached_signature.complete;
        return true;
    }

    bool prepare_wave_chunks(c3x_renderer_frame_v1 const& frame) {
        wave_upload_bytes=0;
        wave_cells_built=wave_cells_reused=0;
        if(!wave_reuse_control && wave_signature==cached_signature.complete)return true;
        if(world_waves)return prepare_retained_wave_chunks(frame);
        reset_waves();wave_signature=cached_signature.complete;
        if(!wave_ready || !frame.tile_count)return true;
        using namespace c3x_renderer::render_core;
        auto anchor=frame.tiles;
        for(unsigned i=0;i<frame.tile_count;++i)if(frame.tiles[i].tile_flags&C3X_RENDERER_TILE_RENDER){anchor=frame.tiles+i;break;}
        auto const& first=*anchor;
        float hw=frame.tile_width*.5f,hh=frame.tile_height*.5f;
        float cu=(first.tile_x+first.tile_y)*.5f,rv=(first.tile_x-first.tile_y)*.5f;
        float dx=geometry_viewport_settings.translation[0],dy=geometry_viewport_settings.translation[1];
        std::map<std::pair<int,int>,bool> cells;
        for(unsigned i=0;i<frame.tile_count;++i){auto const&t=frame.tiles[i];
            if(!(t.tile_flags&C3X_RENDERER_TILE_RENDER))continue;
            int c=(t.tile_x+t.tile_y)/2,r=(t.tile_x-t.tile_y)/2;
            for(int y=r-1;y<=r+1;++y)for(int x=c-1;x<=c+1;++x)cells[{x,y}]=true;
        }
        std::size_t bytes=0;
        for(auto const& entry:cells){
            int c=entry.first.first,r=entry.first.second;
            unsigned visibility=visibility_pass?visibility_coverage.state(c+r,c-r):2;if(!visibility)continue;
            auto identity=world_coast.world().index(c,r);if(identity==std::size_t(-1))continue;
            auto hash=c3x_renderer::stable_hash(unsigned(identity)*193u+71u);
            float seed=c3x_renderer::stable_random(hash),seed2=c3x_renderer::stable_random(hash+23u);
            auto ribbon=coastal_wave_ribbon(world_coast,c,r,.30f+.70f*seed);if(ribbon.empty())continue;
            std::vector<Vertex> vertices;vertices.reserve(ribbon.size());
            CachedVertexChunk chunk;chunk.visual_time=visibility==2?-1.f:0.f;chunk.bounds={LONG_MAX,LONG_MAX,LONG_MIN,LONG_MIN};
            for(unsigned a=0;a<3;++a){chunk.world_bounds.low[a]=1e9f;chunk.world_bounds.high[a]=-1e9f;}
            for(auto const& point:ribbon){
                float u=float(point.position.x)-cu,v=1-(float(point.position.y)-rv);
                Vertex out={};out.x=first.anchor_x+hw+(u-v)*hw-dx;
                out.y=first.anchor_y+(u+v)*hh-dy;out.z=out.y+.08f;
                out.u=point.distance;out.v=point.along;out.panel=1;out.normal_z=1;
                out.shadow_visibility=1;out.ambient_visibility=point.coverage;
                out.base_terrain=seed;out.real_terrain=seed2;out.surface_kind=7;
                out.world_x=float(point.position.x);out.world_y=float(point.position.y);out.world_z=2.5f/112;out.world_valid=1;
                out.macro_u=out.world_x*.5f;out.macro_v=out.world_y*.5f;
                vertices.push_back(out);
                chunk.bounds.left=std::min(chunk.bounds.left,LONG(std::floor(out.x))-2);chunk.bounds.right=std::max(chunk.bounds.right,LONG(std::ceil(out.x))+2);
                chunk.bounds.top=std::min(chunk.bounds.top,LONG(std::floor(out.y))-2);chunk.bounds.bottom=std::max(chunk.bounds.bottom,LONG(std::ceil(out.y))+2);
                float w[]={out.world_x,out.world_y,out.world_z};for(unsigned a=0;a<3;++a){chunk.world_bounds.low[a]=std::min(chunk.world_bounds.low[a],w[a]);chunk.world_bounds.high[a]=std::max(chunk.world_bounds.high[a],w[a]);}
            }
            if(chunk.bounds.right+dx<=0 || chunk.bounds.left+dx>=width || chunk.bounds.bottom+dy<=0 || chunk.bounds.top+dy>=height)continue;
            auto size=unsigned(vertices.size()*sizeof(Vertex));
            if(bytes+size+vertices.size()*4>16u*1024u*1024u){trace.write("coastal-wave-budget","16 MiB active-view cap; no truncation",true);reset_waves();return false;}
            D3D11_BUFFER_DESC desc={};desc.ByteWidth=size;desc.Usage=D3D11_USAGE_IMMUTABLE;desc.BindFlags=D3D11_BIND_VERTEX_BUFFER;
            D3D11_SUBRESOURCE_DATA data={};data.pSysMem=vertices.data();
            if(FAILED(device->CreateBuffer(&desc,&data,&chunk.buffer))){reset_waves();return false;}
            std::vector<unsigned> indices(vertices.size());for(unsigned i=0;i<indices.size();++i)indices[i]=i;
            desc.ByteWidth=unsigned(indices.size()*4);desc.BindFlags=D3D11_BIND_INDEX_BUFFER;data.pSysMem=indices.data();
            if(FAILED(device->CreateBuffer(&desc,&data,&chunk.indices))){release(chunk.buffer);reset_waves();return false;}
            chunk.vertex_stride=sizeof(Vertex);chunk.index_count=unsigned(indices.size());chunk.version=cached_signature.complete;
            wave_chunks.push_back(chunk);bytes+=size+indices.size()*4;
            wave_upload_bytes=wave_geometry_bytes=bytes;
        }
        return true;
    }

    bool compose_resource_animations(c3x_renderer_frame_v1 const & frame) {
        resource_composite_ticks=0;
        if(visibility_pass && !visibility_coverage.capture(frame))return false;
        if (!shared_scene_surface && !frame_has_resource_animation(frame)) {
            moving_resources=visible_resource_animations=visible_wave_animations=0; resource_pixel_signature=0; return true;
        }
        auto clock=resource_clock(frame);
        if (posed_count() && resource_pixel_signature==cached_signature.complete &&
            resource_pixel_clock==clock) return true;
        LARGE_INTEGER started={},finished={};QueryPerformanceCounter(&started);
        visible_resource_animations=moving_resources=0;
        std::array<std::vector<CachedVertexChunk>,geometry_layer_count> buffers; // posed bodies only
        std::vector<D3D11_RECT> rectangles;
        using c3x_renderer::render_core::RasterRegionAxis;
        using c3x_renderer::render_core::raster_anchor_phase;
        bool anchored=world_backdrops && frame.tile_count!=0;
        int anchor_x=anchored?frame.tiles[0].anchor_x:0,anchor_y=anchored?frame.tiles[0].anchor_y:0;
        RasterRegionAxis backdrop_grid_x(width,anchored?raster_anchor_phase(anchor_x,frame.tiles[0].tile_x,frame.tile_width,128):0);
        RasterRegionAxis backdrop_grid_y(height,anchored?raster_anchor_phase(anchor_y,frame.tiles[0].tile_y,frame.tile_height,128):0);
        std::vector<unsigned char> dirty_blocks(std::size_t(backdrop_grid_x.count)*backdrop_grid_y.count,0);
        auto dirty=[&](D3D11_RECT const& visible) {
            if(visible.left>=visible.right || visible.top>=visible.bottom)return;
            for(int y=backdrop_grid_y.at(visible.top);y<=backdrop_grid_y.at(visible.bottom-1);++y)
                for(int x=backdrop_grid_x.at(visible.left);x<=backdrop_grid_x.at(visible.right-1);++x)
                    dirty_blocks[std::size_t(y)*backdrop_grid_x.count+x]=1;
        };
        std::vector<c3x_renderer::FeatureSourceVertex> posed;
        std::vector<std::array<float,12>> vertices;
        std::vector<Vertex> shadow_vertices;
        std::size_t uploaded=0,pool_bytes=0;
        for (auto const & pool:resource_buffers)
            pool_bytes+=pool.capacity+pool.shadow_capacity;
        if(city_profile)for(auto const& asset:resource_animations){
            pool_bytes+=sizeof(asset.source_bounds);
            if(asset.vertices)pool_bytes+=asset.mesh.vertices.size()*sizeof(c3x_renderer::AnimationVertex);
            if(asset.indices)pool_bytes+=asset.mesh.indices.size()*sizeof(unsigned);
        }
        float half_w=frame.tile_width*.5f,half_h=frame.tile_height*.5f;
        float projection=frame.tile_width/224.f,relief=projection*.82f;
        int dx=int(geometry_viewport_settings.translation[0]),dy=int(geometry_viewport_settings.translation[1]);
        auto ticks=clock*std::max<c3x_renderer_i64>(1,frame.presentation_frequency/15);
        for (auto const & anchor:resource_anchors) {
            if (anchor.asset>=resource_animations.size()) return false;
            auto & animation=resource_animations[anchor.asset];
            unsigned visibility=visibility_pass?visibility_coverage.state(anchor.tile_x,anchor.tile_y):2;
            if(!visibility)continue; // Never-explored objects contribute no body or shadow.
            bool advances=visibility==2;
            double time=c3x_renderer::ambient_animation_time(advances?ticks:0,frame.presentation_frequency,
                animation.mesh.duration,anchor.seed);
            if(city_profile) {
                c3x_renderer::AnimationPose pose;
                c3x_renderer::render_core::ResourceSourceBounds::Box posed_bounds;
                if(!c3x_renderer::sample_animation_pose(animation.mesh,time,true,pose) ||
                   !animation.source_bounds.posed(pose,animation.mesh.bones,posed_bounds))return false;
                float cosine=std::cos(animation.yaw),sine=std::sin(animation.yaw);
                float center_x=anchor.anchor_x+half_w+(anchor.u-anchor.v)*half_w;
                float center_y=anchor.anchor_y+(anchor.u+anchor.v)*half_h-anchor.ground*relief;
                CachedVertexChunk chunk,shadow_chunk;
                chunk.bounds=shadow_chunk.bounds={LONG_MAX,LONG_MAX,LONG_MIN,LONG_MIN};
                for(unsigned a=0;a<3;++a){chunk.world_bounds.low[a]=1e9f;chunk.world_bounds.high[a]=-1e9f;}
                // A posed source hull conservatively selects body and ground shadow
                // without skinning all vertices or reading bounds back from the GPU.
                for(unsigned c=0;c<8;++c){float source[3];
                    for(unsigned a=0;a<3;++a)source[a]=(c&(1u<<a))?posed_bounds.high[a]:posed_bounds.low[a];
                    float x=source[0]+animation.offset[0],y=source[1]+animation.offset[1];
                    float lx=(x*cosine-y*sine)*animation.scale,ly=(x*sine+y*cosine)*animation.scale;
                    float lz=(source[2]+animation.offset[2])*animation.scale,feature_height=lz*150.f/.82f;
                    float sx=center_x+(lx-ly)*half_w,sy=center_y+(lx+ly)*half_h-lz*150.f*projection;
                    auto cast=c3x_renderer::lighting::ground_offset(shadow_basis.data()+8,feature_height/112.f);
                    float shadow_x=center_x+(lx-ly)*half_w+(cast[0]+cast[1])*half_w;
                    float shadow_y=center_y+(lx+ly)*half_h+(cast[0]-cast[1])*half_h;
                    if(!std::isfinite(sx) || !std::isfinite(sy) || !std::isfinite(shadow_x) || !std::isfinite(shadow_y) ||
                       std::max({std::abs(sx),std::abs(sy),std::abs(shadow_x),std::abs(shadow_y)})>1e8f)return false;
                    auto extend=[](D3D11_RECT& bounds,float px,float py){
                        bounds.left=std::min(bounds.left,LONG(std::floor(px))-2);bounds.right=std::max(bounds.right,LONG(std::ceil(px))+2);
                        bounds.top=std::min(bounds.top,LONG(std::floor(py))-2);bounds.bottom=std::max(bounds.bottom,LONG(std::ceil(py))+2);
                    };
                    extend(chunk.bounds,sx,sy);extend(chunk.bounds,shadow_x,shadow_y);extend(shadow_chunk.bounds,shadow_x,shadow_y);
                    float world[]={anchor.world_u+anchor.u+lx,anchor.world_v+1-anchor.v-ly,(anchor.ground+2.5f+feature_height)/112.f};
                    for(unsigned a=0;a<3;++a){chunk.world_bounds.low[a]=std::min(chunk.world_bounds.low[a],world[a]);chunk.world_bounds.high[a]=std::max(chunk.world_bounds.high[a],world[a]);}
                }
                D3D11_RECT visible={std::max<LONG>(0,chunk.bounds.left+dx),std::max<LONG>(0,chunk.bounds.top+dy),
                    std::min<LONG>(width,chunk.bounds.right+dx),std::min<LONG>(height,chunk.bounds.bottom+dy)};
                if(selected_output_active){
                    visible={std::max<LONG>(visible.left,selected_output.left-4),std::max<LONG>(visible.top,selected_output.top-4),
                        std::min<LONG>(visible.right,selected_output.right+4),std::min<LONG>(visible.bottom,selected_output.bottom+4)};
                }
                if(visible.left>=visible.right || visible.top>=visible.bottom)continue;
                dirty(visible);
                unsigned source_bytes=unsigned(animation.mesh.vertices.size()*sizeof(c3x_renderer::AnimationVertex));
                unsigned index_bytes=unsigned(animation.mesh.indices.size()*sizeof(unsigned));
                auto room=[&](std::size_t bytes){if(pool_bytes+bytes<=32u*1024u*1024u)return true;
                    trace.write("animation-budget-failed","shared source and instance cap=33554432",true);return false;};
                if(!room((animation.vertices?0:source_bytes)+(animation.indices?0:index_bytes)))return false;
                if(!animation.vertices){
                    D3D11_BUFFER_DESC desc={};desc.ByteWidth=source_bytes;desc.Usage=D3D11_USAGE_IMMUTABLE;desc.BindFlags=D3D11_BIND_VERTEX_BUFFER;
                    D3D11_SUBRESOURCE_DATA data={};data.pSysMem=animation.mesh.vertices.data();
                    if(FAILED(device->CreateBuffer(&desc,&data,&animation.vertices)))return false;
                    pool_bytes+=source_bytes;uploaded+=source_bytes;
                }
                if(!animation.indices){
                    D3D11_BUFFER_DESC desc={};desc.ByteWidth=index_bytes;desc.Usage=D3D11_USAGE_IMMUTABLE;desc.BindFlags=D3D11_BIND_INDEX_BUFFER;
                    D3D11_SUBRESOURCE_DATA data={};data.pSysMem=animation.mesh.indices.data();
                    if(FAILED(device->CreateBuffer(&desc,&data,&animation.indices)))return false;
                    pool_bytes+=index_bytes;uploaded+=index_bytes;
                }
                if(resource_buffers.size()<=visible_resource_animations)resource_buffers.emplace_back();
                auto& pool=resource_buffers[visible_resource_animations];
                unsigned bytes=unsigned(c3x_renderer::render_core::resource_pose_bytes(animation.mesh.bones));
                if(bytes>pool.capacity){
                    if(!room(bytes-pool.capacity))return false;
                    pool_bytes-=pool.capacity;release(pool.vertices);pool.capacity=0;
                    D3D11_BUFFER_DESC desc={};desc.ByteWidth=bytes;desc.Usage=D3D11_USAGE_DYNAMIC;
                    desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;desc.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
                    if(FAILED(device->CreateBuffer(&desc,nullptr,&pool.vertices)))return false;
                    pool.capacity=bytes;pool_bytes+=bytes;
                }
                D3D11_MAPPED_SUBRESOURCE mapped={};
                if(FAILED(context->Map(pool.vertices,0,D3D11_MAP_WRITE_DISCARD,0,&mapped)))return false;
                auto values=static_cast<float*>(mapped.pData);
                float placement[]={cosine,sine,animation.scale,c3x_renderer::lighting::object_height_to_world,
                    animation.offset[0],animation.offset[1],animation.offset[2],anchor.ground,
                    center_x,center_y,anchor.world_u+anchor.u,anchor.world_v+1-anchor.v,
                    half_w,half_h,projection,float(content_view_height)};
                std::copy(std::begin(placement),std::end(placement),values);
                c3x_renderer::render_core::pack_resource_pose(pose,animation.mesh.bones,values);
                context->Unmap(pool.vertices,0);uploaded+=bytes;
                chunk.buffer=shadow_chunk.buffer=animation.vertices;
                chunk.indices=shadow_chunk.indices=animation.indices;
                chunk.resource_instance=shadow_chunk.resource_instance=pool.vertices;
                chunk.vertex_stride=shadow_chunk.vertex_stride=sizeof(c3x_renderer::AnimationVertex);
                chunk.index_count=shadow_chunk.index_count=unsigned(animation.mesh.indices.size());
                chunk.animation_texture=shadow_chunk.animation_texture=animation.view;
                buffers[geometry_shadow].push_back(shadow_chunk);buffers[geometry_feature].push_back(chunk);
                ++visible_resource_animations;if(advances)++moving_resources;continue;
            }
            if (!c3x_renderer::sample_animation_mesh(animation.mesh,time,true,posed)) {
                trace.write("animation-pose-failed",animation.name.c_str(),true);return false;
            }
            vertices.resize(posed.size());
            shadow_vertices.resize(posed.size());
            float cosine=std::cos(animation.yaw),sine=std::sin(animation.yaw);
            float center_x=anchor.anchor_x+half_w+(anchor.u-anchor.v)*half_w;
            float center_y=anchor.anchor_y+(anchor.u+anchor.v)*half_h-anchor.ground*relief;
            CachedVertexChunk chunk; chunk.bounds={LONG_MAX,LONG_MAX,LONG_MIN,LONG_MIN};
            CachedVertexChunk shadow_chunk; shadow_chunk.bounds={LONG_MAX,LONG_MAX,LONG_MIN,LONG_MIN};
            for (unsigned axis=0;axis<3;++axis){chunk.world_bounds.low[axis]=1e9f;chunk.world_bounds.high[axis]=-1e9f;}
            for (std::size_t i=0;i<posed.size();++i) {
                auto const & source=posed[i];
                float x=source.position[0]+animation.offset[0],y=source.position[1]+animation.offset[1];
                float lx=(x*cosine-y*sine)*animation.scale,ly=(x*sine+y*cosine)*animation.scale;
                float lz=(source.position[2]+animation.offset[2])*animation.scale;
                float feature_height=lz*150.f/.82f;
                float sx=center_x+(lx-ly)*half_w,sy=center_y+(lx+ly)*half_h-lz*150.f*projection;
                float depth=center_y+anchor.ground*relief+(lx+ly)*half_h+
                    anchor.ground*relief*.75f+feature_height*.0012f*content_view_height;
                auto & v=vertices[i];
                auto normal=c3x_renderer::lighting::object_normal(
                    source.normal[0]*cosine-source.normal[1]*sine,
                    source.normal[0]*sine+source.normal[1]*cosine,source.normal[2]);
                v={sx,sy,depth,source.uv[0],source.uv[1],
                    normal[0],normal[1],normal[2],21.f,
                    anchor.world_u+anchor.u+lx,anchor.world_v+1-anchor.v-ly,
                    (anchor.ground+2.5f+feature_height)/112.f};
                chunk.bounds.left=std::min(chunk.bounds.left,LONG(std::floor(sx))-2);
                chunk.bounds.top=std::min(chunk.bounds.top,LONG(std::floor(sy))-2);
                chunk.bounds.right=std::max(chunk.bounds.right,LONG(std::ceil(sx))+2);
                chunk.bounds.bottom=std::max(chunk.bounds.bottom,LONG(std::ceil(sy))+2);
                for (unsigned a=0;a<3;++a){chunk.world_bounds.low[a]=std::min(chunk.world_bounds.low[a],v[9+a]);
                    chunk.world_bounds.high[a]=std::max(chunk.world_bounds.high[a],v[9+a]);}

                // Animated resource bodies are composed over a cached static
                // scene, so entering them into the immutable world-shadow
                // atlas would force a complete terrain redraw at 15 fps.
                // Project the actual posed source triangles onto the local
                // ground instead. This preserves silhouette and the shared
                // frame-light basis while keeping redraws bounded.
                float height_world=feature_height/112.f;
                auto cast=c3x_renderer::lighting::ground_offset(shadow_basis.data()+8,height_world);
                float shadow_u=cast[0],shadow_v=cast[1];
                float shadow_x=center_x+(lx-ly)*half_w+(shadow_u+shadow_v)*half_w;
                float shadow_y=center_y+(lx+ly)*half_h+(shadow_u-shadow_v)*half_h;
                Vertex projected={};
                projected.x=shadow_x;projected.y=shadow_y;
                projected.z=shadow_y+anchor.ground*relief*1.75f;
                projected.u=source.uv[0];projected.v=source.uv[1];projected.panel=1.f;
                projected.normal_z=1.f;projected.shadow_visibility=1.f;
                projected.ambient_visibility=1.f;projected.surface_kind=15.f;
                shadow_vertices[i]=projected;
                shadow_chunk.bounds.left=std::min(shadow_chunk.bounds.left,LONG(std::floor(shadow_x))-2);
                shadow_chunk.bounds.top=std::min(shadow_chunk.bounds.top,LONG(std::floor(shadow_y))-2);
                shadow_chunk.bounds.right=std::max(shadow_chunk.bounds.right,LONG(std::ceil(shadow_x))+2);
                shadow_chunk.bounds.bottom=std::max(shadow_chunk.bounds.bottom,LONG(std::ceil(shadow_y))+2);
            }
            chunk.bounds.left=std::min(chunk.bounds.left,shadow_chunk.bounds.left);
            chunk.bounds.top=std::min(chunk.bounds.top,shadow_chunk.bounds.top);
            chunk.bounds.right=std::max(chunk.bounds.right,shadow_chunk.bounds.right);
            chunk.bounds.bottom=std::max(chunk.bounds.bottom,shadow_chunk.bounds.bottom);
            D3D11_RECT visible={std::max<LONG>(0,chunk.bounds.left+dx),std::max<LONG>(0,chunk.bounds.top+dy),
                std::min<LONG>(width,chunk.bounds.right+dx),std::min<LONG>(height,chunk.bounds.bottom+dy)};
            if (visible.left>=visible.right || visible.top>=visible.bottom) continue;
            dirty(visible);
            if (resource_buffers.size()<=visible_resource_animations) resource_buffers.emplace_back();
            auto & pool=resource_buffers[visible_resource_animations];
            unsigned bytes=unsigned(vertices.size()*sizeof(vertices[0]));
            if (bytes>pool.capacity) {
                if (pool_bytes-pool.capacity+bytes>32u*1024u*1024u) {
                    trace.write("animation-budget-failed","vertex buffer pool cap=33554432",true);return false;
                }
                pool_bytes-=pool.capacity; release(pool.vertices); pool.capacity=0;
                D3D11_BUFFER_DESC desc={};desc.ByteWidth=bytes;desc.Usage=D3D11_USAGE_DYNAMIC;
                desc.BindFlags=D3D11_BIND_VERTEX_BUFFER;desc.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
                if (FAILED(device->CreateBuffer(&desc,nullptr,&pool.vertices))) return false;
                pool.capacity=bytes;pool_bytes+=bytes;
            }
            D3D11_MAPPED_SUBRESOURCE mapped={};
            if (FAILED(context->Map(pool.vertices,0,D3D11_MAP_WRITE_DISCARD,0,&mapped))) return false;
            std::memcpy(mapped.pData,vertices.data(),bytes);context->Unmap(pool.vertices,0);uploaded+=bytes;
            unsigned shadow_bytes=unsigned(shadow_vertices.size()*sizeof(shadow_vertices[0]));
            if (shadow_bytes>pool.shadow_capacity) {
                if (pool_bytes-pool.shadow_capacity+shadow_bytes>32u*1024u*1024u) {
                    trace.write("animation-budget-failed","vertex buffer pool cap=33554432",true);return false;
                }
                pool_bytes-=pool.shadow_capacity;release(pool.shadow_vertices);pool.shadow_capacity=0;
                D3D11_BUFFER_DESC desc={};desc.ByteWidth=shadow_bytes;desc.Usage=D3D11_USAGE_DYNAMIC;
                desc.BindFlags=D3D11_BIND_VERTEX_BUFFER;desc.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
                if(FAILED(device->CreateBuffer(&desc,nullptr,&pool.shadow_vertices)))return false;
                pool.shadow_capacity=shadow_bytes;pool_bytes+=shadow_bytes;
            }
            if(FAILED(context->Map(pool.shadow_vertices,0,D3D11_MAP_WRITE_DISCARD,0,&mapped)))return false;
            std::memcpy(mapped.pData,shadow_vertices.data(),shadow_bytes);
            context->Unmap(pool.shadow_vertices,0);uploaded+=shadow_bytes;
            if (!animation.indices) {
                D3D11_BUFFER_DESC desc={};desc.ByteWidth=unsigned(animation.mesh.indices.size()*sizeof(unsigned));
                desc.Usage=D3D11_USAGE_IMMUTABLE;desc.BindFlags=D3D11_BIND_INDEX_BUFFER;
                D3D11_SUBRESOURCE_DATA data={};data.pSysMem=animation.mesh.indices.data();
                if (FAILED(device->CreateBuffer(&desc,&data,&animation.indices))) return false;
            }
            chunk.buffer=pool.vertices;chunk.indices=animation.indices;
            chunk.vertex_stride=48;chunk.index_count=unsigned(animation.mesh.indices.size());
            chunk.animation_texture=animation.view;
            shadow_chunk.buffer=pool.shadow_vertices;shadow_chunk.indices=animation.indices;
            shadow_chunk.vertex_stride=sizeof(Vertex);shadow_chunk.index_count=chunk.index_count;
            // The animated source mesh includes alpha-cutout cards.  The
            // shadow pass must see the same authored coverage as the body or
            // it projects each complete card as an opaque black rectangle.
            shadow_chunk.animation_texture=animation.view;
            buffers[geometry_shadow].push_back(shadow_chunk);
            buffers[geometry_feature].push_back(chunk);++visible_resource_animations;if(advances)++moving_resources;
        }
        if(!prepare_wave_chunks(frame))return false;
        visible_wave_animations=unsigned(std::count_if(wave_chunks.begin(),wave_chunks.end(),[](auto const& chunk){return chunk.visual_time<0;}));
        buffers[geometry_wave]=wave_chunks;
        if(!wave_chunks.empty()){
            wave_time_seconds=float(double(ticks)/std::max<c3x_renderer_i64>(1,frame.presentation_frequency));
            float time[]={wave_time_seconds,0,0,0};
            context->UpdateSubresource(wave_frame,0,nullptr,time,0,0);
            for(auto const& chunk:wave_chunks){
                int wave_dx=chunk.translation_x+dx,wave_dy=chunk.translation_y+dy;
                D3D11_RECT visible={std::max<LONG>(0,chunk.bounds.left+wave_dx),std::max<LONG>(0,chunk.bounds.top+wave_dy),std::min<LONG>(width,chunk.bounds.right+wave_dx),std::min<LONG>(height,chunk.bounds.bottom+wave_dy)};
                dirty(visible);
            }
        }
        if(shared_scene_surface) {
            if(!wave_chunks.empty()){trace.write("scene-surface-failed","waves outside bounded alternative",true);return false;}
            if(!compose_scene_surface(buffers))return false;
            resource_pixel_signature=cached_signature.complete;resource_pixel_clock=clock;
            QueryPerformanceCounter(&finished);resource_composite_ticks=finished.QuadPart-started.QuadPart;
            char detail[128];sprintf_s(detail,"ms=%.3f",trace.milliseconds(resource_composite_ticks));
            trace.write("scene-composition",detail,true);
            return true;
        }
        if (!posed_count()) {resource_pixel_signature=0;return true;}
        // All regions in this composition borrow the same pinned static
        // geometry and light basis, including their reflection passes.
        using Shadow=c3x_renderer::render_core::SourceShadow;
        std::vector<Shadow::Caster> animation_casters;
        Shadow::PreparedCasters animation_prepared;
        std::vector<Shadow::Caster> const* animation_casters_ptr=nullptr;
        Shadow::PreparedCasters* animation_prepared_ptr=nullptr;
        char caster_control[8]={};
        bool share_casters=!(GetEnvironmentVariableA("C3X_RENDERER_COMPOSITION_CASTERS_CONTROL",caster_control,sizeof(caster_control)) && std::strcmp(caster_control,"1")==0);
        if(pickup_profile && share_casters) {
            animation_prepared_ptr=prepare_shadow_submission(geometry_vertex_buffers,animation_casters,animation_prepared);
            animation_casters_ptr=&animation_casters;
        }
        for(int y=0;y<backdrop_grid_y.count;++y)for(int x=0;x<backdrop_grid_x.count;++x)
            if(dirty_blocks[std::size_t(y)*backdrop_grid_x.count+x]) {
                int left=backdrop_grid_x.start(x),top=backdrop_grid_y.start(y);
                rectangles.push_back({left,top,left+128,top+128});
            }
        ++resource_backdrop_epoch;
        if(!ensure_block_targets())return false;
        // City fidelity renders each native 128-pixel block through a guarded
        // 136-pixel target. Preserve that actual scene-linear target so the
        // animated pass accumulates over terrain instead of a cleared block.
        auto & backdrop=city_profile?city_glow.linear:linear_block;
        int backdrop_extent=city_profile?272:fidelity_profile?256:128;
        if(!backdrop.ensure(device,backdrop_extent,backdrop_extent))return false;
        unsigned backdrop_hits=0,backdrop_misses=0;
        bool dependency_backdrops=anchored && city_profile && world_regions && animation_prepared_ptr &&
            c3x_renderer::NavigationOptions::retained(GetEnvironmentVariableA,"C3X_RENDERER_BACKDROP_DEPENDENCIES");
        unsigned backdrop_dependency_hits=0,backdrop_dependency_rejections=0;
        // Posed bodies and their projected shadows have one explicit binding
        // contract. Wave shading continues through its existing scene pass.
        bool prepared_resource_pass=city_profile && !visible_wave_animations &&
            c3x_renderer::NavigationOptions::enabled(GetEnvironmentVariableA,"C3X_RENDERER_PREPARED_RESOURCE_PASS");
        struct PreparedResourceRegion {
            ViewportShaderSettings settings;
            std::array<std::vector<CachedVertexChunk>,geometry_layer_count> draws;
            D3D11_RECT output_damage={136,136,0,0};
            std::size_t shadow_batch=0;
        };
        using ShadowPages=std::set<std::pair<int,int>>;
        std::vector<PreparedResourceRegion> prepared_regions;
        std::vector<ShadowPages> shadow_batches;
        std::size_t prepared_bytes=0;
        constexpr std::size_t prepared_cap=4u*1024u*1024u;
        // Account conservatively for set links/alignment in the x86 allocator.
        constexpr std::size_t shadow_batch_bytes=sizeof(ShadowPages)+32u*64u;
        prepared_resource_pass=prepared_resource_pass && animation_casters_ptr && !clip_dirty_blocks &&
            rectangles.size()<=prepared_cap/(sizeof(PreparedResourceRegion)+shadow_batch_bytes);
        if(prepared_resource_pass) {
            // Borrow pinned pose buffers, preserving exact guarded eligibility
            // and per-layer order. Consecutive regions share a bounded union of
            // required shadow pages; no new atlas or longer resource lifetime.
            try {
                prepared_regions.reserve(rectangles.size());
                shadow_batches.reserve(rectangles.size());
                prepared_bytes=prepared_regions.capacity()*sizeof(PreparedResourceRegion)+
                    shadow_batches.capacity()*shadow_batch_bytes;
                std::vector<Shadow::Bounds> receivers;
                for(auto const& rect:rectangles) {
                    prepared_regions.emplace_back();auto& region=prepared_regions.back();
                    region.settings=geometry_viewport_settings;
                    region.settings.translation[0]+=4-float(rect.left);
                    region.settings.translation[1]+=4-float(rect.top);
                    region.settings.inverse_size[0]=region.settings.inverse_size[1]=1.f/136;
                    D3D11_RECT guard={0,0,136,136};
                    for(auto layer:{geometry_shadow,geometry_feature}) {
                        std::size_t count=0;
                        for(auto const& chunk:buffers[layer])
                            if(chunk_intersects_region(chunk,region.settings,guard,false))++count;
                        if(count>(prepared_cap-std::min(prepared_cap,prepared_bytes))/sizeof(CachedVertexChunk)) {
                            prepared_resource_pass=false;break;
                        }
                        region.draws[layer].reserve(count);
                        prepared_bytes+=region.draws[layer].capacity()*sizeof(CachedVertexChunk);
                        for(auto const& chunk:buffers[layer])
                            if(chunk_intersects_region(chunk,region.settings,guard,false)){
                                region.draws[layer].push_back(chunk);
                                // Existing posed hulls include raster coverage. The
                                // post filter reaches +/-8 high-resolution samples,
                                // or four native pixels. Include guard-only draws too.
                                auto damage=guarded_block_rectangle(chunk.bounds,
                                    int(region.settings.translation[0])+chunk.translation_x,
                                    int(region.settings.translation[1])+chunk.translation_y,4,136);
                                auto& output=region.output_damage;
                                output.left=std::min(output.left,damage.left);output.top=std::min(output.top,damage.top);
                                output.right=std::max(output.right,damage.right);output.bottom=std::max(output.bottom,damage.bottom);
                            }
                    }
                    if(!prepared_resource_pass)break;
                    auto& output=region.output_damage;
                    output.left=std::max<LONG>(4,output.left);output.top=std::max<LONG>(4,output.top);
                    output.right=std::min<LONG>(132,output.right);output.bottom=std::min<LONG>(132,output.bottom);
                    if(output.left>=output.right || output.top>=output.bottom)output={4,4,132,132};
                    receivers.clear();
                    collect_region_receivers(geometry_vertex_buffers,region.settings,{guard},false,receivers);
                    auto pages=Shadow::required_pages(receivers,shadow_basis);
                    if(pages.size()>32){prepared_resource_pass=false;break;}
                    std::size_t additional=pages.size();
                    if(!shadow_batches.empty()) {
                        additional=0;
                        for(auto const& page:pages)if(!shadow_batches.back().count(page))++additional;
                    }
                    if(shadow_batches.empty() || shadow_batches.back().size()+additional>32)
                        shadow_batches.emplace_back();
                    shadow_batches.back().insert(pages.begin(),pages.end());
                    region.shadow_batch=shadow_batches.size()-1;
                }
                if(prepared_bytes>prepared_cap)prepared_resource_pass=false;
            } catch(...) {prepared_resource_pass=false;}
            if(!prepared_resource_pass) {
                std::vector<PreparedResourceRegion>().swap(prepared_regions);
                std::vector<ShadowPages>().swap(shadow_batches);prepared_bytes=0;
            }
        }
        if(prepared_bytes){char detail[160];sprintf_s(detail,"regions=%zu shadow_batches=%zu metadata_bytes=%zu metadata_cap=4194304",
                prepared_regions.size(),shadow_batches.size(),prepared_bytes);
            trace.write("prepared-resource-pass",detail);}
        // Geometry identity includes the entire captured semantic/ownership
        // set, target/zoom, light, wrap, content and device generations, but
        // excludes camera anchors. Conservatively miss when that set changes.
        auto backdrop_signature=anchored?c3x_renderer::render_core::static_region_identity(
            cached_signature.geometry,std::uint64_t(geometry_world_revision)):cached_signature.complete;
        if(profiling){
            poll_animation_gpu();
            if(!animation_gpu.begin(device,context,trace.sequence.load(),unsigned(rectangles.size()*8+8)))
                trace.write("animation-gpu-unavailable","bounded query capacity, pressure or allocation failure",true);
        }
        AnimationGpu::Scope animation_gpu_scope{animation_gpu,context};
        LARGE_INTEGER poses_ready={},background_started={},animation_started={},region_finished={};
        QueryPerformanceCounter(&poses_ready);
        LONGLONG background_ticks=0,animation_ticks=0;
        unsigned atlas_columns=0,atlas_rows=0;
        if(animation_readback_atlas && !rectangles.empty()) {
            while(atlas_columns*atlas_columns<rectangles.size())++atlas_columns;
            atlas_rows=unsigned((rectangles.size()+atlas_columns-1)/atlas_columns);
            unsigned atlas_width=atlas_columns*128,atlas_height=atlas_rows*128;
            if(!animation_readback_texture || animation_readback_width!=atlas_width || animation_readback_height!=atlas_height) {
                release(animation_readback_texture);
                D3D11_TEXTURE2D_DESC desc={};desc.Width=atlas_width;desc.Height=atlas_height;
                desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
                desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.Usage=D3D11_USAGE_STAGING;
                desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
                if(FAILED(device->CreateTexture2D(&desc,nullptr,&animation_readback_texture)))return false;
                animation_readback_width=atlas_width;animation_readback_height=atlas_height;
            }
        }
        std::size_t rectangle_index=0,active_shadow_batch=std::size_t(-1);
        for(auto & rect:rectangles) {
            QueryPerformanceCounter(&background_started);
            int key_x=rect.left-anchor_x,key_y=rect.top-anchor_y;
            auto found=std::find_if(resource_backdrops.begin(),resource_backdrops.end(),[&](auto const& block){
                return !backdrop_reuse_control && block.depth_origin==scene_depth_origin && block.signature==backdrop_signature && block.x==key_x && block.y==key_y;
            });
            ViewportShaderSettings settings=geometry_viewport_settings;
            settings.translation[0]-=float(rect.left);settings.translation[1]-=float(rect.top);
            settings.inverse_size[0]=settings.inverse_size[1]=1.f/128;
            c3x_renderer::render_core::RenderRegionKey backdrop_dependencies;
            if(dependency_backdrops && !backdrop_reuse_control && found==resource_backdrops.end()) {
                // Reuse the existing completed-region dependency contract, but
                // retain BOTH unresolved scene-linear color and depth here.
                // submit_geometry applies this same four-pixel city guard to
                // the 128-pixel animation background before its static draw.
                auto guarded=settings;
                guarded.translation[0]+=4;guarded.translation[1]+=4;
                guarded.inverse_size[0]=guarded.inverse_size[1]=1.f/136;
                bool valid=false;
                try { valid=render_region_key(geometry_vertex_buffers,guarded,*animation_casters_ptr,
                            animation_prepared_ptr,backdrop_dependencies); }
                catch(...) {}
                if(valid) {
                    found=std::find_if(resource_backdrops.begin(),resource_backdrops.end(),[&](auto const& block){
                        return block.depth_origin==scene_depth_origin && !block.dependencies.empty() && block.dependencies==backdrop_dependencies;
                    });
                    if(found!=resource_backdrops.end())++backdrop_dependency_hits;
                } else {backdrop_dependencies={};++backdrop_dependency_rejections;}
            }
            context->OMSetRenderTargets(0,nullptr,nullptr);
            if(found!=resource_backdrops.end()) {
                found->used=resource_backdrop_epoch;
                // Pin a dependency hit to the current view for the existing
                // bounded eviction policy and its cheap unchanged-view lookup.
                found->signature=backdrop_signature;found->x=key_x;found->y=key_y;
                AnimationGpu::Pass gpu_phase(animation_gpu,context,AnimationGpu::import);
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
                if(diagnostic_animation<3)
#endif
                {context->CopyResource(backdrop.color,found->color);
                 context->CopyResource(backdrop.depth_texture,found->depth);}
                ++backdrop_hits;
            } else {
                AnimationGpu::Pass gpu_phase(animation_gpu,context,AnimationGpu::background);
                if(!submit_geometry(geometry_vertex_buffers,{{0,0,128,128}},settings,block_target,block_depth,128,128,
                    nullptr,false,true,nullptr,false,animation_casters_ptr,animation_prepared_ptr,128,0,0,true))return false;
                // Static submission can replace the page table or atlas slots.
                active_shadow_batch=std::size_t(-1);
                ++backdrop_misses;
                // RGBA16F + D24S8, both MSAA4. Cache immutable scene-linear
                // background/depth by static inputs and the region's relative
                // world placement. Animation time and poses never enter it.
                std::size_t bytes=std::size_t(backdrop_extent)*backdrop_extent*48u+
                    backdrop_dependencies.capacity()*sizeof(backdrop_dependencies[0])+sizeof(ResourceBackdrop);
                if(!backdrop_reuse_control && make_resource_backdrop_room(bytes,backdrop_signature)) {
                    resource_backdrops.reserve(resource_backdrops.size()+1);
                    ResourceBackdrop block;block.x=key_x;block.y=key_y;block.bytes=bytes;block.depth_origin=scene_depth_origin;
                    block.signature=backdrop_signature;block.used=resource_backdrop_epoch;
                    block.dependencies=std::move(backdrop_dependencies);
                    D3D11_TEXTURE2D_DESC desc={};backdrop.color->GetDesc(&desc);
                    bool allocated=SUCCEEDED(device->CreateTexture2D(&desc,nullptr,&block.color));
                    backdrop.depth_texture->GetDesc(&desc);
                    if(allocated)allocated=SUCCEEDED(device->CreateTexture2D(&desc,nullptr,&block.depth));
                    if(allocated){
                        context->CopyResource(block.color,backdrop.color);
                        context->CopyResource(block.depth,backdrop.depth_texture);
                        resource_backdrops.push_back(std::move(block));resource_backdrop_bytes+=bytes;
                    }else{
                        release(block.color);release(block.depth);
                        trace.write("animation-backdrop","cache allocation skipped; current backdrop remains valid",true);
                    }
                }
            }
            // Keep the background depth and linear color intact. Reuse static
            // source casters for lighting; animated bodies never invalidate pages.
            QueryPerformanceCounter(&animation_started);
            background_ticks+=animation_started.QuadPart-background_started.QuadPart;
            D3D11_RECT clipped={std::max<LONG>(0,rect.left),std::max<LONG>(0,rect.top),
                std::min<LONG>(width,rect.right),std::min<LONG>(height,rect.bottom)};
            if(prepared_resource_pass) {
                auto const& region=prepared_regions[rectangle_index];
                if(active_shadow_batch!=region.shadow_batch) {
                    AnimationGpu::Pass gpu_phase(animation_gpu,context,AnimationGpu::receivers);
                    if(!prepare_receiver_shadows(geometry_vertex_buffers,region.settings,{{0,0,136,136}},false,
                            *animation_casters_ptr,animation_prepared_ptr,nullptr,&shadow_batches[region.shadow_batch]))return false;
                    active_shadow_batch=region.shadow_batch;
                }
                if(!submit_prepared_resource_region(region.draws,region.settings,region.output_damage))return false;
                clipped.left=std::max(clipped.left,rect.left+region.output_damage.left-4);
                clipped.top=std::max(clipped.top,rect.top+region.output_damage.top-4);
                clipped.right=std::min(clipped.right,rect.left+region.output_damage.right-4);
                clipped.bottom=std::min(clipped.bottom,rect.top+region.output_damage.bottom-4);
            } else if(!submit_geometry(buffers,{{0,0,128,128}},settings,block_target,block_depth,128,128,
                    nullptr,true,true,geometry_vertex_buffers,false,animation_casters_ptr,animation_prepared_ptr))return false;
            {AnimationGpu::Pass gpu_phase(animation_gpu,context,AnimationGpu::transfer);
            D3D11_BOX box={unsigned(clipped.left-rect.left),unsigned(clipped.top-rect.top),0,
                unsigned(clipped.right-rect.left),unsigned(clipped.bottom-rect.top),1};
            if(animation_readback_atlas) {
                unsigned atlas_x=unsigned(rectangle_index%atlas_columns)*128;
                unsigned atlas_y=unsigned(rectangle_index/atlas_columns)*128;
                context->CopySubresourceRegion(animation_readback_texture,0,atlas_x,atlas_y,0,block_texture,0,&box);
            } else {
                context->CopySubresourceRegion(render_texture,0,unsigned(clipped.left),unsigned(clipped.top),0,block_texture,0,&box);
            }
            }
            rect=clipped;
            ++rectangle_index;
            QueryPerformanceCounter(&region_finished);
            animation_ticks+=region_finished.QuadPart-animation_started.QuadPart;
        }
        LARGE_INTEGER readback_started={},readback_submitted={},readback_ready={};
        QueryPerformanceCounter(&readback_started);
        unsigned dirty_pixels=0;
        {AnimationGpu::Pass gpu_phase(animation_gpu,context,AnimationGpu::transfer,!animation_readback_atlas);
        for(auto const & rect:rectangles) {
            if(!animation_readback_atlas) {
                D3D11_BOX box={unsigned(rect.left),unsigned(rect.top),0,unsigned(rect.right),unsigned(rect.bottom),1};
                context->CopySubresourceRegion(readback_texture,0,box.left,box.top,0,render_texture,0,&box);
            }
            dirty_pixels+=unsigned((rect.right-rect.left)*(rect.bottom-rect.top));
        }
        }
        animation_gpu.end(context);
        D3D11_MAPPED_SUBRESOURCE mapped={};
        QueryPerformanceCounter(&readback_submitted);
        auto mapped_texture=animation_readback_atlas?animation_readback_texture:readback_texture;
        if(FAILED(context->Map(mapped_texture,0,D3D11_MAP_READ,0,&mapped))) return false;
        QueryPerformanceCounter(&readback_ready);
        // The immutable base owns every old body position. Never cache posed pixels.
        resource_pixels=pixels;
        rectangle_index=0;
        for(auto const & rect:rectangles) {
            unsigned source_x=animation_readback_atlas?unsigned(rectangle_index%atlas_columns)*128:unsigned(rect.left);
            unsigned source_y=animation_readback_atlas?unsigned(rectangle_index/atlas_columns)*128:unsigned(rect.top);
            for(LONG y=rect.top;y<rect.bottom;++y)
                std::memcpy(resource_pixels.data()+std::size_t(y)*width+rect.left,
                    static_cast<std::uint8_t const*>(mapped.pData)+std::size_t(source_y+y-rect.top)*mapped.RowPitch+std::size_t(source_x)*4,
                    std::size_t(rect.right-rect.left)*4);
            ++rectangle_index;
        }
        context->Unmap(mapped_texture,0);
        resource_pixel_signature=cached_signature.complete;resource_pixel_clock=clock;
        QueryPerformanceCounter(&finished);resource_composite_ticks=finished.QuadPart-started.QuadPart;
        if(profiling)poll_animation_gpu();
        {
            char detail[320];sprintf_s(detail,"pose_prepare_ms=%.3f backdrop_submit_ms=%.3f animated_submit_ms=%.3f readback_submit_ms=%.3f readback_wait_ms=%.3f cpu_copy_ms=%.3f",
                trace.milliseconds(poses_ready.QuadPart-started.QuadPart),trace.milliseconds(background_ticks),
                trace.milliseconds(animation_ticks),trace.milliseconds(readback_submitted.QuadPart-readback_started.QuadPart),
                trace.milliseconds(readback_ready.QuadPart-readback_submitted.QuadPart),trace.milliseconds(finished.QuadPart-readback_ready.QuadPart));
            trace.write("animation-phases",detail);
        }
        if(dependency_backdrops) {
            char detail[128];sprintf_s(detail,"hits=%u rejections=%u entries=%zu bytes=%zu",
                backdrop_dependency_hits,backdrop_dependency_rejections,resource_backdrops.size(),resource_backdrop_bytes);
            trace.write("animation-backdrop-dependencies",detail);
        }
        char detail[640];sprintf_s(detail,"visible=%u waves=%u facing=SE clock=%lld rects=%zu pixels=%u upload_bytes=%zu pool_bytes=%zu backdrop_hits=%u backdrop_misses=%u backdrop_bytes=%zu terrain_built=%u ms=%.3f wave_upload_bytes=%zu wave_geometry_bytes=%zu wave_cells_built=%u wave_cells_reused=%u wave_cell_entries=%zu caster_preparations=%u readback=%s readback_width=%u readback_height=%u resource_material_variants=%u",
            visible_resource_animations,visible_wave_animations,clock,rectangles.size(),dirty_pixels,uploaded,pool_bytes,backdrop_hits,backdrop_misses,
            resource_backdrop_bytes,frame_tiles_built,
            trace.milliseconds(resource_composite_ticks),wave_upload_bytes,wave_geometry_bytes,wave_cells_built,wave_cells_reused,retained_wave_cells.size(),frame_caster_preparations,
            animation_readback_atlas?"atlas":"full",animation_readback_atlas?animation_readback_width:unsigned(width),animation_readback_atlas?animation_readback_height:unsigned(height),unsigned(prepared_resource_pass && !rectangles.empty()));trace.write("animation-frame",detail);
        return true;
    }

    bool submit_scene_pass(GeometryDrawView inputs,std::vector<unsigned> const& order,bool dynamic_pass,
            c3x_renderer::render_core::LinearTarget& target,c3x_renderer::city_fidelity::Glow& glow,
            ViewportShaderSettings const& settings,unsigned w,unsigned h,std::vector<D3D11_RECT> const& physical_rectangles,
            unsigned& batches,unsigned& selected_static,unsigned& selected_dynamic,unsigned& selection_candidates,unsigned& selection_scans) {
        auto select_begin=std::chrono::steady_clock::now();double execute_ms=0;
        using Shadow=c3x_renderer::render_core::SourceShadow;
        std::vector<Shadow::Caster> casters;Shadow::PreparedCasters prepared;
        auto prepared_ptr=prepare_shadow_submission(geometry_vertex_buffers,casters,prepared);
        auto spans=c3x_renderer::render_core::scene_spans<D3D11_RECT>(int(w),int(h),region_origin_x,region_origin_y);
            for(auto span:spans){
            ViewportShaderSettings pass_settings=settings;
            pass_settings.translation[0]+=float(span.x);pass_settings.translation[1]+=float(span.y);
            std::vector<D3D11_RECT> rectangles;
            for(auto rect:physical_rectangles){
                rect={std::max(span.rect.left,rect.left),std::max(span.rect.top,rect.top),
                      std::min(span.rect.right,rect.right),std::min(span.rect.bottom,rect.bottom)};
                if(rect.left<rect.right && rect.top<rect.bottom)rectangles.push_back(rect);
            }
            if(rectangles.empty())continue;
            // The existing view index now selects real submission inputs.
            // It is invalidated with the occurrence assembly, independent of
            // raster surface cells. Dynamic pose lists keep their short scan.
            using Item=c3x_renderer::render_core::RegionContributorIndex::Item;
            std::vector<Item> candidates;
            bool indexed=retained_world && !dynamic_pass && region_contributors.ready;
            if(indexed){
                std::vector<Item> found;
                for(auto const& rect:rectangles){
                    if(!query_region_inputs(0,rect.left-int(pass_settings.translation[0]),
                        rect.top-int(pass_settings.translation[1]),rect.right-rect.left,rect.bottom-rect.top,found) ||
                        candidates.size()+found.size()>region_contributors.budget/(2*sizeof(Item))){indexed=false;break;}
                    candidates.insert(candidates.end(),found.begin(),found.end());
                }
                if(indexed){std::sort(candidates.begin(),candidates.end());
                    candidates.erase(std::unique(candidates.begin(),candidates.end()),candidates.end());}
            }
            GeometryDrawView::Records selected;
            std::set<std::pair<int,int>> pages;
            std::vector<Shadow::Bounds> receivers;
            std::size_t selected_bytes=0;
            auto flush=[&](){
                if(!GeometryDrawView(selected).pass().any())return true;
                if(!pages.empty() && !prepare_receiver_shadows(geometry_vertex_buffers,pass_settings,{span.rect},false,casters,prepared_ptr,nullptr,&pages))return false;
                auto execute_begin=std::chrono::steady_clock::now();
                bool ok=dynamic_pass?submit_prepared_resource_region(selected,pass_settings,span.rect,&glow):
                    submit_geometry(selected,rectangles,pass_settings,target.target,target.depth,int(w),int(h),nullptr,
                        true,false,nullptr,false,&casters,prepared_ptr,128,0,0,false,true);
                execute_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-execute_begin).count();
                for(auto& layer:selected)layer.clear();pages.clear();++batches;return ok;
            };
            for(auto layer:order){
                auto begin=std::lower_bound(candidates.begin(),candidates.end(),Item{layer,0});
                auto end=std::lower_bound(begin,candidates.end(),Item{layer+1,0});
                std::size_t count=indexed?std::size_t(end-begin):inputs[layer].size();
                if(indexed)selection_candidates+=unsigned(count);else selection_scans+=unsigned(count);
                for(std::size_t i=0;i<count;++i){
                    auto item=inputs[layer][indexed?(begin+i)->second:i];
                    bool visible=false;for(auto const& rect:rectangles)visible=visible || chunk_intersects_region(item,pass_settings,rect,false);
                    if(!visible)continue;
                    GeometryDrawRecord record(item.content());
                    record.bounds=item.bounds();record.translation_x=item.translation_x();record.translation_y=item.translation_y();
                    std::copy(item.natural_projection(),item.natural_projection()+4,record.natural_projection);
                    receivers.clear();
                    if(layer!=geometry_shadow)receivers.push_back(item.content().world_bounds);
                    auto needed=Shadow::required_pages(receivers,shadow_basis);
                    // Some non-receiving layers still draw; an empty sentinel
                    // batch uses no extra shadow slots but must be submitted.
                    std::size_t additional=0;for(auto const& page:needed)if(!pages.count(page))++additional;
                    if(pages.size()+additional>32 && !flush())return false;
                    if(needed.size()>32)return false;
                    pages.insert(needed.begin(),needed.end());
                    auto before=selected[layer].capacity();selected[layer].push_back(record);
                    selected_bytes+=(selected[layer].capacity()-before)*sizeof(GeometryDrawRecord);
                    if(selected_bytes>4u*1024u*1024u)return false;
                    if(dynamic_pass)++selected_dynamic;else ++selected_static;
                }
                // Keep adjacent pass inputs together while their receiver page
                // union fits. The executor preserves this exact layer order;
                // only incompatible shadow bindings force a submission split.
            }
            if(!flush())return false;
            }
            frame_scene_execute_ms+=execute_ms;
            frame_scene_select_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-select_begin).count()-execute_ms;
            return true;
    }

    std::vector<unsigned> static_scene_order() const {
        std::vector<unsigned> order={geometry_underlay,geometry_land,geometry_natural_terrain,geometry_natural_mountain,geometry_natural_decal};
        for(unsigned layer=geometry_natural_forest0;layer<geometry_layer_count;++layer)order.push_back(layer);
        for(auto layer:{geometry_bed,geometry_water,geometry_river,geometry_shadow,geometry_route})order.push_back(layer);
        for(unsigned i=0;i<cliff_bundle.assets.size();++i)order.push_back(geometry_cliff0+i);
        for(auto layer:{geometry_feature,geometry_site,geometry_mine,geometry_farm,geometry_city,geometry_wall})order.push_back(layer);
        return order;
    }

    bool draw_scene_guard(std::vector<D3D11_RECT> const& rectangles,bool background=false) {
        if(rectangles.empty())return true;
        auto& glow=region_glow;
        ViewportShaderSettings settings=geometry_viewport_settings;
        settings.translation[0]+=4+scene_guard_pad;settings.translation[1]+=4+scene_guard_pad;
        settings.inverse_size[0]=1.f/scene_guard.width;settings.inverse_size[1]=1.f/scene_guard.height;
        for(auto rect:rectangles){std::vector<D3D11_RECT> one={rect};
            if(!scene_restore.draw(context,scene_scratch,glow.linear.samples,glow.linear.depth_samples,0,0,one,&one))return false;
        }
        std::vector<c3x_renderer::city_fidelity::Lighting const*> lights;
        for(auto const& item:geometry_vertex_buffers[geometry_city])if(item.content().city_lighting){
            auto pointer=item.content().city_lighting.get();
            if(std::find(lights.begin(),lights.end(),pointer)==lights.end())lights.push_back(pointer);
        }
        if(!cities.lights(context,lights))return false;
        unsigned batches=0,selected=0,dynamic=0,candidates=0,scans=0;
        if(!submit_scene_pass(geometry_vertex_buffers,static_scene_order(),false,scene_scratch,glow,settings,
            unsigned(scene_guard.width),unsigned(scene_guard.height),rectangles,batches,selected,dynamic,candidates,scans))return false;
        scene_guard.commit(rectangles);
        std::size_t prepared_pixels=0;for(auto r:rectangles)prepared_pixels+=std::size_t(r.right-r.left)*(r.bottom-r.top);
        char detail[224];sprintf_s(detail,"background=%u pixels=%zu selected=%u batches=%u pending_cells=%zu pad=%d target_bytes=%zu",
            unsigned(background),prepared_pixels,selected,batches,scene_guard.pending,scene_guard_pad,scene_scratch.bytes());
        trace.write("scene-guard-submit",detail,true);return true;
    }

    bool scene_guard_pending() const {
        return world_preparation && shared_scene_surface && scene_guard_pad && !scene_guard_failed &&
            scene_guard.pending && scene_scratch.color && cache_valid && geometry_cache.valid;
    }
    bool prepare_scene_guard(std::atomic<bool> const& cancelled) {
        if(!scene_guard_pending() || cancelled.load(std::memory_order_relaxed))return true;
        auto rectangles=scene_guard.select({},128u*1024u);
        LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
        bool ok=draw_scene_guard(rectangles,true);
        if(!ok){scene_guard.invalidate_all();scene_guard_failed=true;}
        // Submit to the existing GPU owner without readback or a completion wait.
        // A later foreground request executes after these commands in order.
        context->Flush();QueryPerformanceCounter(&end);
        char detail[160];sprintf_s(detail,"ok=%u submit_ms=%.3f pending_cells=%zu readbacks=0",
            unsigned(ok),trace.milliseconds(end.QuadPart-begin.QuadPart),scene_guard.pending);
        trace.write("scene-guard-prepared",detail,true);return ok;
    }

    std::shared_ptr<c3x_renderer::UnitSceneSource> unit_scene_source(){
        // Raw color/depth precede fog. Final composition uses the published
        // full-color underlay, preserving its coverage; hidden units are already
        // excluded by the authoritative unit-input contract.
        if(!shared_scene_surface||!gpu_map_valid||!region_glow.linear.samples)return {};
        auto source=std::make_shared<c3x_renderer::UnitSceneSource>();auto generation=scene_generation;
        source->capture=[this,generation](c3x_gpu_images::Rect r)->std::shared_ptr<c3x_renderer::UnitSceneRegion>{
            int w=r.right-r.left,h=r.bottom-r.top;constexpr std::size_t budget=64u*1024u*1024u;
            auto bytes=std::size_t(w)*h*192u;
            if(generation!=scene_generation||!scene_static_signature||w<1||h<1||w>1024||h>1024||
               r.right<=0||r.bottom<=0||r.left>=width||r.top>=height||bytes>budget){++unit_scene_rejections;return {};}
            auto region=std::make_shared<c3x_renderer::UnitSceneRegion>();
            unit_scene_regions.erase(std::remove_if(unit_scene_regions.begin(),unit_scene_regions.end(),[](auto const& p){return p.expired();}),unit_scene_regions.end());
            while(bytes>budget-*unit_scene_bytes&&!unit_scene_regions.empty()){
                auto old=unit_scene_regions.front().lock();unit_scene_regions.erase(unit_scene_regions.begin());
                if(!old||!old->bytes)continue;
                // Finished native fronts retain their pixels. These raw inputs
                // are a replaceable cache; eviction selects the exact compatible
                // path if the original raw map generation is no longer current.
                if(!region->base.color&&old->base.width==unsigned(w)*2&&old->base.height==unsigned(h)*2)region->base.swap(old->base);
                old->base.reset();*old->charge-=old->bytes;old->bytes=0;++unit_scene_evictions;
            }
            if(bytes>budget-*unit_scene_bytes){++unit_scene_rejections;return {};}
            if(!region->base.ensure(device,unsigned(w)*2,unsigned(h)*2,true,false))return {};
            std::vector<D3D11_RECT> outside;
            if(r.left<0)outside.push_back({0,0,-r.left,h});if(r.top<0)outside.push_back({0,0,w,-r.top});
            if(r.right>width)outside.push_back({width-r.left,0,w,h});if(r.bottom>height)outside.push_back({0,height-r.top,w,h});
            if(!scene_restore.draw(context,region->base,region_glow.linear.samples,region_glow.linear.depth_samples,
                int(region_origin_x%(width+8))-r.left-4,int(region_origin_y%(height+8))-r.top-4,outside,nullptr,
                region_glow.linear.width,region_glow.linear.height,true))return {};
            region->width=w;region->height=h;
            region->charge=unit_scene_bytes;region->bytes=bytes;*unit_scene_bytes+=bytes;unit_scene_regions.push_back(region);++unit_scene_captures;return region;
        };return source;
    }
    bool draw_scene_unit(c3x_renderer::UnitSceneRegion const& region,c3x_renderer_unit_v1 const& unit,unsigned predict,
                         c3x_renderer::UnitSceneSample& sample,int offset_x,int offset_y){
        auto& work=unit_scene_work;
        auto definition=std::find_if(unit_bodies.units.begin(),unit_bodies.units.end(),[&](auto const& model){
            return std::find(model.keys.begin(),model.keys.end(),unit.unit_key)!=model.keys.end();});
        if(definition==unit_bodies.units.end())return false;
        unsigned scale=unsigned(definition->sample_scale),projection=unit.projection_scale_milli?unit.projection_scale_milli:(unit.reduced?500:1000);
        unsigned w=unsigned(unit.sprite_width)*projection/1000,h=unsigned(unit.sprite_height)*projection/1000;
        if((scale!=1&&scale!=2&&scale!=4)||std::size_t(w)*h*scale*scale*48>96u*1024u*1024u){
            ++unit_scene_rejections;return false;
        }
        if(!work.ensure(device,w*scale,h*scale,true,false)||!scene_restore.ensure(device))return false;
        if(!scene_restore.draw(context,work,region.base.samples,region.base.depth_samples,offset_x,offset_y,{},nullptr,
            region.base.width,region.base.height,false,true,int(scale)))return false;
        if(!unit_bodies.render(device,context,unit,[&](auto const& action){return prepare_unit_action(action);},
            nullptr,predict,true,&work))return false;
        // Native composition resolves the changed samples straight into the
        // resident map/UI chain. Only bounded conversion scratch is used; no
        // finished pose cache, CPU readback, or static geometry submission.
        sample=unit_bodies.scene_sample;return true;
    }

    bool compose_scene_surface(GeometryDrawView dynamic) {
        // Failure cannot certify partially changed attachments for later reuse.
        struct Transaction {
            std::uint64_t& signature;bool complete=false;
            ~Transaction(){if(!complete)signature=0;}
        } transaction{scene_static_signature};
        // One scene-linear working set, independent of raster cache cells.
        // Sparse static backup: color/depth only; finishing targets belong to Glow.
        auto& glow=region_glow;
        unsigned w=unsigned(width)+8,h=unsigned(height)+8;
        std::size_t target_bytes=std::size_t(w)*h*240u+std::size_t(w+scene_guard_pad*2)*(h+scene_guard_pad*2)*192u;
        if(!city_profile || !c3x_renderer::render_core::scene_surface_extent(width,height) ||
           target_bytes>(world_preparation?1408u:1152u)*1024u*1024u || reflection.enabled) {
            trace.write("scene-surface-failed","bounded no-reflection view/target contract",true);return false;
        }
        if(glow.native_extent!=w || glow.native_height!=h || !glow.linear.color){
            scene_scratch.reset();scene_guard.invalidate_all();scene_dynamic_damage.clear();scene_static_signature=0;scene_overlap=false;
        }
        if(!glow.ensure(device,fidelity_root,w,h,true) || !scene_scratch.ensure(device,(w+scene_guard_pad*2)*2,(h+scene_guard_pad*2)*2,true,false) ||
           !scene_restore.ensure(device))return false;
        auto& linear=glow.linear;
        ViewportShaderSettings settings=geometry_viewport_settings;
        settings.translation[0]+=4;settings.translation[1]+=4;
        settings.inverse_size[0]=1.f/w;settings.inverse_size[1]=1.f/h;
        D3D11_RECT view={0,0,LONG(w),LONG(h)};
        D3D11_RECT selected_view=selected_output_active?D3D11_RECT{
            std::max<LONG>(0,selected_output.left),std::max<LONG>(0,selected_output.top),
            std::min<LONG>(w,selected_output.right+8),std::min<LONG>(h,selected_output.bottom+8)}:view;
        LARGE_INTEGER begin={},static_end={},dynamic_end={},finish_end={},ready={},copied={};
        QueryPerformanceCounter(&begin);
        bool restored=scene_static_signature==cached_signature.complete;
        bool translated=!restored && scene_overlap && scene_static_signature && scene_static_depth_origin==scene_depth_origin;
        std::vector<D3D11_RECT> static_rectangles=translated?scene_damage:std::vector<D3D11_RECT>{view};
        auto spans=c3x_renderer::render_core::scene_spans<D3D11_RECT>(int(w),int(h),region_origin_x,region_origin_y);
        auto physical=[&](std::vector<D3D11_RECT> const& logical){
            std::vector<D3D11_RECT> result;
            for(auto span:spans)for(auto r:logical){
                r={std::max(span.rect.left,r.left+span.x),std::max(span.rect.top,r.top+span.y),
                   std::min(span.rect.right,r.right+span.x),std::min(span.rect.bottom,r.bottom+span.y)};
                if(r.left<r.right && r.top<r.bottom)result.push_back(r);
            }return result;
        };
        unsigned batches=0,selected_static=0,selected_dynamic=0,selection_candidates=0,selection_scans=0;
        // The working attachment contains the last composed scene. The spare
        // owns only the static samples underneath its animated damage. Restore
        // those samples before accepting camera damage or a new pose.
        context->OMSetRenderTargets(0,nullptr,nullptr);
        if(!scene_guard_pad && (restored || translated)){
            if(!scene_dynamic_damage.empty() && !scene_restore.draw(context,linear,scene_scratch.samples,
                scene_scratch.depth_samples,0,0,{},&scene_dynamic_damage))return false;
            if(translated){
                auto dirty=physical(static_rectangles);
                // Clear only exposed/invalidated physical spans. All unchanged
                // static samples stay at their original world-relative address.
                for(auto rect:dirty){std::vector<D3D11_RECT> one={rect};
                    if(!scene_restore.draw(context,linear,scene_scratch.samples,
                        scene_scratch.depth_samples,0,0,one,&one))return false;}
            }
        }else if(!scene_guard_pad){
            float clear[4]={};context->ClearRenderTargetView(linear.target,clear);
            context->ClearDepthStencilView(linear.depth,D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);
        }
        std::vector<D3D11_RECT> dynamic_damage;
        for(auto layer:{geometry_shadow,geometry_feature})for(auto item:dynamic[layer]){
            auto rect=item.bounds();int dx=item.translation_x()+int(settings.translation[0]),dy=item.translation_y()+int(settings.translation[1]);
            rect={std::max<LONG>(0,rect.left+dx-4),std::max<LONG>(0,rect.top+dy-4),
                  std::min<LONG>(w,rect.right+dx+4),std::min<LONG>(h,rect.bottom+dy+4)};
            if(rect.left<rect.right && rect.top<rect.bottom)dynamic_damage.push_back(rect);
        }
        // Union damage into disjoint scan bands; no duplicate translucent draws
        // or repeated finishing. These are scissors, never miniature scenes.
        auto disjoint=[&](std::vector<D3D11_RECT> const& inputs){
            return c3x_renderer::render_core::scene_damage_union(int(w),int(h),inputs);
        };
        dynamic_damage=disjoint(physical(dynamic_damage));
        auto finish_damage=scene_dynamic_damage;
        finish_damage.insert(finish_damage.end(),dynamic_damage.begin(),dynamic_damage.end());
        char output_control[8]={};GetEnvironmentVariableA("C3X_RENDERER_INCREMENTAL_OUTPUT",output_control,sizeof(output_control));
        bool incremental_output=std::strcmp(output_control,"0")!=0;
        if(incremental_output && translated){
            auto support=c3x_renderer::render_core::scene_filter_damage(int(w),int(h),physical(static_rectangles),4);
            finish_damage.insert(finish_damage.end(),support.begin(),support.end());
        }
        // Animated bounds already carry the four-pixel lens margin above.
        // Expanding them again needlessly increases every stationary update.
        finish_damage=(restored || (incremental_output && translated))?disjoint(finish_damage):std::vector<D3D11_RECT>{view};
        // Static reuse and finished-output validity have different lifetimes.
        // Keep damage outside the selected view pending; unchanged static pixels
        // stay valid. Newly selected old poses cannot disappear from this proof.
        finish_damage.insert(finish_damage.end(),scene_pending_finish.begin(),scene_pending_finish.end());
        auto pending_finish=disjoint(finish_damage);
        if(pending_finish.size()>1024)pending_finish={view}; // bounded conservative damage
        finish_damage=pending_finish;
        if(selected_output_active){
            // Pixels outside this requested view are deliberately not published
            // with the new clock. The wider immutable donor keeps its old sample.
            auto selected_physical=physical({selected_view});
            std::vector<D3D11_RECT> clipped;
            for(auto a:finish_damage)for(auto b:selected_physical){
                D3D11_RECT r={std::max(a.left,b.left),std::max(a.top,b.top),std::min(a.right,b.right),std::min(a.bottom,b.bottom)};
                if(r.left<r.right && r.top<r.bottom)clipped.push_back(r);
            }
            finish_damage=disjoint(clipped);
        }
        // Preserve production layer/occurrence order. Batches change only when
        // selected receivers would exceed the existing 32 shadow-page slots.
        auto submit=[&](GeometryDrawView inputs,std::vector<unsigned> const& order,bool dynamic_pass){
            auto rectangles=dynamic_pass?physical({selected_view}):physical(static_rectangles);
            return submit_scene_pass(inputs,order,dynamic_pass,linear,glow,settings,w,h,rectangles,
                batches,selected_static,selected_dynamic,selection_candidates,selection_scans);
        };
        if(!scene_guard_pad && !restored){
            std::vector<c3x_renderer::city_fidelity::Lighting const*> lights;
            for(auto const& item:geometry_vertex_buffers[geometry_city])if(item.content().city_lighting){
                auto pointer=item.content().city_lighting.get();
                if(std::find(lights.begin(),lights.end(),pointer)==lights.end())lights.push_back(pointer);
            }
            if(!cities.lights(context,lights))return false;
            auto order=static_scene_order();
            if(!submit(geometry_vertex_buffers,order,false))return false;
            context->OMSetRenderTargets(0,nullptr,nullptr);

            scene_static_signature=cached_signature.complete;scene_static_depth_origin=scene_depth_origin;
        }
        if(scene_guard_pad){
            auto visible=c3x_renderer::render_core::scene_physical<D3D11_RECT>(scene_guard.width,scene_guard.height,
                region_origin_x,region_origin_y,{{scene_guard_pad,scene_guard_pad,LONG(w)+scene_guard_pad,LONG(h)+scene_guard_pad}});
            auto missing=scene_guard.select(visible);
            if(!draw_scene_guard(missing))return false;
            auto restore=scene_dynamic_damage;
            if(!restored){auto changed=translated?physical(static_rectangles):std::vector<D3D11_RECT>{view};
                restore.insert(restore.end(),changed.begin(),changed.end());}
            restore=disjoint(restore);
            for(auto transfer:c3x_renderer::render_core::scene_guard_transfers<D3D11_RECT>(int(w),int(h),scene_guard_pad,
                region_origin_x,region_origin_y,restore)){
                std::vector<D3D11_RECT> one={transfer.rect};
                if(!scene_restore.draw(context,linear,scene_scratch.samples,scene_scratch.depth_samples,
                    transfer.x,transfer.y,{},&one,scene_scratch.width,scene_scratch.height))return false;
            }
            scene_static_signature=cached_signature.complete;scene_static_depth_origin=scene_depth_origin;
        }else if(!dynamic_damage.empty() && !scene_restore.draw(context,scene_scratch,linear.samples,
            linear.depth_samples,0,0,{},&dynamic_damage))return false;
        scene_dynamic_damage=std::move(dynamic_damage);
        QueryPerformanceCounter(&static_end);
        if(!submit(dynamic,{geometry_shadow,geometry_feature},true))return false;
        QueryPerformanceCounter(&dynamic_end);
        char selection_detail[192];sprintf_s(selection_detail,"indexed_candidates=%u scanned_candidates=%u selected_static=%u selected_dynamic=%u batches=%u world_records=%zu world_bytes=%zu instances_ready=%u",
            selection_candidates,selection_scans,selected_static,selected_dynamic,batches,topology_cache.size(),topology_cache.bytes(),frame_instances_ready);
        trace.write("world-view-submission",selection_detail,true);
        sprintf_s(selection_detail,"object_builds=%u object_reuses=%u spatial_bytes=%zu spatial_view_entries=%zu affine=%u",
            frame_world_object_builds,frame_world_object_hits,world_pass_index.bytes(),world_pass_occurrences.size(),unsigned(world_pass_affine));
        trace.write("world-content",selection_detail,true);
        char instance_detail[256];sprintf_s(instance_detail,"enabled=%u mesh_bytes=%zu color_instance_bytes=%zu color_batches=%u shadow_instance_bytes=%zu shadow_batches=%u color_discards=%u shadow_discards=%u",
                unsigned(tree_instances_enabled),natural.instance_mesh_bytes,natural.instance_stream.bytes,natural.instance_stream.uploads,
                source_shadow.instance_stream.bytes,source_shadow.instance_stream.uploads,natural.instance_stream.discards,source_shadow.instance_stream.discards);
        trace.write("shared-mesh-instances",instance_detail,true);
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
        // Attribution only: explicit completion boundaries perturb overlap.
        // Never use these serialized durations as production speed evidence.
        auto completion_probe=[&](char const* stage){
            char option[8]={};GetEnvironmentVariableA("C3X_RENDERER_OUTPUT_COMPLETION_PROBE",option,sizeof(option));
            if(std::strcmp(option,"1")!=0)return true;
            ID3D11Query* query=nullptr;D3D11_QUERY_DESC desc={D3D11_QUERY_EVENT,0};
            if(FAILED(device->CreateQuery(&desc,&query)))return false;
            LARGE_INTEGER start={},end={};QueryPerformanceCounter(&start);
            context->End(query);context->Flush();HRESULT hr=S_FALSE;
            do{hr=context->GetData(query,nullptr,0,D3D11_ASYNC_GETDATA_DONOTFLUSH);
                QueryPerformanceCounter(&end);if(hr==S_FALSE)Sleep(0);
            }while(hr==S_FALSE && trace.milliseconds(end.QuadPart-start.QuadPart)<5000);
            query->Release();char detail[160];sprintf_s(detail,"boundary=%s wait_ms=%.3f completed=%u serialized=1",
                stage,trace.milliseconds(end.QuadPart-start.QuadPart),unsigned(hr==S_OK));
            trace.write("output-completion-probe",detail,true);return hr==S_OK;
        };
        if(!completion_probe("scene"))return false;
#endif
        // Keep hardware MSAA resolve: local shader resolves failed exact HDR
        // parity. Reconstruction, glow and display transfer still remain local.
        std::size_t resolved_pixels=finish_damage.empty()?0:std::size_t(w)*h*4;
        bool first=true;
        for(auto const& rect:finish_damage){
            frame_post_lanes+=glow.reconstruct(context,&rect,first,true);first=false;
            linear_output.draw(context,linear,glow.target,display_exposure,1,glow.view,w,h,&rect);
        }
        context->OMSetRenderTargets(0,nullptr,nullptr);
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
        if(!completion_probe("finish"))return false;
        char probe_option[8]={};GetEnvironmentVariableA("C3X_RENDERER_OUTPUT_COMPLETION_PROBE",probe_option,sizeof(probe_option));
        if(std::strcmp(probe_option,"1")==0){
            ID3D11Texture2D* probe=nullptr;D3D11_TEXTURE2D_DESC desc={};
            desc.Width=desc.Height=desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
            desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            if(FAILED(device->CreateTexture2D(&desc,nullptr,&probe)))return false;
            LARGE_INTEGER start={},end={};QueryPerformanceCounter(&start);
            D3D11_BOX box={0,0,0,1,1,1};context->CopySubresourceRegion(probe,0,0,0,0,glow.native,0,&box);
            D3D11_MAPPED_SUBRESOURCE data={};HRESULT hr=context->Map(probe,0,D3D11_MAP_READ,0,&data);
            if(SUCCEEDED(hr))context->Unmap(probe,0);QueryPerformanceCounter(&end);probe->Release();
            char detail[160];sprintf_s(detail,"boundary=one-pixel wait_ms=%.3f completed=%u serialized=1",
                trace.milliseconds(end.QuadPart-start.QuadPart),unsigned(SUCCEEDED(hr)));
            trace.write("output-completion-probe",detail,true);if(FAILED(hr))return false;
        }
#endif
        bool apply_visibility=visibility_pass && !visibility_coverage.tiles.empty();
        if((gpu_output_mode || apply_visibility) && !gpu_map_texture){
            D3D11_TEXTURE2D_DESC d={};d.Width=width;d.Height=height;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
            d.Format=DXGI_FORMAT_B8G8R8A8_UNORM;d.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_RENDER_TARGET;
            if(FAILED(device->CreateTexture2D(&d,nullptr,&gpu_map_texture)))return false;
        }
        std::vector<D3D11_RECT> copies;
        // Scrolling remaps the retained physical image into the exact current
        // camera bitmap. Transfer is cheap; keep one full readback on view changes
        // rather than adding a staging atlas and CPU bitmap translation.
        auto copy_damage=(gpu_output_mode || apply_visibility || cpu_output_stale)?physical({view}):selected_output_active?physical({selected_view}):restored?finish_damage:std::vector<D3D11_RECT>{view};
        for(auto span:spans)for(auto damage:copy_damage){
            D3D11_RECT rect={std::max<LONG>(span.rect.left,damage.left),std::max<LONG>(span.rect.top,damage.top),
                             std::min<LONG>(span.rect.right,damage.right),std::min<LONG>(span.rect.bottom,damage.bottom)};
            rect={std::max<LONG>(rect.left,span.x+4),std::max<LONG>(rect.top,span.y+4),
                  std::min<LONG>(rect.right,span.x+width+4),std::min<LONG>(rect.bottom,span.y+height+4)};
            if(rect.left>=rect.right || rect.top>=rect.bottom)continue;
            D3D11_BOX box={UINT(rect.left),UINT(rect.top),0,UINT(rect.right),UINT(rect.bottom),1};
            context->CopySubresourceRegion((gpu_output_mode || apply_visibility)?gpu_map_texture:readback_texture,0,rect.left-span.x-4,rect.top-span.y-4,0,glow.native,0,&box);
            copies.push_back({rect.left-span.x-4,rect.top-span.y-4,rect.right-span.x-4,rect.bottom-span.y-4});
        }
        if(apply_visibility){
            LARGE_INTEGER coverage_begin={},end={};QueryPerformanceCounter(&coverage_begin);
            if(!visibility_gpu.apply(device,context,gpu_map_texture,visibility_coverage))return false;
            if(!gpu_output_mode){D3D11_BOX box={0,0,0,UINT(width),UINT(height),1};
                context->CopySubresourceRegion(readback_texture,0,0,0,0,gpu_map_texture,0,&box);copies={{0,0,width,height}};}
            QueryPerformanceCounter(&end);char detail[192];sprintf_s(detail,"tiles=%zu gpu=1 upload_bytes=%zu gpu_bytes=%zu submit_ms=%.3f",
                visibility_coverage.tiles.size(),visibility_coverage.tiles.size()*sizeof(c3x_renderer::render_core::VisibilityCoverage::Tile),visibility_gpu.bytes(),trace.milliseconds(end.QuadPart-coverage_begin.QuadPart));
            trace.write("visibility-pass",detail,true);
        }
        QueryPerformanceCounter(&finish_end);
        D3D11_MAPPED_SUBRESOURCE mapped={};
        if(!gpu_output_mode && !copies.empty())++frame_output_readbacks;
        if(!gpu_output_mode && !copies.empty() && FAILED(context->Map(readback_texture,0,D3D11_MAP_READ,0,&mapped)))return false;
        QueryPerformanceCounter(&ready);
        // One persistent composed bitmap also covers animation removal. Output
        // pointer selection must not resurrect a pre-animation CPU bitmap.
        auto& output=pixels;output.resize(std::size_t(width)*height);
        if(!gpu_output_mode)for(auto rect:copies)for(int y=rect.top;y<rect.bottom;++y)std::memcpy(output.data()+std::size_t(y)*width+rect.left,
            static_cast<unsigned char*>(mapped.pData)+std::size_t(y)*mapped.RowPitch+rect.left*4,std::size_t(rect.right-rect.left)*4);
        if(!gpu_output_mode && !copies.empty())context->Unmap(readback_texture,0);
        if(gpu_output_mode){gpu_map_valid=true;cpu_output_stale=true;}
        else cpu_output_stale=false; // the complete CPU image is authoritative again
        QueryPerformanceCounter(&copied);
        char detail[768];sprintf_s(detail,"static_reused=%u translated=%u damage_rects=%zu static_selected=%u dynamic_selected=%u batches=%u target_bytes=%zu target_cap=%zu resolves=%u readbacks=%u full_surface_copies=0 dynamic_damage_rects=%zu copied_rects=%zu static_submit_ms=%.3f dynamic_submit_ms=%.3f finish_submit_ms=%.3f completion_wait_ms=%.3f cpu_copy_ms=%.3f",
            unsigned(restored),unsigned(translated),static_rectangles.size(),selected_static,selected_dynamic,batches,target_bytes,std::size_t(world_preparation?1408u:1152u)*1024u*1024u,unsigned(!finish_damage.empty()),unsigned(!gpu_output_mode && !copies.empty()),scene_dynamic_damage.size(),copies.size(),
            trace.milliseconds(static_end.QuadPart-begin.QuadPart),trace.milliseconds(dynamic_end.QuadPart-static_end.QuadPart),
            trace.milliseconds(finish_end.QuadPart-dynamic_end.QuadPart),trace.milliseconds(ready.QuadPart-finish_end.QuadPart),trace.milliseconds(copied.QuadPart-ready.QuadPart));
        trace.write("shared-scene-surface",detail,true);memory_sample("shared-scene-complete");
        sprintf_s(detail,"content_uploads=%u setups=%llu layers=%llu draws=%llu parameter_updates=%llu bounds_tests=%llu parameter_uploads=%u parameter_records=%u issue_ms=%.3f selection_ms=%.3f execute_ms=%.3f prepared_meshes=%u foreground_meshes=%u prepared_vertex_bytes=%zu",
            frame_content_uploads,frame_pass_setups,frame_active_layers,frame_draw_calls,frame_parameter_updates,frame_bounds_tests,draw_parameters.uploads,draw_parameters.records,
            frame_geometry_issue_ms,frame_scene_select_ms,frame_scene_execute_ms,frame_prepared_meshes,frame_foreground_meshes,frame_prepared_vertex_bytes);
        trace.write("selected-pass-submission",detail,true);
        sprintf_s(detail,"incremental=%u resolve_pixels=%zu finish_rects=%zu readback_rects=%zu",
            unsigned(incremental_output),resolved_pixels,finish_damage.size(),copies.size());
        trace.write("scene-output",detail,true);
        // All pending damage inside the selected spans has now been finished.
        // Retain the complement until that output is requested, not whole views.
        for(auto selected_span:physical({selected_view})){
            std::vector<D3D11_RECT> remaining;
            for(auto r:pending_finish){
                D3D11_RECT overlap={std::max(r.left,selected_span.left),std::max(r.top,selected_span.top),
                    std::min(r.right,selected_span.right),std::min(r.bottom,selected_span.bottom)};
                if(overlap.left>=overlap.right || overlap.top>=overlap.bottom){remaining.push_back(r);continue;}
                for(auto piece:std::array<D3D11_RECT,4>{{{r.left,r.top,r.right,overlap.top},{r.left,overlap.bottom,r.right,r.bottom},
                    {r.left,overlap.top,overlap.left,overlap.bottom},{overlap.right,overlap.top,r.right,overlap.bottom}}})
                    if(piece.left<piece.right && piece.top<piece.bottom)remaining.push_back(piece);
            }
            pending_finish=disjoint(remaining);
        }
        if(pending_finish.size()>1024)pending_finish={view};
        scene_pending_finish=std::move(pending_finish);
        transaction.complete=true;return true;
    }

    bool fill_output(c3x_renderer_frame_v1 const & frame,
                     c3x_renderer_output_v1 & output, c3x_renderer_u32 invalidations,
                     c3x_renderer_i64 renderer_ticks) {
        output.api_version = C3X_RENDERER_API_VERSION;
        output.struct_size = sizeof(output);
        output.width = width;
        output.height = height;
        output.stride_bytes = width * static_cast<int>(sizeof(std::uint32_t));
        output.clip_left = frame.clip_left;
        output.clip_top = frame.clip_top;
        output.clip_right = frame.clip_right;
        output.clip_bottom = frame.clip_bottom;
        output.rendered_tile_count = cached_rendered_tile_count;
        output.fallback_tile_count = cached_fallback_tile_count;
        if (!compose_resource_animations(frame)) return false;
        if(world_regions){
            char detail[320];sprintf_s(detail,"hits=%zu misses=%zu hit_pixels=%zu gpu_bytes=%zu metadata_bytes=%zu entries=%zu rejected=%zu evictions=%llu metadata_cap=%zu local_revisions=%u",
                frame_region_hits,frame_region_misses,frame_region_hit_pixels,render_regions.gpu_bytes,render_regions.metadata_bytes,
                render_regions.entries.size(),frame_region_rejected,render_regions.evictions,render_regions.metadata_limit,unsigned(local_region_revisions));trace.write("render-region-cache",detail,false);
        }
        if(world_regions){
            char detail[320];sprintf_s(detail,"contributors_ms=%.3f lights_ms=%.3f shadows_ms=%.3f tile_validation_ms=%.3f tile_append_ms=%.3f topology_ms=%.3f shadow_proof_hits=%llu shadow_proof_misses=%llu shadow_proof_bytes=%zu contributor_index_bytes=%zu",
                frame_region_phase_ms[0],frame_region_phase_ms[1],frame_region_phase_ms[2],frame_tile_validation_ms,frame_tile_append_ms,frame_topology_ms,
                retained_region_casters.receiver_hits,retained_region_casters.receiver_misses,retained_region_casters.receiver_bytes,region_contributors.bytes);
            trace.write("navigation-phases",detail,false);
            sprintf_s(detail,"ms=%.3f hits=%zu misses=%zu bytes=%zu entries=%zu",frame_center_shore_ms,
                center_shore_cache.hits,center_shore_cache.misses,center_shore_cache.bytes,center_shore_cache.entries.size());
            trace.write("center-shore-cache",detail,false);
        }
        output.bgra_pixels = gpu_output_mode ? nullptr : posed_count() && !shared_scene_surface ? resource_pixels.data() : pixels.data();
        if(visibility_pass && !shared_scene_surface){
            LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
            if(gpu_output_mode){if(!visibility_gpu.apply(device,context,gpu_map_texture,visibility_coverage))return false;}
            else if(!visibility_coverage.tiles.empty()){visibility_coverage.apply(static_cast<std::uint32_t const*>(output.bgra_pixels),visibility_pixels);output.bgra_pixels=visibility_pixels.data();}
            QueryPerformanceCounter(&end);char detail[256];sprintf_s(detail,"tiles=%zu gpu=%u gpu_bytes=%zu cpu_bytes=%zu submit_ms=%.3f",
                visibility_coverage.tiles.size(),unsigned(gpu_output_mode),visibility_gpu.bytes(),visibility_pixels.capacity()*4,trace.milliseconds(end.QuadPart-begin.QuadPart));
            trace.write("visibility-pass",detail,true);
        }
        // Terrain is independent of retained native unit/effect animation.  A
        // cache hit must still report the current frame's animation demand so
        // Civ III keeps driving those overlay planes without rerendering the
        // static map underneath them.
        output.visible_animation_count = frame.visible_animation_count + ambient_count();
        output.request_continuous_redraw = output.visible_animation_count != 0;
        output.renderer_cpu_ticks = renderer_ticks + resource_composite_ticks;
        output.textured_tile_count = cached_textured_tile_count;
        output.fallback_tile_indices = fallback_tile_indices.empty() ? nullptr : fallback_tile_indices.data();
        output.replacement_tile_flags = replacement_tile_flags.empty() ? nullptr : replacement_tile_flags.data();
        output.replacement_tile_count = static_cast<c3x_renderer_u32>(replacement_tile_flags.size());
        output.frame_invalidation_flags = invalidations;
        output.cache_hits = cache_hits;
        output.cache_misses = cache_misses;
        output.cache_evictions = cache_evictions;
        output.cache_stale_rejections = cache_stale_rejections;
        output.cache_entries = static_cast<c3x_renderer_u32>(viewport_cache.size());
        output.cache_capacity = viewport_cache_capacity;
        output.device_generation = device_generation;
        output.device_recoveries = device_recoveries;
        output.content_revision = static_cast<c3x_renderer_i64>(content_revision);
        output.geometry_tiles_built = frame_tiles_built;
        output.geometry_tiles_reused = frame_tiles_reused;
        output.geometry_tiles_evicted = frame_tiles_evicted;
        output.geometry_cache_bytes = static_cast<c3x_renderer_u32>(tile_geometry_cache_bytes);
        output.geometry_upload_bytes = static_cast<c3x_renderer_u32>(frame_upload_bytes);
        output.geometry_ticks = frame_geometry_ticks;
        output.draw_ticks = frame_draw_ticks;
        output.readback_ticks = frame_readback_ticks;
        output.raster_reused_pixels = raster_reused_pixels;
        output.raster_draw_pixels = raster_draw_pixels;
        output.raster_cached_pixels = raster_cached_pixels;
        if (trace.level) {
            char detail[640];
            std::snprintf(detail, sizeof(detail),
                "cache=%s invalidations=%u camera=%u scene=%u environment=%u wrap=%u content=%u ownership=%u device=%u "
                "tiles=%u built=%u reused=%u evicted=%u gpu_bytes=%zu upload_bytes=%zu viewport_bytes=%zu "
                "reused_pixels=%u draw_pixels=%u block_pixels=%u block_bytes=%zu render_ms=%.3f geometry_ms=%.3f draw_submit_ms=%.3f readback_wait_ms=%.3f post_lanes=%zu",
                frame_cache_path, invalidations, (invalidations & 1u) != 0, (invalidations & 2u) != 0,
                (invalidations & 4u) != 0, (invalidations & 8u) != 0, (invalidations & 16u) != 0,
                (invalidations & 32u) != 0, (invalidations & 64u) != 0, frame.tile_count,
                frame_tiles_built, frame_tiles_reused, frame_tiles_evicted,
                tile_geometry_cache_bytes, frame_upload_bytes, viewport_cache_bytes,
                raster_reused_pixels, raster_draw_pixels, raster_cached_pixels, pixel_blocks.bytes, trace.milliseconds(renderer_ticks), trace.milliseconds(frame_geometry_ticks),
                trace.milliseconds(frame_draw_ticks), trace.milliseconds(frame_readback_ticks),frame_post_lanes);
            trace.write("frame", detail, invalidations != 0);
        }
        return true;
    }

    c3x_renderer_u32 invalidations_for(c3x_renderer::TerrainFrameSignature const & signature) const {
        if (!cache_valid)
            return C3X_RENDERER_INVALIDATE_ALL;
        c3x_renderer_u32 flags = 0;
        if (signature.camera != cached_signature.camera)
            flags |= C3X_RENDERER_INVALIDATE_CAMERA;
        if (signature.scene != cached_signature.scene)
            flags |= C3X_RENDERER_INVALIDATE_SCENE;
        if (signature.environment != cached_signature.environment)
            flags |= C3X_RENDERER_INVALIDATE_ENVIRONMENT;
        if (signature.wrap != cached_signature.wrap)
            flags |= C3X_RENDERER_INVALIDATE_WRAP;
        if (signature.ownership != cached_signature.ownership)
            flags |= C3X_RENDERER_INVALIDATE_OWNERSHIP;
        if (content_revision != previous_content_revision)
            flags |= C3X_RENDERER_INVALIDATE_PACK_DEFINITION;
        return flags;
    }

    void release_geometry_vertex_buffers(
        std::array<std::vector<CachedVertexChunk>, geometry_layer_count> & buffers) {
        for (std::vector<CachedVertexChunk> & layer : buffers) {
            for (CachedVertexChunk & chunk : layer) {
                release(chunk.buffer);
                release(chunk.indices);
            }
            std::vector<CachedVertexChunk>().swap(layer);
        }
    }

    void clear_geometry_vertex_buffers() {
        region_contributors.clear();
        resource_anchors.clear();
        geometry_footprints.clear();
        for(auto& layer:geometry_vertex_buffers)std::vector<GeometryDrawRecord>().swap(layer);
    }

    // An abandoned view has no complete contributor set. Keep compiled world
    // content, but never let idle preparation certify pixels from that view.
    void discard_scene_view() {
        cache_valid=false;resource_pixel_signature=0;
        geometry_cache.clear();
        clear_geometry_vertex_buffers();
        scene_guard.invalidate_all();
        scene_static_signature=0;
    }

    void release_resident_content(CachedTileGeometry& owner){
        if(owner.shared_natural)world_pass_index.erase(owner.version);
        resident_content.release(owner.binding);
    }

    void clear_tile_geometry_cache() {
        natural_mesh_cache.clear();natural_mesh_cache_bytes=0;
        ground_grid_cache.clear();ground_grid_cache_bytes=0;
        topology_cache = {};
        resident_content.clear();world_pass_index.clear();world_pass_occurrences.clear();world_pass_affine=false;
        cancel_pixel_preparation();
        pixel_blocks.clear();
        bitmap_footprints.clear();
        for (auto & entry : tile_geometry_cache)
            release_geometry_vertex_buffers(entry.second.buffers);
        tile_geometry_cache.clear();
        for(auto&entry:terrain_patch_indices)release(entry.second);
        terrain_patch_indices.clear();terrain_patch_index_bytes=0;
        tile_geometry_cache_bytes = 0;
        prefetched_geometry_bytes = 0;
    }

    bool make_tile_cache_room(std::size_t bytes) {
        while (tile_geometry_cache_bytes + terrain_patch_index_bytes + bytes > tile_geometry_runtime_budget ||
               tile_geometry_cache.size() >= tile_geometry_cache_capacity) {
            auto oldest = tile_geometry_cache.end();
            auto animation_priority = [&](CachedTileGeometry const& tile) {
                // Static revisits can use their retained bitmap without meshes.
                // Animated revisits still need geometry for depth/lighting.
                // This is a bounded preference, never an additional pin/cap.
                return tile.animation_epoch != 0 &&
                    tile_geometry_epoch-tile.animation_epoch <= viewport_cache_capacity;
            };
            for (auto it = tile_geometry_cache.begin(); it != tile_geometry_cache.end(); ++it) {
                if (it->second.last_used == tile_geometry_epoch)
                    continue; // active frame references this immutable storage
                if (oldest == tile_geometry_cache.end() ||
                    animation_priority(it->second) < animation_priority(oldest->second) ||
                    (animation_priority(it->second) == animation_priority(oldest->second) &&
                     it->second.last_used < oldest->second.last_used))
                    oldest = it;
            }
            if (oldest == tile_geometry_cache.end()) {
                std::size_t resident=0;
                for(auto const& entry:tile_geometry_cache)resident+=entry.second.byte_count;
                char detail[256];sprintf_s(detail,"requested=%zu tracked=%zu resident=%zu pending=%zu entries=%zu epoch=%llu",
                    bytes,tile_geometry_cache_bytes,resident,tile_geometry_cache_bytes-resident,tile_geometry_cache.size(),
                    static_cast<unsigned long long>(tile_geometry_epoch));
                trace.write("tile-cache-budget",detail,true);
                return false;
            }
            tile_geometry_cache_bytes -= oldest->second.byte_count;
            if (oldest->second.prefetched) prefetched_geometry_bytes -= oldest->second.byte_count;
            release_geometry_vertex_buffers(oldest->second.buffers);
            release_resident_content(oldest->second);tile_geometry_cache.erase(oldest);
            ++frame_tiles_evicted;
            if (cache_evictions != 0xffffffffu) ++cache_evictions;
        }
        return true;
    }

    bool cache_geometry_layer(c3x_renderer::render_core::ImmutableMeshUpload& upload,
                              std::vector<Vertex> & vertices,
                              std::vector<CachedVertexChunk> & output,
                              bool prefetch = false, std::size_t pending_bytes = 0,
                              std::atomic<bool> const * foreground_pending = nullptr, bool compact_feature = false, bool natural_vertex = false,
                              NaturalMesh* record=nullptr,NaturalMesh const* cached=nullptr,
                              c3x_renderer::fidelity::GroundProjection const* projection=nullptr,
                              std::vector<UINT> const* grid_indices=nullptr, unsigned projection_kind=0, c3x_renderer::render_core::PreparedMesh const* prepared=nullptr,
                              ID3D11Buffer* prepared_buffer=nullptr, unsigned prepared_vertex_offset=0, bool prepared_indices=false, unsigned prepared_index_offset=0) {
        if(vertices.empty() && (!cached || cached->vertices.empty()) && (!prepared || prepared->empty()))return true;
        c3x_renderer::render_core::PreparedMesh local;
        auto cancelled=[&]{return prefetch && foreground_pending->load(std::memory_order_relaxed);};
        std::vector<Vertex> packed;std::vector<UINT> cached_indices;
        if(cached) {
            cached_indices=cached->indices;packed.resize(cached->vertices.size());
            for(std::size_t i=0;i<packed.size();++i){auto const& p=cached->vertices[i];auto& v=packed[i];
                if(prefetch && (i&255u)==0 && foreground_pending->load(std::memory_order_relaxed))return false;
                v.x=p[0];v.y=p[1];v.z=p[2];v.world_x=p[3];v.world_y=p[4];v.world_z=p[5];v.world_valid=p[6];
                v.normal_x=p[7];v.normal_y=p[8];v.normal_z=p[9];v.u=p[10];v.v=p[11];
                v.material_grass=p[12];v.material_plains=p[13];v.material_desert=p[14];v.material_marsh=p[15];
                v.authored_relief_height=p[16];v.authored_relief_blend=p[17];v.base_terrain=p[18];
                v.relief_owner_u=p[19];v.relief_owner_v=p[20];v.relief_owner_coverage=p[21];v.relief_owner_state=p[22];
                if(projection){auto projected=(*projection)(v.world_x,v.world_y,v.world_z*112.f);
                    v.x=projected.x;v.y=projected.y;v.z=projected.z;}
            }

        }
        if(!prepared){
            c3x_renderer::render_core::MeshFormat format;
            format.pickup=pickup_profile;format.feature=compact_feature;format.natural=natural_vertex;
            format.projection_kind=projection_kind;
            auto topology=cached?&cached_indices:grid_indices;
            auto const& input=cached?packed:vertices;
            if(natural_vertex && grid_indices)format.shared_grid=c3x_renderer::render_core::shared_mesh_grid(input.size(),topology,patch_layouts);
            if(!c3x_renderer::render_core::prepare_mesh(input,topology,format,local,cancelled))return false;
            prepared=&local;++frame_foreground_meshes;
        }else{++frame_prepared_meshes;frame_prepared_vertex_bytes+=prepared->vertices.size();}
        auto const& mesh=*prepared;
        if(mesh.empty())return true;
        CachedVertexChunk chunk;
        chunk.projection_kind=projection_kind;chunk.source_tile_width=(projection_kind==2 || projection_kind==3)?128:shadow_tile_width;
        chunk.version=tile_geometry_version;chunk.vertex_stride=mesh.vertex_stride;
        chunk.bounds={mesh.bounds[0],mesh.bounds[1],mesh.bounds[2],mesh.bounds[3]};
        std::copy(mesh.world_low.begin(),mesh.world_low.end(),chunk.world_bounds.low);
        std::copy(mesh.world_high.begin(),mesh.world_high.end(),chunk.world_bounds.high);
        chunk.projected_bounds=mesh.projected_bounds;
        chunk.index_count=mesh.index_count;chunk.index_format=mesh.index_stride==2?DXGI_FORMAT_R16_UINT:DXGI_FORMAT_R32_UINT;
        unsigned shared_grid=mesh.shared_grid;
        std::size_t index_bytes=mesh.indices.size();
        chunk.byte_count=mesh.vertices.size()+(shared_grid && !prepared_indices?0:index_bytes);
        while (prefetch && prefetched_geometry_bytes + pending_bytes + chunk.byte_count + 3u > 64u*1024u*1024u) {
            auto oldest = tile_geometry_cache.end();
            for (auto it = tile_geometry_cache.begin(); it != tile_geometry_cache.end(); ++it)
                if (it->second.prefetched && it->second.last_used < tile_geometry_epoch-1 &&
                    (oldest == tile_geometry_cache.end() || it->second.last_used < oldest->second.last_used)) oldest = it;
            if (oldest == tile_geometry_cache.end()) return false;
            prefetched_geometry_bytes -= oldest->second.byte_count;
            tile_geometry_cache_bytes -= oldest->second.byte_count;
            release_geometry_vertex_buffers(oldest->second.buffers);
            release_resident_content(oldest->second);tile_geometry_cache.erase(oldest);
            ++frame_tiles_evicted;
            if (cache_evictions != 0xffffffffu) ++cache_evictions;
        }
        if (!make_tile_cache_room(chunk.byte_count+3u)) return false;
        if (prefetch && foreground_pending->load(std::memory_order_relaxed)) return false;
        if(profiling && sampled_geometry_bucket!=tile_geometry_cache_bytes/(32u*1024u*1024u)){
            sampled_geometry_bucket=tile_geometry_cache_bytes/(32u*1024u*1024u);
            char detail[192];sprintf_s(detail,"gpu_request=%zu packed_bytes=%zu index_bytes=%zu prepared=%u",
                chunk.byte_count,mesh.vertices.size(),mesh.indices.size(),unsigned(prepared!=&local));
            trace.write("geometry-allocation",detail,true);
            memory_sample("before-geometry-allocation");
        }
        output.reserve(output.size()+1); // allocate before acquiring COM resources
        if(record){
            record->vertices.resize(mesh.vertices.size()/92u);
            std::memcpy(record->vertices.data(),mesh.vertices.data(),mesh.vertices.size());
            record->indices.resize(mesh.index_count);
            for(unsigned i=0;i<mesh.index_count;++i){
                if(mesh.index_stride==2){std::uint16_t value;std::memcpy(&value,mesh.indices.data()+i*2,2);record->indices[i]=value;}
                else std::memcpy(&record->indices[i],mesh.indices.data()+i*4,4);
            }
        }
        auto before=upload.size();
        if(prepared_buffer){
            // This layer's vertex bytes already live in a buffer created when
            // the content was compiled (worker or foreground fallback); adopt
            // it directly instead of re-copying into the per-tile upload.
            chunk.buffer=prepared_buffer;prepared_buffer->AddRef();
            chunk.vertex_offset=prepared_vertex_offset;
        }else chunk.vertex_offset=upload.append(mesh.vertices.data(),mesh.vertices.size());
        auto shared=terrain_patch_indices.find(shared_grid);
        if(prepared_indices){
            chunk.indices=prepared_buffer;prepared_buffer->AddRef();chunk.index_offset=prepared_index_offset;
        }else if(shared_grid && shared!=terrain_patch_indices.end()){
            chunk.indices=shared->second;chunk.indices->AddRef();++frame_patch_index_reuses;
        }else if(shared_grid){
            if(terrain_patch_index_bytes+index_bytes>4u*1024u*1024u || !make_tile_cache_room(chunk.byte_count+index_bytes))return false;
            D3D11_BUFFER_DESC desc={};desc.ByteWidth=static_cast<UINT>(index_bytes);
            desc.Usage=D3D11_USAGE_IMMUTABLE;desc.BindFlags=D3D11_BIND_INDEX_BUFFER;
            D3D11_SUBRESOURCE_DATA initial={};initial.pSysMem=mesh.indices.data();
            if(FAILED(device->CreateBuffer(&desc,&initial,&chunk.indices)))return false;
            try{terrain_patch_indices.emplace(shared_grid,chunk.indices);}catch(...){release(chunk.indices);throw;}
            chunk.indices->AddRef();terrain_patch_index_bytes+=index_bytes;frame_upload_bytes+=index_bytes;
        }else chunk.index_offset=upload.append(mesh.indices.data(),index_bytes);
        // byte_count tracks true resident bytes for cache/eviction accounting
        // regardless of upload path; frame_upload_bytes tracks only bytes this
        // call actually copied into the foreground per-tile upload buffer.
        auto foreground_bytes=upload.size()-before;
        chunk.byte_count=foreground_bytes+(prepared_buffer?mesh.vertices.size():0)+(prepared_indices?mesh.indices.size()+3u:0);
        tile_geometry_cache_bytes+=chunk.byte_count;frame_upload_bytes+=foreground_bytes;
        output.push_back(chunk);vertices.clear();return true;
    }

    GeometryDrawRecord project_natural_chunk(GeometryDrawRecord chunk, c3x_renderer_tile_v1 const& record) {
        int c=(record.tile_x+record.tile_y)/2,r=(record.tile_x-record.tile_y)/2;
        if(chunk.content().projection_kind==2 || chunk.content().projection_kind==3){
            float scale=float(shadow_tile_width)/float(chunk.content().source_tile_width);
            chunk.bounds={LONG(std::floor(chunk.bounds.left*scale))-2,LONG(std::floor(chunk.bounds.top*scale))-2,
                LONG(std::ceil(chunk.bounds.right*scale))+2,LONG(std::ceil(chunk.bounds.bottom*scale))+2};
            chunk.natural_projection[0]=float(c);chunk.natural_projection[1]=float(r);
            chunk.natural_projection[2]=float(shadow_tile_width);chunk.natural_projection[3]=float(content_view_height);
            return chunk;
        }
        chunk.natural_projection[0]=float(c);chunk.natural_projection[1]=float(r);
        chunk.natural_projection[2]=float(shadow_tile_width);chunk.natural_projection[3]=float(chunk.content().projection_kind==4?content_view_height:height);
        if(tight_natural_bounds && chunk.content().projected_bounds.valid && shadow_tile_height*2==shadow_tile_width){
            auto bounds=chunk.content().projected_bounds.project(c,r,shadow_tile_width);
            chunk.bounds={bounds[0],bounds[1],bounds[2],bounds[3]};return chunk;
        }
        c3x_renderer::fidelity::GroundProjection projection{c,r,shadow_tile_width*.5f,
            shadow_tile_height*.5f,shadow_tile_width/224.f*.82f,float(height)};
        chunk.bounds={LONG_MAX,LONG_MAX,LONG_MIN,LONG_MIN};
        for(unsigned corner=0;corner<8;++corner){
            float p[3];for(unsigned axis=0;axis<3;++axis)p[axis]=(corner&(1u<<axis))?
                chunk.content().world_bounds.high[axis]:chunk.content().world_bounds.low[axis];
            auto v=projection(p[0],p[1],p[2]*112.f);
            chunk.bounds.left=std::min(chunk.bounds.left,LONG(std::floor(v.x))-2);
            chunk.bounds.top=std::min(chunk.bounds.top,LONG(std::floor(v.y))-2);
            chunk.bounds.right=std::max(chunk.bounds.right,LONG(std::ceil(v.x))+2);
            chunk.bounds.bottom=std::max(chunk.bounds.bottom,LONG(std::ceil(v.y))+2);
        }
        return chunk;
    }

    void index_world_content(CachedTileGeometry const& owner){
        if(!owner.world_objects)return;
        c3x_renderer_tile_v1 record={};record.tile_x=owner.tile_x;record.tile_y=owner.tile_y;
        for(auto const& layer:owner.buffers)for(auto const& source:layer){
            auto key=reinterpret_cast<std::uintptr_t>(&source);
            if(world_pass_index.contains(key))continue;
            auto draw=project_natural_chunk(GeometryDrawRecord(source),record);
            double radius=source.city_lighting?1.25:.125; // lighting plus fixed filter/bounds guard
            double scale=1./shadow_tile_width,x=owner.tile_x*.5,y=owner.tile_y*.25;
            if(!world_pass_index.add(owner.version,key,x+draw.bounds.left*scale-radius,y+draw.bounds.top*scale-radius,
                x+draw.bounds.right*scale+radius,y+draw.bounds.bottom*scale+radius))return;
        }
    }

    c3x_renderer::TileFootprint tile_footprint(CachedTileGeometry const & tile,
                                             c3x_renderer_tile_v1 const & record) {
        int anchor_x = record.anchor_x, anchor_y = record.anchor_y;
        c3x_renderer::TileFootprint footprint;
        footprint.coordinate = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(record.tile_x)) << 32) |
            static_cast<std::uint32_t>(record.tile_y);
        footprint.mesh = tile.version;
        footprint.anchor_x = anchor_x; footprint.anchor_y = anchor_y;
        footprint.bounds = {INT_MAX, INT_MAX, INT_MIN, INT_MIN};
        auto include=[&](auto const& buffers,bool natural_world){
        for (auto const & layer : buffers)
            for (auto const & source_chunk : layer) {
                GeometryDrawRecord chunk=(natural_world || source_chunk.projection_kind)?project_natural_chunk(source_chunk,record):GeometryDrawRecord(source_chunk);
                footprint.bounds.left = std::min(footprint.bounds.left, static_cast<int>(chunk.bounds.left));
                footprint.bounds.top = std::min(footprint.bounds.top, static_cast<int>(chunk.bounds.top));
                footprint.bounds.right = std::max(footprint.bounds.right, static_cast<int>(chunk.bounds.right));
                footprint.bounds.bottom = std::max(footprint.bounds.bottom, static_cast<int>(chunk.bounds.bottom));
                if(chunk.content().city_lighting){
                    int radius=int(std::ceil(shadow_tile_width*.85f))+8;
                    footprint.bounds.left=std::min(footprint.bounds.left,int(chunk.bounds.left)-radius);
                    footprint.bounds.right=std::max(footprint.bounds.right,int(chunk.bounds.right)+radius);
                    footprint.bounds.top=std::min(footprint.bounds.top,int(chunk.bounds.top)-radius);
                    footprint.bounds.bottom=std::max(footprint.bounds.bottom,int(chunk.bounds.bottom)+radius);
                }
                if(environment_profile){
                    int shift=int(std::ceil(2*reflection.height_pixels*std::max(0.f,chunk.content().world_bounds.high[2]-2.5f/112.f)))+4;
                    footprint.bounds.bottom=std::max(footprint.bounds.bottom,int(chunk.bounds.bottom)+shift);
                    footprint.bounds.left=std::min(footprint.bounds.left,int(chunk.bounds.left)-4);
                    footprint.bounds.right=std::max(footprint.bounds.right,int(chunk.bounds.right)+4);
                }
                if(pickup_profile) {
                    float z=std::max(0.f,chunk.content().world_bounds.high[2]);
                    float u=-shadow_basis[8]/shadow_basis[10]*z,v=-shadow_basis[9]/shadow_basis[10]*z;
                    int dx=int(std::ceil(std::abs((u+v)*shadow_tile_width*.5f)))+3;
                    int dy=int(std::ceil(std::abs((u-v)*shadow_tile_height*.5f)+z*112*.82f*shadow_tile_width/224))+3;
                    footprint.bounds.left=std::min(footprint.bounds.left,int(chunk.bounds.left)-dx);
                    footprint.bounds.right=std::max(footprint.bounds.right,int(chunk.bounds.right)+dx);
                    footprint.bounds.top=std::min(footprint.bounds.top,int(chunk.bounds.top)-dy);
                    footprint.bounds.bottom=std::max(footprint.bounds.bottom,int(chunk.bounds.bottom)+dy);
                }
            }
        };
        include(tile.buffers,false);
        if(auto natural_tile=resident_content.resolve(tile.natural_content))
            if(natural_tile->shared_natural)include(natural_tile->buffers,true);
        return footprint;
    }

    void append_tile_geometry(CachedTileGeometry & tile, c3x_renderer_tile_v1 const & record, bool animated_view) {
        if (tile.prefetched) {
            tile.prefetched = false;
            prefetched_geometry_bytes -= tile.byte_count;
        }
        int anchor_x = record.anchor_x, anchor_y = record.anchor_y;
        if (record.tile_flags & C3X_RENDERER_TILE_RENDER)
            for (auto anchor : tile.resource_anchors) {
                anchor.anchor_x += anchor_x; anchor.anchor_y += anchor_y;
                anchor.tile_x=record.tile_x;anchor.tile_y=record.tile_y;
                resource_anchors.push_back(anchor);
            }
        if (tile.byte_count != 0) geometry_footprints.push_back(tile_footprint(tile, record));
        tile.last_used = tile_geometry_epoch;
        if(animated_view)tile.animation_epoch=tile_geometry_epoch;
        auto append=[&](CachedTileGeometry& source,bool natural_world){
        source.last_used=tile_geometry_epoch;
        if(animated_view)source.animation_epoch=tile_geometry_epoch;
        for (std::size_t layer = 0; layer < geometry_layer_count; ++layer) {
            // Let push_back grow geometrically. Reserving exactly one tile at
            // a time copied the entire layer again for every visible tile.
            for (auto const& source_chunk : source.buffers[layer]) {
                GeometryDrawRecord chunk(source_chunk);
                if(natural_world || source_chunk.projection_kind)chunk=project_natural_chunk(chunk,record);
                chunk.translation_x = anchor_x - source.anchor_x;
                chunk.translation_y = anchor_y - source.anchor_y;
                geometry_vertex_buffers[layer].push_back(chunk);
                // The current epoch protects this immutable owner from eviction.
                // Draw records acquire no buffer or material references.
            }
        }
        };
        append(tile,false);
        if(auto natural_tile=resident_content.resolve(tile.natural_content))
            if(natural_tile->shared_natural){index_world_content(*natural_tile);append(*natural_tile,true);}
        topology_cache.attach(record,tile.binding);
    }

    c3x_renderer::fidelity::TerrainCompileInput terrain_compile_input(c3x_renderer_tile_v1 const& tile,
            c3x_renderer_frame_v1 const& frame,int ground,bool skip_shore,bool separate,bool indexed,bool heights,bool world_content=false) const {
        c3x_renderer::fidelity::TerrainCompileInput input;
        input.tile_x=tile.tile_x;input.tile_y=tile.tile_y;input.real_terrain_type=tile.real_terrain_type;input.ground=ground;
        input.tile_width=frame.tile_width;input.tile_height=frame.tile_height;input.target_height=content_view_height;
        if(world_content){
            input.tile_width=128;input.tile_height=64;input.target_height=128;
        }
        input.world_revision=frame.world_topology_revision;input.detail=patch_detail;
        input.river_ready=river_assets_ready;input.skip_flat_shore=skip_shore;input.separate_relief=separate;input.indexed=indexed;input.retain_height=heights;
        input.key={std::uint64_t(tile.tile_x),std::uint64_t(tile.tile_y),std::uint64_t(tile.real_terrain_type)*32+unsigned(ground),
            std::uint64_t(input.tile_width),std::uint64_t(input.tile_height),std::uint64_t(input.target_height),std::uint64_t(content_revision),
            std::uint64_t(frame.world_width_tiles),std::uint64_t(frame.world_height_tiles),
            std::uint64_t(frame.world_wrap_x)*2+unsigned(frame.world_wrap_y),std::uint64_t(patch_detail.identity()),
            std::uint64_t(skip_shore)*16+unsigned(separate)*8+unsigned(indexed)*4+unsigned(heights)*2+unsigned(river_assets_ready)};
        return input;
    }
    bool terrain_result_valid(c3x_renderer::fidelity::TerrainSurfaces const& result) {
        for(auto const& input:result.world)if(world_coast.world().at(input.first)!=input.second)return false;
        for(auto const& input:result.coast)if(world_coast.node_revision(input.first)!=input.second)return false;
        return natural.valid(result.rivers);
    }
    std::unique_ptr<c3x_renderer::fidelity::TerrainSurfaces> compile_terrain(
            c3x_renderer::fidelity::TerrainCompileInput const& input,
            c3x_renderer::fidelity::TerrainCompileScratch& scratch,std::function<bool()> cancelled,bool bounded=true) {
        return c3x_renderer::fidelity::compile_terrain_surfaces(natural,terrain_textures,world_coast,input,scratch,cancelled,bounded);
    }
    // Packs every non-empty natural layer's vertex bytes into one immutable
    // buffer and attaches it to the compiled result, so GPU adoption becomes
    // an AddRef instead of a copy+CreateBuffer. Safe to call from a worker
    // thread: the device is created without D3D11_CREATE_DEVICE_SINGLETHREADED,
    // so CreateBuffer is free-threaded. compile_terrain_surfaces stays CPU-only
    // and device-free; this step runs after it, still off the render thread.
    bool attach_terrain_vertex_buffer(c3x_renderer::fidelity::TerrainSurfaces& surfaces) {
        c3x_renderer::render_core::ImmutableMeshUpload vertex_upload;
        for(unsigned layer=0;layer<3;++layer){
            auto const& mesh=surfaces.meshes[layer];
            if(mesh.empty())continue;
            surfaces.vertex_offset[layer]=vertex_upload.append(mesh.vertices.data(),mesh.vertices.size());
        }
        ID3D11Buffer* buffer=nullptr;
        if(!vertex_upload.create(device,&buffer))return false;
        if(buffer)surfaces.vertex_buffer=std::shared_ptr<void>(buffer,[](void* p){static_cast<ID3D11Buffer*>(p)->Release();});
        return true;
    }

    bool attach_object_buffer(c3x_renderer::objects::PreparedObjects& result) {
        c3x_renderer::render_core::ImmutableMeshUpload upload;
        auto append=[&](auto& part){if(part.mesh.empty())return;
            part.vertex_offset=upload.append(part.mesh.vertices.data(),part.mesh.vertices.size());
            part.index_offset=upload.append(part.mesh.indices.data(),part.mesh.indices.size());};
        for(auto& part:result.layers)append(part);
        for(auto& part:result.city)append(part);
        ID3D11Buffer* buffer=nullptr;
        if(!upload.create(device,&buffer))return false;
        result.gpu_bytes=upload.size();
        if(buffer)result.buffer=std::shared_ptr<void>(buffer,[](void* p){static_cast<ID3D11Buffer*>(p)->Release();});
        return true;
    }

    std::vector<std::uint64_t> prepared_view_dependencies() const {
        std::vector<std::uint64_t> result;
        for(auto const& item:tile_geometry_cache){auto const& tile=item.second;
            if(tile.last_used!=tile_geometry_epoch)continue;
            if(!tile.shared_natural)result.push_back(topology_cache.key(tile.tile_x,tile.tile_y));
            for(auto const& d:tile.dependencies)result.push_back(d.first);
            for(auto const& d:tile.appearance_dependencies)result.push_back(d.first);
            for(auto const& d:tile.anchor_dependencies)result.push_back(d.first);
        }
        return result;
    }

    bool tile_content_valid(CachedTileGeometry& cached,c3x_renderer_tile_v1 const& tile) {
        bool valid = !cached.shared_natural;
        if(cached.natural_content.generation){
            auto shared=resident_content.resolve(cached.natural_content);
            valid=valid && shared && shared->shared_natural;
        }
        if(!valid)return false; // Residency may change within a frame.
        if(tile_geometry_epoch && cached.validity_epoch==tile_geometry_epoch &&
           cached.validity_anchor_x==tile.anchor_x && cached.validity_anchor_y==tile.anchor_y)return cached.validity;
        // Source inputs are immutable during this frame's CPU read lease.
        // Preparation selection and view assembly share one complete proof.
        for(auto const& dependency:cached.appearance_dependencies)
            if(topology_cache.appearance_revision(dependency.first)!=dependency.second){valid=false;break;}
        for (auto const & dependency : cached.dependencies) {
            auto current = topology_cache.current(dependency.first);
            auto value = current == nullptr ? 0 : current->semantic;
            if (value != dependency.second) { valid = false; break; }
        }
        for (auto const & dependency : cached.coast_dependencies)
            if (world_coast.node_revision(dependency.first) != dependency.second) { valid = false; break; }
        for (auto const & dependency : cached.world_dependencies)
            if (world_coast.world().at(dependency.first) != dependency.second) { valid = false; break; }
        for (auto const & dependency : cached.anchor_dependencies) {
            auto current = topology_cache.current(dependency.first);
            if (current == nullptr ||
                std::int64_t(current->occurrence.anchor_x-tile.anchor_x)*(cached.world_ground?cached.source_tile_width:1) !=
                    std::int64_t(dependency.second[0])*(cached.world_ground?shadow_tile_width:1) ||
                std::int64_t(current->occurrence.anchor_y-tile.anchor_y)*(cached.world_ground?cached.source_tile_width:1) !=
                    std::int64_t(dependency.second[1])*(cached.world_ground?shadow_tile_width:1)) valid = false;
        }
        cached.validity=valid && natural.valid(cached.river_dependencies);
        cached.validity_epoch=tile_geometry_epoch;
        cached.validity_anchor_x=tile.anchor_x;cached.validity_anchor_y=tile.anchor_y;
        return cached.validity;
    }

    bool restore_viewport_geometry(CachedViewport const& stored,c3x_renderer_frame_v1 const& frame,
                                   c3x_renderer::TerrainFrameSignature const& signature) {
        if(stored.signature.complete!=signature.complete || stored.tile_keys.size()!=frame.tile_count)return false;
        // The complete frame signature already validates content, environment,
        // topology and camera. Only GPU residency/lifetime needs checking again.
        std::vector<CachedTileGeometry*> owners(frame.tile_count,nullptr);
        for(std::size_t i=0;i<owners.size();++i){
            auto handle=stored.tile_keys[i];if(!handle.generation)continue;
            owners[i]=resident_content.resolve(handle);
            if(!owners[i] || owners[i]->shared_natural)return false;
            if(owners[i]->natural_content.generation){
                auto world_mesh=resident_content.resolve(owners[i]->natural_content);
                if(!world_mesh || !world_mesh->shared_natural)return false;
            }
        }
        // Validate the entire set before publishing any partial active geometry.
        for(std::size_t i=0;i<owners.size();++i)
            if(owners[i])append_tile_geometry(*owners[i],frame.tiles[i],true);
        geometry_cache.signature=signature;geometry_cache.tiles=stored.tiles;
        geometry_cache.tile_keys=stored.tile_keys;
        geometry_cache.replacement_flags=stored.replacement_flags;
        geometry_cache.fallback_indices=stored.fallback_indices;
        geometry_cache.rendered_tile_count=stored.rendered_tile_count;
        geometry_cache.fallback_tile_count=stored.fallback_tile_count;
        geometry_cache.textured_tile_count=stored.textured_tile_count;
        geometry_cache.valid=true;geometry_world_revision=frame.world_topology_revision;
        return true;
    }

    static bool append_region_bytes(c3x_renderer::render_core::RenderRegionKey& key,void const* data,std::size_t bytes) {
        auto words=(bytes+7)/8;
        if(key.size()+1+words>c3x_renderer::render_core::RenderRegionCache<ID3D11Texture2D>::key_words_limit)return false;
        key.push_back(bytes);auto source=static_cast<unsigned char const*>(data);
        for(std::size_t i=0;i<bytes;i+=8){std::uint64_t word=0;std::memcpy(&word,source+i,std::min<std::size_t>(8,bytes-i));key.push_back(word);}
        return true;
    }

    bool chunk_intersects_region(GeometryDrawReference const& chunk,ViewportShaderSettings const& settings,
                                 D3D11_RECT const& rect,bool reflected,int radius=0) const {
        int dx=chunk.translation_x()+int(settings.translation[0]),dy=chunk.translation_y()+int(settings.translation[1]);
        float low=reflected?2*reflection.height_pixels*std::max(0.f,chunk.content().world_bounds.low[2]-2.5f/112.f):0;
        float high=reflected?2*reflection.height_pixels*std::max(0.f,chunk.content().world_bounds.high[2]-2.5f/112.f):0;
        return !(chunk.bounds().right+dx+radius<=rect.left || chunk.bounds().left+dx-radius>=rect.right ||
                 chunk.bounds().bottom+dy+high+radius<=rect.top || chunk.bounds().top+dy+low-radius>=rect.bottom);
    }

    bool query_region_inputs(unsigned pass,int left,int top,unsigned extent_x,unsigned extent_y,
            std::vector<std::pair<unsigned,unsigned>>& output){
        if(!region_contributors.query_rectangle(pass,left,top,extent_x,extent_y,output))return false;
        if(!world_pass_affine)return true;
        std::vector<c3x_renderer::render_core::WorldPassIndex::Key> keys;
        double scale=1./shadow_tile_width;
        double margin=pass?world_pass_reflection:0;
        if(!world_pass_index.query((left-world_pass_x)*scale,(top-world_pass_y-margin)*scale,
            (left+double(extent_x)-world_pass_x)*scale,(top+double(extent_y)-world_pass_y)*scale,keys))return false;
        for(auto key:keys){auto found=world_pass_occurrences.find(key);if(found==world_pass_occurrences.end())continue;
            if(output.size()+found->second.size()>region_contributors.budget/(2*sizeof(output[0])))return false;
            output.insert(output.end(),found->second.begin(),found->second.end());
        }
        std::sort(output.begin(),output.end());output.erase(std::unique(output.begin(),output.end()),output.end());return true;
    }
    bool query_region_inputs(unsigned pass,int left,int top,unsigned extent,
            std::vector<std::pair<unsigned,unsigned>>& output){return query_region_inputs(pass,left,top,extent,extent,output);}

    void prepare_region_contributors(GeometryDrawView buffers) {
        if(region_contributors.ready && region_contributors.tile_width==shadow_tile_width &&
           region_contributors.reflection_height==reflection.height_pixels)return;
        region_contributors.clear();world_pass_occurrences.clear();world_pass_affine=false;world_pass_reflection=0;
        try {
            bool first=true,affine=shadow_tile_width>=32;
            for(unsigned layer=0;layer<geometry_layer_count;++layer)for(unsigned i=0;i<buffers[layer].size();++i){
                auto const& chunk=buffers[layer][i];auto key=reinterpret_cast<std::uintptr_t>(&chunk.content());
                if(!world_pass_index.contains(key))continue;
                auto const& p=chunk.natural_projection();
                double x=chunk.translation_x()-(p[0]+p[1])*shadow_tile_width*.5;
                double y=chunk.translation_y()-(p[0]-p[1])*shadow_tile_width*.25;
                if(first){world_pass_x=x;world_pass_y=y;first=false;}
                else if(x!=world_pass_x || y!=world_pass_y)affine=false;
                if(world_pass_occurrences.size()>=c3x_renderer::render_core::WorldPassIndex::budget/256){affine=false;break;}
                world_pass_occurrences[key].push_back({layer,i});
                world_pass_reflection=std::max(world_pass_reflection,
                    2*reflection.height_pixels*std::max(0.f,chunk.content().world_bounds.high[2]-2.5f/112.f));
            }
            world_pass_affine=affine && !first;
            if(!world_pass_affine)world_pass_occurrences.clear();
            for(unsigned pass=0;pass<2;++pass)for(unsigned layer=0;layer<geometry_layer_count;++layer)
                for(unsigned i=0;i<buffers[layer].size();++i){auto const& chunk=buffers[layer][i];
                    if(world_pass_affine && world_pass_occurrences.count(reinterpret_cast<std::uintptr_t>(&chunk.content())))continue;
                    int radius=layer==geometry_city && chunk.content().city_lighting?int(std::ceil(shadow_tile_width*.85f))+8:0;
                    float low=pass?2*reflection.height_pixels*std::max(0.f,chunk.content().world_bounds.low[2]-2.5f/112.f):0;
                    float high=pass?2*reflection.height_pixels*std::max(0.f,chunk.content().world_bounds.high[2]-2.5f/112.f):0;
                    if(!region_contributors.add(pass,{layer,i},double(chunk.bounds().left)+chunk.translation_x()-radius,
                        double(chunk.bounds().top)+chunk.translation_y()+low-radius,
                        double(chunk.bounds().right)+chunk.translation_x()+radius,
                        double(chunk.bounds().bottom)+chunk.translation_y()+high+radius)){
                        region_contributors.clear();return;
                    }
                }
            region_contributors.tile_width=shadow_tile_width;
            region_contributors.reflection_height=reflection.height_pixels;region_contributors.ready=true;
        }catch(...){region_contributors.clear();}
    }

    void collect_region_receivers(GeometryDrawView buffers,
            ViewportShaderSettings const& settings,std::vector<D3D11_RECT> const& rectangles,bool reflected,
            std::vector<c3x_renderer::render_core::SourceShadow::Bounds>& receivers) {
        // The index borrows only the current static assembly. Posed buffers and
        // rejected/unavailable indexes retain the original complete scan.
        std::vector<c3x_renderer::render_core::RegionContributorIndex::Item> candidates;
        bool indexed=false;
        if(composition_receiver_index && buffers.is(geometry_vertex_buffers) && rectangles.size()==1 &&
           region_contributors.tile_width==shadow_tile_width && region_contributors.reflection_height==reflection.height_pixels) {
            auto const& rect=rectangles[0];
            try {indexed=query_region_inputs(reflected?1u:0u,rect.left-int(settings.translation[0]),
                rect.top-int(settings.translation[1]),std::max(rect.right-rect.left,rect.bottom-rect.top),candidates);}
            catch(...) {}
        }
        auto append=[&](unsigned layer,GeometryDrawReference const& chunk) {
            if(layer==geometry_shadow)return;
            bool visible=false;
            for(auto const& rect:rectangles)visible=visible || chunk_intersects_region(chunk,settings,rect,reflected);
            if(visible)receivers.push_back(chunk.content().world_bounds);
        };
        if(indexed)for(auto const& item:candidates)append(item.first,buffers[item.first][item.second]);
        else for(unsigned layer=0;layer<geometry_layer_count;++layer)for(auto const& chunk:buffers[layer])append(layer,chunk);
    }

    bool render_region_key(GeometryDrawView buffers,
            ViewportShaderSettings const& settings,
            std::vector<c3x_renderer::render_core::SourceShadow::Caster> const& casters,
            c3x_renderer::render_core::SourceShadow::PreparedCasters* prepared,
            c3x_renderer::render_core::RenderRegionKey& key,std::vector<std::size_t>* sections=nullptr) {
        using Shadow=c3x_renderer::render_core::SourceShadow;
        if(region_context.empty() || !prepared || !prepared->matches(casters,shadow_basis))return false;
        auto phase_started=std::chrono::steady_clock::now();
        auto phase=[&](unsigned index){auto now=std::chrono::steady_clock::now();
            frame_region_phase_ms[index]+=std::chrono::duration<double,std::milli>(now-phase_started).count();phase_started=now;};
        key=region_context;
        if(sections){sections->clear();sections->push_back(key.size());}
        for(unsigned pass=0;pass<(reflection.enabled?2u:1u);++pass){
            ViewportShaderSettings local=settings;
            int extent=pass?144:136;
            if(pass){local.translation[0]+=4;local.translation[1]+=4;local.inverse_size[0]=local.inverse_size[1]=1.f/float(extent);}
            D3D11_RECT rect={0,0,extent,extent};
            std::vector<Shadow::Bounds> receivers;
            std::vector<c3x_renderer::city_fidelity::Lighting const*> lights;
            key.push_back(0x726567696f6e0000ull+pass);
            std::vector<c3x_renderer::render_core::RegionContributorIndex::Item> candidates;
            bool indexed=false;
            try{indexed=query_region_inputs(pass,-int(local.translation[0]),-int(local.translation[1]),extent,candidates);}catch(...){}
            auto contribute=[&](unsigned layer,GeometryDrawReference const& chunk){
                if(layer==geometry_city && chunk.content().city_lighting &&
                   chunk_intersects_region(chunk,local,rect,pass!=0,int(std::ceil(shadow_tile_width*.85f))+8)){
                    auto light=chunk.content().city_lighting.get();
                    if(std::find(lights.begin(),lights.end(),light)==lights.end())lights.push_back(light);
                }
                if(!chunk_intersects_region(chunk,local,rect,pass!=0))return true;
                if(chunk.content().animation_texture)return false; // Posed pixels never enter this static cache.
                if(layer!=geometry_shadow)receivers.push_back(chunk.content().world_bounds);
                std::uint64_t identity[]={layer,chunk.content().version,chunk.content().index_count,std::uint64_t(chunk.content().index_format),
                    chunk.content().vertex_stride,chunk.content().city_material,std::uint64_t(chunk.content().city_environment),
                    chunk.content().vertex_offset,chunk.content().index_offset,
                    std::uint64_t(reinterpret_cast<std::uintptr_t>(chunk.content().buffer)),
                    std::uint64_t(reinterpret_cast<std::uintptr_t>(chunk.content().indices))};
                if(!append_region_bytes(key,identity,sizeof(identity)))return false;
                auto effective=local;
                std::copy(std::begin(chunk.natural_projection()),std::end(chunk.natural_projection()),effective.natural_projection);
                effective.reserved[1]=layer==geometry_underlay?.5f:layer==geometry_bed?4.f:layer==geometry_water?5.f:0.f;
                effective.translation[0]+=float(chunk.translation_x());effective.translation[1]+=float(chunk.translation_y());
                effective.depth_translation=city_profile?local.depth_translation+float(chunk.translation_y()):effective.translation[1];
                if(!append_region_bytes(key,&effective,sizeof(effective)) ||
                   !append_region_bytes(key,chunk.content().city_atlas,sizeof(chunk.content().city_atlas)))return false;
                return true;
            };
            if(indexed){for(auto const& candidate:candidates)
                if(!contribute(candidate.first,buffers[candidate.first][candidate.second]))return false;
            }else for(unsigned layer=0;layer<geometry_layer_count;++layer)for(auto const& chunk:buffers[layer])
                if(!contribute(layer,chunk))return false;
            if(sections)sections->push_back(key.size());
            phase(0);
            key.push_back(0x6c69676874730000ull);key.push_back(lights.size());
            for(auto light:lights){
                if(!append_region_bytes(key,light->lights.data(),light->lights.size()*sizeof(light->lights[0])) ||
                   !append_region_bytes(key,light->blockers.data(),light->blockers.size()*sizeof(light->blockers[0])))return false;
            }
            if(sections)sections->push_back(key.size());
            phase(1);
            auto proof_start=key.size();
            auto receiver_key=prepared->receiver_key(receivers,region_receiver_shadows);
            auto proof=prepared->find_receiver(receiver_key);
            if(proof)key.insert(key.end(),proof->begin(),proof->end());
            else {
            auto pages=Shadow::required_pages(receivers,shadow_basis);
            if(pages.size()>32)return false;
            key.push_back(0x736861646f770000ull);key.push_back(pages.size());
            for(auto page:pages){
                auto selection=prepared->find(page);Shadow::PreparedCasters::Selection temporary;
                if(!selection){
                    temporary=Shadow::PreparedCasters::select(casters,prepared->bounds,page);
                    selection=prepared->admit(page,std::move(temporary));
                    if(!selection)selection=&temporary;
                }
                Shadow::PreparedCasters::Selection narrowed;
                if(region_receiver_shadows){
                    narrowed=Shadow::PreparedCasters::receiver_selection(*selection,casters,prepared->bounds,receivers,shadow_basis);
                    selection=&narrowed;
                }
                std::uint64_t identity[]={std::uint64_t(page.first),std::uint64_t(page.second),selection->hash,std::uint64_t(selection->indices.size())};
                if(!append_region_bytes(key,identity,sizeof(identity)))return false;
            }
            try{prepared->admit_receiver(std::move(receiver_key),{key.begin()+std::ptrdiff_t(proof_start),key.end()});}catch(...){}
            }
            if(sections)sections->push_back(key.size());
            phase(2);
        }
        return key.size()<=c3x_renderer::render_core::RenderRegionCache<ID3D11Texture2D>::key_words_limit;
    }

    bool draw_cached_geometry(GeometryLayer layer,
            GeometryDrawView buffers,
            std::vector<D3D11_RECT> const & rectangles, ViewportShaderSettings const & viewport_settings,
            std::atomic<bool> const * cancellation, bool reflection_pass=false) {
        if(buffers[layer].empty())return true;
        auto issue_begin=std::chrono::steady_clock::now();
        AnimationGpu::Pass gpu_phase(animation_gpu,context,layer==geometry_shadow?AnimationGpu::shadow:AnimationGpu::body,
            profiling && (layer==geometry_shadow || layer==geometry_feature) && !buffers[layer].empty());
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
        if(diagnostic_routes && layer==geometry_route)return true;
#endif
        if(pickup_profile && layer<geometry_natural_terrain)context->IASetInputLayout(layer>=geometry_feature?feature_input_layout:input_layout);
        ViewportShaderSettings previous = {};
        bool first = true;
        for (D3D11_RECT const & rect : rectangles) {
        D3D11_RECT scaled=rect;
        if(fidelity_profile){scaled.left*=2;scaled.top*=2;scaled.right*=2;scaled.bottom*=2;}
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
        // Keep candidate selection, geometry, projection and draw submission intact.
        // Only main/reflection geometry pixel coverage is reduced; shadow pages,
        // finishing and readback retain their ordinary sizes and responsibilities.
        if(diagnostic_half_pixels)scaled.right=scaled.left+(scaled.right-scaled.left+1)/2;
#endif
        context->RSSetScissorRects(1, &scaled);
        if(layer>=geometry_natural_forest0 && !buffers[layer].empty() && buffers[layer][0].content().instances){
            using Stream=c3x_renderer::render_core::InstanceStream;
            std::vector<Stream::Instance> selected;selected.reserve(256);
            auto const&mesh=buffers[layer][0].content();
            context->UpdateSubresource(viewport_settings_buffer,0,nullptr,&viewport_settings,0,0);++frame_parameter_updates;
            natural.bind_instances(context,unsigned(layer-geometry_natural_forest0));
            auto flush=[&](){
                if(selected.empty())return true;
                if(!natural.instance_stream.upload(device,context,selected))return false;
                ID3D11Buffer*streams[]={mesh.buffer,natural.instance_stream.buffer};UINT strides[]={32,64},offsets[]={0,natural.instance_stream.offset};
                context->IASetVertexBuffers(0,2,streams,strides,offsets);context->IASetIndexBuffer(mesh.indices,mesh.index_format,0);
                context->DrawIndexedInstanced(mesh.index_count,UINT(selected.size()),0,0,0);++frame_draw_calls;
                selected.clear();return true;
            };
            for(auto const&chunk:buffers[layer]){
                ++frame_bounds_tests;if(cancellation && cancellation->load(std::memory_order_relaxed))return false;
                if(!chunk_intersects_region(chunk,viewport_settings,rect,reflection_pass))continue;
                if(!chunk.content().instances || chunk.content().buffer!=mesh.buffer)return false;
                for(auto instance:*chunk.content().instances){
                    if(selected.size()==Stream::limit && !flush())return false;
                    std::copy(std::begin(chunk.natural_projection()),std::end(chunk.natural_projection()),instance.projection);
                    instance.view[0]=viewport_settings.translation[0]+float(chunk.translation_x());
                    instance.view[1]=viewport_settings.translation[1]+float(chunk.translation_y());
                    instance.view[2]=viewport_settings.depth_translation+float(chunk.translation_y());
                    selected.push_back(instance);
                }
            }
            if(!flush())return false;
            first=true;continue;
        }
        using Parameters=c3x_renderer::render_core::DrawParameterStream;
        std::array<ViewportShaderSettings,Parameters::limit> parameters;
        std::vector<GeometryDrawReference> selected;selected.reserve(Parameters::limit);
        bool streamed=draw_parameters.available(device,context);
        auto flush=[&](){
            if(selected.empty())return true;
            if(streamed && !draw_parameters.upload(parameters.data(),unsigned(selected.size())))return false;
            auto issue=[&](GeometryDrawReference const& chunk,ViewportShaderSettings const& settings,unsigned index){
                if(streamed)draw_parameters.bind(1,index);
                else if(first || std::memcmp(&previous,&settings,sizeof(settings))!=0){
                    context->UpdateSubresource(viewport_settings_buffer,0,nullptr,&settings,0,0);
                    ++frame_parameter_updates;previous=settings;first=false;
                }
                UINT stride = chunk.content().vertex_stride, offset = chunk.content().vertex_offset;
                context->IASetVertexBuffers(0, 1, &chunk.content().buffer, &stride, &offset);
                context->IASetIndexBuffer(chunk.content().indices, chunk.content().index_format, chunk.content().index_offset);
                if(chunk.content().city_material!=0xffffffffu){
                    ID3D11SamplerState*samplers[]={natural_wrap,natural_clamp};context->PSSetSamplers(0,2,samplers);
                    context->OMSetBlendState(blend_state,nullptr,0xffffffffu);context->OMSetDepthStencilState(depth_state,0);
                    cities.bind(context,chunk.content().city_material,chunk.content().city_environment,chunk.content().city_atlas,reflection_pass,false);
                    context->DrawIndexed(chunk.content().index_count,0,0);
                    ++frame_draw_calls;
                    if(!cities.library.materials[chunk.content().city_material].ground){
                        cities.bind(context,chunk.content().city_material,chunk.content().city_environment,chunk.content().city_atlas,reflection_pass,true);
                        context->DrawIndexed(chunk.content().index_count,0,0);
                        ++frame_draw_calls;
                    }
                    context->OMSetBlendState(blend_state,nullptr,0xffffffffu);context->OMSetDepthStencilState(depth_state,0);
                    return;
                }
                if(city_profile && layer==geometry_city){
                    context->IASetInputLayout(feature_input_layout);
                    context->VSSetShader(reflection_pass?reflection.vs[1]:feature_vertex_shader,nullptr,0);
                    context->PSSetShader(reflection_pass?reflection.ps[1]:feature_pixel_shader,nullptr,0);
                    context->PSSetShaderResources(116,4,city_emissive_views.data());context->PSSetShaderResources(124,4,city_base_views.data());
                    ID3D11SamplerState*samplers[]={terrain_sampler,decal_sampler};context->PSSetSamplers(0,2,samplers);
                }
                if (chunk.content().animation_texture) context->PSSetShaderResources(116,1,&chunk.content().animation_texture);
                if(chunk.content().resource_instance){
                    context->IASetInputLayout(resource_input_layout);
                    context->VSSetConstantBuffers(8,1,&chunk.content().resource_instance);
                    context->VSSetConstantBuffers(2,1,&shadow_settings_buffer);
                    context->VSSetShader(layer==geometry_shadow?resource_shadow_vertex_shader:resource_body_vertex_shader,nullptr,0);
                }
                if(layer==geometry_wave){float wave_sample[]={chunk.content().visual_time<0?wave_time_seconds:chunk.content().visual_time,0,0,0};
                    context->UpdateSubresource(wave_frame,0,nullptr,wave_sample,0,0);}
                context->DrawIndexed(chunk.content().index_count, 0, 0);
                ++frame_draw_calls;
                if(chunk.content().resource_instance){
                    context->IASetInputLayout(layer==geometry_shadow?input_layout:feature_input_layout);
                    context->VSSetShader(layer==geometry_shadow?vertex_shader:feature_vertex_shader,nullptr,0);
                }
                if (chunk.content().animation_texture) context->PSSetShaderResources(116,1,resource_texture_views.data());
            };
            for(unsigned i=0;i<selected.size();++i)issue(selected[i],parameters[i],i);
            selected.clear();return true;
        };
        for (GeometryDrawReference const & chunk : buffers[layer]) {
            ++frame_bounds_tests;
            if(cancellation && cancellation->load(std::memory_order_relaxed))return false;
            if(reflection_pass && chunk.content().animation_texture)continue;
            if(!chunk_intersects_region(chunk,viewport_settings,rect,reflection_pass))continue;
            ViewportShaderSettings settings = viewport_settings;
            std::copy(std::begin(chunk.natural_projection()),std::end(chunk.natural_projection()),settings.natural_projection);
            settings.padding=float(chunk.content().projection_kind);
            if(pickup_profile)settings.reserved[1]=layer==geometry_underlay?.5f:layer==geometry_bed?4.f:layer==geometry_water?5.f:0.f;
            settings.translation[0] += static_cast<float>(chunk.translation_x());
            settings.translation[1] += static_cast<float>(chunk.translation_y());
            settings.depth_translation = city_profile?viewport_settings.depth_translation+float(chunk.translation_y()):settings.translation[1];
            parameters[selected.size()]=settings;selected.push_back(chunk);
            if(selected.size()==Parameters::limit && !flush())return false;
        }
        if(!flush())return false;
        // Other shader owners (including the existing forest instance stream)
        // retain their ordinary b1 buffer contract across selected pass calls.
        if(streamed)context->VSSetConstantBuffers(1,1,&viewport_settings_buffer);
        }
        if(city_profile && layer==geometry_city){ID3D11SamplerState*samplers[]={terrain_sampler,decal_sampler};context->PSSetSamplers(0,2,samplers);}
        frame_geometry_issue_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-issue_begin).count();
        return true;
    }

    D3D11_RECT guarded_block_rectangle(D3D11_RECT const& rect, int dx, int dy, int guard, int extent, int extent_y=0) {
        if(!extent_y)extent_y=extent;
        return {std::max<LONG>(0,rect.left+dx-guard),std::max<LONG>(0,rect.top+dy-guard),
                std::min<LONG>(extent,rect.right+dx+guard),std::min<LONG>(extent_y,rect.bottom+dy+guard)};
    }

    void collect_shadow_casters(
            GeometryDrawView buffers,
            std::vector<c3x_renderer::render_core::SourceShadow::Caster> & casters) {
        using Shadow=c3x_renderer::render_core::SourceShadow;
        auto dims=world_coast.world().dimensions();
        for(unsigned layer=0;layer<geometry_layer_count;++layer)for(auto const& chunk:buffers[layer]) {
            bool caster=layer==geometry_land || (layer>=geometry_feature && layer!=geometry_natural_decal);
            if(chunk.content().city_material!=0xffffffffu && cities.library.materials[chunk.content().city_material].ground)caster=false;
            if(!caster || chunk.content().animation_texture)continue;
            for(int wy=dims.wrap_y?-1:0;wy<=(dims.wrap_y?1:0);++wy)
                for(int wx=dims.wrap_x?-1:0;wx<=(dims.wrap_x?1:0);++wx) {
                    Shadow::Caster c;c.vertices=chunk.content().buffer;c.indices=chunk.content().indices;c.count=chunk.content().index_count;
                    c.index_format=chunk.content().index_format;
                    c.vertex_offset=chunk.content().vertex_offset;c.index_offset=chunk.content().index_offset;
                    c.instances=chunk.content().instances.get();c.instance_material=chunk.content().instance_material;
                    c.stride=chunk.content().vertex_stride;c.layer=layer;c.version=chunk.content().version;c.bounds=chunk.content().world_bounds;
                    if(chunk.content().city_material!=0xffffffffu)c.binding=10000+chunk.content().city_material;
                    c.offset[0]=float(wx*dims.width+wy*dims.height)*.5f;
                    c.offset[1]=float(wx*dims.width-wy*dims.height)*.5f;casters.push_back(c);
                }
        }
    }

    c3x_renderer::render_core::SourceShadow::PreparedCasters* prepare_shadow_submission(
            GeometryDrawView buffers,
            std::vector<c3x_renderer::render_core::SourceShadow::Caster>& casters,
            c3x_renderer::render_core::SourceShadow::PreparedCasters& prepared) {
        ++frame_caster_preparations;
        collect_shadow_casters(buffers,casters);
        char control[8]={};
        bool reuse=!(GetEnvironmentVariableA("C3X_RENDERER_CASTER_BOUNDS_CONTROL",control,sizeof(control)) && std::strcmp(control,"1")==0);
        char dependency_control[8]={};
        bool retain=world_regions && buffers.is(geometry_vertex_buffers) &&
            !(GetEnvironmentVariableA("C3X_RENDERER_REGION_DEPENDENCY_CONTROL",dependency_control,sizeof(dependency_control)) && std::strcmp(dependency_control,"1")==0);
        char index_control[8]={};
        bool index=world_regions && buffers.is(geometry_vertex_buffers) &&
            !(GetEnvironmentVariableA("C3X_RENDERER_REGION_INDEX_CONTROL",index_control,sizeof(index_control)) && std::strcmp(index_control,"1")==0);
        if(index)prepare_region_contributors(buffers);else region_contributors.clear();
        auto& selected=retain?retained_region_casters:prepared;
        auto result=reuse && selected.build(casters,shadow_basis,retain)?&selected:nullptr;
        char detail[160];sprintf_s(detail,"casters=%zu descriptor_bytes=%zu dirty_block_clip=%u",casters.size(),
            casters.capacity()*sizeof(casters[0]),unsigned(clip_dirty_blocks));
        trace.write("submission-casters",detail,false);
        if(profiling){sprintf_s(detail,"casters=%zu bounds_capacity=%zu",casters.size(),prepared.bounds.capacity()*sizeof(prepared.bounds[0]));
            trace.write("prepared-caster-bounds",detail,false);}
        return result;
    }

    bool prepare_receiver_shadows(
            GeometryDrawView shadow_buffers,
            ViewportShaderSettings const& settings,std::vector<D3D11_RECT> const& rectangles,bool reflection_pass,
            std::vector<c3x_renderer::render_core::SourceShadow::Caster> const& casters,
            c3x_renderer::render_core::SourceShadow::PreparedCasters* prepared_casters_ptr,
            std::atomic<bool> const* cancellation,
            std::set<std::pair<int,int>> const* selected_pages=nullptr) {
        using Shadow=c3x_renderer::render_core::SourceShadow;
        std::vector<Shadow::Bounds> receivers;
        if(!selected_pages)collect_region_receivers(shadow_buffers,settings,rectangles,reflection_pass,receivers);
        std::array<ID3D11ShaderResourceView*,33> alpha{};
        std::copy(feature_texture_views.begin(),feature_texture_views.end(),alpha.begin());
        std::copy(river_rock_texture_views.begin(),river_rock_texture_views.end(),alpha.begin()+8);
        std::copy(bridge_texture_views.begin(),bridge_texture_views.end(),alpha.begin()+13);
        std::copy(resource_texture_views.begin(),resource_texture_views.end(),alpha.begin()+21);
        std::copy(city_base_views.begin(),city_base_views.end(),alpha.begin()+29);
        auto bind=[&](unsigned layer) {
            if(layer>=10000){
                auto material=layer-10000;auto mask=cities.materials[material][6];
                context->PSSetShaderResources(34,1,&mask);return mask!=nullptr;
            }
            if(layer==geometry_land)return false;
            if(layer==geometry_natural_terrain || layer==geometry_natural_mountain)return true;
            if(layer>=geometry_natural_forest0){
                auto const&m=natural.materials[natural.bodies[layer-geometry_natural_forest0].material];
                ID3D11ShaderResourceView*mask=m.channels[6]==0xffffffffu?nullptr:natural.textures[m.channels[6]];
                context->PSSetShaderResources(33,1,&mask);return mask!=nullptr;
            }
            auto views=alpha;
            if(layer==geometry_city)std::copy(city_base_views.begin(),city_base_views.end(),views.begin()+29);
            if(layer==geometry_wall)views[29]=views[30]=views[31]=views[32]=wall_texture_view;
            if(layer==geometry_site)std::copy(site_views.begin(),site_views.end(),views.begin()+21);
            if(layer==geometry_mine)std::copy(mine_base_views.begin(),mine_base_views.end(),views.begin()+21);
            if(layer==geometry_farm)std::copy(farm_base_views.begin(),farm_base_views.end(),views.begin()+21);
            if(layer>=geometry_cliff0 && layer<geometry_natural_terrain) {
                auto const & asset=cliff_bundle.assets[layer-geometry_cliff0];
                views[0]=cliff_views[asset.texture_index];
            }
            context->PSSetShaderResources(0,33,views.data());return true;
        };
        LARGE_INTEGER start={},end={};QueryPerformanceCounter(&start);
        if(!source_shadow.prepare(context,shadow_basis,receivers,casters,bind,cancellation,prepared_casters_ptr,selected_pages)) {
            trace.write("source-shadow-failed","caster pages exceeded budget or preparation interrupted",true);return false;
        }
        QueryPerformanceCounter(&end);
        char message[256];sprintf_s(message,"pages_hit=%u pages_built=%u source_draws=%u casters=%u bytes_cap=134217728 ticks=%lld",
            source_shadow.hits,source_shadow.rebuilt,source_shadow.draws,unsigned(casters.size()),end.QuadPart-start.QuadPart);
        trace.write("source-shadow",message,false);
        return true;
    }

    bool submit_prepared_resource_region(
            GeometryDrawView buffers,
            ViewportShaderSettings const& settings,D3D11_RECT const& output_damage,
            c3x_renderer::city_fidelity::Glow* surface=nullptr) {
        auto& glow=surface?*surface:city_glow;
        std::vector<D3D11_RECT> rectangles={surface?output_damage:D3D11_RECT{0,0,LONG(glow.native_extent),LONG(glow.native_height)}};
        auto& linear=glow.linear;
        context->OMSetRenderTargets(1,&linear.target,linear.depth);
        context->OMSetDepthStencilState(depth_state,0);
        context->OMSetBlendState(blend_state,nullptr,0xffffffffu);
        context->RSSetState(rasterizer_state);
        D3D11_VIEWPORT viewport={0,0,float(linear.width),float(linear.height),0,1};context->RSSetViewports(1,&viewport);
        if(!cities.lights(context,{}))return false;
        // The shadow material variant preserves shading, sample order and depth.
        // Only body/shadow resources are consumed: no terrain table,
        // natural-provider transitions, cliff bindings or empty layer draws.
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->PSSetConstantBuffers(0,1,&terrain_settings_buffer);
        context->VSSetConstantBuffers(1,1,&viewport_settings_buffer);
        context->PSSetConstantBuffers(2,1,&shadow_settings_buffer);
        context->PSSetConstantBuffers(3,1,&world_settings_buffer);
        context->PSSetConstantBuffers(4,1,&source_shadow.table);
        context->PSSetShaderResources(17,1,&source_shadow.view);
        context->PSSetShaderResources(25,1,&source_shadow.view);
        context->PSSetShaderResources(116,8,resource_texture_views.data());
        ID3D11SamplerState* shadow_samplers[]={natural_wrap,natural_clamp};
        context->PSSetSamplers(0,2,shadow_samplers);
        context->VSSetShader(vertex_shader,nullptr,0);
        context->PSSetShader(resource_shadow_shader,nullptr,0);
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
        if(!diagnostic_animation)
#endif
        if(!draw_cached_geometry(geometry_shadow,buffers,rectangles,settings,nullptr))return false;
        ID3D11SamplerState* body_samplers[]={terrain_sampler,decal_sampler};
        context->PSSetSamplers(0,2,body_samplers);
        context->PSSetShaderResources(25,4,feature_texture_views.data());
        context->PSSetShaderResources(94,4,feature_texture_views.data()+4);
        context->VSSetShader(feature_vertex_shader,nullptr,0);
        context->PSSetShader(feature_pixel_shader,nullptr,0);
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
        if(!diagnostic_animation)
#endif
        if(!draw_cached_geometry(geometry_feature,buffers,rectangles,settings,nullptr))return false;
        if(surface)return true; // Shared scene caller finishes and reads back once.
        AnimationGpu::Pass gpu_phase(animation_gpu,context,AnimationGpu::finish);
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
        if(diagnostic_animation>=2){
            // Defined diagnostic output preserves replay determinism. This is
            // a counterfactual cost experiment, never visual acceptance.
            context->OMSetRenderTargets(0,nullptr,nullptr);float clear[4]={};
            context->ClearRenderTargetView(block_target,clear);return true;
        }
#endif
        frame_post_lanes+=city_glow.reconstruct(context,&output_damage);
        linear_output.draw(context,linear,city_glow.target,display_exposure,1,city_glow.view,136,136,&output_damage);
        context->OMSetRenderTargets(0,nullptr,nullptr);
        D3D11_BOX box={unsigned(output_damage.left),unsigned(output_damage.top),0,unsigned(output_damage.right),unsigned(output_damage.bottom),1};
        context->CopySubresourceRegion(block_texture,0,box.left-4,box.top-4,0,city_glow.native,0,&box);
        return true;
    }

    std::array<ID3D11ShaderResourceView*,128> const& compiled_material_views(){
        if(material_views_valid)return material_views;
        auto& views=material_views;views={};
        TerrainTexture const & grass = terrain_textures[2];
        TerrainTexture const & plains = terrain_textures[1];
        TerrainTexture const & desert = terrain_textures[0];
        TerrainTexture const & hills = terrain_textures[5];
        TerrainTexture const & mountain = terrain_textures[6];
        TerrainTexture const & marsh = terrain_textures[9];
        TerrainTexture const & coast = terrain_textures[11];
        TerrainTexture const & ocean = terrain_textures[13];
        views[0] = grass.view; views[1] = grass.material_height_view; views[2] = grass.specular_view;
        views[3] = mountain.material_height_view; views[4] = mountain.material_height_view;
        views[5] = mountain.material_height_view;
        views[6] = mountain.view; views[7] = mountain.elevated_view;
        views[8] = mountain.relief_layer_views[0];
        views[9] = mountain.material_height_view; views[10] = mountain.specular_view;
        views[11] = coast.relief_layer_views[0]; views[12] = terrain_extra_views[0];
        views[13] = terrain_extra_views[1]; views[14] = coast.relief_layer_views[1];
        views[15] = terrain_extra_views[2]; views[16] = terrain_extra_views[3];
        views[17] = coast.view; views[18] = ocean.view;
        views[19] = coast.material_height_view;
        for (std::size_t index = 0; index < coast.water_surface_views.size(); ++index)
            views[20 + index] = coast.water_surface_views[index];
        for (std::size_t index = 0; index < 4; ++index)
            views[25 + index] = feature_texture_views[index];
        views[29] = coast.specular_view; views[30] = ocean.material_height_view;
        views[31] = ocean.specular_view;
        for (std::size_t index = 0; index < 13; ++index)
            views[32 + index] = terrain_extra_views[4 + index];
        views[45] = plains.view; views[46] = plains.material_height_view;
        views[47] = plains.specular_view;
        views[48] = desert.view; views[49] = desert.material_height_view;
        views[50] = desert.specular_view; views[51] = hills.material_height_view;
        views[52] = dune_surface.view; views[53] = dune_surface.material_height_view;
        views[54] = dune_surface.specular_view; views[55] = dune_decal_base_view;
        views[56] = dune_decal_height_view;
        for (std::size_t index = 1; index < mountain.relief_layer_views.size(); ++index)
            views[56 + index] = mountain.relief_layer_views[index];
        views[61] = terrain_extra_views[17]; views[62] = terrain_extra_views[18];
        views[63] = marsh.view; views[64] = marsh.material_height_view;
        views[65] = marsh.specular_view; views[66] = marsh_decal_base_view;
        views[67] = marsh_decal_height_view; views[68] = marsh_decal_specular_view;
        views[69] = volcano_base_view; views[70] = volcano_height_view;
        views[71] = volcano_active_base_view; views[72] = volcano_active_specular_view;
        views[73] = water_clutter_base_view; views[74] = water_clutter_height_view;
        views[75] = grass_clutter_base_view; views[76] = grass_clutter_height_view;
        views[77] = plains_clutter_base_view; views[78] = plains_clutter_height_view;
        for (std::size_t index = 0; index < river_surface_views.size(); ++index)
            views[79 + index] = river_surface_views[index];
        for (std::size_t index = 0; index < river_rock_texture_views.size(); ++index)
            views[89 + index] = river_rock_texture_views[index];
        TerrainTexture const & tundra = terrain_textures[3];
        views[94] = tundra.view;
        views[95] = tundra.material_height_view;
        views[96] = tundra.specular_view;
        for (std::size_t index = 0; index < route_texture_views.size(); ++index)
            views[98 + index] = route_texture_views[index];
        for (std::size_t index = 0; index < bridge_texture_views.size(); ++index)
            views[108 + index] = bridge_texture_views[index];
        for (std::size_t index = 0; index < resource_texture_views.size(); ++index)
            views[116 + index] = resource_texture_views[index];
        for (std::size_t index = 0; index < city_base_views.size(); ++index)
            views[124 + index] = city_base_views[index];
        material_views_valid=true;return material_views;
    }

    bool submit_geometry(GeometryDrawView buffers,
                         std::vector<D3D11_RECT> const & rectangles, ViewportShaderSettings const & settings,
                         ID3D11RenderTargetView * target, ID3D11DepthStencilView * depth,
                         int projection_width, int projection_height,
                         std::atomic<bool> const * cancellation = nullptr,
                         bool accumulate = false, bool finish = true,
                         GeometryDrawView shadow_buffers_ptr = {},
                         bool reflection_pass=false,
                         std::vector<c3x_renderer::render_core::SourceShadow::Caster> const * shadow_casters_ptr=nullptr,
                         c3x_renderer::render_core::SourceShadow::PreparedCasters * prepared_casters_ptr=nullptr,
                         int region_size=128,int grid_x=0,int grid_y=0,bool require_linear_backdrop=false,bool scene_surface_pass=false) {
        int const region_height=region_size==2240?256:region_size;
        auto& active_glow=scene_surface_pass?region_glow:region_size==128?city_glow:region_glow;
        auto& active_reflection=region_size==128?reflection:region_reflection;
        // Geometry owners remain pinned throughout this synchronous submission.
        // Camera blocks and reflection passes borrow one immutable caster list;
        // only receivers depend on their current screen rectangle. No list is
        // retained across a frame, content edit, animation update or eviction.
        std::vector<c3x_renderer::render_core::SourceShadow::Caster> submission_casters;
        c3x_renderer::render_core::SourceShadow::PreparedCasters prepared_casters;
        if(pickup_profile && !shadow_casters_ptr) {
            if(cancellation && cancellation->load(std::memory_order_relaxed))return false;
            prepared_casters_ptr=prepare_shadow_submission(shadow_buffers_ptr?shadow_buffers_ptr:buffers,submission_casters,prepared_casters);
            shadow_casters_ptr=&submission_casters;
        }
        if(fidelity_profile && !reflection_pass && !scene_surface_pass) {
            ID3D11Resource* destination_resource=nullptr;target->GetResource(&destination_resource);
            ID3D11Texture2D* destination_texture=nullptr;
            HRESULT hr=destination_resource->QueryInterface(__uuidof(ID3D11Texture2D),reinterpret_cast<void**>(&destination_texture));
            destination_resource->Release();if(FAILED(hr))return false;
            D3D11_TEXTURE2D_DESC desc={};destination_texture->GetDesc(&desc);
            if((desc.Width>unsigned(region_size) || desc.Height>unsigned(region_height) || city_profile) && !(city_profile && projection_width==region_size+8 && projection_height==region_height+8)){
                if(!ensure_block_targets()){destination_texture->Release();return false;}
                // Bounded guarded region scratch retains RGBA16F MSAA4,
                // 2x reconstruction and the same pixel-sized glow filter.
                // Each native output pixel is reconstructed once from its own
                // 2x2 linear samples. Integer block origins preserve coverage.
                auto floor_region=[](int value,int phase,int size){return c3x_renderer::render_core::raster_region_floor(value,phase,size);};
                for(auto const&rect:rectangles)for(int y=floor_region(rect.top,grid_y,region_height);y<rect.bottom;y+=region_height)
                    for(int x=floor_region(rect.left,grid_x,region_size);x<rect.right;x+=region_size){
                        if(cancellation && cancellation->load(std::memory_order_relaxed)){destination_texture->Release();return false;}
                        ViewportShaderSettings local=settings;
                        int guard=city_profile?4:0,extent=region_size+guard*2,extent_y=region_height+guard*2;
                        local.translation[0]+=float(guard-x);local.translation[1]+=float(guard-y);
                        local.inverse_size[0]=1.f/float(extent);local.inverse_size[1]=1.f/float(extent_y);
                        int l=std::max(x,int(rect.left)),t=std::max(y,int(rect.top));
                        int r=std::min(x+region_size,int(rect.right)),b=std::min(y+region_height,int(rect.bottom));
                        // Only these pixels are copied out. Preserve four native
                        // pixels around them for the city's +/-8 high-res glow
                        // taps; drawing uses the unchanged 2x/MSAA projection.
                        // A finished bitmap cannot restore MSAA linear color/depth for animation accumulation.
                        bool const region_path=!require_linear_backdrop && world_regions && region_size==128 && !accumulate && !shadow_buffers_ptr &&
                            buffers.is(geometry_vertex_buffers);
                        D3D11_RECT block_rect={0,0,extent,extent_y};
                        if(clip_dirty_blocks && !region_path)block_rect=guarded_block_rectangle({l,t,r,b},guard-x,guard-y,guard,extent,extent_y);
                        c3x_renderer::render_core::RenderRegionKey key;
                        std::vector<std::size_t> sections;
                        bool cacheable=false;
                        if(region_path && !world_regions_control){
                            try{cacheable=render_region_key(buffers,local,*shadow_casters_ptr,prepared_casters_ptr,key,region_diagnostics?&sections:nullptr);}
                            catch(...){key.clear();}
                            if(!cacheable)++frame_region_rejected;
                        }
                        auto cached=cacheable?render_regions.find(key):nullptr;
                        if(region_diagnostics && region_path && cacheable){
                            // Diagnostic hashes never authorize cache reuse. Exact value keys above do.
                            std::uint64_t hashes[7]={};std::size_t begin=0;
                            for(std::size_t part=0;part<sections.size() && part<7;++part){
                                auto hash=14695981039346656037ull;
                                for(std::size_t word=begin;word<sections[part];++word){hash^=key[word];hash*=1099511628211ull;}
                                hashes[part]=hash;begin=sections[part];
                            }
                            char detail[512];sprintf_s(detail,"x=%lld y=%lld hit=%u parts=%zu context=%llu draw=%llu lights=%llu shadow=%llu reflected_draw=%llu reflected_lights=%llu reflected_shadow=%llu screen_x=%d screen_y=%d left=%d top=%d right=%d bottom=%d",
                                static_cast<long long>(x-region_origin_x),static_cast<long long>(y-region_origin_y),cached?1u:0u,sections.size(),
                                hashes[0],hashes[1],hashes[2],hashes[3],hashes[4],hashes[5],hashes[6],x,y,l,t,r,b);
                            trace.write("render-region-dependencies",detail,false);
                        }
                        if(cached){++frame_region_hits;frame_region_hit_pixels+=std::size_t(r-l)*(b-t);}
                        else {
                            if(region_path)++frame_region_misses;
                            if(!submit_geometry(buffers,{block_rect},local,city_profile?active_glow.target:block_target,block_depth,extent,extent_y,cancellation,accumulate,true,shadow_buffers_ptr,false,shadow_casters_ptr,prepared_casters_ptr,region_size,grid_x,grid_y)){
                                destination_texture->Release();return false;
                            }
                            if(cacheable){
                                std::size_t bytes=std::size_t(extent)*extent_y*4;
                                ID3D11Texture2D* image=nullptr;bool admitted=false;
                                if(render_regions.make_room(key,bytes)){
                                    D3D11_TEXTURE2D_DESC image_desc={};image_desc.Width=UINT(extent);image_desc.Height=UINT(extent_y);
                                    image_desc.MipLevels=image_desc.ArraySize=image_desc.SampleDesc.Count=1;
                                    image_desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;image_desc.Usage=D3D11_USAGE_DEFAULT;
                                    if(SUCCEEDED(device->CreateTexture2D(&image_desc,nullptr,&image))){
                                        context->OMSetRenderTargets(0,nullptr,nullptr);
                                        context->CopyResource(image,active_glow.native);
                                        try{admitted=render_regions.insert(std::move(key),image,bytes);}catch(...){}
                                        if(admitted)image=nullptr;
                                    }
                                }
                                if(image)image->Release();
                                if(!admitted)++frame_region_rejected;
                            }
                        }
                        D3D11_BOX box={UINT(l-x+guard),UINT(t-y+guard),0,UINT(r-x+guard),UINT(b-y+guard),1};
                        context->OMSetRenderTargets(0,nullptr,nullptr);
                        context->CopySubresourceRegion(destination_texture,0,UINT(l),UINT(t),0,cached?cached:city_profile?active_glow.native:block_texture,0,&box);
                    }
                destination_texture->Release();return true;
            }
            destination_texture->Release();
        }
        if(pickup_profile && !scene_surface_pass && !(city_profile && region_size==2240)) {
            std::vector<D3D11_RECT> pieces;
            for(auto const& rect:rectangles)
                for(LONG y=rect.top;y<rect.bottom;y+=512)for(LONG x=rect.left;x<rect.right;x+=512)
                    pieces.push_back({x,y,std::min(x+512,rect.right),std::min(y+512,rect.bottom)});
            if(pieces.size()>1) {
                for(std::size_t i=0;i<pieces.size();++i)
                    if(!submit_geometry(buffers,{pieces[i]},settings,target,depth,projection_width,projection_height,
                        cancellation,accumulate || i!=0,finish && i+1==pieces.size(),shadow_buffers_ptr,reflection_pass,shadow_casters_ptr,prepared_casters_ptr,region_size,grid_x,grid_y))return false;
                return true;
            }
        }
        auto draw = [&](GeometryLayer layer) {
            if(reflection_pass){
                if(layer==geometry_underlay || layer==geometry_bed || layer==geometry_water ||
                    layer==geometry_river || layer==geometry_route || layer==geometry_shadow || layer==geometry_wave)return true;
                unsigned provider=layer<geometry_feature?0:layer<geometry_natural_terrain?1:
                    layer<=geometry_natural_decal?2:layer==geometry_natural_mountain?3:4;
                context->VSSetShader(active_reflection.vs[provider],nullptr,0);
                context->PSSetShader(active_reflection.ps[provider],nullptr,0);
            }
            return draw_cached_geometry(layer, buffers, rectangles, settings, cancellation,reflection_pass);
        };
        ID3D11RenderTargetView * destination = target;
        c3x_renderer::render_core::LinearTarget * linear = nullptr;
        // Selected scene passes already name their exact MSAA color/depth
        // destination. Only legacy bitmap submissions acquire an intermediate.
        if (pickup_profile && !scene_surface_pass) {
            ID3D11Resource * resource = nullptr;
            target->GetResource(&resource);
            ID3D11Texture2D * texture = nullptr;
            HRESULT hr = resource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&texture));
            resource->Release();
            if (FAILED(hr)) return false;
            D3D11_TEXTURE2D_DESC desc = {}; texture->GetDesc(&desc); texture->Release();
            linear = reflection_pass?&active_reflection.linear:city_profile?&active_glow.linear:desc.Width == 128 && desc.Height == 128 ? &linear_block : &linear_frame;
            if (!linear->ensure(device, reflection_pass?active_reflection.native_extent*2:desc.Width*(fidelity_profile?2:1), reflection_pass?active_reflection.native_height*2:desc.Height*(fidelity_profile?2:1))) {
                trace.write("linear-target-failed", "pickup MSAA4 allocation", true); return false;
            }
            target = linear->target; depth = linear->depth;
        }
        bool reflection_needed=true;
        if(cull_empty_water && environment_profile && !reflection_pass) {
            reflection_needed=false;
            for(auto const& chunk:buffers[geometry_water]) {
                int dx=chunk.translation_x()+int(settings.translation[0]);
                int dy=chunk.translation_y()+int(settings.translation[1]);
                for(auto const& rect:rectangles)
                    reflection_needed=reflection_needed || !(chunk.bounds().right+dx<=rect.left ||
                        chunk.bounds().left+dx>=rect.right || chunk.bounds().bottom+dy<=rect.top || chunk.bounds().top+dy>=rect.bottom);
            }
        }
        if(environment_profile && !reflection_pass && !scene_surface_pass && active_reflection.enabled && reflection_needed){
            ViewportShaderSettings reflected=settings;
            reflected.translation[0]+=4;reflected.translation[1]+=4;
            int reflected_extent=int(active_reflection.native_extent),reflected_height=int(active_reflection.native_height);
            reflected.inverse_size[0]=1.f/float(reflected_extent);reflected.inverse_size[1]=1.f/float(reflected_height);
            std::vector<D3D11_RECT> reflected_rects={{0,0,reflected_extent,reflected_height}};
            if(clip_dirty_blocks){
                reflected_rects.clear();
                // Mirror coordinates add four native pixels. Another four
                // conservatively cover water's <=3 high-res distortion plus
                // bilinear sampling. Main-pass glow padding is already present.
                for(auto const& rect:rectangles)reflected_rects.push_back(guarded_block_rectangle(rect,4,4,4,reflected_extent,reflected_height));
            }
            if(!submit_geometry(buffers,reflected_rects,reflected,destination,depth,reflected_extent,reflected_height,
                cancellation,false,true,shadow_buffers_ptr,true,shadow_casters_ptr,prepared_casters_ptr,region_size,grid_x,grid_y))return false;
        }
        {AnimationGpu::Pass gpu_phase(animation_gpu,context,AnimationGpu::receivers);
        if(pickup_profile && !scene_surface_pass && !prepare_receiver_shadows(shadow_buffers_ptr?shadow_buffers_ptr:buffers,
                settings,rectangles,reflection_pass,*shadow_casters_ptr,prepared_casters_ptr,cancellation))return false;}

        float clear[4] = {0, 0, 0, 0};
        if(!accumulate) {
            context->ClearRenderTargetView(target, clear);
            context->ClearDepthStencilView(depth, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
        }
        context->OMSetRenderTargets(1, &target, depth);
        context->OMSetDepthStencilState(depth_state, 0);
        float blend_factor[4] = {0, 0, 0, 0};
        context->OMSetBlendState(blend_state, blend_factor, 0xffffffffu);
        context->RSSetState(rasterizer_state);
        D3D11_VIEWPORT viewport = {0.0f, 0.0f, static_cast<float>(projection_width*(fidelity_profile?2:1)), static_cast<float>(projection_height*(fidelity_profile?2:1)), 0.0f, 1.0f};
        context->RSSetViewports(1, &viewport);
        // Always retain a complete terrain surface. The output clip is applied
        // only by c3x_renderer_blit when Civ III composites its dirty rectangle.
        D3D11_RECT scissor = {0, 0, projection_width, projection_height};
        context->RSSetScissorRects(1, &scissor);
        if(environment_profile)active_reflection.bind(context);
        if(city_profile && !scene_surface_pass){
            std::vector<c3x_renderer::city_fidelity::Lighting const*> active;
            for(auto const&chunk:buffers[geometry_city])if(chunk.content().city_lighting){
                auto pointer=chunk.content().city_lighting.get();if(std::find(active.begin(),active.end(),pointer)!=active.end())continue;
                bool intersects=false;int radius=int(std::ceil(shadow_tile_width*.85f))+8;
                for(auto const&r:rectangles)intersects=intersects || chunk_intersects_region(chunk,settings,r,reflection_pass,radius);
                if(intersects)active.push_back(pointer);
            }
            if(!cities.lights(context,active)){trace.write("city-composition-failed","facade block capacity; no truncation",true);return false;}
        }

        auto pass=buffers.pass();
        if (pass.any()) {
            ++frame_pass_setups;frame_active_layers+=pass.count();
            bool base_pass=false,cliff_pass=false;
            for(unsigned layer=0;layer<geometry_natural_terrain;++layer)base_pass=base_pass || pass.has(layer);
            for(unsigned layer=geometry_cliff0;layer<geometry_natural_terrain;++layer)cliff_pass=cliff_pass || pass.has(layer);
            if(base_pass){
                context->IASetInputLayout(input_layout);
                context->VSSetShader(vertex_shader, nullptr, 0);
                context->PSSetShader(pixel_shader, nullptr, 0);
            }
            // Register-for-register match with the frozen approved terrain
            // shader. No production-only palette remains in this binding path.
            auto const& views=compiled_material_views();
            if(base_pass)context->PSSetShaderResources(0, static_cast<UINT>(views.size()), views.data());
            context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ID3D11SamplerState * samplers[] = {fidelity_profile?natural_wrap:terrain_sampler, fidelity_profile?natural_clamp:decal_sampler};
            context->PSSetSamplers(0, 2, samplers);
            context->PSSetConstantBuffers(0, 1, &terrain_settings_buffer);
            context->VSSetConstantBuffers(1, 1, &viewport_settings_buffer);
            if (pickup_profile) {
                context->PSSetConstantBuffers(2, 1, &shadow_settings_buffer);
                context->PSSetConstantBuffers(3, 1, &world_settings_buffer);
                context->PSSetConstantBuffers(4, 1, &source_shadow.table);
                context->PSSetShaderResources(25,1,&source_shadow.view);
            }
            // Geometry buffers are immutable until an authoritative geometry
            // fingerprint changes. Camera-only translation therefore issues
            // draws without regenerating or re-uploading the world vertices.
            if (!draw(geometry_underlay) || !draw(geometry_land))return false;
            if(fidelity_profile){
                context->PSSetShaderResources(17,1,&source_shadow.view);
                // Relief-neighborhood chunks now own their complete terrain
                // surface. Draw them before the independent decal/clutter
                // layer so those assets remain genuinely layered on top.
                unsigned fixed_natural_layers[]={geometry_natural_terrain,
                    geometry_natural_mountain,geometry_natural_decal};
                for(unsigned layer:fixed_natural_layers){
                    if(!pass.has(layer))continue;
                    unsigned provider=layer<=geometry_natural_decal?0:layer==geometry_natural_mountain?1:2;
                    context->OMSetDepthStencilState(layer==geometry_natural_decal?natural.decal_depth:depth_state,0);
                    natural.bind(context,provider,provider==2?layer-geometry_natural_forest0:0);
                    // Terrain and mountain surface shaders also consume these
                    // shared volcanic textures outside the natural t0-t30 pack.
                    context->PSSetShaderResources(69,1,views.data()+69);
                    context->PSSetShaderResources(71,1,views.data()+71);
                    if(provider==0){
                        // The generated face meets these same selected rock
                        // bodies; use their material instead of a brown seam.
                        ID3D11ShaderResourceView* cliff_face[]={cliff_views[0],views[15]};
                        context->PSSetShaderResources(31,2,cliff_face);
                    }
                    if(!draw(static_cast<GeometryLayer>(layer)))return false;
                }
                for(unsigned layer=geometry_natural_forest0;layer<geometry_layer_count;layer++){
                    if(!pass.has(layer))continue;
                    context->OMSetDepthStencilState(depth_state,0);
                    natural.bind(context,2,layer-geometry_natural_forest0);
                    if(!draw(static_cast<GeometryLayer>(layer)))return false;
                }
                if(base_pass){
                context->OMSetDepthStencilState(depth_state,0);
                context->PSSetShaderResources(0,UINT(views.size()),views.data());
                context->PSSetShaderResources(25,1,&source_shadow.view);
                context->PSSetConstantBuffers(0,1,&terrain_settings_buffer);
                context->VSSetShader(vertex_shader,nullptr,0);context->PSSetShader(pixel_shader,nullptr,0);
                }
            }
            if(environment_profile && !reflection_pass){
                ID3D11ShaderResourceView*view=active_reflection.enabled?active_reflection.linear.view:nullptr;
                context->PSSetShaderResources(121,1,&view);
            }
            if (!draw(geometry_bed) ||
                !draw(geometry_water) ||
                !draw(geometry_river) ||
                !draw(geometry_shadow) ||
                !draw(geometry_route)) {
                return false;
            }
            if(!reflection_pass && wave_ready && !buffers[geometry_wave].empty()){
                context->PSSetShader(wave_shader,nullptr,0);
                context->PSSetShaderResources(0,3,wave_views.data());context->PSSetConstantBuffers(7,1,&wave_frame);
                context->OMSetDepthStencilState(natural.decal_depth,0);
                if(!draw(geometry_wave))return false;
                context->OMSetDepthStencilState(depth_state,0);
                context->PSSetShader(pixel_shader,nullptr,0);context->PSSetShaderResources(0,3,views.data());
            }
            if(environment_profile)context->PSSetShaderResources(121,1,views.data()+121);
            if(fidelity_profile){ID3D11SamplerState*retained[]={terrain_sampler,decal_sampler};context->PSSetSamplers(0,2,retained);}
            if(pickup_profile)context->PSSetShaderResources(17,1,&source_shadow.view);
            if (pickup_profile && cliff_pass) {
                context->VSSetShader(feature_vertex_shader, nullptr, 0);
                context->PSSetShader(feature_pixel_shader, nullptr, 0);
                ID3D11SamplerState* cliff_sampler=fidelity_profile?natural_clamp:decal_sampler;
                context->PSSetSamplers(0,1,&cliff_sampler);
                for (unsigned i=0;i<cliff_bundle.assets.size();++i) {
                    if(!pass.has(geometry_cliff0+i))continue;
                    auto texture_index=cliff_bundle.assets[i].texture_index;
                    context->PSSetShaderResources(25,4,cliff_views.data()+texture_index);
                    if (!draw(static_cast<GeometryLayer>(geometry_cliff0+i))) return false;
                }
                context->PSSetSamplers(0,1,&terrain_sampler);
                context->PSSetShaderResources(25,4,feature_texture_views.data());
            }
            if (!buffers[geometry_feature].empty()) {
                context->VSSetShader(feature_vertex_shader, nullptr, 0);
                context->PSSetShader(feature_pixel_shader, nullptr, 0);
                context->PSSetShaderResources(94, 4, feature_texture_views.data() + 4);
                if (!draw(geometry_feature))
                    return false;
            }
            if (!buffers[geometry_site].empty()) {
                context->VSSetShader(feature_vertex_shader,nullptr,0);
                context->PSSetShader(feature_pixel_shader,nullptr,0);
                context->PSSetShaderResources(116,8,site_views.data());
                if(!draw(geometry_site))return false;
            }
            if (!buffers[geometry_mine].empty()) {
                context->VSSetShader(feature_vertex_shader, nullptr, 0);
                context->PSSetShader(feature_pixel_shader, nullptr, 0);
                context->PSSetShaderResources(116, 6, mine_base_views.data());
                context->PSSetShaderResources(124, 2, mine_emissive_views.data());
                if (!draw(geometry_mine))
                    return false;
            }
            if (!buffers[geometry_farm].empty()) {
                context->VSSetShader(feature_vertex_shader, nullptr, 0);
                context->PSSetShader(feature_pixel_shader, nullptr, 0);
                context->PSSetShaderResources(116, 6, farm_base_views.data());
                context->PSSetShaderResources(124, 2, farm_emissive_views.data());
                if (!draw(geometry_farm))
                    return false;
            }
            if (!buffers[geometry_city].empty()) {
                context->VSSetShader(feature_vertex_shader, nullptr, 0);
                context->PSSetShader(feature_pixel_shader, nullptr, 0);
                context->PSSetShaderResources(116, 4, city_emissive_views.data());
                context->PSSetShaderResources(124, 4, city_base_views.data());
                if (!draw(geometry_city))
                    return false;
            }
            if (!buffers[geometry_wall].empty()) {
                std::array<ID3D11ShaderResourceView *, 4> no_emissive = {};
                std::array<ID3D11ShaderResourceView *, 4> wall_views = {
                    wall_texture_view, nullptr, nullptr, nullptr};
                context->VSSetShader(feature_vertex_shader, nullptr, 0);
                context->PSSetShader(feature_pixel_shader, nullptr, 0);
                context->PSSetShaderResources(116, 4, no_emissive.data());
                context->PSSetShaderResources(124, 4, wall_views.data());
                if (!draw(geometry_wall))
                    return false;
            }
        }

        AnimationGpu::Pass gpu_finish(animation_gpu,context,AnimationGpu::finish);
        if(reflection_pass){active_reflection.resolve(context);return true;}
        if(city_profile && linear && finish){
            // Only the experimental strip leaf has no accumulated 512-pixel
            // siblings. Its guarded rectangle contains every copied pixel.
            // Other submissions reconstruct the whole target as before.
            auto dirty=bounded_post && region_size==2240 && !accumulate && rectangles.size()==1?&rectangles[0]:nullptr;
            frame_post_lanes+=active_glow.reconstruct(context,dirty);
            linear_output.draw(context,*linear,destination,display_exposure,1,active_glow.view,active_glow.native_extent,active_glow.native_height,dirty);return true;
        }
        if (linear && finish) linear_output.draw(context, *linear, destination, display_exposure, fidelity_profile?2:1);
        return true;
    }

    void cancel_pixel_preparation() {
        pixel_neighborhood.clear(); pixel_prepare_rects.clear(); pixel_prepare_cursor = 0;
        pixel_prepare_started = false; pixel_readback_pending = false;
        pending_pixel_block = {};
    }

    bool block_phase(std::vector<c3x_renderer::TileFootprint> const & tiles,
                     int tile_width, int tile_height, int & x, int & y) {
        if (tiles.empty() || height > 4096) return false;
        auto const & first = tiles.front();
        auto world_x = static_cast<std::int32_t>(first.coordinate >> 32);
        auto world_y = static_cast<std::int32_t>(first.coordinate);
        x = static_cast<int>((first.anchor_x-static_cast<std::int64_t>(world_x)*tile_width/2) % 128);
        y = static_cast<int>((first.anchor_y-static_cast<std::int64_t>(world_y)*tile_height/2) % 128);
        if (x < 0) x += 128; if (y < 0) y += 128;
        // Preserve derivative-quad phase. Exact contributor keys still validate
        // arbitrary projections and wrapped occurrences; a different grid misses.
        return (x & 1) == 0 && (y & 1) == 0;
    }

    void begin_pixel_neighborhood(c3x_renderer_frame_v1 const & frame) {
        cancel_pixel_preparation();
        // The legacy pixel-block producer uses its own screen-space scratch
        // grid. Do not mix those pixels with the world-grid experiment.
        if(world_raster_grid || shared_scene_surface)return;
        bool has_prefetch=false;
        for(unsigned i=0;i<frame.tile_count;++i) has_prefetch=has_prefetch || (frame.tiles[i].tile_flags & C3X_RENDERER_TILE_PREFETCH) != 0;
        if (!has_prefetch) return;
        if (bitmap_footprint_signature.complete != requested_signature ||
            !block_phase(bitmap_footprints, frame.tile_width, frame.tile_height, pixel_phase_x, pixel_phase_y)) return;
        pixel_neighborhood = bitmap_footprints;
    }

    void start_pixel_preparation() {
        if (pixel_prepare_started || pixel_neighborhood.empty()) return;
        pixel_prepare_started = true;
        std::stable_sort(pixel_neighborhood.begin(), pixel_neighborhood.end(), [](auto const & a, auto const & b) {
            return a.anchor_y != b.anchor_y ? a.anchor_y < b.anchor_y : a.anchor_x < b.anchor_x;
        });
        int left=0,top=0,right=width,bottom=height;
        for (auto const & tile : pixel_neighborhood) {
            left=std::min(left,tile.anchor_x+tile.bounds.left); top=std::min(top,tile.anchor_y+tile.bounds.top);
            right=std::max(right,tile.anchor_x+tile.bounds.right); bottom=std::max(bottom,tile.anchor_y+tile.bounds.bottom);
        }
        left=std::max(left,-512); top=std::max(top,-512);
        right=std::min(right,width+512); bottom=std::min(bottom,height+512);
        for(int y=c3x_renderer::block_floor(top,pixel_phase_y); y<bottom; y+=128)
            for(int x=c3x_renderer::block_floor(left,pixel_phase_x); x<right; x+=128) {
                if (pixel_prepare_rects.size() >= 512u) break;
                c3x_renderer::PixelRect rect{x,y,x+128,y+128};
                // The current bitmap already retains interior pixels. Prepare
                // exposure and changed contributors, not a duplicate viewport.
                if (x>=0 && y>=0 && x+128<=width && y+128<=height &&
                    c3x_renderer::block_key(pixel_neighborhood,rect)==c3x_renderer::block_key(bitmap_footprints,rect)) continue;
                pixel_prepare_rects.push_back(rect);
            }
        // The next exposed strip is close to an existing edge. Distant cells
        // and the already-owned middle cannot starve that work.
        auto priority = [&](c3x_renderer::PixelRect rect) {
            return std::min({std::abs(rect.left),std::abs(rect.right-width),std::abs(rect.top),std::abs(rect.bottom-height)});
        };
        std::stable_sort(pixel_prepare_rects.begin(),pixel_prepare_rects.end(),[&](auto a,auto b){return priority(a)<priority(b);});
    }

    unsigned pixel_work_pending() const {
        return static_cast<unsigned>(pixel_prepare_rects.size()-pixel_prepare_cursor);
    }

    bool ensure_block_targets() {
        if (block_readback && block_target && block_depth) return true;
        release(block_readback); release(block_depth); release(block_depth_texture);
        release(block_target); release(block_texture);
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width=desc.Height=128; desc.MipLevels=desc.ArraySize=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM; desc.SampleDesc.Count=1;
        desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_RENDER_TARGET;
        if (FAILED(device->CreateTexture2D(&desc,nullptr,&block_texture)) ||
            FAILED(device->CreateRenderTargetView(block_texture,nullptr,&block_target))) return false;
        desc.Format=DXGI_FORMAT_D24_UNORM_S8_UINT; desc.BindFlags=D3D11_BIND_DEPTH_STENCIL;
        if (FAILED(device->CreateTexture2D(&desc,nullptr,&block_depth_texture)) ||
            FAILED(device->CreateDepthStencilView(block_depth_texture,nullptr,&block_depth))) return false;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM; desc.BindFlags=0;
        desc.Usage=D3D11_USAGE_STAGING; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        return SUCCEEDED(device->CreateTexture2D(&desc,nullptr,&block_readback));
    }

    bool prepare_pixel_block(std::atomic<bool> const & foreground) {
        if (foreground.load(std::memory_order_relaxed) || !pixel_work_pending()) return true;
        if (pixel_readback_pending) {
            D3D11_MAPPED_SUBRESOURCE mapped = {};
            HRESULT hr = context->Map(block_readback,0,D3D11_MAP_READ,D3D11_MAP_FLAG_DO_NOT_WAIT,&mapped);
            if (hr == DXGI_ERROR_WAS_STILL_DRAWING) return true;
            if (FAILED(hr)) return false;
            for (int y=0;y<128;++y)
                std::memcpy(pending_pixel_block.pixels.data()+y*128,
                    static_cast<unsigned char const *>(mapped.pData)+y*mapped.RowPitch,128*sizeof(std::uint32_t));
            context->Unmap(block_readback,0);
            pixel_readback_pending=false;
            pixel_blocks.insert(std::move(pending_pixel_block));
            ++prepared_blocks; ++pixel_prepare_cursor;
            return true;
        }
        auto rect=pixel_prepare_rects[pixel_prepare_cursor];
        auto key=c3x_renderer::block_key(pixel_neighborhood,rect);
        if (pixel_blocks.find(key)>=0) { ++pixel_prepare_cursor; return true; }
        pending_pixel_block = {};
        pending_pixel_block.key=std::move(key);
        pending_pixel_block.pixels.assign(128*128,0u);
        if (pending_pixel_block.key.empty()) {
            pixel_blocks.insert(std::move(pending_pixel_block)); ++pixel_prepare_cursor; return true;
        }
        if (rect.left>=0 && rect.top>=0 && rect.right<=width && rect.bottom<=height &&
            pending_pixel_block.key==c3x_renderer::block_key(bitmap_footprints,rect)) {
            // This exact contributor set already exists in the immutable CPU
            // bitmap. Copy it into the block cache without any GPU work.
            for(int y=0;y<128;++y)
                std::memcpy(pending_pixel_block.pixels.data()+y*128,
                    pixels.data()+(rect.top+y)*width+rect.left,128*sizeof(std::uint32_t));
            pixel_blocks.insert(std::move(pending_pixel_block)); ++pixel_prepare_cursor; return true;
        }
        if (!ensure_block_targets()) return false;
        // Borrow buffers only for this worker call. No extra retained GPU mesh
        // references escape the cache budget; the D3D command stream owns draws.
        GeometryDrawView::Records buffers;
        for (auto const & contributor : pending_pixel_block.key) {
            auto found=std::find_if(tile_geometry_cache.begin(),tile_geometry_cache.end(),[&](auto const & item){
                return !item.second.shared_natural && item.second.version==contributor.mesh;
            });
            if(found==tile_geometry_cache.end()) { ++pixel_prepare_cursor; return true; }
            for(std::size_t layer=0;layer<geometry_layer_count;++layer)
                for(auto const& source_chunk:found->second.buffers[layer]) {
                    GeometryDrawRecord chunk(source_chunk);
                    chunk.translation_x=contributor.x; chunk.translation_y=contributor.y;
                    buffers[layer].push_back(chunk);
                }
            if(found->second.natural_content.generation){
                auto world_mesh=resident_content.resolve(found->second.natural_content);
                if(!world_mesh || !world_mesh->shared_natural){
                    ++pixel_prepare_cursor;return true;
                }
                c3x_renderer_tile_v1 record={};
                record.tile_x=found->second.tile_x;record.tile_y=found->second.tile_y;
                // Shared owners also contain ground, routes and city parts.
                // The old terrain-only range silently dropped their casters
                // from prepared pixels after those layers migrated to world storage.
                for(std::size_t layer=0;layer<geometry_layer_count;++layer)
                    for(auto const& source_chunk:world_mesh->buffers[layer]){
                        auto chunk=project_natural_chunk(source_chunk,record);
                        chunk.translation_x=contributor.x;chunk.translation_y=contributor.y;
                        buffers[layer].push_back(chunk);
                    }
            }
        }
        ViewportShaderSettings settings={};
        settings.inverse_size[0]=settings.inverse_size[1]=1.0f/128;
        settings.reserved[0]=static_cast<float>(height);
        if (!submit_geometry(buffers,{{0,0,128,128}},settings,block_target,block_depth,128,128,&foreground)) return true;
        context->CopyResource(block_readback,block_texture);
        context->Flush(); // submit once; later polling never waits for the GPU
        pixel_readback_pending=true;
        return true;
    }

    std::uint64_t tile_content_signature(c3x_renderer_tile_v1 const & tile) const {
        std::uint64_t hash = 1469598103934665603ull;
        auto mix = [&hash](void const * data, std::size_t size) {
            auto const * bytes = static_cast<std::uint8_t const *>(data);
            for (std::size_t index = 0; index < size; ++index) {
                hash ^= bytes[index];
                hash *= 1099511628211ull;
            }
        };
        if(patch_pixels){auto identity=patch_detail.identity();mix(&identity,sizeof(identity));}
        if(retained_world){
            auto appearance=SceneTopology::content(tile);mix(&appearance,sizeof(appearance));return hash;
        }
        for (auto value : {tile.terrain_type, tile.real_terrain_type,
                           static_cast<c3x_renderer_i32>(tile.variant_seed),
                           static_cast<c3x_renderer_i32>(tile.feature_flags),
                           static_cast<c3x_renderer_i32>(tile.improvement_flags),
                           static_cast<c3x_renderer_i32>(tile.irrigation_mask),
                           static_cast<c3x_renderer_i32>(tile.has_effect),
                           static_cast<c3x_renderer_i32>(tile.river_code),
                           static_cast<c3x_renderer_i32>(tile.road_mask),
                           static_cast<c3x_renderer_i32>(tile.railroad_mask),
                           tile.route_style, tile.resource_id, tile.resource_class, tile.barbarian_tribe_id,
                           tile.city_id, tile.city_owner_id, tile.city_size,
                           tile.city_culture_group, tile.city_era,
                           static_cast<c3x_renderer_i32>(tile.city_flags)})
            mix(&value, sizeof(value));
        mix(tile.resource_name, sizeof(tile.resource_name));
        return hash;
    }

    std::uint64_t tile_topology_signature(c3x_renderer_tile_v1 const & tile) const {
        // Neighbor sampling consumes terrain, river and route connectivity.
        // Retained units and unrelated city/resource changes do not alter it;
        // a lightweight halo supplies exactly these same authoritative fields.
        std::uint64_t hash = 1469598103934665603ull;
        for (auto value : {tile.terrain_type, tile.real_terrain_type,
                static_cast<c3x_renderer_i32>(tile.river_code),
                static_cast<c3x_renderer_i32>(tile.road_mask),
                static_cast<c3x_renderer_i32>(tile.railroad_mask)})
            hash = (hash ^ static_cast<std::uint32_t>(value)) * 1099511628211ull;
        return hash;
    }

    int canonical_world_component(int value, int size, c3x_renderer_u32 wraps) const {
        if (wraps == 0 || size <= 0)
            return value;
        int result = value % size;
        return result < 0 ? result + size : result;
    }

    std::uint64_t world_tile_key(c3x_renderer_tile_v1 const & tile,
                                 c3x_renderer_frame_v1 const & frame) const {
        int x = canonical_world_component(
            tile.tile_x, frame.world_width_tiles, frame.world_wrap_x);
        int y = canonical_world_component(
            tile.tile_y, frame.world_height_tiles, frame.world_wrap_y);
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 32) |
            static_cast<std::uint32_t>(y);
    }

    bool geometry_matches(CachedGeometry const & candidate,
                          c3x_renderer_frame_v1 const & frame,
                          c3x_renderer::TerrainFrameSignature const & signature,
                          int & translation_x, int & translation_y,
                          bool require_translation) const {
        if (!candidate.valid || signature.geometry != candidate.signature.geometry ||
            frame.tile_count == 0 || frame.tile_count != candidate.tiles.size())
            return false;
        translation_x = frame.tiles[0].anchor_x - candidate.tiles[0].anchor_x;
        translation_y = frame.tiles[0].anchor_y - candidate.tiles[0].anchor_y;
        for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
            c3x_renderer_tile_v1 const & current = frame.tiles[index];
            c3x_renderer_tile_v1 const & cached = candidate.tiles[index];
            if (!same_terrain_content(current, cached) ||
                current.anchor_x - cached.anchor_x != translation_x ||
                current.anchor_y - cached.anchor_y != translation_y)
                return false;
        }
        return !require_translation || translation_x != 0 || translation_y != 0;
    }

    bool same_terrain_content(c3x_renderer_tile_v1 const & left,
                              c3x_renderer_tile_v1 const & right) const {
        return left.tile_x == right.tile_x && left.tile_y == right.tile_y &&
            left.terrain_type == right.terrain_type &&
            left.real_terrain_type == right.real_terrain_type &&
            left.variant_seed == right.variant_seed &&
            left.tile_flags == right.tile_flags &&
            left.feature_flags == right.feature_flags &&
            left.improvement_flags == right.improvement_flags &&
            left.barbarian_tribe_id == right.barbarian_tribe_id &&
            left.irrigation_mask == right.irrigation_mask &&
            left.has_effect == right.has_effect &&
            left.river_code == right.river_code &&
            left.road_mask == right.road_mask &&
            left.railroad_mask == right.railroad_mask &&
            left.route_style == right.route_style &&
            left.resource_id == right.resource_id &&
            left.resource_class == right.resource_class &&
            std::memcmp(left.resource_name, right.resource_name,
                        sizeof(left.resource_name)) == 0 &&
            left.city_id == right.city_id &&
            left.city_owner_id == right.city_owner_id &&
            left.city_size == right.city_size &&
            left.city_culture_group == right.city_culture_group &&
            left.city_era == right.city_era &&
            left.city_flags == right.city_flags;
    }

    bool same_terrain_record(c3x_renderer_tile_v1 const & left,
                             c3x_renderer_tile_v1 const & right) const {
        return left.anchor_x == right.anchor_x && left.anchor_y == right.anchor_y &&
            same_terrain_content(left, right);
    }

    bool reuse_geometry_for_translation(
        c3x_renderer_frame_v1 const & frame,
        c3x_renderer::TerrainFrameSignature const & signature,
        int & translation_x, int & translation_y) const {
        if (pickup_profile && frame.world_topology_revision != geometry_world_revision) return false;
        return geometry_matches(geometry_cache, frame, signature,
                                translation_x, translation_y, true);
    }

    bool reuse_cached_subset(c3x_renderer_frame_v1 const & frame,
                             c3x_renderer::TerrainFrameSignature const & signature) {
        if(world_raster_grid)return false; // Subsets may choose a different wrapped raster origin.
        if (!cache_valid || cached_tiles.empty() ||
            frame.tile_count > cached_tiles.size() ||
            signature.camera != cached_signature.camera ||
            signature.environment != cached_signature.environment ||
            signature.wrap != cached_signature.wrap ||
            content_revision != previous_content_revision ||
            cached_replacement_tile_flags.size() != cached_tiles.size())
            return false;
        std::vector<c3x_renderer_u32> subset_flags;
        subset_flags.reserve(frame.tile_count);
        for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
            bool found = false;
            for (std::size_t cached_index = 0; cached_index < cached_tiles.size(); ++cached_index) {
                if (same_terrain_record(frame.tiles[index], cached_tiles[cached_index])) {
                    subset_flags.push_back(cached_replacement_tile_flags[cached_index]);
                    found = true;
                    break;
                }
            }
            if (!found)
                return false;
        }
        replacement_tile_flags = std::move(subset_flags);
        fallback_tile_indices.clear();
        return true;
    }

    bool render(c3x_renderer_frame_v1 const & frame, c3x_renderer_output_v1 & output,
                int prewarm_index = -1, std::atomic<bool> const * foreground_pending = nullptr,
                std::uint64_t prewarm_signature = 0,
                unsigned const* preparation_indices=nullptr,unsigned preparation_count=0,
                c3x_renderer_frame_v1 const* content_view=nullptr,D3D11_RECT const* output_selection=nullptr) {
        ++scene_generation;
        if(prewarm_index<0)frame_output_readbacks=0;
        if(!gpu_output_mode && cpu_output_stale){cache_valid=false;resource_pixel_signature=0;}
        if(gpu_output_mode)resource_pixel_signature=0; // finish/snapshot the demanded map, never return CPU cache bytes
        selected_output_changed=selected_output_active!=bool(output_selection) ||
            (output_selection && std::memcmp(&selected_output,output_selection,sizeof(*output_selection)));
        if(selected_output_changed)resource_pixel_signature=0;
        selected_output_active=output_selection!=nullptr;selected_output=output_selection?*output_selection:D3D11_RECT{};
        auto const& content_source=content_view?*content_view:frame;
        if(content_view_width!=content_source.target_width || content_view_height!=content_source.target_height){
            cache_valid=false;geometry_cache.clear();clear_resource_backdrops();
        }
        content_view_width=content_source.target_width;content_view_height=content_source.target_height;
        // Source mutation is exclusive; CPU jobs resume once this request has
        // established its world inputs, and may continue while GPU work runs.
        struct PreparationLease {
            c3x_renderer::fidelity::TerrainPreparation& preparation;
            PreparationLease(c3x_renderer::fidelity::TerrainPreparation& p):preparation(p){p.pause();}
            ~PreparationLease(){try{preparation.resume();}catch(...){preparation.clear();}}
        } preparation_lease(terrain_preparation);
        bool const prewarming = prewarm_index >= 0;
        bool const batch_preparing=prewarming && preparation_indices && preparation_count;
        if (prewarming) prepared_footprint = {};
        auto cancelled = [&] { return foreground_pending && foreground_pending->load(std::memory_order_relaxed); };
        if (cancelled()) return false;
        if (prewarming && (static_cast<unsigned>(prewarm_index) >= frame.tile_count ||
            (frame.tiles[prewarm_index].tile_flags & C3X_RENDERER_TILE_PREFETCH) == 0 ||
            !cache_valid || cancelled())) return false;
        if(prewarming)terrain_preparation.resume();
        LARGE_INTEGER started = {}, finished = {};
        QueryPerformanceCounter(&started);
        frame_geometry_ticks = frame_draw_ticks = frame_readback_ticks = 0;
        frame_cache_path = "tiles";
        if (!prewarming) {
        bool selected_surface=scene_surface_requested &&
            c3x_renderer::render_core::scene_surface_extent(frame.target_width,frame.target_height);
        if(selected_surface!=shared_scene_surface){
            reset_targets();clear_resource_backdrops();shared_scene_surface=selected_surface;
        }
        if(gpu_output_mode && (!shared_scene_surface || !city_profile || reflection.enabled))return false;
        ++trace.sequence;
        char profile_option[8]={};
        profiling=GetEnvironmentVariableA("C3X_RENDERER_PROFILE",profile_option,sizeof(profile_option)) &&
            std::strcmp(profile_option,"1")==0;
        frame_draw_calls=frame_parameter_updates=frame_bounds_tests=0;
        draw_parameters.uploads=draw_parameters.records=0;frame_content_uploads=frame_prepared_meshes=frame_foreground_meshes=0;frame_prepared_vertex_bytes=0;
        frame_pass_setups=frame_active_layers=0;
        frame_geometry_issue_ms=frame_scene_select_ms=frame_scene_execute_ms=0;
        frame_caster_preparations=0;frame_post_lanes=0;
        frame_region_hits=frame_region_misses=frame_region_hit_pixels=frame_region_rejected=0;
        frame_region_phase_ms={};frame_tile_validation_ms=frame_tile_append_ms=frame_topology_ms=0;
        frame_center_shore_ms=0;center_shore_cache.hits=center_shore_cache.misses=0;
        memory_sample("frame-start");
        trace.write("render-begin", "", false);
        LARGE_INTEGER load_mark={};QueryPerformanceCounter(&load_mark);
        auto load_phase=[&](char const* stage){
            LARGE_INTEGER now={};QueryPerformanceCounter(&now);
            if(!cache_valid){char detail[96];sprintf_s(detail,"ms=%.3f",trace.milliseconds(now.QuadPart-load_mark.QuadPart));
                trace.write(stage,detail,true);}
            load_mark=now;
        };
        if (!initialize()) {
            trace.write("native-failure","initialize",true);
            return false;
        }
        if (!ensure_targets(frame.target_width, frame.target_height)) {
            trace.write("native-failure","targets",true);
            return false;
        }
        memory_sample("targets-ready");
        if(profiling)gpu_telemetry.poll(context,[&](std::uint64_t sequence,bool valid,double draw_ms,double copy_ms,unsigned skipped){
            char detail[320];sprintf_s(detail,"sample_sequence=%llu valid=%u gpu_draw_ms=%.6f gpu_copy_ms=%.6f frequency=%llu draw_ticks=%llu copy_ticks=%llu ring_skipped=%u",
                sequence,unsigned(valid),draw_ms,copy_ms,gpu_telemetry.last_frequency,gpu_telemetry.last_draw_ticks,gpu_telemetry.last_copy_ticks,skipped);
            trace.write("gpu-timing",detail,true);
        });
        for (int index = 0; index < c3x_renderer::terrain_type_count; ++index)
            if (terrain_textures[index].configured && !ensure_pack_texture(index))
                terrain_textures[index].configured = false;
        load_phase("load-device-terrain");
        c3x_renderer_i64 read_ticks=0,hash_ticks=0,texture_ticks=0;
        std::size_t asset_bytes=0;
        auto read_fidelity=[&](std::string const& relative,std::vector<std::uint8_t>& bytes){
                LARGE_INTEGER begin={},read_end={},hash_end={};QueryPerformanceCounter(&begin);
                char path[4*MAX_PATH];
                bool ok=pack_path(fidelity_root.c_str(),relative.c_str(),path,std::size(path)) && read_file(path,bytes);
                QueryPerformanceCounter(&read_end);
                if(ok){mix_content_revision(bytes);asset_bytes+=bytes.size();}
                QueryPerformanceCounter(&hash_end);
                read_ticks+=read_end.QuadPart-begin.QuadPart;hash_ticks+=hash_end.QuadPart-read_end.QuadPart;return ok;
        };
        auto upload_fidelity=[&](auto const& bytes,auto&view){
            LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
            bool ok=ensure_dds_texture(bytes,view,true,true);
            QueryPerformanceCounter(&end);texture_ticks+=end.QuadPart-begin.QuadPart;return ok;
        };
        if(environment_profile && !wave_attempted){
            wave_attempted=true;std::vector<std::uint8_t> header;
            char control[16]={};GetEnvironmentVariableA("C3X_RENDERER_WAVES",control,sizeof(control));
            if(std::strcmp(control,"0") && read_fidelity("Renderer/packs/CoastalWavesRuntime/waves.bin",header) &&
               header.size()==16 && !std::memcmp(header.data(),"CWV1",4) && read_u32(header,4)==1 && read_u32(header,8)==1){
                wave_ready=true;char const* files[]={"crest.dds","auxiliary.dds","delays.dds"};
                for(unsigned i=0;i<3;++i){std::vector<std::uint8_t> bytes;
                    wave_ready=wave_ready && read_fidelity(std::string("Renderer/packs/CoastalWavesRuntime/")+files[i],bytes) && upload_fidelity(bytes,wave_views[i]);}
            }
            trace.write("coastal-wave-pack",wave_ready?"enabled":"disabled or unavailable; terrain retained",true);
        }
        if(fidelity_profile && !natural.load(device,fidelity_root,read_fidelity,upload_fidelity,city_profile?"city_fidelity":environment_profile?"environment_refresh":"source_fidelity")) {
            trace.write("source-fidelity-failed",natural.failure.c_str(),true);return false;
        }
        load_phase("load-natural");
        if(environment_profile && !reflection.ensure(device,fidelity_root,city_profile?"city_fidelity":"environment_refresh",city_profile?144:136)){
            trace.write("reflection-failed","shader initialization",true);return false;
        }
        load_phase("load-reflection");
        if(city_profile && !cities.load(device,fidelity_root,read_fidelity,upload_fidelity)){
            trace.write("city-composition-failed","pack/material initialization; native fallback",true);return false;
        }
        load_phase("load-city");
        if(asset_bytes){char detail[192];sprintf_s(detail,"bytes=%zu read_ms=%.3f hash_ms=%.3f texture_ms=%.3f",asset_bytes,
            trace.milliseconds(read_ticks),trace.milliseconds(hash_ticks),trace.milliseconds(texture_ticks));trace.write("load-assets",detail,true);}
        if(city_profile && !city_glow.ensure(device,fidelity_root)){trace.write("city-composition-failed","guarded glow initialization",true);return false;}
        if(city_profile && scene_region_size!=128 &&
           (!region_glow.ensure(device,fidelity_root,unsigned(scene_region_size+8),unsigned(scene_region_height+8)) ||
            !region_reflection.ensure(device,fidelity_root,"city_fidelity",unsigned(scene_region_size+16),unsigned(scene_region_height+16)))) {
            trace.write("region-scratch-failed","bounded guarded region initialization",true);return false;
        }
        load_phase("load-glow");
        if (!ensure_terrain_textures()) {
            trace.write("native-failure","terrain-textures",true);
            return false;
        }
        load_phase("load-retained-textures");
        if (pickup_profile) {
            if (!frame.world_topology_count) {
                trace.write("profile-incomplete", "pickup requires authoritative world topology", true);
                return false;
            }
            LARGE_INTEGER begin = {}, end = {}; QueryPerformanceCounter(&begin);
            auto updated = world_coast.update({frame.world_width_tiles, frame.world_height_tiles,
                frame.world_wrap_x != 0, frame.world_wrap_y != 0}, frame.world_topology,
                frame.world_topology_count, frame.world_topology_revision);
            if(fidelity_profile){natural.update_rivers(world_coast.world(),frame.world_topology_revision);
                cliff_query_scratch.bind(natural,world_coast.world(),frame.world_topology_revision);}
            // Ground binds its private scratch under each scoped task lease.
            QueryPerformanceCounter(&end);
            if (updated.cells_built) {
                char detail[256];
                std::snprintf(detail, sizeof(detail), "revision=%lld changed=%zu cells=%zu bytes=%zu ms=%.3f",
                    frame.world_topology_revision, updated.topology_changes, updated.cells_built,
                    updated.bytes, trace.milliseconds(end.QuadPart-begin.QuadPart));
                trace.write("world-coast-update", detail, true);
            }
        }
        }
        frame_tiles_built = frame_tiles_reused = frame_tiles_evicted = frame_instances_ready = frame_patch_index_reuses = frame_world_object_hits = frame_world_object_builds = 0;
        frame_natural_hits=0;
        frame_ground_grid_hits=0;
        char world_control[8]={};GetEnvironmentVariableA("C3X_RENDERER_RETAINED_WORLD",world_control,sizeof(world_control));
        retained_world=std::strcmp(world_control,"0")!=0;
        char patch_control[16]={};GetEnvironmentVariableA("C3X_RENDERER_PATCH_PIXELS",patch_control,sizeof(patch_control));
        unsigned next_patch_pixels=unsigned(std::clamp(std::atoi(patch_control),0,8));
        if(patch_pixels!=next_patch_pixels){patch_pixels=next_patch_pixels;++content_revision;}
        patch_detail=c3x_renderer::fidelity::PatchDetail(frame.tile_width,patch_pixels);
        char cpu_option[8]={};GetEnvironmentVariableA("C3X_RENDERER_CPU_PREPARATION",cpu_option,sizeof(cpu_option));
        char preparation_option[8]={};GetEnvironmentVariableA("C3X_RENDERER_WORLD_PREPARATION",preparation_option,sizeof(preparation_option));
        // Accepted full-detail neighborhood path; zero preserves the reproducible control.
        world_preparation=std::strcmp(preparation_option,"0")!=0 && shared_scene_surface;
        cpu_preparation_budget=(world_preparation?64u:16u)*1024u*1024u;
        unsigned requested_workers=!cpu_option[0]?(world_preparation?4u:2u):std::strcmp(cpu_option,"6")==0?6u:std::strcmp(cpu_option,"4")==0?4u:std::strcmp(cpu_option,"2")==0?2u:std::strcmp(cpu_option,"1")==0?1u:0u;
        if(cpu_terrain_workers!=requested_workers){terrain_preparation.clear();cpu_terrain_workers=requested_workers;}
        bool const cpu_terrain_enabled=cpu_terrain_workers && fidelity_profile && retained_world;
        unsigned active_terrain_workers=cpu_terrain_workers;
        bool const animated_view=frame_has_resource_animation(frame);
        char sharing_control[8]={};
        // Civ III uses a 2:1 diamond; preserve the existing CPU projection for
        // diagnostic/non-native aspect ratios accepted by the renderer API.
        bool const share_world_meshes=frame.tile_height*2==frame.tile_width &&
            !(GetEnvironmentVariableA("C3X_RENDERER_SHARED_NATURAL_CONTROL",sharing_control,sizeof(sharing_control)) &&
            std::strcmp(sharing_control,"1")==0);
        char grid_control[8]={};
        bool const index_natural_grids=!(GetEnvironmentVariableA("C3X_RENDERER_GRID_INDEX_CONTROL",grid_control,sizeof(grid_control)) &&
            std::strcmp(grid_control,"1")==0);
        char relief_control[8]={};
        bool const separate_natural_relief=fidelity_profile &&
            !(GetEnvironmentVariableA("C3X_RENDERER_RELIEF_QUERY_CONTROL",relief_control,sizeof(relief_control)) &&
            std::strcmp(relief_control,"1")==0);
        char ground_control[8]={};
        bool const retain_ground_grids=fidelity_profile && share_world_meshes &&
            !(GetEnvironmentVariableA("C3X_RENDERER_GROUND_GRID_CONTROL",ground_control,sizeof(ground_control)) &&
            std::strcmp(ground_control,"1")==0);
        char instance_control[8]={};GetEnvironmentVariableA("C3X_RENDERER_TREE_INSTANCES_CONTROL",instance_control,sizeof(instance_control));
        bool instance_mode=fidelity_profile && retain_ground_grids && !reflection.enabled && std::strcmp(instance_control,"1")!=0;
        if(tree_instances_enabled!=instance_mode){tree_instances_enabled=instance_mode;++content_revision;}
        natural.instance_stream.bytes=natural.instance_stream.uploads=natural.instance_stream.discards=0;
        source_shadow.instance_stream.bytes=source_shadow.instance_stream.uploads=source_shadow.instance_stream.discards=0;
        char nested_control[8]={};
        bool const reuse_nested_ground_grids=!(GetEnvironmentVariableA("C3X_RENDERER_NESTED_GRID_CONTROL",nested_control,sizeof(nested_control)) &&
            std::strcmp(nested_control,"1")==0);
        char flat_shore_control[8]={};
        bool const skip_flat_shore=!(GetEnvironmentVariableA("C3X_RENDERER_FLAT_SHORE_CONTROL",flat_shore_control,sizeof(flat_shore_control)) &&
            std::strcmp(flat_shore_control,"1")==0);
        char center_control[8]={};
        bool const retain_center_shore=world_regions &&
            !(GetEnvironmentVariableA("C3X_RENDERER_CENTER_SHORE_CONTROL",center_control,sizeof(center_control)) && std::strcmp(center_control,"1")==0);
        char height_cache_control[8]={};
        bool const retain_height_samples=!(GetEnvironmentVariableA("C3X_RENDERER_HEIGHT_CACHE_CONTROL",height_cache_control,sizeof(height_cache_control)) &&
            std::strcmp(height_cache_control,"1")==0);
        // Reuse the existing CPU tier, not another reservation. World-owned
        // natural GPU meshes already survive zoom changes independently.
        if(retain_ground_grids){natural_mesh_cache.clear();natural_mesh_cache_bytes=0;}
        else {ground_grid_cache.clear();ground_grid_cache_bytes=0;}
        frame_upload_bytes = 0;
        raster_reused_pixels = raster_draw_pixels = raster_cached_pixels = 0;
        c3x_renderer::TerrainFrameSignature signature = prewarming ? c3x_renderer::TerrainFrameSignature{} :
            c3x_renderer::terrain_frame_signature(frame, content_revision, device_generation);
        if (!prewarming) requested_signature = signature.complete;
        c3x_renderer_u32 invalidations = 0;
        std::vector<std::uint32_t> restored_viewport;
        CachedViewport const* restored_viewport_entry=nullptr;
        if (!prewarming) {
        if (cache_valid && (!frame_has_resource_animation(frame) || geometry_cache.valid) &&
            signature.complete == cached_signature.complete) {
            if (cache_hits != 0xffffffffu)
                ++cache_hits;
            frame_cache_path = "viewport-current";
            replacement_tile_flags = cached_replacement_tile_flags;
            fallback_tile_indices.clear();
            return fill_output(frame, output, 0, 0);
        }
        {
            for (std::size_t cache_index = 0; !shared_scene_surface && cache_index < viewport_cache.size(); ++cache_index) {
                if (viewport_cache[cache_index].signature.complete != signature.complete)
                    continue;
                if(frame_has_resource_animation(frame)) {
                    // Restore only immutable terrain pixels. Reassemble the
                    // geometry/anchors below before composing current poses.
                    restored_viewport=viewport_cache[cache_index].pixels;
                    restored_viewport_entry=&viewport_cache[cache_index];
                    if(cache_hits!=0xffffffffu)++cache_hits;
                    break;
                }
                frame_cache_path = "viewport-lru";
                CachedViewport hit = std::move(viewport_cache[cache_index]);
                viewport_cache.erase(viewport_cache.begin() + static_cast<std::ptrdiff_t>(cache_index));
                pixels = hit.pixels;
                cached_tiles = hit.tiles;
                cached_replacement_tile_flags = hit.replacement_flags;
                replacement_tile_flags = hit.replacement_flags;
                cached_rendered_tile_count = hit.rendered_tile_count;
                cached_fallback_tile_count = hit.fallback_tile_count;
                cached_textured_tile_count = hit.textured_tile_count;
                cached_visible_animation_count = 0;
                cached_request_continuous_redraw = 0;
                cached_signature = hit.signature;
                cache_valid = true;
                viewport_cache.push_back(std::move(hit));
                fallback_tile_indices.clear();
                if (cache_hits != 0xffffffffu)
                    ++cache_hits;
                return fill_output(frame, output, 0, 0);
            }
        }
        if (!shared_scene_surface && !frame_has_resource_animation(frame) && reuse_cached_subset(frame, signature)) {
            frame_cache_path = "viewport-subset";
            if (cache_hits != 0xffffffffu)
                ++cache_hits;
            return fill_output(frame, output, 0, 0);
        }
        invalidations = invalidations_for(signature);
        if (cache_valid) {
            if (cache_stale_rejections != 0xffffffffu)
                ++cache_stale_rejections;
        }
        }
        // The previous bitmap may belong to a recent-view LRU entry instead of
        // the active geometry owner. Validate its own snapshot before shifting.
        int raster_dx = 0, raster_dy = 0;
        char raster_control[8]={};
        bool const retain_raster=!(GetEnvironmentVariableA("C3X_RENDERER_RASTER_REUSE_CONTROL",raster_control,sizeof(raster_control)) &&
            std::strcmp(raster_control,"1")==0);
        bool const allow_odd_raster=world_raster_grid && fidelity_profile;
        auto origin_phase=[&](c3x_renderer_tile_v1 const& origin){
            return std::array<int,2>{
                c3x_renderer::render_core::raster_anchor_phase(origin.anchor_x,origin.tile_x,frame.tile_width,scene_region_size),
                c3x_renderer::render_core::raster_anchor_phase(origin.anchor_y,origin.tile_y,frame.tile_height,scene_region_height)};
        };
        auto raster_grid_phase=world_raster_grid && frame.tile_count?origin_phase(frame.tiles[0]):std::array<int,2>{};
        auto same_grid=[&](int dx,int dy){
            if(!world_raster_grid)return true;
            if(cached_tiles.empty())return false;
            auto old=origin_phase(cached_tiles[0]);
            // A canonical wrapped origin can jump by a map period which is
            // not divisible by the region extent. Reject that donor's pixels.
            return c3x_renderer::render_core::raster_translation_matches(old[0],raster_grid_phase[0],dx,scene_region_size) &&
                c3x_renderer::render_core::raster_translation_matches(old[1],raster_grid_phase[1],dy,scene_region_height);
        };
        bool reuse_raster = retain_raster && !prewarming && cache_valid && frame.tile_count != 0 &&
            cached_tiles.size() == frame.tile_count && signature.geometry == cached_signature.geometry;
        if (reuse_raster) {
            raster_dx = frame.tiles[0].anchor_x - cached_tiles[0].anchor_x;
            raster_dy = frame.tiles[0].anchor_y - cached_tiles[0].anchor_y;
            // The world-grid experiment preserves local raster coordinates.
            // At 2x reconstruction one native pixel moves two shader pixels,
            // retaining derivative-quad phase even for odd native-pixel pans.
            reuse_raster = (raster_dx != 0 || raster_dy != 0) &&
                same_grid(raster_dx,raster_dy) &&
                (allow_odd_raster || ((raster_dx & 1) == 0 && (raster_dy & 1) == 0)) &&
                std::abs(raster_dx) < width && std::abs(raster_dy) < height;
            for (c3x_renderer_u32 i = 0; reuse_raster && i < frame.tile_count; ++i)
                reuse_raster = same_terrain_content(frame.tiles[i], cached_tiles[i]) &&
                    frame.tiles[i].anchor_x - cached_tiles[i].anchor_x == raster_dx &&
                    frame.tiles[i].anchor_y - cached_tiles[i].anchor_y == raster_dy;
        }
        D3D11_RECT overlap = {std::max(0, raster_dx), std::max(0, raster_dy),
                             std::min(width, width + raster_dx), std::min(height, height + raster_dy)};
        if (!prewarming) raster_rects.clear();
        auto dirty_rect = [&](LONG left, LONG top, LONG right, LONG bottom) {
            if (!prewarming && left < right && top < bottom) {
                raster_rects.push_back({left, top, right, bottom});
                raster_draw_pixels += static_cast<c3x_renderer_u32>((right - left) * (bottom - top));
            }
        };
        if (reuse_raster) {
            dirty_rect(0, 0, width, overlap.top);
            dirty_rect(0, overlap.bottom, width, height);
            dirty_rect(0, overlap.top, overlap.left, overlap.bottom);
            dirty_rect(overlap.right, overlap.top, width, overlap.bottom);
            raster_reused_pixels = static_cast<c3x_renderer_u32>(width * height) - raster_draw_pixels;
        } else dirty_rect(0, 0, width, height);
        int geometry_translation_x = 0;
        int geometry_translation_y = 0;
        bool reuse_geometry = !prewarming && reuse_geometry_for_translation(
            frame, signature, geometry_translation_x, geometry_translation_y);
        if (reuse_geometry) {
            frame_cache_path = "geometry-translation";
            if (cache_hits != 0xffffffffu)
                ++cache_hits;
        } else if (!prewarming && cache_misses != 0xffffffffu) {
            ++cache_misses;
        }
        if (!reuse_geometry && !prewarming) {
            geometry_translation_x = geometry_translation_y = 0;
            clear_geometry_vertex_buffers();
            geometry_cache.clear();
            ++tile_geometry_epoch;
        }

        std::vector<c3x_renderer_u32> prewarm_replacement, prewarm_fallback;
        auto & build_replacement = prewarming ? prewarm_replacement : replacement_tile_flags;
        auto & build_fallback = prewarming ? prewarm_fallback : fallback_tile_indices;
        c3x_renderer_u32 draw_record_count = 0;
        for (c3x_renderer_u32 i = 0; i < content_source.tile_count; ++i)
            if ((content_source.tiles[i].tile_flags & C3X_RENDERER_TILE_TOPOLOGY_HALO) == 0 ||
                (content_source.tiles[i].tile_flags & C3X_RENDERER_TILE_RENDER) != 0) ++draw_record_count;
        int const base_ground_grid = frame.tile_width >= 96 ?
            (draw_record_count <= 768 ? 16 : 12) : 8;
        c3x_renderer_i64 ground_ticks=0,feature_ticks=0,cliff_ticks=0,upload_ticks=0,terrain_prep_ticks=0;
        // One frame-local compile lane, accessed only by the single ground
        // task (or serial control) until its join. No cache survives a frame's
        // topology/asset replacement. River pages stay bounded at two.
        c3x_renderer::fidelity::SurfaceQueryScratch ground_compile_scratch;
        double ground_compile_ms=0,ground_join_ms=0;
        unsigned ground_jobs=0,ground_batch_jobs=0,ground_batch_fallbacks=0;
        unsigned object_instances=0,object_routes=0,city_rigid_parts=0,city_deformed_parts=0,city_fallbacks_omitted=0;
        double ground_schedule_ms=0;
        c3x_renderer::fidelity::GroundPreparation::Statistics ground_batch_before=selected_ground_preparation.statistics();
        std::size_t ground_ready_peak=0;
        char ground_worker_option[8]={};
        bool ground_concurrent=pickup_profile && !(GetEnvironmentVariableA("C3X_RENDERER_GROUND_WORKERS",ground_worker_option,sizeof(ground_worker_option)) &&
            std::strcmp(ground_worker_option,"0")==0);
        std::array<c3x_renderer_i64,6> natural_phase_ticks{};
        LARGE_INTEGER natural_phase_mark={};
        auto begin_natural_phase=[&](){if(profiling)QueryPerformanceCounter(&natural_phase_mark);};
        auto record_natural_phase=[&](unsigned phase){if(profiling){
            LARGE_INTEGER now={};QueryPerformanceCounter(&now);
            natural_phase_ticks[phase]+=now.QuadPart-natural_phase_mark.QuadPart;natural_phase_mark=now;
        }};
        c3x_renderer::render_core::ExactPointCache<c3x_renderer::render_core::ShoreSample> shore_samples;
        c3x_renderer::render_core::ExactPointCache<c3x_renderer::render_core::GroundSample> pickup_ground_samples;
        c3x_renderer::render_core::ExactPointCache<std::array<float,2>> natural_height_samples;
        std::size_t pickup_height_queries=0;
        LARGE_INTEGER phase_time={},phase_end={};
        std::vector<Vertex> underlay_vertices;
        std::vector<Vertex> land_vertices;
        std::vector<Vertex> bed_vertices;
        std::vector<Vertex> water_vertices, wave_vertices;
        std::vector<Vertex> river_vertices;
        std::array<std::vector<UINT>, geometry_river+1> ground_indices;
        std::vector<Vertex> route_vertices;
        std::vector<Vertex> shadow_vertices;
        std::vector<Vertex> feature_vertices;
        std::vector<Vertex> city_vertices;
        std::vector<PendingCityChunk> city_chunks;
        std::vector<Vertex> wall_vertices;
        std::vector<Vertex> mine_vertices;
        std::vector<Vertex> farm_vertices;
        std::vector<Vertex> site_vertices;
        std::array<std::vector<Vertex>,8> cliff_vertices;
        std::array<std::vector<Vertex>,25> natural_vertices;
        std::array<std::vector<UINT>,2> natural_grid_indices;
        std::array<std::vector<Vertex> *, geometry_layer_count> tile_layers = {
            &underlay_vertices, &land_vertices, &bed_vertices, &water_vertices,
            &river_vertices, &route_vertices, &shadow_vertices, &wave_vertices, &feature_vertices,
            &city_vertices, &wall_vertices, &mine_vertices, &farm_vertices, &site_vertices,
            &cliff_vertices[0], &cliff_vertices[1], &cliff_vertices[2], &cliff_vertices[3],
            &cliff_vertices[4], &cliff_vertices[5], &cliff_vertices[6], &cliff_vertices[7]};
        for(unsigned i=0;i<25;i++)tile_layers[geometry_natural_terrain+i]=&natural_vertices[i];
        bool const world_objects=city_profile && retained_world && share_world_meshes &&
            frame.tile_height*2==frame.tile_width;
        bool const world_ground=world_objects && frame.tile_width>=96;
        float half_w = static_cast<float>(frame.tile_width) * 0.5f;
        float half_h = static_cast<float>(frame.tile_height) * 0.5f;
        auto ndc_x = [](float x) { return x; };
        auto ndc_y = [](float y) { return y; };
        c3x_renderer_u32 textured_tile_count = 0;
        c3x_renderer_u32 fallback_tile_count = 0;
        build_fallback.clear();
        build_replacement.assign(frame.tile_count, 0u);
        c3x_renderer::EnvironmentState environment = c3x_renderer::evaluate_environment(
            static_cast<float>(frame.hour), frame.season);
        TerrainShaderSettings frame_settings = {};
        frame_settings.height_texel[0] = 1.0f / 2048.0f;
        frame_settings.height_texel[1] = 1.0f / 2048.0f;
        frame_settings.normal_strength = 4.0f;
        frame_settings.exposure = 1.0f;
        float const * key_light = environment.sun_intensity >= environment.moon_intensity
            ? environment.sun_direction : environment.moon_direction;
        std::copy(key_light, key_light + 3,
                  std::begin(frame_settings.light_direction));
        frame_settings.sun_intensity = environment.sun_intensity;
        std::copy(std::begin(environment.sun_color), std::end(environment.sun_color),
                  std::begin(frame_settings.sun_color));
        frame_settings.shadow_strength = environment.shadow_strength;
        std::copy(std::begin(environment.moon_direction), std::end(environment.moon_direction),
                  std::begin(frame_settings.moon_direction));
        frame_settings.moon_intensity = environment.moon_intensity;
        std::copy(std::begin(environment.moon_color), std::end(environment.moon_color),
                  std::begin(frame_settings.moon_color));
        if(pickup_profile) {
            // Every current world receiver uses the same stylized direction
            // as the atlas, including city/feature diffuse and water specular.
            auto light=c3x_renderer::lighting::key_light(environment);
            std::copy(light.direction.begin(),light.direction.end(),frame_settings.light_direction);
            std::copy(light.direction.begin(),light.direction.end(),frame_settings.moon_direction);
        }
        frame_settings.night_activation = environment.night_activation;
        std::copy(std::begin(environment.ambient_color), std::end(environment.ambient_color),
                  std::begin(frame_settings.ambient_color));
        frame_settings.environment_exposure = environment.exposure;
        frame_settings.water_fresnel = environment.water_fresnel;
        frame_settings.water_specular = environment.water_specular;
        frame_settings.emissive_scale = environment.emissive_scale;
        cities.night=environment.night_activation;cities.emissive_scale=environment.emissive_scale;
        if(city_profile){
            char control[8]={};
            if(GetEnvironmentVariableA("C3X_RENDERER_CITY_LIGHT_CONTROL",control,sizeof(control)) && std::strcmp(control,"1")==0)cities.night=0;
            city_glow.gain=GetEnvironmentVariableA("C3X_RENDERER_CITY_GLOW_CONTROL",control,sizeof(control)) && std::strcmp(control,"1")==0?0.f:6.f;
        }
        frame_settings.hour = static_cast<float>(frame.hour);
        reflection.height_pixels=112.f*.82f*float(frame.tile_width)/224.f;
        reflection.depth_metric=112.f*.0016f*float(content_view_height);
        region_reflection.height_pixels=reflection.height_pixels;region_reflection.depth_metric=reflection.depth_metric;
        region_reflection.enabled=reflection.enabled;region_glow.gain=city_glow.gain;
        ViewportShaderSettings viewport_settings = {};
        viewport_settings.translation[0] =
            static_cast<float>(geometry_translation_x);
        viewport_settings.translation[1] =
            static_cast<float>(geometry_translation_y);
        viewport_settings.depth_translation = float(geometry_translation_y);
        if(city_profile && frame.tile_count){
            auto const& reference=frame.tiles[0];
            auto basis=c3x_renderer::render_core::scene_depth_basis(reference.anchor_y,reference.tile_y,
                frame.tile_height,frame.target_height,geometry_translation_y);
            viewport_settings.depth_translation=basis.translation;
            if(!prewarming)scene_depth_origin=basis.world_origin;
        }
        if (!prewarming) {
            context->UpdateSubresource(terrain_settings_buffer, 0, nullptr, &frame_settings, 0, 0);
            display_exposure = environment.exposure;
            if (pickup_profile) {
                int period = frame.world_wrap_x ? frame.world_width_tiles :
                    frame.world_wrap_y ? frame.world_height_tiles : 0;
                if (frame.world_wrap_x && frame.world_wrap_y) {
                    int a = frame.world_width_tiles, b = frame.world_height_tiles;
                    while (b) { int r = a % b; a = b; b = r; } period = a;
                }
                float world[] = {float(frame.world_width_tiles), float(frame.world_height_tiles),
                    float(frame.world_wrap_x), float(frame.world_wrap_y), float(period)*.5f, 0, 0, 0};
                context->UpdateSubresource(world_settings_buffer, 0, nullptr, world, 0, 0);
                shadow_basis=c3x_renderer::fidelity::light_frame(environment);
                float shadow[20]={};std::copy(shadow_basis.begin(),shadow_basis.end(),shadow);
                shadow[16]=fidelity_profile && fidelity_shadow_control?0.f:1.f;shadow[17]=1;
                if(fidelity_profile){char detail[384];sprintf_s(detail,"authority=r13 adapter=natural-coverage-r3 mountain_mask=1 biome_field=1 coast_coverage=1 coast_height=1 trees=22 recipes=25 weight=180 msaa=4 anisotropy=16 scale=2 mip_bias=-1 scratch_max=3670016 river_pages_max=16 L=%.6f,%.6f,%.6f receive=%.0f",shadow_basis[8],shadow_basis[9],shadow_basis[10],shadow[16]);trace.write("source-fidelity",detail,true);}
                context->UpdateSubresource(shadow_settings_buffer, 0, nullptr, shadow, 0, 0);
                if(fidelity_profile)natural.update(context,environment,shadow_basis.data()+8);
                shadow_tile_width=frame.tile_width;shadow_tile_height=frame.tile_height;
            }
        }
        viewport_settings.inverse_size[0] = 1.0f / c3x_renderer::power_of_two_extent(frame.target_width);
        viewport_settings.inverse_size[1] = 1.0f / c3x_renderer::power_of_two_extent(frame.target_height);
        viewport_settings.reserved[0] = static_cast<float>(content_view_height);
        if (!prewarming) geometry_viewport_settings = viewport_settings;
        if(!prewarming && (world_regions || shared_scene_surface)){
            region_origin_x=frame.tile_count?std::int64_t(frame.tiles[0].anchor_x)-std::int64_t(frame.tiles[0].tile_x)*frame.tile_width/2:0;
            region_origin_y=frame.tile_count?std::int64_t(frame.tiles[0].anchor_y)-std::int64_t(frame.tiles[0].tile_y)*frame.tile_height/2:0;
            // Geometry dependency validation still observes every topology revision.
            // Region validity can then follow selected contributor versions and
            // shadow proofs, instead of invalidating pixels for an unrelated edit.
            region_context={content_revision,device_generation,local_region_revisions?0:std::uint64_t(frame.world_topology_revision),
                signature.environment,signature.wrap,std::uint64_t(width),std::uint64_t(height),
                std::uint64_t(frame.tile_width),std::uint64_t(frame.tile_height),
                std::uint64_t(reflection.enabled),std::uint64_t(fidelity_shadow_control),std::uint64_t(cull_empty_water),std::uint64_t(region_receiver_shadows),std::uint64_t(tight_natural_bounds),std::uint64_t(region_input_ring)};
            append_region_bytes(region_context,&frame_settings,sizeof(frame_settings));
            append_region_bytes(region_context,shadow_basis.data(),sizeof(shadow_basis));
            float response[]={display_exposure,city_glow.gain,cities.night,cities.emissive_scale,reflection.height_pixels,reflection.depth_metric};
            append_region_bytes(region_context,response,sizeof(response));
        }
        if(!reuse_geometry && restored_viewport_entry &&
           restore_viewport_geometry(*restored_viewport_entry,frame,signature)){
            reuse_geometry=true;
            if(cache_misses) --cache_misses;
        }
        if (!reuse_geometry) {
        if(!prewarming)geometry_cache.tile_keys.assign(frame.tile_count,{});
        auto canonical_component = [](int value, int size, c3x_renderer_u32 wraps) {
            if (wraps == 0 || size <= 0)
                return value;
            int result = value % size;
            return result < 0 ? result + size : result;
        };
        auto coordinate_key = [&frame, &canonical_component](int x, int y) {
            x = canonical_component(x, frame.world_width_tiles, frame.world_wrap_x);
            y = canonical_component(y, frame.world_height_tiles, frame.world_wrap_y);
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 32) |
                static_cast<std::uint32_t>(y);
        };
        auto & river_nodes = topology_cache.rivers;
        std::uint64_t topology_signature = prewarming ? prewarm_signature : signature.complete;
        // All lookup payloads are owned; a new snapshot address does not change
        // appearance. The complete capture signature still guards view changes.
        if (topology_cache.signature != topology_signature) {
        auto topology_started=std::chrono::steady_clock::now();
        topology_cache.signature = 0;
        if(!topology_cache.begin(frame))return false;
        river_nodes.clear();
        for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
            if ((index & 63u) == 0 && cancelled()) return false;
            c3x_renderer_tile_v1 const & tile = frame.tiles[index];
            if ((tile.tile_flags & (C3X_RENDERER_TILE_RENDER |
                                    C3X_RENDERER_TILE_TOPOLOGY_HALO)) == 0)
                continue;
            int ground = ground_type(tile);
            int relief = relief_type(tile);
            if (relief == 10 && !volcano_assets_ready)relief = -1;
            ground=ground>=0 && ground<c3x_renderer::terrain_type_count && terrain_textures[ground].view?ground:-1;
            relief=relief>=0 && relief<c3x_renderer::terrain_type_count && terrain_textures[relief].view?relief:-1;
            int surface=marsh_assets_ready && tile.real_terrain_type==9 && terrain_textures[9].view?9:
                relief>=0?relief:ground;
            if(!topology_cache.update(tile,ground,relief,surface,tile_topology_signature(tile)))return false;
        }
        std::unordered_map<std::uint64_t, std::size_t> river_index;
        auto river_key = [](int x, int y) {
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 32) | static_cast<std::uint32_t>(y);
        };
        auto river_node_at = [&](int x, int y) -> RiverNode & {
            auto inserted = river_index.try_emplace(river_key(x,y),river_nodes.size());
            if (inserted.second) river_nodes.push_back(RiverNode{x,y,0u,false});
            return river_nodes[inserted.first->second];
        };
        auto add_river_edge = [&river_node_at](int start_x, int start_y,
                                               int endpoint_x, int endpoint_y) {
            river_node_at(start_x, start_y).degree += 1u;
            river_node_at(endpoint_x, endpoint_y).degree += 1u;
        };
        for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
            if ((index & 63u) == 0 && cancelled()) return false;
            c3x_renderer_tile_v1 const & tile = frame.tiles[index];
            if ((tile.tile_flags & (C3X_RENDERER_TILE_RENDER |
                                    C3X_RENDERER_TILE_TOPOLOGY_HALO)) == 0)
                continue;
            unsigned mask = tile.river_code & 170u;
            if ((mask & 2u) != 0)
                add_river_edge(tile.tile_x, tile.tile_y - 1,
                               tile.tile_x + 1, tile.tile_y);
            if ((mask & 8u) != 0)
                add_river_edge(tile.tile_x + 1, tile.tile_y,
                               tile.tile_x, tile.tile_y + 1);
        }
        constexpr int river_corner_offsets[4][2] = {
            {0, -1}, {1, 0}, {0, 1}, {-1, 0}
        };
        for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
            if ((index & 63u) == 0 && cancelled()) return false;
            c3x_renderer_tile_v1 const & tile = frame.tiles[index];
            if ((tile.tile_flags & (C3X_RENDERER_TILE_RENDER |
                                    C3X_RENDERER_TILE_TOPOLOGY_HALO)) == 0 ||
                ground_type(tile) < 11)
                continue;
            for (auto const & offset : river_corner_offsets) {
                auto found = river_index.find(river_key(tile.tile_x+offset[0],tile.tile_y+offset[1]));
                if (found != river_index.end()) river_nodes[found->second].touches_water = true;
            }
        }
        topology_cache.finish();
        topology_cache.signature = topology_signature;
        char scene_detail[160];sprintf_s(scene_detail,"records=%zu bytes=%zu cap=%zu",
            topology_cache.size(),topology_cache.bytes(),topology_cache.record_limit);
        trace.write("captured-scene",scene_detail,true);
        frame_topology_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-topology_started).count();
        }
        char prefetch_foreground_control[8]={};
        bool const offload_prefetch=pickup_profile && !prewarming &&
            GetEnvironmentVariableA("C3X_RENDERER_PREFETCH_FOREGROUND_CONTROL",
                                   prefetch_foreground_control,sizeof(prefetch_foreground_control)) &&
            (std::strcmp(prefetch_foreground_control,"1")==0 ||
             std::strcmp(prefetch_foreground_control,"2")==0);
        int const prefetch_guard_tiles=std::strcmp(prefetch_foreground_control,"2")==0?2:0;
        auto select_river_nodes=[&](c3x_renderer_tile_v1 const& tile){
            std::vector<RiverNode const*> local_river_nodes;
            if (river_assets_ready && (tile.river_code & 170u) != 0) {
                // The shader's source/junction/mouth responses vanish by 24 px.
                // Include that radius plus the whole tile rectangle's diameter:
                // if any point responds, its nearest node is closer than every
                // omitted node at ALL vertices, preserving interpolation too.
                c3x_renderer::RiverNodeWindow node_window(world_ground?96:frame.tile_width, world_ground?48:frame.tile_height);
                for (RiverNode const & node : river_nodes)
                    if (node_window.contains(node.lattice_x-tile.tile_x, node.lattice_y-tile.tile_y))
                        local_river_nodes.push_back(&node);
                std::sort(local_river_nodes.begin(), local_river_nodes.end(), [](auto a, auto b) {
                    return a->lattice_y != b->lattice_y ? a->lattice_y < b->lattice_y : a->lattice_x < b->lattice_x;
                });
            }
            return local_river_nodes;
        };
        auto river_context_for=[](auto const& nodes){
            std::uint64_t river_context=1469598103934665603ull;
            auto mix=[&](auto value){auto bytes=reinterpret_cast<std::uint8_t const*>(&value);
                for(std::size_t i=0;i<sizeof(value);++i)river_context=(river_context^bytes[i])*1099511628211ull;};
            for(auto node:nodes){mix(node->lattice_x);mix(node->lattice_y);mix(node->degree);mix(node->touches_water);}
            return river_context;
        };
        auto compile_context_for=[&](c3x_renderer_tile_v1 const& tile,std::uint64_t river_context){
            auto persistent_instance=topology_cache.retained(coordinate_key(tile.tile_x,tile.tile_y));
            return std::array<std::uint64_t,20>{std::uint64_t(tile.tile_x),std::uint64_t(tile.tile_y),std::uint64_t(world_ground?0:content_view_width),std::uint64_t(world_ground?0:content_view_height),std::uint64_t(world_ground?96:frame.tile_width),std::uint64_t(world_ground?48:frame.tile_height),std::uint64_t(frame.world_width_tiles),std::uint64_t(frame.world_height_tiles),std::uint64_t(frame.world_wrap_x),std::uint64_t(frame.world_wrap_y),std::uint64_t(content_revision),std::uint64_t(device_generation),std::uint64_t(pickup_profile?0:frame.hour),std::uint64_t(pickup_profile?0:frame.season),(std::uint64_t(world_ground?0:base_ground_grid)<<32)|patch_detail.identity(),std::uint64_t(world_ground?0:draw_record_count<=512?0:draw_record_count<=768?1:draw_record_count<=2048?2:3),std::uint64_t(river_context),std::uint64_t(persistent_instance?persistent_instance->revision:0),std::uint64_t(c3x_renderer::render_core::render_core_revision),std::uint64_t(pickup_profile)};
        };
        auto selected_tile=[&](unsigned index){
            auto const& tile=frame.tiles[index];
            bool const guarded_prefetch=prefetch_guard_tiles!=0 &&
                tile.anchor_x+frame.tile_width>=-prefetch_guard_tiles*frame.tile_width &&
                tile.anchor_x<=frame.target_width+prefetch_guard_tiles*frame.tile_width &&
                tile.anchor_y+frame.tile_height>=-prefetch_guard_tiles*frame.tile_height &&
                tile.anchor_y<=frame.target_height+prefetch_guard_tiles*frame.tile_height;
            if(prewarming ? ((tile.tile_flags&C3X_RENDERER_TILE_PREFETCH)==0 || (!batch_preparing && static_cast<int>(index)!=prewarm_index)) :
                (tile.tile_flags & (C3X_RENDERER_TILE_RENDER |
                    (pickup_profile && (!offload_prefetch || guarded_prefetch)?C3X_RENDERER_TILE_PREFETCH:0)))==0)return false;
            if(!prewarming && pickup_profile && !(tile.tile_flags&C3X_RENDERER_TILE_RENDER)){
                int mx=frame.tile_width*region_input_ring,my=frame.tile_height*region_input_ring;
                if(tile.anchor_x+frame.tile_width < -mx || tile.anchor_x>frame.target_width+mx ||
                   tile.anchor_y+frame.tile_height < -my || tile.anchor_y>frame.target_height+my)return false;
            }
            return true;
        };
        auto ground_observations=topology_cache.observation_view();
        auto make_ground_job=[&](c3x_renderer_tile_v1 const& tile,std::vector<RiverNode const*> const& nodes){
            c3x_renderer::fidelity::GroundPreparationInput job;
            auto& input=job.compile;
            input.tile=tile;input.world_ground=world_ground;input.pickup_profile=pickup_profile;input.fidelity_profile=fidelity_profile;
            input.draw_marsh=marsh_assets_ready && tile.real_terrain_type==9;
            input.river_assets_ready=river_assets_ready;input.retain_ground_grids=retain_ground_grids;
            input.reuse_nested_ground_grids=reuse_nested_ground_grids;input.prewarming=prewarming;
            input.ground=ground_type(tile);input.half_w=half_w;input.half_h=half_h;
            input.uv_scale=.26f;input.relief_projection_scale=float(frame.tile_width)/224.f*.82f;
            std::copy(key_light,key_light+3,input.key_light);
            job.tile_width=frame.tile_width;job.tile_height=frame.tile_height;
            job.world_width=frame.world_width_tiles;job.world_height=frame.world_height_tiles;
            job.wrap_x=frame.world_wrap_x!=0;job.wrap_y=frame.world_wrap_y!=0;job.topology_revision=frame.world_topology_revision;
            job.skip_flat_shore=skip_flat_shore;job.separate_natural_relief=separate_natural_relief;
            job.nodes.reserve(nodes.size());for(auto node:nodes)job.nodes.push_back(*node);
            c3x_renderer::render_core::ExactPointCache<c3x_renderer::render_core::ShoreSample> samples;
            auto ignore=[](auto,auto){};
            c3x_renderer::fidelity::SurfaceQueries<decltype(ignore),decltype(ignore)> queries(world_coast,samples,tile.tile_x,tile.tile_y,ignore,ignore,skip_flat_shore);
            if(retain_center_shore){
                try{job.center=center_shore_cache.get(world_coast,tile.tile_x,tile.tile_y,ignore,ignore);}
                catch(...){job.center=queries.shore(queries.center_u,queries.center_v);}
            }else job.center=queries.shore(queries.center_u,queries.center_v);
            int relief=relief_type(tile);
            if((relief==10 && !volcano_assets_ready) || relief<0 || relief>=c3x_renderer::terrain_type_count || !terrain_textures[relief].view)relief=-1;
            input.tile_ground_grid=c3x_renderer::fidelity::ground_grid_detail(tile,relief,
                dune_assets_ready && tile.real_terrain_type==0 && tile.terrain_type==0,pickup_profile,frame.tile_width,draw_record_count,
                [&](int x,int y){return ground_observations.current(ground_observations.key(x,y));},
                [&](int c,int r){return world_coast.world().tile(c,r);});
            input.flat_grid=std::abs(job.center.distance)<1.5 && frame.tile_width>=96?16:8;
            input.shadow_grid=frame.tile_width>=96 && draw_record_count<=512?16:8;
            return job;
        };
        c3x_renderer::objects::Assets object_assets{{&bridge_bundle,&site_bundle,&mine_bundle,&farm_bundle,&city_bundle,&wall_bundle}};
        bool const prepare_objects=fidelity_profile && world_objects;
        char object_control[8]={};
        bool const object_worker=prepare_objects && !prewarming &&
            !(GetEnvironmentVariableA("C3X_RENDERER_OBJECT_WORKERS",object_control,sizeof(object_control)) && std::strcmp(object_control,"0")==0);
        bool const world_batch_enabled=cpu_terrain_enabled && world_ground && retain_ground_grids && ground_concurrent && object_worker;
        c3x_renderer::WorldPreparationLease world_lease;
        unsigned world_jobs=0,world_recovery=0;double world_join_ms=0;
        double world_ground_ms=0,world_terrain_ms=0,world_object_ms=0,world_upload_ms=0;
        std::size_t world_gpu_bytes=0;
        bool const ground_batch_enabled=!world_batch_enabled && ground_concurrent && fidelity_profile && world_ground;
        std::vector<bool> ground_queued(frame.tile_count,false);
        // This scope joins before the immutable observation/asset lease ends.
        // attach() changes only the separate resident-handle map during adoption.
        c3x_renderer::fidelity::GroundPreparationLease ground_batch_lease(selected_ground_preparation);
        if(ground_batch_enabled){
            auto schedule_started=std::chrono::steady_clock::now();
            std::deque<c3x_renderer::fidelity::GroundPreparation::Job> jobs;
            for(unsigned slot=0;slot<(batch_preparing?preparation_count:frame.tile_count);++slot){
                unsigned index=batch_preparing?preparation_indices[slot]:slot;
                if(index>=frame.tile_count)return false;
                if(!selected_tile(index))continue;
                if(cancelled())return false;
                auto const& tile=frame.tiles[index];int ground=ground_type(tile);
                if(ground<0 || ground>=c3x_renderer::terrain_type_count || !terrain_textures[ground].view)continue;
                auto nodes=select_river_nodes(tile);
                auto expected=compile_context_for(tile,river_context_for(nodes));
                auto instance=topology_cache.retained(coordinate_key(tile.tile_x,tile.tile_y));
                bool resident=false;
                if(instance)for(auto handle:instance->compiled_views){
                    auto cached=resident_content.resolve(handle);
                    if(cached && cached->compile_context==expected && tile_content_valid(*cached,tile)){resident=true;break;}
                }
                if(resident)continue;
                jobs.push_back({index,make_ground_job(tile,nodes)});ground_queued[index]=true;
            }
            ground_batch_jobs=unsigned(jobs.size());
            // Share the existing preparation allowance while both systems have
            // selected work; do not place two busy lanes on top of four terrain
            // lanes. Explicit low-concurrency terrain controls remain usable.
            if(!prewarming && ground_batch_jobs && cpu_terrain_workers>2)active_terrain_workers=cpu_terrain_workers-2;
            ground_batch_lease.start(std::move(jobs),[this,ground_observations,foreground_pending](auto const& input,auto& scratch,auto const& stop){
                auto cancelled=[&]{return stop.load(std::memory_order_relaxed) ||
                    (foreground_pending && foreground_pending->load(std::memory_order_relaxed));};
                return c3x_renderer::fidelity::compile_selected_ground(input,natural,world_coast,ground_observations,terrain_textures,scratch,cancelled);
            });
            ground_schedule_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-schedule_started).count();
        }
        c3x_renderer::objects::PreparationLease object_lease;
        c3x_renderer::fidelity::TerrainCompileScratch foreground_object_scratch;
        unsigned object_jobs=0,object_recovery=0;double object_join_ms=0;
        std::size_t object_gpu_bytes=0;
        auto make_object_job=[&](c3x_renderer_tile_v1 const& tile){
            c3x_renderer::objects::PreparationInput input;
            auto& projection=input.projection;projection.tile=tile;
            projection.tile_width=frame.tile_width;projection.content_view_height=frame.target_height;
            projection.half_w=half_w;projection.half_h=half_h;
            projection.relief_projection_scale=float(frame.tile_width)/224.f*.82f;
            projection.feature_projection_scale=float(frame.tile_width)/224.f;
            projection.pickup_profile=true;projection.world_objects=true;
            input.ground=ground_type(tile);input.world_revision=frame.world_topology_revision;
            input.river_ready=river_assets_ready;input.skip_flat_shore=skip_flat_shore;
            input.separate_relief=separate_natural_relief;input.retain_height=retain_height_samples;
            input.route_ready=route_assets_ready;input.mine_ready=mine_assets_ready;input.farm_ready=farm_assets_ready;
            input.city_ready=city_assets_ready;input.composition_ready=cities.ready;
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
            input.routes_enabled=diagnostic_routes!=2;
#endif
            return input;
        };
        auto compile_objects=[&](auto const& input,auto& scratch,auto stop,bool bounded=false){
            auto result=c3x_renderer::objects::prepare(input,object_assets,cities.library,natural,terrain_textures,
                world_coast,ground_observations,scratch,stop,bounded);
            if(result && !stop() && !attach_object_buffer(*result))result.reset();
            return result;
        };
        // The callback borrows compile_objects and frame sources. Join before
        // those locals are destroyed, including every early return/exception.
        struct ObjectJoin {
            c3x_renderer::objects::Preparation& queue;
            c3x_renderer::fidelity::GroundPreparation& ground;
            ~ObjectJoin(){ground.set_ready_notification({});queue.set_ready_notification({});queue.clear();}
        } object_join{object_lease.queue,selected_ground_preparation};
        if(object_worker && !world_batch_enabled){
            std::deque<c3x_renderer::objects::Preparation::Job> jobs;
            for(unsigned index=0;index<frame.tile_count;++index){
                if(!selected_tile(index))continue;
                auto const& tile=frame.tiles[index];int ground=ground_type(tile);
                if(ground<0 || ground>=c3x_renderer::terrain_type_count || !terrain_textures[ground].view)continue;
                auto nodes=select_river_nodes(tile);auto expected=compile_context_for(tile,river_context_for(nodes));
                auto instance=topology_cache.retained(coordinate_key(tile.tile_x,tile.tile_y));bool resident=false;
                if(instance)for(auto handle:instance->compiled_views){auto cached=resident_content.resolve(handle);
                    if(cached && cached->compile_context==expected && tile_content_valid(*cached,tile)){resident=true;break;}}
                if(!resident)jobs.push_back({index,make_object_job(tile)});
            }
            object_jobs=unsigned(jobs.size());
            if(object_jobs && active_terrain_workers>1)--active_terrain_workers;
            object_lease.queue.configure(std::move(jobs),[&](auto const& input,auto const& stop,unsigned){
                return compile_objects(input,object_lease.scratch,[&]{return stop.load(std::memory_order_relaxed) || cancelled();},true);
            });
            object_lease.queue.resume();
        }
        if(cpu_terrain_enabled && !prewarming && !world_batch_enabled){
            using Preparation=c3x_renderer::fidelity::TerrainPreparation;
            try {
            std::deque<Preparation::Job> jobs;
            std::vector<c3x_renderer::fidelity::TerrainCompileInput::Key> needed;
            std::set<c3x_renderer::fidelity::TerrainCompileInput::Key> keys;
            for(unsigned i=0;i<frame.tile_count;++i){
                auto const& tile=frame.tiles[i];
                if(!(tile.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)))continue;
                auto instance=topology_cache.retained(coordinate_key(tile.tile_x,tile.tile_y));
                auto cached=instance?resident_content.resolve(instance->compiled):nullptr;
                if(cached && cached->compile_context[17]==instance->revision && tile_content_valid(*cached,tile))continue;
                auto input=terrain_compile_input(tile,frame,ground_type(tile),skip_flat_shore,separate_natural_relief,index_natural_grids,retain_height_samples,world_objects);
                if(tile.tile_flags&C3X_RENDERER_TILE_RENDER)needed.push_back(input.key);
                if(keys.insert(input.key).second && !terrain_preparation.contains(input.key,
                    [&](auto const& result){return terrain_result_valid(result);}))jobs.push_back({input.key,input});
            }
                terrain_preparation.configure(std::move(jobs),[this,foreground_pending](auto const& input,auto const& stop,unsigned worker){
                    auto cancelled=[&]{return stop.load(std::memory_order_relaxed) ||
                        (foreground_pending && foreground_pending->load(std::memory_order_relaxed));};
                    auto result=compile_terrain(input,terrain_scratch[worker],cancelled);
                    if(result && !attach_terrain_vertex_buffer(*result))result.reset();
                    return result;
                },active_terrain_workers,std::move(needed),cpu_preparation_budget);
                terrain_preparation.resume();
            }catch(...){terrain_preparation.clear();}
        }
        if(cpu_terrain_enabled && !prewarming && prepare_objects && !world_batch_enabled){
            // Return finished lanes at producer completion. Waiting for the
            // next foreground tile can park available workers behind a long
            // terrain compile/join. Notifications touch only synchronized queue
            // state; ObjectJoin unregisters and joins them before local owners
            // disappear. The existing total worker allowance does not grow.
            auto return_lanes=[this,&object_lease,total=cpu_terrain_workers]{
                try {
                auto ground=selected_ground_preparation.statistics();auto objects=object_lease.queue.statistics();
                unsigned reserved=(ground.pending || ground.active?2u:0u)+(objects.pending || objects.active?1u:0u);
                terrain_preparation.expand_workers(total>reserved?total-reserved:1u);
                }catch(...){/* Optional expansion: existing workers and serial recovery remain available. */}
            };
            selected_ground_preparation.set_ready_notification(return_lanes);
            object_lease.queue.set_ready_notification(return_lanes);
            return_lanes(); // includes a producer that finished before registration
        }
        auto compile_world=[&](auto const& input,auto& ground_scratch,auto& surface_scratch,auto stop,bool bounded){
            auto result=std::make_unique<c3x_renderer::PreparedWorld>();
            auto begin=std::chrono::steady_clock::now();
            auto elapsed=[&]{auto now=std::chrono::steady_clock::now();
                double ms=std::chrono::duration<double,std::milli>(now-begin).count();begin=now;return ms;};
            result->ground=c3x_renderer::fidelity::compile_selected_ground(input.ground,natural,world_coast,
                ground_observations,terrain_textures,ground_scratch,stop);
            result->ground_ms=elapsed();if(!result->ground || stop())return std::unique_ptr<c3x_renderer::PreparedWorld>{};
            result->terrain=compile_terrain(input.terrain,surface_scratch,stop,bounded);
            result->terrain_ms=elapsed();if(!result->terrain || stop())return std::unique_ptr<c3x_renderer::PreparedWorld>{};
            result->objects=c3x_renderer::objects::prepare(input.objects,object_assets,cities.library,natural,
                terrain_textures,world_coast,ground_observations,surface_scratch,stop,bounded);
            result->object_ms=elapsed();if(!result->objects || stop())return std::unique_ptr<c3x_renderer::PreparedWorld>{};
            c3x_renderer::render_core::ImmutableMeshUpload upload;
            auto append=[&](auto const& mesh,unsigned& vertices,unsigned& indices){if(mesh.empty())return;
                vertices=upload.append(mesh.vertices.data(),mesh.vertices.size());
                if(!mesh.shared_grid && !mesh.indices.empty())indices=upload.append(mesh.indices.data(),mesh.indices.size());};
            for(unsigned i=0;i<6;++i){
                // Bed/water adopt the underlay's identical ranges on the render owner.
                if(i==2 || i==3)continue;
                append(result->ground->meshes[i],result->ground_vertices[i],result->ground_indices[i]);
            }
            for(unsigned i=0;i<3;++i)append(result->terrain->meshes[i],result->terrain_vertices[i],result->terrain_indices[i]);
            for(auto& part:result->objects->layers)append(part.mesh,part.vertex_offset,part.index_offset);
            for(auto& part:result->objects->city)append(part.mesh,part.vertex_offset,part.index_offset);
            if(stop() || (bounded && result->bytes()+upload.size()>c3x_renderer::WorldPreparation::byte_limit))
                return std::unique_ptr<c3x_renderer::PreparedWorld>{};
            ID3D11Buffer* buffer=nullptr;if(!upload.create(device,&buffer))return std::unique_ptr<c3x_renderer::PreparedWorld>{};
            if(buffer)result->buffer=std::shared_ptr<void>(buffer,[](void* p){static_cast<ID3D11Buffer*>(p)->Release();});
            result->gpu_bytes=upload.size();result->objects->buffer=result->buffer;
            result->upload_ms=elapsed();return result;
        };
        auto make_world_job=[&](auto const& tile,auto const& nodes){
            return c3x_renderer::WorldPreparationInput{make_ground_job(tile,nodes),
                terrain_compile_input(tile,frame,ground_type(tile),skip_flat_shore,separate_natural_relief,index_natural_grids,retain_height_samples,world_objects),
                make_object_job(tile)};
        };
        // Compile callbacks borrow the local source lease. Clear before those
        // callbacks/inputs die on all exits, not merely normal publication.
        struct WorldJoin {c3x_renderer::WorldPreparation& queue;~WorldJoin(){queue.clear();}} world_join{world_lease.queue};
        if(world_batch_enabled){
            terrain_preparation.clear(); // superseded producers must not compete for this request
            std::deque<c3x_renderer::WorldPreparation::Job> jobs;
            for(unsigned index=0;index<frame.tile_count;++index){
                if(!selected_tile(index))continue;
                auto const& tile=frame.tiles[index];int ground=ground_type(tile);
                if(ground<0 || ground>=c3x_renderer::terrain_type_count || !terrain_textures[ground].view)continue;
                auto nodes=select_river_nodes(tile);auto expected=compile_context_for(tile,river_context_for(nodes));
                auto instance=topology_cache.retained(coordinate_key(tile.tile_x,tile.tile_y));bool resident=false;
                if(instance)for(auto handle:instance->compiled_views){auto cached=resident_content.resolve(handle);
                    if(cached && cached->compile_context==expected && tile_content_valid(*cached,tile)){resident=true;break;}}
                if(!resident)jobs.push_back({index,make_world_job(tile,nodes)});
            }
            world_jobs=unsigned(jobs.size());
            world_lease.queue.configure(std::move(jobs),[&](auto const& input,auto const& stop,unsigned worker){
                return compile_world(input,world_ground_scratch[worker],terrain_scratch[worker],[&]{return stop.load(std::memory_order_relaxed) || cancelled();},true);
            },cpu_terrain_workers,{},cpu_preparation_budget);
            world_lease.queue.resume();
        }
        c3x_renderer::FeatureGroup broadleaf_forest;
        c3x_renderer::FeatureGroup const * forest_group =
            c3x_renderer::find_feature_group(feature_bundle, "forest");
        if (forest_group != nullptr) {
            broadleaf_forest.name = "forest";
            for (c3x_renderer::FeaturePlacement const & placement : forest_group->placements)
                if (placement.asset_index < feature_bundle.assets.size() &&
                    feature_bundle.assets[placement.asset_index].id.find(
                        "feature/forest/leafy") != std::string::npos)
                    broadleaf_forest.placements.push_back(placement);
            if (!broadleaf_forest.placements.empty())
                forest_group = &broadleaf_forest;
        }
        c3x_renderer::FeatureGroup const * river_rock_group = river_assets_ready
            ? c3x_renderer::find_feature_group(river_rock_bundle, "river_rock") : nullptr;
        auto stable_feature_hash = [](std::uint32_t value) {
            value ^= value >> 16;
            value *= 0x7feb352du;
            value ^= value >> 15;
            value *= 0x846ca68bu;
            return value ^ (value >> 16);
        };
        c3x_renderer::objects::Projection object_projection;
        object_projection.tile_width=frame.tile_width;object_projection.content_view_height=content_view_height;
        object_projection.half_w=half_w;object_projection.half_h=half_h;
        object_projection.pickup_profile=pickup_profile;object_projection.world_objects=world_objects;
        std::copy(key_light,key_light+3,object_projection.key_light.begin());
        c3x_renderer::objects::Surfaces object_output;
        auto append_object_shadow=[&](auto const& asset,float scale,float x,float y,float h){
            c3x_renderer::objects::append_shadow(object_projection,asset,scale,x,y,h,shadow_vertices);
        };
        // World compilation order is independent of occurrence/pass order.
        // Defer an active helper's tile once, doing other selected work first.
        // The bounded second pass joins any remaining producers; no polling,
        // duplicate builds, extra pool or partial frame publication.
        std::vector<c3x_renderer_u32> compiling_tiles;
        for (c3x_renderer_u32 preparation_slot = 0; preparation_slot < (batch_preparing?preparation_count:frame.tile_count+compiling_tiles.size()); ++preparation_slot) {
            c3x_renderer_u32 index=batch_preparing?preparation_indices[preparation_slot]:
                preparation_slot<frame.tile_count?preparation_slot:compiling_tiles[preparation_slot-frame.tile_count];
            if(index>=frame.tile_count)return false;
            c3x_renderer_tile_v1 const & tile = frame.tiles[index];
            if(!selected_tile(index))continue;
            if (cancelled()) return false;
            if(!world_batch_enabled && active_terrain_workers<cpu_terrain_workers){
                auto ground_work=selected_ground_preparation.statistics();
                auto object_work=object_lease.queue.statistics();
                unsigned reserved=(ground_work.pending || ground_work.active?2u:0u)+(object_work.pending || object_work.active?1u:0u);
                unsigned available=cpu_terrain_workers>reserved?cpu_terrain_workers-reserved:1u;
                if(available>active_terrain_workers){terrain_preparation.expand_workers(available);active_terrain_workers=available;}
            }
            int ground = ground_type(tile);
            int relief = relief_type(tile);
            if (relief == 10 && !volcano_assets_ready)
                relief = -1;
            if (ground < 0 || ground >= c3x_renderer::terrain_type_count ||
                terrain_textures[ground].view == nullptr)
                continue;
            if (relief < 0 || relief >= c3x_renderer::terrain_type_count ||
                terrain_textures[relief].view == nullptr)
                relief = -1;
            bool draw_feature = feature_assets_ready &&
                (tile.real_terrain_type == 7 || tile.real_terrain_type == 8);
            bool draw_marsh = marsh_assets_ready && tile.real_terrain_type == 9;
            bool draw_volcano = volcano_assets_ready && tile.real_terrain_type == 10;
            bool draw_dunes = dune_assets_ready &&
                tile.real_terrain_type == 0 && tile.terrain_type == 0;
            if(!world_batch_enabled && cpu_terrain_enabled && retained_world && !prewarming && preparation_slot<frame.tile_count){
                auto input=terrain_compile_input(tile,frame,ground,skip_flat_shore,separate_natural_relief,index_natural_grids,retain_height_samples,world_objects);
                if(terrain_preparation.compiling(input.key)){compiling_tiles.push_back(index);continue;}
            }
            ++textured_tile_count;
            build_replacement[index] = C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED;
            if (draw_feature || draw_marsh || draw_volcano)
                build_replacement[index] |= C3X_RENDERER_TILE_CUSTOM_FEATURE_REPLACED;
            if (draw_dunes)
                build_replacement[index] |= C3X_RENDERER_TILE_CUSTOM_DUNES_REPLACED;
            if (river_assets_ready && (tile.river_code & 170u) != 0)
                build_replacement[index] |= C3X_RENDERER_TILE_CUSTOM_RIVER_REPLACED;
            if (route_assets_ready && tile.road_mask != 0)
                build_replacement[index] |= C3X_RENDERER_TILE_CUSTOM_ROAD_REPLACED;
            if (route_assets_ready && tile.railroad_mask != 0)
                build_replacement[index] |= C3X_RENDERER_TILE_CUSTOM_RAILROAD_REPLACED;
            if (city_assets_ready && tile.city_id >= 0)
                build_replacement[index] |= C3X_RENDERER_TILE_CUSTOM_CITY_REPLACED;
            unsigned site_flags=tile.improvement_flags & (C3X_RENDERER_IMPROVEMENT_GOODY_HUT | C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP);
            if(site_flags) {
                if(!site_assets_ready)return false;
                if(site_flags&C3X_RENDERER_IMPROVEMENT_GOODY_HUT)build_replacement[index]|=C3X_RENDERER_TILE_CUSTOM_HUT_REPLACED;
                if(site_flags&C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP)build_replacement[index]|=C3X_RENDERER_TILE_CUSTOM_CAMP_REPLACED;
            }
            if (mine_assets_ready &&
                (tile.improvement_flags & C3X_RENDERER_IMPROVEMENT_MINE) != 0)
                build_replacement[index] |= C3X_RENDERER_TILE_CUSTOM_MINE_REPLACED;
            if (farm_assets_ready &&
                (tile.improvement_flags & C3X_RENDERER_IMPROVEMENT_IRRIGATION) != 0)
                build_replacement[index] |= C3X_RENDERER_TILE_CUSTOM_FARM_REPLACED;
            auto local_river_nodes=select_river_nodes(tile);
            auto river_context=river_context_for(local_river_nodes);
            auto persistent_instance=topology_cache.retained(coordinate_key(tile.tile_x,tile.tile_y));
            auto compile_context=compile_context_for(tile,river_context);
            if(retained_world && (!prewarming || world_preparation) && persistent_instance){
                auto validation_begin=std::chrono::steady_clock::now();
                CachedTileGeometry* ready=nullptr;
                for(auto handle:persistent_instance->compiled_views){
                    auto candidate=resident_content.resolve(handle);
                    if(candidate && candidate->compile_context==compile_context && tile_content_valid(*candidate,tile)){ready=candidate;break;}
                }
                bool valid=ready!=nullptr;
                frame_tile_validation_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-validation_begin).count();
                if(valid){
                    ready->last_used=prewarming?tile_geometry_epoch-1:tile_geometry_epoch;
                    if(auto shared=resident_content.resolve(ready->natural_content))shared->last_used=tile_geometry_epoch;
                    if(ready->replaces_resource)build_replacement[index]|=C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED;
                    if(prewarming){topology_cache.attach(tile,ready->binding);prepared_footprint=tile_footprint(*ready,tile);}
                    else geometry_cache.tile_keys[index]=ready->binding;
                    ++frame_tiles_reused;++frame_instances_ready;
                    continue;
                }
            }
            std::unordered_map<std::uint64_t, std::uint64_t> dependencies;
            std::unordered_map<std::uint64_t, std::uint64_t> coast_dependencies;
            std::unordered_map<std::size_t, std::uint32_t> world_dependencies;
            std::unordered_map<std::uint64_t,std::uint64_t> appearance_dependencies;
            c3x_renderer::fidelity::NaturalWorld::CellInputs river_dependencies;
            c3x_renderer::fidelity::NaturalWorld::DependencyScope river_inputs(natural,retained_world?&river_dependencies:nullptr);
            if(retained_world && tile.real_terrain_type==7)
                for(int dr=-2;dr<=2;++dr)for(int dc=-2;dc<=2;++dc){
                    int c=(tile.tile_x+tile.tile_y)/2+dc,r=(tile.tile_x-tile.tile_y)/2+dr;
                    auto id=coordinate_key(c+r,c-r);
                    appearance_dependencies.emplace(id,topology_cache.appearance_revision(id));
                }
            auto observe_world = [&](std::size_t i, std::uint32_t value) { world_dependencies.emplace(i,value); };
            auto observe_coast = [&](auto id,auto revision) { coast_dependencies.emplace(id,revision); };
            c3x_renderer::fidelity::SurfaceQueries queries(world_coast,shore_samples,
                tile.tile_x,tile.tile_y,observe_world,observe_coast,skip_flat_shore);
            auto world_lookup = [&](int c,int r) { return queries.tile(c,r); };
            float shore_center_u=queries.center_u,shore_center_v=queries.center_v;
            pickup_ground_samples.clear();
            natural_height_samples.clear();
            auto shore_sample_at = [&](float u,float v) { return queries.shore(u,v); };
            std::vector<std::pair<std::uint64_t, std::array<int, 2>>> anchor_dependencies;
            auto observed_coordinate_key = [&](int x, int y) {
                auto key = coordinate_key(x, y);
                // Zero denotes a missing captured neighbor; revealing it must invalidate.
                auto inserted = dependencies.try_emplace(key, 0);
                if (inserted.second) {
                    auto found = topology_cache.current(key);
                    inserted.first->second = found == nullptr ? 0 : found->semantic;
                }
                return key;
            };
            // Ground generation always immediately reads the record it just
            // recorded as a dependency; combine both steps into one lookup.
            auto topology_lookup = [&](int x, int y) { return topology_cache.current(observed_coordinate_key(x, y)); };
            float left = 0.0f; // mesh origin is local; Civ III supplies the draw anchor
            float top = 0.0f;
            // The source terrain materials are detail textures, not one enormous
            // decal per viewport.  This scale puts a repeat across roughly six
            // Civ III tiles and is close to the cell-relative density used by
            // modern connected-grid terrain renderers.
            float const uv_scale = 0.26f;
            // The approved BIQ Lab projection uses a 224 px tile and applies
            // 0.82 vertical scale to terrain relief. Preserve that exact ratio
            // at both Civ III zoom levels. Feature bodies use the Lab's full
            // 150-pixel vertical basis rather than terrain's 0.82 multiplier.
            float const relief_projection_scale =
                static_cast<float>(frame.tile_width) / 224.0f * 0.82f;
            float const feature_projection_scale =
                static_cast<float>(frame.tile_width) / 224.0f;
            float ground_slot = static_cast<float>(ground);
            float surface_slot = static_cast<float>(draw_marsh ? 9 :
                (relief >= 0 ? relief : ground));
            int const tile_ground_grid=c3x_renderer::fidelity::ground_grid_detail(tile,relief,draw_dunes,
                pickup_profile,frame.tile_width,draw_record_count,topology_lookup,world_lookup);
            bool coast_detail=false;
            c3x_renderer::render_core::ShoreSample tile_center_shore{};
            if(pickup_profile){
                auto shore_started=std::chrono::steady_clock::now();
                if(retain_center_shore){
                    try{tile_center_shore=center_shore_cache.get(world_coast,tile.tile_x,tile.tile_y,observe_world,observe_coast);queries.prime_center(tile_center_shore);}
                    catch(...){tile_center_shore=shore_sample_at(queries.center_u,queries.center_v);}
                }else tile_center_shore=shore_sample_at(queries.center_u,queries.center_v);
                coast_detail=std::abs(tile_center_shore.distance)<1.5;
                frame_center_shore_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-shore_started).count();
            }
            int const flat_grid=pickup_profile ? (coast_detail && frame.tile_width>=96?16:8) : tile_ground_grid;
            int const shadow_grid = frame.tile_width >= 96 && draw_record_count <= 512
                ? 16 : 8;
            std::uint64_t tile_signature = 1469598103934665603ull;
            auto mix_tile = [&](auto value) {
                auto bytes = reinterpret_cast<std::uint8_t const *>(&value);
                for (std::size_t i = 0; i < sizeof(value); ++i)
                    tile_signature = (tile_signature ^ bytes[i]) * 1099511628211ull;
            };
            mix_tile(tile.tile_x); mix_tile(tile.tile_y);
            mix_tile(tile_content_signature(tile));
            mix_tile(world_ground?0:content_view_width); mix_tile(world_ground?0:content_view_height);
            mix_tile(world_ground?96:frame.tile_width); mix_tile(world_ground?48:frame.tile_height);
            mix_tile(frame.world_wrap_x ? frame.world_width_tiles : 0);
            mix_tile(frame.world_wrap_y ? frame.world_height_tiles : 0);
            mix_tile(frame.world_wrap_x); mix_tile(frame.world_wrap_y);
            mix_tile(content_revision); mix_tile(device_generation);
            if (pickup_profile) {
                mix_tile(c3x_renderer::render_core::render_core_revision);
                mix_tile(frame.world_width_tiles); mix_tile(frame.world_height_tiles);
            }
            if (!pickup_profile) { mix_tile(frame.hour); mix_tile(frame.season); }
            mix_tile(tile_ground_grid); mix_tile(world_ground?0:shadow_grid);
            if(pickup_profile)mix_tile(flat_grid);
            for(auto node:local_river_nodes){mix_tile(node->lattice_x);mix_tile(node->lattice_y);
                mix_tile(node->degree);mix_tile(node->touches_water);}
            auto validation_started=std::chrono::steady_clock::now();
            auto reuse_tile = [&](CachedTileGeometry& cached) {
                bool valid=tile_content_valid(cached,tile);
                if (valid) {
                    auto append_started=std::chrono::steady_clock::now();
                    frame_tile_validation_ms+=std::chrono::duration<double,std::milli>(append_started-validation_started).count();
                    if (prewarming) {
                        topology_cache.attach(tile,cached.binding);
                        cached.last_used = std::max(cached.last_used, tile_geometry_epoch-1);
                        prepared_footprint = tile_footprint(cached, tile);
                        ++frame_tiles_reused; return true;
                    }
                    if (cached.replaces_resource) build_replacement[index] |= C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED;
                    geometry_cache.tile_keys[index]=cached.binding;
                    if(retained_world){
                        cached.last_used=tile_geometry_epoch;
                        if(auto shared=resident_content.resolve(cached.natural_content))shared->last_used=tile_geometry_epoch;
                        topology_cache.attach(tile,cached.binding);
                    }else append_tile_geometry(cached, tile, animated_view);
                    frame_tile_append_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-append_started).count();
                    ++frame_tiles_reused;
                    return true;
                }
                return false;
            };
            // World records carry a non-owning association. Exact context and
            // all dependency checks still apply before using resident content.
            auto record=topology_cache.retained(coordinate_key(tile.tile_x,tile.tile_y));
            auto bound=record?resident_content.resolve(record->compiled):nullptr;
            bool reused_tile=bound && bound->signature==tile_signature && reuse_tile(*bound);
            if(!reused_tile){
                auto candidates=tile_geometry_cache.equal_range(tile_signature);
                for(auto it=candidates.first;it!=candidates.second;++it)
                    if(&it->second!=bound && reuse_tile(it->second)){reused_tile=true;break;}
            }
            if(reused_tile){if(prewarming && !batch_preparing)return true;continue;}
            frame_tile_validation_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-validation_started).count();
            ++frame_tiles_built;
            std::vector<ResourceAnchor> tile_resource_anchors;
            // Thousands of height/material/shadow samples revisit a tiny
            // integer neighborhood. Record each dependency once and resolve
            // its semantic maps once, using bounded per-tile scratch storage.
            struct NeighborhoodSample {
                bool ready = false;
                float ground = 0, surface = 0;
                int relief = -1;
            };
            constexpr int neighborhood_radius = 16;
            constexpr int neighborhood_side = neighborhood_radius * 2 + 1;
            std::array<NeighborhoodSample, neighborhood_side * neighborhood_side> neighborhood = {};
            auto neighborhood_at = [&](int x, int y) {
                int local_x = x - tile.tile_x + neighborhood_radius;
                int local_y = y - tile.tile_y + neighborhood_radius;
                NeighborhoodSample * cached = local_x >= 0 && local_x < neighborhood_side &&
                    local_y >= 0 && local_y < neighborhood_side
                    ? &neighborhood[local_y * neighborhood_side + local_x] : nullptr;
                if (cached != nullptr && cached->ready) return *cached;
                auto key = observed_coordinate_key(x, y);
                auto record = topology_cache.current(key);
                NeighborhoodSample value{true,
                    record ? static_cast<float>(record->ground) : ground_slot,
                    record ? static_cast<float>(record->surface) : surface_slot,
                    record ? record->relief : -1};
                if (cached != nullptr) *cached = value;
                return value;
            };
            auto ground_at = [&](int x, int y) { return neighborhood_at(x, y).ground; };
            auto surface_at = [&](int x, int y) { return neighborhood_at(x, y).surface; };
            auto terrain_at_lattice = [&](int lattice_u, int lattice_v) {
                int x = lattice_u + lattice_v;
                int y = lattice_u - lattice_v;
                return surface_at(x, y);
            };
            auto ground_at_lattice = [&](int u, int v) { return ground_at(u + v, u - v); };
            auto relief_at_lattice = [&](int u, int v) { return neighborhood_at(u + v, u - v).relief; };
            auto periodic_surface_uv = [&](float world_u, float world_v, float frequency) {
                return c3x_renderer::fidelity::ground_surface_uv(tile,frame,world_u,world_v,frequency);
            };
            auto river_segment_distance = [](float point_x, float point_y,
                                              float start_x, float start_y,
                                              float endpoint_x, float endpoint_y) {
                float segment_x = endpoint_x - start_x;
                float segment_y = endpoint_y - start_y;
                float point_offset_x = point_x - start_x;
                float point_offset_y = point_y - start_y;
                float denominator = segment_x * segment_x + segment_y * segment_y;
                float t = denominator > 0.0f
                    ? std::clamp((point_offset_x * segment_x + point_offset_y * segment_y) /
                                     denominator, 0.0f, 1.0f)
                    : 0.0f;
                float delta_x = point_offset_x - segment_x * t;
                float delta_y = point_offset_y - segment_y * t;
                return std::sqrt(delta_x * delta_x + delta_y * delta_y);
            };
            auto river_edge_distance = [&](c3x_renderer_tile_v1 const & river_tile,
                                           float u, float v,
                                           float start_u, float start_v,
                                           float endpoint_u, float endpoint_v,
                                           unsigned direction_bit) {
                auto screen_point = [&](float local_u, float local_v) {
                    return std::array<float, 2>{
                        (local_u - local_v) * half_w,
                        (local_u + local_v - 1.0f) * half_h};
                };
                std::array<float, 2> point = screen_point(u, v);
                std::array<float, 2> start = screen_point(start_u, start_v);
                std::array<float, 2> endpoint = screen_point(endpoint_u, endpoint_v);
                int canonical_river_x = canonical_component(
                    river_tile.tile_x, frame.world_width_tiles, frame.world_wrap_x);
                int canonical_river_y = canonical_component(
                    river_tile.tile_y, frame.world_height_tiles, frame.world_wrap_y);
                int canonical_column = (canonical_river_x + canonical_river_y) / 2;
                int canonical_row = (canonical_river_x - canonical_river_y) / 2;
                unsigned edge_family = 0u;
                if (direction_bit == 32u)
                    canonical_row -= 1;
                else if (direction_bit == 128u) {
                    canonical_column -= 1;
                    edge_family = 1u;
                } else if (direction_bit == 8u)
                    edge_family = 1u;
                unsigned seed = static_cast<unsigned>(canonical_column + 4096) * 73856093u ^
                                static_cast<unsigned>(canonical_row + 4096) * 19349663u ^
                                edge_family * 83492791u;
                seed ^= seed >> 13;
                float direction_x = endpoint[0] - start[0];
                float direction_y = endpoint[1] - start[1];
                float direction_length = std::sqrt(direction_x * direction_x +
                                                   direction_y * direction_y);
                float normal_x = -direction_y / std::max(direction_length, 0.001f);
                float normal_y = direction_x / std::max(direction_length, 0.001f);
                float primary_bend = 5.0f +
                    static_cast<float>((seed >> 5) & 15u) / 15.0f * 8.0f;
                float secondary_bend = 1.5f +
                    static_cast<float>((seed >> 11) & 7u) / 7.0f * 4.0f;
                if ((seed & 1u) != 0)
                    primary_bend = -primary_bend;
                if ((seed & 2u) != 0)
                    secondary_bend = -secondary_bend;
                float distance = 1000.0f;
                float previous_x = start[0];
                float previous_y = start[1];
                for (int segment = 1; segment <= 16; ++segment) {
                    float t = static_cast<float>(segment) / 16.0f;
                    float offset = std::sin(t * 3.14159265f) * primary_bend +
                                   std::sin(t * 6.28318531f) * secondary_bend;
                    float curve_x = start[0] + direction_x * t + normal_x * offset;
                    float curve_y = start[1] + direction_y * t + normal_y * offset;
                    distance = std::min(distance, river_segment_distance(
                        point[0], point[1], previous_x, previous_y, curve_x, curve_y));
                    previous_x = curve_x;
                    previous_y = curve_y;
                }
                return distance;
            };
            if(fidelity_profile)for(int r=(tile.tile_x-tile.tile_y)/2-4;r<=(tile.tile_x-tile.tile_y)/2+4;r++)
                for(int c=(tile.tile_x+tile.tile_y)/2-4;c<=(tile.tile_x+tile.tile_y)/2+4;c++)world_lookup(c,r);
            auto river_distance = [&](c3x_renderer_tile_v1 const & river_tile,
                                      float u, float v) {
                if(fidelity_profile){
                    float x=float(river_tile.tile_x+river_tile.tile_y)*.5f+u,y=float(river_tile.tile_x-river_tile.tile_y)*.5f+1-v;
                    return float(natural.river_sample({x,y}).distance);
                }
                float distance = 1000.0f;
                unsigned mask = river_tile.river_code & 170u;
                if ((mask & 2u) != 0)
                    distance = std::min(distance, river_edge_distance(
                        river_tile, u, v, 0.0f, 0.0f, 1.0f, 0.0f, 2u));
                if ((mask & 8u) != 0)
                    distance = std::min(distance, river_edge_distance(
                        river_tile, u, v, 1.0f, 0.0f, 1.0f, 1.0f, 8u));
                if ((mask & 32u) != 0)
                    distance = std::min(distance, river_edge_distance(
                        river_tile, u, v, 0.0f, 1.0f, 1.0f, 1.0f, 32u));
                if ((mask & 128u) != 0)
                    distance = std::min(distance, river_edge_distance(
                        river_tile, u, v, 0.0f, 0.0f, 0.0f, 1.0f, 128u));
                return distance;
            };
            auto center_material_weights = [&](int lattice_u, int lattice_v) {
                std::array<float, 5> result = {};
                int base = static_cast<int>(ground_at_lattice(lattice_u, lattice_v));
                int surface = static_cast<int>(terrain_at_lattice(lattice_u, lattice_v));
                auto add_material = [&result](int sample_base, int sample_surface, float amount) {
                    if (sample_base >= 11)
                        return;
                    int material = sample_surface == 9 ? 3 :
                        (sample_base == 0 || sample_base == 4 ? 2 :
                         (sample_base == 1 ? 1 : (sample_base == 3 ? 4 : 0)));
                    result[material] += amount;
                };
                if (base < 11)
                    add_material(base, surface, 1.0f);
                else {
                    for (int y = -1; y <= 1; ++y) {
                        for (int x = -1; x <= 1; ++x) {
                            int sample_base = static_cast<int>(
                                ground_at_lattice(lattice_u + x, lattice_v + y));
                            int sample_surface = static_cast<int>(
                                terrain_at_lattice(lattice_u + x, lattice_v + y));
                            add_material(sample_base, sample_surface,
                                         (x == 0 || y == 0) ? 1.0f : 0.70f);
                        }
                    }
                }
                float total = result[0] + result[1] + result[2] + result[3] + result[4];
                if (total <= 0.0f)
                    result[0] = 1.0f;
                else
                    for (float & value : result)
                        value /= total;
                return result;
            };
            auto material_weights_for = [&](float world_u, float world_v) {
                if (pickup_profile) {
                    return queries.weights(world_u,world_v);
                }
                float source_x = world_u + world_v - 1.0f;
                float source_y = world_u - world_v;
                float warp_x = std::sin(source_x * 0.83f + source_y * 1.19f) * 0.10f +
                    std::sin(source_x * 2.31f - source_y * 0.67f) * 0.035f;
                float warp_y = std::sin(source_x * 1.07f - source_y * 0.91f) * 0.10f +
                    std::sin(source_x * 0.59f + source_y * 2.03f) * 0.035f;
                float grid_x = world_u - 0.5f + warp_x;
                float grid_y = world_v - 0.5f + warp_y;
                int x0 = static_cast<int>(std::floor(grid_x));
                int y0 = static_cast<int>(std::floor(grid_y));
                float tx = smoothstep01((grid_x - static_cast<float>(x0) - 0.20f) / 0.60f);
                float ty = smoothstep01((grid_y - static_cast<float>(y0) - 0.20f) / 0.60f);
                std::array<float, 5> c00 = center_material_weights(x0, y0);
                std::array<float, 5> c10 = center_material_weights(x0 + 1, y0);
                std::array<float, 5> c01 = center_material_weights(x0, y0 + 1);
                std::array<float, 5> c11 = center_material_weights(x0 + 1, y0 + 1);
                std::array<float, 5> result = {};
                for (int material = 0; material < 5; ++material) {
                    float top = c00[material] * (1.0f - tx) + c10[material] * tx;
                    float bottom = c01[material] * (1.0f - tx) + c11[material] * tx;
                    result[material] = top * (1.0f - ty) + bottom * ty;
                }
                return result;
            };
            auto water_family_depth = [&](float world_u, float world_v) {
                float grid_x = world_u - 0.5f;
                float grid_y = world_v - 0.5f;
                int x0 = static_cast<int>(std::floor(grid_x));
                int y0 = static_cast<int>(std::floor(grid_y));
                float tx = smoothstep01(grid_x - static_cast<float>(x0));
                float ty = smoothstep01(grid_y - static_cast<float>(y0));
                auto center_depth = [&](int x, int y) {
                    int base = static_cast<int>(ground_at_lattice(x, y));
                    return base >= 11 ? std::clamp((base - 10) * 0.34f, 0.18f, 1.0f) : 0.34f;
                };
                float top = center_depth(x0, y0) * (1.0f - tx) +
                    center_depth(x0 + 1, y0) * tx;
                float bottom = center_depth(x0, y0 + 1) * (1.0f - tx) +
                    center_depth(x0 + 1, y0 + 1) * tx;
                return top * (1.0f - ty) + bottom * ty;
            };
            auto signed_shore_distance = [&](float world_u, float world_v,
                                             float local_u, float local_v) {
                if (pickup_profile) return static_cast<float>(std::clamp(
                    -shore_sample_at(world_u,world_v).distance/.65, -1., 1.));
                float grid_x = world_u - 0.5f;
                float grid_y = world_v - 0.5f;
                int x0 = static_cast<int>(std::floor(grid_x));
                int y0 = static_cast<int>(std::floor(grid_y));
                float tx = smoothstep01(grid_x - static_cast<float>(x0));
                float ty = smoothstep01(grid_y - static_cast<float>(y0));
                auto raw_sign = [&](int x, int y) {
                    return ground_at_lattice(x, y) >= 11.0f ? 1.0f : -1.0f;
                };
                auto center_sign = [&](int x, int y) {
                    return raw_sign(x, y) * 0.62f +
                        (raw_sign(x - 1, y) + raw_sign(x + 1, y) +
                         raw_sign(x, y - 1) + raw_sign(x, y + 1)) * 0.095f;
                };
                float top = center_sign(x0, y0) * (1.0f - tx) +
                    center_sign(x0 + 1, y0) * tx;
                float bottom = center_sign(x0, y0 + 1) * (1.0f - tx) +
                    center_sign(x0 + 1, y0 + 1) * tx;
                float field = top * (1.0f - ty) + bottom * ty;
                float center_dx = local_u - 0.5f;
                float center_dy = local_v - 0.5f;
                float center_anchor = 1.0f - smoothstep01(
                    std::sqrt(center_dx * center_dx + center_dy * center_dy) / 0.34f);
                float own_sign = ground >= 11 ? 1.0f : -1.0f;
                field = field * (1.0f - center_anchor * 0.88f) +
                    own_sign * center_anchor * 0.88f;
                float source_x = world_u + world_v - 1.0f;
                float source_y = world_u - world_v;
                float contour_noise = std::sin(source_x * 1.37f + source_y * 0.71f) * 0.38f +
                    std::sin(source_x * 0.79f - source_y * 1.11f) * 0.20f +
                    std::sin(source_x * 3.83f - source_y * 2.17f) * 0.13f +
                    std::sin(source_x * 0.53f + source_y * 2.91f) * 0.07f;
                float boundary_weight = 1.0f - smoothstep01(std::abs(field) / 0.92f);
                contour_noise *= 1.0f - center_anchor * 0.90f;
                float result = field + contour_noise * boundary_weight;
                for (int row = y0 - 1; row <= y0 + 1; ++row) {
                    for (int column = x0 - 1; column <= x0 + 1; ++column) {
                        if (raw_sign(column, row) < 0.0f)
                            continue;
                        bool connected = raw_sign(column - 1, row) > 0.0f ||
                            raw_sign(column + 1, row) > 0.0f ||
                            raw_sign(column, row - 1) > 0.0f ||
                            raw_sign(column, row + 1) > 0.0f;
                        if (connected)
                            continue;
                        float dx = world_u - (static_cast<float>(column) + 0.5f);
                        float dy = world_v - (static_cast<float>(row) + 0.5f);
                        float basin = 1.0f - smoothstep01(
                            std::sqrt(dx * dx + dy * dy) / 0.46f);
                        result = std::max(result, basin);
                    }
                }
                return std::clamp(result, -1.0f, 1.0f);
            };
            auto sample_mountain = [&](int column, int row,
                                       float local_x, float local_y,
                                       float & height, float & blend) {
                height = 0.0f;
                blend = 0.0f;
                if (relief_at_lattice(column, row) != 6)
                    return;
                int source_x = canonical_component(column + row,
                    frame.world_width_tiles, frame.world_wrap_x);
                int source_y = canonical_component(column - row,
                    frame.world_height_tiles, frame.world_wrap_y);
                unsigned seed = static_cast<unsigned>(source_x * 73 + source_y * 151);
                unsigned variant = (seed >> 3) % 5u;
                TerrainTexture const & mountains = terrain_textures[6];
                if (mountains.relief_height_variants[variant].empty() ||
                    mountains.relief_blend_variants[variant].empty())
                    return;

                // Copied from the canonical Lab mountain sampler. Preserve the
                // five authored silhouettes and only apply its deterministic
                // rigid orientation and connected-massif footprint fit.
                unsigned transform = seed & 7u;
                if ((transform & 1u) != 0)
                    std::swap(local_x, local_y);
                if ((transform & 2u) != 0)
                    local_x = 1.0f - local_x;
                if ((transform & 4u) != 0)
                    local_y = 1.0f - local_y;
                constexpr int offsets[4][2] = {
                    {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
                bool has_relief_neighbor = false;
                for (auto const & offset : offsets) {
                    int neighbor = relief_at_lattice(
                        column + offset[0], row + offset[1]);
                    has_relief_neighbor = has_relief_neighbor ||
                        neighbor == 6 || neighbor == 10;
                }
                float footprint_scale = has_relief_neighbor ? 0.50f : 0.68f;
                float source_u = 0.5f + (local_x - 0.5f) * footprint_scale;
                float source_v = 0.5f + (local_y - 0.5f) * footprint_scale;
                if (source_u < 0.0f || source_u > 1.0f ||
                    source_v < 0.0f || source_v > 1.0f)
                    return;
                float source_edge = std::min(std::min(source_u, 1.0f - source_u),
                    std::min(source_v, 1.0f - source_v));
                float edge = smoothstep01(source_edge / 0.055f);
                height = sample_normalized_field(
                    mountains.relief_height_variants[variant],
                    mountains.relief_variant_widths[variant],
                    mountains.relief_variant_heights[variant],
                    mountains.relief_height_minimum[variant],
                    mountains.relief_height_maximum[variant], source_u, source_v);
                blend = sample_normalized_field(
                    mountains.relief_blend_variants[variant],
                    mountains.relief_variant_widths[variant],
                    mountains.relief_variant_heights[variant],
                    mountains.relief_blend_minimum[variant],
                    mountains.relief_blend_maximum[variant], source_u, source_v) * edge;
            };
            auto sample_volcano = [&](int column, int row,
                                      float local_x, float local_y,
                                      float & height, float & blend) {
                height = 0.0f;
                blend = 0.0f;
                if (!volcano_assets_ready || relief_at_lattice(column, row) != 10)
                    return;
                int source_x = canonical_component(column + row,
                    frame.world_width_tiles, frame.world_wrap_x);
                int source_y = canonical_component(column - row,
                    frame.world_height_tiles, frame.world_wrap_y);
                unsigned seed = static_cast<unsigned>(source_x) * 73856093u ^
                    static_cast<unsigned>(source_y) * 19349663u;
                seed ^= seed >> 13;
                if ((seed & 1u) != 0)
                    std::swap(local_x, local_y);
                if ((seed & 2u) != 0)
                    local_x = 1.0f - local_x;
                constexpr int offsets[4][2] = {
                    {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
                bool has_relief_neighbor = false;
                for (auto const & offset : offsets) {
                    int neighbor = relief_at_lattice(
                        column + offset[0], row + offset[1]);
                    has_relief_neighbor = has_relief_neighbor ||
                        neighbor == 6 || neighbor == 10;
                }
                float footprint_scale = has_relief_neighbor ? 0.60f : 1.0f;
                float aspect = (seed & 4u) != 0 ? 0.88f : 1.12f;
                float source_u = 0.5f + (local_x - 0.5f) * footprint_scale * aspect;
                float source_v = 0.5f + (local_y - 0.5f) * footprint_scale / aspect;
                if (source_u < 0.0f || source_u > 1.0f ||
                    source_v < 0.0f || source_v > 1.0f)
                    return;
                float source_edge = std::min(std::min(source_u, 1.0f - source_u),
                    std::min(source_v, 1.0f - source_v));
                float edge = smoothstep01(source_edge / 0.055f);
                TerrainTexture const & volcano = terrain_textures[10];
                height = sample_normalized_field(volcano.height_pixels,
                    volcano.height_width, volcano.height_height,
                    volcano.height_minimum, volcano.height_maximum,
                    source_u, source_v);
                blend = sample_normalized_field(volcano.blend_pixels,
                    volcano.height_width, volcano.height_height,
                    volcano.blend_minimum, volcano.blend_maximum,
                    source_u, source_v) * edge;
            };
            auto sample_relief_chain = [&](float world_u, float world_v,
                                           float & height, float & blend,
                                           float & displacement,
                                           bool include_volcano) {
                height = 0.0f;
                blend = 0.0f;
                displacement = 0.0f;
                int center_u = static_cast<int>(std::floor(world_u));
                int center_v = static_cast<int>(std::floor(world_v));
                constexpr int candidates[5][2] = {
                    {0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
                for (auto const & offset : candidates) {
                    int candidate_u = center_u + offset[0];
                    int candidate_v = center_v + offset[1];
                    int candidate_relief = relief_at_lattice(candidate_u, candidate_v);
                    if ((candidate_relief != 6 && candidate_relief != 10) ||
                        (!include_volcano && candidate_relief == 10))
                        continue;
                    float local_x = world_u - static_cast<float>(candidate_u);
                    float local_y = 1.0f - (world_v - static_cast<float>(candidate_v));
                    float candidate_height = 0.0f;
                    float candidate_blend = 0.0f;
                    if (candidate_relief == 6)
                        sample_mountain(candidate_u, candidate_v, local_x, local_y,
                                        candidate_height, candidate_blend);
                    else
                        sample_volcano(candidate_u, candidate_v, local_x, local_y,
                                       candidate_height, candidate_blend);
                    int source_x = canonical_component(candidate_u + candidate_v,
                        frame.world_width_tiles, frame.world_wrap_x);
                    int source_y = canonical_component(candidate_u - candidate_v,
                        frame.world_height_tiles, frame.world_wrap_y);
                    unsigned vertical_seed = static_cast<unsigned>(source_x) * 73856093u ^
                        static_cast<unsigned>(source_y) * 19349663u;
                    float candidate_displacement = candidate_height *
                        smoothstep01(candidate_blend / 0.34f) *
                        (candidate_relief == 6 ? 104.0f :
                         (((vertical_seed >> 3) & 1u) != 0 ? 104.0f : 88.0f));
                    if (candidate_displacement > displacement) {
                        height = candidate_height;
                        blend = candidate_blend;
                        displacement = candidate_displacement;
                    }
                }
            };
            auto pickup_source = [&](int kind, unsigned variant, int channel, float u, float v) {
                return c3x_renderer::fidelity::relief_source(terrain_textures,fidelity_profile,kind,variant,channel,u,v);
            };
            auto pickup_river = [&](int c, int r, float u, float v) {
                auto const & world = world_coast.world();
                auto i=world.index(c,r);auto value = world.at(i);
                if(i!=std::size_t(-1))observe_world(i,value);
                if (value == 0xffffffffu || ((value >> 16) & 170u) == 0 || !river_assets_ready) return 1000.0f;
                c3x_renderer_tile_v1 owner = {};
                owner.tile_x = c+r; owner.tile_y = c-r; owner.river_code = (value >> 16) & 255u;
                return river_distance(owner,u,v);
            };
            auto pickup_dune = [&](float u, float v) {
                // Source-fidelity terrain composes localized authored dune
                // decals. Keep the older analytic wave field only for the
                // retained non-fidelity path so it cannot turn every desert
                // region into one continuous parallel ridge carpet.
                return dune_assets_ready && !fidelity_profile ?
                    c3x_renderer::dune_height(u,v,1.0f) : 0.0f;
            };
            auto pickup_activity = [&](int c, int r) {
                auto const & world = world_coast.world();
                auto i = world.index(c,r); auto value = world.at(i);
                if (i != std::size_t(-1)) observe_world(i,value);
                return value != 0xffffffffu && (value >> 24) != 0 ? 1.0f : 0.0f;
            };
            c3x_renderer::fidelity::ReliefSurface pickup_surface(world_coast.world().dimensions(),
                (tile.tile_x+tile.tile_y)/2, (tile.tile_x-tile.tile_y)/2,
                pickup_profile ? shore_sample_at(shore_center_u,shore_center_v).distance : 0,
                world_lookup, pickup_source, shore_sample_at, pickup_river, pickup_dune, pickup_activity,
                pickup_ground_samples, pickup_height_queries,separate_natural_relief);
            auto pickup_ground_at = [&](float u,float v) {return pickup_surface.sample(u,v);};
            auto pickup_height_at = [&](float u,float v) {return pickup_surface.height(u,v);};
            // Only this tile's immutable source/dependency scope may reuse a
            // height. Keep support with it: vegetation queries consume both.
            // Exact float keys, bounded admission, and no resampling or LOD.
            auto natural_height_at = [&](float u,float v,float* support=nullptr) {
                auto compute=[&]() {
                    std::array<float,2> value{};
                    value[0]=queries.height(natural,pickup_height_at,u,v,&value[1]);
                    return value;
                };
                auto value=retain_height_samples?natural_height_samples.get(u,v,compute):compute();
                if(support)*support=value[1];
                return value[0];
            };
            auto relief_at_world = [&](float world_u, float world_v) {
                if (pickup_profile) {
                    auto sample = pickup_ground_at(world_u,world_v);
                    return std::array<float,3>{sample.height,sample.authored_height,sample.authored_blend};
                }
                // This is the approved Lab BIQ hill path, copied literally
                // into production-space accessors. Base-material height maps
                // are intentionally not evaluated here: the Lab BIQ renderer
                // keeps ordinary ground geometrically flat and only raises
                // authored hills, mountain/volcano chains, and dunes.
                int owner_u = static_cast<int>(std::floor(world_u));
                int owner_v = static_cast<int>(std::floor(world_v));
                auto owner_found = topology_cache.current(observed_coordinate_key(
                    owner_u + owner_v, owner_u - owner_v));
                c3x_renderer_tile_v1 const & height_tile =
                    owner_found == nullptr ? tile : owner_found->occurrence;
                TerrainTexture const & hill_asset = terrain_textures[5];
                auto hill_value = [&](float sample_u, float sample_v) {
                    if (hill_asset.relief_profile != 2 || hill_asset.height_pixels.empty())
                        return 0.0f;
                    constexpr float radius = 0.018f;
                    float source_u = 0.11f + sample_u * 0.035f;
                    float source_v = 0.17f + sample_v * 0.035f;
                    auto sample = [&](float u, float v) {
                        return sample_normalized_field(hill_asset.height_pixels,
                            hill_asset.height_width, hill_asset.height_height,
                            hill_asset.height_minimum, hill_asset.height_maximum,
                            u, v);
                    };
                    float center = sample(source_u, source_v) * 4.0f;
                    float cardinal = sample(source_u - radius, source_v) +
                        sample(source_u + radius, source_v) +
                        sample(source_u, source_v - radius) +
                        sample(source_u, source_v + radius);
                    float diagonal = sample(source_u - radius, source_v - radius) +
                        sample(source_u + radius, source_v - radius) +
                        sample(source_u - radius, source_v + radius) +
                        sample(source_u + radius, source_v + radius);
                    float authored_macro =
                        (center + cardinal * 2.0f + diagonal) / 16.0f;
                    return smoothstep01((authored_macro - 0.22f) / 0.38f);
                };
                auto hill_support = [&]() {
                    int center_u = static_cast<int>(std::floor(world_u));
                    int center_v = static_cast<int>(std::floor(world_v));
                    float support = 0.0f;
                    for (int row = center_v - 1; row <= center_v + 1; ++row) {
                        for (int column = center_u - 1; column <= center_u + 1; ++column) {
                            if (relief_at_lattice(column, row) != 5)
                                continue;
                            float dx = (world_u - (static_cast<float>(column) + 0.5f)) / 0.92f;
                            float dy = (world_v - (static_cast<float>(row) + 0.5f)) / 0.78f;
                            float distance = std::sqrt(dx * dx + dy * dy);
                            support = std::max(support,
                                smoothstep01((1.0f - distance) / 0.42f));
                        }
                    }
                    return support;
                };
                float tile_world_u =
                    static_cast<float>(height_tile.tile_x + height_tile.tile_y) * 0.5f;
                float tile_world_v =
                    static_cast<float>(height_tile.tile_x - height_tile.tile_y) * 0.5f;
                float local_u = world_u - tile_world_u;
                float local_v = 1.0f - (world_v - tile_world_v);
                constexpr int source_offsets[4][2] = {
                    {-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
                float distances[4] = {
                    local_u, local_v, 1.0f - local_u, 1.0f - local_v};
                float hill_compatibility = 1.0f;
                for (int edge = 0; edge < 4; ++edge) {
                    auto neighbor = topology_cache.current(observed_coordinate_key(
                        height_tile.tile_x + source_offsets[edge][0],
                        height_tile.tile_y + source_offsets[edge][1]));
                    bool compatible = neighbor != nullptr &&
                        neighbor->ground >= 0 && neighbor->ground < 11;
                    if (!compatible)
                        hill_compatibility *= smoothstep01(distances[edge] / 0.22f);
                }
                float support = hill_support();
                float height = support > 0.0f && hill_compatibility > 0.0f
                    ? hill_value(world_u, world_v) * 52.0f * support * hill_compatibility : 0.0f;
                float authored_height = 0.0f;
                float authored_blend = 0.0f;
                int current_real = height_tile.real_terrain_type;
                if (current_real == 5 || current_real == 6 || current_real == 10) {
                    float relief_envelope = 1.0f;
                    if (current_real == 6 || current_real == 10) {
                        for (int edge = 0; edge < 4; ++edge) {
                            auto neighbor = topology_cache.current(observed_coordinate_key(
                                height_tile.tile_x + source_offsets[edge][0],
                                height_tile.tile_y + source_offsets[edge][1]));
                            int neighbor_real = neighbor == nullptr
                                ? -1 : neighbor->relief;
                            bool continues = neighbor_real == 5 || neighbor_real == 6 ||
                                neighbor_real == 10;
                            if (!continues)
                                relief_envelope *= smoothstep01(distances[edge] / 0.28f);
                        }
                    }
                    float transition_envelope = 1.0f;
                    if (current_real == 5) {
                        transition_envelope = 0.0f;
                        for (int edge = 0; edge < 4; ++edge) {
                            auto neighbor = topology_cache.current(observed_coordinate_key(
                                height_tile.tile_x + source_offsets[edge][0],
                                height_tile.tile_y + source_offsets[edge][1]));
                            if (neighbor != nullptr &&
                                neighbor->relief == 6)
                                transition_envelope = std::max(transition_envelope,
                                    1.0f - smoothstep01(distances[edge] / 0.24f));
                        }
                    }
                    float chain_height = 0.0f;
                    float chain_blend = 0.0f;
                    float chain_displacement = 0.0f;
                    sample_relief_chain(world_u, world_v, chain_height, chain_blend,
                                        chain_displacement, current_real != 5);
                    float mountain_displacement = chain_displacement *
                        relief_envelope * transition_envelope;
                    if (height <= 0.001f || mountain_displacement <= 0.001f)
                        height = std::max(height, mountain_displacement);
                    else {
                        constexpr float blend_width = 12.0f;
                        float weight = std::clamp(0.5f + 0.5f *
                            (mountain_displacement - height) / blend_width, 0.0f, 1.0f);
                        height = height * (1.0f - weight) +
                            mountain_displacement * weight +
                            blend_width * weight * (1.0f - weight);
                    }
                    float material_envelope = 0.0f;
                    if (current_real == 5)
                        material_envelope = transition_envelope * 0.35f;
                    else {
                        material_envelope = 1.0f;
                        for (int edge = 0; edge < 4; ++edge) {
                            auto neighbor = topology_cache.current(observed_coordinate_key(
                                height_tile.tile_x + source_offsets[edge][0],
                                height_tile.tile_y + source_offsets[edge][1]));
                            int neighbor_real = neighbor == nullptr
                                ? -1 : neighbor->relief;
                            if (neighbor_real != 6 && neighbor_real != 10)
                                material_envelope *= smoothstep01(
                                    distances[edge] / 0.28f);
                        }
                    }
                    authored_height = chain_height;
                    authored_blend = chain_blend * material_envelope;
                }
                bool height_tile_has_dunes = dune_assets_ready &&
                    height_tile.real_terrain_type == 0 && height_tile.terrain_type == 0;
                if (height_tile_has_dunes) {
                    float dune_envelope = 1.0f;
                    for (int edge = 0; edge < 4; ++edge) {
                        std::uint64_t key = observed_coordinate_key(
                            height_tile.tile_x + source_offsets[edge][0],
                            height_tile.tile_y + source_offsets[edge][1]);
                        auto neighbor = topology_cache.current(key);
                        bool continues = neighbor && neighbor->ground == 0 && neighbor->real == 0;
                        if (!continues)
                            dune_envelope *= smoothstep01(distances[edge] / 0.16f);
                    }
                    height += c3x_renderer::dune_height(
                        world_u, world_v, 1.0f) * dune_envelope;
                }
                if (river_assets_ready && height_tile.terrain_type < 11 &&
                    (height_tile.river_code & 170u) != 0) {
                    float local_height_u = world_u - tile_world_u;
                    float local_height_v = 1.0f - (world_v - tile_world_v);
                    float distance = river_distance(
                        height_tile, local_height_u, local_height_v);
                    float valley = 1.0f - smoothstep01((distance - 4.0f) / 16.0f);
                    float valley_floor = height * 0.10f;
                    height = height * (1.0f - valley * 0.92f) +
                             valley_floor * valley * 0.92f;
                }
                return std::array<float, 3>{height, authored_height, authored_blend};
            };
            // Retain the actual wrapped occurrence: vertices contain raw world
            // coordinates/UVs, not just canonical gameplay identity.
            auto ground_key=(std::uint64_t(std::uint32_t(tile.tile_x))<<32)|std::uint32_t(tile.tile_y);
            std::uint64_t ground_signature=tile_content_signature(tile);
            for(auto value:{content_revision,retained_world?std::uint64_t(0):std::uint64_t(frame.world_topology_revision),
                    std::uint64_t(frame.world_width_tiles),std::uint64_t(frame.world_height_tiles),
                    std::uint64_t(frame.world_wrap_x),std::uint64_t(frame.world_wrap_y)})
                ground_signature=(ground_signature^value)*1099511628211ull;
            if(retained_world && !local_river_nodes.empty()){
                for(auto value:{frame.tile_width,frame.tile_height})ground_signature=(ground_signature^std::uint64_t(value))*1099511628211ull;
                for(auto node:local_river_nodes)for(auto value:{node->lattice_x,node->lattice_y,int(node->degree),int(node->touches_water)})
                    ground_signature=(ground_signature^std::uint64_t(value))*1099511628211ull;
            }
            auto retained_ground=ground_grid_cache.find(ground_key);
            bool ground_hit=retain_ground_grids && !world_ground && !prewarming && retained_ground!=ground_grid_cache.end() &&
                retained_ground->second.signature==ground_signature;
            if(ground_hit && !natural.valid(retained_ground->second.river_dependencies))ground_hit=false;
            if(ground_hit)for(auto const& dependency:retained_ground->second.dependencies){
                auto found=topology_cache.current(dependency.first);
                if((found==nullptr?0:found->semantic)!=dependency.second){ground_hit=false;break;}
            }
            if(ground_hit)for(auto const& dependency:retained_ground->second.coast_dependencies)
                if(world_coast.node_revision(dependency.first)!=dependency.second){ground_hit=false;break;}
            if(ground_hit)for(auto const& dependency:retained_ground->second.world_dependencies)
                if(world_coast.world().at(dependency.first)!=dependency.second){ground_hit=false;break;}
            if(ground_hit){
                auto& cached=retained_ground->second;cached.used=tile_geometry_epoch;
                river_dependencies.insert(cached.river_dependencies.begin(),cached.river_dependencies.end());
                dependencies.insert(cached.dependencies.begin(),cached.dependencies.end());
                coast_dependencies.insert(cached.coast_dependencies.begin(),cached.coast_dependencies.end());
                world_dependencies.insert(cached.world_dependencies.begin(),cached.world_dependencies.end());
            }else if(retain_ground_grids && !prewarming && retained_ground!=ground_grid_cache.end()){
                ground_grid_cache_bytes-=retained_ground->second.bytes;ground_grid_cache.erase(retained_ground);
            }
            std::vector<CachedGroundGrid> pending_ground_grids;
            object_projection.tile=tile;object_projection.left=left;object_projection.top=top;
            object_projection.relief_projection_scale=relief_projection_scale;
            object_projection.feature_projection_scale=feature_projection_scale;
            auto append_feature_instance=[&](auto const& bundle,auto const& placement,float u,float v,float rotation,
                    float scale,float material,float owner,bool shadow,std::vector<Vertex>& target){
                c3x_renderer::objects::append_instance(object_projection,bundle,placement,u,v,rotation,scale,
                    material,owner,shadow,&bundle==&site_bundle,relief_at_world,natural_height_at,target,shadow_vertices);
            };
            auto adopt_objects=[&](c3x_renderer::objects::Surfaces& result){
                std::vector<Vertex>* layers[]={&route_vertices,&feature_vertices,&city_vertices,&wall_vertices,&mine_vertices,&farm_vertices,&site_vertices};
                for(unsigned layer=0;layer<c3x_renderer::objects::layer_count;++layer){
                    auto& source=result.layers[layer];auto& target=*layers[layer];
                    if(source.empty())continue;
                    if(target.empty())target.swap(source);else {target.insert(target.end(),source.begin(),source.end());source.clear();}
                }
                shadow_vertices.insert(shadow_vertices.end(),result.shadows.begin(),result.shadows.end());result.shadows.clear();
            };
            // Match the standalone terrain stack exactly: a flat material
            // underlay, raised land, submerged bed, then transparent water.
            // Keeping these pass-major vectors prevents a later land tile
            // from overwriting an earlier neighbor's continuous shoreline.
            // A hit must carry its dependency observations into the ordinary
            // GPU tile cache. Never reuse samples across authoritative edits.
            std::uint64_t natural_key=1469598103934665603ull;
            for(auto value:{std::uint64_t(std::uint32_t(tile.tile_x)),std::uint64_t(std::uint32_t(tile.tile_y)),
                    tile_content_signature(tile),content_revision,retained_world?std::uint64_t(0):std::uint64_t(frame.world_topology_revision),
                    std::uint64_t(frame.world_width_tiles),std::uint64_t(frame.world_height_tiles),
                    std::uint64_t(frame.world_wrap_x),std::uint64_t(frame.world_wrap_y),std::uint64_t(world_ground),std::uint64_t(patch_detail.identity()),world_ground?river_context:std::uint64_t(0)})
                natural_key=(natural_key^value)*1099511628211ull;
            // Forest exclusions consume neighboring city composition, while
            // the ordinary topology dependency map intentionally omits it.
            if(tile.real_terrain_type==7)for(int dr=-2;dr<=2;++dr)for(int dc=-2;dc<=2;++dc){
                int c=(tile.tile_x+tile.tile_y)/2+dc,r=(tile.tile_x-tile.tile_y)/2+dr;
                auto neighbor=topology_cache.current(coordinate_key(c+r,c-r));
                auto value=neighbor!=nullptr && neighbor->occurrence.city_id>=0?
                    tile_content_signature(neighbor->occurrence):0;
                natural_key=(natural_key^value)*1099511628211ull;
            }
            // World-space GPU data may outlive any particular camera entry.
            // Pin the owner before ground uploads can evict other entries.
            bool cache_natural=fidelity_profile && tile.city_id<0 && (!prewarming || world_preparation);
            bool share_natural=(cache_natural || world_objects) && share_world_meshes;
            auto shared_natural=tile_geometry_cache.find(natural_key);
            bool shared_hit=share_natural && shared_natural!=tile_geometry_cache.end() && shared_natural->second.shared_natural && shared_natural->second.world_objects==world_objects && shared_natural->second.world_ground==world_ground;
            if(shared_hit && !natural.valid(shared_natural->second.river_dependencies))shared_hit=false;
            if(shared_hit)for(auto const& dependency:shared_natural->second.appearance_dependencies)
                if(topology_cache.appearance_revision(dependency.first)!=dependency.second){shared_hit=false;break;}
            if(shared_hit)for(auto const& dependency:shared_natural->second.dependencies){
                auto current=topology_cache.current(dependency.first);
                if((current==nullptr?0:current->semantic)!=dependency.second){shared_hit=false;break;}
            }
            if(shared_hit)for(auto const& dependency:shared_natural->second.coast_dependencies)
                if(world_coast.node_revision(dependency.first)!=dependency.second){shared_hit=false;break;}
            if(shared_hit)for(auto const& dependency:shared_natural->second.world_dependencies)
                if(world_coast.world().at(dependency.first)!=dependency.second){shared_hit=false;break;}
            if(shared_hit && world_objects)for(auto const& dependency:shared_natural->second.anchor_dependencies){
                auto current=topology_cache.current(dependency.first);
                auto basis=shared_natural->second.source_tile_width;
                if(!current || std::int64_t(current->occurrence.anchor_x-tile.anchor_x)*basis!=std::int64_t(dependency.second[0])*frame.tile_width ||
                    std::int64_t(current->occurrence.anchor_y-tile.anchor_y)*basis!=std::int64_t(dependency.second[1])*frame.tile_width){shared_hit=false;break;}
            }
            if(share_natural && !shared_hit && shared_natural!=tile_geometry_cache.end()){
                if(shared_natural->second.shared_natural && shared_natural->second.last_used!=tile_geometry_epoch){
                    tile_geometry_cache_bytes-=shared_natural->second.byte_count;
                    release_resident_content(shared_natural->second);tile_geometry_cache.erase(shared_natural);
                }else share_natural=false; // Do not replace a pinned owner or a colliding ordinary key.
            }
            if(shared_hit && world_objects){
                tile_resource_anchors=shared_natural->second.resource_anchors;
                if(shared_natural->second.replaces_resource)build_replacement[index]|=C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED;
            }
            if(shared_hit){
                auto& cached=shared_natural->second;cached.last_used=tile_geometry_epoch;
                if(animated_view)cached.animation_epoch=tile_geometry_epoch;
                river_dependencies.insert(cached.river_dependencies.begin(),cached.river_dependencies.end());
                dependencies.insert(cached.dependencies.begin(),cached.dependencies.end());
                coast_dependencies.insert(cached.coast_dependencies.begin(),cached.coast_dependencies.end());
                world_dependencies.insert(cached.world_dependencies.begin(),cached.world_dependencies.end());
            }
            std::unique_ptr<c3x_renderer::PreparedWorld> prepared_world;
            if(world_batch_enabled && !shared_hit){
                auto begin=std::chrono::steady_clock::now();
                prepared_world=world_lease.queue.take(index);
                world_join_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
                if(cancelled())return false;
                if(prepared_world){
                    world_ground_ms+=prepared_world->ground_ms;world_terrain_ms+=prepared_world->terrain_ms;
                    world_object_ms+=prepared_world->object_ms;world_upload_ms+=prepared_world->upload_ms;
                    world_gpu_bytes+=prepared_world->gpu_bytes;
                }else {
                    // Oversized/failed combined work uses the established
                    // per-component compilers below. Do not impose one buffer's
                    // size limit on a valid tile that previously used several.
                    ++world_recovery;
                }
            }
            QueryPerformanceCounter(&phase_time);
            // Copy the cache lease before dispatch; no worker observes the map
            // iterator, renderer counters, or another generator's query scratch.
            auto ground_grid_lease=ground_hit?retained_ground->second.grids:nullptr;
            c3x_renderer::fidelity::PreparedGround ground_cache_proof;
            if(retain_ground_grids && !world_ground && !prewarming){
                ground_cache_proof.topology=dependencies;
                ground_cache_proof.coast=coast_dependencies;
                ground_cache_proof.world=world_dependencies;
                ground_cache_proof.rivers=river_dependencies;
            }
            c3x_renderer::fidelity::GroundCompileInput ground_compile_input;
            ground_compile_input.tile = tile;
            ground_compile_input.world_ground = world_ground;
            ground_compile_input.pickup_profile = pickup_profile;
            ground_compile_input.fidelity_profile = fidelity_profile;
            ground_compile_input.draw_marsh = draw_marsh;
            ground_compile_input.river_assets_ready = river_assets_ready;
            ground_compile_input.retain_ground_grids = retain_ground_grids;
            ground_compile_input.reuse_nested_ground_grids = reuse_nested_ground_grids;
            ground_compile_input.prewarming = prewarming;
            ground_compile_input.ground = ground;
            ground_compile_input.half_w = half_w;
            ground_compile_input.half_h = half_h;
            ground_compile_input.uv_scale = uv_scale;
            ground_compile_input.relief_projection_scale = relief_projection_scale;
            std::copy(key_light, key_light + 3, ground_compile_input.key_light);
            ground_compile_input.left = left;
            ground_compile_input.top = top;
            ground_compile_input.flat_grid = flat_grid;
            ground_compile_input.tile_ground_grid = tile_ground_grid;
            ground_compile_input.shadow_grid = shadow_grid;
            auto compile_ground=[&](std::atomic<bool> const& stop)->std::unique_ptr<c3x_renderer::fidelity::PreparedGround>{
                if(world_ground && shared_hit)return std::make_unique<c3x_renderer::fidelity::PreparedGround>();
                auto ground_cancelled=[&]{return stop.load(std::memory_order_relaxed) || cancelled();};
                return c3x_renderer::fidelity::prepare_ground(ground_compile_input,frame,natural,ground_compile_scratch,world_coast,topology_cache,
                    local_river_nodes,ground_grid_lease,tile_center_shore,ground_slot,skip_flat_shore,separate_natural_relief,
                    coordinate_key,pickup_source,pickup_dune,river_distance,relief_at_world,material_weights_for,
                    signed_shore_distance,periodic_surface_uv,ndc_x,ndc_y,ground_cancelled);
            };
            bool const batched_ground=ground_batch_enabled && (!world_ground || !shared_hit);
            if(batched_ground && !ground_queued[index]){
                if(!selected_ground_preparation.offer({index,make_ground_job(tile,local_river_nodes)},
                        c3x_renderer::fidelity::GroundPreparation::job_limit,true))return false;
                ground_queued[index]=true;++ground_batch_jobs;
            }
            // Compatibility/serial controls retain their original tile scope.
            // Production selected jobs own their inputs and use the frame lease.
            std::unique_ptr<c3x_renderer::fidelity::GroundTask> ground_task;
            if(!batched_ground && !prepared_world)ground_task=std::make_unique<c3x_renderer::fidelity::GroundTask>(ground_preparation,compile_ground,
                ground_concurrent && !world_batch_enabled && (!world_ground || !shared_hit));
            // Legacy callbacks run synchronously and still observe through the
            // caller's maps. Capture those reads before later object queries.
            if(!pickup_profile && retain_ground_grids && !world_ground && !prewarming){
                ground_cache_proof.topology=dependencies;
                ground_cache_proof.coast=coast_dependencies;
                ground_cache_proof.world=world_dependencies;
                ground_cache_proof.rivers=river_dependencies;
            }
            QueryPerformanceCounter(&phase_end);ground_ticks+=phase_end.QuadPart-phase_time.QuadPart;
            phase_time=phase_end;

            if (cancelled()) return false;
            if(world_objects){if(shared_hit)++frame_world_object_hits;else ++frame_world_object_builds;}
            c3x_renderer::city_fidelity::Composition const* tile_city_composition=nullptr;
            std::unique_ptr<c3x_renderer::objects::PreparedObjects> prepared_objects;
            if(prepare_objects && !shared_hit){
                auto begin=std::chrono::steady_clock::now();
                if(prepared_world)prepared_objects=std::move(prepared_world->objects);
                else if(object_worker && !world_batch_enabled)prepared_objects=object_lease.queue.take(index);
                if(!prepared_objects){
                    if(object_worker && !world_batch_enabled)++object_recovery;
                    prepared_objects=compile_objects(make_object_job(tile),foreground_object_scratch,cancelled);
                }
                object_join_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
                if(!prepared_objects || cancelled())return false;
                // Results never outlive this immutable frame lease. All reads,
                // including absent route neighbors, become resident proofs.
                dependencies.insert(prepared_objects->topology.begin(),prepared_objects->topology.end());
                world_dependencies.insert(prepared_objects->world.begin(),prepared_objects->world.end());
                coast_dependencies.insert(prepared_objects->coast.begin(),prepared_objects->coast.end());
                river_dependencies.insert(prepared_objects->rivers.begin(),prepared_objects->rivers.end());
                object_instances+=prepared_objects->instances;object_routes+=prepared_objects->routes;
                object_gpu_bytes+=prepared_objects->gpu_bytes;
                if(prepared_objects->composition!=~0u){
                    tile_city_composition=&cities.library.compositions[prepared_objects->composition];
                    if(city_assets_ready && ground<11)++city_fallbacks_omitted;
                }
                for(auto const& part:prepared_objects->city){if(part.terrain_conforming)++city_deformed_parts;else ++city_rigid_parts;}
            }
            if(!world_objects || !shared_hit){
            if(!prepared_objects){
                c3x_renderer::objects::Plan plan;
                bool routes_enabled=true;
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
                routes_enabled=diagnostic_routes!=2;
#endif
                c3x_renderer::objects::select_routes(tile,object_assets,route_assets_ready,routes_enabled,
                    [&](int x,int y){return topology_cache.current(observed_coordinate_key(x,y));},plan);
                object_instances+=unsigned(plan.instances.size());object_routes+=unsigned(plan.routes.size());
                c3x_renderer::objects::compile(plan,object_projection,object_assets,relief_at_world,natural_height_at,object_output);
                adopt_objects(object_output);
            }
            if (feature_assets_ready &&
                (tile.real_terrain_type == 7 || tile.real_terrain_type == 8) &&
                !(fidelity_profile && tile.real_terrain_type == 7)) {
                char const * group_name = tile.real_terrain_type == 7 ? "forest" : "jungle";
                c3x_renderer::FeatureGroup const * group =
                    tile.real_terrain_type == 7 ? forest_group :
                    c3x_renderer::find_feature_group(feature_bundle, group_name);
                // Forest uses a 6x6 canopy body and jungle a 7x7 body. Stable
                // grid jitter prevents the random
                // interior holes visible in the earlier in-game port.
                unsigned instance_count = tile.real_terrain_type == 7 ? 36u : 49u;
                char const * forest_anchors[] = {"pine_01", "pine_clump_01", "shrub_01"};
                char const * jungle_anchors[] = {
                    "grass_04", "palm_01", "palm_02", "plant_01", "plant_02", "plant_03"};
                char const * const * anchors = tile.real_terrain_type == 7 ?
                    forest_anchors : jungle_anchors;
                unsigned anchor_count = tile.real_terrain_type == 7 ? 3u : 6u;
                float tile_world_u = static_cast<float>(tile.tile_x + tile.tile_y) * 0.5f;
                float tile_world_v = static_cast<float>(tile.tile_x - tile.tile_y) * 0.5f;
                int canonical_feature_x = canonical_component(
                    tile.tile_x, frame.world_width_tiles, frame.world_wrap_x);
                int canonical_feature_y = canonical_component(
                    tile.tile_y, frame.world_height_tiles, frame.world_wrap_y);
                std::uint32_t feature_seed =
                    static_cast<std::uint32_t>(canonical_feature_x * 0x193) ^
                    static_cast<std::uint32_t>(canonical_feature_y * 0x217);
                if (group != nullptr) {
                    for (unsigned instance = 0; instance < instance_count; ++instance) {
                        c3x_renderer::FeaturePlacement const * placement =
                            instance < anchor_count ?
                            c3x_renderer::find_feature_placement_by_suffix(
                                feature_bundle, *group, anchors[instance]) :
                            c3x_renderer::select_feature_placement(
                                *group, feature_seed + instance * 31u);
                        if (placement == nullptr || placement->asset_index >= feature_bundle.assets.size())
                            continue;
                        c3x_renderer::FeatureAsset const & asset =
                            feature_bundle.assets[placement->asset_index];
                        unsigned grid_side = tile.real_terrain_type == 7 ? 6u : 7u;
                        unsigned column = instance % grid_side;
                        unsigned row = instance / grid_side;
                        float jitter_u = c3x_renderer::stable_random(
                            feature_seed + instance * 103u + 59u) - 0.5f;
                        float jitter_v = c3x_renderer::stable_random(
                            feature_seed + instance * 107u + 61u) - 0.5f;
                        float u_t = (static_cast<float>(column) + 0.5f + jitter_u * 0.68f) /
                            static_cast<float>(grid_side);
                        float v_t = (static_cast<float>(row) + 0.5f + jitter_v * 0.68f) /
                            static_cast<float>(grid_side);
                        float u = 0.07f + 0.86f * u_t;
                        float v = 0.07f + 0.86f * v_t;
                        float scale_variation =
                            (c3x_renderer::stable_random(feature_seed + instance * 71u + 23u) * 2.0f - 1.0f) *
                            placement->scale_variation;
                        float scene_feature_scale = tile.real_terrain_type == 7 ? 0.42f : 0.40f;
                        float scale = placement->scale * (1.0f + scale_variation) *
                            scene_feature_scale;
                        float rotation = c3x_renderer::stable_random(
                            feature_seed + instance * 97u + 47u) * 6.28318530718f;
                        float cosine = std::cos(rotation);
                        float sine = std::sin(rotation);
                        std::array<float, 3> ground_sample = relief_at_world(
                            tile_world_u + u, tile_world_v + (1.0f - v));
                        float center_x = left + half_w + (u - v) * half_w;
                        float center_y = top + (u + v) * half_h -
                            ground_sample[0] * relief_projection_scale;
                        append_object_shadow(asset, scale, center_x, center_y,
                            ground_sample[0] * relief_projection_scale);
                        std::vector<Vertex> transformed(asset.vertices.size());
                        for (std::size_t vertex_index = 0; vertex_index < asset.vertices.size(); ++vertex_index) {
                            c3x_renderer::FeatureSourceVertex const & source = asset.vertices[vertex_index];
                            float local_x = (source.position[0] * cosine - source.position[1] * sine) * scale;
                            float local_y = (source.position[0] * sine + source.position[1] * cosine) * scale;
                            float local_z = source.position[2] * scale;
                            float screen_x = center_x + (local_x - local_y) * half_w;
                            float screen_y = center_y + (local_x + local_y) * half_h -
                                local_z * 150.0f * feature_projection_scale;
                            float normal_x = source.normal[0] * cosine - source.normal[1] * sine;
                            float normal_y = source.normal[0] * sine + source.normal[1] * cosine;
                            // Match Lab's BIQ feature depth: position the body
                            // from its ground-plane Y, then pull authored height
                            // toward the camera separately. Using lifted screen_y
                            // here pushed tall trees behind neighboring ground and
                            // clipped their meshes at exact tile diamonds.
                            float ground_height_pixels =
                                ground_sample[0] * relief_projection_scale;
                            float base_ground_y = center_y + ground_height_pixels +
                                (local_x + local_y) * half_h;
                            float feature_height_tiles = local_z * 150.0f *
                        (world_objects?128.f/224.f:feature_projection_scale) /
                        (world_objects?128.f/224.f*.82f:relief_projection_scale);
                            float depth =
                                base_ground_y + ground_height_pixels * 0.75f +
                        feature_height_tiles * 0.0012f * static_cast<float>(content_view_height);
                            transformed[vertex_index] = Vertex{
                                ndc_x(screen_x), ndc_y(screen_y), depth,
                                source.uv[0], source.uv[1], 1.0f,
                                normal_x, normal_y, source.normal[2],
                                1.0f, 1.0f, 0.0f, 0.0f,
                                0.0f, 0.0f,
                                static_cast<float>(asset.texture_index), 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,
                                1000.0f, 0.0f, 1000.0f, 0.0f, -1.0f
                            };
                    if (pickup_profile) {
                        auto & vertex = transformed[vertex_index];
                        vertex.world_x = tile_world_u + u + local_x;
                        vertex.world_y = tile_world_v + 1.0f - v - local_y;
                        vertex.world_z = (ground_sample[0] + 2.5f + feature_height_tiles) / 112.0f;
                        vertex.world_valid = 1.0f;
                        if(world_objects){
                            vertex.x=64.f+(u-v)*64.f+(local_x-local_y)*64.f;
                            vertex.y=((u+v)*32.f-ground_sample[0]*(128.f/224.f*.82f))+(local_x+local_y)*32.f-local_z*150.f*(128.f/224.f);
                            vertex.z=feature_height_tiles;
                        }
                        auto normal=c3x_renderer::lighting::object_normal(normal_x,normal_y,source.normal[2]);
                        vertex.normal_x=normal[0];vertex.normal_y=normal[1];vertex.normal_z=normal[2];
                    }
                        }
                        for (std::uint32_t source_index : asset.indices)
                            feature_vertices.push_back(transformed[source_index]);
                    }
                }
            }
            if (river_rock_group != nullptr && !river_rock_group->placements.empty()) {
                struct RiverRockEdge {
                    unsigned bit;
                    int neighbor_x;
                    int neighbor_y;
                    bool north_edge;
                };
                RiverRockEdge edges[] = {
                    {2u, tile.tile_x + 1, tile.tile_y - 1, true},
                    {8u, tile.tile_x + 1, tile.tile_y + 1, false},
                };
                for (RiverRockEdge const & edge : edges) {
                    if ((tile.river_code & edge.bit) == 0)
                        continue;
                    std::uint32_t seed = stable_feature_hash(
                        static_cast<std::uint32_t>(canonical_component(
                            tile.tile_x, frame.world_width_tiles, frame.world_wrap_x) + 4096) * 0x193u ^
                        static_cast<std::uint32_t>(canonical_component(
                            tile.tile_y, frame.world_height_tiles, frame.world_wrap_y) + 4096) * 0x217u ^
                        edge.bit);
                    if ((seed % 3u) != 0u)
                        continue;
                    c3x_renderer::FeaturePlacement const & placement =
                        river_rock_group->placements[(seed >> 5) %
                            river_rock_group->placements.size()];
                    if (placement.asset_index >= river_rock_bundle.assets.size())
                        continue;
                    float along = 0.28f + c3x_renderer::stable_random(seed ^ 0x73a52u) * 0.44f;
                    c3x_renderer_tile_v1 const * owner = &tile;
                    float local_u = edge.north_edge ? along : 0.975f;
                    float local_v = edge.north_edge ? 0.025f : along;
                    auto neighbor = topology_cache.current(
                        observed_coordinate_key(edge.neighbor_x, edge.neighbor_y));
                    if (ground_type(*owner) >= 11 && neighbor != nullptr &&
                        ground_type(neighbor->occurrence) < 11) {
                        owner = &neighbor->occurrence;
                        local_u = edge.north_edge ? along : 0.025f;
                        local_v = edge.north_edge ? 0.975f : along;
                    }
                    if (ground_type(*owner) >= 11)
                        continue;
                    c3x_renderer::FeatureAsset const & asset =
                        river_rock_bundle.assets[placement.asset_index];
                    float scale = 0.155f +
                        c3x_renderer::stable_random(seed ^ 0x91c37u) * 0.070f;
                    float rotation = c3x_renderer::stable_random(seed ^ 0x4ad91u) *
                        6.28318530718f;
                    float cosine = std::cos(rotation);
                    float sine = std::sin(rotation);
                    if(fidelity_profile){
                        river::P query_point{(owner->tile_x+owner->tile_y)*.5+local_u,(owner->tile_x-owner->tile_y)*.5+1-local_v};
                        bool placed=false;
                        for(unsigned attempt=0;attempt<4 && !placed;attempt++){
                            river::P point;double side=((seed&1u)?1.:-1.)*((attempt&1u)?-1.:1.);
                            double margin=11.+c3x_renderer::stable_random(seed^0x2c07u)*1.5+(attempt/2u)*3.;
                            if(!natural.river_page(query_point.x,query_point.y).bank_point(query_point,margin,side,point))continue;
                            int c=int(std::floor(point.x)),r=int(std::floor(point.y));
                            auto receiving=topology_cache.current(observed_coordinate_key(c+r,c-r));
                            if(receiving==nullptr || ground_type(receiving->occurrence)>=11)continue;
                            float u=std::clamp(float(point.x-c),.00001f,.99999f),v=std::clamp(float(r+1-point.y),.00001f,.99999f);
                            if(shore_sample_at(float(point.x),float(point.y)).distance<.065)continue;
                            bool clear=relief_at_world(float(point.x),float(point.y))[0]+2.5f<18;
                            for(auto const&vertex:asset.vertices){river::P q{point.x+(vertex.position[0]*cosine-vertex.position[1]*sine)*scale,
                                point.y-(vertex.position[0]*sine+vertex.position[1]*cosine)*scale};
                                if(natural.river_sample(q).distance<5.5){clear=false;break;}}
                            if(!clear)continue;
                            owner=&receiving->occurrence;local_u=u;local_v=v;placed=true;
                        }
                        if(!placed)continue;
                    }
                    float owner_world_u =
                        static_cast<float>(owner->tile_x + owner->tile_y) * 0.5f;
                    float owner_world_v =
                        static_cast<float>(owner->tile_x - owner->tile_y) * 0.5f;
                    std::array<float, 3> ground_sample = relief_at_world(
                        owner_world_u + local_u, owner_world_v + (1.0f - local_v));
                    anchor_dependencies.push_back({coordinate_key(owner->tile_x, owner->tile_y),
                        {owner->anchor_x - tile.anchor_x, owner->anchor_y - tile.anchor_y}});
                    float center_x = static_cast<float>(owner->anchor_x - tile.anchor_x) + half_w +
                        (local_u - local_v) * half_w;
                    float center_y = static_cast<float>(owner->anchor_y - tile.anchor_y) +
                        (local_u + local_v) * half_h -
                        ground_sample[0] * relief_projection_scale;
                    append_object_shadow(asset, scale, center_x, center_y,
                        ground_sample[0] * relief_projection_scale);
                    std::vector<Vertex> transformed(asset.vertices.size());
                    for (std::size_t vertex_index = 0;
                         vertex_index < asset.vertices.size(); ++vertex_index) {
                        c3x_renderer::FeatureSourceVertex const & source =
                            asset.vertices[vertex_index];
                        float local_x = (source.position[0] * cosine -
                                         source.position[1] * sine) * scale;
                        float local_y = (source.position[0] * sine +
                                         source.position[1] * cosine) * scale;
                        float local_z = source.position[2] * scale;
                        float screen_x = center_x + (local_x - local_y) * half_w;
                        float screen_y = center_y + (local_x + local_y) * half_h -
                            local_z * 150.0f * feature_projection_scale;
                        float normal_x = source.normal[0] * cosine - source.normal[1] * sine;
                        float normal_y = source.normal[0] * sine + source.normal[1] * cosine;
                        float ground_height_pixels =
                            ground_sample[0] * relief_projection_scale;
                        float base_ground_y = center_y + ground_height_pixels +
                            (local_x + local_y) * half_h;
                        float feature_height_tiles = local_z * 150.0f *
                        (world_objects?128.f/224.f:feature_projection_scale) /
                        (world_objects?128.f/224.f*.82f:relief_projection_scale);
                        float depth =
                            base_ground_y + ground_height_pixels * 0.75f +
                        feature_height_tiles * 0.0012f * static_cast<float>(content_view_height);
                        transformed[vertex_index] = Vertex{
                            ndc_x(screen_x), ndc_y(screen_y), depth,
                            source.uv[0], source.uv[1], 1.0f,
                            normal_x, normal_y, source.normal[2],
                            1.0f, 1.0f, 0.0f, 0.0f,
                            0.0f, 0.0f,
                            static_cast<float>(asset.texture_index + 8u), 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 1.0f,
                            1000.0f, 0.0f, 1000.0f, 0.0f, -1.0f};
                    if (pickup_profile) {
                        auto & vertex = transformed[vertex_index];
                        vertex.world_x = owner_world_u + local_u + local_x;
                        vertex.world_y = owner_world_v + 1.0f - local_v - local_y;
                        vertex.world_z = (ground_sample[0] + 2.5f + feature_height_tiles) / 112.0f;
                        vertex.world_valid = 1.0f;
                        if(world_objects){
                            vertex.x=float(owner->anchor_x-tile.anchor_x)*128.f/frame.tile_width+64.f+(local_u-local_v)*64.f+(local_x-local_y)*64.f;
                            vertex.y=float(owner->anchor_y-tile.anchor_y)*128.f/frame.tile_width+((local_u+local_v)*32.f-ground_sample[0]*(128.f/224.f*.82f))+(local_x+local_y)*32.f-local_z*150.f*(128.f/224.f);
                            vertex.z=feature_height_tiles;
                        }
                        auto normal=c3x_renderer::lighting::object_normal(normal_x,normal_y,source.normal[2]);
                        vertex.normal_x=normal[0];vertex.normal_y=normal[1];vertex.normal_z=normal[2];
                    }
                    }
                    for (std::uint32_t source_index : asset.indices)
                        feature_vertices.push_back(transformed[source_index]);
                }
            }
            int animated_resource = resource_animation_for(tile);
            if (animated_resource >= 0) {
                auto const & animation = resource_animations[animated_resource];
                build_replacement[index] |= C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED;
                for (unsigned body=0;body<animation.count;++body) {
                    float angle=6.28318530718f*float(body)/float(animation.count);
                    float ring=animation.count==1 ? 0.f : (body==0 ? .045f : .10f+.055f*float((body-1)%3));
                    ResourceAnchor anchor; anchor.asset=unsigned(animated_resource);
                    anchor.seed=c3x_renderer::stable_hash(tile.variant_seed*83u+body*97u+29u);
                    anchor.u=.5f+std::cos(angle)*ring;anchor.v=.5f+std::sin(angle)*ring*.78f;
                    anchor.world_u=float(tile.tile_x+tile.tile_y)*.5f;
                    anchor.world_v=float(tile.tile_x-tile.tile_y)*.5f;
                    anchor.ground=relief_at_world(anchor.world_u+anchor.u,anchor.world_v+1-anchor.v)[0];
                    tile_resource_anchors.push_back(anchor);
                }
            } else if (resource_assets_ready && tile.resource_id >= 0) {
                std::string resource_name = tile.resource_name;
                std::transform(resource_name.begin(), resource_name.end(), resource_name.begin(),
                    [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
                char const * group_name = nullptr;
                for (char const * candidate : {"horses", "iron", "uranium", "gold",
                                               "dye", "wheat", "cattle", "fish"}) {
                    if (resource_name.find(candidate) != std::string::npos) {
                        group_name = candidate;
                        break;
                    }
                }
                c3x_renderer::FeatureGroup const * group = group_name == nullptr ? nullptr :
                    c3x_renderer::find_feature_group(resource_bundle, group_name);
                if (group != nullptr && !group->placements.empty()) {
                    build_replacement[index] |= C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED;
                    c3x_renderer::FeaturePlacement const & placement = group->placements.front();
                    unsigned count = std::max(1u, placement.count);
                    for (unsigned body = 0; body < count; ++body) {
                        float angle = 6.28318530718f *
                            (static_cast<float>(body) / static_cast<float>(count) +
                             c3x_renderer::stable_random(tile.variant_seed * 101u + body * 37u) * 0.11f);
                        float ring = count == 1u ? 0.0f :
                            (body == 0u ? 0.045f : 0.10f + 0.055f * static_cast<float>((body - 1u) % 3u));
                        float variation =
                            (c3x_renderer::stable_random(tile.variant_seed * 59u + body * 71u + 13u) *
                             2.0f - 1.0f) * placement.scale_variation;
                        float scale = placement.scale * (1.0f + variation) * 0.72f;
                        float rotation = c3x_renderer::stable_random(
                            tile.variant_seed * 83u + body * 97u + 29u) * 6.28318530718f;
                        bool fish = std::strcmp(group_name, "fish") == 0;
                        append_feature_instance(resource_bundle, placement,
                            0.5f + std::cos(angle) * ring,
                            0.5f + std::sin(angle) * ring * 0.78f,
                            rotation, scale, 21.0f, 0.0f, !fish, feature_vertices);
                    }
                }
            }
            if(!prepared_objects){
                c3x_renderer::objects::Plan plan;
                if(fidelity_profile && city_profile && cities.ready)
                    tile_city_composition=c3x_renderer::city_fidelity::select(cities.library,tile,
                        (tile.tile_x+tile.tile_y)/2,(tile.tile_x-tile.tile_y)/2,world_lookup,shore_sample_at,
                        [&](float x,float y){return natural.river_sample({x,y}).distance;},natural_height_at);
                if(tile_city_composition && city_assets_ready && ground<11)++city_fallbacks_omitted;
                if(!c3x_renderer::objects::select_improvements(tile,object_assets,ground,site_flags,
                        mine_assets_ready,farm_assets_ready,city_assets_ready,tile_city_composition!=nullptr,plan))return false;
                object_instances+=unsigned(plan.instances.size());object_routes+=unsigned(plan.routes.size());
                c3x_renderer::objects::compile(plan,object_projection,object_assets,relief_at_world,natural_height_at,object_output);
                adopt_objects(object_output);
            }
            QueryPerformanceCounter(&phase_end);feature_ticks+=phase_end.QuadPart-phase_time.QuadPart;phase_time=phase_end;
            if (pickup_profile && cliff_assets_ready && coast_detail) {
                c3x_renderer::fidelity::CliffCompileInput cliff_input;
                cliff_input.tile_x=tile.tile_x;cliff_input.tile_y=tile.tile_y;
                cliff_input.left=left;cliff_input.top=top;cliff_input.half_w=half_w;cliff_input.half_h=half_h;
                cliff_input.relief_projection_scale=relief_projection_scale;
                cliff_input.content_view_height=float(content_view_height);
                cliff_input.vertical_basis=150.f/112.f*(world_objects?128.f/224.f:feature_projection_scale)/
                    (world_objects?128.f/224.f*.82f:relief_projection_scale);
                // Compiled into an isolated per-tile result (vertices plus the
                // coast cells actually read) instead of writing directly into
                // the frame-shared cliff_vertices/coast_dependencies, so the
                // compiler has no renderer-owned state to race on if it later
                // moves off this thread. Merged below, in the same order as
                // the previous inline version, so output stays byte-identical.
                c3x_renderer::fidelity::CliffSurfaces cliff_result;
                // Cliffs query height/shore/river data through their own
                // private SurfaceQueries/ReliefSurface/NaturalWorld instead of
                // the ones ground uses, so this whole block can eventually run
                // concurrently with ground/city generation for this tile
                // without racing their mutable caches. Dependencies read
                // through these private objects are recorded locally and
                // merged into the frame-shared maps below, same as ground's.
                std::unordered_map<std::size_t,std::uint32_t> cliff_world_dependencies;
                auto cliff_observe_world=[&](std::size_t i,std::uint32_t value){cliff_world_dependencies.emplace(i,value);};
                auto cliff_observe_coast=[&](auto id,auto revision){cliff_result.coast.emplace(id,revision);};
                cliff_query_scratch.reset_tile();
                c3x_renderer::fidelity::SurfaceQueries cliff_queries(world_coast,cliff_query_scratch.shore_samples,
                    tile.tile_x,tile.tile_y,cliff_observe_world,cliff_observe_coast,skip_flat_shore);
                cliff_queries.prime_center(tile_center_shore);
                auto cliff_world_lookup=[&](int c,int r){ return cliff_queries.tile(c,r); };
                auto cliff_shore_sample_at=[&](float u,float v){ return cliff_queries.shore(u,v); };
                auto cliff_river_distance=[&](c3x_renderer_tile_v1 const & river_tile,float u,float v){
                    if(fidelity_profile){
                        float x=float(river_tile.tile_x+river_tile.tile_y)*.5f+u,y=float(river_tile.tile_x-river_tile.tile_y)*.5f+1-v;
                        return float(cliff_query_scratch.rivers.river_sample({x,y}).distance);
                    }
                    float distance=1000.0f;unsigned mask=river_tile.river_code & 170u;
                    if((mask & 2u)!=0)distance=std::min(distance,river_edge_distance(river_tile,u,v,0.0f,0.0f,1.0f,0.0f,2u));
                    if((mask & 8u)!=0)distance=std::min(distance,river_edge_distance(river_tile,u,v,1.0f,0.0f,1.0f,1.0f,8u));
                    if((mask & 32u)!=0)distance=std::min(distance,river_edge_distance(river_tile,u,v,0.0f,1.0f,1.0f,1.0f,32u));
                    if((mask & 128u)!=0)distance=std::min(distance,river_edge_distance(river_tile,u,v,0.0f,0.0f,0.0f,1.0f,128u));
                    return distance;
                };
                auto cliff_pickup_river=[&](int c,int r,float u,float v){
                    auto const & world=world_coast.world();
                    auto i=world.index(c,r);auto value=world.at(i);
                    if(i!=std::size_t(-1))cliff_observe_world(i,value);
                    if(value==0xffffffffu || ((value>>16)&170u)==0 || !river_assets_ready)return 1000.0f;
                    c3x_renderer_tile_v1 owner={};
                    owner.tile_x=c+r;owner.tile_y=c-r;owner.river_code=(value>>16)&255u;
                    return cliff_river_distance(owner,u,v);
                };
                auto cliff_pickup_activity=[&](int c,int r){
                    auto const & world=world_coast.world();
                    auto i=world.index(c,r);auto value=world.at(i);
                    if(i!=std::size_t(-1))cliff_observe_world(i,value);
                    return value!=0xffffffffu && (value>>24)!=0 ? 1.0f : 0.0f;
                };
                c3x_renderer::fidelity::ReliefSurface cliff_pickup_surface(world_coast.world().dimensions(),
                    (tile.tile_x+tile.tile_y)/2,(tile.tile_x-tile.tile_y)/2,tile_center_shore.distance,
                    cliff_world_lookup,pickup_source,cliff_shore_sample_at,cliff_pickup_river,pickup_dune,cliff_pickup_activity,
                    cliff_query_scratch.pickup_ground_samples,cliff_query_scratch.pickup_height_queries,separate_natural_relief);
                auto cliff_pickup_height_at=[&](float u,float v){ return cliff_pickup_surface.height(u,v); };
                auto cliff_natural_height_at=[&](float u,float v,float* support=nullptr){
                    auto compute=[&](){
                        std::array<float,2> value{};
                        value[0]=cliff_queries.height(natural,cliff_pickup_height_at,u,v,&value[1]);
                        return value;
                    };
                    auto value=retain_height_samples?cliff_query_scratch.height_samples.get(u,v,compute):compute();
                    if(support)*support=value[1];
                    return value[0];
                };
                c3x_renderer::fidelity::compile_cliff_surfaces(world_coast.world().dimensions(),cliff_bundle,cliff_input,
                    cliff_world_lookup,
                    [&](int c,int r){ return world_coast.world().index(c,r); },
                    [&](double u,double v){ return cliff_natural_height_at(float(u),float(v))-2.5f; },
                    [&](double u,double v){ return cliff_shore_sample_at(float(u),float(v)).distance; },
                    [&](unsigned i){ float h=0;for(auto const& v:cliff_bundle.assets[i].vertices)
                        h=std::max(h,v.position[2]*cliff_input.vertical_basis);return h; },
                    [&](int c,int r){return world_coast.cell(c,r,cliff_observe_coast);},
                    [&](bool is_small,unsigned seed){
                        auto group=find_feature_group(cliff_bundle,is_small?"cliff_small":"cliff_large");
                        auto selected=c3x_renderer::select_feature_placement(*group,seed);
                        if(!selected)return c3x_renderer::render_core::CliffRecipe{0,0,0};
                        return c3x_renderer::render_core::CliffRecipe{selected->asset_index,
                            selected->scale,selected->scale_variation};
                    },
                    cancelled,cliff_result);
                for(unsigned i=0;i<cliff_result.vertices.size();++i)if(!cliff_result.vertices[i].empty())
                    cliff_vertices[i].insert(cliff_vertices[i].end(),cliff_result.vertices[i].begin(),cliff_result.vertices[i].end());
                for(auto const& dependency:cliff_result.coast)coast_dependencies.emplace(dependency.first,dependency.second);
                for(auto const& dependency:cliff_world_dependencies)world_dependencies.emplace(dependency.first,dependency.second);
            }
            } // immutable routes and objects already resident on a world hit
            QueryPerformanceCounter(&phase_end);cliff_ticks+=phase_end.QuadPart-phase_time.QuadPart;phase_time=phase_end;
            std::unique_ptr<c3x_renderer::fidelity::TerrainSurfaces> prepared_terrain;
            std::array<std::vector<c3x_renderer::fidelity::MeshInstance>,22> forest_instances;
            std::array<c3x_renderer::render_core::SourceShadow::Bounds,22> forest_bounds;
            std::array<c3x_renderer::render_core::ProjectedMeshBounds,22> forest_projected;
            auto natural_found=natural_mesh_cache.find(natural_key);
            bool natural_hit=fidelity_profile && natural_found!=natural_mesh_cache.end();
            if(natural_hit && !natural.valid(natural_found->second.river_dependencies))natural_hit=false;
            if(natural_hit)for(auto const& dependency:natural_found->second.appearance_dependencies)
                if(topology_cache.appearance_revision(dependency.first)!=dependency.second){natural_hit=false;break;}
            if(natural_hit)for(auto const& dependency:natural_found->second.dependencies){
                auto current=topology_cache.current(dependency.first);
                if((current==nullptr?0:current->semantic)!=dependency.second){natural_hit=false;break;}
            }
            if(natural_hit)for(auto const& dependency:natural_found->second.coast_dependencies)
                if(world_coast.node_revision(dependency.first)!=dependency.second){natural_hit=false;break;}
            if(natural_hit)for(auto const& dependency:natural_found->second.world_dependencies)
                if(world_coast.world().at(dependency.first)!=dependency.second){natural_hit=false;break;}
            // City composition also emits non-natural buffers. Keep that path
            // in the normal compiler; all other natural layers are independent.
            NaturalTile pending_natural;
            if(shared_hit){
                natural_hit=false;++frame_natural_hits;
            }else if(natural_hit && cache_natural){
                auto& cached=natural_found->second;cached.used=tile_geometry_epoch;++frame_natural_hits;
                river_dependencies.insert(cached.river_dependencies.begin(),cached.river_dependencies.end());
                dependencies.insert(cached.dependencies.begin(),cached.dependencies.end());
                coast_dependencies.insert(cached.coast_dependencies.begin(),cached.coast_dependencies.end());
                world_dependencies.insert(cached.world_dependencies.begin(),cached.world_dependencies.end());
            }else{
                natural_hit=false;
                #include "source_fidelity/geometry.h"
            }
            QueryPerformanceCounter(&phase_end);terrain_prep_ticks+=phase_end.QuadPart-phase_time.QuadPart;phase_time=phase_end;
            auto ground_join_started=std::chrono::steady_clock::now();
            auto prepared_ground=prepared_world?std::move(prepared_world->ground):batched_ground?selected_ground_preparation.take(index):ground_task->take();
            if(batched_ground && !prepared_ground && !cancelled()){
                // A rejected/evicted result is never published as complete.
                // The exact compiler remains the bounded recovery path.
                ++ground_batch_fallbacks;
                std::atomic<bool> stop{false};prepared_ground=compile_ground(stop);
            }
            ground_join_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ground_join_started).count();
            if(!prepared_ground || cancelled())return false;
            ground_compile_ms+=prepared_ground->compile_ms;
            ground_ready_peak=std::max(ground_ready_peak,prepared_ground->bytes());
            if(!world_ground || !shared_hit)++ground_jobs;
            frame_ground_grid_hits+=prepared_ground->grid_hits;
            for(auto const& dependency:prepared_ground->world)world_dependencies.emplace(dependency.first,dependency.second);
            for(auto const& dependency:prepared_ground->coast)coast_dependencies.emplace(dependency.first,dependency.second);
            for(auto const& dependency:prepared_ground->topology)dependencies.emplace(dependency.first,dependency.second);
            for(auto const& dependency:prepared_ground->rivers)river_dependencies.emplace(dependency.first,dependency.second);
            // Legacy analytic object shadows append to terrain shadows in the
            // same layer. Keep their original combined packing/order.
            if(!pickup_profile)shadow_vertices.insert(shadow_vertices.begin(),
                prepared_ground->legacy_shadow.begin(),prepared_ground->legacy_shadow.end());
            pending_ground_grids=std::move(prepared_ground->pending_grids);
            if(!pending_ground_grids.empty()){
                CachedGroundTile incoming;
                incoming.signature=ground_signature;incoming.used=tile_geometry_epoch;incoming.x=tile.tile_x;incoming.y=tile.tile_y;
                incoming.grids=std::make_shared<std::vector<CachedGroundGrid>>(std::move(pending_ground_grids));
                ground_cache_proof.topology.insert(prepared_ground->topology.begin(),prepared_ground->topology.end());
                incoming.dependencies.assign(ground_cache_proof.topology.begin(),ground_cache_proof.topology.end());
                ground_cache_proof.coast.insert(prepared_ground->coast.begin(),prepared_ground->coast.end());
                incoming.coast_dependencies.assign(ground_cache_proof.coast.begin(),ground_cache_proof.coast.end());
                ground_cache_proof.rivers.insert(prepared_ground->rivers.begin(),prepared_ground->rivers.end());
                incoming.river_dependencies.assign(ground_cache_proof.rivers.begin(),ground_cache_proof.rivers.end());
                ground_cache_proof.world.insert(prepared_ground->world.begin(),prepared_ground->world.end());
                incoming.world_dependencies.assign(ground_cache_proof.world.begin(),ground_cache_proof.world.end());
                if(ground_hit){
                    // Allocate before moving any retained grids. A failed
                    // allocation must leave the old cache entry usable.
                    incoming.grids->reserve(incoming.grids->size()+retained_ground->second.grids->size());
                    for(auto& grid:*retained_ground->second.grids)incoming.grids->push_back(std::move(grid));
                    ground_grid_cache_bytes-=retained_ground->second.bytes;ground_grid_cache.erase(retained_ground);
                }
                incoming.bytes=sizeof(CachedGroundTile)+natural.proof_bytes(incoming.river_dependencies)+64+incoming.grids->capacity()*sizeof(CachedGroundGrid)+
                    incoming.dependencies.capacity()*sizeof(incoming.dependencies[0])+
                    incoming.coast_dependencies.capacity()*sizeof(incoming.coast_dependencies[0])+
                    incoming.world_dependencies.capacity()*sizeof(incoming.world_dependencies[0]);
                for(auto const& grid:*incoming.grids)incoming.bytes+=grid.vertices.capacity()*sizeof(Vertex)+grid.samples.capacity()*sizeof(grid.samples[0]);
                float cx=float(tile.tile_x)+float(frame.target_width-frame.tile_width-2*tile.anchor_x)/frame.tile_width;
                float cy=float(tile.tile_y)+float(frame.target_height-frame.tile_height-2*tile.anchor_y)/frame.tile_height;
                auto distance=[&](CachedGroundTile const& value){float dx=value.x-cx,dy=value.y-cy;return dx*dx+dy*dy;};
                bool admit=incoming.bytes<=natural_mesh_cache_budget;
                while(admit && !ground_grid_cache.empty() && (ground_grid_cache_bytes+incoming.bytes>natural_mesh_cache_budget ||
                        ground_grid_cache.size()>=natural_mesh_cache_capacity)){
                    auto victim=ground_grid_cache.begin();
                    for(auto it=ground_grid_cache.begin();it!=ground_grid_cache.end();++it)
                        if(distance(it->second)>distance(victim->second))victim=it;
                    if(distance(incoming)>distance(victim->second))admit=false;
                    else {ground_grid_cache_bytes-=victim->second.bytes;ground_grid_cache.erase(victim);}
                }
                if(admit){auto bytes=incoming.bytes;ground_grid_cache.emplace(ground_key,std::move(incoming));ground_grid_cache_bytes+=bytes;}
            }
            QueryPerformanceCounter(&phase_end);ground_ticks+=phase_end.QuadPart-phase_time.QuadPart;phase_time=phase_end;
            if(cache_natural && !natural_hit && !shared_hit){
                pending_natural.appearance_dependencies.assign(appearance_dependencies.begin(),appearance_dependencies.end());
                pending_natural.dependencies.assign(dependencies.begin(),dependencies.end());
                pending_natural.coast_dependencies.assign(coast_dependencies.begin(),coast_dependencies.end());
                pending_natural.river_dependencies.assign(river_dependencies.begin(),river_dependencies.end());
                pending_natural.world_dependencies.assign(world_dependencies.begin(),world_dependencies.end());
                pending_natural.used=tile_geometry_epoch;pending_natural.tile_width=frame.tile_width;
                pending_natural.tile_height=frame.tile_height;
                pending_natural.target_height=content_view_height;
                pending_natural.tile_x=tile.tile_x;pending_natural.tile_y=tile.tile_y;
            }
            c3x_renderer::fidelity::GroundProjection natural_projection{(tile.tile_x+tile.tile_y)/2,
                (tile.tile_x-tile.tile_y)/2,half_w,half_h,relief_projection_scale,float(content_view_height)};
            QueryPerformanceCounter(&phase_end);terrain_prep_ticks+=phase_end.QuadPart-phase_time.QuadPart;phase_time=phase_end;
            if (cancelled()) return false;
            CachedTileGeometry compiled;
            compiled.resource_anchors = std::move(tile_resource_anchors);
            compiled.replaces_resource = (build_replacement[index] & C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED) != 0;
            compiled.signature = tile_signature;
            compiled.compile_context=compile_context;
            compiled.world_objects=world_objects;compiled.world_ground=world_ground;compiled.source_tile_width=frame.tile_width;
            if(shared_hit && world_objects)for(auto const& d:shared_natural->second.anchor_dependencies)
                anchor_dependencies.push_back({d.first,{int(std::int64_t(d.second[0])*frame.tile_width/shared_natural->second.source_tile_width),
                    int(std::int64_t(d.second[1])*frame.tile_width/shared_natural->second.source_tile_width)}});
            compiled.tile_x=tile.tile_x;compiled.tile_y=tile.tile_y;
            compiled.version = ++tile_geometry_version;
            compiled.anchor_x = 0;
            compiled.anchor_y = 0;
            compiled.last_used = prewarming ? tile_geometry_epoch - 1 : tile_geometry_epoch;
            compiled.prefetched = prewarming;
            compiled.appearance_dependencies.assign(appearance_dependencies.begin(),appearance_dependencies.end());
            compiled.dependencies.assign(dependencies.begin(), dependencies.end());
            compiled.coast_dependencies.assign(coast_dependencies.begin(), coast_dependencies.end());
            compiled.river_dependencies.assign(river_dependencies.begin(),river_dependencies.end());
            compiled.world_dependencies.assign(world_dependencies.begin(), world_dependencies.end());
            compiled.anchor_dependencies = std::move(anchor_dependencies);
            std::size_t metadata_bytes = pickup_profile
                ? compiled.coast_dependencies.capacity() * sizeof(compiled.coast_dependencies[0]) +
                  compiled.world_dependencies.capacity() * sizeof(compiled.world_dependencies[0]) : 0;
            metadata_bytes += natural.proof_bytes(compiled.river_dependencies);
            metadata_bytes += sizeof(compiled.compile_context)+24; // per-frame validity receipt, including padding
            metadata_bytes += compiled.appearance_dependencies.capacity()*sizeof(compiled.appearance_dependencies[0]);
            metadata_bytes += compiled.resource_anchors.capacity()*sizeof(ResourceAnchor)+sizeof(compiled.binding);
            if(!city_chunks.empty())metadata_bytes+=sizeof(c3x_renderer::city_fidelity::Lighting)+
                city_chunks.front().lighting->lights.capacity()*sizeof(c3x_renderer::city_fidelity::Light)+
                city_chunks.front().lighting->blockers.capacity()*sizeof(c3x_renderer::city_fidelity::Lighting::Box);
            if(prepared_objects && !prepared_objects->city.empty()){
                auto const& lighting=*prepared_objects->city.front().lighting;
                metadata_bytes+=sizeof(lighting)+lighting.lights.capacity()*sizeof(lighting.lights[0])+lighting.blockers.capacity()*sizeof(lighting.blockers[0]);
            }
            if (!make_tile_cache_room(metadata_bytes)) return false;
            tile_geometry_cache_bytes += metadata_bytes;
            compiled.byte_count = metadata_bytes;
            try {
            std::array<c3x_renderer::render_core::ImmutableMeshUpload,2> mesh_uploads;
            auto shared_start=share_natural?(world_ground?0:world_objects?geometry_route:geometry_natural_terrain):geometry_layer_count;
            // Compute before cache_geometry_layer moves the flat grid into its
            // immutable buffer. Retained tiles carry the resulting absent
            // layers with the same coast/semantic dependencies as that grid.
            bool water_coverage=(world_ground && shared_hit) || !cull_empty_water || !environment_profile ||
                prepared_ground->water_coverage;
            for (std::size_t layer = 0; layer < geometry_layer_count; ++layer) {
                auto& mesh_upload=mesh_uploads[layer>=shared_start?1:0];
                if(world_ground && shared_hit)continue;
                if(prepared_objects){
                    auto adopt=[&](auto const& part,unsigned projection){
                        if(part.mesh.empty())return true;
                        std::vector<Vertex> empty;
                        if(!cache_geometry_layer(mesh_upload,empty,compiled.buffers[layer],prewarming,compiled.byte_count,foreground_pending,
                            false,false,nullptr,nullptr,nullptr,nullptr,projection,&part.mesh,
                            static_cast<ID3D11Buffer*>(prepared_objects->buffer.get()),part.vertex_offset,true,part.index_offset))return false;
                        compiled.byte_count+=compiled.buffers[layer].back().byte_count;return true;
                    };
                    if(layer==geometry_city && !prepared_objects->city.empty()){
                        for(auto const& part:prepared_objects->city){
                            if(part.mesh.empty())continue;
                            if(!adopt(part,4)){tile_geometry_cache_bytes-=compiled.byte_count;return false;}
                            auto& chunk=compiled.buffers[layer].back();chunk.city_material=part.material;chunk.city_environment=part.environment;
                            std::copy(part.atlas.begin(),part.atlas.end(),chunk.city_atlas);chunk.city_lighting=part.lighting;
                        }
                        continue;
                    }
                    unsigned const layers[]={geometry_route,geometry_feature,geometry_city,geometry_wall,geometry_mine,geometry_farm,geometry_site};
                    for(unsigned i=0;i<c3x_renderer::objects::layer_count;++i)if(layer==layers[i]){
                        if(!adopt(prepared_objects->layers[i],2)){tile_geometry_cache_bytes-=compiled.byte_count;return false;}
                    }
                }
                if(layer==geometry_city && !city_chunks.empty()){
                    for(auto&part:city_chunks){
                        if(part.vertices.empty())continue;
                        if(!cache_geometry_layer(mesh_upload,part.vertices,compiled.buffers[layer],prewarming,compiled.byte_count,foreground_pending,false,false,nullptr,nullptr,nullptr,nullptr,world_objects?4u:0u)){
                            tile_geometry_cache_bytes-=compiled.byte_count;return false;
                        }
                        auto&chunk=compiled.buffers[layer].back();chunk.city_material=part.material;chunk.city_environment=part.environment;
                        std::copy(part.atlas,part.atlas+4,chunk.city_atlas);chunk.city_lighting=part.lighting;compiled.byte_count+=chunk.byte_count;
                    }
                    city_chunks.clear();continue;
                }
                if(pickup_profile && (layer==geometry_bed || layer==geometry_water)) {
                    if(!water_coverage)continue;
                    // All flat layers share exact vertex/index data. Surface
                    // kind is the per-draw b1 value; no duplicate allocation.
                    compiled.buffers[layer]=compiled.buffers[geometry_underlay];
                    for(auto& chunk:compiled.buffers[layer]){
                        if(chunk.buffer)chunk.buffer->AddRef();if(chunk.indices)chunk.indices->AddRef();chunk.byte_count=0;
                    }
                    continue;
                }
                bool natural_layer=layer>=geometry_natural_terrain;
                if(shared_hit && (natural_layer || (world_objects && layer>=geometry_route)))continue;
                if(layer>=geometry_natural_forest0 && !forest_instances[layer-geometry_natural_forest0].empty()){
                    unsigned body=unsigned(layer-geometry_natural_forest0);
                    auto&instances=forest_instances[body];
                    if(!natural.ensure_instance_mesh(device,body))return false;
                    CachedVertexChunk chunk;chunk.byte_count=instances.capacity()*sizeof(instances[0])+sizeof(instances)+64;
                    if(!make_tile_cache_room(chunk.byte_count))return false;
                    chunk.instances=std::make_shared<std::vector<c3x_renderer::fidelity::MeshInstance> const>(std::move(instances));
                    compiled.buffers[layer].reserve(1);
                    auto const&mesh=natural.instance_meshes[body];chunk.buffer=mesh.vertices;chunk.indices=mesh.indices;
                    chunk.buffer->AddRef();chunk.indices->AddRef();chunk.index_count=mesh.count;chunk.vertex_stride=32;
                    chunk.version=compiled.version;chunk.world_bounds=forest_bounds[body];chunk.projected_bounds=forest_projected[body];
                    chunk.instance_material=natural.materials[natural.bodies[body].material].repeat?41.f:40.f;
                    auto b=chunk.projected_bounds.project((tile.tile_x+tile.tile_y)/2,(tile.tile_x-tile.tile_y)/2,frame.tile_width);
                    chunk.bounds={b[0],b[1],b[2],b[3]};
                    tile_geometry_cache_bytes+=chunk.byte_count;compiled.byte_count+=chunk.byte_count;
                    compiled.buffers[layer].push_back(std::move(chunk));continue;
                }
                auto const* cached_mesh=natural_layer && natural_hit?&natural_found->second.layers[layer-geometry_natural_terrain]:nullptr;
                auto* record_mesh=natural_layer && cache_natural && !natural_hit && !retain_ground_grids?&pending_natural.layers[layer-geometry_natural_terrain]:nullptr;
                bool reproject=natural_hit && (natural_found->second.tile_width!=frame.tile_width ||
                    natural_found->second.tile_height!=frame.tile_height || natural_found->second.target_height!=content_view_height);
                bool world_mesh=prepared_world && (layer<=geometry_river || layer==geometry_shadow ||
                    (prepared_terrain && layer>=geometry_natural_terrain && layer<=geometry_natural_mountain));
                unsigned world_vertex_offset=0,world_index_offset=0;
                if(world_mesh){
                    if(layer>=geometry_natural_terrain){world_vertex_offset=prepared_world->terrain_vertices[layer-geometry_natural_terrain];
                        world_index_offset=prepared_world->terrain_indices[layer-geometry_natural_terrain];}
                    else {unsigned i=layer==geometry_shadow?5u:unsigned(layer);world_vertex_offset=prepared_world->ground_vertices[i];world_index_offset=prepared_world->ground_indices[i];}
                }
                auto prior_chunks=compiled.buffers[layer].size();
                if (!cache_geometry_layer(mesh_upload,*tile_layers[layer], compiled.buffers[layer], prewarming, compiled.byte_count, foreground_pending, layer>=geometry_feature && layer<geometry_natural_terrain, natural_layer,
                        record_mesh,cached_mesh,reproject?&natural_projection:nullptr,
                        layer<=geometry_river?&ground_indices[layer]:
                        index_natural_grids && (layer==geometry_natural_terrain || layer==geometry_natural_terrain+2)
                            ?&natural_grid_indices[layer==geometry_natural_terrain?0:1]:nullptr,
                        world_ground && layer<geometry_route?3u:world_objects && layer>=geometry_route && layer<geometry_natural_terrain?
                            (layer>=geometry_cliff0?4u:2u):(world_objects && natural_layer?1u:0u),
                        layer<=geometry_river?&prepared_ground->meshes[layer]:layer==geometry_shadow && pickup_profile?&prepared_ground->meshes[5]:
                        prepared_terrain && layer>=geometry_natural_terrain && layer<=geometry_natural_mountain?&prepared_terrain->meshes[layer-geometry_natural_terrain]:nullptr,
                        world_mesh?static_cast<ID3D11Buffer*>(prepared_world->buffer.get()):
                        prepared_terrain && layer>=geometry_natural_terrain && layer<=geometry_natural_mountain?static_cast<ID3D11Buffer*>(prepared_terrain->vertex_buffer.get()):nullptr,
                        world_mesh?world_vertex_offset:
                        prepared_terrain && layer>=geometry_natural_terrain && layer<=geometry_natural_mountain?prepared_terrain->vertex_offset[layer-geometry_natural_terrain]:0u,
                        world_mesh && !(prepared_terrain && layer>=geometry_natural_terrain && layer<=geometry_natural_mountain &&
                            prepared_terrain->meshes[layer-geometry_natural_terrain].shared_grid),world_index_offset)) {
                    char detail[256];sprintf_s(detail,"tile=%d,%d layer=%u vertices=%u bytes=%llu cap=%llu built=%u reused=%u prewarming=%u",
                        tile.tile_x,tile.tile_y,unsigned(layer),unsigned(tile_layers[layer]->size()),
                        static_cast<unsigned long long>(tile_geometry_cache_bytes),static_cast<unsigned long long>(tile_geometry_cache_budget),
                        frame_tiles_built,frame_tiles_reused,unsigned(prewarming));
                    trace.write("mesh-cache-failed",detail,true);
                    tile_geometry_cache_bytes -= compiled.byte_count;
                    release_geometry_vertex_buffers(compiled.buffers);
                    return false;
                }
                for(auto i=prior_chunks;i<compiled.buffers[layer].size();++i)compiled.byte_count+=compiled.buffers[layer][i].byte_count;
            }
            // Keep allocation boundaries equal to residency/eviction owners:
            // camera-specific layers and shared world content can retire alone.
            // The two owners' CreateBuffer calls are independent (different
            // ImmutableMeshUpload instances, same thread-safe device), so when
            // both are populated for this tile run them concurrently instead
            // of serially; this changes nothing about ordering, cancellation,
            // or the success/failure contract below, only wall-clock cost.
            ID3D11Buffer* allocations[2]={nullptr,nullptr};
            bool create_ok[2]={true,true};
            bool second_owner_needed=mesh_uploads[0].size()>0 && mesh_uploads[1].size()>0;
            std::thread second_owner_thread;
            if(second_owner_needed)second_owner_thread=std::thread([&]{
                create_ok[1]=mesh_uploads[1].create(device,&allocations[1]);
            });
            if(mesh_uploads[0].size() && !mesh_uploads[0].create(device,&allocations[0]))create_ok[0]=false;
            if(second_owner_needed)second_owner_thread.join();
            else if(mesh_uploads[1].size() && !mesh_uploads[1].create(device,&allocations[1]))create_ok[1]=false;
            if((mesh_uploads[0].size() && !create_ok[0]) || (mesh_uploads[1].size() && !create_ok[1])){
                if(allocations[0])allocations[0]->Release();
                if(allocations[1])allocations[1]->Release();
                tile_geometry_cache_bytes-=compiled.byte_count;return false;
            }
            for(unsigned owner=0;owner<2;++owner)if(mesh_uploads[owner].size()){
                ID3D11Buffer* allocation=allocations[owner];
                for(unsigned layer=0;layer<geometry_layer_count;++layer)if(unsigned(layer>=shared_start)==owner)
                    for(auto& chunk:compiled.buffers[layer]){
                    // Buffer and indices are independent: a natural layer may
                    // already carry its own worker-created vertex buffer while
                    // still needing this shared per-tile buffer for indices.
                    if(!chunk.buffer){chunk.buffer=allocation;allocation->AddRef();}
                    if(!chunk.indices){chunk.indices=allocation;allocation->AddRef();}
                }
                allocation->Release();++frame_content_uploads;
            }
            for(auto&indices:natural_grid_indices)indices.clear();
            } catch (...) {
                tile_geometry_cache_bytes -= compiled.byte_count;
                return false; // compiled owns every successfully uploaded buffer
            }
            if(cache_natural && !natural_hit && !shared_hit && !retain_ground_grids){
                pending_natural.bytes=sizeof(NaturalTile)+natural.proof_bytes(pending_natural.river_dependencies)+pending_natural.appearance_dependencies.capacity()*sizeof(pending_natural.appearance_dependencies[0])+pending_natural.dependencies.capacity()*sizeof(pending_natural.dependencies[0])+
                    pending_natural.coast_dependencies.capacity()*sizeof(pending_natural.coast_dependencies[0])+
                    pending_natural.world_dependencies.capacity()*sizeof(pending_natural.world_dependencies[0]);
                for(auto const& mesh:pending_natural.layers)pending_natural.bytes+=mesh.vertices.capacity()*sizeof(mesh.vertices[0])+mesh.indices.capacity()*sizeof(UINT);
                auto old=natural_mesh_cache.find(natural_key);
                if(old!=natural_mesh_cache.end()){natural_mesh_cache_bytes-=old->second.bytes;natural_mesh_cache.erase(old);}
                constexpr std::size_t budget=natural_mesh_cache_budget;
                if(pending_natural.bytes<=budget){
                    float center_x=float(tile.tile_x)+float(frame.target_width-frame.tile_width-2*tile.anchor_x)/frame.tile_width;
                    float center_y=float(tile.tile_y)+float(frame.target_height-frame.tile_height-2*tile.anchor_y)/frame.tile_height;
                    auto distance=[&](NaturalTile const& value){float dx=value.tile_x-center_x,dy=value.tile_y-center_y;return dx*dx+dy*dy;};
                    bool admit=true;
                    while(!natural_mesh_cache.empty() && (natural_mesh_cache_bytes+pending_natural.bytes>budget || natural_mesh_cache.size()>=natural_mesh_cache_capacity)){
                        auto oldest=natural_mesh_cache.begin();
                        // Keep the central working set when the wider view
                        // exceeds this tier, avoiding sequential-scan thrash.
                        for(auto it=natural_mesh_cache.begin();it!=natural_mesh_cache.end();++it)if(distance(it->second)>distance(oldest->second))oldest=it;
                        if(distance(pending_natural)>distance(oldest->second)){admit=false;break;}
                        natural_mesh_cache_bytes-=oldest->second.bytes;natural_mesh_cache.erase(oldest);
                    }
                    if(admit){natural_mesh_cache_bytes+=pending_natural.bytes;natural_mesh_cache.emplace(natural_key,std::move(pending_natural));}
                }
            }
            if(share_natural){
                if(shared_hit)compiled.natural_content=shared_natural->second.binding;
                if(!shared_hit){
                    CachedTileGeometry shared;
                    shared.shared_natural=true;shared.signature=natural_key;
                    shared.world_objects=world_objects;shared.world_ground=world_ground;shared.source_tile_width=frame.tile_width;
                    shared.tile_x=tile.tile_x;shared.tile_y=tile.tile_y;
                    shared.version=compiled.version;shared.last_used=tile_geometry_epoch;
                    if(animated_view)shared.animation_epoch=tile_geometry_epoch;
                    try {
                        shared.appearance_dependencies=compiled.appearance_dependencies;
                        if(world_objects){shared.resource_anchors=compiled.resource_anchors;shared.replaces_resource=compiled.replaces_resource;
                            shared.anchor_dependencies=compiled.anchor_dependencies;}
                        shared.dependencies=compiled.dependencies;
                        shared.coast_dependencies=compiled.coast_dependencies;
                        shared.river_dependencies=compiled.river_dependencies;
                        shared.world_dependencies=compiled.world_dependencies;
                    } catch(...) {
                        tile_geometry_cache_bytes-=compiled.byte_count;return false;
                    }
                    // Count this owner's metadata separately from camera metadata.
                    std::size_t metadata=sizeof(CachedTileGeometry)+natural.proof_bytes(shared.river_dependencies)+
                        shared.appearance_dependencies.capacity()*sizeof(shared.appearance_dependencies[0])+
                        shared.dependencies.capacity()*sizeof(shared.dependencies[0])+
                        shared.coast_dependencies.capacity()*sizeof(shared.coast_dependencies[0])+
                        shared.world_dependencies.capacity()*sizeof(shared.world_dependencies[0])+
                        shared.anchor_dependencies.capacity()*sizeof(shared.anchor_dependencies[0])+
                        shared.resource_anchors.capacity()*sizeof(ResourceAnchor);
                    if(!make_tile_cache_room(metadata)){
                        tile_geometry_cache_bytes-=compiled.byte_count;return false;
                    }
                    for(std::size_t layer=world_ground?0:world_objects?geometry_route:geometry_natural_terrain;layer<geometry_layer_count;++layer){
                        shared.buffers[layer]=std::move(compiled.buffers[layer]);
                        for(auto const& chunk:shared.buffers[layer])shared.byte_count+=chunk.byte_count;
                    }
                    if(world_objects && !shared.buffers[geometry_city].empty() && shared.buffers[geometry_city].front().city_lighting){
                        auto const& light=*shared.buffers[geometry_city].front().city_lighting;
                        auto bytes=sizeof(light)+light.lights.capacity()*sizeof(light.lights[0])+light.blockers.capacity()*sizeof(light.blockers[0]);
                        shared.byte_count+=bytes;
                    }
                    compiled.byte_count-=shared.byte_count;
                    shared.byte_count+=metadata;tile_geometry_cache_bytes+=metadata;
                    std::size_t shared_bytes=shared.byte_count;
                    try {
                        auto inserted=tile_geometry_cache.emplace(natural_key,std::move(shared));
                        try {
                            inserted->second.binding=resident_content.bind(inserted->second);
                            if(!inserted->second.binding.generation)throw std::bad_alloc();
                            compiled.natural_content=inserted->second.binding;
                        }catch(...){tile_geometry_cache.erase(inserted);throw;}
                    }
                    catch(...){tile_geometry_cache_bytes-=compiled.byte_count+shared_bytes;return false;}
                }
            }
            if(!make_tile_cache_room(0)){
                tile_geometry_cache_bytes-=compiled.byte_count;return false;
            }
            std::size_t compiled_bytes = compiled.byte_count;
            decltype(tile_geometry_cache)::iterator inserted;
            try {
                inserted = tile_geometry_cache.emplace(tile_signature, std::move(compiled));
                try {
                    inserted->second.binding=resident_content.bind(inserted->second);
                    if(!inserted->second.binding.generation)throw std::bad_alloc();
                }catch(...){tile_geometry_cache.erase(inserted);throw;}
            } catch (...) {
                // Container insertion can allocate after GPU upload. RAII owns
                // the buffers even if it moved the value before allocation failed.
                tile_geometry_cache_bytes -= compiled_bytes;
                return false;
            }
            if (prewarming) {
                topology_cache.attach(tile,inserted->second.binding);
                prefetched_geometry_bytes += inserted->second.byte_count;
                prepared_footprint = tile_footprint(inserted->second, tile);
                if(batch_preparing)continue;
                return true;
            }
            geometry_cache.tile_keys[index]=inserted->second.binding;
            if(retained_world)topology_cache.attach(tile,inserted->second.binding);
            else append_tile_geometry(inserted->second, tile, animated_view);
            QueryPerformanceCounter(&phase_end);upload_ticks+=phase_end.QuadPart-phase_time.QuadPart;
        }
        auto ground_drain_started=std::chrono::steady_clock::now();
        ground_batch_lease.finish();
        object_lease.queue.clear();
        world_lease.queue.clear();
        double ground_drain_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ground_drain_started).count();
        if(pickup_profile && !prewarming) {
            char detail[512];sprintf_s(detail,"built=%u reused=%u ground_ms=%.3f features_ms=%.3f cliffs_ms=%.3f terrain_prep_ms=%.3f upload_ms=%.3f bytes=%llu natural_hits=%u natural_bytes=%zu ground_grid_hits=%u ground_grid_bytes=%zu",
                frame_tiles_built,frame_tiles_reused,trace.milliseconds(ground_ticks),trace.milliseconds(feature_ticks),
                trace.milliseconds(cliff_ticks),trace.milliseconds(terrain_prep_ticks),trace.milliseconds(upload_ticks),static_cast<unsigned long long>(tile_geometry_cache_bytes),frame_natural_hits,natural_mesh_cache_bytes,frame_ground_grid_hits,ground_grid_cache_bytes);
            trace.write("mesh-phases",detail,true);
            if(world_batch_enabled){auto stats=world_lease.queue.statistics();
                sprintf_s(detail,"scheduled=%u consumed=%llu rejected=%llu evicted=%llu recovery=%u workers=%u worker_ms=%.3f join_ms=%.3f ground_ms=%.3f terrain_ms=%.3f object_ms=%.3f upload_ms=%.3f queue_peak_bytes=%zu gpu_bytes=%zu",
                    world_jobs,stats.consumed,stats.rejected,stats.evicted,world_recovery,cpu_terrain_workers,stats.cpu_ms,world_join_ms,
                    world_ground_ms,world_terrain_ms,world_object_ms,world_upload_ms,stats.peak_bytes,world_gpu_bytes);
                trace.write("world-preparation",detail,true);
            }
            sprintf_s(detail,"rigid_instances=%u terrain_routes=%u city_rigid_parts=%u city_deformed_parts=%u city_fallbacks_omitted=%u",
                object_instances,object_routes,city_rigid_parts,city_deformed_parts,city_fallbacks_omitted);
            trace.write("object-descriptions",detail,true);
            auto object_stats=object_lease.queue.statistics();
            sprintf_s(detail,"scheduled=%u consumed=%llu rejected=%llu evicted=%llu recovery=%u worker_ms=%.3f join_ms=%.3f queue_peak_bytes=%zu gpu_bytes=%zu workers=%u",
                object_jobs,object_stats.consumed,object_stats.rejected,object_stats.evicted,object_recovery,
                object_stats.cpu_ms,object_join_ms,object_stats.peak_bytes,object_gpu_bytes,object_worker && !world_batch_enabled?1u:0u);
            trace.write("object-preparation",detail,true);
            sprintf_s(detail,"jobs=%u concurrent=%u compile_ms=%.3f join_ms=%.3f ready_peak_bytes=%zu",
                ground_jobs,unsigned(ground_concurrent),ground_compile_ms,ground_join_ms,ground_ready_peak);
            trace.write("ground-preparation",detail,true);
            auto ground_stats=selected_ground_preparation.statistics();
            sprintf_s(detail,"scheduled=%u built=%llu consumed=%llu cancelled=%llu rejected=%llu evicted=%llu recovery=%u workers=%u schedule_ms=%.3f drain_ms=%.3f worker_ms=%.3f queue_peak_bytes=%zu",
                ground_batch_jobs,ground_stats.built-ground_batch_before.built,ground_stats.consumed-ground_batch_before.consumed,
                ground_stats.cancelled-ground_batch_before.cancelled,ground_stats.rejected-ground_batch_before.rejected,
                ground_stats.evicted-ground_batch_before.evicted,ground_batch_fallbacks,ground_batch_enabled?2u:0u,
                ground_schedule_ms,ground_drain_ms,ground_stats.cpu_ms-ground_batch_before.cpu_ms,ground_stats.peak_bytes);
            trace.write("ground-selected",detail,true);
            sprintf_s(detail,"pixels=%u mountain_cells=%u rocky_cells=%u shared_layouts=%zu shared_index_bytes=%zu shared_index_reuses=%u",
                patch_pixels,patch_detail.mountain,patch_detail.rocky_ground,terrain_patch_indices.size(),terrain_patch_index_bytes,frame_patch_index_reuses);
            trace.write("terrain-patches",detail,true);
            if(profiling){
                sprintf_s(detail,"ground_ms=%.3f surface_decals_ms=%.3f relief_ms=%.3f vegetation_floor_ms=%.3f city_ms=%.3f forest_ms=%.3f",
                    trace.milliseconds(natural_phase_ticks[0]),trace.milliseconds(natural_phase_ticks[1]),
                    trace.milliseconds(natural_phase_ticks[2]),trace.milliseconds(natural_phase_ticks[3]),
                    trace.milliseconds(natural_phase_ticks[4]),trace.milliseconds(natural_phase_ticks[5]));
                trace.write("natural-mesh-phases",detail,true);
            }
            sprintf_s(detail,"shore_hits=%zu shore_misses=%zu material_hits=%zu material_misses=%zu height_queries=%zu scratch_bytes=%zu",
                shore_samples.hits,shore_samples.misses,pickup_ground_samples.hits,pickup_ground_samples.misses,
                pickup_height_queries,
                shore_samples.bytes()+pickup_ground_samples.bytes());
            trace.write("query-cache",detail,true);
            sprintf_s(detail,"hits=%zu misses=%zu bytes=%zu enabled=%u",natural_height_samples.hits,
                natural_height_samples.misses,natural_height_samples.bytes(),unsigned(retain_height_samples));
            trace.write("natural-height-cache",detail,true);
        }
        if (prewarming) return true;
        // Construction is complete. Current occurrences choose protected world
        // content; capture ordering/anchors and replacement ownership stay exact.
        if(retained_world){
            auto assembly_started=std::chrono::steady_clock::now();
            for(std::size_t i=0;i<frame.tile_count;++i){
                auto handle=geometry_cache.tile_keys[i];if(!handle.generation)continue;
                auto owner=resident_content.resolve(handle);if(!owner)return false;
                append_tile_geometry(*owner,frame.tiles[i],animated_view);
            }
            frame_tile_append_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-assembly_started).count();
        }
        // Off-screen caster geometry contributes shadows but never replaces a
        // native draw. Keep the public ownership array aligned with RENDER,
        // including the copies retained for bitmap and translated-cache hits.
        for (c3x_renderer_u32 i = 0; i < frame.tile_count; ++i)
            if ((frame.tiles[i].tile_flags & C3X_RENDERER_TILE_RENDER) == 0)
                build_replacement[i] = 0;
        geometry_world_revision = frame.world_topology_revision;
        geometry_cache.signature = signature;
        if (frame.tile_count != 0)
            geometry_cache.tiles.assign(frame.tiles, frame.tiles + frame.tile_count);
        geometry_cache.replacement_flags = build_replacement;
        geometry_cache.fallback_indices = build_fallback;
        geometry_cache.rendered_tile_count = textured_tile_count;
        geometry_cache.fallback_tile_count = fallback_tile_count;
        geometry_cache.textured_tile_count = textured_tile_count;
        geometry_cache.valid = true;
        if (frame_tiles_built == 0 && frame_tiles_reused != 0) {
            if (cache_hits != 0xffffffffu) ++cache_hits;
            if (cache_misses != 0) --cache_misses;
        }
        } else {
            build_replacement = geometry_cache.replacement_flags;
            build_fallback = geometry_cache.fallback_indices;
            textured_tile_count = geometry_cache.textured_tile_count;
            frame_tiles_reused = textured_tile_count;
            fallback_tile_count = geometry_cache.fallback_tile_count;
        }
        std::vector<c3x_renderer::TileFootprint> current_footprints = geometry_footprints;
        for (auto & footprint : current_footprints) {
            footprint.anchor_x += geometry_translation_x;
            footprint.anchor_y += geometry_translation_y;
        }
        // Only the exact bitmap that owns these footprints can donate pixels.
        // An LRU/subset hit may have changed the bitmap without rebuilding meshes.
        if (retain_raster && !reuse_raster && cache_valid &&
            bitmap_footprint_signature.complete == cached_signature.complete &&
            signature.camera == cached_signature.camera &&
            signature.environment == cached_signature.environment &&
            signature.wrap == cached_signature.wrap) {
            std::vector<c3x_renderer::PixelRect> damage;
            if (c3x_renderer::scroll_damage(bitmap_footprints, current_footprints,
                    width, height, raster_dx, raster_dy, damage, pickup_profile ? 0 : 32) &&
                same_grid(raster_dx,raster_dy) &&
                (allow_odd_raster || ((raster_dx & 1) == 0 && (raster_dy & 1) == 0))) {
                raster_rects.clear(); raster_draw_pixels = 0;
                for (auto const & rect : damage) dirty_rect(rect.left, rect.top, rect.right, rect.bottom);
                raster_reused_pixels = static_cast<c3x_renderer_u32>(width * height) - raster_draw_pixels;
                reuse_raster = raster_reused_pixels != 0;
                overlap = {std::max(0, raster_dx), std::max(0, raster_dy),
                           std::min(width, width + raster_dx), std::min(height, height + raster_dy)};
            }
        }
        if (reuse_raster) frame_cache_path = "raster-scroll";
        struct BlockCopy { D3D11_RECT rect; int cache_index, origin_x, origin_y; };
        std::vector<BlockCopy> block_copies;
        int block_phase_x=0,block_phase_y=0;
        if (retain_raster && !world_raster_grid && !pixel_blocks.blocks.empty() &&
            block_phase(current_footprints,frame.tile_width,frame.tile_height,block_phase_x,block_phase_y)) {
            std::vector<D3D11_RECT> misses;
            for (auto const & dirty : raster_rects)
                for (int y=c3x_renderer::block_floor(dirty.top,block_phase_y);y<dirty.bottom;y+=128)
                    for(int x=c3x_renderer::block_floor(dirty.left,block_phase_x);x<dirty.right;x+=128) {
                        D3D11_RECT part={std::max<LONG>(x,dirty.left),std::max<LONG>(y,dirty.top),
                            std::min<LONG>(x+128,dirty.right),std::min<LONG>(y+128,dirty.bottom)};
                        auto key=c3x_renderer::block_key(current_footprints,{x,y,x+128,y+128});
                        int found=pixel_blocks.find(key);
                        if(found>=0) {
                            block_copies.push_back({part,found,x,y});
                            raster_cached_pixels+=(part.right-part.left)*(part.bottom-part.top);
                        } else misses.push_back(part);
                    }
            // Merge adjacent misses without adding pixels or overlapping draws.
            c3x_renderer::merge_adjacent_rectangles(misses);
            raster_rects=std::move(misses);
            raster_draw_pixels-=raster_cached_pixels;
            if(raster_cached_pixels) frame_cache_path="pixel-blocks";
        }

        if(!restored_viewport.empty()){
            raster_rects.clear();block_copies.clear();reuse_raster=false;
            cache_valid=false; // Pixel identity is committed with metadata below.
            pixels=std::move(restored_viewport);raster_draw_pixels=0;
            raster_cached_pixels=static_cast<c3x_renderer_u32>(width*height);
            frame_cache_path="viewport-lru-animation";
        }
        LARGE_INTEGER geometry_finished = {}, draw_finished = {}, readback_finished = {};
        QueryPerformanceCounter(&geometry_finished);
        frame_geometry_ticks = geometry_finished.QuadPart - started.QuadPart;
        if(cpu_terrain_enabled){auto stats=terrain_preparation.statistics();char detail[384];
            sprintf_s(detail,"workers=%u active_peak=%u built=%llu consumed=%llu cancelled=%llu rejected=%llu evicted=%llu invalidated=%llu ready_bytes=%zu peak_bytes=%zu pending=%zu cpu_ms=%.3f wait_ms=%.3f ready_cap=%zu",
                active_terrain_workers,stats.active_peak,stats.built,stats.consumed,stats.cancelled,stats.rejected,stats.evicted,stats.invalidated,
                stats.bytes,stats.peak_bytes,stats.pending,stats.cpu_ms,stats.wait_ms,cpu_preparation_budget);
            trace.write("cpu-content-preparation",detail,true);
        }
        trace.write("geometry-ready", frame_cache_path);
        if(cancelled())return false;
        memory_sample("geometry-ready");
        if(shared_scene_surface && world_preparation){
            int pad=256;unsigned w=unsigned(width)+8,h=unsigned(height)+8;
            while(pad && std::size_t(w)*h*240u+std::size_t(w+pad*2)*(h+pad*2)*192u>1408u*1024u*1024u)pad-=32;
            bool compatible=scene_guard_pad==pad && scene_guard_context==region_context &&
                scene_guard_depth_origin==std::uint64_t(scene_depth_origin) && scene_static_signature==cached_signature.complete;
            scene_guard_pad=pad;scene_guard_failed=false;
            if(!scene_guard.configure(int(w)+pad*2,int(h)+pad*2))return false;
            auto previous=bitmap_footprints,current=current_footprints;
            for(auto& f:previous){f.anchor_x+=pad+4;f.anchor_y+=pad+4;}
            for(auto& f:current){f.anchor_x+=pad+4;f.anchor_y+=pad+4;}
            int dx=0,dy=0;std::vector<c3x_renderer::PixelRect> damage;
            if(compatible && c3x_renderer::scroll_damage(previous,current,scene_guard.width,scene_guard.height,dx,dy,damage,0)){
                std::vector<D3D11_RECT> logical;for(auto r:damage)logical.push_back({r.left,r.top,r.right,r.bottom});
                scene_guard.invalidate(c3x_renderer::render_core::scene_physical(scene_guard.width,scene_guard.height,region_origin_x,region_origin_y,logical));
            }else scene_guard.invalidate_all();
            scene_guard_context=region_context;scene_guard_depth_origin=std::uint64_t(scene_depth_origin);
        }else if(scene_guard_pad){scene_guard_pad=0;scene_guard.reset();scene_guard_context.clear();scene_scratch.reset();scene_static_signature=0;}
        if(shared_scene_surface){
            scene_overlap=false;scene_damage.clear();
            std::vector<c3x_renderer::PixelRect> damage;
            if(reuse_raster && scene_static_signature==cached_signature.complete &&
               c3x_renderer::scroll_damage(bitmap_footprints,current_footprints,width,height,scene_dx,scene_dy,damage,0) && damage.size()<=16){
                for(auto const& rect:damage)scene_damage.push_back({rect.left==0?0:rect.left+4,rect.top==0?0:rect.top+4,
                    rect.right==width?width+8:rect.right+4,rect.bottom==height?height+8:rect.bottom+4});
                scene_overlap=true;
            }
            raster_rects.clear();block_copies.clear();reuse_raster=false;
        }
        if(profiling && !raster_rects.empty())gpu_telemetry.begin(device,context,trace.sequence.load());
        c3x_renderer::render_core::GpuFrameTelemetry::Scope timing_scope{gpu_telemetry,context};
        if (!raster_rects.empty() && !submit_geometry(geometry_vertex_buffers, raster_rects,
                geometry_viewport_settings, render_target, depth_target,
                c3x_renderer::power_of_two_extent(width), c3x_renderer::power_of_two_extent(height),foreground_pending,
                false,true,nullptr,false,nullptr,nullptr,city_profile?scene_region_size:128,raster_grid_phase[0],raster_grid_phase[1])) return false;
        gpu_telemetry.draw_end(context);

        QueryPerformanceCounter(&draw_finished);
        frame_draw_ticks = draw_finished.QuadPart - geometry_finished.QuadPart;
        if(cancelled())return false;
        trace.write("readback-begin", "includes pending GPU execution");
        // The retained CPU bitmap already owns the unchanged overlap. Transfer
        // only freshly rasterized rectangles; the staging surface outside them
        // may be stale and must never be read below.
        for (D3D11_RECT const & rect : raster_rects) {
            D3D11_BOX box = {static_cast<UINT>(rect.left), static_cast<UINT>(rect.top), 0,
                static_cast<UINT>(rect.right), static_cast<UINT>(rect.bottom), 1};
            context->CopySubresourceRegion(readback_texture, 0, box.left, box.top, 0,
                render_texture, 0, &box);
        }
        gpu_telemetry.end(context);
        LARGE_INTEGER map_begin={},map_end={};QueryPerformanceCounter(&map_begin);
        D3D11_MAPPED_SUBRESOURCE mapped = {};
        if(!raster_rects.empty())++frame_output_readbacks;
        HRESULT hr = raster_rects.empty() ? S_OK : context->Map(readback_texture, 0, D3D11_MAP_READ, 0, &mapped);
        QueryPerformanceCounter(&map_end);
        if (FAILED(hr)) {
            OutputDebugStringA("[C3X renderer] native-failure=readback\n");
            return false;
        }
        if(cancelled()){
            if(!raster_rects.empty())context->Unmap(readback_texture,0);
            return false;
        }
        // Until here cancellation leaves the previous CPU bitmap untouched.
        // From the first write until metadata commit it cannot donate pixels.
        cache_valid=false;
        if (reuse_raster) {
            // Copy in the safe vertical direction, using memmove for horizontal
            // overlap. No second full-resolution CPU scratch bitmap is needed.
            int y = raster_dy > 0 ? overlap.bottom - 1 : overlap.top;
            int end = raster_dy > 0 ? overlap.top - 1 : overlap.bottom;
            int step = raster_dy > 0 ? -1 : 1;
            for (; y != end; y += step)
                std::memmove(pixels.data() + static_cast<std::size_t>(y) * width + overlap.left,
                    pixels.data() + static_cast<std::size_t>(y - raster_dy) * width + overlap.left - raster_dx,
                    static_cast<std::size_t>(overlap.right - overlap.left) * sizeof(std::uint32_t));
        }
        for (auto const & copy : block_copies)
            for (LONG y=copy.rect.top;y<copy.rect.bottom;++y)
                std::memcpy(pixels.data()+static_cast<std::size_t>(y)*width+copy.rect.left,
                    pixel_blocks.blocks[copy.cache_index].pixels.data()+(y-copy.origin_y)*128+copy.rect.left-copy.origin_x,
                    (copy.rect.right-copy.rect.left)*sizeof(std::uint32_t));
        for (D3D11_RECT const & rect : raster_rects)
            for (LONG y = rect.top; y < rect.bottom; ++y)
                std::memcpy(pixels.data() + static_cast<std::size_t>(y) * width + rect.left,
                    static_cast<std::uint8_t const *>(mapped.pData) + static_cast<std::size_t>(y) * mapped.RowPitch +
                        static_cast<std::size_t>(rect.left) * sizeof(std::uint32_t),
                    static_cast<std::size_t>(rect.right - rect.left) * sizeof(std::uint32_t));
        if (!raster_rects.empty()) context->Unmap(readback_texture, 0);
        QueryPerformanceCounter(&readback_finished);
        frame_readback_ticks = readback_finished.QuadPart - draw_finished.QuadPart;
        if(profiling){
            char detail[320];sprintf_s(detail,"map_wait_ms=%.3f cpu_copy_ms=%.3f draws=%llu parameter_updates=%llu bounds_tests=%llu raster_rects=%zu",
                trace.milliseconds(map_end.QuadPart-map_begin.QuadPart),trace.milliseconds(readback_finished.QuadPart-map_end.QuadPart),
                frame_draw_calls,frame_parameter_updates,frame_bounds_tests,raster_rects.size());
            trace.write("submission-phases",detail,true);
            memory_sample("readback-complete");
        }

        bitmap_footprints = std::move(current_footprints);
        bitmap_footprint_signature = signature;
        cached_rendered_tile_count = textured_tile_count;
        cached_fallback_tile_count = fallback_tile_count;
        cached_textured_tile_count = textured_tile_count;
        cached_visible_animation_count = frame.visible_animation_count;
        cached_request_continuous_redraw = frame.visible_animation_count != 0;
        cached_signature = signature;
        previous_signature = signature;
        previous_content_revision = content_revision;
        cache_valid = true;
        if (frame.tile_count == 0)
            cached_tiles.clear();
        else
            cached_tiles.assign(frame.tiles, frame.tiles + frame.tile_count);
        cached_replacement_tile_flags = replacement_tile_flags;
        if (cache_valid && !shared_scene_surface) {
            for (auto existing = viewport_cache.begin(); existing != viewport_cache.end(); ++existing) {
                if (existing->signature.complete == signature.complete) {
                    viewport_cache_bytes -= existing->byte_count;
                    viewport_cache.erase(existing);
                    break;
                }
            }
            CachedViewport stored;
            stored.byte_count = pixels.size() * sizeof(std::uint32_t) +
                cached_tiles.size() * sizeof(c3x_renderer_tile_v1) +
                cached_replacement_tile_flags.size() * sizeof(c3x_renderer_u32) +
                geometry_cache.tile_keys.size()*sizeof(geometry_cache.tile_keys[0]) +
                geometry_cache.fallback_indices.size()*sizeof(c3x_renderer_u32);
            while (!viewport_cache.empty() &&
                   (viewport_cache.size() >= viewport_cache_capacity ||
                    viewport_cache_bytes + stored.byte_count > viewport_cache_budget)) {
                viewport_cache_bytes -= viewport_cache.front().byte_count;
                viewport_cache.erase(viewport_cache.begin());
                if (cache_evictions != 0xffffffffu)
                    ++cache_evictions;
            }
            stored.signature = signature;
            stored.pixels = pixels;
            stored.tiles = cached_tiles;
            stored.replacement_flags = cached_replacement_tile_flags;
            stored.tile_keys=geometry_cache.tile_keys;
            stored.fallback_indices=geometry_cache.fallback_indices;
            stored.rendered_tile_count = cached_rendered_tile_count;
            stored.fallback_tile_count = cached_fallback_tile_count;
            stored.textured_tile_count = cached_textured_tile_count;
            if (stored.byte_count <= viewport_cache_budget) {
                viewport_cache_bytes += stored.byte_count;
                viewport_cache.push_back(std::move(stored));
            }
        }
        QueryPerformanceCounter(&finished);
        return fill_output(frame, output, invalidations, finished.QuadPart - started.QuadPart);
    }

};

// Only the calling/UI thread creates, uses and destroys these GDI resources.
// D3D state and its reset/destructor have no ownership of the blit surface.
class MapBlitter {
    HDC blit_dc = nullptr;
    HBITMAP blit_bitmap = nullptr;
    HGDIOBJ blit_previous_bitmap = nullptr;
    void * blit_bits = nullptr;
    int blit_width = 0, blit_height = 0;
    int blit_destination_bits = -1;
    unsigned blit_destination_green_mask = 0;
    c3x_renderer::ColorRoundingTable blit_round5{}, blit_round6{};
    bool blit_rounding_ready = false;
    int last_black_x=-1,last_black_y=-1,last_black_length=0;
public:
    MapBlitter() = default;
    MapBlitter(MapBlitter const&) = delete;
    MapBlitter& operator=(MapBlitter const&) = delete;
    ~MapBlitter() {reset_blit_surface();}
    void reset_blit_surface() {
        if (blit_dc != nullptr && blit_previous_bitmap != nullptr)
            SelectObject(blit_dc, blit_previous_bitmap);
        blit_previous_bitmap = nullptr;
        if (blit_bitmap != nullptr)
            DeleteObject(blit_bitmap);
        blit_bitmap = nullptr;
        if (blit_dc != nullptr)
            DeleteDC(blit_dc);
        blit_dc = nullptr;
        blit_bits = nullptr;
        blit_width = blit_height = 0;
    }

    template<class Trace>
    bool blit(c3x_renderer_output_v1 const & output, HDC destination, int phase_x, int phase_y, Trace& trace) {
        if (blit_dc == nullptr || blit_bitmap == nullptr ||
            blit_width != output.width || blit_height != output.height) {
            reset_blit_surface();
            BITMAPINFO info = {};
            info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            info.bmiHeader.biWidth = output.width;
            info.bmiHeader.biHeight = -output.height;
            info.bmiHeader.biPlanes = 1;
            info.bmiHeader.biBitCount = 32;
            info.bmiHeader.biCompression = BI_RGB;
            blit_dc = CreateCompatibleDC(nullptr);
            if (blit_dc != nullptr)
                blit_bitmap = CreateDIBSection(
                    blit_dc, &info, DIB_RGB_COLORS, &blit_bits, nullptr, 0);
            if (blit_dc == nullptr || blit_bitmap == nullptr || blit_bits == nullptr) {
                reset_blit_surface();
                return false;
            }
            blit_previous_bitmap = SelectObject(blit_dc, blit_bitmap);
            if (blit_previous_bitmap == nullptr ||
                blit_previous_bitmap == HGDI_ERROR) {
                blit_previous_bitmap = nullptr;
                reset_blit_surface();
                return false;
            }
            blit_width = output.width;
            blit_height = output.height;
        }
        DIBSECTION destination_section = {};
        HGDIOBJ destination_bitmap = GetCurrentObject(destination, OBJ_BITMAP);
        int descriptor_bytes = GetObjectA(destination_bitmap, sizeof(destination_section), &destination_section);
        int destination_bits = descriptor_bytes >= int(sizeof(BITMAP)) ? destination_section.dsBm.bmBitsPixel : 0;
        bool dib16 = descriptor_bytes == int(sizeof(DIBSECTION)) && destination_bits == 16;
        unsigned red_mask = dib16 ? destination_section.dsBitfields[0] : 0;
        unsigned green_mask = dib16 ? destination_section.dsBitfields[1] : 0;
        unsigned blue_mask = dib16 ? destination_section.dsBitfields[2] : 0;
        if (dib16 && destination_section.dsBmih.biCompression == BI_RGB) {
            red_mask=0x7c00;green_mask=0x3e0;blue_mask=0x1f;
        }
        bool round16 = dib16 && blue_mask==0x1f &&
            ((red_mask==0x7c00 && green_mask==0x3e0) || (red_mask==0xf800 && green_mask==0x7e0));
        if (destination_bits != blit_destination_bits || green_mask != blit_destination_green_mask) {
            char detail[192];
            std::snprintf(detail,sizeof(detail),"bits=%d dib=%u masks=%x,%x,%x ordered_rounding=%u",
                destination_bits,unsigned(dib16),red_mask,green_mask,blue_mask,unsigned(round16));
            trace.write("destination-format",detail,true);
            blit_destination_bits=destination_bits;blit_destination_green_mask=green_mask;
        }
        if (round16 && !blit_rounding_ready) {
            blit_round5=c3x_renderer::color_rounding_table(31);
            blit_round6=c3x_renderer::color_rounding_table(63);
            blit_rounding_ready=true;
        }
        // Bounded live diagnostic: distinguish a black strip already present
        // in renderer output from later native erasure/composition. Sampling
        // every eighth row catches the reported 16/32-pixel bars without a
        // second full-image traversal or recording native/UI pixels.
        if(trace.level){
            int black_x=-1,black_y=-1,black_length=0;
            for(int y=output.clip_top;y<output.clip_bottom;y+=8){
                auto row=reinterpret_cast<std::uint32_t const*>(static_cast<std::uint8_t const*>(output.bgra_pixels)+std::size_t(y)*output.stride_bytes);
                int run=0;
                for(int x=output.clip_left;x<output.clip_right;++x){
                    run=(row[x]&0x00ffffffu)?0:run+1;
                    if(run>=32 && run>black_length){black_x=x-run+1;black_y=y;black_length=run;}
                }
            }
            if(black_x!=last_black_x || black_y!=last_black_y || black_length!=last_black_length){
                char detail[192];std::snprintf(detail,sizeof(detail),"x=%d y=%d length=%d width=%d height=%d source=renderer-map sample_rows=8",
                    black_x,black_y,black_length,output.width,output.height);
                trace.write("map-black-span",detail,true);
                last_black_x=black_x;last_black_y=black_y;last_black_length=black_length;
            }
        }
        auto const& green_rounding=green_mask==0x7e0?blit_round6:blit_round5;
        std::size_t row_bytes = static_cast<std::size_t>(
            output.clip_right - output.clip_left) * sizeof(std::uint32_t);
        for (int y = output.clip_top; y < output.clip_bottom; ++y) {
            std::size_t row = static_cast<std::size_t>(y) * output.stride_bytes;
            std::size_t left = static_cast<std::size_t>(output.clip_left) *
                sizeof(std::uint32_t);
            auto* target=static_cast<std::uint8_t *>(blit_bits)+row+left;
            auto const* source=static_cast<std::uint8_t const *>(output.bgra_pixels)+row+left;
            if (!round16) std::memcpy(target,source,row_bytes);
            else for(int x=output.clip_left;x<output.clip_right;++x,source+=4,target+=4) {
                unsigned threshold=c3x_renderer::color_threshold(unsigned(x-phase_x),unsigned(y-phase_y));
                target[0]=blit_round5[threshold][source[0]];
                target[1]=green_rounding[threshold][source[1]];
                target[2]=blit_round5[threshold][source[2]];
                target[3]=source[3];
            }
        }
        return BitBlt(destination, output.clip_left, output.clip_top,
                      output.clip_right - output.clip_left,
                      output.clip_bottom - output.clip_top,
                      blit_dc, output.clip_left, output.clip_top, SRCCOPY) != FALSE;
    }
};

RendererState renderer;

// A presentation owns all pointer-bearing output fields together. Construction
// is transactional: an allocation failure cannot partly replace the image or
// its ownership. Two publications may coexist briefly during commit, each at
// most 32 MiB including ownership arrays. This is not a viewport cache.
struct PublishedMapFrame {
    struct Resident {
        // Opaque immutable storage keeps publication/content proofs independent
        // of D3D. Only the worker creates or submits the underlying texture.
        std::shared_ptr<void> texture;
        int width=0,height=0;
        std::shared_ptr<void> scene;
    };
    Resident resident;
    int source_x=0,source_y=0;
    bool has_image() const {return output.bgra_pixels || bool(resident.texture);}
    c3x_renderer_output_v1 output={};
    std::vector<std::uint32_t> pixels, fallback, replacements;
    std::vector<c3x_renderer_tile_v1> occurrences;
    c3x_renderer_frame_v1 frame={};
    c3x_renderer_camera_identity_v1 identity={};
    std::uint64_t scene_signature=0;
    int phase_x=0,phase_y=0;
    PublishedMapFrame()=default;
    PublishedMapFrame(PublishedMapFrame const&)=delete;
    PublishedMapFrame& operator=(PublishedMapFrame const&)=delete;
    void swap(PublishedMapFrame& other) noexcept {
        std::swap(output,other.output);std::swap(resident,other.resident);
        std::swap(source_x,other.source_x);std::swap(source_y,other.source_y);
        pixels.swap(other.pixels);fallback.swap(other.fallback);replacements.swap(other.replacements);
        occurrences.swap(other.occurrences);std::swap(frame,other.frame);std::swap(identity,other.identity);
        std::swap(scene_signature,other.scene_signature);
        std::swap(phase_x,other.phase_x);std::swap(phase_y,other.phase_y);
    }
    std::size_t bytes() const {
        return (pixels.capacity()+fallback.capacity()+replacements.capacity())*sizeof(std::uint32_t)+
            occurrences.capacity()*sizeof(c3x_renderer_tile_v1)+sizeof(PublishedMapFrame)+
            (resident.texture?std::size_t(resident.width)*resident.height*4:0);
    }
    bool capture(c3x_renderer_output_v1 const& source,int x,int y,
                 c3x_renderer_frame_v1 const* captured=nullptr,c3x_renderer_camera_identity_v1 const& epochs={},
                 Resident const* storage=nullptr,int offset_x=0,int offset_y=0) {
        constexpr std::uint64_t budget=32u*1024u*1024u;
        if(source.width<=0 || source.height<=0 || source.width>8192 || source.height>8192 ||
           source.stride_bytes!=source.width*4 || (!source.bgra_pixels && (!storage || !storage->texture)) ||
           source.fallback_tile_count>8192 || source.replacement_tile_count>8192 ||
           (source.fallback_tile_count && !source.fallback_tile_indices) ||
           (source.replacement_tile_count && !source.replacement_tile_flags))return false;
        if(storage && (!storage->texture || offset_x<0 || offset_y<0 || storage->width<source.width || storage->height<source.height ||
           offset_x>storage->width-source.width || offset_y>storage->height-source.height))return false;
        auto count=storage?std::uint64_t(storage->width)*storage->height:std::uint64_t(source.width)*source.height;
        if(captured && (captured->tile_count>8192 || (captured->tile_count && !captured->tiles) ||
           source.replacement_tile_count!=captured->tile_count || source.width!=captured->target_width ||
           source.height!=captured->target_height))return false;
        if(captured)for(unsigned i=0;i<source.fallback_tile_count;++i)
            if(source.fallback_tile_indices[i]>=captured->tile_count)return false;
        auto occurrence_bytes=captured?std::uint64_t(captured->tile_count)*sizeof(c3x_renderer_tile_v1):0;
        if((count+source.fallback_tile_count+source.replacement_tile_count)*4+occurrence_bytes+sizeof(PublishedMapFrame)>budget)return false;
        try {
            PublishedMapFrame next;
            auto first=static_cast<std::uint32_t const*>(source.bgra_pixels);
            if(storage){next.resident=*storage;next.source_x=offset_x;next.source_y=offset_y;}
            else next.pixels.assign(first,first+std::size_t(count));
            if(source.fallback_tile_count)next.fallback.assign(source.fallback_tile_indices,
                source.fallback_tile_indices+source.fallback_tile_count);
            if(source.replacement_tile_count)next.replacements.assign(source.replacement_tile_flags,
                source.replacement_tile_flags+source.replacement_tile_count);
            if(captured){
                if(captured->tile_count)next.occurrences.assign(captured->tiles,captured->tiles+captured->tile_count);
                next.frame=*captured;
                next.frame.tiles=next.occurrences.empty()?nullptr:next.occurrences.data();
                next.frame.world_topology=nullptr;next.frame.world_topology_count=0;
            }
            // Account allocated capacity, not just requested payload bytes.
            if(next.bytes()>budget)return false;
            next.output=source;next.phase_x=x;next.phase_y=y;next.identity=epochs;
            next.output.bgra_pixels=storage?nullptr:next.pixels.data();
            next.output.fallback_tile_indices=next.fallback.empty()?nullptr:next.fallback.data();
            next.output.replacement_tile_flags=next.replacements.empty()?nullptr:next.replacements.data();
            swap(next);
            return true;
        } catch (...) {return false;}
    }
    bool capture_crop(PublishedMapFrame const& donor,c3x_renderer_output_v1 source,int x,int y,
                      c3x_renderer_frame_v1 const& captured,c3x_renderer_camera_identity_v1 const& epochs) {
        if(x<0 || y<0 || x>donor.output.width-source.width || y>donor.output.height-source.height)return false;
        int anchor_x=captured.tile_count?captured.tiles[0].anchor_x:0,anchor_y=captured.tile_count?captured.tiles[0].anchor_y:0;
        if(donor.resident.texture)return capture(source,anchor_x,anchor_y,&captured,epochs,&donor.resident,donor.source_x+x,donor.source_y+y);
        if(!donor.output.bgra_pixels)return false;
        std::vector<std::uint32_t> cropped(std::size_t(source.width)*source.height);
        for(int row=0;row<source.height;++row)std::copy_n(reinterpret_cast<std::uint32_t const*>(
            static_cast<unsigned char const*>(donor.output.bgra_pixels)+std::size_t(row+y)*donor.output.stride_bytes)+x,
            source.width,cropped.data()+std::size_t(row)*source.width);
        source.bgra_pixels=cropped.data();return capture(source,anchor_x,anchor_y,&captured,epochs);
    }
    bool matches_static_view(c3x_renderer_frame_v1 const& current) const {
        if(!frame.api_version || current.tile_count!=occurrences.size())return false;
        auto left=current,right=frame;
        left.tiles=right.tiles=nullptr;left.world_topology=right.world_topology=nullptr;
        left.world_topology_count=right.world_topology_count=0;
        left.presentation_time_ticks=right.presentation_time_ticks=0;
        left.clip_left=right.clip_left=left.clip_top=right.clip_top=0;
        left.clip_right=right.clip_right=left.clip_bottom=right.clip_bottom=0;
        left.visible_animation_count=right.visible_animation_count=0;
        // A publication retains the topology revision, not its payload. The
        // worker separately checks that payload against its immutable request.
        return !std::memcmp(&left,&right,sizeof(left)) &&
            (!current.tile_count || (current.tiles && !std::memcmp(current.tiles,occurrences.data(),
                std::size_t(current.tile_count)*sizeof(*current.tiles))));
    }
    void clear() {
        std::vector<std::uint32_t>().swap(pixels);
        std::vector<std::uint32_t>().swap(fallback);
        std::vector<std::uint32_t>().swap(replacements);
        std::vector<c3x_renderer_tile_v1>().swap(occurrences);frame={};identity={};
        resident={};source_x=source_y=0;output={};scene_signature=0;phase_x=phase_y=0;
    }
};

// Cheap, deliberately provisional terrain-only view. It consumes the same
// generic normalized base-color DDS inputs and captured screen anchors, never
// prior-camera pixels or uncaptured world cells. No detailed cache is modified.
struct CameraTerrainPreview {
    template<class Textures>
    bool render(c3x_renderer_frame_v1 const& frame,Textures const& textures,
                PublishedMapFrame& target,std::atomic<bool> const& cancelled,
                c3x_renderer_camera_identity_v1 const& identity={}) {
        constexpr unsigned edge=32;
        std::array<std::array<std::uint32_t,edge*edge>,14> colors{};
        std::array<bool,14> ready{};
        auto material=[&](c3x_renderer_tile_v1 const& tile) {
            int type=tile.real_terrain_type;
            if(type!=4 && type!=5 && type!=6 && type!=10 && type<11)type=tile.terrain_type;
            return type;
        };
        auto read32=[](std::vector<std::uint8_t> const& bytes,std::size_t n) {
            return std::uint32_t(bytes[n])|(std::uint32_t(bytes[n+1])<<8)|
                (std::uint32_t(bytes[n+2])<<16)|(std::uint32_t(bytes[n+3])<<24);
        };
        for(unsigned i=0;i<frame.tile_count;++i) {
            if(cancelled.load(std::memory_order_relaxed))return false;
            auto const& tile=frame.tiles[i];
            if(!(tile.tile_flags&C3X_RENDERER_TILE_RENDER))continue;
            int type=material(tile);
            if(type<0 || type>=14 || !textures[type].configured)return false;
            if(ready[type])continue;
            auto const& dds=textures[type].dds;
            if(dds.size()<164 || std::memcmp(dds.data(),"DDS ",4) ||
               read32(dds,4)!=124 || std::memcmp(dds.data()+84,"DX10",4))return false;
            unsigned format=read32(dds,128),w=read32(dds,16),h=read32(dds,12);
            if((format!=77 && format!=78) || !w || !h || w>16384 || h>16384)return false;
            auto bytes=std::uint64_t((w+3)/4)*((h+3)/4)*16;
            if(bytes>dds.size()-148)return false;
            // BC3's color block is opaque BC1 interpolation. Terrain preview
            // ignores alpha, but never accepts another compression format.
            for(unsigned y=0;y<edge;++y)for(unsigned x=0;x<edge;++x) {
                unsigned sx=std::min(w-1,(2*x+1)*w/(2*edge));
                unsigned sy=std::min(h-1,(2*y+1)*h/(2*edge));
                auto at=148+std::size_t((sy/4)*((w+3)/4)+sx/4)*16+8;
                unsigned a=unsigned(dds[at])|(unsigned(dds[at+1])<<8);
                unsigned b=unsigned(dds[at+2])|(unsigned(dds[at+3])<<8);
                unsigned which=(read32(dds,at+4)>>(2*((sy%4)*4+sx%4)))&3;
                unsigned value=0xff000000u;
                for(unsigned channel=0;channel<3;++channel) {
                    unsigned shift=channel==0?11:channel==1?5:0,mask=channel==1?63:31;
                    unsigned ca=((a>>shift)&mask)*255/mask,cb=((b>>shift)&mask)*255/mask;
                    unsigned c=which==0?ca:which==1?cb:which==2?(2*ca+cb)/3:(ca+2*cb)/3;
                    value|=c<<(16-channel*8);
                }
                colors[type][y*edge+x]=value;
            }
            ready[type]=true;
        }
        auto count=std::uint64_t(frame.target_width)*frame.target_height;
        if((count+frame.tile_count)*4+std::uint64_t(frame.tile_count)*sizeof(c3x_renderer_tile_v1)+
           sizeof(PublishedMapFrame)>32u*1024u*1024u)return false;
        target.clear();target.pixels.assign(std::size_t(count),0xff000000u);
        target.replacements.assign(frame.tile_count,0);
        if(frame.tile_count)target.occurrences.assign(frame.tiles,frame.tiles+frame.tile_count);
        target.frame=frame;target.identity=identity;
        target.frame.tiles=target.occurrences.empty()?nullptr:target.occurrences.data();
        target.frame.world_topology=nullptr;target.frame.world_topology_count=0;
        if(target.bytes()>32u*1024u*1024u)return false;
        auto environment=c3x_renderer::evaluate_environment(float(frame.hour),frame.season);
        // A base water texture is the bed, not the water surface. Use a coarse
        // optical-depth estimate per captured water family with the existing
        // absorption/tint response from render_core's scene_material_v1.hlsl.
        // This deliberately omits coast geometry, normals and reflection detail;
        // it is not source-engine evidence or a final-shader parity claim.
        for(unsigned type=0;type<14;++type)if(ready[type]) {
            float light[3]={};c3x_renderer::shade_terrain(environment,int(type),0,0,1.f,light);
            float depth=type==11?.15f:type==12?.35f:.65f;
            float mix=std::clamp((depth-.18f)/(.43f-.18f),0.f,1.f);mix=mix*mix*(3-2*mix);
            float absorption[]={14,7,3},shallow[]={.023f,.074f,.096f},deep[]={.003f,.015f,.040f};
            for(auto& value:colors[type]) {
                unsigned out=0xff000000u;
                for(unsigned c=0;c<3;++c) {
                    float channel=float((value>>(16-c*8))&255)/255.f;
                    if(type>=11) {
                        float linear=channel<=.04045f?channel/12.92f:std::pow((channel+.055f)/1.055f,2.4f);
                        float transmission=std::exp(-depth*3.2f);
                        linear=(linear*std::exp(-depth*absorption[c])*transmission+
                            (shallow[c]+(deep[c]-shallow[c])*mix)*(1-transmission))*light[c];
                        channel=linear<=.0031308f?linear*12.92f:1.055f*std::pow(linear,1/2.4f)-.055f;
                    }else channel*=light[c];
                    out|=unsigned(std::clamp(channel*255.f,0.f,255.f))<<(16-c*8);
                }
                value=out;
            }
        }
        float half_w=frame.tile_width*.5f,half_h=frame.tile_height*.5f;
        unsigned rendered=0;
        for(unsigned i=0;i<frame.tile_count;++i) {
            if(cancelled.load(std::memory_order_relaxed))return false;
            auto const& tile=frame.tiles[i];
            if(!(tile.tile_flags&C3X_RENDERER_TILE_RENDER))continue;
            int type=material(tile);
            auto const& lit=colors[type];
            // Pixel-center diamond coverage: u grows down/right, v down/left.
            // Use 64-bit extents so even off-screen sentinel anchors cannot wrap.
            int left=int(std::clamp<std::int64_t>(tile.anchor_x,0,frame.target_width));
            int right=int(std::clamp<std::int64_t>(std::int64_t(tile.anchor_x)+frame.tile_width,0,frame.target_width));
            int top=int(std::clamp<std::int64_t>(tile.anchor_y,0,frame.target_height));
            int bottom=int(std::clamp<std::int64_t>(std::int64_t(tile.anchor_y)+frame.tile_height,0,frame.target_height));
            for(int y=top;y<bottom;++y) {
                if(cancelled.load(std::memory_order_relaxed))return false;
                float dy=(float(y)+.5f-float(tile.anchor_y))/half_h;
                for(int x=left;x<right;++x) {
                    float dx=(float(x)+.5f-float(tile.anchor_x)-half_w)/half_w;
                    float u=(dx+dy)*.5f,v=(dy-dx)*.5f;
                    if(u<0 || v<0 || u>=1 || v>=1)continue;
                    target.pixels[std::size_t(y)*frame.target_width+x]=lit[unsigned(v*edge)*edge+unsigned(u*edge)];
                }
            }
            target.replacements[i]=C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED;++rendered;
        }
        target.output={C3X_RENDERER_API_VERSION,sizeof(c3x_renderer_output_v1)};
        auto& out=target.output;out.width=frame.target_width;out.height=frame.target_height;out.stride_bytes=out.width*4;
        out.clip_left=frame.clip_left;out.clip_top=frame.clip_top;out.clip_right=frame.clip_right;out.clip_bottom=frame.clip_bottom;
        out.bgra_pixels=target.pixels.data();out.replacement_tile_flags=target.replacements.data();out.replacement_tile_count=frame.tile_count;
        out.rendered_tile_count=out.textured_tile_count=rendered;
        target.phase_x=frame.tile_count?frame.tiles[0].anchor_x:0;
        target.phase_y=frame.tile_count?frame.tiles[0].anchor_y:0;
        return true;
    }
};

// Civ III remains the caller and presenter. One renderer worker owns all
// renderer-state mutation and D3D work, consuming a deep copy of each captured
// frame. The default synchronous ABI permits no stale-frame fallback. Its
// opt-in ambient compatibility mode may retain only an exactly matched static
// camera/scene front while a newer clock tick is in flight; camera or captured
// ownership changes still take over synchronously. No backlog is accumulated.
void CALLBACK renderer_visual_timer(HWND,UINT,UINT_PTR,DWORD);

class RendererWorker {
public:
    explicit RendererWorker(RendererState & state) : renderer_state(state) {
        char option[8]={};
        isolated_publication=GetEnvironmentVariableA("C3X_RENDERER_CAMERA_PUBLICATION",option,sizeof(option)) &&
            std::strcmp(option,"1")==0;
        camera_preview_enabled=GetEnvironmentVariableA("C3X_RENDERER_CAMERA_PREVIEW",option,sizeof(option)) &&
            std::strcmp(option,"1")==0;
        ambient_async_enabled=GetEnvironmentVariableA("C3X_RENDERER_SYNC_AMBIENT",option,sizeof(option)) &&
            std::strcmp(option,"1")==0;
        // The synchronous compatibility path may return its front while the
        // worker mutates render scratch, so its publication must own pixels.
        if(ambient_async_enabled)isolated_publication=true;
        ahead_enabled=!GetEnvironmentVariableA("C3X_RENDERER_PREPARE_AHEAD",option,sizeof(option)) ||
            std::strcmp(option,"1")==0;
        unit_pixels_enabled=!GetEnvironmentVariableA("C3X_RENDERER_UNIT_PIXELS",option,sizeof(option)) ||
            std::strcmp(option,"0")!=0;
        renderer_state.unit_bodies.set_pose_ready_notification([this]{
            std::lock_guard<std::mutex> lock(state_mutex);
            ++unit_content_revision;wake.notify_one();
        });
        // Work ahead isolates its front only after a stable eligible request.
        // Moving-camera synchronous results retain their existing lifetime.
    }

    int set_unit_rendering(int enabled) {
        if(enabled!=0 && enabled!=1)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        std::lock_guard<std::mutex> call_guard(call_mutex);
        std::unique_lock<std::mutex> lock(state_mutex);
        drain_camera_locked(lock);
        // Call serialization excludes foreground configure/render jobs. Idle
        // terrain preparation never reads this unit-only configuration value.
        renderer_state.unit_rendering_enabled=enabled!=0;
        if(!enabled){unit_pixels_queue.clear();unit_instances.clear();renderer_state.unit_bodies.release_pose_leases();}
        renderer_state.trace.write("unit-config",enabled?"enabled; bind at definition load":"disabled; native units",true);
        return C3X_RENDERER_RESULT_OK;
    }

    ~RendererWorker() {
        reset_and_stop();
        // No queue lock: a finishing helper may still be delivering readiness.
        renderer_state.unit_bodies.set_pose_ready_notification({});
    }

    int configure_pack(char const * path) {
        std::lock_guard<std::mutex> call_guard(call_mutex);
        std::unique_lock<std::mutex> lock(state_mutex);
        start_locked();
        drain_camera_locked(lock);
        job_pack_present = path != nullptr;
        job_pack_path = path != nullptr ? path : "";
        return submit_locked(lock, Command::configure_pack);
    }

    int configure_definitions(char const * mod_root, char const * default_path,
                              char const * scenario_path, char const * custom_path) {
        std::lock_guard<std::mutex> call_guard(call_mutex);
        std::unique_lock<std::mutex> lock(state_mutex);
        start_locked();
        drain_camera_locked(lock);
        job_mod_root_present = mod_root != nullptr;
        job_default_path_present = default_path != nullptr;
        job_scenario_path_present = scenario_path != nullptr;
        job_custom_path_present = custom_path != nullptr;
        job_mod_root = mod_root != nullptr ? mod_root : "";
        job_default_path = default_path != nullptr ? default_path : "";
        job_scenario_path = scenario_path != nullptr ? scenario_path : "";
        job_custom_path = custom_path != nullptr ? custom_path : "";
        return submit_locked(lock, Command::configure_definitions);
    }

    int render_gpu(c3x_renderer_camera_request_v1 const& request,c3x_renderer_gpu_frame_v1& view,c3x_renderer_output_v1& metadata){
        LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
        std::lock_guard<std::mutex> calls(call_mutex);std::unique_lock<std::mutex> lock(state_mutex);
        start_locked();drain_camera_locked(lock);advance_visual_clock();foreground_pending.store(true);
        if(!gpu_presentation){nearby.clear();retained_views.clear();gpu_publication.clear();}
        gpu_presentation=true;native_presentation=true;isolated_publication=true;camera_preview_enabled=false;
        auto prior=*request.frame;prior.dirty_flags=gpu_publication.frame.dirty_flags;
        prospective_view_allowed=!gpu_publication.has_image() || gpu_publication.matches_static_view(prior);
        gpu_reused=false;nearby_presented=false;
        if(nearby_available && nearby.map.resident.texture){
            for(auto& area:retained_views)if(area->input.tile_width==request.frame->tile_width &&
                area->project(*request.frame,request.identity,gpu_publication)){
                nearby.swap(*area);gpu_reused=true;break;
            }
            auto const& center=nearby.center;
            auto current=*request.frame;current.dirty_flags=center.frame.dirty_flags;
            // Project performs complete fresh content/topology validation even
            // when a newer selected sample already matches the exact camera.
            PublishedMapFrame selected;
            if(nearby.project(*request.frame,request.identity,selected)){
                if(center.resident.texture && center.frame.presentation_time_ticks<=current.presentation_time_ticks &&
                   center.frame.presentation_time_ticks>=selected.frame.presentation_time_ticks && center.matches_static_view(current))
                    selected.capture(center.output,center.phase_x,center.phase_y,&center.frame,center.identity,&center.resident,center.source_x,center.source_y);
                gpu_publication.swap(selected);gpu_reused=true;
            }
        }
        nearby_presented=gpu_reused;
        if(!gpu_reused)gpu_publication.clear();
        job_frame=*request.frame;job_camera_identity=request.identity;
        job_tiles.clear();job_world_topology.clear();
        try {
            if(job_frame.tile_count)job_tiles.assign(job_frame.tiles,job_frame.tiles+job_frame.tile_count);
            if(job_frame.world_topology_count)job_world_topology.assign(job_frame.world_topology,job_frame.world_topology+job_frame.world_topology_count);
        }catch(...){foreground_pending.store(false);wake.notify_one();return C3X_RENDERER_RESULT_ERROR;}
        job_frame.tiles=job_tiles.data();job_frame.world_topology=job_world_topology.data();
        int result=submit_locked(lock,Command::gpu_render);
        view=gpu_view;metadata=gpu_metadata;QueryPerformanceCounter(&end);
        char detail[224];std::snprintf(detail,sizeof(detail),"result=%d prepared=%u request_ms=%.3f sample_ticks=%lld requested_ticks=%lld",
            result,view.prepared,renderer_state.trace.milliseconds(end.QuadPart-begin.QuadPart),view.presentation_time_ticks,request.frame->presentation_time_ticks);
        renderer_state.trace.write("gpu-map-request",detail,true);return result;
    }
    int images_gpu(c3x_renderer_gpu_images_v1 const& request,c3x_renderer_gpu_result_v1& result,unsigned* readback,unsigned capacity){
        std::lock_guard<std::mutex> calls(call_mutex);std::unique_lock<std::mutex> lock(state_mutex);
        start_locked();drain_camera_locked(lock,gpu_presentation);
        gpu_request=request;gpu_pixels.clear();gpu_commands.clear();
        if(request.action==C3X_GPU_UPLOAD)gpu_pixels.assign(request.pixels,request.pixels+request.pixel_count);
        for(unsigned n=0;n<request.command_count;++n){auto const& c=request.commands[n];
            gpu_commands.push_back({c3x_gpu_images::Kind(c.kind),std::uint64_t(c.destination),std::uint64_t(c.source),
                {c.area[0],c.area[1],c.area[2],c.area[3]},{c.clip[0],c.clip[1],c.clip[2],c.clip[3]},c.source_x,c.source_y,c.color,std::uint64_t(c.background),std::uint64_t(c.detail),std::uint64_t(c.background_detail),c.source_width,c.source_height,std::uint64_t(c.program)});}
        gpu_request.pixels=nullptr;gpu_request.commands=nullptr; // worker receives only owned values
        int code=submit_locked(lock,Command::gpu_images);result=gpu_result;
        if(code==C3X_RENDERER_RESULT_OK && request.action==C3X_GPU_READBACK){
            if(gpu_readback.size()>capacity)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            std::copy(gpu_readback.begin(),gpu_readback.end(),readback);
        }
        return code;
    }

    void stop_visual_timer(){if(visual_timer){KillTimer(nullptr,visual_timer);visual_timer=0;}}
    void advance_visual_clock(){
        LARGE_INTEGER now={},frequency={};QueryPerformanceCounter(&now);QueryPerformanceFrequency(&frequency);
        if(visual_last && visual_allowed && now.QuadPart>=visual_last)visual_ticks+=now.QuadPart-visual_last;
        visual_last=now.QuadPart;visual_frequency=frequency.QuadPart;
    }
    long long visual_clock(){std::lock_guard<std::mutex> calls(call_mutex);advance_visual_clock();return visual_ticks;}
    int visual_policy(unsigned policy){
        std::lock_guard<std::mutex> calls(call_mutex);
        if(policy<2){advance_visual_clock();visual_allowed=policy!=0;}
        auto* session=renderer_state.gpu_composition.get();return session&&session->visual_ready()?1:0;
    }
    int visual_status(c3x_renderer_visual_status_v1& out){
        std::lock_guard<std::mutex> calls(call_mutex);auto* session=renderer_state.gpu_composition.get();
        out={sizeof(out),static_cast<long long>(visual_frames),static_cast<long long>(visual_map_samples),
            static_cast<long long>(visual_unit_samples),static_cast<long long>(visual_pose_changes),
            session?static_cast<long long>(session->visual_bytes()):0,session?static_cast<long long>(session->visual_nodes()):0,visual_ticks,visual_frequency};
        return C3X_RENDERER_RESULT_OK;
    }
    int visual_frame(bool timer=false){
        LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
        // A timer is only transport on the presenter's UI thread. Nested UI
        // dispatch cannot reenter a native transaction or wait on its own gate.
        std::unique_lock<std::mutex> calls(call_mutex,std::try_to_lock);
        if(!calls.owns_lock()||!running||!gpu_presenter.caller_thread())return C3X_RENDERER_RESULT_PENDING;
        if(!visual_allowed || (timer&&(!IsWindowVisible(gpu_present.window?static_cast<HWND>(gpu_present.window):nullptr)||
            GetForegroundWindow()!=GetAncestor(static_cast<HWND>(gpu_present.window),GA_ROOT)))){
            LARGE_INTEGER now={};QueryPerformanceCounter(&now);visual_last=now.QuadPart;return C3X_RENDERER_RESULT_PENDING;
        }
        std::unique_lock<std::mutex> lock(state_mutex);ForegroundCameraPause pause(*this,lock);
        auto* session=renderer_state.gpu_composition.get();
        if(!session||!session->visual_active()||!gpu_presenter.view())return C3X_RENDERER_RESULT_PENDING;
        advance_visual_clock();
        int result=submit_locked(lock,Command::visual_frame);
        if(result==C3X_RENDERER_RESULT_OK){result=gpu_presenter.present();++visual_frames;}
        QueryPerformanceCounter(&end);
        char line[256];std::snprintf(line,sizeof(line),"result=%d frames=%llu request_ms=%.3f retained_bytes=%llu nodes=%zu native_map_calls=0 native_unit_calls=0",
            result,static_cast<unsigned long long>(visual_frames),renderer_state.trace.milliseconds(end.QuadPart-begin.QuadPart),
            static_cast<unsigned long long>(session->visual_bytes()),session->visual_nodes());
        renderer_state.trace.write("visual-frame",line,result==C3X_RENDERER_RESULT_ERROR||visual_frames<=3||visual_frames%128==0);
        return result;
    }
    int present_gpu(c3x_renderer_gpu_present_v1 const& request){
        std::lock_guard<std::mutex> calls(call_mutex);std::unique_lock<std::mutex> lock(state_mutex);
        start_locked();drain_camera_locked(lock,gpu_presentation);
        if(!gpu_presenter.caller_thread())return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(request.action==1){stop_visual_timer();if(renderer_state.gpu_composition)renderer_state.gpu_composition->stop_visuals();gpu_presenter.reset();return C3X_RENDERER_RESULT_OK;}
        if(request.action==2){
            stop_visual_timer();if(renderer_state.gpu_composition)renderer_state.gpu_composition->stop_visuals();
            gpu_present=request;int result=submit_locked(lock,Command::gpu_present);
            if(result==C3X_RENDERER_RESULT_OK)gpu_presenter.release_native();
            return result;
        }
        auto* session=renderer_state.gpu_composition.get();
        if(!session||session->current_ticket()!=request.ticket)return C3X_RENDERER_RESULT_SUPERSEDED;
        bool full=request.area[0]<=0&&request.area[1]<=0&&request.area[2]>=request.width&&request.area[3]>=request.height;
        try{
            if(!gpu_presenter.prepare(static_cast<HWND>(request.window),renderer_state.device,request.width,request.height,full))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            gpu_present=request;
            int result=submit_locked(lock,Command::gpu_present);
            if(result==C3X_RENDERER_RESULT_OK)result=gpu_presenter.present();
            if(result!=C3X_RENDERER_RESULT_OK){stop_visual_timer();gpu_presenter.reset();}
            else if(session->visual_ready()&&!visual_timer){advance_visual_clock();visual_timer=SetTimer(nullptr,0,33,renderer_visual_timer);
                if(!visual_timer){session->stop_visuals();renderer_state.trace.write("visual-timer", "creation failed; native compatibility demand retained",true);}}
            return result;
        }catch(...){gpu_presenter.reset();return C3X_RENDERER_RESULT_ERROR;}
    }

    // Native final transfer is independent of map publication. Existing CPU
    // surfaces stay authoritative until their entire access lifetime is covered.
    int native_screen(c3x_native_images::ScreenSnapshot* screen) {
        std::lock_guard<std::mutex> calls(call_mutex);stop_visual_timer();
        if(renderer_state.gpu_composition)renderer_state.gpu_composition->stop_visuals();std::unique_lock<std::mutex> lock(state_mutex);
        if(!gpu_presenter.caller_thread())return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(!screen){
            if(renderer_state.device){
                ForegroundCameraPause pause(*this,lock);
                gpu_present={sizeof(gpu_present)};gpu_present.action=2;
                int result=submit_locked(lock,Command::gpu_present);
                if(result!=C3X_RENDERER_RESULT_OK)return result;
            }
            gpu_presenter.release_native();native_screen_active=false;return C3X_RENDERER_RESULT_OK;
        }
        start_locked();
        // Final UI transfer has foreground ownership just like a demanded
        // unit. Interruptible preparation resumes with its immutable camera
        // request intact; GPU queue activity never selects a different presenter.
        ForegroundCameraPause pause(*this,lock);
        // Fence any bounded tile/helper work before reading the device on the
        // window thread. The queue remains paused through DXGI presentation.
        screen_upload=false;
        if(submit_locked(lock,Command::native_screen)!=C3X_RENDERER_RESULT_OK || !renderer_state.device)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        try {
            bool full=screen->area.left==0&&screen->area.top==0&&screen->area.right==screen->width&&screen->area.bottom==screen->height;
            if(!gpu_presenter.prepare(screen->window,renderer_state.device,screen->width,screen->height,full)){
                gpu_presenter.release_native();native_screen_active=false;return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            }
            screen_pixels=std::move(screen->pixels);screen_upload=true;screen_width=screen->width;screen_height=screen->height;screen_area=screen->area;screen_format=screen->native_format;
            native_screen_active=true;
            int result=submit_locked(lock,Command::native_screen);
            if(result==C3X_RENDERER_RESULT_OK)result=gpu_presenter.present();
            if(result!=C3X_RENDERER_RESULT_OK){gpu_presenter.release_native();native_screen_active=false;}
            return result;
        }catch(...){gpu_presenter.release_native();native_screen_active=false;return C3X_RENDERER_RESULT_ERROR;}
    }

    int render(c3x_renderer_frame_v1 const & frame, c3x_renderer_output_v1 & output,
               c3x_renderer_camera_identity_v1 const& identity={}) {
        // Diagnostic-only disjoint caller/worker endpoints. The destructor runs
        // after both call locks release, and buffered traces flush at DLL teardown.
        struct CallTiming {
            RendererTrace& trace;
            LARGE_INTEGER entered={},locked={},submitted={},worker_begin={},worker_rendered={},worker_published={};
            bool queued=false;
            CallTiming(RendererTrace& t):trace(t){if(trace.buffered)QueryPerformanceCounter(&entered);}
            ~CallTiming(){if(trace.buffered){
                LARGE_INTEGER returned={};QueryPerformanceCounter(&returned);
                char detail[512];std::snprintf(detail,sizeof(detail),
                    "entered=%lld locked=%lld submitted=%lld worker_begin=%lld worker_rendered=%lld worker_published=%lld returned=%lld queued=%u",
                    entered.QuadPart,locked.QuadPart,submitted.QuadPart,worker_begin.QuadPart,
                    worker_rendered.QuadPart,worker_published.QuadPart,returned.QuadPart,unsigned(queued));
                trace.write("call-endpoints",detail,true);
            }}
        } timing(renderer_state.trace);
        std::lock_guard<std::mutex> call_guard(call_mutex);
        std::unique_lock<std::mutex> lock(state_mutex);
        if(renderer_state.trace.buffered)QueryPerformanceCounter(&timing.locked);
        start_locked();
        if(gpu_presentation){
            drain_camera_locked(lock);gpu_presentation=false;gpu_publication.clear();
            nearby.clear();retained_views.clear();nearby_available=false;
        }
        if(consume_ahead_locked(lock,frame,identity,output))return C3X_RENDERER_RESULT_OK;
        auto const& queued_identity=camera_pending?camera_pending_identity:job_camera_identity;
        bool compatible_camera=camera_result==C3X_RENDERER_RESULT_SUPERSEDED ||
            !std::memcmp(&queued_identity,&identity,sizeof(identity));
        if(ambient_async_enabled && completed_result==C3X_RENDERER_RESULT_OK &&
           completed_resources && publication.output.bgra_pixels && compatible_camera &&
           !std::memcmp(&publication.identity,&identity,sizeof(identity))) {
            // Commit a completed ambient result only when the current capture
            // still describes its exact static view. The active job snapshot
            // retains topology bytes that are intentionally absent from the
            // public display metadata.
            if(camera_ready.output.bgra_pixels && same_ambient_view(frame,job_frame)) {
                c3x_renderer_output_v1 promoted={};
                if(camera_poll_locked(camera_ticket,promoted)==C3X_RENDERER_RESULT_OK) {
                    completed_scene_signature=publication.scene_signature;
                    completed_resources=renderer_state.ambient_count();
                    completed_resource_clock=RendererState::resource_clock(publication.frame);
                }
            }
            if(publication.matches_static_view(frame) && same_ambient_view(frame,job_frame)) {
                auto clock=RendererState::resource_clock(frame);
                if(clock!=completed_resource_clock && !camera_active && !camera_pending &&
                   (camera_result==C3X_RENDERER_RESULT_OK || camera_result==C3X_RENDERER_RESULT_SUPERSEDED)) {
                    c3x_renderer_i64 ignored=0;
                    if(enqueue_camera_locked(frame,identity,ignored)!=C3X_RENDERER_RESULT_PENDING) {
                        drain_camera_locked(lock);
                    }
                }
                if(clock==completed_resource_clock || camera_active || camera_pending ||
                   camera_result==C3X_RENDERER_RESULT_PENDING) {
                    return_current_bitmap(frame,output,"ambient-front");
                    return C3X_RENDERER_RESULT_OK;
                }
            }
        }
        // Native demand may arrive after this exact view was queued through
        // the camera interface. Join its existing owner rather than canceling
        // useful rendering and copying/submitting the same request again.
        // Legacy callers supply zero epochs; explicit native demand supplies its own.
        bool adopt=camera_ticket>0 && !camera_paused &&
            (camera_result==C3X_RENDERER_RESULT_PENDING || camera_result==C3X_RENDERER_RESULT_OK) &&
            (camera_pending?same_camera_request(frame,identity,camera_pending_frame,camera_pending_identity):
                (job_camera_ticket==camera_ticket && !camera_cancelled.load(std::memory_order_relaxed) &&
                 same_camera_request(frame,identity,job_frame,job_camera_identity)));
        if(adopt) {
            auto ticket=camera_ticket;
            bool pending=camera_active || camera_pending;
            bool already_front=camera_front_ticket==ticket && camera_front_result==C3X_RENDERER_RESULT_OK;
            LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
            completed.wait(lock,[this,ticket]{return camera_ticket!=ticket ||
                (!camera_active && !camera_pending && camera_result!=C3X_RENDERER_RESULT_PENDING);});
            int result=camera_poll_locked(ticket,output);
            if(result==C3X_RENDERER_RESULT_OK) {
                completed_scene_signature=publication.scene_signature;
                completed_resources=renderer_state.ambient_count();
                completed_resource_clock=RendererState::resource_clock(publication.frame);
                if(already_front)return_current_bitmap(frame,output,"adopted-front");
            }
            QueryPerformanceCounter(&end);
            char detail[192];sprintf_s(detail,"ticket=%lld result=%d pending=%u wait_ms=%.3f",
                ticket,result,unsigned(pending),renderer_state.trace.milliseconds(end.QuadPart-begin.QuadPart));
            renderer_state.trace.write("worker-adopted-camera",detail,result!=C3X_RENDERER_RESULT_OK);
            return result;
        }
        drain_camera_locked(lock);
        // Idle geometry/pixel preparation never mutates the published frame
        // bitmap or ownership arrays. An identical authoritative appearance
        // can return that immutable publication without cancelling useful work.
        if(completed_scene_signature && completed_result==C3X_RENDERER_RESULT_OK &&
           !std::memcmp(&completed_identity,&identity,sizeof(identity)) &&
           (!completed_resources || completed_resource_clock==RendererState::resource_clock(frame)) &&
           c3x_renderer::terrain_frame_signature(frame,completed_output.content_revision,
                completed_output.device_generation).complete==completed_scene_signature) {
            return_current_bitmap(frame,output,"exact-cache");
            return C3X_RENDERER_RESULT_OK;
        }
        foreground_pending.store(true, std::memory_order_relaxed);
        job_stable_view=completed_result==C3X_RENDERER_RESULT_OK &&
            !std::memcmp(&completed_identity,&identity,sizeof(identity)) &&
            job_frame.api_version && same_ambient_view(frame,job_frame);
        job_frame = frame;job_camera_identity=identity;
        // Reuse bounded snapshot capacity; no allocation/copy on a second vector.
        job_tiles.clear();
        try {
            if (frame.tile_count != 0) job_tiles.assign(frame.tiles, frame.tiles + frame.tile_count);
            job_world_topology.clear();
            if (frame.world_topology_count != 0)
                job_world_topology.assign(frame.world_topology, frame.world_topology + frame.world_topology_count);
        } catch (...) {
            foreground_pending.store(false, std::memory_order_relaxed);
            return C3X_RENDERER_RESULT_ERROR;
        }
        job_frame.tiles = job_tiles.empty() ? nullptr : job_tiles.data();
        job_frame.world_topology = job_world_topology.empty() ? nullptr : job_world_topology.data();
        LARGE_INTEGER submitted = {}, returned = {};
        QueryPerformanceCounter(&submitted);
        if(renderer_state.trace.buffered){timing.submitted=submitted;timing.queued=true;}
        int result = submit_locked(lock, Command::render);
        if(renderer_state.trace.buffered){timing.worker_begin=job_timing_begin;
            timing.worker_rendered=job_timing_rendered;timing.worker_published=job_timing_published;}
        QueryPerformanceCounter(&returned);
        if (renderer_state.trace.level) {
            char detail[480];
            std::snprintf(detail, sizeof(detail),
                "result=%d wait_ms=%.3f tiles=%u prefetch_pending=%u prefetch_built=%u prefetch_unavailable=%u prefetch_cancelled=%u prefetch_bytes=%u cumulative_prewarm_ms=%.3f blocks_pending=%u blocks_built=%u block_bytes=%u",
                result, renderer_state.trace.milliseconds(returned.QuadPart - submitted.QuadPart), frame.tile_count,
                completed_output.prefetch_tiles_pending, completed_output.prefetch_tiles_built,
                completed_output.prefetch_tiles_unavailable, completed_output.prefetch_tiles_cancelled,
                completed_output.prefetch_cache_bytes, renderer_state.trace.milliseconds(completed_output.prefetch_ticks),
                completed_output.prefetch_blocks_pending, completed_output.prefetch_blocks_built, completed_output.pixel_block_cache_bytes);
            renderer_state.trace.write("worker-complete", detail, result != C3X_RENDERER_RESULT_OK);
        }
        output = completed_output;
        return result;
    }

    // Optional Lab-facing camera API. One active immutable snapshot and one
    // replaceable pending snapshot; no backlog and no old-camera success.
    int camera_begin(c3x_renderer_frame_v1 const& frame,c3x_renderer_i64& ticket,
                     c3x_renderer_camera_identity_v1 const& identity={}) {
        std::lock_guard<std::mutex> call_guard(call_mutex);
        std::unique_lock<std::mutex> lock(state_mutex);
        // Later caller demand may submit the same immutable request again. Keep
        // its ticket and ready publication instead of restarting useful work.
        // Compare complete inputs, including time, visibility, order and epochs;
        // no scene hash or old-camera image is sufficient for this decision.
        if(camera_ticket>0 && (camera_result==C3X_RENDERER_RESULT_PENDING || camera_result==C3X_RENDERER_RESULT_OK)) {
            bool same=camera_pending?same_camera_request(frame,identity,camera_pending_frame,camera_pending_identity):
                (job_camera_ticket==camera_ticket && !camera_cancelled.load(std::memory_order_relaxed) &&
                 same_camera_request(frame,identity,job_frame,job_camera_identity));
            if(same){ticket=camera_ticket;return C3X_RENDERER_RESULT_PENDING;}
        }
        return enqueue_camera_locked(frame,identity,ticket);
    }

private:
    bool compatible_ahead(c3x_renderer_frame_v1 const& frame,
                          c3x_renderer_camera_identity_v1 const& identity) const {
        auto candidate=frame;candidate.dirty_flags=ahead_frame.dirty_flags;
        auto clock=RendererState::resource_clock(frame);
        return native_presentation && ahead_frame.api_version && ahead_requested>=0 &&
            !ahead_cancelled.load(std::memory_order_relaxed) && clock>=ahead_requested && clock<=ahead_requested+2 &&
            !std::memcmp(&identity,&ahead_identity,sizeof(identity)) && same_ambient_view(candidate,ahead_frame);
    }

    int enqueue_camera_locked(c3x_renderer_frame_v1 const& frame,
                              c3x_renderer_camera_identity_v1 const& identity,
                              c3x_renderer_i64& ticket) {
        if(camera_ticket==INT64_MAX)return C3X_RENDERER_RESULT_ERROR;
        // Reject an unsupported publication before copying snapshots or asking
        // D3D for a target. Include worst-case fallback/replacement arrays.
        auto required=(std::uint64_t(frame.target_width)*frame.target_height+std::uint64_t(frame.tile_count)*2)*4+
            std::uint64_t(frame.tile_count)*sizeof(c3x_renderer_tile_v1)+sizeof(PublishedMapFrame);
        if(required>32u*1024u*1024u)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        std::vector<c3x_renderer_tile_v1> tiles;
        std::vector<c3x_renderer_u32> topology;
        try {
            if(frame.tile_count)tiles.assign(frame.tiles,frame.tiles+frame.tile_count);
            if(frame.world_topology_count)topology.assign(frame.world_topology,frame.world_topology+frame.world_topology_count);
        }catch(...){return C3X_RENDERER_RESULT_ERROR;}
        // Preserve a synchronous publication before background work may mutate
        // render scratch. Its previous borrowed pointer expires at this begin.
        if(completed_result==C3X_RENDERER_RESULT_OK && completed_output.bgra_pixels &&
           completed_output.bgra_pixels!=publication.output.bgra_pixels) {
            if(!publication.capture(completed_output,completed_phase_x,completed_phase_y,nullptr,completed_identity))
                return C3X_RENDERER_RESULT_ERROR;
            completed_output=publication.output;completed_identity=publication.identity;
        }
        start_locked();
        // A native request inside the proven ambient horizon transfers that
        // producer's result into the camera queue. Do not cancel work it needs.
        // Incompatible scene/view/visibility still cancels the horizon normally.
        if(!compatible_ahead(frame,identity)) {
            ahead_cancelled.store(true,std::memory_order_relaxed);ahead_requested=-1;
            for(auto& ready:ahead_ready)if(ready.output.bgra_pixels){++ahead_discarded;ready.clear();}
        }
        camera_pending_frame=frame;
        camera_pending_identity=identity;
        camera_pending_tiles.swap(tiles);camera_pending_topology.swap(topology);
        camera_pending_frame.tiles=camera_pending_tiles.empty()?nullptr:camera_pending_tiles.data();
        camera_pending_frame.world_topology=camera_pending_topology.empty()?nullptr:camera_pending_topology.data();
        ticket=++camera_ticket;
        camera_result=C3X_RENDERER_RESULT_PENDING;
        camera_ready.clear();
        camera_pending=true;
        camera_cancelled.store(true,std::memory_order_relaxed);
        foreground_pending.store(true,std::memory_order_relaxed);
        completed_scene_signature=0;
        snapshot_memory("camera-begin");
        wake.notify_one();
        return C3X_RENDERER_RESULT_PENDING;
    }

public:
    int camera_poll(c3x_renderer_i64 ticket,c3x_renderer_output_v1& output) {
        std::lock_guard<std::mutex> call_guard(call_mutex);
        std::lock_guard<std::mutex> lock(state_mutex);
        return camera_poll_locked(ticket,output);
    }

    int camera_poll_view(c3x_renderer_i64 ticket,c3x_renderer_camera_view_v1& view) {
        std::lock_guard<std::mutex> call_guard(call_mutex);
        std::lock_guard<std::mutex> lock(state_mutex);
        c3x_renderer_output_v1 output={};
        int result=camera_poll_locked(ticket,output);
        if(result==C3X_RENDERER_RESULT_OK || result==C3X_RENDERER_RESULT_PREVIEW){
            view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
            view.ticket=ticket;view.identity=publication.identity;
            view.frame=publication.frame;view.output=output;
        }
        return result;
    }

    // Pull-only presentation lease. The native caller has recaptured the view
    // it intends to display, including current visibility and local appearance.
    // A time difference can hold ambient animation; a content/view difference
    // cannot acquire pixels or replacement ownership. This never waits for D3D.
    int camera_present_view(c3x_renderer_camera_request_v1 const& request,
                            c3x_renderer_camera_view_v1& view) {
        std::lock_guard<std::mutex> call_guard(call_mutex);
        std::unique_lock<std::mutex> lock(state_mutex);
        if(gpu_presentation){
            drain_camera_locked(lock);gpu_presentation=false;gpu_publication.clear();
            nearby.clear();retained_views.clear();nearby_available=false;
        }
        native_presentation=true;isolated_publication=true;camera_preview_enabled=false;
        auto current=*request.frame;
        auto prior_request=current;prior_request.dirty_flags=publication.frame.dirty_flags;
        prospective_view_allowed=!publication.output.bgra_pixels || publication.matches_static_view(prior_request);
        if(!prospective_view_allowed){
            // Camera/content demand owns the working view. Abandon unstarted
            // alternate pixels during motion; compiled world content survives.
            prospective_views.erase(std::remove_if(prospective_views.begin(),prospective_views.end(),
                [&](auto const& queued){return queued.frame.tile_width!=current.tile_width;}),prospective_views.end());
            if(ahead_active && ahead_prospective && ahead_frame.tile_width!=current.tile_width){
                ahead_view_superseded=true;ahead_cancelled.store(true,std::memory_order_relaxed);
            }
        }
        nearby_presented=false;
        bool acquired=false;
        if(nearby_available && nearby.input.tile_width!=current.tile_width){
            for(auto& area:retained_views)if(area->input.tile_width==current.tile_width && area->project(current,request.identity,publication)){
                nearby.swap(*area);acquired=true;break;
            }
        }
        if(nearby_available && nearby.map.output.bgra_pixels) {
            auto exact_capture=[&](PublishedMapFrame const& candidate){
                auto normalized=current;normalized.dirty_flags=candidate.frame.dirty_flags;
                return candidate.output.bgra_pixels && current.presentation_time_ticks>=candidate.frame.presentation_time_ticks &&
                    candidate.matches_static_view(normalized) &&
                    !std::memcmp(&candidate.identity,&request.identity,sizeof(request.identity)) &&
                    current.world_topology_count==nearby.topology.size() && current.world_topology &&
                    !std::memcmp(current.world_topology,nearby.topology.data(),nearby.topology.size()*4);
            };
            if(exact_capture(publication) && publication.frame.presentation_time_ticks>=nearby.input.presentation_time_ticks)acquired=true;
            if(exact_capture(nearby.center) && (!acquired || nearby.center.frame.presentation_time_ticks>publication.frame.presentation_time_ticks)){
                publication.swap(nearby.center);acquired=true;
            } else if(!acquired)acquired=nearby.project(current,request.identity,publication);
        }
        if(acquired) {
            completed_output=publication.output;completed_identity=request.identity;
            completed_result=C3X_RENDERER_RESULT_OK;
            completed_resources=nearby.map.output.visible_animation_count>nearby.input.visible_animation_count?
                nearby.map.output.visible_animation_count-nearby.input.visible_animation_count:0;
            completed_phase_x=publication.phase_x;completed_phase_y=publication.phase_y;
            publication.scene_signature=c3x_renderer::terrain_frame_signature(current,
                publication.output.content_revision,publication.output.device_generation).complete;
            completed_resource_clock=RendererState::resource_clock(publication.frame);
            completed_scene_signature=publication.scene_signature;
            nearby_presented=true;trim_views();
            view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
            view.identity=publication.identity;view.frame=publication.frame;
            return_current_bitmap(current,view.output,"prepared-current-camera");
            return C3X_RENDERER_RESULT_OK;
        }
        // Dirty flags are scheduling hints. Complete fresh occurrences and the
        // native world/visibility revisions below establish content validity.
        current.dirty_flags=publication.frame.dirty_flags;
        if(completed_result!=C3X_RENDERER_RESULT_OK || !publication.output.bgra_pixels ||
           !publication.matches_static_view(current) ||
           std::memcmp(&publication.identity,&request.identity,sizeof(request.identity)))
            return C3X_RENDERER_RESULT_PENDING;
        c3x_renderer_output_v1 prepared={};
        current.dirty_flags=ahead_frame.dirty_flags;
        if(!camera_active && !camera_pending)consume_ahead_locked(lock,current,request.identity,prepared,false);
        view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
        view.ticket=camera_front_ticket;view.identity=publication.identity;
        view.frame=publication.frame;
        return_current_bitmap(*request.frame,view.output,"native-displayed-view");
        return C3X_RENDERER_RESULT_OK;
    }

    // The caller supplies a fresh authoritative snapshot after composition.
    // One cancellable job fills the same renderer's bounded larger working view.
    int prepare_nearby_view(c3x_renderer_camera_request_v1 const& request) {
        std::lock_guard<std::mutex> call_guard(call_mutex);
        std::unique_lock<std::mutex> lock(state_mutex);
        auto const& frame=*request.frame;
        auto const& displayed=gpu_presentation?gpu_publication:publication;
        char option[8]={};GetEnvironmentVariableA("C3X_RENDERER_PREPARED_VIEW",option,sizeof(option));
        if(!native_presentation || !nearby_available || std::strcmp(option,"0")==0 ||
           !frame.world_topology || !frame.world_topology_count || frame.world_topology_count>1024u*1024u || frame.tile_count>8192 || !frame.tile_count || !frame.tiles ||
           frame.target_width>2240 || frame.target_height>1192 ||
           (frame.target_width>2224 && frame.target_height>1176))return C3X_RENDERER_RESULT_ERROR;
        if(ahead_active || area_pending){
            // Caller-driven priority: do not let speculative zoom construction
            // starve the map the game is currently asking to animate.
            if(ahead_active && ahead_prospective && nearby_presented && frame.presentation_frequency>0 &&
               (!prospective_view_allowed || frame.presentation_time_ticks-displayed.frame.presentation_time_ticks>=frame.presentation_frequency/10)){
                if(!pending_refresh){
                    auto snapshot=std::make_unique<ProspectiveView>();snapshot->frame=frame;snapshot->identity=request.identity;
                    snapshot->tiles.assign(frame.tiles,frame.tiles+frame.tile_count);
                    snapshot->topology.assign(frame.world_topology,frame.world_topology+frame.world_topology_count);
                    pending_refresh=std::move(snapshot);
                }
                ahead_cancelled.store(true,std::memory_order_relaxed);
            }
            return C3X_RENDERER_RESULT_OK;
        }
        if(nearby_presented && nearby.centered(frame) &&
           (nearby.map.output.visible_animation_count<=nearby.input.visible_animation_count || RendererState::resource_clock(frame)==RendererState::resource_clock(displayed.frame)))
            return C3X_RENDERER_RESULT_OK;
        clear_ahead();
        ahead_selected=nearby_presented && nearby.centered(frame) && frame.presentation_frequency>0 &&
            frame.presentation_time_ticks-nearby.input.presentation_time_ticks<frame.presentation_frequency/4;
        if(ahead_selected){ahead_selected_frame=frame;ahead_selected_tiles.assign(frame.tiles,frame.tiles+frame.tile_count);
            ahead_selected_frame.tiles=ahead_selected_tiles.data();}
        if(nearby_presented && nearby.centered(frame)) {
            // Fresh proof has validated this complete contributor set. Animate
            // in its existing world placement; small camera moves only select
            // a crop and must not reconstruct the surrounding static surface.
            ahead_tiles=nearby.source_tiles;ahead_topology=nearby.topology;
            ahead_frame=nearby.input;
            ahead_frame.target_width=nearby.viewport_width;ahead_frame.target_height=nearby.viewport_height;
            ahead_frame.clip_left=frame.clip_left;ahead_frame.clip_top=frame.clip_top;
            ahead_frame.clip_right=frame.clip_right;ahead_frame.clip_bottom=frame.clip_bottom;
            ahead_frame.presentation_time_ticks=frame.presentation_time_ticks;
            ahead_frame.visible_animation_count=frame.visible_animation_count;
        } else {
            ahead_tiles.assign(frame.tiles,frame.tiles+frame.tile_count);
            ahead_topology.assign(frame.world_topology,frame.world_topology+frame.world_topology_count);
            ahead_frame=frame;
        }
        ahead_identity=request.identity;
        ahead_frame.tiles=ahead_tiles.data();ahead_frame.world_topology=ahead_topology.data();
        area_pending=true;ahead_cancelled.store(false,std::memory_order_relaxed);
        wake.notify_one();return C3X_RENDERER_RESULT_OK;
    }

    int prepare_view(c3x_renderer_camera_request_v1 const& request,bool query_only) {
        std::lock_guard<std::mutex> call_guard(call_mutex);
        std::unique_lock<std::mutex> lock(state_mutex);
        auto const& frame=*request.frame;
        if(!native_presentation || !nearby_available || frame.target_width>2240 || frame.target_height>1192 ||
           !frame.tile_count || frame.tile_count>8192 || !frame.world_topology_count || frame.world_topology_count>1024u*1024u)
            return C3X_RENDERER_RESULT_ERROR;
        auto matches=[&](auto const& f,auto const& id){return f.tile_width==frame.tile_width && f.tile_height==frame.tile_height &&
            f.hour==frame.hour && f.season==frame.season && !std::memcmp(&id,&request.identity,sizeof(id));};
        if(matches(nearby.input,nearby.identity))return C3X_RENDERER_RESULT_OK;
        for(auto const& area:retained_views)if(matches(area->input,area->identity))return C3X_RENDERER_RESULT_OK;
        if(ahead_active && matches(ahead_frame,ahead_identity))return C3X_RENDERER_RESULT_OK;
        for(auto const& queued:prospective_views)if(matches(queued.frame,queued.identity))return C3X_RENDERER_RESULT_OK;
        if(query_only)return C3X_RENDERER_RESULT_PENDING;
        auto payload=std::size_t(frame.tile_count)*sizeof(c3x_renderer_tile_v1)+std::size_t(frame.world_topology_count)*4;
        if(!prospective_view_allowed)return C3X_RENDERER_RESULT_PENDING;
        std::size_t bytes=payload;for(auto const& queued:prospective_views)bytes+=queued.tiles.size()*sizeof(c3x_renderer_tile_v1)+queued.topology.size()*4;
        if(prospective_views.size()>=2 || bytes>16u*1024u*1024u)return C3X_RENDERER_RESULT_PENDING;
        ProspectiveView snapshot; snapshot.frame=frame;snapshot.identity=request.identity;
        snapshot.tiles.assign(frame.tiles,frame.tiles+frame.tile_count);
        snapshot.topology.assign(frame.world_topology,frame.world_topology+frame.world_topology_count);
        prospective_views.push_back(std::move(snapshot));wake.notify_one();return C3X_RENDERER_RESULT_OK;
    }
private:
    bool same_ambient_view(c3x_renderer_frame_v1 const& current,
                           c3x_renderer_frame_v1 const& captured)const {
        if(!captured.api_version || current.tile_count!=captured.tile_count ||
           current.world_topology_count!=captured.world_topology_count)return false;
        auto left=current,right=captured;
        left.tiles=right.tiles=nullptr;left.world_topology=right.world_topology=nullptr;
        left.presentation_time_ticks=right.presentation_time_ticks=0;
        left.clip_left=right.clip_left=left.clip_top=right.clip_top=0;
        left.clip_right=right.clip_right=left.clip_bottom=right.clip_bottom=0;
        left.visible_animation_count=right.visible_animation_count=0;
        return !std::memcmp(&left,&right,sizeof(left)) &&
            (!current.tile_count || !std::memcmp(current.tiles,captured.tiles,
                std::size_t(current.tile_count)*sizeof(*current.tiles))) &&
            (!current.world_topology_count || !std::memcmp(current.world_topology,captured.world_topology,
                std::size_t(current.world_topology_count)*sizeof(*current.world_topology)));
    }

    void return_current_bitmap(c3x_renderer_frame_v1 const& frame,
                               c3x_renderer_output_v1& output,char const* reason) {
        if(completed_output.cache_hits!=0xffffffffu)++completed_output.cache_hits;
        if(fast_cache_hits!=0xffffffffu)++fast_cache_hits;
        output=completed_output;
        output.clip_left=frame.clip_left;output.clip_top=frame.clip_top;
        output.clip_right=frame.clip_right;output.clip_bottom=frame.clip_bottom;
        output.visible_animation_count=frame.visible_animation_count+completed_resources;
        output.request_continuous_redraw=output.visible_animation_count!=0;
        output.frame_invalidation_flags=0;
        output.geometry_tiles_built=output.geometry_tiles_reused=output.geometry_tiles_evicted=0;
        output.geometry_upload_bytes=0;output.geometry_ticks=output.draw_ticks=output.readback_ticks=output.renderer_cpu_ticks=0;
        output.raster_reused_pixels=output.raster_draw_pixels=output.raster_cached_pixels=0;
        char detail[224];sprintf_s(detail,"reason=%s tiles=%u pending=%u built=%u cancelled=%u blocks=%u immutable=1",
            reason,frame.tile_count,output.prefetch_tiles_pending,output.prefetch_tiles_built,
            output.prefetch_tiles_cancelled,output.prefetch_blocks_built);
        renderer_state.trace.write("worker-current-bitmap",detail);
    }

    bool same_camera_request(c3x_renderer_frame_v1 const& a,c3x_renderer_camera_identity_v1 const& ai,
                             c3x_renderer_frame_v1 const& b,c3x_renderer_camera_identity_v1 const& bi)const {
        auto left=a,right=b;
        left.tiles=right.tiles=nullptr;left.world_topology=right.world_topology=nullptr;
        // Padding differences can only decline optional reuse. The ordered
        // payloads have no pointers; equal bytes include every semantic field.
        return !std::memcmp(&ai,&bi,sizeof(ai)) && !std::memcmp(&left,&right,sizeof(left)) &&
            (!a.tile_count || !std::memcmp(a.tiles,b.tiles,std::size_t(a.tile_count)*sizeof(*a.tiles))) &&
            (!a.world_topology_count || !std::memcmp(a.world_topology,b.world_topology,std::size_t(a.world_topology_count)*sizeof(*a.world_topology)));
    }

    int camera_poll_locked(c3x_renderer_i64 ticket,c3x_renderer_output_v1& output) {
        if(ticket<=0 || ticket!=camera_ticket)return C3X_RENDERER_RESULT_SUPERSEDED;
        if(camera_result!=C3X_RENDERER_RESULT_OK && camera_result!=C3X_RENDERER_RESULT_PENDING)return camera_result;
        if(camera_ready.output.bgra_pixels) {
            bool stable=native_presentation && publication.matches_static_view(camera_ready.frame) &&
                !std::memcmp(&publication.identity,&camera_ready.identity,sizeof(publication.identity));
            publication.swap(camera_ready);
            camera_ready.clear();
            completed_output=publication.output;completed_identity=publication.identity;
            completed_phase_x=publication.phase_x;completed_phase_y=publication.phase_y;
            completed_result=C3X_RENDERER_RESULT_OK;
            camera_front_ticket=ticket;camera_front_result=camera_ready_result;
            if(native_presentation && camera_ready_result==C3X_RENDERER_RESULT_OK) {
                completed_scene_signature=publication.scene_signature;
                completed_resources=renderer_state.ambient_count();
                completed_resource_clock=RendererState::resource_clock(publication.frame);
                job_stable_view=stable;
                if(!compatible_ahead(job_frame,job_camera_identity))start_ahead();
            }
        }
        if(camera_front_ticket!=ticket)return C3X_RENDERER_RESULT_PENDING;
        output=completed_output;
        return camera_front_result;
    }

public:
    int camera_cancel(c3x_renderer_i64 ticket) {
        std::lock_guard<std::mutex> call_guard(call_mutex);
        std::lock_guard<std::mutex> lock(state_mutex);
        if(ticket<=0 || ticket!=camera_ticket)return C3X_RENDERER_RESULT_SUPERSEDED;
        camera_cancelled.store(true,std::memory_order_relaxed);
        camera_pending=false;camera_result=C3X_RENDERER_RESULT_SUPERSEDED;
        camera_ready.clear();
        camera_pending_tiles.clear();camera_pending_topology.clear();
        if(!camera_active)foreground_pending.store(false,std::memory_order_relaxed);
        wake.notify_one();
        return C3X_RENDERER_RESULT_OK;
    }

    int blit(c3x_renderer_output_v1 const & output, HDC destination) {
        // This method executes on Civ III's calling/UI thread. Holding the call
        // gate excludes foreground rendering/reset. Idle mesh preparation may
        // continue because it never touches this bitmap, its ownership arrays,
        // or the GDI blit resources.
        std::lock_guard<std::mutex> call_guard(call_mutex);
        LARGE_INTEGER started = {}, finished = {};
        QueryPerformanceCounter(&started);
        int result = map_blitter.blit(output, destination, completed_phase_x, completed_phase_y, renderer_state.trace)
            ? C3X_RENDERER_RESULT_OK : C3X_RENDERER_RESULT_ERROR;
        QueryPerformanceCounter(&finished);
        if (renderer_state.trace.level) {
            char detail[160];
            std::snprintf(detail, sizeof(detail), "result=%d blit_ms=%.3f clip=%d,%d,%d,%d",
                result, renderer_state.trace.milliseconds(finished.QuadPart - started.QuadPart),
                output.clip_left, output.clip_top, output.clip_right, output.clip_bottom);
            renderer_state.trace.write("blit", detail, result != C3X_RENDERER_RESULT_OK);
        }
        return result;
    }

    void forget_unit(int id) {
        std::lock_guard<std::mutex> call_guard(call_mutex);
        std::lock_guard<std::mutex> lock(state_mutex);
        unit_instances.forget(id);unit_pixels_queue.forget(id);
        // In-flight poses are immutable shared content. They cannot publish
        // themselves or restore the retired instance's selection.
    }

    int draw_unit(c3x_renderer_unit_v1 const & request,HDC destination,HDC background=nullptr,int* bounds=nullptr,unsigned playback_flags=0,c3x_renderer_gpu_unit_v1 const* gpu_target=nullptr) {
        std::lock_guard<std::mutex> call_guard(call_mutex);
        std::unique_lock<std::mutex> lock(state_mutex);
        if(!renderer_state.unit_rendering_enabled)return C3X_RENDERER_RESULT_ERROR;
        if(playback_flags&C3X_RENDERER_UNIT_HIDDEN){
            if(!(playback_flags&C3X_RENDERER_UNIT_STATE_CAPTURED))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            unit_instances.forget(request.unit_id);unit_pixels_queue.forget(request.unit_id);
            if(bounds){bounds[0]=bounds[2]=request.body_x;bounds[1]=bounds[3]=request.body_y;}
            return C3X_RENDERER_RESULT_OK;
        }
        start_locked();
        auto const& catalog=renderer_state.unit_bodies.units;
        c3x_renderer::render_core::UnitInstances::Selection selection;
        if(!unit_instances.capture(request,playback_flags,catalog,c3x_renderer::native_unit_action,selection)){
            char detail[160];std::snprintf(detail,sizeof(detail),"id=%d key=%.63s action=%d reason=missing-or-invalid-3d-binding",
                request.unit_id,request.unit_key,request.action);
            renderer_state.trace.write("unit-capture-failed",detail,true);
            return C3X_RENDERER_RESULT_ERROR;
        }
        job_unit_selection=selection;
        if(gpu_target)advance_visual_clock();
        if(!unit_instances.sample(selection,gpu_target?visual_ticks:request.presentation_time_ticks,gpu_target?visual_frequency:request.presentation_frequency,
            catalog,job_unit,job_unit_predict))return C3X_RENDERER_RESULT_SUPERSEDED;
        auto definition=unit_instances.definition(selection,catalog);
        auto action_name=c3x_renderer::native_unit_action(job_unit.action);
        auto playback_clip=std::find_if(definition->actions.begin(),definition->actions.end(),
            [&](auto const& action){return action.name==action_name;});
        if(bounds) {
            int projection=request.projection_scale_milli>0?request.projection_scale_milli:(request.reduced?500:1000);
            if(!c3x_renderer::expand_unit_canvas(job_unit.body_x,job_unit.body_y,job_unit.sprite_width,job_unit.sprite_height,
                                   projection,definition->minimum_canvas))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        }
        LARGE_INTEGER started={},finished={};QueryPerformanceCounter(&started);
        // Cached unit pixels are an independent CPU publication. Do not cancel
        // an exact ambient map render merely to copy a pose already in memory.
        if(unit_pixels_enabled) {
            unit_pixels_queue.observe(job_unit,playback_clip->loop,job_unit_predict!=0,job_unit_predict);
            if(job_unit_predict)wake.notify_one();
        }
        unit_gpu_preparation=gpu_target!=nullptr;
        if(gpu_target){
            ForegroundCameraPause pause(*this,lock);gpu_unit=*gpu_target;
            int code=submit_locked(lock,Command::gpu_unit);
            auto const& pose=renderer_state.unit_bodies.resident_pose;
            if(code==C3X_RENDERER_RESULT_OK&&bounds){bounds[0]=job_unit.body_x;bounds[1]=job_unit.body_y;bounds[2]=job_unit.body_x+renderer_state.unit_bodies.image_width;bounds[3]=job_unit.body_y+renderer_state.unit_bodies.image_height;}
            if(code==C3X_RENDERER_RESULT_OK&&pose.prepared)++unit_pixels_hits;
            QueryPerformanceCounter(&finished);auto const& body=renderer_state.unit_bodies;char detail[512];
            std::snprintf(detail,sizeof(detail),"id=%d key=%.63s action=%d cursor=%d/%d result=%d gpu_composition=1 resident_pose=%u direct_scene=%u cache_hit=%u prepared=%u pose_builds=%llu pose_hits=%llu pose_bytes=%zu body_readbacks=%llu composition_uploads=%llu shadow_passes=%llu shadow_input_bytes=%llu cpu_shadow_upload_bytes=%llu readback_ms=%.3f ms=%.3f",
                request.unit_id,request.unit_key,request.action,request.action_cursor,request.frame_count,code,unsigned(!body.direct_scene),unsigned(body.direct_scene),unsigned(body.cache_hit),unsigned(pose.prepared),
                static_cast<unsigned long long>(body.resident_pose_builds),static_cast<unsigned long long>(body.resident_pose_hits),body.resident_pose_bytes,static_cast<unsigned long long>(gpu_unit_output_readbacks),static_cast<unsigned long long>(gpu_unit_composition_uploads),static_cast<unsigned long long>(body.gpu_shadow_passes),static_cast<unsigned long long>(body.gpu_shadow_input_bytes),static_cast<unsigned long long>(body.cpu_shadow_upload_bytes),body.readback_ms,renderer_state.trace.milliseconds(finished.QuadPart-started.QuadPart));
            renderer_state.trace.write("unit-body",detail,code!=C3X_RENDERER_RESULT_OK||!body.cache_hit);return code;
        }
        c3x_renderer::UnitBodyRenderer::PublishedPose cached;
        bool hit=renderer_state.unit_bodies.copy_cached(job_unit,cached);
        if(!hit && unit_pixels_active) {
            // Reserve the next owner turn before releasing the queue mutex.
            // Cancellation alone cannot prevent another speculative batch from
            // starting while this caller reacquires the mutex after notification.
            camera_paused=true;
            foreground_pending.store(true,std::memory_order_relaxed);
            completed.wait(lock,[this]{return !unit_pixels_active;});
            camera_paused=false;
            foreground_pending.store(camera_pending,std::memory_order_relaxed);
            hit=renderer_state.unit_bodies.copy_cached(job_unit,cached);
            wake.notify_one();
        }
        if(hit) {
            unsigned keyed=0;
            int result=renderer_state.unit_bodies.blit(cached,destination,job_unit.body_x,job_unit.body_y,background,keyed)
                ? C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_ERROR;
            if(cached.prepared)++unit_pixels_hits;
            if(result==C3X_RENDERER_RESULT_OK && bounds) {
                bounds[0]=job_unit.body_x;bounds[1]=job_unit.body_y;
                bounds[2]=job_unit.body_x+cached.width;bounds[3]=job_unit.body_y+cached.height;
            }
            QueryPerformanceCounter(&finished);
            char detail[512];std::snprintf(detail,sizeof(detail),
                "id=%d key=%.63s action=%d cursor=%d/%d dir=%d result=%d cache_hit=1 cache_only=%u gpu_composition=%u prepared=%u prepared_hits=%llu keyed=%u shadow_pixels=%u ms=%.3f",
                request.unit_id,request.unit_key,request.action,request.action_cursor,request.frame_count,
                request.direction,result,1u,0u,unsigned(cached.prepared),static_cast<unsigned long long>(unit_pixels_hits),keyed,cached.cast_pixels,
                renderer_state.trace.milliseconds(finished.QuadPart-started.QuadPart));
            renderer_state.trace.write("unit-body",detail,result!=C3X_RENDERER_RESULT_OK);
            return result;
        }
        {
            char detail[256];std::snprintf(detail,sizeof(detail),
                "id=%d key=%.63s action=%d cursor=%d/%d dir=%d reason=%s entries=%zu bytes=%zu",
                request.unit_id,request.unit_key,request.action,request.action_cursor,request.frame_count,
                request.direction,"cache-miss",
                renderer_state.unit_bodies.cached_pose_entries(),renderer_state.unit_bodies.cached_pose_bytes());
            renderer_state.trace.write("unit-cache-only-miss",detail,true);
        }
        ForegroundCameraPause pause(*this,lock);
        int result=submit_locked(lock,Command::unit);
        lock.unlock();
        if(result==C3X_RENDERER_RESULT_OK && !renderer_state.unit_bodies.blit(destination,job_unit.body_x,job_unit.body_y,background)) {
            result=C3X_RENDERER_RESULT_ERROR;renderer_state.unit_bodies.failure_reason="native-canvas-blit";
        }
        if(result==C3X_RENDERER_RESULT_OK && bounds) {
            bounds[0]=job_unit.body_x;bounds[1]=job_unit.body_y;
            bounds[2]=job_unit.body_x+renderer_state.unit_bodies.image_width;
            bounds[3]=job_unit.body_y+renderer_state.unit_bodies.image_height;
        }
        QueryPerformanceCounter(&finished);
        char detail[384];std::snprintf(detail,sizeof(detail),
            "id=%d key=%.63s action=%d queued=%d cursor=%d/%d dir=%d xy=%d,%d reduced=%d color=%06x result=%d reason=%s cache_hit=%d cache_bytes=%zu keyed=%u shadow_pixels=%u ms=%.3f",
            request.unit_id,request.unit_key,request.action,request.queued_action,request.action_cursor,request.frame_count,
            request.direction,request.body_x,request.body_y,request.reduced,request.display_color_rgb,result,renderer_state.unit_bodies.failure_reason,
            renderer_state.unit_bodies.cache_hit?1:0,renderer_state.unit_bodies.cache_bytes,renderer_state.unit_bodies.keyed_pixels,renderer_state.unit_bodies.cast_pixels,
            renderer_state.trace.milliseconds(finished.QuadPart-started.QuadPart));
        renderer_state.trace.write("unit-body",detail,true);
        return result;
    }

#ifdef C3X_RENDERER_BENCHMARK_ORACLE
    int benchmark_trim_to_prepared(c3x_renderer_benchmark_oracle_trim_v1 & result,unsigned mode=0) {
        std::lock_guard<std::mutex> call_guard(call_mutex);
        std::unique_lock<std::mutex> lock(state_mutex);
        start_locked();
        drain_camera_locked(lock);
        benchmark_reset_mode=mode;
        int code=submit_locked(lock,Command::benchmark_trim);
        result=benchmark_trim_result;
        return code;
    }
#endif

    void reset_and_stop() {
        std::unique_lock<std::mutex> call_guard(call_mutex);stop_visual_timer();
        std::unique_lock<std::mutex> lock(state_mutex);
        if (!running)
            return;
        drain_camera_locked(lock);
        renderer_state.unit_bodies.release_pose_leases();
        if(renderer_state.trace.level) {
            char instances[256];std::snprintf(instances,sizeof(instances),"retained=%zu captures=%llu reused=%llu bindings=%llu evictions=%llu",
                unit_instances.size(),static_cast<unsigned long long>(unit_instances.captures),static_cast<unsigned long long>(unit_instances.reused),
                static_cast<unsigned long long>(unit_instances.bindings),static_cast<unsigned long long>(unit_instances.evictions));
            renderer_state.trace.write("unit-instances",instances,true);
            auto stats=renderer_state.unit_bodies.pose_preparation_statistics();char detail[384];
            std::snprintf(detail,sizeof(detail),"built=%llu consumed=%llu cancelled=%llu evicted=%llu rejected=%llu cpu_ms=%.3f join_ms=%.3f peak_ready_bytes=%zu retained_bytes=%zu active_peak=%u",
                static_cast<unsigned long long>(stats.built),static_cast<unsigned long long>(stats.consumed),
                static_cast<unsigned long long>(stats.cancelled),static_cast<unsigned long long>(stats.evicted),
                static_cast<unsigned long long>(stats.rejected),stats.cpu_ms,stats.wait_ms,stats.peak_bytes,
                renderer_state.unit_bodies.pose_retained_bytes(),stats.active_peak);
            renderer_state.trace.write("unit-preparation-summary",detail,true);
        }
        submit_locked(lock, Command::reset);
        stop_requested = true;
        wake.notify_one();
        lock.unlock();
        worker.join();
        lock.lock();
        running = false;
        stop_requested = false;
        map_blitter.reset_blit_surface();
    }

private:
    enum class Command {
        none,
        configure_pack,
        configure_definitions,
        render,
        gpu_render,
        gpu_images,
        gpu_unit,
        gpu_present,
        visual_frame,
        native_screen,
        unit,
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
        benchmark_trim,
#endif
        reset
    };

    c3x_renderer_gpu_frame_v1 gpu_view={sizeof(gpu_view)};
    c3x_renderer_output_v1 gpu_metadata={C3X_RENDERER_API_VERSION,sizeof(gpu_metadata)};
    std::vector<unsigned> gpu_replacements,gpu_fallbacks;
    c3x_renderer_gpu_images_v1 gpu_request={};
    c3x_renderer_gpu_unit_v1 gpu_unit={};
    bool unit_gpu_preparation=false;
    std::uint64_t gpu_unit_output_readbacks=0,gpu_unit_composition_uploads=0;
    c3x_renderer_gpu_result_v1 gpu_result={sizeof(gpu_result)};
    std::vector<unsigned> gpu_pixels,gpu_readback;
    std::vector<c3x_gpu_images::Command> gpu_commands;
    c3x_gpu_images::NativePresenter gpu_presenter;
    bool native_screen_active=false,screen_upload=false;
    std::vector<unsigned short> screen_pixels;unsigned screen_format=1;
    int screen_width=0,screen_height=0;RECT screen_area={};
    c3x_renderer_gpu_present_v1 gpu_present={};
    UINT_PTR visual_timer=0;bool visual_allowed=true;
    long long visual_ticks=0,visual_last=0,visual_frequency=0;
    std::uint64_t visual_frames=0,visual_map_samples=0,visual_unit_samples=0,visual_pose_changes=0;
    c3x_renderer::render_core::DynamicSceneInputs dynamic_inputs;
    c3x_renderer::render_core::UnitInstances::Selection job_unit_selection;
    RendererState & renderer_state;
    LARGE_INTEGER job_timing_begin={},job_timing_rendered={},job_timing_published={};
    MapBlitter map_blitter;
    PublishedMapFrame publication;
    PublishedMapFrame camera_ready;
    bool camera_preview_enabled=false;
    bool ambient_async_enabled=false;
    bool native_presentation=false,gpu_presentation=false,gpu_reused=false;
    PublishedMapFrame gpu_publication;
    c3x_renderer::PreparedViewArea<PublishedMapFrame> nearby;
    bool nearby_available=false,nearby_presented=false,area_pending=false;
    std::vector<std::unique_ptr<c3x_renderer::PreparedViewArea<PublishedMapFrame>>> retained_views;
    struct ProspectiveView {
        c3x_renderer_frame_v1 frame={};c3x_renderer_camera_identity_v1 identity={};
        std::vector<c3x_renderer_tile_v1> tiles;std::vector<c3x_renderer_u32> topology;
    };
    std::deque<ProspectiveView> prospective_views;
    std::unique_ptr<ProspectiveView> pending_refresh;
    bool ahead_prospective=false,prospective_view_allowed=true,ahead_view_superseded=false;
    static constexpr std::size_t retained_view_budget=64u*1024u*1024u;
    c3x_renderer_frame_v1 ahead_selected_frame={};
    std::vector<c3x_renderer_tile_v1> ahead_selected_tiles;
    bool ahead_selected=false;
    std::size_t retained_view_bytes() const {
        auto bytes=nearby.bytes();for(auto const& view:retained_views)bytes+=view->bytes();return bytes;
    }
    void trim_views(){
        if(nearby.bytes()>decltype(nearby)::budget)nearby.center.clear();
        while(!retained_views.empty() && (retained_views.size()>2 || retained_view_bytes()>retained_view_budget))retained_views.erase(retained_views.begin());
    }
    void retain_view(c3x_renderer::PreparedViewArea<PublishedMapFrame>& incoming) {
        // Known local mutations retire stale alternate views immediately.
        // Unknown/outside capture remains a fresh-publication proof obligation.
        for(auto it=retained_views.begin();it!=retained_views.end();){
            auto const& old=**it;
            bool changed=std::memcmp(&old.identity,&incoming.identity,sizeof(old.identity))!=0;
            for(auto const& tile:incoming.source_tiles){
                if(changed)break;
                if(!(tile.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)))continue;
                auto found=old.index.find(old.key(tile));if(found==old.index.end())continue;
                auto const& prior=old.tiles[found->second];
                if((prior.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)) && !old.same_content(prior,tile))changed=true;
            }
            if(changed)it=retained_views.erase(it);else ++it;
        }
        if(nearby.map.has_image() &&
           (nearby.map.output.device_generation!=incoming.map.output.device_generation || nearby.map.output.content_revision!=incoming.map.output.content_revision)){
            nearby.clear();retained_views.clear();
        }
        if(nearby.map.has_image() && incoming.input.tile_width!=nearby.input.tile_width){
            for(auto it=retained_views.begin();it!=retained_views.end();) {
                if((*it)->input.tile_width==nearby.input.tile_width)it=retained_views.erase(it);else ++it;
            }
            auto old=std::make_unique<c3x_renderer::PreparedViewArea<PublishedMapFrame>>();old->swap(nearby);
            retained_views.push_back(std::move(old));
        }
        nearby.swap(incoming);trim_views();
    }
    bool unit_pixels_enabled=false,unit_pixels_active=false,unit_pixels_turn=true;
    c3x_renderer::render_core::UnitFramePreparation unit_pixels_queue;
    std::uint64_t unit_content_revision=0,unit_content_examined=0,unit_offers_examined=0;
    bool unit_preparation_pending() const {
        return unit_pixels_enabled && !unit_pixels_queue.empty() &&
            (!unit_gpu_preparation || unit_content_revision!=unit_content_examined || unit_pixels_queue.offered!=unit_offers_examined);
    }
    std::uint64_t unit_pixels_built=0,unit_pixels_hits=0,unit_pixels_batches=0;
    bool ahead_enabled=false,ahead_active=false,job_stable_view=false;
    std::atomic<bool> ahead_cancelled{false};
    c3x_renderer_frame_v1 ahead_frame={};
    c3x_renderer_camera_identity_v1 ahead_identity={};
    std::vector<c3x_renderer_tile_v1> ahead_tiles;
    std::vector<c3x_renderer_u32> ahead_topology;
    std::array<PublishedMapFrame,2> ahead_ready;
    c3x_renderer_i64 ahead_requested=-1,ahead_next=-1,ahead_active_clock=-1;
    std::uint64_t ahead_built=0,ahead_hits=0,ahead_discarded=0;
    std::size_t ahead_peak_bytes=0;
    static constexpr std::size_t ahead_budget=32u*1024u*1024u;
    c3x_renderer_i64 camera_front_ticket=0;
    int camera_ready_result=C3X_RENDERER_RESULT_OK,camera_front_result=C3X_RENDERER_RESULT_PENDING;
    c3x_renderer_frame_v1 camera_pending_frame={};
    c3x_renderer_camera_identity_v1 camera_pending_identity={},job_camera_identity={},completed_identity={};
    std::vector<c3x_renderer_tile_v1> camera_pending_tiles;
    std::vector<c3x_renderer_u32> camera_pending_topology;
    c3x_renderer_i64 camera_ticket=0,job_camera_ticket=0;
    int camera_result=C3X_RENDERER_RESULT_SUPERSEDED;
    bool camera_active=false,camera_pending=false,camera_paused=false;
    std::atomic<bool> camera_cancelled{false};
    bool isolated_publication=false;
    int completed_phase_x=0,completed_phase_y=0;
    std::mutex call_mutex;
    std::mutex state_mutex;
    std::condition_variable wake;
    std::condition_variable completed;
    std::thread worker;
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
    c3x_renderer_benchmark_oracle_trim_v1 benchmark_trim_result={};
    unsigned benchmark_reset_mode=0;
#endif
    bool running = false;
    bool stop_requested = false;
    bool has_job = false;
    std::atomic<bool> foreground_pending{false};
    c3x_renderer_frame_v1 warm_frame = {};
    std::vector<c3x_renderer_tile_v1> warm_tiles;
    std::vector<c3x_renderer_u32> warm_world_topology;
    std::vector<unsigned> warm_order;
    std::size_t warm_cursor = 0;
    std::uint64_t warm_signature = 0;
    unsigned prepared_tiles = 0, cancelled_tiles = 0, unavailable_tiles = 0;
    c3x_renderer_i64 preparation_ticks = 0;
    Command job_command = Command::none;
    std::uint64_t latest_job_sequence = 0;
    std::uint64_t completed_job_sequence = 0;
    int completed_result = C3X_RENDERER_RESULT_ERROR;
    int last_job_result = C3X_RENDERER_RESULT_ERROR;
    c3x_renderer_unit_v1 job_unit={};
    unsigned job_unit_predict=1;
    c3x_renderer::render_core::UnitInstances unit_instances;
    c3x_renderer_output_v1 completed_output = {};
    std::uint64_t completed_scene_signature=0;
    unsigned fast_cache_hits=0;
    unsigned completed_resources=0;
    c3x_renderer_i64 completed_resource_clock=-1;
    c3x_renderer_frame_v1 job_frame = {};
    std::vector<c3x_renderer_tile_v1> job_tiles;
    std::vector<c3x_renderer_u32> job_world_topology;
    bool job_pack_present = false;
    bool job_mod_root_present = false;
    bool job_default_path_present = false;
    bool job_scenario_path_present = false;
    bool job_custom_path_present = false;
    std::string job_pack_path;
    std::string job_mod_root;
    std::string job_default_path;
    std::string job_scenario_path;
    std::string job_custom_path;

    void start_locked() {
        if (running)
            return;
        stop_requested = false;
        has_job = false;
        running = true;
        worker = std::thread(&RendererWorker::run, this);
    }

    void snapshot_memory(char const* phase) {
        if(!renderer_state.profiling)return;
        char detail[384];std::snprintf(detail,sizeof(detail),
            "phase=%s active_tiles=%zu pending_tiles=%zu warm_tiles=%zu active_topology=%zu pending_topology=%zu warm_topology=%zu front=%zu ready=%zu nearby=%zu area_building=%u active=%u pending=%u paused=%u",
            phase,job_tiles.capacity()*sizeof(c3x_renderer_tile_v1),camera_pending_tiles.capacity()*sizeof(c3x_renderer_tile_v1),
            warm_tiles.capacity()*sizeof(c3x_renderer_tile_v1),job_world_topology.capacity()*4,
            camera_pending_topology.capacity()*4,warm_world_topology.capacity()*4,publication.bytes(),camera_ready.bytes(),nearby.bytes(),unsigned(ahead_active),
            unsigned(camera_active),unsigned(camera_pending),unsigned(camera_paused));
        renderer_state.trace.write("memory-worker",detail,true);
    }

    void drain_camera_locked(std::unique_lock<std::mutex>& lock,bool preserve_preparation=false) {
        // Configuration/reset and incompatible map demand also own the next
        // turn; no optional producer may refill while they drain active work.
        camera_paused=true;
        foreground_pending.store(true,std::memory_order_relaxed);
        completed.wait(lock,[this]{return !unit_pixels_active;});
        pause_ahead_locked(lock,preserve_preparation);
        // Configuration, reset and incompatible synchronous draws replace camera work.
        // Units use a resumable pause instead. Never change
        // job_frame or any renderer-owned object while the camera uses it.
        camera_cancelled.store(true,std::memory_order_relaxed);
        camera_pending=false;camera_result=C3X_RENDERER_RESULT_SUPERSEDED;
        completed.wait(lock,[this]{return !camera_active;});
        camera_paused=false;
        foreground_pending.store(false,std::memory_order_relaxed);
        camera_ready.clear();
        camera_pending_tiles.clear();camera_pending_topology.clear();
        wake.notify_one();
    }

    // A foreground draw/transfer owns the renderer through its UI-thread copy.
    // Save the interrupted immutable map without allocating a third snapshot.
    // A newer pending camera takes precedence over the interrupted one.
    struct ForegroundCameraPause {
        RendererWorker& owner;
        std::unique_lock<std::mutex>& lock;
        ForegroundCameraPause(RendererWorker& worker,std::unique_lock<std::mutex>& gate):owner(worker),lock(gate) {
            owner.camera_paused=true;
            owner.pause_ahead_locked(lock,owner.gpu_presentation);
            bool const interrupted=owner.camera_active;
            owner.foreground_pending.store(true,std::memory_order_relaxed);
            if(interrupted)owner.camera_cancelled.store(true,std::memory_order_relaxed);
            owner.completed.wait(lock,[&]{return !owner.camera_active;});
            if(interrupted && !owner.camera_pending && owner.camera_result==C3X_RENDERER_RESULT_PENDING) {
                owner.camera_pending_frame=owner.job_frame;
                owner.camera_pending_identity=owner.job_camera_identity;
                owner.camera_pending_tiles.swap(owner.job_tiles);
                owner.camera_pending_topology.swap(owner.job_world_topology);
                owner.camera_pending_frame.tiles=owner.camera_pending_tiles.empty()?nullptr:owner.camera_pending_tiles.data();
                owner.camera_pending_frame.world_topology=owner.camera_pending_topology.empty()?nullptr:owner.camera_pending_topology.data();
                owner.camera_pending=true;
            }
        }
        ~ForegroundCameraPause() {
            if(!lock.owns_lock())lock.lock();
            owner.camera_paused=false;
            owner.foreground_pending.store(owner.camera_pending,std::memory_order_relaxed);
            owner.wake.notify_one();
        }
    };

    int submit_locked(std::unique_lock<std::mutex> & lock, Command command) {
        if(command==Command::configure_pack || command==Command::configure_definitions || command==Command::reset){
            if(!gpu_presenter.caller_thread())return C3X_RENDERER_RESULT_BAD_ARGUMENT;gpu_presenter.release_native();native_screen_active=false;
        }
        if(command==Command::configure_pack || command==Command::configure_definitions || command==Command::reset){unit_pixels_queue.clear();unit_instances.clear();unit_gpu_preparation=false;}
        // Fresh demand supersedes unstarted prospective snapshots. The caller
        // registers the updated family after composition; completed views retain
        // their independent content proofs and are not discarded here.
        if(command==Command::render){prospective_views.clear();pending_refresh.reset();}
        if(command!=Command::unit)foreground_pending.store(true, std::memory_order_relaxed);
        job_command = command;
        has_job = true;
        std::uint64_t sequence = ++latest_job_sequence;
        wake.notify_one();
        completed.wait(lock, [this, sequence] {
            return completed_job_sequence == sequence;
        });
        return last_job_result;
    }

    char const * optional_path(bool present, std::string const & path) const {
        return present ? path.c_str() : nullptr;
    }

    std::size_t ahead_bytes() const {
        std::size_t bytes=ahead_tiles.capacity()*sizeof(c3x_renderer_tile_v1)+ahead_topology.capacity()*4;
        for(auto const& ready:ahead_ready)bytes+=ready.bytes();
        return bytes;
    }

    void clear_ahead() {
        area_pending=false;ahead_prospective=false;ahead_view_superseded=false;
        for(auto& ready:ahead_ready)if(ready.output.bgra_pixels){++ahead_discarded;ready.clear();}
        std::vector<c3x_renderer_tile_v1>().swap(ahead_tiles);
        std::vector<c3x_renderer_u32>().swap(ahead_topology);
        ahead_frame={};ahead_identity={};ahead_requested=ahead_next=-1;
    }

    void pause_ahead_locked(std::unique_lock<std::mutex>& lock,bool preserve) {
        if(!preserve){stop_ahead_locked(lock);return;}
        ahead_cancelled.store(true,std::memory_order_relaxed);
        completed.wait(lock,[this]{return !ahead_active;});
        ahead_cancelled.store(false,std::memory_order_relaxed);
    }

    void stop_ahead_locked(std::unique_lock<std::mutex>& lock) {
        ahead_cancelled.store(true,std::memory_order_relaxed);ahead_requested=-1;
        completed.wait(lock,[this]{return !ahead_active;});
        clear_ahead();
    }

    bool consume_ahead_locked(std::unique_lock<std::mutex>& lock,
            c3x_renderer_frame_v1 const& frame,c3x_renderer_camera_identity_v1 const& identity,
            c3x_renderer_output_v1& output,bool allow_wait=true) {
        if(!ahead_enabled || !ahead_frame.api_version || ahead_requested<0 ||
           std::memcmp(&ahead_identity,&identity,sizeof(identity)) || !same_ambient_view(frame,ahead_frame))return false;
        auto clock=RendererState::resource_clock(frame);
        // An unchanged authoritative request can reuse its owned front while
        // future work runs; only a later consumed time advances the horizon.
        if(completed_result==C3X_RENDERER_RESULT_OK && completed_resource_clock==clock &&
           !std::memcmp(&completed_identity,&identity,sizeof(identity))) {
            return_current_bitmap(frame,output,"ahead-current");return true;
        }
        LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
        bool joined=ahead_active && ahead_active_clock==clock;
        if(joined && !allow_wait)return false;
        if(joined)completed.wait(lock,[this]{return !ahead_active;});
        for(auto& ready:ahead_ready)if(ready.output.bgra_pixels && RendererState::resource_clock(ready.frame)==clock) {
            publication.swap(ready);ready.clear();
            completed_output=publication.output;completed_identity=publication.identity;
            completed_phase_x=publication.phase_x;completed_phase_y=publication.phase_y;
            completed_scene_signature=publication.scene_signature;completed_resource_clock=clock;
            completed_result=C3X_RENDERER_RESULT_OK; ++ahead_hits;
            ahead_requested=clock;
            for(auto& old:ahead_ready)if(old.output.bgra_pixels && RendererState::resource_clock(old.frame)<clock){
                ++ahead_discarded;old.clear();
            }
            return_current_bitmap(frame,output,"ahead-ready");
            QueryPerformanceCounter(&end);
            char detail[256];sprintf_s(detail,"clock=%lld joined=%u wait_ms=%.3f hits=%llu built=%llu discarded=%llu bytes=%zu peak_bytes=%zu",
                clock,unsigned(joined),renderer_state.trace.milliseconds(end.QuadPart-begin.QuadPart),
                ahead_hits,ahead_built,ahead_discarded,ahead_bytes(),ahead_peak_bytes);
            renderer_state.trace.write("ahead-consumed",detail,true);
            wake.notify_one();return true;
        }
        return false;
    }

    void start_ahead() {
        clear_ahead();
        if(native_presentation && nearby_available)return;
        if(!ahead_enabled || !job_stable_view || ambient_async_enabled || !renderer_state.can_prepare_ambient() ||
           completed_result!=C3X_RENDERER_RESULT_OK || !publication.output.bgra_pixels ||
           completed_output.bgra_pixels!=publication.output.bgra_pixels ||
           !job_frame.api_version || job_frame.presentation_frequency<=0)return;
        auto payload=(std::uint64_t(job_frame.target_width)*job_frame.target_height+std::uint64_t(job_frame.tile_count)*2)*4+
            std::uint64_t(job_frame.tile_count)*sizeof(c3x_renderer_tile_v1)+sizeof(PublishedMapFrame);
        auto snapshot=std::uint64_t(job_frame.tile_count)*sizeof(c3x_renderer_tile_v1)+std::uint64_t(job_frame.world_topology_count)*4;
        if(payload+snapshot+sizeof(ahead_ready)>ahead_budget)return;
        try {
            ahead_tiles=job_tiles;ahead_topology=job_world_topology;
            if(ahead_bytes()+payload>ahead_budget){clear_ahead();return;}
            ahead_frame=job_frame;ahead_identity=job_camera_identity;
            ahead_frame.tiles=ahead_tiles.empty()?nullptr:ahead_tiles.data();
            ahead_frame.world_topology=ahead_topology.empty()?nullptr:ahead_topology.data();
            ahead_requested=RendererState::resource_clock(job_frame);
            if(ahead_requested>INT64_MAX-3){clear_ahead();return;}
            ahead_next=ahead_requested+1;
            ahead_cancelled.store(false,std::memory_order_relaxed);
        }catch(...){clear_ahead();}
    }

    bool ahead_pending() const {
        if(pending_refresh || !prospective_views.empty())return true;
        if(area_pending)return !ahead_cancelled.load(std::memory_order_relaxed);
        if(ahead_requested<0 || ahead_next<0 || ahead_next>ahead_requested+2 ||
           ahead_cancelled.load(std::memory_order_relaxed))return false;
        auto payload=(std::uint64_t(ahead_frame.target_width)*ahead_frame.target_height+std::uint64_t(ahead_frame.tile_count)*2)*4+
            std::uint64_t(ahead_frame.tile_count)*sizeof(c3x_renderer_tile_v1)+sizeof(PublishedMapFrame);
        if(ahead_bytes()+payload>ahead_budget)return false;
        for(auto const& ready:ahead_ready)if(!ready.output.bgra_pixels)return true;
        return false;
    }

    // One GPU owner executes the existing full-detail path ahead of demand.
    // Only publication adoption runs on the caller. No native pointer, callback,
    // sleep-driven animation clock or second D3D context crosses this boundary.
    struct GpuOutputMode {
        RendererState& state;bool previous;
        GpuOutputMode(RendererState& s,bool gpu):state(s),previous(s.gpu_output_mode){
            state.gpu_output_mode=gpu;if(gpu){state.gpu_map_valid=false;state.cpu_output_stale=true;}
        }
        ~GpuOutputMode(){state.gpu_output_mode=previous;}
    };
    std::shared_ptr<c3x_renderer::UnitSceneProvenance> visual_scene;
    c3x_renderer::UnitSceneProvenance publication_scene(){
        c3x_renderer::UnitSceneProvenance proof;
        auto source=std::static_pointer_cast<c3x_renderer::UnitSceneSource>(gpu_publication.resident.scene);
        if(source)proof.parts.push_back({{0,0,gpu_metadata.width,gpu_metadata.height},std::move(source),gpu_publication.source_x,gpu_publication.source_y});
        return proof;
    }
    c3x_gpu_images::RetainedComposition::Sample retain_visual_map(c3x_renderer_frame_v1 input){
        visual_scene=std::make_shared<c3x_renderer::UnitSceneProvenance>(publication_scene());
        using Texture=c3x_gpu_images::RetainedComposition::Texture;
        // A native prepare is not a displayed-front replacement. Each retained
        // source owns its immutable capture until the compositor releases it.
        // Only renderer reset/configuration invalidates every source at once.
        // Native unit demand does not make the terrain animate. Preserve the
        // existing map sample rate; the visual presenter has its own cadence.
        if(gpu_metadata.visible_animation_count<=job_frame.visible_animation_count)return {};
        int x=gpu_publication.source_x,y=gpu_publication.source_y,w=gpu_metadata.width,h=gpu_metadata.height;
        if(input.target_width!=gpu_publication.resident.width||input.target_height!=gpu_publication.resident.height){input=job_frame;x=y=0;}
        auto capture=dynamic_inputs.capture(input,job_camera_identity);
        auto selected=dynamic_inputs.capture(job_frame,job_camera_identity);
        char detail[192];sprintf_s(detail,"bytes=%zu peak=%zu captured=%llu rejected=%llu unit_records=%zu",
            dynamic_inputs.bytes(),dynamic_inputs.peak,dynamic_inputs.captures,dynamic_inputs.rejected,unit_instances.size());
        renderer_state.trace.write("dynamic-inputs",detail,true);
        if(!capture || !selected)return {}; // Keep the coherent retained image; never sample partial inputs.
        auto* session=renderer_state.gpu_composition.get();
        Texture initial=session->snapshot_bgra(static_cast<ID3D11Texture2D*>(gpu_publication.resident.texture.get()),
            gpu_publication.source_x,gpu_publication.source_y,w,h);
        auto origin=visual_ticks,clock=RendererState::resource_clock(gpu_publication.frame);
        return [this,capture,selected,origin,clock,x,y,w,h,scene=visual_scene,last=std::move(initial)](long long ticks,long long frequency)mutable -> Texture{
            c3x_renderer_frame_v1 frame={},view={};
            if(!capture->valid() || !selected->sample(ticks,frequency,origin,view))return last;
            frame=capture->frame();
            auto next=RendererState::resource_clock(view);if(next==clock)return last;
            frame.presentation_time_ticks=view.presentation_time_ticks;frame.presentation_frequency=view.presentation_frequency;
            c3x_renderer_output_v1 out={C3X_RENDERER_API_VERSION,sizeof(out)};GpuOutputMode mode(renderer_state,true);
            if(!renderer_state.render(frame,out,-1,nullptr,0,nullptr,0,&view)||!renderer_state.gpu_map_valid||renderer_state.frame_output_readbacks)
                throw std::runtime_error("retained map sample failed");
            ++visual_map_samples;last=renderer_state.gpu_composition->snapshot_bgra(renderer_state.gpu_map_texture,x,y,w,h);
            scene->parts.clear();if(auto source=renderer_state.unit_scene_source())scene->parts.push_back({{0,0,w,h},std::move(source),x,y});
            clock=next;return last;
        };
    }
    c3x_gpu_images::RetainedComposition::Direct unit_scene_operation(){
        using Direct=c3x_gpu_images::RetainedComposition::Direct;
        struct Input {c3x_renderer_unit_v1 draw{},identity{};unsigned predict=0;bool animated=false;std::uint64_t revision=1;
            std::shared_ptr<c3x_renderer::UnitSceneSource> source;std::shared_ptr<c3x_renderer::UnitSceneRegion> region;
            c3x_gpu_images::Rect area{};};
        auto state=std::make_shared<Input>();state->draw=job_unit;state->identity=job_unit;
        state->identity.presentation_time_ticks=state->identity.presentation_frequency=0;state->predict=job_unit_predict;
        auto selection=job_unit_selection;Direct operation;
        operation.animated=state->animated=unit_instances.animated(selection,renderer_state.unit_bodies.units);
        operation.revision=[this,state,selection](long long ticks,long long frequency){
            auto const& catalog=renderer_state.unit_bodies.units;auto definition=unit_instances.definition(selection,catalog);
            if(!definition||!unit_instances.sample(selection,ticks,frequency,catalog,state->draw,state->predict))
                throw std::runtime_error("direct unit selection retired");
            auto& draw=state->draw;int projection=draw.projection_scale_milli>0?draw.projection_scale_milli:(draw.reduced?500:1000);
            if(!c3x_renderer::expand_unit_canvas(draw.body_x,draw.body_y,draw.sprite_width,draw.sprite_height,projection,definition->minimum_canvas))
                throw std::runtime_error("direct unit bounds failed");
            auto identity=draw;identity.presentation_time_ticks=identity.presentation_frequency=0;
            if(std::memcmp(&identity,&state->identity,sizeof(identity))){state->identity=identity;++state->revision;++visual_pose_changes;}
            return state->revision;
        };
        operation.draw=[this,state](c3x_gpu_images::Compositor& target,c3x_gpu_images::Command const& command){
            auto& body=renderer_state.unit_bodies;c3x_renderer::UnitSceneSample sample;
            if(unit_pixels_enabled && state->animated)unit_pixels_queue.observe(state->draw,true,state->predict!=0,state->predict);
            std::array<int,4> coverage;
            if(!body.scene_coverage(state->draw,[&](auto const& action){return renderer_state.prepare_unit_action(action);},state->predict,coverage))return false;
            auto selected=c3x_renderer::UnitSceneProvenance::intersect(command.clip,{command.area.left+coverage[0],command.area.top+coverage[1],
                command.area.left+coverage[2],command.area.top+coverage[3]});
            auto proof=target.scene(command.detail).select(selected);
            c3x_gpu_images::Rect area={selected.left+proof.x,selected.top+proof.y,selected.right+proof.x,selected.bottom+proof.y};
            if(proof.source!=state->source||std::memcmp(&area,&state->area,sizeof(area))){
                state->region.reset();state->source=proof.source;state->area=area;
            }
            if(state->region&&!state->region->base.color)state->region.reset();
            if(proof.source && !state->region)state->region=proof.source->capture(area);
            if(state->region&&renderer_state.draw_scene_unit(*state->region,state->draw,state->predict,sample,
                selected.left-command.area.left,selected.top-command.area.top)){
                sample.coverage=coverage;++visual_unit_samples;
                return target.submit(&command,1,&sample);
            }
            // Native overlays or unavailable map samples retain the established
            // bounded GPU pose cache. Admission failure must not force skinning
            // and rasterization of an unchanged compatibility pose every frame.
            if(!body.render(renderer_state.device,renderer_state.context,state->draw,
                [&](auto const& action){return renderer_state.prepare_unit_action(action);},nullptr,state->predict,true))return false;
            auto source=target.attach_source(body.resident_pose.texture.Get());if(!source)return false;
            auto op=command;op.source=source;++visual_unit_samples;
            bool ok=target.submit(&op,1,nullptr,&coverage);target.destroy(source);return ok;
        };
        return operation;
    }
    c3x_gpu_images::RetainedComposition::Sample retain_visual_unit(c3x_gpu_images::RetainedComposition::Texture initial){
        auto selection=job_unit_selection;
        if(!unit_instances.animated(selection,renderer_state.unit_bodies.units))return {};
        return [this,selection,last=std::move(initial)](long long ticks,long long frequency)mutable{
            auto const& catalog=renderer_state.unit_bodies.units;c3x_renderer_unit_v1 draw={};unsigned predict=0;
            auto definition=unit_instances.definition(selection,catalog);
            if(!definition||!unit_instances.sample(selection,ticks,frequency,catalog,draw,predict))return last;
            int projection=draw.projection_scale_milli>0?draw.projection_scale_milli:(draw.reduced?500:1000);
            if(!c3x_renderer::expand_unit_canvas(draw.body_x,draw.body_y,draw.sprite_width,draw.sprite_height,projection,definition->minimum_canvas))
                throw std::runtime_error("retained unit bounds failed");
            // Independent frames also refresh the bounded existing preparation
            // queue; it must not run out of forecasts after native draws stop.
            if(unit_pixels_enabled)unit_pixels_queue.observe(draw,true,predict!=0,predict);
            auto& body=renderer_state.unit_bodies;
            if(!body.render(renderer_state.device,renderer_state.context,draw,[&](auto const& action){return renderer_state.prepare_unit_action(action);},nullptr,predict,true))
                throw std::runtime_error("retained unit sample failed");
            ++visual_unit_samples;if(last.Get()!=body.resident_pose.texture.Get())++visual_pose_changes;last=body.resident_pose.texture;return last;
        };
    }

    bool capture_gpu(PublishedMapFrame& target,c3x_renderer_output_v1 const& output,
                     c3x_renderer_frame_v1 const& frame,c3x_renderer_camera_identity_v1 const& identity) {
        if(!renderer_state.gpu_map_valid || renderer_state.frame_output_readbacks)return false;
        D3D11_TEXTURE2D_DESC desc={};renderer_state.gpu_map_texture->GetDesc(&desc);
        if(desc.Width!=unsigned(output.width)||desc.Height!=unsigned(output.height))return false;
        ID3D11Texture2D* copy=nullptr;
        if(FAILED(renderer_state.device->CreateTexture2D(&desc,nullptr,&copy)))return false;
        PublishedMapFrame::Resident storage={std::shared_ptr<void>(copy,[](void* p){static_cast<ID3D11Texture2D*>(p)->Release();}),int(desc.Width),int(desc.Height)};
        renderer_state.context->CopyResource(copy,renderer_state.gpu_map_texture);
        storage.scene=renderer_state.unit_scene_source();
        return target.capture(output,frame.tile_count?frame.tiles[0].anchor_x:0,frame.tile_count?frame.tiles[0].anchor_y:0,&frame,identity,&storage);
    }

    void prepare_ahead(std::unique_lock<std::mutex>& lock) {
        if(!area_pending && (pending_refresh || !prospective_views.empty())){
            bool prospective=!pending_refresh;
            auto snapshot=prospective?std::move(prospective_views.front()):std::move(*pending_refresh);
            if(prospective)prospective_views.pop_front();else pending_refresh.reset();
            clear_ahead();ahead_prospective=prospective;ahead_frame=snapshot.frame;ahead_identity=snapshot.identity;
            ahead_tiles=std::move(snapshot.tiles);ahead_topology=std::move(snapshot.topology);
            ahead_frame.tiles=ahead_tiles.data();ahead_frame.world_topology=ahead_topology.data();
            ahead_selected=false;area_pending=true;ahead_cancelled.store(false,std::memory_order_relaxed);
        }
        if(area_pending) {
            ahead_active=true;area_pending=false;
            lock.unlock();
            LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
            c3x_renderer::PreparedViewArea<PublishedMapFrame> built;
            bool ok=false;bool resident=gpu_presentation;
            GpuOutputMode mode(renderer_state,resident);
            try {
                if(built.prepare(ahead_frame,ahead_identity)) {
                    c3x_renderer_output_v1 output={C3X_RENDERER_API_VERSION,sizeof(output)};
                    D3D11_RECT selected={};
                    if(ahead_selected){
                        selected={built.pad_x,built.pad_y,built.pad_x+built.viewport_width,built.pad_y+built.viewport_height};
                        for(auto const& tile:ahead_selected_tiles)if(tile.tile_flags&C3X_RENDERER_TILE_RENDER){
                            auto found=built.index.find(built.key(tile));
                            if(found==built.index.end()){ahead_selected=false;break;}
                            auto const& old=built.tiles[found->second];int x=old.anchor_x-tile.anchor_x,y=old.anchor_y-tile.anchor_y;
                            selected={x,y,x+built.viewport_width,y+built.viewport_height};break;
                        }
                    }
                    ok=renderer_state.render(built.input,output,-1,&ahead_cancelled,0,nullptr,0,&ahead_frame,ahead_selected?&selected:nullptr);
                    // A successful render owns a complete immutable result even
                    // if a newer caller arrived during its final GPU/readback.
                    // Unfinished renders still yield at existing safe boundaries.
                    PublishedMapFrame gpu_sample;
                    if(ok && resident)ok=capture_gpu(gpu_sample,output,built.input,ahead_identity);
                    if(ok && ahead_selected){
                        lock.lock();
                        // Publication is transactional; never assign a new clock
                        // to the untouched margins of the wider donor.
                        ok=nearby.input.tile_width==built.input.tile_width && nearby.input.tiles &&
                            nearby.input.tiles[0].anchor_x==built.input.tiles[0].anchor_x && nearby.input.tiles[0].anchor_y==built.input.tiles[0].anchor_y &&
                            nearby.project(ahead_selected_frame,ahead_identity,nearby.center,&output,ahead_frame.presentation_time_ticks,resident?&gpu_sample:nullptr);
                        if(ok)trim_views();
                        lock.unlock();
                    }else if(ok)ok=built.finish(output,renderer_state.prepared_view_dependencies(),resident?&gpu_sample:nullptr);
                }
            }catch(std::exception const& error){renderer_state.trace.write("prepared-area-error",error.what(),true);}
            catch(...){} // Cancellation or unavailable optional content keeps the native front.
            if(!ok){renderer_state.discard_scene_view();}
            QueryPerformanceCounter(&end);lock.lock();
            if(ok && !ahead_selected)retain_view(built);
            if(!ok && resident && !ahead_prospective && ahead_cancelled.load(std::memory_order_relaxed))area_pending=true;
            if(!ok && ahead_prospective && !ahead_view_superseded && prospective_view_allowed && ahead_cancelled.load(std::memory_order_relaxed) && prospective_views.size()<2){
                ProspectiveView retry;retry.frame=ahead_frame;retry.identity=ahead_identity;
                retry.tiles=ahead_tiles;retry.topology=ahead_topology;
                prospective_views.push_front(std::move(retry));
            }
            else if(!ok && !ahead_cancelled.load(std::memory_order_relaxed))nearby_available=false;
            char detail[384];std::snprintf(detail,sizeof(detail),"ok=%u cancelled=%u width=%d height=%d bytes=%zu area_owners_cap=%zu selected=%u retained_views=%zu committed_bytes=%zu prepare_ms=%.3f gpu_resident=%u readbacks=%u",
                unsigned(ok),unsigned(ahead_cancelled.load()),ahead_frame.target_width,ahead_frame.target_height,
                nearby.bytes(),retained_view_budget+decltype(nearby)::budget,unsigned(ahead_selected),retained_views.size()+1,retained_view_bytes(),renderer_state.trace.milliseconds(end.QuadPart-begin.QuadPart),unsigned(resident),renderer_state.frame_output_readbacks);
            renderer_state.trace.write("prepared-area",detail,true);
            ahead_active=false;completed.notify_all();return;
        }

        auto quantum=std::max<c3x_renderer_i64>(1,ahead_frame.presentation_frequency/15);
        if(ahead_next>INT64_MAX/quantum){ahead_requested=-1;return;}
        auto frame=ahead_frame;frame.presentation_time_ticks=ahead_next*quantum;
        auto clock=ahead_next;auto identity=ahead_identity;
        ahead_active=true;ahead_active_clock=clock;
        lock.unlock();
        LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
        PublishedMapFrame prepared;
        c3x_renderer_output_v1 output={C3X_RENDERER_API_VERSION,sizeof(output)};
        bool ok=false;
        try {
            ok=renderer_state.render(frame,output,-1,&ahead_cancelled);
            if(ok && !ahead_cancelled.load(std::memory_order_relaxed)) {
                int x=renderer_state.cached_tiles.empty()?0:renderer_state.cached_tiles.front().anchor_x;
                int y=renderer_state.cached_tiles.empty()?0:renderer_state.cached_tiles.front().anchor_y;
                ok=prepared.capture(output,x,y,&frame,identity);
                if(ok)prepared.scene_signature=c3x_renderer::terrain_frame_signature(
                    frame,output.content_revision,output.device_generation).complete;
            }
        }catch(...){ok=false;}
        // An interrupted assembly is not a device failure. Current CPU pixels
        // belong to publication; the ordinary miss will rebuild its draw view.
        if(!ok){renderer_state.discard_scene_view();}
        QueryPerformanceCounter(&end);
        lock.lock();
        bool cancelled=ahead_cancelled.load(std::memory_order_relaxed);
        if(ok && !cancelled && ahead_requested>=0 && ahead_bytes()+prepared.bytes()<=ahead_budget) {
            for(auto& ready:ahead_ready)if(!ready.output.bgra_pixels){ready.swap(prepared);++ahead_built;break;}
            ++ahead_next;ahead_peak_bytes=std::max(ahead_peak_bytes,ahead_bytes());
        }else {
            ++ahead_discarded;
            // A pressure/error result is optional work, never a retry loop.
            ahead_requested=-1;
        }
        char detail[320];sprintf_s(detail,"clock=%lld ok=%u cancelled=%u ms=%.3f begin=%lld end=%lld geometry_built=%u upload_bytes=%llu bytes=%zu cap=%zu built=%llu hits=%llu discarded=%llu",
            clock,unsigned(ok),unsigned(cancelled),renderer_state.trace.milliseconds(end.QuadPart-begin.QuadPart),
            begin.QuadPart,end.QuadPart,output.geometry_tiles_built,static_cast<unsigned long long>(output.geometry_upload_bytes),
            ahead_bytes(),ahead_budget,ahead_built,ahead_hits,ahead_discarded);
        renderer_state.trace.write("ahead-prepared",detail,true);
        ahead_active=false;ahead_active_clock=-1;
        completed.notify_all();
    }

    void prepare_neighborhood() {
        auto signature = renderer_state.requested_signature;
        // Repeated unit/UI redraws must not restart preparation at the first tile.
        if (signature == warm_signature) return;
        int direction_x = 0, direction_y = 0;
        if (!warm_tiles.empty()) {
            auto const & reference = warm_tiles[warm_tiles.size()/2];
            for (auto const & tile : job_tiles)
                if (tile.tile_x == reference.tile_x && tile.tile_y == reference.tile_y) {
                    direction_x = reference.anchor_x - tile.anchor_x;
                    direction_y = reference.anchor_y - tile.anchor_y;
                    break;
                }
        }
        warm_signature = signature;
        renderer_state.begin_pixel_neighborhood(job_frame);
        warm_frame = job_frame;
        warm_tiles.assign(job_tiles.begin(), job_tiles.end());
        warm_frame.tiles = warm_tiles.data();
        warm_world_topology = job_world_topology;
        warm_frame.world_topology = warm_world_topology.empty() ? nullptr : warm_world_topology.data();
        warm_order.clear(); warm_cursor = 0; unavailable_tiles = 0;
        int left = INT_MAX, top = INT_MAX, right = INT_MIN, bottom = INT_MIN;
        for (auto const & tile : warm_tiles)
            if ((tile.tile_flags & C3X_RENDERER_TILE_RENDER) != 0) {
                left = std::min(left, tile.anchor_x); right = std::max(right, tile.anchor_x);
                top = std::min(top, tile.anchor_y); bottom = std::max(bottom, tile.anchor_y);
            }
        if (left > right || top > bottom) return;
        for (unsigned i = 0; i < warm_tiles.size(); ++i)
            if ((warm_tiles[i].tile_flags & C3X_RENDERER_TILE_PREFETCH) != 0 &&
                (warm_tiles[i].tile_flags & C3X_RENDERER_TILE_RENDER) == 0) warm_order.push_back(i);
        auto priority = [&](unsigned index) {
            auto const & tile = warm_tiles[index];
            int x = tile.anchor_x, y = tile.anchor_y;
            int distance_x = std::max({left-x, x-right, 0}) * 2 / warm_frame.tile_width;
            int distance_y = std::max({top-y, y-bottom, 0}) * 2 / warm_frame.tile_height;
            int away = ((direction_x > 0 && x < left) || (direction_x < 0 && x > right) ||
                        (direction_y > 0 && y < top) || (direction_y < 0 && y > bottom)) ? 16 : 0;
            return renderer_state.world_preparation?std::max(distance_x,distance_y)*32+away:
                std::max(distance_x,distance_y)+away;
        };
        std::stable_sort(warm_order.begin(), warm_order.end(), [&](unsigned a, unsigned b) {
            return priority(a) < priority(b);
        });
        unsigned limit=renderer_state.world_preparation?8192u:renderer_state.pickup_profile?512u:384u;
        if (warm_order.size() > limit) warm_order.resize(limit);
        if (warm_order.empty()) renderer_state.start_pixel_preparation();
    }

    // Called under state_mutex after a worker-only preparation operation.
    void publish_preparation_progress() {
        completed_output.prefetch_tiles_pending=static_cast<unsigned>(warm_order.size()-warm_cursor);
        completed_output.prefetch_tiles_built=prepared_tiles;
        completed_output.prefetch_tiles_unavailable=unavailable_tiles;
        completed_output.prefetch_tiles_cancelled=cancelled_tiles;
        completed_output.prefetch_cache_bytes=static_cast<unsigned>(renderer_state.prefetched_geometry_bytes);
        completed_output.prefetch_ticks=preparation_ticks;
        completed_output.prefetch_blocks_pending=renderer_state.pixel_work_pending();
        completed_output.prefetch_blocks_built=renderer_state.prepared_blocks;
        completed_output.pixel_block_cache_bytes=static_cast<unsigned>(renderer_state.pixel_blocks.bytes);
        completed_output.geometry_cache_bytes=static_cast<unsigned>(renderer_state.tile_geometry_cache_bytes);
    }

    void run() {
        std::unique_lock<std::mutex> lock(state_mutex);
        for (;;) {
            // The next displayed map bucket is due before optional future unit
            // poses; the second ambient bucket can share the remaining window.
            if(!has_job && !stop_requested && !camera_paused && !camera_pending && unit_preparation_pending() &&
               !(ahead_pending() && ahead_next<=ahead_requested+1) &&
               (unit_pixels_turn || !ahead_pending())) {
                std::array<c3x_renderer_unit_v1,2> requests{};
                unsigned count=unit_gpu_preparation?
                    unit_pixels_queue.take_ready(requests.data(),2,[&](auto const& request){return renderer_state.unit_bodies.resident_preparation_ready(request);}):
                    unit_pixels_queue.take(requests.data(),2);
                if(!count){
                    unit_content_examined=unit_content_revision;unit_offers_examined=unit_pixels_queue.offered;
                    continue; // Sleep until CPU readiness or a new caller observation.
                }
                unit_pixels_active=true;unit_pixels_turn=false;bool resident_units=unit_gpu_preparation;
                foreground_pending.store(false,std::memory_order_relaxed);
                lock.unlock();LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
                unsigned built=0;
                try {
                    auto prepare=[&](auto const& action){return renderer_state.prepare_unit_action(action);};
                    built=resident_units?renderer_state.unit_bodies.prepare_resident(renderer_state.device,renderer_state.context,requests.data(),count,prepare,foreground_pending):
                        renderer_state.unit_bodies.prepare_pixels(renderer_state.device,renderer_state.context,requests.data(),count,prepare,foreground_pending);
                }
                catch(...) {} // Optional work never suppresses a demanded native body.
                QueryPerformanceCounter(&end);lock.lock();
                unit_pixels_active=false;unit_pixels_built+=built;if(built)++unit_pixels_batches;
                char detail[256];std::snprintf(detail,sizeof(detail),"built=%u total=%llu batches=%llu hits=%llu queued=%u ms=%.3f gpu_resident=%u staging_cap=%u",
                    built,static_cast<unsigned long long>(unit_pixels_built),static_cast<unsigned long long>(unit_pixels_batches),
                    static_cast<unsigned long long>(unit_pixels_hits),unsigned(!unit_pixels_queue.empty()),
                    renderer_state.trace.milliseconds(end.QuadPart-begin.QuadPart),unsigned(resident_units),resident_units?0u:8388608u);
                renderer_state.trace.write("unit-pixels-prepared",detail,true);
                completed.notify_all();continue;
            }
            if(!has_job && !stop_requested && camera_pending && !camera_paused) {
                unit_pixels_turn=true;
                auto const ticket=camera_ticket;
                job_camera_ticket=ticket;
                job_frame=camera_pending_frame;
                job_camera_identity=camera_pending_identity;
                job_tiles.swap(camera_pending_tiles);job_world_topology.swap(camera_pending_topology);
                camera_pending_tiles.clear();camera_pending_topology.clear();
                job_frame.tiles=job_tiles.empty()?nullptr:job_tiles.data();
                job_frame.world_topology=job_world_topology.empty()?nullptr:job_world_topology.data();
                PublishedMapFrame prepared_camera;
                if(compatible_ahead(job_frame,job_camera_identity)) {
                    auto clock=RendererState::resource_clock(job_frame);
                    for(auto& ready:ahead_ready)if(ready.output.bgra_pixels && RendererState::resource_clock(ready.frame)==clock) {
                        prepared_camera.swap(ready);++ahead_hits;ahead_requested=clock;break;
                    }
                }
                camera_pending=false;camera_active=true;
                camera_cancelled.store(false,std::memory_order_relaxed);
                warm_order.clear();warm_cursor=0;warm_signature=0;
                renderer_state.cancel_pixel_preparation();
                lock.unlock();
                c3x_renderer_output_v1 output={C3X_RENDERER_API_VERSION,sizeof(c3x_renderer_output_v1)};
                int result=C3X_RENDERER_RESULT_ERROR;
                if(camera_preview_enabled) {
                    PublishedMapFrame preview;
                    bool available=false;
                    LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
                    try {available=CameraTerrainPreview{}.render(job_frame,renderer_state.terrain_textures,preview,camera_cancelled,job_camera_identity);}
                    catch(...){available=false;}
                    QueryPerformanceCounter(&end);
                    lock.lock();
                    if(available && ticket==camera_ticket && camera_result==C3X_RENDERER_RESULT_PENDING &&
                       !camera_cancelled.load(std::memory_order_relaxed)) {
                        camera_ready.swap(preview);camera_ready_result=C3X_RENDERER_RESULT_PREVIEW;
                        char detail[128];std::snprintf(detail,sizeof(detail),"ticket=%lld ms=%.3f terrain_only=1",
                            static_cast<long long>(ticket),renderer_state.trace.milliseconds(end.QuadPart-begin.QuadPart));
                        renderer_state.trace.write("camera-preview",detail,true);
                    }
                    lock.unlock();
                }
                try {
                    bool ok=false;
                    if(prepared_camera.output.bgra_pixels) {
                        // The existing ambient proof certifies identical full-detail
                        // output throughout this profile's quantized animation bucket.
                        output=prepared_camera.output;
                        output.clip_left=job_frame.clip_left;output.clip_top=job_frame.clip_top;
                        output.clip_right=job_frame.clip_right;output.clip_bottom=job_frame.clip_bottom;
                        output.geometry_tiles_built=output.geometry_tiles_reused=output.geometry_tiles_evicted=0;
                        output.geometry_upload_bytes=0;
                        output.geometry_ticks=output.draw_ticks=output.readback_ticks=output.renderer_cpu_ticks=0;
                        output.raster_reused_pixels=output.raster_draw_pixels=output.raster_cached_pixels=0;
                        ok=true;
                    } else ok=renderer_state.render(job_frame,output,-1,&camera_cancelled);
                    if(camera_cancelled.load(std::memory_order_relaxed)) {
                        // Completed tile entries and the last committed CPU
                        // bitmap remain individually validated. Render marks
                        // the bitmap invalid before any partial pixel mutation.
                        // Only interrupted draw/animation assemblies are lost.
                        renderer_state.discard_scene_view();
                        result=C3X_RENDERER_RESULT_SUPERSEDED;
                    }else if(ok)result=C3X_RENDERER_RESULT_OK;
                    else {
                        renderer_state.reset();
                        result=C3X_RENDERER_RESULT_DEVICE_ERROR;
                    }
                }catch(c3x_renderer::render_core::CliffPreparationCancelled const&) {
                    // Recursive placement unwinds on ordinary supersession.
                    // Keep resident assets and validated world content, exactly
                    // as for render's non-exception cancellation return above.
                    renderer_state.discard_scene_view();
                    renderer_state.trace.write("camera-cancelled","phase=cliff-placement",true);
                    result=C3X_RENDERER_RESULT_SUPERSEDED;
                }catch(std::exception const& error) {
                    renderer_state.trace.write("camera-error",error.what(),true);
                    renderer_state.reset();
                    result=C3X_RENDERER_RESULT_ERROR;
                }catch(...) {
                    renderer_state.trace.write("camera-error","unknown exception",true);
                    renderer_state.reset();
                    result=C3X_RENDERER_RESULT_ERROR;
                }
                // Copy the immutable completed payload while the worker still
                // owns renderer scratch, without excluding begin/poll. Native
                // takeover waits for camera_active; newer input may supersede
                // this result during the copy and is checked again below.
                PublishedMapFrame finished_frame;
                if(result==C3X_RENDERER_RESULT_OK && !camera_cancelled.load(std::memory_order_relaxed)) {
                    int x=renderer_state.cached_tiles.empty()?0:renderer_state.cached_tiles.front().anchor_x;
                    int y=renderer_state.cached_tiles.empty()?0:renderer_state.cached_tiles.front().anchor_y;
                    if(!finished_frame.capture(output,x,y,&job_frame,job_camera_identity))result=C3X_RENDERER_RESULT_ERROR;
                    else finished_frame.scene_signature=c3x_renderer::terrain_frame_signature(
                        job_frame,output.content_revision,output.device_generation).complete;
                }
                lock.lock();
                if(ticket==camera_ticket && camera_result==C3X_RENDERER_RESULT_PENDING &&
                   !camera_cancelled.load(std::memory_order_relaxed)) {
                    if(result==C3X_RENDERER_RESULT_OK) {
                        camera_ready.swap(finished_frame);camera_ready_result=C3X_RENDERER_RESULT_OK;
                        try {prepare_neighborhood();}
                        catch (...) {warm_order.clear();warm_cursor=0;warm_signature=0;warm_tiles.clear();}
                    }
                    camera_result=result;
                }
                char detail[128];std::snprintf(detail,sizeof(detail),"ticket=%lld current=%lld result=%d",
                    static_cast<long long>(ticket),static_cast<long long>(camera_ticket),result);
                renderer_state.trace.write("camera-complete",detail,true);
                if(prepared_camera.output.bgra_pixels) {
                    char reuse[256];sprintf_s(reuse,"clock=%lld joined=0 wait_ms=0.000 hits=%llu built=%llu discarded=%llu bytes=%zu peak_bytes=%zu camera=1",
                        RendererState::resource_clock(job_frame),ahead_hits,ahead_built,ahead_discarded,ahead_bytes(),ahead_peak_bytes);
                    renderer_state.trace.write("ahead-consumed",reuse,true);
                }
                camera_active=false;
                snapshot_memory("camera-complete");
                foreground_pending.store(camera_pending,std::memory_order_relaxed);
                completed.notify_all();
                // Reclaim an obsolete result/preview outside the queue lock too.
                lock.unlock();finished_frame.clear();prepared_camera.clear();lock.lock();
                continue;
            }
            if(!has_job && !stop_requested && !camera_paused && !camera_pending && ahead_pending()) {
                // Allow bursty foreground callers to take priority before a new
                // non-preemptible GPU submission. Demand wakes this wait early.
                wake.wait_for(lock,std::chrono::milliseconds(2),[this]{return has_job || camera_pending || stop_requested;});
                if(!has_job && !camera_pending && !stop_requested){unit_pixels_turn=true;prepare_ahead(lock);}
                continue;
            }
            if(!has_job && !stop_requested && !camera_paused && !camera_pending && renderer_state.scene_guard_pending()) {
                if(wake.wait_for(lock,std::chrono::milliseconds(2),[this]{return has_job || camera_pending || stop_requested;}))continue;
                lock.unlock();
                try{renderer_state.prepare_scene_guard(foreground_pending);}catch(...){renderer_state.scene_guard_failed=true;}
                lock.lock();continue;
            }
            if (!has_job && !stop_requested && !camera_paused && warm_cursor < warm_order.size()) {
                // Yield between individual tiles. A foreground request wakes
                // this delay and cancels CPU construction before GPU upload.
                if (wake.wait_for(lock, std::chrono::milliseconds(2), [this] {
                    return has_job || camera_pending || stop_requested;
                })) continue;
                unsigned index = warm_order[warm_cursor];
                unsigned batch=renderer_state.world_preparation?unsigned(std::min<std::size_t>(4,warm_order.size()-warm_cursor)):1;
                lock.unlock();
                LARGE_INTEGER begin = {}, end = {};
                QueryPerformanceCounter(&begin);
                c3x_renderer_output_v1 unused = {};
                bool ok = false;
                try {
                    ok = renderer_state.render(warm_frame, unused, static_cast<int>(index), &foreground_pending, warm_signature,
                        batch>1?warm_order.data()+warm_cursor:nullptr,batch);
                } catch (...) {
                    // Optional idle work cannot terminate Civ III on scratch
                    // allocation failure. Published frame state was never changed.
                    ok = false;
                }
                QueryPerformanceCounter(&end);
                lock.lock();
                preparation_ticks += end.QuadPart - begin.QuadPart;
                if (ok) {
                    prepared_tiles += renderer_state.frame_tiles_built;
                    try {
                    if (!renderer_state.pixel_neighborhood.empty() &&
                        (renderer_state.prepared_footprint.bounds.left < renderer_state.prepared_footprint.bounds.right)) {
                        auto const & footprint = renderer_state.prepared_footprint;
                        auto found = std::find_if(renderer_state.pixel_neighborhood.begin(),renderer_state.pixel_neighborhood.end(),
                            [&](auto const & old){return old.coordinate==footprint.coordinate;});
                        if(found==renderer_state.pixel_neighborhood.end()) renderer_state.pixel_neighborhood.push_back(footprint);
                        else *found=footprint;
                    }
                    } catch (...) { renderer_state.cancel_pixel_preparation(); }
                }
                if (foreground_pending.load(std::memory_order_relaxed)) {
                    if (!ok) ++cancelled_tiles;
                } else {
                    if (ok) warm_cursor+=batch;
                    else {
                        unavailable_tiles = static_cast<unsigned>(warm_order.size()-warm_cursor);
                        warm_cursor = warm_order.size(); // bounded cache pressure is not a game failure
                    }
                }
                if (warm_cursor == warm_order.size() && unavailable_tiles == 0) {
                    try { renderer_state.start_pixel_preparation(); }
                    catch (...) { renderer_state.cancel_pixel_preparation(); }
                }
                publish_preparation_progress();
                if (warm_cursor == warm_order.size() || !ok) {
                    char detail[240];
                    std::snprintf(detail, sizeof(detail),
                        "result=%s pending=%zu built=%u cancelled=%u prefetch_bytes=%zu cumulative_ms=%.3f",
                        ok ? "ready" : (foreground_pending.load() ? "interrupted" : "capacity"),
                        warm_order.size()-warm_cursor, prepared_tiles, cancelled_tiles,
                        renderer_state.prefetched_geometry_bytes,
                        renderer_state.trace.milliseconds(preparation_ticks));
                    renderer_state.trace.write("prewarm", detail);
                }
                continue;
            }
            if (!has_job && !stop_requested && !camera_paused && renderer_state.pixel_work_pending()) {
                if (wake.wait_for(lock,std::chrono::milliseconds(2),[this]{return has_job || camera_pending || stop_requested;})) continue;
                lock.unlock();
                bool ok=false;
                try { ok=renderer_state.prepare_pixel_block(foreground_pending); } catch (...) { ok=false; }
                lock.lock();
                if (!ok) renderer_state.cancel_pixel_preparation();
                publish_preparation_progress();
                if (!renderer_state.pixel_work_pending()) {
                    char detail[160];
                    std::snprintf(detail,sizeof(detail),"result=%s built=%u bytes=%zu",ok?"ready":"unavailable",
                        renderer_state.prepared_blocks,renderer_state.pixel_blocks.bytes);
                    renderer_state.trace.write("pixel-prewarm",detail,true);
                }
                continue;
            }
            wake.wait(lock, [this] { return has_job || (camera_pending && !camera_paused) || (!camera_paused && (ahead_pending() || unit_preparation_pending())) || stop_requested; });
            if(!has_job && !stop_requested && !camera_paused && (camera_pending || ahead_pending() || unit_preparation_pending()))continue;
            if (stop_requested && !has_job)
                break;
            Command command = job_command;
            if(renderer_state.trace.buffered)QueryPerformanceCounter(&job_timing_begin);
            std::uint64_t sequence = latest_job_sequence;
            renderer_state.cache_hits=static_cast<unsigned>(std::min<std::uint64_t>(0xffffffffu,
                std::uint64_t(renderer_state.cache_hits)+fast_cache_hits));fast_cache_hits=0;
            lock.unlock();
            int result = C3X_RENDERER_RESULT_ERROR;
            c3x_renderer_output_v1 output = {};
            c3x_renderer::PreparedViewArea<PublishedMapFrame> rendered_area;
            PublishedMapFrame rendered_crop;
            bool rendered_area_ready=false;
            try {
            // Map and unit jobs borrow the device; only configuration/reset owns
            // native composition lifetimes. An ordinary CPU publication cannot
            // invalidate GPU UI/background handles held by the caller.
            if(command==Command::configure_pack || command==Command::configure_definitions || command==Command::reset){dynamic_inputs.invalidate();renderer_state.gpu_composition.reset();}
            if(command==Command::native_screen){
                // Retain the transfer image with the presenter, not with a map
                // ticket. Native UI-only transfers must not retire prepared maps.
                result=renderer_state.initialize_device() && (!screen_upload || gpu_presenter.upload_screen(renderer_state.context,screen_pixels.data(),screen_width,screen_height,screen_area,screen_format))
                    ?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_ERROR;
            }else if(command==Command::gpu_render){
                gpu_view={sizeof(gpu_view)};gpu_metadata={C3X_RENDERER_API_VERSION,sizeof(gpu_metadata)};
                bool supported=renderer_state.scene_surface_requested&&renderer_state.city_profile&&!renderer_state.reflection.enabled&&
                    c3x_renderer::render_core::scene_surface_extent(job_frame.target_width,job_frame.target_height);
                if(!supported)result=C3X_RENDERER_RESULT_BAD_ARGUMENT;
                else {
                    GpuOutputMode mode(renderer_state,true);
                    bool ready=gpu_reused;
                    if(ready){
                        gpu_metadata=gpu_publication.output;
                        gpu_metadata.geometry_tiles_built=gpu_metadata.geometry_tiles_reused=gpu_metadata.geometry_tiles_evicted=0;
                        gpu_metadata.geometry_upload_bytes=0;
                        gpu_metadata.geometry_ticks=gpu_metadata.draw_ticks=gpu_metadata.readback_ticks=gpu_metadata.renderer_cpu_ticks=0;
                        gpu_metadata.frame_invalidation_flags=0;
                    }
                    else {
                        // Demand and preparation use the same full-detail working
                        // extent. A first publication also establishes reusable
                        // coverage, rather than immediately rebuilding it wider.
                        if(rendered_area.prepare(job_frame,job_camera_identity)){
                            PublishedMapFrame finished;
                            if(renderer_state.render(rendered_area.input,gpu_metadata,-1,nullptr,0,nullptr,0,&job_frame) &&
                               capture_gpu(finished,gpu_metadata,rendered_area.input,job_camera_identity) &&
                               rendered_area.finish(gpu_metadata,renderer_state.prepared_view_dependencies(),&finished) &&
                               rendered_area.project(job_frame,job_camera_identity,gpu_publication)){
                                rendered_area_ready=true;ready=true;gpu_metadata=gpu_publication.output;
                            }
                        }
                        if(!ready && renderer_state.render(job_frame,gpu_metadata))ready=capture_gpu(gpu_publication,gpu_metadata,job_frame,job_camera_identity);
                    }
                    if(ready){
                        if(!renderer_state.gpu_composition)renderer_state.gpu_composition=std::make_unique<c3x_gpu_images::Session>(renderer_state.device,renderer_state.context);
                        auto& session=*renderer_state.gpu_composition;
                        auto map_sample=retain_visual_map(rendered_area_ready?rendered_area.input:(gpu_reused&&nearby.map.has_image()?nearby.input:job_frame));
                        auto scene=visual_scene;
                        if(session.publish(static_cast<ID3D11Texture2D*>(gpu_publication.resident.texture.get()),++renderer_state.gpu_serial,
                            gpu_publication.source_x,gpu_publication.source_y,gpu_metadata.width,gpu_metadata.height,std::move(map_sample),*scene,[scene]{return *scene;})){
                            gpu_replacements=gpu_publication.replacements;gpu_fallbacks=gpu_publication.fallback;
                            gpu_metadata.replacement_tile_flags=gpu_replacements.empty()?nullptr:gpu_replacements.data();
                            gpu_metadata.fallback_tile_indices=gpu_fallbacks.empty()?nullptr:gpu_fallbacks.data();
                            gpu_metadata.visible_animation_count=job_frame.visible_animation_count+
                                (gpu_publication.output.visible_animation_count>gpu_publication.frame.visible_animation_count?
                                 gpu_publication.output.visible_animation_count-gpu_publication.frame.visible_animation_count:0);
                            gpu_metadata.request_continuous_redraw=gpu_metadata.visible_animation_count!=0;
                            gpu_metadata.clip_left=job_frame.clip_left;gpu_metadata.clip_top=job_frame.clip_top;
                            gpu_metadata.clip_right=job_frame.clip_right;gpu_metadata.clip_bottom=job_frame.clip_bottom;
                            gpu_view={sizeof(gpu_view),session.current_ticket(),static_cast<c3x_renderer_i64>(session.map_image()),gpu_metadata.width,gpu_metadata.height,gpu_metadata.device_generation,0,gpu_metadata.content_revision,session.session_identity(),unsigned(gpu_reused),gpu_publication.frame.presentation_time_ticks};
                            result=C3X_RENDERER_RESULT_OK;
                        }else result=C3X_RENDERER_RESULT_BAD_ARGUMENT;
                    }
                    char detail[256];std::snprintf(detail,sizeof(detail),"result=%d prepared=%u sample_ticks=%lld requested_ticks=%lld map_readbacks=0 publication_bytes=%zu",
                        result,unsigned(gpu_reused),gpu_publication.frame.presentation_time_ticks,job_frame.presentation_time_ticks,gpu_publication.bytes());
                    renderer_state.trace.write("gpu-map-publication",detail,true);
                }
            }else if(command==Command::gpu_images){
                gpu_result={sizeof(gpu_result)};gpu_readback.clear();
                result=renderer_state.gpu_composition?renderer_state.gpu_composition->execute(gpu_request,gpu_commands,gpu_pixels,gpu_result,gpu_readback):C3X_RENDERER_RESULT_SUPERSEDED;
            }else if(command==Command::gpu_unit){
                result=C3X_RENDERER_RESULT_SUPERSEDED;gpu_unit_output_readbacks=gpu_unit_composition_uploads=0;
                if(renderer_state.gpu_composition&&renderer_state.gpu_composition->current_ticket()==gpu_unit.ticket){
                    auto& body=renderer_state.unit_bodies;auto reads=body.output_readbacks,uploads=renderer_state.gpu_composition->upload_count();
                    if(body.direct_scene){
                        unsigned scale=job_unit.projection_scale_milli?job_unit.projection_scale_milli:(job_unit.reduced?500:1000);
                        result=renderer_state.gpu_composition->draw_unit_scene(gpu_unit,job_unit.sprite_width*scale/1000,job_unit.sprite_height*scale/1000,
                            job_unit.body_x,job_unit.body_y,unit_scene_operation());
                    }else if(body.render(renderer_state.device,renderer_state.context,job_unit,[&](auto const& action){return renderer_state.prepare_unit_action(action);},nullptr,job_unit_predict,true)){
                        auto const& pose=body.resident_pose;result=renderer_state.gpu_composition->compose_resident_unit(gpu_unit,pose.texture.Get(),unsigned(pose.width),unsigned(pose.height),job_unit.body_x,job_unit.body_y,retain_visual_unit(pose.texture));
                    }else result=C3X_RENDERER_RESULT_ERROR;
                    gpu_unit_output_readbacks=body.output_readbacks-reads;gpu_unit_composition_uploads=renderer_state.gpu_composition->upload_count()-uploads;
                    if(body.direct_scene){char detail[512];sprintf_s(detail,"map_draws=%llu region_captures=%llu region_rejections=%llu region_evictions=%llu region_bytes=%zu work_bytes=%zu finished_pose_bytes=%zu compatibility_builds=%llu compatibility_hits=%llu body_readbacks=%llu composition_uploads=%llu",
                        body.map_scene_draws,renderer_state.unit_scene_captures,renderer_state.unit_scene_rejections,renderer_state.unit_scene_evictions,*renderer_state.unit_scene_bytes,renderer_state.unit_scene_work.bytes(),
                        body.resident_pose_bytes,body.resident_pose_builds,body.resident_pose_hits,gpu_unit_output_readbacks,gpu_unit_composition_uploads);
                        renderer_state.trace.write("unit-scene",detail,true);}
                }
            }else if(command==Command::visual_frame){
                int drawn=renderer_state.gpu_composition?renderer_state.gpu_composition->visual_frame(visual_ticks,visual_frequency,
                    gpu_presenter.view(),gpu_presenter.retained(),gpu_presenter.buffer()):0;
                result=drawn==1?C3X_RENDERER_RESULT_OK:drawn==2?C3X_RENDERER_RESULT_PENDING:C3X_RENDERER_RESULT_ERROR;
                if(result==C3X_RENDERER_RESULT_OK)gpu_presenter.gpu_written();
            }else if(command==Command::gpu_present){
                auto const& p=gpu_present;
                if(p.action==2)result=gpu_presenter.preserve_display(renderer_state.context)?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_ERROR;
                else {
                    result=renderer_state.gpu_composition&&renderer_state.gpu_composition->display_to(p.ticket,std::uint64_t(p.image),
                        gpu_presenter.view(),gpu_presenter.retained(),gpu_presenter.buffer(),p.width,p.height,
                        {p.area[0],p.area[1],p.area[2],p.area[3]})?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_BAD_ARGUMENT;
                    if(result==C3X_RENDERER_RESULT_OK)gpu_presenter.gpu_written();
                }
            }else if (command == Command::configure_pack) {
                result = renderer_state.configure_pack(
                    optional_path(job_pack_present, job_pack_path))
                    ? C3X_RENDERER_RESULT_OK : C3X_RENDERER_RESULT_ERROR;
            } else if (command == Command::configure_definitions) {
                result = renderer_state.configure_definitions(
                    optional_path(job_mod_root_present, job_mod_root),
                    optional_path(job_default_path_present, job_default_path),
                    optional_path(job_scenario_path_present, job_scenario_path),
                    optional_path(job_custom_path_present, job_custom_path))
                    ? C3X_RENDERER_RESULT_OK : C3X_RENDERER_RESULT_ERROR;
            } else if (command == Command::render) {
                output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
                // One working extent for demand and preparation. Alternating
                // viewport and area targets destroys the GPU-resident backdrop
                // whenever an animated prepared frame ages out.
                auto render_requested=[&]{
                    char option[8]={};GetEnvironmentVariableA("C3X_RENDERER_PREPARED_VIEW",option,sizeof(option));
                    if(native_presentation && renderer_state.shared_scene_surface && std::strcmp(option,"0")!=0 &&
                       rendered_area.prepare(job_frame,job_camera_identity)) {
                        if(renderer_state.render(rendered_area.input,output,-1,nullptr,0,nullptr,0,&job_frame) &&
                           rendered_area.finish(output,renderer_state.prepared_view_dependencies()) &&
                           rendered_area.project(job_frame,job_camera_identity,rendered_crop)) {
                            output=rendered_crop.output;rendered_area_ready=true;return true;
                        }
                        rendered_area.clear();rendered_crop.clear();
                    }
                    return renderer_state.render(job_frame,output);
                };
                if (render_requested()) {
                    result = C3X_RENDERER_RESULT_OK;
                } else {
                    renderer_state.trace.write("reset", "device and all caches", true);
            renderer_state.reset();
                    if (renderer_state.device_recoveries != 0xffffffffu)
                        ++renderer_state.device_recoveries;
                    result = renderer_state.render(job_frame, output)
                        ? C3X_RENDERER_RESULT_OK : C3X_RENDERER_RESULT_DEVICE_ERROR;
                }
            } else if(command==Command::unit) {
                result=renderer_state.unit_bodies.render(renderer_state.device,renderer_state.context,job_unit,
                    [&](auto const& action){return renderer_state.prepare_unit_action(action);},nullptr,job_unit_predict)
                    ? C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_ERROR;
                if(renderer_state.trace.level) {
                    auto& body=renderer_state.unit_bodies;auto prep=body.pose_preparation_statistics();char detail[512];
                    std::snprintf(detail,sizeof(detail),"payload_ms=%.3f pose_ms=%.3f submission_ms=%.3f readback_ms=%.3f output_ms=%.3f prepared=%u built=%llu consumed=%llu cumulative_cpu_ms=%.3f cumulative_join_ms=%.3f ready_bytes=%zu retained_bytes=%zu active_peak=%u",
                        body.payload_ms,body.pose_ms,body.submission_ms,body.readback_ms,body.output_ms,body.pose_content_hit?1u:0u,
                        static_cast<unsigned long long>(prep.built),static_cast<unsigned long long>(prep.consumed),prep.cpu_ms,prep.wait_ms,prep.bytes,body.pose_retained_bytes(),prep.active_peak);
                    renderer_state.trace.write("unit-stages",detail,true);
                }
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
            } else if(command==Command::benchmark_trim) {
                auto publication_bytes=publication.bytes()+camera_ready.bytes();
                publication.clear();camera_ready.clear();
                renderer_state.trim_to_prepared(benchmark_trim_result,benchmark_reset_mode==0);
                if(benchmark_reset_mode==1) {
                    // Retire all scene-dependent content through its current
                    // owners. Keep immutable assets, shader objects and device.
                    renderer_state.clear_tile_geometry_cache();
                    renderer_state.reset_resource_buffers();
                    renderer_state.reset_waves();
                    renderer_state.world_coast.clear();renderer_state.center_shore_cache.clear();
                    renderer_state.geometry_world_revision=-1;renderer_state.natural.reset_world();
                    renderer_state.source_shadow.clear_cached_pages();
                    benchmark_trim_result.retained_geometry_bytes=0;
                    benchmark_trim_result.retained_natural_bytes=0;
                    benchmark_trim_result.retained_ground_bytes=0;
                    benchmark_trim_result.retained_wave_bytes=0;
                    benchmark_trim_result.retained_other_bytes=0;
                    benchmark_trim_result.retained_geometry_entries=0;
                    benchmark_trim_result.retained_wave_entries=0;
                }
                benchmark_trim_result.cleared_publication_bytes=publication_bytes;
                completed_scene_signature=0;completed_resources=0;completed_resource_clock=-1;
                completed_output={};completed_result=C3X_RENDERER_RESULT_SUPERSEDED;
                completed_phase_x=completed_phase_y=0;
                result=C3X_RENDERER_RESULT_OK;
#endif
            } else if (command == Command::reset) {
                renderer_state.trace.write("reset", "device and all caches", true);
            renderer_state.reset();
                result = C3X_RENDERER_RESULT_OK;
            }
            } catch (...) {
                if(command==Command::unit)renderer_state.unit_bodies.reset_gpu();
                else if(command!=Command::native_screen && command!=Command::visual_frame)renderer_state.reset();
                renderer_state.trace.write("worker-error", "resource allocation or runtime exception", true);
                output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
                result = C3X_RENDERER_RESULT_ERROR;
            }
            if(renderer_state.trace.buffered)QueryPerformanceCounter(&job_timing_rendered);
            lock.lock();
            if (command == Command::render && result == C3X_RENDERER_RESULT_OK) {
                try { prepare_neighborhood(); }
                catch (...) {
                    warm_order.clear(); warm_cursor = 0; warm_signature = 0;
                    warm_tiles.clear(); unavailable_tiles = 1;
                    renderer_state.cancel_pixel_preparation();
                }
                output.prefetch_tiles_pending = static_cast<unsigned>(warm_order.size()-warm_cursor);
                output.prefetch_tiles_built = prepared_tiles;
                output.prefetch_tiles_unavailable = unavailable_tiles;
                output.prefetch_tiles_cancelled = cancelled_tiles;
                output.prefetch_cache_bytes = static_cast<unsigned>(renderer_state.prefetched_geometry_bytes);
                output.prefetch_ticks = preparation_ticks;
                output.prefetch_blocks_pending = renderer_state.pixel_work_pending();
                output.prefetch_blocks_built = renderer_state.prepared_blocks;
                output.pixel_block_cache_bytes = static_cast<unsigned>(renderer_state.pixel_blocks.bytes);
            } else if(command!=Command::unit && command!=Command::native_screen && command!=Command::gpu_images && command!=Command::gpu_unit && command!=Command::gpu_present && command!=Command::visual_frame) {
                warm_order.clear(); warm_cursor = 0; warm_signature = 0;
                warm_tiles.clear();
                renderer_state.cancel_pixel_preparation();
            }
            if(command!=Command::unit && command!=Command::native_screen && command!=Command::gpu_images && command!=Command::gpu_unit && command!=Command::gpu_present && command!=Command::visual_frame) {
            if(command==Command::render && result==C3X_RENDERER_RESULT_OK){
                if(rendered_area_ready){retain_view(rendered_area);nearby_presented=true;}
                completed_phase_x=job_frame.tile_count?job_frame.tiles[0].anchor_x:0;
                completed_phase_y=job_frame.tile_count?job_frame.tiles[0].anchor_y:0;
                bool const preparing_ahead=ahead_enabled && job_stable_view &&
                    renderer_state.can_prepare_ambient() && job_frame.presentation_frequency>0;
                if(isolated_publication || preparing_ahead || rendered_area_ready){
                    auto captured=(ambient_async_enabled || native_presentation)?&job_frame:nullptr;
                    if(publication.capture(output,completed_phase_x,completed_phase_y,captured,job_camera_identity)){
                        publication.scene_signature=c3x_renderer::terrain_frame_signature(
                            job_frame,output.content_revision,output.device_generation).complete;
                        output=publication.output;
                    }
                    else {
                        publication.clear();
                        renderer_state.trace.write("publication-unavailable","synchronous exact output retained",true);
                    }
                }else publication.clear();
            }else {publication.clear();completed_phase_x=completed_phase_y=0;}
            completed_resources = command==Command::render ? renderer_state.ambient_count() : 0;
            completed_resource_clock=command==Command::render ? RendererState::resource_clock(job_frame) : -1;
            completed_output = output;
            completed_identity=command==Command::render && result==C3X_RENDERER_RESULT_OK ?
                job_camera_identity:c3x_renderer_camera_identity_v1{};
            completed_scene_signature=command==Command::render && result==C3X_RENDERER_RESULT_OK ?
                c3x_renderer::terrain_frame_signature(job_frame,output.content_revision,output.device_generation).complete:0;
            completed_result = result;
            }
            if(command==Command::gpu_render && result==C3X_RENDERER_RESULT_OK){
                if(rendered_area_ready){retain_view(rendered_area);nearby_presented=true;}
                nearby_available=renderer_state.shared_scene_surface;
            }else if(command==Command::render && result==C3X_RENDERER_RESULT_OK){
                char area_option[8]={};GetEnvironmentVariableA("C3X_RENDERER_PREPARED_VIEW",area_option,sizeof(area_option));
                nearby_available=renderer_state.shared_scene_surface && std::strcmp(area_option,"0")!=0;
                start_ahead();
            }else if(command!=Command::unit && command!=Command::native_screen && command!=Command::gpu_images && command!=Command::gpu_unit && command!=Command::gpu_present && command!=Command::visual_frame){nearby.clear();retained_views.clear();prospective_views.clear();pending_refresh.reset();nearby_available=false;gpu_publication.clear();gpu_presentation=false;}
            last_job_result=result;
            completed_job_sequence = sequence;
            job_command = Command::none;
            has_job = false;
            foreground_pending.store(false, std::memory_order_relaxed);
            if(renderer_state.trace.buffered)QueryPerformanceCounter(&job_timing_published);
            completed.notify_all();
        }
    }
};

RendererWorker * renderer_worker = nullptr;
void CALLBACK renderer_visual_timer(HWND,UINT,UINT_PTR,DWORD){
    if(renderer_worker)try{renderer_worker->visual_frame(true);}catch(...){OutputDebugStringA("[C3X renderer] visual timer failed\n");}
}
extern "C" __declspec(dllexport) c3x_renderer_i64 c3x_renderer_visual_clock(){return renderer_worker?renderer_worker->visual_clock():0;}
extern "C" __declspec(dllexport) int c3x_renderer_gpu_visual_status(c3x_renderer_visual_status_v1* out){
    if(!out||out->struct_size!=sizeof(*out))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    return renderer_worker?renderer_worker->visual_status(*out):C3X_RENDERER_RESULT_PENDING;
}
extern "C" __declspec(dllexport) int c3x_renderer_gpu_visual_frame(){
    return renderer_worker?renderer_worker->visual_frame():C3X_RENDERER_RESULT_PENDING;
}

RendererWorker & get_renderer_worker() {
    if (renderer_worker == nullptr)
        renderer_worker = new RendererWorker(renderer);
    return *renderer_worker;
}

void destroy_renderer_worker() {
    if (renderer_worker == nullptr) {
        renderer.reset();
        return;
    }
    renderer_worker->reset_and_stop();
    delete renderer_worker;
    renderer_worker = nullptr;
}

bool valid_frame(c3x_renderer_frame_v1 const * frame, c3x_renderer_output_v1 const * output) {
    if (frame == nullptr || output == nullptr)
        return false;
    if (frame->api_version != C3X_RENDERER_API_VERSION || frame->struct_size != sizeof(*frame))
        return false;
    if (output->struct_size != sizeof(*output))
        return false;
    if (frame->target_width <= 0 || frame->target_height <= 0 ||
        frame->target_width > 8192 || frame->target_height > 8192)
        return false;
    if (frame->tile_width <= 0 || frame->tile_height <= 0 || frame->tile_count > 8192u)
        return false;
    if (frame->presentation_time_ticks < 0 || frame->presentation_frequency <= 0)
        return false;
    if (frame->world_width_tiles < 0 || frame->world_height_tiles < 0 ||
        frame->world_width_tiles > 100000 || frame->world_height_tiles > 100000 ||
        (frame->world_wrap_x != 0 && frame->world_width_tiles == 0) ||
        (frame->world_wrap_y != 0 && frame->world_height_tiles == 0))
        return false;
    if (frame->world_topology_count != 0 &&
        (frame->world_topology == nullptr || frame->world_width_tiles <= 0 ||
         frame->world_height_tiles <= 0 || (frame->world_width_tiles & 1) != 0 ||
         (frame->world_wrap_y && (frame->world_height_tiles & 1)) ||
         frame->world_width_tiles > 2048 || frame->world_height_tiles > 2048 ||
         frame->world_topology_count != static_cast<c3x_renderer_u32>(
             frame->world_width_tiles * frame->world_height_tiles / 2)))
        return false;
    if (frame->tile_count != 0 && frame->tiles == nullptr)
        return false;
    if (frame->clip_left < 0 || frame->clip_top < 0 ||
        frame->clip_right > frame->target_width || frame->clip_bottom > frame->target_height ||
        frame->clip_left >= frame->clip_right || frame->clip_top >= frame->clip_bottom)
        return false;
    return true;
}

} // namespace

namespace {
c3x_native_images::CompositionOwner* native_composition=nullptr;
bool drain_native_composition(){
    if(!native_composition)return true;
    try {native_composition->drain();delete native_composition;native_composition=nullptr;return true;}
    catch(std::exception const& e){OutputDebugStringA(e.what());return false;}
}
}

extern "C" __declspec(dllexport) c3x_renderer_u32 c3x_renderer_get_api_version(void) {
    return C3X_RENDERER_API_VERSION;
}

extern "C" __declspec(dllexport) int c3x_renderer_set_pack_path(char const * pack_path) {
    if(!drain_native_composition())return C3X_RENDERER_RESULT_DEVICE_ERROR;
    int result = get_renderer_worker().configure_pack(pack_path);
    if (result != C3X_RENDERER_RESULT_OK)
        destroy_renderer_worker();
    return result;
}

extern "C" __declspec(dllexport) int c3x_renderer_set_definition_paths(
    char const * mod_root, char const * default_path, char const * scenario_path, char const * custom_path) {
    if(!drain_native_composition())return C3X_RENDERER_RESULT_DEVICE_ERROR;
    int result = get_renderer_worker().configure_definitions(
        mod_root, default_path, scenario_path, custom_path);
    if(renderer.trace.level) {
        char effect[16]={};
        GetEnvironmentVariableA("C3X_RENDERER_WAVES",effect,sizeof(effect));
        char detail[384];
        std::snprintf(detail,sizeof(detail),
            "result=%d waves=%u reflections=%u world_regions=%u world_waves=%u world_backdrops=%u receiver_index=%u "
            "viewport_limit=%zu backdrop_limit=%zu unit_pose_limit_mib=%u",
            result,unsigned(std::strcmp(effect,"0")!=0),unsigned(renderer.reflection.enabled),unsigned(renderer.world_regions),
            unsigned(renderer.world_waves),unsigned(renderer.world_backdrops),unsigned(renderer.composition_receiver_index),
            renderer.viewport_cache_budget,renderer.resource_backdrop_cache_budget,
            c3x_renderer::NavigationOptions::unit_pose_mib(GetEnvironmentVariableA));
        renderer.trace.write("usage-settings",detail,true);
    }
    if (result != C3X_RENDERER_RESULT_OK)
        destroy_renderer_worker();
    return result;
}

extern "C" __declspec(dllexport) int c3x_renderer_render(
    c3x_renderer_frame_v1 const * frame, c3x_renderer_output_v1 * output) {
    if (!valid_frame(frame, output))
        return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    auto request=renderer.trace.usage_view(*frame);
    LARGE_INTEGER started={},finished={};
    if(request)QueryPerformanceCounter(&started);
    int result=get_renderer_worker().render(*frame,*output);
    if(request) {
        QueryPerformanceCounter(&finished);
        renderer.trace.usage_result(request,result,finished.QuadPart-started.QuadPart,*output);
    }
    return result;
}

// Optional ordinary-demand entry. Identity does not relax exact captured camera
// or ownership checks and does not arrange another native render call.
extern "C" __declspec(dllexport) int c3x_renderer_render_view(
    c3x_renderer_camera_request_v1 const* request,c3x_renderer_output_v1* output) {
    if(!request || request->version!=C3X_RENDERER_CAMERA_VIEW_VERSION ||
       request->struct_size!=sizeof(*request) || !valid_frame(request->frame,output))
        return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    auto usage=renderer.trace.usage_view(*request->frame);
    LARGE_INTEGER started={},finished={};
    if(usage)QueryPerformanceCounter(&started);
    int result=get_renderer_worker().render(*request->frame,*output,request->identity);
    if(usage){QueryPerformanceCounter(&finished);
        renderer.trace.usage_result(usage,result,finished.QuadPart-started.QuadPart,*output);}
    return result;
}

extern "C" __declspec(dllexport) int c3x_renderer_camera_present_view(
    c3x_renderer_camera_request_v1 const* request,c3x_renderer_camera_view_v1* view) {
    c3x_renderer_output_v1 output={C3X_RENDERER_API_VERSION,sizeof(c3x_renderer_output_v1)};
    if(!request || request->version!=C3X_RENDERER_CAMERA_VIEW_VERSION ||
       request->struct_size!=sizeof(*request) || !view ||
       view->version!=C3X_RENDERER_CAMERA_VIEW_VERSION || view->struct_size!=sizeof(*view) ||
       !valid_frame(request->frame,&output))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    return get_renderer_worker().camera_present_view(*request,*view);
}

extern "C" __declspec(dllexport) int c3x_renderer_camera_begin(
    c3x_renderer_frame_v1 const* frame,c3x_renderer_i64* ticket) {
    c3x_renderer_output_v1 output={C3X_RENDERER_API_VERSION,sizeof(c3x_renderer_output_v1)};
    if(!ticket || !valid_frame(frame,&output))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    return get_renderer_worker().camera_begin(*frame,*ticket);
}

extern "C" __declspec(dllexport) int c3x_renderer_camera_poll(
    c3x_renderer_i64 ticket,c3x_renderer_output_v1* output) {
    if(!output || output->api_version!=C3X_RENDERER_API_VERSION || output->struct_size!=sizeof(*output))
        return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    if(!renderer_worker)return C3X_RENDERER_RESULT_SUPERSEDED;
    return renderer_worker->camera_poll(ticket,*output);
}

extern "C" __declspec(dllexport) int c3x_renderer_camera_cancel(c3x_renderer_i64 ticket) {
    if(!renderer_worker)return C3X_RENDERER_RESULT_SUPERSEDED;
    return renderer_worker->camera_cancel(ticket);
}

extern "C" __declspec(dllexport) int c3x_renderer_camera_begin_view(
    c3x_renderer_camera_request_v1 const* request,c3x_renderer_i64* ticket) {
    c3x_renderer_output_v1 output={C3X_RENDERER_API_VERSION,sizeof(c3x_renderer_output_v1)};
    if(!request || request->version!=C3X_RENDERER_CAMERA_VIEW_VERSION || request->struct_size!=sizeof(*request) ||
       !ticket || !valid_frame(request->frame,&output))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    return get_renderer_worker().camera_begin(*request->frame,*ticket,request->identity);
}

extern "C" __declspec(dllexport) int c3x_renderer_prepare_view(
    c3x_renderer_camera_request_v1 const* request,int query_only) {
    c3x_renderer_output_v1 output={C3X_RENDERER_API_VERSION,sizeof(output)};
    if(!request || request->version!=C3X_RENDERER_CAMERA_VIEW_VERSION || request->struct_size!=sizeof(*request) ||
       !valid_frame(request->frame,&output))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    try{return get_renderer_worker().prepare_view(*request,query_only!=0);}catch(...){return C3X_RENDERER_RESULT_ERROR;}
}

extern "C" __declspec(dllexport) int c3x_renderer_prepare_nearby_view(
    c3x_renderer_camera_request_v1 const* request) {
    c3x_renderer_output_v1 output={C3X_RENDERER_API_VERSION,sizeof(output)};
    if(!request || request->version!=C3X_RENDERER_CAMERA_VIEW_VERSION || request->struct_size!=sizeof(*request) ||
       !valid_frame(request->frame,&output))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    try {return get_renderer_worker().prepare_nearby_view(*request);}
    catch(...) {return C3X_RENDERER_RESULT_ERROR;}
}

extern "C" __declspec(dllexport) int c3x_renderer_camera_poll_view(
    c3x_renderer_i64 ticket,c3x_renderer_camera_view_v1* view) {
    if(!view || view->version!=C3X_RENDERER_CAMERA_VIEW_VERSION || view->struct_size!=sizeof(*view))
        return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    if(!renderer_worker)return C3X_RENDERER_RESULT_SUPERSEDED;
    return renderer_worker->camera_poll_view(ticket,*view);
}

extern "C" __declspec(dllexport) int c3x_renderer_blit(
    c3x_renderer_output_v1 const * output, void * destination_hdc) {
    if (output == nullptr || destination_hdc == nullptr ||
        output->api_version != C3X_RENDERER_API_VERSION || output->struct_size != sizeof(*output) ||
        output->width <= 0 || output->height <= 0 || output->stride_bytes != output->width * 4 ||
        output->clip_left < 0 || output->clip_top < 0 ||
        output->clip_right > output->width || output->clip_bottom > output->height ||
        output->clip_left >= output->clip_right || output->clip_top >= output->clip_bottom ||
        output->bgra_pixels == nullptr)
        return C3X_RENDERER_RESULT_BAD_ARGUMENT;

    if (renderer_worker == nullptr)
        return C3X_RENDERER_RESULT_ERROR;
    return renderer_worker->blit(*output, static_cast<HDC>(destination_hdc));
}

extern "C" __declspec(dllexport) void c3x_renderer_reset(void) {
    if(drain_native_composition())destroy_renderer_worker();
}

#ifdef C3X_RENDERER_BENCHMARK_ORACLE
extern "C" __declspec(dllexport) int c3x_renderer_benchmark_session_reset_v1(
    std::uint32_t mode,c3x_renderer_benchmark_oracle_trim_v1* result) {
    if(!renderer_worker || !result || (mode!=1 && mode!=2) ||
       result->version!=C3X_RENDERER_BENCHMARK_ORACLE_VERSION || result->struct_size!=sizeof(*result))
        return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    return renderer_worker->benchmark_trim_to_prepared(*result,mode);
}

extern "C" __declspec(dllexport) int c3x_renderer_benchmark_trim_to_prepared_v1(
    c3x_renderer_benchmark_oracle_trim_v1 * result) {
    if(!result || result->version!=C3X_RENDERER_BENCHMARK_ORACLE_VERSION ||
       result->struct_size!=sizeof(*result) || !renderer_worker)
        return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    return renderer_worker->benchmark_trim_to_prepared(*result);
}
#endif

extern "C" __declspec(dllexport) int c3x_renderer_unit_draw(
    c3x_renderer_unit_v1 const* unit,void* destination_hdc) {
    if(!unit || unit->struct_size!=sizeof(*unit) || unit->unit_key[63]!=0 || !destination_hdc)
        return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    if(!renderer_worker)return C3X_RENDERER_RESULT_ERROR;
    return renderer_worker->draw_unit(*unit,static_cast<HDC>(destination_hdc));
}

extern "C" __declspec(dllexport) int c3x_renderer_unit_draw_background(
    c3x_renderer_unit_v1 const* unit,void* destination_hdc,void* background_hdc) {
    if(!unit || unit->struct_size!=sizeof(*unit) || unit->unit_key[63]!=0 || !destination_hdc || !background_hdc)
        return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    if(!renderer_worker)return C3X_RENDERER_RESULT_ERROR;
    return renderer_worker->draw_unit(*unit,static_cast<HDC>(destination_hdc),static_cast<HDC>(background_hdc));
}

// Optional extension: success publishes the complete drawn rectangle. Legacy
// callers retain their original strict sprite bounds and separate fallback.
extern "C" __declspec(dllexport) int c3x_renderer_unit_draw_expanded(
    c3x_renderer_unit_v1 const* unit,void* destination_hdc,void* background_hdc,int* bounds) {
    if(!unit || unit->struct_size!=sizeof(*unit) || unit->unit_key[63]!=0 || !destination_hdc || !background_hdc || !bounds)
        return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    if(!renderer_worker)return C3X_RENDERER_RESULT_ERROR;
    return renderer_worker->draw_unit(*unit,static_cast<HDC>(destination_hdc),static_cast<HDC>(background_hdc),bounds);
}

extern "C" __declspec(dllexport) int c3x_renderer_unit_draw_playback(
    c3x_renderer_unit_v1 const* unit,void* destination_hdc,void* background_hdc,int* bounds,unsigned flags) {
    if(!unit || unit->struct_size!=sizeof(*unit) || unit->unit_key[63]!=0 || !destination_hdc || !background_hdc || !bounds ||
       !(flags&C3X_RENDERER_UNIT_STATE_CAPTURED) || (flags&~7u))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    if(!renderer_worker)return C3X_RENDERER_RESULT_ERROR;
    return renderer_worker->draw_unit(*unit,static_cast<HDC>(destination_hdc),static_cast<HDC>(background_hdc),bounds,flags);
}

extern "C" __declspec(dllexport) void c3x_renderer_unit_forget(int unit_id) {
    if(renderer_worker)renderer_worker->forget_unit(unit_id);
}

extern "C" __declspec(dllexport) int c3x_renderer_set_unit_rendering(int enabled) {
    return get_renderer_worker().set_unit_rendering(enabled);
}

// Optional compatibility redraws rebuild native command buttons, whose show
// operation clears pressed-form ownership. Never request one during a mouse
// press. Native actions and resident GPU visual frames have independent owners.
// GetKeyState observes processed state without consuming input transitions.
extern "C" int c3x_renderer_schedule(c3x_renderer_schedule_v1 const*,c3x_renderer_schedule_result_v1*);
extern "C" __declspec(dllexport) int c3x_renderer_schedule_idle(
    c3x_renderer_schedule_v1 const* input,c3x_renderer_schedule_result_v1* output) {
    int result=c3x_renderer_schedule(input,output);
    if(result!=C3X_RENDERER_RESULT_OK)return result;
    unsigned buttons=((GetKeyState(VK_LBUTTON)&0x8000)?1u:0u) |
        ((GetKeyState(VK_RBUTTON)&0x8000)?2u:0u) |
        ((GetKeyState(VK_MBUTTON)&0x8000)?4u:0u);
    bool defer=buttons!=0;
    c3x_renderer_u32 base_request=output->request_redraw;
    c3x_renderer_u32 base_rebase=output->rebase_clock;
    if(defer){output->request_redraw=0;output->dirty_flags=0;output->skipped_frame_count=0;output->rebase_clock=1;}
    char const* reason="cadence-wait";
    if(defer)reason="native-mouse-press";
    else if((input->state_flags&C3X_RENDERER_SCHEDULER_MAP_VISIBLE)==0)reason="map-hidden";
    else if((input->state_flags&C3X_RENDERER_SCHEDULER_FOCUSED)==0)reason="unfocused";
    else if((input->state_flags&C3X_RENDERER_SCHEDULER_MODAL)!=0)reason="modal";
    else if((input->state_flags&C3X_RENDERER_SCHEDULER_DRAWING)!=0)reason="drawing";
    else if((input->state_flags&C3X_RENDERER_SCHEDULER_REDRAW_PENDING)!=0)reason="redraw-pending";
    else if(output->rebase_clock)reason="clock-rebase";
    else if(output->request_redraw)reason="redraw-request";
    char detail[384];
    std::snprintf(detail,sizeof(detail),
        "[C3X renderer] qpc=%lld stage=scheduler-callback buttons=%u state=0x%08x visible=%u base_request=%u base_rebase=%u final_request=%u final_rebase=%u skipped=%u reason=%s\n",
        static_cast<long long>(input->now_ticks),buttons,input->state_flags,input->visible_animation_count,
        base_request,base_rebase,output->request_redraw,output->rebase_clock,output->skipped_frame_count,reason);
    detail[sizeof(detail)-1]='\0';
    OutputDebugStringA(detail);
    return result;
}

// Optional ABI extension: invoked synchronously by native pass-through hooks.
extern "C" __declspec(dllexport) int c3x_renderer_native_observe(c3x_renderer_native_observation const* event) {
    static c3x_native_observation::Capture capture;
    try { return capture.observe(event); }
    catch (...) { capture.ended=true; return 0; }
}

// Optional GPU-only map publication. The live completed-screen compatibility
// callback does not admit map destinations into this exclusive image session.
extern "C" __declspec(dllexport) int c3x_renderer_gpu_render(
    c3x_renderer_camera_request_v1 const* request,c3x_renderer_gpu_frame_v1* view,c3x_renderer_output_v1* metadata){
    if(!request||request->version!=C3X_RENDERER_CAMERA_VIEW_VERSION||request->struct_size!=sizeof(*request)||
       !view||view->struct_size!=sizeof(*view)||!valid_frame(request->frame,metadata)||
       request->frame->target_width>2240||request->frame->target_height>1192)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    *view={sizeof(*view)};
    try{return get_renderer_worker().render_gpu(*request,*view,*metadata);}
    catch(...){return C3X_RENDERER_RESULT_ERROR;}
}
extern "C" __declspec(dllexport) int c3x_renderer_gpu_images(
    c3x_renderer_gpu_images_v1 const* request,c3x_renderer_gpu_result_v1* result,unsigned* readback,unsigned capacity){
    constexpr unsigned max_pixels=2240u*1192u;
    if(!request||request->struct_size!=sizeof(*request)||!result||result->struct_size!=sizeof(*result)||
       request->ticket<=0||request->action<C3X_GPU_CREATE||request->action>C3X_GPU_READBACK||request->pixel_count>max_pixels||
       request->command_count>2048||(request->command_count&&!request->commands)||
       (request->action==C3X_GPU_UPLOAD&&(!request->pixels||!request->pixel_count))||
       (request->action==C3X_GPU_READBACK?(!readback||capacity<request->pixel_count||!capacity):(readback||capacity)))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    bool upload=request->action==C3X_GPU_UPLOAD,submit=request->action==C3X_GPU_SUBMIT;
    bool create=request->action==C3X_GPU_CREATE,read=request->action==C3X_GPU_READBACK;
    if((create?(request->format<C3X_GPU_BGRA32||request->format>C3X_GPU_RGB565):request->format!=0)||
       (!upload && request->pixels)||(!upload && request->revision)||
       (!upload && !read && request->pixel_count)||(!submit && (request->commands||request->command_count||request->command_struct_size))||
       (submit && (!request->command_count||request->command_struct_size!=sizeof(c3x_renderer_gpu_command_v1)))||(create?(request->width<=0||request->height<=0||request->width>2240||request->height>1192||request->image!=0):(request->width||request->height))||
       (!create && !submit && request->image<=0)||(read&&!request->pixel_count))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    *result={sizeof(*result)};
    try{return get_renderer_worker().images_gpu(*request,*result,readback,capacity);}
    catch(...){return C3X_RENDERER_RESULT_ERROR;}
}

// Native/UI thread owns DXGI window operations. The existing worker prepares
// the complete retained display under the same serialized ownership boundary.
extern "C" __declspec(dllexport) int c3x_renderer_gpu_present(c3x_renderer_gpu_present_v1 const* request){
    if(!request||request->struct_size!=sizeof(*request)||request->action<0||request->action>2)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    if(request->action==0&&(request->ticket<=0||request->image<=0||!request->window||request->width<=0||request->height<=0||request->width>2240||request->height>1192||
        request->area[0]>=request->area[2]||request->area[1]>=request->area[3]||request->area[0]>=request->width||request->area[1]>=request->height||request->area[2]<=0||request->area[3]<=0))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    try{return get_renderer_worker().present_gpu(*request);}catch(...){return C3X_RENDERER_RESULT_ERROR;}
}

// Bound behind the hash-verified native hooks. Map preparation activates the
// exclusive owner; before that, completed CPU screens retain compatibility
// presentation. A negative result denies CPU access after a failed barrier.
extern "C" __declspec(dllexport) int c3x_renderer_native_image(int operation,void* image,void* source,void const* from,void const* to,unsigned color){
    if(operation==C3X_NATIVE_VISUAL_POLICY)return renderer_worker?renderer_worker->visual_policy(color):0;
    if(operation==C3X_NATIVE_IMAGE_DRAIN){if(!drain_native_composition())return -1;}
    else if(native_composition&&native_composition->active()){
        try{
            int result=native_composition->operation(operation,image,source,from,to,color);
            if(operation!=C3X_NATIVE_IMAGE_PRESENT || result!=0){
                if(operation==C3X_NATIVE_IMAGE_PRESENT&&result>0){static unsigned frames=0;++frames;
                    if(frames<=3||(frames%128)==0){char line[128];std::snprintf(line,sizeof(line),
                        "[C3X renderer] stage=native-resident-present frames=%u cpu_snapshot=0\n",frames);OutputDebugStringA(line);}}
                return result;
            }
            // An unowned CPU UI source uses the same final presenter. The
            // resident map family remains untouched, including full-color data.
        }
        catch(std::exception const& e){OutputDebugStringA(e.what());return -1;}
    }
    if(operation!=C3X_NATIVE_IMAGE_PRESENT && operation!=C3X_NATIVE_IMAGE_DRAIN)return 0;
    if(operation==C3X_NATIVE_IMAGE_DRAIN){if(renderer_worker)try{renderer_worker->native_screen(nullptr);}catch(...){}return 0;}
    LARGE_INTEGER began={},captured={},ended={},frequency={};QueryPerformanceCounter(&began);captured=began;
    bool presented=false;std::uint64_t uploaded=0;
    try {
        c3x_native_images::ScreenSnapshot screen;
        if(screen.capture(image,source,from)){
            QueryPerformanceCounter(&captured);
            uploaded=std::uint64_t(screen.area.right-screen.area.left)*(screen.area.bottom-screen.area.top)*2;
            presented=get_renderer_worker().native_screen(&screen)==C3X_RENDERER_RESULT_OK;
        }
    }catch(...){}
    // Release DXGI before JGL's original BitBlt. CPU images were never stale.
    if(!presented && renderer_worker)try{renderer_worker->native_screen(nullptr);}catch(...){}
    QueryPerformanceCounter(&ended);QueryPerformanceFrequency(&frequency);
    struct Counts {unsigned calls=0,gpu=0;std::uint64_t bytes=0;double capture_ms=0,callback_ms=0,max_ms=0;};
    static Counts counts;
    ++counts.calls;counts.gpu+=presented;counts.bytes+=presented?uploaded:0;
    double elapsed=1000.*double(ended.QuadPart-began.QuadPart)/double(frequency.QuadPart);
    counts.capture_ms+=1000.*double(captured.QuadPart-began.QuadPart)/double(frequency.QuadPart);counts.callback_ms+=elapsed;counts.max_ms=std::max(counts.max_ms,elapsed);
    if(counts.calls==1||counts.calls%120==0){char line[384];std::snprintf(line,sizeof(line),
        "[C3X renderer] stage=native-screen calls=%u gpu=%u native=%u upload_bytes=%llu snapshot_ms=%.3f callback_ms=%.3f callback_max_ms=%.3f\n",
        counts.calls,counts.gpu,counts.calls-counts.gpu,static_cast<unsigned long long>(counts.bytes),counts.capture_ms,counts.callback_ms,counts.max_ms);OutputDebugStringA(line);}
    return presented?1:0;
}

// Existing unit playback and pose cache, with GPU-native destination ownership.
extern "C" __declspec(dllexport) int c3x_renderer_gpu_unit(c3x_renderer_unit_v1 const* unit,c3x_renderer_gpu_unit_v1 const* target,int* bounds){
    if(!unit||unit->struct_size!=sizeof(*unit)||unit->unit_key[63]!=0||!target||target->struct_size!=sizeof(*target)||!bounds||
       target->ticket<=0||target->destination<=0||target->background<=0||target->detail<0||target->background_detail<0||
       target->clip[0]>target->clip[2]||target->clip[1]>target->clip[3]||(target->playback_flags&~7u)||
       unit->body_x<-32768||unit->body_x>32768||unit->body_y<-32768||unit->body_y>32768)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    if(!renderer_worker)return C3X_RENDERER_RESULT_ERROR;
    try{return renderer_worker->draw_unit(*unit,nullptr,nullptr,bounds,target->playback_flags,target);}catch(...){return C3X_RENDERER_RESULT_ERROR;}
}

// Pinned with the native hooks across renderer configuration/unload. Tracking
// cannot render, request work, or infer that an older image has no CPU aliases.
extern "C" __declspec(dllexport) int c3x_renderer_native_lifetime(int operation,void* image,int context){
    static c3x_native_images::Lifetimes lifetimes;
    bool eligible=lifetimes.observe(operation,image,context,GetCurrentThreadId());
    if(operation==C3X_NATIVE_VERIFY&&!image)OutputDebugStringA("[C3X renderer] stage=native-tracking event=reset\n");
    if(operation==C3X_NATIVE_MAP){
        static unsigned requests=0,accepted=0;++requests;accepted+=eligible;
        if(requests<=8||(requests%128)==0){char line[160];std::snprintf(line,sizeof(line),
            "[C3X renderer] stage=native-lifetime requests=%u eligible=%u accepted=%u\n",requests,unsigned(eligible),accepted);OutputDebugStringA(line);}
    }
    return eligible?1:0;
}

// The live native map seam uses the same GPU producer and image adapter as the
// connected fixture. No CPU bitmap is returned for an admitted map publication.
extern "C" __declspec(dllexport) int c3x_renderer_native_map(int action,void* image,
    c3x_renderer_camera_request_v1 const* request,c3x_renderer_output_v1* output){
    if(action==C3X_NATIVE_MAP_PREPARE&&(!request||request->version!=C3X_RENDERER_CAMERA_VIEW_VERSION||
        request->struct_size!=sizeof(*request)||!valid_frame(request->frame,output)))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    try {
    if(!native_composition){
        if(action==C3X_NATIVE_MAP_CANCEL)return C3X_RENDERER_RESULT_OK;
        if(action!=C3X_NATIVE_MAP_PREPARE||!c3x_renderer_native_lifetime(C3X_NATIVE_MAP,image,0))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        auto jgl=GetModuleHandleA("jgl.dll");if(!c3x_native_observation::verified_module(jgl))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        auto base=reinterpret_cast<char*>(jgl);
        native_composition=new c3x_native_images::CompositionOwner(c3x_renderer_gpu_render,c3x_renderer_gpu_images,c3x_renderer_gpu_present,
            c3x_renderer_gpu_unit,c3x_renderer_native_lifetime,base+0x1b70,base+0x1b90);
    }
    return native_composition->map(action,image,request,output);
    }catch(std::exception const& e){OutputDebugStringA(e.what());return C3X_RENDERER_RESULT_DEVICE_ERROR;}
}

// Native caller adoption includes the honest ambient sample clock. The older
// metadata-only export remains available to explicit integration controls.
extern "C" __declspec(dllexport) int c3x_renderer_native_map_view(int action,void* image,
    c3x_renderer_camera_request_v1 const* request,c3x_renderer_camera_view_v1* view){
    if(action==C3X_NATIVE_MAP_PREPARE && (!view || view->version!=C3X_RENDERER_CAMERA_VIEW_VERSION ||
       view->struct_size!=sizeof(*view)))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    c3x_renderer_output_v1 metadata={C3X_RENDERER_API_VERSION,sizeof(metadata)};
    int result=c3x_renderer_native_map(action,image,request,action==C3X_NATIVE_MAP_PREPARE?&metadata:nullptr);
    if(result==C3X_RENDERER_RESULT_OK && action==C3X_NATIVE_MAP_PREPARE){
        view->ticket=0;view->identity=request->identity;view->frame=*request->frame;view->output=metadata;
        view->frame.presentation_time_ticks=native_composition->sample_ticks();
    }
    return result;
}
