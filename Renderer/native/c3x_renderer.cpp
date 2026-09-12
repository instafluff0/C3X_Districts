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
#include "animation_runtime.h"
#include "benchmark_oracle.h"
#include "environment_runtime.h"
#include "terrain_definition_runtime.h"
#include "renderer_trace.h"
#include "asset_content_hash.h"
#include "scroll_damage.h"
#include "river_node_locality.h"
#include "pixel_block_cache.h"
#include "render_core/render_region_cache.h"
#include "render_core/region_contributor_index.h"
#include "render_core/center_shore_cache.h"
#include "render_core/projected_mesh_bounds.h"
#include "color_quantization.h"
#include "render_core/terrain_query.h"
#include "render_core/world_coast.h"
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
#include "environment_refresh/reflection.h"
#include "unit_body_renderer.h"
#include "city_fidelity/gpu.h"
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
    std::vector<std::pair<std::uint64_t,std::uint64_t>> tile_keys;
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
    std::vector<std::pair<std::uint64_t,std::uint64_t>> tile_keys;
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
    std::size_t byte_count = 0;
    int translation_x = 0, translation_y = 0;
    D3D11_RECT bounds = {};
    c3x_renderer::render_core::SourceShadow::Bounds world_bounds;
    c3x_renderer::render_core::ProjectedMeshBounds projected_bounds;
    std::uint64_t version = 0;
    ID3D11ShaderResourceView * animation_texture = nullptr; // borrowed, dynamic pass only
    unsigned city_material=0xffffffffu;
    bool city_environment=false;
    float city_atlas[4]={};
    std::shared_ptr<c3x_renderer::city_fidelity::Lighting> city_lighting;
    float natural_projection[4] = {};
};

struct PendingCityChunk {
    std::vector<Vertex> vertices;
    unsigned material=0;bool environment=false;float atlas[4]={};
    std::shared_ptr<c3x_renderer::city_fidelity::Lighting> lighting;
};

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
    std::vector<std::uint8_t> dds;
    ID3D11ShaderResourceView * view = nullptr;
    ID3D11Buffer * indices = nullptr;
    float scale = 1, yaw = 0, offset[3] = {};
    unsigned count = 1;
};
struct ResourceAnchor {
    unsigned asset = 0, seed = 0;
    float u = .5f, v = .5f, world_u = 0, world_v = 0, ground = 0;
    int anchor_x = 0, anchor_y = 0;
};
struct ResourceBackdrop {
    int x=0,y=0;
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

struct CachedTileGeometry {
    std::uint64_t signature = 0, version = 0;
    // A separate cache entry owns each world mesh once. Projection entries
    // retain only its key, never unaccounted references to evicted buffers.
    std::uint64_t natural_signature = 0;
    std::uint64_t natural_version = 0;
    bool shared_natural = false;
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
    std::vector<std::array<float,19>> vertices;
    std::vector<UINT> indices;
};
struct NaturalTile {
    std::array<NaturalMesh, geometry_layer_count-geometry_natural_terrain> layers;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> dependencies, coast_dependencies;
    std::vector<std::pair<std::size_t,std::uint32_t>> world_dependencies;
    std::size_t bytes=0;
    std::uint64_t used=0;
    int tile_width=0,tile_height=0,target_height=0,tile_x=0,tile_y=0;
};

// Hash bytes, then compare all bytes: repeated triangle vertices share storage
// without merging seams, normals, materials, or merely similar positions.
struct VertexHash {
    std::size_t stride = sizeof(Vertex);
    bool feature = false;
    std::size_t operator()(Vertex const & vertex) const {
        auto bytes = reinterpret_cast<unsigned char const *>(&vertex);
        float fields[12];
        if(feature){float values[]={vertex.x,vertex.y,vertex.z,vertex.u,vertex.v,
            vertex.normal_x,vertex.normal_y,vertex.normal_z,vertex.base_terrain,
            vertex.world_x,vertex.world_y,vertex.world_z};
            std::memcpy(fields,values,sizeof(fields));bytes=reinterpret_cast<unsigned char const*>(fields);}
        std::uint64_t result = 1469598103934665603ull;
        for (std::size_t i = 0; i < (feature?sizeof(fields):stride); i += sizeof(std::uint32_t)) {
            std::uint32_t word;
            std::memcpy(&word, bytes + i, sizeof(word));
            result = (result ^ word) * 1099511628211ull;
        }
        // Float grids often have identical low mantissa bits. The flat table
        // masks low hash bits, unlike the old prime-bucket unordered_map.
        // Avalanche high coordinate bits before masking to avoid quadratic
        // probing on otherwise ordinary flat terrain. Equality stays exact.
        result ^= result >> 33; result *= 0xff51afd7ed558ccdull;
        result ^= result >> 33; result *= 0xc4ceb9fe1a85ec53ull;
        result ^= result >> 33;
        return static_cast<std::size_t>(result);
    }
};
struct VertexEqual {
    std::size_t stride = sizeof(Vertex);
    bool feature = false;
    bool operator()(Vertex const & a, Vertex const & b) const {
        if(feature)return std::memcmp(&a,&b,20)==0 && std::memcmp(&a.normal_x,&b.normal_x,12)==0 &&
            std::memcmp(&a.base_terrain,&b.base_terrain,4)==0 && std::memcmp(&a.world_x,&b.world_x,12)==0;
        return std::memcmp(&a, &b, stride) == 0;
    }
};

struct GroundPoint {
    float u = 0.0f, v = 0.0f;
    float world_u = 0.0f, world_v = 0.0f;
    float local_ground_x = 0.0f, local_ground_y = 0.0f;
    float material_u = 0.0f, material_v = 0.0f;
    float material_weights[5] = {};
    float signed_shore = 0.0f;
    c3x_renderer::render_core::ShoreSample shore;
    float surface_coordinate = 0.0f;
    float relief[3] = {};
    float normal[3] = {0.0f, 0.0f, 1.0f};
    float normal_delta[2] = {};
    bool terrain_ready = false;
};

struct CachedGroundGrid {
    int divisions=0;
    float layer=0;
    std::vector<Vertex> vertices;
    // Preserve raw height and normal numerators; recovering these from packed
    // world Z or normalized normals would introduce zoom-dependent rounding.
    std::vector<std::array<float,3>> samples;
    int sample_stride(int requested) const {
        if(requested<=0 || divisions<requested || divisions%requested!=0)return 0;
        int stride=divisions/requested;
        // Only reuse genuinely identical sampling coordinates, including the
        // floating-point division. Never interpolate a coarser approximation.
        for(int i=0;i<=requested;++i)
            if(float(i*stride)/divisions!=float(i)/requested)return 0;
        return stride;
    }
    Vertex project(std::size_t index,int width,int height) const {
        Vertex out=vertices[index];
        float u=float(index%std::size_t(divisions+1))/divisions;
        float v=float(index/std::size_t(divisions+1))/divisions;
        float half_w=float(width)*.5f,half_h=float(height)*.5f;
        float h=samples[index][0]*(float(width)/224.f*.82f);
        float base=(u+v)*half_h;
        out.x=half_w+(u-v)*half_w;out.y=base-h;out.z=base+h*.75f;
        if(layer==1.f || layer==9.f){
            float su=samples[index][1]/(2.f*.006f*float(width));
            float sv=samples[index][2]*-1.f/(2.f*.006f*float(width));
            float length=std::sqrt(su*su+sv*sv+1.f);
            out.normal_x=-su/length;out.normal_y=-sv/length;out.normal_z=1.f/length;
        }
        return out;
    }
};
struct CachedGroundTile {
    std::uint64_t signature=0,used=0;
    int x=0,y=0;
    std::size_t bytes=0;
    std::vector<CachedGroundGrid> grids;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> dependencies,coast_dependencies;
    std::vector<std::pair<std::size_t,std::uint32_t>> world_dependencies;
};

struct RiverNode {
    int lattice_x, lattice_y;
    unsigned degree;
    bool touches_water;
};

struct SceneTopology {
    std::uint64_t signature = 0;
    c3x_renderer_tile_v1 const * records = nullptr;
    std::unordered_map<std::uint64_t, int> ground, real, relief, surface;
    std::unordered_map<std::uint64_t, c3x_renderer_tile_v1 const *> tiles;
    std::unordered_map<std::uint64_t, std::uint64_t> semantic;
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
    std::size_t viewport_cache_budget=default_viewport_cache_budget;
    std::size_t resource_backdrop_cache_budget=default_resource_backdrop_cache_budget;
    c3x_renderer::UnitBodyRenderer unit_bodies;
    bool unit_rendering_enabled=false;
    bool fidelity_profile = false, fidelity_shadow_control = false;
    bool environment_profile = false;
    bool city_profile=false;
    bool clip_dirty_blocks=false;
    bool bounded_post=false;
    int scene_region_size=128;
    int scene_region_height=128;
    bool cull_empty_water=false;
    bool world_backdrops=false,backdrop_reuse_control=false;
    bool world_waves=false,wave_reuse_control=false;
    bool animation_readback_atlas=false;
    bool world_raster_grid=false;
    bool world_regions=false,world_regions_control=false;
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
    RendererTrace trace;
    c3x_renderer::render_core::GpuFrameTelemetry gpu_telemetry;
    bool profiling=false;
    std::size_t sampled_geometry_bucket=~std::size_t(0);
    std::uint64_t frame_draw_calls=0,frame_parameter_updates=0,frame_bounds_tests=0;
    unsigned frame_caster_preparations=0;
    std::size_t frame_post_lanes=0;
    void memory_sample(char const* phase) {
        if(!profiling)return;
        auto sample=c3x_renderer::render_core::AddressSpaceSample::capture();
        char detail[640];sprintf_s(detail,
            "phase=%s available_virtual=%llu largest_free_region=%llu committed_va=%llu reserved_va=%llu "
            "gpu_geometry=%zu gpu_geometry_cap=%zu natural_cpu=%zu ground_cpu=%zu natural_cpu_cap=%zu "
            "viewport_cpu=%zu viewport_cpu_cap=%zu backdrop_gpu=%zu backdrop_gpu_cap=%zu pixels_capacity=%zu",
            phase,sample.available,sample.largest,sample.committed,sample.reserved,
            tile_geometry_cache_bytes,tile_geometry_cache_budget,natural_mesh_cache_bytes,ground_grid_cache_bytes,
            natural_mesh_cache_budget,viewport_cache_bytes,viewport_cache_budget,resource_backdrop_bytes,
            resource_backdrop_cache_budget,pixels.capacity()*sizeof(pixels[0]));
        trace.write("memory-sample",detail,true);
        sprintf_s(detail,"linear_frame=%zu linear_block=%zu reflection=%zu glow=%zu region_reflection=%zu region_glow=%zu wave_geometry=%zu",
            linear_frame.bytes(),linear_block.bytes(),reflection.linear.bytes(),city_glow.linear.bytes(),
            region_reflection.linear.bytes(),region_glow.linear.bytes(),wave_geometry_bytes);
        trace.write("memory-linear-scratch",detail,true);
    }
    SceneTopology topology_cache;
    std::uint64_t requested_signature = 0;
    char const * frame_cache_path = "cold";
    c3x_renderer_i64 frame_geometry_ticks = 0, frame_draw_ticks = 0, frame_readback_ticks = 0;
    ID3D11Device * device = nullptr;
    ID3D11DeviceContext * context = nullptr;
    ID3D11VertexShader * vertex_shader = nullptr;
    ID3D11PixelShader * pixel_shader = nullptr;
    ID3D11VertexShader * feature_vertex_shader = nullptr;
    ID3D11PixelShader * feature_pixel_shader = nullptr;
    ID3D11InputLayout * input_layout = nullptr;
    ID3D11InputLayout * feature_input_layout = nullptr;
    ID3D11Buffer * terrain_settings_buffer = nullptr;
    ID3D11Buffer * viewport_settings_buffer = nullptr;
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
    std::uint64_t resource_backdrop_epoch=0;
    std::size_t resource_backdrop_bytes=0;
    c3x_renderer_i64 resource_composite_ticks=0;
    std::vector<std::uint32_t> resource_pixels;
    std::uint64_t resource_pixel_signature = 0;
    c3x_renderer_i64 resource_pixel_clock = -1;
    unsigned visible_resource_animations = 0, visible_wave_animations = 0;
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
    unsigned ambient_count() const {return visible_resource_animations+visible_wave_animations;}
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
    std::array<std::vector<CachedVertexChunk>, geometry_layer_count>
        geometry_vertex_buffers;
    std::unordered_multimap<std::uint64_t, CachedTileGeometry> tile_geometry_cache;
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
        visible_resource_animations = visible_wave_animations = 0;
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
            release_geometry_vertex_buffers(oldest->second.buffers);tile_geometry_cache.erase(oldest);
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
        cancel_pixel_preparation();
        linear_frame.reset(); linear_block.reset(); reflection.linear.reset();region_reflection.linear.reset();region_glow.linear.reset();
        pixel_blocks.clear();
        release(block_readback); release(block_depth); release(block_depth_texture);
        release(block_target); release(block_texture);
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
        resource_pixel_signature = 0; resource_pixel_clock = -1; visible_resource_animations = 0;
        for (auto & animation : resource_animations) {
            release(animation.view); release(animation.indices);
        }
    }

    void reset() {
        memory_sample("before-reset");
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
        tile_geometry_runtime_budget=tile_geometry_cache_budget;
        unit_bodies.benchmark_reset_pose_limit();
#endif
        render_regions.clear();region_context.clear();retained_region_casters={};
        gpu_telemetry.reset();
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
        release(terrain_settings_buffer);
        release(viewport_settings_buffer);
        release(world_settings_buffer); release(shadow_settings_buffer);
        linear_output.reset();
        clear_geometry_vertex_buffers();
        clear_tile_geometry_cache();
        release(feature_pixel_shader);
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

    bool initialize() {
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

        // Production executes the material and feature entry points copied
        // from the approved Lab handoff, isolated from in-progress Lab edits.
        auto compile_terrain_shader = [this](char const * entry, char const * target,
                                             ID3DBlob ** blob) {
            std::string selected_shader=integrated_shader_path;
            if(fidelity_profile && std::strstr(entry,"Feature"))selected_shader=fidelity_root+(city_profile?"/Renderer/native/city_fidelity/feature.hlsl":environment_profile?"/Renderer/native/environment_refresh/feature.hlsl":"/Renderer/native/render_core/terrain_scene.hlsl");
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
        animation_readback_atlas=GetEnvironmentVariableA("C3X_RENDERER_ANIMATION_READBACK_ATLAS",control,sizeof(control)) && std::strcmp(control,"1")==0;
        reflection.enabled=!(GetEnvironmentVariableA("C3X_RENDERER_REFLECTION_CONTROL",control,sizeof(control)) && std::strcmp(control,"1")==0);
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
                    if(m.bytes && !pinned && m.used<oldest){oldest=m.used;mesh_id=int(i);texture_id=-1;}
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
                } else return false;
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
                mesh.animation=std::move(decoded);mesh.bytes=bytes;bodies.resident_bytes+=bytes;++loads;
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
            auto chunk=owner;
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
            auto identity=world_coast.world().index(c,r);if(identity==std::size_t(-1))continue;
            auto hash=c3x_renderer::stable_hash(unsigned(identity)*193u+71u);
            float seed=c3x_renderer::stable_random(hash),seed2=c3x_renderer::stable_random(hash+23u);
            auto ribbon=coastal_wave_ribbon(world_coast,c,r,.30f+.70f*seed);if(ribbon.empty())continue;
            std::vector<Vertex> vertices;vertices.reserve(ribbon.size());
            CachedVertexChunk chunk;chunk.bounds={LONG_MAX,LONG_MAX,LONG_MIN,LONG_MIN};
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
        if (!frame_has_resource_animation(frame)) {
            visible_resource_animations=visible_wave_animations=0; resource_pixel_signature=0; return true;
        }
        auto clock=resource_clock(frame);
        if (ambient_count() && resource_pixel_signature==cached_signature.complete &&
            resource_pixel_clock==clock) return true;
        LARGE_INTEGER started={},finished={};QueryPerformanceCounter(&started);
        visible_resource_animations=0;
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
        float half_w=frame.tile_width*.5f,half_h=frame.tile_height*.5f;
        float projection=frame.tile_width/224.f,relief=projection*.82f;
        int dx=int(geometry_viewport_settings.translation[0]),dy=int(geometry_viewport_settings.translation[1]);
        auto ticks=clock*std::max<c3x_renderer_i64>(1,frame.presentation_frequency/15);
        for (auto const & anchor:resource_anchors) {
            if (anchor.asset>=resource_animations.size()) return false;
            auto & animation=resource_animations[anchor.asset];
            double time=c3x_renderer::ambient_animation_time(ticks,frame.presentation_frequency,
                animation.mesh.duration,anchor.seed);
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
                    anchor.ground*relief*.75f+feature_height*.0012f*frame.target_height;
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
            buffers[geometry_feature].push_back(chunk);++visible_resource_animations;
        }
        if(!prepare_wave_chunks(frame))return false;
        visible_wave_animations=unsigned(wave_chunks.size());
        buffers[geometry_wave]=wave_chunks;
        if(visible_wave_animations){
            float time[]={float(double(ticks)/std::max<c3x_renderer_i64>(1,frame.presentation_frequency)),0,0,0};
            context->UpdateSubresource(wave_frame,0,nullptr,time,0,0);
            for(auto const& chunk:wave_chunks){
                int wave_dx=chunk.translation_x+dx,wave_dy=chunk.translation_y+dy;
                D3D11_RECT visible={std::max<LONG>(0,chunk.bounds.left+wave_dx),std::max<LONG>(0,chunk.bounds.top+wave_dy),std::min<LONG>(width,chunk.bounds.right+wave_dx),std::min<LONG>(height,chunk.bounds.bottom+wave_dy)};
                dirty(visible);
            }
        }
        if (!ambient_count()) {resource_pixel_signature=0;return true;}
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
        // Geometry identity includes the entire captured semantic/ownership
        // set, target/zoom, light, wrap, content and device generations, but
        // excludes camera anchors. Conservatively miss when that set changes.
        auto backdrop_signature=anchored?c3x_renderer::render_core::static_region_identity(
            cached_signature.geometry,std::uint64_t(geometry_world_revision)):cached_signature.complete;
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
        std::size_t rectangle_index=0;
        for(auto & rect:rectangles) {
            QueryPerformanceCounter(&background_started);
            int key_x=rect.left-anchor_x,key_y=rect.top-anchor_y;
            auto found=std::find_if(resource_backdrops.begin(),resource_backdrops.end(),[&](auto const& block){
                return !backdrop_reuse_control && block.signature==backdrop_signature && block.x==key_x && block.y==key_y;
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
                        return !block.dependencies.empty() && block.dependencies==backdrop_dependencies;
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
                context->CopyResource(backdrop.color,found->color);
                context->CopyResource(backdrop.depth_texture,found->depth);++backdrop_hits;
            } else {
                if(!submit_geometry(geometry_vertex_buffers,{{0,0,128,128}},settings,block_target,block_depth,128,128,
                    nullptr,false,true,nullptr,false,animation_casters_ptr,animation_prepared_ptr,128,0,0,true))return false;
                ++backdrop_misses;
                // RGBA16F + D24S8, both MSAA4. Cache immutable scene-linear
                // background/depth by static inputs and the region's relative
                // world placement. Animation time and poses never enter it.
                std::size_t bytes=std::size_t(backdrop_extent)*backdrop_extent*48u+
                    backdrop_dependencies.capacity()*sizeof(backdrop_dependencies[0])+sizeof(ResourceBackdrop);
                if(!backdrop_reuse_control && make_resource_backdrop_room(bytes,backdrop_signature)) {
                    resource_backdrops.reserve(resource_backdrops.size()+1);
                    ResourceBackdrop block;block.x=key_x;block.y=key_y;block.bytes=bytes;
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
            if(!submit_geometry(buffers,{{0,0,128,128}},settings,block_target,block_depth,128,128,
                    nullptr,true,true,&geometry_vertex_buffers,false,animation_casters_ptr,animation_prepared_ptr))return false;
            D3D11_BOX box={unsigned(clipped.left-rect.left),unsigned(clipped.top-rect.top),0,
                unsigned(clipped.right-rect.left),unsigned(clipped.bottom-rect.top),1};
            if(animation_readback_atlas) {
                unsigned atlas_x=unsigned(rectangle_index%atlas_columns)*128;
                unsigned atlas_y=unsigned(rectangle_index/atlas_columns)*128;
                context->CopySubresourceRegion(animation_readback_texture,0,atlas_x,atlas_y,0,block_texture,0,&box);
            } else {
                context->CopySubresourceRegion(render_texture,0,unsigned(clipped.left),unsigned(clipped.top),0,block_texture,0,&box);
            }
            rect=clipped;
            ++rectangle_index;
            QueryPerformanceCounter(&region_finished);
            animation_ticks+=region_finished.QuadPart-animation_started.QuadPart;
        }
        LARGE_INTEGER readback_started={},readback_submitted={},readback_ready={};
        QueryPerformanceCounter(&readback_started);
        unsigned dirty_pixels=0;
        for(auto const & rect:rectangles) {
            if(!animation_readback_atlas) {
                D3D11_BOX box={unsigned(rect.left),unsigned(rect.top),0,unsigned(rect.right),unsigned(rect.bottom),1};
                context->CopySubresourceRegion(readback_texture,0,box.left,box.top,0,render_texture,0,&box);
            }
            dirty_pixels+=unsigned((rect.right-rect.left)*(rect.bottom-rect.top));
        }
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
        char detail[640];sprintf_s(detail,"visible=%u waves=%u facing=SE clock=%lld rects=%zu pixels=%u upload_bytes=%zu pool_bytes=%zu backdrop_hits=%u backdrop_misses=%u backdrop_bytes=%zu terrain_built=%u ms=%.3f wave_upload_bytes=%zu wave_geometry_bytes=%zu wave_cells_built=%u wave_cells_reused=%u wave_cell_entries=%zu caster_preparations=%u readback=%s readback_width=%u readback_height=%u",
            visible_resource_animations,visible_wave_animations,clock,rectangles.size(),dirty_pixels,uploaded,pool_bytes,backdrop_hits,backdrop_misses,
            resource_backdrop_bytes,frame_tiles_built,
            trace.milliseconds(resource_composite_ticks),wave_upload_bytes,wave_geometry_bytes,wave_cells_built,wave_cells_reused,retained_wave_cells.size(),frame_caster_preparations,
            animation_readback_atlas?"atlas":"full",animation_readback_atlas?animation_readback_width:unsigned(width),animation_readback_atlas?animation_readback_height:unsigned(height));trace.write("animation-frame",detail);
        return true;
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
            char detail[320];sprintf_s(detail,"hits=%zu misses=%zu hit_pixels=%zu gpu_bytes=%zu metadata_bytes=%zu entries=%zu rejected=%zu evictions=%llu metadata_cap=%zu",
                frame_region_hits,frame_region_misses,frame_region_hit_pixels,render_regions.gpu_bytes,render_regions.metadata_bytes,
                render_regions.entries.size(),frame_region_rejected,render_regions.evictions,render_regions.metadata_limit);trace.write("render-region-cache",detail,false);
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
        output.bgra_pixels = ambient_count() ? resource_pixels.data() : pixels.data();
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
        release_geometry_vertex_buffers(geometry_vertex_buffers);
    }

    void clear_tile_geometry_cache() {
        natural_mesh_cache.clear();natural_mesh_cache_bytes=0;
        ground_grid_cache.clear();ground_grid_cache_bytes=0;
        topology_cache = {};
        cancel_pixel_preparation();
        pixel_blocks.clear();
        bitmap_footprints.clear();
        for (auto & entry : tile_geometry_cache)
            release_geometry_vertex_buffers(entry.second.buffers);
        tile_geometry_cache.clear();
        tile_geometry_cache_bytes = 0;
        prefetched_geometry_bytes = 0;
    }

    bool make_tile_cache_room(std::size_t bytes) {
        while (tile_geometry_cache_bytes + bytes > tile_geometry_runtime_budget ||
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
            tile_geometry_cache.erase(oldest);
            ++frame_tiles_evicted;
            if (cache_evictions != 0xffffffffu) ++cache_evictions;
        }
        return true;
    }

    bool cache_geometry_layer(std::vector<Vertex> & vertices,
                              std::vector<CachedVertexChunk> & output,
                              bool prefetch = false, std::size_t pending_bytes = 0,
                              std::atomic<bool> const * foreground_pending = nullptr, bool compact_feature = false, bool natural_vertex = false,
                              NaturalMesh* record=nullptr,NaturalMesh const* cached=nullptr,
                              c3x_renderer::fidelity::GroundProjection const* projection=nullptr,
                              std::vector<UINT> const* grid_indices=nullptr) {
        if (vertices.empty() && (!cached || cached->vertices.empty())) return true;
        std::size_t vertex_stride = pickup_profile ? (natural_vertex?76u:compact_feature?48u:sizeof(Vertex)) : 120u;
        std::size_t hash_stride = pickup_profile ? sizeof(Vertex) : 120u;
        std::vector<Vertex> packed;
        std::vector<UINT> indices;
        if(cached) {
            indices=cached->indices;packed.resize(cached->vertices.size());
            for(std::size_t i=0;i<packed.size();++i){auto const& p=cached->vertices[i];auto& v=packed[i];
                if(prefetch && (i&255u)==0 && foreground_pending->load(std::memory_order_relaxed))return false;
                v.x=p[0];v.y=p[1];v.z=p[2];v.world_x=p[3];v.world_y=p[4];v.world_z=p[5];v.world_valid=p[6];
                v.normal_x=p[7];v.normal_y=p[8];v.normal_z=p[9];v.u=p[10];v.v=p[11];
                v.material_grass=p[12];v.material_plains=p[13];v.material_desert=p[14];v.material_marsh=p[15];
                v.authored_relief_height=p[16];v.authored_relief_blend=p[17];v.base_terrain=p[18];
                if(projection){auto projected=(*projection)(v.world_x,v.world_y,v.world_z*112.f);
                    v.x=projected.x;v.y=projected.y;v.z=projected.z;}
            }
        } else if(grid_indices) {
            // The ground compiler already owns unique corners and exact
            // triangle topology. Do not expand and hash them a second time.
            packed.swap(vertices);indices=*grid_indices;
        } else {
        // Index into the packed array instead of allocating a hash node and
        // copying a 168-byte key for every unique vertex. Equality and first
        // occurrence order are identical to the old node-based table.
        std::size_t capacity=1;
        while(capacity<vertices.size()*2u)capacity*=2u;
        std::vector<UINT> slots(capacity,UINT_MAX);
        VertexHash vertex_hash{hash_stride,pickup_profile && compact_feature};
        VertexEqual vertex_equal{hash_stride,pickup_profile && compact_feature};
        packed.reserve(vertices.size()/3u);
        indices.reserve(vertices.size());
        for (Vertex const & vertex : vertices) {
            if (prefetch && (indices.size() & 255u) == 0 && foreground_pending->load(std::memory_order_relaxed)) return false;
            auto slot=vertex_hash(vertex)&(capacity-1);
            while(slots[slot]!=UINT_MAX && !vertex_equal(packed[slots[slot]],vertex))slot=(slot+1)&(capacity-1);
            if(slots[slot]==UINT_MAX){slots[slot]=static_cast<UINT>(packed.size());packed.push_back(vertex);}
            indices.push_back(slots[slot]);
        }
        }
        CachedVertexChunk chunk;
        chunk.bounds = {LONG_MAX, LONG_MAX, LONG_MIN, LONG_MIN};
        chunk.version=tile_geometry_version; chunk.vertex_stride=static_cast<UINT>(vertex_stride);
        for(unsigned i=0;i<3;++i){chunk.world_bounds.low[i]=1e9f;chunk.world_bounds.high[i]=-1e9f;}
        for (Vertex const & vertex : packed) {
            if(natural_vertex)chunk.projected_bounds.include(vertex.world_x,vertex.world_y,vertex.world_z);
            float world_values[]={vertex.world_x,vertex.world_y,vertex.world_z};
            if(pickup_profile)for(unsigned i=0;i<3;++i){float value=world_values[i];
                chunk.world_bounds.low[i]=std::min(chunk.world_bounds.low[i],value);
                chunk.world_bounds.high[i]=std::max(chunk.world_bounds.high[i],value);}
            chunk.bounds.left = std::min(chunk.bounds.left, static_cast<LONG>(std::floor(vertex.x)) - 2);
            chunk.bounds.top = std::min(chunk.bounds.top, static_cast<LONG>(std::floor(vertex.y)) - 2);
            chunk.bounds.right = std::max(chunk.bounds.right, static_cast<LONG>(std::ceil(vertex.x)) + 2);
            chunk.bounds.bottom = std::max(chunk.bounds.bottom, static_cast<LONG>(std::ceil(vertex.y)) + 2);
        }
        chunk.index_count = static_cast<UINT>(indices.size());
        // Most per-tile chunks fit in 16-bit indices, including authored trees.
        // Preserve the complete vertex/triangle order; only storage narrows.
        std::vector<std::uint16_t> narrow_indices;
        if (packed.size() <= 65535u) {
            narrow_indices.reserve(indices.size());
            for (UINT index : indices) narrow_indices.push_back(static_cast<std::uint16_t>(index));
            chunk.index_format = DXGI_FORMAT_R16_UINT;
        }
        std::size_t index_bytes = indices.size() * (narrow_indices.empty() ? sizeof(UINT) : sizeof(std::uint16_t));
        chunk.byte_count = packed.size() * vertex_stride + index_bytes;
        while (prefetch && prefetched_geometry_bytes + pending_bytes + chunk.byte_count > 64u*1024u*1024u) {
            auto oldest = tile_geometry_cache.end();
            for (auto it = tile_geometry_cache.begin(); it != tile_geometry_cache.end(); ++it)
                if (it->second.prefetched && it->second.last_used < tile_geometry_epoch-1 &&
                    (oldest == tile_geometry_cache.end() || it->second.last_used < oldest->second.last_used)) oldest = it;
            if (oldest == tile_geometry_cache.end()) return false;
            prefetched_geometry_bytes -= oldest->second.byte_count;
            tile_geometry_cache_bytes -= oldest->second.byte_count;
            release_geometry_vertex_buffers(oldest->second.buffers);
            tile_geometry_cache.erase(oldest);
            ++frame_tiles_evicted;
            if (cache_evictions != 0xffffffffu) ++cache_evictions;
        }
        if (!make_tile_cache_room(chunk.byte_count)) return false;
        if (prefetch && foreground_pending->load(std::memory_order_relaxed)) return false;
        if(profiling && sampled_geometry_bucket!=tile_geometry_cache_bytes/(32u*1024u*1024u)){
            sampled_geometry_bucket=tile_geometry_cache_bytes/(32u*1024u*1024u);
            char detail[192];sprintf_s(detail,"gpu_request=%zu packed_capacity=%zu index_capacity=%zu narrow_capacity=%zu",
                chunk.byte_count,packed.capacity()*sizeof(packed[0]),indices.capacity()*sizeof(indices[0]),
                narrow_indices.capacity()*sizeof(narrow_indices[0]));
            trace.write("geometry-allocation",detail,true);
            memory_sample("before-geometry-allocation");
        }
        output.reserve(output.size()+1); // allocate before acquiring COM resources
        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = static_cast<UINT>(packed.size() * vertex_stride);
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        D3D11_SUBRESOURCE_DATA initial = {};
        std::vector<std::uint8_t> frozen_vertices;
        initial.pSysMem = packed.data();
        if(natural_vertex) {
            frozen_vertices.resize(packed.size()*76);
            for(std::size_t i=0;i<packed.size();++i){auto const&v=packed[i];
                float data[]={v.x,v.y,v.z,v.world_x,v.world_y,v.world_z,v.world_valid,
                    v.normal_x,v.normal_y,v.normal_z,v.u,v.v,
                    v.material_grass,v.material_plains,v.material_desert,v.material_marsh,
                    v.authored_relief_height,v.authored_relief_blend,v.base_terrain};
                std::memcpy(frozen_vertices.data()+i*76,data,76);
            }
            initial.pSysMem=frozen_vertices.data();
            if(record){record->vertices.resize(packed.size());record->indices=indices;
                std::memcpy(record->vertices.data(),frozen_vertices.data(),frozen_vertices.size());}
        } else if(pickup_profile && compact_feature) {
            frozen_vertices.resize(packed.size()*48);
            for(std::size_t i=0;i<packed.size();++i){auto const& v=packed[i];
                float data[]={v.x,v.y,v.z,v.u,v.v,v.normal_x,v.normal_y,v.normal_z,v.base_terrain,
                    v.world_x,v.world_y,v.world_z};
                std::memcpy(frozen_vertices.data()+i*48,data,48);
            }
            initial.pSysMem=frozen_vertices.data();
        } else if (!pickup_profile) {
            frozen_vertices.resize(packed.size() * vertex_stride);
            for (std::size_t i = 0; i < packed.size(); ++i)
                std::memcpy(frozen_vertices.data() + i*vertex_stride, &packed[i], vertex_stride);
            initial.pSysMem = frozen_vertices.data();
        }
        HRESULT buffer_result=device->CreateBuffer(&desc, &initial, &chunk.buffer);
        if (FAILED(buffer_result)) {
            MEMORYSTATUSEX memory={};memory.dwLength=sizeof(memory);GlobalMemoryStatusEx(&memory);
            char detail[192];sprintf_s(detail,"kind=vertex hr=%08lx bytes=%u virtual_available=%llu",
                static_cast<unsigned long>(buffer_result),desc.ByteWidth,memory.ullAvailVirtual);
            trace.write("mesh-buffer-failed",detail,true);return false;
        }
        desc.ByteWidth = static_cast<UINT>(index_bytes);
        desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
        initial.pSysMem = narrow_indices.empty() ? static_cast<void const*>(indices.data()) : narrow_indices.data();
        buffer_result=device->CreateBuffer(&desc, &initial, &chunk.indices);
        if (FAILED(buffer_result)) {
            MEMORYSTATUSEX memory={};memory.dwLength=sizeof(memory);GlobalMemoryStatusEx(&memory);
            char detail[192];sprintf_s(detail,"kind=index hr=%08lx bytes=%u virtual_available=%llu",
                static_cast<unsigned long>(buffer_result),desc.ByteWidth,memory.ullAvailVirtual);
            trace.write("mesh-buffer-failed",detail,true);
            release(chunk.buffer);
            return false;
        }
        tile_geometry_cache_bytes += chunk.byte_count;
        frame_upload_bytes += chunk.byte_count;
        output.push_back(chunk);
        if(grid_indices)vertices.swap(packed);
        vertices.clear(); // bounded scratch reused for the next tile
        return true;
    }

    CachedVertexChunk project_natural_chunk(CachedVertexChunk chunk, c3x_renderer_tile_v1 const& record) {
        int c=(record.tile_x+record.tile_y)/2,r=(record.tile_x-record.tile_y)/2;
        chunk.natural_projection[0]=float(c);chunk.natural_projection[1]=float(r);
        chunk.natural_projection[2]=float(shadow_tile_width);chunk.natural_projection[3]=float(height);
        if(tight_natural_bounds && chunk.projected_bounds.valid && shadow_tile_height*2==shadow_tile_width){
            auto bounds=chunk.projected_bounds.project(c,r,shadow_tile_width);
            chunk.bounds={bounds[0],bounds[1],bounds[2],bounds[3]};return chunk;
        }
        c3x_renderer::fidelity::GroundProjection projection{c,r,shadow_tile_width*.5f,
            shadow_tile_height*.5f,shadow_tile_width/224.f*.82f,float(height)};
        chunk.bounds={LONG_MAX,LONG_MAX,LONG_MIN,LONG_MIN};
        for(unsigned corner=0;corner<8;++corner){
            float p[3];for(unsigned axis=0;axis<3;++axis)p[axis]=(corner&(1u<<axis))?
                chunk.world_bounds.high[axis]:chunk.world_bounds.low[axis];
            auto v=projection(p[0],p[1],p[2]*112.f);
            chunk.bounds.left=std::min(chunk.bounds.left,LONG(std::floor(v.x))-2);
            chunk.bounds.top=std::min(chunk.bounds.top,LONG(std::floor(v.y))-2);
            chunk.bounds.right=std::max(chunk.bounds.right,LONG(std::ceil(v.x))+2);
            chunk.bounds.bottom=std::max(chunk.bounds.bottom,LONG(std::ceil(v.y))+2);
        }
        return chunk;
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
                auto chunk=natural_world?project_natural_chunk(source_chunk,record):source_chunk;
                footprint.bounds.left = std::min(footprint.bounds.left, static_cast<int>(chunk.bounds.left));
                footprint.bounds.top = std::min(footprint.bounds.top, static_cast<int>(chunk.bounds.top));
                footprint.bounds.right = std::max(footprint.bounds.right, static_cast<int>(chunk.bounds.right));
                footprint.bounds.bottom = std::max(footprint.bounds.bottom, static_cast<int>(chunk.bounds.bottom));
                if(chunk.city_lighting){
                    int radius=int(std::ceil(shadow_tile_width*.85f))+8;
                    footprint.bounds.left=std::min(footprint.bounds.left,int(chunk.bounds.left)-radius);
                    footprint.bounds.right=std::max(footprint.bounds.right,int(chunk.bounds.right)+radius);
                    footprint.bounds.top=std::min(footprint.bounds.top,int(chunk.bounds.top)-radius);
                    footprint.bounds.bottom=std::max(footprint.bounds.bottom,int(chunk.bounds.bottom)+radius);
                }
                if(environment_profile){
                    int shift=int(std::ceil(2*reflection.height_pixels*std::max(0.f,chunk.world_bounds.high[2]-2.5f/112.f)))+4;
                    footprint.bounds.bottom=std::max(footprint.bounds.bottom,int(chunk.bounds.bottom)+shift);
                    footprint.bounds.left=std::min(footprint.bounds.left,int(chunk.bounds.left)-4);
                    footprint.bounds.right=std::max(footprint.bounds.right,int(chunk.bounds.right)+4);
                }
                if(pickup_profile) {
                    float z=std::max(0.f,chunk.world_bounds.high[2]);
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
        if(tile.natural_signature){
            auto natural_tile=tile_geometry_cache.find(tile.natural_signature);
            if(natural_tile!=tile_geometry_cache.end() && natural_tile->second.shared_natural){
                include(natural_tile->second.buffers,true);
            }
        }
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
            for (CachedVertexChunk chunk : source.buffers[layer]) {
                if(natural_world)chunk=project_natural_chunk(chunk,record);
                chunk.translation_x = anchor_x - source.anchor_x;
                chunk.translation_y = anchor_y - source.anchor_y;
                geometry_vertex_buffers[layer].push_back(chunk);
                // Acquire only after the potentially throwing vector growth.
                chunk.buffer->AddRef();
                chunk.indices->AddRef();
            }
        }
        };
        append(tile,false);
        if(tile.natural_signature){
            auto natural_tile=tile_geometry_cache.find(tile.natural_signature);
            if(natural_tile!=tile_geometry_cache.end() && natural_tile->second.shared_natural)
                append(natural_tile->second,true);
        }
    }

    bool restore_viewport_geometry(CachedViewport const& stored,c3x_renderer_frame_v1 const& frame,
                                   c3x_renderer::TerrainFrameSignature const& signature) {
        if(stored.signature.complete!=signature.complete || stored.tile_keys.size()!=frame.tile_count)return false;
        // The complete frame signature already validates content, environment,
        // topology and camera. Only GPU residency/lifetime needs checking again.
        std::vector<CachedTileGeometry*> owners(frame.tile_count,nullptr);
        for(std::size_t i=0;i<owners.size();++i){
            auto key=stored.tile_keys[i];if(!key.second)continue;
            auto range=tile_geometry_cache.equal_range(key.first);
            for(auto it=range.first;it!=range.second;++it)
                if(!it->second.shared_natural && it->second.version==key.second){owners[i]=&it->second;break;}
            if(!owners[i])return false;
            if(owners[i]->natural_signature){
                auto world_mesh=tile_geometry_cache.find(owners[i]->natural_signature);
                if(world_mesh==tile_geometry_cache.end() || !world_mesh->second.shared_natural ||
                   world_mesh->second.version!=owners[i]->natural_version)return false;
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

    bool chunk_intersects_region(CachedVertexChunk const& chunk,ViewportShaderSettings const& settings,
                                 D3D11_RECT const& rect,bool reflected,int radius=0) const {
        int dx=chunk.translation_x+int(settings.translation[0]),dy=chunk.translation_y+int(settings.translation[1]);
        float low=reflected?2*reflection.height_pixels*std::max(0.f,chunk.world_bounds.low[2]-2.5f/112.f):0;
        float high=reflected?2*reflection.height_pixels*std::max(0.f,chunk.world_bounds.high[2]-2.5f/112.f):0;
        return !(chunk.bounds.right+dx+radius<=rect.left || chunk.bounds.left+dx-radius>=rect.right ||
                 chunk.bounds.bottom+dy+high+radius<=rect.top || chunk.bounds.top+dy+low-radius>=rect.bottom);
    }

    void prepare_region_contributors(std::array<std::vector<CachedVertexChunk>,geometry_layer_count> const& buffers) {
        if(region_contributors.ready && region_contributors.tile_width==shadow_tile_width &&
           region_contributors.reflection_height==reflection.height_pixels)return;
        region_contributors.clear();
        try {
            for(unsigned pass=0;pass<2;++pass)for(unsigned layer=0;layer<geometry_layer_count;++layer)
                for(unsigned i=0;i<buffers[layer].size();++i){auto const& chunk=buffers[layer][i];
                    int radius=layer==geometry_city && chunk.city_lighting?int(std::ceil(shadow_tile_width*.85f))+8:0;
                    float low=pass?2*reflection.height_pixels*std::max(0.f,chunk.world_bounds.low[2]-2.5f/112.f):0;
                    float high=pass?2*reflection.height_pixels*std::max(0.f,chunk.world_bounds.high[2]-2.5f/112.f):0;
                    if(!region_contributors.add(pass,{layer,i},double(chunk.bounds.left)+chunk.translation_x-radius,
                        double(chunk.bounds.top)+chunk.translation_y+low-radius,
                        double(chunk.bounds.right)+chunk.translation_x+radius,
                        double(chunk.bounds.bottom)+chunk.translation_y+high+radius)){
                        region_contributors.clear();return;
                    }
                }
            region_contributors.tile_width=shadow_tile_width;
            region_contributors.reflection_height=reflection.height_pixels;region_contributors.ready=true;
        }catch(...){region_contributors.clear();}
    }

    void collect_region_receivers(std::array<std::vector<CachedVertexChunk>,geometry_layer_count> const& buffers,
            ViewportShaderSettings const& settings,std::vector<D3D11_RECT> const& rectangles,bool reflected,
            std::vector<c3x_renderer::render_core::SourceShadow::Bounds>& receivers) {
        // The index borrows only the current static assembly. Posed buffers and
        // rejected/unavailable indexes retain the original complete scan.
        std::vector<c3x_renderer::render_core::RegionContributorIndex::Item> candidates;
        bool indexed=false;
        if(composition_receiver_index && &buffers==&geometry_vertex_buffers && rectangles.size()==1 &&
           region_contributors.tile_width==shadow_tile_width && region_contributors.reflection_height==reflection.height_pixels) {
            auto const& rect=rectangles[0];
            try {indexed=region_contributors.query(reflected?1u:0u,rect.left-int(settings.translation[0]),
                rect.top-int(settings.translation[1]),std::max(rect.right-rect.left,rect.bottom-rect.top),candidates);}
            catch(...) {}
        }
        auto append=[&](unsigned layer,CachedVertexChunk const& chunk) {
            if(layer==geometry_shadow)return;
            bool visible=false;
            for(auto const& rect:rectangles)visible=visible || chunk_intersects_region(chunk,settings,rect,reflected);
            if(visible)receivers.push_back(chunk.world_bounds);
        };
        if(indexed)for(auto const& item:candidates)append(item.first,buffers[item.first][item.second]);
        else for(unsigned layer=0;layer<geometry_layer_count;++layer)for(auto const& chunk:buffers[layer])append(layer,chunk);
    }

    bool render_region_key(std::array<std::vector<CachedVertexChunk>,geometry_layer_count> const& buffers,
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
            try{indexed=region_contributors.query(pass,-int(local.translation[0]),-int(local.translation[1]),extent,candidates);}catch(...){}
            auto contribute=[&](unsigned layer,CachedVertexChunk const& chunk){
                if(layer==geometry_city && chunk.city_lighting &&
                   chunk_intersects_region(chunk,local,rect,pass!=0,int(std::ceil(shadow_tile_width*.85f))+8)){
                    auto light=chunk.city_lighting.get();
                    if(std::find(lights.begin(),lights.end(),light)==lights.end())lights.push_back(light);
                }
                if(!chunk_intersects_region(chunk,local,rect,pass!=0))return true;
                if(chunk.animation_texture)return false; // Posed pixels never enter this static cache.
                if(layer!=geometry_shadow)receivers.push_back(chunk.world_bounds);
                std::uint64_t identity[]={layer,chunk.version,chunk.index_count,std::uint64_t(chunk.index_format),
                    chunk.vertex_stride,chunk.city_material,std::uint64_t(chunk.city_environment)};
                if(!append_region_bytes(key,identity,sizeof(identity)))return false;
                auto effective=local;
                std::copy(std::begin(chunk.natural_projection),std::end(chunk.natural_projection),effective.natural_projection);
                effective.reserved[1]=layer==geometry_underlay?.5f:layer==geometry_bed?4.f:layer==geometry_water?5.f:0.f;
                effective.translation[0]+=float(chunk.translation_x);effective.translation[1]+=float(chunk.translation_y);
                effective.depth_translation=-effective.translation[1]/height;
                if(!append_region_bytes(key,&effective,sizeof(effective)) ||
                   !append_region_bytes(key,chunk.city_atlas,sizeof(chunk.city_atlas)))return false;
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
            std::array<std::vector<CachedVertexChunk>, geometry_layer_count> const & buffers,
            std::vector<D3D11_RECT> const & rectangles, ViewportShaderSettings const & viewport_settings,
            std::atomic<bool> const * cancellation, bool reflection_pass=false) {
        if(pickup_profile && layer<geometry_natural_terrain)context->IASetInputLayout(layer>=geometry_feature?feature_input_layout:input_layout);
        ViewportShaderSettings previous = {};
        bool first = true;
        for (D3D11_RECT const & rect : rectangles) {
        D3D11_RECT scaled=rect;
        if(fidelity_profile){scaled.left*=2;scaled.top*=2;scaled.right*=2;scaled.bottom*=2;}
        context->RSSetScissorRects(1, &scaled);
        for (CachedVertexChunk const & chunk : buffers[layer]) {
            ++frame_bounds_tests;
            if (cancellation && cancellation->load(std::memory_order_relaxed)) return false;
            if(reflection_pass && chunk.animation_texture)continue;
            if(!chunk_intersects_region(chunk,viewport_settings,rect,reflection_pass))continue;
            ViewportShaderSettings settings = viewport_settings;
            std::copy(std::begin(chunk.natural_projection),std::end(chunk.natural_projection),settings.natural_projection);
            if(pickup_profile)settings.reserved[1]=layer==geometry_underlay?.5f:layer==geometry_bed?4.f:layer==geometry_water?5.f:0.f;
            settings.translation[0] += static_cast<float>(chunk.translation_x);
            settings.translation[1] += static_cast<float>(chunk.translation_y);
            settings.depth_translation = -settings.translation[1] / height;
            if (first || std::memcmp(&previous, &settings, sizeof(settings)) != 0) {
                context->UpdateSubresource(viewport_settings_buffer, 0, nullptr, &settings, 0, 0);
                ++frame_parameter_updates;
                previous = settings;
                first = false;
            }
            UINT stride = chunk.vertex_stride, offset = 0;
            context->IASetVertexBuffers(0, 1, &chunk.buffer, &stride, &offset);
            context->IASetIndexBuffer(chunk.indices, chunk.index_format, 0);
            if(chunk.city_material!=0xffffffffu){
                ID3D11SamplerState*samplers[]={natural_wrap,natural_clamp};context->PSSetSamplers(0,2,samplers);
                context->OMSetBlendState(blend_state,nullptr,0xffffffffu);context->OMSetDepthStencilState(depth_state,0);
                cities.bind(context,chunk.city_material,chunk.city_environment,chunk.city_atlas,reflection_pass,false);
                context->DrawIndexed(chunk.index_count,0,0);
                ++frame_draw_calls;
                if(!cities.library.materials[chunk.city_material].ground){
                    cities.bind(context,chunk.city_material,chunk.city_environment,chunk.city_atlas,reflection_pass,true);
                    context->DrawIndexed(chunk.index_count,0,0);
                    ++frame_draw_calls;
                }
                context->OMSetBlendState(blend_state,nullptr,0xffffffffu);context->OMSetDepthStencilState(depth_state,0);
                continue;
            }
            if(city_profile && layer==geometry_city){
                context->IASetInputLayout(feature_input_layout);
                context->VSSetShader(reflection_pass?reflection.vs[1]:feature_vertex_shader,nullptr,0);
                context->PSSetShader(reflection_pass?reflection.ps[1]:feature_pixel_shader,nullptr,0);
                context->PSSetShaderResources(116,4,city_emissive_views.data());context->PSSetShaderResources(124,4,city_base_views.data());
                ID3D11SamplerState*samplers[]={terrain_sampler,decal_sampler};context->PSSetSamplers(0,2,samplers);
            }
            if (chunk.animation_texture) context->PSSetShaderResources(116,1,&chunk.animation_texture);
            context->DrawIndexed(chunk.index_count, 0, 0);
            ++frame_draw_calls;
            if (chunk.animation_texture) context->PSSetShaderResources(116,1,resource_texture_views.data());
        }
        }
        if(city_profile && layer==geometry_city){ID3D11SamplerState*samplers[]={terrain_sampler,decal_sampler};context->PSSetSamplers(0,2,samplers);}
        return true;
    }

    D3D11_RECT guarded_block_rectangle(D3D11_RECT const& rect, int dx, int dy, int guard, int extent, int extent_y=0) {
        if(!extent_y)extent_y=extent;
        return {std::max<LONG>(0,rect.left+dx-guard),std::max<LONG>(0,rect.top+dy-guard),
                std::min<LONG>(extent,rect.right+dx+guard),std::min<LONG>(extent_y,rect.bottom+dy+guard)};
    }

    void collect_shadow_casters(
            std::array<std::vector<CachedVertexChunk>,geometry_layer_count> const & buffers,
            std::vector<c3x_renderer::render_core::SourceShadow::Caster> & casters) {
        using Shadow=c3x_renderer::render_core::SourceShadow;
        auto dims=world_coast.world().dimensions();
        for(unsigned layer=0;layer<geometry_layer_count;++layer)for(auto const& chunk:buffers[layer]) {
            bool caster=layer==geometry_land || (layer>=geometry_feature && layer!=geometry_natural_decal);
            if(chunk.city_material!=0xffffffffu && cities.library.materials[chunk.city_material].ground)caster=false;
            if(!caster || chunk.animation_texture)continue;
            for(int wy=dims.wrap_y?-1:0;wy<=(dims.wrap_y?1:0);++wy)
                for(int wx=dims.wrap_x?-1:0;wx<=(dims.wrap_x?1:0);++wx) {
                    Shadow::Caster c;c.vertices=chunk.buffer;c.indices=chunk.indices;c.count=chunk.index_count;
                    c.index_format=chunk.index_format;
                    c.stride=chunk.vertex_stride;c.layer=layer;c.version=chunk.version;c.bounds=chunk.world_bounds;
                    if(chunk.city_material!=0xffffffffu)c.binding=10000+chunk.city_material;
                    c.offset[0]=float(wx*dims.width+wy*dims.height)*.5f;
                    c.offset[1]=float(wx*dims.width-wy*dims.height)*.5f;casters.push_back(c);
                }
        }
    }

    c3x_renderer::render_core::SourceShadow::PreparedCasters* prepare_shadow_submission(
            std::array<std::vector<CachedVertexChunk>,geometry_layer_count> const& buffers,
            std::vector<c3x_renderer::render_core::SourceShadow::Caster>& casters,
            c3x_renderer::render_core::SourceShadow::PreparedCasters& prepared) {
        ++frame_caster_preparations;
        collect_shadow_casters(buffers,casters);
        char control[8]={};
        bool reuse=!(GetEnvironmentVariableA("C3X_RENDERER_CASTER_BOUNDS_CONTROL",control,sizeof(control)) && std::strcmp(control,"1")==0);
        char dependency_control[8]={};
        bool retain=world_regions && &buffers==&geometry_vertex_buffers &&
            !(GetEnvironmentVariableA("C3X_RENDERER_REGION_DEPENDENCY_CONTROL",dependency_control,sizeof(dependency_control)) && std::strcmp(dependency_control,"1")==0);
        char index_control[8]={};
        bool index=world_regions && &buffers==&geometry_vertex_buffers &&
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

    bool submit_geometry(std::array<std::vector<CachedVertexChunk>, geometry_layer_count> const & buffers,
                         std::vector<D3D11_RECT> const & rectangles, ViewportShaderSettings const & settings,
                         ID3D11RenderTargetView * target, ID3D11DepthStencilView * depth,
                         int projection_width, int projection_height,
                         std::atomic<bool> const * cancellation = nullptr,
                         bool accumulate = false, bool finish = true,
                         std::array<std::vector<CachedVertexChunk>,geometry_layer_count> const * shadow_buffers_ptr = nullptr,
                         bool reflection_pass=false,
                         std::vector<c3x_renderer::render_core::SourceShadow::Caster> const * shadow_casters_ptr=nullptr,
                         c3x_renderer::render_core::SourceShadow::PreparedCasters * prepared_casters_ptr=nullptr,
                         int region_size=128,int grid_x=0,int grid_y=0,bool require_linear_backdrop=false) {
        int const region_height=region_size==2240?256:region_size;
        auto& active_glow=region_size==128?city_glow:region_glow;
        auto& active_reflection=region_size==128?reflection:region_reflection;
        // Geometry owners remain pinned throughout this synchronous submission.
        // Camera blocks and reflection passes borrow one immutable caster list;
        // only receivers depend on their current screen rectangle. No list is
        // retained across a frame, content edit, animation update or eviction.
        std::vector<c3x_renderer::render_core::SourceShadow::Caster> submission_casters;
        c3x_renderer::render_core::SourceShadow::PreparedCasters prepared_casters;
        if(pickup_profile && !shadow_casters_ptr) {
            if(cancellation && cancellation->load(std::memory_order_relaxed))return false;
            prepared_casters_ptr=prepare_shadow_submission(shadow_buffers_ptr?*shadow_buffers_ptr:buffers,submission_casters,prepared_casters);
            shadow_casters_ptr=&submission_casters;
        }
        if(fidelity_profile && !reflection_pass) {
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
                            &buffers==&geometry_vertex_buffers;
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
        if(pickup_profile && !(city_profile && region_size==2240)) {
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
        if (pickup_profile) {
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
                int dx=chunk.translation_x+int(settings.translation[0]);
                int dy=chunk.translation_y+int(settings.translation[1]);
                for(auto const& rect:rectangles)
                    reflection_needed=reflection_needed || !(chunk.bounds.right+dx<=rect.left ||
                        chunk.bounds.left+dx>=rect.right || chunk.bounds.bottom+dy<=rect.top || chunk.bounds.top+dy>=rect.bottom);
            }
        }
        if(environment_profile && !reflection_pass && active_reflection.enabled && reflection_needed){
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
        if (pickup_profile) {
            using Shadow=c3x_renderer::render_core::SourceShadow;
            std::vector<Shadow::Bounds> receivers;
            auto const & casters=*shadow_casters_ptr;
            auto const & shadow_buffers=shadow_buffers_ptr ? *shadow_buffers_ptr : buffers;
            collect_region_receivers(shadow_buffers,settings,rectangles,reflection_pass,receivers);
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
            if(!source_shadow.prepare(context,shadow_basis,receivers,casters,bind,cancellation,prepared_casters_ptr)) {
                trace.write("source-shadow-failed","caster pages exceeded budget or preparation interrupted",true);return false;
            }
            QueryPerformanceCounter(&end);
            char message[256];sprintf_s(message,"pages_hit=%u pages_built=%u source_draws=%u casters=%u bytes_cap=134217728 ticks=%lld",
                source_shadow.hits,source_shadow.rebuilt,source_shadow.draws,unsigned(casters.size()),end.QuadPart-start.QuadPart);
            trace.write("source-shadow",message,false);
        }
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
        if(city_profile){
            std::vector<c3x_renderer::city_fidelity::Lighting const*> active;
            for(auto const&chunk:buffers[geometry_city])if(chunk.city_lighting){
                auto pointer=chunk.city_lighting.get();if(std::find(active.begin(),active.end(),pointer)!=active.end())continue;
                bool intersects=false;int radius=int(std::ceil(shadow_tile_width*.85f))+8;
                for(auto const&r:rectangles)intersects=intersects || chunk_intersects_region(chunk,settings,r,reflection_pass,radius);
                if(intersects)active.push_back(pointer);
            }
            if(!cities.lights(context,active)){trace.write("city-composition-failed","facade block capacity; no truncation",true);return false;}
        }

        bool has_cached_geometry = false;
        for (std::vector<CachedVertexChunk> const & layer : buffers)
            has_cached_geometry = has_cached_geometry || !layer.empty();
        if (has_cached_geometry) {
            context->IASetInputLayout(input_layout);
            context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context->VSSetShader(vertex_shader, nullptr, 0);
            context->PSSetShader(pixel_shader, nullptr, 0);
            // Register-for-register match with the frozen approved terrain
            // shader. No production-only palette remains in this binding path.
            std::array<ID3D11ShaderResourceView *, 128> views = {};
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
            context->PSSetShaderResources(0, static_cast<UINT>(views.size()), views.data());
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
                    unsigned provider=layer<=geometry_natural_decal?0:layer==geometry_natural_mountain?1:2;
                    context->OMSetDepthStencilState(layer==geometry_natural_decal?natural.decal_depth:depth_state,0);
                    natural.bind(context,provider,provider==2?layer-geometry_natural_forest0:0);
                    if(provider==0){
                        // The generated face meets these same selected rock
                        // bodies; use their material instead of a brown seam.
                        ID3D11ShaderResourceView* cliff_face[]={cliff_views[0],views[15]};
                        context->PSSetShaderResources(31,2,cliff_face);
                    }
                    if(!draw(static_cast<GeometryLayer>(layer)))return false;
                }
                for(unsigned layer=geometry_natural_forest0;layer<geometry_layer_count;layer++){
                    context->OMSetDepthStencilState(depth_state,0);
                    natural.bind(context,2,layer-geometry_natural_forest0);
                    if(!draw(static_cast<GeometryLayer>(layer)))return false;
                }
                context->OMSetDepthStencilState(depth_state,0);
                context->PSSetShaderResources(0,UINT(views.size()),views.data());
                context->PSSetShaderResources(25,1,&source_shadow.view);
                context->PSSetConstantBuffers(0,1,&terrain_settings_buffer);
                context->VSSetShader(vertex_shader,nullptr,0);context->PSSetShader(pixel_shader,nullptr,0);
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
            if (pickup_profile) {
                context->PSSetShaderResources(17,1,&source_shadow.view);
                context->VSSetShader(feature_vertex_shader, nullptr, 0);
                context->PSSetShader(feature_pixel_shader, nullptr, 0);
                ID3D11SamplerState* cliff_sampler=fidelity_profile?natural_clamp:decal_sampler;
                context->PSSetSamplers(0,1,&cliff_sampler);
                for (unsigned i=0;i<cliff_bundle.assets.size();++i) {
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
        if(world_raster_grid)return;
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
        std::array<std::vector<CachedVertexChunk>,geometry_layer_count> buffers;
        for (auto const & contributor : pending_pixel_block.key) {
            auto found=std::find_if(tile_geometry_cache.begin(),tile_geometry_cache.end(),[&](auto const & item){
                return !item.second.shared_natural && item.second.version==contributor.mesh;
            });
            if(found==tile_geometry_cache.end()) { ++pixel_prepare_cursor; return true; }
            for(std::size_t layer=0;layer<geometry_layer_count;++layer)
                for(auto chunk:found->second.buffers[layer]) {
                    chunk.translation_x=contributor.x; chunk.translation_y=contributor.y;
                    buffers[layer].push_back(chunk);
                }
            if(found->second.natural_signature){
                auto world_mesh=tile_geometry_cache.find(found->second.natural_signature);
                if(world_mesh==tile_geometry_cache.end() || !world_mesh->second.shared_natural ||
                   world_mesh->second.version!=found->second.natural_version){
                    ++pixel_prepare_cursor;return true;
                }
                c3x_renderer_tile_v1 record={};
                record.tile_x=found->second.tile_x;record.tile_y=found->second.tile_y;
                for(std::size_t layer=geometry_natural_terrain;layer<geometry_layer_count;++layer)
                    for(auto chunk:world_mesh->second.buffers[layer]){
                        chunk=project_natural_chunk(chunk,record);
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
                std::uint64_t prewarm_signature = 0) {
        bool const prewarming = prewarm_index >= 0;
        if (prewarming) prepared_footprint = {};
        auto cancelled = [&] { return foreground_pending && foreground_pending->load(std::memory_order_relaxed); };
        if (cancelled()) return false;
        if (prewarming && (static_cast<unsigned>(prewarm_index) >= frame.tile_count ||
            (frame.tiles[prewarm_index].tile_flags & C3X_RENDERER_TILE_PREFETCH) == 0 ||
            !cache_valid || cancelled())) return false;
        LARGE_INTEGER started = {}, finished = {};
        QueryPerformanceCounter(&started);
        frame_geometry_ticks = frame_draw_ticks = frame_readback_ticks = 0;
        frame_cache_path = "tiles";
        if (!prewarming) {
        ++trace.sequence;
        char profile_option[8]={};
        profiling=GetEnvironmentVariableA("C3X_RENDERER_PROFILE",profile_option,sizeof(profile_option)) &&
            std::strcmp(profile_option,"1")==0;
        frame_draw_calls=frame_parameter_updates=frame_bounds_tests=0;
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
            if(fidelity_profile)natural.update_rivers(world_coast.world(),frame.world_topology_revision);
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
        frame_tiles_built = frame_tiles_reused = frame_tiles_evicted = 0;
        frame_natural_hits=0;
        frame_ground_grid_hits=0;
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
            for (std::size_t cache_index = 0; cache_index < viewport_cache.size(); ++cache_index) {
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
        if (!frame_has_resource_animation(frame) && reuse_cached_subset(frame, signature)) {
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
        for (c3x_renderer_u32 i = 0; i < frame.tile_count; ++i)
            if ((frame.tiles[i].tile_flags & C3X_RENDERER_TILE_TOPOLOGY_HALO) == 0 ||
                (frame.tiles[i].tile_flags & C3X_RENDERER_TILE_RENDER) != 0) ++draw_record_count;
        int const base_ground_grid = frame.tile_width >= 96 ?
            (draw_record_count <= 768 ? 16 : 12) : 8;
        c3x_renderer_i64 ground_ticks=0,feature_ticks=0,cliff_ticks=0,upload_ticks=0;
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
        reflection.depth_metric=112.f*.0016f*float(frame.target_height);
        region_reflection.height_pixels=reflection.height_pixels;region_reflection.depth_metric=reflection.depth_metric;
        region_reflection.enabled=reflection.enabled;region_glow.gain=city_glow.gain;
        ViewportShaderSettings viewport_settings = {};
        viewport_settings.translation[0] =
            static_cast<float>(geometry_translation_x);
        viewport_settings.translation[1] =
            static_cast<float>(geometry_translation_y);
        viewport_settings.depth_translation =
            -static_cast<float>(geometry_translation_y) / frame.target_height;
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
        viewport_settings.reserved[0] = static_cast<float>(frame.target_height);
        if (!prewarming) geometry_viewport_settings = viewport_settings;
        if(!prewarming && world_regions){
            region_origin_x=frame.tile_count?std::int64_t(frame.tiles[0].anchor_x)-std::int64_t(frame.tiles[0].tile_x)*frame.tile_width/2:0;
            region_origin_y=frame.tile_count?std::int64_t(frame.tiles[0].anchor_y)-std::int64_t(frame.tiles[0].tile_y)*frame.tile_height/2:0;
            region_context={content_revision,device_generation,std::uint64_t(frame.world_topology_revision),
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
        auto & ground_by_coordinate = topology_cache.ground;
        auto & real_by_coordinate = topology_cache.real;
        auto & relief_by_coordinate = topology_cache.relief;
        auto & surface_by_coordinate = topology_cache.surface;
        auto & tile_by_coordinate = topology_cache.tiles;
        auto & semantic_by_coordinate = topology_cache.semantic;
        auto & river_nodes = topology_cache.rivers;
        std::uint64_t topology_signature = prewarming ? prewarm_signature : signature.complete;
        // The pointer guard distinguishes the immutable idle snapshot from the
        // foreground job vector, which the caller may overwrite during idle work.
        // A full content key is still required when a vector reuses its address.
        if (topology_cache.signature != topology_signature || topology_cache.records != frame.tiles) {
        auto topology_started=std::chrono::steady_clock::now();
        topology_cache.signature = 0; topology_cache.records = nullptr;
        ground_by_coordinate.clear(); real_by_coordinate.clear(); relief_by_coordinate.clear();
        surface_by_coordinate.clear(); tile_by_coordinate.clear(); semantic_by_coordinate.clear(); river_nodes.clear();
        semantic_by_coordinate.reserve(frame.tile_count);
        ground_by_coordinate.reserve(frame.tile_count);
        real_by_coordinate.reserve(frame.tile_count);
        relief_by_coordinate.reserve(frame.tile_count);
        surface_by_coordinate.reserve(frame.tile_count);
        tile_by_coordinate.reserve(frame.tile_count);
        for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
            if ((index & 63u) == 0 && cancelled()) return false;
            c3x_renderer_tile_v1 const & tile = frame.tiles[index];
            if ((tile.tile_flags & (C3X_RENDERER_TILE_RENDER |
                                    C3X_RENDERER_TILE_TOPOLOGY_HALO)) == 0)
                continue;
            int ground = ground_type(tile);
            int relief = relief_type(tile);
            tile_by_coordinate[coordinate_key(tile.tile_x, tile.tile_y)] = &tile;
            semantic_by_coordinate[coordinate_key(tile.tile_x, tile.tile_y)] = tile_topology_signature(tile);
            if (relief == 10 && !volcano_assets_ready)
                relief = -1;
            ground_by_coordinate[coordinate_key(tile.tile_x, tile.tile_y)] =
                ground >= 0 && ground < c3x_renderer::terrain_type_count &&
                terrain_textures[ground].view != nullptr ? ground : -1;
            real_by_coordinate[coordinate_key(tile.tile_x, tile.tile_y)] =
                tile.real_terrain_type;
            relief_by_coordinate[coordinate_key(tile.tile_x, tile.tile_y)] =
                relief >= 0 && relief < c3x_renderer::terrain_type_count &&
                terrain_textures[relief].view != nullptr ? relief : -1;
            int integrated_marsh = marsh_assets_ready &&
                tile.real_terrain_type == 9 && terrain_textures[9].view != nullptr ? 9 : -1;
            surface_by_coordinate[coordinate_key(tile.tile_x, tile.tile_y)] =
                integrated_marsh >= 0 ? integrated_marsh :
                (relief >= 0 && relief < c3x_renderer::terrain_type_count &&
                 terrain_textures[relief].view != nullptr ? relief :
                 (ground >= 0 && ground < c3x_renderer::terrain_type_count &&
                  terrain_textures[ground].view != nullptr ? ground : -1));
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
        topology_cache.signature = topology_signature; topology_cache.records = frame.tiles;
        frame_topology_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-topology_started).count();
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
        auto append_object_shadow = [&](c3x_renderer::FeatureAsset const & asset,
                                        float scale, float center_x, float center_y,
                                        float ground_height_screen) {
            if (pickup_profile) return;
            float radius = 0.0f;
            float feature_height = 0.0f;
            for (c3x_renderer::FeatureSourceVertex const & vertex : asset.vertices) {
                radius = std::max(radius, std::sqrt(
                    vertex.position[0] * vertex.position[0] +
                    vertex.position[1] * vertex.position[1]) * scale);
                feature_height = std::max(feature_height, vertex.position[2] * scale);
            }
            float shadow_width = std::max(4.0f, radius * half_w * 0.65f);
            float horizontal = std::sqrt(key_light[0] * key_light[0] +
                                         key_light[1] * key_light[1]);
            float cast_world_x = horizontal > 0.001f ? -key_light[0] / horizontal : 0.0f;
            float cast_world_y = horizontal > 0.001f ? -key_light[1] / horizontal : 1.0f;
            float cast_screen_x = cast_world_x - cast_world_y;
            float cast_screen_y = (cast_world_x + cast_world_y) * half_h / half_w;
            float cast_length = std::sqrt(cast_screen_x * cast_screen_x +
                                          cast_screen_y * cast_screen_y);
            if (cast_length > 0.001f) {
                cast_screen_x /= cast_length;
                cast_screen_y /= cast_length;
            }
            float height_shadow_length = feature_height * 150.0f *
                (static_cast<float>(frame.tile_width) / 224.0f) * 0.72f;
            float shadow_length = std::clamp(
                std::max(shadow_width * 2.40f, height_shadow_length),
                shadow_width * 2.55f, std::min(180.0f, shadow_width * 10.0f));
            float perpendicular_x = -cast_screen_y;
            float perpendicular_y = cast_screen_x;
            float ground_base_screen_y = center_y + ground_height_screen;
            auto make_shadow_vertex = [&](float screen_x, float screen_y, float u, float v) {
                float projected_base_y = ground_base_screen_y + (screen_y - center_y);
                float depth =
                    projected_base_y + ground_height_screen * 0.75f;
                return Vertex{
                    ndc_x(screen_x), ndc_y(screen_y), depth, u, v, 1.0f,
                    0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
                    7.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f,
                    1000.0f, 0.0f, 1000.0f, 0.0f, -1.0f};
            };
            float near_left_x = center_x - perpendicular_x * shadow_width * 0.42f;
            float near_left_y = center_y - perpendicular_y * shadow_width * 0.42f;
            float near_right_x = center_x + perpendicular_x * shadow_width * 0.42f;
            float near_right_y = center_y + perpendicular_y * shadow_width * 0.42f;
            float far_right_x = center_x + cast_screen_x * shadow_length +
                                perpendicular_x * shadow_width * 0.72f;
            float far_right_y = center_y + cast_screen_y * shadow_length +
                                perpendicular_y * shadow_width * 0.72f;
            float far_left_x = center_x + cast_screen_x * shadow_length -
                               perpendicular_x * shadow_width * 0.72f;
            float far_left_y = center_y + cast_screen_y * shadow_length -
                               perpendicular_y * shadow_width * 0.72f;
            Vertex near_left = make_shadow_vertex(near_left_x, near_left_y, 0.0f, 0.0f);
            Vertex near_right = make_shadow_vertex(near_right_x, near_right_y, 1.0f, 0.0f);
            Vertex far_right = make_shadow_vertex(far_right_x, far_right_y, 1.0f, 1.0f);
            Vertex far_left = make_shadow_vertex(far_left_x, far_left_y, 0.0f, 1.0f);
            Vertex triangles[] = {near_left, near_right, far_right,
                                  near_left, far_right, far_left};
            shadow_vertices.insert(shadow_vertices.end(),
                                   std::begin(triangles), std::end(triangles));
        };
        char prefetch_foreground_control[8]={};
        bool const offload_prefetch=pickup_profile && !prewarming &&
            GetEnvironmentVariableA("C3X_RENDERER_PREFETCH_FOREGROUND_CONTROL",
                                   prefetch_foreground_control,sizeof(prefetch_foreground_control)) &&
            (std::strcmp(prefetch_foreground_control,"1")==0 ||
             std::strcmp(prefetch_foreground_control,"2")==0);
        int const prefetch_guard_tiles=std::strcmp(prefetch_foreground_control,"2")==0?2:0;
        for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
            c3x_renderer_tile_v1 const & tile = frame.tiles[index];
            bool const guarded_prefetch=prefetch_guard_tiles!=0 &&
                tile.anchor_x+frame.tile_width>=-prefetch_guard_tiles*frame.tile_width &&
                tile.anchor_x<=frame.target_width+prefetch_guard_tiles*frame.tile_width &&
                tile.anchor_y+frame.tile_height>=-prefetch_guard_tiles*frame.tile_height &&
                tile.anchor_y<=frame.target_height+prefetch_guard_tiles*frame.tile_height;
            if (prewarming ? static_cast<int>(index) != prewarm_index :
                (tile.tile_flags & (C3X_RENDERER_TILE_RENDER |
                    (pickup_profile && (!offload_prefetch || guarded_prefetch) ? C3X_RENDERER_TILE_PREFETCH : 0))) == 0) continue;
            if(!prewarming && pickup_profile && (tile.tile_flags&C3X_RENDERER_TILE_RENDER)==0) {
                // Only currently captured full-appearance records are eligible.
                // The optional wider ring stabilizes prepared region inputs;
                // topology-only records and prior captures are never promoted.
                int mx=frame.tile_width*region_input_ring,my=frame.tile_height*region_input_ring;
                if(tile.anchor_x+frame.tile_width < -mx || tile.anchor_x>frame.target_width+mx ||
                   tile.anchor_y+frame.tile_height < -my || tile.anchor_y>frame.target_height+my)continue;
            }
            if (cancelled()) return false;
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
            std::unordered_map<std::uint64_t, std::uint64_t> dependencies;
            std::unordered_map<std::uint64_t, std::uint64_t> coast_dependencies;
            std::unordered_map<std::size_t, std::uint32_t> world_dependencies;
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
                    auto found = semantic_by_coordinate.find(key);
                    inserted.first->second = found == semantic_by_coordinate.end() ? 0 : found->second;
                }
                return key;
            };
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
            int neighbor_coordinates[4][2] = {
                {tile.tile_x - 1, tile.tile_y - 1}, {tile.tile_x + 1, tile.tile_y - 1},
                {tile.tile_x + 1, tile.tile_y + 1}, {tile.tile_x - 1, tile.tile_y + 1}
            };
            bool relief_neighborhood = relief == 5 || relief == 6 || relief == 10 || draw_dunes;
            for (int edge = 0; edge < 4; ++edge) {
                auto found = relief_by_coordinate.find(observed_coordinate_key(
                    neighbor_coordinates[edge][0], neighbor_coordinates[edge][1]));
                relief_neighborhood = relief_neighborhood || (found != relief_by_coordinate.end() &&
                    (found->second == 5 || found->second == 6 || found->second == 10));
            }
            if (pickup_profile) {
                int c = (tile.tile_x + tile.tile_y)/2, r = (tile.tile_x - tile.tile_y)/2;
                for (int dy=-1;dy<=1;dy++) for (int dx=-1;dx<=1;dx++) {
                    int real = world_lookup(c+dx,r+dy).real;
                    relief_neighborhood = relief_neighborhood || real==5 || real==6 || real==10 || real==0;
                }
            }
            // Keep close authored relief dense when the viewport contains only
            // a few hundred tiles, then spend the same geometry budget across
            // wider views.  Unbounded 24x24 patches can exhaust the 32-bit
            // preview/game process and exceed D3D11's per-buffer size limit.
            int const relief_grid = frame.tile_width >= 96 ?
                (draw_record_count <= 512 ? 24 : (draw_record_count <= 768 ? 16 : 12)) :
                (draw_record_count <= 2048 ? 12 : 8);
            int const tile_ground_grid = pickup_profile ?
                (relief_neighborhood ? (frame.tile_width>=96?24:12) : (frame.tile_width>=96?12:8)) :
                relief_neighborhood ? relief_grid : base_ground_grid;
            bool coast_detail=false;
            if(pickup_profile){
                auto shore_started=std::chrono::steady_clock::now();
                c3x_renderer::render_core::ShoreSample center;
                if(retain_center_shore){
                    try{center=center_shore_cache.get(world_coast,tile.tile_x,tile.tile_y,observe_world,observe_coast);queries.prime_center(center);}
                    catch(...){center=shore_sample_at(queries.center_u,queries.center_v);}
                }else center=shore_sample_at(queries.center_u,queries.center_v);
                coast_detail=std::abs(center.distance)<1.5;
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
            mix_tile(frame.target_width); mix_tile(frame.target_height);
            mix_tile(frame.tile_width); mix_tile(frame.tile_height);
            mix_tile(frame.world_wrap_x ? frame.world_width_tiles : 0);
            mix_tile(frame.world_wrap_y ? frame.world_height_tiles : 0);
            mix_tile(frame.world_wrap_x); mix_tile(frame.world_wrap_y);
            mix_tile(content_revision); mix_tile(device_generation);
            if (pickup_profile) {
                mix_tile(c3x_renderer::render_core::render_core_revision);
                mix_tile(frame.world_width_tiles); mix_tile(frame.world_height_tiles);
            }
            if (!pickup_profile) { mix_tile(frame.hour); mix_tile(frame.season); }
            mix_tile(tile_ground_grid); mix_tile(shadow_grid);
            if(pickup_profile)mix_tile(flat_grid);
            std::vector<RiverNode const *> local_river_nodes;
            if (river_assets_ready && (tile.river_code & 170u) != 0) {
                // The shader's source/junction/mouth responses vanish by 24 px.
                // Include that radius plus the whole tile rectangle's diameter:
                // if any point responds, its nearest node is closer than every
                // omitted node at ALL vertices, preserving interpolation too.
                c3x_renderer::RiverNodeWindow node_window(frame.tile_width, frame.tile_height);
                for (RiverNode const & node : river_nodes)
                    if (node_window.contains(node.lattice_x-tile.tile_x, node.lattice_y-tile.tile_y))
                        local_river_nodes.push_back(&node);
                std::sort(local_river_nodes.begin(), local_river_nodes.end(), [](auto a, auto b) {
                    return a->lattice_y != b->lattice_y ? a->lattice_y < b->lattice_y : a->lattice_x < b->lattice_x;
                });
                for (auto node : local_river_nodes) {
                    mix_tile(node->lattice_x); mix_tile(node->lattice_y);
                    mix_tile(node->degree); mix_tile(node->touches_water);
                }
            }
            auto validation_started=std::chrono::steady_clock::now();
            auto candidates = tile_geometry_cache.equal_range(tile_signature);
            bool reused_tile = false;
            for (auto cached_tile = candidates.first; cached_tile != candidates.second; ++cached_tile) {
                bool valid = !cached_tile->second.shared_natural;
                if(cached_tile->second.natural_signature){
                    auto shared=tile_geometry_cache.find(cached_tile->second.natural_signature);
                    valid=valid && shared!=tile_geometry_cache.end() && shared->second.shared_natural &&
                        shared->second.version==cached_tile->second.natural_version;
                }
                for (auto const & dependency : cached_tile->second.dependencies) {
                    auto current = semantic_by_coordinate.find(dependency.first);
                    auto value = current == semantic_by_coordinate.end() ? 0 : current->second;
                    if (value != dependency.second) { valid = false; break; }
                }
                for (auto const & dependency : cached_tile->second.coast_dependencies)
                    if (world_coast.node_revision(dependency.first) != dependency.second) { valid = false; break; }
                for (auto const & dependency : cached_tile->second.world_dependencies)
                    if (world_coast.world().at(dependency.first) != dependency.second) { valid = false; break; }
                for (auto const & dependency : cached_tile->second.anchor_dependencies) {
                    auto current = tile_by_coordinate.find(dependency.first);
                    if (current == tile_by_coordinate.end() ||
                        current->second->anchor_x - tile.anchor_x != dependency.second[0] ||
                        current->second->anchor_y - tile.anchor_y != dependency.second[1]) valid = false;
                }
                if (valid) {
                    auto append_started=std::chrono::steady_clock::now();
                    frame_tile_validation_ms+=std::chrono::duration<double,std::milli>(append_started-validation_started).count();
                    if (prewarming) {
                        cached_tile->second.last_used = std::max(cached_tile->second.last_used, tile_geometry_epoch-1);
                        prepared_footprint = tile_footprint(cached_tile->second, tile);
                        ++frame_tiles_reused; return true;
                    }
                    if (cached_tile->second.replaces_resource) build_replacement[index] |= C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED;
                    geometry_cache.tile_keys[index]={tile_signature,cached_tile->second.version};
                    append_tile_geometry(cached_tile->second, tile, animated_view);
                    frame_tile_append_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-append_started).count();
                    ++frame_tiles_reused;
                    reused_tile = true;
                    break;
                }
            }
            if (reused_tile) continue;
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
                auto base = ground_by_coordinate.find(key);
                auto surface = surface_by_coordinate.find(key);
                auto relief = relief_by_coordinate.find(key);
                NeighborhoodSample value{true,
                    base == ground_by_coordinate.end() ? ground_slot : static_cast<float>(base->second),
                    surface == surface_by_coordinate.end() ? surface_slot : static_cast<float>(surface->second),
                    relief == relief_by_coordinate.end() ? -1 : relief->second};
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
                float map_x = world_u + world_v - 1.0f;
                float map_y = world_u - world_v;
                float half_frequency = frequency * 0.5f;
                float x_component = map_x * half_frequency;
                float y_component = map_y * half_frequency;
                if (frame.world_wrap_x != 0 && frame.world_width_tiles > 0) {
                    float cycles = std::max(1.0f, std::round(frame.world_width_tiles * half_frequency));
                    float canonical_map_x = static_cast<float>(canonical_component(
                        tile.tile_x, frame.world_width_tiles, frame.world_wrap_x)) + map_x - tile.tile_x;
                    x_component = cycles * canonical_map_x / static_cast<float>(frame.world_width_tiles);
                }
                if (frame.world_wrap_y != 0 && frame.world_height_tiles > 0) {
                    float cycles = std::max(1.0f, std::round(frame.world_height_tiles * half_frequency));
                    float canonical_map_y = static_cast<float>(canonical_component(
                        tile.tile_y, frame.world_height_tiles, frame.world_wrap_y)) + map_y - tile.tile_y;
                    y_component = cycles * canonical_map_y / static_cast<float>(frame.world_height_tiles);
                }
                return std::array<float, 2>{x_component + y_component, y_component - x_component};
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
            auto river_node_distance = [&](float u, float v, unsigned node_kind) {
                if(fidelity_profile && node_kind!=1){auto sample=natural.river_sample({(tile.tile_x+tile.tile_y)*.5+u,(tile.tile_x-tile.tile_y)*.5+1-v});return float(node_kind==0?sample.source:sample.mouth);}
                float point_x = static_cast<float>(tile.tile_x) + u - v;
                float point_y = static_cast<float>(tile.tile_y) + u + v - 1.0f;
                float distance = 1000.0f;
                for (auto node_pointer : local_river_nodes) {
                    RiverNode const & node = *node_pointer;
                    bool selected = node_kind == 0u
                        ? node.degree == 1u && !node.touches_water
                        : (node_kind == 1u ? node.degree >= 3u
                                          : node.degree == 1u && node.touches_water);
                    if (!selected)
                        continue;
                    float delta_x = (point_x - static_cast<float>(node.lattice_x)) * half_w;
                    float delta_y = (point_y - static_cast<float>(node.lattice_y)) * half_h;
                    distance = std::min(distance,
                        std::sqrt(delta_x * delta_x + delta_y * delta_y));
                }
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
                auto value = world.at(world.index(c,r));
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
                auto owner_found = tile_by_coordinate.find(observed_coordinate_key(
                    owner_u + owner_v, owner_u - owner_v));
                c3x_renderer_tile_v1 const & height_tile =
                    owner_found == tile_by_coordinate.end() ? tile : *owner_found->second;
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
                    auto neighbor = ground_by_coordinate.find(observed_coordinate_key(
                        height_tile.tile_x + source_offsets[edge][0],
                        height_tile.tile_y + source_offsets[edge][1]));
                    bool compatible = neighbor != ground_by_coordinate.end() &&
                        neighbor->second >= 0 && neighbor->second < 11;
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
                            auto neighbor = relief_by_coordinate.find(observed_coordinate_key(
                                height_tile.tile_x + source_offsets[edge][0],
                                height_tile.tile_y + source_offsets[edge][1]));
                            int neighbor_real = neighbor == relief_by_coordinate.end()
                                ? -1 : neighbor->second;
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
                            auto neighbor = relief_by_coordinate.find(observed_coordinate_key(
                                height_tile.tile_x + source_offsets[edge][0],
                                height_tile.tile_y + source_offsets[edge][1]));
                            if (neighbor != relief_by_coordinate.end() &&
                                neighbor->second == 6)
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
                            auto neighbor = relief_by_coordinate.find(observed_coordinate_key(
                                height_tile.tile_x + source_offsets[edge][0],
                                height_tile.tile_y + source_offsets[edge][1]));
                            int neighbor_real = neighbor == relief_by_coordinate.end()
                                ? -1 : neighbor->second;
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
                        auto neighbor_ground = ground_by_coordinate.find(key);
                        auto neighbor_real = real_by_coordinate.find(key);
                        bool continues = neighbor_ground != ground_by_coordinate.end() &&
                            neighbor_real != real_by_coordinate.end() &&
                            neighbor_ground->second == 0 && neighbor_real->second == 0;
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
            auto cast_shadow_visibility = [&](float world_u, float world_v,
                                              float origin_height) {
                float horizontal = std::sqrt(key_light[0] * key_light[0] +
                                             key_light[1] * key_light[1]);
                if (horizontal < 0.001f) {
                    return 1.0f;
                }
                float direction_u = key_light[0] / horizontal;
                float direction_v = -key_light[1] / horizontal;
                float perpendicular_u = -direction_v;
                float perpendicular_v = direction_u;
                float occlusion = 0.0f;
                for (int lane = -1; lane <= 1; ++lane) {
                    float greatest_obstruction = 0.0f;
                    float lane_offset = static_cast<float>(lane) * 0.075f;
                    for (int step = 1; step <= 48; ++step) {
                        float distance = static_cast<float>(step) * 0.12f;
                        float sample_u = world_u + direction_u * distance +
                                         perpendicular_u * lane_offset;
                        float sample_v = world_v + direction_v * distance +
                                         perpendicular_v * lane_offset;
                        float ray_height = origin_height + 96.0f * distance + 0.8f;
                        // The frozen production height path is bounded: normalized
                        // relief <=104, smooth maximum adds <=3, dunes <=18.6;
                        // hill height <=52 and river carving only lowers it.
                        // Above 128 no later sample can obstruct this rising ray.
                        // This preserves the 48-step result without sampling the
                        // far-away terrain that cannot contribute to its shadow.
                        if (ray_height >= 128.0f) break;
                        float sample_height = relief_at_world(sample_u, sample_v)[0];
                        greatest_obstruction = std::max(
                            greatest_obstruction, sample_height - ray_height);
                        // The lane's final occlusion is already saturated;
                        // subsequent maxima cannot change the result.
                        if (greatest_obstruction >= 10.0f) break;
                    }
                    occlusion += std::clamp(
                        (greatest_obstruction - 0.5f) / 12.0f, 0.0f, 0.78f);
                }
                float visibility = 1.0f - occlusion / 3.0f;
                return visibility;
            };
            std::unordered_map<std::uint64_t, GroundPoint> ground_point_cache;
            ground_point_cache.reserve(2048);
            auto ground_point_key = [](float u, float v) {
                std::uint32_t u_bits = 0, v_bits = 0;
                std::memcpy(&u_bits, &u, sizeof(u_bits));
                std::memcpy(&v_bits, &v, sizeof(v_bits));
                return (static_cast<std::uint64_t>(u_bits) << 32) | v_bits;
            };
            auto ground_point_at = [&](float u, float v) -> GroundPoint & {
                std::uint64_t key = ground_point_key(u, v);
                auto found = ground_point_cache.find(key);
                if (found != ground_point_cache.end())
                    return found->second;
                GroundPoint point = {};
                point.u = u;
                point.v = v;
                point.world_u =
                    (static_cast<float>(tile.tile_x + tile.tile_y) * 0.5f) + u;
                point.world_v =
                    (static_cast<float>(tile.tile_x - tile.tile_y) * 0.5f) + (1.0f - v);
                point.local_ground_x = half_w + (u - v) * half_w;
                point.local_ground_y = (u + v) * half_h;
                std::array<float, 2> material_uv =
                    periodic_surface_uv(point.world_u, point.world_v, uv_scale);
                point.material_u = material_uv[0];
                point.material_v = material_uv[1];
                std::array<float, 5> weights =
                    material_weights_for(point.world_u, point.world_v);
                std::copy(weights.begin(), weights.end(), point.material_weights);
                point.signed_shore = signed_shore_distance(
                    point.world_u, point.world_v, u, v);
                if (pickup_profile) point.shore = shore_sample_at(point.world_u,point.world_v);
                point.surface_coordinate = point.signed_shore <= 0.0f
                    ? point.signed_shore
                    : std::sqrt(smoothstep01(point.signed_shore)) *
                        water_family_depth(point.world_u, point.world_v);
                return ground_point_cache.emplace(key, point).first->second;
            };
            auto make_ground_vertex = [&](float u, float v, float layer) {
                GroundPoint & point = ground_point_at(u, v);
                float world_u = point.world_u;
                float world_v = point.world_v;
                bool land_surface = layer > 0.75f && layer < 1.25f;
                bool terrain_conforming_surface = land_surface ||
                    (layer > 8.5f && layer < 10.5f);
                if (terrain_conforming_surface && !point.terrain_ready) {
                    std::array<float, 3> sampled = relief_at_world(world_u, world_v);
                    std::copy(sampled.begin(), sampled.end(), point.relief);
                    constexpr float normal_step = 0.006f;
                    float left_height = pickup_profile ? pickup_height_at(world_u - normal_step, world_v) : relief_at_world(world_u - normal_step, world_v)[0];
                    float right_height = pickup_profile ? pickup_height_at(world_u + normal_step, world_v) : relief_at_world(world_u + normal_step, world_v)[0];
                    float down_height = pickup_profile ? pickup_height_at(world_u, world_v - normal_step) : relief_at_world(world_u, world_v - normal_step)[0];
                    float up_height = pickup_profile ? pickup_height_at(world_u, world_v + normal_step) : relief_at_world(world_u, world_v + normal_step)[0];
                    point.normal_delta[0]=right_height-left_height;
                    point.normal_delta[1]=up_height-down_height;
                    float slope_u = (right_height - left_height) * (pickup_profile ? 1.0f : relief_projection_scale) /
                        (2.0f * normal_step * static_cast<float>(frame.tile_width));
                    float slope_v = (up_height - down_height) * (pickup_profile ? -1.0f : relief_projection_scale) /
                        (2.0f * normal_step * static_cast<float>(frame.tile_width));
                    float length = std::sqrt(slope_u * slope_u + slope_v * slope_v + 1.0f);
                    point.normal[0] = -slope_u / length;
                    point.normal[1] = -slope_v / length;
                    point.normal[2] = 1.0f / length;
                    point.terrain_ready = true;
                }
                std::array<float, 3> relief_sample = terrain_conforming_surface
                    ? std::array<float, 3>{point.relief[0], point.relief[1], point.relief[2]} :
                      std::array<float, 3>{0.0f, 0.0f, 0.0f};
                float h = relief_sample[0] * relief_projection_scale;
                float signed_shore = point.signed_shore;
                if (!pickup_profile && land_surface && h > 0.0f) {
                    float shore_envelope = smoothstep01((-signed_shore - 0.02f) / 0.42f);
                    h *= shore_envelope;
                    relief_sample[1] *= shore_envelope;
                    relief_sample[2] *= shore_envelope;
                }
                float ground_x = left + point.local_ground_x;
                float ground_y = top + point.local_ground_y;
                // Elevation moves toward the isometric camera as well as up on
                // screen.  Keeping flat-ground depth made steep micro-quads
                // fold over one another and appear as bright contour seams.
                float depth = ground_y + h * 0.75f;
                float normal_x = terrain_conforming_surface ? point.normal[0] : 0.0f;
                float normal_y = terrain_conforming_surface ? point.normal[1] : 0.0f;
                float normal_z = terrain_conforming_surface ? point.normal[2] : 1.0f;
                float surface_coordinate = point.surface_coordinate;
                float shadow_visibility = !pickup_profile && layer > 9.5f
                    ? cast_shadow_visibility(world_u, world_v, relief_sample[0]) : 1.0f;
                // River topology is consumed only by the river surface pass.
                // Computing its curved-edge and global node distances for the
                // four terrain passes and the shadow pass was pure discarded
                // work, and scaled especially badly with a full Civ III view.
                bool river_surface = layer > 8.5f && layer < 9.5f;
                float river_surface_distance = river_surface
                    ? river_distance(tile, u, v) : 1000.0f;
                auto owner_material = pickup_profile && terrain_conforming_surface
                    ? pickup_ground_at(world_u,world_v).owner : std::array<float,4>{};
                return Vertex{
                    ndc_x(ground_x), ndc_y(ground_y - h), depth,
                    point.material_u, point.material_v,
                    1.0f, normal_x, normal_y, normal_z,
                    shadow_visibility, 1.0f, world_u * 0.5f, world_v * 0.5f,
                    layer, surface_coordinate,
                    static_cast<float>(tile.terrain_type),
                    static_cast<float>(tile.real_terrain_type),
                    point.material_weights[0], point.material_weights[1],
                    point.material_weights[2], point.material_weights[3],
                    relief_sample[1], relief_sample[2], signed_shore,
                    river_surface_distance,
                    river_surface ? river_node_distance(u, v, 1u) : 1000.0f,
                    river_surface ? river_node_distance(u, v, 2u) : 1000.0f,
                    river_surface ? river_node_distance(u, v, 0u) : 1000.0f,
                    tile.has_effect != 0 ? 1.0f : 0.0f,
                    point.material_weights[4],
                    world_u, world_v, (relief_sample[0]+2.5f)/112.0f, layer<9.5f ? 1.0f : 0.0f,
                    static_cast<float>(point.shore.distance), static_cast<float>(point.shore.beach_width),
                    static_cast<float>(point.shore.rocky), static_cast<float>(point.shore.depth),
                    owner_material[0], owner_material[1], owner_material[2], owner_material[3]
                };
            };
            // Retain the actual wrapped occurrence: vertices contain raw world
            // coordinates/UVs, not just canonical gameplay identity.
            auto ground_key=(std::uint64_t(std::uint32_t(tile.tile_x))<<32)|std::uint32_t(tile.tile_y);
            std::uint64_t ground_signature=tile_content_signature(tile);
            for(auto value:{content_revision,std::uint64_t(frame.world_topology_revision),
                    std::uint64_t(frame.world_width_tiles),std::uint64_t(frame.world_height_tiles),
                    std::uint64_t(frame.world_wrap_x),std::uint64_t(frame.world_wrap_y)})
                ground_signature=(ground_signature^value)*1099511628211ull;
            auto retained_ground=ground_grid_cache.find(ground_key);
            bool ground_hit=retain_ground_grids && !prewarming && retained_ground!=ground_grid_cache.end() &&
                retained_ground->second.signature==ground_signature;
            if(ground_hit)for(auto const& dependency:retained_ground->second.dependencies){
                auto found=semantic_by_coordinate.find(dependency.first);
                if((found==semantic_by_coordinate.end()?0:found->second)!=dependency.second){ground_hit=false;break;}
            }
            if(ground_hit)for(auto const& dependency:retained_ground->second.coast_dependencies)
                if(world_coast.node_revision(dependency.first)!=dependency.second){ground_hit=false;break;}
            if(ground_hit)for(auto const& dependency:retained_ground->second.world_dependencies)
                if(world_coast.world().at(dependency.first)!=dependency.second){ground_hit=false;break;}
            if(ground_hit){
                auto& cached=retained_ground->second;cached.used=tile_geometry_epoch;
                dependencies.insert(cached.dependencies.begin(),cached.dependencies.end());
                coast_dependencies.insert(cached.coast_dependencies.begin(),cached.coast_dependencies.end());
                world_dependencies.insert(cached.world_dependencies.begin(),cached.world_dependencies.end());
            }else if(retain_ground_grids && !prewarming && retained_ground!=ground_grid_cache.end()){
                ground_grid_cache_bytes-=retained_ground->second.bytes;ground_grid_cache.erase(retained_ground);
            }
            std::vector<CachedGroundGrid> pending_ground_grids;
            auto append_ground_layer = [&](std::vector<Vertex> & target, float layer,
                                           int subdivisions, std::vector<UINT>* indices=nullptr) {
                // Adjacent cells share grid corners. Build and upload each
                // corner once, then preserve the original triangle-list order.
                // This avoids repeated terrain/shadow evaluation for a point.
                int const row_width = subdivisions + 1;
                std::vector<Vertex> expanded_corners;
                auto& grid_vertices=indices?target:expanded_corners;
                grid_vertices.resize(static_cast<std::size_t>(row_width) * row_width);
                CachedGroundGrid const* cached_grid=nullptr;
                int cached_stride=0;
                if(ground_hit)for(auto const& grid:retained_ground->second.grids)
                    if(grid.layer==layer && (grid.divisions==subdivisions || reuse_nested_ground_grids)){
                        int stride=grid.sample_stride(subdivisions);
                        if(stride){cached_grid=&grid;cached_stride=stride;break;}
                    }
                bool record=retain_ground_grids && !prewarming && indices && !cached_grid;
                CachedGroundGrid pending;
                if(record){pending.layer=layer;pending.divisions=subdivisions;pending.samples.resize(grid_vertices.size());}
                if(cached_grid)++frame_ground_grid_hits;
                for (int grid_v = 0; grid_v <= subdivisions; ++grid_v) {
                    if (cancelled()) return;
                    for (int grid_u = 0; grid_u <= subdivisions; ++grid_u) {
                        float u = static_cast<float>(grid_u) / subdivisions;
                        float v = static_cast<float>(grid_v) / subdivisions;
                        auto at=static_cast<std::size_t>(grid_v)*row_width+grid_u;
                        if(cached_grid){
                            auto source_at=static_cast<std::size_t>(grid_v*cached_stride)*(cached_grid->divisions+1)+grid_u*cached_stride;
                            auto vertex=cached_grid->project(source_at,frame.tile_width,frame.tile_height);
                            if(layer==9.f)vertex.river_branch_count=river_node_distance(u,v,1u);
                            grid_vertices[at]=vertex;
                        }else{
                            grid_vertices[at]=make_ground_vertex(u,v,layer);
                            if(record){auto const& point=ground_point_at(u,v);
                                pending.samples[at]={layer==1.f || layer==9.f?point.relief[0]:0.f,
                                    point.normal_delta[0],point.normal_delta[1]};}
                        }
                    }
                }
                if(record){pending.vertices=grid_vertices;pending_ground_grids.push_back(std::move(pending));}
                if(indices){
                    indices->clear();indices->reserve(std::size_t(subdivisions)*subdivisions*6);
                    for(int y=0;y<subdivisions;++y)for(int x=0;x<subdivisions;++x){
                        UINT a=UINT(y*row_width+x),b=a+1,c=b+UINT(row_width),d=a+UINT(row_width);
                        UINT triangles[]={a,b,c,a,c,d};
                        indices->insert(indices->end(),std::begin(triangles),std::end(triangles));
                    }
                    return;
                }
                for (int grid_v = 0; grid_v < subdivisions; ++grid_v) {
                    for (int grid_u = 0; grid_u < subdivisions; ++grid_u) {
                        auto vertex_at = [&](int x, int y) -> Vertex const & {
                            return grid_vertices[
                                static_cast<std::size_t>(y) * row_width + x];
                        };
                        Vertex const & a = vertex_at(grid_u, grid_v);
                        Vertex const & b0 = vertex_at(grid_u + 1, grid_v);
                        Vertex const & c = vertex_at(grid_u + 1, grid_v + 1);
                        Vertex const & d = vertex_at(grid_u, grid_v + 1);
                        Vertex triangles[] = {a, b0, c, a, c, d};
                        target.insert(target.end(), std::begin(triangles), std::end(triangles));
                    }
                }
            };
            auto append_feature_instance = [&](c3x_renderer::FeatureBundle const & bundle,
                                               c3x_renderer::FeaturePlacement const & placement,
                                               float local_u, float local_v, float rotation,
                                               float scale, float material_offset,
                                               float owner_code, bool cast_shadow,
                                               std::vector<Vertex> & target) {
                if (placement.asset_index >= bundle.assets.size())
                    return;
                c3x_renderer::FeatureAsset const & asset = bundle.assets[placement.asset_index];
                float tile_world_u = static_cast<float>(tile.tile_x + tile.tile_y) * 0.5f;
                float tile_world_v = static_cast<float>(tile.tile_x - tile.tile_y) * 0.5f;
                std::array<float, 3> ground_sample = relief_at_world(
                    tile_world_u + local_u, tile_world_v + (1.0f - local_v));
                if(pickup_profile && &bundle==&site_bundle)
                    ground_sample[0]=natural_height_at(
                        tile_world_u+local_u,tile_world_v+1.f-local_v)-2.5f;
                float center_x = left + half_w + (local_u - local_v) * half_w;
                float center_y = top + (local_u + local_v) * half_h -
                    ground_sample[0] * relief_projection_scale;
                if (cast_shadow)
                    append_object_shadow(asset, scale, center_x, center_y,
                                         ground_sample[0] * relief_projection_scale);
                float cosine = std::cos(rotation);
                float sine = std::sin(rotation);
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
                    float ground_height_pixels = ground_sample[0] * relief_projection_scale;
                    float base_ground_y = center_y + ground_height_pixels +
                        (local_x + local_y) * half_h;
                    float feature_height_tiles = local_z * 150.0f *
                        feature_projection_scale / relief_projection_scale;
                    float depth =
                        base_ground_y + ground_height_pixels * 0.75f +
                        feature_height_tiles * 0.0012f * static_cast<float>(frame.target_height);
                    transformed[vertex_index] = Vertex{
                        ndc_x(screen_x), ndc_y(screen_y), depth,
                        source.uv[0], source.uv[1], 1.0f,
                        normal_x, normal_y, source.normal[2],
                        1.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 0.0f,
                        static_cast<float>(asset.texture_index) + material_offset + owner_code,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 1.0f,
                        1000.0f, 0.0f, 1000.0f, 0.0f, -1.0f};
                    if (pickup_profile) {
                        auto & vertex = transformed[vertex_index];
                        vertex.world_x = tile_world_u + local_u + local_x;
                        vertex.world_y = tile_world_v + 1.0f - local_v - local_y;
                        vertex.world_z = (ground_sample[0] + 2.5f + feature_height_tiles) / 112.0f;
                        vertex.world_valid = 1.0f;
                        auto normal=c3x_renderer::lighting::object_normal(normal_x,normal_y,source.normal[2]);
                        vertex.normal_x=normal[0];vertex.normal_y=normal[1];vertex.normal_z=normal[2];
                    }
                }
                for (std::uint32_t source_index : asset.indices)
                    target.push_back(transformed[source_index]);
            };
            auto append_route_segment = [&](float u0, float v0, float u1, float v1,
                                            unsigned style, bool railroad) {
                constexpr int subdivisions = 16;
                float route_half_width = railroad ? 0.076f : 0.105f;
                float atlas_half_width = railroad ? 0.058f : 0.075f;
                float du = u1 - u0, dv = v1 - v0;
                float original_length = std::sqrt(du * du + dv * dv);
                if (original_length < 0.001f)
                    return;
                float direction_u = du / original_length;
                float direction_v = dv / original_length;
                float original_u0 = u0, original_v0 = v0;
                float original_u1 = u1, original_v1 = v1;
                u0 -= direction_u * 0.14f; v0 -= direction_v * 0.14f;
                u1 += direction_u * 0.14f; v1 += direction_v * 0.14f;
                du = u1 - u0; dv = v1 - v0;
                float length = std::sqrt(du * du + dv * dv);
                float perpendicular_u = -dv / length;
                float perpendicular_v = du / length;
                float atlas_dx = 1.0f;
                float atlas_dy = 0.99021526f - 0.90606654f;
                float atlas_length = std::sqrt(atlas_dx * atlas_dx + atlas_dy * atlas_dy);
                float atlas_perpendicular_u = -atlas_dy / atlas_length;
                float atlas_perpendicular_v = atlas_dx / atlas_length;
                float wave_seed = std::fmod(std::fabs(
                    original_u0 * 17.0f + original_v0 * 31.0f +
                    original_u1 * 47.0f + original_v1 * 61.0f), 19.0f) / 19.0f;
                float wave_phase = wave_seed * 6.28318530718f;
                auto route_vertex = [&](float along, float across) {
                    float source_along = (along * length - 0.14f) / original_length;
                    float curve_t = std::clamp(source_along, 0.0f, 1.0f);
                    float curve_envelope = std::sin(curve_t * 3.14159265359f);
                    float curve_amplitude = railroad ? 0.028f : 0.042f;
                    float road_wave = curve_envelope * curve_amplitude *
                        (0.62f * std::sin(wave_phase) +
                         0.38f * std::sin(curve_t * 6.28318530718f + wave_phase));
                    float route_u = u0 + du * along + perpendicular_u *
                        (route_half_width * across + road_wave);
                    float route_v = v0 + dv * along + perpendicular_v *
                        (route_half_width * across + road_wave);
                    float atlas_u = atlas_dx * source_along +
                        atlas_perpendicular_u * atlas_half_width * across;
                    float atlas_v = 0.90606654f + atlas_dy * source_along +
                        atlas_perpendicular_v * atlas_half_width * across;
                    float tile_world_u = static_cast<float>(tile.tile_x + tile.tile_y) * 0.5f;
                    float tile_world_v = static_cast<float>(tile.tile_x - tile.tile_y) * 0.5f;
                    std::array<float, 3> ground_sample = relief_at_world(
                        tile_world_u + route_u, tile_world_v + (1.0f - route_v));
                    float ground_x = left + half_w + (route_u - route_v) * half_w;
                    float ground_y = top + (route_u + route_v) * half_h;
                    float h = ground_sample[0] * relief_projection_scale;
                    float depth =
                        ground_y + h * 0.75f;
                    Vertex vertex{
                        ndc_x(ground_x), ndc_y(ground_y - h), depth,
                        atlas_u, atlas_v, 1.0f, 0.0f, 0.0f, 1.0f,
                        across, curve_t, source_along, 0.90606654f + atlas_dy * source_along,
                        11.0f, 0.0f, static_cast<float>(style), 0.0f,
                        route_u, route_v, 0.0f, 0.0f,
                        0.0f, 0.0f, 1.0f,
                        1000.0f, 0.0f, 1000.0f, 0.0f, -1.0f};
                    if (pickup_profile) {
                        vertex.world_x = tile_world_u + route_u;
                        vertex.world_y = tile_world_v + 1.0f - route_v;
                        vertex.world_z = (ground_sample[0] + 2.5f) / 112.0f;
                        vertex.world_valid = 1.0f;
                    }
                    return vertex;
                };
                for (int segment = 0; segment < subdivisions; ++segment) {
                    float a0 = static_cast<float>(segment) / subdivisions;
                    float a1 = static_cast<float>(segment + 1) / subdivisions;
                    Vertex left0 = route_vertex(a0, -1.0f);
                    Vertex right0 = route_vertex(a0, 1.0f);
                    Vertex right1 = route_vertex(a1, 1.0f);
                    Vertex left1 = route_vertex(a1, -1.0f);
                    Vertex triangles[] = {left0, right0, right1, left0, right1, left1};
                    route_vertices.insert(route_vertices.end(), std::begin(triangles), std::end(triangles));
                }
            };
            // Match the standalone terrain stack exactly: a flat material
            // underlay, raised land, submerged bed, then transparent water.
            // Keeping these pass-major vectors prevents a later land tile
            // from overwriting an earlier neighbor's continuous shoreline.
            QueryPerformanceCounter(&phase_time);
            append_ground_layer(underlay_vertices, 0.5f, flat_grid, &ground_indices[geometry_underlay]);
            if (ground < 11 && (!fidelity_profile || draw_marsh))
                append_ground_layer(land_vertices, 1.0f, tile_ground_grid, &ground_indices[geometry_land]);
            if(!pickup_profile) {
                append_ground_layer(bed_vertices, 4.0f, flat_grid, &ground_indices[geometry_bed]);
                append_ground_layer(water_vertices, 5.0f, flat_grid, &ground_indices[geometry_water]);
            }
            if (river_assets_ready && ((tile.river_code & 170u) != 0 || (fidelity_profile && natural.river_affects((tile.tile_x+tile.tile_y)/2,(tile.tile_x-tile.tile_y)/2))))
                append_ground_layer(river_vertices, 9.0f,
                                    frame.tile_width >= 96 ? 32 : 16, &ground_indices[geometry_river]);
            if (!pickup_profile && ground < 11) {
                // Cast-shadow visibility ray-marches the authored relief field.
                // Retain the approved 16x16 near grid for canonical fixtures,
                // but use the already-approved reduced grid when a live m19
                // capture contains hundreds of companion records.  The shader
                // interpolates visibility across the unchanged terrain body.
                append_ground_layer(shadow_vertices, 10.0f,
                                    shadow_grid);
            }
            if(!pending_ground_grids.empty()){
                CachedGroundTile incoming;
                incoming.signature=ground_signature;incoming.used=tile_geometry_epoch;incoming.x=tile.tile_x;incoming.y=tile.tile_y;
                incoming.grids=std::move(pending_ground_grids);
                incoming.dependencies.assign(dependencies.begin(),dependencies.end());
                incoming.coast_dependencies.assign(coast_dependencies.begin(),coast_dependencies.end());
                incoming.world_dependencies.assign(world_dependencies.begin(),world_dependencies.end());
                if(ground_hit){
                    // Allocate before moving any retained grids. A failed
                    // allocation must leave the old cache entry usable.
                    incoming.grids.reserve(incoming.grids.size()+retained_ground->second.grids.size());
                    for(auto& grid:retained_ground->second.grids)incoming.grids.push_back(std::move(grid));
                    ground_grid_cache_bytes-=retained_ground->second.bytes;ground_grid_cache.erase(retained_ground);
                }
                incoming.bytes=sizeof(CachedGroundTile)+64+incoming.grids.capacity()*sizeof(CachedGroundGrid)+
                    incoming.dependencies.capacity()*sizeof(incoming.dependencies[0])+
                    incoming.coast_dependencies.capacity()*sizeof(incoming.coast_dependencies[0])+
                    incoming.world_dependencies.capacity()*sizeof(incoming.world_dependencies[0]);
                for(auto const& grid:incoming.grids)incoming.bytes+=grid.vertices.capacity()*sizeof(Vertex)+grid.samples.capacity()*sizeof(grid.samples[0]);
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
            QueryPerformanceCounter(&phase_end);ground_ticks+=phase_end.QuadPart-phase_time.QuadPart;
            phase_time=phase_end;
            if (cancelled()) return false;
            if (route_assets_ready && (tile.road_mask != 0 || tile.railroad_mask != 0)) {
                constexpr int route_offsets[4][2] = {
                    {1, -1}, {2, 0}, {1, 1}, {0, 2}
                };
                constexpr unsigned river_edge_bits[4] = {2u, 0u, 8u, 0u};
                constexpr unsigned opposite_river_bits[4] = {32u, 0u, 128u, 0u};
                float base_world_u = static_cast<float>(tile.tile_x + tile.tile_y) * 0.5f;
                float base_world_v = static_cast<float>(tile.tile_x - tile.tile_y) * 0.5f;
                for (int direction = 0; direction < 4; ++direction) {
                    int neighbor_x = tile.tile_x + route_offsets[direction][0];
                    int neighbor_y = tile.tile_y + route_offsets[direction][1];
                    auto found = tile_by_coordinate.find(observed_coordinate_key(neighbor_x, neighbor_y));
                    if (found == tile_by_coordinate.end())
                        continue;
                    c3x_renderer_tile_v1 const & neighbor = *found->second;
                    bool railroad = tile.railroad_mask != 0 && neighbor.railroad_mask != 0;
                    bool road = tile.road_mask != 0 && neighbor.road_mask != 0;
                    if (!railroad && !road)
                        continue;
                    float end_u = (static_cast<float>(neighbor_x + neighbor_y) * 0.5f + 0.5f) -
                        base_world_u;
                    float end_v = 1.0f - ((static_cast<float>(neighbor_x - neighbor_y) * 0.5f + 0.5f) -
                        base_world_v);
                    unsigned style = railroad ? 4u : static_cast<unsigned>(
                        std::clamp(tile.route_style, 0, 3));
                    append_route_segment(0.5f, 0.5f, end_u, end_v, style, railroad);
                    bool bridge = river_edge_bits[direction] != 0 &&
                        (((tile.river_code & river_edge_bits[direction]) != 0) ||
                         ((neighbor.river_code & opposite_river_bits[direction]) != 0));
                    if (bridge) {
                        char const * bridge_style = railroad ? "railroad" :
                            (style >= 3u ? "modern" : (style >= 2u ? "industrial" : "medieval"));
                        std::string group_name = std::string("bridge_") + bridge_style + "_normal";
                        c3x_renderer::FeatureGroup const * bridge_group =
                            c3x_renderer::find_feature_group(bridge_bundle, group_name.c_str());
                        if (bridge_group != nullptr && !bridge_group->placements.empty()) {
                            float rotation = std::atan2(end_v - 0.5f, end_u - 0.5f);
                            c3x_renderer::FeaturePlacement const & placement =
                                bridge_group->placements.front();
                            append_feature_instance(bridge_bundle, placement,
                                (0.5f + end_u) * 0.5f, (0.5f + end_v) * 0.5f,
                                rotation, placement.scale, 13.0f, 0.0f, true,
                                feature_vertices);
                        }
                    }
                }
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
                                feature_projection_scale / relief_projection_scale;
                            float depth =
                                base_ground_y + ground_height_pixels * 0.75f +
                        feature_height_tiles * 0.0012f * static_cast<float>(frame.target_height);
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
                    auto neighbor = tile_by_coordinate.find(
                        observed_coordinate_key(edge.neighbor_x, edge.neighbor_y));
                    if (ground_type(*owner) >= 11 && neighbor != tile_by_coordinate.end() &&
                        ground_type(*neighbor->second) < 11) {
                        owner = neighbor->second;
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
                            auto receiving=tile_by_coordinate.find(observed_coordinate_key(c+r,c-r));
                            if(receiving==tile_by_coordinate.end() || ground_type(*receiving->second)>=11)continue;
                            float u=std::clamp(float(point.x-c),.00001f,.99999f),v=std::clamp(float(r+1-point.y),.00001f,.99999f);
                            if(shore_sample_at(float(point.x),float(point.y)).distance<.065)continue;
                            bool clear=relief_at_world(float(point.x),float(point.y))[0]+2.5f<18;
                            for(auto const&vertex:asset.vertices){river::P q{point.x+(vertex.position[0]*cosine-vertex.position[1]*sine)*scale,
                                point.y-(vertex.position[0]*sine+vertex.position[1]*cosine)*scale};
                                if(natural.river_sample(q).distance<5.5){clear=false;break;}}
                            if(!clear)continue;
                            owner=receiving->second;local_u=u;local_v=v;placed=true;
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
                            feature_projection_scale / relief_projection_scale;
                        float depth =
                            base_ground_y + ground_height_pixels * 0.75f +
                        feature_height_tiles * 0.0012f * static_cast<float>(frame.target_height);
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
            if(site_flags) {
                for(unsigned kind=0;kind<2;++kind) {
                    unsigned flag=kind?C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP:C3X_RENDERER_IMPROVEMENT_GOODY_HUT;
                    if(!(site_flags&flag))continue;
                    unsigned seed=c3x_renderer::stable_hash(tile.variant_seed ^
                        (kind?unsigned(tile.barbarian_tribe_id)*0x9e3779b9u:0u));
                    unsigned buckets[8]={0,1,2,0,1,2,0,1};
                    std::string name=kind?"camp":"hut_"+std::to_string(buckets[seed%8]);
                    auto group=c3x_renderer::find_feature_group(site_bundle,name.c_str());
                    if(!group || group->placements.empty())return false;
                    float rotation=float(seed%4)*1.57079632679f;
                    for(auto const& placement:group->placements) {
                        if(placement.asset_index>=site_bundle.assets.size())return false;
                        append_feature_instance(site_bundle,placement,.5f,.5f,rotation,
                            1.55f,21.f,.18f,false,site_vertices);
                    }
                }
            }
            if (mine_assets_ready && ground < 11 &&
                (tile.improvement_flags & C3X_RENDERER_IMPROVEMENT_MINE) != 0) {
                unsigned era = static_cast<unsigned>(std::clamp(tile.route_style, 0, 3));
                unsigned family = era < 2u ? 0u : 1u;
                unsigned variant = tile.variant_seed % 3u;
                std::string group_name = "mine_" + std::to_string(family * 3u + variant);
                c3x_renderer::FeatureGroup const * group =
                    c3x_renderer::find_feature_group(mine_bundle, group_name.c_str());
                if (group != nullptr && !group->placements.empty()) {
                    float rotation = c3x_renderer::stable_random(
                        static_cast<std::uint32_t>(tile.tile_x * 71 + tile.tile_y * 113) +
                        era * 29u) * 0.48f - 0.24f;
                    for (std::size_t part = 0; part < group->placements.size(); ++part) {
                        c3x_renderer::FeaturePlacement const & placement =
                            group->placements[part];
                        if (placement.asset_index >= mine_bundle.assets.size())
                            continue;
                        c3x_renderer::FeatureAsset const & asset =
                            mine_bundle.assets[placement.asset_index];
                        unsigned emissive_code = 0u;
                        std::size_t marker = asset.id.rfind(":e");
                        if (marker != std::string::npos)
                            emissive_code = static_cast<unsigned>(std::strtoul(
                                asset.id.c_str() + marker + 2u, nullptr, 10));
                        append_feature_instance(mine_bundle, placement,
                            0.5f, 0.5f, rotation, placement.scale, 21.0f,
                            0.01f * static_cast<float>(emissive_code + 1u),
                            part == 0u, mine_vertices);
                    }
                }
            }
            if (farm_assets_ready && ground < 11 &&
                (tile.improvement_flags & C3X_RENDERER_IMPROVEMENT_IRRIGATION) != 0) {
                unsigned era = tile.route_style < 2 ? 0u :
                    static_cast<unsigned>(std::clamp(tile.route_style - 1, 0, 2));
                std::string group_name = "farm_" + std::to_string(era);
                c3x_renderer::FeatureGroup const * group =
                    c3x_renderer::find_feature_group(farm_bundle, group_name.c_str());
                if (group == nullptr || group->placements.empty())
                    return false;
                unsigned connections = 0u;
                for (unsigned bits = tile.irrigation_mask; bits != 0u; bits >>= 1u)
                    connections += bits & 1u;
                bool shadow_emitted = false;
                for (c3x_renderer::FeaturePlacement const & placement : group->placements) {
                    if (placement.asset_index >= farm_bundle.assets.size())
                        return false;
                    c3x_renderer::FeatureAsset const & asset =
                        farm_bundle.assets[placement.asset_index];
                    bool base_part = asset.id.find(":base:") != std::string::npos;
                    bool building_part = asset.id.find(":building:") != std::string::npos;
                    bool crop_part = asset.id.find(":crop:") != std::string::npos;
                    bool include_base = base_part && connections < 4u &&
                        c3x_renderer::stable_hash(
                            static_cast<std::uint32_t>(tile.tile_x * 37 + tile.tile_y * 101)) % 5u == 0u;
                    bool include_building = building_part &&
                        c3x_renderer::stable_hash(
                            static_cast<std::uint32_t>(tile.tile_x * 71 + tile.tile_y * 43)) % 7u == 0u;
                    if (!crop_part && !include_base && !include_building)
                        continue;
                    float scale = crop_part ? 2.22f + 0.025f * static_cast<float>(connections) :
                        (building_part ? 0.94f : 1.05f);
                    unsigned emissive_code = 0u;
                    std::size_t marker = asset.id.rfind(":e");
                    if (marker != std::string::npos)
                        emissive_code = static_cast<unsigned>(std::strtoul(
                            asset.id.c_str() + marker + 2u, nullptr, 10));
                    bool cast_shadow = building_part && !shadow_emitted;
                    append_feature_instance(farm_bundle, placement, 0.5f, 0.5f, 0.0f,
                        scale, 21.0f, 0.01f * static_cast<float>(emissive_code + 1u),
                        cast_shadow, farm_vertices);
                    shadow_emitted = shadow_emitted || cast_shadow;
                }
            }
            if (city_assets_ready && tile.city_id >= 0 && ground < 11) {
                constexpr char const * era_names[] = {
                    "ancient", "medieval", "industrial", "modern"};
                constexpr char const * wall_names[] = {
                    "wall_ancient", "wall_medieval", "wall_industrial"};
                constexpr unsigned counts[] = {4u, 7u, 11u};
                constexpr float radii[] = {0.25f, 0.33f, 0.41f};
                constexpr float size_scales[] = {0.92f, 1.00f, 1.08f};
                constexpr float golden_angle = 2.39996322973f;
                unsigned era = static_cast<unsigned>(std::clamp(tile.city_era, 0, 3));
                unsigned size = static_cast<unsigned>(std::clamp(tile.city_size, 0, 2));
                unsigned culture = static_cast<unsigned>(std::max(0, tile.city_culture_group));
                unsigned owner = static_cast<unsigned>(std::max(0, tile.city_owner_id));
                c3x_renderer::FeatureGroup const * group =
                    c3x_renderer::find_feature_group(city_bundle, era_names[era]);
                if (group != nullptr && !group->placements.empty()) {
                    unsigned component_count = counts[size];
                    for (unsigned slot = 0; slot < component_count; ++slot) {
                        c3x_renderer::FeaturePlacement const & placement = group->placements[
                            (culture + tile.variant_seed + slot) % group->placements.size()];
                        float angle = static_cast<float>(slot) * golden_angle +
                            c3x_renderer::stable_random(tile.variant_seed * 53u + culture * 19u) * 0.72f;
                        float radius = slot == 0u ? 0.0f : radii[size] *
                            std::sqrt(static_cast<float>(slot) /
                                      static_cast<float>(component_count - 1u));
                        float scale = placement.scale * size_scales[size] *
                            (slot == 0u && (tile.city_flags & C3X_RENDERER_CITY_CAPITAL) != 0 ? 1.30f : 1.0f);
                        append_feature_instance(city_bundle, placement,
                            0.5f + std::cos(angle) * radius,
                            0.5f + std::sin(angle) * radius * 0.78f,
                            angle + 0.55f, scale, 29.0f,
                            0.08f * static_cast<float>(owner + 1u), true, city_vertices);
                    }
                }
                if ((tile.city_flags & C3X_RENDERER_CITY_WALLED) != 0) {
                    c3x_renderer::FeatureGroup const * walls = c3x_renderer::find_feature_group(
                        wall_bundle, wall_names[std::min(era, 2u)]);
                    if (walls != nullptr && !walls->placements.empty()) {
                        c3x_renderer::FeaturePlacement const & wall = walls->placements.front();
                        constexpr float offsets[4][3] = {
                            {-0.29f, 0.00f, 0.785398163f},
                            {0.29f, 0.00f, 0.785398163f},
                            {0.00f, -0.23f, -0.785398163f},
                            {0.00f, 0.23f, -0.785398163f},
                        };
                        for (auto const & offset : offsets)
                            append_feature_instance(wall_bundle, wall,
                                0.5f + offset[0], 0.5f + offset[1], offset[2],
                                wall.scale * (size == 0u ? 0.82f : 1.0f), 29.0f,
                                0.08f * static_cast<float>(owner + 1u), true, wall_vertices);
                    }
                }
            }
            QueryPerformanceCounter(&phase_end);feature_ticks+=phase_end.QuadPart-phase_time.QuadPart;phase_time=phase_end;
            if (pickup_profile && cliff_assets_ready && coast_detail) {
                float cu=float(tile.tile_x+tile.tile_y)*.5f;
                float cr=float(tile.tile_x-tile.tile_y)*.5f;
                // Imported cliff bodies use the same vertical projection as
                // every other feature, then convert to the shared relief-world
                // height used by placement, depth, shadows and water.
                float cliff_vertical_basis=150.f/112.f*feature_projection_scale/relief_projection_scale;
                auto placements=c3x_renderer::render_core::cliff_placements(
                    world_coast.world().dimensions(),int(cu),int(cr),world_lookup,
                    [&](int c,int r){ return world_coast.world().index(c,r); },
                    [&](double u,double v){ return natural_height_at(float(u),float(v))-2.5f; },
                    [&](double u,double v){ return shore_sample_at(float(u),float(v)).distance; },
                    [&](unsigned i){ float h=0;for(auto const& v:cliff_bundle.assets[i].vertices)
                        h=std::max(h,v.position[2]*cliff_vertical_basis);return h; },
                    [&](int c,int r){return world_coast.cell(c,r,[&](auto id,auto revision){coast_dependencies.emplace(id,revision);});},
                    [&](bool is_small,unsigned seed){
                        auto group=find_feature_group(cliff_bundle,is_small?"cliff_small":"cliff_large");
                        auto selected=c3x_renderer::select_feature_placement(*group,seed);
                        if(!selected)return c3x_renderer::render_core::CliffRecipe{0,0,0};
                        return c3x_renderer::render_core::CliffRecipe{selected->asset_index,
                            selected->scale,selected->scale_variation};
                    },
                    cancelled);
                c3x_renderer::fidelity::GroundProjection cliff_projection{
                    int(cu),int(cr),half_w,half_h,relief_projection_scale,float(frame.target_height)};
                for(auto const& instance:placements) {
                    auto const& asset=cliff_bundle.assets[instance.asset];
                    c3x_renderer::render_core::CliffTransform transform(instance,cliff_vertical_basis);
                    std::vector<Vertex> transformed(asset.vertices.size());
                    for(std::size_t i=0;i<asset.vertices.size();++i) {
                        auto const& source=asset.vertices[i];auto& v=transformed[i];
                        auto position=transform.position(source.position),normal=transform.normal(source.normal);
                        float wx=position[0],wy=position[1],wz=position[2];
                        auto projected=cliff_projection(wx,wy,wz*112);
                        v.x=left+projected.x;v.y=top+projected.y;v.z=top+projected.z;
                        v.u=source.uv[0];v.v=source.uv[1];v.panel=1;
                        v.normal_x=normal[0];v.normal_y=normal[1];v.normal_z=normal[2];v.base_terrain=.48f;
                        v.shadow_visibility=v.ambient_visibility=1;
                        v.world_x=wx;v.world_y=wy;v.world_z=wz;v.world_valid=1;
                    }
                    for(auto i:asset.indices)cliff_vertices[instance.asset].push_back(transformed[i]);
                }
            }
            // A hit must carry its dependency observations into the ordinary
            // GPU tile cache. Never reuse samples across authoritative edits.
            std::uint64_t natural_key=1469598103934665603ull;
            for(auto value:{std::uint64_t(std::uint32_t(tile.tile_x)),std::uint64_t(std::uint32_t(tile.tile_y)),
                    tile_content_signature(tile),content_revision,std::uint64_t(frame.world_topology_revision),
                    std::uint64_t(frame.world_width_tiles),std::uint64_t(frame.world_height_tiles),
                    std::uint64_t(frame.world_wrap_x),std::uint64_t(frame.world_wrap_y)})
                natural_key=(natural_key^value)*1099511628211ull;
            // Forest exclusions consume neighboring city composition, while
            // the ordinary topology dependency map intentionally omits it.
            if(tile.real_terrain_type==7)for(int dr=-2;dr<=2;++dr)for(int dc=-2;dc<=2;++dc){
                int c=(tile.tile_x+tile.tile_y)/2+dc,r=(tile.tile_x-tile.tile_y)/2+dr;
                auto neighbor=tile_by_coordinate.find(coordinate_key(c+r,c-r));
                auto value=neighbor!=tile_by_coordinate.end() && neighbor->second->city_id>=0?
                    tile_content_signature(*neighbor->second):0;
                natural_key=(natural_key^value)*1099511628211ull;
            }
            // World-space GPU data may outlive any particular camera entry.
            // Pin the owner before ground uploads can evict other entries.
            bool cache_natural=fidelity_profile && tile.city_id<0 && !prewarming;
            bool share_natural=cache_natural && share_world_meshes;
            auto shared_natural=tile_geometry_cache.find(natural_key);
            bool shared_hit=share_natural && shared_natural!=tile_geometry_cache.end() && shared_natural->second.shared_natural;
            if(shared_hit)for(auto const& dependency:shared_natural->second.dependencies){
                auto current=semantic_by_coordinate.find(dependency.first);
                if((current==semantic_by_coordinate.end()?0:current->second)!=dependency.second){shared_hit=false;break;}
            }
            if(shared_hit)for(auto const& dependency:shared_natural->second.coast_dependencies)
                if(world_coast.node_revision(dependency.first)!=dependency.second){shared_hit=false;break;}
            if(shared_hit)for(auto const& dependency:shared_natural->second.world_dependencies)
                if(world_coast.world().at(dependency.first)!=dependency.second){shared_hit=false;break;}
            if(share_natural && !shared_hit && shared_natural!=tile_geometry_cache.end()){
                if(shared_natural->second.shared_natural && shared_natural->second.last_used!=tile_geometry_epoch){
                    tile_geometry_cache_bytes-=shared_natural->second.byte_count;
                    tile_geometry_cache.erase(shared_natural);
                }else share_natural=false; // Do not replace a pinned owner or a colliding ordinary key.
            }
            if(shared_hit){
                auto& cached=shared_natural->second;cached.last_used=tile_geometry_epoch;
                if(animated_view)cached.animation_epoch=tile_geometry_epoch;
                dependencies.insert(cached.dependencies.begin(),cached.dependencies.end());
                coast_dependencies.insert(cached.coast_dependencies.begin(),cached.coast_dependencies.end());
                world_dependencies.insert(cached.world_dependencies.begin(),cached.world_dependencies.end());
            }
            auto natural_found=natural_mesh_cache.find(natural_key);
            bool natural_hit=fidelity_profile && natural_found!=natural_mesh_cache.end();
            if(natural_hit)for(auto const& dependency:natural_found->second.dependencies){
                auto current=semantic_by_coordinate.find(dependency.first);
                if((current==semantic_by_coordinate.end()?0:current->second)!=dependency.second){natural_hit=false;break;}
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
                dependencies.insert(cached.dependencies.begin(),cached.dependencies.end());
                coast_dependencies.insert(cached.coast_dependencies.begin(),cached.coast_dependencies.end());
                world_dependencies.insert(cached.world_dependencies.begin(),cached.world_dependencies.end());
            }else{
                natural_hit=false;
                #include "source_fidelity/geometry.h"
            }
            if(cache_natural && !natural_hit && !shared_hit){
                pending_natural.dependencies.assign(dependencies.begin(),dependencies.end());
                pending_natural.coast_dependencies.assign(coast_dependencies.begin(),coast_dependencies.end());
                pending_natural.world_dependencies.assign(world_dependencies.begin(),world_dependencies.end());
                pending_natural.used=tile_geometry_epoch;pending_natural.tile_width=frame.tile_width;
                pending_natural.tile_height=frame.tile_height;
                pending_natural.target_height=frame.target_height;
                pending_natural.tile_x=tile.tile_x;pending_natural.tile_y=tile.tile_y;
            }
            c3x_renderer::fidelity::GroundProjection natural_projection{(tile.tile_x+tile.tile_y)/2,
                (tile.tile_x-tile.tile_y)/2,half_w,half_h,relief_projection_scale,float(frame.target_height)};
            QueryPerformanceCounter(&phase_end);cliff_ticks+=phase_end.QuadPart-phase_time.QuadPart;phase_time=phase_end;
            if (cancelled()) return false;
            CachedTileGeometry compiled;
            compiled.resource_anchors = std::move(tile_resource_anchors);
            compiled.replaces_resource = (build_replacement[index] & C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED) != 0;
            compiled.signature = tile_signature;
            compiled.tile_x=tile.tile_x;compiled.tile_y=tile.tile_y;
            compiled.version = ++tile_geometry_version;
            compiled.anchor_x = 0;
            compiled.anchor_y = 0;
            compiled.last_used = prewarming ? tile_geometry_epoch - 1 : tile_geometry_epoch;
            compiled.prefetched = prewarming;
            compiled.dependencies.assign(dependencies.begin(), dependencies.end());
            compiled.coast_dependencies.assign(coast_dependencies.begin(), coast_dependencies.end());
            compiled.world_dependencies.assign(world_dependencies.begin(), world_dependencies.end());
            compiled.anchor_dependencies = std::move(anchor_dependencies);
            std::size_t metadata_bytes = pickup_profile
                ? compiled.coast_dependencies.capacity() * sizeof(compiled.coast_dependencies[0]) +
                  compiled.world_dependencies.capacity() * sizeof(compiled.world_dependencies[0]) : 0;
            metadata_bytes += compiled.resource_anchors.capacity()*sizeof(ResourceAnchor);
            if(!city_chunks.empty())metadata_bytes+=sizeof(c3x_renderer::city_fidelity::Lighting)+
                city_chunks.front().lighting->lights.capacity()*sizeof(c3x_renderer::city_fidelity::Light)+
                city_chunks.front().lighting->blockers.capacity()*sizeof(c3x_renderer::city_fidelity::Lighting::Box);
            if (!make_tile_cache_room(metadata_bytes)) return false;
            tile_geometry_cache_bytes += metadata_bytes;
            compiled.byte_count = metadata_bytes;
            try {
            // Compute before cache_geometry_layer moves the flat grid into its
            // immutable buffer. Retained tiles carry the resulting absent
            // layers with the same coast/semantic dependencies as that grid.
            bool water_coverage=!cull_empty_water || !environment_profile ||
                c3x_renderer::render_core::water_surface_can_contribute(underlay_vertices);
            for (std::size_t layer = 0; layer < geometry_layer_count; ++layer) {
                if(layer==geometry_city && !city_chunks.empty()){
                    for(auto&part:city_chunks){
                        if(part.vertices.empty())continue;
                        if(!cache_geometry_layer(part.vertices,compiled.buffers[layer],prewarming,compiled.byte_count,foreground_pending,false,false)){
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
                        chunk.buffer->AddRef();chunk.indices->AddRef();chunk.byte_count=0;
                    }
                    continue;
                }
                bool natural_layer=layer>=geometry_natural_terrain;
                if(natural_layer && shared_hit)continue;
                auto const* cached_mesh=natural_layer && natural_hit?&natural_found->second.layers[layer-geometry_natural_terrain]:nullptr;
                auto* record_mesh=natural_layer && cache_natural && !natural_hit && !retain_ground_grids?&pending_natural.layers[layer-geometry_natural_terrain]:nullptr;
                bool reproject=natural_hit && (natural_found->second.tile_width!=frame.tile_width ||
                    natural_found->second.tile_height!=frame.tile_height || natural_found->second.target_height!=frame.target_height);
                if (!cache_geometry_layer(*tile_layers[layer], compiled.buffers[layer], prewarming, compiled.byte_count, foreground_pending, layer>=geometry_feature && layer<geometry_natural_terrain, natural_layer,
                        record_mesh,cached_mesh,reproject?&natural_projection:nullptr,
                        layer<=geometry_river?&ground_indices[layer]:
                        index_natural_grids && (layer==geometry_natural_terrain || layer==geometry_natural_terrain+2)
                            ?&natural_grid_indices[layer==geometry_natural_terrain?0:1]:nullptr)) {
                    char detail[256];sprintf_s(detail,"tile=%d,%d layer=%u vertices=%u bytes=%llu cap=%llu built=%u reused=%u prewarming=%u",
                        tile.tile_x,tile.tile_y,unsigned(layer),unsigned(tile_layers[layer]->size()),
                        static_cast<unsigned long long>(tile_geometry_cache_bytes),static_cast<unsigned long long>(tile_geometry_cache_budget),
                        frame_tiles_built,frame_tiles_reused,unsigned(prewarming));
                    trace.write("mesh-cache-failed",detail,true);
                    tile_geometry_cache_bytes -= compiled.byte_count;
                    release_geometry_vertex_buffers(compiled.buffers);
                    return false;
                }
                for (auto const & chunk : compiled.buffers[layer]) compiled.byte_count += chunk.byte_count;
            }
            for(auto&indices:natural_grid_indices)indices.clear();
            } catch (...) {
                tile_geometry_cache_bytes -= compiled.byte_count;
                return false; // compiled owns every successfully uploaded buffer
            }
            if(cache_natural && !natural_hit && !shared_hit && !retain_ground_grids){
                pending_natural.bytes=sizeof(NaturalTile)+pending_natural.dependencies.capacity()*sizeof(pending_natural.dependencies[0])+
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
                compiled.natural_signature=natural_key;
                compiled.natural_version=shared_hit?shared_natural->second.version:compiled.version;
                if(!shared_hit){
                    CachedTileGeometry shared;
                    shared.shared_natural=true;shared.signature=natural_key;
                    shared.version=compiled.version;shared.last_used=tile_geometry_epoch;
                    if(animated_view)shared.animation_epoch=tile_geometry_epoch;
                    try {
                        shared.dependencies=compiled.dependencies;
                        shared.coast_dependencies=compiled.coast_dependencies;
                        shared.world_dependencies=compiled.world_dependencies;
                    } catch(...) {
                        tile_geometry_cache_bytes-=compiled.byte_count;return false;
                    }
                    // Count this owner's metadata separately from camera metadata.
                    std::size_t metadata=sizeof(CachedTileGeometry)+
                        shared.dependencies.capacity()*sizeof(shared.dependencies[0])+
                        shared.coast_dependencies.capacity()*sizeof(shared.coast_dependencies[0])+
                        shared.world_dependencies.capacity()*sizeof(shared.world_dependencies[0]);
                    if(!make_tile_cache_room(metadata)){
                        tile_geometry_cache_bytes-=compiled.byte_count;return false;
                    }
                    for(std::size_t layer=geometry_natural_terrain;layer<geometry_layer_count;++layer){
                        shared.buffers[layer]=std::move(compiled.buffers[layer]);
                        for(auto const& chunk:shared.buffers[layer])shared.byte_count+=chunk.byte_count;
                    }
                    compiled.byte_count-=shared.byte_count;
                    shared.byte_count+=metadata;tile_geometry_cache_bytes+=metadata;
                    std::size_t shared_bytes=shared.byte_count;
                    try {tile_geometry_cache.emplace(natural_key,std::move(shared));}
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
            } catch (...) {
                // Container insertion can allocate after GPU upload. RAII owns
                // the buffers even if it moved the value before allocation failed.
                tile_geometry_cache_bytes -= compiled_bytes;
                return false;
            }
            if (prewarming) {
                prefetched_geometry_bytes += inserted->second.byte_count;
                prepared_footprint = tile_footprint(inserted->second, tile);
                return true;
            }
            geometry_cache.tile_keys[index]={tile_signature,inserted->second.version};
            append_tile_geometry(inserted->second, tile, animated_view);
            QueryPerformanceCounter(&phase_end);upload_ticks+=phase_end.QuadPart-phase_time.QuadPart;
        }
        if(pickup_profile && !prewarming) {
            char detail[384];sprintf_s(detail,"built=%u reused=%u ground_ms=%.3f features_ms=%.3f cliffs_ms=%.3f upload_ms=%.3f bytes=%llu natural_hits=%u natural_bytes=%zu ground_grid_hits=%u ground_grid_bytes=%zu",
                frame_tiles_built,frame_tiles_reused,trace.milliseconds(ground_ticks),trace.milliseconds(feature_ticks),
                trace.milliseconds(cliff_ticks),trace.milliseconds(upload_ticks),static_cast<unsigned long long>(tile_geometry_cache_bytes),frame_natural_hits,natural_mesh_cache_bytes,frame_ground_grid_hits,ground_grid_cache_bytes);
            trace.write("mesh-phases",detail,true);
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
        trace.write("geometry-ready", frame_cache_path);
        if(cancelled())return false;
        memory_sample("geometry-ready");
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
        if (cache_valid) {
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
        std::swap(output,other.output);
        pixels.swap(other.pixels);fallback.swap(other.fallback);replacements.swap(other.replacements);
        occurrences.swap(other.occurrences);std::swap(frame,other.frame);std::swap(identity,other.identity);
        std::swap(scene_signature,other.scene_signature);
        std::swap(phase_x,other.phase_x);std::swap(phase_y,other.phase_y);
    }
    std::size_t bytes() const {
        return (pixels.capacity()+fallback.capacity()+replacements.capacity())*sizeof(std::uint32_t)+
            occurrences.capacity()*sizeof(c3x_renderer_tile_v1)+sizeof(PublishedMapFrame);
    }
    bool capture(c3x_renderer_output_v1 const& source,int x,int y,
                 c3x_renderer_frame_v1 const* captured=nullptr,c3x_renderer_camera_identity_v1 const& epochs={}) {
        constexpr std::uint64_t budget=32u*1024u*1024u;
        if(source.width<=0 || source.height<=0 || source.width>8192 || source.height>8192 ||
           source.stride_bytes!=source.width*4 || !source.bgra_pixels ||
           source.fallback_tile_count>8192 || source.replacement_tile_count>8192 ||
           (source.fallback_tile_count && !source.fallback_tile_indices) ||
           (source.replacement_tile_count && !source.replacement_tile_flags))return false;
        auto count=std::uint64_t(source.width)*std::uint64_t(source.height);
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
            next.pixels.assign(first,first+std::size_t(count));
            if(source.fallback_tile_count)next.fallback.assign(source.fallback_tile_indices,
                source.fallback_tile_indices+source.fallback_tile_count);
            if(source.replacement_tile_count)next.replacements.assign(source.replacement_tile_flags,
                source.replacement_tile_flags+source.replacement_tile_count);
            if(captured){
                if(captured->tile_count)next.occurrences.assign(captured->tiles,captured->tiles+captured->tile_count);
                next.frame=*captured;next.identity=epochs;
                next.frame.tiles=next.occurrences.empty()?nullptr:next.occurrences.data();
                next.frame.world_topology=nullptr;next.frame.world_topology_count=0;
            }
            // Account allocated capacity, not just requested payload bytes.
            if(next.bytes()>budget)return false;
            next.output=source;next.phase_x=x;next.phase_y=y;
            pixels.swap(next.pixels);fallback.swap(next.fallback);replacements.swap(next.replacements);
            occurrences.swap(next.occurrences);frame=next.frame;identity=next.identity;
            output=next.output;phase_x=next.phase_x;phase_y=next.phase_y;
            output.bgra_pixels=pixels.data();
            output.fallback_tile_indices=fallback.empty()?nullptr:fallback.data();
            output.replacement_tile_flags=replacements.empty()?nullptr:replacements.data();
            return true;
        } catch (...) {return false;}
    }
    void clear() {
        std::vector<std::uint32_t>().swap(pixels);
        std::vector<std::uint32_t>().swap(fallback);
        std::vector<std::uint32_t>().swap(replacements);
        std::vector<c3x_renderer_tile_v1>().swap(occurrences);frame={};identity={};
        output={};scene_signature=0;phase_x=phase_y=0;
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
    }

    int set_unit_rendering(int enabled) {
        if(enabled!=0 && enabled!=1)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        std::lock_guard<std::mutex> call_guard(call_mutex);
        std::unique_lock<std::mutex> lock(state_mutex);
        drain_camera_locked(lock);
        // Call serialization excludes foreground configure/render jobs. Idle
        // terrain preparation never reads this unit-only configuration value.
        renderer_state.unit_rendering_enabled=enabled!=0;
        renderer_state.trace.write("unit-config",enabled?"enabled; bind at definition load":"disabled; native units",true);
        return C3X_RENDERER_RESULT_OK;
    }

    ~RendererWorker() {
        reset_and_stop();
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

    int render(c3x_renderer_frame_v1 const & frame, c3x_renderer_output_v1 & output) {
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
        if(ambient_async_enabled && completed_result==C3X_RENDERER_RESULT_OK &&
           completed_resources && publication.output.bgra_pixels) {
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
            if(same_ambient_view(frame,job_frame)) {
                auto clock=RendererState::resource_clock(frame);
                if(clock!=completed_resource_clock && !camera_active && !camera_pending &&
                   (camera_result==C3X_RENDERER_RESULT_OK || camera_result==C3X_RENDERER_RESULT_SUPERSEDED)) {
                    c3x_renderer_i64 ignored=0;
                    if(enqueue_camera_locked(frame,{},ignored)!=C3X_RENDERER_RESULT_PENDING) {
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
        drain_camera_locked(lock);
        // Idle geometry/pixel preparation never mutates the published frame
        // bitmap or ownership arrays. An identical authoritative appearance
        // can return that immutable publication without cancelling useful work.
        if(completed_scene_signature && completed_result==C3X_RENDERER_RESULT_OK &&
           (!completed_resources || completed_resource_clock==RendererState::resource_clock(frame)) &&
           c3x_renderer::terrain_frame_signature(frame,completed_output.content_revision,
                completed_output.device_generation).complete==completed_scene_signature) {
            return_current_bitmap(frame,output,"exact-cache");
            return C3X_RENDERER_RESULT_OK;
        }
        foreground_pending.store(true, std::memory_order_relaxed);
        job_frame = frame;
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
        // Completion redraws may submit the same immutable request again. Keep
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
            if(!publication.capture(completed_output,completed_phase_x,completed_phase_y))
                return C3X_RENDERER_RESULT_ERROR;
            completed_output=publication.output;
        }
        start_locked();
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
            publication.swap(camera_ready);
            camera_ready.clear();
            completed_output=publication.output;
            completed_phase_x=publication.phase_x;completed_phase_y=publication.phase_y;
            completed_result=C3X_RENDERER_RESULT_OK;
            camera_front_ticket=ticket;camera_front_result=camera_ready_result;
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

    int draw_unit(c3x_renderer_unit_v1 const & request,HDC destination,HDC background=nullptr,int* bounds=nullptr) {
        std::lock_guard<std::mutex> call_guard(call_mutex);
        std::unique_lock<std::mutex> lock(state_mutex);
        if(!renderer_state.unit_rendering_enabled)return C3X_RENDERER_RESULT_ERROR;
        start_locked();job_unit=request;
        if(bounds) {
            auto const& units=renderer_state.unit_bodies.units;
            auto found=std::find_if(units.begin(),units.end(),[&](auto const& unit){
                return std::find(unit.keys.begin(),unit.keys.end(),request.unit_key)!=unit.keys.end();});
            if(found==units.end())return C3X_RENDERER_RESULT_ERROR;
            int projection=request.projection_scale_milli>0?request.projection_scale_milli:(request.reduced?500:1000);
            if(!c3x_renderer::expand_unit_canvas(job_unit.body_x,job_unit.body_y,job_unit.sprite_width,job_unit.sprite_height,
                                   projection,found->minimum_canvas))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        }
        LARGE_INTEGER started={},finished={};QueryPerformanceCounter(&started);
        // Cached unit pixels are an independent CPU publication. Do not cancel
        // an exact ambient map render merely to copy a pose already in memory.
        if(renderer_state.unit_bodies.restore_cached(job_unit)) {
            int result=renderer_state.unit_bodies.blit(destination,job_unit.body_x,job_unit.body_y,background)
                ? C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_ERROR;
            if(result!=C3X_RENDERER_RESULT_OK)renderer_state.unit_bodies.failure_reason="native-canvas-blit";
            if(result==C3X_RENDERER_RESULT_OK && bounds) {
                bounds[0]=job_unit.body_x;bounds[1]=job_unit.body_y;
                bounds[2]=job_unit.body_x+renderer_state.unit_bodies.image_width;
                bounds[3]=job_unit.body_y+renderer_state.unit_bodies.image_height;
            }
            QueryPerformanceCounter(&finished);
            char detail[384];std::snprintf(detail,sizeof(detail),
                "id=%d key=%.63s action=%d queued=%d cursor=%d/%d dir=%d xy=%d,%d reduced=%d color=%06x result=%d reason=%s cache_hit=1 cache_only=1 cache_bytes=%zu keyed=%u shadow_pixels=%u ms=%.3f",
                request.unit_id,request.unit_key,request.action,request.queued_action,request.action_cursor,request.frame_count,
                request.direction,request.body_x,request.body_y,request.reduced,request.display_color_rgb,result,
                renderer_state.unit_bodies.failure_reason,renderer_state.unit_bodies.cache_bytes,
                renderer_state.unit_bodies.keyed_pixels,renderer_state.unit_bodies.cast_pixels,
                renderer_state.trace.milliseconds(finished.QuadPart-started.QuadPart));
            renderer_state.trace.write("unit-body",detail,true);
            return result;
        }
        {
            char detail[256];std::snprintf(detail,sizeof(detail),
                "id=%d key=%.63s action=%d cursor=%d/%d dir=%d reason=%s entries=%zu bytes=%zu",
                request.unit_id,request.unit_key,request.action,request.action_cursor,request.frame_count,
                request.direction,renderer_state.unit_bodies.failure_reason,
                renderer_state.unit_bodies.cached_pose_entries(),renderer_state.unit_bodies.cache_bytes);
            renderer_state.trace.write("unit-cache-only-miss",detail,true);
        }
        UnitCameraPause pause(*this,lock);
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
        std::unique_lock<std::mutex> call_guard(call_mutex);
        std::unique_lock<std::mutex> lock(state_mutex);
        if (!running)
            return;
        drain_camera_locked(lock);
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
        unit,
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
        benchmark_trim,
#endif
        reset
    };

    RendererState & renderer_state;
    LARGE_INTEGER job_timing_begin={},job_timing_rendered={},job_timing_published={};
    MapBlitter map_blitter;
    PublishedMapFrame publication;
    PublishedMapFrame camera_ready;
    bool camera_preview_enabled=false;
    bool ambient_async_enabled=false;
    c3x_renderer_i64 camera_front_ticket=0;
    int camera_ready_result=C3X_RENDERER_RESULT_OK,camera_front_result=C3X_RENDERER_RESULT_PENDING;
    c3x_renderer_frame_v1 camera_pending_frame={};
    c3x_renderer_camera_identity_v1 camera_pending_identity={},job_camera_identity={};
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
            "phase=%s active_tiles=%zu pending_tiles=%zu warm_tiles=%zu active_topology=%zu pending_topology=%zu warm_topology=%zu front=%zu ready=%zu active=%u pending=%u paused=%u",
            phase,job_tiles.capacity()*sizeof(c3x_renderer_tile_v1),camera_pending_tiles.capacity()*sizeof(c3x_renderer_tile_v1),
            warm_tiles.capacity()*sizeof(c3x_renderer_tile_v1),job_world_topology.capacity()*4,
            camera_pending_topology.capacity()*4,warm_world_topology.capacity()*4,publication.bytes(),camera_ready.bytes(),
            unsigned(camera_active),unsigned(camera_pending),unsigned(camera_paused));
        renderer_state.trace.write("memory-worker",detail,true);
    }

    void drain_camera_locked(std::unique_lock<std::mutex>& lock) {
        // Configuration, reset and synchronous map draws replace camera work.
        // Units use a resumable pause instead. Never change
        // job_frame or any renderer-owned object while the camera uses it.
        camera_cancelled.store(true,std::memory_order_relaxed);
        camera_pending=false;camera_result=C3X_RENDERER_RESULT_SUPERSEDED;
        completed.wait(lock,[this]{return !camera_active;});
        foreground_pending.store(false,std::memory_order_relaxed);
        camera_ready.clear();
        camera_pending_tiles.clear();camera_pending_topology.clear();
    }

    // A unit owns the serialized renderer until its UI-thread copy finishes.
    // Save the interrupted immutable map without allocating a third snapshot.
    // A newer pending camera takes precedence over the interrupted one.
    struct UnitCameraPause {
        RendererWorker& owner;
        std::unique_lock<std::mutex>& lock;
        UnitCameraPause(RendererWorker& worker,std::unique_lock<std::mutex>& gate):owner(worker),lock(gate) {
            owner.camera_paused=true;
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
        ~UnitCameraPause() {
            if(!lock.owns_lock())lock.lock();
            owner.camera_paused=false;
            owner.foreground_pending.store(owner.camera_pending,std::memory_order_relaxed);
            owner.wake.notify_one();
        }
    };

    int submit_locked(std::unique_lock<std::mutex> & lock, Command command) {
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
            return std::max(distance_x, distance_y) + away;
        };
        std::stable_sort(warm_order.begin(), warm_order.end(), [&](unsigned a, unsigned b) {
            return priority(a) < priority(b);
        });
        unsigned limit=renderer_state.pickup_profile?512u:384u;
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
            if(!has_job && !stop_requested && camera_pending && !camera_paused) {
                auto const ticket=camera_ticket;
                job_camera_ticket=ticket;
                job_frame=camera_pending_frame;
                job_camera_identity=camera_pending_identity;
                job_tiles.swap(camera_pending_tiles);job_world_topology.swap(camera_pending_topology);
                camera_pending_tiles.clear();camera_pending_topology.clear();
                job_frame.tiles=job_tiles.empty()?nullptr:job_tiles.data();
                job_frame.world_topology=job_world_topology.empty()?nullptr:job_world_topology.data();
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
                    bool ok=renderer_state.render(job_frame,output,-1,&camera_cancelled);
                    if(camera_cancelled.load(std::memory_order_relaxed)) {
                        // Completed tile entries and the last committed CPU
                        // bitmap remain individually validated. Render marks
                        // the bitmap invalid before any partial pixel mutation.
                        // Only interrupted draw/animation assemblies are lost.
                        renderer_state.geometry_cache.clear();
                        renderer_state.clear_geometry_vertex_buffers();
                        result=C3X_RENDERER_RESULT_SUPERSEDED;
                    }else if(ok)result=C3X_RENDERER_RESULT_OK;
                    else {
                        renderer_state.reset();
                        result=C3X_RENDERER_RESULT_DEVICE_ERROR;
                    }
                }catch(...) {
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
                    }
                    camera_result=result;
                }
                char detail[128];std::snprintf(detail,sizeof(detail),"ticket=%lld current=%lld result=%d",
                    static_cast<long long>(ticket),static_cast<long long>(camera_ticket),result);
                renderer_state.trace.write("camera-complete",detail,true);
                camera_active=false;
                snapshot_memory("camera-complete");
                foreground_pending.store(camera_pending,std::memory_order_relaxed);
                completed.notify_all();
                // Reclaim an obsolete result/preview outside the queue lock too.
                lock.unlock();finished_frame.clear();lock.lock();
                continue;
            }
            if (!has_job && !stop_requested && !camera_paused && warm_cursor < warm_order.size()) {
                // Yield between individual tiles. A foreground request wakes
                // this delay and cancels CPU construction before GPU upload.
                if (wake.wait_for(lock, std::chrono::milliseconds(2), [this] {
                    return has_job || camera_pending || stop_requested;
                })) continue;
                unsigned index = warm_order[warm_cursor];
                lock.unlock();
                LARGE_INTEGER begin = {}, end = {};
                QueryPerformanceCounter(&begin);
                c3x_renderer_output_v1 unused = {};
                bool ok = false;
                try {
                    ok = renderer_state.render(warm_frame, unused, static_cast<int>(index), &foreground_pending, warm_signature);
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
                    if (ok) ++warm_cursor;
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
            wake.wait(lock, [this] { return has_job || (camera_pending && !camera_paused) || stop_requested; });
            if(camera_pending && !camera_paused && !has_job && !stop_requested)continue;
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
            try {
            if (command == Command::configure_pack) {
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
                if (renderer_state.render(job_frame, output)) {
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
                    [&](auto const& action){return renderer_state.prepare_unit_action(action);})
                    ? C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_ERROR;
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
                else renderer_state.reset();
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
            } else if(command!=Command::unit) {
                warm_order.clear(); warm_cursor = 0; warm_signature = 0;
                warm_tiles.clear();
                renderer_state.cancel_pixel_preparation();
            }
            if(command!=Command::unit) {
            if(command==Command::render && result==C3X_RENDERER_RESULT_OK){
                completed_phase_x=renderer_state.cached_tiles.empty()?0:renderer_state.cached_tiles.front().anchor_x;
                completed_phase_y=renderer_state.cached_tiles.empty()?0:renderer_state.cached_tiles.front().anchor_y;
                if(isolated_publication){
                    auto captured=ambient_async_enabled?&job_frame:nullptr;
                    if(publication.capture(output,completed_phase_x,completed_phase_y,captured)){
                        publication.scene_signature=c3x_renderer::terrain_frame_signature(
                            job_frame,output.content_revision,output.device_generation).complete;
                        output=publication.output;
                    }
                    else {
                        publication.clear();
                        renderer_state.trace.write("publication-unavailable","synchronous exact output retained",true);
                    }
                }
            }else {publication.clear();completed_phase_x=completed_phase_y=0;}
            completed_resources = command==Command::render ? renderer_state.ambient_count() : 0;
            completed_resource_clock=command==Command::render ? RendererState::resource_clock(job_frame) : -1;
            completed_output = output;
            completed_scene_signature=command==Command::render && result==C3X_RENDERER_RESULT_OK ?
                c3x_renderer::terrain_frame_signature(job_frame,output.content_revision,output.device_generation).complete:0;
            completed_result = result;
            }
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

extern "C" __declspec(dllexport) c3x_renderer_u32 c3x_renderer_get_api_version(void) {
    return C3X_RENDERER_API_VERSION;
}

extern "C" __declspec(dllexport) int c3x_renderer_set_pack_path(char const * pack_path) {
    int result = get_renderer_worker().configure_pack(pack_path);
    if (result != C3X_RENDERER_RESULT_OK)
        destroy_renderer_worker();
    return result;
}

extern "C" __declspec(dllexport) int c3x_renderer_set_definition_paths(
    char const * mod_root, char const * default_path, char const * scenario_path, char const * custom_path) {
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
    destroy_renderer_worker();
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

extern "C" __declspec(dllexport) int c3x_renderer_set_unit_rendering(int enabled) {
    return get_renderer_worker().set_unit_rendering(enabled);
}

// Ambient redraws are optional. Civ III tells us when the left button belongs to
// a selected-unit map/pathfinder interaction; allow that interaction immediately.
// Keep a bounded guard for other clicks and the first release interval.
// GetKeyState reads processed button state; avoid GetQueueStatus/GetAsyncKeyState,
// whose change bits are consumable.
extern "C" int c3x_renderer_schedule(c3x_renderer_schedule_v1 const*,c3x_renderer_schedule_result_v1*);
extern "C" __declspec(dllexport) int c3x_renderer_schedule_idle(
    c3x_renderer_schedule_v1 const* input,c3x_renderer_schedule_result_v1* output) {
    int result=c3x_renderer_schedule(input,output);
    if(result!=C3X_RENDERER_RESULT_OK)return result;
    unsigned buttons=((GetKeyState(VK_LBUTTON)&0x8000)?1u:0u) |
        ((GetKeyState(VK_RBUTTON)&0x8000)?2u:0u) |
        ((GetKeyState(VK_MBUTTON)&0x8000)?4u:0u);
    bool busy=buttons!=0;
    static bool previous_busy=false;
    static c3x_renderer_i64 busy_started_ticks=0;
    static c3x_renderer_i64 previous_call_ticks=0;
    bool just_pressed=busy && !previous_busy;
    bool just_released=!busy && previous_busy;
    if(just_pressed)busy_started_ticks=input->now_ticks;
    c3x_renderer_i64 held_ticks=input->now_ticks-busy_started_ticks;
    bool pathfinder_hold=(input->state_flags&C3X_RENDERER_SCHEDULER_PATHFINDER_HOLD)!=0;
    bool deciding_click=busy && !pathfinder_hold &&
        (held_ticks<0 || held_ticks<=input->frequency/4);
    bool defer=deciding_click || just_released;
    c3x_renderer_u32 base_request=output->request_redraw;
    c3x_renderer_u32 base_rebase=output->rebase_clock;
    if(!busy)busy_started_ticks=0;
    if(defer){output->request_redraw=0;output->dirty_flags=0;output->skipped_frame_count=0;output->rebase_clock=1;}
    char const* reason="cadence-wait";
    if(defer)reason=just_released?"release-guard":"click-decision-guard";
    else if((input->state_flags&C3X_RENDERER_SCHEDULER_MAP_VISIBLE)==0)reason="map-hidden";
    else if((input->state_flags&C3X_RENDERER_SCHEDULER_FOCUSED)==0)reason="unfocused";
    else if((input->state_flags&C3X_RENDERER_SCHEDULER_MODAL)!=0)reason="modal";
    else if((input->state_flags&C3X_RENDERER_SCHEDULER_DRAWING)!=0)reason="drawing";
    else if((input->state_flags&C3X_RENDERER_SCHEDULER_REDRAW_PENDING)!=0)reason="redraw-pending";
    else if(output->rebase_clock)reason="clock-rebase";
    else if(output->request_redraw)reason="redraw-request";
    double ticks_to_ms=1000.0/double(input->frequency);
    double callback_gap_ms=previous_call_ticks>0?
        double(input->now_ticks-previous_call_ticks)*ticks_to_ms:-1.0;
    double present_age_ms=input->last_presented_ticks>0?
        double(input->now_ticks-input->last_presented_ticks)*ticks_to_ms:-1.0;
    double held_ms=busy?double(held_ticks)*ticks_to_ms:0.0;
    char detail[640];
    std::snprintf(detail,sizeof(detail),
        "[C3X renderer] qpc=%lld stage=scheduler-callback gap_ms=%.3f present_age_ms=%.3f buttons=%u held_ms=%.3f pathfinder=%u state=0x%08x visible=%u base_request=%u base_rebase=%u final_request=%u final_rebase=%u skipped=%u reason=%s\n",
        static_cast<long long>(input->now_ticks),callback_gap_ms,present_age_ms,buttons,held_ms,pathfinder_hold?1u:0u,
        input->state_flags,input->visible_animation_count,base_request,base_rebase,
        output->request_redraw,output->rebase_clock,output->skipped_frame_count,reason);
    detail[sizeof(detail)-1]='\0';
    OutputDebugStringA(detail);
    previous_busy=busy;
    previous_call_ticks=input->now_ticks;
    return result;
}
