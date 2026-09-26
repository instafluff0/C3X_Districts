#pragma once
// Clean compiler boundary for the six ground-family surfaces (underlay, land,
// bed, water, river, terrain-shadow), mirroring cliff_compiler.h and
// terrain_compiler.h's shape.
//
// The caller keeps every lower-level per-tile sampling closure that other
// systems in the same per-tile scope also read (relief/height/material/shore
// queries used by feature, forest, city and cliff-adjacent code) and passes
// them in here as read-only callables. This compiler owns only the
// ground-exclusive mesh-assembly step: turning those samples into the six
// vertex buffers/index arrays plus any cached grid samples worth retaining,
// instead of writing into caller-owned vectors from an inline closure.
//
// Milestone 1.2: the caller may now pass a private, per-generator
// SurfaceQueries/NaturalWorld scratch (mirroring cliff_query_scratch) instead
// of the shared per-tile queries/pickup_surface/natural also read by
// feature/city/cliff generation, so this compiler's own callables and its
// one direct dependency here (river_sample/river_affects, hence the widened
// NaturalWorld& rather than the GPU-owning Natural&) no longer require that
// shared state. The frozen/legacy (!pickup_profile) callables still delegate
// to the shared, topology_cache-reading closures below, since that path is
// foreground-only and never scheduled concurrently.
#include "../../lab/shared/natural/ground.h"
#include "../render_core/terrain_query.h"
#include "../c3x_renderer_api.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>
namespace c3x_renderer { namespace fidelity {

inline float ground_smoothstep01(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    return value * value * (3.0f - 2.0f * value);
}

// Moved here from c3x_renderer.cpp: used only by ground generation, so this
// is now the single definition. c3x_renderer.cpp aliases both names.
struct GroundPoint {
    float u = 0.0f, v = 0.0f;
    float world_u = 0.0f, world_v = 0.0f;
    float local_ground_x = 0.0f, local_ground_y = 0.0f;
    float material_u = 0.0f, material_v = 0.0f;
    float material_weights[5] = {};
    float signed_shore = 0.0f;
    render_core::ShoreSample shore;
    float surface_coordinate = 0.0f;
    float relief[3] = {};
    float normal[3] = {0.0f, 0.0f, 1.0f};
    float normal_delta[2] = {};
    bool terrain_ready = false;
};

struct CachedGroundGrid {
    int divisions=0;
    float layer=0;
    std::vector<MapVertex> vertices;
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
    MapVertex project(std::size_t index,int width,int height) const {
        MapVertex out=vertices[index];
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

struct GroundCompileInput {
    c3x_renderer_tile_v1 tile{};
    bool world_ground=false, pickup_profile=false, fidelity_profile=false, draw_marsh=false;
    bool river_assets_ready=false;
    bool retain_ground_grids=false, reuse_nested_ground_grids=false, prewarming=false;
    int ground=0;
    float half_w=0, half_h=0, uv_scale=1, relief_projection_scale=1;
    float key_light[3]={};
    float left=0, top=0; // mesh origin is local; Civ III supplies the draw anchor
    int flat_grid=0, tile_ground_grid=0, shadow_grid=0;
};

struct GroundSurfaces {
    std::vector<MapVertex> underlay_vertices, land_vertices, bed_vertices,
        water_vertices, river_vertices, shadow_vertices;
    std::vector<UINT> underlay_indices, land_indices, bed_indices,
        water_indices, river_indices;
    std::vector<CachedGroundGrid> pending_grids;
};

template<class RiverNodes,class ReliefAt,class PickupHeightAt,class PickupGroundAt,
    class RiverDistanceAt,class MaterialWeightsAt,class ShoreDistanceAt,class WaterDepthAt,
    class SurfaceUVAt,class ShoreSampleAt,class NdcX,class NdcY,class Cancelled>
void compile_ground_surfaces(GroundCompileInput const & input, c3x_renderer_frame_v1 const & frame,
        NaturalWorld & natural, RiverNodes const & local_river_nodes,
        std::vector<CachedGroundGrid> const * cached_grid_source, unsigned & ground_grid_hit_counter,
        ReliefAt relief_at_world, PickupHeightAt pickup_height_at, PickupGroundAt pickup_ground_at,
        RiverDistanceAt river_distance, MaterialWeightsAt material_weights_for,
        ShoreDistanceAt signed_shore_distance, WaterDepthAt water_family_depth,
        SurfaceUVAt periodic_surface_uv, ShoreSampleAt shore_sample_at,
        NdcX ndc_x, NdcY ndc_y, Cancelled cancelled, GroundSurfaces & destination) {
    c3x_renderer_tile_v1 const & tile = input.tile;
    // River topology is consumed only by the river surface pass. Computing
    // its curved-edge/global node distances for the four terrain passes and
    // the shadow pass was pure discarded work, and scaled especially badly
    // with a full Civ III view.
    auto river_node_distance = [&](float u, float v, unsigned node_kind) {
        if (input.fidelity_profile && node_kind != 1) {
            auto sample = natural.river_sample({(tile.tile_x + tile.tile_y) * .5 + u,
                (tile.tile_x - tile.tile_y) * .5 + 1 - v});
            return float(node_kind == 0 ? sample.source : sample.mouth);
        }
        float point_x = static_cast<float>(tile.tile_x) + u - v;
        float point_y = static_cast<float>(tile.tile_y) + u + v - 1.0f;
        float distance = 1000.0f;
        for (auto node_pointer : local_river_nodes) {
            auto const & node = *node_pointer;
            bool selected = node_kind == 0u
                ? node.degree == 1u && !node.touches_water
                : (node_kind == 1u ? node.degree >= 3u
                                  : node.degree == 1u && node.touches_water);
            if (!selected)
                continue;
            float delta_x = (point_x - static_cast<float>(node.lattice_x)) * (input.world_ground ? 64.f : input.half_w);
            float delta_y = (point_y - static_cast<float>(node.lattice_y)) * (input.world_ground ? 32.f : input.half_h);
            distance = std::min(distance,
                std::sqrt(delta_x * delta_x + delta_y * delta_y));
        }
        return distance;
    };
    auto cast_shadow_visibility = [&](float world_u, float world_v, float origin_height) {
        float horizontal = std::sqrt(input.key_light[0] * input.key_light[0] +
                                     input.key_light[1] * input.key_light[1]);
        if (horizontal < 0.001f) {
            return 1.0f;
        }
        float direction_u = input.key_light[0] / horizontal;
        float direction_v = -input.key_light[1] / horizontal;
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
        point.local_ground_x = input.half_w + (u - v) * input.half_w;
        point.local_ground_y = (u + v) * input.half_h;
        std::array<float, 2> material_uv =
            periodic_surface_uv(point.world_u, point.world_v, input.uv_scale);
        point.material_u = material_uv[0];
        point.material_v = material_uv[1];
        std::array<float, 5> weights =
            material_weights_for(point.world_u, point.world_v);
        std::copy(weights.begin(), weights.end(), point.material_weights);
        point.signed_shore = signed_shore_distance(
            point.world_u, point.world_v, u, v);
        if (input.pickup_profile) point.shore = shore_sample_at(point.world_u, point.world_v);
        point.surface_coordinate = point.signed_shore <= 0.0f
            ? point.signed_shore
            : std::sqrt(ground_smoothstep01(point.signed_shore)) *
                water_family_depth(point.world_u, point.world_v);
        return ground_point_cache.emplace(key, point).first->second;
    };
    bool river_near = input.fidelity_profile && input.river_assets_ready &&
        (((tile.river_code & 170u) != 0) ||
         natural.river_affects((tile.tile_x+tile.tile_y)/2,(tile.tile_x-tile.tile_y)/2));
    auto make_ground_vertex = [&](float u, float v, float layer) {
        GroundPoint & point = ground_point_at(u, v);
        float world_u = point.world_u;
        float world_v = point.world_v;
        bool underlay_surface = layer > 0.4f && layer < 0.6f;
        bool land_surface = layer > 0.75f && layer < 1.25f;
        bool river_surface = layer > 8.5f && layer < 9.5f;
        bool shadow_surface = layer > 9.5f && layer < 10.5f;
        float river_surface_distance = river_surface ||
            (river_near && (underlay_surface || land_surface || shadow_surface))
            ? river_distance(tile, u, v) : 1000.0f;
        bool terrain_conforming_surface = land_surface ||
            (layer > 8.5f && layer < 10.5f);
        if (terrain_conforming_surface && !point.terrain_ready) {
            std::array<float, 3> sampled = relief_at_world(world_u, world_v);
            std::copy(sampled.begin(), sampled.end(), point.relief);
            constexpr float normal_step = 0.006f;
            float left_height = input.pickup_profile ? pickup_height_at(world_u - normal_step, world_v) : relief_at_world(world_u - normal_step, world_v)[0];
            float right_height = input.pickup_profile ? pickup_height_at(world_u + normal_step, world_v) : relief_at_world(world_u + normal_step, world_v)[0];
            float down_height = input.pickup_profile ? pickup_height_at(world_u, world_v - normal_step) : relief_at_world(world_u, world_v - normal_step)[0];
            float up_height = input.pickup_profile ? pickup_height_at(world_u, world_v + normal_step) : relief_at_world(world_u, world_v + normal_step)[0];
            point.normal_delta[0]=right_height-left_height;
            point.normal_delta[1]=up_height-down_height;
            float slope_u = (right_height - left_height) * (input.pickup_profile ? 1.0f : input.relief_projection_scale) /
                (2.0f * normal_step * static_cast<float>(frame.tile_width));
            float slope_v = (up_height - down_height) * (input.pickup_profile ? -1.0f : input.relief_projection_scale) /
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
        if (river_near && (underlay_surface || land_surface || river_surface || shadow_surface))
            relief_sample[0] += river_channel_cut(river_surface_distance);
        float h = relief_sample[0] * input.relief_projection_scale;
        float signed_shore = point.signed_shore;
        if (!input.pickup_profile && land_surface && h > 0.0f) {
            float shore_envelope = ground_smoothstep01((-signed_shore - 0.02f) / 0.42f);
            h *= shore_envelope;
            relief_sample[1] *= shore_envelope;
            relief_sample[2] *= shore_envelope;
        }
        float ground_x = input.left + point.local_ground_x;
        float ground_y = input.top + point.local_ground_y;
        // Elevation moves toward the isometric camera as well as up on
        // screen.  Keeping flat-ground depth made steep micro-quads
        // fold over one another and appear as bright contour seams.
        float depth = ground_y + h * 0.75f;
        float normal_x = terrain_conforming_surface ? point.normal[0] : 0.0f;
        float normal_y = terrain_conforming_surface ? point.normal[1] : 0.0f;
        float normal_z = terrain_conforming_surface ? point.normal[2] : 1.0f;
        float surface_coordinate = point.surface_coordinate;
        float shadow_visibility = !input.pickup_profile && layer > 9.5f
            ? cast_shadow_visibility(world_u, world_v, relief_sample[0]) : 1.0f;
        auto owner_material = input.pickup_profile && terrain_conforming_surface
            ? pickup_ground_at(world_u, world_v).owner : std::array<float,4>{};
        MapVertex vertex{
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
        if(river_surface && input.fidelity_profile){
            auto flow=natural.river_sample({(tile.tile_x+tile.tile_y)*.5+u,(tile.tile_x-tile.tile_y)*.5+1-v}).flow;
            // Relief-owner XY is unused by the river material; carry its
            // immutable world-space flow without enlarging the vertex layout.
            vertex.relief_owner_u=float(flow.x);vertex.relief_owner_v=float(flow.y);
        }
        if (input.world_ground) {
            float elevation=relief_sample[0]*(128.f/224.f*.82f),base=(u+v)*32.f;
            vertex.x=64.f+(u-v)*64.f;vertex.y=base-elevation;vertex.z=base+elevation*.75f;
            if(terrain_conforming_surface){vertex.normal_x=-point.normal_delta[0]/.012f;
                vertex.normal_y=point.normal_delta[1]/.012f;vertex.normal_z=1;}
        }
        return vertex;
    };
    auto append_ground_layer = [&](std::vector<MapVertex> & target, float layer,
                                   int subdivisions, std::vector<UINT> * indices=nullptr) {
        // Adjacent cells share grid corners. Build and upload each
        // corner once, then preserve the original triangle-list order.
        // This avoids repeated terrain/shadow evaluation for a point.
        int const row_width = subdivisions + 1;
        std::vector<MapVertex> expanded_corners;
        auto& grid_vertices=indices?target:expanded_corners;
        grid_vertices.resize(static_cast<std::size_t>(row_width) * row_width);
        CachedGroundGrid const* cached_grid=nullptr;
        int cached_stride=0;
        if(cached_grid_source)for(auto const& grid:*cached_grid_source)
            if(grid.layer==layer && (grid.divisions==subdivisions || input.reuse_nested_ground_grids)){
                int stride=grid.sample_stride(subdivisions);
                if(stride){cached_grid=&grid;cached_stride=stride;break;}
            }
        bool record=input.retain_ground_grids && !input.world_ground && !input.prewarming && indices && !cached_grid;
        CachedGroundGrid pending;
        if(record){pending.layer=layer;pending.divisions=subdivisions;pending.samples.resize(grid_vertices.size());}
        if(cached_grid)++ground_grid_hit_counter;
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
                        pending.samples[at]={layer==.5f || layer==1.f || layer==9.f
                            ? point.relief[0]+(river_near
                                ? river_channel_cut(grid_vertices[at].river_distance) : 0.f) : 0.f,
                            point.normal_delta[0],point.normal_delta[1]};}
                }
            }
        }
        if(record){pending.vertices=grid_vertices;destination.pending_grids.push_back(std::move(pending));}
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
                auto vertex_at = [&](int x, int y) -> MapVertex const & {
                    return grid_vertices[
                        static_cast<std::size_t>(y) * row_width + x];
                };
                MapVertex const & a = vertex_at(grid_u, grid_v);
                MapVertex const & b0 = vertex_at(grid_u + 1, grid_v);
                MapVertex const & c = vertex_at(grid_u + 1, grid_v + 1);
                MapVertex const & d = vertex_at(grid_u, grid_v + 1);
                MapVertex triangles[] = {a, b0, c, a, c, d};
                target.insert(target.end(), std::begin(triangles), std::end(triangles));
            }
        }
    };
    append_ground_layer(destination.underlay_vertices, 0.5f,
                        river_near ? std::max(input.flat_grid,
                            frame.tile_width >= 96 ? 32 : 16) : input.flat_grid,
                        &destination.underlay_indices);
    if (input.ground < 11 && (!input.fidelity_profile || input.draw_marsh))
        append_ground_layer(destination.land_vertices, 1.0f,
                            river_near ? std::max(input.tile_ground_grid,
                                frame.tile_width >= 96 ? 32 : 16) : input.tile_ground_grid,
                            &destination.land_indices);
    if (!input.pickup_profile) {
        append_ground_layer(destination.bed_vertices, 4.0f, input.flat_grid, &destination.bed_indices);
        append_ground_layer(destination.water_vertices, 5.0f, input.flat_grid, &destination.water_indices);
    }
    if (input.river_assets_ready && ((tile.river_code & 170u) != 0 || river_near))
        append_ground_layer(destination.river_vertices, 9.0f,
                            frame.tile_width >= 96 ? 32 : 16, &destination.river_indices);
    if (!input.pickup_profile && input.ground < 11) {
        // Cast-shadow visibility ray-marches the authored relief field.
        // Retain the approved 16x16 near grid for canonical fixtures,
        // but use the already-approved reduced grid when a live m19
        // capture contains hundreds of companion records.  The shader
        // interpolates visibility across the unchanged terrain body.
        append_ground_layer(destination.shadow_vertices, 10.0f, input.shadow_grid);
    }
}

}}
