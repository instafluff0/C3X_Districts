#pragma once
// CPU descriptions and output for existing routes, bridges, sites, improvements
// and fallback city components. Asset IDs are local to the immutable pack lease.
// Query callbacks preserve the caller's dependency recorder; this synchronous
// boundary alone does not authorize concurrent access to its scratch.
#include "terrain_scene_runtime.h"
#include "../lab/shared/natural/vertex.h"
#include "scene_lighting.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
namespace c3x_renderer { namespace objects {
using Vertex=fidelity::MapVertex;
enum Layer {route_layer,feature_layer,city_layer,wall_layer,mine_layer,farm_layer,site_layer,layer_count};
enum Family {bridge_family,site_family,mine_family,farm_family,city_family,wall_family,family_count};
struct Assets {
    std::array<FeatureBundle const*,family_count> bundles;
    FeatureBundle const& operator[](Family family)const{return *bundles[family];}
};
struct Instance {
    Family family; unsigned asset; Layer layer;
    float u,v,rotation,scale,material,owner;
    bool shadow;
};
struct Route {float u0,v0,u1,v1;unsigned style;bool railroad;};
struct Plan {std::vector<Instance> instances;std::vector<Route> routes;};
struct Surfaces {
    std::array<std::vector<Vertex>,layer_count> layers;
    std::array<std::vector<unsigned>,layer_count> indices;
    std::vector<Vertex> shadows;
};
struct Projection {
    c3x_renderer_tile_v1 tile{};
    int tile_width=0,content_view_height=0;
    float left=0,top=0,half_w=0,half_h=0,relief_projection_scale=1,feature_projection_scale=1;
    bool pickup_profile=false,world_objects=false;
    std::array<float,3> key_light{};
};
inline void append_shadow(Projection const& input,FeatureAsset const& asset,float scale,
        float center_x,float center_y,float ground_height_screen,std::vector<Vertex>& shadow_vertices){
    bool pickup_profile=input.pickup_profile;float half_w=input.half_w,half_h=input.half_h;
    auto const& key_light=input.key_light;
    auto ndc_x=[](float x){return x;};auto ndc_y=[](float y){return y;};
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
        (static_cast<float>(input.tile_width) / 224.0f) * 0.72f;
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
}
template<class Relief,class Height>
void append_instance(Projection const& input,FeatureBundle const& bundle,FeaturePlacement const& placement,
        float local_u,float local_v,float rotation,float scale,float material_offset,float owner_code,bool cast_shadow,
        bool site,Relief relief_at_world,Height natural_height_at,std::vector<Vertex>& target,std::vector<Vertex>& shadows,std::vector<unsigned>* topology=nullptr){
    auto const& tile=input.tile;
    float left=input.left,top=input.top,half_w=input.half_w,half_h=input.half_h;
    float relief_projection_scale=input.relief_projection_scale,feature_projection_scale=input.feature_projection_scale;
    int content_view_height=input.content_view_height;
    bool pickup_profile=input.pickup_profile,world_objects=input.world_objects;
    auto ndc_x=[](float x){return x;};auto ndc_y=[](float y){return y;};
    auto append_object_shadow=[&](auto const& asset,float s,float x,float y,float h){append_shadow(input,asset,s,x,y,h,shadows);};
    if (placement.asset_index >= bundle.assets.size())
        return;
    c3x_renderer::FeatureAsset const & asset = bundle.assets[placement.asset_index];
    float tile_world_u = static_cast<float>(tile.tile_x + tile.tile_y) * 0.5f;
    float tile_world_v = static_cast<float>(tile.tile_x - tile.tile_y) * 0.5f;
    std::array<float, 3> ground_sample = relief_at_world(
        tile_world_u + local_u, tile_world_v + (1.0f - local_v));
    if(pickup_profile && site)
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
            if(world_objects){
                vertex.x=64.f+(local_u-local_v)*64.f+(local_x-local_y)*64.f;
                vertex.y=((local_u+local_v)*32.f-ground_sample[0]*(128.f/224.f*.82f))+(local_x+local_y)*32.f-local_z*150.f*(128.f/224.f);
                vertex.z=feature_height_tiles;
            }
            auto normal=c3x_renderer::lighting::object_normal(normal_x,normal_y,source.normal[2]);
            vertex.normal_x=normal[0];vertex.normal_y=normal[1];vertex.normal_z=normal[2];
        }
    }
    if(topology){
        unsigned base=unsigned(target.size());
        target.insert(target.end(),transformed.begin(),transformed.end());
        for(auto index:asset.indices)topology->push_back(base+index);
    }else for (std::uint32_t source_index : asset.indices)
        target.push_back(transformed[source_index]);
}
template<class Relief>
void append_route(Projection const& input,Route const& route,Relief relief_at_world,std::vector<Vertex>& route_vertices){
    float u0=route.u0,v0=route.v0,u1=route.u1,v1=route.v1;unsigned style=route.style;bool railroad=route.railroad;
    auto const& tile=input.tile;
    float left=input.left,top=input.top,half_w=input.half_w,half_h=input.half_h;
    float relief_projection_scale=input.relief_projection_scale;
    bool pickup_profile=input.pickup_profile,world_objects=input.world_objects;
    auto ndc_x=[](float x){return x;};auto ndc_y=[](float y){return y;};
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
            if(world_objects){vertex.x=64.f+(route_u-route_v)*64.f;
                vertex.y=(route_u+route_v)*32.f-ground_sample[0]*(128.f/224.f*.82f);vertex.z=0;}
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
}
template<class Lookup>
void select_routes(c3x_renderer_tile_v1 const& tile,Assets const& assets,bool route_assets_ready,
        bool routes_enabled,Lookup lookup,Plan& plan){
    auto const& bridge_bundle=assets[bridge_family];constexpr Layer feature_vertices=feature_layer;
    auto append_feature_instance=[&](FeatureBundle const& bundle,FeaturePlacement const& placement,
            float u,float v,float rotation,float scale,float material,float owner,bool shadow,Layer layer){
        for(unsigned family=0;family<family_count;++family)if(assets.bundles[family]==&bundle){
            plan.instances.push_back({Family(family),placement.asset_index,layer,u,v,rotation,scale,material,owner,shadow});return;
        }
    };
    auto append_route_segment=[&](float a,float b,float c,float d,unsigned style,bool railroad){plan.routes.push_back({a,b,c,d,style,railroad});};
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
            auto found = lookup(neighbor_x, neighbor_y);
            if (found == nullptr)
                continue;
            c3x_renderer_tile_v1 const & neighbor = found->occurrence;
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
            if(routes_enabled)
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
}
inline bool select_improvements(c3x_renderer_tile_v1 const& tile,Assets const& assets,int ground,unsigned site_flags,
        bool mine_assets_ready,bool farm_assets_ready,bool city_assets_ready,bool composed_city,Plan& plan){
    auto const& site_bundle=assets[site_family];
    auto const& mine_bundle=assets[mine_family];auto const& farm_bundle=assets[farm_family];
    auto const& city_bundle=assets[city_family];auto const& wall_bundle=assets[wall_family];
    constexpr Layer site_vertices=site_layer,mine_vertices=mine_layer,
        farm_vertices=farm_layer,city_vertices=city_layer,wall_vertices=wall_layer;
    auto append_feature_instance=[&](FeatureBundle const& bundle,FeaturePlacement const& placement,
            float u,float v,float rotation,float scale,float material,float owner,bool shadow,Layer layer){
        for(unsigned family=0;family<family_count;++family)if(assets.bundles[family]==&bundle){
            plan.instances.push_back({Family(family),placement.asset_index,layer,u,v,rotation,scale,material,owner,shadow});return;
        }
    };
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
        if (!composed_city && group != nullptr && !group->placements.empty()) {
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
    return true;
}
template<class Relief,class Height>
void compile(Plan const& plan,Projection const& input,Assets const& assets,Relief relief,Height height,Surfaces& output,bool indexed=false){
    for(auto const& route:plan.routes)append_route(input,route,relief,output.layers[route_layer]);
    for(auto const& instance:plan.instances){
        FeaturePlacement placement{};placement.asset_index=instance.asset;
        append_instance(input,assets[instance.family],placement,instance.u,instance.v,instance.rotation,
            instance.scale,instance.material,instance.owner,instance.shadow,instance.family==site_family,
            relief,height,output.layers[instance.layer],output.shadows,indexed?&output.indices[instance.layer]:nullptr);
    }
}
}}
