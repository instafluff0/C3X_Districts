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
#include <cstring>
namespace c3x_renderer { namespace objects {
using Vertex=fidelity::MapVertex;
enum Layer {route_layer,feature_layer,city_layer,wall_layer,mine_layer,farm_layer,site_layer,layer_count};
enum Family {bridge_family,site_family,mine_family,farm_family,city_family,wall_family,family_count};
// Generic connection-mask centerlines: for each 8-neighbor mask (bit k is
// NE, E, SE, S, SW, W, NW, N), polylines in tile-local (u,v). An end equal to
// a direction lies exactly on that neighbor's shared edge midpoint or corner.
struct RoutePatterns {
    struct Line {std::uint32_t first;std::uint16_t count;std::int8_t start,end;};
    // One entry per mask (256), then further variants of the fully connected
    // mask: the railroad sheet has 16 more, which Civ III picks at random.
    std::vector<std::uint32_t> offsets;
    std::vector<Line> lines;
    std::vector<std::array<float,2>> points;
    bool load(std::uint8_t const* data,std::size_t size){
        *this={};
        auto u32=[&](std::size_t at){std::uint32_t value;std::memcpy(&value,data+at,4);return value;};
        if(size<20 || std::memcmp(data,"C3XRPAT1",8)!=0 || u32(8)<256 || u32(8)>512)return false;
        unsigned masks=u32(8);
        std::uint64_t line_count=u32(12),point_count=u32(16);
        std::size_t at=20+(masks+1)*4;
        if(size!=at+line_count*8+point_count*8)return false;
        offsets.resize(masks+1);
        for(unsigned index=0;index<=masks;++index)offsets[index]=u32(20+index*4);
        lines.resize(std::size_t(line_count));points.resize(std::size_t(point_count));
        for(auto& line:lines){
            std::memcpy(&line.first,data+at,4);std::memcpy(&line.count,data+at+4,2);
            line.start=std::int8_t(data[at+6]);line.end=std::int8_t(data[at+7]);at+=8;
        }
        if(point_count)std::memcpy(points.data(),data+at,std::size_t(point_count)*8);
        bool valid=offsets[0]==0 && offsets[masks]==line_count;
        for(unsigned index=0;valid && index<masks;++index)valid=offsets[index]<=offsets[index+1];
        for(auto const& line:lines)valid=valid && line.count>=2 && line.first+std::uint64_t(line.count)<=point_count &&
            line.start>=-1 && line.start<8 && line.end>=-1 && line.end<8;
        for(auto const& point:points)valid=valid && point[0]>-.5f && point[0]<1.5f && point[1]>-.5f && point[1]<1.5f;
        if(!valid)*this={};
        return valid;
    }
    // The pattern drawn for a mask at a tile: a fully connected tile picks
    // one of the sheet's variants by a stable hash of its position.
    unsigned index(unsigned mask,int tile_x,int tile_y)const{
        unsigned variants=offsets.empty()?0u:unsigned(offsets.size()-1);
        if(mask!=255u || variants<=256u)return mask;
        unsigned seed=c3x_renderer::stable_hash(unsigned(tile_x)*73856093u^unsigned(tile_y)*19349663u^0x5bd1e995u);
        return 255u+seed%(variants-255u);
    }
};
struct Assets {
    std::array<FeatureBundle const*,family_count> bundles;
    RoutePatterns const* road_patterns=nullptr;
    RoutePatterns const* rail_patterns=nullptr; // without it railroads follow the road patterns
    FeatureBundle const& operator[](Family family)const{return *bundles[family];}
    // A pattern route's own sheet: railroads (style 4) use the railroad sheet.
    RoutePatterns const* patterns_for(unsigned style)const{return style>=4u && rail_patterns?rail_patterns:road_patterns;}
};
struct Instance {
    Family family; unsigned asset; Layer layer;
    float u,v,rotation,scale,material,owner;
    bool shadow;
};
struct Route {float u0,v0,u1,v1;unsigned style;bool railroad,bridge,reverse,bypass=false;float bridge_t=1.0f;bool bridge_structural=true;bool isolated=false;};
// One centerline of the tile's road pattern. Bridge bit 0/1 marks a start/end
// join crossing a river; points, when present, replaces the pattern's points
// after stretches inside a river channel move onto its bank. joins holds the
// axis shared with the neighbor at the start and end joins, pointing out of
// this tile (zero when absent).
struct PatternRoute {unsigned line,style,bridges=0;std::vector<std::array<float,2>> points;std::array<float,4> joins{};
    std::array<float,4> open{}; // start/end: tile-local way to the open bank where a join touches a river bend
    unsigned fords=0;           // start/end join crosses a river at a corner, without a bridge
    std::vector<std::uint8_t> wet; // points still over water after the bank move: they fade
    std::array<float,2> crossing{}; // start/end: bridged river's center past the join, along its axis
    std::vector<float> fade;        // per point: fades out as a mountain rises under the path
};
// Ground a farm keeps open on its own tile, in tile-local (u,v): its routes
// as capsules around their drawn centerlines and its resource's parts as
// rounded boxes. at() is the signed distance from that ground (negative inside).
struct FarmClearing {
    std::vector<std::array<float,5>> paths; // u0,v0,u1,v1,half width
    std::vector<std::array<float,5>> boxes; // u0,v0,u1,v1,margin
    bool yard=true; // false when the farm plants its resource itself
    float at(float u,float v)const{
        float distance=1.f;
        for(auto const& p:paths){
            float du=p[2]-p[0],dv=p[3]-p[1],length=du*du+dv*dv;
            float t=length>0?std::clamp(((u-p[0])*du+(v-p[1])*dv)/length,0.f,1.f):0.f;
            distance=std::min(distance,std::hypot(u-p[0]-du*t,v-p[1]-dv*t)-p[4]);
        }
        for(auto const& b:boxes){
            float du=std::max({b[0]-u,0.f,u-b[2]}),dv=std::max({b[1]-v,0.f,v-b[3]});
            distance=std::min(distance,(du>0 || dv>0?std::hypot(du,dv):
                -std::min({u-b[0],b[2]-u,v-b[1],b[3]-v}))-b[4]);
        }
        return distance;
    }
};
// farm_kit: the farm pack's kit (a "farm_kit" group) lays out this tile's farm.
struct Plan {std::vector<Instance> instances;std::vector<Route> routes;std::vector<PatternRoute> patterns;
    bool farm_kit=false;FarmClearing farm_clearing;};
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
// A road or railroad bridge rests on the lower of its two banks. The authored
// meshes put their deck ends at the base (z=0), so that end meets its bank
// and the other end settles into a higher bank instead of floating over a
// lower one. One seat keeps the shared rigid transform.
template<class Relief,class Height>
bool seat_route_bridge(Projection const& input,FeatureAsset const& asset,float scale,float rotation,
        float world_u,float world_v,Relief relief_at_world,Height natural_height_at,float& ground){
    if(!input.pickup_profile || asset.id.rfind("route/bridge/",0)!=0 || asset.vertices.empty())return false;
    float reach=0.f;
    for(auto const& source:asset.vertices)reach=std::max(reach,std::abs(source.position[0])*scale);
    ground=1e9f;
    for(float along:{-reach,reach}){
        float u=world_u+std::cos(rotation)*along,v=world_v-std::sin(rotation)*along;
        ground=std::min(ground,std::max(relief_at_world(u,v)[0],natural_height_at(u,v)-2.5f));
    }
    return true;
}
template<class Relief,class Height>
void append_instance(Projection const& input,FeatureBundle const& bundle,FeaturePlacement const& placement,
        float local_u,float local_v,float rotation,float scale,float material_offset,float owner_code,bool cast_shadow,
        bool site,Relief relief_at_world,Height natural_height_at,std::vector<Vertex>& target,std::vector<Vertex>& shadows,std::vector<unsigned>* topology=nullptr,float lift=0.f,float ground_fit=0.f,
        bool drape=false){
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
    bool farm_asset=asset.id.rfind("farm_",0)==0;
    // Pipeline-baked flat ground decal (soft alpha, terrain-conforming, no shadow).
    bool ground_decal=asset.id.rfind("decal/",0)==0;
    bool terrain_wall=pickup_profile && asset.id.rfind("city/walls/",0)==0;
    float wall_source_floor=0.f;
    if(terrain_wall){
        for(auto const& source:asset.vertices)
            wall_source_floor=std::min(wall_source_floor,source.position[2]*scale);
        ground_sample[0]=natural_height_at(tile_world_u+local_u,
            tile_world_v+1.f-local_v)-2.5f-wall_source_floor*(150.f/.82f)+.02f;
    }
    // A draped farm stands on the rendered ground: low relief cannot bury it.
    if(drape && farm_asset)ground_sample[0]=std::max(ground_sample[0],natural_height_at(
        tile_world_u+local_u,tile_world_v+1.f-local_v)-2.5f);
    if (farm_asset && asset.id.find(":base:")!=std::string::npos && ground_sample[2]<.55f)
        return;
    if (farm_asset && asset.id.find(":building:")!=std::string::npos && ground_sample[2]<.14f)
        return;
    if (farm_asset && asset.id.find(":tree:")!=std::string::npos && ground_sample[2]<.11f)
        return;
    if(pickup_profile && site){
        ground_sample[0]=natural_height_at(
            tile_world_u+local_u,tile_world_v+1.f-local_v)-2.5f;
        // Baked ground fit: settle on the lowest ground under the footprint, so
        // a body on a slope sinks uphill rather than floating downhill.
        if(ground_fit>0.f)for(int corner=0;corner<4;++corner)ground_sample[0]=std::min(ground_sample[0],
            natural_height_at(tile_world_u+local_u+(corner&1?ground_fit:-ground_fit),
                tile_world_v+1.f-local_v+(corner&2?ground_fit:-ground_fit))-2.5f);
    }
    seat_route_bridge(input,asset,scale,rotation,tile_world_u+local_u,tile_world_v+1.f-local_v,
        relief_at_world,natural_height_at,ground_sample[0]);
    float center_x = left + half_w + (local_u - local_v) * half_w;
    float center_y = top + (local_u + local_v) * half_h -
        ground_sample[0] * relief_projection_scale;
    if (cast_shadow && !ground_decal)
        append_object_shadow(asset, scale, center_x, center_y,
                             ground_sample[0] * relief_projection_scale);
    float cosine = std::cos(rotation);
    float sine = std::sin(rotation);
    bool farm_decal = false;
    if ((farm_asset || ground_decal) && !asset.vertices.empty()) {
        farm_decal = true;
        float level = asset.vertices.front().position[2];
        for (auto const& vertex : asset.vertices)
            farm_decal = farm_decal && std::abs(vertex.position[2]-level)<1e-5f;
    }
    std::vector<Vertex> transformed(asset.vertices.size());
    std::vector<float> farm_shore(farm_decal ? asset.vertices.size() : 0);
    std::vector<std::array<float,2>> farm_world(farm_decal ? asset.vertices.size() : 0);
    for (std::size_t vertex_index = 0; vertex_index < asset.vertices.size(); ++vertex_index) {
        c3x_renderer::FeatureSourceVertex const & source = asset.vertices[vertex_index];
        float local_x = (source.position[0] * cosine - source.position[1] * sine) * scale;
        float local_y = (source.position[0] * sine + source.position[1] * cosine) * scale;
        // baked sink/raise, tile units; a draped field sits a hair above the
        // ground (still below route strips) so it never z-fights the terrain
        float local_z = source.position[2] * scale + lift + (farm_decal && drape ? .016f : 0.f);
        // A draped patchwork reaches past its tile; vertices far enough out
        // that every triangle using them lies outside skip the terrain queries.
        bool outside = farm_decal && drape && std::max(std::abs(local_u + local_x - .5f),
            std::abs(local_v + local_y - .5f)) > .65f;
        auto vertex_ground = farm_decal && !outside
            ? relief_at_world(tile_world_u + local_u + local_x,
                              tile_world_v + 1.0f - local_v - local_y)
            : ground_sample;
        if (outside) vertex_ground[2] = -1.f;
        if(farm_decal && ground_decal && pickup_profile && site)vertex_ground[0]=natural_height_at(
            tile_world_u+local_u+local_x,tile_world_v+1.f-local_v-local_y)-2.5f;
        // Draped fields lie on the rendered ground, as routes do, so low
        // relief on grassland, plains and hills cannot bury them.
        if(farm_decal && drape && !outside)vertex_ground[0]=std::max(vertex_ground[0],natural_height_at(
            tile_world_u+local_u+local_x,tile_world_v+1.f-local_v-local_y)-2.5f);
        if(terrain_wall)vertex_ground[0]=natural_height_at(
            tile_world_u+local_u+local_x,tile_world_v+1.f-local_v-local_y)-
            2.5f-wall_source_floor*(150.f/.82f)+.02f;
        if (farm_decal) {
            farm_shore[vertex_index] = vertex_ground[2];
            farm_world[vertex_index] = {tile_world_u+local_u+local_x,
                                        tile_world_v+1.0f-local_v-local_y};
        }
        float screen_x = center_x + (local_x - local_y) * half_w;
        float screen_y = center_y + (local_x + local_y) * half_h -
            local_z * 150.0f * feature_projection_scale -
            (vertex_ground[0] - ground_sample[0]) * relief_projection_scale;
        float normal_x = source.normal[0] * cosine - source.normal[1] * sine;
        float normal_y = source.normal[0] * sine + source.normal[1] * cosine;
        float ground_height_pixels = vertex_ground[0] * relief_projection_scale;
        float base_ground_y = center_y +
            ground_sample[0] * relief_projection_scale +
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
            static_cast<float>(asset.texture_index) + material_offset + owner_code + (ground_decal ? .35f : 0.f),
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
            1000.0f, 0.0f, 1000.0f, 0.0f, -1.0f};
        if (pickup_profile) {
            auto & vertex = transformed[vertex_index];
            vertex.world_x = tile_world_u + local_u + local_x;
            vertex.world_y = tile_world_v + 1.0f - local_v - local_y;
            vertex.world_z = (vertex_ground[0] + 2.5f + feature_height_tiles) / 112.0f;
            vertex.world_valid = 1.0f;
            if(world_objects){
                vertex.x=64.f+(local_u-local_v)*64.f+(local_x-local_y)*64.f;
                vertex.y=((local_u+local_v)*32.f-vertex_ground[0]*(128.f/224.f*.82f))+(local_x+local_y)*32.f-local_z*150.f*(128.f/224.f);
                vertex.z=feature_height_tiles;
            }
            auto normal=c3x_renderer::lighting::object_normal(normal_x,normal_y,source.normal[2]);
            vertex.normal_x=normal[0];vertex.normal_y=normal[1];vertex.normal_z=normal[2];
        }
    }
    if (farm_decal) {
        if(asset.id.find(":crop:")!=std::string::npos && !drape){
            float min_u=1e6f,max_u=-1e6f,min_v=1e6f,max_v=-1e6f;
            for(auto const& point:farm_world){
                min_u=std::min(min_u,point[0]);max_u=std::max(max_u,point[0]);
                min_v=std::min(min_v,point[1]);max_v=std::max(max_v,point[1]);
            }
            float min_wet_u=1e6f,max_wet_u=-1e6f,min_wet_v=1e6f,max_wet_v=-1e6f;
            for(unsigned row=0;row<=8u;++row)for(unsigned column=0;column<=8u;++column){
                float u=min_u+(max_u-min_u)*float(column)*.125f;
                float v=min_v+(max_v-min_v)*float(row)*.125f;
                if(relief_at_world(u,v)[2]<.025f){
                    min_wet_u=std::min(min_wet_u,u);max_wet_u=std::max(max_wet_u,u);
                    min_wet_v=std::min(min_wet_v,v);max_wet_v=std::max(max_wet_v,v);
                }
            }
            if(min_wet_u<=max_wet_u){
                float best=.4f,cut=0;unsigned axis=0;float sign=0;
                auto offer=[&](float retained,float limit,unsigned direction,float orientation){
                    if(retained>best){best=retained;cut=limit;axis=direction;sign=orientation;}
                };
                float width=max_u-min_u,height=max_v-min_v;
                if(width>0 && height>0){
                    offer((max_u-max_wet_u-.02f)/width,max_wet_u+.02f,0,1);
                    offer((min_wet_u-min_u-.02f)/width,min_wet_u-.02f,0,-1);
                    offer((max_v-max_wet_v-.02f)/height,max_wet_v+.02f,1,1);
                    offer((min_wet_v-min_v-.02f)/height,min_wet_v-.02f,1,-1);
                }
                if(sign!=0){
                    bool safe=false;
                    for(unsigned shift=0;shift<12u && !safe;++shift){
                        safe=true;
                        for(unsigned sample=0;sample<=16u;++sample){
                            float along=float(sample)*.0625f;
                            float u=axis==0?cut:min_u+width*along;
                            float v=axis==1?cut:min_v+height*along;
                            if(relief_at_world(u,v)[2]<.025f){safe=false;break;}
                        }
                        if(!safe)cut+=sign*.01f;
                    }
                    float remaining=axis==0?(sign>0?(max_u-cut)/width:(cut-min_u)/width):
                        (sign>0?(max_v-cut)/height:(cut-min_v)/height);
                    if(safe && remaining>=.4f)
                        for(std::size_t index=0;index<farm_shore.size();++index)
                            farm_shore[index]=std::min(farm_shore[index],
                                sign*(farm_world[index][axis]-cut));
                }
            }
        }
        struct ShoreVertex { Vertex vertex; float distance; };
        auto intersect=[](ShoreVertex const& a,ShoreVertex const& b){
            float t=a.distance/(a.distance-b.distance);
            ShoreVertex result{a.vertex,0};
            auto* output=reinterpret_cast<float*>(&result.vertex);
            auto const* from=reinterpret_cast<float const*>(&a.vertex);
            auto const* to=reinterpret_cast<float const*>(&b.vertex);
            for(unsigned i=0;i<sizeof(Vertex)/sizeof(float);++i)
                output[i]=from[i]+(to[i]-from[i])*t;
            return result;
        };
        std::size_t first_vertex=target.size(),first_index=topology?topology->size():0;
        float full_area=0,kept_area=0,low_u=1e6f,high_u=-1e6f,low_v=1e6f,high_v=-1e6f;
        auto area=[](float ax,float ay,float bx,float by,float cx,float cy){
            return std::abs((bx-ax)*(cy-ay)-(by-ay)*(cx-ax))*.5f;};
        for(std::size_t triangle=0;triangle+2<asset.indices.size();triangle+=3){
            std::array<ShoreVertex,4> polygon{};
            unsigned count=3;
            for(unsigned corner=0;corner<3;++corner){
                auto index=asset.indices[triangle+corner];
                polygon[corner]={transformed[index],farm_shore[index]};
            }
            if(drape){auto const& a=asset.vertices[asset.indices[triangle]].position;
                auto const& b=asset.vertices[asset.indices[triangle+1]].position;
                auto const& c=asset.vertices[asset.indices[triangle+2]].position;
                full_area+=area(a[0],a[1],b[0],b[1],c[0],c[1])*scale*scale;}
            std::array<ShoreVertex,4> clipped{};
            unsigned kept=0;
            for(unsigned corner=0;corner<count;++corner){
                auto const& a=polygon[corner];auto const& b=polygon[(corner+1)%count];
                bool a_land=a.distance>=0,b_land=b.distance>=0;
                if(a_land)clipped[kept++]=a;
                if(a_land!=b_land)clipped[kept++]=intersect(a,b);
            }
            for(unsigned corner=1;corner+1<kept;++corner){
                if(drape){auto const& a=clipped[0].vertex;auto const& b=clipped[corner].vertex;auto const& c=clipped[corner+1].vertex;
                    kept_area+=area(a.world_x,a.world_y,b.world_x,b.world_y,c.world_x,c.world_y);
                    for(auto const* point:{&a,&b,&c}){low_u=std::min(low_u,point->world_x);high_u=std::max(high_u,point->world_x);
                        low_v=std::min(low_v,point->world_y);high_v=std::max(high_v,point->world_y);}}
                for(unsigned index:{0u,corner,corner+1}){
                    if(topology){topology->push_back(unsigned(target.size()));
                        target.push_back(clipped[index].vertex);}
                    else target.push_back(clipped[index].vertex);
                }
            }
        }
        // A kit field that clipping cuts to a narrow sliver or a scrap is
        // dropped whole: its kept width (area over the kept extent, fringe
        // included) or area is too small for a field.
        if(drape && pickup_profile && kept_area>0 && kept_area<full_area*.5f &&
           (kept_area/std::hypot(high_u-low_u,high_v-low_v)<.07f || kept_area<.006f)){
            target.resize(first_vertex);
            if(topology)topology->resize(first_index);
        }
        return;
    }
    if(topology){
        unsigned base=unsigned(target.size());
        target.insert(target.end(),transformed.begin(),transformed.end());
        for(auto index:asset.indices)topology->push_back(base+index);
    }else for (std::uint32_t source_index : asset.indices)
        target.push_back(transformed[source_index]);
}
template<class Relief,class Height>
void append_route(Projection const& input,Route const& route,Relief relief_at_world,Height height_at_world,std::vector<Vertex>& route_vertices){
    float u0=route.u0,v0=route.v0,u1=route.u1,v1=route.v1;unsigned style=route.style;bool railroad=route.railroad;
    auto const& tile=input.tile;
    float left=input.left,top=input.top,half_w=input.half_w,half_h=input.half_h;
    float relief_projection_scale=input.relief_projection_scale;
    bool pickup_profile=input.pickup_profile,world_objects=input.world_objects;
    auto ndc_x=[](float x){return x;};auto ndc_y=[](float y){return y;};
    float tile_world_u = static_cast<float>(tile.tile_x + tile.tile_y) * 0.5f;
    float tile_world_v = static_cast<float>(tile.tile_x - tile.tile_y) * 0.5f;
    auto surface_height = [&](float world_u,float world_v) {
        return std::max(relief_at_world(world_u,world_v)[0],
            height_at_world(world_u,world_v)-2.5f);
    };
    if(route.bypass){
        auto skirt_anchor=[&](float& local_u,float& local_v){
            // Edge crossings stay fixed for the opposite tile's half. Find a
            // nearby low saddle only for the interior mountain-loop anchors.
            if(local_u<=.10f || local_u>=.90f || local_v<=.10f || local_v>=.90f)return;
            float source_u=local_u,source_v=local_v;
            float best=surface_height(tile_world_u+source_u,tile_world_v+1.0f-source_v);
            if(best-relief_at_world(tile_world_u+source_u,
                    tile_world_v+1.0f-source_v)[0]<10.0f)return;
            for(int y=-3;y<=3;++y)for(int x=-3;x<=3;++x){
                float trial_u=std::clamp(source_u+float(x)*.1f,.06f,.94f);
                float trial_v=std::clamp(source_v+float(y)*.1f,.06f,.94f);
                float distance=std::sqrt((trial_u-source_u)*(trial_u-source_u)+
                    (trial_v-source_v)*(trial_v-source_v));
                float score=surface_height(tile_world_u+trial_u,
                    tile_world_v+1.0f-trial_v)+distance*18.0f;
                if(score<best){best=score;local_u=trial_u;local_v=trial_v;}
            }
        };
        skirt_anchor(u0,v0);skirt_anchor(u1,v1);
    }
    if(!railroad && !route.bypass && std::abs(u1-.5f)+std::abs(v1-.5f)>.35f){
        // All incident routes start at one tile junction. On a raised hill or
        // mountain, search a small interior patch for a lower saddle so the
        // road winds around the crown instead of cutting across its peak.
        float center_u=tile_world_u+u0,center_v=tile_world_v+1.0f-v0;
        float crown=surface_height(center_u,center_v)-relief_at_world(center_u,center_v)[0];
        if(crown>12.0f){
            float best=surface_height(center_u,center_v),best_u=u0,best_v=v0;
            for(int y=-2;y<=2;++y)for(int x=-2;x<=2;++x){
                float candidate_u=std::clamp(u0+float(x)*.09f,.19f,.81f);
                float candidate_v=std::clamp(v0+float(y)*.09f,.19f,.81f);
                float score=surface_height(tile_world_u+candidate_u,
                    tile_world_v+1.0f-candidate_v)+
                    8.0f*std::sqrt(float(x*x+y*y))*.09f;
                if(score<best){best=score;best_u=candidate_u;best_v=candidate_v;}
            }
            u0=best_u;v0=best_v;
        }
    }
    constexpr int subdivisions = 32;
    float route_half_width = railroad ? 0.076f :
        (route.bridge ? (route.bridge_structural ? 0.064f : 0.031f) : 0.028f);
    float atlas_half_width = railroad ? 0.058f : 0.075f;
    float du = u1 - u0, dv = v1 - v0;
    float original_length = std::sqrt(du * du + dv * dv);
    if (original_length < 0.001f)
        return;
    float direction_u = du / original_length;
    float direction_v = dv / original_length;
    float bank_grade=0.0f;
    float bridge_span=route.bridge_structural?.26f:.48f;
    if(route.bridge && !route.bridge_structural){
        float crossing_u=route.u0+(route.u1-route.u0)*route.bridge_t;
        float crossing_v=route.v0+(route.v1-route.v0)*route.bridge_t;
        float a=surface_height(tile_world_u+crossing_u-direction_u*.55f,
            tile_world_v+1.0f-crossing_v+direction_v*.55f);
        float b=surface_height(tile_world_u+crossing_u+direction_u*.55f,
            tile_world_v+1.0f-crossing_v-direction_v*.55f);
        bank_grade=std::min(std::max(a,b),std::min(a,b)+20.0f);
    }
    float original_u1 = u1, original_v1 = v1;
    // Tile junctions and shared edge points are exact joins. Only a small
    // coverage bias is needed; long extensions made multiway knots.
    float start_overhang=0.008f,end_overhang=0.018f;
    u0 -= direction_u * start_overhang; v0 -= direction_v * start_overhang;
    u1 += direction_u * end_overhang; v1 += direction_v * end_overhang;
    du = u1 - u0; dv = v1 - v0;
    float length = std::sqrt(du * du + dv * dv);
    float perpendicular_u = -dv / length;
    float perpendicular_v = du / length;
    float bypass_offset=0.0f;
    if(route.bypass && !route.bridge){
        float best=1e30f;
        for(int candidate=-3;candidate<=3;++candidate){
            float offset=float(candidate)*.15f;
            float score=std::abs(offset)*25.0f;
            for(int sample=1;sample<8;++sample){
                float t=float(sample)*.125f;
                float bend=offset*std::sin(t*3.14159265359f);
                float local_u=u0+du*t+perpendicular_u*bend;
                float local_v=v0+dv*t+perpendicular_v*bend;
                float world_u=tile_world_u+local_u;
                float world_v=tile_world_v+1.0f-local_v;
                float raised=std::max(0.0f,surface_height(world_u,world_v)-
                    relief_at_world(world_u,world_v)[0]);
                score+=std::max(0.0f,raised-12.0f)*
                    std::max(0.0f,raised-12.0f)*.006f;
            }
            if(score<best){best=score;bypass_offset=offset;}
        }
    }
    float atlas_dx = 1.0f;
    float atlas_dy = 0.99021526f - 0.90606654f;
    float atlas_length = std::sqrt(atlas_dx * atlas_dx + atlas_dy * atlas_dy);
    float atlas_perpendicular_u = -atlas_dy / atlas_length;
    float atlas_perpendicular_v = atlas_dx / atlas_length;
    auto route_hash=[](unsigned value){
        value ^= value >> 16; value *= 0x7feb352du;
        value ^= value >> 15; value *= 0x846ca68bu;
        return value ^ (value >> 16);
    };
    // Both halves of a logical edge share the same width and atlas variation.
    // The edge midpoint is independent of which tile prepared its half.
    unsigned edge_u = static_cast<unsigned>((tile_world_u + original_u1) * 2.0f);
    unsigned edge_v = static_cast<unsigned>((tile_world_v + 1.0f - original_v1) * 2.0f);
    unsigned seed = route_hash(edge_u * 73856093u ^ edge_v * 19349663u);
    float wave_phase = float(seed & 0xffffu) / 65536.0f * 6.28318530718f;
    float width_variation = railroad ? 1.0f :
        0.78f + float(route_hash(seed ^ 0x9e3779b9u) & 0xffffu) / 65536.0f * 0.44f;
    auto road_bend = [&](float along) {
        if (route.bridge || railroad) return 0.0f;
        float curve_t = std::clamp((along * length - start_overhang) / original_length, 0.0f, 1.0f);
        if (curve_t <= 0.0f || curve_t >= 1.0f) return 0.0f;
        if(route.bypass)return bypass_offset*std::sin(curve_t*3.14159265359f);
        float center_u = tile_world_u + u0 + du * along;
        float center_v = tile_world_v + 1.0f - (v0 + dv * along);
        float plus_height = surface_height(
            center_u + perpendicular_u * 0.18f,
            center_v - perpendicular_v * 0.18f);
        float minus_height = surface_height(
            center_u - perpendicular_u * 0.18f,
            center_v + perpendicular_v * 0.18f);
        return std::clamp((minus_height - plus_height) * 0.003f,
            -0.16f,0.16f) * std::sin(curve_t * 3.14159265359f);
    };
    auto route_vertex = [&](float along, float across, float terrain_bend, float deck_drop = 0.0f) {
        float source_along = (along * length - start_overhang) / original_length;
        float curve_t = std::clamp(source_along, 0.0f, 1.0f);
        // The midpoint is the shared river edge. Keep its crossing straight
        // and centered under the authored bridge body.
        float curve_envelope = std::sin(curve_t * 6.28318530718f);
        float curve_amplitude = route.bridge ? 0.0f :
            (railroad ? 0.020f : (route.bypass ? 0.025f : 0.085f));
        float road_wave = curve_envelope * curve_amplitude *
            (0.62f * std::sin(wave_phase) +
             0.38f * std::sin(curve_t * 6.28318530718f + wave_phase));
        float center_u=u0+du*along+perpendicular_u*(road_wave+terrain_bend);
        float center_v=v0+dv*along+perpendicular_v*(road_wave+terrain_bend);
        float center_world_u=tile_world_u+center_u;
        float center_world_v=tile_world_v+1.0f-center_v;
        float crown=surface_height(center_world_u,center_world_v)-
            relief_at_world(center_world_u,center_world_v)[0];
        // A large rocky crown can extend into adjacent road tiles. Taper the
        // visible trail into its occluding rock mass instead of projecting a
        // bright stripe up a near-vertical face.
        float rock_clearance=route.bridge &&
            std::abs((curve_t-route.bridge_t)*original_length)<bridge_span?1.0f:
            std::clamp((50.0f-crown)/20.0f,0.0f,1.0f);
        float route_u = u0 + du * along + perpendicular_u *
            (route_half_width * width_variation * across * rock_clearance + road_wave + terrain_bend);
        float route_v = v0 + dv * along + perpendicular_v *
            (route_half_width * width_variation * across * rock_clearance + road_wave + terrain_bend);
        float atlas_along = source_along;
        if (!railroad) {
            float edge_along = route.reverse ? 1.0f - curve_t * 0.5f : curve_t * 0.5f;
            if ((seed & 1u) != 0u) edge_along = 1.0f - edge_along;
            atlas_along = float((seed >> 1) % 3u) * 0.18f + edge_along * 0.60f;
        }
        float atlas_u = atlas_dx * atlas_along +
            atlas_perpendicular_u * atlas_half_width * across;
        float atlas_v = 0.90606654f + atlas_dy * atlas_along +
            atlas_perpendicular_v * atlas_half_width * across;
        float world_u = tile_world_u + route_u;
        float world_v = tile_world_v + (1.0f - route_v);
        float ground_height = surface_height(world_u,world_v);
        if(route.bridge){
            // Authored arches need a continuous roadbed at their crown. A
            // sampled crossing without an arch follows nearby bank grade so
            // its narrow deck stays above the river's carved bed.
            float crossing_distance=std::abs((curve_t-route.bridge_t)*original_length);
            float ramp=std::clamp((bridge_span-crossing_distance)/bridge_span,0.0f,1.0f);
            ramp=ramp*ramp*(3.0f-2.0f*ramp);
            if(route.bridge_structural)ground_height+=13.0f*ramp;
            else ground_height=std::max(ground_height,
                ground_height+(bank_grade+8.0f-ground_height)*ramp);
            ground_height-=deck_drop;
        }
        float ground_x = left + half_w + (route_u - route_v) * half_w;
        float ground_y = top + (route_u + route_v) * half_h;
        // Keep the decal just above the sampled land surface. The bias is in
        // screen pixels and does not flatten relief or detach the route.
        float h = ground_height * relief_projection_scale + 0.65f;
        float depth =
            ground_y + h * 0.75f;
        Vertex vertex{
            ndc_x(ground_x), ndc_y(ground_y - h), depth,
            atlas_u, atlas_v, 1.0f, 0.0f, 0.0f, 1.0f,
            across, curve_t, atlas_dx * atlas_along,
            0.90606654f + atlas_dy * atlas_along,
            11.0f, 0.0f, static_cast<float>(style), 0.0f,
            route_u, route_v, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
            1000.0f, 0.0f, 1000.0f, 0.0f, -1.0f};
        if (!pickup_profile) {
            constexpr float normal_step = 0.025f;
            float slope_u = (surface_height(world_u + normal_step, world_v) -
                surface_height(world_u - normal_step, world_v)) *
                relief_projection_scale / (2.0f * normal_step * input.tile_width);
            float slope_v = (surface_height(world_u, world_v + normal_step) -
                surface_height(world_u, world_v - normal_step)) *
                relief_projection_scale / (2.0f * normal_step * input.tile_width);
            float inverse_length = 1.0f / std::sqrt(slope_u * slope_u + slope_v * slope_v + 1.0f);
            vertex.normal_x = -slope_u * inverse_length;
            vertex.normal_y = -slope_v * inverse_length;
            vertex.normal_z = inverse_length;
        }
        if (pickup_profile) {
            vertex.world_x = world_u;
            vertex.world_y = world_v;
            vertex.world_z = (ground_height + 9.0f) / 112.0f;
            vertex.world_valid = 1.0f;
            if(world_objects){vertex.x=64.f+(route_u-route_v)*64.f;
                vertex.y=(route_u+route_v)*32.f-ground_height*(128.f/224.f*.82f)-.65f;
                vertex.z=0;}
        }
        return vertex;
    };
    std::array<Vertex, subdivisions + 1> left_vertices{},right_vertices{};
    std::array<float, subdivisions + 1> crown_samples{};
    std::array<float, subdivisions + 1> route_heights{};
    for (int sample = 0; sample <= subdivisions; ++sample) {
        float along = static_cast<float>(sample) / subdivisions;
        float bend = road_bend(along);
        left_vertices[sample] = route_vertex(along, -1.0f, bend);
        right_vertices[sample] = route_vertex(along, 1.0f, bend);
        if(!route.bridge && !railroad){
            float center_u=(left_vertices[sample].material_grass+
                right_vertices[sample].material_grass)*.5f;
            float center_v=(left_vertices[sample].material_plains+
                right_vertices[sample].material_plains)*.5f;
            float world_u=tile_world_u+center_u;
            float world_v=tile_world_v+1.0f-center_v;
            route_heights[sample]=surface_height(world_u,world_v);
            crown_samples[sample]=route_heights[sample]-relief_at_world(world_u,world_v)[0];
        }
    }
    for (int segment = 0; segment < subdivisions; ++segment) {
        // An exposed mountain face is too steep for a decal trail. The
        // neighboring skirt pieces remain, while its rock occludes this gap.
        // Trim a few samples around the face so no short steep stubs remain.
        bool exposed=false;
        if(!route.bridge && !railroad && !route.isolated)
            for(int adjacent_sample=std::max(0,segment-4);
                adjacent_sample<=std::min(subdivisions,segment+5);++adjacent_sample){
                exposed|=crown_samples[adjacent_sample]>20.0f;
                if(adjacent_sample<subdivisions)
                    exposed|=std::abs(route_heights[adjacent_sample+1]-
                        route_heights[adjacent_sample])>2.5f;
            }
        if(exposed)continue;
        Vertex const& left0 = left_vertices[segment];
        Vertex const& right0 = right_vertices[segment];
        Vertex const& right1 = right_vertices[segment + 1];
        Vertex const& left1 = left_vertices[segment + 1];
        Vertex triangles[] = {left0, right0, right1, left0, right1, left1};
        route_vertices.insert(route_vertices.end(), std::begin(triangles), std::end(triangles));
    }
    if(route.bridge){
        // A narrow fascia makes the new surface a solid deck when the river
        // and bank are visible beneath the imported side arches.
        constexpr int deck_segments=12;
        float deck_start=std::max(0.0f,route.bridge_t-bridge_span/original_length);
        float deck_end=std::min(1.0f+end_overhang/original_length,
            route.bridge_t+bridge_span/original_length);
        for(int segment=0;segment<deck_segments;++segment){
            float source_a=deck_start+(deck_end-deck_start)*float(segment)/deck_segments;
            float source_b=deck_start+(deck_end-deck_start)*float(segment+1)/deck_segments;
            float a=(source_a*original_length+start_overhang)/length;
            float b=(source_b*original_length+start_overhang)/length;
            for(float side:{-1.0f,1.0f}){
                Vertex top_a=route_vertex(a,side,0.0f),top_b=route_vertex(b,side,0.0f);
                Vertex low_a=route_vertex(a,side,0.0f,4.0f),low_b=route_vertex(b,side,0.0f,4.0f);
                float normal_u=perpendicular_u*side,normal_v=-perpendicular_v*side;
                for(Vertex* vertex:{&top_a,&top_b,&low_a,&low_b}){
                    vertex->normal_x=normal_u;vertex->normal_y=normal_v;
                    vertex->normal_z=0.0f;
                }
                Vertex fascia[]={top_a,low_a,top_b,top_b,low_a,low_b};
                route_vertices.insert(route_vertices.end(),std::begin(fascia),std::end(fascia));
            }
        }
    }
}
// A terrain-draped strip along one road-pattern centerline. Its stroke keeps
// the source pattern's screen width on flat ground and on slopes, so a path
// running down the screen is wider in world units than one running across it,
// and a path across a steep face narrows instead of smearing down it. The
// shader draws a pixel-sharp stroke edge and lets the authored worn track
// shape its opacity, so the terrain's grain reads through the path. The tiled
// path piece keeps one texel aspect along and across and mirrors back and
// forth along longer strips (the decal sampler clamps).
template<class Relief,class Height>
void append_pattern_route(Projection const& input,RoutePatterns const& patterns,PatternRoute const& route,
        Relief relief_at_world,Height height_at_world,std::vector<Vertex>& route_vertices){
    // A railroad (style 4) follows the same pattern as a road, a little wider
    // so its sleepers and both rails read at gameplay zoom.
    bool railroad=route.style>=4u;
    float stroke_across=railroad?5.f:3.5f,stroke_down=railroad?6.f:4.5f; // pixels at a 128-pixel tile
    // Margin beyond the stroke: a road's worn shoulders, a railroad's wider
    // dirt bed (Civ VI lays its rail pieces over a dirt road piece).
    float fringe=railroad?2.f:1.5f;
    constexpr float piece_v=.94814090f,piece_core=.0156f;// tiled path center and opaque half-height
    // Piece units per tile along the path. The road piece keeps its texel
    // aspect; the rail strip (16 sleepers per unit, its stroke spanning 43.5
    // of 256 texels across) keeps its sleepers square at a nominal stroke.
    float texture_scale=railroad?1.98f:piece_core/.031f;
    constexpr float bridge_reach=.2f;                    // a bridged path ends under the bridge's end
    if(route.line>=patterns.lines.size())return;
    auto const& line=patterns.lines[route.line];
    auto const& tile=input.tile;
    float tile_world_u=float(tile.tile_x+tile.tile_y)*.5f,tile_world_v=float(tile.tile_x-tile.tile_y)*.5f;
    std::vector<std::array<float,2>> points=route.points.size()==line.count?route.points:
        std::vector<std::array<float,2>>(patterns.points.begin()+line.first,patterns.points.begin()+line.first+line.count);
    std::size_t count=points.size();
    bool shared_start=line.start>=0 && (route.joins[0]!=0.f || route.joins[1]!=0.f);
    bool shared_end=line.end>=0 && (route.joins[2]!=0.f || route.joins[3]!=0.f);
    // Both tiles at a shared join ease their path onto its shared axis, so
    // the two halves pass through the join tangent to each other instead of
    // meeting at a corner. An authored bridge lies square to its river edge:
    // there the path runs fully onto the axis before it reaches the bridge.
    for(unsigned side=0;side<2;++side){
        if(!(side?shared_end:shared_start))continue;
        bool bridged=route.bridges&(1u<<side);
        float full=bridged?.24f:.04f,ease=bridged?.16f:.18f;
        auto join_point=side?points.back():points.front();
        float axis_u=route.joins[side*2],axis_v=route.joins[side*2+1];
        for(auto& point:points){
            float du=point[0]-join_point[0],dv=point[1]-join_point[1];
            float weight=std::clamp((full+ease-std::hypot(du,dv))/ease,0.f,1.f);
            weight=weight*weight*(3.f-2.f*weight);
            float along=du*axis_u+dv*axis_v;
            point[0]+=(join_point[0]+axis_u*along-point[0])*weight;
            point[1]+=(join_point[1]+axis_v*along-point[1])*weight;
        }
    }
    // A path over water fades out over a few points toward the bank.
    std::vector<float> ford(count,0.f);
    if(route.wet.size()==count)for(std::size_t index=0;index<count;++index)
        for(std::size_t other=index>4?index-4:0;other<std::min(count,index+5);++other)if(route.wet[other])
            ford[index]=std::max(ford[index],1.f-float(index>other?index-other:other-index)/5.f);
    // A path also fades out as it climbs a mountain (the same fade the shader
    // applies to a ford).
    if(route.fade.size()==count)for(std::size_t index=0;index<count;++index)
        ford[index]=std::max(ford[index],route.fade[index]);
    auto direction=[&](std::size_t index){
        if(index==0 && shared_start)return std::array<float,2>{-route.joins[0],-route.joins[1]};
        if(index+1==count && shared_end)return std::array<float,2>{route.joins[2],route.joins[3]};
        auto const& a=points[index==0?0:index-1];auto const& b=points[std::min(index+1,count-1)];
        float du=b[0]-a[0],dv=b[1]-a[1],length=std::max(std::hypot(du,dv),1e-6f);
        return std::array<float,2>{du/length,dv/length};
    };
    // A shared join ends exactly on vertices the neighbor computes too. Other
    // ends (local junctions, a join without a known neighbor) reach past the
    // node so meeting branches leave no notch: about one stroke for a road,
    // about half its own half-width for a railroad, whose structured bed
    // would otherwise show as a stub.
    float core_estimate=stroke_across*std::sqrt(2.f)/128.f*(railroad?.45f:1.f);
    for(unsigned side=0;side<2;++side){
        if(side?shared_end:shared_start)continue;
        float reach=core_estimate;
        auto toward=direction(side?count-1:0);
        auto& point=side?points.back():points.front();
        float sign=side?1.f:-1.f;
        point[0]+=toward[0]*reach*sign;point[1]+=toward[1]*reach*sign;
    }
    std::vector<float> distance(count,0.f);
    for(std::size_t index=1;index<count;++index)
        distance[index]=distance[index-1]+std::hypot(points[index][0]-points[index-1][0],points[index][1]-points[index-1][1]);
    float length=distance.back();
    // The bridge stands on the river's own crossing; its ends move with it.
    float low=route.bridges&1u?bridge_reach-route.crossing[0]:0.f,
        high=length-(route.bridges&2u?bridge_reach-route.crossing[1]:0.f);
    if(length<.002f || high-low<.002f)return;
    // The same visible surface as railroads; pickup relief alone can sit
    // well below the natural ground.
    auto ground=[&](float world_u,float world_v){
        return std::max(relief_at_world(world_u,world_v)[0],height_at_world(world_u,world_v)-2.5f);
    };
    unsigned seed=c3x_renderer::stable_hash(tile.variant_seed^route.line*0x9e3779b9u^
        unsigned(tile.tile_x)*73856093u^unsigned(tile.tile_y)*19349663u);
    // Anchor the piece at a shared join so both tiles meet on the same texel.
    auto texture=[&](float along){
        if(line.start>=0)return along*texture_scale;
        if(line.end>=0)return (length-along)*texture_scale;
        return along*texture_scale+c3x_renderer::stable_random(seed);
    };
    float left=input.left,top=input.top,half_w=input.half_w,half_h=input.half_h;
    float relief_projection_scale=input.relief_projection_scale;
    // The owner's ground material for the shader's height blend: 0 grass,
    // 1 plains, 2 desert, 3 hills, 4 mountain, 5 marsh (Civ III square types).
    int real=tile.real_terrain_type,base=tile.terrain_type;
    float ground_kind=real==5?3.f:real==6 || real==10?4.f:real==9?5.f:base==1?1.f:base==0 || base==4?2.f:0.f;
    struct Station {float u,v,normal_u,normal_v,core,ford,along,coordinate;};
    auto station=[&](std::size_t index){
        auto tangent=direction(index);
        float normal_u=-tangent[1],normal_v=tangent[0];
        // Screen offsets of a tile-local step, flat and on the local slope.
        float world_u=tile_world_u+points[index][0],world_v=tile_world_v+1.f-points[index][1];
        constexpr float e=.03f;
        float slope_u=(ground(world_u+e,world_v)-ground(world_u-e,world_v))/(2*e);
        float slope_v=(ground(world_u,world_v-e)-ground(world_u,world_v+e))/(2*e);
        auto thickness=[&](float lift){
            float tx=(tangent[0]-tangent[1])*half_w,nx=(normal_u-normal_v)*half_w;
            float ty=(tangent[0]+tangent[1])*half_h-(slope_u*tangent[0]+slope_v*tangent[1])*relief_projection_scale*lift;
            float ny=(normal_u+normal_v)*half_h-(slope_u*normal_u+slope_v*normal_v)*relief_projection_scale*lift;
            return std::abs(nx*ty-ny*tx)/std::max(std::hypot(tx,ty),1e-6f);
        };
        float screen=std::hypot(tangent[0]-tangent[1],(tangent[0]+tangent[1])*.5f);
        float down=std::abs(tangent[0]+tangent[1])*.5f/std::max(screen,1e-6f);
        float core=(stroke_across+(stroke_down-stroke_across)*down)*screen/128.f;
        // A slope across the path stretches the flat width down the screen;
        // narrow it there (and widen a foreshortened back slope a little).
        core*=std::clamp(thickness(0.f)/std::max(thickness(1.f),1e-6f),.25f,1.4f);
        return Station{points[index][0],points[index][1],normal_u,normal_v,core,ford[index],
            distance[index]/length,texture(distance[index])};
    };
    auto vertex_at=[&](Station const& s,float across){
        float route_u=s.u+s.normal_u*s.core*fringe*across;
        float route_v=s.v+s.normal_v*s.core*fringe*across;
        float world_u=tile_world_u+route_u,world_v=tile_world_v+1.f-route_v;
        float height=ground(world_u,world_v);
        constexpr float e=.01f;
        float slope_u=(ground(world_u+e,world_v)-ground(world_u-e,world_v))/(2*e);
        float slope_v=(ground(world_u,world_v+e)-ground(world_u,world_v-e))/(2*e);
        float ground_x=left+half_w+(route_u-route_v)*half_w;
        float ground_y=top+(route_u+route_v)*half_h;
        float h=height*relief_projection_scale+.65f;
        // The stroke edge sits at |x|=.575 (the older ribbon shader closes its
        // center ribbon there too). A ford also maps to the piece's
        // transparent margin and closes that ribbon, for shaders that do not
        // read the explicit fade below.
        float shape=across*fringe*.575f;
        Vertex vertex{
            ground_x,ground_y-h,ground_y+h*.75f,
            s.coordinate,piece_v+across*piece_core*fringe+(.9985f-piece_v-across*piece_core*fringe)*s.ford,1.f,0.f,0.f,1.f,
            shape+(2.f-shape)*s.ford,s.along,s.coordinate,piece_v,
            11.f,ground_kind,float(route.style),0.f,
            // The shader clips route pixels to the owner's diamond through
            // these weights; a pattern strip may cross into a neighbor at its
            // join, so it stays inside that test. The third weight fades a
            // ford into its bank; the fourth selects the pattern-road shading.
            .5f,.5f,s.ford,1.f,
            0.f,0.f,1.f,
            1000.f,0.f,1000.f,0.f,-1.f};
        // Same world normal basis as the natural ground under the path.
        float n[]={-slope_u/128,-slope_v/128,1.f};
        float inverse=1.f/std::sqrt(n[0]*n[0]+n[1]*n[1]+1.f);
        vertex.normal_x=n[0]*inverse;vertex.normal_y=n[1]*inverse;vertex.normal_z=inverse;
        if(input.pickup_profile){
            vertex.world_x=world_u;vertex.world_y=world_v;
            vertex.world_z=(height+9.f)/112.f;vertex.world_valid=1.f;
            if(input.world_objects){vertex.x=64.f+(route_u-route_v)*64.f;
                vertex.y=(route_u+route_v)*32.f-height*(128.f/224.f*.82f)-.65f;
                vertex.z=0;}
        }
        return vertex;
    };
    auto between=[](Station const& a,Station const& b,float t){
        Station s;
        s.u=a.u+(b.u-a.u)*t;s.v=a.v+(b.v-a.v)*t;
        s.normal_u=a.normal_u+(b.normal_u-a.normal_u)*t;s.normal_v=a.normal_v+(b.normal_v-a.normal_v)*t;
        float n=std::max(std::hypot(s.normal_u,s.normal_v),1e-6f);s.normal_u/=n;s.normal_v/=n;
        s.core=a.core+(b.core-a.core)*t;s.ford=a.ford+(b.ford-a.ford)*t;
        s.along=a.along+(b.along-a.along)*t;s.coordinate=a.coordinate+(b.coordinate-a.coordinate)*t;
        return s;
    };
    // Mirror the road piece back and forth inside its interior. Joins and
    // turns then never sample the atlas's outer columns, and the path stays
    // continuous without a wrap discontinuity. Quads split at each turn. The
    // rail strip tiles seamlessly, so its sleepers keep an even spacing on an
    // unwrapped coordinate (the shader samples it with a wrapping sampler).
    auto mirrored=[&](float coordinate){
        if(railroad)return coordinate;
        float phase=coordinate-2.f*std::floor(coordinate*.5f);
        return .03f+.94f*(1.f-std::abs(1.f-phase));
    };
    auto quad=[&](Vertex a0,Vertex a1,Vertex b0,Vertex b1){
        for(Vertex* vertex:{&a0,&a1,&b0,&b1}){vertex->u=mirrored(vertex->u);vertex->macro_u=vertex->u;}
        Vertex triangles[]={a0,a1,b1,a0,b1,b0};
        route_vertices.insert(route_vertices.end(),std::begin(triangles),std::end(triangles));
    };
    auto pair=[&](Station const& s){return std::array<Vertex,2>{vertex_at(s,-1.f),vertex_at(s,1.f)};};
    auto span=[&](Station const& a,Station const& b,std::array<Vertex,2> const& from,std::array<Vertex,2> const& to){
        float lowest=std::min(a.coordinate,b.coordinate),highest=std::max(a.coordinate,b.coordinate);
        float turn=std::ceil(lowest);
        if(turn>lowest && turn<highest){
            Station m=between(a,b,(turn-a.coordinate)/(b.coordinate-a.coordinate));
            auto middle=pair(m);
            quad(from[0],from[1],middle[0],middle[1]);
            quad(middle[0],middle[1],to[0],to[1]);
        }else quad(from[0],from[1],to[0],to[1]);
    };
    Station previous=station(0);
    auto previous_pair=pair(previous);
    for(std::size_t index=1;index<count;++index){
        Station next=station(index);
        auto next_pair=pair(next);
        float d0=distance[index-1],d1=distance[index];
        if(d1>low && d0<high && d1>d0){
            // The authored bridge deck carries the path over the river.
            Station a=previous,b=next;auto from=previous_pair,to=next_pair;
            if(d0<low){a=between(previous,next,(low-d0)/(d1-d0));from=pair(a);}
            if(d1>high){b=between(previous,next,(high-d0)/(d1-d0));to=pair(b);}
            span(a,b,from,to);
        }
        previous=next;previous_pair=next_pair;
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
    auto append_route_segment=[&](float a,float b,float c,float d,unsigned style,bool railroad,bool bridge,bool reverse,bool bypass=false){plan.routes.push_back({a,b,c,d,style,railroad,bridge,reverse,bypass});};
    if (route_assets_ready && (tile.road_mask != 0 || tile.railroad_mask != 0)) {
        auto route_hash=[](unsigned value){
            value ^= value >> 16; value *= 0x7feb352du;
            value ^= value >> 15; value *= 0x846ca68bu;
            return value ^ (value >> 16);
        };
        unsigned center_seed=route_hash(tile.variant_seed ^
            static_cast<unsigned>(tile.tile_x)*73856093u ^
            static_cast<unsigned>(tile.tile_y)*19349663u);
        float center_u=0.5f,center_v=0.5f;
        // With a connection-pattern pack, routes follow the source game's
        // rule: one pattern per 8-neighbor mask. A road links road neighbors
        // except where both tiles carry a railroad; a railroad links railroad
        // neighbors and follows the same patterns as roads.
        bool patterns=assets.road_patterns!=nullptr;
        unsigned pattern_mask=0,pattern_bridges=0,rail_mask=0,rail_bridges=0;
        bool mountain_ring=tile.real_terrain_type==6 && tile.road_mask && !tile.railroad_mask && !patterns;
        if (tile.road_mask && !tile.railroad_mask && !mountain_ring) {
            center_u += (float(center_seed & 0xffffu)/65535.0f-0.5f)*0.38f;
            center_v += (float((center_seed >> 16) & 0xffffu)/65535.0f-0.5f)*0.38f;
        }
        constexpr float ring_u[8]={.5f,.88f,.93f,.88f,.5f,.12f,.07f,.12f};
        constexpr float ring_v[8]={.07f,.12f,.5f,.88f,.93f,.88f,.5f,.12f};
        constexpr int route_offsets[8][2] = {
            {1, -1}, {2, 0}, {1, 1}, {0, 2},
            {-1, 1}, {-2, 0}, {-1, -1}, {0, -2}
        };
        constexpr unsigned river_edge_bits[8] = {2u, 0u, 8u, 0u, 32u, 0u, 128u, 0u};
        constexpr unsigned opposite_river_bits[8] = {32u, 0u, 128u, 0u, 2u, 0u, 8u, 0u};
        float base_world_u = static_cast<float>(tile.tile_x + tile.tile_y) * 0.5f;
        float base_world_v = static_cast<float>(tile.tile_x - tile.tile_y) * 0.5f;
        bool connected = false;
        unsigned mountain_spokes=0;
        for (int direction = 0; direction < 8; ++direction) {
            int neighbor_x = tile.tile_x + route_offsets[direction][0];
            int neighbor_y = tile.tile_y + route_offsets[direction][1];
            auto found = lookup(neighbor_x, neighbor_y);
            if (found == nullptr)
                continue;
            c3x_renderer_tile_v1 const & neighbor = found->occurrence;
            bool railroad = tile.railroad_mask != 0 && neighbor.railroad_mask != 0;
            bool road = tile.road_mask != 0 && neighbor.road_mask != 0;
            unsigned edge_seed=route_hash(
                static_cast<unsigned>(tile.tile_x+neighbor_x)*73856093u ^
                static_cast<unsigned>(tile.tile_y+neighbor_y)*19349663u);
            if (road && !railroad && (direction & 1) && !patterns) {
                // A corner-to-corner link is redundant when an incident tile
                // already carries the route around that corner. Civ III has a
                // road bit, not authored edge bits; this keeps dense late-game
                // maps legible while sparse diagonal connections still work.
                int dx = route_offsets[direction][0];
                int dy = route_offsets[direction][1];
                int mid_x = tile.tile_x + dx / 2;
                int mid_y = tile.tile_y + dy / 2;
                for (int side : {-1, 1}) {
                    auto mid = lookup(mid_x + (dx == 0 ? side : 0),
                                      mid_y + (dy == 0 ? side : 0));
                    if (mid != nullptr && mid->occurrence.road_mask != 0) {
                        // Keep a minority of redundant long links: dense
                        // networks gain irregular through-tile crossings.
                        road = edge_seed % 10u == 0u;
                        break;
                    }
                }
            }
            if (!railroad && !road)
                continue;
            connected = true;
            if(mountain_ring && road)mountain_spokes|=1u<<direction;
            float end_u = (static_cast<float>(neighbor_x + neighbor_y) * 0.5f + 0.5f) -
                base_world_u;
            float end_v = 1.0f - ((static_cast<float>(neighbor_x - neighbor_y) * 0.5f + 0.5f) -
                base_world_v);
            unsigned style = railroad ? 4u : static_cast<unsigned>(
                std::clamp(tile.route_style, 0, 3));
            bool bridge = river_edge_bits[direction] != 0 &&
                (((tile.river_code & river_edge_bits[direction]) != 0) ||
                 ((neighbor.river_code & opposite_river_bits[direction]) != 0));
            float edge_u=(0.5f+end_u)*0.5f,edge_v=(0.5f+end_v)*0.5f;
            if (!railroad && !bridge) {
                float axis_u=end_u-0.5f,axis_v=end_v-0.5f;
                float axis_length=std::sqrt(axis_u*axis_u+axis_v*axis_v);
                float shift=(float((edge_seed >> 8)&0xffffu)/65535.0f-0.5f)*0.10f;
                float orientation=direction<4?1.0f:-1.0f;
                edge_u += -axis_v/axis_length*shift*orientation;
                edge_v += axis_u/axis_length*shift*orientation;
            }
            if(patterns){
                unsigned& mask=railroad?rail_mask:pattern_mask;
                unsigned& bridges=railroad?rail_bridges:pattern_bridges;
                mask|=1u<<direction;
                if(bridge)bridges|=1u<<direction;
            }else if(routes_enabled)
                append_route_segment(mountain_ring?ring_u[direction]:center_u,
                    mountain_ring?ring_v[direction]:center_v,edge_u,edge_v,
                    style, railroad, bridge, direction >= 4,mountain_ring);
            if (bridge && direction < 4 && !patterns) {
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
        if(patterns){
            // Both tiles at a join share one axis: the average of their paths'
            // directions there. Strip ends then meet on identical vertices, and
            // an authored bridge over a river edge follows the same road line.
            auto outward=[&](RoutePatterns const& set,unsigned mask,int join,float& u,float& v){
                for(unsigned index=set.offsets[mask];index<set.offsets[mask+1];++index){
                    auto const& line=set.lines[index];
                    if(line.start!=join && line.end!=join)continue;
                    auto const* p=set.points.data()+line.first;unsigned last=line.count-1u;
                    unsigned end=line.end==join?last:0u,back=line.end==join?(last>3u?last-3u:0u):std::min(3u,last);
                    u=p[end][0]-p[back][0];v=p[end][1]-p[back][1];
                    float length=std::hypot(u,v);
                    if(length<1e-5f)return false;
                    u/=length;v/=length;return true;
                }
                return false;
            };
            // A diagonal link crosses a river at a shared corner when the river
            // edges meeting there separate the two linked tiles. Where both
            // diagonals of one corner cross, only the across-screen link
            // carries a bridge, so bridges never stack.
            auto at=[&](int x,int y)->c3x_renderer_tile_v1 const*{
                if(x==tile.tile_x && y==tile.tile_y)return &tile;
                auto found=lookup(x,y);return found?&found->occurrence:nullptr;};
            auto direction_of=[&](int dx,int dy){
                for(int k=0;k<8;++k)if(route_offsets[k][0]==dx && route_offsets[k][1]==dy)return k;
                return -1;
            };
            auto river_edge=[&](c3x_renderer_tile_v1 const* a,c3x_renderer_tile_v1 const* b,int k){
                return a && b && k>=0 && ((a->river_code&river_edge_bits[k])!=0 || (b->river_code&opposite_river_bits[k])!=0);
            };
            // The four river edges that may meet at a corner, as T-P1, X-P1,
            // T-P2 and X-P2 (P1, P2 flank the corner; X is the linked tile).
            auto corner_edges=[&](int tx,int ty,int corner){
                int k1=(corner+7)&7,k2=(corner+1)&7;
                auto t=at(tx,ty);
                auto x=at(tx+route_offsets[corner][0],ty+route_offsets[corner][1]);
                auto p1=at(tx+route_offsets[k1][0],ty+route_offsets[k1][1]);
                auto p2=at(tx+route_offsets[k2][0],ty+route_offsets[k2][1]);
                int x1=direction_of(route_offsets[k1][0]-route_offsets[corner][0],route_offsets[k1][1]-route_offsets[corner][1]);
                int x2=direction_of(route_offsets[k2][0]-route_offsets[corner][0],route_offsets[k2][1]-route_offsets[corner][1]);
                return std::array<bool,4>{river_edge(t,p1,k1),river_edge(x,p1,x1),river_edge(t,p2,k2),river_edge(x,p2,x2)};
            };
            // Bridges stand only on the tile-diagonal river edges (NE, SE, SW,
            // NW). A link through a corner that the river separates fords: it
            // fades into each bank. Roads and railroads are separate networks
            // of the same patterns; roads keep one dirt look across eras.
            auto network=[&](unsigned mask,unsigned bridge_mask,bool railroad,bool draw){
                // Each network draws its own sheet's pattern; a fully connected
                // tile picks that sheet's variant, as its neighbors do for it.
                auto const& set=*assets.patterns_for(railroad?4u:0u);
                unsigned own=set.index(mask,tile.tile_x,tile.tile_y);
                unsigned fords=0;
                std::array<std::array<float,2>,8> open{};
                for(int corner=1;corner<8;corner+=2){
                    if(!(mask>>corner&1u))continue;
                    auto edges=corner_edges(tile.tile_x,tile.tile_y,corner);
                    if((edges[0] || edges[1]) && (edges[2] || edges[3])){fords|=1u<<corner;continue;}
                    // A river that bends at (or ends on) the corner without
                    // separating the linked tiles: the join moves toward the
                    // open bank, away from the river edges meeting there.
                    // Both tiles derive the same direction from the same edges.
                    int k1=(corner+7)&7,k2=(corner+1)&7;
                    auto world=[&](int k){return std::array<float,2>{(route_offsets[k][0]+route_offsets[k][1])*.5f,
                        (route_offsets[k][0]-route_offsets[k][1])*.5f};};
                    auto cx=world(corner),c1=world(k1),c2=world(k2);
                    std::array<float,2> ends[4]={c1,{c1[0]+cx[0],c1[1]+cx[1]},c2,{c2[0]+cx[0],c2[1]+cx[1]}};
                    float u=0,v=0;
                    for(int edge=0;edge<4;++edge)if(edges[edge]){
                        // Each edge leaves the corner toward its midpoint.
                        float du=ends[edge][0]*.5f-cx[0]*.5f,dv=ends[edge][1]*.5f-cx[1]*.5f,length=std::hypot(du,dv);
                        u-=du/length;v-=dv/length;
                    }
                    float length=std::hypot(u,v);
                    if(length>1e-3f)open[corner]={u/length,-v/length}; // tile-local
                }
                std::array<std::array<float,2>,8> axes{};
                for(int direction=0;direction<8;++direction){
                    float u=0,v=0;
                    if(!(mask>>direction&1u))continue;
                    if(bridge_mask>>direction&1u){
                        // A bridge lies square to its river edge, on the tile
                        // diagonal toward the neighbor; both halves follow it.
                        float axis_u=static_cast<float>(route_offsets[direction][0]+route_offsets[direction][1])*.5f;
                        float axis_v=-static_cast<float>(route_offsets[direction][0]-route_offsets[direction][1])*.5f;
                        float length=std::hypot(axis_u,axis_v);
                        axes[direction]={axis_u/length,axis_v/length};
                        continue;
                    }
                    if(!outward(set,own,direction,u,v))continue;
                    int neighbor_x=tile.tile_x+route_offsets[direction][0],neighbor_y=tile.tile_y+route_offsets[direction][1];
                    if(auto found=lookup(neighbor_x,neighbor_y)){
                        auto const& neighbor=found->occurrence;unsigned other=0;
                        for(int k=0;k<8;++k){
                            auto next=lookup(neighbor_x+route_offsets[k][0],neighbor_y+route_offsets[k][1]);
                            if(next && (railroad?next->occurrence.railroad_mask!=0:next->occurrence.road_mask &&
                               !(neighbor.railroad_mask && next->occurrence.railroad_mask)))other|=1u<<k;
                        }
                        float nu=0,nv=0;
                        if(outward(set,set.index(other,neighbor.tile_x,neighbor.tile_y),(direction+4)&7,nu,nv)){u-=nu;v-=nv;}
                    }
                    float length=std::hypot(u,v);
                    if(length>1e-5f)axes[direction]={u/length,v/length};
                }
                if(draw){
                    for(unsigned index=set.offsets[own];index<set.offsets[own+1];++index){
                        auto const& line=set.lines[index];
                        unsigned bridges=(line.start>=0 && (bridge_mask>>line.start&1u)?1u:0u)|
                            (line.end>=0 && (bridge_mask>>line.end&1u)?2u:0u);
                        unsigned crossings=(line.start>=0 && (fords>>line.start&1u)?1u:0u)|
                            (line.end>=0 && (fords>>line.end&1u)?2u:0u);
                        PatternRoute route{index,railroad?4u:0u,bridges,{},{},{},crossings};
                        if(line.start>=0){route.joins[0]=axes[line.start][0];route.joins[1]=axes[line.start][1];
                            route.open[0]=open[line.start][0];route.open[1]=open[line.start][1];}
                        if(line.end>=0){route.joins[2]=axes[line.end][0];route.joins[3]=axes[line.end][1];
                            route.open[2]=open[line.end][0];route.open[3]=open[line.end][1];}
                        plan.patterns.push_back(std::move(route));
                    }
                }
                unsigned style=static_cast<unsigned>(std::clamp(tile.route_style,0,3));
                std::string group_name=std::string("bridge_")+(railroad?"railroad":style>=3u?"modern":style>=2u?"industrial":"medieval")+"_normal";
                c3x_renderer::FeatureGroup const* bridge_group=c3x_renderer::find_feature_group(bridge_bundle,group_name.c_str());
                for(int direction=0;direction<4;direction+=2){
                    if(!(bridge_mask>>direction&1u) || !bridge_group || bridge_group->placements.empty())continue;
                    auto const& placement=bridge_group->placements.front();
                    constexpr float join_u[4]={.5f,1.f,1.f,1.f},join_v[4]={0.f,0.f,.5f,1.f};
                    append_feature_instance(bridge_bundle,placement,join_u[direction],join_v[direction],
                        std::atan2(axes[direction][1],axes[direction][0]),placement.scale,13.0f,0.0f,true,feature_vertices);
                }
            };
            // A road with no road link draws Civ III's mark unless a city or a
            // railroad occupies the tile; a lone railroad unless a city does.
            network(pattern_mask,pattern_bridges,false,routes_enabled && tile.road_mask &&
                (pattern_mask || (tile.city_id<0 && !tile.railroad_mask)));
            network(rail_mask,rail_bridges,true,routes_enabled && tile.railroad_mask &&
                (rail_mask || tile.city_id<0));
        }
        if(mountain_ring && connected && routes_enabled){
            unsigned style=static_cast<unsigned>(std::clamp(tile.route_style,0,3));
            // Join the occupied spokes by the shorter set of skirt arcs.
            // Omit the largest empty gap instead of drawing a redundant loop
            // around every mountain tile.
            unsigned gap_start=0,gap_length=0;
            for(unsigned direction=0;direction<8;++direction)if(mountain_spokes&(1u<<direction)){
                unsigned length=1;
                while(length<8 && !(mountain_spokes&(1u<<((direction+length)&7u))))++length;
                if(length>gap_length){gap_start=direction;gap_length=length;}
            }
            for(unsigned direction=0;direction<8;++direction){
                if(((direction+8u-gap_start)&7u)<gap_length)continue;
                unsigned next=(direction+1u)&7u;
                append_route_segment(ring_u[direction],ring_v[direction],
                    ring_u[next],ring_v[next],style,false,false,false,true);
            }
        }
        if (!connected && routes_enabled && !patterns) {
            bool railroad = tile.railroad_mask != 0;
            unsigned style = railroad ? 4u :
                static_cast<unsigned>(std::clamp(tile.route_style, 0, 3));
            // A single built road still needs a readable mark on the tile.
            if(mountain_ring)
                append_route_segment(ring_u[3],ring_v[3],
                    ring_u[4],ring_v[4],style,false,false,false,true);
            else append_route_segment(center_u-0.14f, center_v,
                center_u+0.14f, center_v, style, railroad, false, false);
            plan.routes.back().isolated=true;
        }
    }
}
template<class River>
void promote_river_crossings(c3x_renderer_tile_v1 const& tile,
        River river_distance,Plan& plan,Assets const* assets=nullptr){
    float tile_world_u=float(tile.tile_x+tile.tile_y)*.5f;
    float tile_world_v=float(tile.tile_x-tile.tile_y)*.5f;
    auto patterns_of=[&](PatternRoute const& route){return assets?assets->patterns_for(route.style):nullptr;};
    // Away from a bridge, a path inside a river channel moves onto the bank:
    // straight toward its own tile's center, since rivers follow tile edges.
    // Only "in water or not" is sampled, which quantized distances answer
    // reliably. A join touching a river bend moves along its shared open-bank
    // direction instead, so both tiles still meet on one point. The move
    // tapers into nearby dry points.
    // The river surface ends 7.4 source pixels from its centerline; keep the
    // whole stroke (a few pixels to each side) on the bank.
    constexpr float bank=11.f;
    // A rendered river bows up to about a quarter tile off its tile edge,
    // around hills and mountains. A route bridge stands on the middle of the
    // water along its axis (both tiles find the same point), so it spans bank
    // to bank instead of standing on a hillside and ending over the river.
    auto crossing=[&](float u,float v,float axis_u,float axis_v){
        auto distance=[&](float t){return river_distance(tile_world_u+u+axis_u*t,tile_world_v+1.0f-(v+axis_v*t));};
        float best=1e9f,center=0.f;
        for(int step=-40;step<=40;++step){
            float t=float(step)*.01f,d=distance(t);
            if(d<best-1e-3f || (d<best+1e-3f && std::abs(t)<std::abs(center))){best=d;center=t;}
        }
        if(best>=7.4f)return 0.f; // no water on this axis
        float low=center,high=center;
        while(low>-.4f && distance(low-.01f)<7.4f)low-=.01f;
        while(high<.4f && distance(high+.01f)<7.4f)high+=.01f;
        return std::clamp((low+high)*.5f,-.3f,.3f);
    };
    if(assets && assets->road_patterns){
        for(auto& instance:plan.instances){
            if(instance.family!=bridge_family || instance.asset>=(*assets)[bridge_family].assets.size())continue;
            auto const& id=(*assets)[bridge_family].assets[instance.asset].id;
            if(id.rfind("route/bridge/",0)!=0)continue;
            float axis_u=std::cos(instance.rotation),axis_v=std::sin(instance.rotation);
            float shift=crossing(instance.u,instance.v,axis_u,axis_v);
            instance.u+=axis_u*shift;instance.v+=axis_v*shift;
        }
        for(auto& route:plan.patterns)if(route.bridges && patterns_of(route) && route.line<patterns_of(route)->lines.size()){
            auto const* patterns=patterns_of(route);
            auto const& line=patterns->lines[route.line];
            auto const* points=patterns->points.data()+line.first;
            for(unsigned side=0;side<2;++side)if(route.bridges&(1u<<side)){
                auto const& join=side?points[line.count-1]:points[0];
                route.crossing[side]=crossing(join[0],join[1],route.joins[side*2],route.joins[side*2+1]);
            }
        }
    }
    for(auto& route:plan.patterns){
        auto const* patterns=patterns_of(route);
        if(!patterns || route.line>=patterns->lines.size())continue;
        auto const& line=patterns->lines[route.line];
        auto const* points=patterns->points.data()+line.first;
        unsigned count=line.count;
        auto dry=[&](float u,float v){return river_distance(tile_world_u+u,tile_world_v+1.0f-v)>=bank;};
        std::vector<std::array<float,2>> shift(count,{0.f,0.f});
        // Near a bridge the path is its deck; near a ford it fades instead.
        auto near_end=[&](unsigned index,unsigned ends){
            return ((ends&1u) && std::hypot(points[index][0]-points[0][0],points[index][1]-points[0][1])<.30f) ||
                ((ends&2u) && std::hypot(points[index][0]-points[count-1][0],points[index][1]-points[count-1][1])<.30f);
        };
        bool any=false;
        for(unsigned index=0;index<count;++index){
            float u=points[index][0],v=points[index][1];
            if(near_end(index,route.bridges|route.fords) || dry(u,v))continue;
            int side=index==0 && line.start>=0?0:index+1==count && line.end>=0?1:-1;
            float du=.5f-u,dv=.5f-v;
            if(side>=0){du=route.open[side*2];dv=route.open[side*2+1];}
            float length=std::hypot(du,dv);
            if(length<1e-5f)continue; // a shared join off any bend stays put
            du/=length;dv/=length;
            float reach=side>=0?.3f:std::min(length,.3f);
            for(float t=.02f;t<=reach+1e-4f;t+=.02f)if(dry(u+du*t,v+dv*t)){
                shift[index]={du*t,dv*t};any=true;break;
            }
        }
        if(any){
        route.points.assign(points,points+count);
        for(unsigned index=0;index<count;++index){
            auto best=shift[index];
            bool join=(index==0 && line.start>=0) || (index+1==count && line.end>=0);
            // A dry point follows its nearest moved point; joins keep their own.
            if(!join && best[0]==0.f && best[1]==0.f)for(unsigned reach=1;reach<5;++reach){
                float weight=1.f-float(reach)/5.f;
                unsigned found=count;
                if(index>=reach && (shift[index-reach][0]!=0.f || shift[index-reach][1]!=0.f))found=index-reach;
                else if(index+reach<count && (shift[index+reach][0]!=0.f || shift[index+reach][1]!=0.f))found=index+reach;
                if(found<count){best={shift[found][0]*weight,shift[found][1]*weight};break;}
            }
            route.points[index][0]+=best[0];route.points[index][1]+=best[1];
        }
        }
        // Whatever still lies over water away from a bridge fades into the bank.
        std::vector<std::uint8_t> wet(count,0u);
        bool fading=false;
        for(unsigned index=0;index<count;++index){
            auto const& point=route.points.empty()?points[index]:route.points[index];
            wet[index]=!near_end(index,route.bridges) && river_distance(tile_world_u+point[0],
                tile_world_v+1.0f-point[1])<9.0f;
            fading=fading || wet[index];
        }
        if(fading)route.wet=std::move(wet);
    }
    for(auto& route:plan.routes){
        if(route.railroad || route.bridge)continue;
        float closest=1000.0f,crossing_t=0.0f;
        for(unsigned sample=0;sample<=24;++sample){
            float t=float(sample)/24.0f;
            float local_u=route.u0+(route.u1-route.u0)*t;
            float local_v=route.v0+(route.v1-route.v0)*t;
            float distance=river_distance(tile_world_u+local_u,
                tile_world_v+1.0f-local_v);
            if(distance<closest){closest=distance;crossing_t=t;}
        }
        // A shared river boundary can place the channel center just beyond
        // one half's endpoint. Raise both half-decks when their edge reaches
        // the near bank, while keeping the interior test on the channel core.
        bool shared_bank=crossing_t>.75f && closest<16.0f;
        if((closest>=5.0f && !shared_bank) || crossing_t<.125f)continue;
        route.bridge=true;route.bridge_t=crossing_t;
        route.bridge_structural=false;
        // The authored arch only fits its documented side-edge orientation.
        // A sampled interior crossing gets a narrow raised deck and fascia;
        // adding the arch here made overlapping structures in dense networks.
    }
}
inline bool select_improvements(c3x_renderer_tile_v1 const& tile,Assets const& assets,int ground,unsigned site_flags,
        bool mine_assets_ready,bool farm_assets_ready,Plan& plan){
    auto const& site_bundle=assets[site_family];
    auto const& mine_bundle=assets[mine_family];auto const& farm_bundle=assets[farm_family];
    constexpr Layer site_vertices=site_layer,mine_vertices=mine_layer,
        farm_vertices=farm_layer;
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
        unsigned seed=c3x_renderer::stable_hash(tile.variant_seed ^
            (tile.irrigation_mask&15u)*0x9e3779b9u ^ unsigned(ground)*0x85ebca6bu);
        unsigned color_mask=(seed>>8)&15u;
        if(color_mask==0u)color_mask=1u<<((seed>>12)&3u);
        if(color_mask==15u)color_mask^=1u<<((seed>>12)&3u);
        float field_rotation=float(seed&3u)*1.57079632679f+
            (c3x_renderer::stable_random(seed+29u)-.5f)*.08f;
        constexpr unsigned palettes[5][2]={{1,2},{1,0},{0,1},{2,1},{0,2}};
        unsigned terrain=unsigned(std::clamp(ground,0,4));
        // A farm kit draws the same green patchwork on every terrain, instead
        // of terrain palettes: its field pieces, at any angle, at their
        // authored size (in tiles) or up to a fifth larger, and shifted so a
        // different part covers each tile. The tile, its routes, resource and
        // water then clip them (farm_relief), and a field cut to a sliver goes.
        plan.farm_kit=c3x_renderer::find_feature_group(farm_bundle,"farm_kit")!=nullptr;
        // Kit pieces add .0035 to their material fraction (emissive codes
        // unchanged): the shaders then give them the natural terrain's
        // height-depth basis, so raised low relief and hills cannot hide them.
        float kit_code=plan.farm_kit?.0035f:0.f;
        if(plan.farm_kit){
            // A resource with its own kit ("farm_kit:<name>", e.g. ripe wheat)
            // swaps the patchwork; a "farm_kit:<name>:crop" kit plants the
            // resource itself, so the farm keeps no yard around it.
            c3x_renderer::FeatureGroup const* fields=group;
            if(tile.resource_id>=0 && tile.city_id<0){
                std::string name="farm_kit:";
                for(char letter:tile.resource_name){if(!letter)break;name+=letter>='A' && letter<='Z'?char(letter+32):letter;}
                if(auto const* own=c3x_renderer::find_feature_group(farm_bundle,name.c_str()))fields=own;
                else if(auto const* crop=c3x_renderer::find_feature_group(farm_bundle,(name+":crop").c_str())){
                    fields=crop;plan.farm_clearing.yard=false;}
            }
            // The crop pieces of the group (the fields of one patchwork) share
            // one placement; pieces that miss the tile are skipped.
            std::vector<unsigned> pieces;
            float low_x=1e6f,high_x=-1e6f,low_y=1e6f,high_y=-1e6f;
            for(auto const& placement:fields->placements){
                if(placement.asset_index>=farm_bundle.assets.size())continue;
                auto const& asset=farm_bundle.assets[placement.asset_index];
                if(asset.id.find(":crop:")==std::string::npos)continue;
                pieces.push_back(placement.asset_index);
                for(auto const& vertex:asset.vertices){
                    low_x=std::min(low_x,vertex.position[0]);high_x=std::max(high_x,vertex.position[0]);
                    low_y=std::min(low_y,vertex.position[1]);high_y=std::max(high_y,vertex.position[1]);
                }
            }
            float side=std::min(high_x-low_x,high_y-low_y);
            if(!pieces.empty() && side>0){
                float size=std::max(1.5f,side*(1.f+.2f*c3x_renderer::stable_random(seed+11u)));
                float rotation=6.28318530718f*c3x_renderer::stable_random(seed+13u);
                // Any shift keeps the whole tile (half-diagonal .71) covered.
                float angle=6.28318530718f*c3x_renderer::stable_random(seed+17u);
                float shift=(size*.5f-.73f)*c3x_renderer::stable_random(seed+19u);
                float scale=size/side,cosine=std::cos(rotation),sine=std::sin(rotation);
                float center_x=(low_x+high_x)*.5f*scale,center_y=(low_y+high_y)*.5f*scale;
                float u=.5f+std::cos(angle)*shift-(center_x*cosine-center_y*sine);
                float v=.5f+std::sin(angle)*shift-(center_x*sine+center_y*cosine);
                for(unsigned piece:pieces){
                    float piece_low_u=1e6f,piece_high_u=-1e6f,piece_low_v=1e6f,piece_high_v=-1e6f;
                    for(auto const& vertex:farm_bundle.assets[piece].vertices){
                        float x=u+(vertex.position[0]*cosine-vertex.position[1]*sine)*scale;
                        float y=v+(vertex.position[0]*sine+vertex.position[1]*cosine)*scale;
                        piece_low_u=std::min(piece_low_u,x);piece_high_u=std::max(piece_high_u,x);
                        piece_low_v=std::min(piece_low_v,y);piece_high_v=std::max(piece_high_v,y);
                    }
                    if(piece_high_u<0 || piece_low_u>1 || piece_high_v<0 || piece_low_v>1)continue;
                    c3x_renderer::FeaturePlacement placement{};placement.asset_index=piece;
                    append_feature_instance(farm_bundle,placement,u,v,rotation,scale,21.0f,
                        .01f+kit_code,false,farm_vertices);
                }
            }
        }
        for (c3x_renderer::FeaturePlacement const & placement : group->placements) {
            if (placement.asset_index >= farm_bundle.assets.size())
                return false;
            c3x_renderer::FeatureAsset const & asset =
                farm_bundle.assets[placement.asset_index];
            bool building_part = asset.id.find(":building:") != std::string::npos;
            bool crop_part = asset.id.find(":crop:") != std::string::npos;
            bool tree_part = asset.id.find(":tree:") != std::string::npos;
            unsigned emissive_code = 0u;
            std::size_t marker = asset.id.rfind(":e");
            if (marker != std::string::npos)
                emissive_code = static_cast<unsigned>(std::strtoul(
                    asset.id.c_str() + marker + 2u, nullptr, 10));
            if (crop_part) {
                if(plan.farm_kit)continue;
                for(unsigned patch=0;patch<4u;++patch){
                    unsigned palette=palettes[terrain][(color_mask>>patch)&1u];
                    if(asset.texture_index!=palette)continue;
                    unsigned patch_seed=c3x_renderer::stable_hash(seed ^ ((patch+1u)*0x9e3779b9u));
                    float rotation=field_rotation+
                        (c3x_renderer::stable_random(patch_seed+29u)-.5f)*.03f;
                    float cosine=std::cos(rotation),sine=std::sin(rotation);
                    float low_x=1e6f,high_x=-1e6f,low_y=1e6f,high_y=-1e6f;
                    for(auto const& vertex:asset.vertices){
                        float x=vertex.position[0]*cosine-vertex.position[1]*sine;
                        float y=vertex.position[0]*sine+vertex.position[1]*cosine;
                        low_x=std::min(low_x,x);high_x=std::max(high_x,x);
                        low_y=std::min(low_y,y);high_y=std::max(high_y,y);
                    }
                    float footprint=.40f+.025f*c3x_renderer::stable_random(patch_seed+13u);
                    float scale=footprint/std::max(high_x-low_x,high_y-low_y);
                    float desired_u=(patch&1u)?.728f:.272f;
                    float desired_v=(patch&2u)?.728f:.272f;
                    float jitter_u=(c3x_renderer::stable_random(patch_seed+37u)-.5f)*.012f;
                    float jitter_v=(c3x_renderer::stable_random(patch_seed+53u)-.5f)*.012f;
                    float u=desired_u-(low_x+high_x)*.5f*scale+jitter_u;
                    float v=desired_v-(low_y+high_y)*.5f*scale+jitter_v;
                    append_feature_instance(farm_bundle,placement,u,v,rotation,
                        scale,21.0f,.01f*float(emissive_code+1u),false,farm_vertices);
                }
            } else if (tree_part) {
                unsigned count=4u+((seed>>5)%3u);
                for(unsigned slot=0;slot<count;++slot){
                    unsigned tree_seed=c3x_renderer::stable_hash(seed ^ ((slot+5u)*0x85ebca6bu));
                    float u=slot<4u?((slot&1u)?.81f:.19f):.5f;
                    float v=slot<4u?((slot&2u)?.81f:.19f):(slot==4u?.19f:.81f);
                    u+=(c3x_renderer::stable_random(tree_seed+19u)-.5f)*.12f;
                    v+=(c3x_renderer::stable_random(tree_seed+31u)-.5f)*.12f;
                    float scale=1.35f+.45f*c3x_renderer::stable_random(tree_seed+47u);
                    append_feature_instance(farm_bundle,placement,u,v,
                        float(tree_seed&3u)*1.57079632679f,scale,21.0f,
                        .01f*float(emissive_code+1u)+kit_code,true,farm_vertices);
                }
            } else if (building_part) {
                append_feature_instance(farm_bundle,placement,
                    .37f+float(seed&1u)*.20f,.38f+float((seed>>1)&1u)*.19f,
                    float((seed>>2)&3u)*1.57079632679f,1.45f,21.0f,
                    .01f*float(emissive_code+1u)+kit_code,true,farm_vertices);
            }
        }
    }
    // CityCompositionRuntime owns complete authored city layouts and walls.
    // Never substitute the retired procedural city or wall meshes.
    return true;
}
// The ground a farm kit keeps open on its tile: its drawn routes with a
// verge, and its resource's parts (tile-local boxes from the caller). Each
// tile draws its own routes up to its edges, so its own route plan suffices.
inline void clear_farm(Plan& plan,Plan const& routes,Assets const& assets,
        std::vector<std::array<float,4>> const& resource){
    if(!plan.farm_kit)return;
    constexpr float road=.095f,rail=.12f,yard=.05f;
    auto& clearing=plan.farm_clearing;
    for(auto const& route:routes.patterns){
        auto const* set=assets.patterns_for(route.style);
        if(!set || route.line>=set->lines.size())continue;
        auto const& line=set->lines[route.line];
        auto point=[&](unsigned index){
            return route.points.size()==line.count?route.points[index]:set->points[line.first+index];};
        for(unsigned index=1;index<line.count;++index){
            auto a=point(index-1),b=point(index);
            clearing.paths.push_back({a[0],a[1],b[0],b[1],route.style>=4u?rail:road});
        }
    }
    for(auto const& route:routes.routes)
        clearing.paths.push_back({route.u0,route.v0,route.u1,route.v1,route.railroad?rail:road});
    if(clearing.yard)for(auto const& box:resource)clearing.boxes.push_back({box[0],box[1],box[2],box[3],yard});
}
// A farm's relief: the clearance channel (dry land) also keeps out of the
// kit's open ground, so fields and props go around routes and resources.
template<class Relief>
auto farm_relief(Plan const& plan,c3x_renderer_tile_v1 const& tile,Relief relief){
    float world_u=float(tile.tile_x+tile.tile_y)*.5f,world_v=float(tile.tile_x-tile.tile_y)*.5f;
    return [&clearing=plan.farm_clearing,kit=plan.farm_kit,relief,world_u,world_v](float x,float y){
        auto sample=relief(x,y);
        float u=x-world_u,v=world_v+1.f-y;
        // A kit's patchwork stops a narrow verge inside its own tile.
        if(kit)sample[2]=std::min(sample[2],std::min(std::min(u,1.f-u),std::min(v,1.f-v))-.02f);
        if(!clearing.paths.empty() || !clearing.boxes.empty())
            sample[2]=std::min(sample[2],clearing.at(u,v));
        return sample;
    };
}
template<class Source>
void settle_farm_fields(Plan& plan,c3x_renderer_tile_v1 const& tile,Assets const& assets,Source source){
    if(plan.farm_kit)return; // the kit's patchwork is clipped, not moved
    auto relief=farm_relief(plan,tile,source);
    constexpr std::array<float,7> scales{{1.f,.94f,.88f,.82f,.75f,.68f,.6f}};
    float world_u=float(tile.tile_x+tile.tile_y)*.5f;
    float world_v=float(tile.tile_x-tile.tile_y)*.5f;
    for(std::size_t index=0;index<plan.instances.size();++index){
        auto& instance=plan.instances[index];
        if(instance.family!=farm_family || instance.asset>=assets[farm_family].assets.size())continue;
        auto const& asset=assets[farm_family].assets[instance.asset];
        if(asset.id.find(":crop:")==std::string::npos || asset.vertices.empty())continue;
        bool right=instance.u>.5f,lower=instance.v>.5f;
        float low_u=right?.505f:.04f,high_u=right?.96f:.495f;
        float low_v=lower?.505f:.04f,high_v=lower?.96f:.495f;
        float cosine=std::cos(instance.rotation),sine=std::sin(instance.rotation);
        bool found=false;
        for(float factor:scales){
            if(found)break;
            for(int radius=0;radius<=2 && !found;++radius)
            for(int du=-radius;du<=radius && !found;++du)
            for(int dv=-radius;dv<=radius;++dv){
                if(std::max(std::abs(du),std::abs(dv))!=radius)continue;
                float u=instance.u+float(du)*.05f,v=instance.v+float(dv)*.05f;
                float min_u=1e6f,max_u=-1e6f,min_v=1e6f,max_v=-1e6f;
                for(auto const& vertex:asset.vertices){
                    float x=(vertex.position[0]*cosine-vertex.position[1]*sine)*instance.scale*factor;
                    float y=(vertex.position[0]*sine+vertex.position[1]*cosine)*instance.scale*factor;
                    min_u=std::min(min_u,u+x);max_u=std::max(max_u,u+x);
                    min_v=std::min(min_v,v+y);max_v=std::max(max_v,v+y);
                }
                if(min_u<low_u || max_u>high_u || min_v<low_v || max_v>high_v)continue;
                bool dry=true;
                for(unsigned row=0;row<5u && dry;++row)for(unsigned column=0;column<5u;++column){
                    float sample_u=min_u+(max_u-min_u)*float(column)*.25f;
                    float sample_v=min_v+(max_v-min_v)*float(row)*.25f;
                    if(relief(world_u+sample_u,world_v+1.f-sample_v)[2]<.025f){dry=false;break;}
                }
                if(dry){instance.u=u;instance.v=v;instance.scale*=factor;found=true;break;}
            }
        }
    }
}
template<class Source>
void settle_farm_props(Plan& plan,c3x_renderer_tile_v1 const& tile,Assets const& assets,Source source){
    auto relief=farm_relief(plan,tile,source);
    constexpr std::array<std::array<float,2>,16> anchors{{
        {{.22f,.22f}},{{.50f,.22f}},{{.78f,.22f}},{{.22f,.50f}},
        {{.78f,.50f}},{{.22f,.78f}},{{.50f,.78f}},{{.78f,.78f}},
        {{.14f,.35f}},{{.14f,.65f}},{{.32f,.35f}},{{.32f,.65f}},
        {{.68f,.35f}},{{.68f,.65f}},{{.86f,.35f}},{{.86f,.65f}}}};
    float world_u=float(tile.tile_x+tile.tile_y)*.5f;
    float world_v=float(tile.tile_x-tile.tile_y)*.5f;
    std::vector<std::array<float,2>> occupied;
    std::vector<unsigned char> keep(plan.instances.size(),1u);
    // Buildings choose their dry position first; trees fill the remaining gaps.
    for(unsigned pass=0;pass<2u;++pass)for(std::size_t index=0;index<plan.instances.size();++index){
        auto& instance=plan.instances[index];
        if(instance.family!=farm_family || instance.asset>=assets[farm_family].assets.size())continue;
        auto const& id=assets[farm_family].assets[instance.asset].id;
        bool building=id.find(":building:")!=std::string::npos;
        bool tree=id.find(":tree:")!=std::string::npos;
        if((pass==0u && !building) || (pass==1u && !tree))continue;
        float clearance=building?.14f:.11f;
        float spacing=building?.17f:.105f;
        auto fits=[&](float u,float v){
            if(relief(world_u+u,world_v+1.f-v)[2]<clearance)return false;
            for(auto const& used:occupied)if(std::hypot(u-used[0],v-used[1])<spacing)return false;
            return true;
        };
        float u=instance.u,v=instance.v;
        bool found=fits(u,v);
        unsigned seed=c3x_renderer::stable_hash(tile.variant_seed ^ unsigned(index+1u)*0x9e3779b9u);
        for(unsigned attempt=0;!found && attempt<anchors.size();++attempt){
            auto const& anchor=anchors[(seed+attempt*5u)%anchors.size()];
            float candidate_u=anchor[0]+(c3x_renderer::stable_random(seed+attempt*17u)-.5f)*.06f;
            float candidate_v=anchor[1]+(c3x_renderer::stable_random(seed+attempt*23u+11u)-.5f)*.06f;
            if(fits(candidate_u,candidate_v)){u=candidate_u;v=candidate_v;found=true;}
        }
        if(found){instance.u=u;instance.v=v;occupied.push_back({u,v});}
        else keep[index]=0u;
    }
    std::size_t index=0;
    plan.instances.erase(std::remove_if(plan.instances.begin(),plan.instances.end(),
        [&](Instance const&){return keep[index++]==0u;}),plan.instances.end());
}
template<class Relief,class Height>
void compile(Plan const& plan,Projection const& input,Assets const& assets,Relief relief,Height height,Surfaces& output,bool indexed=false,
        std::vector<unsigned>* instance_counts=nullptr){
    for(auto const& route:plan.routes)append_route(input,route,relief,height,output.layers[route_layer]);
    for(auto const& route:plan.patterns)if(auto const* patterns=assets.patterns_for(route.style))
        append_pattern_route(input,*patterns,route,relief,height,output.layers[route_layer]);
    auto farm=farm_relief(plan,input.tile,relief);
    for(auto const& instance:plan.instances){
        auto before=indexed?output.indices[instance.layer].size():output.layers[instance.layer].size();
        FeaturePlacement placement{};placement.asset_index=instance.asset;
        if(instance.family==farm_family)append_instance(input,assets[instance.family],placement,instance.u,instance.v,
            instance.rotation,instance.scale,instance.material,instance.owner,instance.shadow,false,
            farm,height,output.layers[instance.layer],output.shadows,indexed?&output.indices[instance.layer]:nullptr,
            0.f,0.f,plan.farm_kit);
        else append_instance(input,assets[instance.family],placement,instance.u,instance.v,instance.rotation,
            instance.scale,instance.material,instance.owner,instance.shadow,instance.family==site_family,
            relief,height,output.layers[instance.layer],output.shadows,indexed?&output.indices[instance.layer]:nullptr);
        if(instance_counts)instance_counts->push_back(unsigned(
            (indexed?output.indices[instance.layer].size():output.layers[instance.layer].size())-before));
    }
}
}}
