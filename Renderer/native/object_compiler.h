#pragma once
// CPU descriptions and output for existing routes, bridges, sites, improvements
// and fallback city components. Asset IDs are local to the immutable pack lease.
// Query callbacks preserve the caller's dependency recorder; this synchronous
// boundary alone does not authorize concurrent access to its scratch.
#include "terrain_scene_runtime.h"
#include "../lab/shared/natural/vertex.h"
#include "../lab/shared/natural/patterns.h"
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
        unsigned seed=c3x_renderer::patterns::feature_hash(unsigned(tile_x)*73856093u^unsigned(tile_y)*19349663u^0x5bd1e995u);
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
    float stretch=1.f; // a farm plot's own x scale, so it fills its strip
    // A farm plot's region (FarmClearing::regions), which clips it; flags:
    // +0x100 joined (cut exactly at tile edges shared with farms), +0x200 clear
    // of routes and yard (water and tile edges only), +0x400 under the plots
    // (the farm ground: a wider, irregular feather into the land around it).
    unsigned region=0;
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
    float bridge_half=0;            // the bridge's half length along its axis (0: default)
    std::vector<float> fade;        // per point: fades out as a mountain rises under the path
    // Up to two bridges' straight approaches through this line (set by
    // compile): the bridged join (u,v), the axis into the tile (u,v), the
    // distance from the join along the network to the line's start and end,
    // and to a point inside it (its distance along the line, or -1: none),
    // and the approach's straight run from the join.
    // [9]: the bridge's deck level (route ground units).
    // [10],[11]: how far from the join the network is carried at the deck's
    // level where its ground lies lower, and over how much more that fades.
    std::array<std::array<float,12>,2> approach{};
    static constexpr float approach_blend=.15f; // past the run, a path eases back onto its pattern within this
    static constexpr float approach_level=.75f,approach_level_fade=.15f; // the river valley's floor, then its wall
    float half_length()const{return bridge_half>0.f?bridge_half:.18f;}
};
// Ground a farm keeps open on its own tile, in tile-local (u,v): its routes
// as capsules around their drawn centerlines and its resource's parts as
// rounded boxes. at() is the signed distance from that ground (negative inside).
// A plot kit also labels the open ground that routes split apart: regions
// holds cells x cells labels (0 closed, 255 a scrap without fields), and
// at() with a region is negative in every other region too.
struct FarmClearing {
    static constexpr int cells=32;
    std::vector<std::array<float,5>> paths; // u0,v0,u1,v1,half width
    std::vector<std::array<float,5>> boxes; // u0,v0,u1,v1,margin
    bool yard=true; // false when the farm plants its resource itself
    std::vector<std::uint8_t> regions;
    std::uint8_t shared=0; // tile edges (u=0, u=1, v=0, v=1) whose next tile is a farm
    bool soft=false;       // a ground kit's farm: an irregular edge toward other land
    // Its ground's strength from its terrain: this tile's, then its u=0, u=1,
    // v=0 and v=1 neighbours' (where they are farms).
    std::array<float,5> strength{{1.f,1.f,1.f,1.f,1.f}};
    // Its ground's tint, the same way: tundra 0, grassland .25, flood plain .4,
    // plains .65, desert 1 (terrain_scene.hlsl farm_kit_ground_tint).
    std::array<float,5> tint{{.25f,.25f,.25f,.25f,.25f}};
    float at(float u,float v,unsigned region=0)const{
        float distance=open(u,v);
        if(region && distance>0 && regions.size()==std::size_t(cells*cells)){
            int i=int(std::floor(u*cells-.5f)),j=int(std::floor(v*cells-.5f));
            bool own=false,other=false;
            for(int k=0;k<4;++k){
                unsigned label=regions[std::clamp(j+(k>>1),0,cells-1)*cells+std::clamp(i+(k&1),0,cells-1)];
                own=own || label==region;other=other || label!=0;
            }
            if(other && !own)distance=-distance;
        }
        return distance;
    }
    float open(float u,float v)const{
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
// farm_route_lines: the tile's route lines when they are in another plan.
// farm_plots: the plot kit that settle_farm_fields lays out, if any.
struct Plan {std::vector<Instance> instances;std::vector<Route> routes;std::vector<PatternRoute> patterns;
    bool farm_kit=false;FarmClearing farm_clearing;unsigned farm_route_lines=0;
    FeatureGroup const* farm_plots=nullptr;};
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
// A baked resource ground decal follows gentle relief but does not climb a steep
// face: no point rises more than half its distance from the decal's centre
// (heights in relief units, 150/.82 per tile), and the face hides the rest
// instead of the decal stretching up it.
inline float decal_ground(float ground,float centre,float distance_tiles){
    return std::min(ground,centre+distance_tiles*(150.f/.82f)*.5f);
}
inline bool steep_decal(float rise,float distance_tiles){
    return rise>distance_tiles*(150.f/.82f)*.5f;
}
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
// The level a road or railroad bridge's deck stands at (both in world u,v
// around its center along its axis; reach: half its length). The river runs
// in a valley, some 15 units below the land beyond about .6 tile from the
// water. A deck on the valley floor sits that far below the paths coming over
// the land, and on the map's fixed oblique view a straight path descending to
// it is drawn bent into the deck's side. The deck stands at the top of the
// lower bank instead (the highest route ground out to .75 on each side, the
// lower of the two sides, at most 20 over its own ends' ground), and its
// paths are carried level to it (see append_pattern_route), so they are
// drawn straight through it. See seat_route_bridge.
template<class Ground>
float route_bridge_level(float world_u,float world_v,float axis_u,float axis_v,float reach,Ground ground){
    float ends=1e9f,banks=1e9f;
    for(float sign:{-1.f,1.f}){
        auto at=[&](float out){return ground(world_u+axis_u*sign*out,world_v+axis_v*sign*out);};
        float end=at(reach),bank=end;
        for(float out:{.45f,.6f,.75f})if(out>reach)bank=std::max(bank,at(out));
        ends=std::min(ends,end);banks=std::min(banks,bank);
    }
    return std::min(banks,ends+20.f);
}
// A road or railroad bridge rests at the top of the lower of its two banks
// (see route_bridge_level). The authored meshes put their deck ends at the
// base (z=0), so that end meets its bank and the other end settles into a
// higher bank instead of floating over a lower one. One seat keeps the shared
// rigid transform.
template<class Relief,class Height>
bool seat_route_bridge(Projection const& input,FeatureAsset const& asset,float scale,float rotation,
        float world_u,float world_v,Relief relief_at_world,Height natural_height_at,float& ground){
    if(!input.pickup_profile || asset.id.rfind("route/bridge/",0)!=0 || asset.vertices.empty())return false;
    float reach=0.f;
    for(auto const& source:asset.vertices)reach=std::max(reach,std::abs(source.position[0])*scale);
    ground=route_bridge_level(world_u,world_v,std::cos(rotation),-std::sin(rotation),reach,
        [&](float u,float v){return std::max(relief_at_world(u,v)[0],natural_height_at(u,v)-2.5f);});
    return true;
}
template<class Relief,class Height>
void append_instance(Projection const& input,FeatureBundle const& bundle,FeaturePlacement const& placement,
        float local_u,float local_v,float rotation,float scale,float material_offset,float owner_code,bool cast_shadow,
        bool site,Relief relief_at_world,Height natural_height_at,std::vector<Vertex>& target,std::vector<Vertex>& shadows,std::vector<unsigned>* topology=nullptr,float lift=0.f,float ground_fit=0.f,
        bool drape=false,float stretch=1.f,unsigned shared_edges=0,float feather=.06f,bool tint=false){
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
        // A resource ground decal centred up a steep face (a mountainside) would
        // stretch down to the ground falling away below it; it is left out. One at
        // the foot stays, and decal_ground keeps its edge from climbing the face.
        if(ground_decal){
            float low=ground_sample[0];
            for(int side=0;side<4;++side)low=std::min(low,natural_height_at(
                tile_world_u+local_u+(side==0?scale:side==1?-scale:0.f),
                tile_world_v+1.f-local_v+(side==2?scale:side==3?-scale:0.f))-2.5f);
            if(steep_decal(ground_sample[0]-low,scale))return;
        }
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
        float local_x = (source.position[0] * stretch * cosine - source.position[1] * sine) * scale;
        float local_y = (source.position[0] * stretch * sine + source.position[1] * cosine) * scale;
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
        if(farm_decal && ground_decal && pickup_profile && site)vertex_ground[0]=decal_ground(natural_height_at(
            tile_world_u+local_u+local_x,tile_world_v+1.f-local_v-local_y)-2.5f,ground_sample[0],
            std::sqrt(local_x*local_x+local_y*local_y));
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
            // A ground kit's farm decals carry their terrain tint t (relief
            // channel 1) as their normal's length, 1.1 + t: compact feature
            // vertices keep the normal (not world_valid), and the shader
            // normalizes it before any lighting.
            if(tint && farm_decal){float length=1.1f+std::clamp(vertex_ground[1],0.f,1.f);
                vertex.normal_x*=length;vertex.normal_y*=length;vertex.normal_z*=length;}
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
        auto blend=[](ShoreVertex const& a,ShoreVertex const& b,float t){
            ShoreVertex result{a.vertex,a.distance+(b.distance-a.distance)*t};
            auto* output=reinterpret_cast<float*>(&result.vertex);
            auto const* from=reinterpret_cast<float const*>(&a.vertex);
            auto const* to=reinterpret_cast<float const*>(&b.vertex);
            for(unsigned i=0;i<sizeof(Vertex)/sizeof(float);++i)
                output[i]=from[i]+(to[i]-from[i])*t;
            return result;
        };
        auto intersect=[&](ShoreVertex const& a,ShoreVertex const& b){
            ShoreVertex result=blend(a,b,a.distance/(a.distance-b.distance));result.distance=0;return result;};
        std::size_t first_vertex=target.size(),first_index=topology?topology->size():0;
        // A kit field's part inside its tile (the .02 verge), measured
        // exactly: the tile edge alone decides whether it is a sliver.
        float full_area=0,edge_area=0,kept_area=0,low_u=1e6f,high_u=-1e6f,low_v=1e6f,high_v=-1e6f;
        auto inside_tile=[&](std::array<std::array<float,2>,3> const& corners){
            std::vector<std::array<float,2>> polygon(corners.begin(),corners.end());
            for(unsigned side=0;side<4 && polygon.size()>=3;++side){
                // An edge a joined plot shares with the next farm cuts no sliver.
                if((shared_edges>>(side==0u?0u:side==1u?2u:side==2u?1u:3u))&1u)continue;
                unsigned axis=side&1u;float limit=side<2u?.02f:.98f,sign=side<2u?1.f:-1.f;
                std::vector<std::array<float,2>> next;
                for(std::size_t k=0;k<polygon.size();++k){
                    auto const& a=polygon[k];auto const& b=polygon[(k+1)%polygon.size()];
                    float da=sign*(a[axis]-limit),db=sign*(b[axis]-limit);
                    if(da>=0)next.push_back(a);
                    if((da>=0)!=(db>=0)){float t=da/(da-db);next.push_back({a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t});}
                }
                polygon.swap(next);
            }
            float area=0;
            for(std::size_t k=1;k+1<polygon.size();++k){
                auto const& a=polygon[0];auto const& b=polygon[k];auto const& c=polygon[k+1];
                area+=std::abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))*.5f;
            }
            for(auto const& point:polygon){low_u=std::min(low_u,point[0]);high_u=std::max(high_u,point[0]);
                low_v=std::min(low_v,point[1]);high_v=std::max(high_v,point[1]);}
            return area;
        };
        for(std::size_t triangle=0;triangle+2<asset.indices.size();triangle+=3){
            std::array<ShoreVertex,8> polygon{};
            unsigned count=3;
            for(unsigned corner=0;corner<3;++corner){
                auto index=asset.indices[triangle+corner];
                polygon[corner]={transformed[index],farm_shore[index]};
            }
            // A joined plot's shared tile edges cut it exactly, unfeathered:
            // the next farm draws the rest of it.
            for(unsigned edge=0;edge<4 && count>=3;++edge){
                if(!((shared_edges>>edge)&1u))continue;
                auto side=[&](ShoreVertex const& point){
                    float u=point.vertex.world_x-tile_world_u,v=tile_world_v+1.f-point.vertex.world_y;
                    return edge==0u?u:edge==1u?1.f-u:edge==2u?v:1.f-v;};
                std::array<ShoreVertex,8> next{};unsigned kept=0;
                for(unsigned corner=0;corner<count;++corner){
                    auto const& a=polygon[corner];auto const& b=polygon[(corner+1)%count];
                    float da=side(a),db=side(b);
                    if(da>=0)next[kept++]=a;
                    if((da>=0)!=(db>=0))next[kept++]=blend(a,b,da/(da-db));
                }
                polygon=next;count=kept;
            }
            if(drape && pickup_profile){
                std::array<std::array<float,2>,3> corners{};
                for(unsigned corner=0;corner<3;++corner){auto const& world=farm_world[asset.indices[triangle+corner]];
                    corners[corner]={world[0]-tile_world_u,tile_world_v+1.f-world[1]};}
                full_area+=std::abs((corners[1][0]-corners[0][0])*(corners[2][1]-corners[0][1])-
                    (corners[1][1]-corners[0][1])*(corners[2][0]-corners[0][0]))*.5f;
                edge_area+=inside_tile(corners);
            }
            std::array<ShoreVertex,8> clipped{};
            unsigned kept=0;
            for(unsigned corner=0;corner<count;++corner){
                auto const& a=polygon[corner];auto const& b=polygon[(corner+1)%count];
                bool a_land=a.distance>=0,b_land=b.distance>=0;
                if(a_land)clipped[kept++]=a;
                if(a_land!=b_land)clipped[kept++]=intersect(a,b);
            }
            // A kit field carries its distance to the nearest cut (up to
            // `feather`, .06 tile; linear, so it interpolates closely across a
            // triangle) in its material's spare digits (.0131-.0134); the
            // shader feathers the field into the ground over half of it.
            if(drape)for(unsigned corner=0;corner<kept;++corner)
                clipped[corner].vertex.base_terrain=static_cast<float>(asset.texture_index)+material_offset+owner_code-
                    .0004f+.0003f*std::clamp(clipped[corner].distance/feather,0.f,1.f);
            for(unsigned corner=1;corner+1<kept;++corner){
                if(drape){auto const& a=clipped[0].vertex;auto const& b=clipped[corner].vertex;auto const& c=clipped[corner+1].vertex;
                    kept_area+=std::abs((b.world_x-a.world_x)*(c.world_y-a.world_y)-(b.world_y-a.world_y)*(c.world_x-a.world_x))*.5f;}
                for(unsigned index:{0u,corner,corner+1}){
                    if(topology){topology->push_back(unsigned(target.size()));
                        target.push_back(clipped[index].vertex);}
                    else target.push_back(clipped[index].vertex);
                }
            }
        }
        // A kit field the tile edge cuts to a narrow sliver (its kept width,
        // area over the kept extent with its fringe, under .07 tile) or a
        // small scrap is dropped whole; so is a field that routes, its yard
        // and water leave only as crumbs. Fields between routes stay.
        if(drape && pickup_profile && kept_area>0 &&
           ((edge_area<full_area*.5f && (edge_area/std::hypot(high_u-low_u,high_v-low_v)<.07f || edge_area<.006f)) ||
            (kept_area<.004f && !shared_edges))){
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
    // A railroad (style 4) is only slightly wider than a road, so a dense
    // network does not outweigh the roads beside it.
    bool railroad=route.style>=4u;
    float stroke_across=railroad?3.85f:3.5f,stroke_down=railroad?4.95f:4.5f; // pixels at a 128-pixel tile
    // Margin beyond the stroke: a road's worn shoulders, a railroad's dirt
    // bed (Civ VI lays its rail pieces over a dirt road piece).
    float fringe=railroad?1.65f:1.5f;
    constexpr float piece_v=.94814090f,piece_core=.0156f;// tiled path center and opaque half-height
    // Piece units per tile along the path. The road piece keeps its texel
    // aspect; the rail strip (16 sleepers per unit, its stroke spanning 43.5
    // of 256 texels across) keeps its sleepers square at its stroke.
    float texture_scale=railroad?2.48f:piece_core/.031f;
    if(route.line>=patterns.lines.size())return;
    auto const& line=patterns.lines[route.line];
    auto const& tile=input.tile;
    float tile_world_u=float(tile.tile_x+tile.tile_y)*.5f,tile_world_v=float(tile.tile_x-tile.tile_y)*.5f;
    std::vector<std::array<float,2>> points=route.points.size()==line.count?route.points:
        std::vector<std::array<float,2>>(patterns.points.begin()+line.first,patterns.points.begin()+line.first+line.count);
    std::size_t count=points.size();
    bool shared_start=line.start>=0 && (route.joins[0]!=0.f || route.joins[1]!=0.f);
    bool shared_end=line.end>=0 && (route.joins[2]!=0.f || route.joins[3]!=0.f);
    // A bridged path skips the stretch its bridge occupies, reaching a little
    // under each end of the deck.
    constexpr float bridge_overlap=.04f;
    float half=route.half_length();
    // A path over water fades out over a few points toward the bank.
    std::vector<float> ford(count,0.f);
    if(route.wet.size()==count)for(std::size_t index=0;index<count;++index)
        for(std::size_t other=index>4?index-4:0;other<std::min(count,index+5);++other)if(route.wet[other])
            ford[index]=std::max(ford[index],1.f-float(index>other?index-other:other-index)/5.f);
    // A path also fades out as it climbs a mountain (the same fade the shader
    // applies to a ford).
    if(route.fade.size()==count)for(std::size_t index=0;index<count;++index)
        ford[index]=std::max(ford[index],route.fade[index]);
    // An authored bridge lies square to its river edge. Out of it the line
    // into the join runs straight along the bridge axis, through the deck and
    // a short clear stretch beyond (the straight run), then eases back onto
    // its pattern (route.approach, from compile). Civ III rail patterns often
    // fork just inside the edge: a line closer to the join along the network
    // than the run's end is hidden up to there, and each part beyond starts
    // at the run's end, moving back onto its pattern along its own length.
    // So no path meets the deck from its side. A segment with a hidden end is
    // not drawn.
    // The same visible surface as railroads; pickup relief alone can sit
    // well below the natural ground.
    auto ground=[&](float world_u,float world_v){
        return std::max(relief_at_world(world_u,world_v)[0],height_at_world(world_u,world_v)-2.5f);
    };
    std::vector<std::uint8_t> hidden(count,0u);
    // Per point: how fully it is drawn at a bridge deck's level (see below), and that level.
    std::vector<float> lift(count,0.f),level(count,-1e9f);
    auto smooth=[](float x){x=std::clamp(x,0.f,1.f);return x*x*(3.f-2.f*x);};
    for(auto const& approach:route.approach){
        if(approach[2]==0.f && approach[3]==0.f)continue;
        float run=approach[8];
        bool into_join=approach[4]==0.f || approach[5]==0.f;
        std::vector<float> arc(count,0.f);
        for(std::size_t index=1;index<count;++index)
            arc[index]=arc[index-1]+std::hypot(points[index][0]-points[index-1][0],points[index][1]-points[index-1][1]);
        float total=arc.back();
        auto walk_at=[&](float at){
            float walk=std::min(approach[4]+at,approach[5]+total-at);
            return approach[6]>=0.f?std::min(walk,approach[7]+std::abs(at-approach[6])):walk;
        };
        // Dense points so the path leaves the axis smoothly, with one exactly
        // where a hidden stretch ends.
        std::vector<std::array<float,2>> next;std::vector<float> next_ford,next_at,next_walk,next_lift,next_level;
        std::vector<std::uint8_t> next_hidden,crossing;
        float carried_lift=0.f,carried_level=-1e9f;
        auto add=[&](std::array<float,2> point,float f,float at,float walk,bool hide,bool cross){
            next.push_back(point);next_ford.push_back(f);next_at.push_back(at);next_walk.push_back(walk);
            next_hidden.push_back(hide?1u:0u);crossing.push_back(cross?1u:0u);
            next_lift.push_back(carried_lift);next_level.push_back(carried_level);
        };
        for(std::size_t index=0;index<count;++index){
            unsigned pieces=index?std::max(1u,unsigned(std::ceil((arc[index]-arc[index-1])/.02f))):1u;
            for(unsigned piece=1;piece<=pieces;++piece){
                float t=index?float(piece)/float(pieces):1.f;
                auto const& from=points[index?index-1:0];auto const& to=points[index];
                std::array<float,2> point{from[0]+(to[0]-from[0])*t,from[1]+(to[1]-from[1])*t};
                float at=index?arc[index-1]+(arc[index]-arc[index-1])*t:0.f,walk=walk_at(at);
                float f=index?ford[index-1]+(ford[index]-ford[index-1])*t:ford[0];
                bool was_hidden=index && piece<pieces?hidden[index-1] && hidden[index]:hidden[index]!=0u;
                carried_lift=index?lift[index-1]+(lift[index]-lift[index-1])*t:lift[0];
                carried_level=index?std::max(level[index-1],level[index]):level[0];
                if(!next.empty() && !into_join && (next_walk.back()<run)!=(walk<run)){
                    float cross=(run-next_walk.back())/(walk-next_walk.back());
                    auto const& last=next.back();
                    add({last[0]+(point[0]-last[0])*cross,last[1]+(point[1]-last[1])*cross},next_ford.back()+(f-next_ford.back())*cross,
                        next_at.back()+(at-next_at.back())*cross,run,next_hidden.back() && was_hidden,true);
                }
                add(point,f,at,walk,was_hidden || (!into_join && walk<run),false);
            }
        }
        std::array<float,2> run_end{approach[0]+approach[2]*run,approach[1]+approach[3]*run};
        // How fully each point is carried at the deck's level (see vertex_at),
        // by its distance from the join along the network.
        for(std::size_t index=0;index<next.size();++index){
            float walk=into_join?(approach[4]==0.f?next_at[index]:total-next_at[index]):next_walk[index];
            float weight=smooth((approach[10]+approach[11]-walk)/std::max(approach[11],1e-3f));
            if(weight>next_lift[index]){next_lift[index]=weight;next_level[index]=approach[9];}
        }
        if(into_join){
            bool from_start=approach[4]==0.f;
            // Its far end stays where the pattern's other lines meet it.
            float blend=std::clamp(total-run-.02f,.04f,PatternRoute::approach_blend);
            for(std::size_t index=0;index<next.size();++index){
                float d=from_start?next_at[index]:total-next_at[index],weight=smooth((run+blend-d)/blend);
                next[index][0]+=(approach[0]+approach[2]*d-next[index][0])*weight;
                next[index][1]+=(approach[1]+approach[3]*d-next[index][1])*weight;
            }
            if(total<run){
                if(from_start){next.push_back(run_end);next_ford.push_back(next_ford.back());next_hidden.push_back(0u);
                    next_lift.push_back(1.f);next_level.push_back(approach[9]);}
                else{next.insert(next.begin(),run_end);next_ford.insert(next_ford.begin(),next_ford.front());next_hidden.insert(next_hidden.begin(),0u);
                    next_lift.insert(next_lift.begin(),1.f);next_level.insert(next_level.begin(),approach[9]);}
            }
        }else{
            std::vector<std::array<float,2>> moved=next;
            for(std::size_t index=0;index<next.size();++index){
                // Hidden points lie along the axis, so a part's first tangent
                // follows the run.
                if(next_hidden[index]){moved[index]={approach[0]+approach[2]*next_walk[index],approach[1]+approach[3]*next_walk[index]};
                    continue;}
                if(!crossing[index])continue;
                int step=index+1<next.size() && !next_hidden[index+1]?1:-1;
                float reach=0.f;
                for(std::ptrdiff_t at=std::ptrdiff_t(index);at>=0 && at<std::ptrdiff_t(next.size()) && !next_hidden[std::size_t(at)];at+=step)
                    reach=std::abs(next_at[std::size_t(at)]-next_at[index]);
                // The move fades out before the part's far end, which stays
                // where the pattern's other lines meet it.
                float length=std::min(PatternRoute::approach_blend,reach-.02f);
                std::array<float,2> move{run_end[0]-next[index][0],run_end[1]-next[index][1]};
                for(std::ptrdiff_t at=std::ptrdiff_t(index);at>=0 && at<std::ptrdiff_t(next.size()) && !next_hidden[std::size_t(at)];at+=step){
                    float t=std::abs(next_at[std::size_t(at)]-next_at[index]);
                    float weight=length>1e-4f?1.f-smooth(t/length):at==std::ptrdiff_t(index)?1.f:0.f;
                    if(weight<=0.f)break;
                    moved[std::size_t(at)][0]+=move[0]*weight;moved[std::size_t(at)][1]+=move[1]*weight;
                }
            }
            next=std::move(moved);
        }
        points=std::move(next);ford=std::move(next_ford);hidden=std::move(next_hidden);count=points.size();
        lift=std::move(next_lift);level=std::move(next_level);
    }
    // Both tiles at a shared join ease their path onto its shared axis, so
    // the two halves pass through the join tangent to each other instead of
    // meeting at a corner.
    for(unsigned side=0;side<2;++side){
        if(!(side?shared_end:shared_start) || (route.bridges&(1u<<side)))continue;
        auto join_point=side?points.back():points.front();
        float axis_u=route.joins[side*2],axis_v=route.joins[side*2+1];
        float full=.04f,ease=.18f;
        for(auto& point:points){
            float du=point[0]-join_point[0],dv=point[1]-join_point[1];
            float weight=std::clamp((full+ease-std::hypot(du,dv))/ease,0.f,1.f);
            weight=weight*weight*(3.f-2.f*weight);
            float along=du*axis_u+dv*axis_v;
            point[0]+=(join_point[0]+axis_u*along-point[0])*weight;
            point[1]+=(join_point[1]+axis_v*along-point[1])*weight;
        }
    }
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
    // The bridge stands on the river's own crossing, so it may sit past the
    // join or inside this tile. Along the path's distance from a bridged join,
    // the deck covers [-c-h, -c+h] (c: crossing outward, h: half length); the
    // path skips that stretch, less the overlap, and keeps any part between
    // the join and a bridge standing farther out.
    if(length<.002f)return;
    std::vector<std::array<float,2>> keep{{0.f,length}};
    auto cut=[&](float a,float b){
        if(b<=a)return;
        std::vector<std::array<float,2>> next;
        for(auto const& r:keep){
            if(b<=r[0] || a>=r[1]){next.push_back(r);continue;}
            if(a>r[0])next.push_back({r[0],a});
            if(b<r[1])next.push_back({b,r[1]});
        }
        keep=std::move(next);
    };
    if(route.bridges&1u)cut(std::max(0.f,-route.crossing[0]-half+bridge_overlap),-route.crossing[0]+half-bridge_overlap);
    if(route.bridges&2u)cut(length-(-route.crossing[1]+half-bridge_overlap),length-std::max(0.f,-route.crossing[1]-half+bridge_overlap));
    keep.erase(std::remove_if(keep.begin(),keep.end(),[](auto const& r){return r[1]-r[0]<.002f;}),keep.end());
    if(keep.empty())return;
    unsigned seed=c3x_renderer::patterns::feature_hash(tile.variant_seed^route.line*0x9e3779b9u^
        unsigned(tile.tile_x)*73856093u^unsigned(tile.tile_y)*19349663u);
    // Anchor the piece at a shared join so both tiles meet on the same texel.
    auto texture=[&](float along){
        if(line.start>=0)return along*texture_scale;
        if(line.end>=0)return (length-along)*texture_scale;
        return along*texture_scale+c3x_renderer::patterns::stable_random(seed);
    };
    float left=input.left,top=input.top,half_w=input.half_w,half_h=input.half_h;
    float relief_projection_scale=input.relief_projection_scale;
    // The owner's ground material for the shader's height blend: 0 grass,
    // 1 plains, 2 desert, 3 hills, 4 mountain, 5 marsh (Civ III square types).
    int real=tile.real_terrain_type,base=tile.terrain_type;
    float ground_kind=real==5?3.f:real==6 || real==10?4.f:real==9?5.f:base==1?1.f:base==0 || base==4?2.f:0.f;
    struct Station {float u,v,normal_u,normal_v,core,ford,along,coordinate,lift,level;};
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
            distance[index]/length,texture(distance[index]),lift[index],level[index]};
    };
    auto vertex_at=[&](Station const& s,float across){
        float route_u=s.u+s.normal_u*s.core*fringe*across;
        float route_v=s.v+s.normal_v*s.core*fringe*across;
        float world_u=tile_world_u+route_u,world_v=tile_world_v+1.f-route_v;
        float height=ground(world_u,world_v);
        // Near a bridge the path is carried at its deck's level where its
        // ground lies lower, like an embankment: route strips draw 6.5 units
        // over their height, a deck's rails 2.5 over its base, so the path is
        // drawn straight through the deck.
        if(s.lift>0.f && s.level-4.f>height)height+=(s.level-4.f-height)*s.lift;
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
        s.lift=a.lift+(b.lift-a.lift)*t;s.level=std::max(a.level,b.level);
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
        // The authored bridge deck carries the path over the river.
        if(d1>d0 && !hidden[index-1] && !hidden[index])for(auto const& r:keep){
            float low=r[0],high=r[1];
            if(!(d1>low && d0<high))continue;
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
                // Pattern bridges are smaller than the source meshes' calibrated
                // scale: they span the river onto both banks without dwarfing
                // the narrow Civ III-width routes.
                constexpr float pattern_bridge_scale=.7f;
                unsigned style=static_cast<unsigned>(std::clamp(tile.route_style,0,3));
                std::string group_name=std::string("bridge_")+(railroad?"railroad":style>=3u?"modern":style>=2u?"industrial":"medieval")+"_normal";
                c3x_renderer::FeatureGroup const* bridge_group=c3x_renderer::find_feature_group(bridge_bundle,group_name.c_str());
                float bridge_half=0.f;
                if(bridge_group && !bridge_group->placements.empty()){
                    auto const& placement=bridge_group->placements.front();
                    if(placement.asset_index<bridge_bundle.assets.size())
                        for(auto const& vertex:bridge_bundle.assets[placement.asset_index].vertices)
                            bridge_half=std::max(bridge_half,std::abs(vertex.position[0])*placement.scale*pattern_bridge_scale);
                }
                if(draw){
                    for(unsigned index=set.offsets[own];index<set.offsets[own+1];++index){
                        auto const& line=set.lines[index];
                        unsigned bridges=(line.start>=0 && (bridge_mask>>line.start&1u)?1u:0u)|
                            (line.end>=0 && (bridge_mask>>line.end&1u)?2u:0u);
                        unsigned crossings=(line.start>=0 && (fords>>line.start&1u)?1u:0u)|
                            (line.end>=0 && (fords>>line.end&1u)?2u:0u);
                        PatternRoute route{index,railroad?4u:0u,bridges,{},{},{},crossings};
                        route.bridge_half=bridge_half;
                        if(line.start>=0){route.joins[0]=axes[line.start][0];route.joins[1]=axes[line.start][1];
                            route.open[0]=open[line.start][0];route.open[1]=open[line.start][1];}
                        if(line.end>=0){route.joins[2]=axes[line.end][0];route.joins[3]=axes[line.end][1];
                            route.open[2]=open[line.end][0];route.open[3]=open[line.end][1];}
                        plan.patterns.push_back(std::move(route));
                    }
                }
                for(int direction=0;direction<4;direction+=2){
                    if(!(bridge_mask>>direction&1u) || !bridge_group || bridge_group->placements.empty())continue;
                    auto const& placement=bridge_group->placements.front();
                    constexpr float join_u[4]={.5f,1.f,1.f,1.f},join_v[4]={0.f,0.f,.5f,1.f};
                    append_feature_instance(bridge_bundle,placement,join_u[direction],join_v[direction],
                        std::atan2(axes[direction][1],axes[direction][0]),placement.scale*pattern_bridge_scale,13.0f,0.0f,true,feature_vertices);
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
            // The mine stands on the visible ground (compile seats mines like
            // sites): centred on flat land and a hill's crown, a little smaller
            // on hills, and at a mountain's camera-facing foot, smaller again.
            bool peak = tile.real_terrain_type == 6 || tile.real_terrain_type == 10;
            float anchor = peak ? 0.78f : 0.5f;
            float fit = peak ? 0.75f : tile.real_terrain_type == 5 ? 0.9f : 1.0f;
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
                // +.0035 marks the natural height-depth basis (as farm kit
                // props), so the hill or mountain under the mine does not
                // hide its lower half; the emissive code digit is unchanged.
                append_feature_instance(mine_bundle, placement,
                    anchor, anchor, rotation, placement.scale * fit, 21.0f,
                    0.01f * static_cast<float>(emissive_code + 1u) + 0.0035f,
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
            // Three or more route lines bound the kit's gap-free patchwork
            // ("farm_kit:dense", "farm_kit:<name>:dense"): its fields fill the
            // ground between the routes, which become the field edges.
            bool dense=plan.patterns.size()+plan.routes.size()+plan.farm_route_lines>=3;
            auto kit_group=[&](std::string const& name){return c3x_renderer::find_feature_group(farm_bundle,name.c_str());};
            c3x_renderer::FeatureGroup const* fields=nullptr;
            std::string name;
            if(tile.resource_id>=0 && tile.city_id<0){
                name="farm_kit:";
                for(char letter:tile.resource_name){if(!letter)break;name+=letter>='A' && letter<='Z'?char(letter+32):letter;}
                if(dense)fields=kit_group(name+":dense");
                if(!fields)fields=kit_group(name);
                if(!fields && dense && (fields=kit_group(name+":dense:crop")))plan.farm_clearing.yard=false;
                if(!fields && (fields=kit_group(name+":crop")))plan.farm_clearing.yard=false;
            }
            if(!fields && dense)fields=kit_group("farm_kit:dense");
            if(!fields)fields=group;
            // A plot kit ("farm_kit:plots", "farm_kit:<name>:plots") lays its
            // rectangular fields in strips instead (settle_farm_fields).
            if(!name.empty())plan.farm_plots=kit_group(name+":plots");
            if(!plan.farm_plots)plan.farm_plots=kit_group("farm_kit:plots");
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
            if(!pieces.empty() && side>0 && !plan.farm_plots){
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
        // A kit marked "farm_kit:sparse" plants fewer trees (2-3) and a
        // farmhouse on about three farms in four.
        bool sparse=plan.farm_kit && c3x_renderer::find_feature_group(farm_bundle,"farm_kit:sparse");
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
                unsigned count=sparse?2u+((seed>>5)&1u):4u+((seed>>5)%3u);
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
            } else if (building_part && !(sparse && ((seed>>9)&3u)==0u)) {
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
// A farm's clearance where the land rises out of it: a mountain lifting the
// rendered ground `rise` (source pixels) above the natural ground, or, unless
// the farm is on a hill itself, a neighbouring hill whose footprint support
// reaches the point. Fields, ground and props stop at the slope's foot
// instead of climbing it (1 = no limit).
inline float farm_slope_clearance(float rise,float hill_support,bool hill_farm){
    float clearance=1.f;
    if(rise>1.f)clearance=std::min(clearance,(10.f-rise)/48.f);
    if(!hill_farm && hill_support>0.f)clearance=std::min(clearance,(.15f-hill_support)*1.1f);
    return clearance;
}
inline void clear_farm(Plan& plan,Plan const& routes,Assets const& assets,
        std::vector<std::array<float,4>> const& resource){
    if(!plan.farm_kit)return;
    // A lone road or railroad keeps a wide verge; a dense junction (every
    // neighbour linked, railroad loops) narrows to about its drawn stroke so
    // the fields between the routes survive.
    float dense=std::clamp((float(routes.patterns.size()+routes.routes.size())-2.f)/8.f,0.f,1.f);
    float road=.095f-.05f*dense,rail=.12f-.06f*dense;
    // A kit marked "farm_kit:narrow" keeps narrower verges in its farmland.
    if(c3x_renderer::find_feature_group(assets[farm_family],"farm_kit:narrow")){road=.065f-.025f*dense;rail=.085f-.035f*dense;}
    constexpr float yard=.05f;
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
// kit's open ground, so fields and props go around routes and resources. A
// plot's relief also keeps it inside its own region; a joined plot (region
// +0x100) does not feather at a tile edge a farm shares (append_instance cuts
// it there exactly).
template<class Relief>
auto farm_relief(Plan const& plan,c3x_renderer_tile_v1 const& tile,Relief relief,unsigned region=0){
    float world_u=float(tile.tile_x+tile.tile_y)*.5f,world_v=float(tile.tile_x-tile.tile_y)*.5f;
    return [&clearing=plan.farm_clearing,kit=plan.farm_kit,relief,world_u,world_v,region](float x,float y){
        auto sample=relief(x,y);
        float u=x-world_u,v=world_v+1.f-y;
        // A kit's patchwork stops at its tile's edge (feathering into it).
        if(kit){
            unsigned shared=region&0x100u?clearing.shared:0u;
            // With a ground kit, fields and props stop up to .04 tile short
            // of other land, irregularly (fixed in world space), so the farm's
            // outline frays into its ground's soft edge instead of a diamond.
            float inset=clearing.soft && !(region&0x400u)?
                .04f*(.5f+.25f*std::sin(x*6.1f+y*2.7f+.4f)+.25f*std::sin(y*5.7f-x*3.1f+2.2f)):0.f;
            float edges[4]={u,1.f-u,v,1.f-v};
            for(unsigned k=0;k<4;++k)if(!((shared>>k)&1u))
                sample[2]=std::min(sample[2],edges[k]-((clearing.shared>>k)&1u?0.f:inset));
            // ...and their plots fade into other land and water over a wider
            // band (props, region 0, keep their clearance tests as they are).
            if(clearing.soft && (region&255u))sample[2]*=.4f;
        }
        if(!(region&0x200u) && (!clearing.paths.empty() || !clearing.boxes.empty()))
            sample[2]=std::min(sample[2],clearing.at(u,v,region&255u));
        // The ground under a farm (0x400) eases into the land and water around
        // it over a wide, irregular edge (fixed in world space, so farms
        // agree), up to .1 tile inside its cut. Its strength follows its
        // terrain (met halfway at a farm of another terrain) and varies gently;
        // capping its distance code below the opaque level lets that much of
        // the terrain show through (the shader's alpha is smoothstep(0,.5,code)).
        // A ground kit's fields and ground carry their terrain tint in relief
        // channel 1 (an authored height they never use), met halfway at a
        // farm of another terrain like the ground's strength.
        float strength=clearing.strength[0],tint=clearing.tint[0];
        float sides[4]={u,1.f-u,v,1.f-v};
        for(unsigned k=0;k<4;++k)if((clearing.shared>>k)&1u){
            float meet=.5f*std::max(0.f,1.f-sides[k]/.35f);
            strength+=(clearing.strength[k+1]-clearing.strength[0])*meet;
            tint+=(clearing.tint[k+1]-clearing.tint[0])*meet;
        }
        if(clearing.soft)sample[1]=tint;
        if(region&0x400u){
            sample[2]-=.1f*(.5f+.25f*std::sin(x*4.7f+y*1.9f)+.25f*std::sin(y*5.3f-x*2.3f+1.7f));
            strength*=.9f+.1f*(.5f+.25f*std::sin(x*2.3f+y*1.1f+.8f)+.25f*std::sin(y*2.7f-x*1.6f+2.9f));
            strength=std::clamp(strength,.05f,1.f);
            sample[2]=std::min(sample[2],.16f*(.5f-std::sin(std::asin(1.f-2.f*strength)/3.f)));
        }
        return sample;
    };
}
// A plot kit's layout. The tile's routes, yard and water split its open
// ground into regions (FarmClearing::regions); each plot is clipped to its
// region, so plots follow the routes' contours and never straddle one.
// A region in a small area of the route network (below) takes that area's
// plots, which every farm in the area lays out alike: they run on across
// tile edges. Elsewhere a region lays strips along the route it borders
// most, from that route's verge outwards (else along its shore or river,
// else its map region's axis); strips along a route divide end to end into
// whole plots, and in open ground a world lattice continues across tiles.
// neighbour(dx,dy) is the tile at that raw offset (null when unknown).
template<class Source,class Neighbour>
void lay_out_farm_plots(Plan& plan,c3x_renderer_tile_v1 const& tile,Assets const& assets,Source water,Neighbour neighbour){
    constexpr float pi=3.14159265359f,gap=.012f;
    constexpr int n=FarmClearing::cells;
    auto const& bundle=assets[farm_family];
    // A ground kit's farms ripen a few plots ("farm_kit:plots:ripe": the same
    // kinds, in order).
    struct Plot {unsigned asset;float width,height;unsigned ripe;};
    std::vector<Plot> plots;
    auto const* ripe=c3x_renderer::find_feature_group(bundle,"farm_kit:plots:ripe");
    for(auto const& placement:plan.farm_plots->placements){
        if(placement.asset_index>=bundle.assets.size())continue;
        float low_x=1e6f,high_x=-1e6f,low_y=1e6f,high_y=-1e6f;
        for(auto const& vertex:bundle.assets[placement.asset_index].vertices){
            low_x=std::min(low_x,vertex.position[0]);high_x=std::max(high_x,vertex.position[0]);
            low_y=std::min(low_y,vertex.position[1]);high_y=std::max(high_y,vertex.position[1]);
        }
        std::size_t kind=std::size_t(&placement-plan.farm_plots->placements.data());
        unsigned ripe_asset=ripe && kind<ripe->placements.size()?ripe->placements[kind].asset_index:placement.asset_index;
        if(high_x>low_x && high_y>low_y)plots.push_back({placement.asset_index,high_x-low_x,high_y-low_y,ripe_asset});
    }
    if(plots.empty())return;
    // A ground kit ("farm_kit:ground") lays one grass decal under the farm's
    // plots, joined with the next farms and over its route verges.
    if(auto const* ground=c3x_renderer::find_feature_group(bundle,"farm_kit:ground"))if(!ground->placements.empty()){
        plan.instances.push_back({farm_family,ground->placements.front().asset_index,farm_layer,.5f,.5f,0.f,1.f,
            21.f,.0135f,false,1.f,0x700u});
        plan.farm_clearing.soft=true;
    }
    // A ditch kit ("farm_kit:ditch") adds sparse irrigation ditches between
    // strips and along routes.
    auto const* ditches=c3x_renderer::find_feature_group(bundle,"farm_kit:ditch");
    unsigned ditch=ditches && !ditches->placements.empty()?ditches->placements.front().asset_index:~0u;
    int c=(tile.tile_x+tile.tile_y)/2,r=(tile.tile_x-tile.tile_y)/2;
    float world_u=float(tile.tile_x+tile.tile_y)*.5f,world_v=float(tile.tile_x-tile.tile_y)*.5f;
    auto& clearing=plan.farm_clearing;
    std::array<float,n> centre{};
    for(int i=0;i<n;++i)centre[i]=(float(i)+.5f)/float(n);
    // Regions: 4-connected open cells; a scrap under .012 tile gets no fields.
    // The water clearance is smooth: sampled on a 9x9 lattice (a full ground
    // query each) and interpolated for the cells.
    std::vector<float> shore(n*n);
    std::vector<std::uint8_t> open(n*n);
    auto& labels=clearing.regions;labels.assign(n*n,0);
    std::array<float,81> water_lattice{};
    for(int k=0;k<81;++k)water_lattice[k]=std::min(water(world_u+float(k%9)/8.f,world_v+1.f-float(k/9)/8.f)[2],1.f);
    for(int cell=0;cell<n*n;++cell){
        float u=centre[cell%n],v=centre[cell/n];
        float x=u*8.f,y=v*8.f;int i=std::min(int(x),7),j=std::min(int(y),7);float a=x-float(i),b=y-float(j);
        shore[cell]=(water_lattice[j*9+i]*(1.f-a)+water_lattice[j*9+i+1]*a)*(1.f-b)+
            (water_lattice[j*9+i+9]*(1.f-a)+water_lattice[j*9+i+10]*a)*b;
        open[cell]=shore[cell]>0 && clearing.open(u,v)>0;
    }
    std::vector<std::vector<int>> regions;
    for(int start=0;start<n*n;++start){
        if(!open[start] || labels[start])continue;
        auto label=std::uint8_t(regions.size()<254?regions.size()+1:255);
        std::vector<int> cells{start};labels[start]=label;
        for(std::size_t k=0;k<cells.size();++k){
            int cell=cells[k],i=cell%n;
            for(int next:{i>0?cell-1:-1,i+1<n?cell+1:-1,cell-n,cell+n})
                if(next>=0 && next<n*n && open[next] && !labels[next]){labels[next]=label;cells.push_back(next);}
        }
        if(label==255 || float(cells.size())<.012f*float(n*n)){for(int cell:cells)labels[cell]=255;cells.clear();}
        regions.push_back(std::move(cells));
    }
    // The terrain of the tile at world lattice (i,j) (i=c, j=-r), or -1.
    auto terrain_at=[&](int i,int j){
        if(i==c && j==-r)return tile.terrain_type;
        auto const* found=neighbour((i-c)-(j+r),(i-c)+(j+r));
        return found?found->terrain_type:-1;};
    // A plot at (along, across) in world coordinates (c+u, v-r) turned by
    // theta: the plot whose aspect best fits, with some variety (a turned plot
    // runs its rows across its strip), stretched within a third to fill it.
    // A few ripen: more on plains, fewer on tundra (by the terrain under the
    // plot's centre, so farms sharing it agree).
    auto emit=[&](float ac,float bc,float la,float lb,float theta,unsigned seed,unsigned region){
        float ca=std::cos(theta),sa=std::sin(theta);
        Plot const* best=nullptr;bool turned=false;float best_score=1e9f;
        for(std::size_t kind=0;kind<plots.size();++kind)for(unsigned quarter=0;quarter<2;++quarter){
            float aspect=quarter?plots[kind].height/plots[kind].width:plots[kind].width/plots[kind].height;
            float score=std::abs(std::log(aspect*lb/la))+
                .45f*c3x_renderer::stable_random(seed+unsigned(kind)*2u+quarter);
            if(score<best_score){best_score=score;best=&plots[kind];turned=quarter!=0;}
        }
        float x=turned?lb:la,y=turned?la:lb;
        float stretch=std::clamp(x/y*best->height/best->width,.75f,1.33f);
        float scale=std::min(x/(best->width*stretch),y/best->height);
        int under=ripe?terrain_at(int(std::floor(ac*ca-bc*sa)),int(std::floor(ac*sa+bc*ca))):-1;
        float share=under==1?.12f:under==3?.02f:under==4?.06f:.05f;
        unsigned asset=ripe && c3x_renderer::stable_random(seed^0x6a09e667u)<share?best->ripe:best->asset;
        plan.instances.push_back({farm_family,asset,farm_layer,ac*ca-bc*sa-float(c),ac*sa+bc*ca+float(r),
            theta+(turned?pi*.5f:0.f)+((seed>>9)&1u?pi:0.f),scale,21.f,.0135f,false,stretch,region});
    };
    // A ditch .02 wide centred at (along, across), its unit length stretched.
    auto emit_ditch=[&](float ac,float bc,float length,float theta,unsigned region){
        if(ditch==~0u || length<.08f)return;
        float ca=std::cos(theta),sa=std::sin(theta);
        plan.instances.push_back({farm_family,ditch,farm_layer,ac*ca-bc*sa-float(c),ac*sa+bc*ca+float(r),
            theta,.02f,21.f,.0135f,false,length/.02f,region});
    };
    // Areas of the route network. Lattice point (i,j)=(c,-r) is a tile
    // centre, at world (i+.5,j+.5); the square of centres from (i,j) splits
    // along its diagonals (crossing at a tile corner) into four triangles
    // {i,j,q}, q from its bottom side counterclockwise. Civ III links every
    // two neighbours that both carry a route, along a side or a diagonal, so
    // an area is the triangles no route separates. An area of up to 32
    // triangles is laid out whole by every tile it touches.
    int i0=c,j0=-r;
    std::vector<std::pair<std::array<int,2>,c3x_renderer_tile_v1 const*>> known;
    auto at_tile=[&](int i,int j)->c3x_renderer_tile_v1 const*{
        if(i==i0 && j==j0)return &tile;
        for(auto const& entry:known)if(entry.first[0]==i && entry.first[1]==j)return entry.second;
        int di=i-i0,dj=j-j0;
        known.push_back({{i,j},neighbour(di-dj,di+dj)});
        return known.back().second;
    };
    auto routed=[&](std::array<int,2> const& k){auto const* t=at_tile(k[0],k[1]);return t && (t->road_mask || t->railroad_mask);};
    auto corner=[](std::array<int,3> const& t,int k){k&=3;return std::array<int,2>{t[0]+(k==1 || k==2),t[1]+(k>=2)};};
    // plots: (along, across, length, width, direction, seed, kind): kind 0 a
    // plot, 1 a ditch between strips, 2 a ditch along the route.
    struct Area {std::vector<std::array<int,3>> triangles;bool whole=true;std::vector<std::array<float,7>> plots;};
    std::vector<Area> areas;
    std::vector<std::pair<std::array<int,3>,unsigned>> area_of;
    auto find_area=[&](std::array<int,3> const& t){
        for(auto const& entry:area_of)if(entry.first==t)return int(entry.second);return -1;};
    auto area_at=[&](std::array<int,3> const& start){
        int found=find_area(start);
        if(found>=0)return unsigned(found);
        unsigned index=unsigned(areas.size());areas.emplace_back();
        std::vector<std::array<int,3>> stack{start};area_of.push_back({start,index});
        while(!stack.empty()){
            auto t=stack.back();stack.pop_back();
            if(areas[index].triangles.size()>=32){areas[index].whole=false;break;}
            areas[index].triangles.push_back(t);
            int q=t[2];
            std::array<int,3> outer=q==0?std::array<int,3>{t[0],t[1]-1,2}:q==1?std::array<int,3>{t[0]+1,t[1],3}:
                q==2?std::array<int,3>{t[0],t[1]+1,0}:std::array<int,3>{t[0]-1,t[1],1};
            std::pair<std::array<int,3>,bool> next[3]={
                {outer,routed(corner(t,q)) && routed(corner(t,q+1))},
                {{t[0],t[1],(q+1)&3},routed(corner(t,q+1)) && routed(corner(t,q+3))},
                {{t[0],t[1],(q+3)&3},routed(corner(t,q)) && routed(corner(t,q+2))}};
            for(auto const& [triangle,blocked]:next)if(!blocked && find_area(triangle)<0){
                area_of.push_back({triangle,index});stack.push_back(triangle);}
        }
        std::sort(areas[index].triangles.begin(),areas[index].triangles.end());
        return index;
    };
    // A small area's plots, as (along, across, length, width, direction,
    // seed): strips along its longest straight route (a side 1, a half
    // diagonal .707), from a fixed verge outwards and evenly over its depth
    // on each side, each divided end to end into whole plots.
    auto lay_area=[&](Area& area){
        std::vector<std::pair<std::array<int,2>,int>> lines; // (orientation, offset) -> length
        auto add=[&](std::array<int,2> const& line,int length){
            for(auto& entry:lines)if(entry.first==line){entry.second+=length;return;}
            lines.push_back({line,length});};
        std::vector<std::array<std::array<float,2>,3>> shapes;
        for(auto const& t:area.triangles){
            int q=t[2];
            auto a=corner(t,q),b=corner(t,q+1);
            shapes.push_back({{{float(a[0])+.5f,float(a[1])+.5f},{float(b[0])+.5f,float(b[1])+.5f},{float(t[0]+1),float(t[1]+1)}}});
            auto diagonal=[&](int k){return k&1?std::array<int,2>{3,t[0]+t[1]+1}:std::array<int,2>{2,t[0]-t[1]};};
            if(routed(a) && routed(b))add(q&1?std::array<int,2>{1,a[0]}:std::array<int,2>{0,a[1]},1000);
            if(routed(b) && routed(corner(t,q+3)))add(diagonal(q+1),707);
            if(routed(a) && routed(corner(t,q+2)))add(diagonal(q),707);
        }
        if(lines.empty()){area.whole=false;return;}
        std::sort(lines.begin(),lines.end());
        auto best=lines.front();
        for(auto const& entry:lines)if(entry.second>best.second)best=entry;
        int orientation=best.first[0];float offset=float(best.first[1])+.5f;
        float theta=float(orientation==0?0.:orientation==1?.5*pi:orientation==2?.25*pi:.75*pi);
        float ca=std::cos(theta),sa=std::sin(theta);
        auto along=[&](std::array<float,2> const& p){return p[0]*ca+p[1]*sa;};
        auto across=[&](std::array<float,2> const& p){return p[1]*ca-p[0]*sa;};
        float line=across(orientation==0?std::array<float,2>{0,offset}:orientation==1?std::array<float,2>{offset,0}:
            std::array<float,2>{offset,.5f});
        unsigned key=c3x_renderer::stable_hash(unsigned(area.triangles.front()[0])*73856093u^
            unsigned(area.triangles.front()[1])*19349663u^unsigned(area.triangles.front()[2])*83492791u);
        constexpr float verge=.06f;
        for(int side=-1;side<=1;side+=2){
            float depth=0;
            for(auto const& shape:shapes)for(auto const& p:shape)depth=std::max(depth,float(side)*(across(p)-line)-verge);
            if(depth<.05f)continue;
            int strips=std::max(1,int(std::lround(depth/.25f)));
            for(int k=0;k<strips;++k){
                float s0=line+float(side)*(verge+depth*float(k)/float(strips));
                float s1=line+float(side)*(verge+depth*float(k+1)/float(strips));
                if(s0>s1)std::swap(s0,s1);
                // The area's extent along this strip: each triangle clipped to it.
                float a0=1e6f,a1=-1e6f;
                for(auto const& shape:shapes){
                    std::vector<std::array<float,2>> polygon(shape.begin(),shape.end());
                    for(int limit=0;limit<2 && !polygon.empty();++limit){
                        std::vector<std::array<float,2>> kept;
                        for(std::size_t e=0;e<polygon.size();++e){
                            auto const& p=polygon[e];auto const& q2=polygon[(e+1)%polygon.size()];
                            float dp=limit?s1-across(p):across(p)-s0,dq=limit?s1-across(q2):across(q2)-s0;
                            if(dp>=0)kept.push_back(p);
                            if((dp>=0)!=(dq>=0)){float f=dp/(dp-dq);kept.push_back({p[0]+(q2[0]-p[0])*f,p[1]+(q2[1]-p[1])*f});}
                        }
                        polygon.swap(kept);
                    }
                    for(auto const& p:polygon){a0=std::min(a0,along(p));a1=std::max(a1,along(p));}
                }
                if(a1-a0<.05f)continue;
                unsigned band=c3x_renderer::stable_hash(key+unsigned((side+1)*64+k)*0x27d4eb2du);
                // Ditches: along the route beside about a third of the first
                // strips, and between strips at about a third of their edges.
                if(k==0 && c3x_renderer::stable_random(band^0x51ed27u)<.35f)
                    area.plots.push_back({(a0+a1)*.5f,line+float(side)*.055f,a1-a0-.04f,0.f,theta,0.f,2.f});
                if(k>0 && c3x_renderer::stable_random(band^0x2545f491u)<.3f)
                    area.plots.push_back({(a0+a1)*.5f,line+float(side)*(verge+depth*float(k)/float(strips)),
                        a1-a0-.04f,0.f,theta,0.f,1.f});
                float length=.26f+.24f*c3x_renderer::stable_random(band);
                int count=std::max(1,int(std::lround((a1-a0)/(length+.08f))));
                length=(a1-a0)/float(count);
                auto stop=[&](int m){return a0+float(m)*length+(m>0 && m<count?
                    (c3x_renderer::stable_random(band^unsigned(m)*0x9e3779b9u)-.5f)*.3f*length:0.f);};
                for(int m=0;m<count;++m){
                    float t0=stop(m),t1=stop(m+1);
                    if(t1-t0-2*gap<.05f || s1-s0-2*gap<.04f)continue;
                    area.plots.push_back({(t0+t1)*.5f,(s0+s1)*.5f,t1-t0-2*gap,s1-s0-2*gap,theta,
                        float(c3x_renderer::stable_hash(band^unsigned(m)*0x85ebca6bu)&0xffffffu),0.f});
                }
            }
        }
    };
    // Tile edges (u=0, u=1, v=0, v=1) where the next tile is a farm, which
    // continues its area's plots.
    auto farm=[&](int di,int dj){auto const* t=at_tile(i0+di,j0+dj);
        return t && (t->improvement_flags&C3X_RENDERER_IMPROVEMENT_IRRIGATION) && t->city_id<0 && t->terrain_type<11;};
    clearing.shared=std::uint8_t((farm(-1,0)?1u:0u)|(farm(1,0)?2u:0u)|(farm(0,-1)?4u:0u)|(farm(0,1)?8u:0u));
    // The ground's strength by terrain: mostly opaque (its tint carries the
    // terrain), letting a little of plains and tundra through and, on desert,
    // plenty of sand.
    auto strength=[&](int i,int j){int terrain=terrain_at(i,j);
        return terrain==0?.45f:terrain==1?.92f:terrain==3?.88f:1.f;};
    clearing.strength={{strength(i0,j0),strength(i0-1,j0),strength(i0+1,j0),strength(i0,j0-1),strength(i0,j0+1)}};
    auto tint=[&](int i,int j){int terrain=terrain_at(i,j);
        return terrain==0?1.f:terrain==1?.65f:terrain==3?0.f:terrain==4?.4f:.25f;};
    clearing.tint={{tint(i0,j0),tint(i0-1,j0),tint(i0+1,j0),tint(i0,j0-1),tint(i0,j0+1)}};
    unsigned map=c3x_renderer::stable_hash(unsigned(c>=0?c/3:(c-2)/3)*73856093u^unsigned(r>=0?r/3:(r-2)/3)*19349663u);
    auto turn=[&](float a,float b){float d=std::fmod(std::abs(a-b),pi);return std::min(d,pi-d);};
    for(std::size_t index=0;index<regions.size();++index){
        auto const& cells=regions[index];
        if(cells.empty())continue;
        // The region's area: the one most of its cells lie in.
        std::vector<std::pair<unsigned,unsigned>> votes;
        for(int cell:cells){
            float x=float(i0)+centre[cell%n]-.5f,y=float(j0)+centre[cell/n]-.5f;
            int si=int(std::floor(x)),sj=int(std::floor(y));
            float a=x-float(si),b=y-float(sj);
            int q=b<std::min(a,1.f-a)?0:a>std::max(b,1.f-b)?1:b>std::max(a,1.f-a)?2:3;
            unsigned area=area_at({si,sj,q});
            bool counted=false;
            for(auto& vote:votes)if(vote.first==area){++vote.second;counted=true;}
            if(!counted)votes.push_back({area,1u});
        }
        auto chosen=*std::max_element(votes.begin(),votes.end(),[](auto const& x,auto const& y){
            return x.second<y.second || (x.second==y.second && x.first>y.first);});
        auto& area=areas[chosen.first];
        if(area.whole && area.plots.empty())lay_area(area);
        if(area.whole){
            // Its area's plots that reach into the region, joined (0x100):
            // their cut at a shared tile edge is hard, as the next farm
            // draws the rest.
            // Plots first, then the ditches over them.
            for(int kind=0;kind<2;++kind)for(auto const& p:area.plots){
                if((p[6]>.5f)!=(kind==1))continue;
                float ca=std::cos(p[4]),sa=std::sin(p[4]);
                bool inside=false;
                for(int cell:cells){
                    float x=float(i0)+centre[cell%n],y=float(j0)+centre[cell/n];
                    float a=x*ca+y*sa-p[0],b=y*ca-x*sa-p[1];
                    if(std::abs(a)<p[2]*.5f+.5f/float(n) && std::abs(b)<std::max(p[3],.06f)*.5f+.5f/float(n)){inside=true;break;}
                }
                if(!inside)continue;
                if(kind==0)emit(p[0],p[1],p[2],p[3],p[4],unsigned(p[5]),unsigned(index+1)|0x100u);
                else emit_ditch(p[0],p[1],p[2],p[4],unsigned(index+1)|(p[6]>1.5f?0x300u:0x100u));
            }
            continue;
        }
        // Cells at a route's verge: that route's direction, centerline point
        // and half width.
        struct Contact {float angle,u,v,half;};
        std::vector<Contact> contacts;
        float mean_u=0,mean_v=0,nearest=1.f,grad_u=0,grad_v=0;
        for(int cell:cells){mean_u+=centre[cell%n];mean_v+=centre[cell/n];}
        mean_u/=float(cells.size());mean_v/=float(cells.size());
        for(int cell:cells){
            float u=centre[cell%n],v=centre[cell/n],best=1.5f/float(n);
            nearest=std::min(nearest,shore[cell]);grad_u+=(u-mean_u)*shore[cell];grad_v+=(v-mean_v)*shore[cell];
            Contact contact{};bool found=false;
            for(auto const& p:clearing.paths){
                float du=p[2]-p[0],dv=p[3]-p[1],length=du*du+dv*dv;
                if(length<1e-10f)continue;
                float t=std::clamp(((u-p[0])*du+(v-p[1])*dv)/length,0.f,1.f),pu=p[0]+du*t,pv=p[1]+dv*t;
                float distance=std::hypot(u-pu,v-pv)-p[4];
                if(distance<best){best=distance;contact={std::atan2(dv,du),pu,pv,p[4]};found=true;}
            }
            if(found)contacts.push_back(contact);
        }
        // The route with the most verge (a direction shared within ~11 degrees).
        std::size_t support=0;float theta=0;
        for(auto const& candidate:contacts){
            std::size_t count=0;
            for(auto const& other:contacts)count+=turn(candidate.angle,other.angle)<.2f;
            if(count>support){support=count;theta=candidate.angle;}
        }
        bool road=support>=4;
        if(road){
            float sum_x=0,sum_y=0;
            for(auto const& contact:contacts)if(turn(theta,contact.angle)<.2f){
                sum_x+=std::cos(2.f*contact.angle);sum_y+=std::sin(2.f*contact.angle);}
            theta=.5f*std::atan2(sum_y,sum_x);
        }else if(nearest<.35f && std::hypot(grad_u,grad_v)>1e-4f)theta=std::atan2(grad_v,grad_u)+pi*.5f;
        else theta=float(map&1u)*pi*.5f+(c3x_renderer::stable_random(map+7u)-.5f)*.2f;
        int degrees=int(std::lround(std::fmod(std::fmod(theta,pi)+pi,pi)*180.f/pi))%180;
        theta=float(degrees)*pi/180.f;
        float ca=std::cos(theta),sa=std::sin(theta);
        // World coordinates sharing the tile-local axes are (c+u, v-r).
        auto along=[&](float u,float v){return (float(c)+u)*ca+(v-float(r))*sa;};
        auto across=[&](float u,float v){return (v-float(r))*ca-(float(c)+u)*sa;};
        float low=1e6f,high=-1e6f;
        for(int cell:cells){float b=across(centre[cell%n],centre[cell/n]);low=std::min(low,b);high=std::max(high,b);}
        low-=.5f/float(n);high+=.5f/float(n);
        // Strips: from the route's verge outwards on each side, evenly over
        // the region's depth there; without a route, a world lattice.
        struct Band {float s0,s1;unsigned key;float road_ditch=1e9f,inner_ditch=1e9f;};
        std::vector<Band> bands;
        unsigned lattice=c3x_renderer::stable_hash(unsigned(degrees+1)*0x9e3779b9u);
        if(road){
            float line=0,half=0,count=0;
            for(auto const& contact:contacts)if(turn(theta,contact.angle)<.2f){
                line+=across(contact.u,contact.v);half+=contact.half;count+=1.f;}
            line/=count;half/=count;
            for(int side=-1;side<=1;side+=2){
                float verge=line+float(side)*half,depth=side>0?high-verge:verge-low;
                if(depth<.05f)continue;
                int strips=std::max(1,int(std::lround(depth/.25f)));
                unsigned key=c3x_renderer::stable_hash(lattice^unsigned(std::lround(line*200.f))*0x27d4eb2du^unsigned(side+1)*0x165667b1u);
                for(int k=0;k<strips;++k){
                    float a=verge+float(side)*depth*float(k)/float(strips),b=verge+float(side)*depth*float(k+1)/float(strips);
                    Band band{std::min(a,b),std::max(a,b),c3x_renderer::stable_hash(key+unsigned(k))};
                    if(k==0 && c3x_renderer::stable_random(band.key^0x51ed27u)<.35f)band.road_ditch=line+float(side)*.055f;
                    if(k>0 && c3x_renderer::stable_random(band.key^0x2545f491u)<.3f)band.inner_ditch=a;
                    bands.push_back(band);
                }
            }
        }else{
            float pitch=.2f+.1f*c3x_renderer::stable_random(lattice);
            auto edge=[&](int k){return float(k)*pitch+(c3x_renderer::stable_random(lattice^unsigned(k)*0x85ebca6bu)-.5f)*.06f;};
            for(int k=int(std::floor(low/pitch))-1;k<=int(std::floor(high/pitch))+1;++k){
                float s0=edge(k),s1=edge(k+1);
                if(s1>low && s0<high)bands.push_back({s0,s1,c3x_renderer::stable_hash(lattice+unsigned(k)*0x27d4eb2du)});
            }
        }
        for(auto const& band:bands){
            float s0=band.s0,s1=band.s1,lb=s1-s0-2*gap;
            if(lb<.04f)continue;
            // The region's cells in this strip, by their place along it.
            std::vector<float> places;
            for(int cell:cells){float u=centre[cell%n],v=centre[cell/n],b=across(u,v);
                if(b>s0 && b<s1)places.push_back(along(u,v));}
            if(places.empty())continue;
            auto [first,last]=std::minmax_element(places.begin(),places.end());
            float a0=*first-.5f/float(n),a1=*last+.5f/float(n);
            float length=.26f+.24f*c3x_renderer::stable_random(band.key),shift=length*c3x_renderer::stable_random(band.key+3u);
            int first_plot=int(std::floor((a0-shift)/length))-1,last_plot=int(std::floor((a1-shift)/length))+1;
            if(road){
                last_plot=std::max(1,int(std::lround((a1-a0)/(length+.08f))));
                length=(a1-a0)/float(last_plot);shift=a0;first_plot=0;--last_plot;
            }
            auto stop=[&](int m){
                if(road && (m<=0 || m>last_plot))return float(m)*length+shift;
                return float(m)*length+shift+(c3x_renderer::stable_random(band.key^unsigned(m)*0x9e3779b9u)-.5f)*(road?.3f*length:.08f);};
            for(int m=first_plot;m<=last_plot;++m){
                float t0=stop(m),t1=stop(m+1),la=t1-t0-2*gap;
                if(t1<a0 || t0>a1 || la<.05f)continue;
                // Skip a plot that would keep only crumbs of the region.
                if(std::count_if(places.begin(),places.end(),[&](float a){return a>t0+gap && a<t1-gap;})<4)continue;
                emit((t0+t1)*.5f,(s0+s1)*.5f,la,lb,theta,c3x_renderer::stable_hash(band.key^unsigned(m)*0x85ebca6bu),unsigned(index+1));
            }
            if(band.road_ditch<1e8f)emit_ditch((a0+a1)*.5f,band.road_ditch,a1-a0-.04f,theta,unsigned(index+1)|0x200u);
            if(band.inner_ditch<1e8f)emit_ditch((a0+a1)*.5f,band.inner_ditch,a1-a0-.04f,theta,unsigned(index+1));
        }
    }
}
// neighbour(dx,dy): the tile at a raw offset, or null, for plots farms share.
template<class Source,class Neighbour>
void settle_farm_fields(Plan& plan,c3x_renderer_tile_v1 const& tile,Assets const& assets,Source source,Neighbour neighbour){
    if(plan.farm_kit){ // the kit's patchwork is clipped, not moved; plots are laid out
        if(plan.farm_plots)lay_out_farm_plots(plan,tile,assets,source,neighbour);
        return;
    }
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
void settle_farm_fields(Plan& plan,c3x_renderer_tile_v1 const& tile,Assets const& assets,Source source){
    settle_farm_fields(plan,tile,assets,source,[](int,int){return static_cast<c3x_renderer_tile_v1 const*>(nullptr);});
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
    // Each bridged join's straight approach (see append_pattern_route): the
    // distance from the join along its network (road or railroad) to every
    // line closer than the straight run's end, and the run. The run reaches
    // a short way past the deck's end, less where the bridge stands deep
    // inside the tile.
    std::vector<PatternRoute> approached;
    if(std::any_of(plan.patterns.begin(),plan.patterns.end(),[](PatternRoute const& route){return route.bridges!=0u;})){
        approached=plan.patterns;
        std::size_t total=plan.patterns.size();
        auto line_of=[&](std::size_t i)->RoutePatterns::Line const*{
            auto const* patterns=assets.patterns_for(plan.patterns[i].style);
            return patterns && plan.patterns[i].line<patterns->lines.size()?&patterns->lines[plan.patterns[i].line]:nullptr;
        };
        auto points_of=[&](std::size_t i){
            auto const& route=plan.patterns[i];auto const& line=*line_of(i);
            return route.points.size()==line.count?route.points.data():assets.patterns_for(route.style)->points.data()+line.first;
        };
        auto end_of=[&](std::size_t i,unsigned k){return points_of(i)[k?line_of(i)->count-1u:0u];};
        auto edge=[&](std::size_t i,unsigned k){return (k?line_of(i)->end:line_of(i)->start)>=0;};
        auto meet=[&](std::array<float,2> const& a,std::array<float,2> const& b){return std::hypot(a[0]-b[0],a[1]-b[1])<1e-3f;};
        std::vector<float> lengths(total,0.f);
        for(std::size_t i=0;i<total;++i)if(auto const* line=line_of(i)){
            auto const* points=points_of(i);
            for(unsigned k=1;k<line->count;++k)lengths[i]+=std::hypot(points[k][0]-points[k-1][0],points[k][1]-points[k-1][1]);
        }
        for(std::size_t index=0;index<total;++index)for(unsigned side=0;side<2;++side){
            auto const& route=plan.patterns[index];
            if(!(route.bridges&(1u<<side)) || !line_of(index))continue;
            bool railroad=route.style>=4u;
            auto in_network=[&](std::size_t i){return line_of(i) && (plan.patterns[i].style>=4u)==railroad;};
            // A loose end that stops just short of another line (the pattern
            // importer leaves a few) joins it at its nearest point.
            struct Link {std::size_t from;unsigned end;std::size_t to;float at,gap;};
            std::vector<Link> links;
            std::vector<float> inside(total,-1.f);
            for(std::size_t i=0;i<total;++i)if(in_network(i))for(unsigned k=0;k<2;++k){
                if(edge(i,k))continue;
                auto node=end_of(i,k);
                bool loose=true;
                for(std::size_t j=0;j<total;++j)if(j!=i && in_network(j))for(unsigned m=0;m<2;++m)
                    loose=loose && (edge(j,m) || !meet(end_of(j,m),node));
                if(!loose)continue;
                Link best{i,k,total,0.f,.06f};
                for(std::size_t j=0;j<total;++j)if(j!=i && in_network(j)){
                    auto const* points=points_of(j);float at=0.f;
                    for(unsigned m=1;m<line_of(j)->count;++m){
                        float du=points[m][0]-points[m-1][0],dv=points[m][1]-points[m-1][1],step=std::hypot(du,dv);
                        float t=step>0.f?std::clamp(((node[0]-points[m-1][0])*du+(node[1]-points[m-1][1])*dv)/(step*step),0.f,1.f):0.f;
                        float gap=std::hypot(points[m-1][0]+du*t-node[0],points[m-1][1]+dv*t-node[1]);
                        if(gap<best.gap)best={i,k,j,at+step*t,gap};
                        at+=step;
                    }
                }
                if(best.to<total && inside[best.to]<0.f){inside[best.to]=best.at;links.push_back(best);}
            }
            // Distance along the network to each line's start, end and inner
            // point; a tile edge other than the bridged join leads nowhere.
            constexpr float unreached=1e9f;
            std::vector<std::array<float,3>> walk(total,{unreached,unreached,unreached});
            walk[index][side]=0.f;
            for(unsigned pass=0;pass<4u*unsigned(total)+4u;++pass){
                bool changed=false;
                auto lower=[&](float& value,float candidate){if(candidate<value-1e-5f){value=candidate;changed=true;}};
                for(std::size_t i=0;i<total;++i)if(in_network(i)){
                    auto& w=walk[i];float length=lengths[i];
                    lower(w[1],w[0]+length);lower(w[0],w[1]+length);
                    if(inside[i]>=0.f){
                        lower(w[2],w[0]+inside[i]);lower(w[2],w[1]+length-inside[i]);
                        lower(w[0],w[2]+inside[i]);lower(w[1],w[2]+length-inside[i]);
                    }
                    for(unsigned k=0;k<2;++k)if(!edge(i,k))for(std::size_t j=0;j<total;++j)if(j!=i && in_network(j))
                        for(unsigned m=0;m<2;++m)if(!edge(j,m) && meet(end_of(j,m),end_of(i,k))){lower(walk[j][m],w[k]);lower(w[k],walk[j][m]);}
                }
                for(auto const& link:links){
                    lower(walk[link.to][2],walk[link.from][link.end]+link.gap);
                    lower(walk[link.from][link.end],walk[link.to][2]+link.gap);
                }
                if(!changed)break;
            }
            float deck_end=std::max(0.f,route.half_length()-route.crossing[side]);
            float run=deck_end+std::clamp(.40f-deck_end,.05f,.12f);
            // Every other tile edge the network reaches keeps a drawn stretch
            // and its own place (its neighbor draws the same point).
            float edge_walk=1e9f;
            for(std::size_t i=0;i<total;++i)if(in_network(i))for(unsigned k=0;k<2;++k)
                if(edge(i,k) && !(i==index && k==side))edge_walk=std::min(edge_walk,walk[i][k]);
            run=std::min(run,std::max(0.f,edge_walk-.08f));
            float fade=PatternRoute::approach_level_fade;
            float held=std::max(run,std::min(PatternRoute::approach_level,edge_walk-.02f-fade));
            auto join=end_of(index,side);
            // The bridge's deck level, as seat_route_bridge finds it.
            float bu=-route.joins[side*2],bv=-route.joins[side*2+1],c=route.crossing[side];
            float tu=float(input.tile.tile_x+input.tile.tile_y)*.5f,tv=float(input.tile.tile_x-input.tile.tile_y)*.5f;
            float deck_level=route_bridge_level(tu+join[0]-bu*c,tv+1.f-(join[1]-bv*c),-bu,bv,route.half_length(),
                [&](float u,float v){return std::max(relief(u,v)[0],height(u,v)-2.5f);});
            for(std::size_t i=0;i<total;++i)if(in_network(i) && std::min({walk[i][0],walk[i][1],walk[i][2]})<std::max(run,held+fade))
                for(auto& slot:approached[i].approach)if(slot[2]==0.f && slot[3]==0.f){
                    slot={join[0],join[1],-route.joins[side*2],-route.joins[side*2+1],walk[i][0],walk[i][1],inside[i],walk[i][2],run,deck_level,held,fade};
                    break;
                }
        }
    }
    for(auto const& route:approached.empty()?plan.patterns:approached)if(auto const* patterns=assets.patterns_for(route.style))
        append_pattern_route(input,*patterns,route,relief,height,output.layers[route_layer]);
    for(auto const& instance:plan.instances){
        auto before=indexed?output.indices[instance.layer].size():output.layers[instance.layer].size();
        FeaturePlacement placement{};placement.asset_index=instance.asset;
        if(instance.family==farm_family)append_instance(input,assets[instance.family],placement,instance.u,instance.v,
            instance.rotation,instance.scale,instance.material,instance.owner,instance.shadow,false,
            farm_relief(plan,input.tile,relief,instance.region),height,output.layers[instance.layer],output.shadows,indexed?&output.indices[instance.layer]:nullptr,
            instance.region&0x400u?-.004f:0.f,0.f,plan.farm_kit,instance.stretch,
            instance.region&0x100u?plan.farm_clearing.shared:0u,instance.region&0x400u?.32f:.06f,
            plan.farm_clearing.soft);
        else append_instance(input,assets[instance.family],placement,instance.u,instance.v,instance.rotation,
            instance.scale,instance.material,instance.owner,instance.shadow,
            instance.family==site_family || instance.family==mine_family,
            relief,height,output.layers[instance.layer],output.shadows,indexed?&output.indices[instance.layer]:nullptr);
        if(instance_counts)instance_counts->push_back(unsigned(
            (indexed?output.indices[instance.layer].size():output.layers[instance.layer].size())-before));
    }
}
}}
