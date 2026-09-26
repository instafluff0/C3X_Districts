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
struct Route {float u0,v0,u1,v1;unsigned style;bool railroad,bridge,reverse,bypass=false;float bridge_t=1.0f;bool bridge_structural=true;bool isolated=false;};
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
template<class Height>
float hill_wall_ground(FeatureAsset const& asset,c3x_renderer_tile_v1 const& tile,
        float local_u,float local_v,float rotation,float scale,Height natural_height_at){
    float world_u=float(tile.tile_x+tile.tile_y)*.5f;
    float world_v=float(tile.tile_x-tile.tile_y)*.5f;
    float highest=natural_height_at(world_u+local_u,world_v+1.f-local_v);
    float lowest_source=0.f,cosine=std::cos(rotation),sine=std::sin(rotation);
    for(auto const& source:asset.vertices){
        float x=(source.position[0]*cosine-source.position[1]*sine)*scale;
        float y=(source.position[0]*sine+source.position[1]*cosine)*scale;
        highest=std::max(highest,natural_height_at(
            world_u+local_u+x,world_v+1.f-local_v-y));
        lowest_source=std::min(lowest_source,source.position[2]*scale);
    }
    return highest-2.5f-lowest_source*(150.f/.82f)+.02f;
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
    bool farm_asset=asset.id.rfind("farm_",0)==0;
    bool terrain_wall=asset.id.rfind("city/walls/",0)==0 &&
        tile.real_terrain_type==5;
    if(terrain_wall)ground_sample[0]=hill_wall_ground(
        asset,tile,local_u,local_v,rotation,scale,natural_height_at);
    if (farm_asset && asset.id.find(":base:")!=std::string::npos && ground_sample[2]<.55f)
        return;
    if (farm_asset && asset.id.find(":building:")!=std::string::npos && ground_sample[2]<.14f)
        return;
    if (farm_asset && asset.id.find(":tree:")!=std::string::npos && ground_sample[2]<.11f)
        return;
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
    bool farm_decal = false;
    if (farm_asset && !asset.vertices.empty()) {
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
        float local_z = source.position[2] * scale;
        auto vertex_ground = farm_decal
            ? relief_at_world(tile_world_u + local_u + local_x,
                              tile_world_v + 1.0f - local_v - local_y)
            : ground_sample;
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
            static_cast<float>(asset.texture_index) + material_offset + owner_code,
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
        if(asset.id.find(":crop:")!=std::string::npos){
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
        for(std::size_t triangle=0;triangle+2<asset.indices.size();triangle+=3){
            std::array<ShoreVertex,4> polygon{};
            unsigned count=3;
            for(unsigned corner=0;corner<3;++corner){
                auto index=asset.indices[triangle+corner];
                polygon[corner]={transformed[index],farm_shore[index]};
            }
            std::array<ShoreVertex,4> clipped{};
            unsigned kept=0;
            for(unsigned corner=0;corner<count;++corner){
                auto const& a=polygon[corner];auto const& b=polygon[(corner+1)%count];
                bool a_land=a.distance>=0,b_land=b.distance>=0;
                if(a_land)clipped[kept++]=a;
                if(a_land!=b_land)clipped[kept++]=intersect(a,b);
            }
            for(unsigned corner=1;corner+1<kept;++corner){
                for(unsigned index:{0u,corner,corner+1}){
                    if(topology){topology->push_back(unsigned(target.size()));
                        target.push_back(clipped[index].vertex);}
                    else target.push_back(clipped[index].vertex);
                }
            }
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
        bool mountain_ring=tile.real_terrain_type==6 && tile.road_mask && !tile.railroad_mask;
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
            if (road && !railroad && (direction & 1)) {
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
            if(routes_enabled)
                append_route_segment(mountain_ring?ring_u[direction]:center_u,
                    mountain_ring?ring_v[direction]:center_v,edge_u,edge_v,
                    style, railroad, bridge, direction >= 4,mountain_ring);
            if (bridge && direction < 4) {
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
        if (!connected && routes_enabled) {
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
        River river_distance,Plan& plan){
    float tile_world_u=float(tile.tile_x+tile.tile_y)*.5f;
    float tile_world_v=float(tile.tile_x-tile.tile_y)*.5f;
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
        unsigned seed=c3x_renderer::stable_hash(tile.variant_seed ^
            (tile.irrigation_mask&15u)*0x9e3779b9u ^ unsigned(ground)*0x85ebca6bu);
        unsigned color_mask=(seed>>8)&15u;
        if(color_mask==0u)color_mask=1u<<((seed>>12)&3u);
        if(color_mask==15u)color_mask^=1u<<((seed>>12)&3u);
        float field_rotation=float(seed&3u)*1.57079632679f+
            (c3x_renderer::stable_random(seed+29u)-.5f)*.08f;
        constexpr unsigned palettes[5][2]={{1,2},{1,0},{0,1},{2,1},{0,2}};
        unsigned terrain=unsigned(std::clamp(ground,0,4));
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
                        .01f*float(emissive_code+1u),true,farm_vertices);
                }
            } else if (building_part) {
                append_feature_instance(farm_bundle,placement,
                    .37f+float(seed&1u)*.20f,.38f+float((seed>>1)&1u)*.19f,
                    float((seed>>2)&3u)*1.57079632679f,1.45f,21.0f,
                    .01f*float(emissive_code+1u),true,farm_vertices);
            }
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
            // A Lab wall bundle can provide the individual pieces used by the
            // fixed city layouts. Ordinary bundles retain the legacy wall path.
            char const * lab_era = era_names[std::min(era, 2u)];
            std::string lab_prefix = std::string("wall_lab_") + lab_era + "_";
            auto const * segment = c3x_renderer::find_feature_group(
                wall_bundle, (lab_prefix + "segment").c_str());
            auto const * gate = c3x_renderer::find_feature_group(
                wall_bundle, (lab_prefix + "gate").c_str());
            auto const * tower = c3x_renderer::find_feature_group(
                wall_bundle, (lab_prefix + "tower").c_str());
            if (composed_city && segment != nullptr && gate != nullptr &&
                tower != nullptr && !segment->placements.empty() &&
                !gate->placements.empty() && !tower->placements.empty()) {
                constexpr float pi = 3.14159265359f;
                constexpr unsigned samples = 2048u;
                float const radius[3] = {.48f, .64f, .76f};
                unsigned const sectors[3] = {16u, 20u, 24u};
                std::array<std::array<float, 2>, samples + 1u> points{};
                std::array<float, samples + 1u> distances{};
                for (unsigned index = 0u; index <= samples; ++index) {
                    float angle = pi / 4.0f + float(index) * 2.0f * pi / float(samples);
                    float cosine = std::cos(angle), sine = std::sin(angle);
                    points[index] = {
                        radius[size] * std::copysign(std::pow(std::abs(cosine), 1.0f / 3.0f), cosine),
                        radius[size] * std::copysign(std::pow(std::abs(sine), 1.0f / 3.0f), sine)};
                    if (index != 0u) {
                        float du = points[index][0] - points[index - 1u][0];
                        float dv = points[index][1] - points[index - 1u][1];
                        distances[index] = distances[index - 1u] + std::sqrt(du * du + dv * dv);
                    }
                }
                std::vector<std::array<float, 3>> ring;
                ring.reserve(sectors[size]);
                for (unsigned sector = 0u; sector < sectors[size]; ++sector) {
                    float target = distances[samples] * float(sector) / float(sectors[size]);
                    unsigned index = 0u;
                    while (index < samples && distances[index] < target) ++index;
                    float blend = index == 0u ? 0.0f :
                        (target - distances[index - 1u]) /
                        (distances[index] - distances[index - 1u]);
                    float x = index == 0u ? points[0u][0] :
                        points[index - 1u][0] * (1.0f - blend) + points[index][0] * blend;
                    float y = index == 0u ? points[0u][1] :
                        points[index - 1u][1] * (1.0f - blend) + points[index][1] * blend;
                    unsigned before = index == 0u ? samples - 1u : index - 1u;
                    unsigned after = std::min(index + 1u, samples);
                    float tangent = std::atan2(points[after][1] - points[before][1],
                                               points[after][0] - points[before][0]);
                    ring.push_back({x, y, tangent - pi / 2.0f});
                }
                for (unsigned sector = 0u; sector < sectors[size]; ++sector) {
                    auto const & position = ring[sector];
                    auto const & part = sector == 0u ? gate->placements.front()
                                                     : segment->placements.front();
                    append_feature_instance(wall_bundle, part,
                        .5f + position[0], .5f + position[1],
                        position[2] + (sector == 0u ? pi / 2.0f : 0.0f),
                        part.scale, 29.0f,
                        .08f * static_cast<float>(owner + 1u), true, wall_vertices);
                }
                for (unsigned sector = 2u; sector < sectors[size]; sector += 4u) {
                    auto const & position = ring[sector];
                    auto const & part = tower->placements.front();
                    append_feature_instance(wall_bundle, part,
                        .5f + position[0], .5f + position[1], position[2],
                        part.scale, 29.0f,
                        .08f * static_cast<float>(owner + 1u), true, wall_vertices);
                }
            } else {
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
    }
    return true;
}
template<class Relief>
void settle_farm_fields(Plan& plan,c3x_renderer_tile_v1 const& tile,Assets const& assets,Relief relief){
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
template<class Relief>
void settle_farm_props(Plan& plan,c3x_renderer_tile_v1 const& tile,Assets const& assets,Relief relief){
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
    for(auto const& instance:plan.instances){
        auto before=indexed?output.indices[instance.layer].size():output.layers[instance.layer].size();
        FeaturePlacement placement{};placement.asset_index=instance.asset;
        append_instance(input,assets[instance.family],placement,instance.u,instance.v,instance.rotation,
            instance.scale,instance.material,instance.owner,instance.shadow,instance.family==site_family,
            relief,height,output.layers[instance.layer],output.shadows,indexed?&output.indices[instance.layer]:nullptr);
        if(instance_counts)instance_counts->push_back(unsigned(
            (indexed?output.indices[instance.layer].size():output.layers[instance.layer].size())-before));
    }
}
}}
