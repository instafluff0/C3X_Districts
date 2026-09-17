#pragma once
// Clean compiler boundary for cliff-body placement/geometry, mirroring
// terrain_compiler.h's shape: callers supply read-only queries as explicit
// function objects, this never writes to renderer-owned state, and every
// dependency actually read is recorded in the returned CliffSurfaces instead
// of being pushed into a caller-owned accumulator via a captured reference.
// cliff_placements() itself (render_core/cliff_placement.h) was already a
// pure function; this module is the missing per-tile output isolation layer
// around it, the same role terrain_mesh_body.h plays for natural terrain.
//
// This alone does not make cliff generation worker-eligible: the queries
// callers pass in (world_lookup/height/shore) still read a per-tile SurfaceQueries
// instance shared with ground/city generation, so running this concurrently
// with them would race that shared cache. A private per-compile scratch
// (mirroring TerrainCompileScratch's independent ExactPointCache instances)
// is the next step before this can move off the foreground thread.
#include "../render_core/cliff_placement.h"
#include "../terrain_scene_runtime.h"
#include "../../lab/shared/natural/ground.h"
namespace c3x_renderer { namespace fidelity {
struct CliffCompileInput {
    int tile_x=0,tile_y=0;
    float left=0,top=0,half_w=0,half_h=0,relief_projection_scale=1,content_view_height=0;
    // Imported cliff bodies use the same vertical projection as every other
    // feature, then convert to the shared relief-world height used by
    // placement, depth, shadows and water. Computed once by the caller (it
    // also feeds the asset_height callback below), not recomputed here.
    float vertical_basis=1;
};
struct CliffSurfaces {
    std::array<std::vector<MapVertex>,8> vertices;
    std::unordered_map<std::uint64_t,std::uint64_t> coast;
};
template<class WorldLookup,class WorldIndex,class HeightAt,class ShoreAt,class AssetHeight,class Contour,class Recipe,class Cancelled>
void compile_cliff_surfaces(render_core::World world,FeatureBundle const& cliff_bundle,CliffCompileInput const& input,
        WorldLookup world_lookup,WorldIndex world_index,HeightAt height_at,ShoreAt shore_at,AssetHeight asset_height,
        Contour contour,Recipe recipe,Cancelled cancelled,CliffSurfaces& destination) {
    float cu=float(input.tile_x+input.tile_y)*.5f,cr=float(input.tile_x-input.tile_y)*.5f;
    auto placements=render_core::cliff_placements(world,int(cu),int(cr),world_lookup,world_index,
        height_at,shore_at,asset_height,contour,recipe,cancelled);
    GroundProjection cliff_projection{int(cu),int(cr),input.half_w,input.half_h,
        input.relief_projection_scale,input.content_view_height};
    for(auto const& instance:placements){
        auto const& asset=cliff_bundle.assets[instance.asset];
        render_core::CliffTransform transform(instance,input.vertical_basis);
        std::vector<MapVertex> transformed(asset.vertices.size());
        for(std::size_t i=0;i<asset.vertices.size();++i){
            auto const& source=asset.vertices[i];auto& v=transformed[i];
            auto position=transform.position(source.position),normal=transform.normal(source.normal);
            float wx=position[0],wy=position[1],wz=position[2];
            auto projected=cliff_projection(wx,wy,wz*112);
            v.x=input.left+projected.x;v.y=input.top+projected.y;v.z=input.top+projected.z;
            v.u=source.uv[0];v.v=source.uv[1];v.panel=1;
            v.normal_x=normal[0];v.normal_y=normal[1];v.normal_z=normal[2];v.base_terrain=.48f;
            v.shadow_visibility=v.ambient_visibility=1;
            v.world_x=wx;v.world_y=wy;v.world_z=wz;v.world_valid=1;
        }
        for(auto i:asset.indices)destination.vertices[instance.asset].push_back(transformed[i]);
    }
}
}}
