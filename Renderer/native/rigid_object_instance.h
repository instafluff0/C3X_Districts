#pragma once
#include "object_compiler.h"
#include "../lab/shared/natural/instance.h"
#include <limits>

namespace c3x_renderer { namespace objects {
inline bool shared_rigid_mesh(FeatureAsset const& asset){
    // Small flat surface/decal meshes retain their grouped packed path instead
    // of turning each component into another draw. Preserve their exact floats.
    // This is a geometry property, independent of pack origin or asset naming.
    if(asset.vertices.empty() || asset.indices.empty())return false;
    auto z=asset.vertices.front().position[2];
    float tolerance=asset.id.rfind("farm_",0)==0?1e-5f:0.0f;
    for(auto const& vertex:asset.vertices)if(std::abs(vertex.position[2]-z)>tolerance)return true;
    return false;
}
// Copied placement of a rigid source asset. The source mesh belongs to the
// immutable pack; the tile owns placement, material and exact culling bounds.
struct PreparedRigid {
    unsigned family=0,asset=0,layer=0;
    fidelity::MeshInstance instance;
    float material=0;
    std::array<int,4> bounds{};
    std::array<float,3> low{},high{};
};
template<class Relief,class Height>
PreparedRigid prepare_rigid(Instance const& source,Projection const& input,Assets const& assets,
        Relief relief,Height height){
    PreparedRigid result;result.family=source.family;result.asset=source.asset;result.layer=source.layer;
    auto const& asset=assets[source.family].assets[source.asset];
    float u=float(input.tile.tile_x+input.tile.tile_y)*.5f,v=float(input.tile.tile_x-input.tile.tile_y)*.5f;
    float ground=relief(u+source.u,v+1-source.v)[0];
    if(source.family==site_family)ground=height(u+source.u,v+1-source.v)-2.5f;
    float values[]={u,v,source.u,source.v,std::cos(source.rotation),std::sin(source.rotation),source.scale,ground};
    std::copy(values,values+8,result.instance.place);
    result.material=float(asset.texture_index)+source.material+source.owner;
    // The same CPU compiler supplies exact conservative bounds once, at world
    // preparation. Expanded vertices are transient and never stored/uploaded.
    FeaturePlacement placement{};placement.asset_index=source.asset;
    std::vector<Vertex> vertices,shadows;std::vector<unsigned> indices;
    append_instance(input,assets[source.family],placement,source.u,source.v,source.rotation,source.scale,
        source.material,source.owner,false,source.family==site_family,relief,height,vertices,shadows,&indices);
    result.bounds={std::numeric_limits<int>::max(),std::numeric_limits<int>::max(),
        std::numeric_limits<int>::min(),std::numeric_limits<int>::min()};
    result.low.fill(1e9f);result.high.fill(-1e9f);
    for(auto const& vertex:vertices){
        result.bounds[0]=std::min(result.bounds[0],int(std::floor(vertex.x))-2);
        result.bounds[1]=std::min(result.bounds[1],int(std::floor(vertex.y))-2);
        result.bounds[2]=std::max(result.bounds[2],int(std::ceil(vertex.x))+2);
        result.bounds[3]=std::max(result.bounds[3],int(std::ceil(vertex.y))+2);
        float position[]={vertex.world_x,vertex.world_y,vertex.world_z};
        for(unsigned axis=0;axis<3;++axis){result.low[axis]=std::min(result.low[axis],position[axis]);
            result.high[axis]=std::max(result.high[axis],position[axis]);}
    }
    return result;
}
}}
