#pragma once
// Typed portable adapters around the same production statement bodies. Native
// includes those bodies in its existing tile compiler to preserve x86 rounding.
#include "data.h"
#include "ground.h"
namespace c3x_renderer { namespace fidelity {
template<class Height,class Shore,class Cancelled>
bool emit_relief_meshes(NaturalData const&natural,int real,Tile owner,GroundProjection project_natural,
                       Height height_natural,Shore shore_sample_at,Cancelled cancelled,
                       std::vector<MapVertex>&decals,std::vector<MapVertex>&mountains) {
    using Vertex=MapVertex;
    int nc=project_natural.column,nr=project_natural.row;
    struct {int real_terrain_type;} tile{real};
    std::array<std::vector<Vertex>*,3> layers{{nullptr,&decals,&mountains}};
    struct LayerView {
        decltype(layers)&values;
        std::vector<Vertex>&operator[](unsigned index){return *values[index];}
    } natural_vertices{layers};
    auto triangle=[](std::vector<Vertex>&out,Vertex const&a,Vertex const&b,Vertex const&c){out.push_back(a);out.push_back(b);out.push_back(c);};
    #include "relief_mesh_body.h"
    return true;
}
struct BuildingBounds {float x0,y0,x1,y1;};
template<class Height,class Shore,class River,class Hash,class Random,class Cancelled,class Layers>
bool emit_forest(NaturalData const&natural,Tile owner,GroundProjection project_natural,
                 std::vector<BuildingBounds> const&buildings,Height height_natural,Shore shore_sample_at,
                 River river_at,Hash hash,Random random,Cancelled cancelled,Layers&natural_vertices) {
    int nc=project_natural.column,nr=project_natural.row;
    #include "forest_mesh_body.h"
    return true;
}
} }
