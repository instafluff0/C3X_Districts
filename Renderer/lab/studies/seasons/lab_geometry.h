#pragma once
// Study adapters around production statement bodies, including inherited
// canopy and vegetation floors. These are never imported into game integration.
#include "../../shared/natural/mesh.h"
#include "../../shared/natural/patterns.h"
namespace c3x_renderer {
std::uint32_t stable_hash(std::uint32_t value){return patterns::feature_hash(value);}
float stable_random(std::uint32_t value){return patterns::stable_random(value);}
namespace fidelity {
template<class Lookup,class Height,class Shore,class River,class Cancelled,class Layers>
bool lab_vegetation(NaturalData const&natural,Tile owner,GroundProjection project_natural,
                   Lookup lookup_natural,Height height_natural,Shore shore_sample_at,
                   River river_at,Cancelled cancelled,Layers&natural_vertices){
    using Vertex=MapVertex;int nc=project_natural.column,nr=project_natural.row;
    struct {int real_terrain_type;}tile{owner.real};
    auto hash=patterns::feature_hash;
    auto random=patterns::stable_random;
    auto triangle=[](auto&out,auto const&a,auto const&b,auto const&c){out.push_back(a);out.push_back(b);out.push_back(c);};
    #include "../../shared/natural/vegetation_floor_mesh_body.h"
    int inherited=native_hill_vegetation(owner.real,[&](int dx,int dy){return lookup_natural(nc+(dx+dy)/2,nr+(dx-dy)/2).real;},native_hill_seed(owner.source_x,owner.source_y));
    bool hill_forest=owner.real==5 && inherited==7,hill_jungle=owner.real==5 && inherited==8;
    bool raised_canopy=(owner.real==6 || owner.real==10) && inherited;
    std::vector<BuildingBounds> buildings;auto emit_forest_instance=[](auto...){return false;};
    if(owner.real==7 || inherited==7){
        #include "../../shared/natural/forest_mesh_body.h"
    }
    if(owner.real==8 || inherited==8){
        #include "../../shared/natural/jungle_mesh_body.h"
    }
    return true;
}
}}
