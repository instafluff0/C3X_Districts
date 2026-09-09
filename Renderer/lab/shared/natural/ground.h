#pragma once
// The production surface compiler, shared by native and local scene builders.
// Callers supply authoritative neighborhood/height/material queries and retain
// ownership of caches, cancellation, layer ordering and graphics resources.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
#include "vertex.h"
#include "../../../native/source_fidelity/kernels.h"
#include "../../../native/source_fidelity/coast_join.h"
namespace c3x_renderer { namespace fidelity {
struct GroundProjection {
    int column, row;
    float half_width, half_height, relief_scale, target_height;
    MapVertex operator()(float x,float y,float height) const {
        MapVertex out={};float dx=x-float(column),dy=y-float(row);
        float base=(dx-dy+1)*half_height;
        out.x=(dx+dy)*half_width;
        out.y=base-(height-2.5f)*relief_scale;
        out.z=base+(height-2.5f)*.0016f*target_height;
        out.world_x=x;out.world_y=y;out.world_z=height/112;out.world_valid=1;
        out.normal_z=1;out.u=x;out.v=y;
        return out;
    }
};

template<class Height,class Shore,class Weights>
MapVertex ground_surface(GroundProjection const&project,float u,float v,
                         Height height,Shore shore_at,Weights weights_at) {
    float x=float(project.column)+u,y=float(project.row)+1-v,support=0;
    float h=height(x,y,&support);
    MapVertex out=project(x,y,h);
    constexpr float e=.006f;
    float n[]={-(height(x+e,y,nullptr)-height(x-e,y,nullptr))/(2*e*128),
        -(height(x,y+e,nullptr)-height(x,y-e,nullptr))/(2*e*128),1};normalize3(n);
    out.normal_x=n[0];out.normal_y=n[1];out.normal_z=n[2];
    out.material_grass=std::max(0.f,(h-2.5f)/112);out.material_plains=1;out.material_desert=support;
    auto shore=shore_at(x,y);
    auto weights=weights_at(x,y);
    float source_weight=std::clamp(1-weights[3],0.f,1.f);
    float desert_weight=std::clamp(weights[2]/std::max(source_weight,.00001f),0.f,1.f);
    float coverage=coast_coverage(float(shore.distance),float(shore.beach_width));
    coverage=coverage*(1-desert_weight)+
        desert_coast_coverage(float(shore.distance))*desert_weight;
    // Complete cliff coverage before its .04-tile rise begins. The beach
    // coverage ramp otherwise makes the elevated rock face translucent.
    float rocky=coast_ramp((float(shore.rocky)-.55f)/.4f);
    coverage=coverage*(1-rocky)+coast_ramp(float(shore.distance)/.04f)*rocky;
    // Values 1..1.25 remain the ground material class (<1.5), carrying
    // the rocky-coast weight to the cliff face shader without another stream.
    out.material_plains=1+.25f*coast_ramp((float(shore.rocky)-.55f)/.4f)*
        (1-coast_ramp((float(shore.distance)-.25f)/.25f));
    // Preserve a broader inland coordinate for softening only the transition
    // receiver. The visible coverage ramp remains unchanged.
    out.world_valid=1+coast_ramp((float(shore.distance)-float(shore.beach_width)-.10f)/.90f);
    // Keep native marsh below the selected source families at their boundary.
    out.base_terrain=-10+coverage*source_weight;
    for(auto&weight:weights)weight/=std::max(source_weight,.00001f);
    out.material_marsh=weights[4];
    out.authored_relief_height=weights[1];out.authored_relief_blend=weights[2];
    return out;
}

// Preserve first-reference order and omit unused corners, just as indexing the
// expanded triangle stream does, without hashing/copying six full vertices per
// cell. A null index destination retains the portable triangle-list adapter.
inline void append_surface_grid(std::vector<MapVertex>&out,std::vector<MapVertex> const&grid,
                                unsigned divisions,bool clip_coast,
                                std::vector<unsigned>*indices=nullptr) {
    unsigned stride=divisions+1;
    std::vector<unsigned> remap;
    if(indices)remap.assign(grid.size(),~0u);
    for(unsigned y=0;y<divisions;y++)for(unsigned x=0;x<divisions;x++){
        unsigned a=y*stride+x,b=a+1,d=a+stride,c=d+1;
        if(clip_coast && std::max({grid[a].base_terrain,grid[b].base_terrain,
                                  grid[c].base_terrain,grid[d].base_terrain})<=-9.999f)continue;
        for(unsigned corner:{a,b,c,a,c,d}){
            if(indices){
                if(remap[corner]==~0u){remap[corner]=unsigned(out.size());out.push_back(grid[corner]);}
                indices->push_back(remap[corner]);
            }else out.push_back(grid[corner]);
        }
    }
}

template<class Surface,class Cancel>
bool emit_ground_grid(std::vector<MapVertex>&out,Surface surface,Cancel cancelled,unsigned divisions=16,
                      std::vector<unsigned>*indices=nullptr) {
    unsigned stride=divisions+1;
    std::vector<MapVertex> grid(stride*stride);
    for(unsigned y=0;y<=divisions;y++){
        if(cancelled())return false;
        for(unsigned x=0;x<=divisions;x++)grid[y*stride+x]=surface(float(x)/divisions,float(y)/divisions);
    }
    append_surface_grid(out,grid,divisions,true,indices);
    return true;
}
}}
