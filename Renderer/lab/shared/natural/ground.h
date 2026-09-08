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
    float coverage=coast_coverage(float(shore.distance),float(shore.beach_width));
    // Preserve a broader inland coordinate for softening only the transition
    // receiver. The visible coverage ramp remains unchanged.
    out.world_valid=1+coast_ramp((float(shore.distance)-float(shore.beach_width)-.10f)/.90f);
    auto weights=weights_at(x,y);
    // Keep native marsh below the selected source families at their boundary.
    float source_weight=std::clamp(1-weights[3],0.f,1.f);
    out.base_terrain=-10+coverage*source_weight;
    for(auto&weight:weights)weight/=std::max(source_weight,.00001f);
    out.material_marsh=weights[4];
    out.authored_relief_height=weights[1];out.authored_relief_blend=weights[2];
    return out;
}

template<class Surface,class Cancel>
bool emit_ground_grid(std::vector<MapVertex>&out,Surface surface,Cancel cancelled) {
    std::array<MapVertex,17*17> grid;
    for(unsigned y=0;y<=16;y++){
        if(cancelled())return false;
        for(unsigned x=0;x<=16;x++)grid[y*17+x]=surface(x/16.f,y/16.f);
    }
    for(unsigned y=0;y<16;y++)for(unsigned x=0;x<16;x++){
        auto&a=grid[y*17+x];auto&b=grid[y*17+x+1];auto&c=grid[(y+1)*17+x+1];auto&d=grid[(y+1)*17+x];
        if(std::max({a.base_terrain,b.base_terrain,c.base_terrain,d.base_terrain})<=-9.999f)continue;
        out.push_back(a);out.push_back(b);out.push_back(c);
        out.push_back(a);out.push_back(c);out.push_back(d);
    }
    return true;
}
}}
