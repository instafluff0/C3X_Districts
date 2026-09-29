#pragma once
#include "data.h"
#include "ground.h"
namespace c3x_renderer { namespace fidelity {
// Draw the authored rock patches on the exact receiver triangles. Independent
// rotated grids intersect steep hills and give the patches unrelated normals.
// UV clipping in the material keeps the footprint without cutting the mesh.
inline void emit_hill_decals(Tile owner,int column,int row,
        std::vector<MapVertex> const& surface,std::vector<unsigned> const* indices,
        std::vector<MapVertex>& decals) {
    Hill hill=composed_hill(owner);std::uint32_t state=hill.seed;
    constexpr unsigned cells[]={0,0,0,0,1,1,1,2,2,2};
    for(unsigned ordinal=0;ordinal<10;++ordinal){
        float keep=random01(state),angle=random01(state)*6.283185307f;
        float radius=std::sqrt(random01(state))*.22f,phase=random01(state)*6.283185307f;
        float scale=.90f+.20f*random01(state);
        if(keep>hill.rockiness)continue;
        float cu=.5f+std::cos(phase)*radius,cv=.5f+std::sin(phase)*radius;
        float co=std::cos(angle),si=std::sin(angle);
        auto project=[&](MapVertex v){
            float u=v.world_x-column-cu,w=float(row)+1-v.world_y-cv;
            v.u=(co*u+si*w)/(.42f*scale)+.5f;
            v.v=(-si*u+co*w)/(.37f*scale)+.5f;
            v.material_grass=std::max(0.f,(v.world_z*112-2.5f)/112);
            v.material_plains=2;v.material_desert=float(cells[ordinal]);
            if(v.base_terrain>=42)v.base_terrain=-10+std::clamp(v.base_terrain-42,0.f,1.f);
            return v;
        };
        auto count=indices?indices->size():surface.size();
        for(std::size_t i=0;i+2<count;i+=3){
            MapVertex v[3];
            for(unsigned j=0;j<3;++j)v[j]=project(surface[indices?(*indices)[i+j]:i+j]);
            if(std::max({v[0].u,v[1].u,v[2].u})<0 || std::min({v[0].u,v[1].u,v[2].u})>1 ||
               std::max({v[0].v,v[1].v,v[2].v})<0 || std::min({v[0].v,v[1].v,v[2].v})>1)continue;
            decals.insert(decals.end(),std::begin(v),std::end(v));
        }
    }
}
}}
