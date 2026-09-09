#pragma once
#include <algorithm>
#include <array>
#include <cmath>

namespace c3x_renderer { namespace render_core {
// Extrema of the actual vertices in the native natural-mesh affine projection.
// No vertex copy, camera identity, resource ownership or persistent allocation.
struct ProjectedMeshBounds {
    std::array<float,4> extent{};
    bool valid=false;
    void include(float x,float y,float z) {
        float u=(x+y)*.5f;
        float v=(x-y+1)*.25f-(z*112.f-2.5f)*(.82f/224.f);
        if(!valid){extent={u,v,u,v};valid=true;}
        else{extent[0]=std::min(extent[0],u);extent[1]=std::min(extent[1],v);
             extent[2]=std::max(extent[2],u);extent[3]=std::max(extent[3],v);}
    }
    std::array<long,4> project(int column,int row,int width) const {
        float x=(float(column)+float(row))*.5f,y=(float(column)-float(row))*.25f;
        return {long(std::floor((extent[0]-x)*width))-2,long(std::floor((extent[1]-y)*width))-2,
                long(std::ceil((extent[2]-x)*width))+2,long(std::ceil((extent[3]-y)*width))+2};
    }
};
} }
