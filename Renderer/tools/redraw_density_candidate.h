#pragma once
// Bounded candidate, used only by the private density fixture until qualified.
// Simplify a retained fine grid, never resample heights/materials from a camera.
#include <array>
#include <cmath>
#include <cstring>
#include <vector>
#include "../lab/shared/natural/vertex.h"
namespace c3x_renderer { namespace fidelity {
struct SurfaceRefinementLeaf {unsigned x,y,size;};
using SurfaceRefinementValue=std::array<float,sizeof(MapVertex)/sizeof(float)>;
inline SurfaceRefinementValue surface_refinement_value(MapVertex const& v){
    SurfaceRefinementValue result;std::memcpy(result.data(),&v,sizeof(v));return result;
}
inline float surface_refinement_limit(unsigned field){
    if(field<3)return .04f; // Canonical projection/depth coordinates.
    if(field>=6 && field<=8)return .003f; // Normal components, before shading.
    if(field==30 || field==31)return .000002f; // World XY stays affine.
    if(field==32)return .04f/112; // Native-pixel height; no world/camera variants.
    return .001f; // UVs, coverage, all material/effect payloads.
}

template<class Cancel>
bool append_refined_surface_grid(std::vector<MapVertex>& out,
        std::vector<MapVertex> const& grid,unsigned cells,Cancel cancelled,
        std::vector<unsigned>* indices){
    if(cells!=64 || grid.size()!=(cells+1)*(cells+1))return false;
    unsigned const stride=cells+1;
    std::vector<SurfaceRefinementValue> values;values.reserve(grid.size());
    for(auto const& v:grid)values.push_back(surface_refinement_value(v));
    auto mix=[](SurfaceRefinementValue const& a,SurfaceRefinementValue const& b,
                SurfaceRefinementValue const& c,float wa,float wb,float wc){
        SurfaceRefinementValue result;
        for(unsigned i=0;i<result.size();++i)result[i]=a[i]*wa+b[i]*wb+c[i]*wc;
        return result;
    };
    auto source=[&](float x,float y){
        unsigned ix=std::min(unsigned(x),cells-1),iy=std::min(unsigned(y),cells-1);
        float u=x-float(ix),v=y-float(iy);unsigned a=iy*stride+ix,b=a+1,d=a+stride,c=d+1;
        return u>=v?mix(values[a],values[b],values[c],1-u,u-v,v):
                    mix(values[a],values[c],values[d],1-v,u,v-u);
    };
    auto coarse=[&](SurfaceRefinementLeaf leaf,float x,float y){
        float u=(x-float(leaf.x))/float(leaf.size),v=(y-float(leaf.y))/float(leaf.size);
        unsigned a=leaf.y*stride+leaf.x,b=a+leaf.size,d=a+leaf.size*stride,c=d+leaf.size;
        unsigned center=a+(leaf.size/2)*(stride+1);
        // Four corner fans. All later stitched fans lie within these triangles.
        if(v<=u && v<=1-u)return mix(values[center],values[a],values[b],2*v,1-u-v,u-v);
        if(u>=v && u>=1-v)return mix(values[center],values[b],values[c],2*(1-u),u-v,u+v-1);
        if(v>=u && v>=1-u)return mix(values[center],values[c],values[d],2*(1-v),u+v-1,v-u);
        return mix(values[center],values[d],values[a],2*u,v-u,1-u-v);
    };
    auto accepts=[&](SurfaceRefinementLeaf leaf){
        // Preserve the complete 64-cell patch boundary for adjacent owners.
        if(!leaf.x || !leaf.y || leaf.x+leaf.size==cells || leaf.y+leaf.size==cells)return false;
        // Original diagonals and corner-fan diagonals intersect at grid points
        // or half-grid points. Their piecewise-affine difference reaches its
        // extrema there. Half the limit leaves room for exact finer neighbors.
        for(unsigned y=0;y<=leaf.size*2;++y)for(unsigned x=0;x<=leaf.size*2;++x){
            float px=float(leaf.x)+float(x)*.5f,py=float(leaf.y)+float(y)*.5f;
            auto original=source(px,py),reduced=coarse(leaf,px,py);
            for(unsigned i=0;i<original.size();++i)
                if(!std::isfinite(original[i]) || !std::isfinite(reduced[i]) ||
                   std::abs(original[i]-reduced[i])>surface_refinement_limit(i)*.5f)return false;
        }
        return true;
    };
    std::vector<SurfaceRefinementLeaf> leaves;
    auto split=[&](auto&& self,SurfaceRefinementLeaf leaf)->void{
        if(leaf.size==1 || accepts(leaf)){leaves.push_back(leaf);return;}
        unsigned half=leaf.size/2;
        for(unsigned y=0;y<2;++y)for(unsigned x=0;x<2;++x)
            self(self,{leaf.x+x*half,leaf.y+y*half,half});
    };
    for(unsigned y=0;y<cells;y+=4){
        if(cancelled())return false;
        for(unsigned x=0;x<cells;x+=4)split(split,{x,y,4});
    }
    std::vector<unsigned char> corners(grid.size(),0);
    for(auto leaf:leaves){unsigned a=leaf.y*stride+leaf.x,b=a+leaf.size,d=a+leaf.size*stride,c=d+leaf.size;
        for(auto at:{a,b,c,d})corners[at]=1;
    }
    std::vector<MapVertex> vertices;std::vector<unsigned> topology;
    std::vector<unsigned> remap(grid.size(),~0u);
    auto append=[&](unsigned at){
        if(indices){
            if(remap[at]==~0u){remap[at]=unsigned(vertices.size());vertices.push_back(grid[at]);}
            topology.push_back(remap[at]);
        }else vertices.push_back(grid[at]);
    };
    for(auto leaf:leaves){
        if(cancelled())return false;
        unsigned a=leaf.y*stride+leaf.x,b=a+leaf.size,d=a+leaf.size*stride,c=d+leaf.size;
        if(leaf.size==1){for(auto at:{a,b,c,a,c,d})append(at);continue;}
        unsigned center=a+(leaf.size/2)*(stride+1);
        std::vector<unsigned> perimeter;
        for(unsigned i=0;i<leaf.size;++i)if(corners[a+i])perimeter.push_back(a+i);
        for(unsigned i=0;i<leaf.size;++i)if(corners[b+i*stride])perimeter.push_back(b+i*stride);
        for(unsigned i=0;i<leaf.size;++i)if(corners[c-i])perimeter.push_back(c-i);
        for(unsigned i=0;i<leaf.size;++i)if(corners[d-i*stride])perimeter.push_back(d-i*stride);
        for(unsigned i=0;i<perimeter.size();++i){
            append(center);append(perimeter[i]);append(perimeter[(i+1)%perimeter.size()]);
        }
    }
    if(cancelled())return false;
    unsigned offset=unsigned(out.size());out.insert(out.end(),vertices.begin(),vertices.end());
    if(indices)for(auto at:topology)indices->push_back(offset+at);
    return true;
}
}}
