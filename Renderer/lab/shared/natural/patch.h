#pragma once
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace c3x_renderer { namespace fidelity {
// A world patch owns surface values, never its regular-grid connectivity.
// These canonical layouts preserve the legacy first-reference triangle order.
struct PatchTopology {
    unsigned divisions=0;
    std::vector<unsigned> corners,indices;
    explicit PatchTopology(unsigned cells):divisions(cells) {
        if(!cells || cells>64)throw std::length_error("terrain patch extent");
        unsigned stride=cells+1;std::vector<unsigned> remap(stride*stride,~0u);
        corners.reserve(stride*stride);indices.reserve(cells*cells*6);
        for(unsigned y=0;y<cells;++y)for(unsigned x=0;x<cells;++x){
            unsigned a=y*stride+x,b=a+1,d=a+stride,c=d+1;
            for(auto at:{a,b,c,a,c,d}){
                if(remap[at]==~0u){remap[at]=unsigned(corners.size());corners.push_back(at);}
                indices.push_back(remap[at]);
            }
        }
    }
};
struct PatchLayouts {
    std::map<unsigned,PatchTopology> layouts;
    PatchTopology const& get(unsigned divisions) {
        auto found=layouts.find(divisions);
        if(found==layouts.end())found=layouts.emplace(divisions,PatchTopology(divisions)).first;
        return found->second;
    }
};
// Zero is exact compatibility. Positive values are explicitly selected visual
// candidates: approximate maximum native-pixel edge length, before supersampling.
// A view uses one common mountain lattice, including every collar/edge neighbor.
struct PatchDetail {
    unsigned mountain=64,rocky_ground=48;
    unsigned identity() const{return mountain*64+rocky_ground;}
    bool operator==(PatchDetail const& b)const{return identity()==b.identity();}
    PatchDetail()=default;
    PatchDetail(int tile_width,unsigned pixels) {
        if(!pixels)return;
        unsigned cells=unsigned(std::max(1.f,std::ceil(float(tile_width)*.559017f/float(pixels))));
        // Power-of-two levels give exact binary world coordinates and stable
        // shared edges. Adjacent patches never independently choose a level.
        unsigned level=8;while(level<cells && level<64)level*=2;
        mountain=level;rocky_ground=std::min(48u,level);
    }
};
} }
