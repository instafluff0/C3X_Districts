#pragma once
#include <cmath>
#include <algorithm>

namespace c3x_renderer { namespace render_core {
// The current city/environment water and bed shaders discard positive-land
// hydrology distance. Inspect the actual uploaded grid samples, not a tile's
// terrain label or an approximate coast query. Interpolation cannot introduce
// a negative distance when every corner is safely positive. Keep a generous
// margin and retain unknown/nonfinite samples conservatively.
template<class Vertices>
bool water_surface_can_contribute(Vertices const& vertices) {
    for(auto const& vertex:vertices)
        if(!std::isfinite(vertex.shore_true_distance) || vertex.shore_true_distance<=.01f)return true;
    return false;
}
// Conservative forward-layer dependency. Tall bodies project over ground
// behind their footprints, so include height and shore/river support. This is
// world-relative and therefore independent of view clipping and camera caches.
template<class World,class Bounds> bool water_under_projection(World const& world,Bounds const& bounds) {
    for(unsigned i=0;i<3;++i)if(!std::isfinite(bounds.low[i]) || !std::isfinite(bounds.high[i]))return true;
    float margin=2.f+2.f*std::max(std::abs(bounds.low[2]),std::abs(bounds.high[2]));
    int left=int(std::floor(bounds.low[0]-margin)),right=int(std::floor(bounds.high[0]+margin));
    int top=int(std::floor(bounds.low[1]-margin)),bottom=int(std::floor(bounds.high[1]+margin));
    if(right<left || bottom<top || right-left>64 || bottom-top>64)return true;
    for(int r=top;r<=bottom;++r)for(int c=left;c<=right;++c){auto value=world.at(world.index(c,r));
        if(value==0xffffffffu || (value&255)>=11 || (value&0x00aa0000u))return true;
    }
    return false;
}
} }
