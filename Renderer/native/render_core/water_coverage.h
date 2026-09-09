#pragma once
#include <cmath>

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
} }
