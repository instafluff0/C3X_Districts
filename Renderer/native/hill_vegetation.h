#pragma once
#include <algorithm>
#include <cstdint>
#include <initializer_list>

// Civ III selects hill, mountain and volcano canopy sheets from their four
// diagonal neighbors, keeping the raised terrain as the visible tile type.
namespace c3x_renderer {
inline unsigned native_hill_seed(int x,int y) {
    std::uint32_t value=std::uint32_t(x*0x193u)^std::uint32_t(y*0x217u);
    value^=value>>16;value*=0x7feb352du;
    value^=value>>15;value*=0x846ca68bu;
    return value^(value>>16);
}
template<class Lookup>
int native_hill_vegetation(int center_real, Lookup neighbor_real, unsigned tie_seed) {
    if (center_real != 5 && center_real != 6 && center_real != 10) return 0;
    int forest = 0, jungle = 0;
    for (int dy : {-1, 1}) for (int dx : {-1, 1}) {
        int real = neighbor_real(dx, dy);
        if (real == 7) ++forest;
        else if (real == 8) ++jungle;
        else if (real != 5 && real != 6 && real != 10) return 0;
    }
    if (forest > jungle) return 7;
    if (jungle > forest) return 8;
    if (forest) return (tie_seed & 1u) ? 7 : 8;
    return 0;
}

// A center height alone can bury a trunk on the uphill side of a slope. Lift
// the placement until every low/root vertex clears the sampled hill surface.
template<class Asset,class Height>
float native_hill_plant_ground(Asset const& asset,float world_u,float world_v,
        float cosine,float sine,float scale,float feature_to_relief,Height surface_height) {
    float lowest=1.0e9f;
    for(auto const& vertex:asset.vertices)lowest=std::min(lowest,vertex.position[2]);
    float ground=surface_height(world_u,world_v)-2.5f;
    for(auto const& vertex:asset.vertices) {
        if(vertex.position[2]>lowest+0.08f)continue;
        float x=(vertex.position[0]*cosine-vertex.position[1]*sine)*scale;
        float y=(vertex.position[0]*sine+vertex.position[1]*cosine)*scale;
        ground=std::max(ground,surface_height(world_u+x,world_v-y)-2.5f-
            vertex.position[2]*scale*feature_to_relief);
    }
    return ground+0.15f;
}
}
