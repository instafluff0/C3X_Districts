#pragma once
#include <array>
#include <cmath>
#include <stdexcept>

namespace q4_relief {
// Height must evaluate the SAME continuous world field across the tile halo.
// It must include final ground, relief and hydrology; never clamp to a caller's
// local tile or select a different field from the incident tile's terrain id.
// height_units_per_tile converts the height scalar to horizontal world units.
template<class Height>
std::array<double,3> continuous_normal(double x,double y,double step,
                                      double height_units_per_tile,Height height) {
    if(!(step>0) || !(height_units_per_tile>0))
        throw std::invalid_argument("invalid continuous derivative scale");
    double dx=(height(x+step,y)-height(x-step,y))/(2*step*height_units_per_tile);
    double dy=(height(x,y+step)-height(x,y-step))/(2*step*height_units_per_tile);
    double length=std::sqrt(dx*dx+dy*dy+1);
    if(!std::isfinite(length))throw std::runtime_error("nonfinite continuous height");
    return {-dx/length,-dy/length,1/length};
}

// Q0 frozen vertex normals use local v, opposite continuous row+(1-v).
// Convert once at that adapter boundary; keep world normals in world space.
inline std::array<double,3> to_local_uv(std::array<double,3> normal) {
    normal[1]=-normal[1];return normal;
}
}
