#pragma once
#include <cmath>
#include <cstdint>

namespace c3x_renderer { namespace render_core {
// Native anchors stay unchanged. A retained raster may move only by whole
// display pixels at the same zoom: filtering would change coverage and depth.
struct StaticRegionShift {
    enum Reason { ready,fractional_phase,guard_bounds };
    int x=0,y=0;
    bool reusable=false;
    Reason reason=fractional_phase;
    static StaticRegionShift between(float zoom,int camera_x,int camera_y,
            int origin_x,int origin_y,int margin_x,int margin_y) {
        auto dx=std::int64_t(camera_x)-origin_x;
        auto dy=std::int64_t(camera_y)-origin_y;
        double px=double(zoom)*dx,py=double(zoom)*dy;
        StaticRegionShift result;
        // Exact phase equality, including negative coordinates. Do not round a
        // fractional displacement into a different raster sample lattice.
        if(!std::isfinite(px)||!std::isfinite(py) ||
                px!=std::floor(px)||py!=std::floor(py))return result;
        if(px < -margin_x || px > margin_x ||
                py < -margin_y || py > margin_y){result.reason=guard_bounds;return result;}
        result.x=int(px);result.y=int(py);
        result.reusable=true;
        result.reason=ready;
        return result;
    }
    template<class Rect> Rect needed(int width,int height,int margin_x,int margin_y)const {
        return {margin_x-x,margin_y-y,margin_x-x+width,margin_y-y+height};
    }
};
}}
