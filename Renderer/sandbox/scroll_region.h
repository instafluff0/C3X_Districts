#pragma once
#include <cmath>
#include <cstdint>

namespace c3x_renderer { namespace render_core {
// A retained raster moves by whole display pixels at the same zoom. Odd shifts
// are allowed: the retained pixels and later strips share one region lattice,
// so the only difference from a fresh render is the 2x2 derivative-quad phase
// of already shaded pixels, which is visually imperceptible. Fractional shifts
// (non-endpoint zoom with a camera move) are rounded; the caller moves the
// whole frame's translation by the sub-pixel remainder so static and dynamic
// layers stay aligned. Native anchors stay unchanged.
struct StaticRegionShift {
    enum Reason { ready,fractional_phase,guard_bounds,derivative_phase };
    int x=0,y=0;
    // Sub-pixel display correction, in display pixels: rounded minus exact.
    double snap_x=0,snap_y=0;
    bool reusable=false;
    Reason reason=fractional_phase;
    static StaticRegionShift between(float zoom,int camera_x,int camera_y,
            int origin_x,int origin_y,int margin_x,int margin_y) {
        auto dx=std::int64_t(camera_x)-origin_x;
        auto dy=std::int64_t(camera_y)-origin_y;
        double px=double(zoom)*dx,py=double(zoom)*dy;
        StaticRegionShift result;
        if(!std::isfinite(px)||!std::isfinite(py))return result;
        double rx=std::floor(px+.5),ry=std::floor(py+.5);
        result.snap_x=rx-px;result.snap_y=ry-py;
        result.x=int(rx);result.y=int(ry);
        if(rx < -margin_x || rx > margin_x ||
                ry < -margin_y || ry > margin_y){result.reason=guard_bounds;return result;}
        result.reusable=true;
        result.reason=ready;
        return result;
    }
    template<class Rect> Rect needed(int width,int height,int margin_x,int margin_y)const {
        return {margin_x-x,margin_y-y,margin_x-x+width,margin_y-y+height};
    }
};
}}
