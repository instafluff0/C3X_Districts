#pragma once
#include "../c3x_renderer_api.h"

namespace c3x_renderer::render_core {
// A translated camera can change the off-screen contributor set even when
// native tile flags and relative anchors match. Retained geometry owns this
// selection policy along with its occurrences.
struct ForegroundSelection {
    int width=0,height=0,tile_width=0,tile_height=0,ring=0,guard=0;
    bool pickup=false,offload=false;
    bool operator==(ForegroundSelection const& other)const {
        return width==other.width&&height==other.height&&tile_width==other.tile_width&&
            tile_height==other.tile_height&&ring==other.ring&&guard==other.guard&&
            pickup==other.pickup&&offload==other.offload;
    }
    bool selects(c3x_renderer_tile_v1 const& tile)const {
        auto x=c3x_renderer_i64(tile.anchor_x),y=c3x_renderer_i64(tile.anchor_y);
        auto inside=[&](int radius){auto mx=c3x_renderer_i64(tile_width)*radius,my=c3x_renderer_i64(tile_height)*radius;
            return x+tile_width>=-mx&&x<=width+mx&&y+tile_height>=-my&&y<=height+my;};
        bool guarded=guard&&inside(guard);
        if(!(tile.tile_flags&(C3X_RENDERER_TILE_RENDER|
            (pickup&&(!offload||guarded)?C3X_RENDERER_TILE_PREFETCH:0))))return false;
        return !pickup||(tile.tile_flags&C3X_RENDERER_TILE_RENDER)||inside(ring);
    }
    bool preserves(c3x_renderer_tile_v1 const& old,c3x_renderer_tile_v1 const& current)const {
        return selects(old)==selects(current);
    }
};
}
