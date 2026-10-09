#pragma once
#include "../c3x_renderer_api.h"
#include <array>

namespace c3x_renderer::render_core {
// A translated camera can change the off-screen contributor set even when
// native tile flags and relative anchors match. Retained geometry owns this
// selection policy along with its occurrences.
struct ForegroundSelection {
    int width=0,height=0,tile_width=0,tile_height=0,ring=0,guard=0;
    bool pickup=false,offload=false;
    // World window (WorldWindow, stage 2a): the explored tiles with
    // appearance whose occurrence lies in `box` (anchor minus camera, in
    // block-aligned world pixels). The box, not the camera, is the identity:
    // camera steps inside one block window keep the same selection.
    bool windowed=false;
    std::array<long long,4> box{};
    long long camera_x=0,camera_y=0;
    bool operator==(ForegroundSelection const& other)const {
        return width==other.width&&height==other.height&&tile_width==other.tile_width&&
            tile_height==other.tile_height&&ring==other.ring&&guard==other.guard&&
            pickup==other.pickup&&offload==other.offload&&windowed==other.windowed&&box==other.box;
    }
    // The same selection rule, whatever the window's position: a block
    // crossing is an ordinary membership change for the incremental diff.
    bool same_rule(ForegroundSelection const& other)const {
        auto a=*this,b=other;a.box=b.box={};return a==b;
    }
    bool selects(c3x_renderer_tile_v1 const& tile)const {
        if(windowed){
            if(!(tile.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)))return false;
            if((tile.tile_flags&C3X_RENDERER_TILE_VISIBILITY_KNOWN) && !(tile.tile_flags&C3X_RENDERER_TILE_EXPLORED))return false;
            auto x=c3x_renderer_i64(tile.anchor_x)-camera_x,y=c3x_renderer_i64(tile.anchor_y)-camera_y;
            return x>=box[0]&&x<box[2]&&y>=box[1]&&y<box[3];
        }
        // Native anchors still describe fog coverage; body work requires
        // explored permission. Older lab inputs have no visibility contract.
        if((tile.tile_flags&C3X_RENDERER_TILE_VISIBILITY_KNOWN) &&
           !(tile.tile_flags&C3X_RENDERER_TILE_EXPLORED))return false;
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
