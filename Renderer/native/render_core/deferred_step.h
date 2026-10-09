#pragma once
#include "world_window.h"
#include <algorithm>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Camera decoupling, stage 2c: a scroll step that keeps the resident world
// window and its content is drawn once, by its adoption on Civ III's next
// tick, instead of by a camera job and again by the adoption (performance
// review, section 38). Civ III draws its own layers for the step from the
// step's tile ownership before that draw, so ownership comes from the last
// drawn window: replacement flags follow tile content, never the camera.
struct DeferredStep {
    // Placement bits change with the camera, and so does which fields a
    // capture is authoritative for (city and overlay facts come with the
    // view, not the margin). The rest of a tile is content.
    static constexpr c3x_renderer_u32 placement=C3X_RENDERER_TILE_RENDER|
        C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_PREFETCH|
        C3X_RENDERER_TILE_CITY_BODY_KNOWN|C3X_RENDERER_TILE_NATIVE_OVERLAYS_KNOWN;
    // The last drawn window, kept to its window tiles, and its window-order
    // replacement flags by content: before the draw clears them for tiles
    // Civ III did not capture to draw (RENDER), which own nothing.
    struct Drawn {
        WorldWindow window;std::vector<c3x_renderer_u32> flags;
        unsigned device=0;std::uint64_t content=0;
        bool valid()const{return window.valid && flags.size()==window.tiles.size();}
        void remember(WorldWindow const& drawn,std::vector<c3x_renderer_u32> const& window_flags,
                      unsigned device_generation,std::uint64_t content_revision){
            window.valid=false;flags.clear();
            if(!drawn.valid || window_flags.size()<drawn.window_count)return;
            window.box=drawn.box;window.window_count=drawn.window_count;window.synthesized=drawn.synthesized;
            window.camera_x=drawn.camera_x;window.camera_y=drawn.camera_y;
            window.tiles.assign(drawn.tiles.begin(),drawn.tiles.begin()+drawn.window_count);
            window.native_index.clear();
            flags.assign(window_flags.begin(),window_flags.begin()+drawn.window_count);
            device=device_generation;content=content_revision;window.valid=true;
        }
    };
    // Civ III's ownership for `next`'s capture (capture order, native_count
    // entries), or false when the step changes the resident set or any window
    // tile's content. same_content compares two tiles with equal placement.
    template<class Same> static bool ownership(Drawn const& drawn,WorldWindow const& next,unsigned native_count,
            Same same_content,std::vector<c3x_renderer_u32>& out){
        if(!drawn.valid() || !next.valid || drawn.window.box!=next.box || drawn.window.window_count!=next.window_count)return false;
        for(unsigned i=0;i<next.window_count;++i){
            auto a=drawn.window.tiles[i],b=next.tiles[i];
            a.anchor_x=b.anchor_x;a.anchor_y=b.anchor_y;
            a.tile_flags&=~placement;b.tile_flags&=~placement;
            if(!same_content(a,b))return false;
        }
        std::vector<c3x_renderer_u32> flags(next.tiles.size(),0u);
        std::copy(drawn.flags.begin(),drawn.flags.begin()+next.window_count,flags.begin());
        next.native_flags(flags,native_count,out);
        return true;
    }
};
}}
