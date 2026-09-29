#pragma once
#include "../c3x_renderer_api.h"
#include <array>
#include <vector>

namespace c3x_renderer::render_core {
// Copied presentation facts. Grade zero is absent; terrain residency never owns
// these records. Each wrapped on-screen occurrence keeps its native anchor.
struct CitySiteOverlay {
    struct Tile {float x,y;unsigned grade,pad;};
    std::vector<Tile> tiles;
    static constexpr std::array<unsigned char,3> white={247,250,244};
    static constexpr std::array<unsigned char,3> green={19,105,52};
    static constexpr float fill_alpha=106.f/255.f,inset_pixels=1.f;
    bool capture(c3x_renderer_frame_v1 const& frame){
        tiles.clear();
        if(frame.tile_count>8192 || (frame.tile_count&&!frame.tiles) ||
           frame.tile_width<=0 || frame.tile_height<=0 ||
           frame.target_width<=0 || frame.target_height<=0)return false;
        for(unsigned i=0;i<frame.tile_count;++i){auto const& tile=frame.tiles[i];
            if(tile.city_site_grade>11)return false;
            if(!tile.city_site_grade || !(tile.tile_flags&C3X_RENDERER_TILE_RENDER) ||
               !(tile.tile_flags&C3X_RENDERER_TILE_VISIBILITY_KNOWN) ||
               !(tile.tile_flags&C3X_RENDERER_TILE_EXPLORED) ||
               ((tile.tile_x+tile.tile_y)&1))continue;
            tiles.push_back({float(tile.anchor_x),float(tile.anchor_y),tile.city_site_grade-1,0});
        }
        return true;
    }
    static std::array<float,3> color(unsigned grade){
        if(grade>10)return {};
        std::array<float,3> out{};
        // C3X commonly clusters legal sites in its top two grades. Spread the
        // light end so those neighboring grades remain readable at map opacity.
        float weight=float(grade)/10.f;weight=weight*weight*weight;
        for(unsigned i=0;i<3;++i)
            out[i]=(float(white[i])+(float(green[i])-white[i])*weight)/255.f;
        return out;
    }
};
}
