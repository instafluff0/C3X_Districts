#pragma once
#include "../c3x_renderer_api.h"
#include <algorithm>
#include <utility>
#include <vector>

namespace c3x_renderer { namespace render_core {

// Civ III's Unit::update_visibility visits 49 neighbors (three rings) about
// both positions. The rectangular bound is intentionally conservative; it is
// still at most 170 valid tiles for two distant positions, not a map sweep.
inline std::vector<std::pair<int,int>> world_move_footprint(
    c3x_renderer_frame_v1 const& frame,int old_x,int old_y,int new_x,int new_y,int radius=6){
    std::vector<std::pair<int,int>> tiles;
    if(frame.world_width_tiles<=0||frame.world_height_tiles<=0||
       (frame.world_width_tiles&1)||frame.world_width_tiles>2048||frame.world_height_tiles>2048||
       radius<0||radius>6)
        return tiles;
    auto add=[&](int cx,int cy){
        for(int dy=-radius;dy<=radius;++dy)for(int dx=-radius;dx<=radius;++dx){
            long long x=static_cast<long long>(cx)+dx,y=static_cast<long long>(cy)+dy;
            if(frame.world_wrap_x){x%=frame.world_width_tiles;if(x<0)x+=frame.world_width_tiles;}
            if(frame.world_wrap_y){y%=frame.world_height_tiles;if(y<0)y+=frame.world_height_tiles;}
            if(x<0||x>=frame.world_width_tiles||y<0||y>=frame.world_height_tiles||((x+y)&1))continue;
            tiles.emplace_back(int(x),int(y));
        }
    };
    add(old_x,old_y);add(new_x,new_y);
    std::sort(tiles.begin(),tiles.end());tiles.erase(std::unique(tiles.begin(),tiles.end()),tiles.end());
    return tiles;
}

}}
