#pragma once
#include "../c3x_renderer_api.h"
#include <algorithm>
#include <utility>
#include <vector>

namespace c3x_renderer { namespace render_core {

inline std::vector<std::pair<int,int>> world_bounded_footprint(
    c3x_renderer_frame_v1 const& frame,int old_x,int old_y,int new_x,int new_y,int radius,bool sight){
    std::vector<std::pair<int,int>> tiles;
    if(frame.world_width_tiles<=0||frame.world_height_tiles<=0||
       (frame.world_width_tiles&1)||frame.world_width_tiles>2048||frame.world_height_tiles>2048||
       radius<0||radius>(sight?7:2))
        return tiles;
    auto add=[&](int cx,int cy){
        for(int v=-radius;v<=radius;++v)for(int u=-radius;u<=radius;++u){
            // Native neighbor_index_to_diff's (2R+1)^2 prefix is the
            // abstract square transformed to Civ III's raw isometric grid.
            int dx=sight?u-v:u,dy=sight?u+v:v;
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

// Unit::update_visibility visits the configured sight rings at both positions.
// Vanilla R=3 visits 49 tiles; C3X's byte-sized iterator clamps R<=7, hence
// at most 450 distinct records in their union, still paged in groups of 128.
inline std::vector<std::pair<int,int>> world_move_footprint(
    c3x_renderer_frame_v1 const& frame,int old_x,int old_y,int new_x,int new_y,int sight_rings=3){
    return world_bounded_footprint(frame,old_x,old_y,new_x,new_y,sight_rings,true);
}

// Physical appearance/connectivity changes retain their separate 13-tile halo.
inline std::vector<std::pair<int,int>> world_change_footprint(
    c3x_renderer_frame_v1 const& frame,int x,int y){
    return world_bounded_footprint(frame,x,y,x,y,2,false);
}

}}
