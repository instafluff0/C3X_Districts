#pragma once
#include <cstdint>
#include <map>
#include <set>
#include <utility>

namespace c3x_renderer { namespace render_core {
template<class Frame>
std::uint64_t wave_geometry_scope(Frame const& frame,std::uint64_t content,std::uint64_t device) {
    std::uint64_t scope=14695981039346656037ull;
    for(auto value:{std::uint64_t(frame.world_topology_revision),std::uint64_t(frame.world_width_tiles),
            std::uint64_t(frame.world_height_tiles),std::uint64_t(frame.world_wrap_x),std::uint64_t(frame.world_wrap_y),
            std::uint64_t(frame.tile_width),std::uint64_t(frame.tile_height),std::uint64_t(frame.target_width),
            std::uint64_t(frame.target_height),content,device})scope=(scope^value)*1099511628211ull;
    return scope;
}
// A canonical coast cell can appear more than once in a wide wrapped view.
// Preserve native screen anchors rather than deriving all cells from the first
// tile: its canonical coordinate jumps at the map seam while screen x does not.
template<class Frame>
std::map<std::pair<int,int>,std::set<std::pair<int,int>>> captured_wave_cells(Frame const& frame,unsigned render_flag) {
    std::map<std::pair<int,int>,std::set<std::pair<int,int>>> cells;
    auto canonical=[](int value,int extent,bool wraps){
        if(!wraps || extent<=0)return value;int result=value%extent;return result<0?result+extent:result;
    };
    for(unsigned i=0;i<frame.tile_count;++i){auto const& tile=frame.tiles[i];
        if(!(tile.tile_flags&render_flag))continue;
        int c=(tile.tile_x+tile.tile_y)/2,r=(tile.tile_x-tile.tile_y)/2;
        for(int y=r-1;y<=r+1;++y)for(int x=c-1;x<=c+1;++x){
            int tx=canonical(x+y,frame.world_width_tiles,frame.world_wrap_x!=0);
            int ty=canonical(x-y,frame.world_height_tiles,frame.world_wrap_y!=0);
            cells[{(tx+ty)/2,(tx-ty)/2}].emplace(
                tile.anchor_x+(x+y-tile.tile_x)*(frame.tile_width/2),
                tile.anchor_y+(x-y-tile.tile_y)*(frame.tile_height/2));
        }
    }
    return cells;
}
} }
