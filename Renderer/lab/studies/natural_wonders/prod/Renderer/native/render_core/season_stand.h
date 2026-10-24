#pragma once
#include <cstdint>
namespace c3x_renderer { namespace render_core {
// Seasons: the C3X seasons (0 Summer, 1 Autumn, 2 Winter, 3 Spring) a chunk draws
// in, one bit each; 0 draws in every season. Forest stands (seasons D4) and
// natural-wonder pines swap this way: the summer stand draws outside Winter, its
// precomputed Winter stand only in Winter. A season flip never recompiles.
constexpr unsigned season_stand_outside_winter=0b1011u,season_stand_winter=0b0100u;
inline bool season_stand_shown(unsigned stand,int season){
    return !stand || ((stand>>(unsigned(season)&3u))&1u)!=0;
}
// Whether going from season a to b shows or hides a chunk of any stand in use
// (bit m of used: some chunk carries stand m). Only then does the selection retire.
inline bool season_stands_differ(std::uint16_t used,int a,int b){
    for(unsigned m=1;m<16;++m)
        if(((used>>m)&1u) && season_stand_shown(m,a)!=season_stand_shown(m,b))return true;
    return false;
}
}}
