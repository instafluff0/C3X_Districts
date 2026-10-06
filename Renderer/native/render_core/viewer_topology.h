#pragma once
#include <cstddef>
#include <cstdint>
#include <exception>
#include <vector>
#include "../c3x_renderer_api.h"

namespace c3x_renderer { namespace render_core {
// World topology as the viewing civ knows it. Unexplored land keeps its biome
// and river code but loses its visible category (hills, mountains, forest,
// marsh, volcano) and effect bit, so hidden relief cannot raise, light or
// shadow revealed neighbours, nor disclose itself through them. Water stays
// water so coastlines are stable. Tiles without a known world input (lab
// inputs, unpaged tiles) stay as captured.
//
// Every unexplored tile also carries hidden_bit (and loses its effect bit), so
// a revealed neighbour can tell where no terrain will be drawn.
//
// The revision equals the captured one while nothing is hidden; otherwise it
// gains a serial above bit 40, so every masked state has its own revision.
struct ViewerTopology {
    static constexpr std::uint32_t hidden_bit=1u<<25;
    static bool hidden(std::uint32_t value){return value!=0xffffffffu && (value&hidden_bit)!=0;}
    std::vector<std::uint32_t> values;
    std::int64_t source=-1,revision=-1;
    std::uint64_t inputs=~std::uint64_t(0),serial=0;
    std::size_t hidden_tiles=0;
    bool masked=false;
    static std::uint32_t hide(std::uint32_t value){
        auto biome=value&255u;
        return (biome>=11 && biome<=13?value&~(1u<<24):(value&0x00ff00ffu)|(biome<<8))|hidden_bit;
    }
    // visit(mark) calls mark(tile_x,tile_y,tile_flags) for every world input.
    // Returns the values to use; `changed` reports a new revision.
    template<class Visit>
    std::uint32_t const* update(std::uint32_t const* raw,std::size_t count,int width,int height,
            std::int64_t raw_revision,std::uint64_t input_sequence,Visit visit,bool* changed=nullptr){
        if(changed)*changed=false;
        bool current=raw_revision==source && values.size()==count;
        if(current && input_sequence==inputs)return masked?values.data():raw;
        std::vector<std::uint32_t> next(raw,raw+count);
        std::size_t hidden=0;
        try{
            visit([&](int x,int y,unsigned flags){
                if((flags&(C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED))!=C3X_RENDERER_TILE_VISIBILITY_KNOWN)return;
                if(x<0 || y<0 || x>=width || y>=height || ((x+y)&1))return;
                auto index=(std::size_t(y)*std::size_t(width)+std::size_t(x))/2;
                if(index>=count)return;
                auto value=hide(next[index]);
                if(value!=next[index]){next[index]=value;++hidden;}
            });
        }catch(std::exception const&){
            // Keep the last mask for this capture rather than briefly
            // revealing (and then recompiling) every hidden neighbour.
            if(current)return masked?values.data():raw;
            throw;
        }
        bool next_masked=hidden!=0;
        if(!current || next_masked!=masked || (next_masked && next!=values)){
            if(changed)*changed=true;
            if(next_masked)++serial;
            revision=next_masked?raw_revision+std::int64_t(serial<<40):raw_revision;
        }
        values.swap(next);source=raw_revision;inputs=input_sequence;masked=next_masked;hidden_tiles=hidden;
        return masked?values.data():raw;
    }
};
} }
