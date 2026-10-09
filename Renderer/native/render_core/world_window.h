#pragma once
#include "../c3x_renderer_api.h"
#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace c3x_renderer { namespace render_core {
// The resident world selected by a window anchored to world blocks, not by
// the tiles Civ III captured around the current camera (camera decoupling
// design, stage 2a). The capture reaches 8 tile coordinates past the view
// (512 px in x, 256 px in y at 1x), so a selection taken from it changed at
// every camera step: casters and contributors entered and left, shadow pages
// and static rasters were re-proven and repaired, and refinement restarted
// (performance review, sections 25-26). The window reaches at least the
// static guard band plus shadow reach past the view and is snapped outward to
// 8x8-coordinate blocks, so it changes only when the camera crosses a block.
// Tiles in it that the capture lacks come from the renderer's retained world:
// the last full capture of the tile while its content is current, otherwise
// the world record. Captured tiles are used as captured.
//
// Anchors are linear in occurrence coordinates (anchor = coordinate * half
// tile + camera), which the capture uses for every tile; a frame that breaks
// that is left unchanged. Window tiles come first in row-major occurrence
// order, so their order is stable across steps; the capture's other tiles
// follow unchanged.
struct WorldWindow {
    static constexpr int block=8,reach_x=8,reach_y=12; // tile coordinates
    bool valid=false;
    std::int64_t camera_x=0,camera_y=0;
    // Occurrence-world pixels, [left,right) x [top,bottom), block-aligned.
    std::array<std::int64_t,4> box{};
    std::vector<c3x_renderer_tile_v1> tiles;
    unsigned window_count=0,synthesized=0;
    // Index in `tiles` of each native tile, or ~0u if it is not drawn from.
    std::vector<unsigned> native_index;
    static std::int64_t floor_div(std::int64_t a,std::int64_t b){return a>=0?a/b:-((-a+b-1)/b);}
    static std::int64_t ceil_div(std::int64_t a,std::int64_t b){return -floor_div(-a,b);}
    static std::uint64_t occurrence(std::int64_t x,std::int64_t y){
        return (std::uint64_t(std::uint32_t(std::int32_t(x)))<<32)|std::uint32_t(std::int32_t(y));
    }
    static bool explored(unsigned flags){
        return !(flags&C3X_RENDERER_TILE_VISIBILITY_KNOWN) || (flags&C3X_RENDERER_TILE_EXPLORED);
    }
    static bool appearance(unsigned flags){
        return (flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH))!=0;
    }
    // The window for a frame's camera, without building it: the camera from
    // the first RENDER tile and the block-snapped box (occurrence pixels).
    static bool locate(c3x_renderer_frame_v1 const& frame,std::int64_t& camera_x,std::int64_t& camera_y,std::array<std::int64_t,4>& box){
        std::int64_t hw=frame.tile_width/2,hh=frame.tile_height/2;
        if(hw<=0 || hh<=0 || !frame.tile_count || !frame.tiles)return false;
        c3x_renderer_tile_v1 const* reference=nullptr;
        for(unsigned i=0;i<frame.tile_count && !reference;++i)
            if(frame.tiles[i].tile_flags&C3X_RENDERER_TILE_RENDER)reference=frame.tiles+i;
        if(!reference)return false;
        camera_x=std::int64_t(reference->anchor_x)-std::int64_t(reference->tile_x)*hw;
        camera_y=std::int64_t(reference->anchor_y)-std::int64_t(reference->tile_y)*hh;
        std::int64_t const bx=block*hw,by=block*hh;
        // Snap the origin and keep a fixed extent that covers any phase, so
        // the window changes once per block of camera motion, not once per
        // edge crossing.
        std::int64_t const left=floor_div(-camera_x-reach_x*hw,bx)*bx,top=floor_div(-camera_y-reach_y*hh,by)*by;
        box={left,top,left+ceil_div(frame.target_width+2*reach_x*hw+bx,bx)*bx,top+ceil_div(frame.target_height+2*reach_y*hh+by,by)*by};
        return true;
    }
    // retained(x, y, tile): fills the retained world's tile for canonical
    // coordinates (any anchors and placement flags), or returns false.
    template<class Retained> bool build(c3x_renderer_frame_v1 const& frame,Retained retained){
        valid=false;tiles.clear();native_index.clear();window_count=synthesized=0;
        std::int64_t hw=frame.tile_width/2,hh=frame.tile_height/2;
        if(!locate(frame,camera_x,camera_y,box))return false;
        std::int64_t const width=frame.world_width_tiles,height=frame.world_height_tiles;
        bool const wrap_x=frame.world_wrap_x && width>0,wrap_y=frame.world_wrap_y && height>0;
        auto canonical=[&](std::int64_t value,std::int64_t size,bool wrap,std::int64_t& out){
            if(wrap){out=value%size;if(out<0)out+=size;return true;}
            out=value;return size<=0 || (value>=0 && value<size);
        };
        // Every native tile must sit on the linear lattice (wrapped copies
        // differ by whole world widths); otherwise keep the native frame.
        std::unordered_map<std::uint64_t,unsigned> native;native.reserve(frame.tile_count*2);
        for(unsigned i=0;i<frame.tile_count;++i){auto const& tile=frame.tiles[i];
            std::int64_t dx=std::int64_t(tile.anchor_x)-camera_x,dy=std::int64_t(tile.anchor_y)-camera_y;
            if(dx%hw || dy%hh)return false;
            std::int64_t ox=dx/hw,oy=dy/hh,cx=0,cy=0;
            if(!canonical(ox,width,wrap_x,cx) || !canonical(oy,height,wrap_y,cy) || cx!=tile.tile_x || cy!=tile.tile_y)return false;
            if(!native.emplace(occurrence(ox,oy),i).second)return false;
        }
        std::vector<bool> used(frame.tile_count,false);
        tiles.reserve(std::size_t((box[2]-box[0])/hw)*std::size_t((box[3]-box[1])/hh)/2+frame.tile_count);
        for(std::int64_t oy=box[1]/hh;oy<box[3]/hh;++oy)for(std::int64_t ox=box[0]/hw;ox<box[2]/hw;++ox){
            if((ox+oy)&1)continue;
            auto found=native.find(occurrence(ox,oy));
            if(found!=native.end() && appearance(frame.tiles[found->second].tile_flags)){
                if(explored(frame.tiles[found->second].tile_flags)){
                    used[found->second]=true;native_index.push_back(found->second);tiles.push_back(frame.tiles[found->second]);}
                continue;
            }
            std::int64_t cx=0,cy=0;
            if(!canonical(ox,width,wrap_x,cx) || !canonical(oy,height,wrap_y,cy))continue;
            c3x_renderer_tile_v1 tile{};
            if(!retained(int(cx),int(cy),tile) || !explored(tile.tile_flags))continue;
            tile.tile_x=int(cx);tile.tile_y=int(cy);
            tile.anchor_x=decltype(tile.anchor_x)(ox*hw+camera_x);tile.anchor_y=decltype(tile.anchor_y)(oy*hh+camera_y);
            tile.tile_flags=(tile.tile_flags&~(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_TOPOLOGY_HALO))|
                C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_TOPOLOGY_HALO;
            if(found!=native.end()){used[found->second]=true;native_index.push_back(found->second);}
            else native_index.push_back(~0u);
            tiles.push_back(tile);++synthesized;
        }
        window_count=unsigned(tiles.size());
        for(unsigned i=0;i<frame.tile_count;++i)if(!used[i]){native_index.push_back(i);tiles.push_back(frame.tiles[i]);}
        valid=window_count!=0;
        return valid;
    }
    // Native-order replacement flags from the window frame's. Civ III owns
    // replacement only for the tiles it captured to draw (RENDER); window
    // tiles drawn from its prefetch margin or the retained world report none.
    void native_flags(std::vector<c3x_renderer_u32> const& window_flags,unsigned native_count,std::vector<c3x_renderer_u32>& out)const{
        out.assign(native_count,0u);
        for(std::size_t i=0;i<window_flags.size() && i<native_index.size();++i)
            if(native_index[i]<native_count && (tiles[i].tile_flags&C3X_RENDERER_TILE_RENDER))out[native_index[i]]=window_flags[i];
    }
    void native_indices(std::vector<c3x_renderer_u32> const& window_indices,unsigned native_count,std::vector<c3x_renderer_u32>& out)const{
        out.clear();
        for(auto index:window_indices)if(index<native_index.size() && native_index[index]<native_count)out.push_back(native_index[index]);
    }
};
}}
