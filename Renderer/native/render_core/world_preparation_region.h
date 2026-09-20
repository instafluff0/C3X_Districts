#pragma once
#include "captured_scene.h"
#include <array>
#include <vector>
#include <unordered_set>

namespace c3x_renderer { namespace render_core {
// An immutable compiler lease assembled from the existing world. Its halo is
// input only; this does not produce a viewport or grant display eligibility.
struct WorldPreparationRegion {
    static constexpr int extent=8,halo=12;
    c3x_renderer_frame_v1 frame{};
    std::vector<c3x_renderer_tile_v1> tiles;
    std::vector<unsigned> selected;
    static int bands(int cells,int pixels,int tile,bool wrap){
        if(!wrap || tile<=0)return 0;
        return std::min((cells+extent-1)/extent,((pixels+tile-1)/tile+halo+extent-1)/extent);
    }
    static unsigned count(c3x_renderer_frame_v1 const& f){
        return unsigned((f.world_width_tiles+extent-1)/extent+2*bands(f.world_width_tiles,f.target_width,f.tile_width,f.world_wrap_x!=0))*
            unsigned((f.world_height_tiles+extent-1)/extent+2*bands(f.world_height_tiles,f.target_height,f.tile_height,f.world_wrap_y!=0));
    }
    static std::array<int,2> core(unsigned index,int cells,int band){
        unsigned columns=unsigned((cells+extent-1)/extent);
        if(index<columns)return {int(index)*extent,0};
        index-=columns;
        if(index<unsigned(band))return {int(index)*extent,cells};
        return {int(columns-unsigned(band)+index-unsigned(band))*extent,-cells};
    }
    bool build(CapturedScene const& scene,c3x_renderer_frame_v1 const& source,unsigned region){
        tiles.clear();selected.clear();frame=source;
        if(!scene.matches_world(source) || source.world_width_tiles<=0 || source.world_height_tiles<=0 ||
           source.world_width_tiles>2048 || source.world_height_tiles>2048 || region>=count(source))return false;
        int horizontal=bands(source.world_width_tiles,source.target_width,source.tile_width,source.world_wrap_x!=0);
        int vertical=bands(source.world_height_tiles,source.target_height,source.tile_height,source.world_wrap_y!=0);
        unsigned columns=unsigned((source.world_width_tiles+extent-1)/extent+2*horizontal);
        auto x_core=core(region%columns,source.world_width_tiles,horizontal);
        auto y_core=core(region/columns,source.world_height_tiles,vertical);
        int left=x_core[0],top=y_core[0];
        std::unordered_set<std::uint64_t> seen;
        // Canonical cores plus bounded edge occurrences cover native wrapping
        // independently of navigation history. Halo coordinates stay unwrapped,
        // matching the authoritative occurrence lattice around each core.
        // Ground and rigid geometry use world units; anchors only supply relative
        // neighbor placement, with the native viewport's projection/detail inputs.
        for(int y=top-halo;y<top+extent+halo;++y)for(int x=left-halo;x<left+extent+halo;++x){
            if((x+y)&1)continue;
            if((!source.world_wrap_x && (x<0 || x>=source.world_width_tiles)) ||
               (!source.world_wrap_y && (y<0 || y>=source.world_height_tiles)))continue;
            auto key=scene.key(x,y);
            // Tiny wrapped worlds use the occurrence closest to the core, not
            // an arbitrary duplicate. They are handled by the demand path until
            // the whole-world lease can prove that projection unambiguously.
            if(!seen.insert(key).second)return false;
            auto record=scene.retained(key);
            if(!record || !record->authoritative)return false;
            auto tile=record->appearance;
            tile.tile_x=x+x_core[1];tile.tile_y=y+y_core[1];
            tile.anchor_x=(x-left)*source.tile_width/2;
            tile.anchor_y=(y-top)*source.tile_height/2;
            tile.visibility_mask=record->visibility_mask;tile.tile_visibility=record->tile_visibility;
            tile.fog_status=record->fog_status;
            tile.tile_flags=record->visibility_flags|C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_PREFETCH;
            if(x>=left && x<left+extent && y>=top && y<top+extent &&
               x<source.world_width_tiles && y<source.world_height_tiles)selected.push_back(unsigned(tiles.size()));
            tiles.push_back(tile);
        }
        frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
        return !selected.empty() && tiles.size()<=CapturedScene::occurrence_limit;
    }
};
}}
