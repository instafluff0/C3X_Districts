#pragma once
#include "captured_scene.h"
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
    static unsigned count(c3x_renderer_frame_v1 const& f){
        return unsigned((f.world_width_tiles+extent-1)/extent)*unsigned((f.world_height_tiles+extent-1)/extent);
    }
    bool build(CapturedScene const& scene,c3x_renderer_frame_v1 const& source,unsigned region){
        tiles.clear();selected.clear();frame=source;
        if(!scene.matches_world(source) || source.world_width_tiles<=0 || source.world_height_tiles<=0 ||
           source.world_width_tiles>2048 || source.world_height_tiles>2048 || region>=count(source))return false;
        int columns=(source.world_width_tiles+extent-1)/extent;
        int left=int(region%unsigned(columns))*extent,top=int(region/unsigned(columns))*extent;
        std::unordered_set<std::uint64_t> seen;
        // Canonical cores and wrapped halo anchors preserve the native lattice.
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
            tile.tile_x=int(std::uint32_t(key>>32));tile.tile_y=int(std::uint32_t(key));
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
