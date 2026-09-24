#pragma once
#include "../c3x_renderer_api.h"
#include <cmath>

namespace c3x_renderer { namespace render_core {
// b10: bounded, non-drifting ocean motion and a camera-relative reflection eye.
// The eye affects material optics only; Civ III's orthographic anchors are unchanged.
struct WaterMaterialFrame {
    float time=0, drift[3]={};
    float camera[4]={}; // natural-world center XY, wrapped raw-world periods XY
};
inline bool water_scene_tile(c3x_renderer_tile_v1 const& tile,bool visibility_pass){
    // Civ III m49 is the underlying biome; m50 is the visible square category.
    // A coast can have a land-like m49 value even though its rendered surface
    // is water. Fogged tiles must not keep the material clock running.
    return (tile.tile_flags&C3X_RENDERER_TILE_RENDER) &&
        (!visibility_pass || (tile.tile_flags&C3X_RENDERER_TILE_VISIBLE)) &&
        (tile.real_terrain_type>=11 || tile.terrain_type>=11 || (tile.river_code&170u));
}
inline WaterMaterialFrame water_material_frame(c3x_renderer_frame_v1 const& frame) {
    WaterMaterialFrame result;
    result.time=float(double(frame.presentation_time_ticks)/double(frame.presentation_frequency>0?frame.presentation_frequency:1));
    result.drift[0]=std::sin(result.time*.21f)*.11f;
    result.drift[1]=(std::sin(result.time*.17f+1.2f)-std::sin(1.2f))*.09f;
    result.drift[2]=(std::sin(result.time*.13f+2.4f)-std::sin(2.4f))*.10f;
    if(frame.tile_count && frame.tiles && frame.tile_width>0 && frame.tile_height>0){
        auto const* tile=frame.tiles;
        for(unsigned i=0;i<frame.tile_count;++i)if(frame.tiles[i].tile_flags&C3X_RENDERER_TILE_RENDER){tile=frame.tiles+i;break;}
        // Tile centers are (c+.5,r+.5). Invert the actual 2:1 screen basis.
        float rawx=float(tile->tile_x)+1+(frame.target_width-2.f*tile->anchor_x-frame.tile_width)/frame.tile_width;
        float rawy=float(tile->tile_y)+(frame.target_height-2.f*tile->anchor_y-frame.tile_height)/frame.tile_height;
        result.camera[0]=(rawx+rawy)*.5f;result.camera[1]=(rawx-rawy)*.5f;
    }
    result.camera[2]=frame.world_wrap_x?float(frame.world_width_tiles):0;
    result.camera[3]=frame.world_wrap_y?float(frame.world_height_tiles):0;
    return result;
}
} }
