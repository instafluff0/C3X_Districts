#pragma once
#include "../c3x_renderer_api.h"
#include "../scene_projection.h"
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Natural-wonder name labels (docs/natural_wonder_rendering.md, Labels): copied
// presentation facts and C3X's draw_map_tile_text placement at the presented zoom.
// Text keeps its pixel size; only its anchor follows the view. Each wrapped
// on-screen occurrence keeps its own native anchor.
struct WonderLabels {
    struct Label {std::string name;float anchor_x,anchor_y;};
    std::vector<Label> labels;
    bool capture(c3x_renderer_frame_v1 const& frame){
        labels.clear();
        if(frame.tile_count>8192 || (frame.tile_count&&!frame.tiles) || frame.tile_width<=0 || frame.tile_height<=0 ||
           frame.target_width<=0 || frame.target_height<=0)return false;
        for(unsigned i=0;i<frame.tile_count;++i){auto const& tile=frame.tiles[i];
            if(!(tile.natural_wonder_flags&C3X_RENDERER_NATURAL_WONDER_NAME) || !tile.natural_wonder[0] ||
               !(tile.tile_flags&C3X_RENDERER_TILE_RENDER) || !(tile.tile_flags&C3X_RENDERER_TILE_VISIBILITY_KNOWN) ||
               !(tile.tile_flags&C3X_RENDERER_TILE_EXPLORED) || ((tile.tile_x+tile.tile_y)&1))continue;
            char name[sizeof(tile.natural_wonder)+1]={};std::memcpy(name,tile.natural_wonder,sizeof(tile.natural_wonder));
            labels.push_back({name,float(tile.anchor_x),float(tile.anchor_y)});
        }
        return true;
    }
    // The text box C3X centres a label in: draw_map_tile_text with the 88-high wonder
    // cell (24 above the tile) at the presented tile width. left/top are output pixels
    // (top is the text line's top, 10 above C3X's text_top); width is the centring span.
    struct Box {int left,top,width;};
    static Box box(c3x_renderer_frame_v1 const& frame,float zoom,float anchor_x,float anchor_y){
        SceneProjection projection(unsigned(frame.target_width),unsigned(frame.target_height),zoom);
        int screen_x=int(std::floor(projection.x(anchor_x)+.5f)),screen_y=int(std::floor(projection.y(anchor_y)+.5f));
        int tile_width=int(std::floor(float(frame.tile_width)*zoom+.5f));
        int screen_height=88*tile_width/128,y_offset=24*tile_width/128;
        int text_width=tile_width-tile_width/16;if(text_width<12)text_width=tile_width;
        int text_left=screen_x+(tile_width-text_width)/2;
        int text_top=screen_y-y_offset+screen_height+tile_width/32;
        return {text_left,text_top-10,text_width};
    }
};
}}
