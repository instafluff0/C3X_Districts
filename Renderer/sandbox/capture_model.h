#pragma once
#include <algorithm>
#include <climits>
#include <cmath>
#include <set>

// Caller-owned copied occurrences, using the existing production GPU camera
// preparation/readiness/adoption APIs. No separate offsets move prepared pixels.
struct SandboxCameraWitness {
    struct View {int x,y;float zoom;char const* name;int native_width=0;unsigned change=0;};
    std::vector<c3x_renderer_tile_v1> canonical;
    c3x_renderer_frame_v1 original;
    explicit SandboxCameraWitness(c3x_renderer_frame_v1 const& frame):original(frame) {
        for(unsigned i=0;i<frame.tile_count;++i){
            auto const& t=frame.tiles[i];
            if(t.tile_x>=0 && t.tile_x<frame.world_width_tiles)canonical.push_back(t);
        }
    }
    std::vector<c3x_renderer_tile_v1> capture(View view) const {
        std::vector<c3x_renderer_tile_v1> result;
        int tile_width=view.native_width?view.native_width:original.tile_width,tile_height=tile_width/2;
        int span=original.world_width_tiles*tile_width/2;
        auto occurrence=[&](auto tile,int wrap){
            tile.tile_x+=wrap*original.world_width_tiles;
            tile.anchor_x=original.target_width/2-tile_width/2+
                (tile.anchor_x+original.tile_width/2-original.target_width/2)*tile_width/original.tile_width+wrap*span+view.x;
            tile.anchor_y=original.target_height/2-tile_height/2+
                (tile.anchor_y+original.tile_height/2-original.target_height/2)*tile_height/original.tile_height+view.y;
            return tile;
        };
        int turn=span?int(std::floor(-double(view.x)/span)):0;
        int min_x=INT_MAX,max_x=INT_MIN,min_y=INT_MAX,max_y=INT_MIN;
        auto visible=[&](auto const& tile){return tile.anchor_x+tile_width>=0 &&
            tile.anchor_x<=original.target_width && tile.anchor_y+tile_height>=0 &&
            tile.anchor_y<=original.target_height;};
        // Synthetic native draw anchors, followed by the actual tile-coordinate
        // rings in capture_custom_renderer_topology. In-place projection zoom
        // does not change native tile dimensions or promote a native zoom ring.
        for(auto const& source:canonical)
            for(int wrap=original.world_wrap_x?turn-1:0;
                    wrap<=(original.world_wrap_x?turn+1:0);++wrap){
                auto tile=occurrence(source,wrap);
                if(!visible(tile))continue;
                min_x=(std::min)(min_x,tile.tile_x);max_x=(std::max)(max_x,tile.tile_x);
                min_y=(std::min)(min_y,tile.tile_y);max_y=(std::max)(max_y,tile.tile_y);
            }
        int appearance=original.world_topology_count?8:4;
        for(auto const& source:canonical)
            for(int wrap=original.world_wrap_x?turn-1:0;
                    wrap<=(original.world_wrap_x?turn+1:0);++wrap){
                auto tile=occurrence(source,wrap);
                if(tile.tile_x<min_x-12 || tile.tile_x>max_x+12 ||
                   tile.tile_y<min_y-12 || tile.tile_y>max_y+12)continue;
                tile.tile_flags&=C3X_RENDERER_TILE_VISIBILITY_BITS;
                if(visible(tile))tile.tile_flags|=C3X_RENDERER_TILE_RENDER;
                else {
                    tile.tile_flags|=C3X_RENDERER_TILE_TOPOLOGY_HALO;
                    if(tile.tile_x>=min_x-appearance && tile.tile_x<=max_x+appearance &&
                       tile.tile_y>=min_y-appearance && tile.tile_y<=max_y+appearance &&
                       (tile.tile_flags&C3X_RENDERER_TILE_EXPLORED))
                        tile.tile_flags|=C3X_RENDERER_TILE_PREFETCH;
                    if(!(tile.tile_flags&C3X_RENDERER_TILE_PREFETCH)){
                        auto full=tile;tile={};
                        tile.tile_x=full.tile_x;tile.tile_y=full.tile_y;
                        tile.anchor_x=full.anchor_x;tile.anchor_y=full.anchor_y;
                        tile.tile_flags=full.tile_flags;tile.terrain_type=full.terrain_type;
                        tile.real_terrain_type=full.real_terrain_type;tile.variant_seed=full.variant_seed;
                        tile.square_parts=full.square_parts;tile.terrain_overlays=full.terrain_overlays;
                        tile.territory_owner_id=full.territory_owner_id;tile.fog_status=full.fog_status;
                        tile.tile_visibility=full.tile_visibility;tile.visibility_mask=full.visibility_mask;
                        tile.river_code=full.river_code;tile.road_mask=full.road_mask;
                        tile.railroad_mask=full.railroad_mask;tile.route_style=full.route_style;
                        tile.resource_id=tile.resource_class=tile.tile_building_id=-1;
                        tile.city_id=tile.city_owner_id=tile.city_size=tile.city_culture_group=tile.city_era=-1;
                        tile.unit_type_id=tile.unit_owner_id=tile.unit_class=tile.unit_state=tile.unit_damage=tile.unit_direction=-1;
                    }
                }
                result.push_back(tile);
            }
        return result;
    }
};

// Real projection zoom on one already adopted native-role capture. Two
// 90-frame cycles cover both directions; the source clock is explicit.
inline float sandbox_capture_zoom(int frame) {
    int phase=frame%90;return phase<=45?1.f+.25f*float(phase)/45.f:
        1.25f-.25f*float(phase-45)/45.f;
}
