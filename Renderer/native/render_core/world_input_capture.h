#pragma once
#include "scene_publication.h"
#include <array>

namespace c3x_renderer { namespace render_core {
// Caller-thread paging state, not another world. Accepted records enter the
// existing publication journal and CapturedScene; a rejected page is retried.
class WorldInputCapture {
    std::array<c3x_renderer_tile_v1,128> storage{};
    c3x_renderer_camera_identity_v1 identity{};
    std::uint64_t configuration=0,requested_sequence=0;
    int width=0,height=0;
public:
    unsigned cursor=0;
    std::uint64_t passes=0,pages=0,records=0;
    void reset(){configuration=0;requested_sequence=0;cursor=0;passes=pages=records=0;}
    c3x_renderer_world_page_v1 page(ScenePublication::State const& state){
        auto const& f=state.metadata;
        if(configuration!=state.configuration || identity.map_epoch!=state.identity.map_epoch ||
           identity.viewer_epoch!=state.identity.viewer_epoch || width!=f.world_width_tiles || height!=f.world_height_tiles){
            reset();configuration=state.configuration;identity=state.identity;
            width=f.world_width_tiles;height=f.world_height_tiles;
        }
        c3x_renderer_world_page_v1 result{};
        result.struct_size=sizeof(result);result.first=cursor;result.capacity=unsigned(storage.size());
        result.identity=state.identity;result.frame=f;result.tiles=storage.data();
        result.frame.tiles=nullptr;result.frame.tile_count=0;
        result.frame.world_topology=state.topology?state.topology->data():nullptr;
        requested_sequence=state.sequence;
        return result;
    }
    bool accept(c3x_renderer_world_page_v1 const& page,ScenePublication& journal){
        auto state=journal.state();
        if(!state||state->sequence!=requested_sequence||state->configuration!=configuration)return false;
        auto const& expected=state->metadata;auto const& scope=state->identity;
        // The callback fills records/count only. It cannot change viewer/map
        // scope or replace a publication that arrived while its lease was open.
        if(page.identity.map_epoch!=scope.map_epoch||page.identity.viewer_epoch!=scope.viewer_epoch||
           page.identity.visibility_epoch!=scope.visibility_epoch||page.identity.scene_epoch!=scope.scene_epoch||
           page.frame.world_width_tiles!=width||page.frame.world_height_tiles!=height||
           page.frame.world_wrap_x!=expected.world_wrap_x||page.frame.world_wrap_y!=expected.world_wrap_y||
           page.frame.world_topology_revision!=expected.world_topology_revision||
           page.frame.world_topology_count!=expected.world_topology_count)return false;
        if(expected.world_topology_count&&(!page.frame.world_topology||!state->topology||
           state->topology->size()!=expected.world_topology_count||
           std::memcmp(page.frame.world_topology,state->topology->data(),state->topology->size()*sizeof(c3x_renderer_u32))))return false;
        if(width<=0 || height<=0 || (width&1) || width>2048 || height>2048 ||
           page.first!=cursor || page.tiles!=storage.data() || page.capacity!=storage.size())return false;
        auto total=unsigned(width)*unsigned(height)/2;
        if(cursor>=total || page.count!=std::min(unsigned(storage.size()),total-cursor))return false;
        for(unsigned i=0;i<page.count;++i){
            auto const& tile=page.tiles[i];auto n=cursor+i;
            int y=int(n/unsigned(width/2)),x=int(2*(n%unsigned(width/2)))+(y&1);
            if(tile.tile_x!=x || tile.tile_y!=y || !(tile.tile_flags&C3X_RENDERER_TILE_PREFETCH) ||
               (tile.tile_flags&C3X_RENDERER_TILE_RENDER))return false;
        }
        // Projection, time and all other metadata remain owned by the issued
        // publication, rather than accepting unrelated callback mutations.
        auto frame=expected;frame.world_topology=state->topology?state->topology->data():nullptr;
        frame.tiles=storage.data();frame.tile_count=page.count;
        if(!journal.capture(frame,page.identity))return false;
        cursor+=page.count;records+=page.count;++pages;
        if(cursor==total){cursor=0;++passes;}
        return true;
    }
};
}}
