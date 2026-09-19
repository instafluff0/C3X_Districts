#pragma once
#include "../c3x_renderer_api.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Captured map effects share one immutable input contract. Resource identities,
// visibility, native occurrences, environment and capture time travel together.
// Existing UnitInstances selections own the corresponding revisioned unit/action
// records. Neither owner grants visibility from resident assets or old geometry.
class DynamicSceneInputs {
    struct Budget {
        std::atomic<std::size_t> bytes{0};
        std::atomic<std::uint64_t> epoch{1};
    };
public:
    class Map {
        friend class DynamicSceneInputs;
        std::shared_ptr<Budget> budget;
        std::uint64_t epoch;
        c3x_renderer_frame_v1 captured;
        c3x_renderer_camera_identity_v1 captured_identity;
        std::vector<c3x_renderer_tile_v1> tiles;
        std::vector<c3x_renderer_u32> topology;
        std::size_t charged=0;
        Map(std::shared_ptr<Budget> owner,c3x_renderer_frame_v1 const& input,c3x_renderer_camera_identity_v1 identity):
            budget(std::move(owner)),epoch(budget->epoch.load()),captured(input),captured_identity(identity) {
            if(input.tile_count)tiles.assign(input.tiles,input.tiles+input.tile_count);
            if(input.world_topology_count)topology.assign(input.world_topology,input.world_topology+input.world_topology_count);
            captured.tiles=nullptr;captured.world_topology=nullptr;
        }
    public:
        Map(Map const&)=delete;Map& operator=(Map const&)=delete;
        ~Map(){budget->bytes.fetch_sub(charged);}
        bool valid()const{return epoch==budget->epoch.load();}
        c3x_renderer_frame_v1 frame()const{
            auto result=captured;result.tiles=tiles.empty()?nullptr:tiles.data();
            result.world_topology=topology.empty()?nullptr:topology.data();return result;
        }
        // A sampled frame borrows only this immutable owner. Native action
        // cursors never pass through this clock; only map effects use elapsed time.
        bool sample(long long ticks,long long frequency,long long origin,c3x_renderer_frame_v1& output)const{
            if(!valid() || ticks<0 || origin<0 || frequency<=0)return false;
            long double elapsed=static_cast<long double>(ticks>origin?ticks-origin:0)*captured.presentation_frequency/frequency;
            if(elapsed!=0 && elapsed>=static_cast<long double>(std::numeric_limits<long long>::max()-captured.presentation_time_ticks))return false;
            output=frame();output.presentation_time_ticks+=static_cast<long long>(elapsed);return true;
        }
        c3x_renderer_camera_identity_v1 identity()const{return captured_identity;}
        std::size_t bytes()const{return charged;}
    };
private:
    std::shared_ptr<Budget> budget=std::make_shared<Budget>();
    std::size_t limit;
public:
    std::size_t peak=0;std::uint64_t captures=0,rejected=0;
    explicit DynamicSceneInputs(std::size_t maximum=16u*1024u*1024u):limit(maximum){}
    std::size_t bytes()const{return budget->bytes.load();}
    // Caller gate serializes captures/reset; retained destruction may be remote.
    // Old retained fronts stay charged until the compositor releases them.
    void invalidate(){auto epoch=budget->epoch.load();if(epoch!=UINT64_MAX)budget->epoch.store(epoch+1);}
    std::shared_ptr<Map const> capture(c3x_renderer_frame_v1 const& input,c3x_renderer_camera_identity_v1 identity){
        auto reject=[&]()->std::shared_ptr<Map const>{++rejected;return {};};
        if(input.api_version!=C3X_RENDERER_API_VERSION || input.struct_size!=sizeof(input) ||
           input.tile_count>8192 || (input.tile_count&&!input.tiles) ||
           input.world_topology_count>1024u*1024u || (input.world_topology_count&&!input.world_topology) ||
           input.presentation_time_ticks<0 || input.presentation_frequency<=0 || budget->epoch.load()==UINT64_MAX)return reject();
        // Include object/control storage, not just the copied arrays. Preflight
        // bounds the one in-flight copy as well as admitted retained records.
        std::size_t needed=sizeof(Map)+64+std::size_t(input.tile_count)*sizeof(c3x_renderer_tile_v1)+std::size_t(input.world_topology_count)*4;
        if(needed>limit || bytes()>limit-needed)return reject();
        auto result=std::shared_ptr<Map>(new Map(budget,input,identity));
        auto actual=sizeof(Map)+64+result->tiles.capacity()*sizeof(c3x_renderer_tile_v1)+result->topology.capacity()*4;
        if(actual>limit || bytes()>limit-actual)return reject();
        result->charged=actual;budget->bytes.fetch_add(actual);peak=std::max(peak,bytes());++captures;
        return result;
    }
};
}}
