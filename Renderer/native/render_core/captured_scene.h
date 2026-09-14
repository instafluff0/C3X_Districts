#pragma once
#include "../c3x_renderer_api.h"
#include "resident_content.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>

namespace c3x_renderer { namespace render_core {
// Persistent worker-owned world instances. Observation eligibility is separate
// from identity and GPU residency; retained appearance never authorizes a draw.
class CapturedScene {
public:
    struct Record {
        c3x_renderer_tile_v1 appearance={};
        std::uint64_t revision=0;
        ContentHandle compiled;
        // Bounded projection variants borrow the same resident content owner.
        ContentHandle compiled_views[3]={};
    };
    struct Observation {
        c3x_renderer_tile_v1 occurrence={};
        std::uint64_t semantic=0,seen=0;
        int ground=-1, real=-1, relief=-1, surface=-1;
    };
    static constexpr std::size_t occurrence_limit=8192;
    // Includes conservative node/bucket overhead. Admission fails instead of
    // evicting identities; typical complete game maps fit well below this cap.
    static constexpr std::size_t budget=128u*1024u*1024u;
    static constexpr std::size_t record_limit=(budget-16u*1024u*1024u)/(sizeof(Record)+64);
private:
    std::unordered_map<std::uint64_t,Record> records;
    std::unordered_map<std::uint64_t,Observation> observations;
    std::uint64_t serial=0,epoch=0;
    int width=0,height=0;
    bool wrap_x=false,wrap_y=false,valid=false;
public:
    std::uint64_t key(int x,int y) const {
        auto canonical=[](int value,int extent,bool wraps){
            if(!wraps || extent<=0)return value;
            int result=value%extent;return result<0?result+extent:result;
        };
        return (std::uint64_t(std::uint32_t(canonical(x,width,wrap_x)))<<32) |
            std::uint32_t(canonical(y,height,wrap_y));
    }
    static c3x_renderer_tile_v1 content(c3x_renderer_tile_v1 tile) {
        tile.tile_x=tile.tile_y=tile.anchor_x=tile.anchor_y=0;
        tile.tile_flags=tile.visibility_mask=tile.tile_visibility=0;
        tile.fog_status=tile.territory_owner_id=0;
        // These selectors/labels belong to native overlays or deferred owners.
        // Match the static frame identity: exact population is not city size.
        tile.square_parts=tile.terrain_overlays=0;
        tile.tile_building_id=tile.city_population=0;
        std::memset(tile.city_owner,0,sizeof(tile.city_owner));
        std::memset(tile.city_civilization,0,sizeof(tile.city_civilization));
        std::memset(tile.city_era_name,0,sizeof(tile.city_era_name));
        // Units have an independent action/body owner, not tile appearance.
        tile.unit_type_id=tile.unit_owner_id=tile.unit_class=tile.unit_state=0;
        tile.unit_damage=tile.unit_direction=0;
        std::memset(tile.unit_owner,0,sizeof(tile.unit_owner));
        std::memset(tile.unit_civilization,0,sizeof(tile.unit_civilization));
        std::memset(tile.unit_era_name,0,sizeof(tile.unit_era_name));
        std::memset(tile.unit_type_name,0,sizeof(tile.unit_type_name));
        return tile;
    }
    bool begin(c3x_renderer_frame_v1 const& frame) {
        valid=false;
        if(frame.tile_count>occurrence_limit || (frame.tile_count && !frame.tiles))return false;
        if(width!=frame.world_width_tiles || height!=frame.world_height_tiles ||
           wrap_x!=(frame.world_wrap_x!=0) || wrap_y!=(frame.world_wrap_y!=0)){records.clear();observations.clear();}
        width=frame.world_width_tiles;height=frame.world_height_tiles;
        wrap_x=frame.world_wrap_x!=0;wrap_y=frame.world_wrap_y!=0;
        if(++epoch==0){observations.clear();++epoch;}
        // Reuse the bounded observation allocation, while world identities are
        // never evicted. Protect this entire capture before reclaiming old views.
        std::size_t missing=0;
        for(std::size_t i=0;i<frame.tile_count;++i){auto const& tile=frame.tiles[i];
            if(!(tile.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_TOPOLOGY_HALO)))continue;
            auto found=observations.find(key(tile.tile_x,tile.tile_y));
            if(found==observations.end())++missing;
            else {found->second.seen=epoch;found->second.occurrence.tile_flags=0;}
        }
        if(observations.size()+missing>occurrence_limit){
            for(auto it=observations.begin();it!=observations.end();){
                if(it->second.seen!=epoch)it=observations.erase(it);else ++it;
            }
        }
        if(observations.empty())observations.reserve(occurrence_limit);
        return true;
    }
    bool update(c3x_renderer_tile_v1 const& tile,int ground,int relief,int surface,
                std::uint64_t semantic) {
        auto id=key(tile.tile_x,tile.tile_y);
        auto found=records.find(id);
        if(found==records.end()){
            if(records.size()==record_limit)return false;
            found=records.try_emplace(id).first;
        }
        auto observed=observations.find(id);
        if(observed==observations.end()){
            if(observations.size()==occurrence_limit)return false;
            observed=observations.try_emplace(id).first;
        }
        auto& observation=observed->second;observation.seen=epoch;
        // Full appearance wins over a lightweight duplicate halo irrespective
        // of traversal order. Wrapped projection remains per input occurrence.
        bool full=(tile.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH))!=0;
        if(full || !(observation.occurrence.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH))) {
            observation.occurrence=tile;
            observation.ground=ground;observation.real=tile.real_terrain_type;
            observation.relief=relief;observation.surface=surface;observation.semantic=semantic;
        }
        if(full){
            auto& record=found->second;auto next=content(tile);
            if(!record.revision || std::memcmp(&next,&record.appearance,sizeof(next))){
                if(serial==~std::uint64_t(0))return false;
                record.appearance=next;record.revision=++serial;record.compiled={};
                for(auto& variant:record.compiled_views)variant={};
            }
        }
        return bytes()<=budget;
    }
    void finish(){valid=true;}
    Observation const* current(std::uint64_t id) const {
        if(!valid)return nullptr;
        auto found=observations.find(id);
        return found!=observations.end() && found->second.seen==epoch?&found->second:nullptr;
    }
    std::uint64_t appearance_revision(std::uint64_t id) const {
        auto observed=current(id);
        if(!observed || !(observed->occurrence.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)))return 0;
        auto record=retained(id);return record?record->revision:0;
    }
    void attach(c3x_renderer_tile_v1 const& tile,ContentHandle handle) {
        if(!(tile.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)))return;
        auto id=key(tile.tile_x,tile.tile_y);auto found=records.find(id);
        if(!current(id) || found==records.end() || !found->second.revision)return;
        auto appearance=content(tile);
        if(!std::memcmp(&appearance,&found->second.appearance,sizeof(appearance))){
            auto& record=found->second;record.compiled=handle;
            unsigned position=2;
            for(unsigned i=0;i<3;++i)if(record.compiled_views[i]==handle){position=i;break;}
            for(unsigned i=position;i>0;--i)record.compiled_views[i]=record.compiled_views[i-1];
            record.compiled_views[0]=handle;
        }
    }
    Record const* retained(std::uint64_t id) const {
        auto found=records.find(id);return found==records.end()?nullptr:&found->second;
    }
    std::size_t size() const{return records.size();}
    // Conservative tracked allocation estimate; allocator/driver residency is
    // reported separately. Record count and reserve requests stay within the cap;
    // the standard library determines the bucket count.
    std::size_t bytes() const {
        return sizeof(*this)+records.bucket_count()*sizeof(void*)+
            records.size()*(sizeof(std::pair<std::uint64_t const,Record>)+4*sizeof(void*))+
            observations.bucket_count()*sizeof(void*)+
            observations.size()*(sizeof(std::pair<std::uint64_t const,Observation>)+4*sizeof(void*));
    }
};
static_assert(sizeof(CapturedScene::Observation)*CapturedScene::occurrence_limit<16u*1024u*1024u,
              "current observation payload must remain bounded");
} }
