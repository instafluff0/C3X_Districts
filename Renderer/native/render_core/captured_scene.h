#pragma once
#include "../c3x_renderer_api.h"
#include "resident_content.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>

namespace c3x_renderer { namespace render_core {
// Worker-owned appearance, independent of capture buffers and projection. This
// bounded observed subset is not authority to draw outside the current capture.
class CapturedScene {
public:
    struct Record {
        c3x_renderer_tile_v1 appearance={};
        c3x_renderer_tile_v1 occurrence={};
        std::uint64_t revision=0, seen=0, semantic=0;
        ContentHandle compiled;
        int ground=-1, real=-1, relief=-1, surface=-1;
    };
    // The existing API admits at most 8192 occurrences. Retain up to that many
    // canonical records, evicting only absent records when a new view needs room.
    static constexpr std::size_t record_limit=8192;
private:
    std::unordered_map<std::uint64_t,Record> records;
    std::uint64_t epoch=0, serial=0;
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
        if(frame.tile_count>record_limit || (frame.tile_count && !frame.tiles))return false;
        if(width!=frame.world_width_tiles || height!=frame.world_height_tiles ||
           wrap_x!=(frame.world_wrap_x!=0) || wrap_y!=(frame.world_wrap_y!=0))records.clear();
        width=frame.world_width_tiles;height=frame.world_height_tiles;
        wrap_x=frame.world_wrap_x!=0;wrap_y=frame.world_wrap_y!=0;
        if(++epoch==0){records.clear();++epoch;}
        // Protect all current coordinates before eviction, including ones later
        // in traversal. Camera order must not cause retained content to churn.
        std::size_t missing=0;
        for(std::size_t i=0;i<frame.tile_count;++i){
            auto const& tile=frame.tiles[i];
            if(!(tile.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_TOPOLOGY_HALO)))continue;
            auto found=records.find(key(tile.tile_x,tile.tile_y));
            if(found==records.end())++missing;else found->second.seen=epoch;
        }
        if(records.size()+missing>record_limit){
            for(auto it=records.begin();it!=records.end();){
                if(it->second.seen!=epoch)it=records.erase(it);else ++it;
            }
        }
        records.reserve(std::min(record_limit,records.size()+missing));
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
        auto& record=found->second;
        record.occurrence=tile;record.seen=epoch;
        record.ground=ground;record.real=tile.real_terrain_type;
        record.relief=relief;record.surface=surface;record.semantic=semantic;
        if(tile.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)){
            auto next=content(tile);
            if(!record.revision || std::memcmp(&next,&record.appearance,sizeof(next))){
                record.appearance=next;record.revision=++serial;record.compiled={};
            }
        }
        return true;
    }
    void finish(){valid=true;}
    Record const* current(std::uint64_t id) const {
        if(!valid)return nullptr;
        auto found=records.find(id);
        return found!=records.end() && found->second.seen==epoch?&found->second:nullptr;
    }
    void attach(c3x_renderer_tile_v1 const& tile,ContentHandle handle) {
        if(!(tile.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)))return;
        auto found=records.find(key(tile.tile_x,tile.tile_y));
        if(!valid || found==records.end() || found->second.seen!=epoch || !found->second.revision)return;
        auto appearance=content(tile);
        if(!std::memcmp(&appearance,&found->second.appearance,sizeof(appearance)))found->second.compiled=handle;
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
            records.size()*(sizeof(std::pair<std::uint64_t const,Record>)+4*sizeof(void*));
    }
};
static_assert(sizeof(CapturedScene::Record)*CapturedScene::record_limit<16u*1024u*1024u,
              "captured appearance payload must remain bounded");
} }
