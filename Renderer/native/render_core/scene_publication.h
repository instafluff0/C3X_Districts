#pragma once
#include "captured_scene.h"
#include <map>
#include <memory>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Caller capture and worker adoption are serialized by RendererWorker's gate.
// One coalesced journal owns accepted changes, independently of camera tickets.
// The existing CapturedScene remains the only persistent world/mesh owner.
class ScenePublication {
public:
    struct State {
        std::uint64_t sequence=0,configuration=0;
        c3x_renderer_camera_identity_v1 identity{};
        c3x_renderer_frame_v1 metadata{};
        std::shared_ptr<std::vector<c3x_renderer_u32> const> topology;
        // Only the last capture, not a second retained world. Exact equality
        // avoids allocating/adopting identical tile changes on every demand.
        std::shared_ptr<std::vector<c3x_renderer_tile_v1> const> tiles;
    };
private:
    using Tiles=std::map<std::uint64_t,c3x_renderer_tile_v1>;
    Tiles updates;
    std::shared_ptr<State const> latest;
    std::uint64_t sequence=0,configuration=1;
    bool pending=false,tile_pending=false;
    std::size_t limit;
    static constexpr std::size_t node_bytes=sizeof(c3x_renderer_tile_v1)+96;
    static void merge_tile(c3x_renderer_tile_v1& old,c3x_renderer_tile_v1 const& next){
        if(next.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH))old=next;
        else {
            // Halo visibility is authoritative; its omitted object fields are not.
            old.tile_flags=(old.tile_flags&~C3X_RENDERER_TILE_VISIBILITY_BITS)|(next.tile_flags&C3X_RENDERER_TILE_VISIBILITY_BITS);
            old.visibility_mask=next.visibility_mask;old.tile_visibility=next.tile_visibility;old.fog_status=next.fog_status;
        }
    }
    static bool same_scope(State const& prior,c3x_renderer_frame_v1 const& f,
                           c3x_renderer_camera_identity_v1 const& id,std::uint64_t config){
        auto const& p=prior.metadata;
        return prior.configuration==config && prior.identity.map_epoch==id.map_epoch && prior.identity.viewer_epoch==id.viewer_epoch &&
            p.world_width_tiles==f.world_width_tiles && p.world_height_tiles==f.world_height_tiles &&
            p.world_wrap_x==f.world_wrap_x && p.world_wrap_y==f.world_wrap_y;
    }
public:
    std::uint64_t accepted=0,rejected=0,applied=0,changed=0,tiles_reused=0;
    std::size_t peak=0;
    explicit ScenePublication(std::size_t budget=16u*1024u*1024u):limit(budget){}
    std::shared_ptr<State const> state()const{return latest;}
    bool ready()const{return pending;}
    std::size_t bytes()const{return sizeof(*this)+updates.size()*node_bytes+
        (latest?sizeof(State)+128+(latest->topology?latest->topology->capacity()*4:0)+
            (latest->tiles?latest->tiles->capacity()*sizeof(c3x_renderer_tile_v1):0):0);}
    void reset(){updates.clear();latest.reset();pending=tile_pending=false;if(configuration!=UINT64_MAX)++configuration;}
    bool capture(c3x_renderer_frame_v1 const& f,c3x_renderer_camera_identity_v1 const& id){
        auto reject=[&]{++rejected;return false;};
        if(f.tile_count>CapturedScene::occurrence_limit || (f.tile_count&&!f.tiles) ||
           f.world_topology_count>1024u*1024u || (f.world_topology_count&&!f.world_topology) ||
           sequence==UINT64_MAX || configuration==UINT64_MAX)return reject();
        // Bound both the admitted journal and this transactional staging copy.
        std::size_t staging=sizeof(State)+128+std::size_t(f.tile_count)*(node_bytes+sizeof(c3x_renderer_tile_v1))+
            std::size_t(f.world_topology_count)*4;
        if(staging>limit || bytes()>limit-staging)return reject();
        try {
            auto next=std::make_shared<State>();next->sequence=sequence+1;next->configuration=configuration;
            next->identity=id;next->metadata=f;next->metadata.tiles=nullptr;next->metadata.tile_count=0;
            next->metadata.world_topology=nullptr;
            bool scope=latest && same_scope(*latest,f,id,configuration);
            if(scope && latest->topology && latest->topology->size()==f.world_topology_count &&
               (!f.world_topology_count || !std::memcmp(latest->topology->data(),f.world_topology,f.world_topology_count*4)))
                next->topology=latest->topology;
            else {
                auto topology=std::make_shared<std::vector<c3x_renderer_u32>>();
                if(f.world_topology_count)topology->assign(f.world_topology,f.world_topology+f.world_topology_count);
                next->topology=std::move(topology);
            }
            auto canonical=[](int value,int extent,bool wrap){if(!wrap||extent<=0)return value;int r=value%extent;return r<0?r+extent:r;};
            auto normalize=[&](c3x_renderer_tile_v1 tile){
                tile.tile_x=canonical(tile.tile_x,f.world_width_tiles,f.world_wrap_x!=0);
                tile.tile_y=canonical(tile.tile_y,f.world_height_tiles,f.world_wrap_y!=0);
                tile.anchor_x=tile.anchor_y=0;return tile;
            };
            bool same_tiles=scope && latest->tiles && latest->tiles->size()==f.tile_count;
            for(unsigned i=0;same_tiles && i<f.tile_count;++i){auto tile=normalize(f.tiles[i]);
                same_tiles=std::memcmp(&tile,&(*latest->tiles)[i],sizeof(tile))==0;
            }
            Tiles staged;
            if(same_tiles)next->tiles=latest->tiles;
            else {
                auto tiles=std::make_shared<std::vector<c3x_renderer_tile_v1>>();tiles->reserve(f.tile_count);
                for(unsigned i=0;i<f.tile_count;++i){auto tile=normalize(f.tiles[i]);tiles->push_back(tile);
                    // A lightweight halo never removes a city/resource. Full prefetch
                    // is authoritative content, but still grants no draw eligibility.
                    if(!(tile.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_TOPOLOGY_HALO)))continue;
                    auto key=(std::uint64_t(std::uint32_t(tile.tile_x))<<32)|std::uint32_t(tile.tile_y);
                    auto found=staged.find(key);
                    if(found!=staged.end()){
                        auto a=CapturedScene::content(found->second),b=CapturedScene::content(tile);
                        if((found->second.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)) &&
                           (tile.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)) && std::memcmp(&a,&b,sizeof(a)))
                            return reject(); // contradictory full wrapped occurrences
                        merge_tile(found->second,tile);
                    }else staged.emplace(key,tile);
                }
                next->tiles=std::move(tiles);
            }
            std::size_t transient=bytes()+sizeof(State)+128+staged.size()*node_bytes+next->topology->capacity()*4+
                next->tiles->capacity()*sizeof(c3x_renderer_tile_v1);
            if(transient>limit)return reject();
            peak=std::max(peak,transient);
            // All allocations and validation finished. Node merge cannot allocate.
            if(!scope || (!tile_pending && !same_tiles))updates.clear();
            for(auto const& entry:staged){auto old=updates.find(entry.first);if(old!=updates.end())merge_tile(old->second,entry.second);}
            updates.merge(staged);latest=std::move(next);++sequence;++accepted;
            if(same_tiles)++tiles_reused;else tile_pending=true;
            pending=true;return true;
        }catch(...){return reject();}
    }
    template<class Scene> bool apply(Scene& scene,bool& content_changed){
        if(!pending)return true;
        // Retained-world admission may allocate. A partial adoption cannot
        // authorize output; keep the journal for retry instead of killing the
        // worker or spinning on the same failed allocation.
        try {
            content_changed=scene.publication_scope(latest->metadata,latest->identity,latest->configuration);
            // Keep the last adopted batch until different inputs arrive, so a
            // device reset can restore authority without allocating a new journal.
            if(content_changed)tile_pending=true;
            if(tile_pending)for(auto const& item:updates)if(!scene.publish(item.second,content_changed)){
                pending=false;return false;
            }
        }catch(...){pending=false;return false;}
        pending=tile_pending=false;++applied;if(content_changed)++changed;return true;
    }
};
}}
