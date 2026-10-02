#pragma once
#include "../c3x_renderer_api.h"
#include "resident_content.h"
#include "raster_dependency_revisions.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace c3x_renderer { namespace render_core {
// Persistent worker-owned world instances. Observation eligibility is separate
// from identity and GPU residency; retained appearance never authorizes a draw.
class CapturedScene {
public:
    struct Record {
        c3x_renderer_tile_v1 appearance={};
        std::uint64_t revision=0;
        bool authoritative=false;
        // Per-field facts share the existing appearance owner. PREFETCH here
        // remembers a standalone full copy; it never grants display eligibility.
        unsigned partial_flags=0;
        std::uint64_t semantic=0,semantic_revision=0;
        std::uint64_t visibility_revision=0;
        unsigned visibility_flags=0,visibility_mask=0,tile_visibility=0;
        int fog_status=0;
        ContentHandle compiled;
        // Bounded projection variants borrow the same resident content owner.
        ContentHandle compiled_views[3]={};
    };
    struct Observation {
        c3x_renderer_tile_v1 occurrence={};
        std::uint64_t semantic=0,seen=0;
        // Last completed native-observation compatibility, separate from the
        // immutable world lease and native anchors.
        std::uint64_t raster_appearance=0;
        int ground=-1, real=-1, relief=-1, surface=-1;
    };
private:
    struct WorldMemory {std::atomic<std::size_t> bytes{0};};
    struct WorldBlock {
        std::unordered_map<std::uint64_t,Observation> values;
        std::shared_ptr<WorldMemory> memory;
        std::size_t charged=0;
        explicit WorldBlock(std::shared_ptr<WorldMemory> account,WorldBlock const* prior=nullptr):memory(std::move(account)){
            if(prior)values=prior->values;else values.reserve(32);
            charge();
        }
        void charge(){auto next=sizeof(*this)+values.bucket_count()*sizeof(void*)+
            values.size()*(sizeof(Observation)+64);memory->bytes.fetch_add(next-charged);charged=next;}
        ~WorldBlock(){memory->bytes.fetch_sub(charged);}
    };
    static std::uint64_t world_block(std::uint64_t id){
        return (std::uint64_t(std::uint32_t(id>>32)/8)<<32)|(std::uint32_t(id)/8);
    }
public:
    // Immutable canonical world inputs. An 8x8 block changes only after a
    // copied game publication; cameras and resident GPU bindings are separate.
    class WorldSnapshot {
        friend class CapturedScene;
        using Blocks=std::unordered_map<std::uint64_t,std::shared_ptr<WorldBlock const>>;
        Blocks blocks;
        int width,height;bool wrap_x,wrap_y;
        std::uint64_t scope;
        std::shared_ptr<WorldMemory> memory;
        std::size_t charged;
    public:
        WorldSnapshot(Blocks owned,int w,int h,bool wx,bool wy,std::uint64_t s,std::shared_ptr<WorldMemory> account):
            blocks(std::move(owned)),width(w),height(h),wrap_x(wx),wrap_y(wy),scope(s),memory(std::move(account)),
            charged(sizeof(*this)+blocks.bucket_count()*sizeof(void*)+blocks.size()*64){memory->bytes.fetch_add(charged);}
        WorldSnapshot(WorldSnapshot const&)=delete;
        ~WorldSnapshot(){memory->bytes.fetch_sub(charged);}
        std::uint64_t key(int x,int y)const{
            if(wrap_x && width>0){x%=width;if(x<0)x+=width;}
            if(wrap_y && height>0){y%=height;if(y<0)y+=height;}
            return (std::uint64_t(std::uint32_t(x))<<32)|std::uint32_t(y);
        }
        Observation const* current(std::uint64_t id)const{
            auto block=blocks.find(world_block(id));if(block==blocks.end())return nullptr;
            auto value=block->second->values.find(id);return value==block->second->values.end()?nullptr:&value->second;
        }
        std::uint64_t scope_sequence()const{return scope;}
        std::size_t bytes()const{
            auto size=charged;for(auto const& block:blocks)size+=block.second->charged;return size;
        }
    };
    class WorldView {
        std::shared_ptr<WorldSnapshot const> lease;
    public:
        explicit WorldView(std::shared_ptr<WorldSnapshot const> owned):lease(std::move(owned)){}
        std::uint64_t key(int x,int y)const{return lease->key(x,y);}
        Observation const* current(std::uint64_t id)const{return lease->current(id);}
        std::uint64_t scope_sequence()const{return lease->scope_sequence();}
    };
    static constexpr std::size_t occurrence_limit=8192;
    // Includes conservative node/bucket overhead. Admission fails instead of
    // evicting identities; typical complete game maps fit well below this cap.
    static constexpr std::size_t budget=128u*1024u*1024u;
    static constexpr std::size_t record_limit=(budget-16u*1024u*1024u)/(sizeof(Record)+sizeof(Observation)+160);
private:
    std::unordered_map<std::uint64_t,Record> records;
    std::unordered_map<std::uint64_t,Observation> observations;
    std::uint64_t serial=0,epoch=0,appearance_epoch=0,scope_epoch=0,visibility_epoch=0,world_input_epoch=0;
    std::size_t authoritative_records=0;
    int width=0,height=0;
    bool wrap_x=false,wrap_y=false,valid=false;
    bool published=false;
    std::uint64_t configuration=0;
    c3x_renderer_i64 map_epoch=0,viewer_epoch=0;
    std::shared_ptr<WorldMemory> world_memory=std::make_shared<WorldMemory>();
    mutable std::shared_ptr<WorldSnapshot const> world_inputs;
    mutable std::unordered_map<std::uint64_t,std::shared_ptr<WorldBlock>> world_changes;
    RasterDependencyRevisions* raster_dependencies=nullptr;
    void touch_raster(RasterDependencyRevisions::Domain domain,std::uint64_t id){
        if(raster_dependencies)raster_dependencies->touch(domain,id);
    }
    std::uint64_t retained_raster_appearance(std::uint64_t id)const{
        auto record=retained(id);return record && record->authoritative?record->revision:0;
    }
    void reset_world_inputs(){
        if(world_input_epoch==UINT64_MAX)throw std::length_error("world input sequence exhausted");
        world_changes.clear();world_inputs.reset();++world_input_epoch;
        if(raster_dependencies)raster_dependencies->invalidate();
    }
    bool update_world_input(std::uint64_t id,c3x_renderer_tile_v1 const& tile,Record const& record){
        Observation next{};
        bool full=record.revision && (record.authoritative ||
                (!published && (record.partial_flags&C3X_RENDERER_TILE_PREFETCH))) &&
            (!(record.visibility_flags&C3X_RENDERER_TILE_VISIBILITY_KNOWN) ||
                (record.visibility_flags&C3X_RENDERER_TILE_EXPLORED));
        next.occurrence=full?record.appearance:content(tile);
        if(!full){
            c3x_renderer_tile_v1 absent{};
            absent.city_id=absent.city_owner_id=absent.city_size=absent.city_culture_group=absent.city_era=-1;
            copy_city_facts(next.occurrence,absent);
            next.occurrence.resource_id=next.occurrence.resource_class=-1;
            std::memset(next.occurrence.resource_name,0,sizeof(next.occurrence.resource_name));
            next.occurrence.barbarian_tribe_id=-1;next.occurrence.improvement_flags=0;
            next.occurrence.irrigation_mask=next.occurrence.has_effect=0;
        }
        // Halo topology is authoritative even when its object fields are omitted.
        next.occurrence.terrain_type=tile.terrain_type;next.occurrence.real_terrain_type=tile.real_terrain_type;
        next.occurrence.river_code=tile.river_code;
        auto partial=record.partial_flags&partial_facts_mask;
        if((record.visibility_flags&(C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED))!=
                (C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED))partial=0;
        if(partial&C3X_RENDERER_TILE_CITY_BODY_KNOWN)copy_city_facts(next.occurrence,record.appearance);
        if(partial&C3X_RENDERER_TILE_NATIVE_OVERLAYS_KNOWN)copy_native_overlays(next.occurrence,record.appearance);
        // Routes also participate in literal visible/legacy halo topology.
        // publish() has already supplied retained routes for omitted fog deltas.
        next.occurrence.road_mask=tile.road_mask;next.occurrence.railroad_mask=tile.railroad_mask;
        next.occurrence.tile_x=static_cast<std::int32_t>(id>>32);
        next.occurrence.tile_y=static_cast<std::int32_t>(id);
        next.occurrence.anchor_x=next.occurrence.anchor_y=0;
        next.occurrence.tile_flags=record.visibility_flags|C3X_RENDERER_TILE_TOPOLOGY_HALO|
            (full?C3X_RENDERER_TILE_PREFETCH:0u)|partial;
        next.occurrence.visibility_mask=record.visibility_mask;next.occurrence.tile_visibility=record.tile_visibility;
        next.occurrence.fog_status=record.fog_status;next.semantic=record.semantic;
        next.real=tile.real_terrain_type;
        next.ground=next.real==4 || next.real>=11?next.real:tile.terrain_type;
        next.relief=next.real==5 || next.real==6 || next.real==10?next.real:-1;
        next.surface=next.real==9?9:next.relief>=0?next.relief:next.ground;
        auto block_id=world_block(id);auto changed=world_changes.find(block_id);
        WorldBlock const* prior=nullptr;
        if(changed!=world_changes.end())prior=changed->second.get();
        else if(world_inputs){auto found=world_inputs->blocks.find(block_id);
            if(found!=world_inputs->blocks.end())prior=found->second.get();}
        if(prior){auto value=prior->values.find(id);
            if(value!=prior->values.end() && !std::memcmp(&value->second.occurrence,&next.occurrence,sizeof(next.occurrence)) &&
               value->second.semantic==next.semantic && value->second.ground==next.ground &&
               value->second.real==next.real && value->second.relief==next.relief && value->second.surface==next.surface)return true;}
        if(world_input_epoch==UINT64_MAX)return false;
        if(changed==world_changes.end()){
            auto allowance=sizeof(WorldBlock)+64*(sizeof(Observation)+64)+128*sizeof(void*);
            if(bytes()>budget || allowance>budget-bytes())return false;
            changed=world_changes.emplace(block_id,std::make_shared<WorldBlock>(world_memory,prior)).first;
        }
        changed->second->values[id]=std::move(next);changed->second->charge();++world_input_epoch;return bytes()<=budget;
    }
public:
    CapturedScene()=default;
    CapturedScene(CapturedScene const&)=delete;
    CapturedScene& operator=(CapturedScene const&)=delete;
    CapturedScene(CapturedScene&&)=default;
    CapturedScene& operator=(CapturedScene&&)=default;
    void bind_raster_dependencies(RasterDependencyRevisions* next){
        if(raster_dependencies==next)return;
        if(raster_dependencies)raster_dependencies->invalidate();
        raster_dependencies=next;if(next)next->invalidate();
    }
    // Called only by the render owner between jobs. Camera cancellation cannot
    // discard these updates; selected observations remain a separate concern.
    bool publication_scope(c3x_renderer_frame_v1 const& frame,
            c3x_renderer_camera_identity_v1 const& identity,std::uint64_t config) {
        bool changed=!published || configuration!=config || map_epoch!=identity.map_epoch ||
            viewer_epoch!=identity.viewer_epoch || width!=frame.world_width_tiles || height!=frame.world_height_tiles ||
            wrap_x!=(frame.world_wrap_x!=0) || wrap_y!=(frame.world_wrap_y!=0);
        if(changed){records.clear();observations.clear();reset_world_inputs();valid=false;authoritative_records=0;++appearance_epoch;++scope_epoch;++visibility_epoch;}
        published=true;configuration=config;map_epoch=identity.map_epoch;viewer_epoch=identity.viewer_epoch;
        width=frame.world_width_tiles;height=frame.world_height_tiles;
        wrap_x=frame.world_wrap_x!=0;wrap_y=frame.world_wrap_y!=0;return changed;
    }
    bool publish(c3x_renderer_tile_v1 const& tile,bool& changed) {
        auto id=key(tile.tile_x,tile.tile_y);auto found=records.find(id);
        if(found==records.end()){
            if(records.size()==record_limit)return false;
            found=records.try_emplace(id).first;
        }
        auto& record=found->second;auto next=content(tile);auto effective=tile;
        bool full=(tile.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH))!=0;
        auto partial=known_partial_flags(tile);
        if(!full){
            next=record.appearance;
            if(partial&C3X_RENDERER_TILE_CITY_BODY_KNOWN)copy_city_facts(next,tile);
            if(partial&C3X_RENDERER_TILE_NATIVE_OVERLAYS_KNOWN)copy_native_overlays(next,tile);
            next=content(next);
        }
        // Hidden minimal deltas omit routes. Full copies or explicit native
        // overlay authority replace them; visible and legacy halos stay literal.
        if(!full && !(partial&C3X_RENDERER_TILE_NATIVE_OVERLAYS_KNOWN) &&
           (record.authoritative || (record.partial_flags&C3X_RENDERER_TILE_NATIVE_OVERLAYS_KNOWN)) &&
           (tile.tile_flags&C3X_RENDERER_TILE_VISIBILITY_KNOWN) &&
           !(tile.tile_flags&C3X_RENDERER_TILE_VISIBLE)){
            effective.road_mask=record.appearance.road_mask;
            effective.railroad_mask=record.appearance.railroad_mask;
        }
        auto semantic=topology(effective);
        if(!record.semantic_revision || record.semantic!=semantic){
            if(serial==UINT64_MAX)return false;
            if(record.semantic_revision || (tile.tile_flags&C3X_RENDERER_TILE_VISIBILITY_KNOWN))changed=true;
            record.semantic=semantic;record.semantic_revision=++serial;
            touch_raster(RasterDependencyRevisions::Domain::semantic,id);
        }
        if((full || partial) && (!record.revision || std::memcmp(&next,&record.appearance,sizeof(next)))){
            if(serial==UINT64_MAX)return false;
            record.appearance=next;record.revision=++serial;record.compiled={};
            for(auto& variant:record.compiled_views)variant={};changed=true;
            ++appearance_epoch;
            touch_raster(RasterDependencyRevisions::Domain::appearance,id);
        }
        record.partial_flags|=partial|(full?C3X_RENDERER_TILE_PREFETCH:0u);
        auto flags=tile.tile_flags&C3X_RENDERER_TILE_VISIBILITY_BITS;
        if(!record.visibility_revision || record.visibility_flags!=flags || record.visibility_mask!=tile.visibility_mask ||
            record.tile_visibility!=tile.tile_visibility || record.fog_status!=tile.fog_status){
            if(serial==UINT64_MAX)return false;
            record.visibility_flags=flags;record.visibility_mask=tile.visibility_mask;
            record.tile_visibility=tile.tile_visibility;record.fog_status=tile.fog_status;record.visibility_revision=++serial;
            ++visibility_epoch;
            touch_raster(RasterDependencyRevisions::Domain::visibility,id);
        }
        if(full && !record.authoritative){record.authoritative=true;++authoritative_records;
            touch_raster(RasterDependencyRevisions::Domain::appearance,id);}
        return update_world_input(id,effective,record) && bytes()<=budget;
    }
    std::uint64_t key(int x,int y) const {
        auto canonical=[](int value,int extent,bool wraps){
            if(!wraps || extent<=0)return value;
            int result=value%extent;return result<0?result+extent:result;
        };
        return (std::uint64_t(std::uint32_t(canonical(x,width,wrap_x)))<<32) |
            std::uint32_t(canonical(y,height,wrap_y));
    }
    // Full records and lightweight halo observations carry these same world
    // inputs. Persist them independently of art eligibility and camera anchors.
    static std::uint64_t topology(c3x_renderer_tile_v1 const& tile){
        std::uint64_t value=1469598103934665603ull;
        for(auto input:{tile.terrain_type,tile.real_terrain_type,
                static_cast<c3x_renderer_i32>(tile.river_code),static_cast<c3x_renderer_i32>(tile.road_mask),
                static_cast<c3x_renderer_i32>(tile.railroad_mask)})
            value=(value^static_cast<std::uint32_t>(input))*1099511628211ull;
        return value;
    }
    static constexpr unsigned partial_facts_mask=C3X_RENDERER_TILE_CITY_BODY_KNOWN|C3X_RENDERER_TILE_NATIVE_OVERLAYS_KNOWN;
    static unsigned known_partial_flags(c3x_renderer_tile_v1 const& tile){
        return (tile.tile_flags&(C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED))==
            (C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED)?tile.tile_flags&partial_facts_mask:0u;
    }
    static void copy_city_facts(c3x_renderer_tile_v1& to,c3x_renderer_tile_v1 const& from){
        to.city_id=from.city_id;to.city_owner_id=from.city_owner_id;to.city_population=from.city_population;
        to.city_size=from.city_size;to.city_culture_group=from.city_culture_group;to.city_era=from.city_era;
        to.city_flags=from.city_flags;
        std::memcpy(to.city_owner,from.city_owner,sizeof(to.city_owner));
        std::memcpy(to.city_civilization,from.city_civilization,sizeof(to.city_civilization));
        std::memcpy(to.city_era_name,from.city_era_name,sizeof(to.city_era_name));
    }
    static void copy_native_overlays(c3x_renderer_tile_v1& to,c3x_renderer_tile_v1 const& from){
        constexpr unsigned mask=C3X_RENDERER_IMPROVEMENT_IRRIGATION|C3X_RENDERER_IMPROVEMENT_MINE|
            C3X_RENDERER_IMPROVEMENT_POLLUTION|C3X_RENDERER_IMPROVEMENT_CRATER|
            C3X_RENDERER_IMPROVEMENT_GOODY_HUT|C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP;
        to.road_mask=from.road_mask;to.railroad_mask=from.railroad_mask;to.route_style=from.route_style;
        to.irrigation_mask=from.irrigation_mask;to.barbarian_tribe_id=from.barbarian_tribe_id;
        to.improvement_flags=(to.improvement_flags&~mask)|(from.improvement_flags&mask);
    }
    static c3x_renderer_tile_v1 content(c3x_renderer_tile_v1 tile) {
        tile.tile_x=tile.tile_y=tile.anchor_x=tile.anchor_y=0;
        tile.tile_flags=tile.visibility_mask=tile.tile_visibility=0;
        tile.city_site_grade=0; // Presentation overlay, never terrain content.
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
           wrap_x!=(frame.world_wrap_x!=0) || wrap_y!=(frame.world_wrap_y!=0)){
            if(published)return false; // An old view cannot replace the published world.
            records.clear();observations.clear();reset_world_inputs();authoritative_records=0;++appearance_epoch;++scope_epoch;
        }
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
                if(it->second.seen!=epoch){
                    if(it->second.raster_appearance!=retained_raster_appearance(it->first))
                        touch_raster(RasterDependencyRevisions::Domain::appearance,it->first);
                    it=observations.erase(it);
                }else ++it;
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
            observed->second.raster_appearance=retained_raster_appearance(id);
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
        if(!published){auto& record=found->second;
            if(!record.semantic_revision || record.semantic!=semantic){
                if(serial==UINT64_MAX)return false;
                record.semantic=semantic;record.semantic_revision=++serial;
                touch_raster(RasterDependencyRevisions::Domain::semantic,id);
            }
        }
        auto partial=known_partial_flags(tile);
        if(full || (!published && partial)){
            auto& record=found->second;auto next=content(tile);
            if(!full){
                next=record.appearance;
                if(partial&C3X_RENDERER_TILE_CITY_BODY_KNOWN)copy_city_facts(next,tile);
                if(partial&C3X_RENDERER_TILE_NATIVE_OVERLAYS_KNOWN)copy_native_overlays(next,tile);
                next=content(next);
            }
            if(!record.authoritative && (!record.revision || std::memcmp(&next,&record.appearance,sizeof(next)))){
                if(serial==~std::uint64_t(0))return false;
                record.appearance=next;record.revision=++serial;record.compiled={};
                for(auto& variant:record.compiled_views)variant={};
                touch_raster(RasterDependencyRevisions::Domain::appearance,id);
            }
        }
        if(!published){auto& record=found->second;
            record.partial_flags|=partial|(full?C3X_RENDERER_TILE_PREFETCH:0u);
            auto flags=tile.tile_flags&C3X_RENDERER_TILE_VISIBILITY_BITS;
            bool visibility_changed=!record.visibility_revision || record.visibility_flags!=flags ||
                record.visibility_mask!=tile.visibility_mask || record.tile_visibility!=tile.tile_visibility || record.fog_status!=tile.fog_status;
            if(visibility_changed){
                if(serial==UINT64_MAX)return false;
                record.visibility_revision=++serial;++visibility_epoch;
                touch_raster(RasterDependencyRevisions::Domain::visibility,id);
            }
            record.visibility_flags=tile.tile_flags&C3X_RENDERER_TILE_VISIBILITY_BITS;
            record.visibility_mask=tile.visibility_mask;record.tile_visibility=tile.tile_visibility;record.fog_status=tile.fog_status;
            if(!update_world_input(id,tile,record))return false;
        }
        return bytes()<=budget;
    }
    void finish(){
        valid=true;
        if(raster_dependencies)for(auto& item:observations){
            auto revision=world_appearance_revision(item.first);
            if(item.second.raster_appearance!=revision)touch_raster(RasterDependencyRevisions::Domain::appearance,item.first);
            item.second.raster_appearance=revision;
        }
    }
    Observation const* current(std::uint64_t id) const {
        if(!valid)return nullptr;
        auto found=observations.find(id);
        return found!=observations.end() && found->second.seen==epoch?&found->second:nullptr;
    }
    // A frame lease may read only observations while the render owner updates
    // Record::compiled/compiled_views through attach(). Those are separate maps;
    // begin/update/finish (and destruction) must wait for all readers to join.
    class ObservationView {
        CapturedScene const& scene;
    public:
        explicit ObservationView(CapturedScene const& owner):scene(owner){}
        std::uint64_t key(int x,int y)const{return scene.key(x,y);}
        Observation const* current(std::uint64_t id)const{return scene.current(id);}
    };
    ObservationView observation_view()const{return ObservationView(*this);}
    // World-space compilers pin immutable canonical inputs. Legacy compilers
    // borrow the native occurrence/anchor map until their foreground lease
    // joins; begin/update/finish and owner destruction must wait for readers.
    class CompilationView {
        std::shared_ptr<WorldSnapshot const> world;
        CapturedScene const* native;
    public:
        CompilationView(CapturedScene const& owner,bool world_content):
            world(world_content?owner.world_snapshot():std::shared_ptr<WorldSnapshot const>{}),
            native(world_content?nullptr:&owner){}
        std::uint64_t key(int x,int y)const{return world?world->key(x,y):native->key(x,y);}
        Observation const* current(std::uint64_t id)const{return world?world->current(id):native->current(id);}
        std::uint64_t scope_sequence()const{return world?world->scope_sequence():native->scope_sequence();}
    };
    CompilationView compilation_view(bool world_content)const{return CompilationView(*this,world_content);}
    // Freeze a whole publication batch once. Each changed block is already
    // private; unchanged blocks and all camera-only generations are shared.
    // Access/publication is render-owner serialized; workers only read leases.
    std::shared_ptr<WorldSnapshot const> world_snapshot()const{
        if(world_inputs && world_changes.empty())return world_inputs;
        auto allowance=world_snapshot_allowance();
        if(bytes()>budget || allowance>budget-bytes())throw std::length_error("world input lease budget");
        WorldSnapshot::Blocks blocks;if(world_inputs)blocks=world_inputs->blocks;
        for(auto const& change:world_changes)blocks[change.first]=change.second;
        auto next=std::make_shared<WorldSnapshot>(std::move(blocks),width,height,wrap_x,wrap_y,scope_epoch,world_memory);
        if(bytes()>budget)throw std::length_error("world input lease budget");
        world_inputs=std::move(next);world_changes.clear();return world_inputs;
    }
    WorldView world_view()const{return WorldView(world_snapshot());}
    std::size_t world_snapshot_allowance()const{
        if(world_inputs && world_changes.empty())return 0;
        return sizeof(WorldSnapshot)+128+((world_inputs?world_inputs->blocks.size():0)+world_changes.size())*96;
    }
    std::uint64_t appearance_revision(std::uint64_t id) const {
        auto observed=current(id);
        if(!observed || !(observed->occurrence.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)))return 0;
        auto record=retained(id);auto appearance=content(observed->occurrence);
        return record && !std::memcmp(&appearance,&record->appearance,sizeof(appearance))?record->revision:0;
    }
    // A compiled world tile may depend on art outside the current camera's
    // observation set. The published full-world record remains its authority;
    // a current full observation must still agree before it can validate art.
    std::uint64_t world_appearance_revision(std::uint64_t id) const {
        auto observed=current(id);
        if(observed && (observed->occurrence.tile_flags&
                (C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)))
            return appearance_revision(id);
        auto record=retained(id);
        return record && record->authoritative?record->revision:0;
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
    std::size_t authoritative_size() const{return authoritative_records;}
    std::uint64_t appearance_sequence() const{return appearance_epoch;}
    std::uint64_t visibility_sequence() const{return visibility_epoch;}
    std::uint64_t scope_sequence() const{return scope_epoch;}
    std::uint64_t world_input_sequence() const{return world_input_epoch;}
    std::uint64_t observation_sequence() const{return epoch;}
    bool matches_world(c3x_renderer_frame_v1 const& frame) const {
        return width==frame.world_width_tiles && height==frame.world_height_tiles &&
            wrap_x==(frame.world_wrap_x!=0) && wrap_y==(frame.world_wrap_y!=0);
    }
    // Conservative tracked allocation estimate; allocator/driver residency is
    // reported separately. Record count and reserve requests stay within the cap;
    // the standard library determines the bucket count.
    std::size_t bytes() const {
        return sizeof(*this)+records.bucket_count()*sizeof(void*)+
            records.size()*(sizeof(std::pair<std::uint64_t const,Record>)+4*sizeof(void*))+
            observations.bucket_count()*sizeof(void*)+
            observations.size()*(sizeof(std::pair<std::uint64_t const,Observation>)+4*sizeof(void*))+
            sizeof(WorldMemory)+world_memory->bytes.load()+world_changes.bucket_count()*sizeof(void*)+world_changes.size()*64;
    }
};
static_assert(sizeof(CapturedScene::Observation)*CapturedScene::occurrence_limit<16u*1024u*1024u,
              "current observation payload must remain bounded");
} }
