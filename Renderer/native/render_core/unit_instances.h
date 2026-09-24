#pragma once
#include "unit_playback.h"
#include <map>
#include <cstdint>

namespace c3x_renderer { namespace render_core {
// Serialized by RendererWorker's caller gate. Instances own copied content and
// compiled catalog indices; selections own occurrence/projection data. Neither
// an instance nor a cached pose grants visibility. Only an explicit selection
// from the native capture may feed a pass.
class UnitInstances {
public:
    struct Selection {
        int id=-1;
        std::uint64_t revision=0;
        c3x_renderer_unit_v1 occurrence{};
        c3x_renderer_unit_visual_v1 visual{};
        c3x_renderer_unit_visual_v1 motion_origin{};
        bool has_visual=false;
    };
private:
    struct Instance {
        c3x_renderer_unit_v1 content{};
        unsigned flags=0;
        std::size_t unit=0,action=0;
        std::uint64_t revision=0,used=0;
    };
    std::map<int,Instance> instances;
    struct Observed {c3x_renderer_unit_visual_v1 value{},origin{};};
    std::map<int,Observed> observations;
    std::map<int,c3x_renderer_unit_move_v1> accepted_moves;
    std::map<int,c3x_renderer_unit_spawn_v1> accepted_spawns;
    std::map<int,c3x_renderer_unit_state_v1> accepted_states;
    UnitPlayback playback;
    std::uint64_t serial=0,access=0;
    // CPU identity metadata only. Shared meshes and completed poses retain
    // their existing independent budgets. Eviction invalidates old selections.
    std::size_t capacity;
    bool newer_event(int id,std::int64_t ticks,std::int64_t frequency)const{
        auto birth=accepted_spawns.find(id);
        if(birth!=accepted_spawns.end()&&birth->second.presentation_frequency==frequency&&
           birth->second.presentation_time_ticks>ticks)return true;
        auto move=accepted_moves.find(id);
        if(move!=accepted_moves.end()&&move->second.presentation_frequency==frequency&&
           move->second.presentation_time_ticks>ticks)return true;
        auto pose=observations.find(id);
        if(pose!=observations.end()&&pose->second.value.presentation_frequency==frequency&&
           pose->second.value.presentation_time_ticks>ticks)return true;
        auto state=accepted_states.find(id);
        return state!=accepted_states.end()&&state->second.presentation_frequency==frequency&&
               state->second.presentation_time_ticks>ticks;
    }
    bool retired(int id)const{
        auto found=accepted_states.find(id);
        return found!=accepted_states.end()&&found->second.kind==C3X_RENDERER_UNIT_STATE_RETIRE;
    }
    void retire_at(int id,std::int64_t ticks,std::int64_t frequency){
        forget(id);
        if(observations.size()>=capacity)observations.erase(observations.begin());
        auto& marker=observations[id].value;
        marker.flags=C3X_RENDERER_UNIT_HIDDEN;
        marker.presentation_time_ticks=ticks;
        marker.presentation_frequency=frequency;
    }
public:
    std::uint64_t captures=0,reused=0,bindings=0,evictions=0;
    explicit UnitInstances(std::size_t limit=4096):capacity(limit){}
    std::size_t size()const{return instances.size();}
    void forget(int id){instances.erase(id);observations.erase(id);accepted_moves.erase(id);accepted_spawns.erase(id);accepted_states.erase(id);playback.forget(id);}
    void clear(){instances.clear();observations.clear();accepted_moves.clear();accepted_spawns.clear();accepted_states.clear();playback.clear();} // serial never reuses a token

    bool state(c3x_renderer_unit_state_v1 const& value){
        if(value.struct_size!=sizeof(value)||value.unit_id<0||value.tile_x<0||value.tile_y<0||
           value.unit_type_id<0||value.owner_id<0||value.owner_id>=32||value.visible>1||
           (value.kind!=C3X_RENDERER_UNIT_STATE_OBSERVE&&value.kind!=C3X_RENDERER_UNIT_STATE_RETIRE)||
           value.presentation_frequency<=0||value.presentation_time_ticks<0||!capacity)return false;
        if(value.kind==C3X_RENDERER_UNIT_STATE_OBSERVE&&
           (value.action<-1||value.damage<0||value.max_hp<=0||value.damage>value.max_hp))return false;
        if(newer_event(value.unit_id,value.presentation_time_ticks,value.presentation_frequency))return false;
        if(value.kind==C3X_RENDERER_UNIT_STATE_OBSERVE&&retired(value.unit_id))return false;
        if(value.kind==C3X_RENDERER_UNIT_STATE_RETIRE||!value.visible){
            retire_at(value.unit_id,value.presentation_time_ticks,value.presentation_frequency);
        }else{
            auto prior=accepted_states.find(value.unit_id);
            if(prior!=accepted_states.end()&&prior->second.action!=value.action){
                instances.erase(value.unit_id);observations.erase(value.unit_id);playback.forget(value.unit_id);
            }
        }
        if(accepted_states.size()>=capacity&&accepted_states.find(value.unit_id)==accepted_states.end())
            accepted_states.erase(accepted_states.begin());
        accepted_states[value.unit_id]=value;
        return true;
    }
    c3x_renderer_unit_state_v1 const* state_of(int id)const{
        auto found=accepted_states.find(id);return found==accepted_states.end()?nullptr:&found->second;
    }

    bool spawn(c3x_renderer_unit_spawn_v1 const& value){
        if(value.struct_size!=sizeof(value)||value.unit_id<0||value.tile_x<0||value.tile_y<0||
           value.unit_type_id<0||value.owner_id<0||value.owner_id>=32||value.visible>1||
           value.presentation_frequency<=0||value.presentation_time_ticks<0||!capacity)return false;
        if(newer_event(value.unit_id,value.presentation_time_ticks,value.presentation_frequency))return false;
        forget(value.unit_id); // Civ III may reuse an ID after despawn.
        if(accepted_spawns.size()>=capacity)accepted_spawns.erase(accepted_spawns.begin());
        accepted_spawns[value.unit_id]=value;
        return true;
    }

    bool move(c3x_renderer_unit_move_v1 const& value){
        if(value.struct_size!=sizeof(value)||value.unit_id<0||value.action<-1||
           value.source_visible>1||value.target_visible>1||
           value.presentation_frequency<=0||value.presentation_time_ticks<0||!capacity||
           (value.old_x==value.new_x&&value.old_y==value.new_y))return false;
        if(newer_event(value.unit_id,value.presentation_time_ticks,value.presentation_frequency))return false;
        if(retired(value.unit_id))return false;
        if(!value.target_visible){
            forget(value.unit_id);
            if(accepted_moves.size()>=capacity)accepted_moves.erase(accepted_moves.begin());
            accepted_moves[value.unit_id]=value; // Retain the fog-loss time against late observations.
            return true;
        }
        auto found=accepted_moves.find(value.unit_id);
        // An accepted move starts a new segment. Do not predict the previous
        // native observation across it; a subsequent body capture supplies the
        // authoritative pixel anchor and FLC action.
        observations.erase(value.unit_id);
        if(found==accepted_moves.end()&&accepted_moves.size()>=capacity)accepted_moves.erase(accepted_moves.begin());
        accepted_moves[value.unit_id]=value;
        return true;
    }

    bool observe(c3x_renderer_unit_visual_v1 const& value){
        if(value.struct_size!=sizeof(value)||value.unit_id<0||value.action<0||
           value.presentation_frequency<=0||value.presentation_time_ticks<0||
           value.projection_scale_milli<=0||value.projection_scale_milli>4000||
           value.max_hp<=0||value.damage<0||value.damage>value.max_hp||
           (value.flags&~7u)||!capacity)return false;
        if(newer_event(value.unit_id,value.presentation_time_ticks,value.presentation_frequency))return false;
        if(retired(value.unit_id))return false;
        if(value.flags&C3X_RENDERER_UNIT_HIDDEN){
            retire_at(value.unit_id,value.presentation_time_ticks,value.presentation_frequency);
            return true;
        }
        auto found=observations.find(value.unit_id);
        Observed next;next.value=value;next.origin=value;
        if(value.action==2&&found!=observations.end()){
            auto const& previous=found->second.value;
            if(previous.action==2&&previous.target_x==value.target_x&&previous.target_y==value.target_y&&
               previous.projection_scale_milli==value.projection_scale_milli&&
               previous.presentation_frequency==value.presentation_frequency)
                next.origin=found->second.origin;
        }
        if(found==observations.end()&&observations.size()>=capacity)observations.erase(observations.begin());
        observations[value.unit_id]=next;
        return true;
    }

    template<class Catalog, class ActionName>
    bool capture(c3x_renderer_unit_v1 request,unsigned flags,Catalog const& catalog,
                 ActionName action_name,Selection& selected) {
        selected={};
        if(request.struct_size!=sizeof(request)||request.unit_key[63]||request.unit_id<0||
           (flags&~7u)||!capacity)return false;
        if(newer_event(request.unit_id,request.presentation_time_ticks,request.presentation_frequency))return false;
        if(retired(request.unit_id))return false;
        if(flags&C3X_RENDERER_UNIT_HIDDEN){
            if(flags&C3X_RENDERER_UNIT_STATE_CAPTURED)
                retire_at(request.unit_id,request.presentation_time_ticks,request.presentation_frequency);
            return false;
        }
        ++captures;
        bool captured=(flags&C3X_RENDERER_UNIT_STATE_CAPTURED)!=0;
        if(captured && !(flags&C3X_RENDERER_UNIT_SELECTED) && request.action==8)request.action=1;
        auto found=instances.find(request.unit_id);
        Instance value{};
        bool bound=found!=instances.end() && found->second.content.action==request.action &&
            !std::memcmp(found->second.content.unit_key,request.unit_key,64);
        if(bound)value=found->second;
        else {
            auto name=action_name(request.action);
            if(!name){forget(request.unit_id);return false;}
            auto unit=std::find_if(catalog.begin(),catalog.end(),[&](auto const& u){
                return std::find(u.keys.begin(),u.keys.end(),request.unit_key)!=u.keys.end();});
            if(unit==catalog.end()){forget(request.unit_id);return false;}
            auto action=std::find_if(unit->actions.begin(),unit->actions.end(),[&](auto const& a){return a.name==name;});
            if(action==unit->actions.end()){forget(request.unit_id);return false;}
            value.unit=std::size_t(unit-catalog.begin());value.action=std::size_t(action-unit->actions.begin());++bindings;
        }
        // Catalog replacement clears this owner before old indices can escape.
        auto const& clip=catalog[value.unit].actions[value.action];
        auto content=request;
        content.body_x=content.body_y=content.sprite_width=content.sprite_height=0;
        content.reduced=content.projection_scale_milli=0;
        content.presentation_time_ticks=content.presentation_frequency=0;
        bool ambient=captured && clip.ambient && (request.action==1 || request.action==11 ||
            (request.action>=13 && request.action<=18));
        if(ambient)content.action_cursor=content.frame_count=0;
        if(found!=instances.end() && found->second.flags==flags &&
           !std::memcmp(&found->second.content,&content,sizeof(content))) {
            value.revision=found->second.revision;++reused;
        }else value.revision=++serial;
        value.content=content;value.flags=flags;value.used=++access;
        if(found==instances.end() && instances.size()>=capacity){
            auto oldest=std::min_element(instances.begin(),instances.end(),[](auto const& a,auto const& b){return a.second.used<b.second.used;});
            forget(oldest->first);++evictions;
        }
        instances[request.unit_id]=value;
        selected.id=request.unit_id;selected.revision=value.revision;selected.occurrence=request;
        auto observed=observations.find(request.unit_id);
        if(observed!=observations.end()&&!(observed->second.value.flags&C3X_RENDERER_UNIT_HIDDEN)&&
           observed->second.value.action==request.action&&
           observed->second.value.presentation_time_ticks==request.presentation_time_ticks&&
           observed->second.value.presentation_frequency==request.presentation_frequency&&
           observed->second.value.body_x==request.body_x&&observed->second.value.body_y==request.body_y){
            selected.visual=observed->second.value;
            selected.motion_origin=observed->second.origin;selected.has_visual=true;
        }
        return true;
    }

    // Clock sampling requires no new native body call. Directed actions retain
    // their captured cursor/anchor; only authored eligible ambient loops advance.
    template<class Catalog>
    bool sample(Selection const& selected,long long ticks,long long frequency,Catalog const& catalog,
                c3x_renderer_unit_v1& output,unsigned& predict) {
        auto found=instances.find(selected.id);
        if(found==instances.end() || found->second.revision!=selected.revision)return false;
        auto const& instance=found->second;
        if(instance.unit>=catalog.size()||instance.action>=catalog[instance.unit].actions.size())return false;
        auto const& clip=catalog[instance.unit].actions[instance.action];
        output=selected.occurrence;output.presentation_time_ticks=ticks;output.presentation_frequency=frequency;predict=1;
        if(instance.flags&C3X_RENDERER_UNIT_STATE_CAPTURED){
            bool selected_unit=(instance.flags&C3X_RENDERER_UNIT_SELECTED)!=0;
            if(!playback.resolve(output,clip,selected_unit,predict))predict=0;
            if(!predict && !clip.ambient && output.action==1){output.action_cursor=0;output.frame_count=1;}
        }
        if(selected.has_visual&&output.action==2&&selected.visual.action==2&&
           selected.motion_origin.presentation_frequency==frequency&&
           ticks>=selected.motion_origin.presentation_time_ticks){
            // Civ III has already accepted this move and supplied its actual
            // pixel target. Sample one continuous segment from the first body
            // observation; later sparse native poses correct it without
            // restarting the clock or pulling the unit backwards.
            auto const& origin=selected.motion_origin;
            double dx=double(origin.target_x)-origin.pixel_x;
            double dy=double(origin.target_y)-origin.pixel_y;
            double distance=std::hypot(dx,2.0*dy);
            double elapsed=double(ticks-origin.presentation_time_ticks)/double(frequency);
            // One ordinary native step is at most a tile. A larger target
            // delta means a camera/projection discontinuity, not travel to
            // predict from this stale screen-space anchor.
            // FLC_Animation advances toward its target with doubled-Y
            // distance / Animation_Info::fast_speed. Civ III's stock ground
            // unit INIs (including Scout and Worker) specify Fast Speed=225;
            // later authoritative poses correct custom-unit speed overrides.
            constexpr double stock_ground_move_speed=225.0;
            double progress=distance>0&&distance<=160.0?
                std::clamp(stock_ground_move_speed*elapsed/distance,0.0,1.0):0.0;
            auto position=[&](double start,double change,double current){
                double predicted=start+change*progress;
                return change>0?std::max(predicted,current):change<0?std::min(predicted,current):current;
            };
            double scale=double(selected.visual.projection_scale_milli)/1000.0;
            output.body_x+=int(std::lround((position(origin.pixel_x,dx,selected.visual.pixel_x)-selected.visual.pixel_x)*scale));
            output.body_y+=int(std::lround((position(origin.pixel_y,dy,selected.visual.pixel_y)-selected.visual.pixel_y)*scale));
            if(output.frame_count>1&&distance>0)
                output.action_cursor=std::max(output.action_cursor,
                    std::min(output.frame_count-1,int(progress*output.frame_count)));
        }
        return true;
    }
    template<class Catalog> bool animated(Selection const& selected,Catalog const& catalog)const{
        auto found=instances.find(selected.id);
        if(found==instances.end()||found->second.revision!=selected.revision)return false;
        auto const& instance=found->second;
        if(!(instance.flags&C3X_RENDERER_UNIT_STATE_CAPTURED)||instance.unit>=catalog.size()||instance.action>=catalog[instance.unit].actions.size())return false;
        auto const& clip=catalog[instance.unit].actions[instance.action];int action=instance.content.action;
        if(selected.has_visual&&action==2&&selected.visual.action==2&&
           (selected.visual.pixel_x!=selected.visual.target_x||selected.visual.pixel_y!=selected.visual.target_y))return true;
        return clip.ambient&&clip.loop&&((action==1&&(instance.flags&C3X_RENDERER_UNIT_SELECTED))||action==11||(action>=13&&action<=18));
    }
    template<class Catalog>
    auto definition(Selection const& selected,Catalog const& catalog)const -> typename Catalog::value_type const* {
        auto found=instances.find(selected.id);
        return found!=instances.end() && found->second.revision==selected.revision && found->second.unit<catalog.size()
            ? &catalog[found->second.unit] : nullptr;
    }
};
}}
