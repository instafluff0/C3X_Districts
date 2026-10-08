#pragma once
#include "unit_playback.h"
#include "unit_locomotion.h"
#include <map>
#include <cstdint>
#include <cmath>
#include <vector>
#include <new>

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
        bool has_visual=false;
    };
private:
    struct Instance {
        c3x_renderer_unit_v1 content{};
        c3x_renderer_unit_v1 occurrence{};
        unsigned flags=0;
        std::size_t unit=0,action=0;
        std::uint64_t revision=0,used=0,pose_identity=0;
        int tile_x=-1,tile_y=-1,display_id=-1;
    };
    std::map<int,Instance> instances;
    struct Motion {
        c3x_renderer_unit_move_v1 event{};
        Instance source{};
        int dx=0,dy=0;
        long long started=-1,available=-1,frequency=0,committed_at=-1;
        // Scene-clock time held (frozen view) since this step started, and how
        // much of it has been recovered by running at double speed.
        long long held=0,recovered=0,sampled=-1;
        // Civ III confirms a step after its animator snaps the unit onto the
        // tile; extra stretches this step's easing to that measured time.
        double extra=0;
        double cycle_distance=0,start_x=0,start_y=0,speed=UnitLocomotion::default_speed;
        double duration()const{return UnitLocomotion::duration(distance(),speed)+extra;}
        double covered(double seconds)const{
            double travel=UnitLocomotion::duration(distance(),speed);
            return UnitLocomotion::sample(extra>0?seconds*travel/(travel+extra):seconds,distance(),speed);
        }
        double seconds(long long ticks,long long clock)const{return std::max(0.,double(ticks-started+recovered)/clock);}
        double distance()const{return std::hypot(double(dx)*64.-start_x,double(dy)*64.-start_y*2.);}
        bool committed=false;
    };
    std::map<int,std::deque<Motion>> motions;
    // Native combat targets include a half-tile approach and a return to the
    // tile center. Retain those accepted endpoints, never intermediate pixels.
    struct PoseOffset {
        double from_x=0,from_y=0,to_x=0,to_y=0;
        long long started=-1,frequency=0;
        std::pair<double,double> sample(long long ticks,long long clock) const {
            double duration=std::hypot(to_x-from_x,to_y-from_y)/225.;
            double t=duration>0&&clock==frequency?std::clamp(double(ticks-started)/clock/duration,0.,1.):1.;
            return {from_x+(to_x-from_x)*t,from_y+(to_y-from_y)*t};
        }
    };
    std::map<int,PoseOffset> pose_offsets;
    struct Observed {c3x_renderer_unit_visual_v1 value{};int display_id=-1;};
    std::map<int,Observed> observations;
    std::map<int,c3x_renderer_unit_move_v1> accepted_moves;
    std::map<int,c3x_renderer_unit_spawn_v1> accepted_spawns;
    std::map<int,c3x_renderer_unit_state_v1> accepted_states;
    UnitPlayback playback;
    std::uint64_t serial=0,access=0;
    std::uint64_t scene_generation=0;
    long long scene_ticks=-1,scene_frequency=0;
    long long motion_pause=-1,motion_held=0;
    // Native travel starts at the move event's QPC time; the scene clock is
    // QPC minus paused visual time. The worker reports their offset whenever
    // it samples the clock (never during replay, whose clock is recorded).
    long long native_offset=0;bool native_offset_known=false;
    // Smoothed native step overhead: confirmation minus start, minus travel.
    double native_overhead=0;bool native_overhead_known=false;
public:
    // One displayed arrival per step: native start (event), displayed start
    // and end, and native confirmation, all on the scene clock. Diagnostic,
    // drained by the worker; bounded.
    // Scene-clock times, except event_qpc (Civ III's move event); shown is
    // the first scene sample that held the step.
    struct Arrival {int unit_id=0;long long native_start=0,visual_start=0,visual_end=0,committed=0,frequency=0,shown=0,event_qpc=0;};
    std::vector<Arrival> arrivals;
    void native_clock(long long qpc,long long visual){native_offset=qpc-visual;native_offset_known=true;}
private:
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
    double travel_seconds_remaining()const{
        if(motions.empty()||scene_frequency<=0||motion_pause>=0)return 0.;
        double remaining=0.;
        for(auto it=motions.begin();it!=motions.end();++it){
            if(it->second.empty())return 0.;
            auto const& motion=it->second.front();
            if(motion.started<0||motion.frequency!=scene_frequency)return 0.;
            double left=motion.duration()-motion.seconds(scene_ticks-motion_held,scene_frequency);
            remaining=it==motions.begin()?left:std::min(remaining,left);
        }
        return std::max(0.,remaining);
    }
    std::size_t motion_count(int id)const{auto found=motions.find(id);return found==motions.end()?0:found->second.size();}
    std::uint64_t generation()const{return scene_generation;}
    std::vector<std::pair<int,long long>> pending_arrivals()const{
        std::vector<std::pair<int,long long>> result;
        for(auto const& item:motions)for(auto const& motion:item.second)
            if(motion.committed)result.emplace_back(item.first,motion.committed_at);
        return result;
    }
    void forget(int id){pose_offsets.erase(id);motions.erase(id);instances.erase(id);observations.erase(id);accepted_moves.erase(id);accepted_spawns.erase(id);accepted_states.erase(id);playback.forget(id);++scene_generation;}
    void clear(){pose_offsets.clear();motions.clear();instances.clear();observations.clear();accepted_moves.clear();accepted_spawns.clear();accepted_states.clear();playback.clear();scene_ticks=-1;scene_frequency=0;motion_pause=-1;motion_held=0;++scene_generation;} // serial never reuses a token

    // Camera preparation freezes the displayed scene until ordered adoption.
    // That interval must not consume much travel that the player cannot yet
    // see. Callers may place the hold after a short allowance so ordinary
    // scroll/recenter transactions do not put moving units in slow motion.
    void pause_motion(long long ticks){if(motion_pause<0)motion_pause=std::max(ticks,scene_ticks);}
    void resume_motion(long long ticks,long long frequency){
        if(motion_pause<0)return;
        if(scene_frequency==frequency)motion_held+=std::max(0ll,ticks-motion_pause);
        motion_pause=-1;
    }

    // Late observations are expected across retirement and native ID reuse.
    // They cannot resurrect a body, and are distinct from malformed input.
    int state_status(c3x_renderer_unit_state_v1 const& value)const{
        if(value.struct_size!=sizeof(value)||value.unit_id<0||value.tile_x<0||value.tile_y<0||
           value.unit_type_id<0||value.owner_id<0||value.owner_id>=32||value.visible>1||
           (value.kind!=C3X_RENDERER_UNIT_STATE_OBSERVE&&value.kind!=C3X_RENDERER_UNIT_STATE_RETIRE)||
           value.presentation_frequency<=0||value.presentation_time_ticks<0||!capacity)
            return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(value.kind==C3X_RENDERER_UNIT_STATE_OBSERVE&&
           (value.action<-1||value.damage<0||value.max_hp<=0||value.damage>value.max_hp))
            return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(newer_event(value.unit_id,value.presentation_time_ticks,value.presentation_frequency)||
           (value.kind==C3X_RENDERER_UNIT_STATE_OBSERVE&&retired(value.unit_id)))
            return C3X_RENDERER_RESULT_SUPERSEDED;
        return C3X_RENDERER_RESULT_OK;
    }
    bool state(c3x_renderer_unit_state_v1 const& value){
        if(state_status(value)!=C3X_RENDERER_RESULT_OK)return false;
        if(value.kind==C3X_RENDERER_UNIT_STATE_RETIRE||!value.visible){
            retire_at(value.unit_id,value.presentation_time_ticks,value.presentation_frequency);
        }else{
            if(value.action!=1&&value.action!=2)motions.erase(value.unit_id);
            auto prior=accepted_states.find(value.unit_id);
            if(prior!=accepted_states.end()&&motions.find(value.unit_id)==motions.end()&&
               (prior->second.tile_x!=value.tile_x||prior->second.tile_y!=value.tile_y))
                pose_offsets.erase(value.unit_id); // Position corrections do not inherit a combat stance.
            if(prior!=accepted_states.end()&&prior->second.action!=value.action&&motions.find(value.unit_id)==motions.end()){
                // Invalidate old native selections atomically, but keep the
                // complete scene body until its replacement capture arrives.
                auto body=instances.find(value.unit_id);
                if(body!=instances.end())body->second.revision=++serial;
                playback.forget(value.unit_id);
            }
        }
        if(accepted_states.size()>=capacity&&accepted_states.find(value.unit_id)==accepted_states.end())
            accepted_states.erase(accepted_states.begin());
        accepted_states[value.unit_id]=value;
        ++scene_generation;
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
        ++scene_generation;
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
            ++scene_generation;
            return true;
        }
        auto motion=motions.find(value.unit_id);
        if(motion!=motions.end()){
            auto pending=std::find_if(motion->second.begin(),motion->second.end(),[&](auto const& m){
                return !m.committed&&m.event.old_x==value.old_x&&m.event.old_y==value.old_y&&
                    m.event.new_x==value.new_x&&m.event.new_y==value.new_y;});
            if(pending!=motion->second.end()){
                pending->committed=true;pending->committed_at=value.presentation_time_ticks;
                if(pending->event.presentation_frequency==value.presentation_frequency){
                    double overhead=double(value.presentation_time_ticks-pending->event.presentation_time_ticks)/
                        value.presentation_frequency-UnitLocomotion::duration(pending->distance(),pending->speed);
                    if(overhead>=0&&overhead<=.5){
                        native_overhead=native_overhead_known?(native_overhead+overhead)*.5:overhead;
                        native_overhead_known=true;
                    }
                }
            }
            else motions.erase(motion); // Teleport/correction cancels stale travel.
        }
        auto found=accepted_moves.find(value.unit_id);
        // An accepted move starts a new segment. Do not predict the previous
        // native observation across it; a subsequent body capture supplies the
        // authoritative pixel anchor and FLC action.
        observations.erase(value.unit_id);
        if(found==accepted_moves.end()&&accepted_moves.size()>=capacity)accepted_moves.erase(accepted_moves.begin());
        accepted_moves[value.unit_id]=value;
        ++scene_generation;
        return true;
    }

    // Called at Civ III's accepted movement target boundary, before its first
    // animation tick. Playback starts on the first renderer sample, so transport
    // delay cannot consume the entire visible movement before it reaches screen.
    bool begin_motion(c3x_renderer_unit_move_v1 const& value,int width,int height,bool wrap_x,bool wrap_y){
        if(value.struct_size!=sizeof(value)||value.unit_id<0||value.action!=2||
           value.presentation_frequency<=0||value.presentation_time_ticks<0||
           value.source_visible>1||value.target_visible>1||width<=0||height<=0||
           value.old_x<0||value.old_x>=width||value.new_x<0||value.new_x>=width||
           value.old_y<0||value.old_y>=height||value.new_y<0||value.new_y>=height||!capacity)return false;
        if(newer_event(value.unit_id,value.presentation_time_ticks,value.presentation_frequency))return false;
        auto found=instances.find(value.unit_id);
        if(!value.source_visible||!value.target_visible||found==instances.end()||retired(value.unit_id))return true;
        int dx=value.new_x-value.old_x,dy=value.new_y-value.old_y;
        if(wrap_x){if(dx>width/2)dx-=width;else if(dx<-width/2)dx+=width;}
        if(wrap_y){if(dy>height/2)dy-=height;else if(dy<-height/2)dy+=height;}
        if((std::abs(dx)+std::abs(dy)!=2)||((dx+dy)&1))return false;
        auto& queue=motions[value.unit_id];
        if(!queue.empty()){
            auto const& previous=queue.back().event;
            if(previous.presentation_frequency==value.presentation_frequency&&
               previous.presentation_time_ticks==value.presentation_time_ticks)return true;
            if(previous.new_x!=value.old_x||previous.new_y!=value.old_y)queue.clear();
        }
        if(queue.size()>=8)return false; // Bounded directed-action admission.
        Motion motion{};motion.event=value;motion.dx=dx;motion.dy=dy;
        if(value.speed>0)motion.speed=value.speed;
        motion.source=queue.empty()?found->second:queue.back().source;
        auto offset=pose_offsets.find(value.unit_id);
        if(queue.empty()&&offset!=pose_offsets.end()){
            // Victory advances from the displayed half-tile combat stance,
            // never back through the original tile center.
            auto current=offset->second.sample(scene_ticks-motion_held,scene_frequency);
            motion.start_x=current.first;motion.start_y=current.second;
        }
        pose_offsets.erase(value.unit_id);
        motion.source.tile_x=value.old_x;motion.source.tile_y=value.old_y;
        queue.push_back(motion);++scene_generation;
        return true;
    }

    bool observe(c3x_renderer_unit_visual_v1 const& value){
        if(value.struct_size!=sizeof(value)||value.unit_id<0||value.action<0||
           value.presentation_frequency<=0||value.presentation_time_ticks<0||
           value.projection_scale_milli<=0||value.projection_scale_milli>4000||
           value.max_hp<=0||value.damage<0||value.damage>value.max_hp||
           (value.flags&~31u)||!capacity)return false;
        if(newer_event(value.unit_id,value.presentation_time_ticks,value.presentation_frequency))return false;
        if(retired(value.unit_id))return false;
        if(value.flags&C3X_RENDERER_UNIT_HIDDEN){
            retire_at(value.unit_id,value.presentation_time_ticks,value.presentation_frequency);
            return true;
        }
        auto found=observations.find(value.unit_id);
        Observed next;next.value=value;
        if(found==observations.end()&&observations.size()>=capacity)observations.erase(observations.begin());
        observations[value.unit_id]=next;
        ++scene_generation;
        return true;
    }

    bool observe_animation(c3x_renderer_unit_animation_v1 const& value){
        if(value.struct_size!=sizeof(value)||value.display_unit_id<0||!observe(value.visual))return false;
        if(value.visual.flags&C3X_RENDERER_UNIT_HIDDEN)return true;
        observations[value.visual.unit_id].display_id=value.display_unit_id;
        return playback.observe(value);
    }

    template<class Catalog, class ActionName>
    bool capture(c3x_renderer_unit_v1 request,unsigned flags,Catalog const& catalog,
                 ActionName action_name,Selection& selected) {
        selected={};
        if(request.struct_size!=sizeof(request)||request.unit_key[63]||request.unit_id<0||
           (flags&~31u)||!capacity)return false;
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
        if(found!=instances.end()&&!std::memcmp(found->second.content.unit_key,request.unit_key,64))
            value.pose_identity=found->second.pose_identity;
        else value.pose_identity=++serial;
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
        value.occurrence=request;
        value.display_id=request.unit_id;
        auto group=observations.find(request.unit_id);
        if(group!=observations.end()&&group->second.display_id>=0&&
           group->second.value.presentation_time_ticks==request.presentation_time_ticks&&
           group->second.value.presentation_frequency==request.presentation_frequency)
            value.display_id=group->second.display_id;
        if(auto state=state_of(request.unit_id)){
            value.tile_x=state->tile_x;value.tile_y=state->tile_y;
        }
        if(found==instances.end() && instances.size()>=capacity){
            auto oldest=std::min_element(instances.begin(),instances.end(),[](auto const& a,auto const& b){return a.second.used<b.second.used;});
            forget(oldest->first);++evictions;
        }
        // Selection is exclusive even if Civ III does not redraw the previous
        // selected body during this update. Retire its cursor immediately.
        if(flags&C3X_RENDERER_UNIT_SELECTED)for(auto& pair:instances)
            if(pair.first!=request.unit_id)
                pair.second.flags&=~(C3X_RENDERER_UNIT_SELECTED|C3X_RENDERER_UNIT_CURSOR);
        instances[request.unit_id]=value;
        ++scene_generation;
        selected.id=request.unit_id;selected.revision=value.revision;selected.occurrence=request;
        auto observed=observations.find(request.unit_id);
        if(observed!=observations.end()&&!(observed->second.value.flags&C3X_RENDERER_UNIT_HIDDEN)&&
           observed->second.value.action==request.action&&
           observed->second.value.presentation_time_ticks==request.presentation_time_ticks&&
           observed->second.value.presentation_frequency==request.presentation_frequency&&
           observed->second.value.body_x==request.body_x&&observed->second.value.body_y==request.body_y){
            selected.visual=observed->second.value;
            selected.has_visual=true;
        }
        return true;
    }

    struct ScenePose {
        c3x_renderer_unit_v1 draw{};
        int tile_x=0,tile_y=0;
        std::size_t unit=0,action=0;
        unsigned predict=0;
        bool animated=false,cursor=false,owner_ring=false;
        int display_id=-1;
        std::uint64_t capture_order=0,pose_identity=0;
        long long pose_ticks=-1;
        bool travelling=false;
    };
    // Civ III starts a step at its move event and keeps native time from there.
    // A late first sample (transport, camera preparation) begins partway into
    // the step instead of extending it, so arrival stays native; at most a
    // quarter of the travel is skipped. The facing turn runs during travel.
    long long native_start(Motion const& motion,long long shown,long long frequency)const{
        if(!native_offset_known||motion.event.presentation_frequency!=frequency)return shown;
        auto start=motion.event.presentation_time_ticks-native_offset-motion_held;
        auto limit=shown-static_cast<long long>(.25*UnitLocomotion::duration(motion.distance(),motion.speed)*frequency);
        return std::clamp(start,limit,shown);
    }
    // A frozen view (camera preparation) holds travel so no unseen distance
    // is skipped. Afterwards the step runs at double speed until the held time
    // is recovered, so arrival returns to native time without a jump.
    void catch_up(Motion& motion,long long ticks)const{
        if(native_offset_known&&motion_pause<0&&motion.sampled>=0){
            auto deficit=(motion_held-motion.held)-motion.recovered;
            if(deficit>0)motion.recovered+=std::min(deficit,std::max(0ll,ticks-motion.sampled));
        }
        motion.sampled=ticks;
    }
    template<class Catalog>
    std::vector<ScenePose> scene_poses(c3x_renderer_frame_v1 const& frame,
            long long ticks,long long frequency,Catalog const& catalog){
        std::vector<ScenePose> result;
        if(!frame.tile_count||!frame.tiles||frame.tile_width<=0||frame.tile_height<=0||frequency<=0)return result;
        // Camera captures can arrive late. Their geometry may be new, but
        // sampling that view must never rewind the current visual scene.
        if(scene_frequency==frequency)ticks=std::max(ticks,scene_ticks);
        else motion_held=0;
        scene_ticks=ticks;scene_frequency=frequency;
        // A hold may start in the future: travel continues until that point.
        auto motion_ticks=(motion_pause>=0?std::min(motion_pause,ticks):ticks)-motion_held;
        // One native capture supplies both visibility and ordered wrap
        // occurrences. Index it once for a busy scene; a few retained actors
        // use the allocation-free linear iterator over the same authority.
        struct Occurrences {
            struct Bucket {std::uint64_t key=0;unsigned first=UINT_MAX,last=UINT_MAX;};
            c3x_renderer_frame_v1 const& frame;
            std::vector<Bucket> table;
            std::vector<unsigned> links;
            std::uint64_t key(int x,int y)const{
                auto canonical=[](int value,int extent,bool wraps){
                    if(!wraps||extent<=0)return value;
                    auto r=value%extent;return r<0?r+extent:r;
                };
                return (std::uint64_t(std::uint32_t(canonical(x,frame.world_width_tiles,frame.world_wrap_x!=0)))<<32)|
                    std::uint32_t(canonical(y,frame.world_height_tiles,frame.world_wrap_y!=0));
            }
            bool captured(unsigned i)const{return frame.tiles[i].tile_flags&
                (C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH);}
            bool visible(unsigned i,bool accepted_move)const{return accepted_move||
                (frame.tiles[i].tile_flags&C3X_RENDERER_TILE_VISIBLE);}
            std::size_t slot(std::uint64_t value)const{
                // Mix both lattice coordinates before power-of-two masking.
                auto hash=value;hash^=hash>>30;hash*=0xbf58476d1ce4e5b9ull;
                hash^=hash>>27;hash*=0x94d049bb133111ebull;hash^=hash>>31;
                auto i=std::size_t(hash)&(table.size()-1);
                while(table[i].first!=UINT_MAX&&table[i].key!=value)i=(i+1)&(table.size()-1);
                return i;
            }
            Occurrences(c3x_renderer_frame_v1 const& capture,bool indexed):frame(capture){
                if(!indexed||frame.tile_count>8192)return;
                std::size_t count=1;while(count<std::size_t(frame.tile_count)*2)count*=2;
                try {table.resize(count);links.assign(frame.tile_count,UINT_MAX);}
                catch(std::bad_alloc const&){
                    // Index admission is optional; retain the exact native
                    // linear iterator if the bounded metadata cannot fit.
                    std::vector<Bucket>().swap(table);std::vector<unsigned>().swap(links);return;
                }
                for(unsigned i=0;i<frame.tile_count;++i)if(captured(i)){
                    auto id=key(frame.tiles[i].tile_x,frame.tiles[i].tile_y);auto& bucket=table[slot(id)];
                    if(bucket.first==UINT_MAX){bucket.key=id;bucket.first=i;}
                    else links[bucket.last]=i;
                    bucket.last=i;
                }
            }
            unsigned next(std::uint64_t id,bool accepted_move,unsigned previous=UINT_MAX)const{
                if(!table.empty()){
                    for(auto i=previous==UINT_MAX?table[slot(id)].first:links[previous];i!=UINT_MAX;i=links[i])
                        if(visible(i,accepted_move))return i;
                    return UINT_MAX;
                }
                for(unsigned i=previous==UINT_MAX?0:previous+1;i<frame.tile_count;++i)
                    if(captured(i)&&visible(i,accepted_move)&&key(frame.tiles[i].tile_x,frame.tiles[i].tile_y)==id)return i;
                return UINT_MAX;
            }
        } occurrences(frame,instances.size()>4);
        for(auto& pair:instances){
            Motion* motion=nullptr;
            auto moving=motions.find(pair.first);
            if(moving!=motions.end()){
                auto& queue=moving->second;
                // A late native step has never been displayed. It cannot
                // inherit elapsed time spent waiting at the preceding tile.
                for(auto& next:queue)if(next.available<0){next.available=motion_ticks;next.frequency=frequency;}
                while(!queue.empty()){
                    auto& next=queue.front();
                    if(next.started<0){
                        next.started=native_start(next,motion_ticks,frequency);next.frequency=frequency;next.held=motion_held;
                        if(native_offset_known)next.extra=native_overhead;
                    }
                    catch_up(next,motion_ticks);
                    double distance=next.distance();
                    double duration=next.duration();
                    bool finished=next.frequency==frequency&&next.seconds(motion_ticks,frequency)>=duration;
                    if(finished&&next.committed){
                        if(queue.size()==1){
                            // Native stack selection may stop drawing this
                            // mover immediately after arrival (capture does
                            // this). A confirmed endpoint must end travel even
                            // without a later idle body sample.
                            auto& body=pair.second;
                            body.tile_x=next.event.new_x;body.tile_y=next.event.new_y;
                            body.content.direction=body.occurrence.direction=UnitLocomotion::direction(next.dx,next.dy);
                            if(body.occurrence.action==2&&body.unit<catalog.size()){
                                auto const& clips=catalog[body.unit].actions;
                                auto idle=std::find_if(clips.begin(),clips.end(),[](auto const& c){return c.name=="idle";});
                                if(idle!=clips.end()){
                                    body.action=std::size_t(idle-clips.begin());
                                    body.content.action=body.occurrence.action=1;
                                    body.occurrence.action_cursor=0;body.occurrence.frame_count=1;
                                    playback.forget(pair.first);
                                }
                            }
                            body.revision=++serial;pose_offsets.erase(pair.first);
                        }
                        long long continuation=next.started-next.recovered+static_cast<long long>(duration*frequency);
                        double cycle_distance=next.cycle_distance+distance;
                        if(arrivals.size()<64)arrivals.push_back({pair.first,
                            next.event.presentation_time_ticks-native_offset,next.started+next.held,
                            continuation+motion_held,next.committed_at-native_offset,frequency,
                            next.available+next.held,next.event.presentation_time_ticks});
                        queue.pop_front();
                        if(!queue.empty()){
                            queue.front().started=std::max(continuation,native_start(queue.front(),queue.front().available,frequency));
                            queue.front().held=motion_held;
                            if(native_offset_known)queue.front().extra=native_overhead;
                            queue.front().cycle_distance=cycle_distance;
                            queue.front().frequency=frequency;
                        }
                        continue;
                    }
                    motion=&next;break;
                }
                if(queue.empty())motions.erase(moving);
            }
            auto const& item=motion?motion->source:pair.second;
            if(!(item.flags&C3X_RENDERER_UNIT_STATE_CAPTURED)||
               (item.flags&C3X_RENDERER_UNIT_HIDDEN))
                continue;
            auto state=state_of(pair.first);
            if(!state||!state->visible||state->kind!=C3X_RENDERER_UNIT_STATE_OBSERVE||
               (!motion&&(state->tile_x!=item.tile_x||state->tile_y!=item.tile_y)))continue;
            auto occurrence_key=occurrences.key(item.tile_x,item.tile_y);
            // Terrain preparation can retain a view captured before this
            // accepted visible step. Its copied anchors remain authoritative
            // even when its tile visibility has not caught up with the unit.
            // Native hide/retirement above still removes the actor immediately.
            auto committed=accepted_moves.find(pair.first);
            auto after_capture=[&](c3x_renderer_unit_move_v1 const& event){
                return event.presentation_frequency==frame.presentation_frequency&&
                    event.presentation_time_ticks>frame.presentation_time_ticks;
            };
            bool accepted_move=motion?after_capture(motion->event):(committed!=accepted_moves.end()&&
                committed->second.target_visible&&after_capture(committed->second)&&
                occurrences.key(committed->second.new_x,committed->second.new_y)==occurrence_key);
            auto occurrence_index=occurrences.next(occurrence_key,accepted_move);
            if(occurrence_index==UINT_MAX)continue;
            auto* occurrence=&frame.tiles[occurrence_index];
            Selection selected{};selected.id=pair.first;selected.revision=item.revision;
            selected.occurrence=item.occurrence;
            auto observed=observations.find(pair.first);
            if(observed!=observations.end()&&
               observed->second.value.action==item.occurrence.action&&
               observed->second.value.presentation_time_ticks==item.occurrence.presentation_time_ticks&&
               observed->second.value.presentation_frequency==item.occurrence.presentation_frequency&&
               observed->second.value.body_x==item.occurrence.body_x&&
               observed->second.value.body_y==item.occurrence.body_y){
                selected.visual=observed->second.value;
                selected.has_visual=true;
            }
            ScenePose pose{};pose.pose_identity=item.pose_identity;pose.pose_ticks=motion_ticks;
            double travel_x=0,travel_y=0;
            if(motion){
                if(item.unit>=catalog.size())continue;
                auto const& actions=catalog[item.unit].actions;
                auto clip=std::find_if(actions.begin(),actions.end(),[](auto const& a){return a.name=="move";});
                if(clip==actions.end()||clip->duration<=0)continue;
                pose.action=std::size_t(clip-actions.begin());
                double seconds=motion->frequency==frequency?motion->seconds(motion_ticks,frequency):0.;
                double distance=motion->distance();
                double covered=motion->covered(seconds);
                double progress=distance>0?covered/distance:1.;
                pose.draw=item.occurrence;pose.draw.action=2;
                pose.draw.direction=UnitLocomotion::direction(motion->dx,motion->dy);
                travel_x=(motion->start_x+(motion->dx*64.-motion->start_x)*progress)*frame.tile_width/128.;
                travel_y=(motion->start_y+(motion->dy*32.-motion->start_y)*progress)*frame.tile_height/64.;
                pose.draw.frame_count=1000;
                pose.draw.action_cursor=int(std::fmod((motion->cycle_distance+covered)/motion->speed,double(clip->duration))/clip->duration*1000.);
                // Native confirmation may arrive after visible travel. Hold
                // the accepted endpoint in idle instead of running in place.
                if(progress>=1.){
                    auto idle=std::find_if(actions.begin(),actions.end(),[](auto const& a){return a.name=="idle";});
                    if(idle!=actions.end()){
                        pose.action=std::size_t(idle-actions.begin());pose.draw.action=1;
                        pose.draw.action_cursor=0;pose.draw.frame_count=1;
                    }
                }
                pose.draw.presentation_time_ticks=ticks;pose.draw.presentation_frequency=frequency;
                pose.predict=1;
            }else {
                if(!sample(selected,ticks,frequency,catalog,pose.draw,pose.predict))continue;
                auto offset=pose_offsets.find(pair.first);
                if(observed!=observations.end()){
                    auto const& native=observed->second.value;
                    double target_x=double(native.target_x)-(double(state->tile_x)+1.)*64.;
                    double target_y=double(native.target_y)-(double(state->tile_y)+1.)*32.;
                    if(frame.world_wrap_x&&frame.world_width_tiles>0)
                        target_x=std::remainder(target_x,double(frame.world_width_tiles)*64.);
                    if(frame.world_wrap_y&&frame.world_height_tiles>0)
                        target_y=std::remainder(target_y,double(frame.world_height_tiles)*32.);
                    // Ordinary tile travel has its own accepted segment. This
                    // owner only accepts native in-tile combat presentation.
                    bool bounded=std::abs(target_x)+2.*std::abs(target_y)<=64.;
                    if(bounded&&(target_x!=0||target_y!=0||offset!=pose_offsets.end())){
                        if(offset==pose_offsets.end())offset=pose_offsets.emplace(pair.first,PoseOffset{}).first;
                        auto& value=offset->second;
                        if(value.started<0||value.to_x!=target_x||value.to_y!=target_y){
                            auto previous=value.started<0?std::make_pair(0.,0.):value.sample(motion_ticks,frequency);
                            value.from_x=previous.first;value.from_y=previous.second;
                            value.to_x=target_x;value.to_y=target_y;
                            value.started=motion_ticks;value.frequency=frequency;
                        }
                    }
                    // A full-tile victory target can precede its accepted move
                    // record. Keep the last bounded stance until that segment
                    // takes ownership instead of snapping back to tile center.
                }
                if(offset!=pose_offsets.end()){
                    auto value=offset->second.sample(motion_ticks,frequency);
                    travel_x=value.first*frame.tile_width/128.;travel_y=value.second*frame.tile_height/64.;
                    pose.animated=value.first!=offset->second.to_x||value.second!=offset->second.to_y;
                }
            }
            // Native set_pixel_target_with_offset places the unit at the tile
            // center (+64,+32 at normal zoom). Use the copied scene's centers
            // for travel AND idle; a delayed native sprite capture can belong
            // to a different camera and must not move the world-space actor.
            int projection=frame.tile_width*1000/128;
            pose.draw.body_x=occurrence->anchor_x+frame.tile_width/2+int(std::lround(travel_x))-
                int(std::int64_t(pose.draw.sprite_width)*projection/2000);
            pose.draw.body_y=occurrence->anchor_y+frame.tile_height/2+int(std::lround(travel_y))-
                int(std::int64_t(pose.draw.sprite_height)*projection/2000);
            pose.draw.projection_scale_milli=projection;
            pose.draw.hour=frame.hour;pose.draw.season=frame.season;
            pose.tile_x=occurrence->tile_x;pose.tile_y=occurrence->tile_y;
            pose.display_id=item.display_id;pose.capture_order=item.used;pose.travelling=motion!=nullptr;
            pose.unit=item.unit;if(!motion)pose.action=item.action;
            pose.cursor=(pair.second.flags&C3X_RENDERER_UNIT_CURSOR)&&(pair.second.flags&C3X_RENDERER_UNIT_SELECTED);
            pose.owner_ring=(pair.second.flags&C3X_RENDERER_UNIT_TEAM_DISC)!=0;
            pose.animated=pose.animated||motion||pose.cursor||animated(selected,catalog);
            result.push_back(pose);
            // Each native captured wrap occurrence is a separate placement; all
            // borrow the same immutable asset and pose preparation identity.
            for(auto i=occurrences.next(occurrence_key,accepted_move,occurrence_index);i!=UINT_MAX;i=occurrences.next(occurrence_key,accepted_move,i)){
                auto const& tile=frame.tiles[i];
                auto dx=std::int64_t(tile.anchor_x)-occurrence->anchor_x,dy=std::int64_t(tile.anchor_y)-occurrence->anchor_y;
                auto x=std::int64_t(pose.draw.body_x)+dx,y=std::int64_t(pose.draw.body_y)+dy;
                if(x<INT_MIN||x>INT_MAX||y<INT_MIN||y>INT_MAX)continue;
                if(std::any_of(result.begin(),result.end(),[&](auto const& old){return old.draw.unit_id==pose.draw.unit_id&&old.draw.body_x==x&&old.draw.body_y==y;}))continue;
                auto wrapped=pose;wrapped.draw.body_x=int(x);wrapped.draw.body_y=int(y);
                wrapped.tile_x=tile.tile_x;wrapped.tile_y=tile.tile_y;result.push_back(wrapped);
            }
        }
        // Native drawing chooses the displayed group on each tile. Retaining
        // a body is not permission to show an older stack selection forever.
        // Travel keeps its source body until arrival; army members share the
        // parent's group instead of competing with their own commander.
        std::map<std::pair<int,int>,std::pair<std::uint64_t,int>> owners;
        for(auto const& pose:result)if(!pose.travelling){
            auto& owner=owners[{pose.tile_x,pose.tile_y}];
            if(pose.capture_order>=owner.first)owner={pose.capture_order,pose.display_id};
        }
        result.erase(std::remove_if(result.begin(),result.end(),[&](auto const& pose){
            return !pose.travelling&&owners.at({pose.tile_x,pose.tile_y}).second!=pose.display_id;
        }),result.end());
        return result;
    }

    // Ambient sampling requires no native callback. Tile movement is owned by
    // scene_poses; ordinary native samples never extrapolate screen position.
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
        return true;
    }
    template<class Catalog> bool animated(Selection const& selected,Catalog const& catalog)const{
        auto found=instances.find(selected.id);
        if(found==instances.end()||found->second.revision!=selected.revision)return false;
        auto const& instance=found->second;
        if(!(instance.flags&C3X_RENDERER_UNIT_STATE_CAPTURED)||instance.unit>=catalog.size()||instance.action>=catalog[instance.unit].actions.size())return false;
        auto const& clip=catalog[instance.unit].actions[instance.action];int action=instance.content.action;
        return playback.directed(selected.id,action)||
            (clip.ambient&&clip.loop&&((action==1&&(instance.flags&C3X_RENDERER_UNIT_SELECTED))||action==11||(action>=13&&action<=18)));
    }
    template<class Catalog>
    auto definition(Selection const& selected,Catalog const& catalog)const -> typename Catalog::value_type const* {
        auto found=instances.find(selected.id);
        return found!=instances.end() && found->second.revision==selected.revision && found->second.unit<catalog.size()
            ? &catalog[found->second.unit] : nullptr;
    }
};
}}
