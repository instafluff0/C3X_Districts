#ifndef C3X_UNIT_PLAYBACK_H
#define C3X_UNIT_PLAYBACK_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include "../c3x_renderer_api.h"

namespace c3x_renderer { namespace render_core {
// UI-caller-owned visual state. Native actions/anchors are never advanced here.
// Resolved requests carry an exact source sample into both caches and helpers.
class UnitPlayback {
    struct Instance {
        int id,action;char key[64];
        long long ticks,frequency;double seconds;
        bool advancing;int cursor,frames;
    };
    std::deque<Instance> instances;
public:
    void clear(){instances.clear();}
    template<class Clip>
    bool resolve(c3x_renderer_unit_v1& request,Clip const& clip,bool selected,unsigned& next_step) {
        next_step=1;
        bool idle=request.action==1;
        bool work=request.action==11 || (request.action>=13 && request.action<=18);
        bool advancing=!idle || selected;
        // One-shots (including movement/combat) keep native lifecycle sampling.
        if(!clip.ambient || (!idle && !work)) {
            instances.erase(std::remove_if(instances.begin(),instances.end(),[&](auto const& s){return s.id==request.unit_id;}),instances.end());
            return advancing;
        }
        if(!clip.loop || !std::isfinite(clip.duration) || clip.duration<=0 || clip.frames<2 ||
           clip.frames>4096 || request.presentation_frequency<=0 || request.presentation_time_ticks<0)
            return false;
        auto old=std::find_if(instances.begin(),instances.end(),[&](auto const& s){return s.id==request.unit_id;});
        Instance state{};state.id=request.unit_id;state.action=request.action;
        std::memcpy(state.key,request.unit_key,64);state.frames=int(clip.frames)-1;
        if(old!=instances.end()) {
            if(old->action==request.action && !std::memcmp(old->key,request.unit_key,64) &&
               old->frames==state.frames)state=*old;
            instances.erase(old);
        }
        double interval=1.0/15; // First observation: current native timer opportunity.
        if(advancing && state.advancing && state.frequency==request.presentation_frequency &&
           request.presentation_time_ticks>=state.ticks) {
            double elapsed=double(request.presentation_time_ticks-state.ticks)/request.presentation_frequency;
            // Hidden units and blocked native calls do not accumulate catch-up.
            if(elapsed>0 && elapsed<=.25) {
                state.seconds=std::fmod(state.seconds+elapsed,double(clip.duration));interval=elapsed;
            }
        }
        state.ticks=request.presentation_time_ticks;state.frequency=request.presentation_frequency;
        state.advancing=advancing;
        state.cursor=int(state.seconds/clip.duration*state.frames+1e-7)%state.frames;
        request.action_cursor=state.cursor;request.frame_count=state.frames;
        int next=int(std::fmod(state.seconds+interval,double(clip.duration))/clip.duration*state.frames+1e-7)%state.frames;
        next_step=unsigned((next-state.cursor+state.frames)%state.frames);
        if(!advancing)next_step=0;
        if(instances.size()==128)instances.pop_front();
        instances.push_back(state);
        return advancing;
    }
};
}}
#endif
