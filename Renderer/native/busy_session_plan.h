#pragma once
#include <algorithm>
#include <vector>

// Standalone scripted input only. No renderer-owned camera or game simulation.
namespace c3x_renderer {
struct BusySessionView {
    int x=0,y=0,width=128,phase=0;
    char const* name="idle";
    bool same_camera(BusySessionView const& other) const {
        return x==other.x && y==other.y && width==other.width;
    }
};
struct BusySessionPlan {
    static constexpr long long duration_us=60000000,slot_us=33333;
    int home_x,home_y,map_width,map_height;
    BusySessionView at(long long us) const {
        long long ms=us/1000;
        BusySessionView view{home_x,home_y,128,0,"idle"};
        if(ms<10000)return view;
        if(ms<20000){
            int t=int(ms-10000);view.phase=1;view.name="scroll";
            view.x+=2*((t<5000?t:10000-t)/500);return view;
        }
        if(ms<28000){
            int widths[]={160,192,160,128};view.width=widths[(ms-20000)/2000];
            view.phase=2;view.name="zoom";return view;
        }
        if(ms<40000){
            view.x=(map_width/4)|1;view.y=(map_height*3/4)|1;
            view.phase=ms<34000?3:4;view.name=ms<34000?"jump1_idle":"jump1_scroll";
            if(ms>=34000)view.x+=2*int((ms-34000)/1000);return view;
        }
        if(ms<50000){
            view.x=(map_width*3/4)|1;view.y=(map_height*3/4)|1;
            view.phase=ms<44000?5:6;view.name=ms<44000?"jump2_idle":"jump2_scroll";
            if(ms>=44000)view.x-=2*int((ms-44000)/1000);return view;
        }
        view.phase=7;view.name="return_idle";return view;
    }
};
struct BusySessionInputs {
    BusySessionPlan plan;
    unsigned next_event=0;
    struct Request {BusySessionView view;long long requested_us;int event=-1;};
    // Discrete zoom and minimap clicks stay queued. Mouse/edge-scroll positions
    // may coalesce while the synchronous renderer blocks the simulated pump.
    Request select(long long now_us) {
        constexpr long long events[]={20000000,22000000,24000000,26000000,28000000,40000000,50000000};
        if(next_event<7 && events[next_event]<=now_us) {
            auto event=next_event++;return {plan.at(events[event]),events[event],int(event)};
        }
        return {plan.at(now_us),now_us/BusySessionPlan::slot_us*BusySessionPlan::slot_us,-1};
    }
    bool finished(long long now_us) const {return now_us>=BusySessionPlan::duration_us && next_event==7;}
};

struct BusyReplayRequest {
    BusySessionView view;
    long long logical_us=0;
    int event=-1;
};

inline std::vector<BusyReplayRequest> fixed_busy_replay(BusySessionPlan const& plan,unsigned samples_per_phase=25) {
    if(samples_per_phase<1)return {};
    constexpr long long bounds[]={0,10000000,20000000,28000000,34000000,40000000,44000000,50000000,60000000};
    constexpr long long events[]={20000000,22000000,24000000,26000000,28000000,40000000,50000000};
    std::vector<BusyReplayRequest> result;
    result.reserve(samples_per_phase*8);
    for(int phase=0;phase<8;++phase) {
        std::vector<long long> times;
        for(auto event:events)if(plan.at(event).phase==phase)times.push_back(event);
        for(unsigned i=0;times.size()<samples_per_phase;++i) {
            auto value=bounds[phase]+(2*static_cast<long long>(i)+1)*(bounds[phase+1]-bounds[phase])/(2*samples_per_phase);
            value=(std::min)(value,bounds[phase+1]-1);
            if(std::find(times.begin(),times.end(),value)==times.end())times.push_back(value);
        }
        std::sort(times.begin(),times.end());
        times.resize(samples_per_phase);
        for(auto logical:times) {
            int event=-1;
            for(unsigned i=0;i<7;++i)if(events[i]==logical)event=int(i);
            result.push_back({plan.at(logical),logical,event});
        }
    }
    return result;
}
}
