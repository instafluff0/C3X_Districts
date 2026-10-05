#pragma once
#include <algorithm>
#include <cmath>

namespace c3x_renderer {
// Image-space presentation of a camera step. A newly committed camera is first
// shown at the previous camera's screen position (its step plus any unfinished
// offset) and slides linearly to rest while the previous world fills the
// trailing strip. The duration follows the recent interval between steps, so
// continuous edge scrolling reads as steady motion instead of one jump per
// completed camera job. Offsets are whole screen pixels at the presented zoom.
class PanTransition {
    double base_x=0,base_y=0;
    int step_x=0,step_y=0;
    long long start=0,duration=1,last_start=0;
    double interval=0; // seconds between recent steps
    bool active=false;
public:
    struct Offset {int x=0,y=0,under_x=0,under_y=0;bool active=false;};
    static constexpr double first_seconds=.6,minimum_seconds=.15,maximum_seconds=1.,restart_seconds=2.;
    void begin(int dx,int dy,long long now,long long frequency){
        if(frequency<=0){active=false;return;}
        auto current=sample(now,frequency);
        double gap=last_start?double(now-last_start)/double(frequency):0.;
        interval=gap>0.&&gap<restart_seconds?(interval>0.?.5*interval+.5*gap:gap):first_seconds;
        last_start=now;step_x=dx;step_y=dy;
        base_x=double(dx)+current.x;base_y=double(dy)+current.y;
        duration=std::max(1LL,(long long)(std::clamp(interval*.9,minimum_seconds,maximum_seconds)*double(frequency)));
        start=now;active=base_x!=0.||base_y!=0.;
    }
    Offset sample(long long now,long long frequency){
        Offset o;if(!active||frequency<=0)return o;
        double f=1.-double(now-start)/double(duration);
        if(f<=0.){active=false;return o;}
        f=std::min(1.,f);
        o.x=int(std::lround(base_x*f));o.y=int(std::lround(base_y*f));
        o.under_x=o.x-step_x;o.under_y=o.y-step_y;o.active=o.x||o.y;return o;
    }
    void cancel(){active=false;last_start=0;}
    bool moving()const{return active;}
};
}
