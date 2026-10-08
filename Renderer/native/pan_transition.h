#pragma once
#include <algorithm>
#include <cmath>

namespace c3x_renderer {
// Image-space presentation of a camera step. A newly committed camera is first
// shown at the previous camera's screen position (its step plus any unfinished
// offset) and slides to rest while the previous world fills the trailing
// strip. Offsets are whole screen pixels at the presented zoom.
//
// An isolated step (a recentre, or the first step of a scroll) eases in and
// out. Steps that follow within restart_seconds are scrolling: they cruise at
// constant speed for the recent interval between steps, then spend the last
// `reserve` of the distance on a cubic ease-out with matching speed. A step
// that arrives on time therefore continues at the same speed, and the final
// step of a scroll glides to rest instead of stopping dead.
class PanTransition {
    double base_x=0,base_y=0;
    int step_x=0,step_y=0;
    long long start=0,last_start=0;
    double interval=0; // seconds between recent steps
    double cruise=1,tail=0,reserve=0; // seconds; tail==0 is an eased shift
    bool active=false;
    // Remaining fraction of the base offset, in [0,1]; 0 once finished.
    double remaining(double t)const{
        if(tail<=0.){
            double u=t/cruise;if(u>=1.)return 0.;
            return 1.-u*u*(3.-2.*u);
        }
        if(t<cruise)return 1.-(1.-reserve)*t/cruise;
        double s=(t-cruise)/tail;if(s>=1.)return 0.;
        return reserve*(1.-s)*(1.-s)*(1.-s);
    }
public:
    struct Offset {int x=0,y=0,under_x=0,under_y=0;bool active=false;};
    static constexpr double first_seconds=.6,minimum_seconds=.15,maximum_seconds=1.,restart_seconds=2.;
    static constexpr double shift_seconds=.28,shift_seconds_per_pixel=.00035,shift_minimum=.3,shift_maximum=.55;
    static constexpr double reserve_maximum=.25,tail_maximum=.45;
    void begin(int dx,int dy,long long now,long long frequency){
        if(frequency<=0){active=false;return;}
        double left=active?remaining(double(now-start)/double(frequency)):0.;
        double current_x=base_x*left,current_y=base_y*left;
        double gap=last_start?double(now-last_start)/double(frequency):0.;
        bool scrolling=gap>0.&&gap<restart_seconds;
        interval=scrolling?(interval>0.?.5*interval+.5*gap:gap):first_seconds;
        last_start=now;step_x=dx;step_y=dy;
        base_x=double(dx)+current_x;base_y=double(dy)+current_y;
        if(scrolling){
            cruise=std::clamp(interval,minimum_seconds,maximum_seconds);
            // A cubic tail starting at the cruise speed lasts 3*reserve/(1-reserve)
            // cruises; keep the final glide short on slow (busy-map) intervals.
            reserve=std::min(reserve_maximum,tail_maximum/(3.*cruise+tail_maximum));
            tail=cruise*3.*reserve/(1.-reserve);
        }else{
            cruise=std::clamp(shift_seconds+std::hypot(base_x,base_y)*shift_seconds_per_pixel,shift_minimum,shift_maximum);
            reserve=tail=0.;
        }
        start=now;active=base_x!=0.||base_y!=0.;
    }
    Offset sample(long long now,long long frequency){
        Offset o;if(!active||frequency<=0)return o;
        double f=remaining(std::max(0.,double(now-start)/double(frequency)));
        if(f<=0.){active=false;return o;}
        o.x=int(std::lround(base_x*f));o.y=int(std::lround(base_y*f));
        o.under_x=o.x-step_x;o.under_y=o.y-step_y;o.active=o.x||o.y;return o;
    }
    void cancel(){active=false;last_start=0;}
    bool moving()const{return active;}
};
}
