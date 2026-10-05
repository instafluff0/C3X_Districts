#pragma once
#include <algorithm>
#include <cmath>

namespace c3x_renderer { namespace render_core {
// Presentation only: accepted native endpoints remain authoritative. Linear
// ramps have continuous velocity and make the run cycle follow distance, so
// slowing the body also slows its stride instead of sliding its feet.
struct UnitLocomotion {
    static constexpr double speed=225.,acceleration=.16,deceleration=.24;
    static int direction(int dx,int dy){
        return dx>0?(dy<0?1:dy>0?3:2):dx<0?(dy<0?7:dy>0?5:6):(dy>0?4:8);
    }
    static double turn(int from,int to){
        if(from<1||from>8||to<1||to>8||from==to)return 0.;
        return .12+.06*std::abs(std::remainder(double(to-from),8.));
    }
    static double duration(double distance){return distance/speed+(acceleration+deceleration)*.5;}
    static double sample(double seconds,double distance){
        double total=duration(distance);
        if(seconds<=0)return 0.;
        if(seconds>=total)return distance;
        if(seconds<acceleration)return speed*seconds*seconds/(2*acceleration);
        if(seconds<total-deceleration)return speed*(seconds-acceleration*.5);
        double remaining=total-seconds;
        return distance-speed*remaining*remaining/(2*deceleration);
    }
};
}}
