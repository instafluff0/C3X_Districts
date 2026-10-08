#pragma once
#include <algorithm>
#include <cmath>

namespace c3x_renderer { namespace render_core {
// Presentation only: accepted native endpoints remain authoritative. A step
// lasts exactly as long as vanilla's constant-speed travel at the art's INI
// Fast Speed, starting when Civ III starts it; the facing turn happens during
// travel, never before it. Linear ramps take fixed fractions of that time, so cruise runs
// 1/(1-(a+d)/2), about 19%, faster. The run cycle follows distance at native
// speed, so slowing the body also slows its stride instead of sliding its feet.
struct UnitLocomotion {
    static constexpr double default_speed=225.,acceleration=.12,deceleration=.20;
    static int direction(int dx,int dy){
        return dx>0?(dy<0?1:dy>0?3:2):dx<0?(dy<0?7:dy>0?5:6):(dy>0?4:8);
    }
    static double duration(double distance,double speed){return distance/speed;}
    static double sample(double seconds,double distance,double speed){
        double total=duration(distance,speed);
        if(seconds<=0)return 0.;
        if(seconds>=total)return distance;
        double a=acceleration*total,d=deceleration*total,cruise=distance/(total-(a+d)*.5);
        if(seconds<a)return cruise*seconds*seconds/(2*a);
        if(seconds<total-d)return cruise*(seconds-a*.5);
        double remaining=total-seconds;
        return distance-cruise*remaining*remaining/(2*d);
    }
};
}}
