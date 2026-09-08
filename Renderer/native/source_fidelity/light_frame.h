#pragma once
// Q6 frame from the authoritative EnvironmentState. Shared by all natural
// face shaders and the existing world source-shadow page projection.
namespace c3x_renderer { namespace fidelity {
inline std::array<float,12> light_frame(EnvironmentState const&e){
    float x=e.sun_direction[0]*e.sun_intensity+e.moon_direction[0]*e.moon_intensity;
    float y=e.sun_direction[1]*e.sun_intensity+e.moon_direction[1]*e.moon_intensity;
    float h=std::hypot(x,y);if(h>1e-6f){x/=h;y/=h;}else{x=-1;y=0;}
    float n=std::sqrt(1+1.35f*1.35f);
    return {-y,x,0,6,-1.35f*x/n,-1.35f*y/n,1/n,1024,x/n,y/n,1.35f/n,0};
}
} }
