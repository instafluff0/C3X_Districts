#pragma once
#include "environment_runtime.h"
#include <array>
#include <algorithm>
#include <cmath>

namespace c3x_renderer { namespace lighting {
#include "../lab/shared/shaders/lighting/shadow_policy.hlsl"
// World XY projects as ((x+y)*half_width, (x-y)*half_height).
// Source object XY projects as ((x-y)*half_width, (x+y)*half_height).
// Keep geometry, inverse-transpose normals and light projection in this one
// contract. Source importers and native sprite placement remain independent.
constexpr float shadow_slope = 1.35f;
constexpr float object_height_to_world = 150.f / (112.f * .82f);
constexpr float shadow_page_span = 6.f;
constexpr unsigned shadow_resolution = 1024;

struct KeyLight {
    std::array<float,3> direction{}, color{};
    float intensity=0;
};

inline KeyLight key_light(EnvironmentState const& e) {
    KeyLight out;
    float x=e.sun_direction[0]*e.sun_intensity+e.moon_direction[0]*e.moon_intensity;
    float y=e.sun_direction[1]*e.sun_intensity+e.moon_direction[1]*e.moon_intensity;
    float h=std::hypot(x,y);
    if(h>1e-6f){x/=h;y/=h;}else{x=-1;y=0;}
    float n=std::sqrt(1+shadow_slope*shadow_slope);
    out.direction={x/n,y/n,shadow_slope/n};
    out.intensity=e.sun_intensity+e.moon_intensity;
    for(unsigned i=0;i<3;++i)
        out.color[i]=(e.sun_color[i]*e.sun_intensity+e.moon_color[i]*e.moon_intensity)/
            std::max(.000001f,out.intensity);
    return out;
}

inline std::array<float,12> shadow_frame(EnvironmentState const& e) {
    auto l=key_light(e).direction;
    float h=std::hypot(l[0],l[1]),x=l[0]/h,y=l[1]/h;
    return {-y,x,0,shadow_page_span,-l[2]*x,-l[2]*y,h,float(shadow_resolution),
            l[0],l[1],l[2],0};
}

inline std::array<float,2> ground_offset(float const* light,float height) {
    if(!std::isfinite(height) || !std::isfinite(light[0]) || !std::isfinite(light[1]) ||
       !std::isfinite(light[2]) || light[2]<=.0001f)return {0,0};
    return {-light[0]/light[2]*height,-light[1]/light[2]*height};
}

inline std::array<float,3> object_normal(float x,float y,float z,
                                       float height_scale=object_height_to_world) {
    std::array<float,3> n={x,-y,z/height_scale};
    float length=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
    if(length>1e-6f)for(auto& value:n)value/=length;
    else n={0,0,1};
    return n;
}
} }
