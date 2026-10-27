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

// Source (BeautyFrame) receivers keep their own coefficients and take the
// environment relative to the approved noon: direct colour and strength with
// moonlight weighted as frame_illumination weights it, and its night-lifted
// sky, so relief, trees and units match cities and water at every hour.
struct SourceResponse {
    std::array<float,3> color{}, ambient{};
    float strength=0;
};

inline SourceResponse source_response(EnvironmentState const& e) {
    auto noon=evaluate_environment(12,0);
    float direct=e.sun_intensity+moon_response*e.moon_intensity;
    float lift=1+night_ambient_lift*e.night_activation;
    SourceResponse out;
    out.strength=direct/(noon.sun_intensity+moon_response*noon.moon_intensity);
    for(unsigned i=0;i<3;++i) {
        out.color[i]=(e.sun_color[i]*e.sun_intensity+moon_response*e.moon_color[i]*e.moon_intensity)/
            std::max(.000001f,direct)/std::max(.001f,noon.sun_color[i]);
        out.ambient[i]=e.ambient_color[i]*lift/std::max(.001f,noon.ambient_color[i]);
    }
    return out;
}

// CPU copy of the gameplay frame_illumination (Q6_GAMEPLAY_NIGHT) on open
// flat ground, lit by the key light. Unit ground shadows blend this opacity.
inline float dynamic_shadow_opacity(EnvironmentState const& e) {
    float z=key_light(e).direction[2],night=e.night_activation;
    float luma[3]={.2126f,.7152f,.0722f},open=0,shaded=0;
    for(unsigned i=0;i<3;++i) {
        float ambient=e.ambient_color[i]*.46f*(1+.8f*night);
        float direct=e.sun_color[i]*e.sun_intensity*(.16f+.84f*z)+
            e.moon_color[i]*e.moon_intensity*(.20f+.80f*z)*.82f*2.2f;
        open+=luma[i]*(ambient+direct);
        // Strength past 1 also shades the sky fill (frame_illumination).
        float s=e.shadow_strength;
        shaded+=luma[i]*(ambient*(1-std::clamp(s-1,0.f,1.f))+direct*(1-std::clamp(s,0.f,1.f)));
    }
    return c3x_dynamic_shadow_opacity(open,shaded);
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
