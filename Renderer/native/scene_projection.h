#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace c3x_renderer {
// Display projection only. Native anchors, scene geometry and gameplay camera
// remain in their captured pixel coordinate system.
struct SceneProjection {
    static constexpr float minimum=1.f,maximum=3.f;
    static constexpr unsigned minimum_q16=unsigned(minimum*65536.f),maximum_q16=unsigned(maximum*65536.f);
    float scale=1.f,cx=0.f,cy=0.f;
    SceneProjection(unsigned width,unsigned height,float zoom=1.f)
        :scale(zoom),cx(float(width/2)),cy(float(height/2)) {
        if(!std::isfinite(scale)||scale<minimum||scale>maximum)
            throw std::invalid_argument("scene projection scale");
    }
    float x(float p)const{return cx+(p-cx)*scale;}
    float y(float p)const{return cy+(p-cy)*scale;}
    // Conservative canonical bounds for a pass's displayed rectangle. Keep
    // the raster guard in the same coordinate system as viewport().
    template<class Rect> Rect source_rect(Rect r,float guard=0.f,float margin_x=0.f,float margin_y=0.f)const {
        if(scale==1.f)return r;
        float x=cx+guard+margin_x,y=cy+guard+margin_y;
        return {int(std::floor(x+(float(r.left)-x)/scale))-1,
                int(std::floor(y+(float(r.top)-y)/scale))-1,
                int(std::ceil(x+(float(r.right)-x)/scale))+1,
                int(std::ceil(y+(float(r.bottom)-y)/scale))+1};
    }
    // A raster viewport applies the affine transform after the existing
    // vertex projection, so every material keeps its authored world basis.
    template<class Viewport> void viewport(Viewport& v,float guard=0.f,
            float margin_x=0.f,float margin_y=0.f,float raster_scale=1.f)const {
        v.TopLeftX=(cx+guard+margin_x)*(1.f-scale)*raster_scale;
        v.TopLeftY=(cy+guard+margin_y)*(1.f-scale)*raster_scale;
        v.Width*=scale;v.Height*=scale;
    }
};
}
