#pragma once
#include <vector>
#include <cmath>
#include <limits>

namespace c3x_renderer { namespace render_core {
// One camera's completed unit placements, published with its completed map.
// Native ink keeps its pixel size and content; only its attachment moves.
struct UnitHudAnchors {
    struct Anchor {int id,x,y;};
    struct Offset {int x=0,y=0;bool visible=true;};
    std::vector<Anchor> anchors;
    template<class Poses> void publish(Poses const& poses,int source_x=0,int source_y=0){
        anchors.clear();
        for(auto const& pose:poses){auto const& d=pose.draw;
            anchors.push_back({d.unit_id,d.body_x+int(static_cast<long long>(d.sprite_width)*d.projection_scale_milli/2000)-source_x,
                d.body_y+int(static_cast<long long>(d.sprite_height)*d.projection_scale_milli/2000)-source_y});
        }
    }
    Offset offset(int id,int native_x,int native_y,int width,int height,double scale)const{
        Anchor const* nearest=nullptr;double distance=std::numeric_limits<double>::max();
        for(auto const& a:anchors)if(a.id==id){
            double dx=double(a.x)-native_x,dy=double(a.y)-native_y,d=dx*dx+dy*dy;
            if(d<distance){nearest=&a;distance=d;}
        }
        if(!nearest)return {0,0,false}; // No visible body grants no retained ink.
        return {int(std::lround((nearest->x-width/2)*scale+width/2))-native_x,
            int(std::lround((nearest->y-height/2)*scale+height/2))-native_y,true};
    }
};
}}
