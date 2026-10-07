#pragma once
// Ground vegetation keeps clear for routes, resources and tile objects. Shared
// by the canopy bodies, the vegetation floor decals and their compilers.
#include "data.h"
#include <cstring>
#include <vector>
namespace c3x_renderer { namespace fidelity {
struct BuildingBounds {float x0,y0,x1,y1;};
// Ground a canopy keeps clear for routes, resources and tile objects, in
// natural world coordinates. A tree also stays out from in front of them, as
// far as the hiding part of its height would cover them on screen.
struct CanopyClearing {
    std::vector<BuildingBounds> boxes;
    // Stationary resource parts (plants, rocks, decals): crowns may stand a
    // little closer, hiding them with half the reach. Animated animals move
    // and keep the full clearance.
    std::vector<BuildingBounds> near_boxes;
    // A resource tile keeps a thinner stand outside this disc (radius 0: none),
    // scaled down when the resource is stationary.
    float open_x=0,open_y=0,open_radius=0,open_scale=1;
    unsigned variety=broadleaf_forest;
    // A body unit of height is 150 pixels at a 128-pixel tile, where a world
    // unit of x-y spans 32 screen pixels. Moving (-1,+1) raises a point by 64.
    // Only the lower part of a crown needs to stay off what lies behind it.
    static constexpr float hide_fraction=.3f,hide_reach=hide_fraction*150.f/64.f;
    // The same spiral spread between the opening and the tile edge; false
    // where the opening reaches the edge in this direction.
    bool open_place(int c,int r,float angle,float ring,float reach,float&u,float&v)const{
        float du=std::cos(angle),dv=std::sin(angle),ou=open_x-float(c),ov=float(r)+1-open_y;
        auto edge=[](float d,float o){return d>1e-4f?(.96f-o)/d:d<-1e-4f?(.04f-o)/d:1e9f;};
        float inner=open_radius*open_scale+reach,outer=std::min(edge(du,ou),edge(dv,ov));
        if(outer<=inner)return false;
        u=ou+du*(inner+(outer-inner)*ring);v=ov+dv*(inner+(outer-inner)*ring);
        return true;
    }
    // The tree footprint swept toward the ground it hides, as a hexagon
    // tested against each box on the x, y and x+y axes.
    bool blocks(float x0,float y0,float x1,float y1,float height)const{
        auto swept=[&](std::vector<BuildingBounds> const&list,float reach){
            for(auto const&b:list)
                if(x0-reach<b.x1 && x1>b.x0 && y0<b.y1 && y1+reach>b.y0 &&
                   x0+y0<b.x1+b.y1 && x1+y1>b.x0+b.y0)return true;
            return false;
        };
        return swept(boxes,height*hide_reach) || swept(near_boxes,height*hide_reach*.5f);
    }
    // A floor decal's visible core keeps off the same ground.
    bool covers(float x0,float y0,float x1,float y1)const{
        for(auto const*list:{&boxes,&near_boxes})
            for(auto const&b:*list)if(x0<b.x1 && x1>b.x0 && y0<b.y1 && y1>b.y0)return true;
        return false;
    }
    // Content identity for the compiled ground that consumes the boxes.
    std::uint64_t hash()const{
        std::uint64_t value=1469598103934665603ull;
        auto mix=[&](float f){std::uint32_t bits;std::memcpy(&bits,&f,4);value=(value^bits)*1099511628211ull;};
        for(auto const*list:{&boxes,&near_boxes}){
            for(auto const&b:*list){mix(b.x0);mix(b.y0);mix(b.x1);mix(b.y1);}
            mix(-1.f);
        }
        mix(open_x);mix(open_y);mix(open_radius);mix(open_scale);
        return boxes.empty() && near_boxes.empty() && open_radius<=0?0:value;
    }
};
} }
