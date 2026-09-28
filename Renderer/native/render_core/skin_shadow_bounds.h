#pragma once
#include "../animation_runtime.h"
#include "../unit_shadow.h"

namespace c3x_renderer { namespace render_core {
// Convex skin weights keep each posed vertex inside the union's AABB. Fit the
// light-space projection of per-joint bind bounds; never skin vertices on CPU.
struct SkinShadowBounds {
    struct Box { std::array<float,3> low{},high{}; bool used=false; };
    std::vector<Box> joints;
    void prepare(AnimationMesh const& mesh){
        joints.assign(mesh.bones,{});
        for(auto const& vertex:mesh.vertices)for(unsigned i=0;i<4;++i)if(vertex.weights[i]>0){
            auto& box=joints[vertex.joints[i]];
            for(unsigned axis=0;axis<3;++axis){
                float p=vertex.source.position[axis];
                box.low[axis]=box.used?std::min(box.low[axis],p):p;
                box.high[axis]=box.used?std::max(box.high[axis],p):p;
            }
            box.used=true;
        }
    }
    void append(float const* palette,float angle,float scale,float offset_z,
                std::vector<UnitShadow::Point>& points)const{
        float c=std::cos(angle),s=std::sin(angle);
        for(unsigned joint=0;joint<joints.size();++joint){
            auto const& box=joints[joint];if(!box.used)continue;
            auto m=palette+joint*16;
            for(unsigned corner=0;corner<8;++corner){
                float p[3];for(unsigned axis=0;axis<3;++axis)
                    p[axis]=(corner&(1u<<axis))?box.high[axis]:box.low[axis];
                float x=p[0]*m[0]+p[1]*m[4]+p[2]*m[8]+m[12];
                float y=p[0]*m[1]+p[1]*m[5]+p[2]*m[9]+m[13];
                float z=p[0]*m[2]+p[1]*m[6]+p[2]*m[10]+m[14];
                // Keep corners below the ground: their light-space extremes
                // can bound a positive-height weighted vertex.
                points.push_back({(x*c-y*s)*scale,(x*s+y*c)*scale,(z+offset_z)*scale});
            }
        }
    }
};
} }
