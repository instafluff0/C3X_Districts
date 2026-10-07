#pragma once
#include "../animation_runtime.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace c3x_renderer { namespace render_core {

// Source bounds belong to the mesh asset. Poses/occurrences borrow this fixed
// metadata; no screen mesh or per-pose cache is retained here.
struct ResourceSourceBounds {
    struct Box {std::array<float,3> low{},high{};bool valid=false;};
    std::array<Box,256> bones{};
    bool prepare(AnimationMesh const& mesh) {
        *this={};
        for(auto const& vertex:mesh.vertices)for(unsigned i=0;i<4;++i){
            if(vertex.weights[i]==0)continue;
            if(vertex.joints[i]>=mesh.bones || mesh.bones>256)return false;
            auto& box=bones[vertex.joints[i]];
            if(!box.valid){std::copy(vertex.source.position,vertex.source.position+3,box.low.begin());box.high=box.low;box.valid=true;}
            else for(unsigned a=0;a<3;++a){
                box.low[a]=std::min(box.low[a],vertex.source.position[a]);
                box.high[a]=std::max(box.high[a],vertex.source.position[a]);
            }
        }
        return true;
    }
    bool posed(AnimationPose const& pose,unsigned count,Box& out)const {
        out={};if(count>256)return false;
        for(unsigned b=0;b<count;++b){auto const& box=bones[b];if(!box.valid)continue;
            auto const& p=pose.positions[b];
            for(unsigned c=0;c<8;++c){float v[3];
                for(unsigned a=0;a<3;++a)v[a]=(c&(1u<<a))?box.high[a]:box.low[a];
                for(unsigned a=0;a<3;++a){
                    float x=v[0]*p[a]+v[1]*p[4+a]+v[2]*p[8+a]+p[12+a];
                    if(!std::isfinite(x))return false;
                    if(!out.valid)out.low[a]=out.high[a]=x;
                    else {out.low[a]=std::min(out.low[a],x);out.high[a]=std::max(out.high[a],x);}
                }
                out.valid=true;
            }
        }
        // Nonnegative decoded weights sum to 1 +/- 1e-5. Include that deviation
        // and scalar float roundoff when enclosing their convex combination.
        if(out.valid)for(unsigned a=0;a<3;++a){
            float pad=1e-4f+std::max(std::abs(out.low[a]),std::abs(out.high[a]))*2e-5f;
            out.low[a]-=pad;out.high[a]+=pad;
        }
        return out.valid;
    }
};

// Four placement vectors followed by 7 float4 values per bone. CPU interpolation
// and inverse transpose are shared with the legacy sampler; shaders only skin
// vertices and project each occurrence. The cap is below D3D11's 64 KiB CB limit.
inline std::size_t resource_pose_bytes(unsigned bones){return bones && bones<=256?(4u+7u*bones)*16u:0;}
inline void pack_resource_pose(AnimationPose const& pose,unsigned bones,float* output){
    for(unsigned b=0;b<bones;++b){auto* dst=output+16+b*28;
        std::copy(pose.positions[b].begin(),pose.positions[b].end(),dst);
        for(unsigned c=0;c<3;++c){for(unsigned a=0;a<3;++a)dst[16+c*4+a]=pose.normals[b][c*3+a];dst[19+c*4]=0;}
    }
}
// Shear packed posed positions onto the ground plane under an animal:
// source z += a*x + b*y + c, so its feet, and a head lowered to graze, follow
// the terrain slope instead of sinking into rising ground. Rows are row vectors
// (position = x*row0 + y*row1 + z*row2 + row3), so each row's z gains a*row.x +
// b*row.y and the translation row also gains c.
inline void slope_resource_pose(unsigned bones,float a,float b,float c,float* output){
    if(a==0 && b==0 && c==0)return;
    for(unsigned bone=0;bone<bones;++bone){float* m=output+16+bone*28;
        for(unsigned row=0;row<4;++row)m[row*4+2]+=a*m[row*4]+b*m[row*4+1];
        m[14]+=c;
    }
}

// Where a layout makes room for routes (an oasis): {shrink, du, dv}. Sizes from
// `largest` down to `smallest` of the group's full reach `radius` are tried; the
// first that has a spot inside the tile (moving up to `reach` tiles) clear of
// every drawn route point (tile-local u,v) by its reach plus a small margin is
// used, at its clearest spot. Among equally clear spots, the one farthest from
// the routes on average wins (the open corner between two arms, not along one).
// If none fits, the smallest size takes its clearest spot.
template<class Points>
std::array<float,3> route_clearance_layout(Points const& points,float radius,float smallest,float largest,
        float reach){
    std::array<float,3> chosen{smallest,0.f,0.f};
    for(int step=0;step<=8;++step){
        float shrink=largest+(smallest-largest)*float(step)/8.f;
        float room=std::max(0.f,.47f-radius*shrink),best=-1.f,best_mean=-1.f;
        for(int k=-1;k<16;++k){
            float angle=6.28318530718f*float(k)/16.f,distance=k<0?0.f:reach;
            float du=std::clamp(std::cos(angle)*distance,-room,room),dv=std::clamp(std::sin(angle)*distance,-room,room);
            float clearance=1e9f,mean=0.f;
            for(auto const& p:points){float d=std::hypot(p[0]-.5f-du,p[1]-.5f-dv);clearance=std::min(clearance,d);mean+=d;}
            mean/=float(std::max<std::size_t>(1,points.size()));
            if(clearance>best+1e-3f || (clearance>best-1e-3f && mean>best_mean)){
                best=std::max(best,clearance);best_mean=mean;chosen={shrink,du,dv};}
        }
        if(best-.04f>=radius*shrink)return chosen;
    }
    return chosen;
}

} }
