#pragma once
#include "../animation_runtime.h"

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

} }
