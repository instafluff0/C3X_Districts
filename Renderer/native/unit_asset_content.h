#pragma once
#include "animation_runtime.h"
#include <string>
#include "render_core/content_preparation.h"

namespace c3x_renderer {
// Path/scalar inputs are immutable. CPU workers read and decode generic packs;
// they never inspect renderer state or touch the immediate context.
struct UnitAssetInput {std::string path;bool mesh=false,move=false;};
struct UnitAssetContent {
    std::shared_ptr<AnimationMesh const> mesh;
    std::vector<std::uint8_t> dds;
    bool failed=false;
    static std::size_t mesh_bytes(AnimationMesh const& value){
        return sizeof(AnimationMesh)+2*sizeof(void*)+value.vertices.capacity()*sizeof(AnimationVertex)+value.indices.capacity()*sizeof(std::uint32_t)+
            value.palettes.capacity()*sizeof(float)+value.rig.poses.capacity()*sizeof(AnimationJointPose)+
            value.rig.parents.capacity()*sizeof(int)+value.rig.skin_joints.capacity()*sizeof(unsigned)+
            value.rig.inverse_bind.capacity()*sizeof(std::array<float,16>);
    }
    std::size_t bytes()const{return sizeof(*this)+dds.capacity()+(mesh?mesh_bytes(*mesh):0);}
};
template<class Read>std::unique_ptr<UnitAssetContent> compile_unit_asset(
        UnitAssetInput const& input,std::atomic<bool> const& cancelled,Read read){
    auto result=std::make_unique<UnitAssetContent>();
    std::vector<std::uint8_t> payload;
    if(cancelled.load(std::memory_order_relaxed))return {};
    if(!read(input.path.c_str(),payload)){result->failed=true;return result;}
    if(cancelled.load(std::memory_order_relaxed))return {};
    if(input.mesh){
        AnimationMesh decoded;
        if(!decode_animation_mesh(payload,decoded,render_core::ContentPreparation<std::uint64_t,UnitAssetInput,UnitAssetContent>::byte_limit-sizeof(UnitAssetContent))){result->failed=true;return result;}
                if(input.move && decoded.frames>1){
                    // Most authored locomotion clips are already anchored. A
                    // few keep common whole-body travel below a stationary
                    // root bone (notably Spearman). Native tile movement owns
                    // that travel; strip only a substantial shared endpoint
                    // delta, leaving the gait's joint and vertical motion.
                    std::vector<float> dx,dy;dx.reserve(decoded.bones);dy.reserve(decoded.bones);
                    for(unsigned bone=0;bone<decoded.bones;++bone){
                        auto first=std::size_t(bone)*16+12;
                        auto last=(std::size_t(decoded.frames-1)*decoded.bones+bone)*16+12;
                        dx.push_back(decoded.palettes[last]-decoded.palettes[first]);
                        dy.push_back(decoded.palettes[last+1]-decoded.palettes[first+1]);
                    }
                    auto middle=decoded.bones/2;
                    std::nth_element(dx.begin(),dx.begin()+middle,dx.end());
                    std::nth_element(dy.begin(),dy.begin()+middle,dy.end());
                    float travel_x=dx[middle],travel_y=dy[middle];
                    float drift=std::hypot(travel_x,travel_y);
                    if(drift>.12f && drift<4.f){
                        for(unsigned frame_index=1;frame_index<decoded.frames;++frame_index){
                            float phase=float(frame_index)/float(decoded.frames-1);
                            for(unsigned bone=0;bone<decoded.bones;++bone){
                                auto at=(std::size_t(frame_index)*decoded.bones+bone)*16+12;
                                decoded.palettes[at]-=travel_x*phase;
                                decoded.palettes[at+1]-=travel_y*phase;
                            }
                            for(std::size_t bone=0;bone<decoded.rig.parents.size();++bone)
                                if(decoded.rig.parents[bone]<0){
                                    auto& root=decoded.rig.poses[std::size_t(frame_index)*decoded.rig.parents.size()+bone];
                                    root.position[0]-=travel_x*phase;root.position[1]-=travel_y*phase;
                                }
                        }
                    }
                }

        result->mesh=std::make_shared<AnimationMesh const>(std::move(decoded));
    }else{
        if(payload.size()<156 || std::memcmp(payload.data(),"DDS ",4) ||
           std::memcmp(payload.data()+84,"DX10",4)){result->failed=true;return result;}
        result->dds=std::move(payload);
    }
    if(cancelled.load(std::memory_order_relaxed))return {};
    if(result->bytes()>render_core::ContentPreparation<std::uint64_t,UnitAssetInput,UnitAssetContent>::byte_limit){result->mesh.reset();result->dds.clear();result->dds.shrink_to_fit();result->failed=true;}
    return result;
}
using UnitAssetPreparation=render_core::ContentPreparation<std::uint64_t,UnitAssetInput,UnitAssetContent>;
}
