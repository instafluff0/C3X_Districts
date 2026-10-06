#pragma once
#include "../animation_runtime.h"
#include <map>
#include <utility>

namespace c3x_renderer { namespace render_core {
using JointPose = AnimationJointPose;
using JointMatrix = std::array<float,16>;

inline JointMatrix joint_matrix(JointPose const& pose){
    auto const& q=pose.rotation;float x=q[0],y=q[1],z=q[2],w=q[3];
    float r[9]={1-2*y*y-2*z*z,2*x*y+2*z*w,2*x*z-2*y*w,
        2*x*y-2*z*w,1-2*x*x-2*z*z,2*y*z+2*x*w,
        2*x*z+2*y*w,2*y*z-2*x*w,1-2*x*x-2*y*y};
    JointMatrix out{};out[15]=1;
    for(unsigned row=0;row<3;++row)for(unsigned col=0;col<3;++col)
        for(unsigned k=0;k<3;++k)out[row*4+col]+=pose.scale[row*3+k]*r[k*3+col];
    std::copy(pose.position.begin(),pose.position.end(),out.begin()+12);return out;
}
inline JointMatrix joint_multiply(JointMatrix const& a,JointMatrix const& b){
    JointMatrix out{};
    for(unsigned r=0;r<4;++r)for(unsigned c=0;c<4;++c)
        for(unsigned k=0;k<4;++k)out[r*4+c]+=a[r*4+k]*b[k*4+c];
    return out;
}
inline JointPose mix_joint(JointPose const& a,JointPose const& b,float t){
    JointPose out;
    for(unsigned i=0;i<3;++i)out.position[i]=a.position[i]+(b.position[i]-a.position[i])*t;
    for(unsigned i=0;i<9;++i)out.scale[i]=a.scale[i]+(b.scale[i]-a.scale[i])*t;
    float dot=0;for(unsigned i=0;i<4;++i)dot+=a.rotation[i]*b.rotation[i];
    float sign=dot<0?-1.f:1.f;dot=std::clamp(std::abs(dot),0.f,1.f);
    float wa=1-t,wb=t;
    if(dot<.9995f){float angle=std::acos(dot),denom=std::sin(angle);
        wa=std::sin((1-t)*angle)/denom;wb=std::sin(t*angle)/denom;}
    float length=0;
    for(unsigned i=0;i<4;++i){out.rotation[i]=wa*a.rotation[i]+wb*sign*b.rotation[i];length+=out.rotation[i]*out.rotation[i];}
    for(auto& q:out.rotation)q/=std::sqrt(length);
    return out;
}

// Only local joints are mixed. World travel, camera placement, native action
// phase and gameplay remain untouched. One sampled palette serves every pass.
class UnitPoseTransitions {
    struct State {
        std::uint64_t incarnation=0;
        int action=-1;
        long long started=0,ticks=-1,frequency=0;
        bool blending=false;
        double duration=.12;
        unsigned sampled_frame=UINT32_MAX;std::uint64_t sampled_source=0;
        std::vector<JointPose> from,current;
        std::array<float,4096> palette{};
    };
    using Key=std::pair<int,std::array<std::uint8_t,32>>;
    std::map<Key,State> states;
    struct Facing {
        std::uint64_t incarnation=0;
        long long ticks=-1,started=0,frequency=0;
        float from=0,target=0,current=0;
        double duration=.12;
    };
    std::map<int,Facing> facings;
public:
    void clear(){states.clear();facings.clear();}
    std::size_t size()const{return states.size();}
    template<class Visible>void retain(Visible const& visible){
        for(auto it=facings.begin();it!=facings.end();){
            bool found=std::any_of(visible.begin(),visible.end(),[&](auto const& p){
                return p.draw.unit_id==it->first&&p.pose_identity==it->second.incarnation;});
            if(found)++it;else it=facings.erase(it);
        }
        for(auto it=states.begin();it!=states.end();){
            bool found=std::any_of(visible.begin(),visible.end(),[&](auto const& p){
                return p.draw.unit_id==it->first.first&&p.pose_identity==it->second.incarnation;});
            if(found)++it;else it=states.erase(it);
        }
    }
    void finish(long long ticks){
        for(auto it=states.begin();it!=states.end();)
            if(it->second.ticks!=ticks)it=states.erase(it);else ++it;
    }
    float facing(int id,std::uint64_t incarnation,long long ticks,long long frequency,float target){
        if(frequency<=0||!std::isfinite(target))return target;
        auto found=facings.find(id);
        if(found==facings.end()||found->second.incarnation!=incarnation||found->second.frequency!=frequency||ticks<found->second.ticks){
            if(found==facings.end()&&facings.size()>=4096)return target;
            facings[id]={incarnation,ticks,ticks,frequency,target,target,target};return target;
        }
        auto& s=found->second;
        float delta=std::remainder(target-s.target,6.28318530718f);
        if(std::abs(delta)>1e-5f){
            s.from=s.current;s.target=s.from+std::remainder(target-s.from,6.28318530718f);s.started=ticks;
            s.duration=.06+.12*std::abs(s.target-s.from)/3.14159265359;
        }
        float t=float(std::clamp(double(ticks-s.started)/frequency/s.duration,0.,1.));t=t*t*(3-2*t);
        s.current=s.from+(s.target-s.from)*t;s.ticks=ticks;return s.current;
    }
    // A null palette uses the original immutable GPU frame, including legacy
    // packs without rig metadata. This never invokes CPU vertex rendering.
    float const* sample(int id,std::uint64_t incarnation,int action,
            long long ticks,long long frequency,AnimationMesh const& mesh,unsigned frame,std::uint64_t source_identity=0){
        auto const& rig=mesh.rig;std::size_t count=rig.parents.size();
        if(!count||frame>=mesh.frames||frequency<=0||ticks<0)return nullptr;
        auto key=Key{id,rig.binding};auto found=states.find(key);
        if(found==states.end()){
            if(states.size()>=4096)return nullptr;
            found=states.emplace(key,State{}).first;
        }
        auto& state=found->second;
        bool fresh=state.incarnation!=incarnation||state.frequency!=frequency||state.current.size()!=count||ticks<state.ticks;
        if(fresh)state=State{};
        if(!fresh&&state.ticks==ticks&&state.action==action)
            return state.blending?state.palette.data():nullptr;
        // Caller identity is an immutable catalogue mesh, independent of the
        // camera and native occurrence. Touch before finish() can retire it.
        if(!fresh&&source_identity&&state.sampled_source==source_identity&&state.sampled_frame==frame&&
                state.action==action&&!state.blending){state.ticks=ticks;return nullptr;}
        bool changed=!fresh&&state.action!=action;
        if(changed){
            state.duration=state.action==2&&action==1?.20:std::min(.12,double(mesh.duration)*.2);
            state.from=state.current;state.started=ticks;state.blending=true;
        }
        state.incarnation=incarnation;state.action=action;state.ticks=ticks;state.frequency=frequency;
        state.current.assign(rig.poses.begin()+frame*count,rig.poses.begin()+(frame+1)*count);
        state.sampled_frame=frame;state.sampled_source=source_identity;
        // Destination time advances during the blend. A new interruption takes
        // the last displayed mixed pose as its source, not an old clip endpoint.
        float t=state.blending?float(std::clamp(double(ticks-state.started)/frequency/state.duration,0.,1.)):1.f;
        state.blending=t<1;
        if(!state.blending){state.from.clear();return nullptr;}
        t=t*t*(3-2*t);
        std::array<JointMatrix,256> worlds{};
        for(std::size_t i=0;i<count;++i){
            state.current[i]=mix_joint(state.from[i],state.current[i],t);
            worlds[i]=joint_matrix(state.current[i]);
            if(rig.parents[i]>=0)worlds[i]=joint_multiply(worlds[i],worlds[rig.parents[i]]);
        }
        for(unsigned i=0;i<mesh.bones;++i){
            auto palette=joint_multiply(rig.inverse_bind[i],worlds[rig.skin_joints[i]]);
            std::copy(palette.begin(),palette.end(),state.palette.begin()+i*16);
        }
        return state.palette.data();
    }
};
} }
