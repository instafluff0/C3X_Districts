#pragma once
#include "../animation_runtime.h"
#include <limits>
#include <new>

namespace c3x_renderer { namespace render_core {
// CPU-only metadata: no palette, vertex payload, GPU binding or asset lease.
// A spherical envelope covers every yaw and every local-joint interpolation,
// including interrupted transitions; endpoint posed AABBs cannot do that.
struct UnitContributionBounds {double radius=0;bool known=false;};
struct UnitMeshContributionBounds {
    bool known=false;
    double baked_radius=0,weight_sum=1;
    std::array<std::uint8_t,32> binding{};
    std::vector<int> parents;
    std::vector<double> translation,scale,bind_radius;
    std::vector<unsigned> skin_joints;
    std::size_t bytes()const{return sizeof(*this)+parents.capacity()*sizeof(int)+
        (translation.capacity()+scale.capacity()+bind_radius.capacity())*sizeof(double)+skin_joints.capacity()*sizeof(unsigned);}
    static double norm(double x,double y,double z){return std::sqrt(x*x+y*y+z*z);}
    static double linear_norm(float const* m,unsigned stride){
        double rows=0,cols=0;
        for(unsigned i=0;i<3;++i){double r=0,c=0;for(unsigned j=0;j<3;++j){
            if(!std::isfinite(m[i*stride+j])||!std::isfinite(m[j*stride+i]))return INFINITY;
            r+=std::abs(double(m[i*stride+j]));c+=std::abs(double(m[j*stride+i]));}
            rows=std::max(rows,r);cols=std::max(cols,c);}
        return std::sqrt(rows*cols);
    }
    bool prepare(AnimationMesh const& mesh){
        *this={};if(!mesh.bones||mesh.bones>256||!mesh.frames||mesh.frames>4096||mesh.vertices.empty()||mesh.vertices.size()>65536||
            mesh.palettes.size()!=std::size_t(mesh.bones)*mesh.frames*16)return false;
        std::vector<double> vertex_radius(mesh.bones,0);
        for(auto const& v:mesh.vertices){double total=0;auto const& p=v.source.position;
            if(!std::isfinite(p[0])||!std::isfinite(p[1])||!std::isfinite(p[2]))return false;
            for(unsigned k=0;k<4;++k){if(v.joints[k]>=mesh.bones||!std::isfinite(v.weights[k])||v.weights[k]<0)return false;
                total+=v.weights[k];if(v.weights[k]>0)vertex_radius[v.joints[k]]=std::max(vertex_radius[v.joints[k]],norm(p[0],p[1],p[2]));}
            weight_sum=std::max(weight_sum,total);
        }
        // A generic mesh path may serve move and non-move actions. Extract
        // from authored data and cover BOTH the original and the exact existing
        // common locomotion-strip variant, without retaining either payload.
        std::vector<float> dx,dy;dx.reserve(mesh.bones);dy.reserve(mesh.bones);
        for(unsigned b=0;b<mesh.bones;++b){auto first=std::size_t(b)*16+12;
            auto last=(std::size_t(mesh.frames-1)*mesh.bones+b)*16+12;
            dx.push_back(mesh.palettes[last]-mesh.palettes[first]);dy.push_back(mesh.palettes[last+1]-mesh.palettes[first+1]);}
        auto middle=mesh.bones/2;std::nth_element(dx.begin(),dx.begin()+middle,dx.end());std::nth_element(dy.begin(),dy.begin()+middle,dy.end());
        float travel_x=dx[middle],travel_y=dy[middle],drift=std::hypot(travel_x,travel_y);
        bool strip=mesh.frames>1&&drift>.12f&&drift<4.f;
        for(unsigned f=0;f<mesh.frames;++f)for(unsigned b=0;b<mesh.bones;++b){
            auto* m=mesh.palettes.data()+(std::size_t(f)*mesh.bones+b)*16;
            auto r=vertex_radius[b]*linear_norm(m,4)+norm(m[12],m[13],m[14]);
            if(strip){float phase=float(f)/float(mesh.frames-1);
                float x=m[12]-travel_x*phase,y=m[13]-travel_y*phase;
                r=std::max(r,vertex_radius[b]*linear_norm(m,4)+norm(x,y,m[14]));}
            if(!std::isfinite(r))return false;baked_radius=std::max(baked_radius,r*weight_sum);
        }
        auto const& rig=mesh.rig;auto count=rig.parents.size();
        if(count){
            if(count>256||rig.poses.size()!=count*mesh.frames||rig.skin_joints.size()!=mesh.bones||rig.inverse_bind.size()!=mesh.bones)return false;
            binding=rig.binding;parents=rig.parents;skin_joints=rig.skin_joints;
            translation.assign(count,0);scale.assign(count,0);bind_radius.assign(mesh.bones,0);
            for(unsigned i=0;i<count;++i){if(parents[i]<-1||parents[i]>=int(i))return false;
                for(unsigned f=0;f<mesh.frames;++f){auto const& j=rig.poses[std::size_t(f)*count+i];
                    auto t=norm(j.position[0],j.position[1],j.position[2]);auto s=linear_norm(j.scale.data(),3);
                    if(strip&&parents[i]<0){float phase=float(f)/float(mesh.frames-1);
                        float x=j.position[0]-travel_x*phase,y=j.position[1]-travel_y*phase;t=std::max(t,norm(x,y,j.position[2]));}
                    if(!std::isfinite(t)||!std::isfinite(s))return false;
                    translation[i]=std::max(translation[i],t);scale[i]=std::max(scale[i],s);}
            }
            for(unsigned b=0;b<mesh.bones;++b)if(skin_joints[b]>=count)return false;
            for(auto const& v:mesh.vertices)for(unsigned k=0;k<4;++k)if(v.weights[k]>0){
                auto b=v.joints[k];auto const& m=rig.inverse_bind[b];auto const& p=v.source.position;
                double x=p[0]*double(m[0])+p[1]*double(m[4])+p[2]*double(m[8])+m[12];
                double y=p[0]*double(m[1])+p[1]*double(m[5])+p[2]*double(m[9])+m[13];
                double z=p[0]*double(m[2])+p[1]*double(m[6])+p[2]*double(m[10])+m[14];
                auto r=norm(x,y,z);if(!std::isfinite(r))return false;bind_radius[b]=std::max(bind_radius[b],r);
            }
        }
        known=std::isfinite(baked_radius)&&std::isfinite(weight_sum);return known;
    }
};
inline UnitContributionBounds unit_contribution_bounds(std::vector<UnitMeshContributionBounds const*> const& parts){
    if(parts.empty())return {};
    double radius=0;
    for(auto* part:parts){if(!part||!part->known)return {};radius=std::max(radius,part->baked_radius);
        if(part->parents.empty())continue;
        auto t=part->translation,s=part->scale;
        // Local position/scale mixes are convex. Merge all clips sharing the
        // rig before propagating the hierarchy, not their posed endpoints.
        for(auto* other:parts)if(other->binding==part->binding&&!other->parents.empty()){
            if(other->parents!=part->parents)return {};
            for(unsigned i=0;i<t.size();++i){t[i]=std::max(t[i],other->translation[i]);s[i]=std::max(s[i],other->scale[i]);}
        }
        std::vector<double> length(t.size()),offset(t.size());
        for(unsigned i=0;i<t.size();++i){auto parent=part->parents[i];double l=parent<0?1:length[unsigned(parent)],u=parent<0?0:offset[unsigned(parent)];
            // Outward roundoff allowance includes normalized quaternion math.
            length[i]=s[i]*l*1.0001;offset[i]=(t[i]*l+u)*1.0001+1e-6;
            if(!std::isfinite(length[i])||!std::isfinite(offset[i]))return {};
        }
        for(unsigned b=0;b<part->skin_joints.size();++b){auto i=part->skin_joints[b];
            radius=std::max(radius,(part->bind_radius[b]*length[i]+offset[i])*part->weight_sum);}
    }
    radius=radius*1.0001+1e-4;return {radius,std::isfinite(radius)};
}
struct UnitContributionRect {double left=0,top=0,right=0,bottom=0;};
enum UnitContributionMask : unsigned {unit_main_body=1,unit_ground_shadow=2,unit_reflection=4};
struct UnitContributionCandidate {
    bool visible=false;
    double anchor_x=0,anchor_y=0,projection_scale=1,model_scale=1,offset_z=0;
    UnitContributionBounds bounds;
    // Bounds on actual low-ground screen displacement; unknown admits all.
    bool ground_known=false;double ground_min=0,ground_max=0;
};
struct UnitContributionView {
    unsigned width=0,height=0;double zoom=1;
    bool shadow=false,reflection=false;
    double shadow_x=0,shadow_y=0;
    std::vector<UnitContributionRect> receivers; // already include water distortion/filter reach
};
// A current palette gives a tighter conservative certificate than the
// all-action admission sphere. Coordinates are pose-local reflected pixels;
// placement and water overlap are evaluated separately for each occurrence.
struct UnitReflectionBounds {
    UnitContributionRect local;
    bool complete=true,has_points=false;
    template<class Points> void append(Points const& points,std::size_t first,
            bool known,double weight_low,double weight_high,double offset_z){
        if(!known||first>=points.size()||!std::isfinite(weight_low)||!std::isfinite(weight_high)||
            weight_low<=0||weight_high<weight_low||!std::isfinite(offset_z)){complete=false;return;}
        constexpr double h=150.*128/224;
        UnitContributionRect part{INFINITY,INFINITY,-INFINITY,-INFINITY};
        for(auto i=first;i<points.size();++i){auto const& p=points[i];
            double x=(double(p[0])-p[1])*64,y=(double(p[0])+p[1])*32+double(p[2])*h;
            if(!std::isfinite(x)||!std::isfinite(y)){complete=false;return;}
            part.left=std::min(part.left,x);part.right=std::max(part.right,x);
            part.top=std::min(part.top,y);part.bottom=std::max(part.bottom,y);
        }
        // The decoder permits small weight-sum error. The weighted palette
        // position scales by that sum; the model's Z offset is applied after it.
        double y0=offset_z*h;
        auto weighted=[](double lo,double hi,double a,double b){return std::array<double,2>{
            std::min({lo*a,lo*b,hi*a,hi*b}),std::max({lo*a,lo*b,hi*a,hi*b})};};
        auto x=weighted(part.left,part.right,weight_low,weight_high);
        auto y=weighted(part.top-y0,part.bottom-y0,weight_low,weight_high);
        part={x[0],y[0]+y0,x[1],y[1]+y0};
        double margin=1e-3+1e-4*std::max({std::abs(part.left),std::abs(part.top),std::abs(part.right),std::abs(part.bottom)});
        if(!std::isfinite(margin)){complete=false;return;}
        part.left-=margin;part.top-=margin;part.right+=margin;part.bottom+=margin;
        if(!has_points)local=part;
        else {local.left=std::min(local.left,part.left);local.top=std::min(local.top,part.top);
            local.right=std::max(local.right,part.right);local.bottom=std::max(local.bottom,part.bottom);}
        has_points=true;
    }
    bool overlaps(UnitContributionView const& view,double anchor_x,double anchor_y,
            double projection_scale,double ground_pixels)const{
        // Hand-authored plans and unavailable certificates retain the early
        // conservative choice. No missing metadata may discard a reflection.
        if(!complete||!has_points||local.left>local.right||local.top>local.bottom||!view.width||!view.height||!std::isfinite(view.zoom)||view.zoom<1||view.zoom>3||
            !std::isfinite(anchor_x)||!std::isfinite(anchor_y)||!std::isfinite(projection_scale)||projection_scale<=0||
            !std::isfinite(ground_pixels))return true;
        double cx=view.width/2,cy=view.height/2;
        UnitContributionRect screen{
            cx+(anchor_x+local.left*projection_scale-cx)*view.zoom+8-4,
            cy+(anchor_y+ground_pixels+local.top*projection_scale-cy)*view.zoom+8-4,
            cx+(anchor_x+local.right*projection_scale-cx)*view.zoom+8+4,
            cy+(anchor_y+ground_pixels+local.bottom*projection_scale-cy)*view.zoom+8+4};
        if(!std::isfinite(screen.left)||!std::isfinite(screen.top)||!std::isfinite(screen.right)||!std::isfinite(screen.bottom))return true;
        for(auto const& receiver:view.receivers){
            if(!std::isfinite(receiver.left)||!std::isfinite(receiver.top)||!std::isfinite(receiver.right)||!std::isfinite(receiver.bottom)||
                receiver.left>receiver.right||receiver.top>receiver.bottom)return true;
            if(screen.left<receiver.right&&screen.right>receiver.left&&screen.top<receiver.bottom&&screen.bottom>receiver.top)return true;
        }
        return false;
    }
};
struct UnitContributionPlan {
    struct Entry {unsigned candidate=0,mask=0;};
    std::vector<Entry> entries;
    UnitContributionView view; // receiver/camera snapshot for current-pose refinement
    unsigned main=0,shadow=0,reflection=0;
    bool valid=true;
    static bool overlap(UnitContributionRect const& a,UnitContributionRect const& b){
        return a.left<b.right&&a.right>b.left&&a.top<b.bottom&&a.bottom>b.top;
    }
    static unsigned select(UnitContributionCandidate const& c,UnitContributionView const& v){
        if(!c.visible)return 0;
        unsigned conservative=unit_main_body|(v.shadow?unit_ground_shadow:0)|
            (v.reflection&&!v.receivers.empty()?unit_reflection:0);
        if(!c.bounds.known||!c.ground_known||!std::isfinite(c.bounds.radius)||c.bounds.radius<0||
            !std::isfinite(c.anchor_x)||!std::isfinite(c.anchor_y)||!std::isfinite(c.projection_scale)||c.projection_scale<=0||
            !std::isfinite(c.model_scale)||!std::isfinite(c.offset_z)||!std::isfinite(c.ground_min)||!std::isfinite(c.ground_max)||
            c.ground_min>c.ground_max||!std::isfinite(v.zoom)||v.zoom<1||v.zoom>3||!v.width||!v.height)return conservative;
        constexpr double h=150.*128/224;
        double r=c.bounds.radius*std::abs(c.model_scale),z=c.offset_z*c.model_scale,s=c.projection_scale;
        double cx=v.width/2,cy=v.height/2;
        auto rect=[&](double x,double y,double rx,double ry,double ground_lo,double ground_hi,double guard){
            return UnitContributionRect{cx+(x-rx-cx)*v.zoom+guard-4,
                cy+(y-ry+ground_lo-cy)*v.zoom+guard-4,
                cx+(x+rx-cx)*v.zoom+guard+4,
                cy+(y+ry+ground_hi-cy)*v.zoom+guard+4};};
        auto body=rect(c.anchor_x,c.anchor_y-s*h*z,s*64*std::sqrt(2.)*r,
            s*std::sqrt(2*32.*32+h*h)*r,-c.ground_max,-c.ground_min,4);
        UnitContributionRect canvas{0,0,double(v.width)+8,double(v.height)+8};
        unsigned mask=overlap(body,canvas)?unit_main_body:0;
        if(v.shadow){
            if(!std::isfinite(v.shadow_x)||!std::isfinite(v.shadow_y))mask|=unit_ground_shadow;
            else {double a=v.shadow_x,b=v.shadow_y;
                auto cast=rect(c.anchor_x+s*64*(a-b)*z,c.anchor_y+s*32*(a+b)*z,
                    s*64*std::sqrt(2+(a-b)*(a-b))*r,s*32*std::sqrt(2+(a+b)*(a+b))*r,-c.ground_max,-c.ground_min,4);
                if(overlap(cast,canvas))mask|=unit_ground_shadow;}
        }
        if(v.reflection&&!v.receivers.empty()){
            auto mirror=rect(c.anchor_x,c.anchor_y+s*h*z,s*64*std::sqrt(2.)*r,
                s*std::sqrt(2*32.*32+h*h)*r,c.ground_min,c.ground_max,8);
            for(auto const& receiver:v.receivers)if(overlap(mirror,receiver)){mask|=unit_reflection;break;}
        }
        return mask;
    }
    static UnitContributionPlan build(std::vector<UnitContributionCandidate> const& candidates,UnitContributionView const& view){
        UnitContributionPlan plan;plan.view=view;
        for(unsigned i=0;i<candidates.size();++i){auto mask=select(candidates[i],view);if(!mask)continue;
            plan.entries.push_back({i,mask});plan.main+=bool(mask&unit_main_body);
            plan.shadow+=bool(mask&unit_ground_shadow);plan.reflection+=bool(mask&unit_reflection);}
        return plan;
    }
    template<class Poses> auto required(Poses const& candidates)const {
        Poses out;out.reserve(entries.size());for(auto const& e:entries)out.push_back(candidates.at(e.candidate));return out;
    }
};
} }
