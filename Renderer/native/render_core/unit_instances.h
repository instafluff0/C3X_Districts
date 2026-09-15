#pragma once
#include "unit_playback.h"
#include <map>
#include <cstdint>

namespace c3x_renderer { namespace render_core {
// Serialized by RendererWorker's caller gate. Instances own copied content and
// compiled catalog indices; selections own occurrence/projection data. Neither
// an instance nor a cached pose grants visibility. Only an explicit selection
// from the native capture may feed a pass.
class UnitInstances {
public:
    struct Selection {
        int id=-1;
        std::uint64_t revision=0;
        c3x_renderer_unit_v1 occurrence{};
    };
private:
    struct Instance {
        c3x_renderer_unit_v1 content{};
        unsigned flags=0;
        std::size_t unit=0,action=0;
        std::uint64_t revision=0,used=0;
    };
    std::map<int,Instance> instances;
    UnitPlayback playback;
    std::uint64_t serial=0,access=0;
    // CPU identity metadata only. Shared meshes and completed poses retain
    // their existing independent budgets. Eviction invalidates old selections.
    std::size_t capacity;
public:
    std::uint64_t captures=0,reused=0,bindings=0,evictions=0;
    explicit UnitInstances(std::size_t limit=4096):capacity(limit){}
    std::size_t size()const{return instances.size();}
    void forget(int id){instances.erase(id);playback.forget(id);}
    void clear(){instances.clear();playback.clear();} // serial never reuses a token

    template<class Catalog, class ActionName>
    bool capture(c3x_renderer_unit_v1 request,unsigned flags,Catalog const& catalog,
                 ActionName action_name,Selection& selected) {
        selected={};
        if(request.struct_size!=sizeof(request)||request.unit_key[63]||request.unit_id<0||
           (flags&~3u)||!capacity)return false;
        ++captures;
        bool captured=(flags&C3X_RENDERER_UNIT_STATE_CAPTURED)!=0;
        if(captured && !(flags&C3X_RENDERER_UNIT_SELECTED) && request.action==8)request.action=1;
        auto found=instances.find(request.unit_id);
        Instance value{};
        bool bound=found!=instances.end() && found->second.content.action==request.action &&
            !std::memcmp(found->second.content.unit_key,request.unit_key,64);
        if(bound)value=found->second;
        else {
            auto name=action_name(request.action);
            if(!name){forget(request.unit_id);return false;}
            auto unit=std::find_if(catalog.begin(),catalog.end(),[&](auto const& u){
                return std::find(u.keys.begin(),u.keys.end(),request.unit_key)!=u.keys.end();});
            if(unit==catalog.end()){forget(request.unit_id);return false;}
            auto action=std::find_if(unit->actions.begin(),unit->actions.end(),[&](auto const& a){return a.name==name;});
            if(action==unit->actions.end()){forget(request.unit_id);return false;}
            value.unit=std::size_t(unit-catalog.begin());value.action=std::size_t(action-unit->actions.begin());++bindings;
        }
        // Catalog replacement clears this owner before old indices can escape.
        auto const& clip=catalog[value.unit].actions[value.action];
        auto content=request;
        content.body_x=content.body_y=content.sprite_width=content.sprite_height=0;
        content.reduced=content.projection_scale_milli=0;
        content.presentation_time_ticks=content.presentation_frequency=0;
        bool ambient=captured && clip.ambient && (request.action==1 || request.action==11 ||
            (request.action>=13 && request.action<=18));
        if(ambient)content.action_cursor=content.frame_count=0;
        if(found!=instances.end() && found->second.flags==flags &&
           !std::memcmp(&found->second.content,&content,sizeof(content))) {
            value.revision=found->second.revision;++reused;
        }else value.revision=++serial;
        value.content=content;value.flags=flags;value.used=++access;
        if(found==instances.end() && instances.size()>=capacity){
            auto oldest=std::min_element(instances.begin(),instances.end(),[](auto const& a,auto const& b){return a.second.used<b.second.used;});
            forget(oldest->first);++evictions;
        }
        instances[request.unit_id]=value;
        selected={request.unit_id,value.revision,request};return true;
    }

    // Clock sampling requires no new native body call. Directed actions retain
    // their captured cursor/anchor; only authored eligible ambient loops advance.
    template<class Catalog>
    bool sample(Selection const& selected,long long ticks,long long frequency,Catalog const& catalog,
                c3x_renderer_unit_v1& output,unsigned& predict) {
        auto found=instances.find(selected.id);
        if(found==instances.end() || found->second.revision!=selected.revision)return false;
        auto const& instance=found->second;
        if(instance.unit>=catalog.size()||instance.action>=catalog[instance.unit].actions.size())return false;
        auto const& clip=catalog[instance.unit].actions[instance.action];
        output=selected.occurrence;output.presentation_time_ticks=ticks;output.presentation_frequency=frequency;predict=1;
        if(instance.flags&C3X_RENDERER_UNIT_STATE_CAPTURED){
            bool selected_unit=(instance.flags&C3X_RENDERER_UNIT_SELECTED)!=0;
            if(!playback.resolve(output,clip,selected_unit,predict))predict=0;
            if(!predict && !clip.ambient && output.action==1){output.action_cursor=0;output.frame_count=1;}
        }
        return true;
    }
    template<class Catalog> bool animated(Selection const& selected,Catalog const& catalog)const{
        auto found=instances.find(selected.id);
        if(found==instances.end()||found->second.revision!=selected.revision)return false;
        auto const& instance=found->second;
        if(!(instance.flags&C3X_RENDERER_UNIT_STATE_CAPTURED)||instance.unit>=catalog.size()||instance.action>=catalog[instance.unit].actions.size())return false;
        auto const& clip=catalog[instance.unit].actions[instance.action];int action=instance.content.action;
        return clip.ambient&&clip.loop&&((action==1&&(instance.flags&C3X_RENDERER_UNIT_SELECTED))||action==11||(action>=13&&action<=18));
    }
    template<class Catalog>
    auto definition(Selection const& selected,Catalog const& catalog)const -> typename Catalog::value_type const* {
        auto found=instances.find(selected.id);
        return found!=instances.end() && found->second.revision==selected.revision && found->second.unit<catalog.size()
            ? &catalog[found->second.unit] : nullptr;
    }
};
}}
