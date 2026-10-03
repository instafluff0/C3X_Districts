#pragma once
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Catalogue identities only. This plan reads no native instance or pose data.
struct UnitSourcePlan {
    struct Action {std::size_t unit=0,action=0;};
    std::vector<std::size_t> types;
    std::vector<Action> actions;
    std::vector<bool> meshes,textures,mixed_motion;
    static constexpr std::size_t type_limit=128,action_limit=4096,resource_limit=8192;
    // A queried headroom is additional capacity, while owner limits bound the
    // complete retained union. Surviving source bytes must not be charged twice.
    static std::size_t owner_allowance(std::uint64_t growth,std::size_t owned,std::size_t ceiling){
        if(owned>=ceiling)return ceiling;
        return owned+std::size_t(std::min<std::uint64_t>(growth,ceiling-owned));
    }
    template<class Catalogue> bool prepare(Catalogue const& catalogue,
            std::vector<std::size_t> indices,bool all){
        if(catalogue.units.size()>type_limit || catalogue.meshes.size()+catalogue.textures.size()>resource_limit)return false;
        if(all){indices.resize(catalogue.units.size());for(std::size_t i=0;i<indices.size();++i)indices[i]=i;}
        if(indices.size()>type_limit)return false;
        std::sort(indices.begin(),indices.end());indices.erase(std::unique(indices.begin(),indices.end()),indices.end());
        UnitSourcePlan next;next.types=std::move(indices);
        next.meshes.assign(catalogue.meshes.size(),false);next.textures.assign(catalogue.textures.size(),false);
        std::vector<unsigned char> motion(catalogue.meshes.size(),0);
        for(auto type:next.types){
            if(type>=catalogue.units.size())return false;
            auto const& unit=catalogue.units[type];if(unit.actions.empty())return false;
            for(std::size_t i=0;i<unit.actions.size();++i){
                if(next.actions.size()==action_limit)return false;
                auto const& action=unit.actions[i];if(action.parts.empty())return false;
                next.actions.push_back({type,i});
                for(auto const& part:action.parts){
                    if(part.mesh>=next.meshes.size() || part.texture>=next.textures.size())return false;
                    next.meshes[part.mesh]=true;motion[part.mesh]|=action.name=="move"?1:2;
                    unsigned requested_textures[]={part.texture,part.material_textures[0],part.material_textures[1],part.material_textures[2],part.material_textures[3]};
                    for(auto texture:requested_textures)if(texture!=UINT32_MAX){
                        if(texture>=next.textures.size())return false;next.textures[texture]=true;}
                }
            }
        }
        next.mixed_motion.resize(motion.size());for(std::size_t i=0;i<motion.size();++i)next.mixed_motion[i]=motion[i]==3;
        *this=std::move(next);return true;
    }
};
} }
