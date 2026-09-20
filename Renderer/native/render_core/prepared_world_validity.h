#pragma once
namespace c3x_renderer { namespace render_core {
// Use the caller's exclusive river scratch; all other inputs belong to the
// immutable world/observation lease shared with the compiler workers.
template<class Result,class World,class Observations,class Rivers>
bool prepared_world_valid(Result const& result,World const& coast,Observations const& observations,Rivers& rivers){
    if(!result.ground || !result.terrain || !result.objects)return false;
    auto valid=[&](auto const& part){
        for(auto const& input:part.world)if(coast.world().at(input.first)!=input.second)return false;
        for(auto const& input:part.coast)if(coast.node_revision(input.first)!=input.second)return false;
        return true;
    };
    for(auto const* part:{&result.ground->topology,&result.objects->topology})for(auto const& input:*part){
        auto current=observations.current(input.first);if((current?current->semantic:0)!=input.second)return false;
    }
    typename Rivers::CellProof ground(result.ground->rivers.begin(),result.ground->rivers.end());
    return valid(*result.ground) && valid(*result.terrain) && valid(*result.objects) &&
        rivers.valid(ground) && rivers.valid(result.terrain->rivers) && rivers.valid(result.objects->rivers);
}
}}
