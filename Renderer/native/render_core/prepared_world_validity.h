#pragma once
#include <cstdint>
namespace c3x_renderer {
enum class WorldPreparationKind : std::uint32_t { combined=0, ground=1, objects=2 };
inline bool world_preparation_kind_valid(WorldPreparationKind kind){
    return kind==WorldPreparationKind::combined || kind==WorldPreparationKind::ground || kind==WorldPreparationKind::objects;
}
inline bool world_preparation_needs_ground(WorldPreparationKind kind){
    return kind==WorldPreparationKind::combined || kind==WorldPreparationKind::ground;
}
inline bool world_preparation_needs_objects(WorldPreparationKind kind){
    return kind==WorldPreparationKind::combined || kind==WorldPreparationKind::objects;
}
namespace render_core {
// This observation read supplies only water-family depth. Presence has its own
// value so a missing observation cannot alias any signed ground family.
template<class Observation> std::uint64_t ground_topology_value(Observation const* observation){
    return observation?std::uint64_t(std::uint32_t(observation->ground))+1:0;
}
// Use the caller's exclusive river scratch; all other inputs belong to the
// immutable world/observation lease shared with the compiler workers.
template<class Result,class World,class Observations,class Rivers>
bool prepared_world_valid(Result const& result,World const& coast,Observations const& observations,Rivers& rivers){
    if(!world_preparation_kind_valid(result.kind))return false;
    bool ground=world_preparation_needs_ground(result.kind),objects=world_preparation_needs_objects(result.kind);
    if((ground && (!result.ground || !result.terrain)) || (objects && !result.objects))return false;
    auto valid=[&](auto const& part){
        for(auto const& input:part.world)if(coast.world().at(input.first)!=input.second)return false;
        for(auto const& input:part.coast)if(coast.node_revision(input.first)!=input.second)return false;
        return true;
    };
    if(ground){
        for(auto const& input:result.ground->topology){
            auto current=observations.current(input.first);
            auto value=result.kind==WorldPreparationKind::ground?ground_topology_value(current):(current?current->semantic:0);
            if(value!=input.second)return false;
        }
        typename Rivers::CellProof proof(result.ground->rivers.begin(),result.ground->rivers.end());
        if(!valid(*result.ground) || !valid(*result.terrain) || !rivers.valid(proof) || !rivers.valid(result.terrain->rivers))return false;
    }
    if(objects){
        for(auto const& input:result.objects->topology){
            auto current=observations.current(input.first);if((current?current->semantic:0)!=input.second)return false;
        }
        if(!valid(*result.objects) || !rivers.valid(result.objects->rivers))return false;
    }
    return true;
}
}}
