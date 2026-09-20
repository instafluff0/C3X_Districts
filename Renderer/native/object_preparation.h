#pragma once
#include "rigid_object_instance.h"
#include "city_fidelity/compiler.h"
#include "source_fidelity/terrain_compiler.h"
namespace c3x_renderer { namespace objects {
// Inputs own capture scalars. Packs, observations and world/coast are immutable
// until the frame lease joins; all mutable query state belongs to this lane.
struct PreparationInput {
    Projection projection;
    int ground=0;
    std::int64_t world_revision=0;
    bool river_ready=false,skip_flat_shore=true,separate_relief=true,retain_height=true;
    bool route_ready=false,routes_enabled=true,mine_ready=false,farm_ready=false,city_ready=false,composition_ready=false;
    bool shared_rigid=false;
};
struct PreparedPart {
    render_core::PreparedMesh mesh;
    unsigned vertex_offset=0,index_offset=0,material=0;
    bool environment=false,terrain_conforming=false;
    std::array<float,4> atlas{};
    std::shared_ptr<city_fidelity::Lighting> lighting;
};
struct PreparedObjects {
    std::array<PreparedPart,layer_count> layers;
    std::vector<PreparedPart> city;
    std::vector<PreparedRigid> rigid;
    unsigned composition=~0u,instances=0,routes=0;
    std::shared_ptr<void> buffer;
    std::size_t gpu_bytes=0;
    std::unordered_map<std::uint64_t,std::uint64_t> topology,coast;
    std::unordered_map<std::size_t,std::uint32_t> world;
    fidelity::NaturalWorld::CellProof rivers;
    std::size_t proof_bytes=0;
    std::size_t bytes()const {
        std::size_t total=sizeof(*this)+city.capacity()*sizeof(PreparedPart)+rigid.capacity()*sizeof(PreparedRigid)+gpu_bytes+proof_bytes;
        for(auto const& part:layers)total+=part.mesh.bytes();
        for(auto const& part:city)total+=part.mesh.bytes();
        if(!city.empty())total+=sizeof(city_fidelity::Lighting)+city.front().lighting->lights.capacity()*sizeof(city_fidelity::Light)+
            city.front().lighting->blockers.capacity()*sizeof(city_fidelity::Lighting::Box);
        return total+(world.size()+coast.size()+topology.size())*64+
            (world.bucket_count()+coast.bucket_count()+topology.bucket_count())*sizeof(void*);
    }
};
using Preparation=render_core::ContentPreparation<unsigned,PreparationInput,PreparedObjects>;
// Destruction order joins the producer before scratch or source leases die.
struct PreparationLease {
    fidelity::TerrainCompileScratch scratch;
    Preparation queue;
};
template<class TerrainAssets,class Observations,class Stop>
std::unique_ptr<PreparedObjects> prepare(PreparationInput const& input,Assets const& assets,
        city_fidelity::Library const& library,fidelity::NaturalData const& natural,
        TerrainAssets const& terrain_assets,render_core::WorldCoast const& world_coast,
        Observations const& observations,fidelity::TerrainCompileScratch& scratch,Stop stop,bool bounded=true) {
    using namespace fidelity;
    if(stop())return {};
    auto result=std::make_unique<PreparedObjects>();
    auto const& capture=input.projection.tile;
    if(!capture.road_mask && !capture.railroad_mask && capture.city_id<0 &&
        !(capture.improvement_flags&(C3X_RENDERER_IMPROVEMENT_MINE|C3X_RENDERER_IMPROVEMENT_IRRIGATION|
            C3X_RENDERER_IMPROVEMENT_GOODY_HUT|C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP)))return result;
    scratch.bind(natural,world_coast.world(),input.world_revision);
    NaturalWorld::CellInputs river_inputs;
    NaturalWorld::DependencyScope scope(scratch.rivers,&river_inputs);
    scratch.pickup.clear();scratch.heights.clear();
    auto observe_world=[&](std::size_t index,std::uint32_t value){result->world.emplace(index,value);};
    auto observe_coast=[&](std::uint64_t index,std::uint64_t revision){result->coast.emplace(index,revision);};
    SurfaceQueries queries(world_coast,scratch.shores,input.projection.tile.tile_x,input.projection.tile.tile_y,observe_world,observe_coast,input.skip_flat_shore);
    auto world_lookup=[&](int c,int r){return queries.tile(c,r);};
    auto lookup_natural=[&](int c,int r){return queries.natural_tile(c,r);};
    auto shore_sample_at=[&](float x,float y){return queries.shore(x,y);};
    auto material_weights_for=[&](float x,float y){return queries.weights(x,y);};
    auto relief_sample=[&](int kind,unsigned variant,int channel,float u,float v){return relief_source(terrain_assets,true,kind,variant,channel,u,v);};
    auto river=[&](int c,int r,float u,float v){
        auto const& world=world_coast.world();auto i=world.index(c,r);auto value=world.at(i);
        if(i!=std::size_t(-1))observe_world(i,value);
        if(value==0xffffffffu || !(value>>16&170u) || !input.river_ready)return 1000.f;
        return float(scratch.rivers.river_sample({float(c)+u,float(r)+1-v}).distance);
    };
    auto dune=[](float,float){return 0.f;};
    auto activity=[&](int c,int r){auto const& world=world_coast.world();auto i=world.index(c,r);auto value=world.at(i);
        if(i!=std::size_t(-1))observe_world(i,value);return value!=0xffffffffu && (value>>24)!=0?1.f:0.f;};
    int nc=(input.projection.tile.tile_x+input.projection.tile.tile_y)/2,nr=(input.projection.tile.tile_x-input.projection.tile.tile_y)/2;
    std::size_t height_queries=0;
    ReliefSurface pickup_surface(world_coast.world().dimensions(),nc,nr,
        shore_sample_at(queries.center_u,queries.center_v).distance,world_lookup,relief_sample,shore_sample_at,river,dune,activity,
        scratch.pickup,height_queries,input.separate_relief);
    auto pickup_height=[&](float x,float y){return pickup_surface.height(x,y);};
    auto height_natural=[&](float x,float y,float* support=nullptr){
        auto compute=[&]{std::array<float,2> value{};value[0]=queries.height(natural,pickup_height,x,y,&value[1]);return value;};
        auto value=input.retain_height?scratch.heights.get(x,y,compute):compute();
        if(support)*support=value[1];return value[0];
    };

    auto relief=[&](float u,float v){auto s=pickup_surface.sample(u,v);return std::array<float,3>{s.height,s.authored_height,s.authored_blend};};
    auto const& tile=input.projection.tile;
    auto composition=input.composition_ready?city_fidelity::select(library,tile,nc,nr,world_lookup,shore_sample_at,
        [&](float x,float y){return scratch.rivers.river_sample({x,y}).distance;},height_natural):nullptr;
    if(composition)result->composition=unsigned(composition-library.compositions.data());
    Plan plan;
    select_routes(tile,assets,input.route_ready,input.routes_enabled,[&](int x,int y){
        auto key=observations.key(x,y);auto record=observations.current(key);
        result->topology.emplace(key,record?record->semantic:0);return record;
    },plan);
    unsigned sites=tile.improvement_flags&(C3X_RENDERER_IMPROVEMENT_GOODY_HUT|C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP);
    if(!select_improvements(tile,assets,input.ground,sites,input.mine_ready,input.farm_ready,input.city_ready,composition!=nullptr,plan))return {};
    result->instances=unsigned(plan.instances.size());result->routes=unsigned(plan.routes.size());
    // Bound worker transients before expanded triangles are constructed. Large
    // valid packs can still use the same synchronous compiler on recovery.
    std::uint64_t raw_bytes=std::uint64_t(plan.routes.size())*96u*sizeof(Vertex);
    for(auto const& instance:plan.instances){
        if(instance.asset>=assets[instance.family].assets.size())continue; // same absent-asset behavior as append_instance
        auto const& asset=assets[instance.family].assets[instance.asset];
        raw_bytes+=(asset.indices.size()+asset.vertices.size())*sizeof(Vertex);}
    if(composition){
        raw_bytes+=(composition->paving.indices.size()+composition->paving.vertices.size())*sizeof(Vertex);
        for(auto const& instance:composition->instances)for(auto const& part:library.models[instance.model].parts)
            raw_bytes+=(part.indices.size()+part.vertices.size())*sizeof(Vertex);
    }
    if(stop() || (bounded && raw_bytes>32u*1024u*1024u))return {};
    Surfaces surfaces;
    if(input.shared_rigid){
        for(auto const& instance:plan.instances){
            if(stop())return {};
            if(instance.asset<assets[instance.family].assets.size())
                result->rigid.push_back(prepare_rigid(instance,input.projection,assets,relief,height_natural));
        }
        plan.instances.clear();
    }
    compile(plan,input.projection,assets,relief,height_natural,surfaces,true);
    for(unsigned layer=0;layer<layer_count;++layer){
        render_core::MeshFormat format;format.feature=layer!=route_layer;format.projection_kind=2;
        if(!render_core::prepare_mesh(surfaces.layers[layer],layer==route_layer?nullptr:&surfaces.indices[layer],format,result->layers[layer].mesh,stop))return {};
        std::vector<Vertex>().swap(surfaces.layers[layer]);
        if(bounded && result->bytes()>Preparation::byte_limit/2)return {};
    }
    if(composition){
        city_fidelity::Surfaces city;
        GroundProjection projection{nc,nr,input.projection.half_w,input.projection.half_h,
            input.projection.relief_projection_scale,float(input.projection.content_view_height)};
        if(!city_fidelity::compile(library,*composition,nc,nr,height_natural,projection,city,stop,true))return {};
        for(auto& chunk:city.chunks){
            PreparedPart part;part.material=chunk.material;part.environment=chunk.environment;
            part.terrain_conforming=chunk.terrain_conforming;part.lighting=chunk.lighting;
            std::copy(chunk.atlas,chunk.atlas+4,part.atlas.begin());
            render_core::MeshFormat format;format.projection_kind=4;format.city=true;
            if(!render_core::prepare_mesh(chunk.vertices,&chunk.indices,format,part.mesh,stop))return {};
            std::vector<Vertex>().swap(chunk.vertices);result->city.push_back(std::move(part));
            if(bounded && result->bytes()>Preparation::byte_limit/2)return {};
        }
    }
    if(stop())return {};
    result->rivers.assign(river_inputs.begin(),river_inputs.end());
    result->proof_bytes=scratch.rivers.proof_bytes(result->rivers);
    return result;
}
}}
