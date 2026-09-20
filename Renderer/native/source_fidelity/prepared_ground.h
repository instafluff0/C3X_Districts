#pragma once
#include "../../lab/shared/natural/world.h"
#include "ground_compiler.h"
#include "surface_query_scratch.h"
#include "../render_core/prepared_mesh.h"
#include "../render_core/scoped_preparation.h"
#include "../render_core/water_coverage.h"
#include <type_traits>

namespace c3x_renderer { namespace fidelity {
struct PreparedGround {
    // Underlay, land, bed, water, river, shadow; route is not ground-owned.
    std::array<render_core::PreparedMesh,6> meshes;
    std::vector<CachedGroundGrid> pending_grids;
    std::vector<MapVertex> legacy_shadow;
    std::unordered_map<std::size_t,std::uint32_t> world;
    std::unordered_map<std::uint64_t,std::uint64_t> coast,topology;
    NaturalWorld::CellInputs rivers;
    unsigned grid_hits=0;
    bool water_coverage=false;
    double compile_ms=0;
    std::size_t bytes() const {
        std::size_t size=sizeof(*this)+legacy_shadow.capacity()*sizeof(MapVertex);
        for(auto const& mesh:meshes)size+=mesh.bytes();
        size+=pending_grids.capacity()*sizeof(CachedGroundGrid);
        for(auto const& grid:pending_grids)size+=grid.vertices.capacity()*sizeof(MapVertex)+grid.samples.capacity()*sizeof(grid.samples[0]);
        size+=(world.size()+coast.size()+topology.size())*64+
            (world.bucket_count()+coast.bucket_count()+topology.bucket_count())*sizeof(void*);
        for(auto const& river:rivers)size+=sizeof(river)+64+sizeof(*river.second)+river.second->values.capacity()*sizeof(std::uint64_t)+
            sizeof(NaturalWorld::PageInputs)+river.second->inputs->flow.capacity()+river.second->inputs->values.capacity()*sizeof(std::pair<std::size_t,std::uint32_t>);
        return size;
    }
};
using GroundTask=render_core::ScopedPreparation<PreparedGround>;

template<class Stop>
bool pack_ground(GroundSurfaces& surfaces,bool pickup,bool world_ground,Stop stop,PreparedGround& result){
    result.water_coverage=render_core::water_surface_can_contribute(surfaces.underlay_vertices);
    std::vector<MapVertex>* vertices[]={&surfaces.underlay_vertices,&surfaces.land_vertices,&surfaces.bed_vertices,
        &surfaces.water_vertices,&surfaces.river_vertices,&surfaces.shadow_vertices};
    std::vector<unsigned>* indices[]={&surfaces.underlay_indices,&surfaces.land_indices,&surfaces.bed_indices,
        &surfaces.water_indices,&surfaces.river_indices,nullptr};
    if(!pickup)result.legacy_shadow=std::move(surfaces.shadow_vertices);
    for(unsigned layer=0;layer<6;++layer){
        render_core::MeshFormat format;format.pickup=pickup;format.projection_kind=world_ground && layer<5?3u:0u;
        if(!render_core::prepare_mesh(*vertices[layer],indices[layer],format,result.meshes[layer],stop))return false;
        std::vector<MapVertex>().swap(*vertices[layer]);
        if(indices[layer])std::vector<unsigned>().swap(*indices[layer]);
    }
    result.pending_grids=std::move(surfaces.pending_grids);
    return !stop();
}

// Borrowed inputs are const for the entire scoped task. Dependencies record
// actual reads, including missing observations, rather than a padded snapshot.
template<class Topology,class RiverNodes,class CoordinateKey,class Source,class Dune,
    class RiverDistance,class Relief,class Weights,class Shore,class SurfaceUV,class NdcX,class NdcY,class Stop>
std::unique_ptr<PreparedGround> prepare_ground(GroundCompileInput const& ground_compile_input,
        c3x_renderer_frame_v1 const& frame,NaturalData const& natural,SurfaceQueryScratch& ground_query_scratch,
        render_core::WorldCoast const& world_coast,Topology const& topology_cache,
        RiverNodes const& local_river_nodes,std::shared_ptr<std::vector<CachedGroundGrid>> const& ground_grid_lease,
        render_core::ShoreSample tile_center_shore,float ground_slot,bool skip_flat_shore,bool separate_natural_relief,
        CoordinateKey coordinate_key,Source pickup_source,Dune pickup_dune,RiverDistance river_distance,
        Relief relief_at_world,Weights material_weights_for,Shore signed_shore_distance,
        SurfaceUV periodic_surface_uv,NdcX ndc_x,NdcY ndc_y,Stop ground_cancelled){
    auto started=std::chrono::steady_clock::now();
    auto result=std::make_unique<PreparedGround>();
    auto const& tile=ground_compile_input.tile;
    bool pickup_profile=ground_compile_input.pickup_profile,fidelity_profile=ground_compile_input.fidelity_profile;
    bool river_assets_ready=ground_compile_input.river_assets_ready,world_ground=ground_compile_input.world_ground;
    using RiverNode=typename std::remove_cv<typename std::remove_pointer<typename RiverNodes::value_type>::type>::type;
    // Ground's own private query/height/river pipeline. The GroundTask
    // caller scope joins before ANY tile-local capture is destroyed,
    // topology is attached/updated, or assets/device state can change.
    // Live topology lookups therefore record exactly the queried keys;
    // no padded neighborhood or duplicated sampling formula is needed.
    // Legacy closures remain synchronous (pickup_profile == false).
    // Exclusive to this compile lane for the frame. Preserve bounded river
    // pages, but reset every tile's point caches and dependency consumer.
    // SurfaceQueries clears shore samples when constructed below.
    ground_query_scratch.reset_tile();
    ground_query_scratch.bind(natural,world_coast.world(),frame.world_topology_revision);
    // Owned, per-tile copies of the exact river nodes this compile
    // reads, instead of pointers into topology_cache.rivers (cleared
    // and rebuilt by the same per-frame pre-pass above whenever the
    // world's topology signature changes). A worker holding
    // local_river_nodes's pointers across that rebuild would read
    // freed memory; RiverNode is a small POD, so copying the already-
    // filtered, already-sorted set by value is cheap and preserves
    // exact order/content.
    std::vector<RiverNode> ground_river_node_values;
    ground_river_node_values.reserve(local_river_nodes.size());
    for(auto node:local_river_nodes)ground_river_node_values.push_back(*node);
    std::vector<RiverNode const *> ground_river_nodes;
    ground_river_nodes.reserve(ground_river_node_values.size());
    for(auto const& node:ground_river_node_values)ground_river_nodes.push_back(&node);
    auto& ground_world_dependencies=result->world;
    auto& ground_coast_dependencies=result->coast;
    auto& ground_topology_dependencies=result->topology;
    auto& ground_river_dependencies=result->rivers;
    c3x_renderer::fidelity::NaturalWorld::DependencyScope ground_river_inputs(
        ground_query_scratch.rivers,&ground_river_dependencies);
    auto ground_observe_world=[&](std::size_t i,std::uint32_t value){ground_world_dependencies.emplace(i,value);};
    auto ground_observe_coast=[&](auto id,auto revision){ground_coast_dependencies.emplace(id,revision);};
    c3x_renderer::fidelity::SurfaceQueries<decltype(ground_observe_world),decltype(ground_observe_coast)> ground_queries(world_coast,ground_query_scratch.shore_samples,
        tile.tile_x,tile.tile_y,ground_observe_world,ground_observe_coast,skip_flat_shore);
    ground_queries.prime_center(tile_center_shore);
    auto ground_world_lookup=[&](int c,int r){ return ground_queries.tile(c,r); };
    auto ground_shore_sample_at=[&](float u,float v){ return ground_queries.shore(u,v); };
    auto ground_river_distance=[&](c3x_renderer_tile_v1 const & river_tile,float u,float v){
        if(fidelity_profile){
            float x=float(river_tile.tile_x+river_tile.tile_y)*.5f+u,y=float(river_tile.tile_x-river_tile.tile_y)*.5f+1-v;
            return float(ground_query_scratch.rivers.river_sample({x,y}).distance);
        }
        return river_distance(river_tile,u,v); // pure river_edge_distance math; no shared state to isolate
    };
    auto ground_pickup_river=[&](int c,int r,float u,float v){
        auto const & world=world_coast.world();
        auto i=world.index(c,r);auto value=world.at(i);
        if(i!=std::size_t(-1))ground_observe_world(i,value);
        if(value==0xffffffffu || ((value>>16)&170u)==0 || !river_assets_ready)return 1000.0f;
        c3x_renderer_tile_v1 owner={};
        owner.tile_x=c+r;owner.tile_y=c-r;owner.river_code=(value>>16)&255u;
        return ground_river_distance(owner,u,v);
    };
    auto ground_pickup_activity=[&](int c,int r){
        auto const & world=world_coast.world();
        auto i=world.index(c,r);auto value=world.at(i);
        if(i!=std::size_t(-1))ground_observe_world(i,value);
        return value!=0xffffffffu && (value>>24)!=0 ? 1.0f : 0.0f;
    };
    c3x_renderer::fidelity::ReliefSurface<decltype(ground_world_lookup),decltype(pickup_source),decltype(ground_shore_sample_at),
        decltype(ground_pickup_river),decltype(pickup_dune),decltype(ground_pickup_activity)> ground_pickup_surface(world_coast.world().dimensions(),
        (tile.tile_x+tile.tile_y)/2,(tile.tile_x-tile.tile_y)/2,tile_center_shore.distance,
        ground_world_lookup,pickup_source,ground_shore_sample_at,ground_pickup_river,pickup_dune,ground_pickup_activity,
        ground_query_scratch.pickup_ground_samples,ground_query_scratch.pickup_height_queries,separate_natural_relief);
    auto ground_pickup_ground_at=[&](float u,float v){ return ground_pickup_surface.sample(u,v); };
    auto ground_pickup_height_at=[&](float u,float v){ return ground_pickup_surface.height(u,v); };
    auto ground_relief_at_world=[&](float world_u,float world_v)->std::array<float,3>{
        if(pickup_profile){
            auto sample=ground_pickup_ground_at(world_u,world_v);
            return {sample.height,sample.authored_height,sample.authored_blend};
        }
        return relief_at_world(world_u,world_v);
    };
    auto ground_material_weights_for=[&](float world_u,float world_v){
        return pickup_profile ? ground_queries.weights(world_u,world_v)
               : material_weights_for(world_u,world_v);
    };
    auto ground_signed_shore_distance=[&](float world_u,float world_v,float local_u,float local_v){
        return pickup_profile
            ? static_cast<float>(std::clamp(-ground_shore_sample_at(world_u,world_v).distance/.65,-1.,1.))
            : signed_shore_distance(world_u,world_v,local_u,local_v);
    };
    auto ground_observed_coordinate_key=[&](int x,int y){
        auto key=coordinate_key(x,y);
        auto inserted=ground_topology_dependencies.try_emplace(key,0);
        if(inserted.second){
            auto found=topology_cache.current(key);
            inserted.first->second=found==nullptr?0:found->semantic;
        }
        return key;
    };
    auto ground_ground_at_lattice=[&](int u,int v){
        auto record=topology_cache.current(ground_observed_coordinate_key(u+v,u-v));
        return record?static_cast<float>(record->ground):ground_slot;
    };
    auto ground_water_family_depth=[&](float world_u,float world_v){
        float grid_x=world_u-0.5f;
        float grid_y=world_v-0.5f;
        int x0=static_cast<int>(std::floor(grid_x));
        int y0=static_cast<int>(std::floor(grid_y));
        float tx=ground_smoothstep01(grid_x-static_cast<float>(x0));
        float ty=ground_smoothstep01(grid_y-static_cast<float>(y0));
        auto center_depth=[&](int x,int y){
            int base=static_cast<int>(ground_ground_at_lattice(x,y));
            return base>=11?std::clamp((base-10)*0.34f,0.18f,1.0f):0.34f;
        };
        float top=center_depth(x0,y0)*(1.0f-tx)+center_depth(x0+1,y0)*tx;
        float bottom=center_depth(x0,y0+1)*(1.0f-tx)+center_depth(x0+1,y0+1)*tx;
        return top*(1.0f-ty)+bottom*ty;
    };
    c3x_renderer::fidelity::GroundSurfaces ground_surfaces;
    c3x_renderer::fidelity::compile_ground_surfaces(ground_compile_input, frame, ground_query_scratch.rivers, ground_river_nodes,
        ground_grid_lease.get(), result->grid_hits,
        ground_relief_at_world, ground_pickup_height_at, ground_pickup_ground_at, ground_river_distance, ground_material_weights_for,
        ground_signed_shore_distance, ground_water_family_depth, periodic_surface_uv, ground_shore_sample_at,
        ndc_x, ndc_y, ground_cancelled, ground_surfaces);
    if(!c3x_renderer::fidelity::pack_ground(ground_surfaces,pickup_profile,world_ground,ground_cancelled,*result))return {};

    result->compile_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    return result;
}
}}
