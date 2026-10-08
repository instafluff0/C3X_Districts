#pragma once
#include "../terrain_scene_runtime.h"
#include "../hill_vegetation.h"
#include "../../lab/shared/natural/world.h"
#include "../../lab/shared/natural/queries.h"
#include "../../lab/shared/natural/relief.h"
#include "../../lab/shared/natural/ground.h"
#include "../../lab/shared/natural/hill_decals.h"
#include "../../lab/shared/natural/canopy_clearing.h"
#include "../../lab/shared/natural/mountain_shape.h"
#include "../render_core/content_preparation.h"
#include "../render_core/prepared_mesh.h"
#include "../../lab/studies/borders/mesh_export.h"
namespace c3x_renderer { namespace fidelity {
struct TerrainCompileInput {
    using Key=std::array<std::uint64_t,13>;
    Key key{};
    // Routes, resources and sites the vegetation floor keeps clear; its
    // hash is the last key word.
    CanopyClearing clearing;
    int tile_x=0,tile_y=0,real_terrain_type=0,ground=0,tile_width=0,tile_height=0,target_height=0;
    std::int64_t world_revision=0;
    PatchDetail detail;
    bool river_ready=false,skip_flat_shore=true,separate_relief=true,indexed=true,retain_height=true;
};
struct TerrainSurfaces {
    std::array<render_core::PreparedMesh,3> meshes;
    // Populated only by the renderer, after compilation, once meshes are
    // packed: a type-erased handle to a GPU buffer holding every non-empty
    // layer's vertex bytes, and each layer's offset within it. Kept opaque so
    // this CPU-only compiler never depends on a graphics device.
    std::shared_ptr<void> vertex_buffer;
    std::array<unsigned,3> vertex_offset{};
    std::unordered_map<std::size_t,std::uint32_t> world;
    std::unordered_map<std::uint64_t,std::uint64_t> coast;
    NaturalWorld::CellProof rivers;
    std::size_t proof_bytes=0;
    std::size_t buffer_bytes() const {
        std::size_t bytes=0;for(auto const& mesh:meshes)bytes+=mesh.bytes();return bytes;
    }
    std::size_t bytes() const {return sizeof(*this)+buffer_bytes()+proof_bytes+
        (world.size()+coast.size())*64+(world.bucket_count()+coast.bucket_count())*sizeof(void*);}
};
struct TerrainCompileScratch {
    NaturalWorld rivers;
    PatchLayouts layouts;
    render_core::ExactPointCache<render_core::ShoreSample> shores;
    render_core::ExactPointCache<render_core::GroundSample> pickup;
    render_core::ExactPointCache<std::array<float,2>> heights;
    TerrainCompileScratch(){rivers.river_page_limit=2;}
    render_core::World dimensions{};
    void reset(){rivers.reset_world();dimensions={};}
    void bind(NaturalData const& natural,render_core::WorldTopology const& world,std::int64_t revision){
        auto next=world.dimensions();
        if(next.width!=dimensions.width || next.height!=dimensions.height ||
           next.wrap_x!=dimensions.wrap_x || next.wrap_y!=dimensions.wrap_y)reset();
        dimensions=next;rivers.borrowed_data=&natural;rivers.update_rivers(world,revision);
    }
};
using TerrainPreparation=render_core::ContentPreparation<TerrainCompileInput::Key,TerrainCompileInput,TerrainSurfaces>;

// Asset payloads and the coast/world are immutable during the read lease.
// Every worker has private river/query/layout scratch. The foreground fallback
// calls this same compiler, so worker availability cannot change float results.
template<class Assets,class Cancelled>
bool emit_terrain_surfaces(NaturalData const& natural,Assets const& assets,
        render_core::WorldCoast const& world_coast,TerrainCompileInput const& input,
        TerrainCompileScratch& scratch,Cancelled stop,TerrainSurfaces& destination,bool bounded) {
    using Vertex=MapVertex;
    auto result=&destination;
    std::array<std::vector<Vertex>,3> natural_vertices;
    std::array<std::vector<unsigned>,2> natural_grid_indices;
    bool const compact_hill_decals=input.indexed && bounded;
    std::vector<unsigned> hill_decal_indices;
    HillDecalOutput hill_decals(natural_vertices[1],hill_decal_indices);
    auto cancelled=[&]{
        std::size_t bytes=result->buffer_bytes();
        for(auto const& layer:natural_vertices)bytes+=layer.capacity()*sizeof(Vertex);
        for(auto const& index:natural_grid_indices)bytes+=index.capacity()*sizeof(unsigned);
        bytes+=hill_decal_indices.capacity()*sizeof(unsigned)+hill_decals.scratch_bytes();
        return stop() || hill_decals.rejected() || (bounded && bytes>8u*1024u*1024u);
    };
    auto admit_hill=[&](std::size_t extra){
        std::size_t bytes=result->buffer_bytes()+hill_decal_indices.capacity()*sizeof(unsigned)+hill_decals.scratch_bytes();
        for(auto const& layer:natural_vertices)bytes+=layer.capacity()*sizeof(Vertex);
        for(auto const& index:natural_grid_indices)bytes+=index.capacity()*sizeof(unsigned);
        return !stop() && bytes<=8u*1024u*1024u && extra<=8u*1024u*1024u-bytes;
    };
    if(cancelled())return {};
    scratch.bind(natural,world_coast.world(),input.world_revision);
    NaturalWorld::CellInputs river_inputs;
    NaturalWorld::DependencyScope scope(scratch.rivers,&river_inputs);
    scratch.pickup.clear();scratch.heights.clear();
    auto observe_world=[&](std::size_t index,std::uint32_t value){result->world.emplace(index,value);};
    auto observe_coast=[&](std::uint64_t index,std::uint64_t revision){result->coast.emplace(index,revision);};
    SurfaceQueries queries(world_coast,scratch.shores,input.tile_x,input.tile_y,observe_world,observe_coast,input.skip_flat_shore,&scratch.rivers);
    auto world_lookup=[&](int c,int r){return queries.tile(c,r);};
    auto lookup_natural=[&](int c,int r){return queries.natural_tile(c,r);};
    auto mountain_flags=[&](int c,int r){auto t=queries.tile(c,r);
        return (t.snow?1u:0u)|(t.active?2u:0u)|(t.erupting?4u:0u);};
    auto shore_sample_at=[&](float x,float y){return queries.shore(x,y);};
    auto material_weights_for=[&](float x,float y){return queries.weights(x,y);};
    auto relief_sample=[&](int kind,unsigned variant,int channel,float u,float v){return relief_source(assets,true,kind,variant,channel,u,v);};
    auto river=[&](int c,int r,float u,float v){
        auto const& world=world_coast.world();auto i=world.index(c,r);auto value=world.at(i);
        if(i!=std::size_t(-1))observe_world(i,value);
        if(value==0xffffffffu || !(value>>16&170u) || !input.river_ready)return 1000.f;
        return float(scratch.rivers.river_sample({float(c)+u,float(r)+1-v}).distance);
    };
    auto dune=[](float,float){return 0.f;};
    auto activity=[&](int c,int r){auto const& world=world_coast.world();auto i=world.index(c,r);auto value=world.at(i);
        if(i!=std::size_t(-1))observe_world(i,value);return value!=0xffffffffu && ((value>>24)&1u)!=0?1.f:0.f;};
    int nc=(input.tile_x+input.tile_y)/2,nr=(input.tile_x-input.tile_y)/2;
    auto const& tile=input;int ground=input.ground;auto owner=lookup_natural(nc,nr);
    HillMaterialFootprint hill_material={owner.real==5,
        lookup_natural(nc-1,nr).real==5,lookup_natural(nc+1,nr).real==5,
        lookup_natural(nc,nr+1).real==5,lookup_natural(nc,nr-1).real==5};
    std::size_t height_queries=0;
    ReliefSurface pickup_surface(world_coast.world().dimensions(),nc,nr,
        shore_sample_at(queries.center_u,queries.center_v).distance,world_lookup,relief_sample,shore_sample_at,river,dune,activity,
        scratch.pickup,height_queries,input.separate_relief);
    auto pickup_height=[&](float x,float y){return pickup_surface.height(x,y);};
    auto const river_field=scratch.rivers.bind_river_page(float(nc)+.5,float(nr)+.5);
    auto river_at=[&](float x,float y){return river_field.sample({x,y}).distance;};
    bool river_terrain_near=input.river_ready && scratch.rivers.river_affects(nc,nr);
    auto height_natural=[&](float x,float y,float* support=nullptr){
        auto compute=[&]{std::array<float,2> value{};value[0]=queries.height(natural,pickup_height,x,y,&value[1]);return value;};
        auto value=input.retain_height?scratch.heights.get(x,y,compute):compute();
        if(support)*support=value[1];return value[0]+
            (river_terrain_near ? river_channel_cut(float(river_at(x,y))) : 0.f);
    };
    GroundProjection project_natural{nc,nr,input.tile_width*.5f,input.tile_height*.5f,
        float(input.tile_width)/224.f*.82f,float(input.target_height)};
    auto surface=[&](float u,float v){
        auto out=ground_surface(project_natural,u,v,height_natural,shore_sample_at,material_weights_for);
        out.material_desert*=hill_material(u,v);
        out.river_distance=river_terrain_near ? float(river_at(out.world_x,out.world_y)) : 1000.f;
        return out;
    };
    auto triangle=[&](std::vector<Vertex>& out,Vertex const& a,Vertex const& b,Vertex const& c){
        if(input.real_terrain_type==5 && compact_hill_decals && &out==&natural_vertices[1]){Vertex v[]={a,b,c};hill_decals.append(v,admit_hill);}
        else{out.push_back(a);out.push_back(b);out.push_back(c);}
    };
    auto emit_hill_decal_surface=[&](Tile source,int column,int row,
            std::vector<Vertex> const& receiver,std::vector<unsigned> const* indices,
            std::vector<Vertex>& output){
        if(!compact_hill_decals){emit_hill_decals(source,column,row,receiver,indices,output);return true;}
        // Ten rock patches plus at most16 hill-canopy floor patches of8x8 quads.
        auto source_count=indices?indices->size():receiver.size();
        if(source_count>(8u*1024u*1024u/8-6144u)/10u ||
           !hill_decals.initialize(source_count*10u+6144u,admit_hill))return false;
        return emit_hill_decal_triangles(source,column,row,receiver,indices,
            [&](Vertex const* v){return hill_decals.append(v,admit_hill);});
    };
    auto patch_detail=input.detail;auto& patch_layouts=scratch.layouts;
    bool index_natural_grids=input.indexed;
    auto record_natural_phase=[](unsigned){}; // Worker CPU time is recorded by its queue.
    auto hidden_taper_at=[&](float x,float y){return queries.hidden_taper(x,y);};
    auto const& clearing=input.clearing;
    #include "terrain_mesh_body.h"
    if(cancelled())return false;
    hill_decals.release_scratch();
    if(!export_border_ground_mesh(input.tile_x,input.tile_y,
                                  natural_vertices,natural_grid_indices))return false;
    for(unsigned layer=0;layer<3;++layer){
        auto topology=layer==1 && tile.real_terrain_type==5 && compact_hill_decals?&hill_decal_indices:
            input.indexed && layer!=1?&natural_grid_indices[layer==0?0:1]:nullptr;
        render_core::MeshFormat format;format.natural=true;
        format.shared_grid=layer==1?0:render_core::shared_mesh_grid(natural_vertices[layer].size(),topology,patch_layouts);
        if(!render_core::prepare_mesh(natural_vertices[layer],topology,format,result->meshes[layer],stop))return false;
        // Ready storage replaces raw compiler output before the next layer.
        // A worker owns at most one layer's packing transient in addition to
        // the existing bounded raw input; no raw arrays enter the ready queue.
        std::vector<Vertex>().swap(natural_vertices[layer]);
    }
    if(cancelled())return false;
    result->rivers.assign(river_inputs.begin(),river_inputs.end());
    result->proof_bytes=scratch.rivers.proof_bytes(result->rivers);
    return true;
}
template<class Assets,class Cancelled>
std::unique_ptr<TerrainSurfaces> compile_terrain_surfaces(NaturalData const& natural,Assets const& assets,
        render_core::WorldCoast const& world_coast,TerrainCompileInput const& input,
        TerrainCompileScratch& scratch,Cancelled stop,bool bounded=true) {
    auto result=std::make_unique<TerrainSurfaces>();
    if(!emit_terrain_surfaces(natural,assets,world_coast,input,scratch,stop,*result,bounded))return {};
    return result;
}
}}
