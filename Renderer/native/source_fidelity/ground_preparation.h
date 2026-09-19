#pragma once
#include "prepared_ground.h"

namespace c3x_renderer { namespace fidelity {
struct GroundRiverNode {
    int lattice_x=0,lattice_y=0;
    unsigned degree=0;
    bool touches_water=false;
};

// Own scalar capture data, never a frame.tiles/game pointer or tile-local
// closure. World observations and assets are borrowed for the enclosing frame.
struct GroundPreparationInput {
    GroundCompileInput compile;
    int tile_width=0,tile_height=0,world_width=0,world_height=0;
    bool wrap_x=false,wrap_y=false,skip_flat_shore=true,separate_natural_relief=false;
    std::int64_t topology_revision=0;
    render_core::ShoreSample center;
    std::vector<GroundRiverNode> nodes;
};

inline std::array<float,2> ground_surface_uv(c3x_renderer_tile_v1 const& tile,
        c3x_renderer_frame_v1 const& frame,float world_u,float world_v,float frequency) {
    auto canonical=[](int value,int extent){int result=value%extent;return result<0?result+extent:result;};
    float map_x=world_u+world_v-1.0f,map_y=world_u-world_v;
    float half_frequency=frequency*0.5f;
    float x_component=map_x*half_frequency,y_component=map_y*half_frequency;
    if(frame.world_wrap_x && frame.world_width_tiles>0){
        float cycles=std::max(1.0f,std::round(frame.world_width_tiles*half_frequency));
        float canonical_map_x=static_cast<float>(canonical(tile.tile_x,frame.world_width_tiles))+map_x-tile.tile_x;
        x_component=cycles*canonical_map_x/static_cast<float>(frame.world_width_tiles);
    }
    if(frame.world_wrap_y && frame.world_height_tiles>0){
        float cycles=std::max(1.0f,std::round(frame.world_height_tiles*half_frequency));
        float canonical_map_y=static_cast<float>(canonical(tile.tile_y,frame.world_height_tiles))+map_y-tile.tile_y;
        y_component=cycles*canonical_map_y/static_cast<float>(frame.world_height_tiles);
    }
    return {x_component+y_component,y_component-x_component};
}

// Shared selection math: both the foreground proof recorder and owned-job
// capture observe the same neighbors and select the same existing grid detail.
template<class TopologyLookup,class WorldLookup>
int ground_grid_detail(c3x_renderer_tile_v1 const& tile,int relief,bool dunes,bool pickup,
        int width,unsigned records,TopologyLookup topology,WorldLookup world) {
    bool neighborhood=relief==5 || relief==6 || relief==10 || dunes;
    int offsets[4][2]={{-1,-1},{1,-1},{1,1},{-1,1}};
    for(auto const& offset:offsets){
        auto found=topology(tile.tile_x+offset[0],tile.tile_y+offset[1]);
        neighborhood=neighborhood || (found && (found->relief==5 || found->relief==6 || found->relief==10));
    }
    if(pickup){
        int c=(tile.tile_x+tile.tile_y)/2,r=(tile.tile_x-tile.tile_y)/2;
        for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
            int real=world(c+dx,r+dy).real;
            neighborhood=neighborhood || real==5 || real==6 || real==10 || real==0;
        }
        return neighborhood?(width>=96?24:12):(width>=96?12:8);
    }
    int base=width>=96?(records<=768?16:12):8;
    int detailed=width>=96?(records<=512?24:records<=768?16:12):(records<=2048?12:8);
    return neighborhood?detailed:base;
}

template<class Topology,class Assets,class Stop>
std::unique_ptr<PreparedGround> compile_selected_ground(GroundPreparationInput const& job,
        NaturalData const& natural,render_core::WorldCoast const& coast,Topology const& observations,
        Assets const& assets,SurfaceQueryScratch& scratch,Stop stop) {
    if(!job.compile.pickup_profile || !job.compile.fidelity_profile || !job.compile.world_ground)return {};
    c3x_renderer_frame_v1 frame{};
    frame.tile_width=job.tile_width;frame.tile_height=job.tile_height;
    frame.world_width_tiles=job.world_width;frame.world_height_tiles=job.world_height;
    frame.world_wrap_x=job.wrap_x;frame.world_wrap_y=job.wrap_y;frame.world_topology_revision=job.topology_revision;
    std::vector<GroundRiverNode const*> nodes;nodes.reserve(job.nodes.size());
    for(auto const& node:job.nodes)nodes.push_back(&node);
    auto key=[&](int x,int y){return observations.key(x,y);};
    auto source=[&](int kind,unsigned variant,int channel,float u,float v){return relief_source(assets,true,kind,variant,channel,u,v);};
    auto dune=[](float,float){return 0.f;};
    // These legacy callbacks are unreachable in the guarded fidelity path.
    auto river=[](auto const&,float,float)->float{throw std::logic_error("legacy ground river callback");};
    auto relief=[](float,float)->std::array<float,3>{throw std::logic_error("legacy ground relief callback");};
    auto weights=[](float,float)->std::array<float,5>{throw std::logic_error("legacy ground weights callback");};
    auto shore=[](float,float,float,float)->float{throw std::logic_error("legacy ground shore callback");};
    auto uv=[&](float u,float v,float scale){return ground_surface_uv(job.compile.tile,frame,u,v,scale);};
    auto ndc=[](float value){return value;};
    return prepare_ground(job.compile,frame,natural,scratch,coast,observations,nodes,{},job.center,
        float(job.compile.ground),job.skip_flat_shore,job.separate_natural_relief,
        key,source,dune,river,relief,weights,shore,uv,ndc,ndc,stop);
}

using GroundPreparation=render_core::ContentPreparation<unsigned,GroundPreparationInput,PreparedGround>;
class GroundPreparationLease {
    GroundPreparation& queue;
    std::array<SurfaceQueryScratch,2> scratch;
public:
    explicit GroundPreparationLease(GroundPreparation& owner):queue(owner){queue.clear();}
    GroundPreparationLease(GroundPreparationLease const&)=delete;
    ~GroundPreparationLease(){queue.clear();} // joins before scratch and borrowed frame inputs die
    template<class Compile> void start(std::deque<GroundPreparation::Job> jobs,Compile compile){
        queue.configure(std::move(jobs),[this,compile](auto const& input,auto const& stop,unsigned worker){
            return compile(input,scratch[worker],stop);
        },unsigned(scratch.size()));
        // Selected work uses the ordinary half-budget refill gate. Only the
        // actual adoption demand bypasses it; marking the whole view urgent
        // would evict ready results while the render owner is still uploading.
        queue.resume();
    }
    void finish(){queue.clear();}
};
}}
