#pragma once
// Per-tile production queries shared with the local scene compiler. World data
// and scratch storage belong to the caller; observations retain native cache
// invalidation authority. Construct a fresh query scope for each owner tile.
#include <unordered_map>
#include "data.h"
#include "../../../native/source_fidelity/coast_join.h"
#include "../../../native/render_core/world_coast.h"
#include "../../../native/render_core/exact_point_cache.h"
namespace c3x_renderer { namespace fidelity {
template<class ObserveWorld,class ObserveCoast> class SurfaceQueries {
    render_core::WorldCoast const& coast;
    render_core::ExactPointCache<render_core::ShoreSample>& samples;
    ObserveWorld observe_world;
    ObserveCoast observe_coast;
    std::unordered_map<std::uint64_t,render_core::Tile> far_tiles;
    std::array<render_core::Tile,81> nearby_tiles{};
    std::array<bool,81> nearby_ready{};
    int nearby_c,nearby_r;
    bool skip_flat_shore;
    render_core::ShoreSample center{};
    bool center_ready=false,patch_attempted=false;
    render_core::WorldCoast::Patch patch;
    // Finite-difference points almost never repeat, but their integer
    // neighborhoods do. Keep authored hill placements for the owner and its
    // eight adjacent sample cells; distant object queries use the ordinary path.
    struct Hills {bool ready=false;unsigned count=0;std::array<Hill,9> bodies;};
    std::array<Hills,9> hills;
public:
    float const center_u,center_v;
    SurfaceQueries(render_core::WorldCoast const& world,
                   render_core::ExactPointCache<render_core::ShoreSample>& scratch,
                   int tile_x,int tile_y,ObserveWorld observe,ObserveCoast nodes,
                   bool skip_flat=true)
        :coast(world),samples(scratch),observe_world(observe),observe_coast(nodes),
         nearby_c((tile_x+tile_y)/2-4),nearby_r((tile_x-tile_y)/2-4),
         skip_flat_shore(skip_flat),
         center_u(float(tile_x+tile_y)*.5f+.5f),center_v(float(tile_x-tile_y)*.5f+.5f) {
        samples.clear();
    }
    render_core::Tile tile(int c,int r) {
        int x=c-nearby_c,y=r-nearby_r;
        if(x>=0 && x<9 && y>=0 && y<9) {
            auto n=std::size_t(y*9+x);
            if(!nearby_ready[n]) {
                auto const& topology=coast.world();auto i=topology.index(c,r);
                if(i!=std::size_t(-1))observe_world(i,topology.at(i));
                nearby_tiles[n]=topology.tile(c,r);nearby_ready[n]=true;
            }
            return nearby_tiles[n];
        }
        std::uint64_t key=(std::uint64_t(std::uint32_t(c))<<32)|std::uint32_t(r);
        auto found=far_tiles.find(key);if(found!=far_tiles.end())return found->second;
        auto const& topology=coast.world();auto i=topology.index(c,r);
        if(i!=std::size_t(-1))observe_world(i,topology.at(i));
        auto value=topology.tile(c,r);far_tiles.emplace(key,value);return value;
    }
    void prime_center(render_core::ShoreSample const& value) {
        center=value;center_ready=true;
        samples.get(center_u,center_v,[&](){return value;});
    }
    render_core::ShoreSample shore(float u,float v) {
        // Preserve the production center certificate and exact query cache.
        if(center_ready && center.distance>1.5+std::hypot(u-center_u,v-center_v))
            return render_core::ShoreSample{2,0,0,0};
        return samples.get(u,v,[&]() {
            if(center_ready && !patch_attempted) {
                patch=coast.prepare({center_u,center_v},.73,std::abs(center.distance),observe_coast);
                patch_attempted=true;
            }
            auto sample=coast.sample_with_lookup({u,v},observe_coast,
                [&](int c,int r){return tile(c,r);},&patch);
            if(u==center_u && v==center_v){center=sample;center_ready=true;}
            return sample;
        });
    }
    std::array<float,5> weights(float u,float v) {
        auto w=render_core::material_weights({u,v},coast.world().dimensions(),
            [&](int c,int r){return tile(c,r);});
        std::array<float,5> result;
        for(int i=0;i<5;i++)result[i]=static_cast<float>(w[i]);
        return result;
    }
    Tile natural_tile(int c,int r) {
        auto value=tile(c,r);auto world=coast.world().dimensions();
        int x=c+r,y=c-r;
        if(world.wrap_x && world.width>0)x=render_core::mod(x,world.width);
        if(world.wrap_y && world.height>0)y=render_core::mod(y,world.height);
        return Tile{x,y,c,r,value.real};
    }
    template<class Height>
    float height(NaturalData const& natural,Height pickup_height,float x,float y,float* support=nullptr) {
        render_core::ShoreSample sample{};
        if(!skip_flat_shore)sample=shore(x,y);
        int c=int(std::floor(x)),r=int(std::floor(y));
        int local_c=c-(nearby_c+3),local_r=r-(nearby_r+3);
        float authored=2.5f,s=0;
        if(local_c>=0 && local_c<3 && local_r>=0 && local_r<3){
            auto& entry=hills[std::size_t(local_r*3+local_c)];
            if(!entry.ready){
                for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
                    auto t=natural_tile(c+dx,r+dy);
                    if(t.real==5)entry.bodies[entry.count++]=composed_hill(t);
                }
                entry.ready=true;
            }
            for(unsigned i=0;i<entry.count;i++)natural.hill_height(entry.bodies[i],x,y,authored,s);
            if(support)*support=s;
        }else authored=natural.height(x,y,[&](int nc,int nr){return natural_tile(nc,nr);},support);
        float pickup=pickup_height(x,y);
        // Both coastal branches multiply displacement above the 2.5 datum.
        // With both sources exactly flat the result is independent of shore;
        // the source queries above still observe all height dependencies.
        if(skip_flat_shore && authored==2.5f && pickup==0.f)return 2.5f;
        if(skip_flat_shore)sample=shore(x,y);
        float coastal=coast_relief(float(sample.distance),float(sample.beach_width));
        float h=std::max(authored,2.5f+pickup);
        float rocky=coast_ramp((float(sample.rocky)-.55f)/.4f);
        // Both hill and pickup relief meet the steep rocky coast. The ordinary
        // beach envelope must not flatten a hill behind the cliff faces.
        if(rocky>0){
            float cliff=coast_ramp((float(sample.distance)-.04f)/.14f);
            return 2.5f+std::max((authored-2.5f)*(coastal+(1-coastal)*rocky*cliff),
                pickup*(coastal+(1-coastal)*rocky));
        }
        return 2.5f+(h-2.5f)*coastal;
    }
};
}}
