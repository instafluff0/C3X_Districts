// Compare the shared sampler to the production query expressions it replaces.
// This test protects query values AND cache-invalidation observations.
#include "queries.h"
#include <cassert>
#include <iostream>
#include <map>
using c3x_renderer::profile_v2::WorldCoast;
using c3x_renderer::profile_v2::World;
using c3x_renderer::profile_v2::ExactPointCache;
using c3x_renderer::profile_v2::ShoreSample;
using c3x_renderer::fidelity::NaturalData;
using c3x_renderer::fidelity::SurfaceQueries;
struct Report {
    std::vector<double> values;
    std::map<std::size_t,std::uint32_t> world;
    std::map<std::uint64_t,std::uint64_t> coast;
};
template<class Lookup,class Shore,class Weights,class Height>
void exercise(Report& report,float cu,float cv,Lookup lookup,Shore shore,Weights weights,Height height) {
    auto point=[&](float u,float v) {
        auto s=shore(u,v);
        for(double value:{s.distance,s.beach_width,s.rocky,s.depth})report.values.push_back(value);
        for(float value:weights(u,v))report.values.push_back(value);
        float support=0;report.values.push_back(height(u,v,&support));report.values.push_back(support);
    };
    point(cu,cv);
    for(float dy:{-.6f,0.f,.4f})for(float dx:{-.6f,0.f,.4f}) {
        point(cu+dx,cv+dy);point(cu+dx,cv+dy);
    }
    for(int offset:{-25,-4,0,4,25}) {
        auto t=lookup(int(cu)+offset,int(cv)-offset);
        report.values.insert(report.values.end(),{double(t.base),double(t.real),double(t.present)});
        auto again=lookup(int(cu)+offset,int(cv)-offset);
        assert(t.base==again.base && t.real==again.real && t.present==again.present);
    }
}
float pickup(float x,float y) {return std::max(0.f,std::sin(x*.31f)*std::cos(y*.2f));}
Report original(WorldCoast const& world_coast,ExactPointCache<ShoreSample>& shore_samples,
                NaturalData const& natural,int tx,int ty) {
    Report result;
    auto& world_dependencies=result.world;auto& coast_dependencies=result.coast;
    struct {int tile_x,tile_y;} tile{tx,ty};
            std::unordered_map<std::uint64_t,c3x_renderer::profile_v2::Tile> world_lookup_cache;
            auto observe_world = [&](std::size_t i, std::uint32_t value) { world_dependencies.emplace(i,value); };
            std::array<c3x_renderer::profile_v2::Tile,81> nearby_world_tiles{};
            std::array<bool,81> nearby_world_ready{};
            int nearby_c=(tile.tile_x+tile.tile_y)/2-4,nearby_r=(tile.tile_x-tile.tile_y)/2-4;
            auto world_lookup = [&](int c, int r) {
                int x=c-nearby_c,y=r-nearby_r;
                if(x>=0 && x<9 && y>=0 && y<9) {
                    auto n=std::size_t(y*9+x);
                    if(!nearby_world_ready[n]) {
                        auto const& topology=world_coast.world();auto i=topology.index(c,r);
                        if(i!=std::size_t(-1))observe_world(i,topology.at(i));
                        nearby_world_tiles[n]=topology.tile(c,r);nearby_world_ready[n]=true;
                    }
                    return nearby_world_tiles[n];
                }
                std::uint64_t key=(std::uint64_t(std::uint32_t(c))<<32)|std::uint32_t(r);
                auto found=world_lookup_cache.find(key);if(found!=world_lookup_cache.end())return found->second;
                auto const & topology = world_coast.world();
                auto i = topology.index(c,r);
                if (i != std::size_t(-1)) observe_world(i, topology.at(i));
                auto value=topology.tile(c,r);world_lookup_cache.emplace(key,value);return value;
            };
            float shore_center_u=float(tile.tile_x+tile.tile_y)*.5f+.5f;
            float shore_center_v=float(tile.tile_x-tile.tile_y)*.5f+.5f;
            c3x_renderer::profile_v2::ShoreSample shore_center{};bool shore_center_ready=false;
            shore_samples.clear();
            c3x_renderer::profile_v2::WorldCoast::Patch shore_patch;
            bool shore_patch_attempted=false;
            auto shore_sample_at = [&](float u,float v) {
                // Distance to a closed contour is 1-Lipschitz. Once the
                // center certificate proves this query beyond every land
                // response collar, its saturated values are exact. The same
                // certificate detects any newly closer coast on terrain edits.
                if(shore_center_ready && shore_center.distance>1.5+std::hypot(u-shore_center_u,v-shore_center_v))
                    return c3x_renderer::profile_v2::ShoreSample{2,0,0,0};
                return shore_samples.get(u,v,[&]() {
                    if(shore_center_ready && !shore_patch_attempted) {
                        shore_patch=world_coast.prepare({shore_center_u,shore_center_v},.73,std::abs(shore_center.distance),
                            [&](auto id,auto revision){coast_dependencies.emplace(id,revision);});
                        shore_patch_attempted=true;
                    }
                    auto sample = world_coast.sample_with_lookup({u,v},
                        [&](auto id,auto revision) { coast_dependencies.emplace(id,revision); }, world_lookup, &shore_patch);
                    if(u==shore_center_u && v==shore_center_v){shore_center=sample;shore_center_ready=true;}
                    return sample;
                });
            };
    auto world=world_coast.world().dimensions();
    auto weights=[&](float u,float v) {
        auto values=c3x_renderer::profile_v2::material_weights({u,v},world,world_lookup);
        std::array<float,5> r;for(int i=0;i<5;i++)r[i]=static_cast<float>(values[i]);return r;
    };
    auto lookup_natural=[&](int c,int r) {
        auto value=world_lookup(c,r);int x=c+r,y=c-r;
        auto canonical=[](int value,int size,bool wraps) {
            if(!wraps || size<=0)return value;
            int result=value%size;return result<0?result+size:result;
        };
        return c3x_renderer::fidelity::Tile{canonical(x,world.width,world.wrap_x),
            canonical(y,world.height,world.wrap_y),c,r,value.real};
    };
    auto height=[&](float x,float y,float* support) {
        auto shore=shore_sample_at(x,y);
        float h=std::max(natural.height(x,y,lookup_natural,support),2.5f+pickup(x,y));
        return 2.5f+(h-2.5f)*c3x_renderer::fidelity::coast_relief(float(shore.distance),float(shore.beach_width));
    };
    exercise(result,shore_center_u,shore_center_v,world_lookup,shore_sample_at,weights,height);
    return result;
}
Report shared(WorldCoast const& coast,ExactPointCache<ShoreSample>& scratch,
              NaturalData const& natural,int tx,int ty) {
    Report result;
    auto observe=[&](auto i,auto value){result.world.emplace(i,value);};
    auto nodes=[&](auto id,auto revision){result.coast.emplace(id,revision);};
    SurfaceQueries query(coast,scratch,tx,ty,observe,nodes);
    exercise(result,query.center_u,query.center_v,
        [&](int c,int r){return query.tile(c,r);},
        [&](float u,float v){return query.shore(u,v);},
        [&](float u,float v){return query.weights(u,v);},
        [&](float u,float v,float* support){return query.height(natural,pickup,u,v,support);});
    return result;
}
int main() {
    NaturalData natural;
    natural.fields.resize(1);auto& f=natural.fields[0];f.width=f.height=8;
    for(unsigned i=0;i<64;i++)f.pixels.push_back(std::uint8_t(i*4));
    unsigned scopes=0,samples=0;
    for(unsigned wraps=0;wraps<3;wraps++) {
        World world{16,16,wraps>0,wraps>1};WorldCoast coast;
        std::vector<std::uint32_t> data(128);
        for(unsigned i=0;i<data.size();i++) {
            int y=int(i/8),x=int(i%8)*2+(y&1);
            unsigned base=x<8?(y<8?2:1):12;
            unsigned real=(x==4 && y==6)?5:base;
            data[i]=base|(real<<8);
        }
        ExactPointCache<ShoreSample> old_scratch,new_scratch;
        for(int revision=0;revision<2;revision++) {
            if(revision)data[(6*16+6)/2]=12|(12<<8);
            coast.update(world,data.data(),data.size(),revision);
            for(auto owner:std::array<std::array<int,2>,3>{{{{2,2}},{{6,6}},{{14,12}}}}) {
                auto a=original(coast,old_scratch,natural,owner[0],owner[1]);
                auto b=shared(coast,new_scratch,natural,owner[0],owner[1]);
                assert(a.values==b.values);assert(a.world==b.world);assert(a.coast==b.coast);
                assert(old_scratch.hits==new_scratch.hits && old_scratch.misses==new_scratch.misses);
                samples+=unsigned(a.values.size());scopes++;
            }
        }
    }
    std::cout<<"PASS production surface queries: "<<scopes<<" scopes, "<<samples
             <<" exact values, world/coast observations and cache statistics; wrapping and terrain edits\n";
}
