// Compare the shared sampler to the production query expressions it replaces.
// This test protects query values AND cache-invalidation observations.
#include "queries.h"
#include <cassert>
#include <iostream>
#include <map>
using c3x_renderer::render_core::WorldCoast;
using c3x_renderer::render_core::World;
using c3x_renderer::render_core::ExactPointCache;
using c3x_renderer::render_core::ShoreSample;
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
            std::unordered_map<std::uint64_t,c3x_renderer::render_core::Tile> world_lookup_cache;
            auto observe_world = [&](std::size_t i, std::uint32_t value) { world_dependencies.emplace(i,value); };
            std::array<c3x_renderer::render_core::Tile,81> nearby_world_tiles{};
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
            c3x_renderer::render_core::ShoreSample shore_center{};bool shore_center_ready=false;
            shore_samples.clear();
            c3x_renderer::render_core::WorldCoast::Patch shore_patch;
            bool shore_patch_attempted=false;
            auto shore_sample_at = [&](float u,float v) {
                // Distance to a closed contour is 1-Lipschitz. Once the
                // center certificate proves this query beyond every land
                // response collar, its saturated values are exact. The same
                // certificate detects any newly closer coast on terrain edits.
                if(shore_center_ready && shore_center.distance>1.5+std::hypot(u-shore_center_u,v-shore_center_v))
                    return c3x_renderer::render_core::ShoreSample{2,0,0,0};
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
        auto values=c3x_renderer::render_core::material_weights({u,v},world,world_lookup);
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
        float h=natural.height(x,y,lookup_natural,support);
        h=std::max(h,2.5f+pickup(x,y));
        return 2.5f+(h-2.5f)*c3x_renderer::fidelity::coast_relief(float(shore.distance),float(shore.beach_width));
    };
    exercise(result,shore_center_u,shore_center_v,world_lookup,shore_sample_at,weights,height);
    return result;
}
Report shared(WorldCoast const& coast,ExactPointCache<ShoreSample>& scratch,
              NaturalData const& natural,int tx,int ty,bool skip_flat=false) {
    Report result;
    auto observe=[&](auto i,auto value){result.world.emplace(i,value);};
    auto nodes=[&](auto id,auto revision){result.coast.emplace(id,revision);};
    SurfaceQueries query(coast,scratch,tx,ty,observe,nodes,skip_flat);
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
    natural.terrain[30]=0;
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
                ExactPointCache<ShoreSample> fast_scratch;
                auto fast=shared(coast,fast_scratch,natural,owner[0],owner[1],true);
                assert(fast.values==a.values);
                samples+=unsigned(a.values.size());scopes++;
            }
        }
    }
    // A full rocky coast must retain the already-shaped pickup rim, even
    // where the ordinary beach envelope is zero. Lowland beaches keep it.
    for(bool rocky:{false,true}) {
        WorldCoast coast;std::vector<std::uint32_t> data(128);
        for(unsigned i=0;i<data.size();i++) {
            int x=int(i%8)*2+int(i/8)%2;
            unsigned real=x<8?(rocky?5:2):12;data[i]=(x<8?2:12)|(real<<8);
        }
        coast.update(World{16,16,false,false},data.data(),data.size(),1);
        ExactPointCache<ShoreSample> scratch;
        auto observe=[](auto,auto){};
        SurfaceQueries query(coast,scratch,6,6,observe,observe);
        bool witnessed=false;
        for(int i=0;i<125 && !witnessed;i++)for(int j=0;j<40 && !witnessed;j++) {
            float x=4+i*.04f,y=-1+j*.1f;auto sample=query.shore(x,y);
            if(sample.distance<=.03 || sample.distance>=.18 || (rocky && sample.rocky<.95))continue;
            auto h=query.height(natural,[](float,float){return 40.f;},x,y);
            if(rocky)assert(h>=42.5f);
            else assert(h==2.5f);
            witnessed=true;
        }
        assert(witnessed);
        if(rocky){
            bool hill_meets_cliff=false;
            for(int i=0;i<125 && !hill_meets_cliff;i++)for(int j=0;j<40 && !hill_meets_cliff;j++){
                float x=4+i*.04f,y=-1+j*.1f;auto sample=query.shore(x,y);
                if(sample.distance<.18 || sample.distance>.22 || sample.rocky<.95)continue;
                float authored=natural.height(x,y,[&](int c,int r){return query.natural_tile(c,r);});
                if(authored<8)continue;
                auto h=query.height(natural,[](float,float){return 0.f;},x,y);
                // The cliff slope has finished where the ordinary beach slope
                // has not begun. Keep the hill's height at this contact.
                assert(h>=2.5f+(authored-2.5f)*.95f);
                hill_meets_cliff=true;
            }
            assert(hill_meets_cliff);
        }
    }
    unsigned flat_samples=0,flat_avoided=0;
    for(unsigned wraps=0;wraps<3;wraps++)for(bool hills:{false,true}) {
        WorldCoast coast;World world{16,16,wraps>0,wraps>1};
        std::vector<std::uint32_t> data(128);
        for(unsigned i=0;i<data.size();++i){
            int y=int(i/8),x=int(i%8)*2+(y&1);
            unsigned base=x<8?2:12;
            unsigned real=hills && x==6 && y==6?5:base;
            data[i]=base|(real<<8);
        }
        coast.update(world,data.data(),data.size(),1);
        for(auto owner:std::array<std::array<int,2>,3>{{{{6,6}},{{-2,6}},{{18,6}}}}){
            ExactPointCache<ShoreSample> slow_cache,fast_cache;
            Report slow_dependencies,fast_dependencies;
            auto slow_world=[&](auto i,auto v){slow_dependencies.world.emplace(i,v);};
            auto fast_world=[&](auto i,auto v){fast_dependencies.world.emplace(i,v);};
            auto nodes=[](auto,auto){};
            SurfaceQueries slow(coast,slow_cache,owner[0],owner[1],slow_world,nodes,false);
            SurfaceQueries fast(coast,fast_cache,owner[0],owner[1],fast_world,nodes,true);
            for(float displacement:{0.f,-0.f,-1.f,.000001f,40.f})
            for(int y=-8;y<=8;++y)for(int x=-8;x<=8;++x){
                float u=slow.center_u+x*.173f,v=slow.center_v+y*.193f;
                auto source=[&](float,float){return displacement;};
                float slow_support=-1,fast_support=-1;
                float a=slow.height(natural,source,u,v,&slow_support);
                float b=fast.height(natural,source,u,v,&fast_support);
                assert(a==b && slow_support==fast_support);++flat_samples;
            }
            assert(fast_cache.hits+fast_cache.misses<=slow_cache.hits+slow_cache.misses);
            flat_avoided+=unsigned(slow_cache.hits+slow_cache.misses-fast_cache.hits-fast_cache.misses);
        }
        if(!hills && !wraps){
            ExactPointCache<ShoreSample> scratch;Report observed;
            auto observe=[&](auto i,auto value){observed.world.emplace(i,value);};
            auto nodes=[&](auto i,auto value){observed.coast.emplace(i,value);};
            SurfaceQueries flat(coast,scratch,6,6,observe,nodes);
            assert(flat.height(natural,[](float,float){return 0.f;},6.5f,.5f)==2.5f);
            assert(observed.world.size()==9 && observed.coast.empty());
            assert(scratch.hits==0 && scratch.misses==0);
            // Introducing relief invalidates the exact source observations,
            // even though this flat height needed no coast query at all.
            auto center=coast.world().index(6,0);
            assert(observed.world.count(center));data[center]=2u|(5u<<8);
            coast.update(world,data.data(),data.size(),2);
            assert(coast.world().at(center)!=observed.world.at(center));
        }
    }
    assert(flat_avoided>0);
    std::cout<<"PASS flat height certificate: "<<flat_samples<<" exact heights/supports, "
             <<flat_avoided<<" redundant shore calls avoided; source edit observations retained\n";
    std::cout<<"PASS production surface queries: "<<scopes<<" scopes, "<<samples
             <<" exact values, world/coast observations and cache statistics; wrapping and terrain edits\n";
}
