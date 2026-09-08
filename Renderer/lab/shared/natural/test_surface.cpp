#include "mesh.h"
#include <cassert>
#include <cstring>
#include <iostream>
using namespace c3x_renderer::fidelity;

struct Shore {float distance,beach_width;};

int main() {
    NaturalData natural;
    for(unsigned biome=0;biome<3;biome++) {
        unsigned first=unsigned(natural.surface_vertices.size());
        SurfaceVertex quad[]={
            {-.4f,-.3f,.10f,.20f},{.4f,-.3f,.90f,.20f},{.4f,.3f,.90f,.80f},
            {-.4f,-.3f,.10f,.20f},{.4f,.3f,.90f,.80f},{-.4f,.3f,.10f,.80f}};
        natural.surface_vertices.insert(natural.surface_vertices.end(),quad,quad+6);
        natural.surface_recipes.push_back({biome,biome==2?5.5f:2.f,.2f,biome+2,.8f,.6f,first,6});
    }
    auto height=[](float x,float y,float*){return 2.5f+x*.07f+y*.11f;};
    auto shore=[](float,float){return Shore{2,.2f};};
    unsigned verified=0;
    for(int real:{2,1}) {
        unsigned biome=unsigned(2-real);Tile owner{17+real,-9+real,3,-4,real};
        GroundProjection projection{3,-4,64,32,128.f/224*.82f,480};
        auto weights=[&](float,float){std::array<float,5>w{};w[biome]=1;return w;};
        auto emit=[&](std::vector<MapVertex>&out,unsigned stop=0) {
            unsigned calls=0;return emit_surface_decals(natural,owner,projection,height,shore,weights,
                [&](){return stop && ++calls==stop;},out);
        };
        std::vector<MapVertex>a,b;assert(emit(a)&&emit(b));
        std::size_t expected=30;
        assert(a.size()==expected&&b.size()==expected&&!std::memcmp(a.data(),b.data(),a.size()*sizeof(MapVertex)));
        for(auto const&v:a) {
            assert(v.material_plains==5+float(biome));assert(v.material_desert==1);
            assert(v.u>=.1f&&v.u<=.9f&&v.v>=.2f&&v.v<=.8f);assert(v.base_terrain==-9);
            assert(std::isfinite(v.world_x)&&std::isfinite(v.world_y)&&std::isfinite(v.world_z));
        }
        std::vector<MapVertex>cancelled;assert(!emit(cancelled,2));assert(cancelled.empty());
        verified+=unsigned(a.size());
    }
    {
        unsigned emitted=0,empty=0;
        for(int coordinate=0;coordinate<32;coordinate++) {
            Tile owner{coordinate,-coordinate,coordinate,-coordinate,0};
            GroundProjection projection{coordinate,-coordinate,64,32,128.f/224*.82f,480};
            auto weights=[](float,float){return std::array<float,5>{0,0,1,0,0};};
            std::vector<MapVertex>a,b;
            auto run=[&](std::vector<MapVertex>&out){return emit_surface_decals(natural,owner,projection,height,shore,weights,[]{return false;},out);};
            assert(run(a)&&run(b)&&a.size()==b.size());
            assert(a.empty()||a.size()==18);
            if(a.empty())empty++;else {emitted++;verified+=unsigned(a.size());}
        }
        assert(emitted>0&&empty>0);
    }
    {
        Tile owner{2,4,3,-4,2};GroundProjection projection{3,-4,64,32,.4f,480};
        auto weak=[](float,float){return std::array<float,5>{.01f,.99f,0,0,0};};
        std::vector<MapVertex>out;assert(emit_surface_decals(natural,owner,projection,height,shore,weak,[]{return false;},out));
        assert(out.empty());
    }
    std::cout<<"PASS source surface composition: 3 biomes, "<<verified
             <<" deterministic vertices, biome fade and cancellation\n";
}
