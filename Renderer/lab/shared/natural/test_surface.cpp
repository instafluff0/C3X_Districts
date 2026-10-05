#include "mesh.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <set>
using namespace c3x_renderer::fidelity;

struct Shore {float distance,beach_width;};

int main() {
    NaturalData natural;
    for(unsigned biome=0;biome<4;biome++) {
        unsigned first=unsigned(natural.surface_vertices.size());
        SurfaceVertex quad[]={
            {-.4f,-.3f,.10f,.20f},{.4f,-.3f,.90f,.20f},{.4f,.3f,.90f,.80f},
            {-.4f,-.3f,.10f,.20f},{.4f,.3f,.90f,.80f},{-.4f,.3f,.10f,.80f}};
        natural.surface_vertices.insert(natural.surface_vertices.end(),quad,quad+6);
        natural.surface_recipes.push_back({biome,biome==2?5.5f:2.f,.2f,biome+2,.8f,.6f,first,6});
    }
    auto height=[](float x,float y,float*){return 2.5f+x*.07f+y*.11f;};
    auto shore=[](float,float){return Shore{2,.2f};};
    auto lookup=[](int c,int r){return Tile{c+r,c-r,c,r,4};};
    unsigned verified=0;
    for(int real:{1,4}) {
        unsigned biome=real==4?3u:unsigned(2-real);Tile owner{17+real,-9+real,3,-4,real};
        GroundProjection projection{3,-4,64,32,128.f/224*.82f,480};
        auto weights=[&](float,float){std::array<float,5>w{};w[biome==3?1:biome]=1;return w;};
        auto emit=[&](std::vector<MapVertex>&out,unsigned stop=0) {
            unsigned calls=0;return emit_surface_decals(natural,owner,projection,lookup,height,shore,weights,
                [&](){return stop && ++calls==stop;},out);
        };
        std::vector<MapVertex>a,b;assert(emit(a)&&emit(b));
        assert(a.size()>0&&a.size()<=132&&a.size()%6==0&&b.size()==a.size()&&
               !std::memcmp(a.data(),b.data(),a.size()*sizeof(MapVertex)));
        for(auto const&v:a) {
            assert(v.material_plains==5+float(biome));assert(v.material_desert==1);
            assert(v.u>=.1f&&v.u<=.9f&&v.v>=.2f&&v.v<=.8f);assert(v.base_terrain==-9);
            assert(std::isfinite(v.world_x)&&std::isfinite(v.world_y)&&std::isfinite(v.world_z));
            assert(std::abs(v.world_z-(2.5f+v.world_x*.07f+v.world_y*.11f+.18f)/112.f)<.00001f);
            assert(v.normal_z>.99f&&v.normal_x<0&&v.normal_y<0);
        }
        std::vector<MapVertex>cancelled;assert(!emit(cancelled,2));assert(cancelled.empty());
        verified+=unsigned(a.size());
    }
    {
        for(int coordinate=0;coordinate<24;coordinate++) {
            Tile owner{coordinate,7,3,-4,2};
            GroundProjection projection{3,-4,64,32,128.f/224*.82f,480};
            auto weights=[](float,float){return std::array<float,5>{1,0,0,0,0};};
            std::vector<MapVertex>out;
            assert(emit_surface_decals(natural,owner,projection,lookup,height,shore,weights,
                                       []{return false;},out));
            // Grass source recipes are disconnected triangles. Their optional
            // mesh is omitted so its straight edges cannot mark the map.
            assert(out.empty());
        }
    }
    {
        unsigned emitted=0,empty=0;
        for(int coordinate=0;coordinate<32;coordinate++) {
            Tile owner{coordinate,-coordinate,coordinate,-coordinate,0};
            GroundProjection projection{coordinate,-coordinate,64,32,128.f/224*.82f,480};
            auto weights=[](float,float){return std::array<float,5>{0,0,1,0,0};};
            std::vector<MapVertex>a,b;
            auto run=[&](std::vector<MapVertex>&out){return emit_surface_decals(natural,owner,projection,lookup,height,shore,weights,[]{return false;},out);};
            assert(run(a)&&run(b)&&a.size()==b.size());
            assert(a.empty()||a.size()==18);
            if(a.empty())empty++;else {
                for(auto const&v:a) {
                    assert(std::abs(v.world_z-2.68f/112.f)<.00001f);
                    assert(v.normal_x==0&&v.normal_y==0&&v.normal_z==1);
                }
                emitted++;verified+=unsigned(a.size());
            }
        }
        assert(emitted>0&&empty>0);
    }
    {
        // Coastal decals straddle a changing shore inside one owner tile.
        // Tile-center coverage cannot describe those vertices, including a
        // submerged tip of an otherwise inland plains/desert patch.
        unsigned partial=0,wet=0,dry=0;
        auto coast=[](float x,float){return Shore{.2f+(x-3.5f)*1.5f,.15f};};
        for(int real:{0,1})for(int coordinate=0;coordinate<32;coordinate++) {
            Tile owner{coordinate,7,3,-4,real};
            GroundProjection projection{3,-4,64,32,.4f,480};
            auto weights=[&](float,float){std::array<float,5>w{};w[2-real]=1;return w;};
            std::vector<MapVertex>out;
            assert(emit_surface_decals(natural,owner,projection,lookup,height,coast,weights,
                                       []{return false;},out));
            for(auto const&v:out) {
                auto s=coast(v.world_x,v.world_y);
                float coverage=real==0?desert_coast_coverage(s.distance):coast_coverage(s.distance,s.beach_width);
                assert(v.base_terrain==-10+coverage);
                assert(v.world_valid==1+coast_ramp((s.distance-s.beach_width-.10f)/.90f));
                if(coverage==0)wet++;else if(coverage==1)dry++;else partial++;
            }
        }
        assert(partial>0 && wet>0 && dry>0);
    }
    {
        Tile owner{2,4,3,-4,2};GroundProjection projection{3,-4,64,32,.4f,480};
        auto weak=[](float,float){return std::array<float,5>{.01f,.99f,0,0,0};};
        std::vector<MapVertex>out;assert(emit_surface_decals(natural,owner,projection,lookup,height,shore,weak,[]{return false;},out));
        assert(out.empty());
    }
    {
        Tile owner{7,7,4,0,4};GroundProjection projection{4,0,64,32,.4f,480};
        auto isolated=[](int c,int r){return Tile{c+r,c-r,c,r,c==4&&r==0?4:1};};
        auto plains=[](float,float){return std::array<float,5>{0,1,0,0,0};};
        std::vector<MapVertex>out;
        assert(emit_surface_decals(natural,owner,projection,isolated,height,shore,plains,[]{return false;},out));
        assert(!out.empty());
        bool faded=false;
        for(auto const&v:out){assert(v.material_desert>=0&&v.material_desert<=1);
            faded=faded||v.material_desert<.99f;}
        assert(faded);
    }
    std::cout<<"PASS source surface composition: grass omitted, 3 active biomes, "<<verified
             <<" deterministic vertices, biome fade and cancellation\n";
}
