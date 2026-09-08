#pragma once
// Production river pages and natural CPU inputs; no graphics API ownership.
#include "data.h"
#include "../../../native/profile_v2/world_topology.h"
#include "../../../native/source_fidelity/river_corridor.h"
namespace c3x_renderer { namespace fidelity {
struct NaturalWorld : NaturalData {
    struct RiverPage {int c=0,r=0;std::uint64_t used=0;river::Corridor field;};
    std::vector<RiverPage> river_pages;
    profile_v2::WorldTopology const*river_world=nullptr;
    std::uint64_t river_epoch=0;
    std::int64_t river_revision=-1;
    void reset_world(){river_pages.clear();river_world=nullptr;river_revision=-1;}
    void update_rivers(profile_v2::WorldTopology const&w,std::int64_t revision){
        river_world=&w;
        if(river_revision!=revision){river_pages.clear();river_revision=revision;}
    }
    river::Corridor const& river_page(double x,double y){
        int pc=int(std::floor(x/8)),pr=int(std::floor(y/8));++river_epoch;
        for(auto&p:river_pages)if(p.c==pc && p.r==pr){p.used=river_epoch;return p.field;}
        // Sixteen 8x8 pages, each with a four-cell authoritative support halo.
        // Distant jumps evict LRU fields instead of building the whole map.
        if(river_pages.size()==16){auto it=std::min_element(river_pages.begin(),river_pages.end(),[](auto const&a,auto const&b){return a.used<b.used;});river_pages.erase(it);}
        RiverPage page;page.c=pc;page.r=pr;page.used=river_epoch;
        auto const&w=*river_world;auto dims=w.dimensions();hydro::Field field;
        field.map_width=dims.width;field.map_height=dims.height;field.wraps=dims.wrap_x;
        for(int r=pr*8-4;r<pr*8+12;r++)for(int c=pc*8-4;c<pc*8+12;c++){
            auto bits=w.at(w.index(c,r));if(bits==0xffffffffu)continue;
            int rx=c+r,ry=c-r;if(dims.wrap_x)rx=profile_v2::mod(rx,dims.width);if(dims.wrap_y)ry=profile_v2::mod(ry,dims.height);
            field.tiles[{c,r}]={c,r,rx,ry,int(bits&255),int((bits>>8)&255),unsigned((bits>>16)&255)};
        }
        auto lookup=[&](int c,int r){auto t=w.tile(c,r);int rx=c+r,ry=c-r;
            if(dims.wrap_x)rx=profile_v2::mod(rx,dims.width);if(dims.wrap_y)ry=profile_v2::mod(ry,dims.height);
            return Tile{rx,ry,c,r,t.present?t.real:-1};};
        page.field.build(field,[&](double u,double v){return height(float(u),float(v),lookup);});
        river_pages.push_back(std::move(page));return river_pages.back().field;
    }
    river::Sample river_sample(hydro::P p){return river_page(p.x,p.y).sample(p);}
    bool river_affects(int c,int r){return river_page(c+.5,r+.5).affects(c,r);}
};
} }
