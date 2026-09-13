#pragma once
// Production river pages and natural CPU inputs; no graphics API ownership.
#include "data.h"
#include <memory>
#include "../../../native/render_core/world_topology.h"
#include "../../../native/source_fidelity/river_corridor.h"
namespace c3x_renderer { namespace fidelity {
struct NaturalWorld : NaturalData {
    struct RiverPage {
        int c=0,r=0;std::uint64_t used=0;
        std::shared_ptr<river::Corridor> field;
        std::vector<std::pair<std::size_t,std::uint32_t>> dependencies;
    };
    std::vector<RiverPage> river_pages;
    render_core::WorldTopology const*river_world=nullptr;
    std::uint64_t river_epoch=0;
    std::int64_t river_revision=-1;
    render_core::World river_dimensions;
    void reset_world(){river_pages.clear();river_world=nullptr;river_revision=-1;}
    void update_rivers(render_core::WorldTopology const&w,std::int64_t revision){
        auto dims=w.dimensions();
        if(river_world!=&w || dims.width!=river_dimensions.width || dims.height!=river_dimensions.height ||
           dims.wrap_x!=river_dimensions.wrap_x || dims.wrap_y!=river_dimensions.wrap_y)river_pages.clear();
        else if(river_revision!=revision) {
            river_pages.erase(std::remove_if(river_pages.begin(),river_pages.end(),[&](auto const& page){
                for(auto const& input:page.dependencies)if(w.at(input.first)!=input.second)return true;
                return false;
            }),river_pages.end());
        }
        river_world=&w;river_dimensions=dims;river_revision=revision;
    }
    RiverPage& river_page_entry(double x,double y){
        int pc=int(std::floor(x/8)),pr=int(std::floor(y/8));++river_epoch;
        for(auto&p:river_pages)if(p.c==pc && p.r==pr){p.used=river_epoch;return p;}
        // Sixteen 8x8 pages, each with a four-cell authoritative support halo.
        // Distant jumps evict LRU fields instead of building the whole map.
        if(river_pages.size()==16){auto it=std::min_element(river_pages.begin(),river_pages.end(),[](auto const&a,auto const&b){return a.used<b.used;});river_pages.erase(it);}
        RiverPage page;page.c=pc;page.r=pr;page.used=river_epoch;
        auto const&w=*river_world;auto dims=w.dimensions();hydro::Field field;
        // Record topology and every terrain-height lookup, including absent
        // edge cells. Current 16x16 support plus curve/height collars fits well
        // below this bound: at most 16 pages * 1024 dependency pairs coexist.
        std::map<std::size_t,std::uint32_t> inputs;
        auto read=[&](int c,int r){
            auto index=w.index(c,r);auto value=w.at(index);inputs.emplace(index,value);
            if(inputs.size()>1024)throw std::runtime_error("river page dependency limit");
            return value;
        };
        field.map_width=dims.width;field.map_height=dims.height;field.wraps=dims.wrap_x;
        for(int r=pr*8-4;r<pr*8+12;r++)for(int c=pc*8-4;c<pc*8+12;c++){
            auto bits=read(c,r);if(bits==0xffffffffu)continue;
            int rx=c+r,ry=c-r;if(dims.wrap_x)rx=render_core::mod(rx,dims.width);if(dims.wrap_y)ry=render_core::mod(ry,dims.height);
            field.tiles[{c,r}]={c,r,rx,ry,int(bits&255),int((bits>>8)&255),unsigned((bits>>16)&255)};
        }
        auto lookup=[&](int c,int r){auto bits=read(c,r);int rx=c+r,ry=c-r;
            if(dims.wrap_x)rx=render_core::mod(rx,dims.width);if(dims.wrap_y)ry=render_core::mod(ry,dims.height);
            return Tile{rx,ry,c,r,bits==0xffffffffu?-1:int((bits>>8)&255)};};
        page.field=std::make_shared<river::Corridor>();
        page.field->build(field,[&](double u,double v){return height(float(u),float(v),lookup);});
        page.dependencies.assign(inputs.begin(),inputs.end());
        river_pages.push_back(std::move(page));return river_pages.back();
    }
    template<class Observe> river::Corridor const& river_page(double x,double y,Observe observe){
        auto& page=river_page_entry(x,y);observe(page);return *page.field;
    }
    river::Corridor const& river_page(double x,double y){return river_page(x,y,[](auto const&){});}
    // A tile compiler may query neighboring pages while using its bound field.
    // Keep that one immutable field alive across vector moves and LRU eviction;
    // release it when this tile finishes, without growing the 16-page cache.
    template<class Observe> std::shared_ptr<river::Corridor const> retain_river_page(double x,double y,Observe observe){
        auto& page=river_page_entry(x,y);observe(page);return page.field;
    }
    std::shared_ptr<river::Corridor const> retain_river_page(double x,double y){return retain_river_page(x,y,[](auto const&){});}
    template<class Observe> river::Sample river_sample(hydro::P p,Observe observe){return river_page(p.x,p.y,observe).sample(p);}
    river::Sample river_sample(hydro::P p){return river_page(p.x,p.y).sample(p);}
    template<class Observe> bool river_affects(int c,int r,Observe observe){return river_page(c+.5,r+.5,observe).affects(c,r);}
    bool river_affects(int c,int r){return river_page(c+.5,r+.5).affects(c,r);}
};
} }
