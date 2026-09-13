#pragma once
// Production river pages and natural CPU inputs; no graphics API ownership.
#include "data.h"
#include <memory>
#include <unordered_map>
#include "../../../native/render_core/world_topology.h"
#include "../../../native/source_fidelity/river_corridor.h"
namespace c3x_renderer { namespace fidelity {
struct NaturalWorld : NaturalData {
    using Dependencies=std::unordered_map<std::size_t,std::uint32_t>;
    using CellKey=std::array<int,4>; // source page, then sampled world cell
    struct PageInputs {
        std::vector<std::pair<std::size_t,std::uint32_t>> values;
        render_core::WorldTopology const* checked_world=nullptr;
        std::int64_t checked_revision=-1;bool current=false;
        bool valid(render_core::WorldTopology const& world,std::int64_t revision){
            if(checked_world!=&world || checked_revision!=revision){
                current=true;for(auto const& input:values)if(world.at(input.first)!=input.second){current=false;break;}
                checked_world=&world;checked_revision=revision;
            }return current;
        }
    };
    struct CellContent {
        std::vector<std::uint64_t> values;
        std::shared_ptr<PageInputs> inputs;
    };
    using CellValue=std::shared_ptr<CellContent const>;
    using CellInputs=std::map<CellKey,CellValue>;
    using CellProof=std::vector<std::pair<CellKey,CellValue>>;
    struct CellCache : std::map<std::pair<int,int>,CellValue> {
        std::shared_ptr<PageInputs> inputs=std::make_shared<PageInputs>();
        std::size_t bytes=0;
    };
    struct RiverPage {int c=0,r=0;std::uint64_t used=0;
        std::shared_ptr<river::Corridor> field;
        std::shared_ptr<CellCache> cells=std::make_shared<CellCache>();
    };
    CellInputs* consumer=nullptr;
    CellKey last_cell={};bool last_cell_valid=false;
    struct DependencyScope {
        NaturalWorld& world;
        DependencyScope(NaturalWorld& owner,CellInputs* inputs):world(owner){
            world.consumer=inputs;world.last_cell_valid=false;
        }
        ~DependencyScope(){world.consumer=nullptr;world.last_cell_valid=false;}
        DependencyScope(DependencyScope const&)=delete;
    };
    // Exact ordered query data, not a page generation or a lossy digest. Empty
    // buckets are proofs too: a newly inserted river must invalidate consumers.
    CellValue cell_value(river::Corridor const& field,CellCache& cells,int c,int r){
        auto key=std::make_pair(c,r);auto old=cells.find(key);if(old!=cells.end())return old->second;
        std::vector<std::uint64_t> values;
        auto real=[&](double v){std::uint64_t bits;std::memcpy(&bits,&v,sizeof(bits));values.push_back(bits);};
        auto segments=field.buckets.find(key);values.push_back(segments!=field.buckets.end());
        values.push_back(segments==field.buckets.end()?0:segments->second.size());
        if(segments!=field.buckets.end())for(auto const& segment:segments->second){
            real(segment.a.x);real(segment.a.y);real(segment.b.x);real(segment.b.y);
        }
        // A separator carries the segment count; no ambiguous concatenations.
        auto nodes=field.terminal_buckets.find(key);
        values.push_back(nodes!=field.terminal_buckets.end());
        values.push_back(nodes==field.terminal_buckets.end()?0:nodes->second.size());
        if(nodes!=field.terminal_buckets.end())for(auto i:nodes->second){
            auto const& t=field.terminals[i];real(t.p.x);real(t.p.y);values.push_back(t.mouth);
            values.push_back(t.profile);real(t.yaw);
            auto count=field.pool_profiles.empty()?0:field.pool_profiles[t.profile].size();values.push_back(count);
            if(count)for(auto radius:field.pool_profiles[t.profile])real(radius);
        }
        auto bytes=sizeof(CellContent)+values.capacity()*sizeof(std::uint64_t)+96;
        if(values.size()>8192 || cells.size()>=4096 || bytes>512u*1024u-cells.bytes)throw std::length_error("river cell proof budget");
        cells.bytes+=bytes;
        auto result=std::make_shared<CellContent const>(CellContent{std::move(values),cells.inputs});
        cells.emplace(key,result);return result;
    }
    void observe(int pc,int pr,river::Corridor const& field,CellCache& cells,int c,int r){
        if(!consumer)return;
        CellKey key={pc,pr,c,r};if(last_cell_valid && key==last_cell)return;
        last_cell=key;last_cell_valid=true;
        if(consumer->find(key)==consumer->end()){
            if(consumer->size()>=256)throw std::length_error("river consumer proof budget");
            consumer->emplace(key,cell_value(field,cells,c,r));
        }
    }
    struct BoundRiver {
        NaturalWorld& owner;int c,r;
        std::shared_ptr<river::Corridor const> field;
        std::shared_ptr<CellCache> cells;
        river::Sample sample(hydro::P p) const {
            owner.observe(c,r,*field,*cells,int(std::floor(p.x)),int(std::floor(p.y)));
            return field->sample(p);
        }
    };
    bool valid(CellProof const& proof){
        if(!proof.empty() && !river_world)return false;
        for(auto const& input:proof){
            // Source proofs outlive page/GPU eviction. A camera move needs no
            // field construction merely to validate unchanged compiled content.
            if(river_world && input.second->inputs->valid(*river_world,river_revision))continue;
            auto const& key=input.first;
            auto& page=river_page_entry(double(key[0])*8+.5,double(key[1])*8+.5);
            auto value=cell_value(*page.field,*page.cells,key[2],key[3]);
            if(value!=input.second && value->values!=input.second->values)return false;
        }return true;
    }
    std::size_t proof_bytes(CellProof const& proof) const {
        std::size_t bytes=proof.capacity()*sizeof(proof[0]);
        // Conservatively charge shared payload to each retaining cache entry.
        for(auto const& input:proof)bytes+=sizeof(*input.second)+input.second->values.capacity()*sizeof(std::uint64_t)+
            sizeof(PageInputs)+input.second->inputs->values.capacity()*sizeof(std::pair<std::size_t,std::uint32_t>);
        return bytes;
    }
    std::vector<RiverPage> river_pages;
    render_core::WorldTopology const*river_world=nullptr;
    std::uint64_t river_epoch=0;
    std::int64_t river_revision=-1;
    void reset_world(){river_pages.clear();river_world=nullptr;river_revision=-1;}
    void update_rivers(render_core::WorldTopology const&w,std::int64_t revision){
        river_world=&w;
        if(river_revision!=revision){river_pages.clear();river_revision=revision;}
    }
    RiverPage& river_page_entry(double x,double y){
        int pc=int(std::floor(x/8)),pr=int(std::floor(y/8));++river_epoch;
        for(auto&p:river_pages)if(p.c==pc && p.r==pr){p.used=river_epoch;return p;}
        // Sixteen 8x8 pages, each with a four-cell authoritative support halo.
        // Distant jumps evict LRU fields instead of building the whole map.
        if(river_pages.size()==16){auto it=std::min_element(river_pages.begin(),river_pages.end(),[](auto const&a,auto const&b){return a.used<b.used;});river_pages.erase(it);}
        RiverPage page;page.c=pc;page.r=pr;page.used=river_epoch;
        auto const&w=*river_world;auto dims=w.dimensions();hydro::Field field;
        Dependencies inputs;
        auto value=[&](int c,int r){auto i=w.index(c,r);auto bits=w.at(i);
            if(i!=std::size_t(-1))inputs.emplace(i,bits);
            if(inputs.size()>4096)throw std::length_error("river dependency budget");
            return bits;};
        field.map_width=dims.width;field.map_height=dims.height;field.wraps=dims.wrap_x;
        for(int r=pr*8-4;r<pr*8+12;r++)for(int c=pc*8-4;c<pc*8+12;c++){
            auto bits=value(c,r);if(bits==0xffffffffu)continue;
            int rx=c+r,ry=c-r;if(dims.wrap_x)rx=render_core::mod(rx,dims.width);if(dims.wrap_y)ry=render_core::mod(ry,dims.height);
            field.tiles[{c,r}]={c,r,rx,ry,int(bits&255),int((bits>>8)&255),unsigned((bits>>16)&255)};
        }
        auto lookup=[&](int c,int r){auto bits=value(c,r);int rx=c+r,ry=c-r;
            if(dims.wrap_x)rx=render_core::mod(rx,dims.width);if(dims.wrap_y)ry=render_core::mod(ry,dims.height);
            return Tile{rx,ry,c,r,bits!=0xffffffffu?int((bits>>8)&255):-1};};
        page.field=std::make_shared<river::Corridor>();
        page.field->build(field,[&](double u,double v){return height(float(u),float(v),lookup);});
        page.cells->inputs->values.assign(inputs.begin(),inputs.end());
        river_pages.push_back(std::move(page));return river_pages.back();
    }
    river::Corridor const& river_page(double x,double y){auto& page=river_page_entry(x,y);
        observe(page.c,page.r,*page.field,*page.cells,int(std::floor(x)),int(std::floor(y)));return *page.field;}
    // A tile compiler may query neighboring pages while using its bound field.
    // Keep that one immutable field alive across vector moves and LRU eviction;
    // release it when this tile finishes, without growing the 16-page cache.
    std::shared_ptr<river::Corridor const> retain_river_page(double x,double y){return river_page_entry(x,y).field;}
    BoundRiver bind_river_page(double x,double y){auto& page=river_page_entry(x,y);
        return {*this,page.c,page.r,page.field,page.cells};}
    river::Sample river_sample(hydro::P p){return river_page(p.x,p.y).sample(p);}
    bool river_affects(int c,int r){return river_page(c+.5,r+.5).affects(c,r);}
};
} }
