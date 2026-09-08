// Migration witness: production river-page expressions versus their shared owner.
#include "world.h"
#include <cassert>
#include <iostream>
using namespace c3x_renderer;
using namespace c3x_renderer::fidelity;
struct OriginalWorld : NaturalData {
    struct RiverPage {int c=0,r=0;std::uint64_t used=0;river::Corridor field;};
    std::vector<RiverPage> river_pages;
    profile_v2::WorldTopology const*river_world=nullptr;
    std::uint64_t river_epoch=0;
    std::int64_t river_revision=-1;
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

template<class A,class B> void equal_pages(A const& a,B const& b) {
    assert(a.river_epoch==b.river_epoch && a.river_revision==b.river_revision);
    assert(a.river_pages.size()==b.river_pages.size() && a.river_pages.size()<=16);
    for(unsigned i=0;i<a.river_pages.size();i++) {
        auto const& x=a.river_pages[i];auto const& y=b.river_pages[i];
        assert(x.c==y.c && x.r==y.r && x.used==y.used);
        assert(x.field.edges.size()==y.field.edges.size());
        assert(x.field.terminals.size()==y.field.terminals.size());
    }
}
int main() {
    unsigned samples=0,near_rivers=0;
    for(unsigned wraps=0;wraps<3;wraps++) {
        profile_v2::World dims{64,64,wraps>0,wraps>1};profile_v2::WorldTopology world;
        std::vector<std::uint32_t> bits(2048,2|(2<<8));
        for(unsigned i=0;i<bits.size();i++) {
            int y=int(i/32),x=int(i%32)*2+(y&1);
            if(x>=32)bits[i]=12|(12<<8);
            if(x==16 && y>=4 && y<=24)bits[i]|=10u<<16;
            if(x==4 && y==6)bits[i]=2|(5<<8);
        }
        OriginalWorld old;NaturalWorld shared;
        old.fields.resize(1);old.fields[0].width=old.fields[0].height=2;
        old.fields[0].pixels={0,64,128,255};shared.fields=old.fields;
        for(int revision=0;revision<2;revision++) {
            if(revision)bits[(16*64+16)/2]&=~(255u<<16);
            world.update(dims,bits.data(),bits.size());old.update_rivers(world,revision);shared.update_rivers(world,revision);
            assert(old.river_pages.empty() && shared.river_pages.empty());
            // Nearby channels and a hill, then >16 pages to exercise LRU eviction.
            for(unsigned i=0;i<26;i++) {
                double x=i<5?double(8+i*2)+.5:double(int(i%6)-2)*8+.5;
                double y=i<5?4.5:double(int(i/6)-2)*8+.5;
                for(double d:{0.,.25}) {
                    auto a=old.river_sample({x+d,y-d}),b=shared.river_sample({x+d,y-d});
                    assert(a.distance==b.distance && a.source==b.source && a.mouth==b.mouth);
                    assert(old.river_affects(int(x),int(y))==shared.river_affects(int(x),int(y)));
                    equal_pages(old,shared);samples++;if(a.distance<1000)near_rivers++;
                }
            }
            auto epoch=shared.river_epoch;auto count=shared.river_pages.size();
            old.update_rivers(world,revision);shared.update_rivers(world,revision);
            assert(shared.river_epoch==epoch && shared.river_pages.size()==count);equal_pages(old,shared);
        }
        auto epoch=shared.river_epoch;shared.reset_world();
        assert(shared.river_pages.empty() && !shared.river_world && shared.river_revision==-1);
        assert(shared.river_epoch==epoch && shared.fields.size()==1);
    }
    assert(near_rivers>0);
    std::cout<<"PASS shared river world: "<<samples<<" exact samples; wrapped topology, revision invalidation and 16-page LRU\n";
}
