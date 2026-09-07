#include "../../systems/hydrology/river_corridor.h"
#include "../../systems/objects/canopy_layout.h"
#include <cassert>
#include <iostream>
#include <set>

int main(int argc,char** argv) {
    assert(argc==4);
    for(unsigned count:{36u,49u}) {
        std::set<unsigned> first_slots;
        for(unsigned x=0;x<100;++x) {
            unsigned seed=x*0x193u ^ 58u*0x217u;
            auto a=canopy::slots(seed,count),b=canopy::slots(seed,count);
            assert(a==b);std::set<unsigned> unique(a.begin(),a.end());
            assert(unique.size()==count && *unique.begin()==0 && *unique.rbegin()==count-1);
            assert(a!=canopy::slots(seed ^ 0x217u,count));
            first_slots.insert(a[0]);
        }
        assert(first_slots.size()>count*3/4); // Anchors cannot remain in one corner.
    }
    river::Corridor a,b;
    a.load_pool_profiles(argv[3]);b.load_pool_profiles(argv[3]);
    a.build(argv[1],[](double,double){return 2.5;});
    b.build(argv[2],[](double,double){return 2.5;});
    unsigned matched=0;
    for(auto const& e:a.edges) {
        if(e.a.x<3 || e.a.x>8 || e.a.y<1 || e.a.y>8)continue;
        auto f=std::find_if(b.edges.begin(),b.edges.end(),[&](auto const& q){return e.id==q.id;});
        assert(f!=b.edges.end());
        for(unsigned j=0;j<e.points.size();++j)
            assert(hydro::length(e.points[j]-(f->points[j]+hydro::P{2,0}))<1e-9);
        for(auto const& p:e.points)assert(a.sample(p).distance<1e-7);
        for(double side:{-1.,1.}) {
            river::P pa,pb;
            assert(a.bank_point(e.points[16],11.5,side,pa));
            assert(b.bank_point(f->points[16],11.5,side,pb));
            assert(hydro::length(pa-(pb+hydro::P{2,0}))<1e-8);
        }
        ++matched;
    }
    assert(matched>0);
    river::Corridor straight;straight.insert({0,0},{1,0});
    for(double side:{-1.,1.}) {
        river::P point;
        assert(straight.bank_point({.5,0},11.5,side,point));
        assert(std::abs(straight.sample(point).distance-11.5)<1e-8);
    }
    river::Corridor relief;
    auto mountain=[](double x,double y){return 2.5+90*std::exp(-((x-4.15)*(x-4.15)+(y-4.8)*(y-4.8))/.3);};
    relief.build(argv[1],mountain);
    unsigned improved=0;
    for(auto const& e:relief.edges) {
        assert(e.cost<=e.original_cost+1e-8);
        improved+=e.cost+1<e.original_cost;
        assert(hydro::length(e.points.front()-e.a)<1e-9);
        assert(hydro::length(e.points.back()-e.b)<1e-9);
    }
    assert(improved>0);
    for(auto const& t:a.terminals)if(!t.mouth) {
        assert(a.sample(t.p).distance<0);
        assert(a.sample(t.p+hydro::P{.15,0}).distance<6);
        if(t.p.x>3 && t.p.x<8 && t.p.y>1 && t.p.y<8) {
            double small=1,large=0;
            for(int j=0;j<32;++j) {
                double theta=j*6.283185307179586/32;
                river::P delta{std::cos(theta)*.22,std::sin(theta)*.22};
                double radius=a.pool_radius(t,delta);small=std::min(small,radius);large=std::max(large,radius);
                assert(std::abs(a.sample(t.p+delta).distance-b.sample(t.p+delta-hydro::P{2,0}).distance)<1e-8);
            }
            assert(large-small>.02);
        }
    }
    std::cout<<"PASS stable canopy permutations; "<<matched<<" crop-matched river edges; "<<improved<<" relief-aware bows; connected headwater pools\n";
}
