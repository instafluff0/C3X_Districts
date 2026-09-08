#include "render_core/terrain_query.h"
#include "render_core/coast_index.h"
#include "render_core/world_topology.h"
#include "render_core/world_coast.h"
#include "render_core/relief_query.h"
#include "render_core/exact_point_cache.h"
#include "render_core/source/systems/terrain/surface.h"
#include "render_core/source/systems/hydrology/field.h"
#include "render_core/source/systems/relief/continuous_normal.h"
#include <cassert>
#include <iostream>

namespace port = c3x_renderer::render_core;
void near(double a,double b,double tolerance=1e-10) {
    if(std::abs(a-b)>tolerance) { std::cerr<<a<<" != "<<b<<"\n"; std::abort(); }
}
int main() {
    // A certified inland tile has exactly zero relief/owner channels even
    // with maximal source height, rocky coast response and arbitrary rivers.
    // Neighbour edits must revoke the certificate; no cache approximation.
    {
        int changed_x=99, changed_y=99, changed_real=2, observed=0;
        auto lookup=[&](int x,int y) {++observed;
            int kind=(x==changed_x && y==changed_y) ? changed_real : 2;
            return port::Tile{kind==0 ? 0 : 2,kind,true};};
        auto source=[](int,unsigned,int,float,float){return 1.f;};
        auto shore=[](float,float){return port::ShoreSample{.86,.2,1,.1};};
        auto river=[](int,int,float,float){return 0.f;};
        auto dune=[](float,float){return 18.6f;};
        auto active=[](int,int){return 1.f;};
        port::ReliefQuery query(port::World{100,100,true,false},lookup,source,shore,river,dune,active);
        port::FlatGroundRegion flat(20,20,1.6+1e-6,lookup);
        assert(flat.certified && observed==25);
        for(int y=0;y<=102;y++)for(int x=0;x<=102;x++) {
            float u=19.99f+x*.01f,v=19.99f+y*.01f;
            assert(flat.contains(u,v));auto sample=query.sample(u,v);
            assert(sample.height==0 && sample.authored_height==0 && sample.authored_blend==0);
            assert((sample.owner==std::array<float,4>{}));
        }
        assert(!flat.contains(19.98f,20.f));
        assert(!port::FlatGroundRegion(20,20,1.6,lookup).certified);
        for(int kind:{0,5,6,10})for(int y=18;y<=22;y++)for(int x=18;x<=22;x++) {
            changed_x=x;changed_y=y;changed_real=kind;
            assert(!port::FlatGroundRegion(20,20,3,lookup).certified);
        }
    }
    {
        port::ExactPointCache<std::array<double,4>> cache;
        int calls=0;
        auto query=[&](float x,float y) {return cache.get(x,y,[&]() {
            ++calls;return std::array<double,4>{x,y,x+y,x-y};});};
        auto first=query(1.f,2.f);assert(query(1.f,2.f)==first && calls==1);
        query(0.f,0.f);int before=calls;query(-0.f,0.f);assert(calls==before);
        query(std::nextafter(1.f,2.f),2.f);assert(calls==before+1);
        for(int i=0;i<30000;i++) {
            float x=i*.037f,y=i*-.081f;auto actual=query(x,y);
            assert(actual[0]==x && actual[1]==y && actual[2]==double(x+y));
        }
        assert(cache.bytes()<=2u*1024u*1024u);
        assert(query(1.f,2.f)==first);
        cache.clear();before=calls;assert(query(1.f,2.f)==first && calls==before+1);
        assert(query(1.f,2.f)==first && calls==before+1);
    }
    // A translated crop, with every material, diagonal coast and a cliff owner.
    q2::Surface reference; reference.width=100; reference.height=80;
    hydro::Field shore; shore.cols=4; shore.rows=4; shore.map_width=100;
    shore.map_height=80; shore.origin_x=54; shore.origin_y=46;
    shore.wraps=true; shore.shoreline_profile=2;
    for(int y=-5;y<=8;y++) for(int x=-5;x<=8;x++) {
        int base=(x+y<2) ? 11 : port::mod(x*3+y,4);
        int real=x==2 ? 5 : y==3 ? 9 : y==2 ? 4 : base;
        reference.tiles.push_back({x,y,54+x+y,46+x-y,base,real});
        shore.tiles[{x,y}]={x,y,54+x+y,46+x-y,base,real,0};
    }
    auto lookup=[&](int x,int y) {
        auto t=reference.at(x-50,y-4);
        return t ? port::Tile{t->base,t->real,true} : port::Tile{};
    };
    port::World world{100,80,true,false};
    port::ShoreField field(world,lookup);
    for(int y=-10;y<=50;y++) for(int x=-10;x<=50;x++) {
        double px=x*.07,py=y*.07;
        auto expected=reference.sample(px,py);
        auto actual=port::material_weights({px+50,py+4},world,lookup);
        for(int i=0;i<5;i++) near(expected.weights[i],actual[i]);
        near(shore.world_noise({px,py},.24,781),field.world_noise({px+50,py+4},.24,781));
        near(shore.occupancy({px,py}),field.occupancy({px+50,py+4}));
        near(shore.signed_coverage({px,py}),field.coverage({px+50,py+4}));
        auto shifted=port::material_weights({px+100,py+54},world,[&](int c,int r){return lookup(c-50,r-50);});
        for(int i=0;i<5;i++) near(actual[i],shifted[i]);
    }
    shore.build();
    std::vector<port::CoastSegment> contour;
    for(int y=-1;y<4;y++) for(int x=-1;x<4;x++) {
        auto const& c=field.cell(x+50,y+4); contour.insert(contour.end(),c.begin(),c.end());
    }
    assert(contour.size()==shore.coast.size());
    for(int y=0;y<20;y++) for(int x=0;x<20;x++) {
        double px=x*.15,py=y*.15;
        auto expected=shore.sample({px,py}); auto actual=field.sample({px+50,py+4},contour);
        near(expected.shore_distance,actual.distance);
        near(expected.beach_width,actual.beach_width);
        near(expected.rocky,actual.rocky); near(expected.depth,actual.depth);
    }
    for(int c=-2;c<6;c++) for(int r=-2;r<6;r++) {
        double value=field.coverage({double(c+50),double(r+4)});
        assert((value<0)==port::water(lookup(c+50,r+4)));
    }
    port::World torus{100,80,true,true};
    port::ShoreField wrapped(torus,lookup);
    for(double frequency:{.24,.82,2.4,.38,.66}) {
        double n=wrapped.world_noise({50.125,4.375},frequency,781);
        near(n,wrapped.world_noise({100.125,54.375},frequency,781));
        near(n,wrapped.world_noise({90.125,-35.625},frequency,781));
    }
    auto height=[](double x,double y){return 112*(x*x*.05+std::sin(y)*.1);};
    auto actual=port::continuous_normal({1.2,2.3},.006,112,height);
    auto expected=q4_relief::continuous_normal(1.2,2.3,.006,112,height);
    for(int i=0;i<3;i++) near(actual[i],expected[i]);
    near(port::relief_support(.5,.5),1); near(port::relief_support(1.25,.5),0);
    near(port::relief_support(-.25,.5),0);
    near(port::volcano_uv(1,.5)[0],.69375);
    // The nearest-coast acceleration must remain exact, including distant
    // coast, and validate local edits without observing a global root revision.
    port::CoastIndex index(-16,-16,32);
    std::vector<port::CoastSegment> segments;
    for(int y=-15;y<15;y++) for(int x=-15;x<15;x++) {
        if(port::hash(unsigned(x*47+y*73))%7) continue;
        port::CoastSegment segment{{x+.1,y+.2},{x+.9,y+.8},.5};
        index.set_cell(x,y,{segment}); segments.push_back(segment);
    }
    for(int i=0;i<1000;i++) {
        port::Point p{double(port::hash(unsigned(i*29))%30000)*.001-15,
                      double(port::hash(unsigned(i*59))%30000)*.001-15};
        auto accelerated=index.nearest(p,[](auto,auto){});
        double brute=1e6;
        for(auto const& segment:segments) {
            auto ab=segment.b-segment.a;
            double t=std::clamp(port::dot(p-segment.a,ab)/port::dot(ab,ab),0.,1.);
            auto delta=p-(segment.a+ab*t); brute=std::min(brute,port::dot(delta,delta));
        }
        near(accelerated.squared,brute);
    }
    port::CoastIndex edits(0,0,16);
    edits.set_cell(0,0,{{{.4,0},{.4,1},.5}});
    std::map<std::uint64_t,std::uint64_t> dependencies;
    auto observed=[&](auto id,auto version){dependencies[id]=version;};
    edits.nearest({1.25,.25},observed);
    auto valid=[&](){for(auto pair:dependencies) if(edits.revision(pair.first)!=pair.second)return false;return true;};
    edits.set_cell(10,10,{{{10.4,10},{10.4,11},.5}});
    assert(valid()); // Distant edits do not rebuild the receiving mesh.
    edits.set_cell(1,0,{{{1.4,0},{1.4,1},.5}});
    assert(!valid()); // A new closer coast invalidates an empty-subtree certificate.
    dependencies.clear();
    auto closer=edits.nearest({1.25,.25},observed); near(closer.squared,.15*.15);
    edits.set_cell(1,0,{}); assert(!valid());
    port::WorldTopology topology;
    bool rejected_odd_wrap=false;std::vector<std::uint32_t> odd(100*79/2,2|(2<<8));
    try {topology.update({100,79,true,true},odd.data(),odd.size());}
    catch(std::invalid_argument const&) {rejected_odd_wrap=true;}
    assert(rejected_odd_wrap);
    std::vector<std::uint32_t> packed(100*80/2,2u|(2u<<8));
    auto changed=topology.update({100,80,true,true},packed.data(),packed.size());
    assert(changed.size()==packed.size());
    assert(topology.update({100,80,true,true},packed.data(),packed.size()).empty());
    for(int y=0;y<80;y++) for(int x=y&1;x<100;x+=2) {
        auto i=std::size_t((y*100+x)/2);
        assert(topology.index((x+y)/2,(x-y)/2)==i);
        assert(topology.index((x+y)/2+50,(x-y)/2+50)==i);
        assert(topology.index((x+y)/2+40,(x-y)/2-40)==i);
    }
    packed[125]=11u|(11u<<8);
    changed=topology.update({100,80,true,true},packed.data(),packed.size());
    assert(changed.size()==1);
    assert(port::water(topology.tile(changed[0].column,changed[0].row)));
    std::vector<std::uint32_t> compact(32*24/2);
    for(int y=0;y<24;y++) for(int x=y&1;x<32;x+=2) {
        unsigned base=((x/8+y/6)%3)==0 ? 11 : 2;
        compact[(y*32+x)/2]=base|((base==11 ? 11u : x%3==0 ? 5u : 2u)<<8);
    }
    port::WorldCoast coast_world;
    auto first=coast_world.update({32,24,true,true},compact.data(),compact.size(),1);
    assert(first.cells_built>0 && first.bytes<32u*1024u*1024u);
    assert(coast_world.update({32,24,true,true},compact.data(),compact.size(),1).cells_built==0);
    auto no_node=[](auto,auto){}; auto no_tile=[](auto,auto){};
    for(int i=0;i<20;i++) {
        port::Point p{double(i)*.7,double(i)*.13};
        auto original=coast_world.sample(p,no_node,no_tile);
        auto wrapped_x=coast_world.sample(p+port::Point{16,16},no_node,no_tile);
        auto wrapped_y=coast_world.sample(p+port::Point{12,-12},no_node,no_tile);
        near(original.distance,wrapped_x.distance,1e-8); near(original.distance,wrapped_y.distance,1e-8);
        near(original.rocky,wrapped_x.rocky,1e-8); near(original.rocky,wrapped_y.rocky,1e-8);
    }
    compact[0]=2u|(5u<<8); // Edit at both wrap seams, including mirrored support.
    coast_world.update({32,24,true,true},compact.data(),compact.size(),2);
    port::WorldCoast rebuilt;
    rebuilt.update({32,24,true,true},compact.data(),compact.size(),2);
    for(int i=0;i<100;i++) {
        port::Point p{double(i)*.13,double(i)*.07};
        auto incremental=coast_world.sample(p,no_node,no_tile);
        auto cold=rebuilt.sample(p,no_node,no_tile);
        near(incremental.distance,cold.distance); near(incremental.rocky,cold.rocky);
        near(incremental.beach_width,cold.beach_width); near(incremental.depth,cold.depth);
    }
    // Prepared disks reproduce exact nearest segments, including both wrap
    // axes, normal-sampling collars and out-of-domain fallback. A coast edit
    // invalidates the certificate before reusing the prepared query domain.
    for(int i=0;i<24;i++) {
        port::Point center{double(i)*.73,double(i)*.21};
        auto center_sample=coast_world.sample(center,no_node,no_tile);
        auto patch=coast_world.prepare(center,.73,std::abs(center_sample.distance),no_node);
        assert(patch.ready && patch.edges.size()<=2048);
        for(int y=-6;y<=6;y++)for(int x=-6;x<=6;x++) {
            auto point=center+port::Point{x*.083,y*.083};
            auto expected=coast_world.sample(point,no_node,no_tile);
            auto actual=coast_world.sample(point,no_node,no_tile,&patch);
            near(actual.distance,expected.distance,1e-12);near(actual.rocky,expected.rocky,1e-12);
            near(actual.beach_width,expected.beach_width,1e-12);near(actual.depth,expected.depth,1e-12);
        }
        auto outside=center+port::Point{2,2};
        assert(!patch.contains(outside));
        near(coast_world.sample(outside,no_node,no_tile,&patch).distance,
             coast_world.sample(outside,no_node,no_tile).distance);
    }
    {
        std::map<std::uint64_t,std::uint64_t> nodes;
        std::map<std::size_t,std::uint32_t> tiles;
        auto on_node=[&](auto id,auto revision){nodes.emplace(id,revision);};
        auto on_tile=[&](auto id,auto value){tiles.emplace(id,value);};
        port::Point center{.5,.5};
        auto center_sample=coast_world.sample(center,on_node,on_tile);
        auto patch=coast_world.prepare(center,.73,std::abs(center_sample.distance),on_node);
        coast_world.sample(center,on_node,on_tile,&patch);
        compact[0]=11u|(11u<<8);
        coast_world.update({32,24,true,true},compact.data(),compact.size(),3);
        bool valid=true;
        for(auto item:nodes)valid=valid && coast_world.node_revision(item.first)==item.second;
        for(auto item:tiles)valid=valid && coast_world.world().at(item.first)==item.second;
        assert(!valid);
    }
    int relief_kind=6;
    auto owners=[&](int c,int r){return port::Tile{2,c==0&&r==0 ? relief_kind : 2,true};};
    auto source=[](int kind,unsigned,int,float,float){return kind==5 ? 0.f : 1.f;};
    auto dry=[](float,float){return port::ShoreSample{100,0,0,0};};
    auto no_river=[](int,int,float,float){return 1000.f;};
    auto no_dune=[](float,float){return 0.f;};
    auto active=[](int,int){return 1.f;};
    port::ReliefQuery relief(port::World{100,80,true,false},owners,source,dry,no_river,no_dune,active);
    near(relief.sample(.5f,.5f).height,104*1.3,1e-4);
    near(relief.sample(1.f,.5f).height,104*1.3,1e-4);
    near(relief.sample(1.25f,.5f).height,0);
    relief_kind=10;
    auto skirt=relief.sample(1.f,.5f);
    near(skirt.height,88*1.6,1e-4);
    near(skirt.owner[0],.69375,1e-6); near(skirt.owner[1],.5);
    near(skirt.owner[2],1); near(skirt.owner[3],1);
    auto before_edge=relief.sample(1.f-1e-5f,.5f);
    near(before_edge.height,skirt.height,1e-3);
    near(before_edge.owner[0],skirt.owner[0],1e-5);
    for(int kind:{0,2,5,6,10}) {
        relief_kind=kind;
        for(int y=-5;y<=15;y++)for(int x=-5;x<=15;x++) {
            auto full=relief.sample(x*.1f,y*.1f);
            auto height_only=relief.sample(x*.1f,y*.1f,false);
            near(full.height,height_only.height,0);
            assert(height_only.authored_blend==0 && height_only.owner[3]==0);
        }
    }
    std::cout<<"render_core: material/shore/normal parity, wrap and relief support passed\n";
}
